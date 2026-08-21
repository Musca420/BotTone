from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from adaptive_bot.adapters.bitunix.collector import ECONOMIC_QUOTE_PROTOCOL_HASH
from adaptive_bot.bitunix_fees import futures_fee_bps

TAPE_ROOT = Path("data/raw/bitunix_microstructure")
REPORT = Path("data/reports/musca_v5_post_only_replay.json")
LOT_SIZE = Decimal("0.001")
NOTIONAL_USDT = Decimal("10000")
QUOTE_LIFETIME_SECONDS = 30
POSITION_LIFETIME_SECONDS = 300
SUBMISSION_LATENCY_MS = 250
TEN_THOUSAND = Decimal("10000")


def load_tape(root: Path = TAPE_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("btcusdt_post_only_tape_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("at"):
                    rows.append(row)
    if not rows:
        return []
    tape_start = min(_at(row["at"]) for row in rows) - timedelta(minutes=5)
    tape_end = max(_at(row["at"]) for row in rows) + timedelta(minutes=5)
    context_root = root.parent / "btc_context"
    for path in sorted(context_root.glob("btc_context_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for record in payload.get("records", []):
                    if not isinstance(record, dict) or record.get("exchange") != "bitunix":
                        continue
                    available_at = record.get("available_at")
                    if available_at is None or not tape_start <= _at(available_at) <= tape_end:
                        continue
                    rate = _decimal(record.get("funding_rate"))
                    if rate is None:
                        continue
                    if record.get("funding_rate_unit") != "fraction_of_notional":
                        rate /= Decimal("100")
                    rows.append(
                        {
                            "type": "funding",
                            "at": available_at,
                            "rate": str(rate),
                            "next_funding_at": record.get("next_funding_timestamp"),
                        }
                    )
    return sorted(rows, key=lambda row: _at(row["at"]))


def replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    observed = [
        row for row in rows if row.get("type") in {"book", "funding", "quote", "trade"}
    ]
    funding: dict[str, Any] = {"rate": None, "next": None}
    for row in observed:
        now = _at(row["at"])
        for state in states.values():
            orders = state.get("orders")
            if orders and now > orders["expires"]:
                state["orders"] = None
                state["expired_pairs"] += 1
        if row.get("type") == "quote":
            if row.get("protocol_hash") != ECONOMIC_QUOTE_PROTOCOL_HASH:
                continue
            fair = _decimal(row.get("fair_value"))
            quotes = row.get("quotes")
            if fair is None or not isinstance(quotes, dict):
                continue
            quantity = (NOTIONAL_USDT / fair / LOT_SIZE).to_integral_value(
                rounding=ROUND_DOWN
            ) * LOT_SIZE
            for profile, raw_quote in quotes.items():
                if not isinstance(raw_quote, dict):
                    continue
                state = states.setdefault(
                    str(profile),
                    {
                        "orders": None,
                        "position": None,
                        "pair_cycles": 0,
                        "expired_pairs": 0,
                        "post_only_rejections": 0,
                        "proxy_fills": 0,
                        "closed": [],
                    },
                )
                if state["orders"] is not None or state["position"] is not None:
                    continue
                bid = _order(raw_quote, "bid", quantity)
                ask = _order(raw_quote, "ask", quantity)
                if bid is None or ask is None:
                    continue
                state["orders"] = {
                    "bid": bid,
                    "ask": ask,
                    "fair": fair,
                    "placed": now,
                    "active": now + timedelta(milliseconds=SUBMISSION_LATENCY_MS),
                    "expires": now
                    + timedelta(
                        milliseconds=SUBMISSION_LATENCY_MS,
                        seconds=QUOTE_LIFETIME_SECONDS,
                    ),
                }
                state["pair_cycles"] += 1
        elif row.get("type") == "trade":
            _apply_trade(states, row, now)
        elif row.get("type") == "book":
            _apply_book(states, row, now)
        elif row.get("type") == "funding":
            _apply_funding(states, funding, row, now)
    return _report(states, observed)


def run(root: Path = TAPE_ROOT, report_path: Path = REPORT) -> dict[str, Any]:
    report = replay(load_tape(root))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(report_path)
    return report


def _order(raw: dict[str, Any], side: str, quantity: Decimal) -> dict[str, Decimal] | None:
    price = _decimal(raw.get(f"{side}_price"))
    queue = _decimal(raw.get(f"{side}_queue_btc"))
    if price is None or queue is None or price <= 0 or queue < 0 or quantity <= 0:
        return None
    return {"price": price, "queue": queue, "remaining": quantity, "quantity": quantity}


def _apply_trade(states: dict[str, dict[str, Any]], row: dict[str, Any], now: datetime) -> None:
    price = _decimal(row.get("price"))
    quantity = _decimal(row.get("quantity"))
    aggressor = str(row.get("aggressor_side", "")).lower()
    if price is None or quantity is None or quantity <= 0:
        return
    for state in states.values():
        orders = state.get("orders")
        if not orders or now < orders["active"]:
            continue
        side = "bid" if aggressor == "sell" else "ask" if aggressor == "buy" else None
        if side is None:
            continue
        order = orders[side]
        through = price < order["price"] if side == "bid" else price > order["price"]
        at_quote = price == order["price"]
        if not through and not at_quote:
            continue
        available = quantity
        if not through:
            consumed = min(order["queue"], available)
            order["queue"] -= consumed
            available -= consumed
        if available <= 0:
            continue
        order["remaining"] -= min(order["remaining"], available)
        if order["remaining"] > 0:
            continue
        state["position"] = {
            "side": "long" if side == "bid" else "short",
            "entry": order["price"],
            "fair": orders["fair"],
            "quantity": order["quantity"],
            "entered": now,
            "mfe_bps": Decimal("0"),
            "mae_bps": Decimal("0"),
            "funding_bps": Decimal("0"),
        }
        state["orders"] = None
        state["proxy_fills"] += 1


def _apply_book(states: dict[str, dict[str, Any]], row: dict[str, Any], now: datetime) -> None:
    bid = _decimal(row.get("best_bid"))
    ask = _decimal(row.get("best_ask"))
    if bid is None or ask is None or bid <= 0 or ask <= bid:
        return
    midpoint = (bid + ask) / 2
    for profile, state in states.items():
        orders = state.get("orders")
        if orders and now >= orders["active"] and (
            orders["bid"]["price"] >= ask or orders["ask"]["price"] <= bid
        ):
            state["orders"] = None
            state["post_only_rejections"] += 1
        position = state.get("position")
        if not position:
            continue
        direction = Decimal("1") if position["side"] == "long" else Decimal("-1")
        excursion = direction * (midpoint - position["entry"]) / position["entry"] * TEN_THOUSAND
        position["mfe_bps"] = max(position["mfe_bps"], excursion)
        position["mae_bps"] = min(position["mae_bps"], excursion)
        target = bid >= position["fair"] if direction > 0 else ask <= position["fair"]
        timed_out = now >= position["entered"] + timedelta(seconds=POSITION_LIFETIME_SECONDS)
        adverse_stop = excursion <= Decimal("-100")
        if not target and not timed_out and not adverse_stop:
            continue
        exit_price = bid if direction > 0 else ask
        gross = direction * (exit_price - position["entry"]) / position["entry"] * TEN_THOUSAND
        level = int(profile.removeprefix("VIP"))
        maker, taker = futures_fee_bps(level)
        net = gross - Decimal(str(maker + taker)) + position["funding_bps"]
        state["closed"].append(
            {
                "entered_at": position["entered"].isoformat(),
                "exited_at": now.isoformat(),
                "side": position["side"],
                "entry": str(position["entry"]),
                "exit": str(exit_price),
                "gross_bps": float(gross),
                "net_bps": float(net),
                "mfe_bps": float(position["mfe_bps"]),
                "mae_bps": float(position["mae_bps"]),
                "funding_bps": float(position["funding_bps"]),
                "reason": (
                    "VWAP_TARGET" if target else "ADVERSE_STOP" if adverse_stop else "TIMEOUT"
                ),
            }
        )
        state["position"] = None


def _apply_funding(
    states: dict[str, dict[str, Any]],
    funding: dict[str, Any],
    row: dict[str, Any],
    now: datetime,
) -> None:
    rate = _decimal(row.get("rate"))
    next_value = row.get("next_funding_at")
    next_at = _at(next_value) if next_value else None
    previous_next = funding.get("next")
    previous_rate = funding.get("rate")
    if (
        isinstance(previous_next, datetime)
        and isinstance(previous_rate, Decimal)
        and now >= previous_next
        and (next_at is None or next_at > previous_next)
    ):
        for state in states.values():
            position = state.get("position")
            if not position:
                continue
            direction = Decimal("1") if position["side"] == "long" else Decimal("-1")
            position["funding_bps"] -= direction * previous_rate * TEN_THOUSAND
    funding["rate"] = rate
    funding["next"] = next_at


def _report(states: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = [_at(row["at"]) for row in rows]
    days = len({at.date() for at in coverage})
    profiles: dict[str, Any] = {}
    for profile, state in sorted(states.items()):
        trades = state["closed"]
        net = [float(trade["net_bps"]) for trade in trades]
        level = int(profile.removeprefix("VIP"))
        maker, taker = futures_fee_bps(level)
        stressed = [value - (maker + taker) for value in net]
        wins = sum(value for value in net if value > 0)
        losses = -sum(value for value in net if value < 0)
        cumulative = peak = drawdown = 0.0
        for value in net:
            cumulative += value
            peak = max(peak, cumulative)
            drawdown = max(drawdown, peak - cumulative)
        profiles[profile] = {
            "pair_cycles": state["pair_cycles"],
            "expired_pairs": state["expired_pairs"],
            "post_only_rejections": state["post_only_rejections"],
            "proxy_fills": state["proxy_fills"],
            "closed_trades": len(trades),
            "trades_per_observed_day": len(trades) / days if days else 0.0,
            "expectancy_net_bps": sum(net) / len(net) if net else None,
            "expectancy_2x_cost_bps": sum(stressed) / len(stressed) if stressed else None,
            "net_pnl_usdt_at_10000_notional": sum(net),
            "profit_factor": wins / losses if losses else None,
            "max_drawdown_usdt_at_10000_notional": drawdown,
            "positive_fraction": sum(value > 0 for value in net) / len(net) if net else None,
            "target_fraction": (
                sum(trade["reason"] == "VWAP_TARGET" for trade in trades) / len(trades)
                if trades
                else None
            ),
            "research_gate_10_days_100_trades": days >= 10 and len(trades) >= 100,
            "trades": trades,
        }
    enough = any(value["research_gate_10_days_100_trades"] for value in profiles.values())
    return {
        "status": "PUBLIC_FILL_PROXY_RESEARCH_ONLY",
        "protocol_hash": ECONOMIC_QUOTE_PROTOCOL_HASH,
        "observed_days": days,
        "start": coverage[0].isoformat() if coverage else None,
        "end": coverage[-1].isoformat() if coverage else None,
        "events": len(rows),
        "profiles": profiles,
        "analysis_authorized": enough,
        "paper_change_authorized": False,
        "private_fill_truth": False,
        "warning": (
            "Public trades and visible queue are conservative proxy labels, "
            "not Bitunix account fills."
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _at(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _decimal(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
