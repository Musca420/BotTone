from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.binance_l2_dataset import load_records

REPORT = Path("data/reports/musca_vwap_liquidity_shadow.json")
FILTERED_REPORT = Path("data/reports/musca_vwap_liquidity_filtered_shadow.json")
ORDER_QUANTITY = 0.001
MAX_INVENTORY = 0.002
MAKER_FEE_BPS = 2.0
QUOTE_LIFETIME_SECONDS = 10
FILTER_START_SECOND = 1_785_997_700
FILTER_PROTOCOL = {
    "name": "signed_flow_depth_microprice_majority_v1",
    "start_second": FILTER_START_SECOND,
    "inventory_reduction_override": True,
}
FILTER_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(FILTER_PROTOCOL, sort_keys=True).encode()
).hexdigest()


def _queue_fill(
    queue_ahead: float, remaining: float, traded_quantity: float
) -> tuple[float, float, float]:
    consumed = min(queue_ahead, traded_quantity)
    queue_ahead -= consumed
    excess = traded_quantity - consumed
    filled = min(remaining, excess) if excess > 0 else 0.0
    if filled < 1e-12:
        filled = 0.0
    remaining = max(0.0, remaining - filled)
    return queue_ahead, remaining, filled


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def simulate(records: pd.DataFrame, *, filter_toxicity: bool = False) -> dict[str, Any]:
    cash, inventory = 10_000.0, 0.0
    orders: dict[str, dict[str, float] | None] = {"BUY": None, "SELL": None}
    fills: list[dict[str, Any]] = []
    placed = canceled = 0
    equity_curve: list[float] = []
    telemetry: list[dict[str, Any]] = []
    mids: dict[int, float] = {}
    recent_flow: deque[tuple[float, float]] = deque(maxlen=5)
    cumulative_price_volume = cumulative_volume = 0.0
    for row in records.sort_values("exchange_second").to_dict("records"):
        second = int(row["exchange_second"])
        bid, bid_size = float(row["bids"][0][0]), float(row["bids"][0][1])
        ask, ask_size = float(row["asks"][0][0]), float(row["asks"][0][1])
        mid = (bid + ask) / 2
        mids[second] = mid
        buy_quote = float(row["buy_quote"])
        sell_quote = float(row["sell_quote"])
        recent_flow.append((buy_quote - sell_quote, buy_quote + sell_quote))
        flow_total = sum(value[1] for value in recent_flow)
        flow_imbalance = sum(value[0] for value in recent_flow) / flow_total if flow_total else 0.0
        bid_depth = sum(float(value[1]) for value in row["bids"][:5])
        ask_depth = sum(float(value[1]) for value in row["asks"][:5])
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
        vote = sum(
            1 if value > 0 else -1 if value < 0 else 0
            for value in (flow_imbalance, depth_imbalance, microprice - mid)
        )
        for price_text, quantity_text, aggressor in row.get("aggregate_trades", []):
            price, quantity = float(price_text), float(quantity_text)
            cumulative_price_volume += price * quantity
            cumulative_volume += quantity
            side = "BUY" if aggressor == "SELL" else "SELL"
            order = orders[side]
            if order is None or abs(price - order["price"]) > 1e-9:
                continue
            queue, remaining, filled = _queue_fill(
                order["queue_ahead"], order["remaining"], quantity
            )
            order["queue_ahead"], order["remaining"] = queue, remaining
            if filled:
                signed = filled if side == "BUY" else -filled
                fee = filled * price * MAKER_FEE_BPS / 10_000
                cash -= signed * price + fee
                inventory += signed
                fills.append(
                    {
                        "exchange_timestamp": datetime.fromtimestamp(second, UTC).isoformat(),
                        "exchange_second": second,
                        "side": side.lower(),
                        "price": price,
                        "quantity": filled,
                        "fee": fee,
                        "commission": fee,
                        "slippage": 0.0,
                        "client_order_id": f"MM-{second}-{side}",
                    }
                )
            if remaining <= 1e-12:
                orders[side] = None

        for side, levels in (("BUY", row["bids"]), ("SELL", row["asks"])):
            order = orders[side]
            if order is None:
                continue
            level = next(
                (value for value in levels if abs(float(value[0]) - order["price"]) < 1e-9),
                None,
            )
            order["queue_ahead"] = min(
                order["queue_ahead"], float(level[1]) if level else 0.0
            )
            crossing = order["price"] >= ask if side == "BUY" else order["price"] <= bid
            expired = second - int(order["placed_second"]) >= QUOTE_LIFETIME_SECONDS
            if crossing or expired:
                orders[side] = None
                canceled += 1

        allow_buy = not filter_toxicity or vote >= 0 or inventory < 0
        allow_sell = not filter_toxicity or vote <= 0 or inventory > 0
        if orders["BUY"] is None and inventory < MAX_INVENTORY - 1e-12 and allow_buy:
            orders["BUY"] = {
                "price": bid,
                "remaining": ORDER_QUANTITY,
                "queue_ahead": bid_size,
                "placed_second": float(second),
            }
            placed += 1
        if orders["SELL"] is None and inventory > -MAX_INVENTORY + 1e-12 and allow_sell:
            orders["SELL"] = {
                "price": ask,
                "remaining": ORDER_QUANTITY,
                "queue_ahead": ask_size,
                "placed_second": float(second),
            }
            placed += 1
        equity_curve.append(cash + inventory * mid)
        current_vwap = (
            cumulative_price_volume / cumulative_volume if cumulative_volume else None
        )
        telemetry.append(
            {
                "timestamp": datetime.fromtimestamp(second, UTC).isoformat(),
                "activity": "PASSIVE_QUOTES" if any(orders.values()) else "INVENTORY_LIMIT",
                "close": mid,
                "center": current_vwap,
                "anchored_vwap": current_vwap,
                "atr": None,
                "entry_score": vote,
                "regime": "liquidity",
                "decision_reason": f"flow/depth/microprice vote {vote}",
            }
        )

    for fill in fills:
        direction = 1 if fill["side"] == "buy" else -1
        for horizon in (5, 30, 60):
            future = mids.get(fill["exchange_second"] + horizon)
            fill[f"markout_{horizon}s_bps"] = (
                None
                if future is None
                else direction * (future - fill["price"]) / fill["price"] * 10_000
            )
    peak, drawdown = equity_curve[0] if equity_curve else cash, 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    final_mid = next(reversed(mids.values()), 0.0)
    anchored_vwap = (
        cumulative_price_volume / cumulative_volume if cumulative_volume else None
    )
    final_equity = cash + inventory * final_mid
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 1,
        "strategy_profile": (
            "musca_vwap_liquidity_filtered"
            if filter_toxicity
            else "musca_vwap_liquidity_baseline"
        ),
        "validation_status": "RESEARCH_ONLY",
        "live_orders_enabled": False,
        "source": "Binance routed USD-M L2 depth20 + aggTrade",
        "records": len(records),
        "orders_placed": placed,
        "orders_canceled": canceled,
        "signals": placed,
        "rejected_signals": canceled,
        "fills": fills,
        "fill_count": len(fills),
        "inventory_btc": inventory,
        "initial_equity": 10_000.0,
        "final_equity": final_equity,
        "net_pnl": final_equity - 10_000.0,
        "gross_pnl": final_equity - 10_000.0 + sum(fill["fee"] for fill in fills),
        "fees": sum(fill["fee"] for fill in fills),
        "slippage": 0.0,
        "max_drawdown": drawdown,
        "risk_per_trade": 0.01,
        "max_leverage": 10,
        "policy_action": "PASSIVE_QUOTE_OR_FLAT",
        "maker_fee_bps_assumption": MAKER_FEE_BPS,
        "queue_model": "visible_size_ahead_exact_depletion_no_fill",
        "toxicity_filter": FILTER_PROTOCOL if filter_toxicity else None,
        "protocol_hash": FILTER_PROTOCOL_HASH if filter_toxicity else None,
        "collector_anchored_vwap": anchored_vwap,
        "distance_from_collector_vwap_bps": (
            (final_mid - anchored_vwap) / anchored_vwap * 10_000
            if anchored_vwap
            else None
        ),
        "equity_curve": [
            {"timestamp": point["timestamp"], "equity": equity}
            for point, equity in zip(telemetry, equity_curve, strict=True)
        ],
        "telemetry": telemetry[-600:],
        "updated_at": datetime.now(UTC).isoformat(),
    }


def run() -> dict[str, Any]:
    records = load_records()
    records = records.loc[records["schema_version"].eq(3)] if not records.empty else records
    baseline = simulate(records)
    future = records.loc[records["exchange_second"].ge(FILTER_START_SECOND)]
    same_period_baseline = simulate(future)
    filtered = simulate(future, filter_toxicity=True)
    filtered["same_period_baseline"] = {
        key: same_period_baseline[key]
        for key in ("records", "fill_count", "net_pnl", "gross_pnl", "fees")
    }
    _write(REPORT, baseline)
    _write(FILTERED_REPORT, filtered)
    return {"baseline": baseline, "filtered_future": filtered}


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca VWAP liquidity shadow")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        print(json.dumps(run(), indent=2))
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
