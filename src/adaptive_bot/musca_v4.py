from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from adaptive_bot.musca_v2 import _atomic_json
from adaptive_bot.musca_v4_research import (
    BUNDLE,
    FEATURES,
    PROTOCOL_HASH,
    build_events,
    build_features,
)
from adaptive_bot.services.bitunix_paper_service import read_collected_candles

INPUT = Path("data/raw/bitunix_btcusdt_last_futures_5m.jsonl")
REPORT = Path("data/reports/musca_v4_shadow.json")
STATE = Path("data/research/musca_v4_shadow_state.json")
V2_STATE = Path("data/research/musca_v2_shadow_state.json")
RISK = Decimal("0.01")
MAX_LEVERAGE = Decimal("10")
COST_BPS = Decimal("8")
BREAKOUT_BARS = 3
REQUIRE_RESTART_SPOT = True
STRATEGY_PROFILE = "musca_v4_multi_anchor_vwap"
CLIENT_PREFIX = "musca-v4"
SOURCE_NAME = "musca_v4_binance_alpha_bitunix_shadow"
STRATEGY_PROTOCOL_HASH = PROTOCOL_HASH


def _get_json(url: str) -> list[list[Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "musca-v4-shadow/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Binance kline response must be a list")
    return cast(list[list[Any]], payload)


def _market_frame(get_json: Any = _get_json) -> pd.DataFrame:
    endpoints = {
        "perp": "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=1000",
        "spot": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000",
    }
    frames: list[pd.DataFrame] = []
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    for prefix, url in endpoints.items():
        rows = [row for row in get_json(url) if int(row[6]) < now_ms]
        frame = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_buy_volume",
                "taker_buy_quote",
                "ignore",
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.drop(columns=["close_time", "ignore"]).rename(
            columns={column: f"{prefix}_{column}" for column in frame.columns[1:]}
        )
        frames.append(frame)
    data = frames[0].merge(frames[1], on="timestamp", validate="one_to_one")
    for column in data.columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["data_valid"] = data.notna().all(axis=1)
    return data.loc[data["data_valid"]].reset_index(drop=True)


def _initial_state(baseline: pd.Timestamp) -> dict[str, Any]:
    timestamp = baseline.isoformat()
    return {
        "baseline": timestamp,
        "last_processed": timestamp,
        "last_signal_at": timestamp,
        "equity": "10000",
        "peak_equity": "10000",
        "max_drawdown": "0",
        "fees": "0",
        "position": None,
        "pending": None,
        "signals": 0,
        "rejected_signals": 0,
        "fills": [],
        "closed_trades": [],
        "equity_curve": [{"timestamp": timestamp, "equity": "10000"}],
        "last_model_ev_bps": None,
        "last_model_win_probability": None,
    }


def _load_state(latest: pd.Timestamp) -> dict[str, Any]:
    if STATE.exists():
        return cast(dict[str, Any], json.loads(STATE.read_text(encoding="utf-8")))
    baseline = latest
    if V2_STATE.exists():
        baseline = pd.Timestamp(json.loads(V2_STATE.read_text(encoding="utf-8"))["baseline"])
    state = _initial_state(baseline)
    _atomic_json(STATE, state)
    return state


def _fill(
    timestamp: pd.Timestamp, side: str, price: Decimal, quantity: Decimal, sequence: int
) -> dict[str, str]:
    return {
        "exchange_timestamp": timestamp.isoformat(),
        "received_timestamp": timestamp.isoformat(),
        "source": SOURCE_NAME,
        "instrument": "BTCUSDT",
        "client_order_id": f"{CLIENT_PREFIX}-{timestamp.value}-{sequence}",
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "commission": "0",
        "slippage": "0",
        "liquidity_role": "simulated_conservative",
    }


def _book_exit(
    state: dict[str, Any], timestamp: pd.Timestamp, price: Decimal, quantity: Decimal, reason: str
) -> None:
    position = state["position"]
    direction = Decimal(position["direction"])
    entry = Decimal(position["entry"])
    gross = direction * (price - entry) * quantity
    cost = entry * quantity * COST_BPS / Decimal(10_000)
    state["equity"] = str(Decimal(state["equity"]) + gross - cost)
    state["fees"] = str(Decimal(state["fees"]) + cost)
    state["fills"].append(
        _fill(
            timestamp,
            "sell" if direction > 0 else "buy",
            price,
            quantity,
            len(state["fills"]),
        )
    )
    position["quantity"] = str(Decimal(position["quantity"]) - quantity)
    position["realized_pnl"] = str(Decimal(position["realized_pnl"]) + gross - cost)
    if Decimal(position["quantity"]) == 0:
        peak = max(Decimal(state["peak_equity"]), Decimal(state["equity"]))
        state["peak_equity"] = str(peak)
        state["max_drawdown"] = str(
            max(Decimal(state["max_drawdown"]), (peak - Decimal(state["equity"])) / peak)
        )
        state["closed_trades"].append(
            {
                "entry_timestamp": position["opened_at"],
                "exit_timestamp": timestamp.isoformat(),
                "net_pnl": position["realized_pnl"],
                "reason": reason,
            }
        )
        state["equity_curve"].append(
            {"timestamp": timestamp.isoformat(), "equity": state["equity"]}
        )
        state["position"] = None


def _position_quantity(equity: Decimal, entry: Decimal, stop: Decimal) -> Decimal:
    return min(equity * RISK / abs(entry - stop), equity * MAX_LEVERAGE / entry)


def _execution_avwap(
    candles: pd.DataFrame, anchor_at: str | pd.Timestamp, through: pd.Timestamp
) -> Decimal:
    anchor = pd.Timestamp(anchor_at)
    timestamps = pd.to_datetime(candles["timestamp"], utc=True)
    if timestamps.empty or anchor < timestamps.min():
        raise ValueError("Execution AVWAP anchor predates observed Bitunix history")
    selected = candles.loc[timestamps.ge(anchor) & timestamps.le(through)]
    volume = Decimal(str(selected["volume"].sum()))
    if volume <= 0:
        raise ValueError("Execution AVWAP requires positive observed Bitunix volume")
    return Decimal(str(selected["quote_volume"].sum())) / volume


def _score(event: pd.Series) -> tuple[float | None, float | None]:
    if not BUNDLE.exists():
        return None, None
    bundle = joblib.load(BUNDLE)
    row = pd.DataFrame([{feature: event.get(feature) for feature in FEATURES}])
    return (
        float(bundle["ev_model"].predict(row)[0]),
        float(bundle["win_model"].predict_proba(row)[0, 1]),
    )


def _advance(
    candles: pd.DataFrame, alpha: pd.DataFrame
) -> tuple[dict[str, Any], pd.Series, pd.Series | None]:
    if not candles["price_type"].eq("LAST_PRICE").all():
        raise ValueError("Musca V4 fails closed unless every Bitunix candle is LAST_PRICE")
    market = build_features(alpha)
    events = build_events(
        alpha,
        breakout_bars=BREAKOUT_BARS,
        require_restart_spot=REQUIRE_RESTART_SPOT,
    )
    events = events.loc[events["event_family"].eq("IMPULSE_PULLBACK")].sort_values("available_at")
    latest = pd.Timestamp(candles.iloc[-1]["timestamp"])
    state = _load_state(latest)
    last_processed = pd.Timestamp(state["last_processed"])
    new_rows = candles.loc[pd.to_datetime(candles["timestamp"], utc=True).gt(last_processed)]
    for raw_index, row in new_rows.iterrows():
        index = cast(int, raw_index)
        timestamp = pd.Timestamp(row["timestamp"])
        available = events.loc[
            pd.to_datetime(events["available_at"], utc=True).le(timestamp)
            & pd.to_datetime(events["available_at"], utc=True).gt(
                pd.Timestamp(state["last_signal_at"])
            )
        ]
        if state["position"] is None and state["pending"] is None and not available.empty:
            event = available.iloc[-1]
            state["last_signal_at"] = pd.Timestamp(event["available_at"]).isoformat()
            state["signals"] += 1
            ev, probability = _score(event)
            state["last_model_ev_bps"], state["last_model_win_probability"] = ev, probability
            signal_close = Decimal(str(alpha.iloc[int(event["signal_index"])]["perp_close"]))
            stop = Decimal(str(event["stop_price"]))
            state["pending"] = {
                "direction": int(event["direction"]),
                "risk_fraction": str(abs(signal_close - stop) / signal_close),
                "anchor_at": pd.Timestamp(event["impulse_anchor_at"]).isoformat(),
                "operating_vwap": str(event["operating_vwap"]),
                "operating_sigma": str(event["operating_sigma"]),
                "signal_at": pd.Timestamp(event["available_at"]).isoformat(),
            }
        if state["pending"] is not None and state["position"] is None:
            pending = state["pending"]
            state["pending"] = None
            direction = int(pending["direction"])
            entry = Decimal(str(row["open"]))
            stop = entry * (Decimal(1) - Decimal(direction) * Decimal(pending["risk_fraction"]))
            quantity = _position_quantity(Decimal(state["equity"]), entry, stop)
            state["position"] = {
                "direction": direction,
                "entry": str(entry),
                "stop": str(stop),
                "target": str(entry + Decimal(direction) * Decimal("1.5") * abs(entry - stop)),
                "quantity": str(quantity),
                "initial_quantity": str(quantity),
                "opened_at": timestamp.isoformat(),
                "bars_open": 0,
                "tp1": False,
                "opposite_closes": 0,
                "anchor_at": pending["anchor_at"],
                "operating_vwap": pending["operating_vwap"],
                "operating_sigma": pending["operating_sigma"],
                "realized_pnl": "0",
            }
            state["fills"].append(
                _fill(
                    timestamp,
                    "buy" if direction > 0 else "sell",
                    entry,
                    quantity,
                    len(state["fills"]),
                )
            )
        position = state["position"]
        if position is not None:
            direction = int(position["direction"])
            stop, target = Decimal(position["stop"]), Decimal(position["target"])
            high, low, open_price = (
                Decimal(str(row["high"])),
                Decimal(str(row["low"])),
                Decimal(str(row["open"])),
            )
            stopped = low <= stop if direction > 0 else high >= stop
            target_hit = high >= target if direction > 0 else low <= target
            if stopped:
                exit_price = min(open_price, stop) if direction > 0 else max(open_price, stop)
                _book_exit(
                    state, timestamp, exit_price, Decimal(position["quantity"]), "STRUCTURAL_STOP"
                )
            elif target_hit and not position["tp1"]:
                half = Decimal(position["quantity"]) / 2
                _book_exit(state, timestamp, target, half, "TP1")
                if state["position"] is not None:
                    position["tp1"] = True
                    entry = Decimal(position["entry"])
                    protected = entry + Decimal(direction) * entry * COST_BPS / Decimal(10_000)
                    position["stop"] = str(
                        max(stop, protected) if direction > 0 else min(stop, protected)
                    )
            if state["position"] is not None:
                position = state["position"]
                position["bars_open"] += 1
                operating = _execution_avwap(candles, position["anchor_at"], timestamp)
                position["current_anchored_vwap"] = str(operating)
                wrong_side = (Decimal(str(row["close"])) - operating) * Decimal(direction) < 0
                position["opposite_closes"] = position["opposite_closes"] + 1 if wrong_side else 0
                if position["tp1"] and index >= 3:
                    completed = candles.iloc[max(0, index - 3) : index]
                    proposal = Decimal(
                        str(completed["low"].min() if direction > 0 else completed["high"].max())
                    )
                    current_stop = Decimal(position["stop"])
                    position["stop"] = str(
                        max(current_stop, proposal)
                        if direction > 0
                        else min(current_stop, proposal)
                    )
                reason = (
                    "AVWAP_ACCEPTANCE_FAILURE"
                    if position["opposite_closes"] >= 3
                    else "TIMEOUT_6H"
                    if position["bars_open"] >= 72
                    else None
                )
                if reason:
                    _book_exit(
                        state,
                        timestamp,
                        Decimal(str(row["close"])),
                        Decimal(position["quantity"]),
                        reason,
                    )
        state["last_processed"] = timestamp.isoformat()
    _atomic_json(STATE, state)
    latest_event = events.iloc[-1] if not events.empty else None
    return state, market.iloc[-1], latest_event


def make_report(
    state: dict[str, Any],
    row: pd.Series,
    event: pd.Series | None,
    candles: pd.DataFrame,
) -> dict[str, Any]:
    position = state["position"]
    anchor_at = (
        position["anchor_at"]
        if position
        else None
        if event is None
        else pd.Timestamp(event["impulse_anchor_at"]).isoformat()
    )
    anchored = (
        None
        if anchor_at is None
        else float(
            _execution_avwap(
                candles, anchor_at, pd.Timestamp(candles.iloc[-1]["timestamp"])
            )
        )
    )
    alpha_anchored = None if event is None else float(event["operating_vwap"])
    activity = (
        "POSITION_OPEN"
        if position
        else "ORDER_PENDING"
        if state["pending"]
        else "WATCHING_IMPULSE_PULLBACK"
    )
    equity = Decimal(state["equity"])
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 5,
        "strategy_profile": STRATEGY_PROFILE,
        "benchmark_venue": "Binance USD-M BTCUSDT",
        "execution_venue": "Bitunix BTCUSDT",
        "vwap_session": "UTC day (00:00 reset)",
        "validation_status": "RESEARCH_ONLY_FUTURE_SHADOW_REQUIRED",
        "policy_action": activity,
        "live_orders_enabled": False,
        "initial_equity": "10000",
        "final_equity": str(equity),
        "gross_pnl": str(equity - Decimal("10000") + Decimal(state["fees"])),
        "net_pnl": str(equity - Decimal("10000")),
        "fees": state["fees"],
        "slippage": "0",
        "max_drawdown": state["max_drawdown"],
        "risk_per_trade": str(RISK),
        "max_leverage": str(MAX_LEVERAGE),
        "signals": state["signals"],
        "rejected_signals": state["rejected_signals"],
        "kill_switches": 0,
        "fills": state["fills"],
        "equity_curve": state["equity_curve"],
        "telemetry": [
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "activity": activity,
                "close": float(row["perp_close"]),
                "center": float(row["daily_vwap"]),
                "anchored_vwap": anchored,
                "alpha_anchored_vwap": alpha_anchored,
                "anchor_at": anchor_at,
                "atr": float(row["atr"]),
                "entry_score": state["last_model_ev_bps"],
                "win_probability": state["last_model_win_probability"],
                "regime": "trend-up"
                if int(row["direction"]) > 0
                else "trend-down"
                if int(row["direction"]) < 0
                else "neutral",
                "decision_reason": activity,
            }
        ],
        "protocol_hash": STRATEGY_PROTOCOL_HASH,
        "shadow_closed_trades": len(state["closed_trades"]),
        "shadow_position": position,
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def worker(input_path: Path, report_path: Path, *, once: bool) -> None:
    while True:
        try:
            candles = read_collected_candles(input_path)
            state, row, event = _advance(candles, _market_frame())
            _atomic_json(report_path, make_report(state, row, event, candles))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _atomic_json(
                report_path,
                {
                    "mode": "shadow",
                    "strategy_profile": STRATEGY_PROFILE,
                    "validation_status": "RESEARCH_ONLY_FAIL_CLOSED",
                    "policy_action": "DATA_UNAVAILABLE",
                    "detail": str(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        if once:
            return
        await asyncio.sleep(300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca V4 multi-anchor VWAP shadow")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset-from-v2", action="store_true")
    args = parser.parse_args()
    if args.reset_from_v2 and STATE.exists():
        STATE.unlink()
    asyncio.run(worker(args.input, args.report, once=args.once))


if __name__ == "__main__":
    main()
