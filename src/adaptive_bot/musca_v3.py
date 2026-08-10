from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd

from adaptive_bot.musca_v2 import INPUT, _atomic_json, _live_features
from adaptive_bot.services.bitunix_paper_service import read_collected_candles

REPORT = Path("data/reports/musca_v3_shadow.json")
STATE = Path("data/research/musca_v3_shadow_state.json")
V2_STATE = Path("data/research/musca_v2_shadow_state.json")
RISK = Decimal("0.01")
MAX_LEVERAGE = Decimal("10")
COST_BPS = Decimal("8")
MAX_BARS = 96
ARMED_MAX_BARS = 6
ANCHOR_MAX_DISTANCE_ATR = 0.5
ANCHOR_MIN_SLOPE_15M = 0.02
ANCHOR_MIN_SLOPE_1H = 0.05


def _features(candles: pd.DataFrame) -> pd.DataFrame:
    data = _live_features(candles)
    data["return_15m"] = data["perp_close"].pct_change(3)
    data["return_1h"] = data["perp_close"].pct_change(12)
    data["local_low"] = (
        data["perp_low"].shift(1).rolling(12, min_periods=12).min() - 0.1 * data["atr"]
    )
    data["local_high"] = (
        data["perp_high"].shift(1).rolling(12, min_periods=12).max() + 0.1 * data["atr"]
    )
    return data


def _tighten_stop(direction: int, stop: Decimal, extreme: Decimal, atr_value: Decimal) -> Decimal:
    proposal = extreme - Decimal(direction) * Decimal("2") * atr_value
    return max(stop, proposal) if direction > 0 else min(stop, proposal)


def _protect_costs(direction: int, stop: Decimal, entry: Decimal) -> Decimal:
    break_even = entry + Decimal(direction) * entry * COST_BPS / Decimal(10_000)
    return max(stop, break_even) if direction > 0 else min(stop, break_even)


def _position_quantity(equity: Decimal, entry: Decimal, stop: Decimal) -> Decimal:
    risk_sized = equity * RISK / abs(entry - stop)
    leverage_capped = equity * MAX_LEVERAGE / entry
    return min(risk_sized, leverage_capped)


def _close_exit_reason(
    row: pd.Series, previous: pd.Series, direction: int, bars_open: int
) -> str | None:
    local_reversal = (
        float(row["return_15m"]) * direction < 0 and float(row["return_1h"]) * direction < 0
    )
    vwap_failed = (
        float(row["perp_close"] - row["daily_vwap"]) * direction < 0
        and float(previous["perp_close"] - previous["daily_vwap"]) * direction < 0
    )
    if vwap_failed:
        return "VWAP_FAILURE"
    if local_reversal:
        return "LOCAL_TREND_REVERSAL"
    return "TIMEOUT_8H" if bars_open >= MAX_BARS else None


def _continuation_signal(
    row: pd.Series,
    previous: pd.Series,
    direction: int,
    anchor: float,
    armed: bool,
    anchor_slope_15m: float | None,
    anchor_slope_1h: float | None,
) -> tuple[bool, bool]:
    atr_value = float(row["atr"])
    touched = (
        float(row["perp_low"]) <= anchor + 0.25 * atr_value
        if direction > 0
        else float(row["perp_high"]) >= anchor - 0.25 * atr_value
    )
    armed = armed or touched
    correct_daily_side = float(row["perp_close"] - row["daily_vwap"]) * direction > 0
    correct_anchor_side = float(row["perp_close"] - anchor) * direction > 0
    near_anchor = abs(float(row["perp_close"] - anchor)) <= ANCHOR_MAX_DISTANCE_ATR * atr_value
    local_trend = (
        float(row["return_15m"]) * direction > 0 and float(row["return_1h"]) * direction > 0
    )
    price_confirmation = (
        float(row["perp_close"]) > float(previous["perp_high"])
        if direction > 0
        else float(row["perp_close"]) < float(previous["perp_low"])
    )
    anchor_trend = (
        anchor_slope_15m is not None
        and anchor_slope_1h is not None
        and anchor_slope_15m * direction >= ANCHOR_MIN_SLOPE_15M
        and anchor_slope_1h * direction >= ANCHOR_MIN_SLOPE_1H
    )
    return armed, bool(
        armed
        and correct_daily_side
        and correct_anchor_side
        and near_anchor
        and local_trend
        and price_confirmation
        and anchor_trend
    )


def _daily_signal(row: pd.Series, previous: pd.Series) -> bool:
    direction = int(row["direction"])
    price_confirmation = (
        float(row["perp_close"]) > float(previous["perp_high"])
        if direction > 0
        else float(row["perp_close"]) < float(previous["perp_low"])
    )
    return bool(
        row["reclaim"]
        and float(row["return_15m"]) * direction > 0
        and float(row["return_1h"]) * direction > 0
        and price_confirmation
    )


def _initial_state(baseline: pd.Timestamp) -> dict[str, Any]:
    timestamp = baseline.isoformat()
    return {
        "baseline": timestamp,
        "last_processed": timestamp,
        "equity": "10000",
        "position": None,
        "pending_entry": None,
        "pending_exit": None,
        "fills": [],
        "closed_trades": [],
        "signals": 0,
        "fees": "0",
        "peak_equity": "10000",
        "max_drawdown": "0",
        "continuation_anchor_at": None,
        "continuation_direction": None,
        "continuation_armed": False,
        "continuation_touch_at": None,
        "continuation_armed_bars": 0,
        "equity_curve": [{"timestamp": timestamp, "equity": "10000"}],
    }


def _load_state(latest: pd.Timestamp) -> dict[str, Any]:
    if STATE.exists():
        loaded = cast(dict[str, Any], json.loads(STATE.read_text(encoding="utf-8")))
        loaded.setdefault("continuation_anchor_at", None)
        loaded.setdefault("continuation_direction", None)
        loaded.setdefault("continuation_armed", False)
        loaded.setdefault("continuation_touch_at", None)
        loaded.setdefault("continuation_armed_bars", 0)
        return loaded
    state = _initial_state(latest)
    _atomic_json(STATE, state)
    return state


def _fill(
    timestamp: pd.Timestamp, side: str, price: Decimal, quantity: Decimal, sequence: int
) -> dict[str, str]:
    return {
        "exchange_timestamp": timestamp.isoformat(),
        "received_timestamp": timestamp.isoformat(),
        "source": "musca_v3_intraday_shadow",
        "instrument": "BTCUSDT",
        "client_order_id": f"musca-v3-{timestamp.value}-{sequence}",
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "commission": "0",
        "slippage": "0",
        "liquidity_role": "simulated",
    }


def _close_position(
    state: dict[str, Any], timestamp: pd.Timestamp, exit_price: Decimal, reason: str
) -> None:
    position = state["position"]
    direction = int(position["direction"])
    entry = Decimal(position["entry"])
    quantity = Decimal(position["quantity"])
    gross = Decimal(direction) * (exit_price - entry) * quantity
    cost = entry * quantity * COST_BPS / Decimal(10_000)
    state["equity"] = str(Decimal(state["equity"]) + gross - cost)
    state["fees"] = str(Decimal(state["fees"]) + cost)
    peak = max(Decimal(state["peak_equity"]), Decimal(state["equity"]))
    state["peak_equity"] = str(peak)
    state["max_drawdown"] = str(
        max(Decimal(state["max_drawdown"]), (peak - Decimal(state["equity"])) / peak)
    )
    state["equity_curve"].append({"timestamp": timestamp.isoformat(), "equity": state["equity"]})
    state["fills"].append(
        _fill(
            timestamp,
            "sell" if direction > 0 else "buy",
            exit_price,
            quantity,
            len(state["fills"]),
        )
    )
    state["closed_trades"].append(
        {
            "entry_timestamp": position["opened_at"],
            "exit_timestamp": timestamp.isoformat(),
            "net_pnl": str(gross - cost),
            "reason": reason,
        }
    )
    state["continuation_anchor_at"] = position["anchor_at"]
    state["continuation_direction"] = direction
    state["continuation_armed"] = False
    state["continuation_touch_at"] = None
    state["continuation_armed_bars"] = 0
    state["position"] = None
    state["pending_exit"] = None


def _anchored_vwap(data: pd.DataFrame, start: pd.Timestamp, end_index: int) -> float:
    window = data.loc[
        pd.to_datetime(data["timestamp"], utc=True).ge(start) & data.index.to_series().le(end_index)
    ]
    return float(window["perp_quote_volume"].sum() / window["perp_volume"].sum())


def _anchored_slopes(
    data: pd.DataFrame, start: pd.Timestamp, end_index: int, atr_value: float
) -> tuple[float, float] | None:
    if (
        end_index < 12
        or atr_value <= 0
        or pd.Timestamp(data.iloc[end_index - 12]["timestamp"]) < start
    ):
        return None
    current = _anchored_vwap(data, start, end_index)
    return (
        (current - _anchored_vwap(data, start, end_index - 3)) / atr_value,
        (current - _anchored_vwap(data, start, end_index - 12)) / atr_value,
    )


def _advance(candles: pd.DataFrame) -> tuple[dict[str, Any], pd.Series, float | None]:
    data = _features(candles)
    latest = pd.Timestamp(data.iloc[-1]["timestamp"])
    state = _load_state(latest)
    if state["continuation_anchor_at"] is None and state["closed_trades"]:
        past = data.loc[
            pd.to_datetime(data["timestamp"], utc=True).le(pd.Timestamp(state["last_processed"]))
            & data["reclaim"]
        ]
        if not past.empty:
            state["continuation_anchor_at"] = pd.Timestamp(past.iloc[-1]["timestamp"]).isoformat()
            state["continuation_direction"] = int(past.iloc[-1]["direction"])
    last = pd.Timestamp(state["last_processed"])
    new_indices = data.index[pd.to_datetime(data["timestamp"], utc=True).gt(last)]
    anchor_value: float | None = None
    for raw_index in new_indices:
        index = int(raw_index)
        row = data.iloc[index]
        previous = data.iloc[index - 1]
        timestamp = pd.Timestamp(row["timestamp"])
        if state["position"] is not None and state["pending_exit"] is not None:
            _close_position(state, timestamp, Decimal(str(row["perp_open"])), state["pending_exit"])
        if state["pending_entry"] is not None and state["position"] is None:
            pending = state["pending_entry"]
            state["pending_entry"] = None
            entry = Decimal(str(row["perp_open"]))
            stop = Decimal(str(pending["stop"]))
            distance = Decimal(pending["direction"]) * (entry - stop)
            if distance > 0:
                quantity = _position_quantity(Decimal(state["equity"]), entry, stop)
                state["position"] = {
                    "direction": int(pending["direction"]),
                    "entry": str(entry),
                    "stop": str(stop),
                    "quantity": str(quantity),
                    "opened_at": timestamp.isoformat(),
                    "anchor_at": pending["anchor_at"],
                    "bars_open": 0,
                    "highest": str(entry),
                    "lowest": str(entry),
                }
                state["continuation_anchor_at"] = pending["anchor_at"]
                state["continuation_direction"] = int(pending["direction"])
                state["continuation_armed"] = False
                state["continuation_touch_at"] = None
                state["continuation_armed_bars"] = 0
                state["fills"].append(
                    _fill(
                        timestamp,
                        "buy" if int(pending["direction"]) > 0 else "sell",
                        entry,
                        quantity,
                        len(state["fills"]),
                    )
                )
        position = state["position"]
        if position is not None:
            direction = int(position["direction"])
            stop = Decimal(position["stop"])
            position["bars_open"] += 1
            position["highest"] = str(
                max(Decimal(position["highest"]), Decimal(str(row["perp_high"])))
            )
            position["lowest"] = str(
                min(Decimal(position["lowest"]), Decimal(str(row["perp_low"])))
            )
            stopped = (
                Decimal(str(row["perp_low"])) <= stop
                if direction > 0
                else Decimal(str(row["perp_high"])) >= stop
            )
            if stopped:
                market = Decimal(str(row["perp_open"]))
                exit_price = min(market, stop) if direction > 0 else max(market, stop)
                _close_position(state, timestamp, exit_price, "STOP")
            else:
                entry = Decimal(position["entry"])
                atr_value = Decimal(str(row["atr"]))
                extreme = Decimal(position["highest"] if direction > 0 else position["lowest"])
                if Decimal(direction) * (extreme - entry) >= atr_value:
                    position["stop"] = str(
                        _protect_costs(
                            direction,
                            _tighten_stop(direction, stop, extreme, atr_value),
                            entry,
                        )
                    )
                state["pending_exit"] = _close_exit_reason(
                    row, previous, direction, int(position["bars_open"])
                )
                anchor_value = _anchored_vwap(data, pd.Timestamp(position["anchor_at"]), index)
        if (
            state["position"] is None
            and state["pending_entry"] is None
            and bool(row["reclaim"])
        ):
            direction = int(row["direction"])
            state["continuation_anchor_at"] = timestamp.isoformat()
            state["continuation_direction"] = direction
            state["continuation_armed"] = False
            state["continuation_touch_at"] = None
            state["continuation_armed_bars"] = 0
            stop = row["local_low"] if direction > 0 else row["local_high"]
            if _daily_signal(row, previous) and pd.notna(stop):
                state["pending_entry"] = {
                    "direction": direction,
                    "stop": str(float(stop)),
                    "anchor_at": timestamp.isoformat(),
                    "setup": "DAILY_VWAP_RECLAIM",
                }
                state["signals"] += 1
        if (
            state["position"] is None
            and state["pending_entry"] is None
            and state["continuation_anchor_at"] is not None
            and int(row["direction"]) == int(state["continuation_direction"])
        ):
            direction = int(state["continuation_direction"])
            anchor = _anchored_vwap(data, pd.Timestamp(state["continuation_anchor_at"]), index)
            slopes = _anchored_slopes(
                data,
                pd.Timestamp(state["continuation_anchor_at"]),
                index,
                float(row["atr"]),
            )
            was_armed = bool(state["continuation_armed"])
            armed, signal = _continuation_signal(
                row,
                previous,
                direction,
                anchor,
                was_armed,
                *(slopes or (None, None)),
            )
            state["continuation_armed"] = armed
            if armed and not was_armed:
                state["continuation_touch_at"] = timestamp.isoformat()
                state["continuation_armed_bars"] = 0
            elif armed:
                state["continuation_armed_bars"] += 1
            if state["continuation_armed_bars"] > ARMED_MAX_BARS:
                armed = signal = False
                state["continuation_armed"] = False
                state["continuation_armed_bars"] = 0
                state["continuation_touch_at"] = None
            stop = row["local_low"] if direction > 0 else row["local_high"]
            if signal and pd.notna(stop):
                state["pending_entry"] = {
                    "direction": direction,
                    "stop": str(float(stop)),
                    "anchor_at": state["continuation_touch_at"] or timestamp.isoformat(),
                    "setup": "ANCHORED_VWAP_CONTINUATION",
                }
                state["continuation_armed"] = False
                state["continuation_touch_at"] = None
                state["continuation_armed_bars"] = 0
                state["signals"] += 1
        state["last_processed"] = timestamp.isoformat()
    if state["position"] is not None and anchor_value is None:
        anchor_value = _anchored_vwap(
            data, pd.Timestamp(state["position"]["anchor_at"]), int(data.index[-1])
        )
    elif state["position"] is None and state["continuation_anchor_at"] is not None:
        anchor_value = _anchored_vwap(
            data, pd.Timestamp(state["continuation_anchor_at"]), int(data.index[-1])
        )
    _atomic_json(STATE, state)
    return state, data.iloc[-1], anchor_value


def make_report(
    state: dict[str, Any], row: pd.Series, anchor_value: float | None
) -> dict[str, Any]:
    anchor_at = (
        state["position"]["anchor_at"]
        if state["position"]
        else state["pending_entry"]["anchor_at"]
        if state["pending_entry"]
        else state["continuation_anchor_at"]
    )
    activity = (
        "EXIT_PENDING"
        if state["pending_exit"] is not None
        else "POSITION_OPEN"
        if state["position"] is not None
        else "ORDER_PENDING"
        if state["pending_entry"] is not None
        else "WATCHING_DAILY_OR_ANCHORED_VWAP"
    )
    equity = Decimal(state["equity"])
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 5,
        "strategy_profile": "musca_v3_intraday_vwap",
        "validation_status": "RESEARCH_ONLY_UNAUDITED_INTRADAY_VARIANT",
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
        "rejected_signals": 0,
        "kill_switches": 0,
        "fills": state["fills"],
        "equity_curve": state["equity_curve"],
        "telemetry": [
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "activity": activity,
                "close": float(row["perp_close"]),
                "center": float(row["daily_vwap"]),
                "anchored_vwap": anchor_value,
                "anchor_at": anchor_at,
                "atr": float(row["atr"]),
                "entry_score": float(row["trend_strength"]) * 10_000,
                "regime": "trend-up" if int(row["direction"]) > 0 else "trend-down",
                "decision_reason": state["pending_exit"]
                or (
                    "intraday position management"
                    if state["position"]
                    else state["pending_entry"]["setup"]
                    if state["pending_entry"]
                    else "waiting for daily or anchored VWAP setup"
                ),
            }
        ],
        "shadow_closed_trades": len(state["closed_trades"]),
        "shadow_position": state["position"],
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def worker(input_path: Path, report_path: Path, *, once: bool) -> None:
    while True:
        state, row, anchor_value = _advance(read_collected_candles(input_path))
        _atomic_json(report_path, make_report(state, row, anchor_value))
        if once:
            return
        await asyncio.sleep(300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca v3 intraday VWAP research shadow")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--replay-from-v2", action="store_true")
    args = parser.parse_args()
    if args.replay_from_v2:
        v2_state = json.loads(V2_STATE.read_text(encoding="utf-8"))
        _atomic_json(STATE, _initial_state(pd.Timestamp(v2_state["baseline"])))
    asyncio.run(worker(args.input, args.report, once=args.once))


if __name__ == "__main__":
    main()
