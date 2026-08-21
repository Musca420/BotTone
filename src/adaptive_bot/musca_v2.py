from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.indicators.atr import atr
from adaptive_bot.services.bitunix_paper_service import read_collected_candles

OOS_TRADES = Path("data/ml/musca_v2/oos_trades.parquet")
BUNDLE = Path("data/models/musca_v2/bundle.joblib")
REPORT = Path("data/reports/musca_v2_shadow.json")
AUDIT = Path("data/reports/musca_v2_audit.json")
RESEARCH_REPORT = Path("data/reports/musca_v2_research.json")
INPUT = Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
HISTORY = Path("data/ml/musca_v2/btc_spot_perp_5m.parquet")
STATE = Path("data/research/musca_v2_shadow_state.json")


def select_ranked_oos(rows: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    """Select by EV rank inside each untouched fold, then enforce one position."""
    ranked = pd.concat(
        [
            fold.loc[fold["expected_net_bps"].ge(fold["expected_net_bps"].quantile(quantile))]
            for _, fold in rows.groupby("outer_fold", sort=True)
        ]
    ).sort_values(["entry_timestamp", "expected_net_bps"], ascending=[True, False])
    accepted: list[Any] = []
    busy_until: dict[str, pd.Timestamp] = {}
    daily: Counter[tuple[str, str]] = Counter()
    for index, row in ranked.iterrows():
        timestamp = pd.Timestamp(row["entry_timestamp"])
        asset = str(row["asset"])
        key = asset, timestamp.strftime("%Y-%m-%d")
        if daily[key] >= 2 or timestamp < busy_until.get(
            asset, pd.Timestamp.min.tz_localize("UTC")
        ):
            continue
        accepted.append(index)
        daily[key] += 1
        busy_until[asset] = pd.Timestamp(row["exit_timestamp"])
    return ranked.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def build_bundle() -> tuple[dict[str, Any], pd.DataFrame]:
    audit = json.loads(RESEARCH_REPORT.read_text(encoding="utf-8"))
    decisions = pd.read_parquet(OOS_TRADES)
    _atomic_json(AUDIT, audit)
    return audit, decisions


def _market_telemetry(candles: pd.DataFrame) -> dict[str, Any]:
    row = _live_features(candles).iloc[-1]
    center, atr_value = float(row["daily_vwap"]), float(row["atr"])
    return {
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "activity": "WATCHING_VWAP_EVENT",
        "close": float(row["perp_close"]),
        "center": center,
        "lower_band": None,
        "upper_band": None,
        "atr": atr_value,
        "adx": None,
        "z_score": None,
        "entry_score": float(row["trend_strength"]) * 10_000,
        "atr_percentile": None,
        "ema_slope": None,
        "spread_bps": None,
        "regime": "trend-up" if int(row["direction"]) > 0 else "trend-down",
        "decision_reason": "waiting for a complete causal V25 trend/pullback candidate",
    }


@lru_cache(maxsize=1)
def _history_tail() -> pd.DataFrame:
    columns = [
        "timestamp",
        "perp_open",
        "perp_high",
        "perp_low",
        "perp_close",
        "perp_volume",
        "perp_quote_volume",
    ]
    return pd.read_parquet(HISTORY, columns=columns).tail(10_000).copy()


def _live_features(candles: pd.DataFrame) -> pd.DataFrame:
    current = candles.sort_values("timestamp").copy()
    current = current.rename(
        columns={
            "open": "perp_open",
            "high": "perp_high",
            "low": "perp_low",
            "close": "perp_close",
            "volume": "perp_volume",
        }
    )
    current["perp_quote_volume"] = current["quote_volume"]
    first = pd.Timestamp(current["timestamp"].min())
    history = _history_tail()
    history = history.loc[pd.to_datetime(history["timestamp"], utc=True).lt(first)]
    data = pd.concat([history, current[history.columns]], ignore_index=True)
    data = (
        data.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    time = pd.to_datetime(data["timestamp"], utc=True)
    day = time.dt.floor("D")
    data["atr"] = atr(data["perp_high"], data["perp_low"], data["perp_close"], 14)
    data["daily_vwap"] = (
        data["perp_quote_volume"].groupby(day).cumsum() / data["perp_volume"].groupby(day).cumsum()
    )
    data["trend_strength"] = 0.6 * data["perp_close"].pct_change(2016) + 0.4 * data[
        "perp_close"
    ].pct_change(8640)
    data["direction"] = np.sign(data["trend_strength"])
    data["structural_low"] = (
        data["perp_low"].shift(1).rolling(2016, min_periods=2016).min() - 0.1 * data["atr"]
    )
    data["structural_high"] = (
        data["perp_high"].shift(1).rolling(2016, min_periods=2016).max() + 0.1 * data["atr"]
    )
    data["reclaim"] = (
        data["direction"].gt(0)
        & data["perp_close"].gt(data["daily_vwap"])
        & data["perp_close"].shift(1).le(data["daily_vwap"].shift(1))
    ) | (
        data["direction"].lt(0)
        & data["perp_close"].lt(data["daily_vwap"])
        & data["perp_close"].shift(1).ge(data["daily_vwap"].shift(1))
    )
    return data


def _load_shadow_state(latest: pd.Timestamp) -> dict[str, Any]:
    if STATE.exists():
        loaded = cast(dict[str, Any], json.loads(STATE.read_text(encoding="utf-8")))
        loaded.setdefault("signals", 0)
        loaded.setdefault("fees", "0")
        loaded.setdefault("peak_equity", loaded["equity"])
        loaded.setdefault("max_drawdown", "0")
        loaded.setdefault(
            "equity_curve", [{"timestamp": loaded["baseline"], "equity": loaded["equity"]}]
        )
        return loaded
    state: dict[str, Any] = {
        "baseline": latest.isoformat(),
        "last_processed": latest.isoformat(),
        "equity": "10000",
        "position": None,
        "pending": None,
        "fills": [],
        "closed_trades": [],
        "signals": 0,
        "fees": "0",
        "peak_equity": "10000",
        "max_drawdown": "0",
        "equity_curve": [{"timestamp": latest.isoformat(), "equity": "10000"}],
    }
    _atomic_json(STATE, state)
    return state


def _advance_shadow(candles: pd.DataFrame) -> dict[str, Any]:
    data = _live_features(candles)
    latest = pd.Timestamp(data.iloc[-1]["timestamp"])
    state = _load_shadow_state(latest)
    last = pd.Timestamp(state["last_processed"])
    new_indices = data.index[pd.to_datetime(data["timestamp"], utc=True).gt(last)]
    for raw_index in new_indices:
        index = int(raw_index)
        row = data.iloc[index]
        timestamp = pd.Timestamp(row["timestamp"])
        if state["pending"] is not None and state["position"] is None:
            pending = state["pending"]
            state["pending"] = None
            entry = Decimal(str(row["perp_open"]))
            stop = Decimal(str(pending["stop"]))
            distance = abs(entry - stop)
            if distance > 0:
                quantity = Decimal(state["equity"]) * Decimal("0.0025") / distance
                state["position"] = {
                    "direction": int(pending["direction"]),
                    "entry": str(entry),
                    "stop": str(stop),
                    "initial_stop": str(stop),
                    "quantity": str(quantity),
                    "opened_at": timestamp.isoformat(),
                    "bars_open": 0,
                }
                side = "buy" if int(pending["direction"]) > 0 else "sell"
                state["fills"].append(_fill(timestamp, side, entry, quantity, len(state["fills"])))
        position = state["position"]
        if position is not None:
            direction = int(position["direction"])
            entry = Decimal(position["entry"])
            stop = Decimal(position["stop"])
            quantity = Decimal(position["quantity"])
            position["bars_open"] += 1
            stopped = (
                Decimal(str(row["perp_low"])) <= stop
                if direction > 0
                else Decimal(str(row["perp_high"])) >= stop
            )
            reversed_trend = position["bars_open"] > 288 and int(row["direction"]) == -direction
            timed_out = position["bars_open"] >= 4032
            if stopped or reversed_trend or timed_out:
                market = Decimal(str(row["perp_open"] if stopped else row["perp_close"]))
                exit_price = (
                    min(market, stop)
                    if stopped and direction > 0
                    else max(market, stop)
                    if stopped
                    else market
                )
                gross = Decimal(direction) * (exit_price - entry) * quantity
                cost = entry * quantity * Decimal(8) / Decimal(10_000)
                state["equity"] = str(Decimal(state["equity"]) + gross - cost)
                state["fees"] = str(Decimal(state["fees"]) + cost)
                peak = max(Decimal(state["peak_equity"]), Decimal(state["equity"]))
                state["peak_equity"] = str(peak)
                state["max_drawdown"] = str(
                    max(
                        Decimal(state["max_drawdown"]),
                        (peak - Decimal(state["equity"])) / peak,
                    )
                )
                state["equity_curve"].append(
                    {"timestamp": timestamp.isoformat(), "equity": state["equity"]}
                )
                side = "sell" if direction > 0 else "buy"
                state["fills"].append(
                    _fill(timestamp, side, exit_price, quantity, len(state["fills"]))
                )
                state["closed_trades"].append(
                    {
                        "entry_timestamp": position["opened_at"],
                        "exit_timestamp": timestamp.isoformat(),
                        "net_pnl": str(gross - cost),
                        "reason": "STOP"
                        if stopped
                        else "TREND_REVERSAL"
                        if reversed_trend
                        else "TIMEOUT",
                    }
                )
                state["position"] = None
            elif position["bars_open"] >= 576 and timestamp.hour == 0 and timestamp.minute == 0:
                proposal = Decimal(
                    str(row["structural_low"] if direction > 0 else row["structural_high"])
                )
                tightened = max(stop, proposal) if direction > 0 else min(stop, proposal)
                position["stop"] = str(tightened)
        if state["position"] is None and state["pending"] is None and bool(row["reclaim"]):
            direction = int(row["direction"])
            stop = row["structural_low"] if direction > 0 else row["structural_high"]
            if pd.notna(stop):
                state["pending"] = {"direction": direction, "stop": str(float(stop))}
                state["signals"] += 1
        state["last_processed"] = timestamp.isoformat()
    _atomic_json(STATE, state)
    return state


def make_report(
    decisions: pd.DataFrame, candles: pd.DataFrame, shadow: dict[str, Any]
) -> dict[str, Any]:
    historical_curve = (1 + decisions["net_return_r"].astype(float) * 0.0025).cumprod() * 10_000
    historical_equity = float(historical_curve.iloc[-1]) if len(historical_curve) else 10_000.0
    current = _market_telemetry(candles)
    current["activity"] = (
        "POSITION_OPEN"
        if shadow["position"] is not None
        else "ORDER_PENDING"
        if shadow["pending"] is not None
        else "WATCHING_VWAP_RECLAIM"
    )
    current["decision_reason"] = (
        "managing structural trailing stop"
        if shadow["position"] is not None
        else "next-bar entry pending"
        if shadow["pending"] is not None
        else "no daily VWAP reclaim on the latest completed 5m bar"
    )
    position = shadow["position"]
    if position is not None:
        bars_open = int(position["bars_open"])
        current["decision_reason"] = (
            "VWAP recross is not an exit; structural stop "
            f"{position['stop']}; trend reversal armed in "
            f"{max(0, 289 - bars_open)} bars"
        )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    shadow_equity = Decimal(shadow["equity"])
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 5,
        "strategy_profile": "musca_v2_long_horizon_vwap",
        "validation_status": audit["verdict"],
        "policy_action": current["activity"],
        "live_orders_enabled": False,
        "initial_equity": "10000",
        "final_equity": str(shadow_equity),
        "gross_pnl": str(shadow_equity - Decimal("10000") + Decimal(shadow["fees"])),
        "net_pnl": str(shadow_equity - Decimal("10000")),
        "fees": shadow["fees"],
        "slippage": "0",
        "max_drawdown": shadow["max_drawdown"],
        "risk_per_trade": "0.0025",
        "signals": shadow["signals"],
        "rejected_signals": 0,
        "kill_switches": 0,
        "fills": shadow["fills"],
        "equity_curve": shadow["equity_curve"],
        "telemetry": [current],
        "cost_scenarios": {
            "8bps_expectancy": audit["oos_metrics"]["expectancy_bps"],
            "16bps_expectancy": audit["stress_metrics"]["expectancy_bps"],
        },
        "historical_replay": {
            "trades": len(decisions),
            "final_equity": historical_equity,
            "expectancy_bps": audit["oos_metrics"]["expectancy_bps"],
            "profit_factor": audit["oos_metrics"]["profit_factor"],
            "max_drawdown": audit["oos_metrics"]["max_drawdown_fraction"],
        },
        "shadow_closed_trades": len(shadow["closed_trades"]),
        "shadow_position": shadow["position"],
        "shadow_exit_policy": {
            "vwap_recross_exit": False,
            "trend_reversal_after_bars": 288,
            "structural_trailing_after_bars": 576,
            "timeout_bars": 4032,
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def worker(input_path: Path, report_path: Path, *, once: bool) -> None:
    audit, decisions = build_bundle()
    del audit
    while True:
        candles = read_collected_candles(input_path)
        shadow = _advance_shadow(candles)
        _atomic_json(report_path, make_report(decisions, candles, shadow))
        if once:
            return
        await asyncio.sleep(300)


def _fill(
    timestamp: pd.Timestamp, side: str, price: Decimal, quantity: Decimal, sequence: int
) -> dict[str, str]:
    return {
        "exchange_timestamp": timestamp.isoformat(),
        "received_timestamp": timestamp.isoformat(),
        "source": "musca_v2_oos_shadow",
        "instrument": "BTCUSDT",
        "client_order_id": f"musca-v2-{timestamp.value}-{sequence}",
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "commission": "0",
        "slippage": "0",
        "liquidity_role": "simulated",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca v2 adaptive VWAP research shadow")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    asyncio.run(worker(args.input, args.report, once=args.once))


if __name__ == "__main__":
    main()
