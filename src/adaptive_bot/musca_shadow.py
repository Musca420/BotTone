from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.config import load_config
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import atr_percentile, normalized_distance
from adaptive_bot.indicators.vwap import rolling_vwap
from adaptive_bot.services.bitunix_paper_service import read_collected_candles

INPUT = Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
REPORT = Path("data/reports/musca_shadow.json")


def replay_musca(
    candles: pd.DataFrame,
    *,
    trade_after: pd.Timestamp,
    initial_equity: Decimal = Decimal("10000"),
    risk_fraction: Decimal = Decimal("0.01"),
    round_trip_cost_bps: Decimal = Decimal("19"),
) -> dict[str, Any]:
    bars = candles.sort_values("timestamp").reset_index(drop=True).copy()
    bars["atr"] = atr(bars["high"], bars["low"], bars["close"], 14)
    bars["adx"] = adx(bars["high"], bars["low"], bars["close"], 14)["adx"]
    bars["center"] = rolling_vwap(bars["high"], bars["low"], bars["close"], bars["volume"], 288)
    bars["z_score"] = normalized_distance(bars["close"], bars["center"], bars["atr"])
    bars["atr_percentile"] = atr_percentile(bars["atr"], 100)
    bars["ema_slope"] = normalized_ema_slope(bars["close"], bars["atr"])
    bars["prior_low"] = bars["low"].shift(1).rolling(24, min_periods=24).min()
    equity = initial_equity
    peak = equity
    maximum_drawdown = Decimal(0)
    fills: list[dict[str, Any]] = []
    equity_curve: list[dict[str, str]] = []
    telemetry: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    pending_signal = False
    signals = 0
    gross_pnl = Decimal(0)
    fees = Decimal(0)
    slippage = Decimal(0)
    for index, (_, row) in enumerate(bars.iterrows()):
        timestamp = pd.Timestamp(row["timestamp"])
        if pending_signal and position is None:
            entry = Decimal(str(row["open"]))
            atr_value = Decimal(str(bars.iloc[index - 1]["atr"]))
            risk_distance = Decimal(2) * atr_value
            quantity = (equity * risk_fraction / risk_distance).quantize(
                Decimal("0.0001"), rounding=ROUND_DOWN
            )
            if quantity > 0:
                position = {
                    "entry": entry,
                    "stop": entry + risk_distance,
                    "target": entry - risk_distance,
                    "quantity": quantity,
                    "opened_index": index,
                }
                fills.append(_fill(timestamp, "sell", entry, quantity, len(fills)))
            pending_signal = False
        if position is not None:
            stop_hit = Decimal(str(row["high"])) >= position["stop"]
            target_hit = Decimal(str(row["low"])) <= position["target"]
            expired = index - position["opened_index"] >= 32
            if stop_hit or target_hit or expired:
                # Worst case is deliberate when both barriers occur in one candle.
                exit_price = (
                    position["stop"]
                    if stop_hit
                    else position["target"]
                    if target_hit
                    else Decimal(str(row["close"]))
                )
                quantity = position["quantity"]
                gross = (position["entry"] - exit_price) * quantity
                notional = (position["entry"] + exit_price) * quantity
                cost = notional * round_trip_cost_bps / Decimal(20_000)
                gross_pnl += gross
                fees += notional * Decimal(12) / Decimal(20_000)
                slippage += notional * Decimal(7) / Decimal(20_000)
                equity += gross - cost
                fills.append(_fill(timestamp, "buy", exit_price, quantity, len(fills)))
                position = None
                peak = max(peak, equity)
                maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
        signal = bool(
            position is None
            and timestamp > trade_after
            and pd.notna(row["prior_low"])
            and pd.notna(row["adx"])
            and float(row["close"]) < float(row["prior_low"])
            and float(row["adx"]) >= 20
        )
        if signal:
            pending_signal = True
            signals += 1
        activity = "SHORT_SIGNAL" if signal else "SHORT_OPEN" if position else "FLAT"
        breakout_strength = (
            max(
                0.0,
                (float(row["prior_low"]) - float(row["close"])) / float(row["atr"]),
            )
            if pd.notna(row["prior_low"]) and pd.notna(row["atr"])
            else 0.0
        )
        momentum_score = (
            breakout_strength + max(0.0, (float(row["adx"]) - 20) / 20)
            if pd.notna(row["adx"])
            else 0.0
        )
        center = _finite(row["center"])
        diagnostic_atr = _finite(row["atr"])
        telemetry.append(
            {
                "timestamp": timestamp.isoformat(),
                "activity": activity,
                "close": float(row["close"]),
                "center": center,
                "lower_band": center - 2 * diagnostic_atr
                if center is not None and diagnostic_atr is not None
                else None,
                "upper_band": center + 2 * diagnostic_atr
                if center is not None and diagnostic_atr is not None
                else None,
                "atr": diagnostic_atr,
                "adx": _finite(row["adx"]),
                "z_score": _finite(row["z_score"]),
                "entry_score": momentum_score,
                "atr_percentile": _finite(row["atr_percentile"]),
                "ema_slope": _finite(row["ema_slope"]),
                "spread_bps": _finite(row.get("spread_bps")),
                "regime": "momentum" if _finite(row["adx"]) and float(row["adx"]) >= 20 else "flat",
            }
        )
        equity_curve.append({"timestamp": timestamp.isoformat(), "equity": str(equity)})
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 5,
        "strategy_profile": "musca_v8_momentum_short_control",
        "validation_status": "NONVALIDATED_V8_CANDIDATE",
        "live_orders_enabled": False,
        "initial_equity": str(initial_equity),
        "final_equity": str(equity),
        "gross_pnl": str(gross_pnl),
        "net_pnl": str(equity - initial_equity),
        "fees": str(fees),
        "slippage": str(slippage),
        "max_drawdown": str(maximum_drawdown),
        "risk_per_trade": str(risk_fraction),
        "signals": signals,
        "rejected_signals": 0,
        "kill_switches": 0,
        "fills": fills,
        "equity_curve": equity_curve,
        "telemetry": telemetry,
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def run_worker(
    config_path: Path, input_path: Path, report_path: Path, *, once: bool, poll_seconds: float = 15
) -> None:
    config = load_config(config_path)
    baseline_path = report_path.with_suffix(".start")
    while True:
        frame = read_collected_candles(input_path)
        bars = frame.sort_values("timestamp").reset_index(drop=True)
        if baseline_path.exists():
            baseline = pd.Timestamp(baseline_path.read_text(encoding="utf-8").strip())
        else:
            baseline = pd.Timestamp(bars.iloc[-1]["timestamp"])
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(baseline.isoformat(), encoding="utf-8")
        report = replay_musca(
            frame,
            trade_after=baseline,
            initial_equity=config.backtest.initial_equity,
            risk_fraction=config.risk.risk_per_trade,
            round_trip_cost_bps=(
                Decimal(2) * config.backtest.taker_fee_bps
                + Decimal(2) * config.backtest.slippage_bps
                + config.backtest.spread_bps
            ),
        )
        _atomic_json(report_path, report)
        if once:
            return
        await asyncio.sleep(poll_seconds)


def _fill(
    timestamp: pd.Timestamp, side: str, price: Decimal, quantity: Decimal, sequence: int
) -> dict[str, str]:
    return {
        "exchange_timestamp": timestamp.isoformat(),
        "received_timestamp": timestamp.isoformat(),
        "source": "musca_shadow",
        "instrument": "BTCUSDT",
        "client_order_id": f"musca-{timestamp.value}-{sequence}",
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "commission": "0",
        "slippage": "0",
        "liquidity_role": "simulated",
    }


def _finite(value: object) -> float | None:
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MUSCA BTC shadow paper worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(
        run_worker(arguments.config, arguments.input, arguments.report, once=arguments.once)
    )


if __name__ == "__main__":
    main()
