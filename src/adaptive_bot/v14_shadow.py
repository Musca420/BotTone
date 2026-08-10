from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from adaptive_bot.config import load_config
from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Instrument
from adaptive_bot.hybrid_policy_v14 import (
    FEATURES,
    first_reentry_events,
    predict_meta_model,
    protocol_hash,
)
from adaptive_bot.hybrid_policy_v14_forward import (
    FORWARD_EXTERNAL_ROOT,
    FORWARD_MICROSTRUCTURE_ROOT,
    FORWARD_MINUTES_PATH,
    FORWARD_ORDERFLOW_PATH,
    FORWARD_WARMUP_META_PATH,
    FORWARD_WARMUP_PATH,
    build_bitunix_forward_minutes,
    build_forward_policy_events,
    combined_bitunix_alpha_minutes,
    download_binance_live_orderflow,
    download_bitunix_warmup,
    download_external_forward,
    observed_trade_candles,
    resample_trade_minutes,
)
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import atr_percentile
from adaptive_bot.indicators.vwap import rolling_vwap
from adaptive_bot.risk.position_sizing import SizingInput, size_position

REPORT = Path("data/reports/v14_vwap_shadow.json")
LOCK = Path("data/models/expert_policy/v14/forward_protocol.lock.json")
BUNDLE = Path("data/models/expert_policy/v14/research_shadow_bundle.joblib")
TIMEFRAME_MINUTES = 5
VWAP_BARS = 24 * 60 // TIMEFRAME_MINUTES
MAX_HOLDING_BARS = 8 * 60 // TIMEFRAME_MINUTES


def _feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    bars = candles.sort_values("timestamp").reset_index(drop=True).copy()
    bars["atr"] = atr(bars["high"], bars["low"], bars["close"], 14)
    bars["adx"] = adx(bars["high"], bars["low"], bars["close"], 14)["adx"]
    bars["vwap"] = rolling_vwap(bars["high"], bars["low"], bars["close"], bars["volume"], VWAP_BARS)
    bars["distance_vwap_atr"] = (bars["close"] - bars["vwap"]) / bars["atr"]
    bars["atr_percentile"] = atr_percentile(bars["atr"], 100)
    bars["ema_slope"] = normalized_ema_slope(bars["close"], bars["atr"])
    bars["regime_code"] = 0.0
    bars.loc[(bars["adx"] >= 25) & (bars["ema_slope"] > 0), "regime_code"] = 1.0
    bars.loc[(bars["adx"] >= 25) & (bars["ema_slope"] < 0), "regime_code"] = 2.0
    bars.loc[bars["atr_percentile"] > 90, "regime_code"] = 4.0
    coverage = bars[["atr", "adx", "vwap", "atr_percentile", "ema_slope"]].notna().all(axis=1)
    bars.loc[~coverage, "regime_code"] = 3.0
    continuity = bars["timestamp"].diff().eq(pd.Timedelta(minutes=TIMEFRAME_MINUTES))
    if "data_valid" in bars:
        continuity &= bars["data_valid"].fillna(False).astype(bool)
    bars["data_valid"] = continuity
    bars["local_feature_coverage"] = coverage
    return bars


def replay_v14_shadow(
    candles: pd.DataFrame,
    *,
    trade_after: pd.Timestamp,
    initial_equity: Decimal = Decimal("10000"),
    risk_fraction: Decimal = Decimal("0.01"),
    round_trip_cost_bps: Decimal = Decimal("19"),
    instrument: Instrument,
    hard_notional_cap: Decimal,
    target_exposure_fraction: Decimal,
    policy_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bars = _feature_frame(candles)
    events = {side: first_reentry_events(bars, side) for side in ("long", "short")}
    equity = initial_equity
    peak = equity
    maximum_drawdown = Decimal(0)
    gross_pnl = Decimal(0)
    fees = Decimal(0)
    slippage = Decimal(0)
    round_trip_notional = Decimal(0)
    signals = 0
    rejected = 0
    fills: list[dict[str, str]] = []
    trades: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    equity_curve: list[dict[str, str]] = []
    position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    policy_decisions: list[dict[str, Any]] = []

    for index, (_, row) in enumerate(bars.iterrows()):
        timestamp = pd.Timestamp(row["timestamp"])
        activity = "FLAT"
        if pending is not None and position is None:
            direction = Decimal(1 if pending["side"] == "long" else -1)
            entry = Decimal(str(row["open"]))
            stop = Decimal(str(pending["stop"]))
            target = Decimal(str(pending["target"]))
            entry_atr = Decimal(str(pending["atr"]))
            risk = direction * (entry - stop)
            reward = direction * (target - entry)
            if risk > 0 and risk <= Decimal(2) * entry_atr and reward > 0:
                sizing = size_position(
                    SizingInput(
                        equity=equity,
                        buying_power=equity * instrument.max_leverage,
                        entry_price=entry,
                        stop_price=stop,
                        estimated_cost_per_unit=entry * round_trip_cost_bps / Decimal(10_000),
                        risk_fraction=risk_fraction,
                        hard_notional_cap=min(hard_notional_cap, equity * target_exposure_fraction),
                        side=Side.BUY if direction > 0 else Side.SELL,
                    ),
                    instrument,
                )
                quantity = sizing.quantity.quantize(instrument.lot_size, rounding=ROUND_DOWN)
                if sizing.approved and quantity > 0:
                    position = {
                        **pending,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "quantity": quantity,
                        "opened_index": index,
                        "opened_at": timestamp,
                    }
                    fills.append(
                        _fill(
                            timestamp,
                            "buy" if direction > 0 else "sell",
                            entry,
                            quantity,
                            len(fills),
                        )
                    )
                    activity = f"{pending['side'].upper()}_OPEN"
                else:
                    rejected += 1
            else:
                rejected += 1
            pending = None

        if position is not None:
            direction = Decimal(1 if position["side"] == "long" else -1)
            high = Decimal(str(row["high"]))
            low = Decimal(str(row["low"]))
            stop_hit = low <= position["stop"] if direction > 0 else high >= position["stop"]
            target_hit = high >= position["target"] if direction > 0 else low <= position["target"]
            expired = index - position["opened_index"] >= MAX_HOLDING_BARS
            invalid_path = not bool(row["data_valid"])
            if stop_hit or target_hit or expired or invalid_path:
                if stop_hit:  # Worst case when stop and target share a candle.
                    opened = Decimal(str(row["open"]))
                    exit_price = (
                        min(position["stop"], opened)
                        if direction > 0
                        else max(position["stop"], opened)
                    )
                    reason = "stop"
                elif target_hit:
                    exit_price = position["target"]
                    reason = "target"
                else:
                    exit_price = Decimal(str(row["open"] if invalid_path else row["close"]))
                    reason = "data_integrity" if invalid_path else "time"
                quantity = position["quantity"]
                gross = direction * (exit_price - position["entry"]) * quantity
                notional = (position["entry"] + exit_price) * quantity
                cost = notional * round_trip_cost_bps / Decimal(20_000)
                trade_fees = notional * Decimal(12) / Decimal(20_000)
                trade_slippage = notional * Decimal(7) / Decimal(20_000)
                gross_pnl += gross
                round_trip_notional += notional
                fees += trade_fees
                slippage += trade_slippage
                equity += gross - cost
                fills.append(
                    _fill(
                        timestamp,
                        "sell" if direction > 0 else "buy",
                        exit_price,
                        quantity,
                        len(fills),
                        commission=trade_fees,
                        slippage=trade_slippage,
                    )
                )
                trades.append(
                    {
                        "side": position["side"].upper(),
                        "signal_timestamp": position["signal_timestamp"].isoformat(),
                        "entry_timestamp": position["opened_at"].isoformat(),
                        "exit_timestamp": timestamp.isoformat(),
                        "entry_price": str(position["entry"]),
                        "exit_price": str(exit_price),
                        "stop_price": str(position["stop"]),
                        "target_price": str(position["target"]),
                        "quantity": str(quantity),
                        "gross_pnl": str(gross),
                        "net_pnl": str(gross - cost),
                        "gross_return_bps": str(
                            direction
                            * (exit_price - position["entry"])
                            / position["entry"]
                            * Decimal(10_000)
                        ),
                        "round_trip_notional": str(notional),
                        "exit_reason": reason,
                    }
                )
                activity = f"{position['side'].upper()}_{reason.upper()}"
                position = None
                peak = max(peak, equity)
                maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
            elif activity == "FLAT":
                activity = f"{position['side'].upper()}_OPEN"

        candidates = [
            side
            for side in ("long", "short")
            if bool(events[side].loc[index, "event_signal"]) and timestamp > trade_after
        ]
        signals += len(candidates)
        if candidates and position is None and pending is None:
            side = candidates[0]
            event = events[side].loc[index]
            event_values = {str(name): value for name, value in event.items()}
            decision = score_policy_event(
                {**event_values, "side": side, "timeframe_minutes": TIMEFRAME_MINUTES},
                policy_bundle,
            )
            decision["signal_timestamp"] = timestamp.isoformat()
            policy_decisions.append(decision)
            pending = {
                "side": side,
                "signal_timestamp": timestamp,
                "stop": float(str(event["event_stop_price"])),
                "target": float(row["vwap"]),
                "atr": float(row["atr"]),
            }
            activity = f"{side.upper()}_SIGNAL"
        elif candidates:
            rejected += len(candidates)

        center = _finite(row["vwap"])
        atr_value = _finite(row["atr"])
        telemetry.append(
            {
                "timestamp": timestamp.isoformat(),
                "activity": activity,
                "policy_action": (
                    policy_decisions[-1]["action"] if policy_decisions else "FLAT_NO_EVENT"
                ),
                "diagnostic_action": activity,
                "close": float(row["close"]),
                "center": center,
                "lower_band": center - 1.5 * atr_value
                if center is not None and atr_value is not None
                else None,
                "upper_band": center + 1.5 * atr_value
                if center is not None and atr_value is not None
                else None,
                "atr": atr_value,
                "adx": _finite(row["adx"]),
                "z_score": _finite(row["distance_vwap_atr"]),
                "entry_score": abs(_finite(row["distance_vwap_atr"]) or 0.0),
                "atr_percentile": _finite(row["atr_percentile"]),
                "ema_slope": _finite(row["ema_slope"]),
                "spread_bps": _finite(row.get("spread_bps")),
                "regime": _regime(row["regime_code"]),
            }
        )
        equity_curve.append({"timestamp": timestamp.isoformat(), "equity": str(equity)})

    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "market_data_source": "bitunix-official-public-websocket-trades",
        "strategy_profile": "v14_vwap_diagnostic_shadow",
        "validation_status": "RESEARCH_SHADOW_DIAGNOSTIC_NOT_APPROVED",
        "policy_action": (policy_decisions[-1]["action"] if policy_decisions else "FLAT_NO_EVENT"),
        "diagnostic_trades_ignore_policy_lcb": True,
        "live_orders_enabled": False,
        "protocol_sha256": protocol_hash(),
        "forward_cutoff": trade_after.isoformat(),
        "initial_equity": str(initial_equity),
        "final_equity": str(equity),
        "gross_pnl": str(gross_pnl),
        "net_pnl": str(equity - initial_equity),
        "fees": str(fees),
        "slippage": str(slippage),
        "cost_scenarios": {
            "gross_0bps": _scenario(gross_pnl, round_trip_notional, Decimal(0)),
            "maker_fee_only_4bps": _scenario(gross_pnl, round_trip_notional, Decimal(4)),
            "taker_fee_only_12bps": _scenario(gross_pnl, round_trip_notional, Decimal(12)),
            "configured_conservative_19bps": _scenario(
                gross_pnl, round_trip_notional, round_trip_cost_bps
            ),
        },
        "max_drawdown": str(maximum_drawdown),
        "risk_per_trade": str(risk_fraction),
        "signals": signals,
        "rejected_signals": rejected,
        "kill_switches": 0,
        "fills": fills,
        "trades": trades,
        "policy_decisions": policy_decisions,
        "equity_curve": equity_curve,
        "telemetry": telemetry,
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def run_worker(
    config_path: Path,
    microstructure_root: Path,
    report_path: Path,
    lock_path: Path,
    bundle_path: Path,
    *,
    once: bool,
    poll_seconds: float = 300,
) -> None:
    config = load_config(config_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bundle = joblib.load(bundle_path)
    if bundle["protocol"]["protocol_sha256"] != lock["protocol_sha256"]:
        raise RuntimeError("V14 bundle and permanent protocol lock do not match")
    trade_after = pd.Timestamp(lock["cutoff"])
    if not FORWARD_WARMUP_PATH.exists() or not FORWARD_WARMUP_META_PATH.exists():
        warmup_end = datetime.now(UTC).replace(second=0, microsecond=0)
        download_bitunix_warmup(warmup_end - timedelta(days=5), warmup_end)
    warmup = json.loads(FORWARD_WARMUP_META_PATH.read_text(encoding="utf-8"))
    model_after = max(
        trade_after, pd.Timestamp(lock["created_at"]), pd.Timestamp(warmup["available_at"])
    )
    while True:
        trade_minutes = observed_trade_candles(microstructure_root, timeframe_minutes=1)
        forward_minutes = build_bitunix_forward_minutes(
            cutoff=trade_after,
            microstructure_root=microstructure_root,
            output_path=FORWARD_MINUTES_PATH,
            trade_minutes=trade_minutes,
        )
        context_errors: list[str] = []
        now = pd.Timestamp.now(tz="UTC").floor("min")
        try:
            download_external_forward(trade_after.to_pydatetime(), now.to_pydatetime())
        except Exception as exc:  # Network failure must leave the policy flat, not kill shadow.
            context_errors.append(f"external:{type(exc).__name__}:{exc}")
        try:
            download_binance_live_orderflow(trade_after, now)
        except Exception as exc:
            context_errors.append(f"orderflow:{type(exc).__name__}:{exc}")
        report = replay_v14_shadow(
            resample_trade_minutes(trade_minutes, timeframe_minutes=TIMEFRAME_MINUTES),
            trade_after=trade_after,
            initial_equity=config.backtest.initial_equity,
            risk_fraction=config.risk.risk_per_trade,
            round_trip_cost_bps=(
                Decimal(2) * config.backtest.taker_fee_bps
                + Decimal(2) * config.backtest.slippage_bps
                + config.backtest.spread_bps
            ),
            instrument=config.instrument,
            hard_notional_cap=config.risk.hard_notional_cap,
            target_exposure_fraction=config.risk.target_exposure_fraction,
            policy_bundle=bundle,
        )
        external_ready = all(
            (FORWARD_EXTERNAL_ROOT / f"{name}_btcusdt_1m.parquet").exists()
            for name in ("binance", "okx", "bybit")
        )
        orderflow_ready = FORWARD_ORDERFLOW_PATH.exists()
        readiness = model_input_readiness(
            combined_bitunix_alpha_minutes(forward_minutes),
            external_ready=external_ready,
            orderflow_ready=orderflow_ready,
        )
        model_decisions: list[dict[str, Any]] = []
        if readiness["ready"]:
            for _, event in build_forward_policy_events(config).iterrows():
                signal_at = pd.Timestamp(event["signal_timestamp"])
                if signal_at <= model_after:
                    continue
                decision = score_policy_event(
                    {str(key): value for key, value in event.to_dict().items()}, bundle
                )
                decision["signal_timestamp"] = signal_at.isoformat()
                decision["timeframe_minutes"] = int(event["timeframe_minutes"])
                model_decisions.append(decision)
        report["model_policy_decisions"] = model_decisions
        report["policy_action"] = (
            model_decisions[-1]["action"]
            if model_decisions
            else "FLAT_WAITING_FOR_VWAP_EVENT"
            if readiness["ready"]
            else "FLAT_INPUTS_NOT_READY"
        )
        readiness["context_errors"] = context_errors
        report["model_input_readiness"] = readiness
        _atomic_json(report_path, report)
        if once:
            return
        await asyncio.sleep(poll_seconds)


def _fill(
    timestamp: pd.Timestamp,
    side: str,
    price: Decimal,
    quantity: Decimal,
    sequence: int,
    *,
    commission: Decimal = Decimal(0),
    slippage: Decimal = Decimal(0),
) -> dict[str, str]:
    return {
        "exchange_timestamp": timestamp.isoformat(),
        "received_timestamp": timestamp.isoformat(),
        "source": "v14_vwap_diagnostic_shadow",
        "instrument": "BTCUSDT",
        "client_order_id": f"v14-shadow-{timestamp.value}-{sequence}",
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "commission": str(commission),
        "slippage": str(slippage),
        "liquidity_role": "simulated",
    }


def _finite(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _regime(value: object) -> str:
    regimes = {0.0: "range", 1.0: "trend_up", 2.0: "trend_down", 3.0: "unknown", 4.0: "shock"}
    code = _finite(value)
    return regimes.get(code, "unknown") if code is not None else "unknown"


def score_policy_event(event: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, Any]:
    if bundle is None:
        return {"action": "FLAT_MODEL_UNAVAILABLE"}
    timeframe = int(event.get("timeframe_minutes", 0))
    supported = tuple(bundle["protocol"]["primary_rule"]["timeframes_minutes"])
    if timeframe not in supported:
        return {
            "action": f"FLAT_UNSUPPORTED_TIMEFRAME_{timeframe}M",
            "supported_timeframes_minutes": list(supported),
        }
    missing = [name for name in FEATURES if _finite(event.get(name)) is None]
    if missing:
        return {"action": "FLAT_FEATURES_INCOMPLETE", "missing_features": missing}
    predicted = predict_meta_model(
        pd.DataFrame([{**event, "feature_coverage": True}]), bundle["models"]
    )
    if predicted.empty:
        return {"action": "FLAT_MODEL_DISABLED"}
    scored = predicted.iloc[0]
    lcb = float(scored["lcb_net"])
    side_key = str(scored["side"])
    side = side_key.upper()
    return {
        "action": side if lcb > 0 else "FLAT_NONPOSITIVE_LCB",
        "side": side,
        "probability": float(scored["probability"]),
        "ev_net_r": float(scored["ev_net"]),
        "lcb_net_r": lcb,
        "champion": bundle["models"]["sides"][side_key]["champion"],
    }


def model_input_readiness(
    minutes: pd.DataFrame, *, external_ready: bool = False, orderflow_ready: bool = False
) -> dict[str, Any]:
    required_bars = 100
    readiness: dict[str, Any] = {}
    for timeframe in (15, 30, 60):
        consecutive = 0
        if not minutes.empty:
            grouped = minutes.set_index("timestamp").resample(
                f"{timeframe}min", origin="epoch", closed="left", label="left"
            )
            bars = grouped.agg(rows=("close", "count"), valid=("data_valid", "all"))
            complete = bars["rows"].eq(timeframe) & bars["valid"]
            if len(bars) and int(bars.iloc[-1]["rows"]) < timeframe:
                complete = complete.iloc[:-1]
            for valid in reversed(complete.tolist()):
                if not valid:
                    break
                consecutive += 1
        readiness[f"{timeframe}m"] = {
            "consecutive_complete_bars": consecutive,
            "required_bars": required_bars,
            "local_ready": consecutive >= required_bars,
        }
    local_ready = any(item["local_ready"] for item in readiness.values())
    return {
        "ready": local_ready and external_ready and orderflow_ready,
        "local_ready_any_timeframe": local_ready,
        "valid_minutes": int(minutes["data_valid"].sum()) if not minutes.empty else 0,
        "latest_minute": (
            pd.Timestamp(minutes["timestamp"].max()).isoformat() if not minutes.empty else None
        ),
        "timeframes": readiness,
        "missing": [
            *([] if local_ready else ["consecutive_bitunix_trade_and_mark_history"]),
            *([] if external_ready else ["current_cross_exchange_context"]),
            *([] if orderflow_ready else ["current_binance_orderflow_context"]),
        ],
    }


def _scenario(gross: Decimal, notional: Decimal, cost_bps: Decimal) -> dict[str, str]:
    cost = notional * cost_bps / Decimal(20_000)
    return {"cost_bps": str(cost_bps), "cost": str(cost), "net_pnl": str(gross - cost)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="V14 BTC VWAP diagnostic shadow worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--microstructure-root", type=Path, default=FORWARD_MICROSTRUCTURE_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(
        run_worker(
            arguments.config,
            arguments.microstructure_root,
            arguments.report,
            arguments.lock,
            arguments.bundle,
            once=arguments.once,
        )
    )


if __name__ == "__main__":
    main()
