from __future__ import annotations

import ipaddress
import json
import math
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import normalized_distance
from adaptive_bot.indicators.vwap import rolling_vwap
from adaptive_bot.services.bitunix_paper_service import (
    ADX_VARIANTS,
    WEIGHTED_PROFILE,
    WEIGHTED_V2_PROFILE,
    WEIGHTED_V11_PROFILE,
    profile_report_path,
    variant_report_path,
)

MILESTONES = (
    {"name": "Milestone 1", "label": "Simulation core", "status": "complete"},
    {"name": "Milestone 2", "label": "Alpaca Paper", "status": "complete"},
    {"name": "Milestone 3", "label": "Research & stress", "status": "next"},
    {"name": "Milestone 4", "label": "Bitunix BTCUSDT Futures", "status": "planned"},
    {"name": "Milestone 5", "label": "IBKR Paper", "status": "planned"},
)
LIVE_DATA_PATH = Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
MUSCA_REPORT_PATH = Path("data/reports/musca_shadow.json")
MUSCA_V2_REPORT_PATH = Path("data/reports/musca_v2_shadow.json")
MUSCA_V4_REPORT_PATH = Path("data/reports/musca_v4_shadow.json")
MUSCA_V5_REPORT_PATH = Path("data/reports/musca_v5_shadow.json")
MUSCA_V8_BINANCE_REPORT_PATH = Path("data/reports/musca_v8_binance_shadow.json")
MUSCA_LIQUIDITY_REPORT_PATH = Path("data/reports/musca_vwap_liquidity_filtered_shadow.json")
V14_SHADOW_REPORT_PATH = Path("data/reports/v14_vwap_shadow.json")
CROSS_EXCHANGE_AUDIT_PATH = Path("data/reports/btc_cross_exchange_forward_audit.json")
MUSCA_V8_BINANCE_AUDIT_PATH = Path("data/reports/musca_v8_binance_paper.json")
MUSCA_AUTO_MOE_AUDIT_PATH = Path("data/reports/musca_btc_auto_moe.json")
MUSCA_V5_ECONOMIC_ALPHA_PATH = Path("data/reports/btc_vwap_alpha_v1.json")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


LIVE_WARMUP_BARS = 288


def build_ml_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    status_source = source.with_name(f"{source.stem}.status.json")
    if not status_source.exists() and source.name == "ml_expert_research_v5.json":
        status_source = source.with_name("ml_expert_research.status.json")
    status: dict[str, Any] = {}
    with suppress(OSError, TypeError, json.JSONDecodeError):
        status = json.loads(status_source.read_text(encoding="utf-8"))
    if not source.exists():
        return {"available": False, "error": "No ML research run is available.", "status": status}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return {"available": True, "status": status, **payload}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"available": False, "error": f"Invalid ML report: {error}", "status": status}


def build_musca_v5_economic_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"available": False, "status": "NOT_TRAINED", "profiles": []}
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        if "vip_policy_audit" in report:
            audits = report.get("vip_policy_audit", {})
            profile_gates = report.get("vip_gates", {})
            profiles = []
            for level in range(6):
                name = f"VIP{level}"
                audit = audits.get(name, {})
                stress = audit.get("stress_2x", {})
                gates = profile_gates.get(name, {})
                profiles.append(
                    {
                        "profile": name,
                        "trades": audit.get("trades", 0),
                        "expectancy_bps": audit.get("expectancy_bps"),
                        "lcb_95_bps": audit.get("bootstrap_expectancy_lcb_95_bps"),
                        "profit_factor": audit.get("profit_factor"),
                        "max_drawdown": audit.get("max_drawdown"),
                        "stress_2x_expectancy_bps": stress.get("expectancy_bps"),
                        "base_financial_gates_passed": bool(gates)
                        and all(
                            bool(passed)
                            for gate, passed in gates.items()
                            if gate not in {"stress_2x_nonnegative", "walk_forward_trades_300"}
                        ),
                        "stress_gate_passed": bool(gates.get("stress_2x_nonnegative", False)),
                        "trade_count_gate_passed": bool(
                            gates.get("walk_forward_trades_300", False)
                        ),
                    }
                )
            return {
                "available": True,
                "status": report.get("status", "REPORT_INVALID"),
                "champion": "local Ridge/XGBoost experts",
                "challenger": "XGBoost GPU",
                "challenger_status": report.get("champions", {}),
                "base_viable_profiles": report.get("paper_eligible_profiles", []),
                "holdout_opened": bool(report.get("holdout_opened", False)),
                "profiles": profiles,
                "protocol_hash": report.get("protocol_hash"),
                "updated_at": report.get("updated_at"),
            }
        challenger = report.get("shadow_challenger", {})
        audits = challenger.get("vip_audit", {})
        profile_gates = challenger.get("profile_financial_gates", {})
        profiles = []
        for level in range(6):
            name = f"VIP{level}"
            audit = audits.get(name, {})
            normal = audit.get("audit_metrics", {})
            stress = audit.get("audit_cost_stress_2x_metrics", {})
            gates = profile_gates.get(name, {})
            profiles.append(
                {
                    "profile": name,
                    "trades": normal.get("trades", 0),
                    "expectancy_bps": normal.get("expectancy_bps"),
                    "lcb_95_bps": normal.get("expectancy_bootstrap_lcb_95_bps"),
                    "profit_factor": normal.get("profit_factor"),
                    "max_drawdown": normal.get("max_account_drawdown"),
                    "stress_2x_expectancy_bps": stress.get("expectancy_bps"),
                    "base_financial_gates_passed": bool(gates)
                    and all(
                        passed
                        for gate, passed in gates.items()
                        if gate != "cost_stress_2x_nonnegative"
                    ),
                    "stress_gate_passed": bool(gates.get("cost_stress_2x_nonnegative", False)),
                    "trade_count_gate_passed": int(normal.get("trades", 0)) >= 300,
                }
            )
        return {
            "available": True,
            "status": report.get("status", "REPORT_INVALID"),
            "champion": report.get("champion"),
            "challenger": challenger.get("model"),
            "challenger_status": challenger.get("status"),
            "base_viable_profiles": challenger.get("base_viable_profiles", []),
            "holdout_opened": False,
            "profiles": profiles,
            "updated_at": report.get("updated_at"),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "status": "REPORT_INVALID",
            "error": str(error),
            "profiles": [],
        }


def build_binance_alpha_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        if "historical_audit" in report:
            audit = report["historical_audit"]
            metrics = audit["metrics"]
            paper_gates = audit["paper_gates"]
            live_gates = audit["live_gates"]
            paper_passed = all(bool(value) for value in paper_gates.values())
            return {
                "available": True,
                "status": report["verdict"],
                "champion": f"{report['gating_champion']} adaptive expert gate",
                "challenger": "XGBoost rejected by the common OOS audit",
                "base_viable_profiles": ["BINANCE"] if paper_passed else [],
                "holdout_opened": bool(report.get("future_holdout_rows_read", 0)),
                "protocol_hash": report["protocol_hash"],
                "profiles": [
                    {
                        "profile": "BINANCE",
                        "trades": metrics["trades"],
                        "expectancy_bps": metrics["expectancy_bps"],
                        "lcb_95_bps": metrics["bootstrap_lcb_95_bps"],
                        "profit_factor": metrics["profit_factor"],
                        "max_drawdown": metrics["max_drawdown"],
                        "stress_2x_expectancy_bps": metrics[
                            "stress_2x_expectancy_bps"
                        ],
                        "base_financial_gates_passed": paper_passed,
                        "stress_gate_passed": bool(live_gates["stress_1_5x"]),
                        "trade_count_gate_passed": bool(
                            live_gates["minimum_historical_trades_300"]
                        ),
                    }
                ],
                "paper_gates": paper_gates,
                "live_gates": live_gates,
                "updated_at": report.get("created_at"),
            }
        alpha = report["alpha"]
        audit = alpha["paper_profiles"]["BINANCE"]
        normal = audit["oos_2026"]
        stress = audit["oos_2026_stress_2x"]
        return {
            "available": True,
            "status": alpha["status"],
            "champion": "frozen multi-horizon impulse/pullback policy",
            "challenger": "none authorized for execution",
            "base_viable_profiles": ["BINANCE"] if audit["paper_eligible"] else [],
            "holdout_opened": False,
            "protocol_hash": alpha["protocol_hash"],
            "profiles": [
                {
                    "profile": "BINANCE",
                    "trades": normal["trades"],
                    "expectancy_bps": normal["expectancy_bps"],
                    "lcb_95_bps": None,
                    "profit_factor": normal["profit_factor"],
                    "max_drawdown": normal["max_drawdown"],
                    "stress_2x_expectancy_bps": stress["expectancy_bps"],
                    "base_financial_gates_passed": bool(audit["paper_eligible"]),
                    "stress_gate_passed": stress["expectancy_bps"] >= 0,
                    "trade_count_gate_passed": normal["trades"] >= 100,
                }
            ],
            "updated_at": report.get("updated_at"),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"available": False, "status": "REPORT_INVALID", "error": str(error)}


def build_research_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    status_source = source.with_name(f"{source.stem}.status.json")
    status: dict[str, Any] = {}
    with suppress(OSError, TypeError, json.JSONDecodeError):
        status = json.loads(status_source.read_text(encoding="utf-8"))
    if not source.exists():
        return {
            "available": False,
            "error": "No completed research run is available yet.",
            "status": status,
        }
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        evaluations = payload["evaluations"]
        return {
            "available": True,
            "run_id": payload["run_id"],
            "completed_at": payload["completed_at"],
            "shadow_updated_at": payload.get("shadow_updated_at", payload["completed_at"]),
            "data_start": payload["data_start"],
            "data_end": payload["data_end"],
            "fixed": payload["fixed"],
            "selection_bias": payload.get("selection_bias", {}),
            "champions": payload.get("champions", []),
            "evaluations": evaluations,
            "counts": {
                status: sum(item["status"] == status for item in evaluations)
                for status in ("validated", "provisional", "insufficient")
            },
            "status": status,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"available": False, "error": f"Invalid research report: {error}"}


def build_live_market_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"available": False, "error": "Waiting for the Bitunix collector."}
    rows: dict[int, dict[str, Any]] = {}
    invalid_rows = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            envelope = json.loads(line)
            candle = envelope["candle"]
            timestamp_ms = int(candle["time"])
            values = {name: float(candle[name]) for name in ("open", "high", "low", "close")}
            volume = float(candle.get("quoteVol", candle.get("volume", 0)))
            quote_volume = float(candle.get("baseVol", values["close"] * volume))
            if min(*values.values()) <= 0 or volume <= 0 or quote_volume <= 0:
                raise ValueError("invalid OHLCV")
            envelope_high = max(values["open"], values["high"], values["close"])
            envelope_low = min(values["open"], values["low"], values["close"])
            deviation_bps = (
                ((envelope_high - values["high"]) + (values["low"] - envelope_low))
                / values["close"]
                * 10_000
            )
            if (
                not all(math.isfinite(value) for value in (*values.values(), volume, quote_volume))
                or min(*values.values(), volume) < 0
                or values["high"] < values["low"]
                or deviation_bps > 1
            ):
                raise ValueError("invalid OHLCV")
            if envelope_high != values["high"] or envelope_low != values["low"]:
                invalid_rows += 1
                values["high"] = envelope_high
                values["low"] = envelope_low
            rows[timestamp_ms] = {
                "timestamp": datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat(),
                **values,
                "volume": volume,
                "quote_volume": quote_volume,
                "collected_at": envelope["collected_at"],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid_rows += 1
    candles = [rows[key] for key in sorted(rows)]
    if not candles:
        return {"available": False, "error": "No valid closed Bitunix candles yet."}
    frame = pd.DataFrame(candles)
    atr_values = atr(frame["high"], frame["low"], frame["close"])
    session = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")
    session_volume = frame["volume"].groupby(session).cumsum()
    daily_vwap = frame["quote_volume"].groupby(session).cumsum() / session_volume
    center = rolling_vwap(
        frame["high"], frame["low"], frame["close"], frame["volume"], LIVE_WARMUP_BARS
    )
    adx_values = adx(frame["high"], frame["low"], frame["close"])["adx"]
    slope = normalized_ema_slope(frame["close"], atr_values)
    z_score = normalized_distance(frame["close"], center, atr_values)
    for index, candle in enumerate(candles):
        center_value = _finite_or_none(center.iloc[index])
        atr_value = _finite_or_none(atr_values.iloc[index])
        candle["center"] = center_value
        candle["daily_vwap"] = _finite_or_none(daily_vwap.iloc[index])
        candle["lower_band"] = (
            None if center_value is None or atr_value is None else center_value - 2 * atr_value
        )
        candle["upper_band"] = (
            None if center_value is None or atr_value is None else center_value + 2 * atr_value
        )
    latest = candles[-1] | {
        "atr": _finite_or_none(atr_values.iloc[-1]),
        "adx": _finite_or_none(adx_values.iloc[-1]),
        "ema_slope": _finite_or_none(slope.iloc[-1]),
        "z_score": _finite_or_none(z_score.iloc[-1]),
        "atr_percentile": None,
        "spread_bps": None,
        "regime": "unknown",
    }
    quote = _live_quote(source.with_suffix(".quote.json"))
    latest["spread_bps"] = quote.get("spread_bps")
    collected_at = datetime.fromisoformat(latest["collected_at"])
    age_seconds = max(0, (datetime.now(UTC) - collected_at).total_seconds())
    ready = len(candles) >= LIVE_WARMUP_BARS and latest["center"] is not None
    return {
        "available": True,
        "status": "live" if age_seconds <= 420 else "stale",
        "age_seconds": round(age_seconds),
        "bars": len(candles),
        "warmup_bars": LIVE_WARMUP_BARS,
        "ready": ready,
        "invalid_rows": invalid_rows,
        "activity": (
            "Indicators ready; paper decisions use closed candles and the observed spread."
            if ready
            else f"Collecting strategy warm-up data: {len(candles)}/{LIVE_WARMUP_BARS} closed bars."
        ),
        "latest": latest,
        "candles": candles[-160:],
        "quote": quote,
    }


def _finite_or_none(value: Any) -> float | None:
    return float(value) if pd.notna(value) else None


def _live_quote(path: Path) -> dict[str, Any]:
    try:
        quote = json.loads(path.read_text(encoding="utf-8"))
        bid = float(quote["best_bid"])
        ask = float(quote["best_ask"])
        spread = float(quote["spread_bps"])
        observed_at = datetime.fromisoformat(quote["observed_at"])
        if not (0 < bid < ask and spread >= 0 and observed_at.tzinfo is not None):
            raise ValueError("invalid quote")
        return {
            "available": True,
            "best_bid": bid,
            "best_ask": ask,
            "spread_bps": spread,
            "observed_at": observed_at.astimezone(UTC).isoformat(),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"available": False}


def build_dashboard_payload(
    report_path: str | Path, profile: str | float | None = None
) -> dict[str, Any]:
    base_path = Path(report_path)
    profile_id = f"adx{profile:g}" if isinstance(profile, (int, float)) else profile
    binance_profile = profile_id == "musca-v5-binance"
    musca_v5_profile = (
        binance_profile or profile_id == "musca-v5" or str(profile_id).startswith("musca-v5-vip")
    )
    selected_fee_profile = (
        "BINANCE"
        if binance_profile
        else str(profile_id).rsplit("-", 1)[-1].upper()
        if str(profile_id).startswith("musca-v5-vip")
        else "VIP0"
    )
    path = (
        V14_SHADOW_REPORT_PATH
        if profile_id == "v14-vwap"
        else MUSCA_LIQUIDITY_REPORT_PATH
        if profile_id == "musca-vwap-liquidity"
        else MUSCA_V8_BINANCE_REPORT_PATH
        if binance_profile
        else MUSCA_V5_REPORT_PATH
        if musca_v5_profile
        else MUSCA_V4_REPORT_PATH
        if profile_id == "musca-v4"
        else MUSCA_V2_REPORT_PATH
        if profile_id == "musca-v2"
        else MUSCA_REPORT_PATH
        if profile_id == "musca"
        else profile_report_path(base_path, profile_id)
        if profile_id in {WEIGHTED_PROFILE, WEIGHTED_V11_PROFILE, WEIGHTED_V2_PROFILE}
        or profile_id in {f"adx{x:g}" for x in ADX_VARIANTS}
        else base_path
    )
    if not path.exists() and base_path.exists():
        path = base_path
    if not path.exists():
        return {
            "available": False,
            "error": f"No report found at {path}",
            "generated_at": datetime.now(UTC).isoformat(),
            "milestones": MILESTONES,
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "error": f"Report cannot be read: {error}",
            "generated_at": datetime.now(UTC).isoformat(),
            "milestones": MILESTONES,
        }
    telemetry = report.get("telemetry", [])
    equity_curve = report.get("equity_curve", [])
    fills = report.get("fills", [])
    musca = report.get("strategy_profile") == "musca_v8_momentum_short_control"
    musca_v2 = report.get("strategy_profile") == "musca_v2_long_horizon_vwap"
    musca_v4 = report.get("strategy_profile") == "musca_v4_multi_anchor_vwap"
    musca_v5 = report.get("strategy_profile") in {
        "musca_v5_stable_multi_horizon_vwap",
        "musca_btc_auto_moe_vwap",
    }
    musca_liquidity = report.get("strategy_profile") == "musca_vwap_liquidity_filtered"
    v14_shadow = report.get("strategy_profile") == "v14_vwap_diagnostic_shadow"
    latest = telemetry[-1] if telemetry else None
    operations, position = _operations(fills)
    forward_audit = (
        build_ml_payload(MUSCA_V8_BINANCE_AUDIT_PATH)
        if binance_profile
        else build_ml_payload(CROSS_EXCHANGE_AUDIT_PATH)
        if musca_v5
        else {}
    )
    if forward_audit:
        forward_audit["selected_fee_profile"] = selected_fee_profile
    return {
        "available": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "report_updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "summary": {
            "mode": report.get("mode", "backtest"),
            "instrument": report.get("instrument", "—"),
            "timeframe_minutes": report.get("timeframe_minutes"),
            "range_adx_threshold": report.get("range_adx_threshold", 20),
            "profile_id": (
                "v14-vwap"
                if v14_shadow
                else "musca-vwap-liquidity"
                if musca_liquidity
                else str(profile_id or "musca-v5-vip0")
                if musca_v5
                else "musca-v4"
                if musca_v4
                else "musca-v2"
                if musca_v2
                else "musca"
                if musca
                else WEIGHTED_V2_PROFILE
                if report.get("strategy_profile") == "weighted_reversion_v2"
                else WEIGHTED_V11_PROFILE
                if report.get("strategy_profile") == "weighted_reversion_v11"
                else WEIGHTED_PROFILE
                if report.get("strategy_profile") == "weighted_reversion"
                else f"adx{report.get('range_adx_threshold', 20):g}"
            ),
            "profile_label": (
                "V14 VWAP · LONG/SHORT diagnostic"
                if v14_shadow
                else "MUSCA VWAP · Liquidity Shadow"
                if musca_liquidity
                else "MUSCA BTC · AUTO-MoE PAPER"
                if binance_profile
                else "MUSCA V5 · BTC VWAP Alpha"
                if musca_v5
                else "MUSCA V4 · Multi-Anchor VWAP"
                if musca_v4
                else "MUSCA V2 · Trend + VWAP"
                if musca_v2
                else "MUSCA · Momentum SHORT V8"
                if musca
                else "MR SCORE v2"
                if report.get("strategy_profile") == "weighted_reversion_v2"
                else "MR SCORE v1.1"
                if report.get("strategy_profile") == "weighted_reversion_v11"
                else "MR SCORE v1"
                if report.get("strategy_profile") == "weighted_reversion"
                else f"ADX {report.get('range_adx_threshold', 20):g}"
            ),
            "initial_equity": report.get("initial_equity", "0"),
            "final_equity": report.get("final_equity", "0"),
            "net_pnl": report.get("net_pnl", "0"),
            "gross_pnl": report.get("gross_pnl", "0"),
            "max_drawdown": report.get("max_drawdown", "0"),
            "fees": report.get("fees", "0"),
            "slippage": report.get("slippage", "0"),
            "signals": report.get("signals", 0),
            "rejected_signals": report.get("rejected_signals", 0),
            "kill_switches": report.get("kill_switches", 0),
            "risk_per_trade": report.get("risk_per_trade", "0.01"),
            "max_daily_loss": report.get("max_daily_loss", "0.02"),
            "max_weekly_loss": report.get("max_weekly_loss", "0.10"),
            "fills": len(fills),
            "operations": len(operations),
            "validation_status": report.get("validation_status"),
            "cost_scenarios": report.get("cost_scenarios", {}),
            "policy_action": report.get("policy_action"),
            "benchmark_venue": report.get("benchmark_venue"),
            "execution_venue": report.get("execution_venue"),
            "vwap_session": report.get("vwap_session"),
            "model_input_readiness": report.get("model_input_readiness", {}),
            "historical_replay": report.get("historical_replay", {}),
            "fee_profile": selected_fee_profile if musca_v5 else None,
            "forward_audit": forward_audit,
            "economic_alpha": build_binance_alpha_payload(MUSCA_AUTO_MOE_AUDIT_PATH)
            if binance_profile
            else build_musca_v5_economic_payload(MUSCA_V5_ECONOMIC_ALPHA_PATH)
            if musca_v5
            else {},
        },
        "latest": latest,
        "current_position": position,
        "no_trade_reason": (
            None
            if operations
            else f"No order was filled. Latest decision: {(latest or {}).get('activity', 'none')}"
        ),
        "equity_curve": _sample(equity_curve, 500),
        "telemetry": telemetry[-160:],
        "fills": fills[-50:][::-1],
        "operations": operations[-50:][::-1],
        "milestones": MILESTONES,
        "variants": _variant_summaries(base_path),
        "safety": {
            "live_enabled": False,
            "risk_per_trade": float(report.get("risk_per_trade", 0.01)),
            "max_drawdown_limit": 0.08,
            "max_daily_loss": 0.02,
            "max_weekly_loss": 0.10,
            "max_open_positions": 1,
        },
    }


def _variant_summaries(report_path: Path) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    try:
        report = json.loads(MUSCA_V8_BINANCE_REPORT_PATH.read_text(encoding="utf-8"))
        audit = json.loads(MUSCA_V8_BINANCE_AUDIT_PATH.read_text(encoding="utf-8"))
        account = audit["one_position_diagnostics"]["paper_account"]
        variants.append(
            {
                "range_adx_threshold": None,
                "profile_id": "musca-v5-binance",
                "profile_label": "MUSCA BTC · AUTO-MoE PAPER",
                "final_equity": account.get("final_equity", 10_000),
                "net_pnl": account.get("net_pnl", 0),
                "max_drawdown": account.get("max_drawdown", 0),
                "signals": report.get("signals", 0),
                "operations": len(account.get("trades", [])),
            }
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        pass
    for threshold in ADX_VARIANTS:
        path = variant_report_path(report_path, threshold)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": threshold,
                "profile_id": f"adx{threshold:g}",
                "profile_label": f"ADX {threshold:g}",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    for profile, label in (
        (WEIGHTED_PROFILE, "MR SCORE v1"),
        (WEIGHTED_V11_PROFILE, "MR SCORE v1.1"),
        (WEIGHTED_V2_PROFILE, "MR SCORE v2"),
    ):
        path = profile_report_path(report_path, profile)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": report.get("range_adx_threshold", 20),
                "profile_id": profile,
                "profile_label": label,
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    try:
        report = json.loads(MUSCA_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": 20,
                "profile_id": "musca",
                "profile_label": "MUSCABOT · Momentum SHORT 5M",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        report = json.loads(MUSCA_V2_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": 20,
                "profile_id": "musca-v2",
                "profile_label": "MUSCA V2 · Trend + VWAP",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        report = json.loads(MUSCA_V4_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": 20,
                "profile_id": "musca-v4",
                "profile_label": "MUSCA V4 · Multi-Anchor VWAP",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        report = json.loads(MUSCA_V5_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": 20,
                "profile_id": "musca-v5",
                "profile_label": "MUSCA V5 · Bitunix Net EV V4",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        audit = json.loads(CROSS_EXCHANGE_AUDIT_PATH.read_text(encoding="utf-8"))
        accounts = audit.get("one_position_diagnostics", {}).get("paper_accounts", {})
        accepted = audit.get("alpha", {}).get("accepted_candidates_by_profile", {})
        for level in range(6):
            fee_profile = f"VIP{level}"
            account = accounts.get(fee_profile, {})
            variants.append(
                {
                    "range_adx_threshold": 20,
                    "profile_id": f"musca-v5-vip{level}",
                    "profile_label": f"MUSCA V5 · {fee_profile}",
                    "final_equity": account.get("final_equity", 10_000),
                    "net_pnl": account.get("net_pnl", 0),
                    "max_drawdown": account.get("max_drawdown", 0),
                    "signals": accepted.get(fee_profile, 0),
                    "operations": len(account.get("trades", [])),
                }
            )
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    try:
        report = json.loads(MUSCA_LIQUIDITY_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": None,
                "profile_id": "musca-vwap-liquidity",
                "profile_label": "MUSCA VWAP · Liquidity Shadow",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        report = json.loads(V14_SHADOW_REPORT_PATH.read_text(encoding="utf-8"))
        operations, _ = _operations(report.get("fills", []))
        variants.append(
            {
                "range_adx_threshold": 25,
                "profile_id": "v14-vwap",
                "profile_label": "V14 VWAP SHADOW · LONG/SHORT 5M",
                "final_equity": report.get("final_equity", "0"),
                "net_pnl": report.get("net_pnl", "0"),
                "max_drawdown": report.get("max_drawdown", "0"),
                "signals": report.get("signals", 0),
                "operations": len(operations),
            }
        )
    except (OSError, json.JSONDecodeError):
        pass
    return variants


def _operations(fills: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    signed_position = Decimal("0")
    operations: list[dict[str, Any]] = []
    for fill in fills:
        quantity = Decimal(str(fill["quantity"]))
        before = signed_position
        signed_position += quantity if fill["side"] == "buy" else -quantity
        if before == 0:
            event = "OPEN LONG" if signed_position > 0 else "OPEN SHORT"
        elif signed_position == 0:
            event = "CLOSE"
        elif before * signed_position < 0:
            event = "REVERSE"
        elif abs(signed_position) < abs(before):
            event = "REDUCE"
        else:
            event = "INCREASE"
        operations.append(
            {
                "timestamp": fill["exchange_timestamp"],
                "event": event,
                "side": fill["side"],
                "quantity": str(quantity),
                "price": fill["price"],
                "position_after": str(signed_position),
                "details": (
                    f"Fee {fill.get('commission', '0')} · slippage {fill.get('slippage', '0')} · "
                    f"{fill['client_order_id']}"
                ),
            }
        )
    side = "LONG" if signed_position > 0 else "SHORT" if signed_position < 0 else "FLAT"
    return operations, {"status": side, "quantity": str(abs(signed_position))}


def _sample(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return values
    step = max(1, len(values) // maximum)
    sampled = values[::step]
    return sampled if sampled[-1] is values[-1] else [*sampled, values[-1]]


def serve_dashboard(
    report_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    live_data_path: str | Path = LIVE_DATA_PATH,
    research_report_path: str | Path = Path("data/reports/research.json"),
    ml_report_path: str | Path = Path("data/reports/ml_expert_research_v5.json"),
) -> None:
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("dashboard host must be loopback-only")
    handler = _handler(
        Path(report_path), Path(live_data_path), Path(research_report_path), Path(ml_report_path)
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Dashboard available at http://{host}:{port}")
        server.serve_forever()


def _handler(
    report_path: Path,
    live_data_path: Path,
    research_report_path: Path,
    ml_report_path: Path,
) -> type[BaseHTTPRequestHandler]:
    assets = files("adaptive_bot.dashboard.static")
    routes = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/styles-v5.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/app-v5.js": ("app.js", "text/javascript; charset=utf-8"),
    }
    response_cache: dict[str, tuple[float, bytes]] = {}
    response_cache_lock = Lock()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path)
            path = request.path
            if path == "/api/dashboard":
                query = parse_qs(request.query)
                profile = query.get("profile", query.get("adx", [None]))[0]
                self._send_cached_json(
                    f"dashboard:{profile}",
                    lambda: build_dashboard_payload(report_path, profile),
                )
                return
            if path == "/api/live":
                self._send_cached_json("live", lambda: build_live_market_payload(live_data_path))
                return
            if path == "/api/research":
                self._send_json(build_research_payload(research_report_path))
                return
            if path == "/api/ml":
                self._send_json(build_ml_payload(ml_report_path))
                return
            if path == "/api/health":
                self._send_json({"status": "healthy", "timestamp": datetime.now(UTC).isoformat()})
                return
            if path == "/mobile-check":
                self._send(
                    b"<!doctype html><meta name=viewport content='width=device-width'>"
                    b"<h1>BotTone OK</h1>"
                    b"<p>Tailscale and dashboard server are reachable.</p>",
                    "text/html; charset=utf-8",
                )
                return
            asset = routes.get(path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            name, content_type = asset
            self._send(assets.joinpath(name).read_bytes(), content_type)

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send(self._json_body(payload), "application/json")

        def _send_cached_json(self, key: str, build: Callable[[], dict[str, Any]]) -> None:
            now = time.monotonic()
            with response_cache_lock:
                cached = response_cache.get(key)
            if cached is not None and now - cached[0] < 4.5:
                self._send(cached[1], "application/json")
                return
            body = self._json_body(build())
            with response_cache_lock:
                response_cache[key] = (now, body)
            self._send(body, "application/json")

        @staticmethod
        def _json_body(payload: dict[str, Any]) -> bytes:
            return json.dumps(_json_safe(payload), separators=(",", ":"), allow_nan=False).encode()

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self' wss://fstream.binance.com; "
                "img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return DashboardHandler
