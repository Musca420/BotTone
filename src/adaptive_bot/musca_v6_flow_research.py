from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import COST_BPS, _metrics, _non_overlapping
from adaptive_bot.musca_v5_micro_model import load_5s
from adaptive_bot.musca_v5_research import BARS, _models, build_features

ROOT = Path("data/ml/musca_v6")
REPORT = Path("data/reports/musca_v6_flow_research.json")
STATUS = Path("data/reports/musca_v6_flow_research.status.json")
EVENT_Z = 1.5
PROTOCOL = {
    "name": "musca_v6_order_flow_vwap_context_v1",
    "market": "BINANCE_BTCUSDT_PERPETUAL",
    "decision": "causal_1m_order_flow_or_velocity_event",
    "event_z": EVENT_Z,
    "actions": ["LONG_30M", "SHORT_30M", "LONG_60M", "SHORT_60M"],
    "entry": "first_5s_close_available_after_decision",
    "exit": "30m_or_60m_time_exit_with_2ATR_adaptive_protective_stop",
    "vwap_role": "daily_and_weekly_context_benchmark_not_direction_rule",
    "cost_bps": COST_BPS,
    "stress_cost_bps": 2 * COST_BPS,
    "fit": "2026-01",
    "calibration": "2026-02",
    "development_validation": "2026-03",
    "unopened_audit": "2026-04",
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
FEATURES = (
    "action_direction",
    "horizon_minutes",
    "ofi_1m",
    "ofi_5m",
    "ofi_z",
    "flow_acceleration",
    "trade_intensity",
    "return_1m_bps",
    "return_5m_bps",
    "velocity_z",
    "absorption",
    "distance_daily_vwap_atr",
    "distance_weekly_vwap_atr",
    "daily_vwap_slope_atr",
    "trend_vote",
    "taker_imbalance_15m",
    "spot_taker_imbalance_15m",
    "relative_volume",
    "atr_percentile",
    "funding_z",
    "basis_bps",
    "oi_change_1h",
    "hour_sin",
    "hour_cos",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": percent,
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def build_states(five_seconds: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    minute = (
        five_seconds.set_index("timestamp")
        .resample("1min")
        .agg(
            quote_volume=("quote_volume", "sum"),
            signed_quote_volume=("signed_quote_volume", "sum"),
            trade_count=("trade_count", "sum"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .dropna()
        .reset_index()
    )
    minute["available_at"] = minute["timestamp"] + pd.Timedelta(minutes=1)
    minute["ofi_1m"] = minute["signed_quote_volume"] / minute["quote_volume"].replace(
        0, np.nan
    )
    minute["ofi_5m"] = minute["signed_quote_volume"].rolling(5).sum() / minute[
        "quote_volume"
    ].rolling(5).sum().replace(0, np.nan)
    minute["flow_acceleration"] = minute["ofi_1m"] - minute["ofi_5m"]
    baseline = minute["trade_count"].shift(1).rolling(1_440, min_periods=480).median()
    minute["trade_intensity"] = minute["trade_count"] / baseline.replace(0, np.nan)
    minute["return_1m_bps"] = minute["close"].pct_change() * 10_000
    minute["return_5m_bps"] = minute["close"].pct_change(5) * 10_000
    minute["absorption"] = minute["ofi_1m"].abs() / (
        minute["return_1m_bps"].abs() + 0.5
    )
    for source, target in (("ofi_1m", "ofi_z"), ("return_1m_bps", "velocity_z")):
        prior = minute[source].shift(1).rolling(1_440, min_periods=480)
        minute[target] = (minute[source] - prior.mean()) / prior.std().replace(0, np.nan)

    context = build_features(bars)
    context["available_at"] = pd.to_datetime(context["available_at"], utc=True)
    context["distance_daily_vwap_atr"] = (
        context["perp_close"] - context["daily_vwap"]
    ) / context["atr"]
    context["distance_weekly_vwap_atr"] = (
        context["perp_close"] - context["weekly_vwap"]
    ) / context["atr"]
    context["daily_vwap_slope_atr"] = context["daily_vwap"].diff(3) / context["atr"]
    columns = [
        "available_at",
        "atr",
        "distance_daily_vwap_atr",
        "distance_weekly_vwap_atr",
        "daily_vwap_slope_atr",
        "trend_vote",
        "taker_imbalance_15m",
        "spot_taker_imbalance_15m",
        "relative_volume",
        "atr_percentile",
        "funding_z",
        "basis_bps",
        "oi_change_1h",
    ]
    result = pd.merge_asof(
        minute.sort_values("available_at"),
        context[columns].sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    hour = result["available_at"].dt.hour + result["available_at"].dt.minute / 60
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    required = [
        feature
        for feature in FEATURES
        if feature not in {"action_direction", "horizon_minutes"}
    ]
    result = result.dropna(subset=[*required, "atr"])
    return result.loc[result[["ofi_z", "velocity_z"]].abs().max(axis=1).ge(EVENT_Z)]


def label_actions(
    states: pd.DataFrame,
    path: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (30, 60),
) -> pd.DataFrame:
    path = path.sort_values("timestamp").reset_index(drop=True)
    timestamp_ns = pd.to_datetime(path["timestamp"], utc=True).to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    high, low, close = (path[column].to_numpy(float) for column in ("high", "low", "close"))
    available = pd.to_datetime(path["available_at"], utc=True)
    rows: list[dict[Any, Any]] = []
    for state in states.to_dict("records"):
        entry_index = int(
            np.searchsorted(timestamp_ns, pd.Timestamp(state["available_at"]).value, side="left")
        )
        if entry_index >= len(path):
            continue
        entry = close[entry_index]
        risk = 2 * float(state["atr"])
        if not 12 <= risk / entry * 10_000 <= 300:
            continue
        for horizon in horizons:
            last = min(entry_index + horizon * 12, len(path) - 1)
            for side in (-1, 1):
                stop = entry - side * risk
                adverse = (
                    low[entry_index + 1 : last + 1]
                    if side > 0
                    else high[entry_index + 1 : last + 1]
                )
                hit = (
                    np.flatnonzero(adverse <= stop)
                    if side > 0
                    else np.flatnonzero(adverse >= stop)
                )
                exit_index = entry_index + 1 + int(hit[0]) if hit.size else last
                exit_price = stop if hit.size else close[exit_index]
                gross = side * (exit_price - entry) / entry * 10_000
                rows.append(
                    state
                    | {
                        "protocol_hash": PROTOCOL_HASH,
                        "signal_timestamp": state["available_at"],
                        "action_direction": side,
                        "horizon_minutes": horizon,
                        "entry_timestamp": available.iat[entry_index],
                        "exit_timestamp": available.iat[exit_index],
                        "entry_price": entry,
                        "stop_price": stop,
                        "gross_return_bps": gross,
                        "net_return_bps": gross - COST_BPS,
                        "stress_return_bps": gross - 2 * COST_BPS,
                        "net_return_r": (gross - COST_BPS) / (risk / entry * 10_000),
                    }
                )
    return pd.DataFrame(rows)


def _select(rows: pd.DataFrame, scores: np.ndarray, coverage: float) -> tuple[pd.DataFrame, float]:
    threshold = float(np.quantile(scores, 1 - coverage))
    candidates = rows.loc[scores >= threshold].copy()
    candidates["score"] = scores[scores >= threshold]
    best = candidates.sort_values("score").groupby("available_at", as_index=False).tail(1)
    return _non_overlapping(best), threshold


def train() -> dict[str, Any]:
    _status("states", 10, "causal one-minute order-flow states")
    path = load_5s()
    states = build_states(path, pd.read_parquet(BARS))
    _status("labels", 35, f"{len(states)} preregistered market events")
    actions = label_actions(states, path)
    ROOT.mkdir(parents=True, exist_ok=True)
    actions.to_parquet(ROOT / "actions.parquet", index=False)
    timestamp = pd.to_datetime(actions["entry_timestamp"], utc=True)
    fit = actions.loc[timestamp.lt("2026-02-01")]
    calibration = actions.loc[timestamp.ge("2026-02-01") & timestamp.lt("2026-03-01")]
    validation = actions.loc[timestamp.ge("2026-03-01") & timestamp.lt("2026-04-01")]
    _status(
        "gpu",
        60,
        f"fit={len(fit)} calibration={len(calibration)} validation={len(validation)}",
    )
    choices: list[tuple[float, str, float, Any, dict[str, float]]] = []
    audits: list[dict[str, Any]] = []
    for name, model in _models().items():
        if hasattr(model, "predict_proba"):
            continue
        model.fit(fit[list(FEATURES)], fit["net_return_bps"])
        scores = np.asarray(model.predict(calibration[list(FEATURES)]))
        for coverage in (0.005, 0.01, 0.02, 0.05):
            selected, threshold = _select(calibration, scores, coverage)
            metrics = _metrics(selected)
            stress = _metrics(selected, "stress_return_bps")
            audit = {
                "model": name,
                "coverage": coverage,
                "threshold": threshold,
                "metrics": metrics,
                "stress": stress,
            }
            audits.append(audit)
            if (
                len(selected) / 28 >= 0.5
                and stress["expectancy_bps"] > 0
                and stress["profit_factor"] >= 1.10
            ):
                choices.append((stress["expectancy_bps"], name, threshold, model, metrics))
    selected = validation.iloc[:0].copy()
    champion = "FLAT"
    calibration_metrics = _metrics(selected)
    if choices:
        _, champion, threshold, model, calibration_metrics = max(choices, key=lambda row: row[0])
        scores = np.asarray(model.predict(validation[list(FEATURES)]))
        candidates = validation.loc[scores >= threshold].copy()
        candidates["score"] = scores[scores >= threshold]
        selected = _non_overlapping(
            candidates.sort_values("score").groupby("available_at", as_index=False).tail(1)
        )
    metrics = _metrics(selected)
    stress = _metrics(selected, "stress_return_bps")
    gates = {
        "validation_trades_15": len(selected) >= 15,
        "frequency_0_5_day": len(selected) / 31 >= 0.5,
        "expectancy_positive": metrics["expectancy_bps"] > 0,
        "profit_factor_1_10": metrics["profit_factor"] >= 1.10,
        "stress_nonnegative": stress["expectancy_bps"] >= 0,
        "drawdown_8pct": metrics["max_drawdown"] <= 0.08,
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": {
            "states": len(states),
            "actions": len(actions),
            "fit": len(fit),
            "calibration": len(calibration),
            "validation": len(validation),
        },
        "calibration_audit": audits,
        "champion": champion,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": metrics,
        "validation_stress_metrics": stress,
        "gates": gates,
        "verdict": "READY_FOR_NEW_APRIL_AUDIT" if all(gates.values()) else "NO_FLOW_EDGE",
        "april_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    _status("complete", 100, str(report["verdict"]))
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
