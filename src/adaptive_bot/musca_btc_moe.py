from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

from adaptive_bot.btc_vwap_alpha import _historical_features

SOURCE = Path("data/ml/hybrid_v25/asset=BTCUSDT/minutes.parquet")
ROOT = Path("data/ml/musca_btc_moe")
MATRIX = ROOT / "matrix.parquet"
CHECKPOINTS = ROOT / "checkpoints"
REPORT = Path("data/reports/musca_btc_moe.json")
STATUS = Path("data/reports/musca_btc_moe.status.json")
BUNDLE = Path("data/models/musca_btc_moe/research_bundle.joblib")

SYMBOL = "BTCUSDT"
HORIZONS = (5, 15, 30, 60)
SIDES = (1, -1)
FINAL_SEEDS = (20260810, 20260811, 20260812, 20260813, 20260814)
ROUND_TRIP_COST_BPS = 11.0
MINIMUM_NET_TARGET_BPS = 2.0
MAX_TARGET_BPS = 300.0
MAX_STOP_BPS = 200.0
FUTURE_HOLDOUT_START = pd.Timestamp("2026-08-10T00:00:00Z")
META_START = pd.Timestamp("2025-01-01T00:00:00Z")
META_END = pd.Timestamp("2026-01-01T00:00:00Z")
MODEL_AUDIT_END = pd.Timestamp("2026-02-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-03-01T00:00:00Z")
POLICY_SELECTION_END = pd.Timestamp("2026-05-01T00:00:00Z")
HISTORICAL_AUDIT_END = pd.Timestamp("2026-08-04T00:00:00Z")
PURGE = pd.Timedelta(minutes=max(HORIZONS))
COVERAGES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

FEATURES = (
    "return_1m_bps",
    "return_2m_bps",
    "return_3m_bps",
    "return_5m_bps",
    "return_10m_bps",
    "return_15m_bps",
    "return_30m_bps",
    "return_60m_bps",
    "vwap_distance_bps",
    "vwap_distance_5m_bps",
    "vwap_distance_15m_bps",
    "vwap_distance_240m_bps",
    "vwap_slope_bps",
    "vwap_slope_change_bps",
    "vwap_distance_velocity_3m_bps",
    "vwap_tests_30m",
    "vwap_rejections_30m",
    "time_since_vwap_cross_minutes",
    "vwap_rejection_strength_bps",
    "vwap_band_position",
    "range_60s_bps",
    "atr_1m_bps",
    "atr_5m_bps",
    "atr_15m_bps",
    "atr_30m_bps",
    "realized_volatility_30m_bps",
    "volatility_percentile",
    "volume_percentile",
    "taker_imbalance_60s",
    "taker_imbalance_15m",
    "taker_imbalance_60m",
    "taker_imbalance_change_5m",
    "trade_count_zscore",
    "aggressive_volume_zscore",
    "candle_body_bps",
    "wick_imbalance_bps",
    "efficiency_15m",
    "efficiency_60m",
    "range_position_15m",
    "range_position_60m",
    "oi_change_1h",
    "return_oi_interaction_raw",
    "basis_bps",
    "funding_z",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)

VWAP_VIEW = (
    "return_1m_bps",
    "return_5m_bps",
    "return_15m_bps",
    "vwap_distance_bps",
    "vwap_distance_5m_bps",
    "vwap_distance_15m_bps",
    "vwap_distance_240m_bps",
    "vwap_slope_bps",
    "vwap_slope_change_bps",
    "vwap_distance_velocity_3m_bps",
    "vwap_tests_30m",
    "vwap_rejections_30m",
    "time_since_vwap_cross_minutes",
    "vwap_rejection_strength_bps",
    "vwap_band_position",
    "atr_5m_bps",
    "realized_volatility_30m_bps",
)
TREND_VIEW = (
    "return_1m_bps",
    "return_2m_bps",
    "return_3m_bps",
    "return_5m_bps",
    "return_10m_bps",
    "return_15m_bps",
    "return_30m_bps",
    "return_60m_bps",
    "vwap_slope_bps",
    "efficiency_15m",
    "efficiency_60m",
    "range_position_15m",
    "range_position_60m",
)
FLOW_VIEW = (
    "return_1m_bps",
    "return_5m_bps",
    "taker_imbalance_60s",
    "taker_imbalance_15m",
    "taker_imbalance_60m",
    "taker_imbalance_change_5m",
    "trade_count_zscore",
    "aggressive_volume_zscore",
    "volume_percentile",
    "candle_body_bps",
    "wick_imbalance_bps",
)
REGIME_VIEW = (
    "return_5m_bps",
    "return_30m_bps",
    "return_60m_bps",
    "range_60s_bps",
    "atr_1m_bps",
    "atr_5m_bps",
    "atr_15m_bps",
    "atr_30m_bps",
    "realized_volatility_30m_bps",
    "volatility_percentile",
    "volume_percentile",
    "oi_change_1h",
    "return_oi_interaction_raw",
    "basis_bps",
    "funding_z",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
VIEWS: dict[str, tuple[str, ...]] = {
    "full": FEATURES,
    "vwap": VWAP_VIEW,
    "trend": TREND_VIEW,
    "flow": FLOW_VIEW,
    "regime": REGIME_VIEW,
}
DIRECTIONAL_FEATURES = frozenset(
    {
        *(name for name in FEATURES if name.startswith("return_")),
        *(name for name in FEATURES if name.startswith("vwap_distance")),
        "vwap_slope_bps",
        "vwap_slope_change_bps",
        "vwap_distance_velocity_3m_bps",
        "vwap_rejection_strength_bps",
        "vwap_band_position",
        "taker_imbalance_60s",
        "taker_imbalance_15m",
        "taker_imbalance_60m",
        "taker_imbalance_change_5m",
        "candle_body_bps",
        "wick_imbalance_bps",
        "range_position_15m",
        "range_position_60m",
        "oi_change_1h",
        "return_oi_interaction_raw",
        "basis_bps",
        "funding_z",
    }
)
EXPERT_COLUMNS = tuple(f"expert_{horizon}m_{view}" for horizon in HORIZONS for view in VIEWS)
GATING_CONTEXT = (
    "vwap_distance_bps",
    "vwap_tests_30m",
    "vwap_rejections_30m",
    "time_since_vwap_cross_minutes",
    "vwap_band_position",
    "atr_5m_bps",
    "atr_30m_bps",
    "realized_volatility_30m_bps",
    "volatility_percentile",
    "volume_percentile",
    "efficiency_15m",
    "efficiency_60m",
    "oi_change_1h",
    "basis_bps",
    "funding_z",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
META_FEATURES = (
    *GATING_CONTEXT,
    *EXPERT_COLUMNS,
    "side",
    "horizon_fraction",
    "target_1_bps",
    "target_2_bps",
    "stop_bps",
    "trailing_bps",
    "predicted_favorable_q50_bps",
    "predicted_favorable_q75_bps",
    "predicted_adverse_q75_bps",
)

PROTOCOL = {
    "name": "musca_btc_mixture_of_experts_v1",
    "symbol": SYMBOL,
    "source": "Binance official USD-M and spot one-minute archives",
    "decision_cadence_minutes": 5,
    "entry": "next-minute open after completed decision candle",
    "horizons_minutes": list(HORIZONS),
    "sides": ["LONG", "SHORT"],
    "feature_views": {name: list(columns) for name, columns in VIEWS.items()},
    "return_experts": {
        "roles": len(HORIZONS) * len(VIEWS),
        "temporal_bootstrap_seeds": len(FINAL_SEEDS),
        "final_components": len(HORIZONS) * len(VIEWS) * len(FINAL_SEEDS),
    },
    "quantile_tools": {
        "per_side_horizon": ["favorable_q50", "favorable_q75", "adverse_q75"],
        "components": len(HORIZONS) * len(SIDES) * 3,
    },
    "gating": (
        "regime context plus OOF expert predictions; Ridge/XGBoost EV heads and "
        "five-seed XGBoost pairwise ranker"
    ),
    "management": "half at q50, half at q75, q75 adverse stop and non-widening trail",
    "same_minute": "stop wins",
    "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    "minimum_net_target_bps": MINIMUM_NET_TARGET_BPS,
    "risk_per_trade": 0.01,
    "maximum_leverage": 10.0,
    "maximum_positions": 1,
    "meta_period": [META_START.isoformat(), META_END.isoformat()],
    "model_audit_end": MODEL_AUDIT_END.isoformat(),
    "calibration_end": CALIBRATION_END.isoformat(),
    "policy_selection_end": POLICY_SELECTION_END.isoformat(),
    "historical_audit_end": HISTORICAL_AUDIT_END.isoformat(),
    "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
    "holdout_status": "sealed and empty at protocol freeze",
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    payload = {
        "phase": phase,
        "detail": detail,
        "percent": round(percent, 2),
        "updated_at": datetime.now(UTC).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
    }
    _atomic_json(STATUS, payload)
    print(f"[{payload['percent']:6.2f}%] {phase}: {detail}", flush=True)


def _load_source() -> pd.DataFrame:
    rows = pd.read_parquet(SOURCE)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.loc[rows["timestamp"].lt(FUTURE_HOLDOUT_START)].copy()
    if rows["timestamp"].ge(FUTURE_HOLDOUT_START).any():
        raise RuntimeError("future holdout was read")
    return rows.sort_values("timestamp").reset_index(drop=True)


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        cached = pd.read_parquet(MATRIX)
        if len(cached) and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached
    _status("matrix", "feature causali ogni 5 minuti", 2)
    source = _load_source()
    rows = _historical_features(source)
    timestamp = pd.to_datetime(rows["timestamp"], utc=True)
    rows["hour_sin"] = np.sin(2 * np.pi * (timestamp.dt.hour + timestamp.dt.minute / 60) / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * (timestamp.dt.hour + timestamp.dt.minute / 60) / 24)
    rows["weekday_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    rows["weekday_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    available = pd.to_datetime(rows["feature_available_at"], utc=True)
    valid = (
        rows["is_available"].fillna(False).to_numpy(bool)
        & rows["feature_contract_valid"].fillna(False).to_numpy(bool)
        & rows["oi_feature_available"].fillna(False).to_numpy(bool)
        & available.dt.minute.mod(5).eq(0).to_numpy(bool)
        & available.lt(HISTORICAL_AUDIT_END).to_numpy(bool)
        & np.isfinite(rows.loc[:, FEATURES].to_numpy(float)).all(axis=1)
    )
    valid[-(max(HORIZONS) + 1) :] = False
    positions = np.flatnonzero(valid)
    matrix = rows.loc[positions, ["timestamp", "feature_available_at", *FEATURES]].copy()
    matrix = matrix.rename(columns={"feature_available_at": "available_at"})
    matrix["decision_position"] = positions
    matrix["entry_timestamp"] = source.loc[positions + 1, "timestamp"].to_numpy()
    if not matrix["available_at"].le(matrix["entry_timestamp"]).all():
        raise RuntimeError("feature availability exceeds entry time")

    open_price = source["perp_open"].to_numpy(float)
    high = source["perp_high"].to_numpy(float)
    low = source["perp_low"].to_numpy(float)
    close = source["perp_close"].to_numpy(float)
    entry = open_price[positions + 1]
    for horizon in HORIZONS:
        offsets = positions[:, None] + np.arange(1, horizon + 1)
        high_path = high[offsets]
        low_path = low[offsets]
        matrix[f"terminal_{horizon}m_bps"] = (close[positions + horizon] / entry - 1) * 10_000
        matrix[f"max_up_{horizon}m_bps"] = (high_path.max(axis=1) / entry - 1) * 10_000
        matrix[f"max_down_{horizon}m_bps"] = (1 - low_path.min(axis=1) / entry) * 10_000
    numeric = matrix.loc[
        :,
        [
            *FEATURES,
            *(f"terminal_{horizon}m_bps" for horizon in HORIZONS),
            *(f"max_up_{horizon}m_bps" for horizon in HORIZONS),
            *(f"max_down_{horizon}m_bps" for horizon in HORIZONS),
        ],
    ].to_numpy(float)
    matrix = matrix.loc[np.isfinite(numeric).all(axis=1)].copy()
    matrix["protocol_hash"] = PROTOCOL_HASH
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    temporary.replace(MATRIX)
    return matrix.reset_index(drop=True)


def _xgb_regressor(seed: int, *, quantile: float | None = None) -> XGBRegressor:
    parameters: dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 140,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 20.0,
        "n_jobs": 4,
        "random_state": seed,
    }
    if quantile is None:
        parameters["objective"] = "reg:squarederror"
    else:
        parameters["objective"] = "reg:quantileerror"
        parameters["quantile_alpha"] = quantile
    return XGBRegressor(**parameters)


def _block_bootstrap(size: int, seed: int, block: int = 288) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = min(block, size)
    starts = rng.integers(0, max(1, size - width + 1), size=int(np.ceil(size / width)))
    return np.concatenate([np.arange(start, start + width) for start in starts])[:size]


def _fit_expert_pool(
    rows: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    status_prefix: str,
    status_start: float,
    status_width: float,
) -> dict[str, Any]:
    returns: dict[int, dict[str, list[Any]]] = {}
    quantiles: dict[int, dict[int, dict[str, Any]]] = {}
    total = len(HORIZONS) * len(VIEWS) * len(seeds) + len(HORIZONS) * len(SIDES) * 3
    completed = 0
    for horizon in HORIZONS:
        returns[horizon] = {}
        target = rows[f"terminal_{horizon}m_bps"].to_numpy(float)
        for view, columns in VIEWS.items():
            x = rows.loc[:, columns].to_numpy(float)
            models: list[Any] = []
            for seed in seeds:
                indexes = (
                    np.arange(len(rows)) if len(seeds) == 1 else _block_bootstrap(len(rows), seed)
                )
                model = _xgb_regressor(seed).fit(x[indexes], target[indexes], verbose=False)
                models.append(model)
                completed += 1
                if completed == total or completed % 5 == 0:
                    _status(
                        "experts",
                        f"{status_prefix} {completed}/{total}",
                        status_start + status_width * completed / total,
                    )
            returns[horizon][view] = models
        quantiles[horizon] = {}
        for side in SIDES:
            favorable = rows[f"max_{'up' if side > 0 else 'down'}_{horizon}m_bps"].to_numpy(float)
            adverse = rows[f"max_{'down' if side > 0 else 'up'}_{horizon}m_bps"].to_numpy(float)
            values: dict[str, Any] = {}
            for name, target, alpha in (
                ("favorable_q50", favorable, 0.50),
                ("favorable_q75", favorable, 0.75),
                ("adverse_q75", adverse, 0.75),
            ):
                values[name] = _xgb_regressor(20260810 + horizon + side, quantile=alpha).fit(
                    rows.loc[:, FEATURES].to_numpy(float), target, verbose=False
                )
                completed += 1
                if completed == total or completed % 5 == 0:
                    _status(
                        "experts",
                        f"{status_prefix} {completed}/{total}",
                        status_start + status_width * completed / total,
                    )
            quantiles[horizon][side] = values
    return {"returns": returns, "quantiles": quantiles, "component_count": total}


def _predict_experts(rows: pd.DataFrame, pool: dict[str, Any]) -> pd.DataFrame:
    output = pd.DataFrame(index=rows.index)
    for horizon in HORIZONS:
        for view, columns in VIEWS.items():
            x = rows.loc[:, columns].to_numpy(float)
            values = np.vstack(
                [
                    np.asarray(model.predict(x), dtype=float)
                    for model in pool["returns"][horizon][view]
                ]
            )
            output[f"expert_{horizon}m_{view}"] = values.mean(axis=0)
        x_full = rows.loc[:, FEATURES].to_numpy(float)
        for side in SIDES:
            for name, model in pool["quantiles"][horizon][side].items():
                output[f"{name}_{horizon}m_{side}"] = np.maximum(
                    np.asarray(model.predict(x_full), dtype=float), 0.0
                )
    return output


def _tighten_stop(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Directional stop coordinates can only increase, which reduces risk."""
    return np.asarray(np.maximum(current, candidate), dtype=float)


def _simulate_management(
    source: pd.DataFrame,
    positions: np.ndarray,
    side: int,
    horizon: int,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    open_price = source["perp_open"].to_numpy(float)
    high = source["perp_high"].to_numpy(float)
    low = source["perp_low"].to_numpy(float)
    close = source["perp_close"].to_numpy(float)
    entry = open_price[positions + 1]
    size = len(positions)
    gross = np.full(size, np.nan)
    exit_minutes = np.full(size, horizon, dtype=np.int16)
    outcome = np.full(size, "TIMEOUT", dtype=object)
    active = np.ones(size, dtype=bool)
    first_filled = np.zeros(size, dtype=bool)
    stop_level = -stop.copy()
    peak = np.zeros(size)
    half_realized = np.zeros(size)

    for minute in range(1, horizon + 1):
        indexes = positions + minute
        open_return = side * (open_price[indexes] / entry - 1) * 10_000
        best = np.where(
            side > 0,
            (high[indexes] / entry - 1) * 10_000,
            (1 - low[indexes] / entry) * 10_000,
        )
        worst = np.where(
            side > 0,
            (low[indexes] / entry - 1) * 10_000,
            (1 - high[indexes] / entry) * 10_000,
        )

        gap = active & (open_return <= stop_level + 1e-9)
        gross[gap] = np.where(
            first_filled[gap], half_realized[gap] + 0.5 * open_return[gap], open_return[gap]
        )
        exit_minutes[gap] = minute
        outcome[gap] = "STOP_GAP"
        active[gap] = False

        before = active & ~first_filled
        stopped = before & (worst <= stop_level + 1e-9)
        gross[stopped] = stop_level[stopped]
        exit_minutes[stopped] = minute
        outcome[stopped] = "STOP"
        active[stopped] = False

        first = active & ~first_filled & (best >= target_1 - 1e-9)
        second_same_bar = first & (best >= target_2 - 1e-9)
        gross[second_same_bar] = 0.5 * target_1[second_same_bar] + 0.5 * target_2[second_same_bar]
        exit_minutes[second_same_bar] = minute
        outcome[second_same_bar] = "TARGET_2"
        active[second_same_bar] = False
        first &= active
        first_filled[first] = True
        half_realized[first] = 0.5 * target_1[first]
        stop_level[first] = _tighten_stop(stop_level[first], np.zeros(first.sum()))

        after = active & first_filled & ~first
        stopped_after = after & (worst <= stop_level + 1e-9)
        gross[stopped_after] = half_realized[stopped_after] + 0.5 * stop_level[stopped_after]
        exit_minutes[stopped_after] = minute
        outcome[stopped_after] = "TRAIL_STOP"
        active[stopped_after] = False
        second = active & first_filled & (best >= target_2 - 1e-9)
        gross[second] = half_realized[second] + 0.5 * target_2[second]
        exit_minutes[second] = minute
        outcome[second] = "TARGET_2"
        active[second] = False

        peak[active & first_filled] = np.maximum(
            peak[active & first_filled], best[active & first_filled]
        )
        managed = active & first_filled
        stop_level[managed] = _tighten_stop(stop_level[managed], peak[managed] - trailing[managed])

    terminal = side * (close[positions + horizon] / entry - 1) * 10_000
    gross[active] = np.where(
        first_filled[active], half_realized[active] + 0.5 * terminal[active], terminal[active]
    )
    return gross.astype(float), exit_minutes, outcome.astype(str)


def _action_rows(
    rows: pd.DataFrame, predictions: pd.DataFrame, source: pd.DataFrame
) -> pd.DataFrame:
    actions: list[pd.DataFrame] = []
    positions = rows["decision_position"].to_numpy(int)
    for horizon in HORIZONS:
        for side in SIDES:
            favorable_q50 = predictions[f"favorable_q50_{horizon}m_{side}"].to_numpy(float)
            favorable_q75 = predictions[f"favorable_q75_{horizon}m_{side}"].to_numpy(float)
            adverse_q75 = predictions[f"adverse_q75_{horizon}m_{side}"].to_numpy(float)
            target_1 = np.clip(
                np.maximum(favorable_q50, ROUND_TRIP_COST_BPS + MINIMUM_NET_TARGET_BPS),
                ROUND_TRIP_COST_BPS + MINIMUM_NET_TARGET_BPS,
                MAX_TARGET_BPS - 1,
            )
            target_2 = np.clip(
                np.maximum(favorable_q75, target_1 + 1), target_1 + 1, MAX_TARGET_BPS
            )
            stop = np.clip(adverse_q75, 3.0, MAX_STOP_BPS)
            trailing = np.clip(adverse_q75, 3.0, MAX_STOP_BPS)
            gross, exit_minutes, outcome = _simulate_management(
                source, positions, side, horizon, target_1, target_2, stop, trailing
            )
            action = rows.loc[:, ["available_at", "entry_timestamp", "decision_position"]].copy()
            for feature in FEATURES:
                values = rows[feature].to_numpy(float)
                action[feature] = values * side if feature in DIRECTIONAL_FEATURES else values
            for column in EXPERT_COLUMNS:
                action[column] = predictions[column].to_numpy(float) * side
            action["side"] = float(side)
            action["horizon_fraction"] = horizon / max(HORIZONS)
            action["horizon_minutes"] = horizon
            action["target_1_bps"] = target_1
            action["target_2_bps"] = target_2
            action["stop_bps"] = stop
            action["trailing_bps"] = trailing
            action["predicted_favorable_q50_bps"] = favorable_q50
            action["predicted_favorable_q75_bps"] = favorable_q75
            action["predicted_adverse_q75_bps"] = adverse_q75
            action["gross_bps"] = gross
            action["net_bps"] = gross - ROUND_TRIP_COST_BPS
            action["stress_1_5x_bps"] = gross - 1.5 * ROUND_TRIP_COST_BPS
            action["stress_2x_bps"] = gross - 2 * ROUND_TRIP_COST_BPS
            action["exit_minutes"] = exit_minutes
            action["exit_timestamp"] = action["entry_timestamp"] + pd.to_timedelta(
                exit_minutes, unit="min"
            )
            action["outcome"] = outcome
            actions.append(action)
    return (
        pd.concat(actions, ignore_index=True)
        .sort_values(["entry_timestamp", "side", "horizon_minutes"])
        .reset_index(drop=True)
    )


def _period(rows: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    available = pd.to_datetime(rows["available_at"], utc=True)
    mask = available.lt(end - PURGE)
    if start is not None:
        mask &= available.ge(start)
    return rows.loc[mask].copy()


def _checkpoint(
    path: Path, train: pd.DataFrame, seeds: tuple[int, ...], **status: Any
) -> dict[str, Any]:
    if path.exists():
        cached = joblib.load(path)
        if cached.get("protocol_hash") == PROTOCOL_HASH:
            return cast(dict[str, Any], cached["pool"])
    pool = _fit_expert_pool(train, seeds=seeds, **status)
    _atomic_joblib(path, {"protocol_hash": PROTOCOL_HASH, "pool": pool})
    return pool


def _oof_actions(matrix: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    folds = (
        (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-05-01T00:00:00Z")),
        (pd.Timestamp("2025-05-01T00:00:00Z"), pd.Timestamp("2025-09-01T00:00:00Z")),
        (pd.Timestamp("2025-09-01T00:00:00Z"), META_END),
    )
    pieces: list[pd.DataFrame] = []
    for number, (start, end) in enumerate(folds, start=1):
        training = _period(matrix, None, start)
        testing = _period(matrix, start, end)
        pool = _checkpoint(
            CHECKPOINTS / f"oof_{number}.joblib",
            training,
            (20260810 + number,),
            status_prefix=f"OOF {number}/{len(folds)}",
            status_start=5 + (number - 1) * 12,
            status_width=12,
        )
        predictions = _predict_experts(testing, pool)
        pieces.append(_action_rows(testing, predictions, source))
    return (
        pd.concat(pieces, ignore_index=True).sort_values("entry_timestamp").reset_index(drop=True)
    )


def _meta_x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, META_FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("meta features must be finite")
    return values


def _fit_meta(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values(["entry_timestamp", "side", "horizon_minutes"]).reset_index(
        drop=True
    )
    x = _meta_x(ordered)
    y = ordered["net_bps"].to_numpy(float)
    positive = (y > 0).astype(int)
    ridge_reg = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(x, y)
    ridge_cls = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2_000, random_state=20260810)
    ).fit(x, positive)
    xgb_reg: list[Any] = []
    xgb_cls: list[Any] = []
    rankers: list[Any] = []
    groups = ordered.groupby("entry_timestamp", sort=False).size().to_numpy(int)
    relevance = (
        ordered.groupby("entry_timestamp", sort=False)["net_bps"].rank(method="first").to_numpy(int)
        - 1
    )
    for number, seed in enumerate(FINAL_SEEDS, start=1):
        regressor = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=260,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=seed,
        ).fit(x, y, verbose=False)
        classifier = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_estimators=260,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=seed,
        ).fit(x, positive, verbose=False)
        ranker = XGBRanker(
            objective="rank:pairwise",
            tree_method="hist",
            device="cuda",
            n_estimators=260,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=seed,
        ).fit(x, relevance, group=groups, verbose=False)
        xgb_reg.append(regressor)
        xgb_cls.append(classifier)
        rankers.append(ranker)
        _status("gating", f"XGBoost meta seed {number}/{len(FINAL_SEEDS)}", 83 + number)
    return {
        "ridge": {"regressors": [ridge_reg], "classifiers": [ridge_cls]},
        "xgboost": {"regressors": xgb_reg, "classifiers": xgb_cls},
        "ranker": {"models": rankers},
    }


def _raw_meta(
    rows: pd.DataFrame, model: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _meta_x(rows)
    evs = np.vstack([np.asarray(item.predict(x), dtype=float) for item in model["regressors"]])
    probabilities = np.vstack(
        [np.asarray(item.predict_proba(x)[:, 1], dtype=float) for item in model["classifiers"]]
    )
    return evs.mean(axis=0), probabilities.mean(axis=0), evs.std(axis=0)


def _ranker_score(rows: pd.DataFrame, ranker: dict[str, Any]) -> np.ndarray:
    x = _meta_x(rows)
    values = np.vstack([np.asarray(item.predict(x), dtype=float) for item in ranker["models"]])
    return np.asarray(values.mean(axis=0), dtype=float)


def _decision_regret(
    rows: pd.DataFrame, prediction: np.ndarray, *, allow_flat: bool = True
) -> float:
    scored = rows.loc[:, ["entry_timestamp", "net_bps"]].copy()
    scored["prediction"] = prediction
    regrets: list[float] = []
    for _, group in scored.groupby("entry_timestamp", sort=False):
        predicted = group["prediction"].to_numpy(float)
        actual = group["net_bps"].to_numpy(float)
        choice = int(np.argmax(predicted))
        chosen = actual[choice] if not allow_flat or predicted[choice] > 0 else 0.0
        regrets.append(max(0.0, float(actual.max())) - chosen)
    return float(np.mean(regrets))


def _model_metrics(rows: pd.DataFrame, model: dict[str, Any]) -> dict[str, float]:
    ev, probability, _ = _raw_meta(rows, model)
    actual = rows["net_bps"].to_numpy(float)
    return {
        "mae_bps": float(mean_absolute_error(actual, ev)),
        "brier": float(brier_score_loss(actual > 0, probability)),
        "decision_regret_bps": _decision_regret(rows, ev),
    }


def _fit_calibrators(rows: pd.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    ev, probability, _ = _raw_meta(rows, model)
    actual = rows["net_bps"].to_numpy(float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return {
        "ev": IsotonicRegression(out_of_bounds="clip").fit(ev, actual),
        "probability": LogisticRegression(C=1.0, max_iter=2_000).fit(
            logit, (actual > 0).astype(int)
        ),
    }


def _score_meta(
    rows: pd.DataFrame,
    model: dict[str, Any],
    calibrators: dict[str, Any],
    ranker: dict[str, Any] | None = None,
) -> pd.DataFrame:
    raw_ev, raw_probability, dispersion = _raw_meta(rows, model)
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    output = rows.copy()
    output["score"] = raw_ev if ranker is None else _ranker_score(rows, ranker)
    output["calibrated_ev_bps"] = calibrators["ev"].predict(raw_ev)
    output["probability_net_positive"] = calibrators["probability"].predict_proba(logit)[:, 1]
    output["ensemble_dispersion_bps"] = dispersion
    return output


def _execute(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    winners = (
        scored.sort_values(
            ["entry_timestamp", "score", "calibrated_ev_bps", "side", "horizon_minutes"],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates("entry_timestamp", keep="first")
        .loc[lambda value: value["score"].ge(threshold)]
    )
    accepted: list[int] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    for index, row in winners.iterrows():
        if pd.Timestamp(row["entry_timestamp"]) < free_at:
            continue
        accepted.append(cast(int, index))
        free_at = pd.Timestamp(row["exit_timestamp"])
    return winners.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def _bootstrap_lcb(trades: pd.DataFrame, seed: int = 20260810) -> float | None:
    if trades.empty:
        return None
    daily = (
        trades.assign(day=pd.to_datetime(trades["entry_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["net_bps"]
        .agg(["sum", "count"])
    )
    if len(daily) < 10:
        return None
    values = daily.to_numpy(float)
    rng = np.random.default_rng(seed)
    block = min(5, len(values))
    draws = np.empty(1_000)
    for number in range(len(draws)):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < len(values):
            start = int(rng.integers(0, len(values) - block + 1))
            pieces.append(values[start : start + block])
        sample = np.concatenate(pieces)[: len(values)]
        draws[number] = sample[:, 0].sum() / sample[:, 1].sum()
    return float(np.quantile(draws, 0.05))


def _spa_pvalue(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    days = pd.date_range(start.floor("D"), end.floor("D") - pd.Timedelta(days=1), freq="1D")
    if len(days) < 10:
        return None
    daily = (
        trades.assign(day=pd.to_datetime(trades["entry_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["net_bps"]
        .sum()
        .reindex(days, fill_value=0.0)
    )
    test = SPA(
        np.zeros(len(daily)),
        (-daily.to_numpy(float)).reshape(-1, 1),
        block_size=min(5, len(daily)),
        reps=1_000,
        bootstrap="stationary",
        seed=20260810,
    )
    test.compute()
    return float(test.pvalues["consistent"])


def _metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    calendar = pd.date_range(start.floor("D"), end.floor("D") - pd.Timedelta(days=1), freq="1D")
    days = max(1, len(calendar))
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "positive_calendar_days": 0.0,
            "max_drawdown": None,
            "bootstrap_lcb_95_bps": None,
            "spa_pvalue": None,
        }
    net = trades["net_bps"].to_numpy(float)
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    notional_multiple = np.minimum(
        10.0, 0.01 / ((trades["stop_bps"].to_numpy(float) + ROUND_TRIP_COST_BPS) / 10_000)
    )
    strategy_returns = notional_multiple * net / 10_000
    equity = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    daily = (
        trades.assign(day=pd.to_datetime(trades["entry_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["net_bps"]
        .sum()
        .reindex(calendar, fill_value=0.0)
    )
    return {
        "trades": len(trades),
        "trades_per_day": float(len(trades) / days),
        "expectancy_bps": float(net.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((net > 0).mean()),
        "positive_calendar_days": float((daily > 0).mean()),
        "max_drawdown": float((1 - equity / peak).max(initial=0.0)),
        "bootstrap_lcb_95_bps": _bootstrap_lcb(trades),
        "spa_pvalue": _spa_pvalue(trades, start, end),
        "stress_1_5x_expectancy_bps": float(trades["stress_1_5x_bps"].mean()),
        "stress_2x_expectancy_bps": float(trades["stress_2x_bps"].mean()),
        "long": int(trades["side"].gt(0).sum()),
        "short": int(trades["side"].lt(0).sum()),
        "horizons": {
            str(key): int(value) for key, value in trades["horizon_minutes"].value_counts().items()
        },
    }


def _selection_gates(value: dict[str, Any]) -> dict[str, bool]:
    return {
        "minimum_trades": int(value["trades"]) >= 120,
        "frequency": float(value["trades_per_day"]) >= 3.0,
        "expectancy": value["expectancy_bps"] is not None and float(value["expectancy_bps"]) > 0,
        "profit_factor": value["profit_factor"] is not None
        and float(value["profit_factor"]) >= 1.10,
        "positive_days": float(value["positive_calendar_days"]) > 0.5,
        "drawdown": value["max_drawdown"] is not None and float(value["max_drawdown"]) <= 0.10,
    }


def _audit_gates(value: dict[str, Any]) -> dict[str, bool]:
    lcb = value["bootstrap_lcb_95_bps"]
    pvalue = value["spa_pvalue"]
    return {
        "minimum_trades": int(value["trades"]) >= 300,
        "frequency": float(value["trades_per_day"]) >= 3.0,
        "expectancy": value["expectancy_bps"] is not None and float(value["expectancy_bps"]) > 0,
        "profit_factor": value["profit_factor"] is not None
        and float(value["profit_factor"]) >= 1.15,
        "positive_days": float(value["positive_calendar_days"]) > 0.5,
        "drawdown": value["max_drawdown"] is not None and float(value["max_drawdown"]) <= 0.10,
        "bootstrap_lcb": lcb is not None and float(lcb) > 0,
        "spa": pvalue is not None and float(pvalue) <= 0.05,
    }


def train(*, force_matrix: bool = False, resume: bool = True) -> dict[str, Any]:
    del resume  # checkpoints are always protocol-hash guarded and safe to reuse
    _status("start", "BTCUSDT Mixture of Experts", 0)
    matrix = build_matrix(force=force_matrix)
    source = _load_source()
    oof_path = CHECKPOINTS / "oof_actions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        if not len(oof) or oof["protocol_hash"].iloc[0] != PROTOCOL_HASH:
            oof = pd.DataFrame()
    else:
        oof = pd.DataFrame()
    if oof.empty:
        oof = _oof_actions(matrix, source)
        oof["protocol_hash"] = PROTOCOL_HASH
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = oof_path.with_suffix(".parquet.tmp")
        oof.to_parquet(temporary, index=False)
        temporary.replace(oof_path)

    final_train = _period(matrix, None, META_END)
    final_pool = _checkpoint(
        CHECKPOINTS / "final_experts.joblib",
        final_train,
        FINAL_SEEDS,
        status_prefix="finale 100 esperti + 24 quantili",
        status_start=42,
        status_width=35,
    )
    future_rows = _period(matrix, META_END, HISTORICAL_AUDIT_END)
    future_actions_path = CHECKPOINTS / "future_actions.parquet"
    if future_actions_path.exists():
        future_actions = pd.read_parquet(future_actions_path)
        if not len(future_actions) or future_actions["protocol_hash"].iloc[0] != PROTOCOL_HASH:
            future_actions = pd.DataFrame()
    else:
        future_actions = pd.DataFrame()
    if future_actions.empty:
        future_predictions = _predict_experts(future_rows, final_pool)
        future_actions = _action_rows(future_rows, future_predictions, source)
        future_actions["protocol_hash"] = PROTOCOL_HASH
        temporary = future_actions_path.with_suffix(".parquet.tmp")
        future_actions.to_parquet(temporary, index=False)
        temporary.replace(future_actions_path)

    meta_fit = oof.loc[
        pd.to_datetime(oof["entry_timestamp"], utc=True).ge(META_START)
        & pd.to_datetime(oof["entry_timestamp"], utc=True).lt(META_END - PURGE)
    ].copy()
    meta_models_path = CHECKPOINTS / "gating.joblib"
    if meta_models_path.exists():
        cached_meta = joblib.load(meta_models_path)
        meta_models = (
            cached_meta["models"] if cached_meta.get("protocol_hash") == PROTOCOL_HASH else None
        )
    else:
        meta_models = None
    if meta_models is None:
        meta_models = _fit_meta(meta_fit)
        _atomic_joblib(meta_models_path, {"protocol_hash": PROTOCOL_HASH, "models": meta_models})

    timestamps = pd.to_datetime(future_actions["entry_timestamp"], utc=True)
    model_audit = future_actions.loc[timestamps.lt(MODEL_AUDIT_END - PURGE)].copy()
    calibration = future_actions.loc[
        timestamps.ge(MODEL_AUDIT_END) & timestamps.lt(CALIBRATION_END - PURGE)
    ].copy()
    policy_selection = future_actions.loc[
        timestamps.ge(CALIBRATION_END) & timestamps.lt(POLICY_SELECTION_END - PURGE)
    ].copy()
    historical_audit = future_actions.loc[
        timestamps.ge(POLICY_SELECTION_END) & timestamps.lt(HISTORICAL_AUDIT_END - PURGE)
    ].copy()
    candidate_metrics = {
        name: _model_metrics(model_audit, meta_models[name]) for name in ("ridge", "xgboost")
    }
    ranker_audit_score = _ranker_score(model_audit, meta_models["ranker"])
    candidate_metrics["ranker"] = {
        "decision_regret_bps": _decision_regret(model_audit, ranker_audit_score, allow_flat=False)
    }
    ridge_metrics = candidate_metrics["ridge"]
    xgb_metrics = candidate_metrics["xgboost"]
    xgb_wins = all(
        float(xgb_metrics[key]) < float(ridge_metrics[key])
        for key in ("mae_bps", "brier", "decision_regret_bps")
    )
    ev_champion = "xgboost" if xgb_wins else "ridge"
    ranker_wins = float(candidate_metrics["ranker"]["decision_regret_bps"]) < float(
        candidate_metrics[ev_champion]["decision_regret_bps"]
    )
    gating_champion = "xgboost_ranker" if ranker_wins else ev_champion
    active_ranker = meta_models["ranker"] if ranker_wins else None
    calibrators = _fit_calibrators(calibration, meta_models[ev_champion])
    selection_scored = _score_meta(
        policy_selection, meta_models[ev_champion], calibrators, active_ranker
    )
    curve: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        threshold = float(selection_scored["score"].quantile(1 - coverage))
        trades = _execute(selection_scored, threshold)
        value = _metrics(trades, CALIBRATION_END, POLICY_SELECTION_END)
        curve.append(
            {
                "coverage": coverage,
                "threshold": threshold,
                "metrics": value,
                "gates": _selection_gates(value),
            }
        )
    passing = [point for point in curve if all(point["gates"].values())]
    selected = max(
        passing,
        key=lambda point: float(point["metrics"]["trades_per_day"]),
        default=None,
    )
    diagnostic = max(
        curve,
        key=lambda point: (
            sum(point["gates"].values()),
            float(point["metrics"]["expectancy_bps"] or -np.inf),
        ),
    )
    frozen = selected or diagnostic
    audit_scored = _score_meta(
        historical_audit, meta_models[ev_champion], calibrators, active_ranker
    )
    audit_threshold = float(frozen["threshold"])
    audit_trades = _execute(audit_scored, audit_threshold)
    audit_metrics = _metrics(audit_trades, POLICY_SELECTION_END, HISTORICAL_AUDIT_END)
    audit_gates = _audit_gates(audit_metrics)
    historical_pass = selected is not None and all(audit_gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": SYMBOL,
        "matrix_rows": len(matrix),
        "oof_action_rows": len(oof),
        "meta_fit_action_rows": len(meta_fit),
        "future_action_rows": len(future_actions),
        "final_model_components": int(final_pool["component_count"]) + 17,
        "candidate_metrics": candidate_metrics,
        "ev_champion": ev_champion,
        "gating_champion": gating_champion,
        "policy_selection": {"curve": curve, "selected": selected},
        "diagnostic_when_no_selection": None if selected is not None else diagnostic,
        "historical_audit": {
            "coverage": frozen["coverage"],
            "threshold": audit_threshold,
            "metrics": audit_metrics,
            "gates": audit_gates,
            "selection_was_gate_passing": selected is not None,
        },
        "causal_checks": {
            "future_holdout_rows_read": int(
                pd.to_datetime(matrix["available_at"], utc=True).ge(FUTURE_HOLDOUT_START).sum()
            ),
            "future_feature_violations": int(
                (
                    pd.to_datetime(matrix["available_at"], utc=True)
                    > pd.to_datetime(matrix["entry_timestamp"], utc=True)
                ).sum()
            ),
            "same_minute_stop_wins": True,
            "one_position": True,
        },
        "verdict": (
            "HISTORICAL_ALPHA_READY_FOR_FUTURE_HOLDOUT"
            if historical_pass
            else "NO_HISTORICALLY_STABLE_BTC_MOE_ALPHA"
        ),
        "future_holdout": {
            "starts_at": FUTURE_HOLDOUT_START.isoformat(),
            "minimum_days": 10,
            "minimum_trades": 100,
            "opened": False,
        },
        "research_only": True,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "real_capital_allowed": False,
    }
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "expert_pool": final_pool,
        "meta_model": meta_models[ev_champion],
        "ranker": active_ranker,
        "calibrators": calibrators,
        "ev_champion": ev_champion,
        "gating_champion": gating_champion,
        "coverage": frozen["coverage"],
        "score_threshold": audit_threshold,
        "historical_pass": historical_pass,
        "research_only": True,
        "orders_enabled": False,
    }
    _atomic_joblib(BUNDLE, bundle)
    _atomic_json(REPORT, report)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Binance Mixture of Experts research training")
    parser.add_argument("--force-matrix", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            train(force_matrix=arguments.force_matrix, resume=not arguments.no_resume),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
