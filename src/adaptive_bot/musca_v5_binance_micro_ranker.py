from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.musca_v5_micro_model import build_micro_features

RAW_ROOT = Path("data/ml/musca_v5/aggtrades")
RAW_MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 7))
MARKET_STATE_SOURCE = Path("data/ml/musca_v5/market_state_features.parquet")
MACRO_SOURCE = Path("data/ml/hybrid_v25/asset=BTCUSDT/bars_5m.parquet")
FUNDING_SOURCE = Path("data/ml/hybrid_v24/joined_minutes.parquet")
MATRIX = Path("data/ml/musca_v5/binance_30s_policy_matrix.parquet")
REPORT = Path("data/reports/musca_v5_binance_policy.json")
STATUS = Path("data/reports/musca_v5_binance_policy.status.json")
BUNDLE = Path("data/models/musca_v5_binance_policy/bundle.joblib")

COST_BPS = 9.0
STRESS_COST_BPS = 18.0
DECISION_SECONDS = 30
FIT_END = pd.Timestamp("2026-03-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-03-15T00:00:00Z")
MODEL_SELECTION_END = pd.Timestamp("2026-04-01T00:00:00Z")
POLICY_SELECTION_END = pd.Timestamp("2026-05-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-07-01T00:00:00Z")
HOLDOUT_START = AUDIT_END
COVERAGES = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)

MICRO_FEATURES = (
    "ofi_15s",
    "ofi_1m",
    "ofi_5m",
    "ofi_persistence_1m",
    "trade_intensity_15s",
    "trade_intensity_1m",
    "absorption_1m",
    "price_velocity_15s",
    "price_velocity_1m",
)
MARKET_FEATURES = (
    "daily_vwap_distance_bps",
    "rolling_vwap_distance_bps",
    "rolling_vwap_slope_bps",
    "rolling_vwap_slope_change_bps",
    "rolling_vwap_tests_1h",
    "rolling_vwap_rejections_1h",
    "time_since_rolling_vwap_cross_minutes",
    "rolling_vwap_rejection_strength_bps",
    "rolling_vwap_band_position",
    "swing_long_distance_bps",
    "swing_short_distance_bps",
    "swing_long_slope_bps",
    "swing_short_slope_bps",
    "swing_long_slope_change_bps",
    "swing_short_slope_change_bps",
    "swing_long_anchor_age_bars",
    "swing_short_anchor_age_bars",
    "swing_long_return_since_anchor_bps",
    "swing_short_return_since_anchor_bps",
    "swing_long_tests_1h",
    "swing_short_tests_1h",
    "swing_long_rejections_1h",
    "swing_short_rejections_1h",
    "swing_long_rejection_strength_bps",
    "swing_short_rejection_strength_bps",
    "rolling_swing_long_distance_bps",
    "rolling_swing_short_distance_bps",
    "rolling_swing_long_convergence_bps",
    "rolling_swing_short_convergence_bps",
    "atr_1m_bps",
    "atr_5m_bps",
    "atr_15m_bps",
    "atr_30m_bps",
    "trend_1m",
    "trend_5m",
    "trend_15m",
    "trend_30m",
    "vwap_state_1m",
    "vwap_state_5m",
    "vwap_state_15m",
    "vwap_state_30m",
    "realized_volatility_30m_bps",
    "volume_percentile",
)
MACRO_FEATURES = (
    "perp_return_1h_bps",
    "perp_return_4h_bps",
    "spot_return_1h_bps",
    "spot_return_4h_bps",
    "perp_taker_imbalance",
    "spot_taker_imbalance",
    "oi_change_1h",
    "return_oi_interaction",
    "mark_distance_bps",
    "spot_perp_basis_bps",
    "perp_daily_vwap_distance_bps",
    "spot_daily_vwap_distance_bps",
    "relative_quote_volume",
    "trade_count_z",
    "aggressive_volume_z",
    "funding_rate_bps",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
FEATURES = (*MICRO_FEATURES, *MARKET_FEATURES, *MACRO_FEATURES)


@dataclass(frozen=True)
class Plan:
    name: str
    minimum_target_bps: float
    maximum_target_bps: float
    reward_to_risk: float
    atr_stop_multiple: float
    minimum_stop_bps: float
    maximum_stop_bps: float
    horizon_seconds: int


PLANS = (
    Plan("FAST", 20.0, 36.0, 1.5, 1.5, 12.0, 24.0, 180),
    Plan("STANDARD", 30.0, 50.0, 1.5, 2.0, 16.0, 32.0, 300),
    Plan("RUNNER", 45.0, 70.0, 1.75, 2.5, 20.0, 40.0, 600),
    Plan("TREND_15M", 60.0, 90.0, 1.5, 3.5, 30.0, 50.0, 900),
    Plan("TREND_30M", 90.0, 140.0, 1.6, 5.0, 45.0, 75.0, 1_800),
    Plan("TREND_60M", 120.0, 200.0, 1.75, 7.0, 60.0, 100.0, 3_600),
)
MAX_HORIZON_SECONDS = max(plan.horizon_seconds for plan in PLANS)
PROTOCOL = {
    "name": "musca_v5_official_binance_high_frequency_policy",
    "market": "Binance BTCUSDT USD-M perpetual",
    "feature_venues": ["Binance BTCUSDT USD-M", "Binance BTCUSDT spot"],
    "cross_exchange_features": False,
    "decision_interval_seconds": DECISION_SECONDS,
    "features": list(FEATURES),
    "vwap_role": (
        "daily, rolling and causal swing-anchored VWAP are state/benchmark features; "
        "direction is estimated from Binance price, flow, volatility and context"
    ),
    "actions": [asdict(plan) for plan in PLANS],
    "directions": ["LONG", "SHORT"],
    "flat": "implicit when the selected causal score is below the frozen coverage threshold",
    "entry": "first 5-second open at or after feature availability",
    "labels": "target-before-stop, stop-before-target or timeout; same 5-second bar means stop",
    "models": {
        "baseline": "Ridge/Logistic",
        "challenger": "XGBoost hist on CUDA",
        "heads": (
            "one calibrated three-class outcome model and one timeout-return model "
            "per plan/side"
        ),
    },
    "chronology": {
        "fit": "2026-01-01/2026-03-01",
        "calibration": "2026-03-01/2026-03-15",
        "model_selection": "2026-03-15/2026-04-01",
        "policy_selection": "2026-04-01/2026-05-01",
        "reused_discovery_audit": "2026-05-01/2026-07-01",
        "sealed_holdout": "from 2026-07-01; never read",
        "purge_seconds": MAX_HORIZON_SECONDS,
    },
    "coverage_frontier": list(COVERAGES),
    "costs": {
        "normal_round_trip_bps": COST_BPS,
        "stress_round_trip_bps": STRESS_COST_BPS,
        "funding": "observed Binance funding events over the realized holding interval",
    },
    "risk": {"one_position": True, "risk_per_trade": 0.01, "leverage_cap": 10},
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class Predictor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _nanoseconds(values: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(values, utc=True)
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy(np.int64)
    )


def _macro_features() -> pd.DataFrame:
    columns = [
        "available_at",
        "max_input_available_at",
        "is_available",
        "context_coverage",
        "perp_close",
        "perp_quote_volume",
        "perp_trade_count",
        "perp_taker_buy_quote",
        "spot_close",
        "spot_quote_volume",
        "spot_taker_buy_quote",
        "perp_mark_close",
        "perp_funding_rate",
        "oi_change_1h",
        "perp_daily_vwap",
        "spot_daily_vwap",
        "perp_taker_imbalance",
        "spot_taker_imbalance",
    ]
    rows = pd.read_parquet(MACRO_SOURCE, columns=columns).sort_values("available_at")
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    rows["max_input_available_at"] = pd.to_datetime(rows["max_input_available_at"], utc=True)
    rows = rows.loc[
        rows["is_available"]
        & rows["context_coverage"]
        & rows["max_input_available_at"].le(rows["available_at"])
    ].copy()
    rows["perp_return_1h_bps"] = rows["perp_close"].pct_change(12) * 10_000
    rows["perp_return_4h_bps"] = rows["perp_close"].pct_change(48) * 10_000
    rows["spot_return_1h_bps"] = rows["spot_close"].pct_change(12) * 10_000
    rows["spot_return_4h_bps"] = rows["spot_close"].pct_change(48) * 10_000
    rows["return_oi_interaction"] = (
        rows["perp_return_1h_bps"] * rows["oi_change_1h"]
    )
    rows["mark_distance_bps"] = (
        rows["perp_mark_close"] / rows["perp_close"] - 1
    ) * 10_000
    rows["spot_perp_basis_bps"] = (
        rows["perp_close"] / rows["spot_close"] - 1
    ) * 10_000
    rows["perp_daily_vwap_distance_bps"] = (
        rows["perp_close"] / rows["perp_daily_vwap"] - 1
    ) * 10_000
    rows["spot_daily_vwap_distance_bps"] = (
        rows["spot_close"] / rows["spot_daily_vwap"] - 1
    ) * 10_000
    quote_median = rows["perp_quote_volume"].shift(1).rolling(288, min_periods=96).median()
    rows["relative_quote_volume"] = rows["perp_quote_volume"] / quote_median
    count_mean = rows["perp_trade_count"].shift(1).rolling(288, min_periods=96).mean()
    count_std = rows["perp_trade_count"].shift(1).rolling(288, min_periods=96).std()
    rows["trade_count_z"] = (rows["perp_trade_count"] - count_mean) / count_std
    aggressive = rows["perp_taker_imbalance"].abs() * rows["perp_quote_volume"]
    aggressive_mean = aggressive.shift(1).rolling(288, min_periods=96).mean()
    aggressive_std = aggressive.shift(1).rolling(288, min_periods=96).std()
    rows["aggressive_volume_z"] = (aggressive - aggressive_mean) / aggressive_std
    rows["funding_rate_bps"] = rows["perp_funding_rate"] * 10_000
    hour = rows["available_at"].dt.hour + rows["available_at"].dt.minute / 60
    weekday = rows["available_at"].dt.dayofweek
    rows["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    rows["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    rows["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return rows.loc[:, ["available_at", *MACRO_FEATURES]]


def _market_features() -> pd.DataFrame:
    rows = pd.read_parquet(
        MARKET_STATE_SOURCE, columns=["available_at", *MARKET_FEATURES, "coverage_valid"]
    ).sort_values("available_at")
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    return rows.loc[rows["coverage_valid"], ["available_at", *MARKET_FEATURES]]


def _load_raw() -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            RAW_ROOT / f"BTCUSDT-aggTrades-5s-{month}.parquet",
            columns=[
                "timestamp",
                "available_at",
                "quote_volume",
                "base_volume",
                "signed_quote_volume",
                "trade_count",
                "buy_count",
                "open",
                "high",
                "low",
                "close",
            ],
        )
        for month in RAW_MONTHS
    ]
    rows = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    if not rows["timestamp"].is_monotonic_increasing:
        raise ValueError("Binance 5-second timestamps are not monotonic")
    if rows["timestamp"].duplicated().any():
        raise ValueError("Binance 5-second timestamps contain duplicates")
    if not rows["available_at"].gt(rows["timestamp"]).all():
        raise ValueError("Binance feature availability must follow the source bucket")
    return rows.reset_index(drop=True)


def _plan_levels(atr_1m_bps: np.ndarray, plan: Plan) -> tuple[np.ndarray, np.ndarray]:
    stop = np.clip(
        plan.atr_stop_multiple * atr_1m_bps,
        plan.minimum_stop_bps,
        plan.maximum_stop_bps,
    )
    target = np.clip(
        np.maximum(plan.minimum_target_bps, plan.reward_to_risk * stop),
        plan.minimum_target_bps,
        plan.maximum_target_bps,
    )
    return target, stop


def _barrier_outcomes(
    *,
    entry: np.ndarray,
    entry_indexes: np.ndarray,
    side: int,
    target: np.ndarray,
    stop: np.ndarray,
    horizon_seconds: int,
    raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bars = horizon_seconds // 5
    result_outcome = np.empty(len(entry), dtype=np.int8)
    result_realized = np.empty(len(entry), dtype=float)
    result_exit = np.empty(len(entry), dtype=np.int32)
    high_view = np.lib.stride_tricks.sliding_window_view(
        raw["high"].to_numpy(float), bars
    )
    low_view = np.lib.stride_tricks.sliding_window_view(
        raw["low"].to_numpy(float), bars
    )
    raw_open = raw["open"].to_numpy(float)
    raw_close = raw["close"].to_numpy(float)
    for first in range(0, len(entry), 50_000):
        last = min(first + 50_000, len(entry))
        selection = slice(first, last)
        indexes = entry_indexes[selection]
        selected_entry = entry[selection]
        high_windows = high_view[indexes]
        low_windows = low_view[indexes]
        favorable = np.where(
            side > 0,
            (high_windows / selected_entry[:, None] - 1) * 10_000,
            (1 - low_windows / selected_entry[:, None]) * 10_000,
        )
        adverse = np.where(
            side > 0,
            (1 - low_windows / selected_entry[:, None]) * 10_000,
            (high_windows / selected_entry[:, None] - 1) * 10_000,
        )
        target_hit = favorable >= target[selection, None]
        stop_hit = adverse >= stop[selection, None]
        has_target = target_hit.any(axis=1)
        has_stop = stop_hit.any(axis=1)
        first_target = np.where(has_target, target_hit.argmax(axis=1), bars)
        first_stop = np.where(has_stop, stop_hit.argmax(axis=1), bars)
        outcome = np.select(
            (
                has_target & (first_target < first_stop),
                has_stop & (first_stop <= first_target),
            ),
            (0, 1),
            default=2,
        ).astype(np.int8)
        exit_offset = np.select(
            (outcome == 0, outcome == 1),
            (first_target, first_stop),
            default=bars - 1,
        ).astype(np.int32)
        terminal = side * (
            raw_close[indexes + bars - 1] / selected_entry - 1
        ) * 10_000
        realized = np.select(
            (outcome == 0, outcome == 1),
            (target[selection], -stop[selection]),
            default=terminal,
        ).astype(float)
        stopped = outcome == 1
        if stopped.any():
            stop_indexes = indexes[stopped] + first_stop[stopped]
            gap = side * (
                raw_open[stop_indexes] / selected_entry[stopped] - 1
            ) * 10_000
            realized[stopped] = np.minimum(-stop[selection][stopped], gap)
        result_outcome[selection] = outcome
        result_realized[selection] = realized
        result_exit[selection] = (exit_offset + 1) * 5
    return result_outcome, result_realized, result_exit


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        cached = pd.read_parquet(MATRIX)
        if len(cached) and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached
    _status("matrix", "Loading official Binance 5-second aggTrades", 3)
    raw = _load_raw()
    micro = build_micro_features(raw)
    micro["available_at"] = pd.to_datetime(micro["available_at"], utc=True)
    decision = micro.loc[
        micro["available_at"].dt.second.mod(DECISION_SECONDS).eq(0)
    ].copy()
    decision = pd.merge_asof(
        decision.sort_values("available_at"),
        _market_features().sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    decision = pd.merge_asof(
        decision.sort_values("available_at"),
        _macro_features().sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    raw_timestamp_ns = _nanoseconds(raw["timestamp"])
    available_ns = _nanoseconds(decision["available_at"])
    entry_indexes = np.searchsorted(raw_timestamp_ns, available_ns, side="left")
    maximum_bars = MAX_HORIZON_SECONDS // 5
    valid_path = entry_indexes + maximum_bars <= len(raw)
    finite = np.isfinite(decision.loc[:, FEATURES].to_numpy(float)).all(axis=1)
    decision = decision.loc[valid_path & finite].copy().reset_index(drop=True)
    entry_indexes = entry_indexes[valid_path & finite]
    decision["entry_timestamp"] = pd.to_datetime(
        raw["timestamp"].iloc[entry_indexes].to_numpy(), utc=True
    )
    decision["entry_price"] = raw["open"].to_numpy(float)[entry_indexes]
    entry = decision["entry_price"].to_numpy(float)
    atr = decision["atr_1m_bps"].to_numpy(float)
    for plan_number, plan in enumerate(PLANS, start=1):
        _status(
            "matrix",
            f"Barrier labels {plan_number}/{len(PLANS)}: {plan.name}",
            8 + 12 * plan_number,
        )
        target, stop = _plan_levels(atr, plan)
        for side, suffix in ((1, "long"), (-1, "short")):
            outcome, gross, exit_seconds = _barrier_outcomes(
                entry=entry,
                entry_indexes=entry_indexes,
                side=side,
                target=target,
                stop=stop,
                horizon_seconds=plan.horizon_seconds,
                raw=raw,
            )
            stem = f"{plan.name.lower()}_{suffix}"
            decision[f"{stem}_target_bps"] = target
            decision[f"{stem}_stop_bps"] = stop
            decision[f"{stem}_outcome"] = outcome
            decision[f"{stem}_gross_bps"] = gross
            decision[f"{stem}_exit_seconds"] = exit_seconds
    decision["protocol_hash"] = PROTOCOL_HASH
    decision["day"] = decision["entry_timestamp"].dt.floor("D")
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    decision.to_parquet(temporary, index=False)
    temporary.replace(MATRIX)
    _status("matrix", f"{len(decision):,} causal decisions every 30 seconds", 48)
    return decision


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Binance policy features must be complete; imputation is forbidden")
    return np.asarray(values, dtype=float)


def _classifier(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1_000, random_state=seed),
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda",
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=20.0,
        random_state=seed,
        n_jobs=4,
    )


def _regressor(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=300,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=20.0,
        random_state=seed,
        n_jobs=4,
    )


def _three_class_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    predicted = np.asarray(model.predict_proba(values), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    result = np.zeros((len(values), 3), dtype=float)
    result[:, classes] = predicted
    return result


def _fit_head(
    kind: str,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    plan: Plan,
    side: int,
) -> dict[str, Any]:
    suffix = "long" if side > 0 else "short"
    stem = f"{plan.name.lower()}_{suffix}"
    truth = fit[f"{stem}_outcome"].to_numpy(int)
    seed = 100 * list(PLANS).index(plan) + (0 if side > 0 else 1)
    model = _classifier(kind, seed).fit(_x(fit), truth)
    raw = np.clip(_three_class_probabilities(model, _x(calibration)), 1e-6, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1_000, random_state=seed)
    calibrator.fit(np.log(raw), calibration[f"{stem}_outcome"].to_numpy(int))
    timeout = truth == 2
    if timeout.sum() < 100:
        raise ValueError(f"Insufficient timeout labels for {stem}")
    timeout_model = _regressor(kind, seed + 50).fit(
        _x(fit.loc[timeout]), fit.loc[timeout, f"{stem}_gross_bps"].to_numpy(float)
    )
    return {
        "name": stem,
        "plan": plan,
        "side": side,
        "outcome_model": model,
        "calibrator": calibrator,
        "timeout_model": timeout_model,
    }


def _fit_models(kind: str, fit: pd.DataFrame, calibration: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        _fit_head(kind, fit, calibration, plan, side)
        for plan in PLANS
        for side in (1, -1)
    ]


def _score_head(head: dict[str, Any], rows: pd.DataFrame) -> dict[str, np.ndarray]:
    values = _x(rows)
    raw = np.clip(
        _three_class_probabilities(head["outcome_model"], values), 1e-6, 1
    )
    probability = _three_class_probabilities(head["calibrator"], np.log(raw))
    timeout = np.asarray(head["timeout_model"].predict(values), dtype=float)
    stem = cast(str, head["name"])
    target = rows[f"{stem}_target_bps"].to_numpy(float)
    stop = rows[f"{stem}_stop_bps"].to_numpy(float)
    expected_gross = (
        probability[:, 0] * target
        - probability[:, 1] * stop
        + probability[:, 2] * timeout
    )
    return {
        "score": expected_gross - COST_BPS,
        "p_target": probability[:, 0],
        "p_stop": probability[:, 1],
        "p_timeout": probability[:, 2],
        "timeout": timeout,
    }


def _funding_events() -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_parquet(FUNDING_SOURCE, columns=["timestamp", "funding_event_rate"])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.loc[rows["funding_event_rate"].ne(0)].sort_values("timestamp")
    times = _nanoseconds(rows["timestamp"])
    cumulative = np.r_[0.0, np.cumsum(rows["funding_event_rate"].to_numpy(float))]
    return times, cumulative


def _observed_funding_bps(
    entry: pd.Series, exit_time: pd.Series, side: np.ndarray
) -> np.ndarray:
    times, cumulative = _funding_events()
    entry_ns = _nanoseconds(entry)
    exit_ns = _nanoseconds(exit_time)
    left = np.searchsorted(times, entry_ns, side="right")
    right = np.searchsorted(times, exit_ns, side="right")
    return np.asarray(-side * (cumulative[right] - cumulative[left]) * 10_000, dtype=float)


def score(rows: pd.DataFrame, heads: list[dict[str, Any]]) -> pd.DataFrame:
    best = np.full(len(rows), -np.inf)
    chosen = np.zeros(len(rows), dtype=int)
    p_target = np.zeros(len(rows))
    p_stop = np.zeros(len(rows))
    p_timeout = np.zeros(len(rows))
    for number, head in enumerate(heads):
        values = _score_head(head, rows)
        better = values["score"] > best
        best[better] = values["score"][better]
        chosen[better] = number
        p_target[better] = values["p_target"][better]
        p_stop[better] = values["p_stop"][better]
        p_timeout[better] = values["p_timeout"][better]
    output = rows.loc[:, ["available_at", "entry_timestamp", "entry_price", "day"]].copy()
    output["score"] = best
    output["p_target"] = p_target
    output["p_stop"] = p_stop
    output["p_timeout"] = p_timeout
    output["plan"] = [heads[number]["plan"].name for number in chosen]
    output["side"] = np.asarray([heads[number]["side"] for number in chosen], dtype=int)
    indexes = np.arange(len(rows))

    def selected_values(suffix: str, dtype: type[float] | type[int]) -> np.ndarray:
        matrix = np.column_stack(
            [rows[f"{head['name']}_{suffix}"].to_numpy(dtype) for head in heads]
        )
        return np.asarray(matrix[indexes, chosen], dtype=dtype)

    output["gross_bps"] = selected_values("gross_bps", float)
    output["target_bps"] = selected_values("target_bps", float)
    output["stop_bps"] = selected_values("stop_bps", float)
    exit_seconds = selected_values("exit_seconds", int)
    output["exit_timestamp"] = pd.to_datetime(
        output["entry_timestamp"], utc=True
    ) + pd.to_timedelta(exit_seconds, unit="s")
    funding = _observed_funding_bps(
        output["entry_timestamp"], output["exit_timestamp"], output["side"].to_numpy(int)
    )
    output["funding_bps"] = funding
    output["net_bps"] = output["gross_bps"] + funding - COST_BPS
    output["stress_bps"] = output["gross_bps"] + funding - STRESS_COST_BPS
    return output.sort_values("entry_timestamp").reset_index(drop=True)


def execute(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = scored.loc[scored["score"].ge(threshold)].sort_values("entry_timestamp")
    accepted: list[Hashable] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in candidates.iterrows():
        if row["entry_timestamp"] < blocked_until:
            continue
        accepted.append(index)
        blocked_until = pd.Timestamp(row["exit_timestamp"])
    return candidates.loc[accepted].reset_index(drop=True)


def _bootstrap_daily_lcb(trades: pd.DataFrame, seed: int = 42) -> float | None:
    if trades.empty:
        return None
    daily = trades.groupby("day")["net_bps"].agg(["sum", "count"])
    if len(daily) < 10:
        return None
    values = daily[["sum", "count"]].to_numpy(float)
    block = min(5, len(values))
    random = np.random.default_rng(seed)
    means = np.empty(1_000)
    for sample in range(len(means)):
        joined: list[np.ndarray] = []
        while sum(len(item) for item in joined) < len(values):
            start = int(random.integers(0, len(values) - block + 1))
            joined.append(values[start : start + block])
        draw = np.concatenate(joined)[: len(values)]
        means[sample] = draw[:, 0].sum() / draw[:, 1].sum()
    return float(np.quantile(means, 0.05))


def metrics(
    trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, stress: bool = False
) -> dict[str, float | None]:
    value_column = "stress_bps" if stress else "net_bps"
    days = int((end.floor("D") - start.floor("D")).days)
    if trades.empty:
        return {
            "trades": 0.0,
            "trades_per_calendar_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "positive_calendar_days": 0.0,
            "max_drawdown": None,
            "bootstrap_lcb_95_bps": None,
        }
    values = trades[value_column].to_numpy(float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    daily = trades.groupby("day")[value_column].sum().reindex(
        pd.date_range(start.floor("D"), end.floor("D"), inclusive="left", freq="D"),
        fill_value=0.0,
    )
    risk_return = np.maximum(values / (trades["stop_bps"].to_numpy(float) + COST_BPS), -2) * 0.01
    equity = np.cumprod(1 + risk_return)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    lcb_rows = trades.copy()
    lcb_rows["net_bps"] = values
    return {
        "trades": float(len(trades)),
        "trades_per_calendar_day": float(len(trades) / days),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()),
        "positive_calendar_days": float((daily > 0).mean()),
        "max_drawdown": float((1 - equity / peaks).max()),
        "bootstrap_lcb_95_bps": _bootstrap_daily_lcb(lcb_rows),
    }


def _gates(value: dict[str, float | None]) -> dict[str, bool]:
    return {
        "frequency_3_per_calendar_day": float(value["trades_per_calendar_day"] or 0) >= 3,
        "expectancy_positive": value["expectancy_bps"] is not None
        and float(value["expectancy_bps"]) > 0,
        "profit_factor_1_15": value["profit_factor"] is not None
        and float(value["profit_factor"]) >= 1.15,
        "majority_positive_calendar_days": float(value["positive_calendar_days"] or 0) > 0.5,
        "drawdown_8pct": value["max_drawdown"] is not None
        and float(value["max_drawdown"]) <= 0.08,
        "bootstrap_lcb_positive": value["bootstrap_lcb_95_bps"] is not None
        and float(value["bootstrap_lcb_95_bps"]) > 0,
    }


def _period(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    purge_end = end - pd.Timedelta(seconds=MAX_HORIZON_SECONDS)
    return rows.loc[
        rows["available_at"].ge(start) & rows["available_at"].lt(purge_end)
    ].copy()


def _diagnostic(
    score_history: pd.DataFrame,
    scored: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    coverage: float = 0.001,
) -> dict[str, Any]:
    threshold = float(score_history["score"].quantile(1 - coverage))
    trades = execute(scored, threshold)
    return {
        "coverage": coverage,
        "threshold": threshold,
        "metrics": metrics(trades, start, end),
        "stress_costs_2x": metrics(trades, start, end, stress=True),
    }


def train() -> dict[str, Any]:
    rows = build_matrix()
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    split = {
        "fit": _period(rows, start, FIT_END),
        "calibration": _period(rows, FIT_END, CALIBRATION_END),
        "model_selection": _period(rows, CALIBRATION_END, MODEL_SELECTION_END),
        "policy_selection": _period(rows, MODEL_SELECTION_END, POLICY_SELECTION_END),
        "audit": _period(rows, POLICY_SELECTION_END, AUDIT_END),
    }
    if min(len(value) for value in split.values()) < 20_000:
        raise ValueError("Insufficient causal Binance 30-second decisions in a split")
    candidates: dict[str, dict[str, Any]] = {}
    for number, kind in enumerate(("ridge", "xgboost"), start=1):
        _status("training", f"{kind} probabilistic heads {number}/2", 52 + 14 * number)
        heads = _fit_models(kind, split["fit"], split["calibration"])
        calibration_scored = score(split["calibration"], heads)
        selection_scored = score(split["model_selection"], heads)
        candidates[kind] = {
            "heads": heads,
            "calibration": calibration_scored,
            "model_selection": selection_scored,
            "diagnostic": _diagnostic(
                calibration_scored,
                selection_scored,
                CALIBRATION_END,
                MODEL_SELECTION_END,
            ),
        }
    ridge = candidates["ridge"]["diagnostic"]["metrics"]
    xgboost = candidates["xgboost"]["diagnostic"]["metrics"]
    champion = (
        "xgboost"
        if float(xgboost["expectancy_bps"] or -np.inf)
        > float(ridge["expectancy_bps"] or -np.inf)
        and float(xgboost["profit_factor"] or 0) >= float(ridge["profit_factor"] or 0)
        else "ridge"
    )
    chosen = candidates[champion]
    _status("policy_selection", f"{champion}: April causal frequency frontier", 84)
    policy_scored = score(split["policy_selection"], chosen["heads"])
    curve: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        threshold = float(chosen["model_selection"]["score"].quantile(1 - coverage))
        trades = execute(policy_scored, threshold)
        value = metrics(trades, MODEL_SELECTION_END, POLICY_SELECTION_END)
        stress = metrics(
            trades, MODEL_SELECTION_END, POLICY_SELECTION_END, stress=True
        )
        gates = _gates(value)
        gates["stress_costs_2x_nonnegative"] = (
            stress["expectancy_bps"] is not None
            and float(stress["expectancy_bps"]) >= 0
        )
        curve.append(
            {
                "coverage": coverage,
                "threshold": threshold,
                "metrics": value,
                "stress_costs_2x": stress,
                "gates": gates,
            }
        )
    passing = [point for point in curve if all(point["gates"].values())]
    selected = max(
        passing,
        key=lambda point: float(point["metrics"]["trades_per_calendar_day"] or 0),
        default=None,
    )
    _status("audit", "Frozen April coverage on May-June reused discovery audit", 94)
    audit_scored = score(split["audit"], chosen["heads"])
    if selected is not None:
        audit_threshold = float(
            policy_scored["score"].quantile(1 - float(selected["coverage"]))
        )
        audit_trades = execute(audit_scored, audit_threshold)
    else:
        audit_threshold = None
        audit_trades = audit_scored.iloc[0:0].copy()
    audit_metrics = metrics(audit_trades, POLICY_SELECTION_END, AUDIT_END)
    audit_stress = metrics(
        audit_trades, POLICY_SELECTION_END, AUDIT_END, stress=True
    )
    audit_gates = _gates(audit_metrics)
    audit_gates["stress_costs_2x_nonnegative"] = (
        audit_stress["expectancy_bps"] is not None
        and float(audit_stress["expectancy_bps"]) >= 0
    )
    passed = selected is not None and all(audit_gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "rows": len(rows),
            "features": len(FEATURES),
            "rows_by_split": {name: len(value) for name, value in split.items()},
            "sealed_july_rows_read": 0,
        },
        "causal_checks": {
            "entry_at_or_after_feature_availability": bool(
                rows["entry_timestamp"].ge(rows["available_at"]).all()
            ),
            "same_bar_stop_wins": True,
            "missing_features_imputed": False,
            "cross_exchange_features": False,
            "observed_funding": True,
            "one_position": True,
        },
        "model_selection": {
            kind: value["diagnostic"] for kind, value in candidates.items()
        },
        "champion": champion,
        "policy_selection_april": {"curve": curve, "selected": selected},
        "reused_discovery_audit_may_june": {
            "threshold": audit_threshold,
            "metrics": audit_metrics,
            "stress_costs_2x": audit_stress,
            "gates": audit_gates,
            "long": int(audit_trades["side"].gt(0).sum()),
            "short": int(audit_trades["side"].lt(0).sum()),
            "plans": {
                str(key): int(value)
                for key, value in audit_trades["plan"].value_counts().items()
            },
        },
        "diagnostic_audit_at_0_1pct_if_no_policy": (
            None
            if selected is not None
            else _diagnostic(
                policy_scored,
                audit_scored,
                POLICY_SELECTION_END,
                AUDIT_END,
            )
        ),
        "verdict": (
            "BINANCE_HIGH_FREQUENCY_RESEARCH_POLICY_READY"
            if passed
            else "NO_SUSTAINABLE_BINANCE_HIGH_FREQUENCY_ALPHA"
        ),
        "paper_policy_changed": False,
        "sealed_july_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    if passed:
        assert selected is not None
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        temporary = BUNDLE.with_suffix(".tmp")
        joblib.dump(
            {
                "protocol": PROTOCOL,
                "protocol_hash": PROTOCOL_HASH,
                "features": FEATURES,
                "heads": chosen["heads"],
                "champion": champion,
                "coverage": selected["coverage"],
                "score_threshold": audit_threshold,
                "research_only": True,
                "live_orders_enabled": False,
            },
            temporary,
        )
        temporary.replace(BUNDLE)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


if __name__ == "__main__":
    train()
