from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_bot.binance_public_book_depth import FEATURES as BOOK_DEPTH_FEATURES
from adaptive_bot.bitunix_fees import FUTURES_VIP_FEE_BPS
from adaptive_bot.musca_v5_cross_exchange import CROSS_FEATURES
from adaptive_bot.musca_v5_cross_exchange import OUTPUT as CROSS_EXCHANGE_FEATURES
from adaptive_bot.musca_v5_market_state import MARKET_STATE_FEATURES, MARKET_STATE_METADATA
from adaptive_bot.musca_v5_market_state import OUTPUT as MARKET_STATE_PATH
from adaptive_bot.musca_v5_micro_model import (
    DIRECTIONAL_MICRO_FEATURES,
    MICRO_FEATURES,
    build_micro_features,
    load_5s,
)
from adaptive_bot.musca_v5_research import EVENTS, ML_FEATURES

ROOT = Path("data/ml/musca_v5")
MATRIX = ROOT / "event_policy_matrix.parquet"
BUNDLE = Path("data/models/musca_v5/event_policy.joblib")
REPORT = Path("data/reports/musca_v5_event_policy.json")
STATUS = Path("data/reports/musca_v5_event_policy.status.json")

NORMAL_COST_BPS = 2 * FUTURES_VIP_FEE_BPS[0][1] + 1.0
STRESS_COST_BPS = 2 * NORMAL_COST_BPS
MINIMUM_GROSS_TO_COST = 3.0
CALIBRATION_THRESHOLDS_BPS = (0.0, 2.0, 4.0)
MICRO_STOP_LOOKBACK_BUCKETS = 36
MINIMUM_STOP_BPS = 12.0
MAXIMUM_MICRO_STOP_BPS = 60.0
INVALIDATION_BUCKETS = 6
ENTRY_CONFIRMATION_BUCKETS = 60
MINIMUM_HOLD_BUCKETS = 60
LABEL_HORIZONS_MINUTES = (5, 15, 30, 60)
BARRIER_TARGETS_BPS = (10, 20, 30, 50)


@dataclass(frozen=True)
class ManagementPlan:
    name: str
    first_target_bps: float
    second_target_bps: float
    first_exit_fraction: float
    trailing_giveback_bps: float
    maximum_minutes: int
    first_risk_multiple: float = 0.0
    second_risk_multiple: float = 0.0


PLANS = (
    ManagementPlan("MICRO_30_55", 30.0, 55.0, 0.50, 10.0, 30, 1.0, 1.5),
    ManagementPlan("STANDARD_40_70", 40.0, 70.0, 0.50, 12.0, 60, 1.2, 2.0),
    ManagementPlan("EXTENDED_50_100", 50.0, 100.0, 0.40, 16.0, 120, 1.5, 3.0),
)

PROTOCOL = {
    "name": "musca_v5_event_driven_management",
    "asset": "BTCUSDT",
    "feature_venue": "Binance USD-M and spot",
    "execution_calibration_venue": "Bitunix VIP0 shadow",
    "entry": (
        "after a completed 5m candidate, wait at most 5 minutes for aligned 15s price restart, "
        "15s/1m aggressive flow and trade intensity; execute at next 5s bucket open"
    ),
    "plans": [asdict(plan) for plan in PLANS],
    "normal_round_trip_cost_bps": NORMAL_COST_BPS,
    "stress_round_trip_cost_bps": STRESS_COST_BPS,
    "same_bucket": "stop wins",
    "stop": (
        "candidate structural pullback/VWAP invalidation; require at least 12 bps and no more "
        "than 100 bps for a micro setup; never widens"
    ),
    "invalidation": (
        "adverse multi-window flow is recorded for the management model; it is not an "
        "automatic exit because discovery audit demonstrated fee churn"
    ),
    "management": (
        "partial first target; then choose close or runner; economic break-even protection; "
        "trailing and second target; event-driven exit before time-stop"
    ),
    "minimum_weighted_planned_gross_to_cost": MINIMUM_GROSS_TO_COST,
    "labels": {
        "mfe_mae_minutes": LABEL_HORIZONS_MINUTES,
        "barriers_bps": BARRIER_TARGETS_BPS,
        "same_bucket": "stop wins",
    },
    "split": (
        "January fit; February calibration; March discovery audit. March has already "
        "been inspected and is not an untouched final holdout; confirmation requires "
        "strictly later data."
    ),
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

DEPTH_FEATURES = (
    "depth_imbalance_1pct",
    "depth_imbalance_2pct",
    "depth_imbalance_5pct",
    "depth_skew_change_1m",
    "depth_skew_change_5m",
    "log_depth_1pct",
)
POLICY_CROSS_FEATURES = (
    "aligned_median_return_1m_bps",
    "dispersion_return_1m_bps",
    "aligned_binance_lag_1m_bps",
    "aligned_median_return_5m_bps",
    "dispersion_return_5m_bps",
    "aligned_binance_lag_5m_bps",
    "aligned_median_return_15m_bps",
    "dispersion_return_15m_bps",
    "aligned_binance_lag_15m_bps",
    "aligned_bybit_basis_bps",
    "aligned_okx_basis_bps",
)
POLICY_MARKET_FEATURES = (
    "aligned_daily_vwap_distance_bps",
    "aligned_rolling_vwap_distance_bps",
    "aligned_rolling_vwap_slope_bps",
    "aligned_rolling_vwap_slope_change_bps",
    "rolling_vwap_tests_1h",
    "rolling_vwap_rejections_1h",
    "time_since_rolling_vwap_cross_minutes",
    "rolling_vwap_rejection_strength_bps",
    "aligned_rolling_vwap_band_position",
    "aligned_swing_distance_bps",
    "aligned_swing_slope_bps",
    "aligned_swing_slope_change_bps",
    "swing_anchor_age_bars",
    "swing_anchor_price",
    "swing_volume_since_anchor",
    "aligned_swing_return_since_anchor_bps",
    "swing_tests_1h",
    "swing_rejections_1h",
    "swing_rejection_strength_bps",
    "aligned_rolling_swing_distance_bps",
    "rolling_swing_convergence_bps",
    *(f"atr_{minutes}m_bps" for minutes in (1, 5, 15, 30)),
    *(f"aligned_trend_{minutes}m" for minutes in (1, 5, 15, 30)),
    *(f"aligned_vwap_state_{minutes}m" for minutes in (1, 5, 15, 30)),
    "realized_volatility_30m_bps",
    "volume_percentile",
)

POLICY_FEATURES = (
    *ML_FEATURES,
    *MICRO_FEATURES,
    *DEPTH_FEATURES,
    *POLICY_CROSS_FEATURES,
    *POLICY_MARKET_FEATURES,
    "plan_first_target_bps",
    "plan_second_target_bps",
    "plan_first_exit_fraction",
    "plan_trailing_giveback_bps",
    "plan_maximum_minutes",
)
MODEL_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        {"label_protocol_hash": PROTOCOL_HASH, "policy_features": POLICY_FEATURES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
CONTINUATION_FEATURES = (
    *POLICY_FEATURES,
    "time_to_first_target_seconds",
    "tp1_ofi_15s",
    "tp1_ofi_1m",
    "tp1_ofi_persistence_1m",
    "tp1_trade_intensity_15s",
    "tp1_price_velocity_15s",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def calculate_net_ev_bps(
    probability_gross_win: np.ndarray | float,
    conditional_gross_gain_bps: np.ndarray | float,
    conditional_gross_loss_bps: np.ndarray | float,
    expected_total_cost_bps: np.ndarray | float,
) -> np.ndarray:
    """Return economic EV, keeping market movement and execution costs separate."""
    probability = np.asarray(probability_gross_win, dtype=float)
    gain = np.asarray(conditional_gross_gain_bps, dtype=float)
    loss = np.asarray(conditional_gross_loss_bps, dtype=float)
    cost = np.asarray(expected_total_cost_bps, dtype=float)
    if (
        np.any((probability < 0) | (probability > 1))
        or np.any(gain < 0)
        or np.any(loss < 0)
        or np.any(cost < 0)
    ):
        raise ValueError("Probabilities and economic inputs are outside valid bounds")
    result = probability * gain - (1.0 - probability) * loss - cost
    return cast(np.ndarray, result)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": round(percent, 2),
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _directional_return(side: int, entry: float, price: float) -> float:
    return side * (price / entry - 1.0) * 10_000


def _stop_fill_bps(side: int, entry: float, open_price: float, stop_bps: float) -> float:
    open_return = _directional_return(side, entry, open_price)
    return min(open_return, stop_bps)


def _economically_protected_stop(
    plan: ManagementPlan,
    first_target_bps: float | None = None,
    cost_bps: float = NORMAL_COST_BPS,
) -> float:
    remaining = 1.0 - plan.first_exit_fraction
    target = plan.first_target_bps if first_target_bps is None else first_target_bps
    required = (
        cost_bps
        + 2.0
        - plan.first_exit_fraction * target
    ) / remaining
    return max(0.0, required)


def _weighted_planned_gross_bps(
    plan: ManagementPlan, first_target_bps: float, second_target_bps: float
) -> float:
    return (
        plan.first_exit_fraction * first_target_bps
        + (1.0 - plan.first_exit_fraction) * second_target_bps
    )


def _causal_micro_stop_bps(
    side: int, entry: float, data: pd.DataFrame, entry_index: int
) -> float | None:
    start = max(0, entry_index - MICRO_STOP_LOOKBACK_BUCKETS)
    history = data.iloc[start:entry_index]
    if len(history) < 12:
        return None
    bucket_range_bps = (
        (history["high"].to_numpy(float) - history["low"].to_numpy(float))
        / history["close"].to_numpy(float)
        * 10_000
    )
    buffer_bps = max(2.0, 0.25 * float(np.median(bucket_range_bps)))
    extreme = float(history["low"].min() if side > 0 else history["high"].max())
    extreme_bps = _directional_return(side, entry, extreme) - buffer_bps
    distance = max(MINIMUM_STOP_BPS, -extreme_bps)
    if not np.isfinite(distance) or distance > MAXIMUM_MICRO_STOP_BPS:
        return None
    return -distance


def _causal_entry_index(
    event: dict[str, Any], data: pd.DataFrame, available_ns: np.ndarray
) -> int | None:
    available = pd.Timestamp(event["available_at"])
    first = int(np.searchsorted(available_ns, available.value, side="left"))
    side = int(event["direction"])
    last = min(first + ENTRY_CONFIRMATION_BUCKETS, len(data) - 2)
    for current in range(first, last + 1):
        row = data.iloc[current]
        confirmed = (
            side * float(row["price_velocity_15s"]) >= 0.5
            and side * float(row["ofi_15s"]) >= 0.05
            and side * float(row["ofi_1m"]) >= -0.02
            and float(row["trade_intensity_15s"]) >= 0.8
        )
        if confirmed:
            return current + 1
    return None


def _multi_horizon_path_labels(
    data: pd.DataFrame,
    *,
    entry_index: int,
    side: int,
    entry: float,
    stop_bps: float,
    first_target_bps: float,
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    maximum = min(
        entry_index + max(LABEL_HORIZONS_MINUTES) * 12,
        len(data) - 1,
    )
    maximum_path = data.iloc[entry_index + 1 : maximum + 1]
    favorable_series = maximum_path["high"] if side > 0 else maximum_path["low"]
    adverse_series = maximum_path["low"] if side > 0 else maximum_path["high"]
    favorable_path = side * (favorable_series.to_numpy(float) / entry - 1.0) * 10_000
    adverse_path = side * (adverse_series.to_numpy(float) / entry - 1.0) * 10_000
    for minutes in LABEL_HORIZONS_MINUTES:
        end = min(entry_index + minutes * 12, len(data) - 1)
        count = end - entry_index
        labels[f"mfe_{minutes}m_bps"] = float(
            np.max(favorable_path[:count], initial=0.0)
        )
        labels[f"mae_{minutes}m_bps"] = float(
            np.min(adverse_path[:count], initial=0.0)
        )
        labels[f"label_available_at_{minutes}m"] = pd.Timestamp(
            data.iloc[end]["available_at"]
        )

    targets = (*BARRIER_TARGETS_BPS, first_target_bps)
    outcomes: dict[float, tuple[bool, float]] = {}
    stop_positions = np.flatnonzero(adverse_path <= stop_bps)
    first_stop = int(stop_positions[0]) if len(stop_positions) else len(adverse_path)
    for target in targets:
        target_positions = np.flatnonzero(favorable_path >= target)
        first_target = (
            int(target_positions[0]) if len(target_positions) else len(favorable_path)
        )
        # Zero-based positions are equal when both barriers occur in the same bucket.
        hit = first_target < first_stop and first_target < len(favorable_path)
        seconds = float((first_target + 1) * 5) if hit else np.nan
        outcomes[float(target)] = hit, seconds

    for target in BARRIER_TARGETS_BPS:
        hit, seconds = outcomes[float(target)]
        labels[f"hit_{target}bps_before_stop"] = hit
        labels[f"time_to_{target}bps_seconds"] = seconds
    first_hit, first_seconds = outcomes[float(first_target_bps)]
    for minutes in LABEL_HORIZONS_MINUTES:
        labels[f"first_target_within_{minutes}m"] = bool(
            first_hit and first_seconds <= minutes * 60
        )
    return labels


def simulate_management(
    event: dict[str, Any],
    data: pd.DataFrame,
    plan: ManagementPlan,
    *,
    cost_bps: float = NORMAL_COST_BPS,
    minimum_gross_to_cost: float = MINIMUM_GROSS_TO_COST,
    stop_style: str = "structural",
    available_ns: np.ndarray | None = None,
    entry_style: str = "micro_confirmation",
    dynamic_protected_trailing: bool = False,
) -> dict[str, Any] | None:
    """Simulate one causal path; the maximum horizon is only a time-stop."""
    if available_ns is None:
        available_ns = pd.to_datetime(data["available_at"], utc=True).to_numpy(
            dtype="datetime64[ns]"
        ).astype("int64")
    if entry_style == "micro_confirmation":
        entry_index = _causal_entry_index(event, data, available_ns)
    elif entry_style == "immediate_next_bucket":
        current = int(
            np.searchsorted(
                available_ns, pd.Timestamp(event["available_at"]).value, side="left"
            )
        )
        entry_index = current + 1 if current + 1 < len(data) else None
    else:
        raise ValueError(f"Unsupported entry style: {entry_style}")
    if entry_index is None:
        return None
    entry = float(data.iloc[entry_index]["open"])
    side = int(event["direction"])
    if stop_style == "micro":
        micro_stop = _causal_micro_stop_bps(side, entry, data, entry_index)
        if micro_stop is None:
            return None
        initial_stop_bps = micro_stop
    elif stop_style == "structural":
        structural_stop_bps = _directional_return(
            side, entry, float(event["stop_price"])
        )
        if (
            not np.isfinite(structural_stop_bps)
            or structural_stop_bps > -MINIMUM_STOP_BPS
            or structural_stop_bps < -100.0
        ):
            return None
        initial_stop_bps = structural_stop_bps
    else:
        raise ValueError(f"Unsupported stop style: {stop_style}")
    risk_bps = -initial_stop_bps
    first_target_bps = max(
        plan.first_target_bps, plan.first_risk_multiple * risk_bps
    )
    second_target_bps = max(
        plan.second_target_bps, plan.second_risk_multiple * risk_bps
    )
    if (
        _weighted_planned_gross_bps(plan, first_target_bps, second_target_bps)
        < minimum_gross_to_cost * cost_bps
    ):
        return None
    last = min(entry_index + plan.maximum_minutes * 12, len(data) - 1)
    if last <= entry_index:
        return None

    remaining = 1.0
    realized_gross = 0.0
    stop_bps = initial_stop_bps
    peak_bps = 0.0
    first_target_index: int | None = None
    effective_trailing_giveback_bps = plan.trailing_giveback_bps
    runner_gross_bps: float | None = None
    exit_index = last
    exit_reason = f"TIME_STOP_{plan.maximum_minutes}M"
    final_gross = _directional_return(side, entry, float(data.iloc[last]["close"]))
    mfe_bps = 0.0
    mae_bps = 0.0
    invalidation_count = 0
    maximum_adverse_flow_run = 0

    for current in range(entry_index + 1, last + 1):
        row = data.iloc[current]
        high = float(row["high"])
        low = float(row["low"])
        favorable_price = high if side > 0 else low
        adverse_price = low if side > 0 else high
        favorable = _directional_return(side, entry, favorable_price)
        adverse = _directional_return(side, entry, adverse_price)
        mfe_bps = max(mfe_bps, favorable)
        mae_bps = min(mae_bps, adverse)

        # With no tick ordering inside a 5-second bucket, the adverse outcome wins.
        if adverse <= stop_bps:
            fill = _stop_fill_bps(side, entry, float(row["open"]), stop_bps)
            final_gross = realized_gross + remaining * fill
            exit_index = current
            exit_reason = "INITIAL_STOP" if first_target_index is None else "PROTECTED_TRAIL"
            runner_gross_bps = fill if first_target_index is not None else None
            break

        if first_target_index is None and favorable >= first_target_bps:
            first_target_index = current
            realized_gross = plan.first_exit_fraction * first_target_bps
            remaining = 1.0 - plan.first_exit_fraction
            peak_bps = first_target_bps
            protected_stop = _economically_protected_stop(
                plan, first_target_bps, cost_bps
            )
            stop_bps = max(stop_bps, protected_stop)
            if dynamic_protected_trailing:
                effective_trailing_giveback_bps = max(
                    0.0, first_target_bps - protected_stop
                )
            continue

        if first_target_index is not None:
            if favorable >= second_target_bps:
                runner_gross_bps = second_target_bps
                final_gross = realized_gross + remaining * runner_gross_bps
                exit_index = current
                exit_reason = "SECOND_TARGET"
                break
            peak_bps = max(peak_bps, favorable)
            stop_bps = max(
                stop_bps, peak_bps - effective_trailing_giveback_bps
            )

        adverse_flow = (
            current - entry_index >= MINIMUM_HOLD_BUCKETS
            and side * float(row["ofi_15s"]) < -0.15
            and side * float(row["ofi_1m"]) < -0.08
            and side * float(row["ofi_5m"]) < -0.03
            and side * float(row["price_velocity_1m"]) < -0.5
        )
        invalidation_count = invalidation_count + 1 if adverse_flow else 0
        maximum_adverse_flow_run = max(maximum_adverse_flow_run, invalidation_count)

    if exit_reason.startswith("TIME_STOP") and first_target_index is not None:
        runner_gross_bps = _directional_return(
            side, entry, float(data.iloc[exit_index]["close"])
        )
        final_gross = realized_gross + remaining * runner_gross_bps

    close_at_first_gross = (
        first_target_bps if first_target_index is not None else final_gross
    )
    continuation_increment = (
        final_gross - close_at_first_gross if first_target_index is not None else np.nan
    )
    entry_available = pd.Timestamp(data.iloc[entry_index]["available_at"])
    entry_decision_available = pd.Timestamp(data.iloc[entry_index - 1]["available_at"])
    exit_available = pd.Timestamp(data.iloc[exit_index]["available_at"])
    result = event | {
        "protocol_hash": PROTOCOL_HASH,
        "plan": plan.name,
        "plan_first_target_bps": first_target_bps,
        "plan_second_target_bps": second_target_bps,
        "plan_first_exit_fraction": plan.first_exit_fraction,
        "plan_trailing_giveback_bps": plan.trailing_giveback_bps,
        "effective_trailing_giveback_bps": effective_trailing_giveback_bps,
        "plan_maximum_minutes": plan.maximum_minutes,
        "expected_round_trip_cost_bps": cost_bps,
        "entry_timestamp": entry_available,
        "entry_decision_available_at": entry_decision_available,
        "exit_timestamp": exit_available,
        "entry_price": entry,
        "initial_stop_bps": initial_stop_bps,
        "final_stop_bps": stop_bps,
        "first_target_hit": first_target_index is not None,
        "time_to_first_target_seconds": (
            float((first_target_index - entry_index) * 5)
            if first_target_index is not None
            else np.nan
        ),
        "runner_gross_bps": runner_gross_bps,
        "close_at_first_net_bps": close_at_first_gross - cost_bps,
        "continuation_increment_bps": continuation_increment,
        "gross_return_bps": final_gross,
        "net_return_bps": final_gross - cost_bps,
        "stress_return_bps": final_gross - 2 * cost_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "duration_seconds": float((exit_index - entry_index) * 5),
        "maximum_adverse_flow_run": maximum_adverse_flow_run,
        "exit_reason": exit_reason,
    }
    result |= _multi_horizon_path_labels(
        data,
        entry_index=entry_index,
        side=side,
        entry=entry,
        stop_bps=initial_stop_bps,
        first_target_bps=first_target_bps,
    )
    if first_target_index is not None:
        target_row = data.iloc[first_target_index]
        result |= {
            "tp1_ofi_15s": side * float(target_row["ofi_15s"]),
            "tp1_ofi_1m": side * float(target_row["ofi_1m"]),
            "tp1_ofi_persistence_1m": side
            * float(target_row["ofi_persistence_1m"]),
            "tp1_trade_intensity_15s": float(target_row["trade_intensity_15s"]),
            "tp1_price_velocity_15s": side * float(target_row["price_velocity_15s"]),
        }
    return result


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        cached = pd.read_parquet(MATRIX)
        if len(cached) and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
            if "expected_round_trip_cost_bps" not in cached:
                cached["expected_round_trip_cost_bps"] = NORMAL_COST_BPS
            return _attach_market_state(
                _attach_cross_exchange(_attach_historical_depth(cached))
            )
    _status("event_labels", 5, "Loading verified Binance 5-second aggregate trades")
    raw = load_5s()
    micro = build_micro_features(raw)
    data = raw.merge(micro, on="available_at", how="inner", validate="one_to_one")
    events = pd.read_parquet(EVENTS)
    event_time = pd.to_datetime(events["available_at"], utc=True)
    events = events.loc[event_time.ge("2026-01-01") & event_time.lt("2026-04-01")].copy()
    event_micro = pd.merge_asof(
        events.sort_values("available_at"),
        micro.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=5),
    ).dropna(subset=list(MICRO_FEATURES))
    for feature in DIRECTIONAL_MICRO_FEATURES:
        event_micro[feature] = event_micro[feature] * event_micro["direction"]
    timestamp = pd.to_datetime(event_micro["available_at"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    weekday = timestamp.dt.dayofweek
    event_micro["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    event_micro["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    event_micro["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    event_micro["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    rows: list[dict[str, Any]] = []
    records = cast(list[dict[str, Any]], event_micro.to_dict("records"))
    for number, event in enumerate(records, start=1):
        for plan in PLANS:
            outcome = simulate_management(event, data, plan)
            if outcome is not None:
                rows.append(outcome)
        if number % 250 == 0:
            _status(
                "event_labels",
                5 + 40 * number / max(1, len(records)),
                f"{number:,}/{len(records):,} causal candidates",
            )
    matrix = pd.DataFrame(rows)
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    temporary.replace(MATRIX)
    return _attach_market_state(_attach_cross_exchange(_attach_historical_depth(matrix)))


def _attach_historical_depth(matrix: pd.DataFrame) -> pd.DataFrame:
    if not BOOK_DEPTH_FEATURES.exists():
        raise ValueError("Official Binance historical book-depth features are unavailable")
    depth = pd.read_parquet(BOOK_DEPTH_FEATURES)
    depth = depth.loc[depth["coverage_valid"]].copy()
    depth["log_depth_1pct"] = np.log1p(
        depth["bid_depth_1pct"] + depth["ask_depth_1pct"]
    )
    source = matrix.drop(columns=list(DEPTH_FEATURES), errors="ignore").copy()
    source["available_at"] = pd.to_datetime(source["available_at"], utc=True)
    depth["available_at"] = pd.to_datetime(depth["available_at"], utc=True)
    joined = pd.merge_asof(
        source.sort_values("available_at"),
        depth.sort_values("available_at")[["available_at", *DEPTH_FEATURES]],
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    )
    return joined.dropna(subset=list(DEPTH_FEATURES)).reset_index(drop=True)


def _attach_cross_exchange(matrix: pd.DataFrame) -> pd.DataFrame:
    if not CROSS_EXCHANGE_FEATURES.exists():
        raise ValueError("BTC Binance/Bybit/OKX causal feature history is unavailable")
    context = pd.read_parquet(CROSS_EXCHANGE_FEATURES)
    context = context.loc[context["coverage_valid"]].copy()
    context["available_at"] = pd.to_datetime(context["available_at"], utc=True)
    source = matrix.drop(columns=list(POLICY_CROSS_FEATURES), errors="ignore").copy()
    source["available_at"] = pd.to_datetime(source["available_at"], utc=True)
    joined = pd.merge_asof(
        source.sort_values("available_at"),
        context.sort_values("available_at")[["available_at", *CROSS_FEATURES]],
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    )
    side = joined["direction"].to_numpy(float)
    for minutes in (1, 5, 15):
        joined[f"aligned_median_return_{minutes}m_bps"] = (
            side * joined[f"median_return_{minutes}m_bps"]
        )
        joined[f"aligned_binance_lag_{minutes}m_bps"] = (
            side * joined[f"binance_lag_{minutes}m_bps"]
        )
    joined["aligned_bybit_basis_bps"] = side * joined["bybit_basis_to_binance_bps"]
    joined["aligned_okx_basis_bps"] = side * joined["okx_basis_to_binance_bps"]
    return joined.dropna(subset=list(POLICY_CROSS_FEATURES)).reset_index(drop=True)


def _attach_market_state(matrix: pd.DataFrame) -> pd.DataFrame:
    if not MARKET_STATE_PATH.exists():
        raise ValueError("BTC multi-timeframe VWAP market state is unavailable")
    state = pd.read_parquet(MARKET_STATE_PATH)
    state = state.loc[state["coverage_valid"]].copy()
    state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
    source = matrix.drop(columns=list(POLICY_MARKET_FEATURES), errors="ignore").copy()
    source["available_at"] = pd.to_datetime(source["available_at"], utc=True)
    joined = pd.merge_asof(
        source.sort_values("available_at"),
        state.sort_values("available_at")[
            ["available_at", *MARKET_STATE_METADATA, *MARKET_STATE_FEATURES]
        ],
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=6),
    )
    side = joined["direction"].to_numpy(float)
    for name in (
        "daily_vwap_distance_bps",
        "rolling_vwap_distance_bps",
        "rolling_vwap_slope_bps",
        "rolling_vwap_slope_change_bps",
        "rolling_vwap_band_position",
    ):
        joined[f"aligned_{name}"] = side * joined[name]
    joined["aligned_swing_distance_bps"] = np.where(
        side > 0,
        side * joined["swing_long_distance_bps"],
        side * joined["swing_short_distance_bps"],
    )
    joined["aligned_swing_slope_bps"] = np.where(
        side > 0,
        side * joined["swing_long_slope_bps"],
        side * joined["swing_short_slope_bps"],
    )
    joined["anchor_timestamp"] = np.where(
        side > 0,
        joined["swing_long_anchor_timestamp"],
        joined["swing_short_anchor_timestamp"],
    )
    joined["anchor_type"] = np.where(
        side > 0, joined["swing_long_anchor_type"], joined["swing_short_anchor_type"]
    )
    joined["aligned_swing_slope_change_bps"] = np.where(
        side > 0,
        side * joined["swing_long_slope_change_bps"],
        side * joined["swing_short_slope_change_bps"],
    )
    for target, source_long, source_short in (
        ("swing_anchor_age_bars", "swing_long_anchor_age_bars", "swing_short_anchor_age_bars"),
        ("swing_anchor_price", "swing_long_anchor_price", "swing_short_anchor_price"),
        (
            "swing_volume_since_anchor",
            "swing_long_volume_since_anchor",
            "swing_short_volume_since_anchor",
        ),
        ("swing_tests_1h", "swing_long_tests_1h", "swing_short_tests_1h"),
        (
            "swing_rejections_1h",
            "swing_long_rejections_1h",
            "swing_short_rejections_1h",
        ),
        (
            "swing_rejection_strength_bps",
            "swing_long_rejection_strength_bps",
            "swing_short_rejection_strength_bps",
        ),
        (
            "rolling_swing_convergence_bps",
            "rolling_swing_long_convergence_bps",
            "rolling_swing_short_convergence_bps",
        ),
    ):
        joined[target] = np.where(side > 0, joined[source_long], joined[source_short])
    joined["aligned_swing_return_since_anchor_bps"] = np.where(
        side > 0,
        side * joined["swing_long_return_since_anchor_bps"],
        side * joined["swing_short_return_since_anchor_bps"],
    )
    joined["aligned_rolling_swing_distance_bps"] = np.where(
        side > 0,
        side * joined["rolling_swing_long_distance_bps"],
        side * joined["rolling_swing_short_distance_bps"],
    )
    for minutes in (1, 5, 15, 30):
        joined[f"aligned_trend_{minutes}m"] = side * joined[f"trend_{minutes}m"]
        joined[f"aligned_vwap_state_{minutes}m"] = (
            side * joined[f"vwap_state_{minutes}m"]
        )
    return joined.dropna(subset=list(POLICY_MARKET_FEATURES)).reset_index(drop=True)


def _x(rows: pd.DataFrame, features: tuple[str, ...] = POLICY_FEATURES) -> np.ndarray:
    values = rows.loc[:, features].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Musca V5 policy features must be finite and causally covered")
    return values


def _regressor(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=seed,
    )


def _classifier(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2_000, random_state=seed),
        )
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=seed,
    )


def _fit_probability_head(
    kind: str, fit: pd.DataFrame, calibration: pd.DataFrame, target: str, seed: int
) -> dict[str, Any]:
    model = _classifier(kind, seed).fit(_x(fit), fit[target].astype(int))
    raw = np.asarray(model.predict_proba(_x(calibration))[:, 1], float)
    calibrator = LogisticRegression(C=1.0, random_state=seed).fit(
        raw.reshape(-1, 1), calibration[target].astype(int)
    )
    return {"model": model, "calibrator": calibrator}


def _predict_probability(head: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(head["model"].predict_proba(_x(rows))[:, 1], float)
    return np.asarray(
        head["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1], float
    )


def _fit_economic_heads(
    kind: str, fit: pd.DataFrame, calibration: pd.DataFrame, seed: int
) -> dict[str, Any]:
    fit = fit.copy()
    calibration = calibration.copy()
    fit["net_win"] = fit["net_return_bps"].gt(0)
    calibration["net_win"] = calibration["net_return_bps"].gt(0)
    fit["gross_win"] = fit["gross_return_bps"].gt(0)
    calibration["gross_win"] = calibration["gross_return_bps"].gt(0)
    net_probability = _fit_probability_head(kind, fit, calibration, "net_win", seed)
    gross_probability = _fit_probability_head(
        kind, fit, calibration, "gross_win", seed + 1
    )
    first_target = _fit_probability_head(
        kind, fit, calibration, "first_target_hit", seed + 2
    )
    positive = fit["gross_win"]
    negative = ~positive
    gain = _regressor(kind, seed + 3).fit(
        _x(fit.loc[positive]), fit.loc[positive, "gross_return_bps"]
    )
    loss = _regressor(kind, seed + 4).fit(
        _x(fit.loc[negative]), -fit.loc[negative, "gross_return_bps"]
    )
    mfe = _regressor(kind, seed + 5).fit(_x(fit), fit["mfe_bps"])
    mae = _regressor(kind, seed + 6).fit(_x(fit), -fit["mae_bps"])
    target_rows = fit["first_target_hit"] & fit["time_to_first_target_seconds"].notna()
    time_to_target = _regressor(kind, seed + 7).fit(
        _x(fit.loc[target_rows]), fit.loc[target_rows, "time_to_first_target_seconds"] / 60
    )
    horizon_mfe = {
        minutes: _regressor(kind, seed + 10 + minutes).fit(
            _x(fit), fit[f"mfe_{minutes}m_bps"]
        )
        for minutes in LABEL_HORIZONS_MINUTES
    }
    horizon_mae = {
        minutes: _regressor(kind, seed + 20 + minutes).fit(
            _x(fit), -fit[f"mae_{minutes}m_bps"]
        )
        for minutes in LABEL_HORIZONS_MINUTES
    }
    barrier_probability = {
        target: _fit_probability_head(
            kind,
            fit,
            calibration,
            f"hit_{target}bps_before_stop",
            seed + 30 + target,
        )
        for target in BARRIER_TARGETS_BPS
    }
    target_within_probability = {
        minutes: _fit_probability_head(
            kind,
            fit,
            calibration,
            f"first_target_within_{minutes}m",
            seed + 90 + minutes,
        )
        for minutes in LABEL_HORIZONS_MINUTES
    }
    return {
        "p_net_win": net_probability,
        "p_gross_win": gross_probability,
        "p_first_target": first_target,
        "conditional_gross_gain": gain,
        "conditional_gross_loss": loss,
        "expected_mfe": mfe,
        "expected_mae": mae,
        "expected_time_to_target": time_to_target,
        "expected_mfe_by_horizon": horizon_mfe,
        "expected_mae_by_horizon": horizon_mae,
        "barrier_probability": barrier_probability,
        "target_within_probability": target_within_probability,
    }


def _economic_predictions(heads: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    net_probability = _predict_probability(heads["p_net_win"], rows)
    gross_probability = _predict_probability(heads["p_gross_win"], rows)
    target_probability = _predict_probability(heads["p_first_target"], rows)
    gain = np.maximum(0.0, heads["conditional_gross_gain"].predict(_x(rows)))
    loss = np.maximum(0.0, heads["conditional_gross_loss"].predict(_x(rows)))
    expected_cost = rows["expected_round_trip_cost_bps"].to_numpy(float)
    gross_ev = gross_probability * gain - (1.0 - gross_probability) * loss
    output: dict[str, Any] = {
            "predicted_win_probability": net_probability,
            "predicted_gross_win_probability": gross_probability,
            "predicted_target_probability": target_probability,
            "predicted_conditional_gross_gain_bps": gain,
            "predicted_conditional_gross_loss_bps": loss,
            "predicted_gross_ev_bps": gross_ev,
            "predicted_total_cost_bps": expected_cost,
            "predicted_ev_bps": calculate_net_ev_bps(
                gross_probability, gain, loss, expected_cost
            ),
            "predicted_mfe_bps": np.maximum(0.0, heads["expected_mfe"].predict(_x(rows))),
            "predicted_mae_bps": np.maximum(0.0, heads["expected_mae"].predict(_x(rows))),
            "predicted_time_to_target_minutes": np.maximum(
                0.0, heads["expected_time_to_target"].predict(_x(rows))
            ),
    }
    for minutes in LABEL_HORIZONS_MINUTES:
        output[f"predicted_mfe_{minutes}m_bps"] = np.maximum(
            0.0, heads["expected_mfe_by_horizon"][minutes].predict(_x(rows))
        )
        output[f"predicted_mae_{minutes}m_bps"] = np.maximum(
            0.0, heads["expected_mae_by_horizon"][minutes].predict(_x(rows))
        )
        output[f"predicted_p_target_within_{minutes}m"] = _predict_probability(
            heads["target_within_probability"][minutes], rows
        )
    for target in BARRIER_TARGETS_BPS:
        output[f"predicted_p_{target}bps_before_stop"] = _predict_probability(
            heads["barrier_probability"][target], rows
        )
    return pd.DataFrame(output, index=rows.index)


def score_economic_candidates(
    bundle: dict[str, Any],
    candidates: pd.DataFrame,
    *,
    expected_total_cost_bps: float,
) -> pd.DataFrame:
    """Score complete setup-plan rows with the current venue cost estimate."""
    if expected_total_cost_bps < 0 or not np.isfinite(expected_total_cost_bps):
        raise ValueError("Current execution cost must be finite and non-negative")
    if tuple(bundle.get("policy_features", ())) != POLICY_FEATURES:
        raise ValueError("The V5 model feature protocol does not match the scorer")
    rows = candidates.copy()
    rows["expected_round_trip_cost_bps"] = expected_total_cost_bps
    predictions = _economic_predictions(bundle["economic_heads"], rows)
    for column in predictions:
        rows[column] = predictions[column]
    rows["planned_gross_bps"] = (
        rows["plan_first_exit_fraction"] * rows["plan_first_target_bps"]
        + (1 - rows["plan_first_exit_fraction"]) * rows["plan_second_target_bps"]
    )
    rows["movement_covers_cost"] = rows["planned_gross_bps"].ge(
        MINIMUM_GROSS_TO_COST * expected_total_cost_bps
    )
    return rows


def select_economic_decision(
    scored: pd.DataFrame,
    *,
    entry_threshold_bps: float,
) -> dict[str, Any]:
    """Apply the economic gate after setup detection and before deterministic risk checks."""
    if scored.empty:
        return {"action": "WAIT", "reason": "NO_VWAP_AVWAP_CANDIDATE"}
    eligible = scored.loc[scored["movement_covers_cost"]].copy()
    if eligible.empty:
        best = cast(
            dict[str, Any],
            scored.loc[scored["predicted_ev_bps"].idxmax()].to_dict(),
        )
        return {
            "action": "NO_TRADE",
            "reason": "EXPECTED_MOVEMENT_BELOW_3X_COST",
            "best_predicted_ev_bps": float(best["predicted_ev_bps"]),
            "expected_total_cost_bps": float(best["predicted_total_cost_bps"]),
            "planned_gross_bps": float(best["planned_gross_bps"]),
        }
    best = cast(
        dict[str, Any],
        eligible.loc[eligible["predicted_ev_bps"].idxmax()].to_dict(),
    )
    threshold = max(0.0, float(entry_threshold_bps))
    action = "LONG" if int(best["direction"]) > 0 else "SHORT"
    if float(best["predicted_ev_bps"]) < threshold:
        action = "NO_TRADE"
    return {
        "action": action,
        "reason": "POSITIVE_NET_EV" if action != "NO_TRADE" else "NET_EV_BELOW_THRESHOLD",
        "setup": str(best["event_family"]),
        "direction": "LONG" if int(best["direction"]) > 0 else "SHORT",
        "management_plan": str(best["plan"]),
        "entry_price": float(best["entry_price"]),
        "technical_stop_bps": abs(float(best["initial_stop_bps"])),
        "first_target_bps": float(best["plan_first_target_bps"]),
        "second_target_bps": float(best["plan_second_target_bps"]),
        "first_exit_fraction": float(best["plan_first_exit_fraction"]),
        "maximum_holding_minutes": int(best["plan_maximum_minutes"]),
        "predicted_net_win_probability": float(best["predicted_win_probability"]),
        "predicted_gross_win_probability": float(
            best["predicted_gross_win_probability"]
        ),
        "predicted_target_probability": float(best["predicted_target_probability"]),
        "predicted_gross_ev_bps": float(best["predicted_gross_ev_bps"]),
        "predicted_net_ev_bps": float(best["predicted_ev_bps"]),
        "expected_total_cost_bps": float(best["predicted_total_cost_bps"]),
        "expected_mfe_bps": float(best["predicted_mfe_bps"]),
        "expected_mae_bps": float(best["predicted_mae_bps"]),
        "expected_time_to_target_minutes": float(
            best["predicted_time_to_target_minutes"]
        ),
        "anchor_type": str(best.get("anchor_type", "UNKNOWN")),
        "anchor_timestamp": str(best.get("anchor_timestamp", "")),
        "model_protocol_hash": MODEL_PROTOCOL_HASH,
    }


def _metrics(rows: pd.DataFrame, column: str = "net_return_bps") -> dict[str, float]:
    if column not in rows:
        return {
            "trades": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown_bps": 0.0,
        }
    values = rows[column].to_numpy(float)
    if not len(values):
        return {"trades": 0.0, "expectancy_bps": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "max_drawdown_bps": 0.0}
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    equity = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.r_[0.0, equity])[1:] - equity
    return {
        "trades": float(len(values)),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "win_rate": float((values > 0).mean()),
        "max_drawdown_bps": float(drawdown.max(initial=0.0)),
    }


def _one_position(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values(["entry_timestamp", "predicted_ev_bps"], ascending=[True, False])
    accepted: list[Any] = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    seen: set[pd.Timestamp] = set()
    for index, row in ordered.iterrows():
        entry = pd.Timestamp(row["entry_timestamp"])
        signal = pd.Timestamp(row["available_at"])
        if signal in seen or entry < busy_until:
            continue
        seen.add(signal)
        accepted.append(index)
        busy_until = pd.Timestamp(row["exit_timestamp"])
    return ordered.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def _bootstrap_lcb(values: np.ndarray, *, seed: int = 20260808) -> float:
    if len(values) < 10:
        return float("-inf")
    rng = np.random.default_rng(seed)
    block = max(2, round(np.sqrt(len(values))))
    samples = np.empty(2_000)
    starts = np.arange(max(1, len(values) - block + 1))
    for number in range(len(samples)):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < len(values):
            start = int(rng.choice(starts))
            pieces.append(values[start : start + block])
        samples[number] = np.concatenate(pieces)[: len(values)].mean()
    return float(np.quantile(samples, 0.05))


def _setup_audit(matrix: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for (family, plan), rows in matrix.groupby(["event_family", "plan"], observed=True):
        gross_profit = rows.loc[rows["gross_return_bps"].gt(0), "gross_return_bps"].sum()
        metrics = _metrics(rows)
        result[f"{family}/{plan}"] = {
            "trade_count": float(len(rows)),
            "gross_win_rate": float(rows["gross_return_bps"].gt(0).mean()),
            "net_win_rate": float(rows["net_return_bps"].gt(0).mean()),
            "mean_mfe_bps": float(rows["mfe_bps"].mean()),
            "median_mfe_bps": float(rows["mfe_bps"].median()),
            "mean_mae_bps": float(rows["mae_bps"].mean()),
            "median_mae_bps": float(rows["mae_bps"].median()),
            "mean_time_to_target_minutes": float(
                rows["time_to_first_target_seconds"].mean() / 60
            ),
            "median_time_to_target_minutes": float(
                rows["time_to_first_target_seconds"].median() / 60
            ),
            "mean_cost_bps": float(rows["expected_round_trip_cost_bps"].mean()),
            "mean_net_pnl_bps": float(rows["net_return_bps"].mean()),
            "profit_factor": metrics["profit_factor"],
            "expectancy_bps": metrics["expectancy_bps"],
            "tp_before_sl_probability": float(rows["first_target_hit"].mean()),
            "cost_to_gross_profit_ratio": float(
                rows["expected_round_trip_cost_bps"].sum() / gross_profit
            )
            if gross_profit > 0
            else float("inf"),
        }
    return result


def _selected(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = scored.loc[scored["predicted_ev_bps"].ge(threshold)].copy()
    if candidates.empty:
        return candidates
    best = candidates.loc[candidates.groupby("available_at")["predicted_ev_bps"].idxmax()]
    return _one_position(best)


def train(*, force_matrix: bool = False) -> dict[str, Any]:
    matrix = build_matrix(force=force_matrix).sort_values("entry_timestamp").reset_index(drop=True)
    timestamp = pd.to_datetime(matrix["entry_timestamp"], utc=True)
    fit = matrix.loc[timestamp.lt("2026-02-01")].copy()
    calibration = matrix.loc[timestamp.ge("2026-02-01") & timestamp.lt("2026-03-01")].copy()
    audit = matrix.loc[timestamp.ge("2026-03-01") & timestamp.lt("2026-04-01")].copy()
    if min(len(fit), len(calibration), len(audit)) < 100:
        raise RuntimeError("Insufficient independent chronological rows for V5 event policy")
    _status("gpu_training", 50, f"fit={len(fit)} calibration={len(calibration)} audit={len(audit)}")

    candidates: dict[str, dict[str, Any]] = {}
    trained: dict[str, Any] = {}
    for position, kind in enumerate(("ridge", "xgboost"), start=1):
        heads = _fit_economic_heads(kind, fit, calibration, 20260808)
        predictions = _economic_predictions(heads, calibration)
        calibration_scored = calibration.copy()
        for column in predictions:
            calibration_scored[column] = predictions[column]
        choices: list[tuple[float, float, dict[str, float]]] = []
        for threshold in CALIBRATION_THRESHOLDS_BPS:
            selected = _selected(calibration_scored, threshold)
            metric = _metrics(selected)
            if metric["trades"] >= 20 and metric["profit_factor"] >= 1.0:
                choices.append((metric["expectancy_bps"], threshold, metric))
        threshold = max(choices, default=(-float("inf"), float("inf"), {}))[1]
        calibration_metrics = max(choices, default=(0.0, threshold, _metrics(pd.DataFrame())))[2]
        candidates[kind] = {
            "calibration_ev_mae_bps": float(
                mean_absolute_error(
                    calibration["net_return_bps"], predictions["predicted_ev_bps"]
                )
            ),
            "calibration_win_brier": float(
                brier_score_loss(
                    calibration["net_return_bps"].gt(0),
                    predictions["predicted_win_probability"],
                )
            ),
            "calibration_gross_win_brier": float(
                brier_score_loss(
                    calibration["gross_return_bps"].gt(0),
                    predictions["predicted_gross_win_probability"],
                )
            ),
            "calibration_target_brier": float(
                brier_score_loss(
                    calibration["first_target_hit"],
                    predictions["predicted_target_probability"],
                )
            ),
            "calibration_barrier_brier": float(
                np.mean(
                    [
                        brier_score_loss(
                            calibration[f"hit_{target}bps_before_stop"],
                            predictions[f"predicted_p_{target}bps_before_stop"],
                        )
                        for target in BARRIER_TARGETS_BPS
                    ]
                )
            ),
            "calibration_horizon_excursion_mae_bps": float(
                np.mean(
                    [
                        mean_absolute_error(
                            calibration[f"mfe_{minutes}m_bps"],
                            predictions[f"predicted_mfe_{minutes}m_bps"],
                        )
                        + mean_absolute_error(
                            -calibration[f"mae_{minutes}m_bps"],
                            predictions[f"predicted_mae_{minutes}m_bps"],
                        )
                        for minutes in LABEL_HORIZONS_MINUTES
                    ]
                )
            ),
            "threshold_bps": threshold,
            "calibration_policy": calibration_metrics,
        }
        trained[kind] = heads
        _status("gpu_training", 50 + position * 15, f"{kind} complete")

    ridge_policy = candidates["ridge"]["calibration_policy"]
    xgb_policy = candidates["xgboost"]["calibration_policy"]
    xgb_wins = (
        candidates["xgboost"]["calibration_ev_mae_bps"]
        < candidates["ridge"]["calibration_ev_mae_bps"]
        and candidates["xgboost"]["calibration_win_brier"]
        < candidates["ridge"]["calibration_win_brier"]
        and candidates["xgboost"]["calibration_gross_win_brier"]
        < candidates["ridge"]["calibration_gross_win_brier"]
        and candidates["xgboost"]["calibration_target_brier"]
        < candidates["ridge"]["calibration_target_brier"]
        and candidates["xgboost"]["calibration_barrier_brier"]
        < candidates["ridge"]["calibration_barrier_brier"]
        and candidates["xgboost"]["calibration_horizon_excursion_mae_bps"]
        < candidates["ridge"]["calibration_horizon_excursion_mae_bps"]
        and xgb_policy.get("expectancy_bps", -np.inf)
        > ridge_policy.get("expectancy_bps", -np.inf)
        and xgb_policy.get("profit_factor", 0.0) >= ridge_policy.get("profit_factor", 0.0)
    )
    champion = "xgboost" if xgb_wins else "ridge"
    chosen = trained[champion]
    audit_predictions = _economic_predictions(chosen, audit)
    for column in audit_predictions:
        audit[column] = audit_predictions[column]
    selected = _selected(audit, float(candidates[champion]["threshold_bps"]))
    audit_metrics = _metrics(selected)
    stress_metrics = _metrics(selected, "stress_return_bps")
    lcb = _bootstrap_lcb(selected["net_return_bps"].to_numpy(float))

    continuation = matrix.loc[matrix["first_target_hit"]].dropna(
        subset=list(CONTINUATION_FEATURES)
    )
    continuation_time = pd.to_datetime(continuation["entry_timestamp"], utc=True)
    continuation_fit = continuation.loc[continuation_time.lt("2026-03-01")]
    continuation_audit = continuation.loc[continuation_time.ge("2026-03-01")].copy()
    continuation_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        _x(continuation_fit, CONTINUATION_FEATURES),
        continuation_fit["continuation_increment_bps"],
    )
    continuation_audit["predicted_increment_bps"] = continuation_model.predict(
        _x(continuation_audit, CONTINUATION_FEATURES)
    )
    close_or_continue = np.where(
        continuation_audit["predicted_increment_bps"].to_numpy(float) > 0,
        continuation_audit["net_return_bps"].to_numpy(float),
        continuation_audit["close_at_first_net_bps"].to_numpy(float),
    )
    continuation_policy = {
        "decisions": len(continuation_audit),
        "continue_rate": float(
            continuation_audit["predicted_increment_bps"].gt(0).mean()
        ),
        "dynamic_expectancy_bps": float(close_or_continue.mean()),
        "always_close_expectancy_bps": float(
            continuation_audit["close_at_first_net_bps"].mean()
        ),
        "always_continue_expectancy_bps": float(continuation_audit["net_return_bps"].mean()),
    }

    oracle = matrix.loc[timestamp.ge("2026-03-01") & timestamp.lt("2026-04-01")]
    oracle = oracle.loc[oracle.groupby("available_at")["net_return_bps"].idxmax()]
    oracle = oracle.loc[oracle["net_return_bps"].gt(0)]
    oracle = _one_position(oracle.assign(predicted_ev_bps=oracle["net_return_bps"]))
    oracle_metrics = _metrics(oracle)
    gates = {
        "perfect_foresight_opportunities_exist": oracle_metrics["trades"] > 0,
        "audit_trades_30": audit_metrics["trades"] >= 30,
        "audit_expectancy_positive": audit_metrics["expectancy_bps"] > 0,
        "audit_profit_factor_1_10": audit_metrics["profit_factor"] >= 1.10,
        "bootstrap_lcb_positive": lcb > 0,
        "stress_expectancy_nonnegative": audit_metrics["trades"] > 0
        and stress_metrics["expectancy_bps"] >= 0,
        "continuation_beats_static": continuation_policy["dynamic_expectancy_bps"]
        >= max(
            continuation_policy["always_close_expectancy_bps"],
            continuation_policy["always_continue_expectancy_bps"],
        ),
    }
    verdict = "RESEARCH_POLICY_READY" if all(gates.values()) else "NO_DEPLOYABLE_POLICY"
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "model_protocol_hash": MODEL_PROTOCOL_HASH,
        "champion": champion,
        "economic_heads": chosen,
        "entry_threshold_bps": candidates[champion]["threshold_bps"],
        "continuation_model": continuation_model,
        "policy_features": POLICY_FEATURES,
        "continuation_features": CONTINUATION_FEATURES,
        "plans": [asdict(plan) for plan in PLANS],
        "live_orders_enabled": False,
        "verdict": verdict,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(BUNDLE)
    report = {
        "status": verdict,
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "model_protocol_hash": MODEL_PROTOCOL_HASH,
        "model_output": {
            "probabilities": [
                "P(net profitable after configured cost)",
                "P(gross profitable)",
                "P(first target before stop/time-stop)",
            ],
            "movement": [
                "conditional gross gain/loss bps",
                "expected MFE/MAE at 5m, 15m, 30m and 60m",
                "expected time to target minutes",
            ],
            "barriers": [
                "P(10/20/30/50 bps before technical stop)",
                "P(proposed target within 5m/15m/30m/60m)",
            ],
            "decision_equation": (
                "P(gross win)*E[gross gain|win] - P(gross loss)*"
                "E[gross loss|loss] - fee - observed spread - slippage - funding"
            ),
            "dynamic_target": (
                "score every preregistered target/stop/maximum-duration plan and choose "
                "the highest positive net EV; require planned gross movement >= 3x cost"
            ),
        },
        "cost_feature_coverage": {
            "historical_alpha_spread": "UNAVAILABLE_NOT_IMPUTED",
            "historical_training_cost_bps": NORMAL_COST_BPS,
            "bitunix_live_spread": "OBSERVED_FROM_BEST_BID_ASK",
            "bitunix_depth": "OBSERVED_L2_SHADOW_ONLY",
            "binance_historical_depth": (
                "OFFICIAL_DAILY_BOOK_DEPTH_CHECKSUM_VERIFIED_2026_01_TO_2026_03"
            ),
            "execution_rule": (
                "fail closed if current spread/depth/staleness is unavailable; execution "
                "features are not backfilled into Alpha"
            ),
        },
        "rows": {"matrix": len(matrix), "fit": len(fit), "calibration": len(calibration),
                 "audit": len(audit)},
        "setup_statistics": _setup_audit(matrix),
        "candidate_models": candidates,
        "champion": champion,
        "perfect_foresight_opportunity_ceiling": oracle_metrics,
        "selected_audit": audit_metrics,
        "selected_audit_stress_2x": stress_metrics,
        "bootstrap_expectancy_lcb_95_bps": lcb,
        "continuation_audit": continuation_policy,
        "gates": gates,
        "bundle": str(BUNDLE),
        "real_capital_allowed": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT, report)
    _status("complete", 100, verdict)
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=str))
