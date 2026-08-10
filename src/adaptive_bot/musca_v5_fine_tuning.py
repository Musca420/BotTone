from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.musca_v4_research import BARS, MINUTES, build_events, label_events
from adaptive_bot.musca_v5_event_policy import NORMAL_COST_BPS, PLANS, ManagementPlan
from adaptive_bot.musca_v5_frequency_audit import CANDIDATES
from adaptive_bot.musca_v5_market_state import (
    MARKET_STATE_FEATURES,
)
from adaptive_bot.musca_v5_market_state import (
    OUTPUT as MARKET_STATE_PATH,
)
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    HORIZONS,
    profile_cost_bps,
)
from adaptive_bot.musca_v8_multi_horizon import (
    PROTOCOL_HASH as BASE_PROTOCOL_HASH,
)

ROOT = Path("data/ml/musca_v5")
MATRIX = ROOT / "fine_tuning_matrix.parquet"
REPORT = Path("data/reports/musca_v5_fine_tuning.json")
BASE_REPORT = Path("data/reports/musca_v8_multi_horizon.json")
STATUS = Path("data/reports/musca_v5_fine_tuning.status.json")
BUNDLE = Path("data/models/musca_v5/fine_tuning.joblib")
POLICY_MATRIX = ROOT / "cost_linked_policy_matrix.parquet"
POLICY_REPORT = Path("data/reports/musca_v5_policy_economics.json")
POLICY_STATUS = Path("data/reports/musca_v5_policy_economics.status.json")
WALK_FORWARD_REPORT = Path("data/reports/musca_v5_policy_walk_forward.json")
WALK_FORWARD_STATUS = Path("data/reports/musca_v5_policy_walk_forward.status.json")
WALK_FORWARD_BUNDLE = Path("data/models/musca_v5/policy_walk_forward.joblib")
BINANCE_CONTEXT = Path("data/ml/hybrid_v19/binance_context_5m.parquet")
BINANCE_MARKET = Path("data/ml/hybrid_v25/asset=BTCUSDT/features_15m.parquet")
STOP_LOOKBACK_MINUTES = 5
MINIMUM_STOP_BPS = 12.0
MAXIMUM_STOP_BPS = 60.0
MINIMUM_SELECTION_TRADES = 50
COST_TARGET_MULTIPLIERS = (1.5, 2.0)
COST_TARGET_MAXIMUM_MINUTES = 60
RUNNER_FIRST_EXIT_FRACTION = 0.5
RUNNER_PROTECTED_PROFIT_BPS = 1.0
POLICY_COVERAGE_LEVELS = tuple(value / 10 for value in range(10, 0, -1))
WALK_FORWARD_STEP = pd.Timedelta(weeks=4)
WALK_FORWARD_CALIBRATION = pd.Timedelta(weeks=13)
WALK_FORWARD_SELECTION = pd.Timedelta(weeks=13)
BASE_PLAN = ManagementPlan("BASE_V8_1_5R", 0.0, 0.0, 0.50, 0.0, 360, 1.5, 0.0)
SCALP_PLAN = ManagementPlan("SCALP_20_50", 20.0, 50.0, 0.50, 10.0, 30, 0.75, 1.5)
MANAGEMENT_PLANS = (BASE_PLAN, SCALP_PLAN, *PLANS)
ENTRY_MICRO_FEATURES = (
    *(f"entry_return_{minutes}m_bps" for minutes in (1, 5, 15)),
    *(f"entry_taker_imbalance_{minutes}m" for minutes in (1, 5, 15)),
    "entry_trade_count_z_1h",
    "entry_aggressive_volume_z_1h",
    "entry_mark_divergence_bps",
    "entry_funding_rate",
    "entry_range_1m_bps",
    "entry_mark_spot_basis_bps",
    "entry_basis_change_1h_bps",
    "entry_funding_z",
    *(f"signed_entry_return_{minutes}m_bps" for minutes in (1, 5, 15)),
    *(f"signed_entry_taker_imbalance_{minutes}m" for minutes in (1, 5, 15)),
)
BINANCE_CONTEXT_FEATURES = (
    "binance_taker_imbalance_15m",
    "binance_taker_imbalance_1h",
    "binance_trade_count_z",
    "binance_aggressive_volume_z",
    "oi_change_1h",
    "oi_change_4h",
    "oi_change_24h",
    "oi_value_change_1h",
    "top_position_ratio",
    "global_long_short_ratio",
    "taker_long_short_ratio",
    "metrics_taker_ratio_z",
    "book_imbalance_1pct",
    "book_depth_log_z",
    "signed_binance_taker_imbalance_15m",
    "signed_binance_taker_imbalance_1h",
    "signed_book_imbalance_1pct",
    "signed_top_position_deviation",
    "signed_global_position_deviation",
    "signed_taker_ratio_deviation",
    "directional_price_oi_interaction_1h",
    "directional_price_oi_interaction_4h",
    "directional_price_oi_interaction_24h",
)
POLICY_STATE_INTERACTIONS = (
    "target_to_atr_1m",
    "target_to_atr_5m",
    "target_to_realized_volatility_30m",
    "stop_to_atr_5m",
    "flow_volume_confirmation",
    "orderflow_depth_confirmation",
)
FEATURES = (
    "direction",
    "return_1h",
    "return_4h",
    "ema_spread_atr",
    "relative_volume",
    "taker_imbalance",
    "spot_return_1h",
    "spot_perp_divergence_bps",
    "daily_distance_sigma",
    "impulse_distance_sigma",
    "swing_distance_sigma",
    "confluence_score",
    "remaining_zone_count",
    "pullback_depth_atr",
    "room_bps",
    "atr_percentile",
    "hour_sin",
    "hour_cos",
    "expert_breakout_bars",
    "anchor_cycle",
    "initial_stop_bps",
    "plan_first_target_bps",
    "plan_second_target_bps",
    "plan_first_exit_fraction",
    "plan_trailing_giveback_bps",
    "plan_maximum_minutes",
    "planned_gross_bps",
    *MARKET_STATE_FEATURES,
    *ENTRY_MICRO_FEATURES,
    *BINANCE_CONTEXT_FEATURES,
    *POLICY_STATE_INTERACTIONS,
)
PROTOCOL = {
    "name": "musca_v5_frequency_frontier_v1",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "candidate_source": str(CANDIDATES),
    "base_outcomes": "direct_frozen_v8_generator_and_labeler",
    "direction": "frozen_v8_only",
    "entry": "first_valid_1m_open_at_or_after_signal_available_at",
    "stop": {
        "style": "tighter_of_pullback_structure_and_prior_5m_swing",
        "buffer": "max_2bps_or_quarter_median_1m_range",
        "minimum_bps": MINIMUM_STOP_BPS,
        "maximum_bps": MAXIMUM_STOP_BPS,
        "never_widens": True,
    },
    "plans": [plan.name for plan in MANAGEMENT_PLANS],
    "features": list(FEATURES),
    "historical_context": {
        "source": str(BINANCE_CONTEXT),
        "market_source": str(BINANCE_MARKET),
        "join": "backward_only_15m_tolerance",
        "missing": "fail_closed_no_imputation",
    },
    "management_cost_protection_bps": NORMAL_COST_BPS,
    "economic_admission": "predicted_net_EV_and_prudent_bound_positive",
    "cost_stress": "2x_evaluated_separately_not_an_entry_multiple",
    "same_minute": "stop_wins",
    "model": "ridge_champion_xgboost_gpu_challenger",
    "target_decomposition": (
        "P(first_target_hit)*E[gross|hit] + "
        "P(no_first_target_hit)*E[signed_gross|no_hit]"
    ),
    "split": {
        "fit": "2024",
        "calibration": "2025_H1",
        "selection": "2025_H2",
        "audit": "2026_before_sealed_holdout",
    },
    "selection": "maximum_frequency_subject_to_positive_EV_PF_1_15_DD_8pct",
    "holdout_opened": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
POLICY_PROTOCOL = {
    "name": "musca_v5_cost_linked_runner_policy_real_cost_gate",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "candidate_source": str(CANDIDATES),
    "candidate_deduplication": "highest_preregistered_breakout_horizon_per_timestamp_side",
    "direction": "frozen_v8_only",
    "target_multipliers": list(COST_TARGET_MULTIPLIERS),
    "target": "tp1_max_cost_floor_1R_then_tp2_max_cost_floor_2R",
    "first_exit_fraction": RUNNER_FIRST_EXIT_FRACTION,
    "protected_net_profit_bps": RUNNER_PROTECTED_PROFIT_BPS,
    "trailing": "tp1_minus_cost_protected_stop_never_widens",
    "stop": "causal_structural_12_to_60bps_never_widened",
    "maximum_minutes": COST_TARGET_MAXIMUM_MINUTES,
    "intrabar": "stop_wins",
    "invalid_minute": "fail_closed",
    "selection": "policy_level_not_individual_trade_lcb",
    "economic_gate_costs": "observed_profile_costs_1x",
    "cost_stress": "2x_reported_as_diagnostic_only",
    "holdout_opened": False,
    "real_capital_allowed": False,
}
POLICY_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(POLICY_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
WALK_FORWARD_PROTOCOL = {
    "name": "musca_v5_cost_linked_policy_walk_forward_real_cost_gate",
    "outcome_protocol_hash": POLICY_PROTOCOL_HASH,
    "fit": "expanding_history_before_calibration",
    "calibration_weeks": 13,
    "selection_weeks": 13,
    "test_step_weeks": 4,
    "purge": "actual_exit_before_next_partition",
    "coverage_levels": list(POLICY_COVERAGE_LEVELS),
    "selection": "maximum_policy_coverage_subject_to_aggregate_gates",
    "ranking": (
        "continuous_uncalibrated_score; "
        "calibrated_EV_is_reported_not_used_for_tie_expansion"
    ),
    "economic_gate_costs": "observed_profile_costs_1x",
    "cost_stress": "2x_reported_as_diagnostic_only",
    "individual_positive_ev_required": False,
    "probability_calibration": "isotonic_on_separate_chronological_window",
    "ev_calibration": "isotonic_on_separate_chronological_window",
    "ridge": "default_champion",
    "xgboost_gpu": "challenger_same_splits",
    "features": list(FEATURES),
    "holdout_opened": False,
    "real_capital_allowed": False,
}
WALK_FORWARD_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(WALK_FORWARD_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _technical_stop_bps(
    event: dict[str, Any], minutes: pd.DataFrame, entry_index: int, entry: float
) -> float | None:
    side = int(event["direction"])
    history = minutes.iloc[max(0, entry_index - STOP_LOOKBACK_MINUTES) : entry_index]
    if len(history) < STOP_LOOKBACK_MINUTES:
        return None
    if "data_valid" in history and not history["data_valid"].fillna(False).all():
        return None
    ranges = (
        (history["perp_high"].to_numpy(float) - history["perp_low"].to_numpy(float))
        / history["perp_close"].to_numpy(float)
        * 10_000
    )
    buffer_bps = max(2.0, 0.25 * float(np.median(ranges)))
    recent_extreme = float(
        history["perp_low"].min() if side > 0 else history["perp_high"].max()
    )
    recent_stop = _directional_return(side, entry, recent_extreme) - buffer_bps
    structural_stop = _directional_return(side, entry, float(event["stop_price"]))
    valid = [value for value in (recent_stop, structural_stop) if value < 0]
    if not valid:
        return None
    distance = max(MINIMUM_STOP_BPS, -max(valid))
    if not np.isfinite(distance) or distance > MAXIMUM_STOP_BPS:
        return None
    return -distance


def _protected_stop_bps(
    plan: ManagementPlan,
    first_target_bps: float,
    cost_bps: float = NORMAL_COST_BPS,
) -> float:
    remaining = 1.0 - plan.first_exit_fraction
    required = (
        cost_bps
        + RUNNER_PROTECTED_PROFIT_BPS
        - plan.first_exit_fraction * first_target_bps
    ) / remaining
    return max(0.0, required)


def _simulate_plan(
    event: dict[str, Any],
    minutes: pd.DataFrame,
    times: pd.Series,
    plan: ManagementPlan,
    protection_cost_bps: float = NORMAL_COST_BPS,
) -> dict[str, Any] | None:
    entry_index = int(
        times.searchsorted(pd.Timestamp(event["available_at"]), side="left")
    )
    if entry_index >= len(minutes):
        return None
    if not bool(minutes.iloc[entry_index].get("data_valid", False)):
        return None
    entry = float(minutes.iloc[entry_index]["perp_open"])
    side = int(event["direction"])
    stop_bps = _technical_stop_bps(event, minutes, entry_index, entry)
    if stop_bps is None:
        return None
    risk_bps = -stop_bps
    first_target = max(plan.first_target_bps, plan.first_risk_multiple * risk_bps)
    second_target = max(plan.second_target_bps, plan.second_risk_multiple * risk_bps)
    planned_gross = (
        plan.first_exit_fraction * first_target
        + (1.0 - plan.first_exit_fraction) * second_target
    )
    last = min(entry_index + plan.maximum_minutes, len(minutes) - 1)
    if last <= entry_index:
        return None

    remaining = 1.0
    realized = 0.0
    funding = 0.0
    peak = 0.0
    mfe = 0.0
    mae = 0.0
    first_target_index: int | None = None
    funding_at_first_target = float("nan")
    exit_index = last
    exit_reason = f"TIME_STOP_{plan.maximum_minutes}M"
    active_stop = stop_bps
    for current in range(entry_index, last + 1):
        row = minutes.iloc[current]
        if not bool(row.get("data_valid", False)):
            return None
        favorable_price = float(row["perp_high"] if side > 0 else row["perp_low"])
        adverse_price = float(row["perp_low"] if side > 0 else row["perp_high"])
        favorable = _directional_return(side, entry, favorable_price)
        adverse = _directional_return(side, entry, adverse_price)
        peak = max(peak, favorable)
        mfe = max(mfe, favorable)
        mae = min(mae, adverse)
        raw_funding = row.get("funding_event_rate", 0.0)
        funding_event = float(raw_funding) if pd.notna(raw_funding) else 0.0
        funding -= side * funding_event * 10_000 * remaining

        if adverse <= active_stop:
            open_return = _directional_return(side, entry, float(row["perp_open"]))
            fill = min(open_return, active_stop)
            realized += remaining * fill
            remaining = 0.0
            exit_index = current
            exit_reason = "DYNAMIC_STOP"
            break
        if first_target_index is None and favorable >= first_target:
            realized += plan.first_exit_fraction * first_target
            remaining -= plan.first_exit_fraction
            first_target_index = current
            funding_at_first_target = funding
            if remaining <= 0:
                remaining = 0.0
                exit_index = current
                exit_reason = "FIRST_TARGET_FULL_EXIT"
                break
            active_stop = max(
                active_stop,
                _protected_stop_bps(plan, first_target, protection_cost_bps),
            )
        if first_target_index is not None and favorable >= second_target:
            realized += remaining * second_target
            remaining = 0.0
            exit_index = current
            exit_reason = "SECOND_TARGET"
            break
        if first_target_index is not None:
            active_stop = max(active_stop, peak - plan.trailing_giveback_bps)

    if remaining:
        close_return = _directional_return(
            side, entry, float(minutes.iloc[exit_index]["perp_close"])
        )
        realized += remaining * close_return
    gross = realized + funding
    close_at_first_gross = (
        first_target + funding_at_first_target
        if first_target_index is not None
        else gross
    )
    return {
        "entry_timestamp": pd.Timestamp(times.iat[entry_index]),
        "exit_timestamp": pd.Timestamp(times.iat[exit_index]),
        "entry_price": entry,
        "initial_stop_bps": risk_bps,
        "first_target_bps": first_target,
        "second_target_bps": second_target,
        "first_target_hit": first_target_index is not None,
        "first_target_timestamp": (
            pd.Timestamp(times.iat[first_target_index])
            if first_target_index is not None
            else pd.NaT
        ),
        "first_target_state_available_at": (
            pd.Timestamp(times.iat[first_target_index]) + pd.Timedelta(minutes=1)
            if first_target_index is not None
            else pd.NaT
        ),
        "time_to_first_target_minutes": (
            first_target_index - entry_index
            if first_target_index is not None
            else np.nan
        ),
        "gross_market_return_bps": realized,
        "funding_return_bps": funding,
        "gross_return_bps": gross,
        "close_at_first_gross_bps": close_at_first_gross,
        "continuation_increment_bps": (
            gross - close_at_first_gross
            if first_target_index is not None
            else np.nan
        ),
        "mfe_bps": mfe,
        "mae_bps": mae,
        "duration_minutes": exit_index - entry_index + 1,
        "exit_reason": exit_reason,
        "plan": plan.name,
        "plan_first_target_bps": plan.first_target_bps,
        "plan_second_target_bps": plan.second_target_bps,
        "plan_first_exit_fraction": plan.first_exit_fraction,
        "plan_trailing_giveback_bps": plan.trailing_giveback_bps,
        "plan_maximum_minutes": plan.maximum_minutes,
        "planned_gross_bps": planned_gross,
    }


def _matrix_worker(horizon: int) -> pd.DataFrame:
    candidates = pd.read_parquet(CANDIDATES)
    candidates = candidates.loc[candidates["expert_breakout_bars"].eq(horizon)]
    minutes = pd.read_parquet(MINUTES).sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(minutes["timestamp"], utc=True)
    output: list[dict[str, Any]] = []
    for event in candidates.to_dict("records"):
        for plan in MANAGEMENT_PLANS:
            if plan.name == BASE_PLAN.name:
                continue
            outcome = _simulate_plan(event, minutes, times, plan)
            if outcome is not None:
                output.append(event | outcome | {"fine_tuning_protocol_hash": PROTOCOL_HASH})
    return pd.DataFrame(output)


def _base_worker(horizon: int) -> pd.DataFrame:
    events = build_events(pd.read_parquet(BARS), breakout_bars=horizon)
    rows = label_events(events, pd.read_parquet(MINUTES))
    if rows.empty:
        return rows
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    rows = rows.loc[
        rows["event_family"].eq("IMPULSE_PULLBACK") & time.lt(HOLDOUT_START)
    ].copy()
    risk_bps = (
        rows["direction"]
        * (rows["entry_price"] - rows["stop_price"])
        / rows["entry_price"]
        * 10_000
    )
    rows["expert_breakout_bars"] = horizon
    rows["initial_stop_bps"] = risk_bps
    rows["first_target_bps"] = 1.5 * risk_bps
    rows["second_target_bps"] = 0.0
    rows["first_target_hit"] = rows["tp1"].astype(bool)
    rows["time_to_first_target_minutes"] = np.nan
    rows["gross_market_return_bps"] = rows["gross_return_bps"]
    rows["funding_return_bps"] = 0.0
    rows["duration_minutes"] = (
        pd.to_datetime(rows["exit_timestamp"], utc=True)
        - pd.to_datetime(rows["entry_timestamp"], utc=True)
    ).dt.total_seconds() / 60
    rows["plan"] = BASE_PLAN.name
    rows["plan_first_target_bps"] = rows["first_target_bps"]
    rows["plan_second_target_bps"] = 0.0
    rows["plan_first_exit_fraction"] = 0.5
    rows["plan_trailing_giveback_bps"] = 0.0
    rows["plan_maximum_minutes"] = 360
    rows["planned_gross_bps"] = rows["room_bps"].clip(lower=0)
    rows["fine_tuning_protocol_hash"] = PROTOCOL_HASH
    rows["candidate_source"] = "BASE_V8_FROZEN"
    return rows


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        existing = pd.read_parquet(MATRIX)
        if (
            "fine_tuning_protocol_hash" in existing
            and existing["fine_tuning_protocol_hash"].eq(PROTOCOL_HASH).all()
        ):
            return existing
    _status("target_labels", 5, "Causal minute paths for frozen V8 candidates")
    with ProcessPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_matrix_worker, HORIZONS))
    with ProcessPoolExecutor(max_workers=4) as pool:
        base_parts = list(pool.map(_base_worker, HORIZONS))
    matrix = pd.concat([*parts, *base_parts], ignore_index=True)
    matrix = _attach_context(matrix)
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    os.replace(temporary, MATRIX)
    return matrix


def _deduplicate_broad_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.sort_values(
            ["signal_timestamp", "expert_breakout_bars"], ascending=[True, False]
        )
        .drop_duplicates(["signal_timestamp", "direction"], keep="first")
        .sort_values("signal_timestamp")
        .reset_index(drop=True)
    )


def _cost_target_plan(cost_bps: float, multiplier: float) -> ManagementPlan:
    target = cost_bps * multiplier
    return ManagementPlan(
        f"FULL_TP1_{multiplier:g}X_COST",
        target,
        target,
        1.0,
        0.0,
        COST_TARGET_MAXIMUM_MINUTES,
        1.0,
        2.0,
    )


def _cost_runner_plan(cost_bps: float, multiplier: float) -> ManagementPlan:
    first_target = cost_bps * multiplier
    second_target = 2.0 * first_target
    protected_stop = (
        cost_bps
        + RUNNER_PROTECTED_PROFIT_BPS
        - RUNNER_FIRST_EXIT_FRACTION * first_target
    ) / (1.0 - RUNNER_FIRST_EXIT_FRACTION)
    return ManagementPlan(
        f"RUNNER_TP1_{multiplier:g}X_TP2_{2 * multiplier:g}X_COST",
        first_target,
        second_target,
        RUNNER_FIRST_EXIT_FRACTION,
        first_target - max(0.0, protected_stop),
        COST_TARGET_MAXIMUM_MINUTES,
        0.0,
        0.0,
    )


def _policy_profile_worker(level: int) -> pd.DataFrame:
    profile = f"VIP{level}"
    cost = profile_cost_bps(profile)
    candidates = _deduplicate_broad_candidates(pd.read_parquet(CANDIDATES))
    candidates = candidates.loc[
        pd.to_datetime(candidates["available_at"], utc=True).lt(HOLDOUT_START)
    ]
    minutes = pd.read_parquet(MINUTES).sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(minutes["timestamp"], utc=True)
    output: list[dict[str, Any]] = []
    for multiplier in COST_TARGET_MULTIPLIERS:
        plan = _cost_runner_plan(cost, multiplier)
        for event in candidates.to_dict("records"):
            outcome = _simulate_plan(
                event,
                minutes,
                times,
                plan,
                protection_cost_bps=cost,
            )
            if outcome is None:
                continue
            output.append(
                event
                | outcome
                | {
                    "fee_profile": profile,
                    "round_trip_cost_bps": cost,
                    "target_cost_multiple": multiplier,
                    "policy_action": "PARTIAL_RUNNER",
                    "net_return_bps": outcome["gross_return_bps"] - cost,
                    "stress_return_bps": outcome["gross_return_bps"] - 2 * cost,
                    "policy_protocol_hash": POLICY_PROTOCOL_HASH,
                }
            )
    return pd.DataFrame(output)


def build_cost_linked_policy_matrix(*, force: bool = False) -> pd.DataFrame:
    if POLICY_MATRIX.exists() and not force:
        cached = pd.read_parquet(POLICY_MATRIX)
        if (
            "policy_protocol_hash" in cached
            and cached["policy_protocol_hash"].eq(POLICY_PROTOCOL_HASH).all()
        ):
            return cached
    _atomic_json(
        POLICY_STATUS,
        {
            "phase": "cost_linked_labels",
            "percent": 10,
            "detail": "6 fee profiles x 2 preregistered partial-runner policies",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(_policy_profile_worker, range(6)))
    matrix = pd.concat(parts, ignore_index=True)
    POLICY_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = POLICY_MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    os.replace(temporary, POLICY_MATRIX)
    return matrix


def _entry_micro_state(minutes: pd.DataFrame) -> pd.DataFrame:
    data = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True)
    close = data["perp_close"]
    quote = data["perp_quote_volume"].replace(0, np.nan)
    signed_quote = 2 * data["perp_taker_buy_quote"] - data["perp_quote_volume"]
    output = pd.DataFrame(
        {"available_at": pd.to_datetime(data["timestamp"], utc=True) + pd.Timedelta(minutes=1)}
    )
    for minutes_count in (1, 5, 15):
        output[f"entry_return_{minutes_count}m_bps"] = (
            close.pct_change(minutes_count) * 10_000
        )
        output[f"entry_taker_imbalance_{minutes_count}m"] = (
            signed_quote.rolling(minutes_count, min_periods=minutes_count).sum()
            / quote.rolling(minutes_count, min_periods=minutes_count).sum()
        )
    prior_trade_mean = data["perp_trade_count"].shift(1).rolling(60, min_periods=20).mean()
    prior_trade_std = data["perp_trade_count"].shift(1).rolling(60, min_periods=20).std()
    aggressive = signed_quote.abs()
    prior_aggressive_mean = aggressive.shift(1).rolling(60, min_periods=20).mean()
    prior_aggressive_std = aggressive.shift(1).rolling(60, min_periods=20).std()
    output["entry_trade_count_z_1h"] = (
        data["perp_trade_count"] - prior_trade_mean
    ) / prior_trade_std.replace(0, np.nan)
    output["entry_aggressive_volume_z_1h"] = (
        aggressive - prior_aggressive_mean
    ) / prior_aggressive_std.replace(0, np.nan)
    output["entry_mark_divergence_bps"] = (
        data["mark_close"] / close - 1.0
    ) * 10_000
    output["entry_funding_rate"] = data["funding_rate"]
    output["entry_range_1m_bps"] = (
        (data["perp_high"] - data["perp_low"]) / close * 10_000
    )
    output["entry_return_24h_bps"] = close.pct_change(24 * 60) * 10_000
    mark_spot_basis = (data["mark_close"] / data["spot_close"] - 1.0) * 10_000
    output["entry_mark_spot_basis_bps"] = mark_spot_basis
    output["entry_basis_change_1h_bps"] = mark_spot_basis.diff(60)
    return output


def _causal_context_coverage(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["binance_context_coverage"].fillna(False).astype(bool)
        & rows["binance_context_available_at"].le(rows["available_at"])
        & rows["max_input_available_at"].le(rows["available_at"])
        & rows.loc[:, FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )


def _attach_context(matrix: pd.DataFrame) -> pd.DataFrame:
    output = matrix.copy()
    output["available_at"] = pd.to_datetime(output["available_at"], utc=True)
    state = pd.read_parquet(MARKET_STATE_PATH)
    state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
    output = pd.merge_asof(
        output.sort_values("available_at"),
        state.loc[:, ["available_at", *MARKET_STATE_FEATURES]].sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    micro = _entry_micro_state(pd.read_parquet(MINUTES))
    output = pd.merge_asof(
        output.sort_values("available_at"),
        micro.sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    market = pd.read_parquet(BINANCE_MARKET)[
        ["available_at", "max_input_available_at", "funding_z"]
    ].copy()
    market["available_at"] = pd.to_datetime(market["available_at"], utc=True)
    market["max_input_available_at"] = pd.to_datetime(
        market["max_input_available_at"], utc=True
    )
    market = market.rename(
        columns={
            "available_at": "binance_market_available_at",
            "funding_z": "entry_funding_z",
        }
    )
    output = pd.merge_asof(
        output.sort_values("available_at"),
        market.sort_values("binance_market_available_at"),
        left_on="available_at",
        right_on="binance_market_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    context = pd.read_parquet(BINANCE_CONTEXT)[
        ["context_available_at", "context_coverage", *BINANCE_CONTEXT_FEATURES[:14]]
    ].copy()
    context["context_available_at"] = pd.to_datetime(
        context["context_available_at"], utc=True
    )
    context = context.rename(
        columns={
            "context_available_at": "binance_context_available_at",
            "context_coverage": "binance_context_coverage",
        }
    )
    output = pd.merge_asof(
        output.sort_values("available_at"),
        context.sort_values("binance_context_available_at"),
        left_on="available_at",
        right_on="binance_context_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    for minutes_count in (1, 5, 15):
        output[f"signed_entry_return_{minutes_count}m_bps"] = (
            output["direction"] * output[f"entry_return_{minutes_count}m_bps"]
        )
        output[f"signed_entry_taker_imbalance_{minutes_count}m"] = (
            output["direction"] * output[f"entry_taker_imbalance_{minutes_count}m"]
        )
    output["signed_binance_taker_imbalance_15m"] = (
        output["direction"] * output["binance_taker_imbalance_15m"]
    )
    output["signed_binance_taker_imbalance_1h"] = (
        output["direction"] * output["binance_taker_imbalance_1h"]
    )
    output["signed_book_imbalance_1pct"] = (
        output["direction"] * output["book_imbalance_1pct"]
    )
    output["signed_top_position_deviation"] = output["direction"] * (
        output["top_position_ratio"] - 1.0
    )
    output["signed_global_position_deviation"] = output["direction"] * (
        output["global_long_short_ratio"] - 1.0
    )
    output["signed_taker_ratio_deviation"] = output["direction"] * (
        output["taker_long_short_ratio"] - 1.0
    )
    output["directional_price_oi_interaction_1h"] = (
        output["direction"] * output["return_1h"] * output["oi_change_1h"]
    )
    output["directional_price_oi_interaction_4h"] = (
        output["direction"] * output["return_4h"] * output["oi_change_4h"]
    )
    output["directional_price_oi_interaction_24h"] = (
        output["direction"] * output["entry_return_24h_bps"] / 10_000
        * output["oi_change_24h"]
    )
    output["target_to_atr_1m"] = output["first_target_bps"] / output[
        "atr_1m_bps"
    ].replace(0, np.nan)
    output["target_to_atr_5m"] = output["first_target_bps"] / output[
        "atr_5m_bps"
    ].replace(0, np.nan)
    output["target_to_realized_volatility_30m"] = output[
        "first_target_bps"
    ] / output["realized_volatility_30m_bps"].replace(0, np.nan)
    output["stop_to_atr_5m"] = output["initial_stop_bps"] / output[
        "atr_5m_bps"
    ].replace(0, np.nan)
    output["flow_volume_confirmation"] = (
        output["signed_entry_taker_imbalance_15m"] * output["volume_percentile"]
    )
    output["orderflow_depth_confirmation"] = (
        output["signed_binance_taker_imbalance_15m"]
        * output["signed_book_imbalance_1pct"]
    )
    output["model_feature_coverage_valid"] = _causal_context_coverage(output)
    return output.sort_values("signal_timestamp").reset_index(drop=True)


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, FEATURES].to_numpy(float)
    distance_columns = tuple(
        FEATURES.index(name)
        for name in (
            "daily_distance_sigma",
            "impulse_distance_sigma",
            "swing_distance_sigma",
        )
    )
    values[:, distance_columns] = np.clip(values[:, distance_columns], -20, 20)
    return values


def _weights(rows: pd.DataFrame) -> np.ndarray:
    actions = rows.groupby("signal_timestamp")["signal_timestamp"].transform("size")
    return 1.0 / actions.to_numpy(float)


def _models(kind: str) -> tuple[Any, Any]:
    if kind == "ridge":
        return (
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=10.0),
            ),
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=2000, random_state=20260809),
            ),
        )
    common = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=20260809,
    )
    return (
        XGBRegressor(objective="reg:squarederror", **common),
        XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common),
    )


def _fit(kind: str, fit: pd.DataFrame) -> dict[str, Any]:
    if not fit["model_feature_coverage_valid"].all():
        raise ValueError("model fitting received rows without complete causal coverage")
    regressor, classifier = _models(kind)
    gain_regressor, _ = _models(kind)
    miss_regressor, _ = _models(kind)
    weights = _weights(fit)
    hits = fit["first_target_hit"].astype(bool)
    if kind == "ridge":
        regressor.fit(_x(fit), fit["gross_return_bps"], ridge__sample_weight=weights)
        gain_regressor.fit(
            _x(fit.loc[hits]),
            fit.loc[hits, "gross_return_bps"],
            ridge__sample_weight=weights[hits.to_numpy()],
        )
        miss_regressor.fit(
            _x(fit.loc[~hits]),
            fit.loc[~hits, "gross_return_bps"],
            ridge__sample_weight=weights[(~hits).to_numpy()],
        )
        classifier.fit(
            _x(fit),
            fit["first_target_hit"].astype(int),
            logisticregression__sample_weight=weights,
        )
    else:
        regressor.fit(_x(fit), fit["gross_return_bps"], sample_weight=weights)
        gain_regressor.fit(
            _x(fit.loc[hits]),
            fit.loc[hits, "gross_return_bps"],
            sample_weight=weights[hits.to_numpy()],
        )
        miss_regressor.fit(
            _x(fit.loc[~hits]),
            fit.loc[~hits, "gross_return_bps"],
            sample_weight=weights[(~hits).to_numpy()],
        )
        classifier.fit(_x(fit), fit["first_target_hit"].astype(int), sample_weight=weights)
    return {
        "regressor": regressor,
        "gain_regressor": gain_regressor,
        "miss_regressor": miss_regressor,
        "classifier": classifier,
    }


def _raw_probability(model: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model["classifier"].predict_proba(_x(rows))[:, 1], dtype=float)


def _conditional_ev(
    probability: np.ndarray, gain: np.ndarray, miss: np.ndarray
) -> np.ndarray:
    return probability * gain + (1.0 - probability) * miss


def _score(
    model: dict[str, Any], calibrator: IsotonicRegression, rows: pd.DataFrame
) -> pd.DataFrame:
    output = rows.copy()
    output["predicted_direct_gross_bps"] = model["regressor"].predict(_x(rows))
    output["predicted_target_probability"] = calibrator.predict(
        _raw_probability(model, rows)
    )
    output["predicted_conditional_gain_bps"] = np.maximum(
        0.0, model["gain_regressor"].predict(_x(rows))
    )
    output["predicted_conditional_miss_bps"] = model["miss_regressor"].predict(
        _x(rows)
    )
    output["predicted_conditional_loss_bps"] = np.maximum(
        0.0, -output["predicted_conditional_miss_bps"]
    )
    probability = output["predicted_target_probability"]
    output["predicted_gross_bps"] = _conditional_ev(
        probability.to_numpy(float),
        output["predicted_conditional_gain_bps"].to_numpy(float),
        output["predicted_conditional_miss_bps"].to_numpy(float),
    )
    return output


def _fit_chronologically_calibrated(
    kind: str, fit: pd.DataFrame, calibration: pd.DataFrame
) -> dict[str, Any]:
    model = _fit(kind, fit)
    probability_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        _raw_probability(model, calibration),
        calibration["first_target_hit"].astype(int),
    )
    calibration_scores = _score(model, probability_calibrator, calibration)
    ev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_scores["predicted_gross_bps"],
        calibration["gross_return_bps"],
    )
    return {
        "model": model,
        "probability_calibrator": probability_calibrator,
        "ev_calibrator": ev_calibrator,
    }


def _score_chronologically_calibrated(
    fitted: dict[str, Any], rows: pd.DataFrame
) -> pd.DataFrame:
    output = _score(
        fitted["model"], fitted["probability_calibrator"], rows
    )
    output["predicted_uncalibrated_gross_bps"] = output["predicted_gross_bps"]
    output["predicted_gross_bps"] = fitted["ev_calibrator"].predict(
        output["predicted_uncalibrated_gross_bps"]
    )
    return output


def _one_position(rows: pd.DataFrame) -> pd.DataFrame:
    accepted: list[Any] = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in rows.sort_values("entry_timestamp").iterrows():
        if pd.Timestamp(row["entry_timestamp"]) >= busy_until:
            accepted.append(index)
            busy_until = pd.Timestamp(row["exit_timestamp"])
    return rows.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def _best_actions(rows: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    candidates = rows.copy()
    candidates["predicted_net_ev_bps"] = candidates["predicted_gross_bps"] - cost_bps
    return (
        candidates.sort_values("predicted_net_ev_bps", ascending=False)
        .drop_duplicates("signal_timestamp")
        .reset_index(drop=True)
    )


def _metrics(rows: pd.DataFrame, cost_bps: float) -> dict[str, float]:
    values = rows["gross_return_bps"].to_numpy(float) - cost_bps
    if not len(values):
        return {
            "trades": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "positive_trade_days": 0.0,
        }
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    returns_r = values / rows["initial_stop_bps"].to_numpy(float)
    curve = np.cumsum(returns_r * 0.01)
    peak = np.maximum.accumulate(np.r_[0.0, curve])
    drawdown = float(np.max(peak - np.r_[0.0, curve]))
    daily = pd.Series(
        returns_r,
        index=pd.to_datetime(rows["exit_timestamp"], utc=True).dt.floor("D"),
    ).groupby(level=0).sum()
    return {
        "trades": float(len(values)),
        "expectancy_bps": float(values.mean()),
        "profit_factor": gains / losses if losses else float("inf"),
        "win_rate": float((values > 0).mean()),
        "max_drawdown": drawdown,
        "positive_trade_days": float((daily > 0).mean()),
    }


def _frequency_frontier(rows: pd.DataFrame, cost_bps: float) -> dict[str, Any] | None:
    best = _best_actions(rows, cost_bps)
    thresholds = np.sort(
        best.loc[best["predicted_net_ev_bps"].gt(0), "predicted_net_ev_bps"].unique()
    )
    champion: dict[str, Any] | None = None
    for threshold in thresholds:
        selected = _one_position(best.loc[best["predicted_net_ev_bps"].ge(threshold)])
        metrics = _metrics(selected, cost_bps)
        if (
            metrics["trades"] >= MINIMUM_SELECTION_TRADES
            and metrics["expectancy_bps"] > 0
            and metrics["profit_factor"] >= 1.15
            and metrics["max_drawdown"] <= 0.08
            and (champion is None or metrics["trades"] > champion["metrics"]["trades"])
        ):
            champion = {
                "threshold_bps": float(threshold),
                "metrics": metrics,
                "selected_indices": selected.index.tolist(),
            }
    return champion


def _frequency_diagnostic(rows: pd.DataFrame, cost_bps: float) -> dict[str, Any]:
    best = _best_actions(rows, cost_bps)
    positive = best.loc[best["predicted_net_ev_bps"].gt(0)]
    available = _one_position(positive)
    thresholds = np.sort(positive["predicted_net_ev_bps"].unique())
    economic: dict[str, Any] | None = None
    best_at_least_50: dict[str, Any] | None = None
    for threshold in thresholds:
        selected = _one_position(
            best.loc[best["predicted_net_ev_bps"].ge(threshold)]
        )
        metrics = _metrics(selected, cost_bps)
        candidate = {"threshold_bps": float(threshold), "metrics": metrics}
        if (
            metrics["expectancy_bps"] > 0
            and metrics["profit_factor"] >= 1.15
            and metrics["max_drawdown"] <= 0.08
            and (economic is None or metrics["trades"] > economic["metrics"]["trades"])
        ):
            economic = candidate
        if (
            metrics["trades"] >= MINIMUM_SELECTION_TRADES
            and (
                best_at_least_50 is None
                or metrics["expectancy_bps"]
                > best_at_least_50["metrics"]["expectancy_bps"]
            )
        ):
            best_at_least_50 = candidate
    return {
        "unique_signals": len(best),
        "predicted_positive_signals": len(positive),
        "predicted_positive_nonoverlap": len(available),
        "all_predicted_positive_metrics": _metrics(available, cost_bps),
        "highest_frequency_economic_without_count_gate": economic,
        "best_expectancy_with_at_least_50": best_at_least_50,
    }


def _rank_policy_actions(rows: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["predicted_net_ev_bps"] = ranked["predicted_gross_bps"] - cost_bps
    rank_source = (
        ranked["predicted_uncalibrated_gross_bps"]
        if "predicted_uncalibrated_gross_bps" in ranked
        else ranked["predicted_gross_bps"]
    )
    ranked["policy_rank_score_bps"] = rank_source - cost_bps
    return (
        ranked.sort_values("policy_rank_score_bps", ascending=False)
        .drop_duplicates(["signal_timestamp", "direction"], keep="first")
        .reset_index(drop=True)
    )


def _policy_coverage_evaluations(
    rows: pd.DataFrame, cost_bps: float
) -> list[dict[str, Any]]:
    ranked = _rank_policy_actions(rows, cost_bps)
    if ranked.empty:
        return []
    evaluations: list[dict[str, Any]] = []
    for coverage in POLICY_COVERAGE_LEVELS:
        count = max(1, int(np.ceil(len(ranked) * coverage)))
        threshold = float(ranked.iloc[count - 1]["policy_rank_score_bps"])
        selected_before_lock = ranked.iloc[:count]
        selected = _one_position(selected_before_lock)
        metrics = _metrics(selected, cost_bps)
        stress = _metrics(selected, 2 * cost_bps)
        lcb = _bootstrap_lcb(selected, cost_bps)
        gates = {
            "trades_50": metrics["trades"] >= 50,
            "expectancy_positive": metrics["expectancy_bps"] > 0,
            "pf_1_15": metrics["profit_factor"] >= 1.15,
            "drawdown_8pct": metrics["max_drawdown"] <= 0.08,
            "majority_positive_active_days": metrics["positive_trade_days"] > 0.5,
            "expectancy_lcb_positive": bool(np.isfinite(lcb) and lcb > 0),
        }
        evaluations.append(
            {
                "coverage": coverage,
                "threshold_bps": threshold,
                "selected_before_position_lock": len(selected_before_lock),
                "metrics": metrics,
                "stress_2x": stress,
                "stress_2x_nonnegative_diagnostic": stress["expectancy_bps"] >= 0,
                "expectancy_lcb_95_bps": lcb,
                "gates": gates,
                "operational_pass": all(gates.values()),
            }
        )
    return evaluations


def _policy_coverage_frontier(
    rows: pd.DataFrame, cost_bps: float
) -> dict[str, Any] | None:
    passing = [
        candidate
        for candidate in _policy_coverage_evaluations(rows, cost_bps)
        if candidate["operational_pass"]
    ]
    if not passing:
        return None
    champion = max(passing, key=lambda candidate: candidate["metrics"]["trades"])
    return champion | {"candidate_policies_evaluated": len(POLICY_COVERAGE_LEVELS)}


def _apply_policy_threshold(
    rows: pd.DataFrame, cost_bps: float, threshold_bps: float
) -> pd.DataFrame:
    ranked = _rank_policy_actions(rows, cost_bps)
    return _one_position(
        ranked.loc[ranked["policy_rank_score_bps"].ge(threshold_bps)]
    )


def _apply_threshold(rows: pd.DataFrame, cost_bps: float, threshold: float) -> pd.DataFrame:
    best = _best_actions(rows, cost_bps)
    return _one_position(best.loc[best["predicted_net_ev_bps"].ge(threshold)])


def _bootstrap_lcb(rows: pd.DataFrame, cost_bps: float, seed: int = 20260809) -> float:
    if len(rows) < 20:
        return float("nan")
    values = rows["gross_return_bps"].to_numpy(float) - cost_bps
    rng = np.random.default_rng(seed)
    block = min(10, max(2, int(np.sqrt(len(values)))))
    means = np.empty(1000)
    for sample in range(len(means)):
        output: list[float] = []
        while len(output) < len(values):
            start = int(rng.integers(0, max(1, len(values) - block + 1)))
            output.extend(values[start : start + block])
        means[sample] = np.mean(output[: len(values)])
    return float(np.quantile(means, 0.05))


def _policy_period_summary(
    rows: pd.DataFrame, cost_bps: float, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    selected = rows.loc[
        pd.to_datetime(rows["entry_timestamp"], utc=True).between(
            start, end, inclusive="left"
        )
    ].copy()
    metrics = _metrics(selected, cost_bps)
    stress = _metrics(selected, 2 * cost_bps)
    values = selected["gross_return_bps"].to_numpy(float) - cost_bps
    days = pd.to_datetime(selected["exit_timestamp"], utc=True).dt.floor("D")
    daily = pd.DataFrame({"day": days, "net_bps": values}).groupby("day").agg(
        net_bps=("net_bps", "sum"), trades=("net_bps", "size")
    )
    calendar_days = max(1, int((end - start).total_seconds() // 86_400))
    return {
        "metrics": metrics,
        "stress_2x": stress,
        "expectancy_lcb_95_bps": _bootstrap_lcb(selected, cost_bps),
        "calendar_days": calendar_days,
        "trades_per_calendar_day": len(selected) / calendar_days,
        "active_trade_days": len(daily),
        "trades_per_active_day": float(daily["trades"].mean()) if len(daily) else 0.0,
        "maximum_trades_in_one_day": int(daily["trades"].max()) if len(daily) else 0,
        "positive_active_days": float((daily["net_bps"] > 0).mean()) if len(daily) else 0.0,
        "target_hit_rate": float(selected["first_target_hit"].mean())
        if len(selected)
        else 0.0,
        "median_duration_minutes": float(selected["duration_minutes"].median())
        if len(selected)
        else 0.0,
        "exit_reasons": selected["exit_reason"].value_counts().to_dict(),
    }


def _prior_policy_gate(
    year_2024: dict[str, Any],
    year_2025: dict[str, Any],
    combined: dict[str, Any],
) -> dict[str, bool]:
    return {
        "at_least_100_prior_trades": combined["metrics"]["trades"] >= 100,
        "positive_both_years": year_2024["metrics"]["expectancy_bps"] > 0
        and year_2025["metrics"]["expectancy_bps"] > 0,
        "pf_1_05_both_years": year_2024["metrics"]["profit_factor"] >= 1.05
        and year_2025["metrics"]["profit_factor"] >= 1.05,
        "combined_pf_1_15": combined["metrics"]["profit_factor"] >= 1.15,
        "combined_drawdown_8pct": combined["metrics"]["max_drawdown"] <= 0.08,
    }


def _audit_policy_gate(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "trades_50": summary["metrics"]["trades"] >= 50,
        "expectancy_positive": summary["metrics"]["expectancy_bps"] > 0,
        "pf_1_15": summary["metrics"]["profit_factor"] >= 1.15,
        "drawdown_8pct": summary["metrics"]["max_drawdown"] <= 0.08,
        "majority_positive_active_days": summary["positive_active_days"] > 0.5,
        "expectancy_lcb_positive": bool(
            np.isfinite(summary["expectancy_lcb_95_bps"])
            and summary["expectancy_lcb_95_bps"] > 0
        ),
    }


def run_cost_linked_policy_audit(*, force_matrix: bool = False) -> dict[str, Any]:
    matrix = build_cost_linked_policy_matrix(force=force_matrix)
    profiles: dict[str, Any] = {}
    eligible_profiles: list[str] = []
    boundaries = {
        "2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
        "2025": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        "prior": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        "audit": (pd.Timestamp("2026-01-01", tz="UTC"), HOLDOUT_START),
    }
    for level in range(6):
        profile = f"VIP{level}"
        cost = profile_cost_bps(profile)
        policies: dict[str, Any] = {}
        prior_eligible: list[tuple[float, float]] = []
        for multiplier in COST_TARGET_MULTIPLIERS:
            rows = matrix.loc[
                matrix["fee_profile"].eq(profile)
                & matrix["target_cost_multiple"].eq(multiplier)
            ]
            rows = _one_position(rows)
            summaries = {
                name: _policy_period_summary(rows, cost, *period)
                for name, period in boundaries.items()
            }
            prior_gate = _prior_policy_gate(
                summaries["2024"], summaries["2025"], summaries["prior"]
            )
            audit_gate = _audit_policy_gate(summaries["audit"])
            prior_pass = all(prior_gate.values())
            audit_pass = all(audit_gate.values())
            policies[f"{multiplier:g}x"] = {
                "target_gross_bps": cost * multiplier,
                "target_gross_percent": cost * multiplier / 100,
                "periods": summaries,
                "prior_gates": prior_gate,
                "prior_pass": prior_pass,
                "audit_gates": audit_gate,
                "audit_pass": audit_pass,
                "stress_2x_diagnostic": {
                    name: summary["stress_2x"]["expectancy_bps"] >= 0
                    for name, summary in summaries.items()
                },
            }
            if prior_pass:
                prior_eligible.append(
                    (multiplier, summaries["prior"]["metrics"]["trades"])
                )
        selected_multiplier = (
            max(prior_eligible, key=lambda item: item[1])[0] if prior_eligible else None
        )
        profile_eligible = bool(
            selected_multiplier is not None
            and policies[f"{selected_multiplier:g}x"]["audit_pass"]
        )
        if profile_eligible:
            eligible_profiles.append(profile)
        profiles[profile] = {
            "round_trip_cost_bps": cost,
            "break_even_percent": cost / 100,
            "policies": policies,
            "selected_on_prior": (
                f"{selected_multiplier:g}x" if selected_multiplier is not None else None
            ),
            "paper_eligible": profile_eligible,
        }
    report = {
        "protocol": POLICY_PROTOCOL,
        "protocol_hash": POLICY_PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "unique_candidates": int(
            matrix[["signal_timestamp", "direction"]].drop_duplicates().shape[0]
        ),
        "profiles": profiles,
        "eligible_profiles": eligible_profiles,
        "verdict": (
            "COST_LINKED_POLICY_READY"
            if eligible_profiles
            else "BASE_ONLY_NO_COST_LINKED_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(POLICY_REPORT, report)
    _atomic_json(
        POLICY_STATUS,
        {
            "phase": "complete",
            "percent": 100,
            "detail": report["verdict"],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return report


def _purged_partition(
    matrix: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    entry = pd.to_datetime(matrix["entry_timestamp"], utc=True)
    exit_time = pd.to_datetime(matrix["exit_timestamp"], utc=True)
    mask = exit_time.lt(end) & matrix["model_feature_coverage_valid"].fillna(False)
    if start is None:
        mask &= entry.lt(end)
    else:
        mask &= entry.ge(start) & entry.lt(end)
    return matrix.loc[mask].copy()


def _walk_forward_profile_gate(
    discovery: dict[str, Any], audit: dict[str, Any]
) -> dict[str, bool]:
    return {
        "discovery_trades_100": discovery["metrics"]["trades"] >= 100,
        "discovery_expectancy_positive": discovery["metrics"]["expectancy_bps"] > 0,
        "discovery_pf_1_15": discovery["metrics"]["profit_factor"] >= 1.15,
        "discovery_drawdown_8pct": discovery["metrics"]["max_drawdown"] <= 0.08,
        "discovery_lcb_positive": bool(
            np.isfinite(discovery["expectancy_lcb_95_bps"])
            and discovery["expectancy_lcb_95_bps"] > 0
        ),
        "audit_trades_50": audit["metrics"]["trades"] >= 50,
        "audit_expectancy_positive": audit["metrics"]["expectancy_bps"] > 0,
        "audit_pf_1_15": audit["metrics"]["profit_factor"] >= 1.15,
        "audit_drawdown_8pct": audit["metrics"]["max_drawdown"] <= 0.08,
        "audit_majority_positive_days": audit["positive_active_days"] > 0.5,
        "audit_lcb_positive": bool(
            np.isfinite(audit["expectancy_lcb_95_bps"])
            and audit["expectancy_lcb_95_bps"] > 0
        ),
    }


def run_policy_walk_forward(*, force_outcomes: bool = False) -> dict[str, Any]:
    matrix = _attach_context(
        build_cost_linked_policy_matrix(force=force_outcomes)
    )
    test_start = pd.Timestamp("2025-01-01", tz="UTC")
    fold_reports: list[dict[str, Any]] = []
    selected_rows: dict[str, list[pd.DataFrame]] = {
        f"VIP{level}": [] for level in range(6)
    }
    last_bundle: dict[str, Any] | None = None
    fold_number = 0
    while test_start < HOLDOUT_START:
        fold_number += 1
        test_end = min(test_start + WALK_FORWARD_STEP, HOLDOUT_START)
        selection_start = test_start - WALK_FORWARD_SELECTION
        calibration_start = selection_start - WALK_FORWARD_CALIBRATION
        fit = _purged_partition(matrix, None, calibration_start)
        calibration = _purged_partition(matrix, calibration_start, selection_start)
        selection = _purged_partition(matrix, selection_start, test_start)
        test = _purged_partition(matrix, test_start, test_end)
        _atomic_json(
            WALK_FORWARD_STATUS,
            {
                "phase": "walk_forward_gpu",
                "percent": round(
                    5 + 90 * (test_start - pd.Timestamp("2025-01-01", tz="UTC"))
                    / (HOLDOUT_START - pd.Timestamp("2025-01-01", tz="UTC")),
                    2,
                ),
                "detail": f"Fold {fold_number}: {test_start.date()} to {test_end.date()}",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if min(len(fit), len(calibration), len(selection), len(test)) == 0:
            fold_reports.append(
                {
                    "fold": fold_number,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "status": "SKIPPED_EMPTY_PARTITION",
                }
            )
            test_start = test_end
            continue
        kinds: dict[str, Any] = {}
        for kind in ("ridge", "xgboost_gpu"):
            fitted = _fit_chronologically_calibrated(kind, fit, calibration)
            scored_selection = _score_chronologically_calibrated(fitted, selection)
            scored_test = _score_chronologically_calibrated(fitted, test)
            evaluations = {
                f"VIP{level}": _policy_coverage_evaluations(
                    scored_selection.loc[
                        scored_selection["fee_profile"].eq(f"VIP{level}")
                    ],
                    profile_cost_bps(f"VIP{level}"),
                )
                for level in range(6)
            }
            frontiers = {
                profile: (
                    max(
                        (
                            candidate
                            for candidate in candidates
                            if candidate["operational_pass"]
                        ),
                        key=lambda candidate: candidate["metrics"]["trades"],
                        default=None,
                    )
                )
                for profile, candidates in evaluations.items()
            }
            kinds[kind] = {
                "fitted": fitted,
                "selection": scored_selection,
                "test": scored_test,
                "mae": float(
                    mean_absolute_error(
                        selection["gross_return_bps"],
                        scored_selection["predicted_gross_bps"],
                    )
                ),
                "brier": float(
                    brier_score_loss(
                        selection["first_target_hit"],
                        scored_selection["predicted_target_probability"],
                    )
                ),
                "frontiers": frontiers,
                "evaluations": evaluations,
            }
        ridge_valid = [value for value in kinds["ridge"]["frontiers"].values() if value]
        xgb_valid = [
            value for value in kinds["xgboost_gpu"]["frontiers"].values() if value
        ]
        ridge_trades = sum(value["metrics"]["trades"] for value in ridge_valid)
        xgb_trades = sum(value["metrics"]["trades"] for value in xgb_valid)
        champion = (
            "xgboost_gpu"
            if kinds["xgboost_gpu"]["mae"] < kinds["ridge"]["mae"]
            and kinds["xgboost_gpu"]["brier"] < kinds["ridge"]["brier"]
            and len(xgb_valid) >= len(ridge_valid)
            and xgb_trades > ridge_trades
            else "ridge"
        )
        chosen = kinds[champion]
        profile_reports: dict[str, Any] = {}
        for level in range(6):
            profile = f"VIP{level}"
            cost = profile_cost_bps(profile)
            frontier = chosen["frontiers"][profile]
            if frontier is None:
                profile_reports[profile] = {
                    "selection_frontier": None,
                    "selection_diagnostics": chosen["evaluations"][profile],
                    "test_trades": 0,
                }
                continue
            test_profile = chosen["test"].loc[
                chosen["test"]["fee_profile"].eq(profile)
            ]
            accepted = _apply_policy_threshold(
                test_profile, cost, float(frontier["threshold_bps"])
            )
            accepted["walk_forward_fold"] = fold_number
            accepted["walk_forward_model"] = champion
            accepted["walk_forward_threshold_bps"] = frontier["threshold_bps"]
            selected_rows[profile].append(accepted)
            profile_reports[profile] = {
                "selection_frontier": frontier,
                "selection_diagnostics": chosen["evaluations"][profile],
                "test_trades": len(accepted),
                "test_metrics": _metrics(accepted, cost),
            }
        fold_reports.append(
            {
                "fold": fold_number,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "rows": {
                    "fit": len(fit),
                    "calibration": len(calibration),
                    "selection": len(selection),
                    "test": len(test),
                },
                "model_comparison": {
                    kind: {
                        "selection_gross_mae": value["mae"],
                        "selection_target_brier": value["brier"],
                        "valid_profile_frontiers": sum(
                            frontier is not None
                            for frontier in value["frontiers"].values()
                        ),
                    }
                    for kind, value in kinds.items()
                },
                "champion": champion,
                "profiles": profile_reports,
            }
        )
        last_bundle = {
            "fold_start": test_start,
            "fold_end": test_end,
            "champion": champion,
            "fitted": chosen["fitted"],
            "frontiers": chosen["frontiers"],
        }
        test_start = test_end

    profiles: dict[str, Any] = {}
    eligible_profiles: list[str] = []
    for level in range(6):
        profile = f"VIP{level}"
        cost = profile_cost_bps(profile)
        if selected_rows[profile]:
            rows = _one_position(pd.concat(selected_rows[profile], ignore_index=True))
        else:
            rows = matrix.iloc[0:0].copy()
        discovery = _policy_period_summary(
            rows,
            cost,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
        )
        audit = _policy_period_summary(
            rows,
            cost,
            pd.Timestamp("2026-01-01", tz="UTC"),
            HOLDOUT_START,
        )
        gates = _walk_forward_profile_gate(discovery, audit)
        eligible = all(gates.values())
        if eligible:
            eligible_profiles.append(profile)
        profiles[profile] = {
            "round_trip_cost_bps": cost,
            "walk_forward_trades": len(rows),
            "discovery_2025": discovery,
            "audit_2026_pre_holdout": audit,
            "gates": gates,
            "paper_eligible": eligible,
        }
    report = {
        "protocol": WALK_FORWARD_PROTOCOL,
        "protocol_hash": WALK_FORWARD_PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_rows": len(matrix),
        "causal_coverage": float(matrix["model_feature_coverage_valid"].mean()),
        "folds": fold_reports,
        "profiles": profiles,
        "eligible_profiles": eligible_profiles,
        "verdict": (
            "WALK_FORWARD_FREQUENCY_ALPHA_READY"
            if eligible_profiles
            else "BASE_ONLY_NO_WALK_FORWARD_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    if last_bundle is not None:
        WALK_FORWARD_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        temporary = WALK_FORWARD_BUNDLE.with_suffix(".joblib.tmp")
        joblib.dump(
            {
                "protocol": WALK_FORWARD_PROTOCOL,
                "protocol_hash": WALK_FORWARD_PROTOCOL_HASH,
                "features": FEATURES,
                "last_fold": last_bundle,
                "profiles": profiles,
                "research_only": True,
            },
            temporary,
        )
        os.replace(temporary, WALK_FORWARD_BUNDLE)
    _atomic_json(WALK_FORWARD_REPORT, report)
    _atomic_json(
        WALK_FORWARD_STATUS,
        {
            "phase": "complete",
            "percent": 100,
            "detail": report["verdict"],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return report


def train(*, force_matrix: bool = False) -> dict[str, Any]:
    matrix = build_matrix(force=force_matrix)
    time = pd.to_datetime(matrix["entry_timestamp"], utc=True)
    coverage = matrix["model_feature_coverage_valid"].fillna(False).astype(bool)
    raw_split_rows = {
        "fit": int(time.dt.year.eq(2024).sum()),
        "calibration": int(
            time.between(
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-07-01", tz="UTC"),
                inclusive="left",
            ).sum()
        ),
        "selection": int(
            time.between(
                pd.Timestamp("2025-07-01", tz="UTC"),
                pd.Timestamp("2026-01-01", tz="UTC"),
                inclusive="left",
            ).sum()
        ),
        "audit": int(
            time.between(
                pd.Timestamp("2026-01-01", tz="UTC"), HOLDOUT_START, inclusive="left"
            ).sum()
        ),
    }
    fit = matrix.loc[time.dt.year.eq(2024) & coverage].copy()
    calibration = matrix.loc[
        time.between(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-07-01", tz="UTC"),
            inclusive="left",
        )
        & coverage
    ].copy()
    selection = matrix.loc[
        time.between(
            pd.Timestamp("2025-07-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
            inclusive="left",
        )
        & coverage
    ].copy()
    audit = matrix.loc[
        time.between(pd.Timestamp("2026-01-01", tz="UTC"), HOLDOUT_START, inclusive="left")
        & coverage
    ].copy()
    _status("gpu_models", 65, f"{len(matrix):,} action-plan labels")

    candidates: dict[str, Any] = {}
    fitted: dict[str, dict[str, Any]] = {}
    selection_scores: dict[str, pd.DataFrame] = {}
    audit_scores: dict[str, pd.DataFrame] = {}
    for kind in ("ridge", "xgboost_gpu"):
        model = _fit(kind, fit)
        raw_calibration = _raw_probability(model, calibration)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            raw_calibration, calibration["first_target_hit"].astype(int)
        )
        scored_selection = _score(model, calibrator, selection)
        candidates[kind] = {
            "selection_gross_mae": float(
                mean_absolute_error(
                    selection["gross_return_bps"], scored_selection["predicted_gross_bps"]
                )
            ),
            "selection_target_brier": float(
                brier_score_loss(
                    selection["first_target_hit"],
                    scored_selection["predicted_target_probability"],
                )
            ),
        }
        fitted[kind] = {"model": model, "calibrator": calibrator}
        selection_scores[kind] = scored_selection
        audit_scores[kind] = _score(model, calibrator, audit)
    for kind, scored in selection_scores.items():
        candidates[kind]["selection_frontiers"] = {
            f"VIP{level}": _frequency_frontier(
                scored, profile_cost_bps(f"VIP{level}")
            )
            for level in range(6)
        }
        candidates[kind]["selection_diagnostics"] = {
            f"VIP{level}": _frequency_diagnostic(
                scored, profile_cost_bps(f"VIP{level}")
            )
            for level in range(6)
        }
        research_audit: dict[str, Any] = {}
        for level in range(6):
            profile = f"VIP{level}"
            cost = profile_cost_bps(profile)
            diagnostic = candidates[kind]["selection_diagnostics"][profile][
                "highest_frequency_economic_without_count_gate"
            ]
            if diagnostic is None:
                research_audit[profile] = None
                continue
            selected = _apply_threshold(
                audit_scores[kind], cost, float(diagnostic["threshold_bps"])
            )
            research_audit[profile] = {
                "selection_threshold_bps": diagnostic["threshold_bps"],
                "selection_metrics": diagnostic["metrics"],
                "audit_metrics": _metrics(selected, cost),
                "audit_stress_2x": _metrics(selected, 2 * cost),
                "audit_expectancy_lcb_95_bps": _bootstrap_lcb(selected, cost),
                "plan_counts": selected["plan"].value_counts().to_dict(),
                "horizon_counts": selected["expert_breakout_bars"]
                .value_counts()
                .sort_index()
                .to_dict(),
                "direction_counts": selected["direction"].value_counts().to_dict(),
                "research_only_reason": "selection_count_below_50",
            }
        candidates[kind]["research_audit_without_count_gate"] = research_audit
    ridge = candidates["ridge"]
    xgboost = candidates["xgboost_gpu"]
    ridge_frontiers = ridge["selection_frontiers"]
    xgboost_frontiers = xgboost["selection_frontiers"]
    ridge_valid = [frontier for frontier in ridge_frontiers.values() if frontier]
    xgboost_valid = [frontier for frontier in xgboost_frontiers.values() if frontier]
    ridge_trades = sum(frontier["metrics"]["trades"] for frontier in ridge_valid)
    xgboost_trades = sum(frontier["metrics"]["trades"] for frontier in xgboost_valid)
    champion = (
        "xgboost_gpu"
        if xgboost["selection_gross_mae"] < ridge["selection_gross_mae"]
        and xgboost["selection_target_brier"] < ridge["selection_target_brier"]
        and len(xgboost_valid) >= len(ridge_valid)
        and xgboost_trades > ridge_trades
        else "ridge"
    )
    selected_model = fitted[champion]
    scored_selection = _score(
        selected_model["model"], selected_model["calibrator"], selection
    )
    scored_audit = _score(selected_model["model"], selected_model["calibrator"], audit)

    profiles: dict[str, Any] = {}
    for level in range(6):
        profile = f"VIP{level}"
        cost = profile_cost_bps(profile)
        frontier = candidates[champion]["selection_frontiers"][profile]
        if frontier is None:
            profiles[profile] = {
                "cost_bps": cost,
                "selection": None,
                "audit": None,
                "paper_eligible": False,
            }
            continue
        selected_audit = _apply_threshold(
            scored_audit, cost, float(frontier["threshold_bps"])
        )
        audit_metrics = _metrics(selected_audit, cost)
        stress_metrics = _metrics(selected_audit, 2 * cost)
        lcb = _bootstrap_lcb(selected_audit, cost)
        profiles[profile] = {
            "cost_bps": cost,
            "selection": {
                "threshold_bps": frontier["threshold_bps"],
                "metrics": frontier["metrics"],
                "calendar_days": 184,
                "trades_per_calendar_day": frontier["metrics"]["trades"] / 184,
            },
            "audit": {
                "metrics": audit_metrics,
                "stress_2x": stress_metrics,
                "expectancy_lcb_95_bps": lcb,
                "calendar_days": int((HOLDOUT_START - pd.Timestamp("2026-01-01", tz="UTC")).days),
                "trades_per_calendar_day": audit_metrics["trades"]
                / max(1, (HOLDOUT_START - pd.Timestamp("2026-01-01", tz="UTC")).days),
            },
            "paper_eligible": bool(
                audit_metrics["trades"] >= 50
                and audit_metrics["expectancy_bps"] > 0
                and audit_metrics["profit_factor"] >= 1.15
                and audit_metrics["max_drawdown"] <= 0.08
                and stress_metrics["expectancy_bps"] >= 0
                and np.isfinite(lcb)
                and lcb > 0
            ),
        }
    eligible_profiles = [
        profile for profile, result in profiles.items() if result["paper_eligible"]
    ]
    frozen_base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    base_ready = frozen_base.get("verdict") == "RESEARCH_BASE_ALPHA_READY"
    verdict = (
        "FREQUENCY_ALPHA_READY"
        if eligible_profiles
        else "BASE_ONLY_NO_FREQUENCY_GAIN"
        if base_ready
        else "NO_FREQUENCY_ALPHA"
    )
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "context_audit": {
            "binance_context_rows": len(pd.read_parquet(BINANCE_CONTEXT)),
            "matrix_coverage": float(coverage.mean()),
            "excluded_rows": int((~coverage).sum()),
            "context_lookahead_violations": int(
                (
                    matrix["binance_context_available_at"] > matrix["available_at"]
                ).fillna(False).sum()
            ),
            "market_lookahead_violations": int(
                (matrix["max_input_available_at"] > matrix["available_at"])
                .fillna(False)
                .sum()
            ),
        },
        "raw_split_rows": raw_split_rows,
        "split_rows": {
            "fit": len(fit),
            "calibration": len(calibration),
            "selection": len(selection),
            "audit": len(audit),
        },
        "model_comparison": candidates,
        "champion": champion,
        "profiles": profiles,
        "eligible_profiles": eligible_profiles,
        "frozen_base": {
            "protocol_hash": frozen_base.get("protocol_hash"),
            "verdict": frozen_base.get("verdict"),
            "prior_combined": frozen_base.get("prior_combined"),
            "audit_2026": frozen_base.get("test"),
            "audit_2026_stress_2x": frozen_base.get("test_stress"),
            "paper_profiles": frozen_base.get("paper_profiles"),
            "remains_official_fallback": bool(base_ready and not eligible_profiles),
        },
        "verdict": verdict,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary_bundle = BUNDLE.with_suffix(".joblib.tmp")
    joblib.dump(
        {
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "features": FEATURES,
            "champion": champion,
            "model": selected_model["model"],
            "calibrator": selected_model["calibrator"],
            "profiles": profiles,
            "frozen_base_protocol_hash": frozen_base.get("protocol_hash"),
            "fallback": "frozen_v8" if base_ready and not eligible_profiles else None,
            "research_only": True,
        },
        temporary_bundle,
    )
    os.replace(temporary_bundle, BUNDLE)
    _atomic_json(REPORT, report)
    _status("complete", 100, report["verdict"])
    return report


if __name__ == "__main__":
    print(json.dumps(_json_safe(train()), indent=2, allow_nan=False, default=str))
