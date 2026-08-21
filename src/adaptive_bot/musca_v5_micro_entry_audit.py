from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from adaptive_bot.musca_v4_research import MINUTES
from adaptive_bot.musca_v5_event_policy import simulate_management
from adaptive_bot.musca_v5_fine_tuning import (
    COST_TARGET_MULTIPLIERS,
    FEATURES,
    HOLDOUT_START,
    POLICY_PROTOCOL_HASH,
    _apply_policy_threshold,
    _atomic_json,
    _attach_context,
    _conditional_ev,
    _cost_runner_plan,
    _deduplicate_broad_candidates,
    _metrics,
    _models,
    _one_position,
    _policy_coverage_evaluations,
    _policy_period_summary,
    _weights,
    profile_cost_bps,
)
from adaptive_bot.musca_v5_frequency_audit import CANDIDATES
from adaptive_bot.musca_v5_micro_model import (
    DIRECTIONAL_ONE_SECOND_FEATURES,
    ONE_SECOND_MICRO_FEATURES,
    build_micro_features,
    build_one_second_features,
)
from adaptive_bot.musca_v5_microstructure import ROOT as AGGTRADES_ROOT

REPORT = Path("data/reports/musca_v5_micro_entry_audit.json")
STATUS = Path("data/reports/musca_v5_micro_entry_audit.status.json")
BUNDLE = Path("data/models/musca_v5/micro_entry_audit.joblib")
TIMING_MATRIX = Path("data/ml/musca_v5/micro_timing_matrix.parquet")
MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 6))
ENTRY_STYLES = ("immediate_next_bucket", "micro_confirmation")
MICRO_FEATURES = tuple(f"micro_{name}" for name in ONE_SECOND_MICRO_FEATURES)
MODEL_FEATURES = (*FEATURES, *MICRO_FEATURES, "round_trip_cost_bps")
TIMING_PROTOCOL = {
    "name": "musca_v5_five_second_restart_timing",
    "outcome_protocol_hash": POLICY_PROTOCOL_HASH,
    "source": "Binance official USD-M aggTrades 1s",
    "months": list(MONTHS),
    "latest_input": HOLDOUT_START.isoformat(),
    "join": "backward_only_two_second_tolerance",
    "features": list(MICRO_FEATURES),
    "direction": "frozen_v8_only",
    "restart_confirmation": {
        "maximum_minutes": 5,
        "signed_price_velocity_15s_bps": 0.5,
        "signed_ofi_15s": 0.05,
        "signed_ofi_1m": -0.02,
        "trade_intensity_15s": 0.8,
        "entry": "next_5s_bucket_open",
    },
    "entry_styles": list(ENTRY_STYLES),
    "path": "observed_5s; same_bucket_stop_wins",
    "stop": "causal_micro_12_to_60bps_never_widens",
    "targets": "TP1=max(profile_cost_floor,1R); TP2=max(profile_cost_floor,2R)",
    "partial_exit": "50pct_at_TP1_then_cost_protected_dynamic_trailing",
}
TIMING_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(TIMING_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
LEGACY_TIMING_PROTOCOL_HASH = (
    "97f2cf092f9bd46735626d9769d6b24488752d69a6a9ee44225c70ddb39f574a"
)
PROTOCOL = {
    "name": "musca_v5_conditional_net_ev_micro_audit",
    "timing_protocol_hash": TIMING_PROTOCOL_HASH,
    "timing_protocol": TIMING_PROTOCOL,
    "model": {
        "probability": "P(net_return_bps > 0)",
        "conditional_gain": "E(gross_return_bps | net_return_bps > 0)",
        "conditional_miss": "E(gross_return_bps | net_return_bps <= 0)",
        "composition": "P*conditional_gain + (1-P)*conditional_miss",
        "probability_calibration": "chronological_isotonic",
        "gross_ev_calibration": "chronological_isotonic",
        "ranking": "continuous_uncalibrated_conditional_EV",
        "profile_cost": "causal_feature_and_subtracted_once_by_policy_gate",
    },
    "splits": [
        {
            "fit": "January",
            "calibration": "February",
            "selection": "March",
            "test": "April",
        },
        {
            "fit": "January-February",
            "calibration": "March",
            "selection": "April",
            "test": "May_before_sealed_holdout",
        },
    ],
    "ridge": "default_champion",
    "xgboost_gpu": "challenger_same_rows",
    "economic_gate_costs": "observed_profile_costs_1x",
    "cost_stress": "2x_reported_as_diagnostic_only",
    "changes_to_active_paper": False,
    "holdout_opened": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _write_timing_matrix(matrix: pd.DataFrame) -> None:
    TIMING_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = TIMING_MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    os.replace(temporary, TIMING_MATRIX)


def _funding_return_bps(
    outcome: dict[str, Any], side: int, funding: pd.DataFrame
) -> float:
    entry = pd.Timestamp(outcome["entry_timestamp"])
    exit_time = pd.Timestamp(outcome["exit_timestamp"])
    events = funding.loc[
        funding["timestamp"].ge(entry) & funding["timestamp"].le(exit_time)
    ]
    if events.empty:
        return 0.0
    first_target = (
        entry + pd.Timedelta(seconds=float(outcome["time_to_first_target_seconds"]))
        if bool(outcome["first_target_hit"])
        else pd.NaT
    )
    total = 0.0
    for event in events.itertuples(index=False):
        fraction = 0.5 if pd.notna(first_target) and event.timestamp > first_target else 1.0
        total -= side * float(event.funding_event_rate) * 10_000 * fraction
    return total


def build_timing_matrix(*, force: bool = False) -> pd.DataFrame:
    if TIMING_MATRIX.exists() and not force:
        cached = pd.read_parquet(TIMING_MATRIX)
        if cached["timing_protocol_hash"].eq(TIMING_PROTOCOL_HASH).all():
            return cached
        if cached["timing_protocol_hash"].eq(LEGACY_TIMING_PROTOCOL_HASH).all():
            cached["timing_protocol_hash"] = TIMING_PROTOCOL_HASH
            _write_timing_matrix(cached)
            return cached
    raw_parts: list[pd.DataFrame] = []
    for month in MONTHS:
        filters = (
            [("available_at", "<", HOLDOUT_START.to_pydatetime())]
            if month == "2026-05"
            else None
        )
        raw_parts.append(
            pd.read_parquet(
                AGGTRADES_ROOT / f"BTCUSDT-aggTrades-5s-{month}.parquet",
                filters=filters,
            )
        )
    raw = pd.concat(raw_parts, ignore_index=True).sort_values("available_at")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["available_at"] = pd.to_datetime(raw["available_at"], utc=True)
    data = (
        raw.merge(
            build_micro_features(raw),
            on="available_at",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("available_at")
        .reset_index(drop=True)
    )
    available_ns = (
        data["available_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    )
    candidates = _deduplicate_broad_candidates(pd.read_parquet(CANDIDATES))
    candidates["available_at"] = pd.to_datetime(candidates["available_at"], utc=True)
    latest_setup = HOLDOUT_START - pd.Timedelta(minutes=65)
    candidates = candidates.loc[
        candidates["available_at"].ge(data["available_at"].min())
        & candidates["available_at"].lt(latest_setup)
    ]
    funding = pd.read_parquet(MINUTES)[["timestamp", "funding_event_rate"]]
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
    funding = funding.loc[
        funding["funding_event_rate"].fillna(0).ne(0)
        & funding["timestamp"].ge(data["timestamp"].min())
        & funding["timestamp"].lt(HOLDOUT_START)
    ]
    records: list[dict[str, Any]] = []
    events = candidates.to_dict("records")
    for event_number, event in enumerate(events, start=1):
        side = int(event["direction"])
        for entry_style in ENTRY_STYLES:
            for level in range(6):
                profile = f"VIP{level}"
                cost = profile_cost_bps(profile)
                for multiplier in COST_TARGET_MULTIPLIERS:
                    plan = _cost_runner_plan(cost, multiplier)
                    outcome = simulate_management(
                        event,
                        data,
                        plan,
                        cost_bps=cost,
                        minimum_gross_to_cost=0.0,
                        stop_style="micro",
                        available_ns=available_ns,
                        entry_style=entry_style,
                        dynamic_protected_trailing=True,
                    )
                    if outcome is None:
                        continue
                    funding_bps = _funding_return_bps(outcome, side, funding)
                    market_gross = float(outcome["gross_return_bps"])
                    gross = market_gross + funding_bps
                    first_target_time = (
                        pd.Timestamp(outcome["entry_timestamp"])
                        + pd.Timedelta(
                            seconds=float(outcome["time_to_first_target_seconds"])
                        )
                        if bool(outcome["first_target_hit"])
                        else pd.NaT
                    )
                    records.append(
                        outcome
                        | {
                        "setup_available_at": event["available_at"],
                        "available_at": outcome["entry_decision_available_at"],
                        "fee_profile": profile,
                        "round_trip_cost_bps": cost,
                        "target_cost_multiple": multiplier,
                        "entry_style": entry_style,
                        "policy_action": "FIVE_SECOND_PARTIAL_RUNNER",
                        "gross_market_return_bps": market_gross,
                        "funding_return_bps": funding_bps,
                        "gross_return_bps": gross,
                        "net_return_bps": gross - cost,
                        "stress_return_bps": gross - 2 * cost,
                        "initial_stop_bps": abs(float(outcome["initial_stop_bps"])),
                        "first_target_bps": float(outcome["plan_first_target_bps"]),
                        "second_target_bps": float(outcome["plan_second_target_bps"]),
                        "first_target_timestamp": first_target_time,
                        "first_target_state_available_at": first_target_time,
                        "close_at_first_gross_bps": (
                            float(outcome["close_at_first_net_bps"]) + cost
                        ),
                        "duration_minutes": float(outcome["duration_seconds"]) / 60,
                        "planned_gross_bps": (
                            plan.first_exit_fraction * plan.first_target_bps
                            + (1 - plan.first_exit_fraction)
                            * plan.second_target_bps
                        ),
                        "timing_protocol_hash": TIMING_PROTOCOL_HASH,
                        }
                    )
        if event_number % 100 == 0:
            _atomic_json(
                STATUS,
                {
                    "phase": "five_second_paths",
                    "percent": 5 + 20 * event_number / len(events),
                    "detail": f"{event_number:,}/{len(events):,} V8 setups",
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
    matrix = pd.DataFrame(records)
    _write_timing_matrix(matrix)
    return matrix


def merge_causal_micro(rows: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    renamed = micro.rename(
        columns={
            "available_at": "micro_available_at",
            **{name: f"micro_{name}" for name in ONE_SECOND_MICRO_FEATURES},
        }
    )
    output = pd.merge_asof(
        rows.sort_values("available_at"),
        renamed.sort_values("micro_available_at"),
        left_on="available_at",
        right_on="micro_available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=2),
    )
    for name in DIRECTIONAL_ONE_SECOND_FEATURES:
        output[f"micro_{name}"] *= output["direction"]
    output["micro_feature_coverage_valid"] = (
        output["micro_available_at"].notna()
        & output["micro_available_at"].le(output["available_at"])
        & output.loc[:, MICRO_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    return output


def attach_one_second_micro(rows: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    available = pd.to_datetime(rows["available_at"], utc=True)
    exits = pd.to_datetime(rows["exit_timestamp"], utc=True)
    source = rows.loc[
        available.ge(start) & available.lt(HOLDOUT_START) & exits.lt(HOLDOUT_START)
    ].copy()
    parts: list[pd.DataFrame] = []
    for number, month in enumerate(MONTHS, start=1):
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        left = source.loc[
            source["available_at"].ge(month_start)
            & source["available_at"].lt(min(month_end, HOLDOUT_START))
        ]
        if left.empty:
            continue
        filters = (
            [("available_at", "<", HOLDOUT_START.to_pydatetime())]
            if month == "2026-05"
            else None
        )
        raw = pd.read_parquet(
            AGGTRADES_ROOT / f"BTCUSDT-aggTrades-1s-{month}.parquet",
            filters=filters,
        )
        raw["available_at"] = pd.to_datetime(raw["available_at"], utc=True)
        parts.append(merge_causal_micro(left, build_one_second_features(raw)))
        _atomic_json(
            STATUS,
            {
                "phase": "micro_features",
                "percent": 5 + 25 * number / len(MONTHS),
                "detail": f"{month}: {len(left):,} action rows",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    if not parts:
        raise ValueError("No pre-holdout V8 actions overlap official one-second data")
    return pd.concat(parts, ignore_index=True).sort_values("available_at")


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, MODEL_FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Micro entry model received incomplete causal features")
    return values


def _fit(kind: str, fit: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    gain_regressor, classifier = _models(kind)
    miss_regressor, _ = _models(kind)
    weights = _weights(fit)
    positive = fit["net_return_bps"].gt(0)
    if positive.all() or not positive.any():
        raise ValueError("Conditional EV fitting requires both economic outcomes")
    if kind == "ridge":
        gain_regressor.fit(
            _x(fit.loc[positive]),
            fit.loc[positive, "gross_return_bps"],
            ridge__sample_weight=weights[positive.to_numpy()],
        )
        miss_regressor.fit(
            _x(fit.loc[~positive]),
            fit.loc[~positive, "gross_return_bps"],
            ridge__sample_weight=weights[(~positive).to_numpy()],
        )
        classifier.fit(
            _x(fit),
            positive.astype(int),
            logisticregression__sample_weight=weights,
        )
    else:
        gain_regressor.fit(
            _x(fit.loc[positive]),
            fit.loc[positive, "gross_return_bps"],
            sample_weight=weights[positive.to_numpy()],
        )
        miss_regressor.fit(
            _x(fit.loc[~positive]),
            fit.loc[~positive, "gross_return_bps"],
            sample_weight=weights[(~positive).to_numpy()],
        )
        classifier.fit(_x(fit), positive.astype(int), sample_weight=weights)
    raw_probability = np.asarray(
        classifier.predict_proba(_x(calibration))[:, 1], dtype=float
    )
    probability_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_probability, calibration["net_return_bps"].gt(0).astype(int)
    )
    probability = probability_calibrator.predict(raw_probability)
    gain = np.asarray(gain_regressor.predict(_x(calibration)), dtype=float)
    miss = np.asarray(miss_regressor.predict(_x(calibration)), dtype=float)
    raw_ev = _conditional_ev(probability, gain, miss)
    ev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_ev, calibration["gross_return_bps"]
    )
    return {
        "classifier": classifier,
        "gain_regressor": gain_regressor,
        "miss_regressor": miss_regressor,
        "probability_calibrator": probability_calibrator,
        "ev_calibrator": ev_calibrator,
    }


def _score(fitted: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    raw_probability = np.asarray(
        fitted["classifier"].predict_proba(_x(rows))[:, 1], dtype=float
    )
    probability = fitted["probability_calibrator"].predict(raw_probability)
    gain = np.asarray(fitted["gain_regressor"].predict(_x(rows)), dtype=float)
    miss = np.asarray(fitted["miss_regressor"].predict(_x(rows)), dtype=float)
    raw_ev = _conditional_ev(probability, gain, miss)
    output["predicted_net_positive_probability"] = probability
    output["predicted_conditional_gain_bps"] = gain
    output["predicted_conditional_miss_bps"] = miss
    output["predicted_uncalibrated_gross_bps"] = raw_ev
    output["predicted_gross_bps"] = fitted["ev_calibrator"].predict(raw_ev)
    return output


def _partition(
    rows: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp
) -> pd.DataFrame:
    entries = pd.to_datetime(rows["entry_timestamp"], utc=True)
    exits = pd.to_datetime(rows["exit_timestamp"], utc=True)
    mask = entries.lt(end) & exits.lt(end)
    if start is not None:
        mask &= entries.ge(start)
    return rows.loc[mask].copy()


def _frontiers(scored: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluations = {
        f"VIP{level}": _policy_coverage_evaluations(
            scored.loc[scored["fee_profile"].eq(f"VIP{level}")],
            profile_cost_bps(f"VIP{level}"),
        )
        for level in range(6)
    }
    frontiers = {
        profile: max(
            (
                candidate
                for candidate in candidates
                if candidate["operational_pass"]
            ),
            key=lambda candidate: candidate["metrics"]["trades"],
            default=None,
        )
        for profile, candidates in evaluations.items()
    }
    return evaluations, frontiers


def run() -> dict[str, Any]:
    matrix = attach_one_second_micro(
        _attach_context(build_timing_matrix())
    )
    matrix = matrix.loc[
        matrix["model_feature_coverage_valid"]
        & matrix["micro_feature_coverage_valid"]
    ].copy()
    specs = (
        (
            pd.Timestamp("2026-02-01", tz="UTC"),
            pd.Timestamp("2026-03-01", tz="UTC"),
            pd.Timestamp("2026-04-01", tz="UTC"),
            pd.Timestamp("2026-05-01", tz="UTC"),
        ),
        (
            pd.Timestamp("2026-03-01", tz="UTC"),
            pd.Timestamp("2026-04-01", tz="UTC"),
            pd.Timestamp("2026-05-01", tz="UTC"),
            HOLDOUT_START,
        ),
    )
    selected: dict[str, list[pd.DataFrame]] = {
        f"VIP{level}": [] for level in range(6)
    }
    folds: list[dict[str, Any]] = []
    last_bundle: dict[str, Any] | None = None
    for fold_number, (fit_end, calibration_end, selection_end, test_end) in enumerate(
        specs, start=1
    ):
        fit = _partition(matrix, None, fit_end)
        calibration = _partition(matrix, fit_end, calibration_end)
        selection = _partition(matrix, calibration_end, selection_end)
        test = _partition(matrix, selection_end, test_end)
        kinds: dict[str, Any] = {}
        for kind in ("ridge", "xgboost_gpu"):
            fitted = _fit(kind, fit, calibration)
            scored_selection = _score(fitted, selection)
            evaluations, frontiers = _frontiers(scored_selection)
            kinds[kind] = {
                "fitted": fitted,
                "test": _score(fitted, test),
                "evaluations": evaluations,
                "frontiers": frontiers,
                "selection_mae": float(
                    mean_absolute_error(
                        selection["gross_return_bps"],
                        scored_selection["predicted_gross_bps"],
                    )
                ),
                "selection_brier": float(
                    brier_score_loss(
                        selection["net_return_bps"].gt(0),
                        scored_selection["predicted_net_positive_probability"],
                    )
                ),
            }
        ridge_valid = sum(
            value is not None for value in kinds["ridge"]["frontiers"].values()
        )
        xgb_valid = sum(
            value is not None
            for value in kinds["xgboost_gpu"]["frontiers"].values()
        )
        champion = (
            "xgboost_gpu"
            if kinds["xgboost_gpu"]["selection_mae"]
            < kinds["ridge"]["selection_mae"]
            and kinds["xgboost_gpu"]["selection_brier"]
            < kinds["ridge"]["selection_brier"]
            and xgb_valid > ridge_valid
            else "ridge"
        )
        chosen = kinds[champion]
        profile_reports: dict[str, Any] = {}
        for level in range(6):
            profile = f"VIP{level}"
            cost = profile_cost_bps(profile)
            frontier = chosen["frontiers"][profile]
            diagnostics = chosen["evaluations"][profile]
            if frontier is None:
                profile_reports[profile] = {
                    "selection_frontier": None,
                    "selection_diagnostics": diagnostics,
                    "test_trades": 0,
                }
                continue
            accepted = _apply_policy_threshold(
                chosen["test"].loc[chosen["test"]["fee_profile"].eq(profile)],
                cost,
                float(frontier["threshold_bps"]),
            )
            accepted["micro_fold"] = fold_number
            accepted["micro_model"] = champion
            selected[profile].append(accepted)
            profile_reports[profile] = {
                "selection_frontier": frontier,
                "selection_diagnostics": diagnostics,
                "test_trades": len(accepted),
                "test_metrics": _metrics(accepted, cost),
            }
        folds.append(
            {
                "fold": fold_number,
                "boundaries": {
                    "fit_end": fit_end,
                    "calibration_end": calibration_end,
                    "selection_end": selection_end,
                    "test_end": test_end,
                },
                "rows": {
                    "fit": len(fit),
                    "calibration": len(calibration),
                    "selection": len(selection),
                    "test": len(test),
                },
                "model_comparison": {
                    kind: {
                        "selection_mae": value["selection_mae"],
                        "selection_net_positive_brier": value["selection_brier"],
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
        last_bundle = {"champion": champion, "fitted": chosen["fitted"]}
        _atomic_json(
            STATUS,
            {
                "phase": "micro_walk_forward_gpu",
                "percent": 30 + 60 * fold_number / len(specs),
                "detail": f"Fold {fold_number}/{len(specs)}",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    profiles: dict[str, Any] = {}
    research_profiles: list[str] = []
    for level in range(6):
        profile = f"VIP{level}"
        cost = profile_cost_bps(profile)
        rows = (
            _one_position(pd.concat(selected[profile], ignore_index=True))
            if selected[profile]
            else matrix.iloc[0:0].copy()
        )
        summary = _policy_period_summary(
            rows,
            cost,
            pd.Timestamp("2026-04-01", tz="UTC"),
            HOLDOUT_START,
        )
        gates = {
            "trades_50": summary["metrics"]["trades"] >= 50,
            "expectancy_positive": summary["metrics"]["expectancy_bps"] > 0,
            "pf_1_15": summary["metrics"]["profit_factor"] >= 1.15,
            "drawdown_8pct": summary["metrics"]["max_drawdown"] <= 0.08,
            "expectancy_lcb_positive": bool(
                np.isfinite(summary["expectancy_lcb_95_bps"])
                and summary["expectancy_lcb_95_bps"] > 0
            ),
            "majority_positive_active_days": summary["positive_active_days"] > 0.5,
        }
        if all(gates.values()):
            research_profiles.append(profile)
        profiles[profile] = {
            "round_trip_cost_bps": cost,
            "summary": summary,
            "gates": gates,
            "stress_2x_nonnegative_diagnostic": (
                summary["stress_2x"]["expectancy_bps"] >= 0
            ),
            "research_signal": all(gates.values()),
        }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "timing_protocol_hash": TIMING_PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "micro_lookahead_violations": int(
            (matrix["micro_available_at"] > matrix["available_at"]).sum()
        ),
        "micro_coverage": float(matrix["micro_feature_coverage_valid"].mean()),
        "folds": folds,
        "profiles": profiles,
        "research_profiles": research_profiles,
        "verdict": (
            "MICRO_ENTRY_SIGNAL_RESEARCH_ONLY"
            if research_profiles
            else "BASE_ONLY_NO_MICRO_ENTRY_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    if last_bundle is not None:
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        temporary = BUNDLE.with_suffix(".joblib.tmp")
        joblib.dump(
            {
                "protocol": PROTOCOL,
                "protocol_hash": PROTOCOL_HASH,
                "features": MODEL_FEATURES,
                "last_fold": last_bundle,
                "profiles": profiles,
                "research_only": True,
            },
            temporary,
        )
        os.replace(temporary, BUNDLE)
    _atomic_json(REPORT, report)
    _atomic_json(
        STATUS,
        {
            "phase": "complete",
            "percent": 100,
            "detail": report["verdict"],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
