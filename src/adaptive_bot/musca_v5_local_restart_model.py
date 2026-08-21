from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from adaptive_bot.musca_v4_research import MINUTES, _non_overlapping
from adaptive_bot.musca_v5_fine_tuning import _atomic_json, _models
from adaptive_bot.musca_v5_local_restart_frontier import MATRIX
from adaptive_bot.musca_v5_local_restart_frontier import (
    PROTOCOL_HASH as OUTCOME_PROTOCOL_HASH,
)
from adaptive_bot.musca_v5_room_frontier import _audit_gates, _summary
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    PAPER_PROFILES,
    profile_cost_bps,
)

REPORT = Path("data/reports/musca_v5_local_restart_model.json")
STATUS = Path("data/reports/musca_v5_local_restart_model.status.json")
BUNDLE = Path("data/models/musca_v5/local_restart_model_research.joblib")
QUALITY_REPORT = Path("data/reports/musca_v5_local_restart_quality_model.json")
QUALITY_BUNDLE = Path("data/models/musca_v5/local_restart_quality_model_research.joblib")
FIT_END = pd.Timestamp("2024-09-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2024-11-01", tz="UTC")
SELECTION_END = pd.Timestamp("2025-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-01-01", tz="UTC")
COVERAGES = (0.10, 0.05, 0.025, 0.01)
FEATURES = (
    "direction",
    "expert_breakout_bars",
    "trend_strength",
    "directional_return_15m",
    "directional_return_60m",
    "directional_spot_return_15m",
    "directional_vwap_60m_slope",
    "relative_volume",
    "directional_perp_taker_1m",
    "directional_perp_taker_5m",
    "directional_spot_taker_5m",
    "pullback_depth_atr",
    "risk_bps_at_signal",
    "room_bps",
    "reward_risk_at_signal",
    "directional_vwap_distance_bps",
    "hour_sin",
    "hour_cos",
)
QUALITY_FEATURES = (
    *FEATURES,
    "atr_bps",
    "atr_relative_to_previous_day",
    "directional_return_1m",
    "directional_return_5m",
    "directional_acceleration_1m_5m",
    "directional_close_location",
    "directional_rejection_wick",
    "body_fraction",
    "directional_restart_break_bps",
    "trade_count_z60",
    "quote_volume_z60",
    "directional_taker_change_1m_5m",
    "directional_basis_bps",
    "directional_basis_change_5m",
    "directional_spot_perp_divergence_1m",
    "directional_spot_perp_divergence_5m",
    "impulse_age_minutes",
    "volume_ratio_to_impulse",
    "directional_taker_change_from_impulse",
)
PROTOCOL = {
    "name": "musca_v5_local_restart_conditional_ev_v1",
    "outcome_protocol_hash": OUTCOME_PROTOCOL_HASH,
    "features": list(FEATURES),
    "targets": {
        "probability": "P(impulse_extreme_target_before_other_exit)",
        "conditional_gain": "E(gross_bps|target)",
        "conditional_miss": "E(gross_bps|not_target)",
        "composition": "P*gain+(1-P)*miss",
    },
    "models": ["ridge_default_champion", "xgboost_hist_cuda_challenger"],
    "fit_end": FIT_END.isoformat(),
    "calibration_end": CALIBRATION_END.isoformat(),
    "selection_end": SELECTION_END.isoformat(),
    "validation_end": VALIDATION_END.isoformat(),
    "audit_end": HOLDOUT_START.isoformat(),
    "coverage_frontier": list(COVERAGES),
    "selection": "model_and_absolute_score_threshold_on_2024_selection_only",
    "validation": "2025_unseen_during_selection",
    "audit": "2026_preholdout_reused_discovery_only",
    "flat": "zero_and_selected_when_no_predicted_net_positive_action",
    "position_limit": 1,
    "economic_costs": "profile_taker_costs_1x",
    "cost_stress": "2x_diagnostic_only",
    "holdout_opened": False,
    "changes_to_active_paper": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
QUALITY_PROTOCOL = PROTOCOL | {
    "name": "musca_v5_local_restart_quality_conditional_ev_v1",
    "features": list(QUALITY_FEATURES),
    "feature_extension": "causal_restart_bar_impulse_comparison_and_mark_basis",
}
QUALITY_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(QUALITY_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


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


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    side = data["direction"].to_numpy(float)
    signal_close = data["target_price"] / (1 + side * data["room_bps"] / 10_000)
    data["trend_strength"] = side * data["trend_score"]
    for source, target in (
        ("return_15m", "directional_return_15m"),
        ("return_60m", "directional_return_60m"),
        ("spot_return_15m", "directional_spot_return_15m"),
        ("vwap_60m_slope", "directional_vwap_60m_slope"),
        ("perp_taker_1m", "directional_perp_taker_1m"),
        ("perp_taker_5m", "directional_perp_taker_5m"),
        ("spot_taker_5m", "directional_spot_taker_5m"),
    ):
        data[target] = side * data[source]
    data["reward_risk_at_signal"] = data["room_bps"] / data["risk_bps_at_signal"].replace(0, np.nan)
    data["directional_vwap_distance_bps"] = (
        side * (signal_close - data["operating_vwap"]) / signal_close * 10_000
    )
    time = pd.to_datetime(data["signal_timestamp"], utc=True)
    hour = time.dt.hour + time.dt.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["target_hit"] = data["exit_reason"].astype(str).str.endswith("TARGET")
    coverage: pd.Series = (
        data.loc[:, list(FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis="columns")
    )
    data["model_feature_coverage_valid"] = coverage
    return data.loc[data["model_feature_coverage_valid"]].reset_index(drop=True)


def attach_restart_quality(rows: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    path = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True).copy()
    path["timestamp"] = pd.to_datetime(path["timestamp"], utc=True)
    previous = path["perp_close"].shift(1)
    true_range = pd.concat(
        [
            path["perp_high"] - path["perp_low"],
            (path["perp_high"] - previous).abs(),
            (path["perp_low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_baseline = atr.shift(1).rolling(1_440, min_periods=480).median()
    bar_range = (path["perp_high"] - path["perp_low"]).replace(0, np.nan)
    close_location = (path["perp_close"] - path["perp_low"]) / bar_range
    lower_wick = (path[["perp_open", "perp_close"]].min(axis=1) - path["perp_low"]) / bar_range
    upper_wick = (path["perp_high"] - path[["perp_open", "perp_close"]].max(axis=1)) / bar_range
    taker_1m = 2 * path["perp_taker_buy_quote"] / path["perp_quote_volume"].replace(0, np.nan) - 1
    taker_5m = (
        2
        * path["perp_taker_buy_quote"].rolling(5).sum()
        / path["perp_quote_volume"].rolling(5).sum().replace(0, np.nan)
        - 1
    )
    return_1m = path["perp_close"].pct_change()
    return_5m = path["perp_close"].pct_change(5)
    spot_return_1m = path["spot_close"].pct_change()
    spot_return_5m = path["spot_close"].pct_change(5)
    basis = (path["mark_close"] - path["perp_close"]) / path["perp_close"] * 10_000

    def previous_zscore(values: pd.Series) -> pd.Series:
        mean = values.shift(1).rolling(60, min_periods=30).mean()
        std = values.shift(1).rolling(60, min_periods=30).std().replace(0, np.nan)
        return (values - mean) / std

    quality = pd.DataFrame(
        {
            "signal_timestamp": path["timestamp"],
            "atr_bps": atr / path["perp_close"] * 10_000,
            "atr_relative_to_previous_day": atr / atr_baseline,
            "return_1m_quality": return_1m,
            "return_5m_quality": return_5m,
            "close_location": close_location,
            "lower_wick": lower_wick,
            "upper_wick": upper_wick,
            "body_fraction": (path["perp_close"] - path["perp_open"]).abs() / bar_range,
            "long_restart_break_bps": (
                (path["perp_close"] - path["perp_high"].shift(1)) / path["perp_close"] * 10_000
            ),
            "short_restart_break_bps": (
                (path["perp_low"].shift(1) - path["perp_close"]) / path["perp_close"] * 10_000
            ),
            "trade_count_z60": previous_zscore(path["perp_trade_count"]),
            "quote_volume_z60": previous_zscore(path["perp_quote_volume"]),
            "taker_1m_quality": taker_1m,
            "taker_5m_quality": taker_5m,
            "basis_bps_quality": basis,
            "basis_change_5m": basis.diff(5),
            "spot_perp_divergence_1m": spot_return_1m - return_1m,
            "spot_perp_divergence_5m": spot_return_5m - return_5m,
            "signal_quote_volume": path["perp_quote_volume"],
        }
    )
    impulse = pd.DataFrame(
        {
            "impulse_anchor_at": path["timestamp"],
            "impulse_quote_volume": path["perp_quote_volume"],
            "impulse_taker_1m": taker_1m,
        }
    )
    output = rows.copy()
    output["signal_timestamp"] = pd.to_datetime(output["signal_timestamp"], utc=True)
    output["impulse_anchor_at"] = pd.to_datetime(output["impulse_anchor_at"], utc=True)
    output = output.merge(quality, on="signal_timestamp", how="left", validate="many_to_one")
    output = output.merge(impulse, on="impulse_anchor_at", how="left", validate="many_to_one")
    side = output["direction"].to_numpy(float)
    output["directional_return_1m"] = side * output["return_1m_quality"]
    output["directional_return_5m"] = side * output["return_5m_quality"]
    output["directional_acceleration_1m_5m"] = side * (
        output["return_1m_quality"] - output["return_5m_quality"] / 5
    )
    output["directional_close_location"] = np.where(
        side > 0, output["close_location"], 1 - output["close_location"]
    )
    output["directional_rejection_wick"] = np.where(
        side > 0, output["lower_wick"], output["upper_wick"]
    )
    output["directional_restart_break_bps"] = np.where(
        side > 0, output["long_restart_break_bps"], output["short_restart_break_bps"]
    )
    output["directional_taker_change_1m_5m"] = side * (
        output["taker_1m_quality"] - output["taker_5m_quality"]
    )
    output["directional_basis_bps"] = side * output["basis_bps_quality"]
    output["directional_basis_change_5m"] = side * output["basis_change_5m"]
    output["directional_spot_perp_divergence_1m"] = side * output["spot_perp_divergence_1m"]
    output["directional_spot_perp_divergence_5m"] = side * output["spot_perp_divergence_5m"]
    output["impulse_age_minutes"] = (
        output["signal_timestamp"] - output["impulse_anchor_at"]
    ).dt.total_seconds() / 60
    output["volume_ratio_to_impulse"] = output["signal_quote_volume"] / output[
        "impulse_quote_volume"
    ].replace(0, np.nan)
    output["directional_taker_change_from_impulse"] = side * (
        output["taker_1m_quality"] - output["impulse_taker_1m"]
    )
    coverage = (
        output.loc[:, list(QUALITY_FEATURES)]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis="columns")
    )
    return output.loc[coverage].reset_index(drop=True)


def _x(rows: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> np.ndarray:
    values = rows.loc[:, features].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Local restart model received incomplete causal features")
    return values


def _weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("signal_timestamp")["signal_timestamp"].transform("size")
    return 1.0 / counts.to_numpy(float)


def _fit(kind: str, rows: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> dict[str, Any]:
    direct, classifier = _models(kind)
    gain, _ = _models(kind)
    miss, _ = _models(kind)
    weights = _weights(rows)
    hits = rows["target_hit"].to_numpy(bool)
    fit_args = {"ridge__sample_weight": weights} if kind == "ridge" else {"sample_weight": weights}
    direct.fit(_x(rows, features), rows["gross_return_bps"], **fit_args)
    gain_args = (
        {"ridge__sample_weight": weights[hits]}
        if kind == "ridge"
        else {"sample_weight": weights[hits]}
    )
    miss_args = (
        {"ridge__sample_weight": weights[~hits]}
        if kind == "ridge"
        else {"sample_weight": weights[~hits]}
    )
    gain.fit(_x(rows.loc[hits], features), rows.loc[hits, "gross_return_bps"], **gain_args)
    miss.fit(_x(rows.loc[~hits], features), rows.loc[~hits, "gross_return_bps"], **miss_args)
    classifier_args = (
        {"logisticregression__sample_weight": weights}
        if kind == "ridge"
        else {"sample_weight": weights}
    )
    classifier.fit(_x(rows, features), hits.astype(int), **classifier_args)
    return {"direct": direct, "classifier": classifier, "gain": gain, "miss": miss}


def _raw_probability(
    model: dict[str, Any], rows: pd.DataFrame, features: tuple[str, ...] = FEATURES
) -> np.ndarray:
    return np.asarray(model["classifier"].predict_proba(_x(rows, features))[:, 1], dtype=float)


def _raw_score(
    model: dict[str, Any],
    probability: np.ndarray,
    rows: pd.DataFrame,
    features: tuple[str, ...] = FEATURES,
) -> pd.DataFrame:
    output = rows.copy()
    output["predicted_target_probability"] = probability
    output["predicted_conditional_gain_bps"] = np.maximum(
        0.0, model["gain"].predict(_x(rows, features))
    )
    output["predicted_conditional_miss_bps"] = model["miss"].predict(_x(rows, features))
    output["predicted_direct_gross_bps"] = model["direct"].predict(_x(rows, features))
    output["predicted_gross_bps"] = (
        probability * output["predicted_conditional_gain_bps"]
        + (1 - probability) * output["predicted_conditional_miss_bps"]
    )
    return output


def _fit_calibrated(
    kind: str,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    features: tuple[str, ...] = FEATURES,
) -> dict[str, Any]:
    model = _fit(kind, fit, features)
    probability_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        _raw_probability(model, calibration, features),
        calibration["target_hit"].astype(int),
    )
    probability = probability_calibrator.predict(_raw_probability(model, calibration, features))
    scores = _raw_score(model, probability, calibration, features)
    ev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        scores["predicted_gross_bps"], calibration["gross_return_bps"]
    )
    return {
        "model": model,
        "probability_calibrator": probability_calibrator,
        "ev_calibrator": ev_calibrator,
    }


def _score(
    fitted: dict[str, Any], rows: pd.DataFrame, features: tuple[str, ...] = FEATURES
) -> pd.DataFrame:
    model = fitted["model"]
    probability = fitted["probability_calibrator"].predict(_raw_probability(model, rows, features))
    output = _raw_score(model, probability, rows, features)
    output["predicted_uncalibrated_gross_bps"] = output["predicted_gross_bps"]
    output["predicted_gross_bps"] = fitted["ev_calibrator"].predict(
        output["predicted_uncalibrated_gross_bps"]
    )
    return output


def _split(rows: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    entry = pd.to_datetime(rows["entry_timestamp"], utc=True)
    exit_time = pd.to_datetime(rows["exit_timestamp"], utc=True)
    selected = exit_time.lt(end)
    if start is not None:
        selected &= entry.ge(start)
    return rows.loc[selected].copy()


def _best_actions(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.sort_values("predicted_gross_bps", ascending=False)
        .drop_duplicates("signal_timestamp", keep="first")
        .reset_index(drop=True)
    )


def _selected(rows: pd.DataFrame, cost: float, threshold: float) -> pd.DataFrame:
    best = _best_actions(rows)
    selected = best.loc[
        (best["predicted_gross_bps"] > cost) & (best["predicted_gross_bps"] >= threshold)
    ]
    return _non_overlapping(selected)


def _selection_frontier(rows: pd.DataFrame, cost: float) -> list[dict[str, Any]]:
    best = _best_actions(rows).sort_values("predicted_gross_bps", ascending=False)
    positive = best.loc[best["predicted_gross_bps"] > cost]
    evaluations: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        count = min(len(positive), max(1, int(np.ceil(len(best) * coverage))))
        if not count:
            continue
        threshold = float(positive.iloc[count - 1]["predicted_gross_bps"])
        selected = _selected(rows, cost, threshold)
        summary = _summary(selected, cost)
        gates = {
            "trades_20": summary["metrics"]["trades"] >= 20,
            "expectancy_positive": summary["metrics"]["expectancy_bps"] > 0,
            "pf_1_05": summary["metrics"]["profit_factor"] >= 1.05,
        }
        evaluations.append(
            {
                "coverage": coverage,
                "threshold_gross_bps": threshold,
                "summary": summary,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )
    return evaluations


def _diagnostics(scored: pd.DataFrame) -> dict[str, float]:
    return {
        "gross_mae_bps": float(
            mean_absolute_error(scored["gross_return_bps"], scored["predicted_gross_bps"])
        ),
        "target_brier": float(
            brier_score_loss(
                scored["target_hit"].astype(int), scored["predicted_target_probability"]
            )
        ),
    }


def _choose_model(models: dict[str, Any], profile: str) -> str | None:
    ridge = models["ridge"]["profiles"][profile]
    xgb = models["xgboost"]["profiles"][profile]
    ridge_pass = ridge["selected"] is not None
    xgb_pass = xgb["selected"] is not None
    xgb_better = (
        models["xgboost"]["selection_diagnostics"]["gross_mae_bps"]
        < models["ridge"]["selection_diagnostics"]["gross_mae_bps"]
        and models["xgboost"]["selection_diagnostics"]["target_brier"]
        < models["ridge"]["selection_diagnostics"]["target_brier"]
    )
    if xgb_pass and xgb_better:
        return "xgboost"
    if ridge_pass:
        return "ridge"
    return None


def _run_model(
    data: pd.DataFrame,
    *,
    features: tuple[str, ...],
    protocol: dict[str, Any],
    protocol_hash: str,
    report_path: Path,
    bundle_path: Path,
    pass_verdict: str,
    fail_verdict: str,
) -> dict[str, Any]:
    fit = _split(data, None, FIT_END)
    calibration = _split(data, FIT_END, CALIBRATION_END)
    selection = _split(data, CALIBRATION_END, SELECTION_END)
    validation = _split(data, SELECTION_END, VALIDATION_END)
    audit = _split(data, VALIDATION_END, HOLDOUT_START)
    scored: dict[str, dict[str, pd.DataFrame]] = {}
    fitted_models: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}
    for number, kind in enumerate(("ridge", "xgboost"), start=1):
        _status("model_fit", 10 + 25 * (number - 1), kind)
        fitted = _fit_calibrated(kind, fit, calibration, features)
        fitted_models[kind] = fitted
        scored[kind] = {
            "selection": _score(fitted, selection, features),
            "validation": _score(fitted, validation, features),
            "audit": _score(fitted, audit, features),
        }
        selection_profiles: dict[str, Any] = {}
        for profile in PAPER_PROFILES:
            cost = profile_cost_bps(profile)
            frontier = _selection_frontier(scored[kind]["selection"], cost)
            passing = [candidate for candidate in frontier if candidate["pass"]]
            selection_profiles[profile] = {
                "round_trip_cost_bps_1x": cost,
                "frontier": frontier,
                "selected": (
                    max(passing, key=lambda item: item["summary"]["metrics"]["trades"])
                    if passing
                    else None
                ),
            }
        model_reports[kind] = {
            "selection_diagnostics": _diagnostics(scored[kind]["selection"]),
            "profiles": selection_profiles,
        }
    profiles: dict[str, Any] = {}
    research_profiles: list[str] = []
    for number, profile in enumerate(PAPER_PROFILES, start=1):
        champion = _choose_model(model_reports, profile)
        profile_report: dict[str, Any] = {
            "champion_selected_on_2024": champion,
            "ridge_selection": model_reports["ridge"]["profiles"][profile],
            "xgboost_selection": model_reports["xgboost"]["profiles"][profile],
            "validation": None,
            "validation_gates": None,
            "audit": None,
            "audit_gates": None,
            "research_signal": False,
        }
        if champion is not None:
            chosen = model_reports[champion]["profiles"][profile]["selected"]
            assert chosen is not None
            threshold = float(chosen["threshold_gross_bps"])
            cost = profile_cost_bps(profile)
            validation_rows = _selected(scored[champion]["validation"], cost, threshold)
            validation_summary = _summary(validation_rows, cost)
            validation_gates = {
                "trades_100": validation_summary["metrics"]["trades"] >= 100,
                "expectancy_positive": validation_summary["metrics"]["expectancy_bps"] > 0,
                "pf_1_15": validation_summary["metrics"]["profit_factor"] >= 1.15,
                "drawdown_8pct": validation_summary["metrics"]["max_drawdown"] <= 0.08,
                "expectancy_lcb_positive": bool(
                    np.isfinite(validation_summary["expectancy_lcb_95_bps"])
                    and validation_summary["expectancy_lcb_95_bps"] > 0
                ),
                "majority_positive_active_days": validation_summary["positive_active_days"] > 0.5,
            }
            profile_report["validation"] = validation_summary
            profile_report["validation_gates"] = validation_gates
            if all(validation_gates.values()):
                audit_rows = _selected(scored[champion]["audit"], cost, threshold)
                audit_summary = _summary(audit_rows, cost)
                audit_gates = _audit_gates(audit_summary)
                profile_report["audit"] = audit_summary
                profile_report["audit_gates"] = audit_gates
                profile_report["research_signal"] = all(audit_gates.values())
                if profile_report["research_signal"]:
                    research_profiles.append(profile)
        profiles[profile] = profile_report
        _status("model_validation", 60 + 35 * number / len(PAPER_PROFILES), profile)
    result: dict[str, Any] = {
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": {
            "fit": len(fit),
            "calibration": len(calibration),
            "selection": len(selection),
            "validation": len(validation),
            "audit": len(audit),
        },
        "model_diagnostics": {
            kind: report["selection_diagnostics"] for kind, report in model_reports.items()
        },
        "profiles": profiles,
        "research_profiles": research_profiles,
        "verdict": pass_verdict if research_profiles else fail_verdict,
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    if research_profiles:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "protocol": protocol,
                "protocol_hash": protocol_hash,
                "models": fitted_models,
                "profiles": profiles,
                "features": features,
                "research_only": True,
            },
            bundle_path,
        )
    _atomic_json(report_path, result)
    _status("complete", 100, result["verdict"])
    return result


def run() -> dict[str, Any]:
    _status("model_features", 5, "Preparing causal FT-016 candidate features")
    return _run_model(
        prepare(pd.read_parquet(MATRIX)),
        features=FEATURES,
        protocol=PROTOCOL,
        protocol_hash=PROTOCOL_HASH,
        report_path=REPORT,
        bundle_path=BUNDLE,
        pass_verdict="LOCAL_RESTART_MODEL_RESEARCH_ONLY",
        fail_verdict="NO_LOCAL_RESTART_MODEL",
    )


def run_quality() -> dict[str, Any]:
    _status("quality_features", 5, "Attaching causal restart quality features")
    data = prepare(pd.read_parquet(MATRIX))
    data = attach_restart_quality(data, pd.read_parquet(MINUTES))
    return _run_model(
        data,
        features=QUALITY_FEATURES,
        protocol=QUALITY_PROTOCOL,
        protocol_hash=QUALITY_PROTOCOL_HASH,
        report_path=QUALITY_REPORT,
        bundle_path=QUALITY_BUNDLE,
        pass_verdict="LOCAL_RESTART_QUALITY_MODEL_RESEARCH_ONLY",
        fail_verdict="NO_LOCAL_RESTART_QUALITY_MODEL",
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
