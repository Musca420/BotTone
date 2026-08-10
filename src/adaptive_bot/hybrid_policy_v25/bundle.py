from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import joblib
import pandas as pd

from adaptive_bot.hybrid_policy_v22.protocol import sha256
from adaptive_bot.hybrid_policy_v25.models import (
    _apply_calibrator,
    _baseline_pair,
    _calibrate_probability,
    _choose_threshold,
    _deterministic_predictions,
    _fit_pair,
    _pairs,
    _predict,
    _quantile_models,
)
from adaptive_bot.hybrid_policy_v25.protocol import (
    BUNDLE_ROOT,
    FEATURE_COLUMNS,
    HOLDOUT_WEEKS,
    PROTOCOL_PATH,
    atomic_json,
)

BUNDLE_PATH = BUNDLE_ROOT / "bundle.joblib"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"


class PaperState(StrEnum):
    IDLE = "IDLE"
    REGIME_ACTIVE = "REGIME_ACTIVE"
    IMPULSE_DETECTED = "IMPULSE_DETECTED"
    PULLBACK_TRACKING = "PULLBACK_TRACKING"
    CANDIDATE_READY = "CANDIDATE_READY"
    MODEL_APPROVED = "MODEL_APPROVED"
    ORDER_PENDING = "ORDER_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    POSITION_CLOSED = "POSITION_CLOSED"
    COOLDOWN = "COOLDOWN"
    HALTED = "HALTED"


ALLOWED_TRANSITIONS = {
    PaperState.IDLE: {PaperState.REGIME_ACTIVE, PaperState.HALTED},
    PaperState.REGIME_ACTIVE: {PaperState.IMPULSE_DETECTED, PaperState.IDLE, PaperState.HALTED},
    PaperState.IMPULSE_DETECTED: {PaperState.PULLBACK_TRACKING, PaperState.IDLE, PaperState.HALTED},
    PaperState.PULLBACK_TRACKING: {PaperState.CANDIDATE_READY, PaperState.IDLE, PaperState.HALTED},
    PaperState.CANDIDATE_READY: {PaperState.MODEL_APPROVED, PaperState.COOLDOWN, PaperState.HALTED},
    PaperState.MODEL_APPROVED: {PaperState.ORDER_PENDING, PaperState.COOLDOWN, PaperState.HALTED},
    PaperState.ORDER_PENDING: {PaperState.POSITION_OPEN, PaperState.COOLDOWN, PaperState.HALTED},
    PaperState.POSITION_OPEN: {
        PaperState.PARTIAL_EXIT,
        PaperState.POSITION_CLOSED,
        PaperState.HALTED,
    },
    PaperState.PARTIAL_EXIT: {PaperState.POSITION_CLOSED, PaperState.HALTED},
    PaperState.POSITION_CLOSED: {PaperState.COOLDOWN, PaperState.HALTED},
    PaperState.COOLDOWN: {PaperState.IDLE, PaperState.HALTED},
    PaperState.HALTED: set(),
}


@dataclass
class PaperMachine:
    state: PaperState = PaperState.IDLE
    halted_reason: str | None = None

    def transition(self, target: PaperState) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid V25 transition: {self.state} -> {target}")
        self.state = target

    def halt(self, reason: str) -> None:
        self.state = PaperState.HALTED
        self.halted_reason = reason


def _schema_hash() -> str:
    return hashlib.sha256(json.dumps(FEATURE_COLUMNS).encode()).hexdigest()


def _fit_final(outcomes: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    times = pd.to_datetime(outcomes["entry_timestamp"], utc=True)
    btc_end = times.loc[outcomes["asset"].eq("BTCUSDT")].max()
    holdout_start = btc_end - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    preholdout = outcomes.loc[times.lt(holdout_start)].copy()
    pre_times = pd.to_datetime(preholdout["entry_timestamp"], utc=True)
    calibration_start = holdout_start - pd.Timedelta(weeks=8)
    train = preholdout.loc[pre_times.lt(calibration_start)].copy()
    calibration = preholdout.loc[pre_times.ge(calibration_start)].copy()
    model_name = max(
        audit.get("chosen_models", {"deterministic": 1}),
        key=audit.get("chosen_models", {"deterministic": 1}).get,
    )
    variants = [fold.get("variant", "base") for fold in audit.get("folds", [])]
    variant = max(set(variants), key=variants.count) if variants else "base"
    target = "net_return_bps_4" if variant == "base" else "flow_variant_net_return_bps_4"
    if variant == "flow":
        train = train.loc[train["flow_gate_pass"]]
        calibration = calibration.loc[calibration["flow_gate_pass"]]
    quantiles = _quantile_models(train, target)
    if model_name == "deterministic":
        fitted = None
        cal_expected, cal_probability = _deterministic_predictions(calibration)
    else:
        template = next((pair for pair in _pairs() if pair.name == model_name), _baseline_pair())
        fitted = _fit_pair(template, train, target)
        cal_expected, cal_probability = _predict(fitted, calibration)
    observed = calibration[target].to_numpy(float)
    calibrator, calibrated, calibration_method = _calibrate_probability(
        cal_probability, (observed > 0).astype(int)
    )
    bias = float((observed - cal_expected).mean())
    calibration_frame = calibration.copy()
    calibration_frame["expected_net_bps"] = cal_expected + bias
    calibration_frame["calibrated_probability"] = calibrated
    calibration_frame["q25_net_bps"] = quantiles[0.25].predict(calibration.loc[:, FEATURE_COLUMNS])
    calibration_frame["observed_net_bps"] = observed
    threshold, threshold_trials = _choose_threshold(calibration_frame)
    return {
        "model_name": model_name,
        "variant": variant,
        "target": target,
        "model_pair": fitted,
        "quantile_models": quantiles,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "regression_bias_bps": bias,
        "probability_threshold": threshold,
        "threshold_trials": threshold_trials,
        "training_end": calibration_start.isoformat(),
        "calibration_end": holdout_start.isoformat(),
    }


def build_bundle(
    outcomes: pd.DataFrame, protocol: dict[str, Any], model_audit: dict[str, Any]
) -> dict[str, Any]:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    fitted = _fit_final(outcomes, model_audit)
    payload = {
        "bundle_type": "RESEARCH_ONLY",
        "allowed_modes": ["research", "paper", "shadow"],
        "real_capital_allowed": False,
        "auto_promotion": False,
        "protocol_hash": protocol["protocol_hash"],
        "feature_schema_hash": _schema_hash(),
        "feature_columns": FEATURE_COLUMNS,
        "risk": {
            "risk_per_trade": 0.0025,
            "max_open_positions": 2,
            "max_same_direction_positions": 1,
            "daily_loss_limit": 0.01,
            "weekly_loss_limit": 0.03,
            "max_drawdown_kill": 0.08,
        },
        "drift": {"psi_halt": 0.25, "mandatory_missing_halt": 0.01},
        "model_audit_verdict": model_audit["verdict"],
        **fitted,
    }
    joblib.dump(payload, BUNDLE_PATH)
    manifest = {
        "bundle_type": "RESEARCH_ONLY",
        "bundle_sha256": sha256(BUNDLE_PATH),
        "protocol_hash": protocol["protocol_hash"],
        "feature_schema_hash": _schema_hash(),
        "model_hash": hashlib.sha256(joblib.hash(fitted).encode()).hexdigest(),
        "training_end": fitted["training_end"],
        "allowed_modes": payload["allowed_modes"],
        "real_capital_allowed": False,
        "auto_promotion": False,
    }
    atomic_json(MANIFEST_PATH, manifest)
    return manifest


def load_bundle(mode: str, *, feature_columns: tuple[str, ...] = FEATURE_COLUMNS) -> dict[str, Any]:
    if mode not in {"research", "paper", "shadow"}:
        raise PermissionError("V25 RESEARCH_ONLY bundle refuses live capital")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["bundle_sha256"] != sha256(BUNDLE_PATH):
        raise RuntimeError("V25 bundle hash mismatch")
    if (
        hashlib.sha256(json.dumps(feature_columns).encode()).hexdigest()
        != manifest["feature_schema_hash"]
    ):
        raise RuntimeError("V25 feature schema mismatch")
    frozen_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if frozen_protocol["protocol_hash"] != manifest["protocol_hash"]:
        raise RuntimeError("V25 protocol manifest mismatch")
    bundle = cast(dict[str, Any], joblib.load(BUNDLE_PATH))
    if mode not in bundle["allowed_modes"] or bundle["real_capital_allowed"]:
        raise PermissionError("V25 manifest does not authorize this mode")
    return bundle


def score_candidate(bundle: dict[str, Any], candidate: pd.DataFrame) -> dict[str, Any]:
    if not set(FEATURE_COLUMNS).issubset(candidate.columns):
        raise ValueError("V25 candidate is missing mandatory features")
    features = candidate.loc[:, list(FEATURE_COLUMNS)].copy()
    if bundle["model_name"] == "deterministic":
        expected, raw_probability = _deterministic_predictions(features)
    else:
        expected, raw_probability = _predict(bundle["model_pair"], features)
    probability = _apply_calibrator(
        bundle["calibrator"], bundle["calibration_method"], raw_probability
    )
    quantiles = {
        quantile: float(model.predict(features)[0])
        for quantile, model in bundle["quantile_models"].items()
    }
    expected_value = float(expected[0] + bundle["regression_bias_bps"])
    approved = (
        expected_value >= 8
        and float(probability[0]) >= bundle["probability_threshold"]
        and quantiles[0.25] >= -2
        and float(candidate.iloc[0]["cost_to_stop_ratio"]) <= 0.33
    )
    return {
        "decision": "MODEL_APPROVED" if approved else "NO_TRADE",
        "expected_net_bps": expected_value,
        "calibrated_probability": float(probability[0]),
        "q25_net_bps": quantiles[0.25],
        "q50_net_bps": quantiles[0.5],
        "q75_net_bps": quantiles[0.75],
    }
