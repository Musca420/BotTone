from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor

from adaptive_bot import musca_btc_moe as base

ROOT = Path("data/ml/musca_btc_auto_moe")
CANDIDATES = ROOT / "candidates.parquet"
LIBRARY = ROOT / "expert_library.joblib"
ACTIONS = ROOT / "expert_actions.parquet"
REPORT = Path("data/reports/musca_btc_auto_moe.json")
STATUS = Path("data/reports/musca_btc_auto_moe.status.json")
BUNDLE = Path("data/models/musca_btc_auto_moe/research_bundle.joblib")

DISCOVERY_FIT_END = pd.Timestamp("2025-10-01T00:00:00Z")
LIBRARY_FREEZE_END = pd.Timestamp("2026-01-01T00:00:00Z")
GATE_FIT_END = pd.Timestamp("2026-03-01T00:00:00Z")
MODEL_AUDIT_END = pd.Timestamp("2026-04-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-05-01T00:00:00Z")
HISTORICAL_AUDIT_END = base.HISTORICAL_AUDIT_END
FUTURE_HOLDOUT_START = base.FUTURE_HOLDOUT_START

TREE_BATCH = 16
MAX_TREES_PER_ACTION = 128
SATURATION_PATIENCE = 2
MIN_FIT_OPPORTUNITIES = 1_000
MIN_VALIDATION_OPPORTUNITIES = 90
MAX_SIGNAL_JACCARD = 0.90
GATE_SEEDS = (20260820, 20260821, 20260822)

EXPERT_META_FEATURES = (
    "side",
    "horizon_fraction",
    "target_1_bps",
    "target_2_bps",
    "stop_bps",
    "trailing_bps",
    "fit_expectancy_bps",
    "validation_expectancy_bps",
    "validation_profit_factor",
    "validation_activation_rate",
    "generator_score_bps",
)
GATE_FEATURES = (*base.GATING_CONTEXT, *EXPERT_META_FEATURES)

PROTOCOL = {
    "name": "musca_btc_automatic_two_stage_mixture_of_experts",
    "parent_protocol_hash": base.PROTOCOL_HASH,
    "symbol": base.SYMBOL,
    "source": base.PROTOCOL["source"],
    "phase_1": {
        "generator": "XGBRFRegressor GPU; every learned leaf is a candidate strategy",
        "horizons_seconds": list(base.HORIZONS),
        "sides": ["LONG", "SHORT"],
        "tree_batch": TREE_BATCH,
        "emergency_tree_ceiling_per_action": MAX_TREES_PER_ACTION,
        "saturation_patience_batches": SATURATION_PATIENCE,
        "final_expert_limit": None,
        "minimum_fit_opportunities": MIN_FIT_OPPORTUNITIES,
        "minimum_validation_opportunities": MIN_VALIDATION_OPPORTUNITIES,
        "maximum_signal_jaccard": MAX_SIGNAL_JACCARD,
        "validation_profit_factor": 1.05,
        "validation_cost_stress": 1.5,
        "positive_validation_months": "at least 2 of 3",
    },
    "phase_2": {
        "inputs": "forward-OOS expert activations, frozen expert metadata and regime context",
        "champion": "Ridge/logistic",
        "challenger": "three-seed XGBoost GPU",
        "challenger_rule": "strictly better MAE, Brier and decision regret",
        "decision": "highest calibrated EV if positive, otherwise neutral FLAT",
    },
    "management": base.PROTOCOL["management"],
    "entry": base.PROTOCOL["entry"],
    "same_5s_bucket": base.PROTOCOL["same_5s_bucket"],
    "round_trip_cost_bps": base.ROUND_TRIP_COST_BPS,
    "risk_per_trade": 0.01,
    "maximum_leverage": 10.0,
    "maximum_positions": 1,
    "chronology": {
        "discovery_fit_end": DISCOVERY_FIT_END.isoformat(),
        "library_freeze_end": LIBRARY_FREEZE_END.isoformat(),
        "gate_fit_end": GATE_FIT_END.isoformat(),
        "model_audit_end": MODEL_AUDIT_END.isoformat(),
        "calibration_end": CALIBRATION_END.isoformat(),
        "historical_audit_end": HISTORICAL_AUDIT_END.isoformat(),
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
    },
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


def _period(rows: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    available = pd.to_datetime(rows["available_at"], utc=True)
    mask = available.lt(end - base.PURGE)
    if start is not None:
        mask &= available.ge(start)
    return rows.loc[mask].copy()


def _expert_id(side: int, horizon: int, tree: int, leaf: int) -> str:
    identity = {
        "protocol_hash": PROTOCOL_HASH,
        "side": side,
        "horizon_seconds": horizon,
        "tree": tree,
        "leaf": leaf,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"btc-{'long' if side > 0 else 'short'}-{horizon}s-{digest}"


def _generator(seed: int) -> XGBRFRegressor:
    return XGBRFRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=MAX_TREES_PER_ACTION,
        max_depth=4,
        min_child_weight=500,
        subsample=0.75,
        colsample_bynode=0.65,
        reg_lambda=20.0,
        n_jobs=4,
        random_state=seed,
    )


def _terminal_net(
    rows: pd.DataFrame,
    side: int,
    horizon: int,
    funding: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    gross = side * rows[f"terminal_{horizon}s_bps"].to_numpy(float)
    exit_timestamp = rows["entry_timestamp"] + pd.Timedelta(seconds=horizon)
    funding_bps = base._funding_pnl_bps(
        rows["entry_timestamp"], exit_timestamp, np.full(len(rows), side), funding
    )
    return np.asarray(gross + funding_bps - base.ROUND_TRIP_COST_BPS, dtype=float)


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses else float("inf")


def _candidate(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    fit_net: np.ndarray,
    validation_net: np.ndarray,
    *,
    side: int,
    horizon: int,
    tree: int,
    leaf: int,
) -> dict[str, Any]:
    fit_values = fit_net[fit_indexes]
    validation_values = validation_net[validation_indexes]
    reasons: list[str] = []
    if len(fit_indexes) < MIN_FIT_OPPORTUNITIES:
        reasons.append("fit_opportunities")
    if len(validation_indexes) < MIN_VALIDATION_OPPORTUNITIES:
        reasons.append("validation_opportunities")
    fit_expectancy = float(fit_values.mean()) if len(fit_values) else float("-inf")
    validation_expectancy = (
        float(validation_values.mean()) if len(validation_values) else float("-inf")
    )
    validation_pf = _profit_factor(validation_values) if len(validation_values) else 0.0
    stress_expectancy = validation_expectancy - 0.5 * base.ROUND_TRIP_COST_BPS
    if fit_expectancy <= 0:
        reasons.append("fit_expectancy")
    if validation_expectancy <= 0:
        reasons.append("validation_expectancy")
    if validation_pf < 1.05:
        reasons.append("validation_profit_factor")
    if stress_expectancy < 0:
        reasons.append("cost_stress_1_5x")

    timestamps = pd.to_datetime(validation.iloc[validation_indexes]["entry_timestamp"], utc=True)
    months = pd.DataFrame(
        {"month": timestamps.dt.to_period("M").astype(str), "net": validation_values}
    )
    monthly = months.groupby("month")["net"].mean() if len(months) else pd.Series(dtype=float)
    positive_months = int(monthly.gt(0).sum())
    if positive_months < 2:
        reasons.append("monthly_stability")

    favorable_name = f"max_{'up' if side > 0 else 'down'}_{horizon}s_bps"
    adverse_name = f"max_{'down' if side > 0 else 'up'}_{horizon}s_bps"
    favorable = fit.iloc[fit_indexes][favorable_name].to_numpy(float)
    adverse = fit.iloc[fit_indexes][adverse_name].to_numpy(float)
    favorable_q50 = float(np.quantile(favorable, 0.50)) if len(favorable) else 0.0
    favorable_q75 = float(np.quantile(favorable, 0.75)) if len(favorable) else 0.0
    adverse_q50 = float(np.quantile(adverse, 0.50)) if len(adverse) else 0.0
    adverse_q75 = float(np.quantile(adverse, 0.75)) if len(adverse) else 0.0
    minimum_target = base.ROUND_TRIP_COST_BPS + base.MINIMUM_NET_TARGET_BPS
    target_1 = float(np.clip(max(favorable_q50, minimum_target), minimum_target, 299.0))
    target_2 = float(np.clip(max(favorable_q75, target_1 + 1.0), target_1 + 1.0, 300.0))
    stop = float(np.clip(adverse_q75, 3.0, base.MAX_STOP_BPS))
    trailing = float(np.clip(adverse_q50, 3.0, stop))
    signature = hashlib.sha256(
        np.packbits(np.isin(np.arange(len(validation)), validation_indexes)).tobytes()
    ).hexdigest()
    stability_penalty = float(monthly.std(ddof=0)) if len(monthly) else 1_000.0
    robust_score = validation_expectancy + 0.25 * fit_expectancy - 0.10 * stability_penalty
    return {
        "expert_id": _expert_id(side, horizon, tree, leaf),
        "side": side,
        "horizon_seconds": horizon,
        "tree_index": tree,
        "leaf_id": leaf,
        "fit_opportunities": len(fit_indexes),
        "validation_opportunities": len(validation_indexes),
        "fit_expectancy_bps": fit_expectancy,
        "validation_expectancy_bps": validation_expectancy,
        "validation_profit_factor": validation_pf,
        "validation_stress_1_5x_bps": stress_expectancy,
        "positive_validation_months": positive_months,
        "validation_activation_rate": len(validation_indexes) / max(1, len(validation)),
        "target_1_bps": target_1,
        "target_2_bps": target_2,
        "stop_bps": stop,
        "trailing_bps": trailing,
        "robust_score": robust_score,
        "signal_signature": signature,
        "accepted_by_economics": not reasons,
        "rejection_reasons": ",".join(reasons),
    }


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 1.0


def _select_diverse(
    candidates: list[dict[str, Any]], signals: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    for candidate in sorted(
        candidates, key=lambda value: float(value["robust_score"]), reverse=True
    ):
        signature = cast(str, candidate["signal_signature"])
        if signature in seen_signatures:
            rejected["duplicate_signal"] += 1
            continue
        peers = [
            expert
            for expert in selected
            if expert["side"] == candidate["side"]
            and expert["horizon_seconds"] == candidate["horizon_seconds"]
        ]
        if any(
            _jaccard(
                signals[cast(str, peer["expert_id"])], signals[cast(str, candidate["expert_id"])]
            )
            >= MAX_SIGNAL_JACCARD
            for peer in peers
        ):
            rejected["correlated_signal"] += 1
            continue
        selected.append(candidate)
        seen_signatures.add(signature)
    return selected, rejected


def discover_library(matrix: pd.DataFrame, *, force: bool = False) -> dict[str, Any]:
    if LIBRARY.exists() and not force:
        cached = joblib.load(LIBRARY)
        if cached.get("protocol_hash") == PROTOCOL_HASH:
            return cast(dict[str, Any], cached)

    fit = _period(matrix, None, DISCOVERY_FIT_END)
    validation = _period(matrix, DISCOVERY_FIT_END, LIBRARY_FREEZE_END)
    funding = base._funding_curve()
    x_fit = fit.loc[:, base.FEATURES].to_numpy(np.float32)
    x_validation = validation.loc[:, base.FEATURES].to_numpy(np.float32)
    all_candidates: list[dict[str, Any]] = []
    signals: dict[str, np.ndarray] = {}
    generators: dict[str, Any] = {}
    saturation: dict[str, Any] = {}
    actions = [(side, horizon) for horizon in base.HORIZONS for side in base.SIDES]

    for action_number, (side, horizon) in enumerate(actions, start=1):
        key = f"{side}:{horizon}"
        fit_net = _terminal_net(fit, side, horizon, funding)
        validation_net = _terminal_net(validation, side, horizon, funding)
        model = _generator(20260810 + action_number).fit(x_fit, fit_net, verbose=False)
        fit_leaves = np.asarray(model.apply(x_fit), dtype=np.int32)
        validation_leaves = np.asarray(model.apply(x_validation), dtype=np.int32)
        generators[key] = model
        empty_batches = 0
        trees_used = 0
        action_candidates: list[dict[str, Any]] = []
        action_signals: dict[str, np.ndarray] = {}
        for start in range(0, MAX_TREES_PER_ACTION, TREE_BATCH):
            batch_new = 0
            for tree in range(start, min(start + TREE_BATCH, fit_leaves.shape[1])):
                leaves = np.union1d(
                    np.unique(fit_leaves[:, tree]), np.unique(validation_leaves[:, tree])
                )
                for leaf in leaves:
                    fit_indexes = np.flatnonzero(fit_leaves[:, tree] == leaf)
                    validation_indexes = np.flatnonzero(validation_leaves[:, tree] == leaf)
                    candidate = _candidate(
                        fit,
                        validation,
                        fit_indexes,
                        validation_indexes,
                        fit_net,
                        validation_net,
                        side=side,
                        horizon=horizon,
                        tree=tree,
                        leaf=int(leaf),
                    )
                    all_candidates.append(candidate)
                    if not candidate["accepted_by_economics"]:
                        continue
                    action_candidates.append(candidate)
                    signal = validation_leaves[:, tree] == leaf
                    action_signals[cast(str, candidate["expert_id"])] = signal
                    batch_new += 1
            trees_used = min(start + TREE_BATCH, fit_leaves.shape[1])
            if batch_new == 0:
                empty_batches += 1
                if empty_batches >= SATURATION_PATIENCE:
                    break
            else:
                empty_batches = 0
        selected, _ = _select_diverse(action_candidates, action_signals)
        signals.update(action_signals)
        saturation[key] = {
            "trees_generated": MAX_TREES_PER_ACTION,
            "trees_evaluated": trees_used,
            "economically_valid_leaves": len(action_candidates),
            "selected_after_diversity": len(selected),
            "stopped_by_saturation": trees_used < MAX_TREES_PER_ACTION,
        }
        _status(
            "expert_discovery",
            f"azione {action_number}/{len(actions)}: {len(selected)} esperti {key}",
            5 + 45 * action_number / len(actions),
        )

    economically_valid = [item for item in all_candidates if item["accepted_by_economics"]]
    selected, diversity_rejections = _select_diverse(economically_valid, signals)
    selected_ids = {item["expert_id"] for item in selected}
    for item in all_candidates:
        item["selected"] = item["expert_id"] in selected_ids
    catalog = pd.DataFrame(all_candidates)
    catalog["protocol_hash"] = PROTOCOL_HASH
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANDIDATES.with_suffix(".parquet.tmp")
    catalog.to_parquet(temporary, index=False)
    temporary.replace(CANDIDATES)
    payload = {
        "protocol_hash": PROTOCOL_HASH,
        "experts": selected,
        "generators": generators,
        "candidate_count": len(all_candidates),
        "economically_valid_count": len(economically_valid),
        "diversity_rejections": dict(diversity_rejections),
        "saturation": saturation,
    }
    _atomic_joblib(LIBRARY, payload)
    return payload


def _expert_action_rows(
    rows: pd.DataFrame, library: dict[str, Any], source: pd.DataFrame
) -> pd.DataFrame:
    if not library["experts"]:
        return pd.DataFrame()
    funding = base._funding_curve()
    x = rows.loc[:, base.FEATURES].to_numpy(np.float32)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for expert in library["experts"]:
        grouped[f"{expert['side']}:{expert['horizon_seconds']}"].append(expert)
    pieces: list[pd.DataFrame] = []
    for key, experts in grouped.items():
        model = library["generators"][key]
        leaves = np.asarray(model.apply(x), dtype=np.int32)
        generator_score = np.asarray(model.predict(x), dtype=np.float32)
        for expert in experts:
            indexes = np.flatnonzero(leaves[:, int(expert["tree_index"])] == int(expert["leaf_id"]))
            if not len(indexes):
                continue
            active = rows.iloc[indexes].copy()
            side = int(expert["side"])
            horizon = int(expert["horizon_seconds"])
            gross, exit_seconds, outcome = base._simulate_management(
                source,
                active["decision_position"].to_numpy(int),
                side,
                horizon,
                np.full(len(active), expert["target_1_bps"], dtype=float),
                np.full(len(active), expert["target_2_bps"], dtype=float),
                np.full(len(active), expert["stop_bps"], dtype=float),
                np.full(len(active), expert["trailing_bps"], dtype=float),
            )
            action = active.loc[:, ["available_at", "entry_timestamp", "decision_position"]].copy()
            for feature in base.GATING_CONTEXT:
                values = active[feature].to_numpy(np.float32)
                action[feature] = values * side if feature in base.DIRECTIONAL_FEATURES else values
            action["expert_id"] = expert["expert_id"]
            action["side"] = float(side)
            action["horizon_fraction"] = horizon / max(base.HORIZONS)
            action["horizon_seconds"] = horizon
            for name in (
                "target_1_bps",
                "target_2_bps",
                "stop_bps",
                "trailing_bps",
                "fit_expectancy_bps",
                "validation_expectancy_bps",
                "validation_profit_factor",
                "validation_activation_rate",
            ):
                action[name] = float(expert[name])
            action["generator_score_bps"] = generator_score[indexes]
            action["gross_bps"] = gross
            action["exit_seconds"] = exit_seconds
            action["outcome"] = outcome
            action["exit_timestamp"] = action["entry_timestamp"] + pd.to_timedelta(
                exit_seconds, unit="s"
            )
            action["funding_bps"] = base._funding_pnl_bps(
                action["entry_timestamp"],
                action["exit_timestamp"],
                np.full(len(action), side),
                funding,
            )
            action["net_bps"] = (
                action["gross_bps"] + action["funding_bps"] - base.ROUND_TRIP_COST_BPS
            )
            action["stress_1_5x_bps"] = (
                action["gross_bps"] + action["funding_bps"] - 1.5 * base.ROUND_TRIP_COST_BPS
            )
            action["stress_2x_bps"] = (
                action["gross_bps"] + action["funding_bps"] - 2 * base.ROUND_TRIP_COST_BPS
            )
            pieces.append(action)
    if not pieces:
        return pd.DataFrame()
    output = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["entry_timestamp", "expert_id"])
        .reset_index(drop=True)
    )
    output["protocol_hash"] = PROTOCOL_HASH
    return output


def build_actions(
    matrix: pd.DataFrame, library: dict[str, Any], *, force: bool = False
) -> pd.DataFrame:
    if ACTIONS.exists() and not force:
        protocol = pd.read_parquet(ACTIONS, columns=["protocol_hash"])
        if not protocol.empty and protocol["protocol_hash"].eq(PROTOCOL_HASH).all():
            return pd.read_parquet(ACTIONS)
    rows = _period(matrix, LIBRARY_FREEZE_END, HISTORICAL_AUDIT_END)
    source = base._load_micro_source()
    actions = _expert_action_rows(rows, library, source)
    if not actions.empty:
        temporary = ACTIONS.with_suffix(".parquet.tmp")
        actions.to_parquet(temporary, index=False)
        temporary.replace(ACTIONS)
    return actions


def _gate_x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, GATE_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("gating features must be finite")
    return values


def _timestamp_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("entry_timestamp")["entry_timestamp"].transform("size").to_numpy(float)
    return np.asarray(1.0 / counts, dtype=float)


def _fit_gate(rows: pd.DataFrame) -> dict[str, Any]:
    x = _gate_x(rows)
    y = rows["net_bps"].to_numpy(np.float32)
    positive = (y > 0).astype(int)
    weights = _timestamp_weights(rows)
    ridge_reg = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(
        x, y, ridge__sample_weight=weights
    )
    ridge_cls = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2_000, random_state=20260820)
    ).fit(x, positive, logisticregression__sample_weight=weights)
    xgb_reg: list[Any] = []
    xgb_cls: list[Any] = []
    for number, seed in enumerate(GATE_SEEDS, start=1):
        xgb_reg.append(
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                n_estimators=220,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=30.0,
                n_jobs=4,
                random_state=seed,
            ).fit(x, y, sample_weight=weights, verbose=False)
        )
        xgb_cls.append(
            XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                device="cuda",
                n_estimators=220,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=30.0,
                n_jobs=4,
                random_state=seed,
            ).fit(x, positive, sample_weight=weights, verbose=False)
        )
        _status("gating", f"XGBoost challenger {number}/{len(GATE_SEEDS)}", 76 + 3 * number)
    return {
        "ridge": {"regressors": [ridge_reg], "classifiers": [ridge_cls]},
        "xgboost": {"regressors": xgb_reg, "classifiers": xgb_cls},
    }


def _raw_gate(rows: pd.DataFrame, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = _gate_x(rows)
    ev = np.vstack([np.asarray(item.predict(x), dtype=float) for item in model["regressors"]])
    probability = np.vstack(
        [np.asarray(item.predict_proba(x)[:, 1], dtype=float) for item in model["classifiers"]]
    )
    return ev.mean(axis=0), probability.mean(axis=0)


def _decision_regret(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    values = rows.loc[:, ["entry_timestamp", "net_bps"]].copy()
    values["prediction"] = prediction
    regrets: list[float] = []
    for _, group in values.groupby("entry_timestamp", sort=False):
        predicted = group["prediction"].to_numpy(float)
        actual = group["net_bps"].to_numpy(float)
        choice = int(np.argmax(predicted))
        chosen = actual[choice] if predicted[choice] > 0 else 0.0
        regrets.append(max(0.0, float(actual.max())) - chosen)
    return float(np.mean(regrets))


def _gate_metrics(rows: pd.DataFrame, model: dict[str, Any]) -> dict[str, float]:
    ev, probability = _raw_gate(rows, model)
    actual = rows["net_bps"].to_numpy(float)
    return {
        "mae_bps": float(mean_absolute_error(actual, ev)),
        "brier": float(brier_score_loss(actual > 0, probability)),
        "decision_regret_bps": _decision_regret(rows, ev),
    }


def _fit_calibrators(rows: pd.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    ev, probability = _raw_gate(rows, model)
    actual = rows["net_bps"].to_numpy(float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return {
        "ev": IsotonicRegression(out_of_bounds="clip").fit(ev, actual),
        "probability": LogisticRegression(C=1.0, max_iter=2_000).fit(
            logit, (actual > 0).astype(int)
        ),
    }


def _score(rows: pd.DataFrame, model: dict[str, Any], calibrators: dict[str, Any]) -> pd.DataFrame:
    ev, probability = _raw_gate(rows, model)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    output = rows.copy()
    output["raw_ev_bps"] = ev
    output["calibrated_ev_bps"] = calibrators["ev"].predict(ev)
    output["probability_net_positive"] = calibrators["probability"].predict_proba(logit)[:, 1]
    return output


def _execute(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    winners = (
        scored.sort_values(
            ["entry_timestamp", "calibrated_ev_bps", "raw_ev_bps", "expert_id"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("entry_timestamp", keep="first")
        .loc[lambda value: value["calibrated_ev_bps"].gt(0)]
    )
    accepted: list[int] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    for index, row in winners.iterrows():
        if pd.Timestamp(row["entry_timestamp"]) < free_at:
            continue
        accepted.append(cast(int, index))
        free_at = pd.Timestamp(row["exit_timestamp"])
    return winners.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def _library_spa(actions: pd.DataFrame) -> float | None:
    if actions.empty:
        return None
    pivot = actions.pivot_table(
        index=pd.to_datetime(actions["entry_timestamp"], utc=True).dt.floor("D"),
        columns="expert_id",
        values="net_bps",
        aggfunc="sum",
        fill_value=0.0,
    )
    if len(pivot) < 10 or not len(pivot.columns):
        return None
    test = SPA(
        np.zeros(len(pivot)),
        -pivot.to_numpy(float),
        block_size=min(5, len(pivot)),
        reps=1_000,
        bootstrap="stationary",
        seed=20260820,
    )
    test.compute()
    return float(test.pvalues["consistent"])


def train(*, force: bool = False) -> dict[str, Any]:
    _status("start", "BTC automatic expert discovery", 0)
    matrix = base.build_matrix(force=False)
    library = discover_library(matrix, force=force)
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": base.SYMBOL,
        "parent_protocol_hash": base.PROTOCOL_HASH,
        "matrix_rows": len(matrix),
        "candidate_experts_evaluated": int(library["candidate_count"]),
        "economically_valid_before_diversity": int(library["economically_valid_count"]),
        "frozen_expert_count": len(library["experts"]),
        "saturation": library["saturation"],
        "future_holdout_rows_read": int(
            pd.to_datetime(matrix["available_at"], utc=True).ge(FUTURE_HOLDOUT_START).sum()
        ),
        "research_only": True,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "real_capital_allowed": False,
    }
    if not library["experts"]:
        report["verdict"] = "NO_DISCOVERED_EXPERT_LIBRARY"
        _atomic_json(REPORT, report)
        _atomic_joblib(
            BUNDLE,
            {"protocol": PROTOCOL, "protocol_hash": PROTOCOL_HASH, "orders_enabled": False},
        )
        _status("complete", cast(str, report["verdict"]), 100)
        return report

    _status("expert_replay", f"{len(library['experts'])} esperti congelati", 55)
    actions = build_actions(matrix, library, force=force)
    if actions.empty:
        report["verdict"] = "NO_FORWARD_EXPERT_ACTIONS"
        _atomic_json(REPORT, report)
        _status("complete", cast(str, report["verdict"]), 100)
        return report
    timestamp = pd.to_datetime(actions["entry_timestamp"], utc=True)
    gate_fit = actions.loc[timestamp.lt(GATE_FIT_END - base.PURGE)].copy()
    model_audit = actions.loc[
        timestamp.ge(GATE_FIT_END) & timestamp.lt(MODEL_AUDIT_END - base.PURGE)
    ].copy()
    calibration = actions.loc[
        timestamp.ge(MODEL_AUDIT_END) & timestamp.lt(CALIBRATION_END - base.PURGE)
    ].copy()
    historical_audit = actions.loc[
        timestamp.ge(CALIBRATION_END) & timestamp.lt(HISTORICAL_AUDIT_END - base.PURGE)
    ].copy()
    models = _fit_gate(gate_fit)
    candidate_metrics = {
        name: _gate_metrics(model_audit, models[name]) for name in ("ridge", "xgboost")
    }
    ridge = candidate_metrics["ridge"]
    xgb = candidate_metrics["xgboost"]
    champion = (
        "xgboost"
        if all(
            float(xgb[key]) < float(ridge[key])
            for key in ("mae_bps", "brier", "decision_regret_bps")
        )
        else "ridge"
    )
    calibrators = _fit_calibrators(calibration, models[champion])
    audit_scored = _score(historical_audit, models[champion], calibrators)
    trades = _execute(audit_scored)
    metrics = base._metrics(trades, CALIBRATION_END, HISTORICAL_AUDIT_END)
    gates = base._audit_gates(metrics)
    historical_pass = all(gates.values())
    report.update(
        {
            "forward_action_rows": len(actions),
            "gate_fit_rows": len(gate_fit),
            "model_audit_rows": len(model_audit),
            "calibration_rows": len(calibration),
            "historical_audit_rows": len(historical_audit),
            "library_spa_pvalue": _library_spa(actions.loc[timestamp.lt(GATE_FIT_END)]),
            "candidate_metrics": candidate_metrics,
            "gating_champion": champion,
            "historical_audit": {"metrics": metrics, "gates": gates},
            "verdict": (
                "HISTORICAL_ALPHA_READY_FOR_FUTURE_HOLDOUT"
                if historical_pass
                else "NO_DEPLOYABLE_POLICY"
            ),
        }
    )
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "expert_library": library,
        "meta_model": models[champion],
        "calibrators": calibrators,
        "gating_champion": champion,
        "historical_pass": historical_pass,
        "research_only": True,
        "orders_enabled": False,
    }
    _atomic_joblib(BUNDLE, bundle)
    _atomic_json(REPORT, report)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC automatic two-stage expert training")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(train(force=arguments.force), indent=2, default=str))


if __name__ == "__main__":
    main()
