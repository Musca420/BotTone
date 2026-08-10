from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
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

ROOT = Path(f"data/ml/musca_{base.ASSET_SLUG}_auto_moe")
CANDIDATES = ROOT / "candidates.parquet"
LIBRARY = ROOT / "expert_library.joblib"
ACTIONS = ROOT / "expert_actions.parquet"
AUDIT_TRADES = ROOT / "audit_trades.parquet"
REPORT = Path(f"data/reports/musca_{base.ASSET_SLUG}_auto_moe.json")
STATUS = Path(f"data/reports/musca_{base.ASSET_SLUG}_auto_moe.status.json")
BUNDLE = Path(f"data/models/musca_{base.ASSET_SLUG}_auto_moe/research_bundle.joblib")

DISCOVERY_FIT_END = pd.Timestamp("2026-01-01T00:00:00Z")
LIBRARY_FREEZE_END = pd.Timestamp("2026-03-01T00:00:00Z")
GATE_TUNE_END = pd.Timestamp("2026-04-01T00:00:00Z")
GATE_FIT_END = pd.Timestamp("2026-05-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-06-01T00:00:00Z")
HISTORICAL_AUDIT_END = base.HISTORICAL_AUDIT_END
FUTURE_HOLDOUT_START = base.FUTURE_HOLDOUT_START

TREE_BATCH = 16
MAX_TREES_PER_ACTION = 128
SATURATION_PATIENCE = 2
MIN_FIT_OPPORTUNITIES = 1_000
MIN_VALIDATION_OPPORTUNITIES = 90
MAX_SIGNAL_JACCARD = 0.90
GATE_SEEDS = (20260820, 20260821, 20260822)
EXPERT_VALIDATION_PROFIT_FACTOR = 1.02
MAX_PROFIT_FACTOR_FEATURE = 100.0
LIVE_OFFICIAL_DERIVATIVE_FEATURES = frozenset(
    {"basis_bps", "funding_z"}
    | (
        {"oi_change_1h", "return_oi_interaction_raw"}
        if base.REQUIRE_OPEN_INTEREST
        else set()
    )
)
MODEL_FEATURES = base.FEATURES
GATE_CONTEXT = base.GATING_CONTEXT

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
GATE_FEATURES = (*GATE_CONTEXT, *EXPERT_META_FEATURES)

PROTOCOL = {
    "name": f"musca_{base.ASSET_SLUG}_automatic_two_stage_mixture_of_experts",
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
        "model_features": list(MODEL_FEATURES),
        "live_official_derivative_features": sorted(LIVE_OFFICIAL_DERIVATIVE_FEATURES),
        "live_feature_sources": {
            "basis_bps": "closed GET /fapi/v1/markPriceKlines / spot api/v3/klines",
            "funding_z": "GET /fapi/v1/fundingRate reconstructed on the minute grid",
        }
        | (
            {"oi_change_1h": "GET /futures/data/openInterestHist period=5m"}
            if base.REQUIRE_OPEN_INTEREST
            else {}
        ),
        "validation_profit_factor": EXPERT_VALIDATION_PROFIT_FACTOR,
        "expert_selection_target": "managed net PnL using the production exit rules",
        "validation_cost_stress": "reported only; applied to the combined policy audit",
        "positive_validation_months": "both January and February using managed net PnL",
    },
    "phase_2": {
        "inputs": "forward-OOS expert activations, frozen expert metadata and regime context",
        "champion": "Ridge/logistic",
        "challenger": "three-seed XGBoost GPU",
        "challenger_rule": "strictly better MAE, Brier and decision regret",
        "decision": "highest calibrated EV if positive, otherwise neutral FLAT",
        "adaptation": "monthly prequential refit; prior outcomes only",
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
        "gate_tune_end": GATE_TUNE_END.isoformat(),
        "gate_fit_end": GATE_FIT_END.isoformat(),
        "calibration_end": CALIBRATION_END.isoformat(),
        "historical_audit_end": HISTORICAL_AUDIT_END.isoformat(),
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
    },
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
LEGACY_DISCOVERY_PROTOCOL_HASHES = frozenset(
    {"57167b3200409981efb2cbd73133e07d6e8d0f8f5cc65864104d41f59637c7af"}
    if base.SYMBOL == "DOGEUSDT"
    else set()
)


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
    with suppress(BrokenPipeError, OSError):
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
    return f"{base.ASSET_SLUG}-{'long' if side > 0 else 'short'}-{horizon}s-{digest}"


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


def _managed_net(
    rows: pd.DataFrame,
    indexes: np.ndarray,
    source: pd.DataFrame,
    funding: tuple[np.ndarray, np.ndarray],
    *,
    side: int,
    horizon: int,
    target_1: float,
    target_2: float,
    stop: float,
    trailing: float,
) -> np.ndarray:
    if not len(indexes):
        return np.empty(0, dtype=float)
    active = rows.iloc[indexes]
    gross, exit_seconds, _ = base._simulate_management(
        source,
        active["decision_position"].to_numpy(int),
        side,
        horizon,
        np.full(len(active), target_1, dtype=float),
        np.full(len(active), target_2, dtype=float),
        np.full(len(active), stop, dtype=float),
        np.full(len(active), trailing, dtype=float),
    )
    exit_timestamp = active["entry_timestamp"] + pd.to_timedelta(exit_seconds, unit="s")
    funding_bps = base._funding_pnl_bps(
        active["entry_timestamp"],
        exit_timestamp,
        np.full(len(active), side),
        funding,
    )
    return np.asarray(gross + funding_bps - base.ROUND_TRIP_COST_BPS, dtype=float)


def _candidate(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    fit_net: np.ndarray,
    validation_net: np.ndarray,
    source: pd.DataFrame,
    funding: tuple[np.ndarray, np.ndarray],
    *,
    side: int,
    horizon: int,
    tree: int,
    leaf: int,
) -> dict[str, Any]:
    terminal_fit_values = fit_net[fit_indexes]
    terminal_validation_values = validation_net[validation_indexes]
    reasons: list[str] = []
    if len(fit_indexes) < MIN_FIT_OPPORTUNITIES:
        reasons.append("fit_opportunities")
    if len(validation_indexes) < MIN_VALIDATION_OPPORTUNITIES:
        reasons.append("validation_opportunities")
    terminal_fit_expectancy = (
        float(terminal_fit_values.mean()) if len(terminal_fit_values) else float("-inf")
    )
    terminal_validation_expectancy = (
        float(terminal_validation_values.mean())
        if len(terminal_validation_values)
        else float("-inf")
    )
    terminal_validation_pf = (
        min(_profit_factor(terminal_validation_values), MAX_PROFIT_FACTOR_FEATURE)
        if len(terminal_validation_values)
        else 0.0
    )
    if terminal_fit_expectancy <= 0:
        reasons.append("fit_expectancy")
    if terminal_validation_expectancy <= 0:
        reasons.append("validation_expectancy")
    if terminal_validation_pf < EXPERT_VALIDATION_PROFIT_FACTOR:
        reasons.append("validation_profit_factor")

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
    fit_values = np.empty(0, dtype=float)
    validation_values = np.empty(0, dtype=float)
    managed_evaluated = not reasons
    if managed_evaluated:
        fit_values = _managed_net(
            fit,
            fit_indexes,
            source,
            funding,
            side=side,
            horizon=horizon,
            target_1=target_1,
            target_2=target_2,
            stop=stop,
            trailing=trailing,
        )
        validation_values = _managed_net(
            validation,
            validation_indexes,
            source,
            funding,
            side=side,
            horizon=horizon,
            target_1=target_1,
            target_2=target_2,
            stop=stop,
            trailing=trailing,
        )
    fit_expectancy = float(fit_values.mean()) if len(fit_values) else float("-inf")
    validation_expectancy = (
        float(validation_values.mean()) if len(validation_values) else float("-inf")
    )
    validation_pf = (
        min(_profit_factor(validation_values), MAX_PROFIT_FACTOR_FEATURE)
        if len(validation_values)
        else 0.0
    )
    if managed_evaluated and fit_expectancy <= 0:
        reasons.append("managed_fit_expectancy")
    if managed_evaluated and validation_expectancy <= 0:
        reasons.append("managed_validation_expectancy")
    if managed_evaluated and validation_pf < EXPERT_VALIDATION_PROFIT_FACTOR:
        reasons.append("managed_validation_profit_factor")
    stress_expectancy = validation_expectancy - 0.5 * base.ROUND_TRIP_COST_BPS
    timestamps = pd.to_datetime(validation.iloc[validation_indexes]["entry_timestamp"], utc=True)
    months = (
        pd.DataFrame(
            {"month": timestamps.dt.strftime("%Y-%m").to_numpy(), "net": validation_values}
        )
        if len(validation_values)
        else pd.DataFrame(columns=["month", "net"])
    )
    monthly = months.groupby("month")["net"].mean() if len(months) else pd.Series(dtype=float)
    positive_months = int(monthly.gt(0).sum())
    if managed_evaluated and positive_months < 2:
        reasons.append("managed_monthly_stability")
    signal = np.zeros(len(validation), dtype=bool)
    signal[validation_indexes] = True
    signature = hashlib.sha256(np.packbits(signal).tobytes()).hexdigest()
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
        "terminal_fit_expectancy_bps": terminal_fit_expectancy,
        "terminal_validation_expectancy_bps": terminal_validation_expectancy,
        "terminal_validation_profit_factor": terminal_validation_pf,
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


def _packed_jaccard(
    left: np.ndarray, right: np.ndarray, left_count: int, right_count: int
) -> float:
    intersection = int(np.bitwise_count(np.bitwise_and(left, right)).sum())
    union = left_count + right_count - intersection
    return intersection / union if union else 1.0


def _select_diverse(
    candidates: list[dict[str, Any]], signals: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    selected_by_action: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    packed = {key: np.packbits(signal) for key, signal in signals.items()}
    counts = {key: int(signal.sum()) for key, signal in signals.items()}
    for candidate in sorted(
        candidates, key=lambda value: float(value["robust_score"]), reverse=True
    ):
        signature = cast(str, candidate["signal_signature"])
        expert_id = cast(str, candidate["expert_id"])
        if signature in seen_signatures:
            rejected["duplicate_signal"] += 1
            continue
        action = (int(candidate["side"]), int(candidate["horizon_seconds"]))
        correlated = False
        for peer in selected_by_action[action]:
            peer_id = cast(str, peer["expert_id"])
            smaller, larger = sorted((counts[peer_id], counts[expert_id]))
            if not larger or smaller / larger < MAX_SIGNAL_JACCARD:
                continue
            if (
                _packed_jaccard(
                    packed[peer_id], packed[expert_id], counts[peer_id], counts[expert_id]
                )
                >= MAX_SIGNAL_JACCARD
            ):
                correlated = True
                break
        if correlated:
            rejected["correlated_signal"] += 1
            continue
        selected.append(candidate)
        selected_by_action[action].append(candidate)
        seen_signatures.add(signature)
    return selected, rejected


def discover_library(matrix: pd.DataFrame, *, force: bool = False) -> dict[str, Any]:
    if LIBRARY.exists() and not force:
        cached = joblib.load(LIBRARY)
        if cached.get("protocol_hash") == PROTOCOL_HASH:
            return cast(dict[str, Any], cached)
        previous_hash = str(cached.get("protocol_hash", ""))
        if previous_hash in LEGACY_DISCOVERY_PROTOCOL_HASHES:
            cached["protocol_hash"] = PROTOCOL_HASH
            cached["migrated_from_protocol_hash"] = previous_hash
            cached["migration_reason"] = (
                "DOGE discovery inputs were unchanged; unavailable open-interest fields "
                "were removed only from the downstream regime and gating views"
            )
            _atomic_joblib(LIBRARY, cached)
            if CANDIDATES.exists():
                catalog = pd.read_parquet(CANDIDATES)
                catalog_hashes = set(catalog["protocol_hash"].astype(str).unique())
                if catalog_hashes == {previous_hash}:
                    catalog["protocol_hash"] = PROTOCOL_HASH
                    temporary = CANDIDATES.with_suffix(".parquet.migration.tmp")
                    catalog.to_parquet(temporary, index=False)
                    temporary.replace(CANDIDATES)
            return cast(dict[str, Any], cached)

    fit = _period(matrix, None, DISCOVERY_FIT_END)
    validation = _period(matrix, DISCOVERY_FIT_END, LIBRARY_FREEZE_END)
    funding = base._funding_curve()
    source = base._load_micro_source()
    x_fit = fit.loc[:, MODEL_FEATURES].to_numpy(np.float32)
    x_validation = validation.loc[:, MODEL_FEATURES].to_numpy(np.float32)
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
                        source,
                        funding,
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
            _status(
                "expert_candidates",
                (
                    f"azione {action_number}/{len(actions)}: alberi "
                    f"{trees_used}/{MAX_TREES_PER_ACTION}, validi {len(action_candidates)}"
                ),
                5
                + 45
                * (
                    action_number
                    - 0.5
                    + 0.5 * trees_used / MAX_TREES_PER_ACTION
                )
                / len(actions),
            )
            if batch_new == 0:
                empty_batches += 1
                if empty_batches >= SATURATION_PATIENCE:
                    break
            else:
                empty_batches = 0
        _status(
            "expert_diversity",
            f"azione {action_number}/{len(actions)}: {len(action_candidates)} regole valide",
            5 + 45 * (action_number - 0.25) / len(actions),
        )
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
    x = rows.loc[:, MODEL_FEATURES].to_numpy(np.float32)
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
            for feature in GATE_CONTEXT:
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
    frame = rows.loc[:, GATE_FEATURES].copy()
    frame["validation_profit_factor"] = frame["validation_profit_factor"].clip(
        upper=MAX_PROFIT_FACTOR_FEATURE
    )
    values = frame.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("gating features must be finite")
    return values


def _timestamp_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("entry_timestamp")["entry_timestamp"].transform("size").to_numpy(float)
    return np.asarray(1.0 / counts, dtype=float)


def _fit_gate(
    rows: pd.DataFrame, candidates: tuple[str, ...] = ("ridge", "xgboost")
) -> dict[str, Any]:
    x = _gate_x(rows)
    y = rows["net_bps"].to_numpy(np.float32)
    positive = (y > 0).astype(int)
    weights = _timestamp_weights(rows)
    models: dict[str, Any] = {}
    if "ridge" in candidates:
        ridge_reg = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(
            x, y, ridge__sample_weight=weights
        )
        ridge_cls = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2_000, random_state=20260820)
        ).fit(x, positive, logisticregression__sample_weight=weights)
        models["ridge"] = {"regressors": [ridge_reg], "classifiers": [ridge_cls]}
    if "xgboost" in candidates:
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
        models["xgboost"] = {"regressors": xgb_reg, "classifiers": xgb_cls}
    return models


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


@lru_cache(maxsize=1)
def _paper_bundle() -> dict[str, Any]:
    if not BUNDLE.exists():
        raise ValueError("Auto-MoE paper bundle is missing")
    bundle = cast(dict[str, Any], joblib.load(BUNDLE))
    if bundle.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("Auto-MoE paper bundle protocol is stale")
    if not bundle.get("paper_orders_enabled", False):
        raise ValueError("Auto-MoE paper economics gate is closed")
    return bundle


def _live_micro_features(
    records: pd.DataFrame, evaluated_at: pd.Timestamp
) -> dict[str, float] | None:
    required = {
        "exchange_second",
        "available_at",
        "buy_quote",
        "sell_quote",
        "trade_count",
        "aggregate_trades",
    }
    if records.empty or required - set(records):
        return None
    frame = records.loc[:, sorted(required)].copy()
    frame["exchange_second"] = pd.to_numeric(frame["exchange_second"], errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, format="mixed")
    for name in ("buy_quote", "sell_quote", "trade_count"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna().sort_values("exchange_second")
    if frame.empty:
        return None
    evaluated_at = (
        evaluated_at.tz_localize("UTC")
        if evaluated_at.tzinfo is None
        else evaluated_at.tz_convert("UTC")
    )
    frame["trade_close"] = frame["aggregate_trades"].map(
        lambda trades: (
            float(trades[-1][0]) if isinstance(trades, list) and trades else float("nan")
        )
    )
    frame["bucket"] = pd.to_datetime(
        frame["exchange_second"].astype(np.int64), unit="s", utc=True
    ).dt.floor("5s")
    buckets = frame.groupby("bucket", sort=True).agg(
        close=("trade_close", "last"),
        buy_quote=("buy_quote", "sum"),
        sell_quote=("sell_quote", "sum"),
        trade_count=("trade_count", "sum"),
        observed_seconds=("exchange_second", "nunique"),
    )
    buckets = buckets.loc[buckets.index + pd.Timedelta(seconds=5) <= evaluated_at].copy()
    buckets["close"] = buckets["close"].ffill()
    buckets = buckets.dropna(subset=["close"])
    if (
        len(buckets) < 121
        or buckets.index[-60:].to_series().diff().dropna().ne(pd.Timedelta(seconds=5)).any()
        or buckets["observed_seconds"].iloc[-60:].lt(5).any()
    ):
        return None
    signed = buckets["buy_quote"] - buckets["sell_quote"]
    quote = buckets["buy_quote"] + buckets["sell_quote"]

    def imbalance(window: int) -> float:
        denominator = float(quote.iloc[-window:].sum())
        return float(signed.iloc[-window:].sum() / denominator) if denominator > 0 else float("nan")

    baseline = float(
        buckets["trade_count"].shift(1).rolling(720, min_periods=120).median().iloc[-1]
    )
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    ofi_1m = imbalance(12)
    price_velocity_1m = float(
        (buckets["close"].iloc[-1] / buckets["close"].iloc[-13] - 1) * 10_000
    )
    result = {
        "ofi_15s": imbalance(3),
        "ofi_1m": ofi_1m,
        "ofi_5m": imbalance(60),
        "ofi_persistence_1m": float(np.sign(signed.iloc[-12:]).mean()),
        "trade_intensity_15s": float(buckets["trade_count"].iloc[-3:].sum() / baseline),
        "trade_intensity_1m": float(
            buckets["trade_count"].iloc[-12:].sum() / (12 * baseline)
        ),
        "absorption_1m": float(abs(ofi_1m) / (abs(price_velocity_1m) + 0.1)),
        "price_velocity_15s": float(
            (buckets["close"].iloc[-1] / buckets["close"].iloc[-4] - 1) * 10_000
        ),
        "price_velocity_1m": price_velocity_1m,
    }
    return result if np.isfinite(np.fromiter(result.values(), dtype=float)).all() else None


def live_candidate(
    context: pd.DataFrame,
    l2_records: pd.DataFrame,
    evaluated_at: pd.Timestamp,
) -> dict[str, Any] | None:
    bundle = _paper_bundle()
    if context.empty or l2_records.empty:
        return None
    latest = context.sort_values("available_at").iloc[-1]
    available_at = pd.Timestamp(latest["available_at"])
    available_at = (
        available_at.tz_localize("UTC")
        if available_at.tzinfo is None
        else available_at.tz_convert("UTC")
    )
    if available_at > evaluated_at or evaluated_at - available_at > pd.Timedelta(seconds=90):
        return None
    l2 = l2_records.copy()
    l2["available_at"] = pd.to_datetime(l2["available_at"], utc=True, format="mixed")
    l2 = l2.loc[l2["available_at"].le(evaluated_at)].copy()
    micro = _live_micro_features(l2, available_at)
    if micro is None:
        return None
    timestamp = available_at
    features: dict[str, float] = {}
    for name in MODEL_FEATURES:
        if name in micro:
            features[name] = micro[name]
        elif name == "hour_sin":
            hour = timestamp.hour + timestamp.minute / 60
            features[name] = float(np.sin(2 * np.pi * hour / 24))
        elif name == "hour_cos":
            hour = timestamp.hour + timestamp.minute / 60
            features[name] = float(np.cos(2 * np.pi * hour / 24))
        elif name == "weekday_sin":
            features[name] = float(np.sin(2 * np.pi * timestamp.dayofweek / 7))
        elif name == "weekday_cos":
            features[name] = float(np.cos(2 * np.pi * timestamp.dayofweek / 7))
        else:
            features[name] = float(latest.get(name, float("nan")))
    values = np.array([features[name] for name in MODEL_FEATURES], dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    library = cast(dict[str, Any], bundle["expert_library"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for expert in library["experts"]:
        grouped[f"{expert['side']}:{expert['horizon_seconds']}"].append(expert)
    action_rows: list[dict[str, Any]] = []
    x = values.reshape(1, -1)
    for key, experts in grouped.items():
        generator = library["generators"][key]
        leaves = np.asarray(generator.apply(x), dtype=np.int32)[0]
        generator_score = float(np.asarray(generator.predict(x), dtype=float)[0])
        for expert in experts:
            if leaves[int(expert["tree_index"])] != int(expert["leaf_id"]):
                continue
            side = int(expert["side"])
            row: dict[str, Any] = {
                "entry_timestamp": available_at,
                "exit_timestamp": available_at
                + pd.Timedelta(seconds=int(expert["horizon_seconds"])),
                "expert_id": expert["expert_id"],
            }
            for name in GATE_CONTEXT:
                value = features[name]
                row[name] = value * side if name in base.DIRECTIONAL_FEATURES else value
            for name in EXPERT_META_FEATURES:
                if name == "horizon_fraction":
                    row[name] = int(expert["horizon_seconds"]) / max(base.HORIZONS)
                elif name == "generator_score_bps":
                    row[name] = generator_score
                else:
                    row[name] = expert[name]
            action_rows.append(row)
    if not action_rows:
        return None
    scored = _score(
        pd.DataFrame(action_rows),
        cast(dict[str, Any], bundle["meta_model"]),
        cast(dict[str, Any], bundle["calibrators"]),
    ).sort_values(["calibrated_ev_bps", "raw_ev_bps", "expert_id"], ascending=[False, False, True])
    winner = scored.iloc[0]
    if float(winner["calibrated_ev_bps"]) <= 0:
        return None
    expert = next(
        item for item in library["experts"] if item["expert_id"] == winner["expert_id"]
    )
    side = int(expert["side"])
    decision_second = int(available_at.timestamp())
    decision_books = l2.loc[
        pd.to_numeric(l2["exchange_second"], errors="coerce").lt(decision_second)
    ].sort_values("exchange_second")
    if decision_books.empty:
        return None
    latest_book = decision_books.iloc[-1]
    price = (
        float(latest_book["mid"])
        if "mid" in latest_book and pd.notna(latest_book["mid"])
        else (
            float(latest_book["bids"][0][0]) + float(latest_book["asks"][0][0])
        )
        / 2
    )
    direction = "LONG" if side > 0 else "SHORT"
    return {
        "setup": "AUTO_MOE_VWAP_CONTROLLER",
        "candidate": True,
        "setup_active": True,
        "direction": direction,
        "available_at": available_at.isoformat(),
        "expert_id": expert["expert_id"],
        "policy_source": f"{base.SYMBOL}_AUTO_MOE_RESEARCH_PAPER",
        "operating_vwap": float(latest["rolling_vwap"]),
        "impulse_anchor_at": None,
        "stop_price": price * (1 - side * float(expert["stop_bps"]) / 10_000),
        "stop_bps": float(expert["stop_bps"]),
        "target_1_bps": float(expert["target_1_bps"]),
        "target_2_bps": float(expert["target_2_bps"]),
        "trailing_bps": float(expert["trailing_bps"]),
        "maximum_hold_minutes": int(expert["horizon_seconds"]) // 60,
        "management_style": "HALF_AT_Q50_Q75_NON_WIDENING_TRAIL",
        "partial_target_fraction": 0.5,
        "target_probability": float(winner["probability_net_positive"]),
        "expected_net_ev_bps": float(winner["calibrated_ev_bps"]),
        "robust_expected_gross_bps": float(winner["calibrated_ev_bps"])
        + base.ROUND_TRIP_COST_BPS,
        "passed_checks": 1,
        "total_checks": 1,
        "first_failed_check": None,
        "checks": [{"name": "auto_moe_calibrated_ev", "passed": True}],
    }


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


def _simple_trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "expectancy_bps": None, "profit_factor": None, "win_rate": None}
    net = trades["net_bps"].to_numpy(float)
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    return {
        "trades": len(trades),
        "expectancy_bps": float(net.mean()),
        "profit_factor": gains / losses if losses else None,
        "win_rate": float((net > 0).mean()),
    }


def _trade_breakdown(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"gross_expectancy_bps": None, "by_side": {}, "by_horizon": {}, "outcomes": {}}
    return {
        "gross_expectancy_bps": float(trades["gross_bps"].mean()),
        "funding_expectancy_bps": float(trades["funding_bps"].mean()),
        "round_trip_cost_bps": base.ROUND_TRIP_COST_BPS,
        "by_side": {
            str(key): _simple_trade_metrics(group) for key, group in trades.groupby("side")
        },
        "by_horizon": {
            str(key): _simple_trade_metrics(group)
            for key, group in trades.groupby("horizon_seconds")
        },
        "outcomes": {
            str(key): int(value) for key, value in trades["outcome"].value_counts().items()
        },
    }


def _policy_metrics(
    trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    metrics = base._metrics(trades, start, end)
    calendar = pd.date_range(start.floor("D"), end.floor("D") - pd.Timedelta(days=1), freq="1D")
    daily = (
        trades.assign(day=pd.to_datetime(trades["entry_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["net_bps"]
        .sum()
        if not trades.empty
        else pd.Series(dtype=float)
    )
    metrics["active_days"] = len(daily)
    metrics["flat_calendar_days"] = int(max(0, len(calendar) - len(daily)))
    metrics["positive_active_days"] = float(daily.gt(0).mean()) if len(daily) else 0.0
    metrics["nonnegative_calendar_days"] = float(
        daily.reindex(calendar, fill_value=0.0).ge(0).mean()
    )
    return metrics


def _policy_gates(metrics: dict[str, Any], *, live: bool) -> dict[str, bool]:
    expectancy = metrics.get("expectancy_bps")
    profit_factor = metrics.get("profit_factor")
    common = {
        "expectancy": expectancy is not None and float(expectancy) > 0,
        "profit_factor": profit_factor is not None
        and float(profit_factor) >= (1.15 if live else 1.05),
        "drawdown": metrics.get("max_drawdown") is not None
        and float(metrics["max_drawdown"]) <= 0.10,
        "risk_budget": int(metrics.get("risk_budget_violations", 0)) == 0,
    }
    if not live:
        return {"minimum_research_trades_50": int(metrics.get("trades", 0)) >= 50, **common}
    lcb = metrics.get("bootstrap_lcb_95_bps")
    pvalue = metrics.get("spa_pvalue")
    return {
        "minimum_historical_trades_300": int(metrics.get("trades", 0)) >= 300,
        "frequency_3_per_day": float(metrics.get("trades_per_day", 0.0)) >= 3.0,
        **common,
        "positive_active_days": float(metrics.get("positive_active_days", 0.0)) > 0.5,
        "stress_1_5x": metrics.get("stress_1_5x_expectancy_bps") is not None
        and float(metrics["stress_1_5x_expectancy_bps"]) >= 0,
        "bootstrap_lcb": lcb is not None and float(lcb) > 0,
        "spa": pvalue is not None and float(pvalue) <= 0.05,
    }


def _prequential_audit(
    actions: pd.DataFrame, champion: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    timestamp = pd.to_datetime(actions["entry_timestamp"], utc=True)
    trade_pieces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    latest_model: dict[str, Any] | None = None
    latest_calibrators: dict[str, Any] | None = None
    for test_start in pd.date_range(
        CALIBRATION_END, HISTORICAL_AUDIT_END, freq="MS", inclusive="left"
    ):
        test_end = min(test_start + pd.offsets.MonthBegin(1), HISTORICAL_AUDIT_END)
        calibration_start = test_start - pd.offsets.MonthBegin(1)
        fit = actions.loc[timestamp.lt(calibration_start - base.PURGE)].copy()
        calibration = actions.loc[
            timestamp.ge(calibration_start) & timestamp.lt(test_start - base.PURGE)
        ].copy()
        test = actions.loc[
            timestamp.ge(test_start) & timestamp.lt(test_end - base.PURGE)
        ].copy()
        if fit.empty or calibration.empty or test.empty:
            continue
        latest_model = _fit_gate(fit, (champion,))[champion]
        latest_calibrators = _fit_calibrators(calibration, latest_model)
        scored = _score(test, latest_model, latest_calibrators)
        trades = _execute(scored)
        trade_pieces.append(trades)
        diagnostics.append(
            {
                "month": test_start.strftime("%Y-%m"),
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "positive_candidate_fraction": float(
                    scored.groupby("entry_timestamp")["calibrated_ev_bps"].max().gt(0).mean()
                ),
                "metrics": _policy_metrics(trades, test_start, test_end),
            }
        )
    if latest_model is None or latest_calibrators is None:
        raise ValueError("prequential audit has no complete fit/calibration/test window")
    combined = (
        pd.concat(trade_pieces, ignore_index=True)
        if trade_pieces
        else pd.DataFrame(columns=actions.columns)
    )
    return combined, diagnostics, latest_model, latest_calibrators


def train(*, force: bool = False) -> dict[str, Any]:
    _status("start", f"{base.SYMBOL} automatic expert discovery", 0)
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
        "shadow_collection_enabled": True,
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
    gate_tune = actions.loc[timestamp.lt(GATE_TUNE_END - base.PURGE)].copy()
    model_audit = actions.loc[
        timestamp.ge(GATE_TUNE_END) & timestamp.lt(GATE_FIT_END - base.PURGE)
    ].copy()
    calibration = actions.loc[
        timestamp.ge(GATE_FIT_END) & timestamp.lt(CALIBRATION_END - base.PURGE)
    ].copy()
    historical_audit = actions.loc[
        timestamp.ge(CALIBRATION_END) & timestamp.lt(HISTORICAL_AUDIT_END - base.PURGE)
    ].copy()
    candidate_models = _fit_gate(gate_tune)
    candidate_metrics = {
        name: _gate_metrics(model_audit, candidate_models[name]) for name in ("ridge", "xgboost")
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
    gate_fit = actions.loc[timestamp.lt(GATE_FIT_END - base.PURGE)].copy()
    trades, prequential, _, _ = _prequential_audit(actions, champion)
    forward_calibration_start = HISTORICAL_AUDIT_END - pd.offsets.MonthBegin(1)
    forward_fit = actions.loc[
        timestamp.lt(forward_calibration_start - base.PURGE)
    ].copy()
    forward_calibration = actions.loc[
        timestamp.ge(forward_calibration_start)
        & timestamp.lt(HISTORICAL_AUDIT_END - base.PURGE)
    ].copy()
    champion_model = _fit_gate(forward_fit, (champion,))[champion]
    calibrators = _fit_calibrators(forward_calibration, champion_model)
    AUDIT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    temporary_trades = AUDIT_TRADES.with_suffix(".parquet.tmp")
    trades.to_parquet(temporary_trades, index=False)
    temporary_trades.replace(AUDIT_TRADES)
    metrics = _policy_metrics(trades, CALIBRATION_END, HISTORICAL_AUDIT_END)
    paper_gates = _policy_gates(metrics, live=False)
    live_gates = _policy_gates(metrics, live=True)
    paper_pass = all(paper_gates.values())
    historical_pass = all(live_gates.values())
    months: dict[str, Any] = {}
    for start in pd.date_range(CALIBRATION_END, HISTORICAL_AUDIT_END, freq="MS", inclusive="left"):
        end = min(start + pd.offsets.MonthBegin(1), HISTORICAL_AUDIT_END)
        timestamp_trades = pd.to_datetime(trades["entry_timestamp"], utc=True)
        monthly_trades = trades.loc[timestamp_trades.ge(start) & timestamp_trades.lt(end)].copy()
        months[start.strftime("%Y-%m")] = _policy_metrics(monthly_trades, start, end)
    report.update(
        {
            "forward_action_rows": len(actions),
            "gate_tune_rows": len(gate_tune),
            "gate_fit_rows": len(gate_fit),
            "model_audit_rows": len(model_audit),
            "calibration_rows": len(calibration),
            "historical_audit_rows": len(historical_audit),
            "library_spa_pvalue": _library_spa(gate_tune),
            "candidate_metrics": candidate_metrics,
            "gating_champion": champion,
            "prequential_monthly_refits": prequential,
            "forward_model_fit_rows": len(forward_fit),
            "forward_model_calibration_rows": len(forward_calibration),
            "historical_audit": {
                "metrics": metrics,
                "paper_gates": paper_gates,
                "live_gates": live_gates,
                "months": months,
                "trade_breakdown": _trade_breakdown(trades),
                "flat_days_are_neutral": True,
            },
            "paper_orders_enabled": paper_pass,
            "verdict": (
                "HISTORICAL_ALPHA_READY_FOR_FUTURE_HOLDOUT"
                if historical_pass
                else "RESEARCH_PAPER"
                if paper_pass
                else "NO_DEPLOYABLE_POLICY"
            ),
        }
    )
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "expert_library": library,
        "meta_model": champion_model,
        "calibrators": calibrators,
        "gating_champion": champion,
        "historical_pass": historical_pass,
        "paper_pass": paper_pass,
        "research_only": True,
        "orders_enabled": False,
        "paper_orders_enabled": paper_pass,
        "live_orders_enabled": False,
    }
    _atomic_joblib(BUNDLE, bundle)
    _atomic_json(REPORT, report)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{base.SYMBOL} automatic two-stage expert training"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(train(force=arguments.force), indent=2, default=str))


if __name__ == "__main__":
    main()
