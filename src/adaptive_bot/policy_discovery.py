from __future__ import annotations

import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from adaptive_bot.config import AppConfig, MachineLearningConfig
from adaptive_bot.ml_research import FEATURE_COLUMNS, write_ml_status
from adaptive_bot.research import combinatorial_pbo, deflated_sharpe_probability
from adaptive_bot.scientific_ml import (
    INVALID_SCORE,
    OofEvaluation,
    StrategyCandidate,
    _append_outer_predictions,
    _atomic_json,
    _candidate_frame,
    _evaluate_candidate,
    _event_driven_replay,
    _fit_calibrated,
    _gate,
    _logistic,
    _metrics,
    _non_overlapping_events,
    _non_overlapping_returns,
    _outer_fold_returns_for_events,
    _payoff_priors,
    _robust_score,
    _scientific_config,
    _third_cost_events,
    generate_candidates,
    gpu_preflight,
    reality_check_pvalue,
    resample_observed,
    rolling_nested_splits,
)

PROTOCOL_VERSION = "adaptive_policy_v3"
REPORT_PATH = Path("data/reports/ml_policy_research.json")
STATUS_PATH = Path("data/reports/ml_policy_research.status.json")
MODEL_ROOT = Path("data/models/policy_candidates")
MIN_OPPORTUNITY_Z = 0.25
MIN_CLASS_SAMPLES = 5


def policy_candidates() -> tuple[StrategyCandidate, ...]:
    """Small preregistered exit family; entry timing is learned from market state."""
    candidates: list[StrategyCandidate] = []
    for timeframe in (5, 15, 30):
        cooldown = max(1, 60 // timeframe)
        for vwap_hours in (8, 48):
            for profile, stop_atr, exit_z, holding_hours in (
                ("fast", 2.0, 0.5, 4),
                ("full", 3.0, 0.0, 12),
            ):
                candidates.append(
                    StrategyCandidate(
                        candidate_id=f"policy-{timeframe}m-{vwap_hours}h-{profile}",
                        timeframe_minutes=timeframe,
                        vwap_hours=vwap_hours,
                        atr_period=14,
                        adx_period=14,
                        entry_z_long=MIN_OPPORTUNITY_Z,
                        entry_z_short=MIN_OPPORTUNITY_Z,
                        range_adx_threshold=20.0,
                        regime_policy="any_nonshock",
                        entry_rule="touch",
                        confirmation_bars=0,
                        stop_atr=stop_atr,
                        exit_z=exit_z,
                        time_stop_hours=holding_hours,
                        cooldown_bars=cooldown,
                    )
                )
    return tuple(candidates)


def opportunity_mask(data: pd.DataFrame, candidate: StrategyCandidate, side: str) -> pd.Series:
    """All causal Adaptive Range opportunities, without the old ADX/entry-rule gate."""
    z = data["distance_vwap_atr"]
    direction = z.le(-MIN_OPPORTUNITY_Z) if side == "long" else z.ge(MIN_OPPORTUNITY_Z)
    valid = (
        data["data_valid"].rolling(candidate.vwap_bars, min_periods=candidate.vwap_bars).min().eq(1)
    )
    atr_change = data["atr"].pct_change(fill_method=None)
    cumulative_move = (data["close"] - data["close"].shift(3)) / data["atr"]
    shock = data["atr_percentile"].gt(90) | atr_change.gt(0.5) | cumulative_move.abs().gt(3)
    executable = data[f"net_return_{side}"].notna() & data[f"net_return_{side}"].ne(0)
    return direction & valid & ~shock & data[f"target_{side}"].ge(0) & executable


def _policy_rows(data: pd.DataFrame, indexes: np.ndarray, mask: pd.Series) -> pd.DataFrame:
    rows = data.iloc[indexes].loc[mask.iloc[indexes]].copy()
    rows["event_index"] = rows.index
    for side in ("long", "short"):
        rows[f"target_{side}"] = rows[f"net_return_{side}"].gt(0).astype(np.int8)
    return rows.reset_index(drop=True)


def _class_supported(data: pd.DataFrame, target: str, minimum: int = MIN_CLASS_SAMPLES) -> bool:
    counts = data[target].value_counts()
    return len(counts) == 2 and int(counts.min()) >= minimum


def _xgb(rounds: int, seed: int, *, early_stopping: bool = False) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=rounds,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        early_stopping_rounds=40 if early_stopping else None,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )


def _fit_policy_xgb(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    side: str,
    seed: int,
) -> tuple[Any, int]:
    target = f"target_{side}"
    if calibration.empty:
        raise ValueError("calibration window is empty")
    first_calibration_event = int(calibration.iloc[0]["event_index"])
    train = train.loc[train[f"exit_index_{side}"].lt(first_calibration_event)].copy()
    split = int(len(train) * 0.8)
    base = train.iloc[:split].copy()
    early = train.iloc[split:].copy()
    if early.empty:
        raise ValueError("early-stopping window is empty")
    first_early_event = int(early.iloc[0]["event_index"])
    base = base.loc[base[f"exit_index_{side}"].lt(first_early_event)]
    if min(len(base), len(early), len(calibration)) < 100:
        raise ValueError("policy fold has insufficient chronological samples")
    if not all(_class_supported(frame, target) for frame in (base, early, calibration)):
        raise ValueError("policy fold has insufficient class support")
    probe = _xgb(1200, seed, early_stopping=True)
    probe.fit(
        base.loc[:, FEATURE_COLUMNS],
        base[target],
        eval_set=[(early.loc[:, FEATURE_COLUMNS], early[target])],
        verbose=False,
    )
    rounds = max(20, int(getattr(probe, "best_iteration", 399)) + 1)
    return _fit_calibrated(_xgb(rounds, seed), train, calibration, side), rounds


def _classification(oof: OofEvaluation) -> dict[str, float | int | None]:
    if not oof.targets:
        return {"samples": 0, "average_precision": None, "brier": None, "log_loss": None}
    target = np.asarray(oof.targets, dtype=int)
    probability = np.asarray(oof.probabilities, dtype=float)
    return {
        "samples": len(target),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


def _evaluate_policy(
    app: AppConfig,
    frame: pd.DataFrame,
    candidate: StrategyCandidate,
    config: MachineLearningConfig,
    *,
    position: int,
    total: int,
    run_id: str,
) -> dict[str, Any]:
    data = _candidate_frame(frame, app, candidate)
    data["exit_index_max"] = data[["exit_index_long", "exit_index_short"]].max(axis=1)
    folds = rolling_nested_splits(
        data,
        train_weeks=config.outer_train_weeks,
        calibration_weeks=config.outer_calibration_weeks,
        test_weeks=config.outer_test_weeks,
        step_weeks=config.outer_step_weeks,
        exit_column="exit_index_max",
    )
    xgb_oof = {side: OofEvaluation([], [], [], []) for side in ("long", "short")}
    logistic_oof = {side: OofEvaluation([], [], [], []) for side in ("long", "short")}
    rounds: dict[str, list[int]] = {"long": [], "short": []}
    masks = {side: opportunity_mask(data, candidate, side) for side in ("long", "short")}
    completed = 0
    fold_total = len(folds) * 2
    for fold_number, fold in enumerate(folds, 1):
        for side in ("long", "short"):
            completed += 1
            train = _policy_rows(data, fold.train, masks[side])
            calibration = _policy_rows(data, fold.calibration, masks[side])
            test = _policy_rows(data, fold.test, masks[side])
            target = f"target_{side}"
            if min(len(train), len(calibration), len(test)) >= 100 and all(
                _class_supported(part, target) for part in (train, calibration)
            ):
                try:
                    model, best_rounds = _fit_policy_xgb(
                        train, calibration, side, config.random_seed + fold_number
                    )
                    baseline = _fit_calibrated(
                        _logistic(config.random_seed), train, calibration, side
                    )
                except ValueError:
                    pass
                else:
                    rounds[side].append(best_rounds)
                    _append_outer_predictions(xgb_oof[side], model, train, test, candidate, side)
                    _append_outer_predictions(
                        logistic_oof[side], baseline, train, test, candidate, side
                    )
            write_ml_status(
                config,
                "policy_walk_forward",
                f"{candidate.candidate_id} {side}: fold {fold_number}/{len(folds)}",
                5 + ((position - 1) + completed / fold_total) / total * 80,
                run_id=run_id,
                completed=completed,
                total=fold_total,
                backend="cuda",
                action_space="LONG/SHORT/FLAT",
                opportunity_rows={name: int(mask.sum()) for name, mask in masks.items()},
            )
    events = [event for result in xgb_oof.values() for event in result.events]
    stress_events = [event for result in xgb_oof.values() for event in result.stress_events]
    stress_3x_events = _third_cost_events(events, stress_events)
    executed = _non_overlapping_events(events, candidate.cooldown_bars)
    returns = [event[2] for event in executed]
    stress_returns = _non_overlapping_returns(stress_events, candidate.cooldown_bars)
    stress_3x_returns = _non_overlapping_returns(stress_3x_events, candidate.cooldown_bars)
    fold_returns = _outer_fold_returns_for_events(data, candidate, events, config)
    executed_set = set(executed)
    event_sets = {
        side: set(result.events).intersection(executed_set) for side, result in xgb_oof.items()
    }
    logistic_events = [event for result in logistic_oof.values() for event in result.events]
    evaluation: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "parameters": asdict(candidate),
        "score": _robust_score(fold_returns),
        "metrics": _metrics(returns),
        "stress_2x_metrics": _metrics(stress_returns),
        "stress_3x_metrics": _metrics(stress_3x_returns),
        "sides": {
            side: _metrics([event[2] for event in executed if event in event_sets[side]])
            for side in ("long", "short")
        },
        "classification": {side: _classification(xgb_oof[side]) for side in ("long", "short")},
        "logistic_benchmark": _metrics(
            _non_overlapping_returns(logistic_events, candidate.cooldown_bars)
        ),
        "best_rounds": {
            side: round(statistics.median(values)) if values else None
            for side, values in rounds.items()
        },
        "fold_returns": fold_returns,
        "returns": returns,
        "stress_returns": stress_returns,
        "stress_3x_returns": stress_3x_returns,
        "side_event_indexes": {
            side: [event[0] for event in events] for side, events in event_sets.items()
        },
    }
    return evaluation


def _fit_final_models(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    config: MachineLearningConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    calibration_start = timestamps.max() - pd.Timedelta(weeks=config.outer_calibration_weeks)
    train_indexes = np.flatnonzero(timestamps.lt(calibration_start))
    calibration_indexes = np.flatnonzero(timestamps.ge(calibration_start))
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for side in ("long", "short"):
        mask = opportunity_mask(data, candidate, side)
        train = _policy_rows(data, train_indexes, mask)
        calibration = _policy_rows(data, calibration_indexes, mask)
        model, rounds = _fit_policy_xgb(train, calibration, side, config.random_seed)
        win, loss = _payoff_priors(train, side, candidate)
        models[side] = model
        metadata[side] = {
            "best_rounds": rounds,
            "expected_win": win,
            "expected_loss": loss,
            "break_even_probability": loss / (win + loss) if win + loss > 0 else 1.0,
            "train_samples": len(train),
            "calibration_samples": len(calibration),
        }
    return models, metadata


def policy_research_config(app: AppConfig) -> MachineLearningConfig:
    config = _scientific_config(app)
    return config.model_copy(
        update={
            "report_path": REPORT_PATH,
            "status_path": STATUS_PATH,
            "model_path": Path("data/models/adaptive_range_policy.joblib"),
            "candidate_directory": MODEL_ROOT,
        }
    )


def run_policy_discovery(app: AppConfig, one_minute: pd.DataFrame) -> dict[str, Any]:
    config = policy_research_config(app)
    if app.risk.risk_per_trade != Decimal("0.01"):
        raise ValueError("policy discovery requires fixed risk_per_trade=1%")
    if app.bitunix is None or str(app.bitunix.leverage) != "10":
        raise ValueError("policy discovery requires fixed leverage=10x")
    gpu = gpu_preflight(required=config.gpu_required)
    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("policyv3-%Y%m%dT%H%M%SZ")
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True)
    holdout_end = timestamps.max() + pd.Timedelta(minutes=1)
    holdout_start = holdout_end - pd.Timedelta(weeks=config.holdout_weeks)
    development_raw = one_minute.loc[timestamps.lt(holdout_start)].copy()
    if development_raw.empty:
        raise ValueError("no development data precedes the sealed holdout")
    candidates = policy_candidates()
    write_ml_status(
        config,
        "policy_setup",
        "Real archive loaded; resampling causal timeframes",
        2,
        run_id=run_id,
        completed=0,
        total=len(candidates),
        backend=gpu["backend"],
    )
    frames = {
        minutes: resample_observed(development_raw, minutes, float(app.instrument.tick_size))
        for minutes in sorted({candidate.timeframe_minutes for candidate in candidates})
    }
    write_ml_status(
        config,
        "policy_setup",
        "Real archive loaded; causal LONG/SHORT/FLAT policy discovery",
        5,
        run_id=run_id,
        completed=0,
        total=len(candidates),
        backend=gpu["backend"],
    )
    evaluations: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, 1):
        evaluation = _evaluate_policy(
            app,
            frames[candidate.timeframe_minutes],
            candidate,
            config,
            position=position,
            total=len(candidates),
            run_id=run_id,
        )
        evaluations.append(evaluation)
        elapsed = time.monotonic() - started
        remaining = elapsed / position * (len(candidates) - position)
        ranked = [item for item in evaluations if item["score"] > INVALID_SCORE]
        best = max(ranked, key=lambda item: float(item["score"])) if ranked else evaluation
        write_ml_status(
            config,
            "policy_candidate_complete",
            f"Completed {candidate.candidate_id}",
            5 + position / len(candidates) * 80,
            run_id=run_id,
            completed=position,
            total=len(candidates),
            backend=gpu["backend"],
            eta_seconds=round(remaining),
            candidate=candidate.candidate_id,
            candidate_score=round(float(evaluation["score"]), 6),
            candidate_metrics=evaluation["metrics"],
            best_candidate=best["candidate_id"],
            best_score=round(float(best["score"]), 6),
            best_metrics=best["metrics"],
        )
    valid = [item for item in evaluations if item["score"] > INVALID_SCORE and item["fold_returns"]]
    if not valid:
        payload = {
            "schema_version": 3,
            "protocol": PROTOCOL_VERSION,
            "run_id": run_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "development_gate_passed": False,
            "development_verdict": "NO_DEMONSTRABLE_EDGE",
            "gate_failures": ["no_valid_policy_candidate"],
            "holdout": {"status": "sealed", "opened": False},
            "accepted": False,
            "evaluations": evaluations,
        }
        _atomic_json(config.report_path, payload)
        write_ml_status(
            config,
            "development_complete",
            "No valid learned policy; final holdout remains sealed",
            100,
            run_id=run_id,
            gate_passed=False,
        )
        return payload
    valid.sort(key=lambda item: (float(item["score"]), item["candidate_id"]), reverse=True)
    champion = valid[0]
    candidate = StrategyCandidate(**champion["parameters"])
    champion_data = _candidate_frame(frames[candidate.timeframe_minutes], app, candidate)
    pbo = combinatorial_pbo([item["fold_returns"] for item in valid])
    trial_sharpes = [
        statistics.mean(item["returns"]) / statistics.stdev(item["returns"])
        if len(item["returns"]) > 1 and statistics.stdev(item["returns"])
        else 0.0
        for item in valid
    ]
    dsr = deflated_sharpe_probability(champion["returns"], trial_sharpes)
    baseline_candidate = next(
        item for item in generate_candidates(config) if item.candidate_id == "baseline-adx20"
    )
    baseline_data = _candidate_frame(
        frames[baseline_candidate.timeframe_minutes], app, baseline_candidate
    )
    baseline = _evaluate_candidate(baseline_data, baseline_candidate, config)
    reality_pvalue = reality_check_pvalue(
        [item["fold_returns"] for item in valid], baseline["fold_returns"], config.random_seed
    )
    allowed_entries = {
        (pd.Timestamp(champion_data.iloc[index]["timestamp"]).to_pydatetime(), side)
        for side, indexes in champion["side_event_indexes"].items()
        for index in indexes
    }
    write_ml_status(
        config,
        "event_replay",
        "Learned policy replay through shared execution and risk engine",
        90,
        run_id=run_id,
        backend=gpu["backend"],
        candidate=candidate.candidate_id,
        allowed_entries=len(allowed_entries),
    )
    replay = _event_driven_replay(
        app, frames[candidate.timeframe_minutes], candidate, allowed_entries
    )
    accepted, failures = _gate(
        champion["metrics"],
        champion["stress_2x_metrics"],
        returns=champion["returns"],
        config=config,
        pbo=cast(float | None, pbo.get("pbo")),
        dsr=dsr,
        reality_pvalue=reality_pvalue,
        side_metrics=champion["sides"],
    )
    if not replay["passed"]:
        accepted = False
        failures.append("event_driven_replay_failed")
    models, model_metadata = _fit_final_models(champion_data, candidate, config)
    artifact = config.candidate_directory / run_id / f"{candidate.candidate_id}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "protocol": PROTOCOL_VERSION,
            "run_id": run_id,
            "candidate": asdict(candidate),
            "models": models,
            "model_metadata": model_metadata,
            "features": FEATURE_COLUMNS,
            "action_space": ("LONG", "SHORT", "FLAT"),
            "trained_until": holdout_start.isoformat(),
        },
        artifact,
    )
    for item in evaluations:
        for key in (
            "fold_returns",
            "returns",
            "stress_returns",
            "stress_3x_returns",
            "side_event_indexes",
        ):
            item.pop(key, None)
    champion_public = next(
        item for item in evaluations if item["candidate_id"] == candidate.candidate_id
    )
    champion_public["artifact"] = str(artifact)
    payload = {
        "schema_version": 3,
        "protocol": PROTOCOL_VERSION,
        "run_id": run_id,
        "mode": "research_development_only",
        "started_at": (
            datetime.now(UTC) - timedelta(seconds=time.monotonic() - started)
        ).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "data": {
            "development_start": pd.to_datetime(development_raw["timestamp"], utc=True)
            .min()
            .isoformat(),
            "development_end": holdout_start.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "holdout_end": holdout_end.isoformat(),
            "prices_mark_funding": "observed Bitunix archive",
            "historical_spread": "unavailable_not_estimated",
            "spread_assumption_bps": float(app.backtest.spread_bps),
            "slippage_assumption_bps_per_side": float(app.backtest.slippage_bps),
        },
        "compute": gpu,
        "search": {
            "entry_policy": "learned from every valid signed VWAP opportunity",
            "action_space": ["LONG", "SHORT", "FLAT"],
            "preregistered_exit_candidates": len(candidates),
            "timeframes": sorted(frames),
            "random_splits": False,
            "outer_walk_forward_weeks": {
                "train": config.outer_train_weeks,
                "calibration": config.outer_calibration_weeks,
                "test": config.outer_test_weeks,
                "step": config.outer_step_weeks,
            },
        },
        "selection_bias": {
            **pbo,
            "candidate_count": len(valid),
            "deflated_sharpe_probability": dsr,
            "white_style_reality_check_pvalue": reality_pvalue,
            "baseline": "baseline-adx20",
        },
        "champion": champion_public,
        "development_metrics": champion_public["metrics"],
        "stress_2x_metrics": champion_public["stress_2x_metrics"],
        "stress_3x_metrics": champion_public["stress_3x_metrics"],
        "development_gate_passed": accepted,
        "development_verdict": (
            "ELIGIBLE_FOR_FINAL_HOLDOUT" if accepted else "NO_DEMONSTRABLE_EDGE"
        ),
        "gate_failures": failures,
        "event_driven_replay": replay,
        "holdout": {"status": "sealed", "opened": False},
        "accepted": False,
        "risk_policy": {
            "risk_per_trade": "0.01",
            "leverage": "10",
            "model_controls_size": False,
            "automatic_promotion": False,
        },
        "evaluations": evaluations,
    }
    _atomic_json(config.report_path, payload)
    write_ml_status(
        config,
        "development_complete",
        "Policy discovery complete; final holdout remains sealed",
        100,
        run_id=run_id,
        backend=gpu["backend"],
        elapsed_seconds=round(time.monotonic() - started),
        gate_passed=accepted,
    )
    return payload
