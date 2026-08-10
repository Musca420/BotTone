from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot import musca_btc_daily_portfolio as daily
from adaptive_bot import musca_btc_moe as base

REPORT = Path("data/reports/musca_btc_daily_portfolio_fqi.json")
STATUS = Path("data/reports/musca_btc_daily_portfolio_fqi.status.json")
BUNDLE = Path("data/models/musca_btc_daily_portfolio_fqi/research_bundle.joblib")
AUDIT_TRADES = Path("data/ml/musca_btc_daily_portfolio_fqi/audit_trades.parquet")

FQI_ITERATIONS = 4
ACTION_COUNT = len(base.HORIZONS) * len(base.SIDES)
WAIT_FEATURES = (*base.GATING_CONTEXT, *base.EXPERT_COLUMNS, "minutes_to_utc_close")
MODEL_FEATURES = daily.FEATURES

PROTOCOL = {
    "name": "musca_btc_finite_horizon_fitted_q_iteration_v1",
    "replaces_failed_protocol_hash": daily.PROTOCOL_HASH,
    "parent_action_protocol_hash": base.PROTOCOL_HASH,
    "positive_control_protocol_hash": daily.PROTOCOL["positive_control_protocol_hash"],
    "symbol": "BTCUSDT",
    "venue": "Binance USD-M futures",
    "objective": (
        "expected end-of-day portfolio log equity; fitted Bellman value, not hindsight oracle"
    ),
    "actions": {
        "market_actions": "LONG/SHORT x 1/5/15/60/360 minutes",
        "flat": "separate learned wait value for the next minute",
        "open_position": "frozen 5-second managed exit; no overlapping position",
    },
    "fitted_q": {
        "iterations": FQI_ITERATIONS,
        "terminal_value": 0,
        "next_value": "previous-iteration model prediction only",
        "realized_future_maximum_used": False,
        "partial_utc_days": "excluded",
        "action_features": list(MODEL_FEATURES),
        "wait_features": list(WAIT_FEATURES),
    },
    "chronology": daily.PROTOCOL["chronology"],
    "models": {
        "default": "ridge",
        "challenger": "xgboost_cuda_squared_error",
        "challenger_rule": (
            "lower Bellman TD MAE and higher terminal replay mean daily return on 2025-Q3"
        ),
        "calibration": "isotonic Bellman advantage on 2025-Q4",
        "threshold_search": False,
        "entry": "best calibrated action advantage > 0; otherwise FLAT",
    },
    "economics": daily.PROTOCOL["economics"],
    "gates": daily.PROTOCOL["gates"],
    "future_holdout_opened": False,
    "paper_orders_enabled": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class Regressor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Transitions:
    actions: pd.DataFrame
    states: pd.DataFrame
    valid_action: np.ndarray
    current_state: np.ndarray
    next_action_state: np.ndarray
    next_wait_state: np.ndarray
    reward_bps: np.ndarray


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
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


def _action_x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, MODEL_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("FQI action features must be finite")
    return values


def _wait_x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, WAIT_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("FQI wait features must be finite")
    return values


def _complete_utc_days(rows: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(rows["entry_timestamp"], utc=True)
    day = timestamp.dt.floor("D")
    decisions = rows.assign(_day=day).groupby("_day")["entry_timestamp"].nunique()
    complete = decisions.loc[decisions.eq(24 * 60)].index
    return rows.loc[day.isin(complete)].copy()


def build_transitions(rows: pd.DataFrame, *, require_complete_days: bool = True) -> Transitions:
    if require_complete_days:
        rows = _complete_utc_days(rows)
    actions = rows.sort_values(
        ["entry_timestamp", "side", "horizon_seconds"], kind="stable"
    ).reset_index(drop=True)
    counts = actions.groupby("entry_timestamp", sort=False).size().to_numpy(int)
    if not len(counts) or not np.all(counts == ACTION_COUNT):
        raise ValueError("each FQI state must contain every side/horizon action")
    state_indexes = actions.index[
        actions["side"].eq(1) & actions["horizon_seconds"].eq(min(base.HORIZONS))
    ].to_numpy(int)
    states = actions.iloc[state_indexes].reset_index(drop=True)
    if len(states) != len(counts):
        raise ValueError("one canonical wait-state row is required per timestamp")

    current_state = np.repeat(np.arange(len(states), dtype=int), ACTION_COUNT)
    state_time = pd.to_datetime(states["entry_timestamp"], utc=True)
    state_ns = state_time.astype("datetime64[ns, UTC]").astype("int64").to_numpy(np.int64)
    action_time = pd.to_datetime(actions["entry_timestamp"], utc=True)
    action_ns = action_time.astype("datetime64[ns, UTC]").astype("int64").to_numpy(np.int64)
    horizon_ns = actions["horizon_seconds"].to_numpy(np.int64) * 1_000_000_000
    exit_ns = action_ns + horizon_ns
    day_end_ns = (
        (action_time.dt.floor("D") + pd.Timedelta(days=1))
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy(np.int64)
    )
    valid_action = exit_ns <= day_end_ns
    next_action_state = np.full(len(actions), -1, dtype=int)
    next_wait_state = np.full(len(states), -1, dtype=int)
    state_days = state_time.dt.floor("D")

    for _, state_indexes_value in states.groupby(state_days, sort=True).groups.items():
        day_states = np.asarray(state_indexes_value, dtype=int)
        first_state = int(day_states[0])
        last_state = int(day_states[-1])
        day_times = state_ns[day_states]
        day_action_indexes = np.arange(
            first_state * ACTION_COUNT, (last_state + 1) * ACTION_COUNT, dtype=int
        )
        local_next = np.searchsorted(day_times, exit_ns[day_action_indexes], side="left")
        inside = local_next < len(day_states)
        next_action_state[day_action_indexes[inside]] = day_states[local_next[inside]]
        next_wait_state[day_states[:-1]] = day_states[1:]

    portfolio = daily._portfolio_return(actions)
    if np.any(portfolio <= -0.999):
        raise ValueError("FQI action can lose the complete portfolio")
    reward_bps = np.log1p(portfolio) * 10_000
    return Transitions(
        actions=actions,
        states=states,
        valid_action=np.asarray(valid_action, dtype=bool),
        current_state=current_state,
        next_action_state=next_action_state,
        next_wait_state=next_wait_state,
        reward_bps=np.asarray(reward_bps, dtype=float),
    )


def _fit_regressor(kind: str, values: np.ndarray, target: np.ndarray) -> Regressor:
    if kind == "ridge":
        return cast(
            Regressor,
            make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(values, target),
        )
    if kind != "xgboost_cuda":
        raise ValueError(f"unknown FQI model kind: {kind}")
    return cast(
        Regressor,
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=20260810,
        ).fit(values, target, verbose=False),
    )


def _predict_values(
    transitions: Transitions, action_model: Regressor, wait_model: Regressor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_q = np.full(len(transitions.actions), -np.inf, dtype=float)
    action_q[transitions.valid_action] = np.asarray(
        action_model.predict(_action_x(transitions.actions.loc[transitions.valid_action])),
        dtype=float,
    )
    wait_q = np.asarray(wait_model.predict(_wait_x(transitions.states)), dtype=float)
    maximum_action = np.full(len(transitions.states), -np.inf, dtype=float)
    np.maximum.at(
        maximum_action,
        transitions.current_state[transitions.valid_action],
        action_q[transitions.valid_action],
    )
    value = np.maximum(0.0, np.maximum(maximum_action, wait_q))
    if not np.isfinite(value).all():
        raise ValueError("FQI predicted state values must be finite")
    return action_q, wait_q, value


def fit_fqi(
    rows: pd.DataFrame, kind: str, *, status_start: float, status_width: float
) -> dict[str, Any]:
    transitions = build_transitions(rows)
    action_rows = transitions.actions.loc[transitions.valid_action]
    action_values = _action_x(action_rows)
    wait_values = _wait_x(transitions.states)
    value = np.zeros(len(transitions.states), dtype=float)
    action_model: Regressor | None = None
    wait_model: Regressor | None = None
    diagnostics: list[dict[str, float]] = []

    for iteration in range(1, FQI_ITERATIONS + 1):
        next_action = transitions.next_action_state[transitions.valid_action]
        action_target = transitions.reward_bps[transitions.valid_action] + np.where(
            next_action >= 0, value[np.maximum(next_action, 0)], 0.0
        )
        wait_target = np.where(
            transitions.next_wait_state >= 0,
            value[np.maximum(transitions.next_wait_state, 0)],
            0.0,
        )
        action_model = _fit_regressor(kind, action_values, action_target)
        wait_model = _fit_regressor(kind, wait_values, wait_target)
        _, _, updated = _predict_values(transitions, action_model, wait_model)
        diagnostics.append(
            {
                "iteration": float(iteration),
                "mean_value_bps": float(updated.mean()),
                "q95_value_bps": float(np.quantile(updated, 0.95)),
                "maximum_value_bps": float(updated.max()),
            }
        )
        value = updated
        _status(
            "fitted_q",
            f"{kind} Bellman iteration {iteration}/{FQI_ITERATIONS}",
            status_start + status_width * iteration / FQI_ITERATIONS,
        )
    if action_model is None or wait_model is None:
        raise ValueError("FQI produced no model")
    return {
        "kind": kind,
        "action": action_model,
        "wait": wait_model,
        "iterations": diagnostics,
    }


def score_advantage(rows: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    transitions = build_transitions(rows)
    action_q, wait_q, _ = _predict_values(
        transitions, cast(Regressor, model["action"]), cast(Regressor, model["wait"])
    )
    output = transitions.actions.loc[transitions.valid_action].copy()
    output["raw_action_q_bps"] = action_q[transitions.valid_action]
    output["raw_wait_q_bps"] = wait_q[transitions.current_state[transitions.valid_action]]
    output["raw_advantage_bps"] = output["raw_action_q_bps"] - output["raw_wait_q_bps"]
    return output


def _bellman_targets(rows: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    transitions = build_transitions(rows)
    action_q, wait_q, value = _predict_values(
        transitions, cast(Regressor, model["action"]), cast(Regressor, model["wait"])
    )
    valid = transitions.valid_action
    next_state = transitions.next_action_state[valid]
    target_q = transitions.reward_bps[valid] + np.where(
        next_state >= 0, value[np.maximum(next_state, 0)], 0.0
    )
    output = transitions.actions.loc[valid].copy()
    output["raw_action_q_bps"] = action_q[valid]
    output["raw_wait_q_bps"] = wait_q[transitions.current_state[valid]]
    output["raw_advantage_bps"] = output["raw_action_q_bps"] - output["raw_wait_q_bps"]
    output["bellman_action_q_bps"] = target_q
    output["bellman_advantage_bps"] = target_q - output["raw_wait_q_bps"].to_numpy(float)
    return output


def _model_metrics(rows: pd.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    values = _bellman_targets(rows, model)
    td_mae = mean_absolute_error(values["bellman_action_q_bps"], values["raw_action_q_bps"])
    trades = daily._terminal_replay(values, "raw_advantage_bps")
    start = pd.to_datetime(rows["entry_timestamp"], utc=True).min().floor("D")
    end = pd.to_datetime(rows["entry_timestamp"], utc=True).max().floor("D") + pd.Timedelta(days=1)
    metrics = daily._trade_metrics(trades, start, end)
    return {
        "bellman_td_mae_bps": float(td_mae),
        "terminal_replay": metrics,
    }


def _fit_advantage_calibrator(rows: pd.DataFrame, model: dict[str, Any]) -> IsotonicRegression:
    values = _bellman_targets(rows, model)
    return cast(
        IsotonicRegression,
        IsotonicRegression(out_of_bounds="clip").fit(
            values["raw_advantage_bps"], values["bellman_advantage_bps"]
        ),
    )


def _calibrated_advantage(
    rows: pd.DataFrame, model: dict[str, Any], calibrator: IsotonicRegression
) -> pd.DataFrame:
    scored = score_advantage(rows, model)
    scored["calibrated_advantage_bps"] = calibrator.predict(
        scored["raw_advantage_bps"].to_numpy(float)
    )
    return scored


def _myopic_control(oof: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = _within_day(_complete_utc_days(oof))
    future = _within_day(_complete_utc_days(future))
    timestamp = pd.to_datetime(oof["entry_timestamp"], utc=True)
    fit = oof.loc[timestamp.lt(daily.MODEL_SELECTION_END)].copy()
    calibration = oof.loc[
        timestamp.ge(daily.MODEL_SELECTION_END) & timestamp.lt(daily.CALIBRATION_END)
    ].copy()
    fit["immediate_portfolio_bps"] = daily._portfolio_return(fit) * 10_000
    calibration["immediate_portfolio_bps"] = daily._portfolio_return(calibration) * 10_000
    model = daily._fit_ridge(fit, "immediate_portfolio_bps")
    calibrator = daily._fit_calibrator(calibration, model, "immediate_portfolio_bps")
    scored = daily._calibrated_score(future, model, calibrator, "calibrated_immediate_bps")
    return scored, {"model": model, "calibrator": calibrator}


def _within_day(rows: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(rows["entry_timestamp"], utc=True)
    planned_exit = timestamp + pd.to_timedelta(rows["horizon_seconds"], unit="s")
    return rows.loc[planned_exit.le(timestamp.dt.floor("D") + pd.Timedelta(days=1))].copy()


def train() -> dict[str, Any]:
    _status("load", "Reusing frozen 2025 Binance OOF actions", 2)
    oof = daily.load_actions(daily.OOF_ACTIONS)
    timestamp = pd.to_datetime(oof["entry_timestamp"], utc=True)
    fit = oof.loc[timestamp.lt(daily.FIT_END)].copy()
    model_selection = oof.loc[
        timestamp.ge(daily.FIT_END) & timestamp.lt(daily.MODEL_SELECTION_END)
    ].copy()
    calibration = oof.loc[
        timestamp.ge(daily.MODEL_SELECTION_END) & timestamp.lt(daily.CALIBRATION_END)
    ].copy()
    if min(len(fit), len(model_selection), len(calibration)) < 1_000_000:
        raise ValueError("insufficient chronological FQI actions")

    ridge = fit_fqi(fit, "ridge", status_start=5, status_width=15)
    xgboost = fit_fqi(fit, "xgboost_cuda", status_start=20, status_width=30)
    candidate_metrics = {
        "ridge": _model_metrics(model_selection, ridge),
        "xgboost_cuda": _model_metrics(model_selection, xgboost),
    }
    ridge_metrics = candidate_metrics["ridge"]
    xgb_metrics = candidate_metrics["xgboost_cuda"]
    champion = (
        "xgboost_cuda"
        if float(xgb_metrics["bellman_td_mae_bps"]) < float(ridge_metrics["bellman_td_mae_bps"])
        and float(xgb_metrics["terminal_replay"]["mean_daily_return"])
        > float(ridge_metrics["terminal_replay"]["mean_daily_return"])
        else "ridge"
    )

    _status("refit", f"{champion} on 2025-Q2-Q3", 55)
    fit_and_selection = oof.loc[timestamp.lt(daily.MODEL_SELECTION_END)].copy()
    model = fit_fqi(fit_and_selection, champion, status_start=55, status_width=15)
    calibrator = _fit_advantage_calibrator(calibration, model)

    _status("audit", "Model frozen; loading 2026 discovery audit", 72)
    future = daily.load_actions(daily.FUTURE_ACTIONS)
    future_timestamp = pd.to_datetime(future["entry_timestamp"], utc=True)
    if future_timestamp.ge(daily.FUTURE_HOLDOUT_START).any():
        raise ValueError("future holdout rows must remain unread")
    future = future.loc[future_timestamp.lt(daily.AUDIT_END)].copy()
    scored = _calibrated_advantage(future, model, calibrator)
    myopic_scored, myopic_bundle = _myopic_control(oof, future)

    _status("managed_replay", "Exact 5-second Binance management", 82)
    trades = daily._managed_replay(scored, "calibrated_advantage_bps")
    myopic_trades = daily._managed_replay(myopic_scored, "calibrated_immediate_bps")
    audit_start = pd.Timestamp("2026-01-01T00:00:00Z")
    metrics = daily._trade_metrics(trades, audit_start, daily.AUDIT_END)
    myopic_metrics = daily._trade_metrics(myopic_trades, audit_start, daily.AUDIT_END)
    gates = daily._gates(metrics, myopic_metrics)
    passed = all(gates.values())

    AUDIT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_TRADES.with_suffix(".parquet.tmp")
    trades.to_parquet(temporary, index=False)
    temporary.replace(AUDIT_TRADES)
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "oof_rows": len(oof),
            "fit_rows": len(fit),
            "model_selection_rows": len(model_selection),
            "calibration_rows": len(calibration),
            "audit_rows": len(future),
            "future_holdout_rows_read": 0,
        },
        "causal_checks": {
            "realized_future_maximum_used": False,
            "bellman_next_value_from_previous_model": True,
            "realized_return_excluded_from_features": "net_bps" not in MODEL_FEATURES,
            "flat_has_separate_wait_model": True,
            "one_position_enforced": True,
            "test_used_for_selection_or_calibration": False,
        },
        "candidate_metrics_2025_q3": candidate_metrics,
        "champion": champion,
        "fitted_q_iterations": model["iterations"],
        "audit_discovery_2026": {
            "fitted_q_policy": metrics,
            "myopic_same_split_control": myopic_metrics,
            "gates": gates,
        },
        "verdict": (
            "FQI_DAILY_PORTFOLIO_RESEARCH_CHALLENGER"
            if passed
            else "NO_INCREMENTAL_FQI_DAILY_ALPHA"
        ),
        "paper_policy_changed": False,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "future_holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_joblib(
        BUNDLE,
        {
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "model": model,
            "calibrator": calibrator,
            "myopic_control": myopic_bundle,
            "champion": champion,
            "research_only": True,
            "paper_orders_enabled": False,
            "live_orders_enabled": False,
        },
    )
    _atomic_json(REPORT, report)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


if __name__ == "__main__":
    train()
