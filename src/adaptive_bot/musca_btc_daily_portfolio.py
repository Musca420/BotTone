from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot import musca_btc_moe as base

OOF_ACTIONS = base.CHECKPOINTS / "oof_actions.parquet"
FUTURE_ACTIONS = base.CHECKPOINTS / "future_actions.parquet"
ROOT = Path("data/ml/musca_btc_daily_portfolio")
REPORT = Path("data/reports/musca_btc_daily_portfolio.json")
STATUS = Path("data/reports/musca_btc_daily_portfolio.status.json")
BUNDLE = Path("data/models/musca_btc_daily_portfolio/research_bundle.joblib")
AUDIT_TRADES = ROOT / "audit_trades.parquet"

FIT_END = pd.Timestamp("2025-07-01T00:00:00Z")
MODEL_SELECTION_END = pd.Timestamp("2025-10-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-08-01T00:00:00Z")
FUTURE_HOLDOUT_START = base.FUTURE_HOLDOUT_START

RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.02
MAXIMUM_LEVERAGE = 10.0
ROUND_TRIP_COST_BPS = base.ROUND_TRIP_COST_BPS
XGB_SEED = 20260810

STATE_FEATURES = (
    "minutes_to_utc_close",
    "horizon_to_remaining_day",
    "portfolio_leverage",
)
FEATURES = (*base.META_FEATURES, *STATE_FEATURES)
REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            *base.META_FEATURES,
            "available_at",
            "entry_timestamp",
            "decision_position",
            "exit_timestamp",
            "side",
            "horizon_seconds",
            "target_1_bps",
            "target_2_bps",
            "stop_bps",
            "trailing_bps",
            "funding_bps",
            "gross_bps",
            "net_bps",
            "protocol_hash",
        )
    )
)

PROTOCOL = {
    "name": "musca_btc_finite_horizon_daily_portfolio_q_v1",
    "parent_protocol_hash": base.PROTOCOL_HASH,
    "positive_control_protocol_hash": (
        "a195b75e41bf7bfd40dd2ca23103720cee1ea5dfbc7333cd34dcca4c7171a360"
    ),
    "symbol": "BTCUSDT",
    "venue": "Binance USD-M futures",
    "question": (
        "Given market state, available experts, realized daily PnL, residual risk and open "
        "position, choose the action maximizing expected end-of-day net equity after costs"
    ),
    "source": {
        "fit_and_selection": str(OOF_ACTIONS),
        "audit": str(FUTURE_ACTIONS),
        "source_protocol_hash": base.PROTOCOL_HASH,
    },
    "objective": (
        "finite-horizon semi-Markov action advantage over waiting one decision; "
        "portfolio log return after Binance costs and stop-based sizing"
    ),
    "actions": {
        "sides": ["LONG", "SHORT"],
        "horizons_seconds": list(base.HORIZONS),
        "flat": "implicit next decision value",
        "open_position": "manage frozen TP1/TP2/stop/non-widening trail; no second position",
    },
    "features": list(FEATURES),
    "chronology": {
        "fit_end": FIT_END.isoformat(),
        "model_selection_end": MODEL_SELECTION_END.isoformat(),
        "calibration_end": CALIBRATION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
    },
    "models": {
        "champion_default": "ridge",
        "challenger": "xgboost_cuda_pseudohuber",
        "challenger_rule": (
            "strictly lower Q MAE and decision regret plus higher terminal replay daily return "
            "on 2025-Q3"
        ),
        "threshold_search": False,
        "entry_threshold": "calibrated advantage strictly greater than zero",
    },
    "economics": {
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "stress_round_trip_cost_bps": 2 * ROUND_TRIP_COST_BPS,
        "risk_per_trade": RISK_PER_TRADE,
        "maximum_leverage": MAXIMUM_LEVERAGE,
        "maximum_daily_loss": MAX_DAILY_LOSS,
        "maximum_positions": 1,
    },
    "gates": {
        "expectancy_bps": 0,
        "profit_factor": 1.15,
        "maximum_drawdown": 0.08,
        "positive_active_days": 0.50,
        "stress_2x_expectancy_bps": 0,
        "bootstrap_daily_lcb": 0,
        "risk_budget_violations": 0,
        "beats_myopic_daily_return": True,
    },
    "future_holdout_opened": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class Regressor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


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


def _portfolio_leverage(stop_bps: np.ndarray) -> np.ndarray:
    risk_distance = (np.asarray(stop_bps, dtype=float) + ROUND_TRIP_COST_BPS) / 10_000
    return np.asarray(
        np.minimum(MAXIMUM_LEVERAGE, RISK_PER_TRADE / np.maximum(risk_distance, 1e-9)),
        dtype=float,
    )


def _portfolio_return(rows: pd.DataFrame, return_column: str = "net_bps") -> np.ndarray:
    leverage = _portfolio_leverage(rows["stop_bps"].to_numpy(float))
    return np.asarray(leverage * rows[return_column].to_numpy(float) / 10_000, dtype=float)


def _derive_state_features(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    timestamp = pd.to_datetime(output["entry_timestamp"], utc=True)
    close = timestamp.dt.floor("D") + pd.Timedelta(days=1)
    remaining_seconds = (close - timestamp).dt.total_seconds().to_numpy(float)
    output["minutes_to_utc_close"] = remaining_seconds / 60
    output["horizon_to_remaining_day"] = output["horizon_seconds"].to_numpy(float) / np.maximum(
        remaining_seconds, 1.0
    )
    output["portfolio_leverage"] = _portfolio_leverage(output["stop_bps"].to_numpy(float))
    return output


def load_actions(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    hashes = set(rows["protocol_hash"].astype(str).unique())
    if hashes != {base.PROTOCOL_HASH}:
        raise ValueError(f"unexpected parent protocol hashes: {sorted(hashes)}")
    for column in ("available_at", "entry_timestamp", "exit_timestamp"):
        rows[column] = pd.to_datetime(rows[column], utc=True)
    rows = _derive_state_features(rows)
    values = rows.loc[:, FEATURES].to_numpy(np.float32)
    finite = np.isfinite(values).all(axis=1)
    labels = np.isfinite(rows["net_bps"].to_numpy(float))
    rows = rows.loc[finite & labels].copy()
    return rows.sort_values(
        ["entry_timestamp", "side", "horizon_seconds"], kind="stable"
    ).reset_index(drop=True)


def _daily_advantage_targets(rows: pd.DataFrame) -> pd.DataFrame:
    """Build a non-tradable backward oracle separately inside each UTC day."""
    output = rows.sort_values(
        ["entry_timestamp", "side", "horizon_seconds"], kind="stable"
    ).reset_index(drop=True)
    rewards = _portfolio_return(output)
    if np.any(rewards <= -0.999):
        raise ValueError("portfolio reward at or below total loss")
    reward_log = np.log1p(rewards)
    targets = np.full(len(output), np.nan, dtype=float)
    timestamps = pd.to_datetime(output["entry_timestamp"], utc=True)
    days = timestamps.dt.floor("D")

    for day, indexes_value in output.groupby(days, sort=True).groups.items():
        indexes = np.asarray(indexes_value, dtype=int)
        day_rows = output.iloc[indexes]
        day_time_ns = (
            pd.to_datetime(day_rows["entry_timestamp"], utc=True)
            .astype("datetime64[ns, UTC]")
            .astype("int64")
            .to_numpy(np.int64)
        )
        unique_times, counts = np.unique(day_time_ns, return_counts=True)
        action_count = len(base.HORIZONS) * len(base.SIDES)
        if not np.all(counts == action_count):
            raise ValueError("each decision must expose every side/horizon action")
        horizon = day_rows["horizon_seconds"].to_numpy(int).reshape(-1, action_count)
        if not np.all(horizon == horizon[0]):
            raise ValueError("side/horizon action ordering must be stable")
        reward = reward_log[indexes].reshape(-1, action_count)
        exit_values = unique_times[:, None] + horizon.astype(np.int64) * 1_000_000_000
        next_position = np.searchsorted(unique_times, exit_values, side="left")
        value = np.zeros(len(unique_times) + 1, dtype=float)
        day_end = pd.Timestamp(cast(Any, day)) + pd.Timedelta(days=1)
        valid = exit_values <= day_end.value
        day_targets = np.full_like(reward, np.nan)
        for position in range(len(unique_times) - 1, -1, -1):
            flat_value = value[position + 1]
            q_value = reward[position] + value[next_position[position]]
            q_value[~valid[position]] = -np.inf
            day_targets[position] = np.where(
                valid[position], (q_value - flat_value) * 10_000, np.nan
            )
            value[position] = max(flat_value, float(np.max(q_value, initial=-np.inf)))
        targets[indexes] = day_targets.reshape(-1)

    output["daily_q_advantage_bps"] = targets
    return output


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("daily portfolio features must be finite")
    return values


def _target(rows: pd.DataFrame, name: str) -> np.ndarray:
    values = rows[name].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"target {name} must be finite")
    return values


def _fit_ridge(rows: pd.DataFrame, target: str) -> Regressor:
    return cast(
        Regressor,
        make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(_x(rows), _target(rows, target)),
    )


def _fit_xgboost(rows: pd.DataFrame, target: str) -> Regressor:
    return cast(
        Regressor,
        XGBRegressor(
            objective="reg:pseudohubererror",
            tree_method="hist",
            device="cuda",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=XGB_SEED,
        ).fit(_x(rows), _target(rows, target), verbose=False),
    )


def _score(rows: pd.DataFrame, model: Regressor, name: str = "raw_score_bps") -> pd.DataFrame:
    output = rows.copy()
    output[name] = np.asarray(model.predict(_x(output)), dtype=float)
    return output


def _decision_regret(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = rows.loc[:, ["entry_timestamp", "daily_q_advantage_bps"]].copy()
    scored["prediction"] = prediction
    selected = scored.loc[scored.groupby("entry_timestamp", sort=False)["prediction"].idxmax()]
    selected_value = np.where(
        selected["prediction"].to_numpy(float) > 0,
        selected["daily_q_advantage_bps"].to_numpy(float),
        0.0,
    )
    oracle = (
        scored.groupby("entry_timestamp", sort=False)["daily_q_advantage_bps"]
        .max()
        .clip(lower=0)
        .to_numpy(float)
    )
    return float(np.mean(oracle - selected_value))


def _terminal_replay(scored: pd.DataFrame, score_column: str) -> pd.DataFrame:
    winners = (
        scored.sort_values(
            ["entry_timestamp", score_column, "side", "horizon_seconds"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("entry_timestamp", keep="first")
        .loc[lambda value: value[score_column].gt(0)]
    )
    accepted: list[int] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    equity = 1.0
    current_day: pd.Timestamp | None = None
    day_start_equity = 1.0
    trades_today = 0
    risk_blocked = 0
    for index, row in winners.iterrows():
        entry = pd.Timestamp(row["entry_timestamp"])
        day = entry.floor("D")
        if current_day is None or day != current_day:
            current_day = day
            day_start_equity = equity
            trades_today = 0
        if entry < free_at:
            continue
        leverage = float(_portfolio_leverage(np.asarray([float(row["stop_bps"])]))[0])
        worst_risk = leverage * (float(row["stop_bps"]) + ROUND_TRIP_COST_BPS) / 10_000
        residual = equity / day_start_equity - (1 - MAX_DAILY_LOSS)
        if residual + 1e-12 < worst_risk:
            risk_blocked += 1
            continue
        trade_return = leverage * float(row["net_bps"]) / 10_000
        accepted.append(cast(int, index))
        equity *= 1 + trade_return
        free_at = entry + pd.Timedelta(seconds=int(row["horizon_seconds"]))
        trades_today += 1
    trades = winners.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)
    if trades.empty:
        return trades
    trades["portfolio_return"] = _portfolio_return(trades)
    trades["stress_2x_bps"] = trades["net_bps"] - ROUND_TRIP_COST_BPS
    trades["selection_exit_timestamp"] = trades["entry_timestamp"] + pd.to_timedelta(
        trades["horizon_seconds"], unit="s"
    )
    trades.attrs["risk_blocked_decisions"] = risk_blocked
    return trades


def _model_metrics(rows: pd.DataFrame, model: Regressor) -> dict[str, float]:
    prediction = np.asarray(model.predict(_x(rows)), dtype=float)
    actual = rows["daily_q_advantage_bps"].to_numpy(float)
    trades = _terminal_replay(rows.assign(raw_score_bps=prediction), "raw_score_bps")
    daily = _daily_returns(trades, rows["entry_timestamp"].min(), rows["entry_timestamp"].max())
    return {
        "mae_bps": float(mean_absolute_error(actual, prediction)),
        "decision_regret_bps": _decision_regret(rows, prediction),
        "terminal_replay_mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
        "terminal_replay_trades": float(len(trades)),
    }


def _fit_calibrator(rows: pd.DataFrame, model: Regressor, target: str) -> IsotonicRegression:
    prediction = np.asarray(model.predict(_x(rows)), dtype=float)
    return cast(
        IsotonicRegression,
        IsotonicRegression(out_of_bounds="clip").fit(prediction, _target(rows, target)),
    )


def _calibrated_score(
    rows: pd.DataFrame, model: Regressor, calibrator: IsotonicRegression, name: str
) -> pd.DataFrame:
    output = rows.copy()
    output["raw_score_bps"] = np.asarray(model.predict(_x(output)), dtype=float)
    output[name] = calibrator.predict(output["raw_score_bps"].to_numpy(float))
    return output


def _daily_returns(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    calendar = pd.date_range(start.floor("D"), end.floor("D"), freq="1D", inclusive="left")
    if trades.empty:
        return pd.Series(0.0, index=calendar)
    daily = (
        trades.assign(day=pd.to_datetime(trades["entry_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["portfolio_return"]
        .apply(lambda value: float(np.prod(1 + value.to_numpy(float)) - 1))
    )
    return daily.reindex(calendar, fill_value=0.0)


def _bootstrap_daily_lcb(daily: pd.Series, seed: int = XGB_SEED) -> float | None:
    values = daily.to_numpy(float)
    if len(values) < 20:
        return None
    rng = np.random.default_rng(seed)
    block = min(5, len(values))
    means = np.empty(1_000, dtype=float)
    for draw in range(len(means)):
        sampled: list[np.ndarray] = []
        while sum(len(value) for value in sampled) < len(values):
            start = int(rng.integers(0, len(values) - block + 1))
            sampled.append(values[start : start + block])
        means[draw] = np.concatenate(sampled)[: len(values)].mean()
    return float(np.quantile(means, 0.05))


def _spa_pvalue(daily: pd.Series) -> float | None:
    if len(daily) < 20:
        return None
    test = SPA(
        np.zeros(len(daily)),
        (-daily.to_numpy(float)).reshape(-1, 1),
        block_size=min(5, len(daily)),
        reps=1_000,
        bootstrap="stationary",
        seed=XGB_SEED,
    )
    test.compute()
    return float(test.pvalues["consistent"])


def _trade_metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    daily = _daily_returns(trades, start, end)
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "mean_daily_return": 0.0,
            "positive_active_days": 0.0,
            "maximum_drawdown": None,
            "stress_2x_expectancy_bps": None,
            "bootstrap_daily_lcb": None,
            "spa_pvalue": None,
            "risk_budget_violations": 0,
        }
    net = trades["net_bps"].to_numpy(float)
    portfolio = trades["portfolio_return"].to_numpy(float)
    gains = portfolio[portfolio > 0].sum()
    losses = -portfolio[portfolio < 0].sum()
    equity = np.cumprod(1 + portfolio)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    active = daily.loc[daily.ne(0)]
    return {
        "trades": len(trades),
        "trades_per_day": float(len(trades) / max(1, len(daily))),
        "expectancy_bps": float(net.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((portfolio > 0).mean()),
        "mean_daily_return": float(daily.mean()),
        "positive_calendar_days": float(daily.gt(0).mean()),
        "positive_active_days": float(active.gt(0).mean()) if len(active) else 0.0,
        "maximum_drawdown": float((1 - equity / peak).max(initial=0.0)),
        "stress_2x_expectancy_bps": float(trades["stress_2x_bps"].mean()),
        "bootstrap_daily_lcb": _bootstrap_daily_lcb(daily),
        "spa_pvalue": _spa_pvalue(daily),
        "risk_budget_violations": int(daily.lt(-MAX_DAILY_LOSS - 1e-9).sum()),
        "long": int(trades["side"].gt(0).sum()),
        "short": int(trades["side"].lt(0).sum()),
        "horizons": {
            str(key): int(value) for key, value in trades["horizon_seconds"].value_counts().items()
        },
    }


def _managed_replay(scored: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Replay selected actions with the exact frozen 5-second management engine."""
    winners = (
        scored.sort_values(
            ["entry_timestamp", score_column, "side", "horizon_seconds"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("entry_timestamp", keep="first")
        .loc[lambda value: value[score_column].gt(0)]
    )
    source = base._load_micro_source()
    funding = base._funding_curve()
    accepted: list[pd.Series[Any]] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    equity = 1.0
    current_day: pd.Timestamp | None = None
    day_start_equity = 1.0
    for _, row in winners.iterrows():
        entry = pd.Timestamp(row["entry_timestamp"])
        day = entry.floor("D")
        if current_day is None or day != current_day:
            current_day = day
            day_start_equity = equity
        if entry < free_at:
            continue
        leverage = float(_portfolio_leverage(np.asarray([float(row["stop_bps"])]))[0])
        worst_risk = leverage * (float(row["stop_bps"]) + ROUND_TRIP_COST_BPS) / 10_000
        residual = equity / day_start_equity - (1 - MAX_DAILY_LOSS)
        if residual + 1e-12 < worst_risk:
            continue
        gross, exit_seconds, outcome = base._simulate_management(
            source,
            np.asarray([int(row["decision_position"])]),
            int(row["side"]),
            int(row["horizon_seconds"]),
            np.asarray([float(row["target_1_bps"])]),
            np.asarray([float(row["target_2_bps"])]),
            np.asarray([float(row["stop_bps"])]),
            np.asarray([float(row["trailing_bps"])]),
        )
        actual_exit = entry + pd.Timedelta(seconds=int(exit_seconds[0]))
        funding_bps = float(
            base._funding_pnl_bps(
                pd.Series([entry]),
                pd.Series([actual_exit]),
                np.asarray([int(row["side"])]),
                funding,
            )[0]
        )
        net_bps = float(gross[0]) + funding_bps - ROUND_TRIP_COST_BPS
        trade_return = leverage * net_bps / 10_000
        chosen = row.copy()
        chosen["gross_bps"] = float(gross[0])
        chosen["funding_bps"] = funding_bps
        chosen["net_bps"] = net_bps
        chosen["stress_2x_bps"] = net_bps - ROUND_TRIP_COST_BPS
        chosen["exit_seconds"] = int(exit_seconds[0])
        chosen["exit_timestamp"] = actual_exit
        chosen["outcome"] = str(outcome[0])
        chosen["portfolio_return"] = trade_return
        chosen["daily_pnl_before"] = equity / day_start_equity - 1
        chosen["risk_remaining_before"] = residual
        chosen["position_state_before"] = "FLAT"
        accepted.append(chosen)
        equity *= 1 + trade_return
        free_at = actual_exit
    return pd.DataFrame(accepted).reset_index(drop=True) if accepted else winners.iloc[:0].copy()


def _gates(metrics: dict[str, Any], myopic: dict[str, Any]) -> dict[str, bool]:
    return {
        "expectancy_positive": float(metrics.get("expectancy_bps") or 0) > 0,
        "profit_factor_1_15": float(metrics.get("profit_factor") or 0) >= 1.15,
        "drawdown_8pct": float(metrics.get("maximum_drawdown") or 1) <= 0.08,
        "majority_positive_active_days": float(metrics.get("positive_active_days") or 0) > 0.5,
        "stress_2x_nonnegative": float(metrics.get("stress_2x_expectancy_bps") or -1) >= 0,
        "bootstrap_daily_lcb_positive": float(metrics.get("bootstrap_daily_lcb") or -1) > 0,
        "risk_budget_respected": int(metrics.get("risk_budget_violations", 1)) == 0,
        "beats_myopic_daily_return": float(metrics.get("mean_daily_return") or 0)
        > float(myopic.get("mean_daily_return") or 0),
    }


def train() -> dict[str, Any]:
    _status("load", "Existing Binance OOF actions; no download and no regeneration", 2)
    oof = _daily_advantage_targets(load_actions(OOF_ACTIONS))
    valid_q = oof["daily_q_advantage_bps"].notna()
    oof = oof.loc[valid_q].copy()
    timestamp = pd.to_datetime(oof["entry_timestamp"], utc=True)
    fit = oof.loc[timestamp.lt(FIT_END)].copy()
    model_selection = oof.loc[timestamp.ge(FIT_END) & timestamp.lt(MODEL_SELECTION_END)].copy()
    calibration = oof.loc[timestamp.ge(MODEL_SELECTION_END) & timestamp.lt(CALIBRATION_END)].copy()
    if min(len(fit), len(model_selection), len(calibration)) < 100_000:
        raise ValueError("insufficient chronological OOF daily-Q rows")

    _status("model_selection", f"Ridge baseline on {len(fit):,} Q2 actions", 18)
    candidates: dict[str, Regressor] = {"ridge": _fit_ridge(fit, "daily_q_advantage_bps")}
    _status("model_selection", "One preregistered XGBoost CUDA challenger", 30)
    candidates["xgboost_cuda"] = _fit_xgboost(fit, "daily_q_advantage_bps")
    candidate_metrics = {
        name: _model_metrics(model_selection, model) for name, model in candidates.items()
    }
    ridge_metrics = candidate_metrics["ridge"]
    xgb_metrics = candidate_metrics["xgboost_cuda"]
    champion = (
        "xgboost_cuda"
        if float(xgb_metrics["mae_bps"]) < float(ridge_metrics["mae_bps"])
        and float(xgb_metrics["decision_regret_bps"]) < float(ridge_metrics["decision_regret_bps"])
        and float(xgb_metrics["terminal_replay_mean_daily_return"])
        > float(ridge_metrics["terminal_replay_mean_daily_return"])
        else "ridge"
    )

    fit_and_selection = oof.loc[timestamp.lt(MODEL_SELECTION_END)].copy()
    _status("refit", f"{champion} frozen using Q2-Q3 only", 50)
    model = (
        _fit_xgboost(fit_and_selection, "daily_q_advantage_bps")
        if champion == "xgboost_cuda"
        else _fit_ridge(fit_and_selection, "daily_q_advantage_bps")
    )
    calibrator = _fit_calibrator(calibration, model, "daily_q_advantage_bps")

    _status("myopic_control", "Same Ridge/splits with immediate return target", 60)
    oof["immediate_portfolio_bps"] = _portfolio_return(oof) * 10_000
    fit_and_selection = oof.loc[timestamp.lt(MODEL_SELECTION_END)].copy()
    calibration = oof.loc[timestamp.ge(MODEL_SELECTION_END) & timestamp.lt(CALIBRATION_END)].copy()
    myopic_model = _fit_ridge(fit_and_selection, "immediate_portfolio_bps")
    myopic_calibrator = _fit_calibrator(calibration, myopic_model, "immediate_portfolio_bps")

    _status("audit", "Loading untouched 2026 actions after model and calibration freeze", 68)
    future = load_actions(FUTURE_ACTIONS)
    future_timestamp = pd.to_datetime(future["entry_timestamp"], utc=True)
    if future_timestamp.ge(FUTURE_HOLDOUT_START).any():
        raise ValueError("future holdout rows must remain unread")
    future = future.loc[future_timestamp.lt(AUDIT_END)].copy()
    future_q = _daily_advantage_targets(future)
    future = future.loc[future_q["daily_q_advantage_bps"].notna()].copy()
    future["daily_q_advantage_bps"] = future_q.loc[
        future_q["daily_q_advantage_bps"].notna(), "daily_q_advantage_bps"
    ].to_numpy(float)
    q_scored = _calibrated_score(future, model, calibrator, "calibrated_advantage_bps")
    myopic_scored = _calibrated_score(
        future, myopic_model, myopic_calibrator, "calibrated_immediate_bps"
    )

    _status("managed_replay", "Exact 5-second TP/stop/trailing replay; one position", 78)
    q_trades = _managed_replay(q_scored, "calibrated_advantage_bps")
    myopic_trades = _managed_replay(myopic_scored, "calibrated_immediate_bps")
    audit_start = pd.Timestamp("2026-01-01T00:00:00Z")
    q_metrics = _trade_metrics(q_trades, audit_start, AUDIT_END)
    myopic_metrics = _trade_metrics(myopic_trades, audit_start, AUDIT_END)
    gates = _gates(q_metrics, myopic_metrics)
    passed = all(gates.values())

    ROOT.mkdir(parents=True, exist_ok=True)
    temporary_trades = AUDIT_TRADES.with_suffix(".parquet.tmp")
    q_trades.to_parquet(temporary_trades, index=False)
    temporary_trades.replace(AUDIT_TRADES)

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
            "audit_days": int(future_timestamp.dt.floor("D").nunique()),
            "future_holdout_rows_read": 0,
        },
        "causal_checks": {
            "features_exclude_realized_action_return": "net_bps" not in FEATURES,
            "daily_oracle_used_only_as_training_target": True,
            "utc_day_boundary_known_at_decision": True,
            "actions_crossing_day_excluded": True,
            "one_position_enforced": True,
            "test_not_used_for_model_or_calibrator": True,
        },
        "candidate_model_metrics_2025_q3": candidate_metrics,
        "champion": champion,
        "audit_discovery_2026": {
            "daily_q_policy": q_metrics,
            "myopic_same_split_control": myopic_metrics,
            "gates": gates,
            "oracle_is_not_tradable": True,
        },
        "verdict": (
            "DAILY_PORTFOLIO_RESEARCH_CHALLENGER" if passed else "NO_INCREMENTAL_DAILY_Q_ALPHA"
        ),
        "paper_policy_changed": False,
        "future_holdout_opened": False,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "real_capital_allowed": False,
    }
    _atomic_joblib(
        BUNDLE,
        {
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "features": FEATURES,
            "model": model,
            "calibrator": calibrator,
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
