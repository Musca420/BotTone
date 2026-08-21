from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL_START = pd.Timestamp("2026-08-08T09:45:00Z")
EXPECTED_OUTCOME_PROTOCOL_HASH = (
    "ad7cc649566082df60768274aad0c0c63942452fc50cfab32550e311ee4c9a0f"
)
FEATURES = (
    "median_return_1m_bps",
    "median_return_5m_bps",
    "dispersion_return_1m_bps",
    "dispersion_return_5m_bps",
    "rolling_vwap_5m_distance_bps",
    "spread_bps",
    "range_60s_bps",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_15s",
    "trade_arrival_rate_30s",
    "depth_imbalance_5bps",
    "depth_imbalance_10bps",
    "depth_imbalance_20bps",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
    "spread_change_bps",
)
EXPERTS = (
    "anchor_continuation_dynamic",
    "anchor_failure_dynamic",
    "pullback_continuation_dynamic",
    "vwap_reversion_dynamic",
)
FIT_DECISIONS = 200
CALIBRATION_DECISIONS = 60
CHAMPION_AUDIT_DECISIONS = 60
MINIMUM_HOLDOUT_TRADES = 100
MINIMUM_HOLDOUT_DAYS = 10
MODEL_ROOT = Path("data/models/btc_vwap_forward_selector")
BUNDLE = MODEL_ROOT / "research_selector_v4.joblib"
REPORT = Path("data/reports/btc_vwap_forward_selector.json")
BARRIERS = (20, 30, 50)
OUTCOME_COLUMNS = (
    *(f"mfe_{minutes}m_bps" for minutes in (5, 15, 30, 60)),
    *(f"mae_{minutes}m_bps" for minutes in (5, 15, 30, 60)),
    *(f"time_to_{barrier}bps_seconds" for barrier in (10, 20, 30, 50)),
)
HORIZON_PROBABILITY_COLUMNS = tuple(
    f"target_30bps_within_{minutes}m" for minutes in (5, 15, 30, 60)
)
PROTOCOL = {
    "name": "btc_vwap_forward_selector_v2_bitunix",
    "protocol_start": PROTOCOL_START.isoformat(),
    "outcome_protocol_hash": EXPECTED_OUTCOME_PROTOCOL_HASH,
    "features": list(FEATURES),
    "experts": list(EXPERTS),
    "blocks": {
        "fit_decision_timestamps": FIT_DECISIONS,
        "calibration_decision_timestamps": CALIBRATION_DECISIONS,
        "champion_audit_decision_timestamps": CHAMPION_AUDIT_DECISIONS,
        "holdout": "all later rows; minimum 100 selected trades and 10 UTC days",
    },
    "target": "net_bps (book execution, taker entry/exit, reserve and realized funding)",
    "champion": (
        "Ridge unless five-seed CUDA XGBoost strictly wins EV error, calibration, "
        "decision regret and LCB regret"
    ),
    "decision": "highest positive lower confidence EV among causal candidates; otherwise FLAT=0",
    "missing_data": "fail closed; no imputation",
    "live_orders_enabled": False,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.joblib")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 1_000:
        return pd.DataFrame()
    rows = pd.read_parquet(path)
    required = {
        "protocol_hash",
        "signal_at",
        "feature_available_at",
        "entry_at",
        "exit_at",
        "expert",
        "side",
        "gross_bps",
        "net_bps",
        "stress_bps",
        "stop_bps",
        "fee_bps",
        "slippage_reserve_bps",
        *(f"target_{barrier}bps_before_stop" for barrier in BARRIERS),
        *HORIZON_PROBABILITY_COLUMNS,
        *OUTCOME_COLUMNS,
        *FEATURES,
    }
    if not required.issubset(rows.columns):
        return pd.DataFrame()
    signal = pd.to_datetime(rows["signal_at"], utc=True)
    available = pd.to_datetime(rows["feature_available_at"], utc=True)
    entry = pd.to_datetime(rows["entry_at"], utc=True)
    exit_at = pd.to_datetime(rows["exit_at"], utc=True)
    valid = (
        rows["protocol_hash"].eq(EXPECTED_OUTCOME_PROTOCOL_HASH)
        & signal.ge(PROTOCOL_START)
        & available.le(signal)
        & entry.gt(signal)
        & exit_at.ge(entry)
        & rows["expert"].isin(EXPERTS)
        & rows[
            [
                *FEATURES,
                "gross_bps",
                "net_bps",
                "stress_bps",
                "stop_bps",
                "fee_bps",
                "slippage_reserve_bps",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    clean = rows.loc[valid].copy()
    if clean.empty:
        return clean
    numeric = clean[
        [
            *FEATURES,
            "gross_bps",
            "net_bps",
            "stress_bps",
            "stop_bps",
            "fee_bps",
            "slippage_reserve_bps",
        ]
    ].to_numpy(float)
    clean = clean.loc[np.isfinite(numeric).all(axis=1)].copy()
    return clean.drop_duplicates(["signal_at", "expert"], keep="last").sort_values(
        ["signal_at", "expert"]
    ).reset_index(drop=True)


def design(rows: pd.DataFrame) -> np.ndarray:
    numeric = rows.loc[:, FEATURES].to_numpy(float)
    side = np.where(rows["side"].eq("LONG"), 1.0, -1.0)[:, None]
    experts = np.column_stack(
        [rows["expert"].eq(expert).to_numpy(float) for expert in EXPERTS]
    )
    values = np.column_stack((numeric, side, experts))
    if not np.isfinite(values).all():
        raise ValueError("selector features must be completely covered and finite")
    return values


def _bootstrap_indexes(size: int, seed: int, block: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = min(block, size)
    starts = np.arange(max(1, size - width + 1))
    pieces = [
        np.arange(start, min(start + width, size))
        for start in rng.choice(starts, size=int(np.ceil(size / width)))
    ]
    return np.concatenate(pieces)[:size]


def _mean_lower_bound(values: np.ndarray, seed: int) -> float:
    width = min(20, len(values))
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - width + 1))
    means = np.empty(1_000)
    for position in range(len(means)):
        chosen = rng.choice(starts, size=int(np.ceil(len(values) / width)))
        sample = np.concatenate([values[start : start + width] for start in chosen])[
            : len(values)
        ]
        means[position] = sample.mean()
    return float(np.quantile(means, 0.05))


def _predict(models: list[Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([np.asarray(model.predict(x), dtype=float) for model in models])
    return values.mean(axis=0), values.std(axis=0)


def _fit_outcomes(fitting: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    """Fit interpretable barrier and excursion heads only from chronological observations."""
    x_fit, x_cal = design(fitting), design(calibration)
    barrier_models: dict[int, dict[str, Any]] = {}
    for barrier in BARRIERS:
        column = f"target_{barrier}bps_before_stop"
        y_fit = fitting[column].astype(int).to_numpy()
        y_cal = calibration[column].astype(int).to_numpy()
        if min(np.bincount(y_fit, minlength=2)) < 5 or len(np.unique(y_cal)) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, random_state=42),
        ).fit(x_fit, y_fit)
        raw = model.predict_proba(x_cal)[:, 1]
        calibrator = LogisticRegression(C=1.0, random_state=42).fit(
            raw.reshape(-1, 1), y_cal
        )
        barrier_models[barrier] = {"model": model, "calibrator": calibrator}
    horizon_models: dict[str, dict[str, Any]] = {}
    for column in HORIZON_PROBABILITY_COLUMNS:
        y_fit = fitting[column].astype(int).to_numpy()
        y_cal = calibration[column].astype(int).to_numpy()
        if min(np.bincount(y_fit, minlength=2)) < 5 or len(np.unique(y_cal)) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, random_state=42),
        ).fit(x_fit, y_fit)
        raw = model.predict_proba(x_cal)[:, 1]
        horizon_models[column] = {
            "model": model,
            "calibrator": LogisticRegression(C=1.0, random_state=42).fit(
                raw.reshape(-1, 1), y_cal
            ),
        }
    regressions: dict[str, Any] = {}
    for column in OUTCOME_COLUMNS:
        available = fitting[column].notna()
        if int(available.sum()) < 30:
            continue
        target = fitting.loc[available, column].to_numpy(float)
        log_target = column.startswith("time_to_")
        if log_target:
            target = np.log1p(target)
        regressions[column] = {
            "model": make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
                design(fitting.loc[available]), target
            ),
            "log_target": log_target,
        }
    return {
        "barriers": barrier_models,
        "horizons": horizon_models,
        "regressions": regressions,
    }


def _predict_outcomes(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    x = design(rows)
    outcomes = bundle.get("outcomes", {})
    for barrier, pair in outcomes.get("barriers", {}).items():
        raw = pair["model"].predict_proba(x)[:, 1]
        result[f"p_target_{barrier}bps_before_stop"] = pair[
            "calibrator"
        ].predict_proba(raw.reshape(-1, 1))[:, 1]
    for column, pair in outcomes.get("horizons", {}).items():
        raw = pair["model"].predict_proba(x)[:, 1]
        result[f"p_{column}"] = pair["calibrator"].predict_proba(
            raw.reshape(-1, 1)
        )[:, 1]
    for column, item in outcomes.get("regressions", {}).items():
        values = np.asarray(item["model"].predict(x), dtype=float)
        result[f"expected_{column}"] = np.expm1(values) if item["log_target"] else values
    return result


def _decision_regret(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = rows.loc[:, ["signal_at", "net_bps"]].copy()
    scored["prediction"] = prediction
    regrets: list[float] = []
    for _, group in scored.groupby("signal_at", sort=True):
        actual = group["net_bps"].to_numpy(float)
        predicted = group["prediction"].to_numpy(float)
        chosen = int(np.argmax(predicted))
        chosen_value = actual[chosen] if predicted[chosen] > 0 else 0.0
        regrets.append(max(0.0, float(actual.max())) - chosen_value)
    return float(np.mean(regrets)) if regrets else float("inf")


def _audit_metrics(
    rows: pd.DataFrame,
    prediction: np.ndarray,
    lower_adjustment: float,
) -> dict[str, float]:
    truth = rows["net_bps"].to_numpy(float)
    return {
        "ev_mse": float(np.mean((prediction - truth) ** 2)),
        "calibration_error": float(abs(np.mean(prediction - truth))),
        "decision_regret": _decision_regret(rows, prediction),
        "lcb_decision_regret": _decision_regret(rows, prediction + lower_adjustment),
    }


def _fit_bundle(rows: pd.DataFrame) -> dict[str, Any] | None:
    signals = pd.DatetimeIndex(pd.to_datetime(rows["signal_at"], utc=True).unique()).sort_values()
    boundary_calibration = pd.Timestamp(signals[FIT_DECISIONS])
    boundary_audit = pd.Timestamp(signals[FIT_DECISIONS + CALIBRATION_DECISIONS])
    audit_end = pd.Timestamp(
        signals[FIT_DECISIONS + CALIBRATION_DECISIONS + CHAMPION_AUDIT_DECISIONS - 1]
    )
    row_signals = pd.to_datetime(rows["signal_at"], utc=True)
    exits = pd.to_datetime(rows["exit_at"], utc=True)
    fitting = rows.loc[row_signals.lt(boundary_calibration) & exits.lt(boundary_calibration)]
    calibration = rows.loc[
        row_signals.ge(boundary_calibration)
        & row_signals.lt(boundary_audit)
        & exits.lt(boundary_audit)
    ]
    audit = rows.loc[row_signals.ge(boundary_audit) & row_signals.le(audit_end)]
    if len(fitting) < 180 or len(calibration) < 50 or len(audit) < 50:
        return None
    x_fit, y_fit = design(fitting), fitting["net_bps"].to_numpy(float)
    x_cal, y_cal = design(calibration), calibration["net_bps"].to_numpy(float)
    x_audit = design(audit)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x_fit, y_fit)
    ridge_cal_raw, _ = _predict([ridge], x_cal)
    ridge_bias = float(np.mean(y_cal - ridge_cal_raw))
    ridge_residuals = y_cal - ridge_cal_raw - ridge_bias
    ridge_lower = _mean_lower_bound(ridge_residuals, 42)
    ridge_audit, _ = _predict([ridge], x_audit)
    ridge_audit += ridge_bias
    ridge_metrics = _audit_metrics(audit, ridge_audit, ridge_lower)

    from xgboost import XGBRegressor

    xgb_models: list[Any] = []
    for seed in range(5):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            tree_method="hist",
            device="cuda",
            n_jobs=4,
            random_state=42 + seed,
        )
        indexes = _bootstrap_indexes(len(fitting), 42 + seed)
        model.fit(x_fit[indexes], y_fit[indexes], verbose=False)
        xgb_models.append(model)
    xgb_cal_raw, _ = _predict(xgb_models, x_cal)
    xgb_bias = float(np.mean(y_cal - xgb_cal_raw))
    xgb_residuals = y_cal - xgb_cal_raw - xgb_bias
    xgb_lower = _mean_lower_bound(xgb_residuals, 42)
    xgb_audit, _ = _predict(xgb_models, x_audit)
    xgb_audit += xgb_bias
    xgb_metrics = _audit_metrics(audit, xgb_audit, xgb_lower)
    xgb_wins = all(
        xgb_metrics[key] < ridge_metrics[key]
        for key in ("ev_mse", "calibration_error", "decision_regret")
    ) and xgb_metrics["lcb_decision_regret"] <= ridge_metrics["lcb_decision_regret"]
    champion = "xgboost" if xgb_wins else "ridge"
    outcomes = _fit_outcomes(fitting, calibration)
    return {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "champion": champion,
        "models": xgb_models if xgb_wins else [ridge],
        "bias": xgb_bias if xgb_wins else ridge_bias,
        "residual_lower": xgb_lower if xgb_wins else ridge_lower,
        "candidate_audit": {"ridge": ridge_metrics, "xgboost": xgb_metrics},
        "outcomes": outcomes,
        "probability_status": (
            "CALIBRATED"
            if len(outcomes["barriers"]) == len(BARRIERS)
            else "INSUFFICIENT_OBSERVED_BITUNIX_LABELS"
        ),
        "frozen_after_signal": pd.Timestamp(audit["signal_at"].max()).isoformat(),
        "frozen_after_exit": pd.Timestamp(audit["exit_at"].max()).isoformat(),
        "fit_rows": len(fitting),
        "calibration_rows": len(calibration),
        "champion_audit_rows": len(audit),
        "live_orders_enabled": False,
    }


def _holdout_metrics(rows: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    if rows.empty:
        return {"candidates": 0, "trades": 0, "days": 0, "eligible": False}
    prediction, dispersion = _predict(bundle["models"], design(rows))
    lcb = prediction + float(bundle["bias"]) + float(bundle["residual_lower"]) - dispersion
    scored = rows.copy()
    scored["ev_mean"] = prediction + float(bundle["bias"])
    scored["ev_lcb"] = lcb
    predicted_outcomes = _predict_outcomes(bundle, scored)
    scored[predicted_outcomes.columns] = predicted_outcomes
    probability_ready = all(
        f"p_target_{barrier}bps_before_stop" in scored for barrier in BARRIERS
    )
    if probability_ready:
        costs = scored["fee_bps"] + scored["slippage_reserve_bps"]
        stop = scored["stop_bps"]
        evs = pd.DataFrame(
            {
                barrier: scored[f"p_target_{barrier}bps_before_stop"]
                * (barrier - costs)
                - (1 - scored[f"p_target_{barrier}bps_before_stop"])
                * (stop + costs)
                for barrier in BARRIERS
            },
            index=scored.index,
        )
        scored["probability_ev_bps"] = evs.max(axis=1)
        scored["proposed_target_bps"] = evs.idxmax(axis=1)
    else:
        scored["probability_ev_bps"] = np.nan
        scored["proposed_target_bps"] = np.nan
    choices: list[pd.Series] = []
    decisions: list[dict[str, Any]] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    for _, group in scored.groupby("signal_at", sort=True):
        signal = pd.Timestamp(group.iloc[0]["signal_at"])
        if signal < free_at:
            continue
        winner = group.sort_values(
            ["ev_lcb", "ev_mean", "expert"], ascending=[False, False, True]
        ).iloc[0]
        accepted = (
            probability_ready
            and float(winner["ev_mean"]) > 0
            and float(winner["ev_lcb"]) > 0
            and float(winner["probability_ev_bps"]) > 0
        )
        target = int(winner["proposed_target_bps"]) if probability_ready else None
        decisions.append(
            {
                "signal_at": signal.isoformat(),
                "setup": str(winner["expert"]),
                "direction": str(winner["side"]),
                "expected_net_ev_bps": float(winner["ev_mean"]),
                "prudent_ev_bps": float(winner["ev_lcb"]),
                "proposed_target_bps": target,
                "target_before_stop_probability": (
                    float(winner[f"p_target_{target}bps_before_stop"])
                    if target is not None
                    else None
                ),
                "expected_mfe_bps": {
                    str(minutes): (
                        None
                        if pd.isna(winner.get(f"expected_mfe_{minutes}m_bps"))
                        else float(winner[f"expected_mfe_{minutes}m_bps"])
                    )
                    for minutes in (5, 15, 30, 60)
                },
                "expected_mae_bps": {
                    str(minutes): (
                        None
                        if pd.isna(winner.get(f"expected_mae_{minutes}m_bps"))
                        else float(winner[f"expected_mae_{minutes}m_bps"])
                    )
                    for minutes in (5, 15, 30, 60)
                },
                "technical_stop_bps": float(winner["stop_bps"]),
                "expected_cost_bps": float(
                    winner["fee_bps"] + winner["slippage_reserve_bps"]
                ),
                "execution": "TAKER_MARKET_REPLAY",
                "decision": "TRADE" if accepted else "NO_TRADE",
                "reason": (
                    "positive calibrated target EV and prudent net EV"
                    if accepted
                    else "probability model unavailable or net EV not positive"
                ),
            }
        )
        if accepted:
            choices.append(winner)
            free_at = pd.Timestamp(winner["exit_at"])
    selected = pd.DataFrame(choices)
    if selected.empty:
        return {
            "candidates": len(rows),
            "trades": 0,
            "flat_decisions": int(scored["signal_at"].nunique()),
            "days": int(pd.to_datetime(rows["signal_at"], utc=True).dt.floor("D").nunique()),
            "eligible": False,
            "latest_decision": decisions[-1] if decisions else None,
        }
    returns = selected["net_bps"].to_numpy(float)
    stressed = selected["stress_bps"].to_numpy(float)
    gross = selected["gross_bps"].to_numpy(float)
    curve = np.cumsum(returns)
    drawdown = np.maximum.accumulate(np.r_[0.0, curve])[1:] - curve
    gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
    timestamps = pd.to_datetime(selected["signal_at"], utc=True)
    daily = pd.Series(returns, index=pd.DatetimeIndex(timestamps)).resample("1D").sum()
    lower = _mean_lower_bound(returns, 43)
    profit_factor = float(gains / losses) if losses else None
    positive_days = float(daily.gt(0).mean())
    days = int(pd.to_datetime(rows["signal_at"], utc=True).dt.floor("D").nunique())
    gates = {
        "minimum_trades": len(selected) >= MINIMUM_HOLDOUT_TRADES,
        "minimum_days": days >= MINIMUM_HOLDOUT_DAYS,
        "positive_expectancy": float(returns.mean()) > 0,
        "positive_lcb": lower > 0,
        "profit_factor": (profit_factor or 0) >= 1.15,
        "drawdown": float(drawdown.max(initial=0.0)) <= 800,
        "stress_non_negative": float(stressed.mean()) >= 0,
        "majority_positive_days": positive_days > 0.5,
        "gross_move_three_x_cost": float(gross.mean()) >= 12,
    }
    return {
        "candidates": len(rows),
        "trades": len(selected),
        "flat_decisions": int(scored["signal_at"].nunique()) - len(selected),
        "days": days,
        "net_expectancy_bps": float(returns.mean()),
        "stress_expectancy_bps": float(stressed.mean()),
        "stress_expectancy_lcb_95_bps": lower,
        "gross_expectancy_bps": float(gross.mean()),
        "profit_factor": profit_factor,
        "max_drawdown_bps": float(drawdown.max(initial=0.0)),
        "positive_day_fraction": positive_days,
        "gates": gates,
        "eligible": all(gates.values()),
        "latest_decision": decisions[-1] if decisions else None,
    }


def evaluate(matrix_path: Path) -> dict[str, Any]:
    rows = load_matrix(matrix_path)
    frozen_decisions = FIT_DECISIONS + CALIBRATION_DECISIONS + CHAMPION_AUDIT_DECISIONS
    available_decisions = (
        int(pd.to_datetime(rows["signal_at"], utc=True).nunique()) if len(rows) else 0
    )
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "available_rows": len(rows),
        "available_decisions": available_decisions,
        "required_before_freeze": frozen_decisions,
        "live_orders_enabled": False,
        "probability_status": "NOT_TRAINED",
    }
    if not BUNDLE.exists():
        if available_decisions < frozen_decisions:
            payload |= {
                "status": "INSUFFICIENT_FORWARD_DATA",
                "decisions_remaining": frozen_decisions - available_decisions,
                "gpu_training_started": False,
            }
            _atomic_json(REPORT, payload)
            return payload
        bundle = _fit_bundle(rows.copy())
        if bundle is None:
            payload |= {
                "status": "INSUFFICIENT_PURGED_DATA",
                "gpu_training_started": False,
            }
            _atomic_json(REPORT, payload)
            return payload
        _atomic_joblib(BUNDLE, bundle)
    bundle = joblib.load(BUNDLE)
    if bundle.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("selector bundle belongs to a different frozen protocol")
    holdout = rows.loc[
        pd.to_datetime(rows["signal_at"], utc=True).gt(
            pd.Timestamp(bundle["frozen_after_exit"])
        )
    ].copy()
    holdout_metrics = _holdout_metrics(holdout, bundle)
    payload |= {
        "status": (
            "ELIGIBLE_FOR_SHADOW"
            if holdout_metrics.get("eligible")
            else "FROZEN_COLLECTING_HOLDOUT"
        ),
        "champion": bundle["champion"],
        "candidate_audit": bundle["candidate_audit"],
        "probability_status": bundle.get("probability_status"),
        "outcome_heads": {
            "barriers": sorted(bundle.get("outcomes", {}).get("barriers", {})),
            "regressions": sorted(bundle.get("outcomes", {}).get("regressions", {})),
            "target_within_horizons": sorted(
                bundle.get("outcomes", {}).get("horizons", {})
            ),
        },
        "bundle": str(BUNDLE),
        "holdout": holdout_metrics,
    }
    _atomic_json(REPORT, payload)
    return payload
