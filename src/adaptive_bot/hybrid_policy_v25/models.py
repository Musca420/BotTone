from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.expert_policy import moving_block_lower_bound
from adaptive_bot.hybrid_policy_v22.path_audit import atomic_parquet
from adaptive_bot.hybrid_policy_v25.protocol import (
    FEATURE_COLUMNS,
    HOLDOUT_WEEKS,
    RANDOM_SEED,
    ROOT,
    atomic_json,
    status,
)
from adaptive_bot.research import combinatorial_pbo, deflated_sharpe_probability

PREDICTION_PATH = ROOT / "oos_predictions.parquet"
DECISION_PATH = ROOT / "oos_decisions.parquet"


@dataclass
class ModelPair:
    name: str
    regressor: Any
    classifier: Any


@dataclass
class ConstantCalibrator:
    probability: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        rows = len(values)
        return np.column_stack(
            [np.full(rows, 1 - self.probability), np.full(rows, self.probability)]
        )


def _baseline_pair() -> ModelPair:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ModelPair(
        "ridge",
        Pipeline([("numeric", clone(numeric)), ("model", Ridge(alpha=10.0))]),
        Pipeline(
            [
                ("numeric", clone(numeric)),
                (
                    "model",
                    LogisticRegression(C=0.1, max_iter=2_000, random_state=RANDOM_SEED),
                ),
            ]
        ),
    )


def _xgb_pair(depth: int, learning_rate: float, min_child_weight: int) -> ModelPair:
    common = {
        "n_estimators": 400,
        "max_depth": depth,
        "learning_rate": learning_rate,
        "min_child_weight": min_child_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
    }
    name = f"xgb-d{depth}-lr{learning_rate}-mcw{min_child_weight}"
    return ModelPair(
        name,
        XGBRegressor(objective="reg:squarederror", **common),
        XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common),
    )


def _pairs() -> list[ModelPair]:
    return [_baseline_pair()] + [
        _xgb_pair(depth, rate, weight)
        for depth in (4, 6)
        for rate in (0.03, 0.07)
        for weight in (3, 10)
    ]


def _deterministic_predictions(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    score = (
        (rows["regime_score"].to_numpy(float) - 0.6) * 20
        + rows["pullback_efficiency"].to_numpy(float) * 4
        + rows["spot_confirmation"].to_numpy(float) * 2
        - rows["cost_to_stop_ratio"].to_numpy(float) * 8
    )
    probability = 1 / (1 + np.exp(-score / 5))
    return score, probability


def _fit_pair(
    pair: ModelPair,
    train: pd.DataFrame,
    target: str,
    validation: pd.DataFrame | None = None,
) -> ModelPair:
    x = train.loc[:, FEATURE_COLUMNS]
    y = train[target].to_numpy(float)
    positive = (y > 0).astype(int)
    regressor, classifier = clone(pair.regressor), clone(pair.classifier)
    if pair.name.startswith("xgb-") and validation is not None:
        validation_x = validation.loc[:, FEATURE_COLUMNS]
        validation_y = validation[target].to_numpy(float)
        regressor.set_params(early_stopping_rounds=30)
        classifier.set_params(early_stopping_rounds=30)
        regressor.fit(x, y, eval_set=[(validation_x, validation_y)], verbose=False)
        classifier.fit(
            x,
            positive,
            eval_set=[(validation_x, (validation_y > 0).astype(int))],
            verbose=False,
        )
    else:
        regressor.fit(x, y)
        classifier.fit(x, positive)
    return ModelPair(pair.name, regressor, classifier)


def _predict(pair: ModelPair, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = rows.loc[:, FEATURE_COLUMNS]
    return (
        np.asarray(pair.regressor.predict(x), dtype=float),
        np.asarray(pair.classifier.predict_proba(x)[:, 1], dtype=float),
    )


def _inner_splits(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ordered = train.sort_values("entry_timestamp")
    times = pd.to_datetime(ordered["entry_timestamp"], utc=True)
    boundaries = np.linspace(0.4, 1.0, 4)
    output: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for left, right in pairwise(boundaries):
        fit_end = pd.Timestamp(times.quantile(left))
        validation_end = pd.Timestamp(times.quantile(right))
        fit = ordered.loc[times.lt(fit_end)].copy()
        fit = fit.loc[
            pd.to_datetime(fit["exit_timestamp"], utc=True).lt(fit_end - pd.Timedelta(hours=6))
        ]
        validation = ordered.loc[times.ge(fit_end) & times.lt(validation_end)].copy()
        if len(fit) >= 100 and len(validation) >= 30:
            output.append((fit, validation))
    return output


def _select_model(
    train: pd.DataFrame, target: str, *, context: str = ""
) -> tuple[ModelPair | None, list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    splits = _inner_splits(train)
    if len(splits) < 3:
        return None, scores
    pairs = _pairs()
    for number, pair in enumerate(pairs, start=1):
        prefix = f"{context} " if context else ""
        status(
            "model_inner",
            f"{prefix}candidate {number}/{len(pairs)} {pair.name}",
            76,
        )
        fold_scores = []
        best_iterations: list[int] = []
        try:
            for fit, validation in splits:
                fitted = _fit_pair(pair, fit, target, validation)
                expected, probability = _predict(fitted, validation)
                y = validation[target].to_numpy(float)
                fold_scores.append(
                    mean_absolute_error(y, expected)
                    + 10 * brier_score_loss((y > 0).astype(int), probability)
                )
                if pair.name.startswith("xgb-"):
                    iterations = [
                        int(getattr(model, "best_iteration", 399)) + 1
                        for model in (fitted.regressor, fitted.classifier)
                    ]
                    best_iterations.append(max(iterations))
        except Exception as error:
            scores.append({"model": pair.name, "error": f"{type(error).__name__}: {error}"})
            continue
        scores.append(
            {
                "model": pair.name,
                "inner_loss": float(np.mean(fold_scores)),
                "n_estimators": int(np.median(best_iterations)) if best_iterations else None,
            }
        )
    deterministic_losses = []
    for _, validation in splits:
        expected, probability = _deterministic_predictions(validation)
        y = validation[target].to_numpy(float)
        deterministic_losses.append(
            mean_absolute_error(y, expected)
            + 10 * brier_score_loss((y > 0).astype(int), probability)
        )
    scores.append({"model": "deterministic", "inner_loss": float(np.mean(deterministic_losses))})
    valid = [row for row in scores if "inner_loss" in row]
    if not valid:
        return None, scores
    chosen_name = str(min(valid, key=lambda row: (row["inner_loss"], row["model"]))["model"])
    if chosen_name == "deterministic":
        return ModelPair("deterministic", None, None), scores
    chosen = next(pair for pair in _pairs() if pair.name == chosen_name)
    chosen_score = next(row for row in scores if row["model"] == chosen_name)
    if chosen.name.startswith("xgb-") and chosen_score["n_estimators"]:
        chosen.regressor.set_params(n_estimators=chosen_score["n_estimators"])
        chosen.classifier.set_params(n_estimators=chosen_score["n_estimators"])
    return chosen, scores


def _quantile_models(train: pd.DataFrame, target: str) -> dict[float, Any]:
    models: dict[float, Any] = {}
    for quantile in (0.25, 0.5, 0.75):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=quantile,
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        model.fit(train.loc[:, FEATURE_COLUMNS], train[target].to_numpy(float))
        models[quantile] = model
    return models


def _calibrate_probability(
    raw_probability: np.ndarray, y: np.ndarray
) -> tuple[Any, np.ndarray, str]:
    if len(np.unique(y)) < 2:
        probability = float(y.mean()) if len(y) else 0.0
        calibrator = ConstantCalibrator(probability)
        return calibrator, np.full(len(y), probability), "constant"
    if len(y) >= 500 and len(np.unique(raw_probability)) >= 20:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_probability, y)
        return calibrator, np.asarray(calibrator.predict(raw_probability)), "isotonic"
    calibrator = LogisticRegression(C=1.0, random_state=RANDOM_SEED).fit(
        raw_probability.reshape(-1, 1), y
    )
    return (
        calibrator,
        calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1],
        "platt",
    )


def _apply_calibrator(calibrator: Any, method: str, values: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(calibrator.predict(values), dtype=float)
    return np.asarray(calibrator.predict_proba(values.reshape(-1, 1))[:, 1], dtype=float)


def _sharpe(values: np.ndarray) -> float:
    deviation = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(values.mean() / deviation) if deviation > 0 else 0.0


def _statistical_controls(predictions: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "spa_pvalue": 1.0,
            "dsr": 0.0,
            "pbo": {"pbo": None, "splits": 0, "blocks": 0},
            "coverage_curve": [],
            "asset_metrics": {},
        }
    indexed = predictions.copy()
    indexed["date"] = pd.to_datetime(indexed["entry_timestamp"], utc=True).dt.floor("D")
    indexed["policy_return"] = indexed["observed_net_bps"].where(indexed["model_gate"], 0.0)
    indexed["base_return"] = indexed["net_return_bps_4"]
    indexed["flow_return"] = indexed["flow_variant_net_return_bps_4"].where(
        indexed["flow_gate_pass"], 0.0
    )
    daily = indexed.groupby("date")[["policy_return", "base_return", "flow_return"]].sum()
    active_columns = [column for column in daily if daily[column].std(ddof=0) > 0]
    active_daily = daily.loc[:, active_columns]
    spa_pvalue = 1.0
    if len(active_daily) >= 20 and active_columns:
        spa = SPA(
            np.zeros(len(active_daily)),
            -active_daily.to_numpy(float),
            reps=1_000,
            block_size=max(2, int(math.sqrt(len(daily)))),
            bootstrap="stationary",
            seed=RANDOM_SEED,
        )
        spa.compute()
        spa_pvalue = float(spa.pvalues["consistent"])
    trial_sharpes = [_sharpe(daily[column].to_numpy(float)) for column in active_columns]
    selected_daily = daily["policy_return"].to_numpy(float)
    weeks = pd.DatetimeIndex(daily.index).to_period("W")
    candidates = [
        [group.to_list() for _, group in daily[column].groupby(weeks)] for column in active_columns
    ]
    coverage_curve = []
    for fraction in (1.0, 0.5, 0.25, 0.1):
        count = max(1, int(len(indexed) * fraction))
        values = indexed.nlargest(count, "policy_score")["observed_net_bps"].to_numpy(float)
        coverage_curve.append(
            {"coverage": fraction, "events": len(values), "expectancy_bps": float(values.mean())}
        )
    return {
        "spa_pvalue": spa_pvalue,
        "dsr": deflated_sharpe_probability(selected_daily, trial_sharpes)
        if len(decisions) >= 3
        else 0.0,
        "pbo": combinatorial_pbo(candidates),
        "coverage_curve": coverage_curve,
        "asset_metrics": {asset: _metrics(group) for asset, group in decisions.groupby("asset")}
        if "asset" in decisions
        else {},
    }


def _profit_factor(values: np.ndarray) -> float:
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return float(gains / losses) if losses > 0 else 999.0 if gains > 0 else 0.0


def _drawdown_bps(values: np.ndarray) -> float:
    equity = np.r_[0.0, np.cumsum(values)]
    return float(np.max(np.maximum.accumulate(equity) - equity))


def _choose_threshold(calibration: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        chosen = calibration.loc[
            calibration["expected_net_bps"].ge(2)
            & calibration["calibrated_probability"].ge(threshold)
            & calibration["q25_net_bps"].ge(-2)
            & calibration["cost_to_stop_ratio"].le(0.33)
        ]
        values = chosen["observed_net_bps"].to_numpy(float)
        utility = (
            float(values.mean())
            - 0.02 * _drawdown_bps(values)
            - 0.1 * len(values) / len(calibration)
            if len(values) >= 20 and _profit_factor(values) >= 1
            else -math.inf
        )
        trials.append(
            {
                "threshold": threshold,
                "trades": len(values),
                "expectancy_bps": float(values.mean()) if len(values) else 0.0,
                "profit_factor": _profit_factor(values),
                "utility": utility,
            }
        )
    valid = [row for row in trials if math.isfinite(row["utility"])]
    return (
        float(max(valid, key=lambda row: (row["utility"], row["threshold"]))["threshold"])
        if valid
        else 1.0
    ), trials


def _top_k(rows: pd.DataFrame) -> pd.DataFrame:
    accepted: list[Any] = []
    last_asset: dict[str, pd.Timestamp] = {}
    daily_count: Counter[str] = Counter()
    for index, row in rows.sort_values(
        ["entry_timestamp", "policy_score"], ascending=[True, False]
    ).iterrows():
        timestamp = pd.Timestamp(row["entry_timestamp"])
        day = timestamp.strftime("%Y-%m-%d")
        prior = last_asset.get(str(row["asset"]))
        if daily_count[day] >= 2 or (
            prior is not None and timestamp < prior + pd.Timedelta(hours=4)
        ):
            continue
        accepted.append(index)
        last_asset[str(row["asset"])] = timestamp
        daily_count[day] += 1
    return rows.loc[accepted].sort_values("entry_timestamp")


def _metrics(rows: pd.DataFrame) -> dict[str, float]:
    values = rows.get("observed_net_bps", pd.Series(dtype=float)).to_numpy(float)
    stress = rows.get("observed_net_bps_8", pd.Series(dtype=float)).to_numpy(float)
    return {
        "trades": float(len(values)),
        "expectancy_net_bps_4": float(values.mean()) if len(values) else 0.0,
        "expectancy_net_bps_8": float(stress.mean()) if len(stress) else 0.0,
        "profit_factor": _profit_factor(values),
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "max_drawdown_bps": _drawdown_bps(values),
        "bootstrap_lcb_bps": moving_block_lower_bound(
            values, block_size=min(20, len(values)), seed=RANDOM_SEED
        )
        if len(values)
        else 0.0,
    }


def _predict_frame(
    pair: ModelPair,
    train: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    quantiles: dict[float, Any],
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    if pair.name == "deterministic":
        expected, probability = _deterministic_predictions(rows)
    else:
        expected, probability = _predict(_fit_pair(pair, train, target), rows)
    quantile_values = {
        quantile: np.asarray(model.predict(rows.loc[:, FEATURE_COLUMNS]), dtype=float)
        for quantile, model in quantiles.items()
    }
    return expected, probability, quantile_values


def train_walk_forward(
    outcomes: pd.DataFrame, *, resume: bool, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any] | None]:
    if resume and PREDICTION_PATH.exists() and DECISION_PATH.exists() and not smoke:
        import json

        loaded_predictions = pd.read_parquet(PREDICTION_PATH)
        loaded_decisions = pd.read_parquet(DECISION_PATH)
        audit = json.loads(Path("data/reports/ml_hybrid_v25_model_audit.json").read_text())
        if audit.get("diagnostic_version") == 2:
            return loaded_predictions, loaded_decisions, audit, None
    if outcomes.empty or not outcomes["asset"].eq("BTCUSDT").any():
        empty = outcomes.iloc[:0].copy()
        audit = {
            "verdict": "INSUFFICIENT_DATA",
            "oos_events": 0,
            "oos_trades": 0,
            "gates_passed": False,
            "reason": "no BTC outcomes available",
            "folds": [],
            "chosen_models": {},
        }
        return empty, empty, audit, None
    data = outcomes.copy()
    times = pd.to_datetime(data["entry_timestamp"], utc=True)
    btc_end = times.loc[data["asset"].eq("BTCUSDT")].max()
    holdout_start = btc_end - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    research = data.loc[times.lt(holdout_start)].copy()
    research_times = pd.to_datetime(research["entry_timestamp"], utc=True)
    first = research_times.min().floor("D") + pd.Timedelta(weeks=52)
    last = research_times.max().floor("D")
    starts = list(pd.date_range(first, last - pd.Timedelta(weeks=16), freq="8W", tz="UTC"))
    if smoke:
        starts = starts[-1:]
    predictions: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    chosen_models: list[str] = []
    for fold, start in enumerate(starts, start=1):
        train = research.loc[
            research_times.ge(start - pd.Timedelta(weeks=52)) & research_times.lt(start)
        ].copy()
        train = train.loc[
            pd.to_datetime(train["exit_timestamp"], utc=True).lt(start - pd.Timedelta(hours=6))
        ]
        calibration_start, test_start = start, start + pd.Timedelta(weeks=8)
        test_end = test_start + pd.Timedelta(weeks=8)
        calibration = research.loc[
            research_times.ge(calibration_start) & research_times.lt(test_start)
        ].copy()
        test = research.loc[research_times.ge(test_start) & research_times.lt(test_end)].copy()
        if min(len(train), len(calibration), len(test)) == 0:
            continue
        base_target = "net_return_bps_4"
        variant_train = train.loc[train["flow_gate_pass"]].copy()
        candidates = [("base", train, base_target)]
        if len(variant_train) >= 200:
            candidates.append(("flow", variant_train, "flow_variant_net_return_bps_4"))
        variant_results = []
        for variant, variant_rows, target in candidates:
            pair, search = _select_model(
                variant_rows,
                target,
                context=f"Fold {fold}/{len(starts)} {variant.upper()}",
            )
            valid_losses = [row["inner_loss"] for row in search if "inner_loss" in row]
            if pair is not None and valid_losses:
                variant_results.append((min(valid_losses), variant, pair, target, search))
        if not variant_results:
            continue
        _, variant, selected, target, search = min(
            variant_results, key=lambda value: (value[0], value[1], value[2].name)
        )
        selected_train = train if variant == "base" else train.loc[train["flow_gate_pass"]]
        selected_calibration = (
            calibration if variant == "base" else calibration.loc[calibration["flow_gate_pass"]]
        )
        selected_test = test if variant == "base" else test.loc[test["flow_gate_pass"]]
        if min(len(selected_train), len(selected_calibration), len(selected_test)) == 0:
            continue
        quantiles = _quantile_models(selected_train, target)
        cal_expected, cal_probability, cal_q = _predict_frame(
            selected, selected_train, selected_calibration, target, quantiles
        )
        observed_calibration = selected_calibration[target].to_numpy(float)
        calibrator, calibrated_cal_probability, method = _calibrate_probability(
            cal_probability, (observed_calibration > 0).astype(int)
        )
        bias = float((observed_calibration - cal_expected).mean())
        calibration_frame = selected_calibration.copy()
        calibration_frame["expected_net_bps"] = cal_expected + bias
        calibration_frame["calibrated_probability"] = calibrated_cal_probability
        calibration_frame["q25_net_bps"] = cal_q[0.25]
        calibration_frame["observed_net_bps"] = observed_calibration
        threshold, threshold_trials = _choose_threshold(calibration_frame)
        test_expected, test_probability, test_q = _predict_frame(
            selected, selected_train, selected_test, target, quantiles
        )
        test_frame = selected_test.copy()
        test_frame["outer_fold"] = fold
        test_frame["variant"] = variant
        test_frame["model"] = selected.name
        test_frame["expected_net_bps"] = test_expected + bias
        test_frame["expected_gross_bps"] = test_frame["expected_net_bps"] + 4.0
        test_frame["calibrated_probability"] = _apply_calibrator(
            calibrator, method, test_probability
        )
        for quantile, values in test_q.items():
            test_frame[f"q{int(quantile * 100)}_net_bps"] = values
        test_frame["predicted_mae_bps"] = selected_train["mae_bps"].mean()
        test_frame["policy_score"] = (
            test_frame["expected_net_bps"]
            + 5 * (test_frame["calibrated_probability"] - 0.5)
            + 0.25 * test_frame["q25_net_bps"]
            - 0.10 * test_frame["predicted_mae_bps"]
        )
        test_frame["probability_threshold"] = threshold
        test_frame["model_gate"] = (
            test_frame["expected_net_bps"].ge(2)
            & test_frame["expected_gross_bps"].ge(12)
            & test_frame["calibrated_probability"].ge(threshold)
            & test_frame["q25_net_bps"].ge(-2)
            & test_frame["cost_to_stop_ratio"].le(0.33)
            & test_frame["is_available"].astype(bool)
        )
        test_frame["observed_net_bps"] = test_frame[target]
        stress_target = "net_return_bps_8" if variant == "base" else "flow_variant_net_return_bps_8"
        test_frame["observed_net_bps_8"] = test_frame[stress_target]
        sign_stability = {
            feature: float(selected_test[feature].corr(selected_test[target], method="spearman"))
            if selected_test[feature].notna().sum() >= 20
            else 0.0
            for feature in FEATURE_COLUMNS
        }
        importance: dict[str, float] = {}
        if selected.name != "deterministic" and len(selected_test) >= 20:
            diagnostic = _fit_pair(selected, selected_train, target)
            permutation = permutation_importance(
                diagnostic.regressor,
                selected_test.loc[:, FEATURE_COLUMNS],
                selected_test[target],
                scoring="neg_mean_absolute_error",
                n_repeats=2,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
            importance = dict(
                zip(FEATURE_COLUMNS, permutation.importances_mean.tolist(), strict=True)
            )
        transfer_audit: dict[str, Any] = {"available": False}
        btc_train = selected_train.loc[selected_train["asset"].eq("BTCUSDT")]
        btc_calibration = selected_calibration.loc[selected_calibration["asset"].eq("BTCUSDT")]
        eth_test = selected_test.loc[selected_test["asset"].eq("ETHUSDT")]
        if min(len(btc_train), len(btc_calibration), len(eth_test)) >= 20:
            transfer_pair = (
                selected
                if selected.name == "deterministic"
                else _fit_pair(selected, btc_train, target)
            )
            if selected.name == "deterministic":
                btc_expected, btc_probability = _deterministic_predictions(btc_calibration)
                eth_expected, eth_probability = _deterministic_predictions(eth_test)
            else:
                btc_expected, btc_probability = _predict(transfer_pair, btc_calibration)
                eth_expected, eth_probability = _predict(transfer_pair, eth_test)
            transfer_calibrator, _, transfer_method = _calibrate_probability(
                btc_probability,
                (btc_calibration[target].to_numpy(float) > 0).astype(int),
            )
            transfer_bias = float((btc_calibration[target].to_numpy(float) - btc_expected).mean())
            transfer_probability = _apply_calibrator(
                transfer_calibrator, transfer_method, eth_probability
            )
            transfer_gate = (eth_expected + transfer_bias >= 8) & (
                transfer_probability >= threshold
            )
            transfer_values = eth_test.loc[transfer_gate, target].to_numpy(float)
            transfer_audit = {
                "available": True,
                "train_asset": "BTCUSDT",
                "test_asset": "ETHUSDT",
                "test_events": len(eth_test),
                "selected_events": int(transfer_gate.sum()),
                "expectancy_bps": float(transfer_values.mean()) if len(transfer_values) else 0.0,
                "profit_factor": _profit_factor(transfer_values),
            }
        predictions.append(test_frame)
        chosen_models.append(selected.name)
        fold_reports.append(
            {
                "fold": fold,
                "train_start": (start - pd.Timedelta(weeks=52)).isoformat(),
                "train_end": start.isoformat(),
                "calibration_start": calibration_start.isoformat(),
                "calibration_end": test_start.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "train_events": len(selected_train),
                "calibration_events": len(selected_calibration),
                "test_events": len(selected_test),
                "variant": variant,
                "model": selected.name,
                "calibration_method": method,
                "threshold": threshold,
                "threshold_trials": threshold_trials,
                "inner_search": search,
                "permutation_importance_oos": importance,
                "feature_direction_oos": sign_stability,
                "btc_train_eth_test": transfer_audit,
            }
        )
        status(
            "model_oos", f"Fold {fold}/{len(starts)} {selected.name}", 76 + 16 * fold / len(starts)
        )
    all_predictions: pd.DataFrame = (
        pd.concat(predictions, ignore_index=True) if predictions else research.iloc[:0].copy()
    )
    eligible: pd.DataFrame = (
        all_predictions.loc[all_predictions["model_gate"]].copy()
        if "model_gate" in all_predictions
        else all_predictions.copy()
    )
    decisions = _top_k(eligible) if not eligible.empty else eligible
    metrics = _metrics(decisions)
    fold_ev: pd.Series[Any] = (
        decisions.groupby("outer_fold")["observed_net_bps"].mean()
        if not decisions.empty
        else pd.Series(dtype=float)
    )
    positive_pnl: pd.Series[Any] = (
        decisions.groupby("outer_fold")["observed_net_bps"].sum().clip(lower=0)
        if not decisions.empty
        else pd.Series(dtype=float)
    )
    concentration = (
        float(positive_pnl.max() / positive_pnl.sum()) if positive_pnl.sum() > 0 else 1.0
    )
    gates = {
        "oos_events_500": len(all_predictions) >= 500,
        "oos_trades_150": len(decisions) >= 150,
        "expectancy_positive": metrics["expectancy_net_bps_4"] > 0,
        "profit_factor_1_10": metrics["profit_factor"] >= 1.10,
        "positive_fold_half": float(fold_ev.gt(0).mean()) >= 0.5 if len(fold_ev) else False,
        "fold_concentration_40pct": concentration <= 0.40,
    }
    if not all_predictions.empty:
        all_predictions["score_decile"] = pd.qcut(
            all_predictions["policy_score"], 10, labels=False, duplicates="drop"
        )
    deciles = (
        all_predictions.groupby("score_decile")["observed_net_bps"]
        .agg(["count", "mean"])
        .to_dict("index")
        if "score_decile" in all_predictions
        else {}
    )
    calibration_payload: dict[str, Any] = {}
    if not all_predictions.empty:
        probability_true, probability_pred = calibration_curve(
            (all_predictions["observed_net_bps"] > 0).astype(int),
            all_predictions["calibrated_probability"],
            n_bins=10,
            strategy="quantile",
        )
        calibration_payload = {
            "predicted": probability_pred.tolist(),
            "observed": probability_true.tolist(),
            "precision": precision_score(
                (all_predictions["observed_net_bps"] > 0).astype(int),
                all_predictions["model_gate"].astype(int),
                zero_division=0,
            ),
            "recall": recall_score(
                (all_predictions["observed_net_bps"] > 0).astype(int),
                all_predictions["model_gate"].astype(int),
                zero_division=0,
            ),
        }
    statistical_controls = _statistical_controls(all_predictions, decisions)
    audit = {
        "diagnostic_version": 2,
        "verdict": "META_EDGE_CANDIDATE" if all(gates.values()) else "NO_META_EDGE",
        "holdout": {
            "asset": "BTCUSDT",
            "start": holdout_start.isoformat(),
            "end": btc_end.isoformat(),
            "opened": False,
            "rows_excluded": int((data["asset"].eq("BTCUSDT") & times.ge(holdout_start)).sum()),
        },
        "oos_events": len(all_predictions),
        "oos_trades": len(decisions),
        "metrics": metrics,
        "gates": gates,
        "gates_passed": all(gates.values()),
        "fold_positive_fraction": float(fold_ev.gt(0).mean()) if len(fold_ev) else 0.0,
        "fold_pnl_concentration": concentration,
        "folds": fold_reports,
        "chosen_models": dict(Counter(chosen_models)),
        "score_deciles": {str(key): value for key, value in deciles.items()},
        "calibration": calibration_payload,
        "statistical_controls": statistical_controls,
    }
    final_model: dict[str, Any] | None = None
    if chosen_models:
        final_model = {"preferred_model": Counter(chosen_models).most_common(1)[0][0]}
    if not smoke:
        atomic_parquet(PREDICTION_PATH, all_predictions)
        atomic_parquet(DECISION_PATH, decisions)
        atomic_json(Path("data/reports/ml_hybrid_v25_model_audit.json"), audit)
    return all_predictions, decisions, audit, final_model
