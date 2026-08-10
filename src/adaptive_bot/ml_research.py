from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlencode

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_bot.adapters.bitunix.market_data import JsonGetter, _get_json
from adaptive_bot.config import AppConfig, MachineLearningConfig
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.domain.exceptions import DataQualityError
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import atr_percentile
from adaptive_bot.indicators.vwap import rolling_vwap

FEATURE_COLUMNS = (
    "distance_vwap_pct",
    "distance_vwap_atr",
    "vwap_zscore",
    "vwap_slope_3",
    "vwap_slope_6",
    "vwap_slope_12",
    "bars_above_vwap",
    "bars_below_vwap",
    "crossed_vwap_12",
    "band_width_pct",
    "position_inside_band",
    "distance_upper_atr",
    "distance_lower_atr",
    "atr_pct",
    "atr_percentile",
    "rolling_std_pct",
    "realized_volatility",
    "high_low_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "adx",
    "adx_slope",
    "ema20_slope",
    "ema50_slope",
    "relative_volume",
    "volume_zscore",
    "volume_change",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "rsi",
    "close_position_in_candle",
    "previous_high_break",
    "previous_low_break",
    "bullish_reversal",
    "bearish_reversal",
    "quote_volume_ratio",
    "funding_rate",
    "mark_last_divergence",
    "mark_return_1",
)


@dataclass(frozen=True)
class DatasetInfo:
    rows: int
    duplicates_removed: int
    missing_candles: int
    spread_source: Literal["unavailable"]
    envelope_corrections: int
    max_envelope_deviation_bps: float


OBSERVED_MARKET_COLUMNS = (
    "mark_close",
    "funding_rate",
    "quote_volume",
    "round_trip_cost_bps",
)
PROVENANCE_COLUMNS = (
    "market_data_source",
    "price_source",
    "volume_source",
    "funding_source",
    "spread_source",
    "cost_source",
)


def write_ml_status(
    config: MachineLearningConfig,
    phase: str,
    detail: str,
    percent: float,
    **progress: Any,
) -> None:
    config.status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "detail": detail,
        "percent": round(percent, 1),
        "updated_at": datetime.now(UTC).isoformat(),
        **progress,
    }
    temporary = config.status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(config.status_path)


def download_official_ml_history(
    app: AppConfig,
    start: datetime,
    end: datetime,
    *,
    get_json: JsonGetter = _get_json,
) -> pd.DataFrame:
    config = app.machine_learning
    if config is None or not config.enabled:
        raise ValueError("machine-learning research is disabled")
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("history boundaries must be timezone-aware and ordered")
    write_ml_status(config, "download", "Downloading official Bitunix LAST_PRICE", 1)
    last = _download_klines(get_json, start, end, "LAST_PRICE")
    write_ml_status(config, "download", "Downloading official Bitunix MARK_PRICE", 45)
    mark = _download_klines(get_json, start, end, "MARK_PRICE").rename(
        columns={name: f"mark_{name}" for name in ("open", "high", "low", "close")}
    )
    write_ml_status(config, "download", "Downloading official Bitunix funding", 88)
    funding = _download_funding(get_json, start - timedelta(days=1), end)
    last["raw_high"] = last["high"]
    last["raw_low"] = last["low"]
    last["high"] = last.loc[:, ["open", "high", "close"]].max(axis=1)
    last["low"] = last.loc[:, ["open", "low", "close"]].min(axis=1)
    data = last.merge(
        mark.loc[:, ["timestamp", "mark_open", "mark_high", "mark_low", "mark_close"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp")
    data = pd.merge_asof(
        data, funding.sort_values("timestamp"), on="timestamp", direction="backward"
    )
    data["round_trip_cost_bps"] = config.taker_fee_bps * 2
    data["market_data_source"] = "observed:bitunix-official-rest"
    data["price_source"] = "observed"
    data["volume_source"] = "observed"
    data["funding_source"] = "observed"
    data["spread_source"] = "unavailable"
    data["cost_source"] = "official_vip0_taker_fee_no_spread"
    if data["funding_rate"].isna().any():
        raise ValueError("official funding history does not cover every candle")
    config.history_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(config.history_path, index=False)
    write_ml_status(config, "download_complete", f"Saved {len(data):,} official candles", 100)
    return data


def _download_klines(
    get_json: JsonGetter,
    start: datetime,
    end: datetime,
    price_type: Literal["LAST_PRICE", "MARK_PRICE"],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    rows: dict[int, dict[str, Any]] = {}
    pages = 0
    while cursor > start_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": cursor,
                "interval": interval,
                "limit": 200,
                "type": price_type,
            }
        )
        payload = get_json(f"https://fapi.bitunix.com/api/v1/futures/market/kline?{query}")
        batch = payload.get("data")
        if payload.get("code") not in (0, "0") or not isinstance(batch, list):
            raise ValueError(f"Bitunix {price_type} history is unavailable")
        valid = [item for item in batch if isinstance(item, dict)]
        if not valid:
            break
        for item in valid:
            timestamp = int(item["time"])
            if start_ms <= timestamp < int(end.timestamp() * 1000):
                rows[timestamp] = {
                    "timestamp": pd.to_datetime(timestamp, unit="ms", utc=True),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["quoteVol"]),
                    "quote_volume": float(item["baseVol"]),
                }
        next_cursor = min(int(item["time"]) for item in valid) - 1
        if next_cursor >= cursor:
            raise RuntimeError(f"Bitunix {price_type} pagination did not advance")
        cursor = next_cursor
        pages += 1
        if progress is not None and pages % 20 == 0:
            progress(cursor, len(rows))
        time.sleep(0.11)
    if progress is not None:
        progress(max(cursor, start_ms), len(rows))
    return pd.DataFrame(rows.values()).sort_values("timestamp").reset_index(drop=True)


def _download_funding(
    get_json: JsonGetter,
    start: datetime,
    end: datetime,
    *,
    symbol: str = "BTCUSDT",
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    rows: dict[int, dict[str, Any]] = {}
    pages = 0
    while cursor > start_ms:
        query = urlencode(
            {"symbol": symbol, "startTime": start_ms, "endTime": cursor, "limit": 200}
        )
        payload = get_json(
            "https://fapi.bitunix.com/api/v1/futures/market/get_funding_rate_history?" + query
        )
        batch = payload.get("data")
        if payload.get("code") not in (0, "0") or not isinstance(batch, list):
            raise ValueError("Bitunix funding history is unavailable")
        valid = [item for item in batch if isinstance(item, dict)]
        if not valid:
            break
        for item in valid:
            timestamp = int(item["fundingTime"])
            if start_ms <= timestamp < int(end.timestamp() * 1000):
                rows[timestamp] = {
                    "timestamp": pd.to_datetime(timestamp, unit="ms", utc=True),
                    "funding_rate": float(item["fundingRate"]),
                }
        next_cursor = min(int(item["fundingTime"]) for item in valid) - 1
        if next_cursor >= cursor:
            raise RuntimeError("Bitunix funding pagination did not advance")
        cursor = next_cursor
        pages += 1
        if progress is not None and pages % 20 == 0:
            progress(cursor, len(rows))
        time.sleep(0.11)
    if progress is not None:
        progress(max(cursor, start_ms), len(rows))
    return pd.DataFrame(rows.values()).sort_values("timestamp").reset_index(drop=True)


def _cell_float(frame: pd.DataFrame, index: int, column: str) -> float:
    return float(cast(Any, frame.at[index, column]))


def build_ml_features(
    frame: pd.DataFrame,
    app: AppConfig,
    *,
    alpha_gross: bool = False,
) -> tuple[pd.DataFrame, DatasetInfo]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    duplicates = int(data["timestamp"].duplicated().sum())
    data = (
        data.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    correction_count = 0
    max_deviation = 0.0
    if {"raw_high", "raw_low"} <= set(data.columns):
        deviation = (
            ((data["high"] - data["raw_high"]) + (data["raw_low"] - data["low"]))
            / data["close"]
            * 10_000
        )
        correction_count = int((deviation > 0).sum())
        max_deviation = float(deviation.max())
        if max_deviation > 100:
            raise ValueError("official OHLC envelope deviation exceeds 100 bps")
    report = validate_candles(
        data, timeframe_minutes=app.strategy.timeframe_minutes, calendar_name=None
    )
    fatal_errors = [error for error in report.errors if "candles are missing" not in error]
    if fatal_errors or report.score < 0.995:
        raise DataQualityError("; ".join(fatal_errors or report.errors))
    missing_market = [
        column for column in (*OBSERVED_MARKET_COLUMNS, *PROVENANCE_COLUMNS) if column not in data
    ]
    if missing_market:
        raise ValueError(
            "ML training requires official observed data; missing: " + ", ".join(missing_market)
        )
    numeric_columns = tuple(
        column
        for column in OBSERVED_MARKET_COLUMNS
        if not (alpha_gross and column == "funding_rate")
    )
    if data.loc[:, (*numeric_columns, *PROVENANCE_COLUMNS)].isna().any().any():
        raise ValueError("ML training refuses null official market fields")
    if not data["market_data_source"].astype(str).str.startswith("observed:").all():
        raise ValueError("market_data_source must identify an observed archive")
    observed_sources = ("price_source", "volume_source")
    if not all(data[column].eq("observed").all() for column in observed_sources):
        raise ValueError("estimated price, volume or funding data are forbidden")
    allowed_funding = ("observed", "unavailable") if alpha_gross else ("observed",)
    if not data["funding_source"].isin(allowed_funding).all():
        raise ValueError("estimated funding data are forbidden")
    if not data["spread_source"].eq("unavailable").all():
        raise ValueError("historical spread must be explicitly unavailable")
    expected_cost_source = (
        "alpha_gross_no_execution_cost" if alpha_gross else "official_vip0_taker_fee_no_spread"
    )
    if not data["cost_source"].eq(expected_cost_source).all():
        raise ValueError(f"unexpected training cost source: expected {expected_cost_source}")
    numeric_market = data.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_market.to_numpy(dtype=float)).all():
        raise ValueError("observed market fields must be finite numbers")
    data.loc[:, numeric_columns] = numeric_market
    data["funding_rate"] = pd.to_numeric(data["funding_rate"], errors="coerce")
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    open_ = data["open"].astype(float)
    volume = data["volume"].astype(float)
    atr_values = atr(high, low, close, app.strategy.atr_period)
    vwap = rolling_vwap(high, low, close, volume, app.strategy.crypto_vwap_window)
    rolling_std = close.rolling(app.strategy.crypto_vwap_window).std()
    adx_values = adx(high, low, close, app.strategy.adx_period)["adx"]
    upper = vwap + app.strategy.range_multiplier * atr_values
    lower = vwap - app.strategy.range_multiplier * atr_values
    price_range = (high - low).replace(0, np.nan)
    returns = close.pct_change(fill_method=None)

    features = data.copy()
    features["vwap"] = vwap
    features["atr"] = atr_values
    features["upper_band"] = upper
    features["lower_band"] = lower
    features["distance_vwap_pct"] = (close - vwap) / vwap
    features["distance_vwap_atr"] = (close - vwap) / atr_values.replace(0, np.nan)
    features["vwap_zscore"] = (close - vwap) / rolling_std.replace(0, np.nan)
    for bars in (3, 6, 12):
        features[f"vwap_slope_{bars}"] = (vwap - vwap.shift(bars)) / (
            atr_values.replace(0, np.nan) * bars
        )
    features["bars_above_vwap"] = _consecutive(close > vwap)
    features["bars_below_vwap"] = _consecutive(close < vwap)
    crossed = (close > vwap) != (close.shift(1) > vwap.shift(1))
    features["crossed_vwap_12"] = crossed.rolling(12).sum()
    features["band_width_pct"] = (upper - lower) / vwap
    features["position_inside_band"] = (close - lower) / (upper - lower).replace(0, np.nan)
    features["distance_upper_atr"] = (close - upper) / atr_values.replace(0, np.nan)
    features["distance_lower_atr"] = (close - lower) / atr_values.replace(0, np.nan)
    features["atr_pct"] = atr_values / close
    features["atr_percentile"] = atr_percentile(atr_values, app.strategy.atr_percentile_window)
    features["rolling_std_pct"] = rolling_std / close
    features["realized_volatility"] = returns.rolling(12).std() * math.sqrt(12)
    features["high_low_range_pct"] = price_range / close
    features["candle_body_pct"] = (close - open_).abs() / open_
    features["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    features["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
    features["adx"] = adx_values
    features["adx_slope"] = adx_values.diff(3) / 3
    features["ema20_slope"] = normalized_ema_slope(close, atr_values, 20, 5)
    features["ema50_slope"] = normalized_ema_slope(close, atr_values, 50, 5)
    volume_mean = volume.rolling(96).mean()
    volume_std = volume.rolling(96).std()
    features["relative_volume"] = volume / volume_mean.replace(0, np.nan)
    features["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    features["volume_change"] = volume.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    for bars in (1, 3, 6, 12):
        features[f"return_{bars}"] = close.pct_change(bars, fill_method=None)
    features["rsi"] = _rsi(close)
    features["close_position_in_candle"] = (close - low) / price_range
    features["previous_high_break"] = (close > high.shift(1)).astype(float)
    features["previous_low_break"] = (close < low.shift(1)).astype(float)
    features["bullish_reversal"] = (
        (close > open_) & (close.shift(1) < open_.shift(1)) & (close > open_.shift(1))
    ).astype(float)
    features["bearish_reversal"] = (
        (close < open_) & (close.shift(1) > open_.shift(1)) & (close < open_.shift(1))
    ).astype(float)
    features["quote_volume_ratio"] = data["quote_volume"] / data["quote_volume"].rolling(96).mean()
    mark_close = data["mark_close"].astype(float)
    features["mark_last_divergence"] = (mark_close - close) / close
    features["mark_return_1"] = mark_close.pct_change(fill_method=None)
    return features, DatasetInfo(
        len(data),
        duplicates,
        report.missing_candles,
        "unavailable",
        correction_count,
        max_deviation,
    )


def add_triple_barrier_labels(
    features: pd.DataFrame,
    *,
    max_holding_bars: int,
    stop_atr: float,
    target_z: float = 0.0,
) -> pd.DataFrame:
    result = features.copy()
    size = len(result)
    centers = result["vwap"].to_numpy(dtype=float)
    atr_values = result["atr"].to_numpy(dtype=float)
    opens = result["open"].to_numpy(dtype=float)
    highs = result["high"].to_numpy(dtype=float)
    lows = result["low"].to_numpy(dtype=float)
    closes = result["close"].to_numpy(dtype=float)
    costs = result["round_trip_cost_bps"].to_numpy(dtype=float) / 10_000
    for side in ("long", "short"):
        target = np.full(size, -1, dtype=np.int8)
        net_return = np.full(size, np.nan)
        exit_index = np.full(size, -1, dtype=np.int64)
        for index in range(size - max_holding_bars - 1):
            center = centers[index]
            atr_value = atr_values[index]
            if not math.isfinite(center) or not math.isfinite(atr_value) or atr_value <= 0:
                continue
            entry = opens[index + 1]
            target_price = (
                center - target_z * atr_value if side == "long" else center + target_z * atr_value
            )
            if (side == "long" and target_price <= entry) or (
                side == "short" and target_price >= entry
            ):
                # The signal existed at candle close, but the next executable price has
                # already crossed its target. Record the canceled opportunity instead of
                # deleting the sample using future information.
                target[index] = 0
                net_return[index] = 0.0
                exit_index[index] = index + 1
                continue
            stop = entry - stop_atr * atr_value if side == "long" else entry + stop_atr * atr_value
            for future in range(index + 1, index + max_holding_bars + 1):
                hit_stop = lows[future] <= stop if side == "long" else highs[future] >= stop
                hit_target = (
                    highs[future] >= target_price
                    if side == "long"
                    else lows[future] <= target_price
                )
                if hit_stop or hit_target:
                    won = hit_target and not hit_stop
                    exit_price = target_price if won else stop
                    gross = (
                        (exit_price - entry) / entry
                        if side == "long"
                        else (entry - exit_price) / entry
                    )
                    target[index] = int(won)
                    net_return[index] = gross - costs[index + 1]
                    exit_index[index] = future
                    break
            if target[index] < 0:
                future = index + max_holding_bars
                exit_price = closes[future]
                gross = (
                    (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
                )
                target[index] = 0
                net_return[index] = gross - costs[index + 1]
                exit_index[index] = future
        result[f"target_{side}"] = target
        result[f"net_return_{side}"] = net_return
        result[f"exit_index_{side}"] = exit_index
    return result


def run_ml_research(app: AppConfig, frame: pd.DataFrame) -> dict[str, Any]:
    config = app.machine_learning
    if config is None or not config.enabled:
        raise ValueError("machine-learning research is disabled")
    write_ml_status(config, "features", "Building causal VWAP feature matrix", 2)
    features, info = build_ml_features(frame, app)
    labelled = add_triple_barrier_labels(
        features,
        max_holding_bars=config.max_holding_bars,
        stop_atr=config.stop_atr,
    )
    usable = labelled.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if len(usable) < 10_000:
        raise ValueError("at least 10,000 fully featured candles are required")
    final_start = int(len(usable) * 0.8)
    development = usable.iloc[:final_start].reset_index(drop=True)
    final_test = usable.iloc[final_start:]
    write_ml_status(config, "optimization", "Optuna TPE walk-forward", 8)
    study = _optimize(app, development, config)
    params = dict(study.best_params)
    train_end = int(len(development) * 0.75)
    train = development.iloc[:train_end]
    calibration = development.iloc[train_end + config.max_holding_bars :]
    models: dict[str, ClassifierMixin] = {}
    baselines: dict[str, ClassifierMixin] = {}
    test_metrics: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    write_ml_status(config, "final_test", "Training frozen models; opening final holdout", 90)
    for side in ("long", "short"):
        train_side = _eligible(train, side, params)
        calibration_side = _eligible(calibration, side, params)
        test_side = _eligible(final_test, side, params)
        model = _fit_calibrated("hist", params, train_side, calibration_side, side, config)
        baseline = _fit_calibrated("logistic", params, train_side, calibration_side, side, config)
        models[side] = model
        baselines[side] = baseline
        probability = model.predict_proba(test_side.loc[:, FEATURE_COLUMNS])[:, 1]
        baseline_probability = baseline.predict_proba(test_side.loc[:, FEATURE_COLUMNS])[:, 1]
        predictions[side] = probability
        test_metrics[side] = _classification_metrics(
            test_side[f"target_{side}"].to_numpy(), probability, float(params["threshold"])
        )
        baseline_metrics[side] = _classification_metrics(
            test_side[f"target_{side}"].to_numpy(),
            baseline_probability,
            float(params["threshold"]),
        )
        test_metrics[side]["feature_importance"] = _importance(
            model, test_side, side, config.random_seed
        )
    scenarios = {
        name: _combined_economic_metrics(final_test, params, models, multiplier)
        for name, multiplier in (("optimistic", 0.5), ("base", 1.0), ("stress", 2.0))
    }
    payload = {
        "schema_version": 1,
        "run_id": datetime.now(UTC).strftime("ml-%Y%m%dT%H%M%SZ"),
        "completed_at": datetime.now(UTC).isoformat(),
        "mode": "research_shadow_only",
        "data": {
            **info.__dict__,
            "start": str(usable["timestamp"].min()),
            "end": str(usable["timestamp"].max()),
            "feature_rows": len(usable),
            "features": list(FEATURE_COLUMNS),
            "historical_spread": "unavailable_not_estimated",
            "source": "Bitunix official REST",
            "execution_cost": "official VIP0 taker fee; spread excluded",
        },
        "split": {
            "development_rows": len(development),
            "final_test_rows": len(final_test),
            "final_test_start": str(final_test["timestamp"].iloc[0]),
            "gap_bars": config.max_holding_bars,
            "random_split": False,
        },
        "labels": {
            side: {
                str(int(cast(Any, key))): int(value)
                for key, value in usable[f"target_{side}"].value_counts().sort_index().items()
            }
            for side in ("long", "short")
        },
        "optimization": {
            "trials": len(study.trials),
            "best_value": study.best_value,
            "best_parameters": params,
        },
        "baseline_logistic": baseline_metrics,
        "hist_gradient_boosting": test_metrics,
        "economic_scenarios": scenarios,
        "accepted": _accepted(scenarios["base"], scenarios["stress"]),
        "risk_policy": {
            "risk_per_trade": str(app.risk.risk_per_trade),
            "leverage": str(app.bitunix.leverage if app.bitunix else 1),
            "model_controls_size": False,
            "automatic_promotion": False,
        },
    }
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "features": FEATURE_COLUMNS,
            "parameters": params,
            "model_version": payload["run_id"],
            "trained_until": str(calibration["timestamp"].iloc[-1]),
        },
        config.model_path,
    )
    _write_report(config.report_path, payload)
    _write_model_card(config.model_card_path, payload)
    write_ml_status(config, "complete", "Model and untouched holdout report ready", 100)
    return payload


def _optimize(
    app: AppConfig, development: pd.DataFrame, config: MachineLearningConfig
) -> optuna.Study:
    storage = f"sqlite:///{config.optuna_path.resolve().as_posix()}"
    config.optuna_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        study_name=f"adaptive-range-{datetime.now(UTC):%Y%m%dT%H%M%S}",
    )

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 260),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 120),
            "l2": trial.suggest_float("l2", 1e-3, 10, log=True),
            "threshold": trial.suggest_float("threshold", 0.25, 0.60),
            "z_min": trial.suggest_float("z_min", 0.8, 2.5),
            "adx_max": trial.suggest_float("adx_max", 18, 45),
            "atr_min": trial.suggest_float("atr_min", 0, 35),
            "atr_max": trial.suggest_float("atr_max", 65, 100),
        }
        splitter = TimeSeriesSplit(
            n_splits=config.n_splits,
            gap=config.max_holding_bars,
        )
        folds: list[dict[str, float]] = []
        for fold, (train_index, validation_index) in enumerate(splitter.split(development)):
            train = development.iloc[train_index]
            validation = development.iloc[validation_index]
            probabilities: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
            for side in ("long", "short"):
                train_side = _eligible(train, side, params)
                validation_side = _eligible(validation, side, params)
                if min(len(train_side), len(validation_side)) < 100:
                    raise optuna.TrialPruned()
                model = _fit_temporal("hist", params, train_side, side, config)
                probabilities[side] = (
                    validation_side,
                    model.predict_proba(validation_side.loc[:, FEATURE_COLUMNS])[:, 1],
                )
            metrics = _economic_metrics(probabilities, params, 0.0)
            folds.append(metrics)
            partial = _objective_score(folds)
            trial.report(partial, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _objective_score(folds)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial
        completed = len(study.trials)
        write_ml_status(
            config,
            "optimization",
            f"Optuna trial {completed}/{config.trials}",
            8 + completed / config.trials * 80,
        )

    study.optimize(objective, n_trials=config.trials, callbacks=[callback], gc_after_trial=True)
    return study


def _eligible(data: pd.DataFrame, side: str, params: dict[str, Any]) -> pd.DataFrame:
    direction = data["distance_vwap_atr"] < -float(params["z_min"])
    if side == "short":
        direction = data["distance_vwap_atr"] > float(params["z_min"])
    mask = (
        direction
        & (data["adx"] <= float(params["adx_max"]))
        & data["atr_percentile"].between(float(params["atr_min"]), float(params["atr_max"]))
        & (data[f"target_{side}"] >= 0)
    )
    return data.loc[mask].copy()


def _estimator(kind: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    if kind == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=0.5, max_iter=1000, random_state=seed),
                ),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    max_iter=int(params["max_iter"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    l2_regularization=float(params["l2"]),
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_temporal(
    kind: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    side: str,
    config: MachineLearningConfig,
) -> ClassifierMixin:
    split = int(len(train) * 0.8)
    base = train.iloc[:split]
    calibration = train.iloc[split + config.max_holding_bars :]
    return _fit_calibrated(kind, params, base, calibration, side, config)


def _fit_calibrated(
    kind: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    side: str,
    config: MachineLearningConfig,
) -> ClassifierMixin:
    estimator = _estimator(kind, params, config.random_seed)
    estimator.fit(train.loc[:, FEATURE_COLUMNS], train[f"target_{side}"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    calibrated.fit(calibration.loc[:, FEATURE_COLUMNS], calibration[f"target_{side}"])
    return calibrated


def _combined_economic_metrics(
    test: pd.DataFrame,
    params: dict[str, Any],
    models: dict[str, ClassifierMixin],
    cost_multiplier: float,
) -> dict[str, float]:
    probabilities = {
        side: (
            eligible := _eligible(test, side, params),
            models[side].predict_proba(eligible.loc[:, FEATURE_COLUMNS])[:, 1],
        )
        for side in ("long", "short")
    }
    return _economic_metrics(probabilities, params, cost_multiplier - 1)


def _economic_metrics(
    probabilities: dict[str, tuple[pd.DataFrame, np.ndarray]],
    params: dict[str, Any],
    extra_cost_multiplier: float,
) -> dict[str, float]:
    signals: list[tuple[int, int, float]] = []
    for side, (data, probability) in probabilities.items():
        extra_cost = (
            data["round_trip_cost_bps"].to_numpy(dtype=float) * extra_cost_multiplier / 10_000
        )
        returns = data[f"net_return_{side}"].to_numpy(dtype=float) - extra_cost
        # Payoffs from the evaluated period must never decide which signals are accepted.
        # Expected value belongs in a preceding training/calibration window; this legacy
        # evaluator therefore uses only the probability threshold.
        accepted = probability >= float(params["threshold"])
        for position, value in zip(np.flatnonzero(accepted), returns[accepted], strict=True):
            index = int(data.index[int(position)])
            signals.append((index, int(data.iloc[position][f"exit_index_{side}"]), float(value)))
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    blocked_until = -1
    trades: list[float] = []
    for index, exit_index, value in sorted(signals):
        if index <= blocked_until:
            continue
        trades.append(value)
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        blocked_until = exit_index
    win_total = sum((value for value in trades if value > 0), 0.0)
    loss_total = abs(sum((value for value in trades if value < 0), 0.0))
    return {
        "net_return": sum(trades),
        "max_drawdown": drawdown,
        "profit_factor": win_total / loss_total if loss_total else (999.0 if win_total else 0.0),
        "trades": float(len(trades)),
        "win_rate": sum(value > 0 for value in trades) / len(trades) if trades else 0.0,
    }


def _objective_score(folds: list[dict[str, float]]) -> float:
    net = np.asarray([fold["net_return"] for fold in folds])
    drawdown = max(fold["max_drawdown"] for fold in folds)
    profit_factor = np.median([min(fold["profit_factor"], 3) for fold in folds])
    trades = sum(fold["trades"] for fold in folds)
    return float(
        np.median(net)
        - 1.5 * drawdown
        + 0.005 * profit_factor
        - np.std(net)
        - max(0.0, 100 - trades) * 0.0001
    )


def _classification_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = probability >= threshold
    fraction, mean = calibration_curve(target, probability, n_bins=10, strategy="quantile")
    return {
        "precision": precision_score(target, predicted, zero_division=0),
        "recall": recall_score(target, predicted, zero_division=0),
        "pr_auc": average_precision_score(target, probability),
        "roc_auc": roc_auc_score(target, probability),
        "brier_score": brier_score_loss(target, probability),
        "confusion_matrix": confusion_matrix(target, predicted).tolist(),
        "calibration_curve": [
            {"predicted": float(x), "observed": float(y)}
            for x, y in zip(mean, fraction, strict=True)
        ],
        "samples": len(target),
    }


def _importance(
    model: ClassifierMixin, data: pd.DataFrame, side: str, seed: int
) -> list[dict[str, float | str]]:
    sample = data.tail(min(3000, len(data)))
    result = permutation_importance(
        model,
        sample.loc[:, FEATURE_COLUMNS],
        sample[f"target_{side}"],
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    ranked = sorted(
        zip(FEATURE_COLUMNS, result.importances_mean, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    return [{"feature": name, "importance": float(value)} for name, value in ranked]


def _accepted(base: dict[str, float], stress: dict[str, float]) -> bool:
    return bool(
        base["net_return"] > 0
        and base["profit_factor"] > 1
        and stress["net_return"] >= 0
        and base["trades"] >= 100
    )


def _consecutive(mask: pd.Series) -> pd.Series:
    groups = (mask != mask.shift()).cumsum()
    return mask.astype(int).groupby(groups).cumsum().where(mask, 0).astype(float)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _write_model_card(path: Path, payload: dict[str, Any]) -> None:
    base = payload["economic_scenarios"]["base"]
    stress = payload["economic_scenarios"]["stress"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Adaptive Range ML model card",
                "",
                f"- Version: `{payload['run_id']}`",
                f"- Accepted: `{payload['accepted']}`",
                f"- Final holdout begins: `{payload['split']['final_test_start']}`",
                f"- Base net return: `{base['net_return']:.6f}`",
                f"- Base profit factor: `{base['profit_factor']:.3f}`",
                f"- Stress net return: `{stress['net_return']:.6f}`",
                "- Prices, volumes and funding: observed via Bitunix official REST.",
                "- Historical spread: unavailable and not estimated.",
                "- Costs: official VIP0 taker fee; spread excluded pending shadow validation.",
                "- Usage: research/shadow only. Automatic promotion and live orders are disabled.",
            ]
        ),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
