from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v9 import spa_reality_check
from adaptive_bot.hybrid_policy_v11 import (
    BASE_COST_BPS,
    EXCHANGES,
    SHADOW_COST_BPS,
    V11Expert,
    btc_inventory,
    build_feature_frame,
    cross_exchange_features,
    evaluate_expert,
    return_metrics,
)

PROTOCOL = "hybrid_v14_vwap_event_meta_label"
ROOT = Path("data/ml/hybrid_v14")
MATRIX_ROOT = ROOT / "event_matrix"
OOS_ROOT = ROOT / "oos_folds"
MODEL_ROOT = Path("data/models/expert_policy/v14")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
BUNDLE_PATH = MODEL_ROOT / "bundle.joblib"
RESEARCH_BUNDLE_PATH = MODEL_ROOT / "research_shadow_bundle.joblib"
FORWARD_LOCK_PATH = MODEL_ROOT / "forward_protocol.lock.json"
REPORT_PATH = Path("data/reports/ml_hybrid_v14.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v14.status.json")
FORWARD_STATUS_PATH = Path("data/reports/ml_hybrid_v14_forward.status.json")
ORDERFLOW_PATH = ROOT / "binance_reference_orderflow_1m.parquet"
BITUNIX_CANDLES_PATH = Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
BITUNIX_MICROSTRUCTURE_ROOT = Path("data/raw/bitunix_microstructure")
EXCURSION_Z = 1.5
REENTRY_Z = 1.25
RESET_Z = 0.25
STOP_ATR = 2.0
STOP_BUFFER_ATR = 0.25
MIN_SIGNAL_STOP_ATR = 0.75
HOLDING_BARS = 32
EMBARGO_HOURS = HOLDING_BARS * 15 // 60
TRAIN_WEEKS = 52
CALIBRATION_WEEKS = 8
TEST_WEEKS = 4
TIMEFRAMES = (15, 30, 60)
FEATURES = (
    "distance_vwap_atr",
    "vwap_zscore",
    "vwap_slope_3",
    "vwap_slope_6",
    "vwap_slope_12",
    "bars_above_vwap",
    "bars_below_vwap",
    "crossed_vwap_12",
    "atr_pct",
    "atr_percentile",
    "realized_volatility",
    "adx",
    "adx_slope",
    "ema20_slope",
    "ema50_slope",
    "relative_volume",
    "volume_zscore",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_48",
    "rsi",
    "close_position_in_candle",
    "upper_wick_pct",
    "lower_wick_pct",
    "mark_last_divergence",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "regime_code",
    "cross_exchange_return_median",
    "cross_exchange_return_dispersion",
    "excursion_max_z",
    "excursion_bars",
    "reentry_speed",
    "side_code",
    "prospective_reward_r",
    "prospective_cost_r",
    "timeframe_minutes",
    "vwap_hours",
    "context_code",
    "binance_taker_imbalance_15m",
    "binance_taker_imbalance_1h",
    "binance_trade_count_z",
    "binance_aggressive_volume_z",
)

ORDERFLOW_COLUMNS = (
    "binance_taker_imbalance_15m",
    "binance_taker_imbalance_1h",
    "binance_trade_count_z",
    "binance_aggressive_volume_z",
)
BINANCE_ARCHIVE = "https://data.binance.vision/data/futures/um"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_names(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start.tz_localize(None), end=end.tz_localize(None), freq="M")
    current = end.tz_localize(None).to_period("M")
    names = [
        f"monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip"
        for month in months
        if month < current
    ]
    names.extend(
        f"daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day:%Y-%m-%d}.zip"
        for day in pd.date_range(current.start_time, end.tz_localize(None).floor("D"), freq="D")
    )
    return names


def _download_checked(relative: str) -> bytes:
    url = f"{BINANCE_ARCHIVE}/{relative}"
    request = Request(url, headers={"User-Agent": "adaptive-range-research/14"})
    checksum_request = Request(f"{url}.CHECKSUM", headers=request.headers)
    with urlopen(checksum_request, timeout=60) as response:
        expected = response.read().decode("ascii").split()[0].lower()
    with urlopen(request, timeout=180) as response:
        payload = cast(bytes, response.read())
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Binance checksum mismatch for {relative}")
    return payload


def _parse_binance_kline_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise RuntimeError("unexpected Binance archive layout")
        frame = pd.read_csv(archive.open(members[0]), header=None)
    frame = frame.iloc[:, [0, 7, 8, 10]].copy()
    frame.columns = ["timestamp", "quote_volume", "trade_count", "taker_buy_quote"]
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True
    )
    for column in ("quote_volume", "trade_count", "taker_buy_quote"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def build_orderflow_context(minutes: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "quote_volume", "trade_count", "taker_buy_quote"}
    if not required.issubset(minutes):
        raise ValueError(f"missing Binance order-flow fields: {sorted(required - set(minutes))}")
    minute = minutes.copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], format="mixed", utc=True)
    minute = minute.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    expected = minute.index.to_series().diff().eq(pd.Timedelta(minutes=1)).rolling(15).sum().eq(15)
    minute["aggressive_net"] = 2 * minute["taker_buy_quote"] - minute["quote_volume"]
    bars = minute[["quote_volume", "trade_count", "taker_buy_quote", "aggressive_net"]].resample(
        "15min", label="left", closed="left"
    ).sum(min_count=15)
    continuity = expected.resample("15min", label="left", closed="left").min().fillna(False)
    sell = bars["quote_volume"] - bars["taker_buy_quote"]
    bars["binance_taker_imbalance_15m"] = (
        (bars["taker_buy_quote"] - sell) / bars["quote_volume"].replace(0, np.nan)
    )
    buy_hour = bars["taker_buy_quote"].rolling(4, min_periods=4).sum()
    volume_hour = bars["quote_volume"].rolling(4, min_periods=4).sum()
    bars["binance_taker_imbalance_1h"] = (2 * buy_hour - volume_hour) / volume_hour.replace(
        0, np.nan
    )
    for source, target in (
        ("trade_count", "binance_trade_count_z"),
        ("aggressive_net", "binance_aggressive_volume_z"),
    ):
        history = bars[source].shift(1).rolling(2_688, min_periods=672)
        bars[target] = (bars[source] - history.mean()) / history.std(ddof=0).replace(0, np.nan)
    bars["orderflow_available_at"] = bars.index + pd.Timedelta(minutes=15)
    bars["orderflow_coverage"] = continuity & bars[list(ORDERFLOW_COLUMNS)].notna().all(axis=1)
    return bars.reset_index()[
        ["orderflow_available_at", "orderflow_coverage", *ORDERFLOW_COLUMNS]
    ]


def load_orderflow_context(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if ORDERFLOW_PATH.exists():
        cached = pd.read_parquet(ORDERFLOW_PATH)
        times = pd.to_datetime(cached["timestamp"], format="mixed", utc=True)
        if times.min() <= start.floor("min") and times.max() >= end.floor("min"):
            return build_orderflow_context(cached.loc[times.between(start, end)])
    frames: list[pd.DataFrame] = []
    archives = _archive_names(start, end)
    for number, relative in enumerate(archives, start=1):
        _status(
            "orderflow_download",
            f"Binance official order-flow archive {number}/{len(archives)}",
            2 + 3 * number / len(archives),
            block="1/4",
        )
        frames.append(_parse_binance_kline_zip(_download_checked(relative)))
    minutes = pd.concat(frames, ignore_index=True)
    minutes = minutes.loc[
        pd.to_datetime(minutes["timestamp"], utc=True).between(start.floor("min"), end.floor("min"))
    ].drop_duplicates("timestamp").sort_values("timestamp")
    if minutes.empty or minutes["timestamp"].diff().dropna().max() > pd.Timedelta(minutes=1):
        raise RuntimeError("Binance order-flow history has missing minutes")
    ORDERFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(ORDERFLOW_PATH, minutes)
    return build_orderflow_context(minutes)


def primary_expert(side: str, timeframe_minutes: int = 15) -> V11Expert:
    return V11Expert(
        f"vwap_first_reentry_{timeframe_minutes}m",
        "mean_reversion",
        cast(Any, side),
        "confirmed_reentry",
        EXCURSION_Z,
        0,
        STOP_ATR,
        None,
        0.0,
        None,
        8 * 60 // timeframe_minutes,
    )


def first_reentry_events(features: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return one causal signal per completed VWAP excursion."""
    result = features.copy()
    signed = result["distance_vwap_atr"].astype(float) * (1 if side == "short" else -1)
    extreme = (
        (result["high"] - result["vwap"]) / result["atr"]
        if side == "short"
        else (result["vwap"] - result["low"]) / result["atr"]
    )
    regime = result["regime_code"]
    regime_confirmed = regime.isin((0.0, 1.0, 2.0))
    regime_confirmed &= regime.eq(regime.shift(1)) & regime.eq(regime.shift(2))
    safe = (
        result["data_valid"].fillna(False).astype(bool)
        & result["local_feature_coverage"].fillna(False).astype(bool)
        & result["atr_percentile"].le(90)
        & regime_confirmed
    )
    signals = np.zeros(len(result), dtype=bool)
    maxima = np.full(len(result), np.nan)
    durations = np.full(len(result), np.nan)
    speeds = np.full(len(result), np.nan)
    armed = False
    consumed = False
    maximum = 0.0
    duration = 0
    previous = np.nan
    extreme_values = extreme.to_numpy(dtype=float)
    for index, raw_value in enumerate(signed.to_numpy(dtype=float)):
        value = float(raw_value)
        extreme_value = float(extreme_values[index])
        if not bool(safe.iloc[index]) or not np.isfinite(value):
            armed = consumed = False
            maximum = 0.0
            duration = 0
            previous = value
            continue
        if value <= RESET_Z:
            armed = consumed = False
            maximum = 0.0
            duration = 0
        elif value >= EXCURSION_Z and not consumed:
            armed = True
            maximum = max(maximum, extreme_value)
            duration += 1
        elif armed:
            maximum = max(maximum, extreme_value)
            duration += 1
            if value <= REENTRY_Z and np.isfinite(previous) and value < previous:
                signals[index] = True
                maxima[index] = maximum
                durations[index] = duration
                speeds[index] = previous - value
                armed = False
                consumed = True
        previous = value
    result["excursion_max_z"] = maxima
    result["excursion_bars"] = durations
    result["reentry_speed"] = speeds
    result["side_code"] = 1.0 if side == "long" else -1.0
    if side == "long":
        swing_stop = result["low"].rolling(2).min() - STOP_BUFFER_ATR * result["atr"]
        minimum_stop = result["close"] - MIN_SIGNAL_STOP_ATR * result["atr"]
        result["event_stop_price"] = np.minimum(swing_stop, minimum_stop)
    else:
        swing_stop = result["high"].rolling(2).max() + STOP_BUFFER_ATR * result["atr"]
        minimum_stop = result["close"] + MIN_SIGNAL_STOP_ATR * result["atr"]
        result["event_stop_price"] = np.maximum(swing_stop, minimum_stop)
    prospective_risk = result["side_code"] * (result["close"] - result["event_stop_price"])
    valid_risk = prospective_risk.gt(0)
    result["prospective_reward_r"] = np.where(
        valid_risk,
        result["side_code"] * (result["vwap"] - result["close"]) / prospective_risk,
        np.nan,
    )
    result["prospective_cost_r"] = np.where(
        valid_risk,
        BASE_COST_BPS / (prospective_risk / result["close"] * 10_000),
        np.nan,
    )
    result["event_signal"] = signals
    return result


def build_exchange_events(
    path: Path,
    app: AppConfig,
    exchange: str,
    timeframe_minutes: int,
    orderflow: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, features = build_feature_frame(
        path,
        app,
        exchange,
        timeframe_minutes=timeframe_minutes,
        vwap_hours=24,
    )
    features = pd.merge_asof(
        features.sort_values("signal_timestamp"),
        orderflow.sort_values("orderflow_available_at"),
        left_on="signal_timestamp",
        right_on="orderflow_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    features["orderflow_lookahead_valid"] = pd.to_datetime(
        features["orderflow_available_at"], utc=True
    ).le(pd.to_datetime(features["signal_timestamp"], utc=True))
    features["local_feature_coverage"] &= (
        features["orderflow_coverage"].fillna(False).astype(bool)
        & features["orderflow_lookahead_valid"].fillna(False).astype(bool)
    )
    rows: list[pd.DataFrame] = []
    for side in ("long", "short"):
        event_features = first_reentry_events(features, side)
        outcome = evaluate_expert(
            event_features,
            raw,
            primary_expert(side, timeframe_minutes),
            cost_bps=BASE_COST_BPS,
            entry_mask=event_features["event_signal"],
            stop_prices=event_features["event_stop_price"],
            timeframe_minutes=timeframe_minutes,
            vwap_hours=24.0,
        )
        if outcome.empty:
            continue
        direction = 1.0 if side == "long" else -1.0
        risk = (outcome["entry_price"] - outcome["initial_stop_price"]).abs()
        outcome["reward_r"] = (direction * (outcome["vwap"] - outcome["entry_price"]) / risk).clip(
            lower=0
        )
        outcome["target_hit"] = outcome["exit_reason"].eq("target").astype(int)
        outcome["meta_label"] = outcome["net_return_r_1x"].gt(0).astype(int)
        outcome["context_code"] = (
            outcome["side_code"] * 1_000
            + outcome["regime_code"] * 100
            + outcome["timeframe_minutes"]
        )
        rows.append(outcome.loc[outcome["execution_valid"].astype(bool)])
    if not rows:
        raise RuntimeError(f"no VWAP re-entry events for {exchange}")
    return features, pd.concat(rows, ignore_index=True).sort_values("signal_timestamp")


def build_matrices(
    app: AppConfig,
    inventory: dict[str, dict[str, Any]],
    orderflow: pd.DataFrame,
    *,
    resume: bool,
) -> tuple[dict[int, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    local: dict[int, dict[str, pd.DataFrame]] = {timeframe: {} for timeframe in TIMEFRAMES}
    matrices: dict[str, pd.DataFrame] = {}
    fresh: dict[str, list[pd.DataFrame]] = {exchange: [] for exchange in EXCHANGES}
    completed = 0
    total = len(TIMEFRAMES) * len(EXCHANGES)
    for timeframe in TIMEFRAMES:
        for exchange in EXCHANGES:
            source = Path(inventory[exchange]["path"])
            features, events = build_exchange_events(
                source, app, exchange, timeframe, orderflow
            )
            local[timeframe][exchange] = features
            fresh[exchange].append(events)
            completed += 1
            _status(
                "event_matrix",
                f"{exchange.upper()} {timeframe}m events ({completed}/{total})",
                5 + 15 * completed / total,
                block="1/4",
            )
    for number, exchange in enumerate(EXCHANGES, start=1):
        new_matrix = pd.concat(fresh[exchange], ignore_index=True).sort_values("signal_timestamp")
        target = MATRIX_ROOT / f"{exchange}.parquet"
        if resume and target.exists():
            cached = pd.read_parquet(target)
            if (
                cached.get("protocol_sha256", pd.Series(dtype=str)).eq(protocol_hash()).all()
                and cached.get("orderflow_sha256", pd.Series(dtype=str))
                .eq(_sha256(ORDERFLOW_PATH))
                .all()
            ):
                matrices[exchange] = cached
            else:
                matrices[exchange] = new_matrix
        else:
            matrices[exchange] = new_matrix
        matrices[exchange]["protocol_sha256"] = protocol_hash()
        matrices[exchange]["orderflow_sha256"] = _sha256(ORDERFLOW_PATH)
        _atomic_parquet(target, matrices[exchange])
        _status(
            "event_matrix",
            f"{exchange.upper()} {len(matrices[exchange]):,} first-reentry events",
            5 + number * 5,
            block="1/4",
        )
    return local, matrices


def _weights(rows: pd.DataFrame) -> np.ndarray:
    overlap = rows.groupby(["exchange", "signal_timestamp"])["meta_label"].transform("size")
    weights = pd.Series(1.0 / overlap.to_numpy(float), index=rows.index)
    totals = weights.groupby([rows["exchange"], rows["side"]]).transform("sum")
    balanced = weights / totals
    return np.asarray((balanced * len(balanced) / balanced.sum()).to_numpy(float), dtype=float)


def _complete(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURES)).copy()
    return result.loc[result["feature_coverage"].fillna(False).astype(bool)].sort_values(
        "signal_timestamp"
    )


def _sigmoid_calibrator(raw: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1_000_000, solver="lbfgs", random_state=20260804).fit(
        logits, labels
    )


def _calibrate(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(logits)[:, 1], dtype=float)


def _quality(rows: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    labels = rows["meta_label"].to_numpy(int)
    fraction, predicted = calibration_curve(labels, probability, n_bins=8, strategy="quantile")
    return {
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "calibration_mae": float(np.mean(np.abs(fraction - predicted))),
    }


def _raw_probability(model: Any, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(rows[list(FEATURES)])[:, 1], dtype=float)


def _candidate_model(train: pd.DataFrame, kind: str, parameters: dict[str, Any]) -> Any:
    values = train[list(FEATURES)]
    labels = train["meta_label"].to_numpy(int)
    weights = _weights(train)
    if kind == "ridge_logistic":
        continuous = [
            feature
            for feature in FEATURES
            if feature not in {"context_code", "regime_code", "side_code", "timeframe_minutes"}
        ]
        model = make_pipeline(
            ColumnTransformer(
                [
                    ("context", OneHotEncoder(handle_unknown="ignore"), ["context_code"]),
                    ("continuous", StandardScaler(), continuous),
                ]
            ),
            LogisticRegression(
                C=float(parameters["C"]),
                solver="lbfgs",
                max_iter=2000,
                random_state=20260804,
            ),
        )
    else:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            learning_rate=0.03,
            min_child_weight=int(parameters["min_child_weight"]),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=10.0,
            random_state=20260804,
        )
    return model.fit(
        values,
        labels,
        **(
            {"sample_weight": weights}
            if kind != "ridge_logistic"
            else {"logisticregression__sample_weight": weights}
        ),
    )


def _tuned_candidate(
    train: pd.DataFrame, kind: str
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    grids: dict[str, tuple[dict[str, Any], ...]] = {
        "ridge_logistic": ({"C": 0.01}, {"C": 0.1}, {"C": 1.0}),
        "xgboost": (
            {"n_estimators": 200, "max_depth": 2, "min_child_weight": 20},
            {"n_estimators": 300, "max_depth": 3, "min_child_weight": 20},
            {"n_estimators": 300, "max_depth": 2, "min_child_weight": 50},
        ),
    }
    times = pd.to_datetime(train["signal_timestamp"], utc=True)
    cutoff = times.max() - pd.Timedelta(weeks=8)
    inner_train = train.loc[
        times.lt(cutoff) & pd.to_datetime(train["exit_timestamp"], utc=True).lt(cutoff)
    ]
    inner_valid = train.loc[times.ge(cutoff)]
    if len(inner_train) < 300 or len(inner_valid) < 80:
        selected = grids[kind][0]
        return _candidate_model(train, kind, selected), selected, []
    scores: list[dict[str, Any]] = []
    labels = inner_valid["meta_label"].to_numpy(int)
    for parameters in grids[kind]:
        model = _candidate_model(inner_train, kind, parameters)
        probability = _raw_probability(model, inner_valid)
        scores.append(
            {
                "parameters": parameters,
                "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
                "brier": float(brier_score_loss(labels, probability)),
            }
        )
    best = min(scores, key=lambda item: (item["log_loss"], item["brier"]))
    selected = cast(dict[str, Any], best["parameters"])
    return _candidate_model(train, kind, selected), selected, scores


def _score_candidates(
    rows: pd.DataFrame,
    probability: np.ndarray,
    residual_lower_r: float = 0.0,
    win_r: float = 1.0,
    loss_r: float = 1.0,
) -> pd.DataFrame:
    result = rows.copy()
    result["probability"] = probability
    result["ev_net"] = probability * win_r - (1 - probability) * loss_r
    result["lcb_net"] = result["ev_net"] + residual_lower_r
    return result


def select_non_overlapping(rows: pd.DataFrame, *, require_lcb: bool = True) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    score = "lcb_net" if require_lcb else "ev_net"
    proposals = rows.loc[rows[score].gt(0)].sort_values(
        ["exchange", "signal_timestamp", score], ascending=[True, True, False]
    )
    chosen: list[pd.DataFrame] = []
    for _, venue in proposals.groupby("exchange", sort=True):
        blocked_until = pd.Timestamp("1900-01-01", tz="UTC")
        for _, row in venue.iterrows():
            signal = pd.to_datetime(row["signal_timestamp"], utc=True)
            if signal <= blocked_until:
                continue
            chosen.append(row.to_frame().T)
            blocked_until = pd.to_datetime(row["exit_timestamp"], utc=True) + pd.Timedelta(hours=1)
    return pd.concat(chosen, ignore_index=True) if chosen else rows.iloc[:0].copy()


def _decision_ev(
    rows: pd.DataFrame, probability: np.ndarray, win_r: float = 1.0, loss_r: float = 1.0
) -> tuple[float, int]:
    chosen = select_non_overlapping(
        _score_candidates(rows, probability, win_r=win_r, loss_r=loss_r), require_lcb=False
    )
    return (float(chosen["net_return_r_1x"].mean()), len(chosen)) if len(chosen) else (0.0, 0)


def _fit_meta_side(train: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    train = _complete(train)
    calibration = _complete(calibration)
    if len(train) < 500 or len(calibration) < 100 or train["meta_label"].nunique() < 2:
        return {"enabled": False, "reason": "insufficient_event_history"}
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    calibrate = calibration.loc[times.lt(midpoint)]
    admission = calibration.loc[times.ge(midpoint)]
    if (
        min(len(calibrate), len(admission)) < 40
        or min(calibrate["meta_label"].nunique(), admission["meta_label"].nunique()) < 2
    ):
        return {"enabled": False, "reason": "insufficient_chronological_calibration"}
    wins = calibrate.loc[calibrate["meta_label"].eq(1), "net_return_r_1x"]
    losses = calibrate.loc[calibrate["meta_label"].eq(0), "net_return_r_1x"].abs()
    win_r, loss_r = float(wins.mean()), float(losses.mean())
    fitted: dict[str, dict[str, Any]] = {}
    for kind in ("ridge_logistic", "xgboost"):
        model, parameters, inner_scores = _tuned_candidate(train, kind)
        calibrator = _sigmoid_calibrator(
            _raw_probability(model, calibrate), calibrate["meta_label"].to_numpy(int)
        )
        probability = _calibrate(calibrator, _raw_probability(model, admission))
        scored = _score_candidates(admission, probability, win_r=win_r, loss_r=loss_r)
        fitted[kind] = {
            "model": model,
            "parameters": parameters,
            "inner_scores": inner_scores,
            "calibrator": calibrator,
            "quality": _quality(admission, probability),
            "decision_ev": _decision_ev(admission, probability, win_r, loss_r)[0],
            "decision_trades": _decision_ev(admission, probability, win_r, loss_r)[1],
            "win_r": win_r,
            "loss_r": loss_r,
            "residual_lower_r": moving_block_lower_bound(
                admission["net_return_r_1x"].to_numpy(float) - scored["ev_net"].to_numpy(float),
                block_size=min(7, len(admission)),
                seed=20260804,
            ),
        }
    ridge = fitted["ridge_logistic"]
    xgb = fitted["xgboost"]
    challenger_wins = (
        xgb["quality"]["brier"] < ridge["quality"]["brier"]
        and xgb["quality"]["log_loss"] < ridge["quality"]["log_loss"]
        and xgb["quality"]["calibration_mae"] <= ridge["quality"]["calibration_mae"]
        and xgb["decision_ev"] > ridge["decision_ev"]
        and xgb["decision_trades"] >= 20
    )
    champion = "xgboost" if challenger_wins else "ridge_logistic"
    return {
        "enabled": True,
        "champion": champion,
        "features": FEATURES,
        **fitted[champion],
        "benchmark": fitted,
    }


def fit_meta_model(train: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    sides = {
        side: _fit_meta_side(
            train.loc[train["side"].eq(side)], calibration.loc[calibration["side"].eq(side)]
        )
        for side in ("long", "short")
    }
    return {
        "enabled": any(model.get("enabled") for model in sides.values()),
        "reason": (
            None
            if any(model.get("enabled") for model in sides.values())
            else "insufficient_event_history_both_sides"
        ),
        "sides": sides,
    }


def predict_meta_model(rows: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    if not fitted.get("enabled"):
        return rows.iloc[:0].copy()
    predictions: list[pd.DataFrame] = []
    for side, side_model in fitted["sides"].items():
        if not side_model.get("enabled"):
            continue
        complete = _complete(rows.loc[rows["side"].eq(side)])
        if complete.empty:
            continue
        probability = _calibrate(
            side_model["calibrator"], _raw_probability(side_model["model"], complete)
        )
        predictions.append(
            _score_candidates(
                complete,
                probability,
                float(side_model["residual_lower_r"]),
                float(side_model["win_r"]),
                float(side_model["loss_r"]),
            )
        )
    return pd.concat(predictions, ignore_index=True) if predictions else rows.iloc[:0].copy()


def _with_context(
    matrices: dict[str, pd.DataFrame],
    local: dict[int, dict[str, pd.DataFrame]],
    allowed: tuple[str, ...],
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    columns = [
        "timestamp",
        "cross_exchange_return_median",
        "cross_exchange_return_dispersion",
        "feature_coverage",
    ]
    for timeframe in TIMEFRAMES:
        context = cross_exchange_features(local[timeframe], allowed)
        for exchange in EXCHANGES:
            selected = matrices[exchange].loc[matrices[exchange]["timeframe_minutes"].eq(timeframe)]
            base = selected.drop(columns=[c for c in columns[1:] if c in selected])
            output.append(
                base.merge(
                    context[exchange][columns],
                    on="timestamp",
                    how="left",
                    validate="many_to_one",
                )
            )
    return (
        pd.concat(output, ignore_index=True).sort_values("signal_timestamp").reset_index(drop=True)
    )


def run_v14(
    app: AppConfig, config_path: Path, *, resume: bool = False, smoke: bool = False
) -> dict[str, Any]:
    inventory = btc_inventory()
    _status("data_audit", "BTC VWAP event protocol", 1, block="1/4")
    start = min(pd.Timestamp(item["start"]) for item in inventory.values())
    end = min(pd.Timestamp(item["end"]) for item in inventory.values())
    orderflow = load_orderflow_context(start, end)
    protocol = preregister(config_path)
    local, matrices = build_matrices(app, inventory, orderflow, resume=resume)
    candidates: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    evaluated_events = 0
    total = 0
    scenarios: list[tuple[str, pd.DataFrame, tuple[Any, ...]]] = []
    for held_out in EXCHANGES:
        allowed = tuple(exchange for exchange in EXCHANGES if exchange != held_out)
        matrix = _with_context(matrices, local, allowed)
        folds = purged_expert_folds(
            matrix,
            train_weeks=TRAIN_WEEKS,
            calibration_weeks=CALIBRATION_WEEKS,
            test_weeks=TEST_WEEKS,
            step_weeks=4,
            embargo_hours=EMBARGO_HOURS,
        )
        if smoke:
            folds = folds[-2:]
        scenarios.append((held_out, matrix, folds))
        total += len(folds)
    completed = 0
    OOS_ROOT.mkdir(parents=True, exist_ok=True)
    for held_out, matrix, folds in scenarios:
        for number, fold in enumerate(folds, start=1):
            checkpoint = OOS_ROOT / f"{held_out}_fold_{number:02d}.joblib"
            if resume and checkpoint.exists():
                saved = joblib.load(checkpoint)
                if saved.get("protocol_sha256") == protocol["protocol_sha256"]:
                    candidates.append(saved["candidates"])
                    diagnostics.append(saved["diagnostics"])
                    evaluated_events += int(saved["diagnostics"].get("test_events", 0))
                    completed += 1
                    continue
            train = matrix.iloc[fold.train].loc[matrix.iloc[fold.train]["exchange"].ne(held_out)]
            calibration = matrix.iloc[fold.calibration].loc[
                matrix.iloc[fold.calibration]["exchange"].ne(held_out)
            ]
            testing = matrix.iloc[fold.test].loc[matrix.iloc[fold.test]["exchange"].eq(held_out)]
            evaluated_events += len(testing)
            _status(
                "meta_training",
                f"{held_out.upper()} fold {number}/{len(folds)} Ridge + XGBoost GPU",
                20 + 65 * completed / max(total, 1),
                block="2/4",
            )
            fitted = fit_meta_model(train, calibration)
            predicted = predict_meta_model(testing, fitted)
            selected = select_non_overlapping(predicted)
            selected["held_out_exchange"] = held_out
            selected["outer_fold"] = number
            candidates.append(selected)
            diagnostic = {
                "held_out": held_out,
                "fold": number,
                "champion": {
                    side: model.get("champion", "disabled")
                    for side, model in fitted.get("sides", {}).items()
                },
                "reason": fitted.get("reason"),
                "test_events": len(testing),
                "prudent_candidates": len(selected),
                "benchmark": {
                    side: {
                        kind: {
                            "quality": values["quality"],
                            "decision_ev": values["decision_ev"],
                            "decision_trades": values["decision_trades"],
                            "parameters": values["parameters"],
                        }
                        for kind, values in model.get("benchmark", {}).items()
                    }
                    for side, model in fitted.get("sides", {}).items()
                },
            }
            diagnostics.append(diagnostic)
            _atomic_joblib(
                checkpoint,
                {
                    "protocol_sha256": protocol["protocol_sha256"],
                    "candidates": selected,
                    "diagnostics": diagnostic,
                },
            )
            completed += 1
    decisions = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    _status("audit", "OOS costs, bootstrap and multiple comparisons", 90, block="3/4")
    audit_result = audit_oos(decisions, evaluated_events)
    deployable = bool(audit_result["ready"] and not smoke)
    report = {
        "protocol": PROTOCOL,
        "run_id": protocol["run_id"],
        "verdict": "ELIGIBLE_FOR_FORWARD_SHADOW" if deployable else "NO_DEPLOYABLE_POLICY",
        "action": "SHADOW_LONG_SHORT_FLAT" if deployable else "FLAT",
        "deployable": False,
        "shadow_eligible": deployable,
        "paper_enabled": False,
        "live_enabled": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "symbols": ["BTCUSDT"],
        "primary_rule": protocol["primary_rule"],
        "data": inventory,
        "orderflow": {
            "source": "Binance USD-M official public archive",
            "sha256": _sha256(ORDERFLOW_PATH),
            "rows": int(pd.read_parquet(ORDERFLOW_PATH, columns=["timestamp"]).shape[0]),
            "features": list(ORDERFLOW_COLUMNS),
        },
        "audit": audit_result,
        "model_diagnostics": diagnostics,
        "holdout": {
            "status": "forward_only",
            "opened": False,
            "starts_after": max(item["end"] for item in inventory.values()),
        },
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if deployable:
        _atomic_joblib(
            BUNDLE_PATH,
            {
                "protocol": protocol,
                "status": "SHADOW_ONLY",
                "note": "Refit is intentionally deferred until forward admission.",
            },
        )
    report_path = REPORT_PATH.with_name("ml_hybrid_v14_smoke.json") if smoke else REPORT_PATH
    _atomic_json(report_path, report)
    _status("complete", report["verdict"], 100, block="4/4")
    return report


def audit_oos(decisions: pd.DataFrame, candidates_evaluated: int) -> dict[str, Any]:
    if decisions.empty:
        return {
            "ready": False,
            "reason": "no_positive_prudent_ev_events",
            "gates": {"trades": False},
            "total_candidates_evaluated": candidates_evaluated,
        }
    metrics = {
        name: {
            exchange: return_metrics(rows, column)
            for exchange, rows in decisions.groupby("exchange")
        }
        for name, column in {
            "gross": "gross_return_r",
            "net_1x": "net_return_r_1x",
            "stress_2x": "net_return_r_2x",
            "bitunix_19bps": "net_return_r_shadow_19bps",
        }.items()
    }
    lower = {
        exchange: moving_block_lower_bound(
            rows["net_return_r_1x"].to_numpy(float), block_size=min(7, len(rows)), seed=20260804
        )
        for exchange, rows in decisions.groupby("exchange")
    }
    months = pd.to_datetime(decisions["signal_timestamp"], utc=True).dt.to_period("M")
    positive_months = float(decisions["net_return_r_1x"].groupby(months).sum().gt(0).mean())
    calendar = pd.date_range(
        pd.to_datetime(decisions["signal_timestamp"], utc=True).min().floor("D"),
        pd.to_datetime(decisions["signal_timestamp"], utc=True).max().ceil("D"),
        freq="1D",
    )
    daily = (
        decisions.assign(day=pd.to_datetime(decisions["signal_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["net_return_r_1x"]
        .sum()
        .reindex(calendar, fill_value=0.0)
    )
    comparison = spa_reality_check(
        pd.DataFrame({"v14_policy": daily, "flat": np.zeros(len(daily))}, index=calendar),
        control_expert_id="flat",
    )
    present = set(metrics["net_1x"]) == set(EXCHANGES)
    gates = {
        "three_exchanges_present": present,
        "minimum_100_trades_each_exchange": present
        and all(v["trades"] >= 100 for v in metrics["net_1x"].values()),
        "expectancy_positive_each_exchange": present
        and all(v["expectancy_r"] > 0 for v in metrics["net_1x"].values()),
        "lower_bound_positive_each_exchange": present
        and all(lower.get(e, -np.inf) > 0 for e in EXCHANGES),
        "profit_factor_each_exchange": present
        and all(v["profit_factor"] >= 1.15 for v in metrics["net_1x"].values()),
        "drawdown_each_exchange": present
        and all(v["max_drawdown"] <= 0.08 for v in metrics["net_1x"].values()),
        "stress_2x_each_exchange": present
        and all(v["expectancy_r"] >= 0 for v in metrics["stress_2x"].values()),
        "positive_window_majority": positive_months > 0.5,
        "spa": comparison["spa_pvalue"] <= 0.05,
        "reality_check": comparison["reality_check_pvalue"] <= 0.05,
    }
    return {
        "ready": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "expectancy_lower_bound_r": lower,
        "positive_month_fraction": positive_months,
        "multiple_comparison": comparison,
        "total_candidates_evaluated": candidates_evaluated,
    }


def protocol_hash() -> str:
    payload = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_rule": primary_rule(),
        "features": FEATURES,
        "cost_bps": BASE_COST_BPS,
        "shadow_cost_bps": SHADOW_COST_BPS,
        "walk_forward": {
            "train_weeks": TRAIN_WEEKS,
            "calibration_weeks": CALIBRATION_WEEKS,
            "test_weeks": TEST_WEEKS,
            "step_weeks": 4,
            "embargo_hours": EMBARGO_HOURS,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def primary_rule() -> dict[str, Any]:
    return {
        "center": "rolling_vwap_24h",
        "timeframes_minutes": list(TIMEFRAMES),
        "excursion_z": EXCURSION_Z,
        "reentry_z": REENTRY_Z,
        "reset_z": RESET_Z,
        "stop_atr": STOP_ATR,
        "stop_rule": "two_bar_reentry_swing_plus_0.25_atr_min_0.75_atr_max_2.0_atr",
        "target": "fixed_signal_vwap",
        "maximum_holding_hours": 8,
        "one_event_per_excursion": True,
        "regime_gate": (
            "stable_3_bars_in_range_or_trend; shock_and_unknown_excluded; "
            "model_decides_trade_or_flat"
        ),
        "same_bar_stop_target": "worst_case_stop",
    }


def preregister(config_path: Path) -> dict[str, Any]:
    if FORWARD_LOCK_PATH.exists():
        locked = json.loads(FORWARD_LOCK_PATH.read_text(encoding="utf-8"))
        if locked.get("protocol_sha256") != protocol_hash():
            raise RuntimeError(
                "V14 forward protocol is locked; source changes require a new version"
            )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    payload = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "run_id": f"hybrid-v14-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "config_sha256": config_hash,
        "primary_rule": primary_rule(),
        "features": list(FEATURES),
        "orderflow_sha256": _sha256(ORDERFLOW_PATH),
        "training_exchanges": list(EXCHANGES),
        "symbol": "BTCUSDT",
        "target_exchange": "bitunix_shadow_only",
        "confirmation": "strictly_forward_after_historical_cutoff",
        "walk_forward": {
            "train_weeks": TRAIN_WEEKS,
            "calibration_weeks": CALIBRATION_WEEKS,
            "test_weeks": TEST_WEEKS,
            "step_weeks": 4,
            "embargo_hours": EMBARGO_HOURS,
        },
    }
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def forward_readiness(
    *,
    cutoff: pd.Timestamp,
    candles_path: Path = BITUNIX_CANDLES_PATH,
    microstructure_root: Path = BITUNIX_MICROSTRUCTURE_ROOT,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    today = current.date()
    candle_counts: dict[str, set[int]] = {}
    if candles_path.exists():
        with candles_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                    timestamp = int(payload["candle"]["time"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                instant = pd.Timestamp(timestamp, unit="ms", tz="UTC")
                if instant <= cutoff or instant.date() >= today:
                    continue
                candle_counts.setdefault(instant.date().isoformat(), set()).add(timestamp)
    complete_candle_days = sorted(
        day for day, timestamps in candle_counts.items() if len(timestamps) >= 274
    )
    completed_micro_days: list[str] = []
    for path in microstructure_root.glob("btcusdt_*.parquet"):
        try:
            day = datetime.strptime(path.stem.removeprefix("btcusdt_"), "%Y-%m-%d").date()
            if day <= cutoff.date() or day >= today or path.stat().st_size == 0:
                continue
            frame = pd.read_parquet(path, columns=["exchange_timestamp", "event_type"])
            timestamps = pd.to_datetime(frame["exchange_timestamp"], format="mixed", utc=True)
            coverage = timestamps.max() - timestamps.min()
            if coverage >= pd.Timedelta(hours=12) and set(frame["event_type"]) >= {"book", "trade"}:
                completed_micro_days.append(day.isoformat())
        except (KeyError, OSError, ValueError):
            continue
    status_path = microstructure_root / "status_btcusdt.json"
    collector = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    result = {
        "cutoff": cutoff.isoformat(),
        "alpha_complete_days": complete_candle_days,
        "alpha_distinct_days": len(complete_candle_days),
        "execution_complete_days": sorted(completed_micro_days),
        "execution_distinct_days": len(completed_micro_days),
        "minimum_days": 30,
        "ready_for_forward_audit": (
            len(complete_candle_days) >= 30 and len(completed_micro_days) >= 30
        ),
        "collector_connected": bool(collector.get("connected")),
        "collector_updated_at": collector.get("updated_at"),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(FORWARD_STATUS_PATH, result)
    return result


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def freeze_research_candidate(app: AppConfig, config_path: Path) -> dict[str, Any]:
    if FORWARD_LOCK_PATH.exists():
        raise FileExistsError(f"forward protocol already frozen: {FORWARD_LOCK_PATH}")
    inventory = btc_inventory()
    cutoff = min(pd.Timestamp(item["end"]) for item in inventory.values())
    start = min(pd.Timestamp(item["start"]) for item in inventory.values())
    orderflow = load_orderflow_context(start, cutoff)
    protocol = preregister(config_path)
    local, matrices = build_matrices(app, inventory, orderflow, resume=True)
    matrix = _with_context(matrices, local, EXCHANGES)
    times = pd.to_datetime(matrix["signal_timestamp"], utc=True)
    calibration_start = cutoff - pd.Timedelta(weeks=8)
    train_start = calibration_start - pd.Timedelta(weeks=52)
    train = matrix.loc[
        times.between(train_start, calibration_start, inclusive="left")
        & pd.to_datetime(matrix["exit_timestamp"], utc=True).lt(calibration_start)
    ]
    calibration = matrix.loc[times.between(calibration_start, cutoff, inclusive="both")]
    fitted = fit_meta_model(train, calibration)
    if not fitted.get("enabled"):
        raise RuntimeError(f"research candidate cannot be fitted: {fitted.get('reason')}")
    _atomic_joblib(
        RESEARCH_BUNDLE_PATH,
        {
            "status": "RESEARCH_SHADOW_ONLY",
            "paper_enabled": False,
            "live_enabled": False,
            "protocol": protocol,
            "cutoff": cutoff.isoformat(),
            "models": fitted,
            "features": FEATURES,
        },
    )
    lock = {
        "status": "PERMANENT_FORWARD_PROTOCOL_LOCK",
        "protocol_sha256": protocol["protocol_sha256"],
        "bundle_sha256": _sha256(RESEARCH_BUNDLE_PATH),
        "cutoff": cutoff.isoformat(),
        "source_sha256": {exchange: item["sha256"] for exchange, item in inventory.items()},
        "orderflow_sha256": _sha256(ORDERFLOW_PATH),
        "minimum_forward_days": 30,
        "minimum_forward_trades": 100,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _exclusive_json(FORWARD_LOCK_PATH, lock)
    return lock


def write_v14_failure(error: Exception) -> None:
    _status("failed", f"{type(error).__name__}: {error}", 0, block="FAILED")


def _status(phase: str, detail: str, percent: float, **extra: Any) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-forward", action="store_true")
    mode.add_argument("--forward-status", action="store_true")
    arguments = parser.parse_args()
    try:
        app = load_config(arguments.config)
        if arguments.freeze_forward:
            print(json.dumps(freeze_research_candidate(app, arguments.config), indent=2))
        elif arguments.forward_status:
            if not FORWARD_LOCK_PATH.exists():
                raise RuntimeError("freeze the V14 forward protocol before checking readiness")
            lock = json.loads(FORWARD_LOCK_PATH.read_text(encoding="utf-8"))
            print(json.dumps(forward_readiness(cutoff=pd.Timestamp(lock["cutoff"])), indent=2))
        else:
            run_v14(app, arguments.config, resume=arguments.resume, smoke=arguments.smoke)
    except Exception as error:
        write_v14_failure(error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
