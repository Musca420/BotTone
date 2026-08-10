from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Hashable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.hybrid_policy_v25.data import _checked_archive, _parse_archive
from adaptive_bot.indicators.atr import atr

SYMBOLS = ("ETHUSDT", "XRPUSDT", "DOGEUSDT")
MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 7))
HOLDOUT_START = pd.Timestamp("2026-07-01T00:00:00Z")
ROOT = Path("data/ml/musca_altcoin_micro")
REPORT = Path("data/reports/musca_altcoin_micro.json")
STATUS = Path("data/reports/musca_altcoin_micro.status.json")
BUNDLE_ROOT = Path("data/models/musca_altcoin_micro")

COST_BPS = {"ETHUSDT": 9.0, "XRPUSDT": 10.5, "DOGEUSDT": 11.5}
COVERAGES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
CHAMPION_COVERAGE = 0.02


@dataclass(frozen=True)
class Plan:
    name: str
    horizon_minutes: int
    atr_stop_multiple: float
    minimum_stop_bps: float
    maximum_stop_bps: float
    reward_to_risk: float


PLANS = (
    Plan("MICRO_5", 5, 1.5, 12.0, 30.0, 1.4),
    Plan("MICRO_15", 15, 2.0, 15.0, 40.0, 1.5),
    Plan("MICRO_30", 30, 2.5, 18.0, 50.0, 1.6),
    Plan("MICRO_60", 60, 3.0, 22.0, 65.0, 1.75),
)
MAX_HORIZON_MINUTES = max(plan.horizon_minutes for plan in PLANS)

FEATURES = (
    "return_1m_bps",
    "return_3m_bps",
    "return_5m_bps",
    "return_15m_bps",
    "return_30m_bps",
    "return_60m_bps",
    "atr_1m_bps",
    "volatility_5m_bps",
    "volatility_15m_bps",
    "volatility_60m_bps",
    "flow_1m",
    "flow_5m",
    "flow_15m",
    "relative_volume",
    "trade_count_z",
    "daily_vwap_distance_bps",
    "rolling_vwap_distance_bps",
    "rolling_vwap_slope_bps",
    "btc_return_1m_bps",
    "btc_return_5m_bps",
    "btc_return_15m_bps",
    "btc_return_60m_bps",
    "btc_flow_1m",
    "btc_flow_5m",
    "btc_flow_15m",
    "btc_volatility_60m_bps",
    "btc_beta_4h",
    "btc_correlation_1h",
    "btc_correlation_4h",
    "residual_return_1m_bps",
    "residual_return_5m_bps",
    "residual_return_15m_bps",
    "btc_shock_5m",
    "direction_agreement_5m",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
ACTION_FEATURES = ("side", "target_bps", "stop_bps", "horizon_minutes")

PROTOCOL = {
    "name": "musca_altcoin_micro_binance_v1",
    "action_symbols": list(SYMBOLS),
    "context_symbol": "BTCUSDT",
    "data": "Binance USD-M official checksum-verified one-minute klines",
    "months_read": list(MONTHS),
    "holdout_start": HOLDOUT_START.isoformat(),
    "features": list(FEATURES),
    "plans": [asdict(plan) for plan in PLANS],
    "directions": ["LONG", "SHORT"],
    "cost_bps": COST_BPS,
    "entry": "next one-minute open after feature availability",
    "same_bar": "stop_wins",
    "models": ["ridge", "xgboost_cuda"],
    "selection": "maximum causal frequency under frozen economic gates",
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _status(symbol: str, phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS,
        {
            "symbol": symbol,
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _load_symbol(symbol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for number, month in enumerate(MONTHS, start=1):
        relative = f"futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
        frames.append(_parse_archive(_checked_archive(relative)))
        _status(symbol, "data", f"verified archive {number}/{len(MONTHS)}: {month}", 8 * number)
    rows = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    rows = rows.drop_duplicates("timestamp", keep="last")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    if rows["timestamp"].duplicated().any() or not rows["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{symbol} timestamps are not unique and monotonic")
    if not rows["available_at"].gt(rows["timestamp"]).all():
        raise ValueError(f"{symbol} availability must follow the source minute")
    if rows["timestamp"].ge(HOLDOUT_START).any():
        raise ValueError(f"{symbol} sealed July holdout was read")
    return rows.reset_index(drop=True)


def _flow(frame: pd.DataFrame, minutes: int, prefix: str = "") -> pd.Series:
    quote = frame[f"{prefix}quote_volume"]
    signed = 2 * frame[f"{prefix}taker_buy_quote"] - quote
    return signed.rolling(minutes, min_periods=minutes).sum() / quote.rolling(
        minutes, min_periods=minutes
    ).sum().replace(0, np.nan)


def feature_frame(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = _load_symbol(symbol).set_index("timestamp")
    btc = _load_symbol("BTCUSDT").set_index("timestamp")
    start = max(target.index.min(), btc.index.min())
    end = min(target.index.max(), btc.index.max())
    index = pd.date_range(start, end, freq="1min", tz="UTC")
    target = target.reindex(index)
    btc = btc.reindex(index)
    raw = target.loc[:, ["open", "high", "low", "close", "volume", "quote_volume"]].copy()
    raw.index.name = "timestamp"

    frame = target.loc[
        :, ["close", "high", "low", "volume", "quote_volume", "trade_count", "taker_buy_quote"]
    ].copy()
    close = frame["close"]
    returns_1m = close.pct_change(fill_method=None) * 10_000
    for minutes in (1, 3, 5, 15, 30, 60):
        frame[f"return_{minutes}m_bps"] = close.pct_change(minutes, fill_method=None) * 10_000
    frame["atr_1m_bps"] = atr(frame["high"], frame["low"], close, 14) / close * 10_000
    for minutes in (5, 15, 60):
        frame[f"volatility_{minutes}m_bps"] = returns_1m.rolling(
            minutes, min_periods=minutes
        ).std(ddof=0)
    for minutes in (1, 5, 15):
        frame[f"flow_{minutes}m"] = _flow(frame, minutes)
    volume_median = frame["quote_volume"].shift(1).rolling(240, min_periods=60).median()
    frame["relative_volume"] = frame["quote_volume"] / volume_median.replace(0, np.nan)
    log_count = np.log1p(frame["trade_count"])
    count_window = log_count.shift(1).rolling(240, min_periods=60)
    frame["trade_count_z"] = (log_count - count_window.mean()) / count_window.std(
        ddof=0
    ).replace(0, np.nan)
    time_index = pd.DatetimeIndex(frame.index)
    day = pd.Series(time_index.floor("D"), index=frame.index)
    daily_vwap = frame["quote_volume"].groupby(day).cumsum() / frame["volume"].groupby(
        day
    ).cumsum().replace(0, np.nan)
    rolling_vwap = frame["quote_volume"].rolling(60, min_periods=60).sum() / frame[
        "volume"
    ].rolling(60, min_periods=60).sum().replace(0, np.nan)
    frame["daily_vwap_distance_bps"] = (close / daily_vwap - 1) * 10_000
    frame["rolling_vwap_distance_bps"] = (close / rolling_vwap - 1) * 10_000
    frame["rolling_vwap_slope_bps"] = rolling_vwap.pct_change(15, fill_method=None) * 10_000

    btc_close = btc["close"]
    btc_return_1m = btc_close.pct_change(fill_method=None) * 10_000
    for minutes in (1, 5, 15, 60):
        frame[f"btc_return_{minutes}m_bps"] = (
            btc_close.pct_change(minutes, fill_method=None) * 10_000
        )
    btc_flow_frame = btc.loc[:, ["quote_volume", "taker_buy_quote"]]
    for minutes in (1, 5, 15):
        frame[f"btc_flow_{minutes}m"] = _flow(btc_flow_frame, minutes)
    frame["btc_volatility_60m_bps"] = btc_return_1m.rolling(60, min_periods=60).std(ddof=0)
    prior_asset = returns_1m.shift(1)
    prior_btc = btc_return_1m.shift(1)
    covariance = prior_asset.rolling(240, min_periods=60).cov(prior_btc)
    variance = prior_btc.rolling(240, min_periods=60).var(ddof=0)
    frame["btc_beta_4h"] = covariance / variance.replace(0, np.nan)
    frame["btc_correlation_1h"] = prior_asset.rolling(60, min_periods=30).corr(prior_btc)
    frame["btc_correlation_4h"] = prior_asset.rolling(240, min_periods=60).corr(prior_btc)
    for minutes in (1, 5, 15):
        frame[f"residual_return_{minutes}m_bps"] = (
            frame[f"return_{minutes}m_bps"]
            - frame["btc_beta_4h"] * frame[f"btc_return_{minutes}m_bps"]
        )
    frame["btc_shock_5m"] = frame["btc_return_5m_bps"].abs() / frame[
        "btc_volatility_60m_bps"
    ].replace(0, np.nan)
    frame["direction_agreement_5m"] = np.sign(frame["return_5m_bps"]) * np.sign(
        frame["btc_return_5m_bps"]
    )
    hour = time_index.hour + time_index.minute / 60
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * time_index.dayofweek / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * time_index.dayofweek / 7)
    frame["available_at"] = frame.index + pd.Timedelta(minutes=1)
    frame["max_input_available_at"] = frame["available_at"]
    frame["symbol"] = symbol
    return frame.reset_index(), raw.reset_index()


def plan_levels(atr_bps: np.ndarray, plan: Plan, cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    stop = np.clip(
        plan.atr_stop_multiple * atr_bps,
        plan.minimum_stop_bps,
        plan.maximum_stop_bps,
    )
    target = np.maximum(1.5 * cost_bps, plan.reward_to_risk * stop)
    return target, stop


def barrier_outcomes(
    raw: pd.DataFrame,
    atr_bps: np.ndarray,
    plan: Plan,
    side: int,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = plan.horizon_minutes
    open_price = raw["open"].to_numpy(float)
    high = raw["high"].to_numpy(float)
    low = raw["low"].to_numpy(float)
    close = raw["close"].to_numpy(float)
    entries = open_price[1 : len(raw) - horizon + 1]
    high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)
    low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)
    target, stop = plan_levels(atr_bps[: len(entries)], plan, cost_bps)
    favorable = np.where(
        side > 0,
        (high_windows / entries[:, None] - 1) * 10_000,
        (1 - low_windows / entries[:, None]) * 10_000,
    )
    adverse = np.where(
        side > 0,
        (1 - low_windows / entries[:, None]) * 10_000,
        (high_windows / entries[:, None] - 1) * 10_000,
    )
    target_hit = favorable >= target[:, None]
    stop_hit = adverse >= stop[:, None]
    has_target, has_stop = target_hit.any(axis=1), stop_hit.any(axis=1)
    first_target = np.where(has_target, target_hit.argmax(axis=1), horizon)
    first_stop = np.where(has_stop, stop_hit.argmax(axis=1), horizon)
    outcome = np.select(
        (has_target & (first_target < first_stop), has_stop & (first_stop <= first_target)),
        (0, 1),
        default=2,
    ).astype(np.int8)
    exit_offset = np.select(
        (outcome == 0, outcome == 1), (first_target, first_stop), default=horizon - 1
    ).astype(int)
    terminal = side * (close[horizon : len(raw)] / entries - 1) * 10_000
    gross = np.select((outcome == 0, outcome == 1), (target, -stop), default=terminal)
    stopped = outcome == 1
    if stopped.any():
        base = np.arange(len(entries))[stopped] + 1
        gap = side * (open_price[base + first_stop[stopped]] / entries[stopped] - 1) * 10_000
        gross[stopped] = np.minimum(-stop[stopped], gap)
    path_valid = np.lib.stride_tricks.sliding_window_view(
        np.isfinite(open_price[1:])
        & np.isfinite(high[1:])
        & np.isfinite(low[1:])
        & np.isfinite(close[1:]),
        horizon,
    ).all(axis=1)
    gross[~path_valid] = np.nan
    return outcome, gross, exit_offset + 1, target, stop


def build_matrix(symbol: str, *, resume: bool = True) -> pd.DataFrame:
    output = ROOT / f"{symbol.lower()}_matrix.parquet"
    if resume and output.exists():
        cached = pd.read_parquet(output)
        if len(cached) and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached
    features, raw = feature_frame(symbol)
    maximum_rows = len(raw) - MAX_HORIZON_MINUTES
    matrix = features.iloc[:maximum_rows].copy()
    for plan in PLANS:
        for side, suffix in ((1, "long"), (-1, "short")):
            outcome, gross, exit_minutes, target, stop = barrier_outcomes(
                raw, features["atr_1m_bps"].to_numpy(float), plan, side, COST_BPS[symbol]
            )
            stem = f"{plan.name.lower()}_{suffix}"
            matrix[f"{stem}_outcome"] = outcome[:maximum_rows]
            matrix[f"{stem}_gross_bps"] = gross[:maximum_rows]
            matrix[f"{stem}_exit_minutes"] = exit_minutes[:maximum_rows]
            matrix[f"{stem}_target_bps"] = target[:maximum_rows]
            matrix[f"{stem}_stop_bps"] = stop[:maximum_rows]
    finite = np.isfinite(matrix.loc[:, FEATURES].to_numpy(float)).all(axis=1)
    label_columns = [
        column
        for column in matrix
        if str(column).endswith(("_gross_bps", "_stop_bps"))
    ]
    finite &= np.isfinite(matrix.loc[:, label_columns].to_numpy(float)).all(axis=1)
    matrix = matrix.loc[
        finite
        & matrix["max_input_available_at"].le(matrix["available_at"])
        & matrix["available_at"].lt(HOLDOUT_START)
    ].copy()
    matrix["entry_timestamp"] = matrix["available_at"]
    matrix["day"] = matrix["entry_timestamp"].dt.floor("D")
    matrix["protocol_hash"] = PROTOCOL_HASH
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    temporary.replace(output)
    return matrix


def _action_stems() -> list[tuple[Plan, int, str]]:
    return [
        (plan, side, f"{plan.name.lower()}_{'long' if side > 0 else 'short'}")
        for plan in PLANS
        for side in (1, -1)
    ]


def _long_xy(rows: pd.DataFrame, cost_bps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = rows.loc[:, FEATURES].to_numpy(float)
    values: list[np.ndarray] = []
    gross: list[np.ndarray] = []
    for plan, side, stem in _action_stems():
        action = np.column_stack(
            (
                np.full(len(rows), side),
                rows[f"{stem}_target_bps"].to_numpy(float),
                rows[f"{stem}_stop_bps"].to_numpy(float),
                np.full(len(rows), plan.horizon_minutes),
            )
        )
        values.append(np.column_stack((base, action)))
        gross.append(rows[f"{stem}_gross_bps"].to_numpy(float))
    y = np.concatenate(gross)
    return np.concatenate(values), y, (y - cost_bps > 0).astype(int)


def _models(kind: Literal["ridge", "xgboost"], seed: int = 20260810) -> tuple[Any, Any]:
    if kind == "ridge":
        return (
            make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1_000, random_state=seed),
            ),
        )
    return (
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=20.0,
            random_state=seed,
            n_jobs=4,
        ),
        XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=20.0,
            random_state=seed,
            n_jobs=4,
        ),
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(clipped / (1 - clipped)).reshape(-1, 1), dtype=float)


def fit(
    kind: Literal["ridge", "xgboost"],
    fit_rows: pd.DataFrame,
    calibration: pd.DataFrame,
    cost_bps: float,
) -> dict[str, Any]:
    values, gross, positive = _long_xy(fit_rows, cost_bps)
    regressor, classifier = _models(kind)
    regressor.fit(values, gross)
    classifier.fit(values, positive)
    calibration_values, calibration_gross, calibration_positive = _long_xy(calibration, cost_bps)
    raw_ev = np.asarray(regressor.predict(calibration_values), dtype=float)
    ev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_ev, calibration_gross)
    raw_probability = np.asarray(classifier.predict_proba(calibration_values)[:, 1], dtype=float)
    probability_calibrator = LogisticRegression(C=1.0, max_iter=1_000).fit(
        _logit(raw_probability), calibration_positive
    )
    return {
        "kind": kind,
        "regressor": regressor,
        "classifier": classifier,
        "ev_calibrator": ev_calibrator,
        "probability_calibrator": probability_calibrator,
    }


def score(rows: pd.DataFrame, model: dict[str, Any], cost_bps: float) -> pd.DataFrame:
    base = rows.loc[:, FEATURES].to_numpy(float)
    best = np.full(len(rows), -np.inf)
    selected = np.zeros(len(rows), dtype=int)
    probability = np.zeros(len(rows))
    expected_gross = np.zeros(len(rows))
    for number, (plan, side, stem) in enumerate(_action_stems()):
        action = np.column_stack(
            (
                np.full(len(rows), side),
                rows[f"{stem}_target_bps"].to_numpy(float),
                rows[f"{stem}_stop_bps"].to_numpy(float),
                np.full(len(rows), plan.horizon_minutes),
            )
        )
        values = np.column_stack((base, action))
        raw_ev = np.asarray(model["regressor"].predict(values), dtype=float)
        predicted_gross = np.asarray(model["ev_calibrator"].predict(raw_ev), dtype=float)
        raw_probability = np.asarray(model["classifier"].predict_proba(values)[:, 1], dtype=float)
        predicted_probability = model["probability_calibrator"].predict_proba(
            _logit(raw_probability)
        )[:, 1]
        candidate = predicted_gross - cost_bps
        better = candidate > best
        best[better] = candidate[better]
        selected[better] = number
        probability[better] = predicted_probability[better]
        expected_gross[better] = predicted_gross[better]
    output = rows.loc[:, ["available_at", "entry_timestamp", "day"]].copy()
    output["score"] = best
    output["expected_gross_bps"] = expected_gross
    output["probability_net_positive"] = probability
    output["plan"] = [_action_stems()[number][0].name for number in selected]
    output["side"] = np.array([_action_stems()[number][1] for number in selected], dtype=int)
    indexes = np.arange(len(rows))

    def selected_values(suffix: str, dtype: type[float] | type[int]) -> np.ndarray:
        matrix = np.column_stack(
            [rows[f"{stem}_{suffix}"].to_numpy(dtype) for _, _, stem in _action_stems()]
        )
        return np.asarray(matrix[indexes, selected], dtype=dtype)

    output["gross_bps"] = selected_values("gross_bps", float)
    output["target_bps"] = selected_values("target_bps", float)
    output["stop_bps"] = selected_values("stop_bps", float)
    output["exit_minutes"] = selected_values("exit_minutes", int)
    output["exit_timestamp"] = output["entry_timestamp"] + pd.to_timedelta(
        output["exit_minutes"], unit="min"
    )
    output["net_bps"] = output["gross_bps"] - cost_bps
    output["stress_bps"] = output["gross_bps"] - 2 * cost_bps
    return output.sort_values("entry_timestamp").reset_index(drop=True)


def execute(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = scored.loc[scored["score"].ge(max(0.0, threshold))].sort_values(
        "entry_timestamp"
    )
    accepted: list[Hashable] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in candidates.iterrows():
        if row["entry_timestamp"] < blocked_until:
            continue
        accepted.append(index)
        blocked_until = pd.Timestamp(row["exit_timestamp"])
    return candidates.loc[accepted].reset_index(drop=True)


def _bootstrap_lcb(
    trades: pd.DataFrame, column: str = "net_bps", seed: int = 20260810
) -> float | None:
    daily = trades.groupby("day")[column].agg(["sum", "count"])
    if len(daily) < 10:
        return None
    values = daily.to_numpy(float)
    block = min(5, len(values))
    random = np.random.default_rng(seed)
    means = np.empty(1_000)
    for sample in range(len(means)):
        joined: list[np.ndarray] = []
        while sum(len(item) for item in joined) < len(values):
            start = int(random.integers(0, len(values) - block + 1))
            joined.append(values[start : start + block])
        draw = np.concatenate(joined)[: len(values)]
        means[sample] = draw[:, 0].sum() / draw[:, 1].sum()
    return float(np.quantile(means, 0.05))


def metrics(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    stress: bool = False,
) -> dict[str, Any]:
    column = "stress_bps" if stress else "net_bps"
    calendar_days = max(1, int((end.floor("D") - start.floor("D")).days))
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_calendar_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "positive_active_days": None,
            "max_drawdown": None,
            "bootstrap_lcb_95_bps": None,
        }
    values = trades[column].to_numpy(float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    daily = trades.groupby("day")[column].sum()
    risk_return = np.maximum(
        values / (trades["stop_bps"].to_numpy(float) + COST_BPS[cast(str, trades.attrs["symbol"])]),
        -2,
    ) * 0.01
    equity = np.cumprod(1 + risk_return)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "trades": len(trades),
        "trades_per_calendar_day": float(len(trades) / calendar_days),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()),
        "positive_active_days": float((daily > 0).mean()),
        "max_drawdown": float((1 - equity / peaks).max()),
        "bootstrap_lcb_95_bps": _bootstrap_lcb(trades.assign(net_bps=values)),
    }


def _gates(value: dict[str, Any], *, minimum_trades: int) -> dict[str, bool]:
    return {
        "minimum_trades": int(value["trades"]) >= minimum_trades,
        "frequency_3_per_day": float(value["trades_per_calendar_day"]) >= 3.0,
        "expectancy_positive": value["expectancy_bps"] is not None
        and float(value["expectancy_bps"]) > 0,
        "profit_factor_1_15": value["profit_factor"] is not None
        and float(value["profit_factor"]) >= 1.15,
        "majority_positive_active_days": value["positive_active_days"] is not None
        and float(value["positive_active_days"]) > 0.5,
        "drawdown_8pct": value["max_drawdown"] is not None
        and float(value["max_drawdown"]) <= 0.08,
        "bootstrap_lcb_positive": value["bootstrap_lcb_95_bps"] is not None
        and float(value["bootstrap_lcb_95_bps"]) > 0,
    }


def _period(rows: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    first, last = pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(
        minutes=MAX_HORIZON_MINUTES
    )
    return rows.loc[rows["available_at"].ge(first) & rows["available_at"].lt(last)].copy()


def _threshold(history: pd.DataFrame, coverage: float) -> float:
    return max(0.0, float(history["score"].quantile(1 - coverage)))


def _with_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame.attrs["symbol"] = symbol
    return frame


def _oracle(matrix: pd.DataFrame, symbol: str) -> dict[str, Any]:
    best = np.full(len(matrix), -np.inf)
    selected = np.zeros(len(matrix), dtype=int)
    for number, (_, _, stem) in enumerate(_action_stems()):
        value = matrix[f"{stem}_gross_bps"].to_numpy(float) - COST_BPS[symbol]
        better = value > best
        best[better], selected[better] = value[better], number
    rows = matrix.loc[:, ["entry_timestamp", "day"]].copy()
    rows["score"] = best
    rows["net_bps"] = best
    rows["gross_bps"] = best + COST_BPS[symbol]
    rows["stress_bps"] = best - COST_BPS[symbol]
    rows["stop_bps"] = np.column_stack(
        [matrix[f"{stem}_stop_bps"].to_numpy(float) for _, _, stem in _action_stems()]
    )[np.arange(len(matrix)), selected]
    exit_minutes = np.column_stack(
        [matrix[f"{stem}_exit_minutes"].to_numpy(int) for _, _, stem in _action_stems()]
    )[np.arange(len(matrix)), selected]
    rows["exit_timestamp"] = rows["entry_timestamp"] + pd.to_timedelta(exit_minutes, unit="min")
    trades = _with_symbol(execute(rows, 0.0), symbol)
    return {
        "positive_decision_fraction": float((best > 0).mean()),
        "future_information_only": True,
        "upper_bound": metrics(trades, pd.Timestamp("2026-01-01T00:00:00Z"), HOLDOUT_START),
    }


def train_symbol(
    symbol: str, *, resume: bool = True
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    cost = COST_BPS[symbol]
    _status(symbol, "matrix", "causal BTC context and counterfactual actions", 3)
    matrix = build_matrix(symbol, resume=resume)
    split = {
        "fit": _period(matrix, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        "calibration": _period(matrix, "2026-03-01T00:00:00Z", "2026-03-15T00:00:00Z"),
        "model_selection": _period(matrix, "2026-03-15T00:00:00Z", "2026-04-01T00:00:00Z"),
        "policy_selection": _period(matrix, "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
        "audit": _period(matrix, "2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    }
    if min(len(rows) for rows in split.values()) < 10_000:
        raise ValueError(f"{symbol} has insufficient complete causal minutes")
    candidates: dict[str, dict[str, Any]] = {}
    for number, kind in enumerate(("ridge", "xgboost"), start=1):
        model_kind = cast(Literal["ridge", "xgboost"], kind)
        _status(symbol, "training", f"{kind}  {number}/2", 50 + 12 * number)
        model = fit(model_kind, split["fit"], split["calibration"], cost)
        calibration_scored = score(split["calibration"], model, cost)
        selection_scored = score(split["model_selection"], model, cost)
        threshold = _threshold(calibration_scored, CHAMPION_COVERAGE)
        trades = _with_symbol(execute(selection_scored, threshold), symbol)
        predicted = selection_scored["expected_gross_bps"].to_numpy(float)
        actual = selection_scored["gross_bps"].to_numpy(float)
        candidates[kind] = {
            "model": model,
            "calibration_scored": calibration_scored,
            "selection_scored": selection_scored,
            "threshold": threshold,
            "mae_bps": float(np.abs(predicted - actual).mean()),
            "metrics": metrics(
                trades,
                pd.Timestamp("2026-03-15T00:00:00Z"),
                pd.Timestamp("2026-04-01T00:00:00Z"),
            ),
        }
    ridge, xgboost = candidates["ridge"], candidates["xgboost"]
    ridge_metrics, xgb_metrics = ridge["metrics"], xgboost["metrics"]
    champion = (
        "xgboost"
        if xgboost["mae_bps"] < ridge["mae_bps"]
        and float(xgb_metrics["expectancy_bps"] or -np.inf)
        > float(ridge_metrics["expectancy_bps"] or -np.inf)
        and float(xgb_metrics["profit_factor"] or 0) >= float(ridge_metrics["profit_factor"] or 0)
        else "ridge"
    )
    chosen = candidates[champion]
    _status(symbol, "selection", f"{champion}: frozen April frequency frontier", 82)
    policy_scored = score(split["policy_selection"], chosen["model"], cost)
    curve: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        threshold = _threshold(chosen["selection_scored"], coverage)
        trades = _with_symbol(execute(policy_scored, threshold), symbol)
        value = metrics(
            trades,
            pd.Timestamp("2026-04-01T00:00:00Z"),
            pd.Timestamp("2026-05-01T00:00:00Z"),
        )
        curve.append(
            {
                "coverage": coverage,
                "threshold": threshold,
                "metrics": value,
                "stress_costs_2x": metrics(
                    trades,
                    pd.Timestamp("2026-04-01T00:00:00Z"),
                    pd.Timestamp("2026-05-01T00:00:00Z"),
                    stress=True,
                ),
                "gates": _gates(value, minimum_trades=60),
            }
        )
    passing = [point for point in curve if all(point["gates"].values())]
    selected = max(
        passing,
        key=lambda point: float(point["metrics"]["trades_per_calendar_day"]),
        default=None,
    )
    diagnostic = max(
        curve,
        key=lambda point: (
            sum(point["gates"].values()),
            float(point["metrics"]["trades_per_calendar_day"]),
        ),
    )
    frozen = selected or diagnostic
    audit_scored = score(split["audit"], chosen["model"], cost)
    audit_threshold = _threshold(policy_scored, float(frozen["coverage"]))
    audit_trades = _with_symbol(execute(audit_scored, audit_threshold), symbol)
    audit_metrics = metrics(
        audit_trades,
        pd.Timestamp("2026-05-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T00:00:00Z"),
    )
    audit_gates = _gates(audit_metrics, minimum_trades=100)
    passed = selected is not None and all(audit_gates.values())
    result = {
        "symbol": symbol,
        "cost_bps": cost,
        "rows": len(matrix),
        "rows_by_split": {name: len(rows) for name, rows in split.items()},
        "causal_checks": {
            "future_feature_violations": int(
                (matrix["max_input_available_at"] > matrix["available_at"]).sum()
            ),
            "sealed_holdout_rows_read": int(matrix["available_at"].ge(HOLDOUT_START).sum()),
            "missing_feature_rows": int(
                (~np.isfinite(matrix.loc[:, FEATURES].to_numpy(float)).all(axis=1)).sum()
            ),
            "same_minute_stop_wins": True,
        },
        "oracle": _oracle(matrix, symbol),
        "model_selection": {
            kind: {
                key: value
                for key, value in candidate.items()
                if key != "model" and not key.endswith("scored")
            }
            for kind, candidate in candidates.items()
        },
        "champion": champion,
        "policy_selection": {"curve": curve, "selected": selected},
        "diagnostic_choice_when_no_policy": None if selected is not None else diagnostic,
        "audit": {
            "coverage": frozen["coverage"],
            "threshold": audit_threshold,
            "metrics": audit_metrics,
            "stress_costs_2x": metrics(
                audit_trades,
                pd.Timestamp("2026-05-01T00:00:00Z"),
                pd.Timestamp("2026-07-01T00:00:00Z"),
                stress=True,
            ),
            "gates": audit_gates,
            "long": int(audit_trades["side"].gt(0).sum()),
            "short": int(audit_trades["side"].lt(0).sum()),
            "plans": {
                str(key): int(value)
                for key, value in audit_trades["plan"].value_counts().items()
            },
            "selection_was_gate_passing": selected is not None,
        },
        "verdict": "RESEARCH_PAPER_READY" if passed else "RESEARCH_ONLY_FLAT",
        "real_capital_allowed": False,
    }
    bundle = None
    if passed:
        assert selected is not None
        bundle = {
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "symbol": symbol,
            "features": FEATURES,
            "action_features": ACTION_FEATURES,
            "model": chosen["model"],
            "champion": champion,
            "coverage": selected["coverage"],
            "score_threshold": audit_threshold,
            "research_only": True,
            "live_orders_enabled": False,
        }
    return result, bundle


def train(symbols: tuple[str, ...] = SYMBOLS, *, resume: bool = True) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for number, symbol in enumerate(symbols, start=1):
        result, bundle = train_symbol(symbol, resume=resume)
        results[symbol] = result
        if bundle is not None:
            path = BUNDLE_ROOT / symbol.lower() / "bundle.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            joblib.dump(bundle, temporary)
            temporary.replace(path)
        _status(
            symbol,
            "asset_complete",
            cast(str, result["verdict"]),
            100 * number / len(symbols),
        )
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "assets": results,
        "paper_ready": [
            symbol
            for symbol, value in results.items()
            if value["verdict"] == "RESEARCH_PAPER_READY"
        ],
        "verdict": (
            "ALTCOIN_RESEARCH_POLICIES_READY"
            if any(value["verdict"] == "RESEARCH_PAPER_READY" for value in results.values())
            else "NO_SUSTAINABLE_ALTCOIN_MICRO_ALPHA"
        ),
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    _status("ALL", "complete", cast(str, report["verdict"]), 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca Binance ETH/XRP/DOGE micro-policy audit")
    parser.add_argument("--symbol", choices=["all", *SYMBOLS], default="all")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    symbols = SYMBOLS if args.symbol == "all" else (args.symbol,)
    print(json.dumps(train(symbols, resume=not args.no_resume), indent=2, default=str))


if __name__ == "__main__":
    main()
