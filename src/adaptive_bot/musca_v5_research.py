from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.musca_v4_research import (
    COST_BPS,
    _metrics,
    _non_overlapping,
)

ROOT = Path("data/ml/musca_v5")
BARS = Path("data/ml/hybrid_v25/asset=BTCUSDT/bars_5m.parquet")
MINUTES = Path("data/ml/hybrid_v25/asset=BTCUSDT/minutes.parquet")
EVENTS = ROOT / "events.parquet"
LABELED = ROOT / "events_with_outcomes.parquet"
REPORT = Path("data/reports/musca_v5_research.json")
STATUS = Path("data/reports/musca_v5_research.status.json")
HOLDOUT_START = pd.Timestamp("2026-05-11T11:30:00Z")
PROTOCOL = {
    "name": "musca_v5_frequent_multi_vwap_v3",
    "market": "BINANCE_BTCUSDT_SPOT_PERP",
    "timeframe_minutes": 5,
    "direction": "continuous_vote_return_1h_4h_ema_weekly_vwap_spot_taker; abs>=2",
    "centers": ["DAILY_VWAP", "ROLLING_4H_VWAP", "CONFIRMED_SWING_AVWAP"],
    "context": ["OPEN_INTEREST", "MARK_SPOT_BASIS", "FUNDING", "AGGRESSIVE_VOLUME"],
    "actions": {
        "FOLLOW": "center touch/cross then continuation beyond previous high/low",
        "FADE": "extension at least 0.5ATR then reversal toward center",
    },
    "entry": {
        "FOLLOW": "next_1m_open_after_confirmed_vwap_retest",
        "FADE": "next_1m_open",
    },
    "stop": "beyond_three_bar_pullback_and_center; 12..150bps; never_widens",
    "management": {
        "FOLLOW": "no_target; protect_after_1R; trailing_30m_swing; timeout_12h",
        "FADE": "full_exit_at_operating_vwap; structural_stop; timeout_2h",
    },
    "minimum_planned_move_bps": 3 * COST_BPS,
    "cooldown_minutes": 30,
    "base_cost_bps": COST_BPS,
    "stress_cost_bps": 2 * COST_BPS,
    "oos_frequency": {"minimum_trades": 100, "minimum_per_day": 0.5},
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
ML_FEATURES = (
    "event_family_code",
    "direction",
    "trend_vote",
    "return_1h",
    "return_4h",
    "ema_spread_atr",
    "taker_imbalance_15m",
    "spot_taker_imbalance_15m",
    "distance_vwap_atr",
    "risk_bps",
    "relative_volume",
    "trade_count_z",
    "aggressive_volume_z",
    "atr_percentile",
    "funding_z",
    "basis_bps",
    "oi_change_1h",
    "return_oi_interaction",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": percent,
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _session_vwap(data: pd.DataFrame, period: pd.Series) -> pd.Series:
    volume = data["perp_volume"].groupby(period).cumsum()
    quote = data["perp_quote_volume"].groupby(period).cumsum()
    return quote / volume.replace(0, np.nan)


def _swing_avwap(data: pd.DataFrame, *, long: bool) -> np.ndarray:
    low = data["perp_low"].to_numpy(float)
    high = data["perp_high"].to_numpy(float)
    volume = data["perp_volume"].to_numpy(float)
    quote = data["perp_quote_volume"].to_numpy(float)
    result = np.full(len(data), np.nan)
    anchor = 0
    cumulative_volume = cumulative_quote = 0.0
    for current in range(len(data)):
        pivot = current - 2
        if current >= 4:
            window = low[current - 4 : current + 1] if long else high[current - 4 : current + 1]
            confirmed = (
                low[pivot] <= np.nanmin(window)
                if long
                else high[pivot] >= np.nanmax(window)
            )
            if confirmed:
                anchor = pivot
                cumulative_volume = float(np.nansum(volume[anchor : current + 1]))
                cumulative_quote = float(np.nansum(quote[anchor : current + 1]))
            else:
                cumulative_volume += volume[current]
                cumulative_quote += quote[current]
        else:
            cumulative_volume += volume[current]
            cumulative_quote += quote[current]
        if cumulative_volume > 0:
            result[current] = cumulative_quote / cumulative_volume
    return result


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    validity = "is_available" if "is_available" in bars else "data_valid"
    data = bars.loc[bars[validity]].sort_values("timestamp").reset_index(drop=True).copy()
    timestamp = pd.to_datetime(data["timestamp"], utc=True)
    previous = data["perp_close"].shift(1)
    true_range = pd.concat(
        [
            data["perp_high"] - data["perp_low"],
            (data["perp_high"] - previous).abs(),
            (data["perp_low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["return_1h"] = data["perp_close"].pct_change(12)
    data["return_4h"] = data["perp_close"].pct_change(48)
    data["spot_return_1h"] = data["spot_close"].pct_change(12)
    ema20 = data["perp_close"].ewm(span=20, adjust=False).mean()
    ema50 = data["perp_close"].ewm(span=50, adjust=False).mean()
    data["ema_spread_atr"] = (ema20 - ema50) / data["atr"]
    week = timestamp.dt.tz_localize(None).dt.to_period("W-SUN")
    day = timestamp.dt.floor("D")
    data["daily_vwap"] = _session_vwap(data, day)
    data["weekly_vwap"] = _session_vwap(data, week)
    data["rolling_vwap_4h"] = (
        data["perp_quote_volume"].rolling(48, min_periods=48).sum()
        / data["perp_volume"].rolling(48, min_periods=48).sum().replace(0, np.nan)
    )
    data["swing_avwap_long"] = _swing_avwap(data, long=True)
    data["swing_avwap_short"] = _swing_avwap(data, long=False)
    data["taker_imbalance_15m"] = (
        2 * data["perp_taker_buy_quote"].rolling(3).sum()
        / data["perp_quote_volume"].rolling(3).sum().replace(0, np.nan)
        - 1
    )
    data["spot_taker_imbalance_15m"] = (
        2 * data["spot_taker_buy_quote"].rolling(3).sum()
        / data["spot_quote_volume"].rolling(3).sum().replace(0, np.nan)
        - 1
    )
    quote_median = data["perp_quote_volume"].shift(1).rolling(288, min_periods=96).median()
    data["relative_volume"] = data["perp_quote_volume"] / quote_median
    trade_mean = data["perp_trade_count"].shift(1).rolling(288, min_periods=96).mean()
    trade_std = data["perp_trade_count"].shift(1).rolling(288, min_periods=96).std()
    data["trade_count_z"] = (data["perp_trade_count"] - trade_mean) / trade_std.replace(0, np.nan)
    aggressive = data["taker_imbalance_15m"].abs() * data["perp_quote_volume"]
    aggressive_mean = aggressive.shift(1).rolling(288, min_periods=96).mean()
    aggressive_std = aggressive.shift(1).rolling(288, min_periods=96).std()
    data["aggressive_volume_z"] = (aggressive - aggressive_mean) / aggressive_std.replace(
        0, np.nan
    )
    data["atr_percentile"] = (
        data["atr"].shift(1).rolling(2016, min_periods=288).rank(pct=True) * 100
    )
    funding_mean = data["perp_funding_rate"].shift(1).rolling(2016, min_periods=288).mean()
    funding_std = data["perp_funding_rate"].shift(1).rolling(2016, min_periods=288).std()
    data["funding_z"] = (data["perp_funding_rate"] - funding_mean) / funding_std.replace(
        0, np.nan
    )
    data["basis_bps"] = (
        (data["perp_mark_close"] - data["spot_close"]) / data["spot_close"] * 10_000
    )
    data["return_oi_interaction"] = data["return_1h"] * data["oi_change_1h"]
    components = np.column_stack(
        [
            np.sign(data["return_1h"]),
            np.sign(data["return_4h"]),
            np.sign(data["spot_return_1h"]),
            np.sign(data["ema_spread_atr"]),
            np.sign(data["perp_close"] - data["weekly_vwap"]),
            np.sign(data["taker_imbalance_15m"]),
        ]
    )
    data["trend_vote"] = np.nansum(components, axis=1)
    data["direction"] = np.where(
        data["trend_vote"].ge(2), 1, np.where(data["trend_vote"].le(-2), -1, 0)
    )
    data["available_at"] = timestamp + pd.Timedelta(minutes=5)
    return data


def build_events(bars: pd.DataFrame) -> pd.DataFrame:
    data = build_features(bars)
    events: list[dict[str, Any]] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for current in range(50, len(data) - 1):
        row = data.iloc[current]
        available = pd.Timestamp(row["available_at"])
        side = int(row["direction"])
        required = row[
            [
                "atr",
                "relative_volume",
                "trade_count_z",
                "aggressive_volume_z",
                "atr_percentile",
                "funding_z",
                "basis_bps",
                "oi_change_1h",
            ]
        ].to_numpy(float)
        if side == 0 or available < blocked_until or not np.isfinite(required).all():
            continue
        previous = data.iloc[current - 1]
        confirms = (
            row["perp_close"] > previous["perp_high"]
            if side > 0
            else row["perp_close"] < previous["perp_low"]
        )
        flow = side * row["taker_imbalance_15m"] > 0
        spot = side * row["spot_return_1h"] > 0
        if not (confirms and flow and spot):
            continue
        centers = {
            "DAILY_RECLAIM": float(row["daily_vwap"]),
            "ROLLING_4H_RETEST": float(row["rolling_vwap_4h"]),
            "SWING_AVWAP_RETEST": float(
                row["swing_avwap_long"] if side > 0 else row["swing_avwap_short"]
            ),
        }
        atr_value = float(row["atr"])
        band = 0.25 * atr_value
        touched = {
            family: center
            for family, center in centers.items()
            if np.isfinite(center)
            and (
                float(previous["perp_low"]) <= center + band
                and float(previous["perp_high"]) >= center - band
            )
        }
        if not touched:
            continue
        family, center = min(
            touched.items(), key=lambda item: abs(float(previous["perp_close"]) - item[1])
        )
        pullback = data.iloc[current - 2 : current + 1]
        extreme = (
            float(pullback["perp_low"].min())
            if side > 0
            else float(pullback["perp_high"].max())
        )
        zone_edge = center - side * band
        stop = (
            min(extreme, zone_edge) - 0.1 * atr_value
            if side > 0
            else max(extreme, zone_edge) + 0.1 * atr_value
        )
        risk_bps = side * (float(row["perp_close"]) - stop) / float(row["perp_close"]) * 10_000
        if not 12 <= risk_bps <= 150 or 1.5 * risk_bps < 3 * COST_BPS:
            continue
        events.append(
            {
                "protocol_hash": PROTOCOL_HASH,
                "signal_timestamp": pd.Timestamp(row["timestamp"]),
                "available_at": available,
                "direction": side,
                "event_family": family,
                "event_family_code": list(centers).index(family),
                "operating_vwap": center,
                "stop_price": stop,
                "trend_vote": float(row["trend_vote"]),
                "return_1h": side * float(row["return_1h"]),
                "return_4h": side * float(row["return_4h"]),
                "ema_spread_atr": side * float(row["ema_spread_atr"]),
                "taker_imbalance_15m": side * float(row["taker_imbalance_15m"]),
                "spot_taker_imbalance_15m": side
                * float(row["spot_taker_imbalance_15m"]),
                "distance_vwap_atr": side
                * (float(row["perp_close"]) - center)
                / atr_value,
                "risk_bps": risk_bps,
                "relative_volume": float(row["relative_volume"]),
                "trade_count_z": float(row["trade_count_z"]),
                "aggressive_volume_z": float(row["aggressive_volume_z"]),
                "atr_percentile": float(row["atr_percentile"]),
                "funding_z": side * float(row["funding_z"]),
                "basis_bps": side * float(row["basis_bps"]),
                "oi_change_1h": side * float(row["oi_change_1h"]),
                "return_oi_interaction": float(row["return_oi_interaction"]),
            }
        )
        blocked_until = available + pd.Timedelta(minutes=30)
    fade_blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for current in range(50, len(data) - 1):
        row = data.iloc[current]
        previous = data.iloc[current - 1]
        available = pd.Timestamp(row["available_at"])
        required = row[
            [
                "atr",
                "relative_volume",
                "trade_count_z",
                "aggressive_volume_z",
                "atr_percentile",
                "funding_z",
                "basis_bps",
                "oi_change_1h",
            ]
        ].to_numpy(float)
        if available < fade_blocked_until or not np.isfinite(required).all():
            continue
        atr_value = float(row["atr"])
        centers = {
            "DAILY_FADE": float(row["daily_vwap"]),
            "ROLLING_4H_FADE": float(row["rolling_vwap_4h"]),
        }
        candidates: list[tuple[str, float, int, float]] = []
        for family, center in centers.items():
            if not np.isfinite(center):
                continue
            distance = (float(previous["perp_close"]) - center) / atr_value
            if not 0.5 <= abs(distance) <= 3.0:
                continue
            side = -1 if distance > 0 else 1
            confirms = (
                row["perp_close"] > previous["perp_high"]
                if side > 0
                else row["perp_close"] < previous["perp_low"]
            )
            room_bps = (
                side
                * (center - float(row["perp_close"]))
                / float(row["perp_close"])
                * 10_000
            )
            if (
                confirms
                and side * float(row["taker_imbalance_15m"]) > 0
                and room_bps >= 3 * COST_BPS
            ):
                candidates.append((family, center, side, room_bps))
        if not candidates:
            continue
        family, center, side, room_bps = max(candidates, key=lambda candidate: candidate[3])
        pullback = data.iloc[current - 2 : current + 1]
        extreme = (
            float(pullback["perp_low"].min())
            if side > 0
            else float(pullback["perp_high"].max())
        )
        stop = extreme - side * 0.1 * atr_value
        risk_bps = side * (float(row["perp_close"]) - stop) / float(row["perp_close"]) * 10_000
        if not 12 <= risk_bps <= 150:
            continue
        events.append(
            {
                "protocol_hash": PROTOCOL_HASH,
                "signal_timestamp": pd.Timestamp(row["timestamp"]),
                "available_at": available,
                "direction": side,
                "event_family": family,
                "event_family_code": 3 if family == "DAILY_FADE" else 4,
                "exit_style": "CENTER",
                "operating_vwap": center,
                "target_price": center,
                "stop_price": stop,
                "trend_vote": side * float(row["trend_vote"]),
                "return_1h": side * float(row["return_1h"]),
                "return_4h": side * float(row["return_4h"]),
                "ema_spread_atr": side * float(row["ema_spread_atr"]),
                "taker_imbalance_15m": side * float(row["taker_imbalance_15m"]),
                "spot_taker_imbalance_15m": side
                * float(row["spot_taker_imbalance_15m"]),
                "distance_vwap_atr": room_bps * float(row["perp_close"]) / 10_000 / atr_value,
                "risk_bps": risk_bps,
                "relative_volume": float(row["relative_volume"]),
                "trade_count_z": float(row["trade_count_z"]),
                "aggressive_volume_z": float(row["aggressive_volume_z"]),
                "atr_percentile": float(row["atr_percentile"]),
                "funding_z": side * float(row["funding_z"]),
                "basis_bps": side * float(row["basis_bps"]),
                "oi_change_1h": side * float(row["oi_change_1h"]),
                "return_oi_interaction": float(row["return_oi_interaction"]),
            }
        )
        fade_blocked_until = available + pd.Timedelta(minutes=30)
    result = pd.DataFrame(events)
    if not result.empty:
        result["exit_style"] = result.get("exit_style", pd.Series(index=result.index)).fillna(
            "TREND"
        )
    return result.sort_values(["signal_timestamp", "event_family_code"]).reset_index(drop=True)


def label_v5_events(events: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    trend = events.loc[events["exit_style"].eq("TREND")]
    validity = "data_valid" if "data_valid" in minutes else "is_available"
    data = minutes.loc[minutes[validity]].sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(data["timestamp"], utc=True)
    output: list[pd.DataFrame] = []
    follows: list[dict[Any, Any]] = []
    for event in trend.to_dict("records"):
        entry_index = int(times.searchsorted(pd.Timestamp(event["available_at"]), side="left"))
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(data.iloc[entry_index]["perp_open"])
        stop = float(event["stop_price"])
        risk = side * (entry - stop)
        if risk <= 0 or risk / entry * 10_000 < 12:
            continue
        gross, mfe, mae = 0.0, 0.0, 0.0
        exit_index = min(entry_index + 720, len(data) - 1)
        reason, armed = "TIMEOUT_12H", False
        for current in range(entry_index, exit_index + 1):
            row = data.iloc[current]
            high, low, open_price = float(row["perp_high"]), float(row["perp_low"]), float(
                row["perp_open"]
            )
            mfe = max(mfe, side * ((high if side > 0 else low) - entry))
            mae = min(mae, side * ((low if side > 0 else high) - entry))
            stopped = low <= stop if side > 0 else high >= stop
            if stopped:
                fill = min(open_price, stop) if side > 0 else max(open_price, stop)
                gross = side * (fill - entry) / entry * 10_000
                exit_index, reason = current, "STRUCTURAL_OR_TRAILING_STOP"
                break
            if not armed and mfe >= risk:
                armed = True
                stop = (
                    max(stop, entry + entry * COST_BPS / 10_000)
                    if side > 0
                    else min(stop, entry - entry * COST_BPS / 10_000)
                )
            if armed and current - entry_index >= 30:
                completed = data.iloc[current - 30 : current]
                proposal = (
                    float(completed["perp_low"].min())
                    if side > 0
                    else float(completed["perp_high"].max())
                )
                stop = max(stop, proposal) if side > 0 else min(stop, proposal)
        if reason == "TIMEOUT_12H":
            exit_price = float(data.iloc[exit_index]["perp_close"])
            gross = side * (exit_price - entry) / entry * 10_000
        risk_bps = risk / entry * 10_000
        follows.append(
            event
            | {
                "entry_timestamp": times.iat[entry_index],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps": gross - COST_BPS,
                "stress_return_bps": gross - 2 * COST_BPS,
                "net_return_r": (gross - COST_BPS) / risk_bps,
                "mfe_bps": mfe / entry * 10_000,
                "mae_bps": mae / entry * 10_000,
                "tp1": armed,
                "exit_reason": reason,
                "entry_order_type": "NEXT_1M_MARKET_CONSERVATIVE",
            }
        )
    if follows:
        output.append(pd.DataFrame(follows))
    fades: list[dict[Any, Any]] = []
    for event in events.loc[events["exit_style"].eq("CENTER")].to_dict("records"):
        entry_index = int(times.searchsorted(pd.Timestamp(event["available_at"]), side="left"))
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(data.iloc[entry_index]["perp_open"])
        stop, target = float(event["stop_price"]), float(event["target_price"])
        risk = side * (entry - stop)
        reward = side * (target - entry)
        if risk <= 0 or reward / entry * 10_000 < 3 * COST_BPS:
            continue
        exit_index = min(entry_index + 120, len(data) - 1)
        reason, exit_price = "TIMEOUT_2H", float(data.iloc[exit_index]["perp_close"])
        mfe = mae = 0.0
        for current in range(entry_index, exit_index + 1):
            row = data.iloc[current]
            high, low, open_price = float(row["perp_high"]), float(row["perp_low"]), float(
                row["perp_open"]
            )
            mfe = max(mfe, side * ((high if side > 0 else low) - entry))
            mae = min(mae, side * ((low if side > 0 else high) - entry))
            stopped = low <= stop if side > 0 else high >= stop
            target_hit = high >= target if side > 0 else low <= target
            if stopped:
                exit_index, reason = current, "STRUCTURAL_STOP"
                exit_price = min(open_price, stop) if side > 0 else max(open_price, stop)
                break
            if target_hit:
                exit_index, reason, exit_price = current, "VWAP_TARGET", target
                break
        gross = side * (exit_price - entry) / entry * 10_000
        risk_bps = risk / entry * 10_000
        fades.append(
            event
            | {
                "entry_timestamp": times.iat[entry_index],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps": gross - COST_BPS,
                "stress_return_bps": gross - 2 * COST_BPS,
                "net_return_r": (gross - COST_BPS) / risk_bps,
                "mfe_bps": mfe / entry * 10_000,
                "mae_bps": mae / entry * 10_000,
                "tp1": reason == "VWAP_TARGET",
                "exit_reason": reason,
            }
        )
    if fades:
        output.append(pd.DataFrame(fades))
    return pd.concat(output, ignore_index=True) if output else events.iloc[:0].copy()


def _model_rows(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    timestamp = pd.to_datetime(result["entry_timestamp"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    weekday = timestamp.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return result


def _models() -> dict[str, Any]:
    common = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": 4,
        "random_state": 20260805,
    }
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
        ),
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, random_state=20260805),
        ),
        "xgboost_gpu": XGBRegressor(objective="reg:squarederror", **common),
        "xgboost_classifier_gpu": XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", **common
        ),
    }


def _fit_scores(model: Any, fit: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
    classification = hasattr(model, "predict_proba")
    target = fit["stress_return_bps"].gt(0) if classification else fit["net_return_bps"]
    model.fit(fit[list(ML_FEATURES)], target)
    if classification:
        return np.asarray(model.predict_proba(predict[list(ML_FEATURES)])[:, 1], dtype=float)
    return np.asarray(model.predict(predict[list(ML_FEATURES)]), dtype=float)


def walk_forward_meta(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    data = _model_rows(rows).sort_values("entry_timestamp").reset_index(drop=True)
    timestamp = pd.to_datetime(data["entry_timestamp"], utc=True)
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    test_start = pd.Timestamp("2025-03-01T00:00:00Z")
    fold_number = 0
    while test_start + pd.Timedelta(weeks=8) <= HOLDOUT_START:
        fold_number += 1
        test_end = test_start + pd.Timedelta(weeks=8)
        calibration_start = test_start - pd.Timedelta(weeks=8)
        calibration = data.loc[
            timestamp.ge(calibration_start) & timestamp.lt(test_start)
        ]
        test = data.loc[timestamp.ge(test_start) & timestamp.lt(test_end)]
        if min(len(calibration), len(test)) < 100:
            test_start += pd.Timedelta(weeks=8)
            continue
        choices: list[tuple[float, str, int, float, Any, dict[str, float]]] = []
        for training_weeks in (12, 26, 52):
            fit_start = calibration_start - pd.Timedelta(weeks=training_weeks)
            fit = data.loc[timestamp.ge(fit_start) & timestamp.lt(calibration_start)]
            if len(fit) < 100:
                continue
            for name, model in _models().items():
                scores = _fit_scores(model, fit, calibration)
                for coverage in (0.05, 0.10, 0.20, 0.30):
                    threshold = float(np.quantile(scores, 1 - coverage))
                    selected = _non_overlapping(calibration.loc[scores >= threshold])
                    metrics = _metrics(selected, "stress_return_bps")
                    if (
                        len(selected) / 56 >= 0.5
                        and metrics["expectancy_bps"] > 0
                        and metrics["profit_factor"] >= 1.05
                    ):
                        choices.append(
                            (
                                metrics["expectancy_bps"],
                                name,
                                training_weeks,
                                threshold,
                                model,
                                metrics,
                            )
                        )
        if choices:
            _, name, training_weeks, threshold, model, calibration_metrics = max(
                choices, key=lambda choice: choice[0]
            )
            test_scores = (
                np.asarray(model.predict_proba(test[list(ML_FEATURES)])[:, 1], dtype=float)
                if hasattr(model, "predict_proba")
                else np.asarray(model.predict(test[list(ML_FEATURES)]), dtype=float)
            )
            selected_test = _non_overlapping(test.loc[test_scores >= threshold].copy())
            selected_test["model_score"] = (
                model.predict_proba(selected_test[list(ML_FEATURES)])[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(selected_test[list(ML_FEATURES)])
            )
            predictions.append(selected_test)
            folds.append(
                {
                    "fold": fold_number,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "champion": name,
                    "training_weeks": training_weeks,
                    "threshold": threshold,
                    "calibration_metrics": calibration_metrics,
                    "test_metrics": _metrics(selected_test),
                }
            )
        else:
            folds.append(
                {
                    "fold": fold_number,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "champion": "FLAT",
                }
            )
        test_start = test_end
        _status("meta_gpu", 55 + min(35, fold_number * 7), f"fold {fold_number}")
    return (
        pd.concat(predictions, ignore_index=True) if predictions else data.iloc[:0].copy(),
        folds,
    )


def train() -> dict[str, Any]:
    bars = pd.read_parquet(BARS)
    minutes = pd.read_parquet(MINUTES)
    cached = pd.read_parquet(LABELED) if LABELED.exists() else pd.DataFrame()
    if not cached.empty and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
        events = pd.read_parquet(EVENTS)
        labeled = cached
        _status("resume", 45, f"{len(labeled):,} cached causal outcomes")
    else:
        if EVENTS.exists():
            # ponytail: reuse only while candidate generation is unchanged; delete EVENTS otherwise.
            events = pd.read_parquet(EVENTS)
            events["protocol_hash"] = PROTOCOL_HASH
            _status("events", 20, f"{len(events):,} unchanged causal candidates")
        else:
            _status("events", 5, "BTC-only multi-VWAP event scan")
            events = build_events(bars)
        ROOT.mkdir(parents=True, exist_ok=True)
        events.to_parquet(EVENTS, index=False)
        _status("labels", 35, f"{len(events):,} causal candidates")
        labeled = label_v5_events(events, minutes)
        labeled.to_parquet(LABELED, index=False)
    pre_holdout = labeled.loc[
        pd.to_datetime(labeled["entry_timestamp"], utc=True).lt(HOLDOUT_START)
    ]
    family_metrics = {
        family: _metrics(_non_overlapping(rows))
        for family, rows in pre_holdout.groupby("event_family")
    }
    train_rows = pre_holdout.loc[
        pd.to_datetime(pre_holdout["entry_timestamp"], utc=True).lt("2025-01-01")
    ]
    eligible = [
        family
        for family, rows in train_rows.groupby("event_family")
        if (metric := _metrics(_non_overlapping(rows)))["expectancy_bps"] > 0
        and metric["profit_factor"] >= 1.10
    ]
    oos = _non_overlapping(
        pre_holdout.loc[
            pd.to_datetime(pre_holdout["entry_timestamp"], utc=True).ge("2025-01-01")
            & pre_holdout["event_family"].isin(eligible)
        ]
    )
    metrics = _metrics(oos)
    stress = _metrics(oos, "stress_return_bps")
    days = max(
        1.0,
        (
            min(HOLDOUT_START, pd.to_datetime(pre_holdout["entry_timestamp"], utc=True).max())
            - pd.Timestamp("2025-01-01T00:00:00Z")
        ).total_seconds()
        / 86400,
    )
    frequency = len(oos) / days
    meta_oos, meta_folds = walk_forward_meta(pre_holdout)
    meta_metrics = _metrics(meta_oos)
    meta_stress = _metrics(meta_oos, "stress_return_bps")
    meta_days = max(
        1.0,
        (
            min(HOLDOUT_START, pd.to_datetime(pre_holdout["entry_timestamp"], utc=True).max())
            - pd.Timestamp("2025-03-01T00:00:00Z")
        ).total_seconds()
        / 86400,
    )
    meta_frequency = len(meta_oos) / meta_days
    gates = {
        "train_positive_family": bool(eligible),
        "oos_trades_100": len(oos) >= 100,
        "oos_frequency_0_5_day": frequency >= 0.5,
        "oos_expectancy_positive": metrics["expectancy_bps"] > 0,
        "oos_profit_factor_1_10": metrics["profit_factor"] >= 1.10,
        "oos_drawdown_8pct": metrics["max_drawdown"] <= 0.08,
        "stress_nonnegative": stress["expectancy_bps"] >= 0,
    }
    meta_gates = {
        "oos_trades_100": len(meta_oos) >= 100,
        "oos_frequency_0_5_day": meta_frequency >= 0.5,
        "oos_expectancy_positive": meta_metrics["expectancy_bps"] > 0,
        "oos_profit_factor_1_10": meta_metrics["profit_factor"] >= 1.10,
        "oos_drawdown_8pct": meta_metrics["max_drawdown"] <= 0.08,
        "stress_nonnegative": meta_stress["expectancy_bps"] >= 0,
        "majority_folds_positive": sum(
            fold.get("test_metrics", {}).get("expectancy_bps", 0) > 0 for fold in meta_folds
        )
        > len(meta_folds) / 2,
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {"bars_5m": len(bars), "minutes_1m": len(minutes), "assets": ["BTCUSDT"]},
        "candidates": len(events),
        "labeled": len(labeled),
        "eligible_families_train_only": eligible,
        "family_metrics": family_metrics,
        "oos_metrics": metrics,
        "oos_stress_metrics": stress,
        "oos_trades_per_day": frequency,
        "gates": gates,
        "meta_oos_metrics": meta_metrics,
        "meta_oos_stress_metrics": meta_stress,
        "meta_oos_trades_per_day": meta_frequency,
        "meta_folds": meta_folds,
        "meta_gates": meta_gates,
        "verdict": "FREQUENT_META_EDGE_FOUND" if all(meta_gates.values()) else "NO_FREQUENT_EDGE",
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    _status("complete", 100, str(report["verdict"]))
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
