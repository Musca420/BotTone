from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.indicators.atr import atr

BARS = Path("data/ml/hybrid_v24/bars_5m.parquet")
MINUTES = Path("data/ml/hybrid_v24/joined_minutes.parquet")
ROOT = Path("data/ml/musca_v4")
EVENTS = ROOT / "events.parquet"
OOS = ROOT / "oos.parquet"
REPORT = Path("data/reports/musca_v4_research.json")
STATUS = Path("data/reports/musca_v4_research.status.json")
BUNDLE = Path("data/models/musca_v4/bundle.joblib")
COST_BPS = 8.0
PROTOCOL = {
    "name": "musca_v4_multi_anchor_v1",
    "market": "BINANCE_BTCUSDT_SPOT_PERP",
    "timeframe_minutes": 5,
    "anchors": ["SESSION_UTC", "IMPULSE_15M", "STRUCTURAL_SWING"],
    "families": {
        "primary": "IMPULSE_PULLBACK",
        "challengers": ["IMPULSE_REENTRY", "SESSION_SWING_PULLBACK"],
        "gate": "enabled train-only if EV>0 and PF>=1.10",
    },
    "anchor_band": "1_volume_weighted_sigma",
    "impulse": "15m_breakout; relative_volume>=1.0; taker_flow_and_spot_confirm",
    "anchor_lifetime_bars": 288,
    "restart_expiry_bars": 6,
    "entry": "next_1m_open_after_closed_5m_restart",
    "restart": "close leaves the pullback zone; at most one active zone remains",
    "stop": "beyond_pullback_and_operating_avwap_band; never_widens",
    "management": "half_at_1.5R; cost_protected; trailing_15m_swing; timeout_6h",
    "cost_bps": COST_BPS,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
FEATURES = (
    "event_family_code",
    "direction",
    "return_1h",
    "return_4h",
    "ema_spread_atr",
    "relative_volume",
    "taker_imbalance",
    "spot_return_1h",
    "spot_perp_divergence_bps",
    "daily_distance_sigma",
    "impulse_distance_sigma",
    "swing_distance_sigma",
    "confluence_score",
    "pullback_depth_atr",
    "room_bps",
    "atr_percentile",
    "hour_sin",
    "hour_cos",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    time = pd.to_datetime(data["timestamp"], utc=True)
    bar_vwap = data["perp_quote_volume"] / data["perp_volume"].replace(0, np.nan)
    day = time.dt.floor("D")
    cumulative_volume = data["perp_volume"].groupby(day).cumsum()
    cumulative_quote = data["perp_quote_volume"].groupby(day).cumsum()
    cumulative_second = (data["perp_volume"] * bar_vwap.pow(2)).groupby(day).cumsum()
    data["daily_vwap"] = cumulative_quote / cumulative_volume
    data["daily_sigma"] = np.sqrt(
        (cumulative_second / cumulative_volume - data["daily_vwap"].pow(2)).clip(lower=0)
    )
    data["atr"] = atr(data["perp_high"], data["perp_low"], data["perp_close"], 14)
    data["return_15m"] = data["perp_close"].pct_change(3)
    data["return_1h"] = data["perp_close"].pct_change(12)
    data["return_4h"] = data["perp_close"].pct_change(48)
    data["spot_return_15m"] = data["spot_close"].pct_change(3)
    data["spot_return_1h"] = data["spot_close"].pct_change(12)
    ema_fast = data["perp_close"].ewm(span=12, adjust=False).mean()
    ema_slow = data["perp_close"].ewm(span=48, adjust=False).mean()
    data["ema_spread_atr"] = (ema_fast - ema_slow) / data["atr"]
    data["direction"] = np.where(
        data["return_1h"].gt(0)
        & data["return_4h"].gt(0)
        & data["spot_return_1h"].gt(0)
        & data["ema_spread_atr"].gt(0),
        1,
        np.where(
            data["return_1h"].lt(0)
            & data["return_4h"].lt(0)
            & data["spot_return_1h"].lt(0)
            & data["ema_spread_atr"].lt(0),
            -1,
            0,
        ),
    )
    data["relative_volume"] = (
        data["perp_quote_volume"]
        / data["perp_quote_volume"].shift(1).rolling(288, min_periods=96).median()
    )
    data["taker_imbalance"] = (
        2 * data["perp_taker_buy_quote"] / data["perp_quote_volume"].replace(0, np.nan) - 1
    )
    data["spot_taker_imbalance"] = (
        2 * data["spot_taker_buy_quote"] / data["spot_quote_volume"].replace(0, np.nan) - 1
    )
    data["spot_perp_divergence_bps"] = (data["spot_return_15m"] - data["return_15m"]) * 10_000
    data["atr_percentile"] = (
        data["atr"].shift(1).rolling(2016, min_periods=288).rank(pct=True) * 100
    )
    hour = time.dt.hour + time.dt.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["available_at"] = time + pd.Timedelta(minutes=5)
    return data


def anchored_vwap_band(
    volume: np.ndarray, quote: np.ndarray, start: int, end: int
) -> tuple[float, float]:
    weights = volume[start : end + 1]
    total = float(np.nansum(weights))
    if total <= 0:
        return np.nan, np.nan
    prices = quote[start : end + 1] / np.where(weights > 0, weights, np.nan)
    center = float(np.nansum(quote[start : end + 1]) / total)
    sigma = float(np.sqrt(max(0.0, np.nansum(weights * (prices - center) ** 2) / total)))
    return center, sigma


def build_events(
    frame: pd.DataFrame,
    audit: dict[str, int] | None = None,
    *,
    breakout_bars: int = 3,
    relative_volume_min: float = 1.0,
    pullback_depth_atr: float = 0.25,
    restart_expiry_bars: int = 6,
    require_quieter_pullback: bool = True,
    require_restart_spot: bool = True,
    max_restart_confluence: int = 1,
    minimum_room_bps: float | None = 3 * COST_BPS,
    risk_bps_range: tuple[float, float] | None = (12.0, 200.0),
    _prepared_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = build_features(frame) if _prepared_features is None else _prepared_features
    arrays = {
        column: data[column].to_numpy(float)
        for column in data.columns
        if column != "timestamp" and pd.api.types.is_numeric_dtype(data[column])
    }
    high, low, close = arrays["perp_high"], arrays["perp_low"], arrays["perp_close"]
    volume, quote = arrays["perp_volume"], arrays["perp_quote_volume"]
    direction = arrays["direction"].astype(int)
    prior_high = (
        data["perp_high"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .max()
        .to_numpy(float)
    )
    prior_low = (
        data["perp_low"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .min()
        .to_numpy(float)
    )
    spot_high = (
        data["spot_high"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .max()
        .to_numpy(float)
    )
    spot_low = (
        data["spot_low"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .min()
        .to_numpy(float)
    )
    impulse = (
        (direction > 0)
        & (close > prior_high)
        & (arrays["spot_close"] > spot_high)
        & (arrays["relative_volume"] >= relative_volume_min)
        & (arrays["taker_imbalance"] > 0.05)
        & (arrays["spot_taker_imbalance"] > 0)
    ) | (
        (direction < 0)
        & (close < prior_low)
        & (arrays["spot_close"] < spot_low)
        & (arrays["relative_volume"] >= relative_volume_min)
        & (arrays["taker_imbalance"] < -0.05)
        & (arrays["spot_taker_imbalance"] < 0)
    )
    stages: dict[str, set[int]] | None = (
        {
            name: set()
            for name in (
                "scanned",
                "armed",
                "restart",
                "confirmed",
                "room",
                "risk",
                "event",
            )
        }
        if audit is not None
        else None
    )
    events: list[dict[str, Any]] = []
    busy_until = -1
    for raw_index in np.flatnonzero(impulse):
        index = int(raw_index)
        if index <= busy_until:
            continue
        if stages is not None:
            stages["scanned"].add(index)
        side = int(direction[index])
        swing_slice = slice(max(0, index - 24), index + 1)
        swing_index = max(0, index - 24) + int(
            np.nanargmin(low[swing_slice]) if side > 0 else np.nanargmax(high[swing_slice])
        )
        impulse_extreme = high[index] if side > 0 else low[index]
        pullback_extreme = impulse_extreme
        armed_at: int | None = None
        operating_center = operating_sigma = np.nan
        confluence = 0
        touch_confluence = 0
        anchor_cycle = 0
        for current in range(index + 1, min(index + 289, len(data) - 1)):
            pullback_extreme = (
                min(pullback_extreme, low[current])
                if side > 0
                else max(pullback_extreme, high[current])
            )
            impulse_center, impulse_sigma = anchored_vwap_band(volume, quote, index, current)
            swing_center, swing_sigma = anchored_vwap_band(volume, quote, swing_index, current)
            centers = (arrays["daily_vwap"][current], impulse_center, swing_center)
            sigmas = (
                arrays["daily_sigma"][current],
                impulse_sigma,
                swing_sigma,
            )
            zones = [
                abs(close[current] - center) <= max(sigma, 0.25 * arrays["atr"][current])
                for center, sigma in zip(centers, sigmas, strict=True)
                if np.isfinite(center) and np.isfinite(sigma)
            ]
            confluence = sum(zones)
            depth = side * (impulse_extreme - pullback_extreme) / arrays["atr"][current]
            quieter = arrays["perp_quote_volume"][current] < arrays["perp_quote_volume"][index]
            if (
                armed_at is None
                and confluence >= 1
                and depth >= pullback_depth_atr
                and (quieter or not require_quieter_pullback)
            ):
                armed_at = current
                if stages is not None:
                    stages["armed"].add(index)
                touch_confluence = confluence
                distances = [abs(close[current] - center) for center in centers]
                selected = int(np.nanargmin(distances))
                operating_center, operating_sigma = centers[selected], sigmas[selected]
            if armed_at is None:
                continue
            if current - armed_at > restart_expiry_bars:
                break
            restarted = (
                close[current] > high[current - 1]
                if side > 0
                else close[current] < low[current - 1]
            )
            if restarted and stages is not None:
                stages["restart"].add(index)
            flow = arrays["taker_imbalance"][current] * side > 0
            spot = arrays["spot_return_15m"][current] * side > 0
            accepted = (close[current] - operating_center) * side > 0
            if not (
                restarted
                and flow
                and (spot or not require_restart_spot)
                and accepted
                and confluence <= max_restart_confluence
            ):
                continue
            if stages is not None:
                stages["confirmed"].add(index)
            entry_proxy = close[current]
            room = side * (impulse_extreme - entry_proxy) / entry_proxy * 10_000
            if minimum_room_bps is not None and room < minimum_room_bps:
                continue
            if stages is not None:
                stages["room"].add(index)
            zone_edge = operating_center - side * max(
                operating_sigma, 0.25 * arrays["atr"][current]
            )
            stop = (
                min(pullback_extreme, zone_edge) - 0.1 * arrays["atr"][current]
                if side > 0
                else max(pullback_extreme, zone_edge) + 0.1 * arrays["atr"][current]
            )
            risk_bps = side * (entry_proxy - stop) / entry_proxy * 10_000
            if risk_bps_range is not None and not (
                risk_bps_range[0] <= risk_bps <= risk_bps_range[1]
            ):
                continue
            if stages is not None:
                stages["risk"].add(index)
            row = data.iloc[current]
            events.append(
                {
                    "protocol_hash": PROTOCOL_HASH,
                    "signal_index": current,
                    "signal_timestamp": pd.Timestamp(row["timestamp"]),
                    "available_at": pd.Timestamp(row["available_at"]),
                    "direction": side,
                    "event_family": (
                        "IMPULSE_PULLBACK" if anchor_cycle == 0 else "IMPULSE_REENTRY"
                    ),
                    "event_family_code": 0 if anchor_cycle == 0 else 2,
                    "anchor_cycle": anchor_cycle,
                    "impulse_anchor_at": pd.Timestamp(data.iloc[index]["timestamp"]),
                    "swing_anchor_at": pd.Timestamp(data.iloc[swing_index]["timestamp"]),
                    "anchor_available_at": pd.Timestamp(data.iloc[index]["available_at"]),
                    "operating_vwap": operating_center,
                    "operating_sigma": operating_sigma,
                    "stop_price": stop,
                    "return_1h": float(row["return_1h"]),
                    "return_4h": float(row["return_4h"]),
                    "ema_spread_atr": float(row["ema_spread_atr"]),
                    "relative_volume": float(row["relative_volume"]),
                    "taker_imbalance": float(row["taker_imbalance"]),
                    "spot_return_1h": float(row["spot_return_1h"]),
                    "spot_perp_divergence_bps": float(row["spot_perp_divergence_bps"]),
                    "daily_distance_sigma": float(
                        (close[current] - arrays["daily_vwap"][current])
                        / max(arrays["daily_sigma"][current], 1e-9)
                    ),
                    "impulse_distance_sigma": float(
                        (close[current] - impulse_center) / max(impulse_sigma, 1e-9)
                    ),
                    "swing_distance_sigma": float(
                        (close[current] - swing_center) / max(swing_sigma, 1e-9)
                    ),
                    "confluence_score": touch_confluence,
                    "remaining_zone_count": confluence,
                    "pullback_depth_atr": depth,
                    "room_bps": room,
                    "atr_percentile": float(row["atr_percentile"]),
                    "hour_sin": float(row["hour_sin"]),
                    "hour_cos": float(row["hour_cos"]),
                }
            )
            if stages is not None:
                stages["event"].add(index)
            busy_until = current
            anchor_cycle += 1
            armed_at = None
            pullback_extreme = high[current] if side > 0 else low[current]
    rolling_high = data["perp_high"].shift(1).rolling(12, min_periods=12).max().to_numpy(float)
    rolling_low = data["perp_low"].shift(1).rolling(12, min_periods=12).min().to_numpy(float)
    active: dict[str, Any] | None = None
    cooldown_until = -1
    latest_impulse = {1: -1, -1: -1}
    for current in range(288, len(data) - 1):
        side = int(direction[current])
        if impulse[current] and side:
            latest_impulse[side] = current
        if side == 0 or current <= cooldown_until:
            continue
        if active is not None and side != active["direction"]:
            active = None
        if active is None:
            start = current - 24
            swing_slice = slice(start, current + 1)
            swing_index = start + int(
                np.nanargmin(low[swing_slice]) if side > 0 else np.nanargmax(high[swing_slice])
            )
            impulse_index = latest_impulse[side]
            if impulse_index < current - 288:
                impulse_index = -1
            impulse_center, impulse_sigma = (
                anchored_vwap_band(volume, quote, impulse_index, current)
                if impulse_index >= 0
                else (np.nan, np.nan)
            )
            swing_center, swing_sigma = anchored_vwap_band(volume, quote, swing_index, current)
            centers = (arrays["daily_vwap"][current], impulse_center, swing_center)
            sigmas = (arrays["daily_sigma"][current], impulse_sigma, swing_sigma)
            zone_count = sum(
                abs(close[current] - center) <= max(sigma, 0.25 * arrays["atr"][current])
                for center, sigma in zip(centers, sigmas, strict=True)
                if np.isfinite(center) and np.isfinite(sigma)
            )
            counter_move = arrays["return_15m"][current] * side <= 0
            quieter = arrays["relative_volume"][current] <= 1.5
            if zone_count and counter_move and quieter:
                active = {
                    "direction": side,
                    "armed_at": current,
                    "swing_index": swing_index,
                    "impulse_index": impulse_index,
                    "pullback_extreme": low[current] if side > 0 else high[current],
                    "touch_confluence": zone_count,
                }
            continue
        armed_at = int(active["armed_at"])
        if current - armed_at > 6:
            active = None
            continue
        pullback_extreme = float(active["pullback_extreme"])
        pullback_extreme = (
            min(pullback_extreme, low[current])
            if side > 0
            else max(pullback_extreme, high[current])
        )
        active["pullback_extreme"] = pullback_extreme
        swing_index = int(active["swing_index"])
        impulse_index = int(active["impulse_index"])
        impulse_center, impulse_sigma = (
            anchored_vwap_band(volume, quote, impulse_index, current)
            if impulse_index >= 0
            else (np.nan, np.nan)
        )
        swing_center, swing_sigma = anchored_vwap_band(volume, quote, swing_index, current)
        centers = (arrays["daily_vwap"][current], impulse_center, swing_center)
        sigmas = (arrays["daily_sigma"][current], impulse_sigma, swing_sigma)
        remaining = sum(
            abs(close[current] - center) <= max(sigma, 0.25 * arrays["atr"][current])
            for center, sigma in zip(centers, sigmas, strict=True)
            if np.isfinite(center) and np.isfinite(sigma)
        )
        restarted = (
            close[current] > high[current - 1] if side > 0 else close[current] < low[current - 1]
        )
        if not (
            restarted
            and arrays["taker_imbalance"][current] * side > 0
            and arrays["spot_return_15m"][current] * side > 0
            and remaining <= 1
        ):
            continue
        local_extreme = rolling_high[current] if side > 0 else rolling_low[current]
        room = side * (local_extreme - close[current]) / close[current] * 10_000
        if minimum_room_bps is not None and room < minimum_room_bps:
            continue
        distances = [abs(close[current] - center) for center in centers]
        selected = int(np.nanargmin(distances))
        operating_center, operating_sigma = centers[selected], sigmas[selected]
        zone_edge = operating_center - side * max(operating_sigma, 0.25 * arrays["atr"][current])
        stop = (
            min(pullback_extreme, zone_edge) - 0.1 * arrays["atr"][current]
            if side > 0
            else max(pullback_extreme, zone_edge) + 0.1 * arrays["atr"][current]
        )
        risk_bps = side * (close[current] - stop) / close[current] * 10_000
        if risk_bps_range is not None and not (
            risk_bps_range[0] <= risk_bps <= risk_bps_range[1]
        ):
            active = None
            continue
        row = data.iloc[current]
        events.append(
            {
                "protocol_hash": PROTOCOL_HASH,
                "signal_index": current,
                "signal_timestamp": pd.Timestamp(row["timestamp"]),
                "available_at": pd.Timestamp(row["available_at"]),
                "direction": side,
                "event_family": "SESSION_SWING_PULLBACK",
                "event_family_code": 1,
                "impulse_anchor_at": (
                    pd.Timestamp(data.iloc[impulse_index]["timestamp"])
                    if impulse_index >= 0
                    else pd.NaT
                ),
                "swing_anchor_at": pd.Timestamp(data.iloc[swing_index]["timestamp"]),
                "anchor_available_at": pd.Timestamp(data.iloc[armed_at]["available_at"]),
                "operating_vwap": operating_center,
                "operating_sigma": operating_sigma,
                "stop_price": stop,
                "return_1h": float(row["return_1h"]),
                "return_4h": float(row["return_4h"]),
                "ema_spread_atr": float(row["ema_spread_atr"]),
                "relative_volume": float(row["relative_volume"]),
                "taker_imbalance": float(row["taker_imbalance"]),
                "spot_return_1h": float(row["spot_return_1h"]),
                "spot_perp_divergence_bps": float(row["spot_perp_divergence_bps"]),
                "daily_distance_sigma": float(
                    (close[current] - arrays["daily_vwap"][current])
                    / max(arrays["daily_sigma"][current], 1e-9)
                ),
                "impulse_distance_sigma": (
                    float((close[current] - impulse_center) / max(impulse_sigma, 1e-9))
                    if np.isfinite(impulse_center)
                    else np.nan
                ),
                "swing_distance_sigma": float(
                    (close[current] - swing_center) / max(swing_sigma, 1e-9)
                ),
                "confluence_score": int(active["touch_confluence"]),
                "remaining_zone_count": remaining,
                "pullback_depth_atr": float(
                    side * (local_extreme - pullback_extreme) / arrays["atr"][current]
                ),
                "room_bps": room,
                "atr_percentile": float(row["atr_percentile"]),
                "hour_sin": float(row["hour_sin"]),
                "hour_cos": float(row["hour_cos"]),
            }
        )
        active = None
        cooldown_until = current + 2
    result = pd.DataFrame(events)
    if audit is not None and stages is not None:
        audit.update({name: len(indices) for name, indices in stages.items()})
    if result.empty:
        return pd.DataFrame(
            columns=[
                "signal_timestamp",
                "available_at",
                "event_family",
                "event_family_code",
                "direction",
            ]
        )
    return (
        result
        .sort_values(["signal_timestamp", "event_family_code"])
        .drop_duplicates(["signal_timestamp", "direction"], keep="first")
        .reset_index(drop=True)
    )


def label_events(events: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    data = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(data["timestamp"], utc=True)
    output: list[dict[str, Any]] = []
    for raw_event in events.to_dict("records"):
        event = cast(dict[str, Any], raw_event)
        entry_index = int(times.searchsorted(pd.Timestamp(event["available_at"]), side="left"))
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(data.iloc[entry_index]["perp_open"])
        stop = float(event["stop_price"])
        risk = side * (entry - stop)
        if risk <= 0 or risk / entry * 10_000 < 12:
            continue
        target = entry + side * 1.5 * risk
        remaining = 1.0
        gross = 0.0
        mfe = mae = 0.0
        exit_index = min(entry_index + 360, len(data) - 1)
        reason = "TIMEOUT_6H"
        tp1 = False
        for current in range(entry_index, exit_index + 1):
            row = data.iloc[current]
            high, low, open_price = (
                float(row["perp_high"]),
                float(row["perp_low"]),
                float(row["perp_open"]),
            )
            favorable = side * ((high if side > 0 else low) - entry)
            adverse = side * ((low if side > 0 else high) - entry)
            mfe, mae = max(mfe, favorable), min(mae, adverse)
            stopped = low <= stop if side > 0 else high >= stop
            target_hit = high >= target if side > 0 else low <= target
            if stopped:
                fill = min(open_price, stop) if side > 0 else max(open_price, stop)
                gross += remaining * side * (fill - entry) / entry * 10_000
                exit_index, reason, remaining = current, "STRUCTURAL_STOP", 0
                break
            if not tp1 and target_hit:
                gross += 0.5 * side * (target - entry) / entry * 10_000
                remaining, tp1 = 0.5, True
                stop = (
                    max(stop, entry + entry * COST_BPS / 10_000)
                    if side > 0
                    else min(stop, entry - entry * COST_BPS / 10_000)
                )
            if tp1 and current - entry_index >= 15:
                completed = data.iloc[current - 15 : current]
                proposal = (
                    float(completed["perp_low"].min())
                    if side > 0
                    else float(completed["perp_high"].max())
                )
                stop = max(stop, proposal) if side > 0 else min(stop, proposal)
        if remaining:
            exit_price = float(data.iloc[exit_index]["perp_close"])
            gross += remaining * side * (exit_price - entry) / entry * 10_000
        output.append(
            event
            | {
                "entry_timestamp": times.iat[entry_index],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps": gross - COST_BPS,
                "stress_return_bps": gross - 2 * COST_BPS,
                "net_return_r": (gross - COST_BPS) / (risk / entry * 10_000),
                "mfe_bps": mfe / entry * 10_000,
                "mae_bps": mae / entry * 10_000,
                "tp1": tp1,
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(output)


def _non_overlapping(rows: pd.DataFrame) -> pd.DataFrame:
    accepted: list[Any] = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in rows.sort_values(["entry_timestamp", "signal_timestamp"]).iterrows():
        if pd.Timestamp(row["entry_timestamp"]) >= busy_until:
            accepted.append(index)
            busy_until = pd.Timestamp(row["exit_timestamp"])
    return rows.loc[accepted].sort_values("entry_timestamp").reset_index(drop=True)


def _metrics(rows: pd.DataFrame, target: str = "net_return_bps") -> dict[str, float]:
    values = rows[target].to_numpy(float)
    if not len(values):
        return {
            "trades": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
        }
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    curve = np.cumsum(rows["net_return_r"].to_numpy(float) * 0.01)
    drawdown = np.max(np.maximum.accumulate(np.r_[0.0, curve]) - np.r_[0.0, curve])
    return {
        "trades": float(len(rows)),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else float("inf"),
        "win_rate": float((values > 0).mean()),
        "max_drawdown": float(drawdown),
    }


def _models() -> dict[str, tuple[Any, Any]]:
    ridge = (
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, random_state=20260805),
        ),
    )
    common = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=20260805,
    )
    return {
        "ridge": ridge,
        "xgboost_gpu": (
            XGBRegressor(objective="reg:squarederror", **common),
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common),
        ),
    }


def train() -> dict[str, Any]:
    _status("deterministic_events", 5, "BTC multi-anchor causal scan")
    events = build_events(pd.read_parquet(BARS))
    labeled = label_events(events, pd.read_parquet(MINUTES))
    ROOT.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(EVENTS, index=False)
    family_metrics = {
        family: _metrics(_non_overlapping(rows)) for family, rows in labeled.groupby("event_family")
    }
    base = _non_overlapping(labeled.loc[labeled["event_family"].eq("IMPULSE_PULLBACK")])
    base_metrics = _metrics(base)
    _status("deterministic_audit", 40, f"{len(base):,} non-overlapping trades")
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    times = pd.to_datetime(labeled["entry_timestamp"], utc=True)
    for fold, year in enumerate(range(2025, 2027), 1):
        prior = labeled.loc[times.lt(pd.Timestamp(f"{year}-01-01", tz="UTC"))]
        test = labeled.loc[times.dt.year.eq(year)]
        eligible_families = [
            family
            for family, rows in prior.groupby("event_family")
            if (
                (metrics := _metrics(_non_overlapping(rows)))["expectancy_bps"] > 0
                and metrics["profit_factor"] >= 1.10
            )
        ]
        prior = prior.loc[prior["event_family"].isin(eligible_families)]
        test = test.loc[test["event_family"].isin(eligible_families)]
        if len(prior) < 40 or len(test) < 10:
            continue
        split = int(len(prior) * 0.8)
        fit, calibration = prior.iloc[:split], prior.iloc[split:]
        choices: list[tuple[float, str, Any, Any]] = []
        for name, (ev_model, win_model) in _models().items():
            ev_model.fit(fit[list(FEATURES)], fit["stress_return_bps"])
            win_model.fit(fit[list(FEATURES)], fit["stress_return_bps"].gt(0))
            ev = ev_model.predict(calibration[list(FEATURES)])
            probability = win_model.predict_proba(calibration[list(FEATURES)])[:, 1]
            selected = _non_overlapping(calibration.loc[(ev > 0) & (probability > 0.5)])
            score = (
                _metrics(selected, "stress_return_bps")["expectancy_bps"]
                if len(selected) >= 10
                else -np.inf
            )
            choices.append((score, name, ev_model, win_model))
        ridge = choices[0]
        challenger = max(choices, key=lambda choice: choice[0])
        champion = (
            challenger if challenger[1] != "xgboost_gpu" or challenger[0] > ridge[0] else ridge
        )
        deterministic_calibration = _metrics(_non_overlapping(calibration), "stress_return_bps")[
            "expectancy_bps"
        ]
        if champion[0] <= deterministic_calibration:
            name = "deterministic"
            selected = _non_overlapping(test.copy())
        else:
            _, name, ev_model, win_model = champion
            ev_model.fit(prior[list(FEATURES)], prior["stress_return_bps"])
            win_model.fit(prior[list(FEATURES)], prior["stress_return_bps"].gt(0))
            ev = ev_model.predict(test[list(FEATURES)])
            probability = win_model.predict_proba(test[list(FEATURES)])[:, 1]
            selected = _non_overlapping(test.loc[(ev > 0) & (probability > 0.5)].copy())
        selected["outer_year"], selected["champion"] = year, name
        predictions.append(selected)
        folds.append(
            {
                "year": year,
                "eligible_families": eligible_families,
                "champion": name,
                "metrics": _metrics(selected),
            }
        )
        _status("walk_forward_gpu", 40 + fold * 25, f"OOS {year}: {name}")
    oos = pd.concat(predictions, ignore_index=True) if predictions else labeled.iloc[:0].copy()
    oos.to_parquet(OOS, index=False)
    oos_metrics = _metrics(oos)
    annual = base.groupby(pd.to_datetime(base["entry_timestamp"], utc=True).dt.year)[
        "net_return_bps"
    ].mean()
    gates = {
        "base_expectancy_positive": base_metrics["expectancy_bps"] > 0,
        "base_profit_factor_1_10": base_metrics["profit_factor"] >= 1.10,
        "oos_trades_100": len(oos) >= 100,
        "oos_expectancy_positive": oos_metrics["expectancy_bps"] > 0,
        "oos_profit_factor_1_15": oos_metrics["profit_factor"] >= 1.15,
        "oos_drawdown_8pct": oos_metrics["max_drawdown"] <= 0.08,
        "stress_nonnegative": _metrics(oos, "stress_return_bps")["expectancy_bps"] >= 0,
        "majority_years_positive": float(annual.gt(0).mean()) > 0.5,
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "bars_5m": len(pd.read_parquet(BARS, columns=["timestamp"])),
            "minutes_1m": len(pd.read_parquet(MINUTES, columns=["timestamp"])),
        },
        "candidates": len(events),
        "labeled": len(labeled),
        "family_metrics": family_metrics,
        "base_metrics": base_metrics,
        "oos_metrics": oos_metrics,
        "stress_metrics": _metrics(oos, "stress_return_bps"),
        "annual_expectancy_bps": {str(year): float(value) for year, value in annual.items()},
        "folds": folds,
        "gates": gates,
        "verdict": "RESEARCH_ONLY_SHADOW" if all(gates.values()) else "NO_VALIDATED_INTRADAY_EDGE",
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    if len(labeled) >= 100:
        model_rows = labeled.loc[labeled["event_family"].eq("IMPULSE_PULLBACK")]
        ev_model, win_model = _models()["ridge"]
        ev_model.fit(model_rows[list(FEATURES)], model_rows["stress_return_bps"])
        win_model.fit(model_rows[list(FEATURES)], model_rows["stress_return_bps"].gt(0))
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "bundle_type": "RESEARCH_ONLY",
                "strategy_profile": "musca_v4_multi_anchor_vwap",
                "protocol": PROTOCOL,
                "protocol_hash": PROTOCOL_HASH,
                "features": FEATURES,
                "enabled_families": ["IMPULSE_PULLBACK"],
                "ev_model": ev_model,
                "win_model": win_model,
                "validated": all(gates.values()),
                "real_capital_allowed": False,
            },
            BUNDLE,
        )
    _status("complete", 100, str(report["verdict"]))
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
