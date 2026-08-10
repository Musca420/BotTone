from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_bot.musca_v5_research import BARS, MINUTES, build_features

OUTPUT = Path("data/ml/musca_v5/market_state_features.parquet")
TIMEFRAMES = (1, 5, 15, 30)
MARKET_STATE_FEATURES = (
    "daily_vwap_distance_bps",
    "rolling_vwap_distance_bps",
    "rolling_vwap_slope_bps",
    "rolling_vwap_slope_change_bps",
    "rolling_vwap_tests_1h",
    "rolling_vwap_rejections_1h",
    "time_since_rolling_vwap_cross_minutes",
    "rolling_vwap_rejection_strength_bps",
    "rolling_vwap_band_position",
    "swing_long_distance_bps",
    "swing_short_distance_bps",
    "swing_long_slope_bps",
    "swing_short_slope_bps",
    "swing_long_slope_change_bps",
    "swing_short_slope_change_bps",
    "swing_long_anchor_age_bars",
    "swing_short_anchor_age_bars",
    "swing_long_anchor_price",
    "swing_short_anchor_price",
    "swing_long_volume_since_anchor",
    "swing_short_volume_since_anchor",
    "swing_long_return_since_anchor_bps",
    "swing_short_return_since_anchor_bps",
    "swing_long_tests_1h",
    "swing_short_tests_1h",
    "swing_long_rejections_1h",
    "swing_short_rejections_1h",
    "swing_long_rejection_strength_bps",
    "swing_short_rejection_strength_bps",
    "rolling_swing_long_distance_bps",
    "rolling_swing_short_distance_bps",
    "rolling_swing_long_convergence_bps",
    "rolling_swing_short_convergence_bps",
    *(f"atr_{minutes}m_bps" for minutes in TIMEFRAMES),
    *(f"trend_{minutes}m" for minutes in TIMEFRAMES),
    *(f"vwap_state_{minutes}m" for minutes in TIMEFRAMES),
    "realized_volatility_30m_bps",
    "volume_percentile",
)
MARKET_STATE_METADATA = (
    "swing_long_anchor_timestamp",
    "swing_short_anchor_timestamp",
    "swing_long_anchor_type",
    "swing_short_anchor_type",
)


def _swing_anchor_context(data: pd.DataFrame, *, long: bool) -> pd.DataFrame:
    """Return causal swing-anchor metadata; a pivot is known two completed bars later."""
    low = data["perp_low"].to_numpy(float)
    high = data["perp_high"].to_numpy(float)
    volume = data["perp_volume"].to_numpy(float)
    anchor_index = np.zeros(len(data), dtype=int)
    anchor_price = np.full(len(data), np.nan)
    cumulative_volume = 0.0
    anchor = 0
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
            else:
                cumulative_volume += volume[current]
        else:
            cumulative_volume += volume[current]
        anchor_index[current] = anchor
        anchor_price[current] = low[anchor] if long else high[anchor]
    cumulative = np.nancumsum(volume)
    before_anchor = np.where(anchor_index > 0, cumulative[np.maximum(anchor_index - 1, 0)], 0.0)
    return pd.DataFrame(
        {
            "anchor_index": anchor_index,
            "anchor_price": anchor_price,
            "anchor_age_bars": np.arange(len(data)) - anchor_index,
            "volume_since_anchor": cumulative - before_anchor,
        },
        index=data.index,
    )


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _multi_timeframe(minutes: pd.DataFrame) -> pd.DataFrame:
    source = minutes.loc[minutes["is_available"]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
    source = source.set_index("timestamp").sort_index()
    result: pd.DataFrame | None = None
    for timeframe in TIMEFRAMES:
        bars = source.resample(f"{timeframe}min").agg(
            open=("perp_open", "first"),
            high=("perp_high", "max"),
            low=("perp_low", "min"),
            close=("perp_close", "last"),
            volume=("perp_volume", "sum"),
            quote=("perp_quote_volume", "sum"),
        ).dropna()
        atr = _true_range(bars).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        ema_fast = bars["close"].ewm(span=12, adjust=False).mean()
        ema_slow = bars["close"].ewm(span=48, adjust=False).mean()
        rolling_vwap = bars["quote"].rolling(96, min_periods=24).sum() / bars[
            "volume"
        ].rolling(96, min_periods=24).sum().replace(0, np.nan)
        part = pd.DataFrame(
            {
                "available_at": bars.index + pd.Timedelta(minutes=timeframe),
                f"atr_{timeframe}m_bps": atr / bars["close"] * 10_000,
                f"trend_{timeframe}m": (ema_fast - ema_slow) / atr.replace(0, np.nan),
                f"vwap_state_{timeframe}m": (
                    bars["close"] - rolling_vwap
                ) / atr.replace(0, np.nan),
            }
        ).reset_index(drop=True)
        result = part if result is None else pd.merge_asof(
            result.sort_values("available_at"),
            part.sort_values("available_at"),
            on="available_at",
            direction="backward",
        )
    if result is None:
        raise ValueError("Multi-timeframe BTC state is empty")
    return result


def build_market_state(bars: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    state = build_features(bars).copy()
    state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
    close = state["perp_close"]
    atr = state["atr"].replace(0, np.nan)
    rolling = state["rolling_vwap_4h"]
    distance = close - rolling
    touch = distance.abs().le(0.25 * atr)
    cross = distance.ge(0).ne(distance.shift(1).ge(0))
    rejection = touch.shift(1, fill_value=False) & distance.abs().gt(0.25 * atr)
    last_cross = state["available_at"].where(cross).ffill()
    state["daily_vwap_distance_bps"] = (close / state["daily_vwap"] - 1) * 10_000
    state["rolling_vwap_distance_bps"] = (close / rolling - 1) * 10_000
    state["rolling_vwap_slope_bps"] = rolling.pct_change(3) * 10_000
    state["rolling_vwap_slope_change_bps"] = state["rolling_vwap_slope_bps"].diff(3)
    state["rolling_vwap_tests_1h"] = touch.rolling(12, min_periods=1).sum()
    state["rolling_vwap_rejections_1h"] = rejection.rolling(12, min_periods=1).sum()
    state["time_since_rolling_vwap_cross_minutes"] = (
        state["available_at"] - last_cross
    ).dt.total_seconds() / 60
    state["rolling_vwap_rejection_strength_bps"] = (
        state["rolling_vwap_distance_bps"].abs().where(rejection, 0.0)
    )
    rolling_sigma = close.shift(1).rolling(48, min_periods=12).std()
    state["rolling_vwap_band_position"] = distance / rolling_sigma.replace(0, np.nan)
    for side in ("long", "short"):
        center = state[f"swing_avwap_{side}"]
        anchor = _swing_anchor_context(state, long=side == "long")
        state[f"swing_{side}_distance_bps"] = (close / center - 1) * 10_000
        state[f"swing_{side}_slope_bps"] = center.pct_change(3) * 10_000
        state[f"swing_{side}_slope_change_bps"] = state[
            f"swing_{side}_slope_bps"
        ].diff(3)
        state[f"swing_{side}_anchor_age_bars"] = anchor["anchor_age_bars"]
        state[f"swing_{side}_anchor_price"] = anchor["anchor_price"]
        state[f"swing_{side}_volume_since_anchor"] = anchor["volume_since_anchor"]
        state[f"swing_{side}_anchor_timestamp"] = state["timestamp"].iloc[
            anchor["anchor_index"].to_numpy(int)
        ].to_numpy()
        state[f"swing_{side}_anchor_type"] = "SWING_LOW" if side == "long" else "SWING_HIGH"
        state[f"swing_{side}_return_since_anchor_bps"] = (
            close / anchor["anchor_price"] - 1
        ) * 10_000
        swing_distance = close - center
        swing_touch = swing_distance.abs().le(0.25 * atr)
        swing_rejection = swing_touch.shift(1, fill_value=False) & swing_distance.abs().gt(
            0.25 * atr
        )
        state[f"swing_{side}_tests_1h"] = swing_touch.rolling(12, min_periods=1).sum()
        state[f"swing_{side}_rejections_1h"] = swing_rejection.rolling(
            12, min_periods=1
        ).sum()
        state[f"swing_{side}_rejection_strength_bps"] = state[
            f"swing_{side}_distance_bps"
        ].abs().where(swing_rejection, 0.0)
        state[f"rolling_swing_{side}_distance_bps"] = (rolling / center - 1) * 10_000
        state[f"rolling_swing_{side}_convergence_bps"] = state[
            f"rolling_swing_{side}_distance_bps"
        ].abs().diff().mul(-1)
    context = _multi_timeframe(minutes)
    state = pd.merge_asof(
        state.sort_values("available_at"),
        context.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    )
    minute_source = minutes.loc[minutes["is_available"]].copy()
    minute_source["available_at"] = pd.to_datetime(minute_source["available_at"], utc=True)
    minute_source = minute_source.sort_values("available_at")
    returns = minute_source["perp_close"].pct_change()
    minute_source["realized_volatility_30m_bps"] = returns.rolling(30).std() * 10_000
    minute_source["volume_percentile"] = (
        minute_source["perp_quote_volume"].shift(1).rolling(360, min_periods=60).rank(pct=True)
    )
    state = pd.merge_asof(
        state.sort_values("available_at"),
        minute_source[
            ["available_at", "realized_volatility_30m_bps", "volume_percentile"]
        ],
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=2),
    )
    state["coverage_valid"] = np.isfinite(state[list(MARKET_STATE_FEATURES)]).all(axis=1)
    return state[
        ["available_at", *MARKET_STATE_METADATA, *MARKET_STATE_FEATURES, "coverage_valid"]
    ]


def materialize() -> pd.DataFrame:
    bars = pd.read_parquet(BARS)
    minutes = pd.read_parquet(MINUTES)
    state = build_market_state(bars, minutes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    state.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)
    return state


if __name__ == "__main__":
    result = materialize()
    print(f"{len(result):,} causal BTC multi-timeframe states -> {OUTPUT}")
