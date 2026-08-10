from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_bot.bitunix_fees import FUTURES_VIP_FEE_BPS

SOURCE = Path("data/ml/hybrid_v25/asset=BTCUSDT/minutes.parquet")
ROOT = Path("data/ml/btc_vwap_alpha_v1")
MATRIX = ROOT / "events.parquet"
BUNDLE = Path("data/models/btc_vwap_alpha_v1/bundle.joblib")
REPORT = Path("data/reports/btc_vwap_alpha_v1.json")
STATUS = Path("data/reports/btc_vwap_alpha_v1.status.json")
HORIZONS = (5, 15, 30, 60)
BARRIERS = (20, 30, 50)
MIN_STOP_BPS = 15.0
MAX_TECHNICAL_STOP_BPS = 200.0
STOP_RANGE_MULTIPLIER = 2.0
FAMILIES = (
    "VWAP_PULLBACK_CONTINUATION",
    "VWAP_REVERSION",
    "ROLLING_VWAP_REENTRY",
)
LOCAL_EXPERTS = tuple(
    f"{family}:{side}" for family in FAMILIES for side in ("LONG", "SHORT")
)
FEATURES = (
    "directional_return_1m_bps",
    "directional_return_2m_bps",
    "directional_return_3m_bps",
    "directional_return_5m_bps",
    "directional_return_10m_bps",
    "directional_return_15m_bps",
    "directional_return_30m_bps",
    "directional_return_60m_bps",
    "directional_vwap_distance_bps",
    "absolute_vwap_distance_bps",
    "directional_vwap_distance_5m_bps",
    "directional_vwap_distance_15m_bps",
    "directional_vwap_distance_240m_bps",
    "directional_vwap_slope_bps",
    "directional_vwap_slope_change_bps",
    "directional_vwap_distance_velocity_3m_bps",
    "vwap_tests_30m",
    "vwap_rejections_30m",
    "time_since_vwap_cross_minutes",
    "directional_vwap_rejection_strength_bps",
    "directional_vwap_band_position",
    "structural_stop_bps",
    "range_60s_bps",
    "atr_1m_bps",
    "atr_5m_bps",
    "atr_15m_bps",
    "atr_30m_bps",
    "realized_volatility_30m_bps",
    "volatility_percentile",
    "volume_percentile",
    "directional_taker_imbalance_60s",
    "directional_taker_imbalance_15m",
    "directional_taker_imbalance_60m",
    "directional_taker_imbalance_change_5m",
    "trade_count_zscore",
    "aggressive_volume_zscore",
    "directional_candle_body_bps",
    "directional_wick_imbalance_bps",
    "efficiency_15m",
    "efficiency_60m",
    "directional_range_position_15m",
    "directional_range_position_60m",
    "directional_oi_change_1h",
    "return_oi_interaction",
    "directional_basis_bps",
    "directional_funding_z",
    "hour_sin",
    "hour_cos",
)
VIP_LEVELS = tuple(range(6))
LIVE_REQUIRED = {
    "signal_at",
    "side",
    "expert",
    "alpha_feature_contract_valid",
    "alpha_return_1m_bps",
    "alpha_return_2m_bps",
    "alpha_return_3m_bps",
    "alpha_return_5m_bps",
    "alpha_return_10m_bps",
    "alpha_return_15m_bps",
    "alpha_return_30m_bps",
    "alpha_return_60m_bps",
    "alpha_vwap_distance_bps",
    "alpha_vwap_distance_5m_bps",
    "alpha_vwap_distance_15m_bps",
    "alpha_vwap_distance_240m_bps",
    "alpha_vwap_slope_bps",
    "alpha_range_60s_bps",
    "alpha_taker_imbalance_60s",
    "stop_bps",
    "alpha_vwap_slope_change_bps",
    "alpha_vwap_distance_velocity_3m_bps",
    "alpha_vwap_tests_30m",
    "alpha_vwap_rejections_30m",
    "alpha_time_since_vwap_cross_minutes",
    "alpha_vwap_rejection_strength_bps",
    "alpha_vwap_band_position",
    "alpha_atr_1m_bps",
    "alpha_atr_5m_bps",
    "alpha_atr_15m_bps",
    "alpha_atr_30m_bps",
    "alpha_realized_volatility_30m_bps",
    "alpha_volatility_percentile",
    "alpha_volume_percentile",
    "alpha_taker_imbalance_15m",
    "alpha_taker_imbalance_60m",
    "alpha_taker_imbalance_change_5m",
    "alpha_trade_count_zscore",
    "alpha_aggressive_volume_zscore",
    "alpha_candle_body_bps",
    "alpha_wick_imbalance_bps",
    "alpha_efficiency_15m",
    "alpha_efficiency_60m",
    "alpha_range_position_15m",
    "alpha_range_position_60m",
    "oi_change_1h_binance",
    "basis_bps_binance",
    "funding_z_binance",
}
SLIPPAGE_RESERVE_BPS_PER_SIDE = 0.5
HOLDOUT_WEEKS = 12
PROTOCOL = {
    "name": "musca_v5_btc_vwap_alpha_aligned",
    "asset": "BTCUSDT",
    "source": "Binance official perpetual and spot 1m archives",
    "entry": "next one-minute open after the completed signal minute",
    "decision_cadence_minutes": 5,
    "events": list(FAMILIES),
    "local_experts": list(LOCAL_EXPERTS),
    "model_pooling": "separate model and calibration for every setup family and side",
    "event_rules": {
        "episode_rule": "one candidate on the false-to-true edge of each family and side",
        "continuation": (
            "direction=sign(15m return + 30m return + VWAP slope votes); both returns "
            "aligned; directional VWAP distance -3..15bps; 1m return and taker flow aligned"
        ),
        "reversion": "abs(VWAP distance)>=5bps; 1m return and taker flow point back to VWAP",
        "reentry": (
            "VWAP side changed in the five completed minutes since the previous decision; "
            "current 1m return and taker flow remain aligned"
        ),
    },
    "horizons_minutes": list(HORIZONS),
    "barriers_bps": list(BARRIERS),
    "same_bar": "stop wins",
    "barrier_outcomes": "TARGET, STOP, or TIMEOUT at 60m; timeout uses observed close",
    "funding": (
        "Binance funding is retained as causal Alpha context and a venue-specific "
        "diagnostic only; it is never substituted for Bitunix funding. Live scoring "
        "subtracts observed Bitunix funding through the execution quote"
    ),
    "model_target": (
        "joint multiclass P(target first), P(stop first), P(timeout), plus conditional "
        "timeout return; "
        "EV gross = P(target)*target - P(stop)*structural stop + P(timeout)*timeout return"
    ),
    "calibration": (
        "chronologically separate Platt probability, isotonic EV, and moving-block "
        "residual confidence windows"
    ),
    "position_lock": "one position until observed target/stop minute, at most 60m",
    "paper_management": (
        "chosen state-dependent target and structural stop remain fixed; only independent "
        "catastrophic/risk exits may override the Alpha plan"
    ),
    "stop_bps": (
        "structural 5m extreme plus volatility buffer; "
        f"clip({MIN_STOP_BPS:g}, {MAX_TECHNICAL_STOP_BPS:g})"
    ),
    "holdout_weeks": HOLDOUT_WEEKS,
    "operating_fee_profile": "Bitunix futures VIP0 taker/taker",
    "slippage_reserve_bps_per_side": SLIPPAGE_RESERVE_BPS_PER_SIDE,
    "scenario_fee_profiles": [f"VIP{level}" for level in VIP_LEVELS],
    "feature_contract": {
        "cadence": "completed Binance one-minute trade bars; 60m VWAP center",
        "availability": (
            "minute close plus one minute; no partial current candle; 420-minute "
            "contiguous warm-up; OI context available_at cannot exceed decision time"
        ),
        "shared_builder": "canonical_minute_market_features",
        "live_support": "fail closed outside fitted feature support",
        "excluded": {
            "anchored_vwap": "live anchor state did not reproduce the historical reset rule",
            "order_book_alpha": "no matching multi-year Binance L2 history",
            "cross_exchange_confirmation": "not part of Binance-only Alpha",
        },
    },
    "real_capital_allowed": False,
    "features": list(FEATURES),
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def canonical_minute_market_features(minutes: pd.DataFrame) -> pd.DataFrame:
    """Build the market features shared by historical and live Alpha paths."""
    required = {
        "timestamp",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_quote",
        "trade_count",
    }
    missing = required - set(minutes.columns)
    if missing:
        raise ValueError(f"Missing canonical minute columns: {sorted(missing)}")
    rows = minutes.sort_values("timestamp").reset_index(drop=True).copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    price = pd.to_numeric(rows["close"], errors="coerce")
    quote = pd.to_numeric(rows["quote_volume"], errors="coerce")
    volume = pd.to_numeric(rows["volume"], errors="coerce")
    rolling_vwaps = {
        window: quote.rolling(window, min_periods=window).sum()
        / volume.rolling(window, min_periods=window).sum().replace(0, np.nan)
        for window in (5, 15, 60, 240)
    }
    rolling_vwap = rolling_vwaps[60]
    buy = pd.to_numeric(rows["taker_buy_quote"], errors="coerce")
    rows["rolling_vwap"] = rolling_vwap
    for window, values in rolling_vwaps.items():
        rows[f"rolling_vwap_{window}m"] = values
    rows["return_1m_bps"] = price.pct_change() * 10_000
    rows["return_2m_bps"] = price.pct_change(2) * 10_000
    rows["return_3m_bps"] = price.pct_change(3) * 10_000
    rows["return_5m_bps"] = price.pct_change(5) * 10_000
    rows["return_10m_bps"] = price.pct_change(10) * 10_000
    rows["return_15m_bps"] = price.pct_change(15) * 10_000
    rows["return_30m_bps"] = price.pct_change(30) * 10_000
    rows["return_60m_bps"] = price.pct_change(60) * 10_000
    rows["vwap_distance_bps"] = (price / rolling_vwap - 1) * 10_000
    for window in (5, 15, 240):
        rows[f"vwap_distance_{window}m_bps"] = (
            price / rolling_vwaps[window] - 1
        ) * 10_000
    rows["vwap_slope_bps"] = rolling_vwap.pct_change() * 10_000
    rows["vwap_slope_change_bps"] = rows["vwap_slope_bps"].diff(5)
    rows["vwap_distance_velocity_3m_bps"] = rows["vwap_distance_bps"].diff(3)
    rows["range_60s_bps"] = (rows["high"] - rows["low"]) / price * 10_000
    rows["recent_low_5m"] = pd.to_numeric(rows["low"], errors="coerce").rolling(5).min()
    rows["recent_high_5m"] = pd.to_numeric(rows["high"], errors="coerce").rolling(5).max()
    rows["taker_imbalance_60s"] = (2 * buy - quote) / quote.replace(0, np.nan)
    for window in (15, 60):
        rolling_quote = quote.rolling(window, min_periods=window).sum()
        rolling_buy = buy.rolling(window, min_periods=window).sum()
        rows[f"taker_imbalance_{window}m"] = (
            2 * rolling_buy - rolling_quote
        ) / rolling_quote.replace(0, np.nan)
    rows["taker_imbalance_change_5m"] = rows["taker_imbalance_60m"].diff(5)
    trades = pd.to_numeric(rows["trade_count"], errors="coerce")
    trade_mean = trades.shift(1).rolling(1_440, min_periods=240).mean()
    trade_std = trades.shift(1).rolling(1_440, min_periods=240).std()
    volume_mean = quote.shift(1).rolling(1_440, min_periods=240).mean()
    volume_std = quote.shift(1).rolling(1_440, min_periods=240).std()
    rows["trade_count_zscore"] = (trades - trade_mean) / trade_std.replace(0, np.nan)
    rows["aggressive_volume_zscore"] = (quote - volume_mean) / volume_std.replace(0, np.nan)
    candle_body = price - pd.to_numeric(rows["open"], errors="coerce")
    upper_wick = pd.to_numeric(rows["high"], errors="coerce") - pd.concat(
        [pd.to_numeric(rows["open"], errors="coerce"), price], axis=1
    ).max(axis=1)
    lower_wick = pd.concat(
        [pd.to_numeric(rows["open"], errors="coerce"), price], axis=1
    ).min(axis=1) - pd.to_numeric(rows["low"], errors="coerce")
    rows["candle_body_bps"] = candle_body / price * 10_000
    rows["wick_imbalance_bps"] = (lower_wick - upper_wick) / price * 10_000
    absolute_return = rows["return_1m_bps"].abs()
    for window in (15, 60):
        path_length = absolute_return.rolling(window, min_periods=window).sum()
        rows[f"efficiency_{window}m"] = (
            rows[f"return_{window}m_bps"].abs() / path_length.replace(0, np.nan)
        )
        rolling_high = pd.to_numeric(rows["high"], errors="coerce").rolling(window).max()
        rolling_low = pd.to_numeric(rows["low"], errors="coerce").rolling(window).min()
        rows[f"range_position_{window}m"] = (
            2 * (price - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan) - 1
        )
    rows["feature_available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    distance = rows["vwap_distance_bps"]
    touch = distance.abs().le(1.0)
    cross = distance.ge(0).ne(distance.shift(1).ge(0))
    rejection = touch.shift(1, fill_value=False) & distance.abs().gt(2.0)
    rows["vwap_tests_30m"] = touch.rolling(30, min_periods=1).sum()
    rows["vwap_rejections_30m"] = rejection.rolling(30, min_periods=1).sum()
    last_cross = rows["feature_available_at"].where(cross).ffill()
    rows["time_since_vwap_cross_minutes"] = (
        rows["feature_available_at"] - last_cross
    ).dt.total_seconds() / 60
    rows["vwap_rejection_strength_bps"] = distance.abs().where(rejection, 0.0)
    rolling_sigma = price.shift(1).rolling(60, min_periods=20).std()
    rows["vwap_band_position"] = (price - rolling_vwap) / rolling_sigma.replace(0, np.nan)

    previous = price.shift(1)
    true_range = pd.concat(
        [
            rows["high"] - rows["low"],
            (rows["high"] - previous).abs(),
            (rows["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for timeframe in (1, 5, 15, 30):
        rows[f"atr_{timeframe}m_bps"] = (
            true_range.rolling(14 * timeframe, min_periods=14 * timeframe).mean()
            / price
            * 10_000
        )
    returns = price.pct_change()
    rows["realized_volatility_30m_bps"] = returns.rolling(30).std() * 10_000
    rows["volatility_percentile"] = rows["realized_volatility_30m_bps"].shift(1).rolling(
        1_440, min_periods=240
    ).rank(pct=True)
    rows["volume_percentile"] = quote.shift(1).rolling(1_440, min_periods=240).rank(
        pct=True
    )
    gap = rows["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    gap.iloc[0] = False
    rows["feature_contract_valid"] = (
        (~gap).rolling(420, min_periods=420).sum().eq(420)
        & rows[["open", "high", "low", "close", "volume", "quote_volume"]]
        .notna()
        .all(axis=1)
        & rows["available_at"].ge(rows["timestamp"] + pd.Timedelta(minutes=1))
    )
    return rows


def _historical_features(minutes: pd.DataFrame) -> pd.DataFrame:
    rows = minutes.sort_values("timestamp").reset_index(drop=True).copy()
    for source, target in (
        ("perp_open", "open"),
        ("perp_high", "high"),
        ("perp_low", "low"),
        ("perp_close", "close"),
        ("perp_volume", "volume"),
        ("perp_quote_volume", "quote_volume"),
        ("perp_taker_buy_quote", "taker_buy_quote"),
        ("perp_trade_count", "trade_count"),
    ):
        rows[target] = pd.to_numeric(rows[source], errors="coerce")
    rows = canonical_minute_market_features(rows)
    funding_mean = rows["perp_funding_rate"].shift(1).rolling(10_080, min_periods=1_440).mean()
    funding_std = rows["perp_funding_rate"].shift(1).rolling(10_080, min_periods=1_440).std()
    rows["funding_z"] = (rows["perp_funding_rate"] - funding_mean) / funding_std.replace(
        0, np.nan
    )
    rows["basis_bps"] = (rows["perp_mark_close"] / rows["spot_close"] - 1) * 10_000
    rows["return_oi_interaction_raw"] = rows["return_5m_bps"] * rows["oi_change_1h"]
    context_available = pd.to_datetime(
        rows["context_available_at"], utc=True, errors="coerce"
    )
    rows["oi_feature_available"] = (
        rows["oi_change_1h"].notna()
        & context_available.notna()
        & context_available.le(rows["feature_available_at"])
    )
    return rows


def setup_conditions(
    rows: pd.DataFrame, family: str, *, prefix: str = ""
) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    """Canonical setup definition shared by historical and live decision paths."""
    return_1m = rows[f"{prefix}return_1m_bps"]
    return_15m = rows[f"{prefix}return_15m_bps"]
    return_30m = rows[f"{prefix}return_30m_bps"]
    distance = rows[f"{prefix}vwap_distance_bps"]
    flow = rows[f"{prefix}taker_imbalance_60s"]
    trend_score = (
        np.sign(return_15m)
        + np.sign(return_30m)
        + np.sign(rows[f"{prefix}vwap_slope_bps"])
    )
    direction = np.sign(trend_score)
    if family == "VWAP_PULLBACK_CONTINUATION":
        side = direction
        stages = [
            ("direction", direction.notna() & direction.ne(0)),
            ("trend_15m", return_15m.mul(direction).gt(0)),
            ("trend_30m", return_30m.mul(direction).gt(0)),
            ("vwap_zone", distance.mul(direction).between(-3.0, 15.0)),
            ("price_restart", return_1m.mul(direction).gt(0)),
            ("taker_flow_restart", flow.mul(direction).gt(0)),
        ]
    elif family == "VWAP_REVERSION":
        side = -np.sign(distance)
        stages = [
            ("vwap_extension", distance.abs().ge(5.0)),
            ("price_reversal", return_1m.mul(side).gt(0)),
            ("taker_flow_reversal", flow.mul(side).gt(0)),
        ]
    elif family == "ROLLING_VWAP_REENTRY":
        side = np.sign(distance)
        previous_side = np.sign(distance.shift(1))
        crossed = (
            side.notna()
            & previous_side.notna()
            & side.ne(0)
            & previous_side.ne(0)
            & previous_side.ne(side)
        )
        stages = [
            (
                "vwap_cross",
                crossed.rolling(5, min_periods=1).max().fillna(False).astype(bool),
            ),
            ("price_restart", return_1m.mul(side).gt(0)),
            ("taker_flow_restart", flow.mul(side).gt(0)),
        ]
    else:
        raise ValueError(f"Unknown family: {family}")
    return pd.Series(side, index=rows.index), stages


def _candidate_indexes(rows: pd.DataFrame) -> list[tuple[int, int, str]]:
    required = [
        "return_1m_bps",
        "return_15m_bps",
        "return_30m_bps",
        "vwap_distance_bps",
        "vwap_slope_bps",
        "taker_imbalance_60s",
        "volatility_percentile",
        "volume_percentile",
        "oi_change_1h",
        "funding_z",
        "basis_bps",
    ]
    covered = (
        rows["is_available"].to_numpy(bool)
        & rows["feature_contract_valid"].fillna(False).to_numpy(bool)
        & rows["oi_feature_available"].fillna(False).to_numpy(bool)
        & np.isfinite(rows[required].to_numpy(float)).all(axis=1)
    )
    covered[:6] = False
    covered[-61:] = False
    decision_time = pd.to_datetime(rows["feature_available_at"], utc=True)
    decision_positions = np.flatnonzero(
        covered & decision_time.dt.minute.mod(5).eq(0).to_numpy()
    )
    result: list[tuple[int, int, str]] = []
    for family in FAMILIES:
        side_values, stages = setup_conditions(rows, family)
        mask = covered.copy()
        for _, condition in stages:
            mask &= condition.fillna(False).to_numpy(bool)
        sides = side_values.fillna(0).to_numpy(np.int8)
        for side in (-1, 1):
            active = (mask & (sides == side))[decision_positions]
            rising = active & ~np.r_[False, active[:-1]]
            indexes = decision_positions[np.flatnonzero(rising)]
            if len(indexes):
                result.extend((int(index), side, family) for index in indexes)
    return sorted(result, key=lambda item: (item[0], item[2]))


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists():
        cached = pd.read_parquet(MATRIX)
        if (
            not force
            and len(cached)
            and "protocol_hash" in cached.columns
            and cached["protocol_hash"].nunique() == 1
            and cached["protocol_hash"].iloc[0] == PROTOCOL_HASH
        ):
            return cached
        if len(cached) and "protocol_hash" in cached:
            old_hash = str(cached["protocol_hash"].iloc[0])
            archive = MATRIX.parent / "protocol_archive" / f"events_{old_hash}.parquet"
            if old_hash != PROTOCOL_HASH and not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(MATRIX, archive)
    _status("alpha_dataset", "Caricamento storico BTC Binance", 5)
    rows = _historical_features(pd.read_parquet(SOURCE))
    indexes = _candidate_indexes(rows)
    candidate_index = np.asarray([item[0] for item in indexes], dtype=int)
    sides = np.asarray([item[1] for item in indexes], dtype=float)
    families = np.asarray([item[2] for item in indexes], dtype=object)
    entry_price = rows["perp_open"].to_numpy(float)[candidate_index + 1]
    high = np.lib.stride_tricks.sliding_window_view(rows["perp_high"].to_numpy(float), 60)[
        candidate_index + 1
    ]
    low = np.lib.stride_tricks.sliding_window_view(rows["perp_low"].to_numpy(float), 60)[
        candidate_index + 1
    ]
    funding_path = np.lib.stride_tricks.sliding_window_view(
        pd.to_numeric(rows["perp_funding_event_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(float),
        60,
    )[candidate_index + 1]
    favorable = np.where(
        sides[:, None] > 0,
        (high / entry_price[:, None] - 1) * 10_000,
        (entry_price[:, None] / low - 1) * 10_000,
    )
    adverse = np.where(
        sides[:, None] > 0,
        (entry_price[:, None] / low - 1) * 10_000,
        (high / entry_price[:, None] - 1) * 10_000,
    )
    signal = rows.iloc[candidate_index]
    close_60 = rows["perp_close"].to_numpy(float)[candidate_index + 60]
    hours = pd.to_datetime(signal["feature_available_at"], utc=True).dt.hour.to_numpy()
    entry_times = pd.DatetimeIndex(
        pd.to_datetime(rows.iloc[candidate_index + 1]["timestamp"], utc=True)
    )
    minute_in_funding_cycle = (entry_times.hour % 8) * 60 + entry_times.minute
    minutes_to_next_funding = np.where(
        minute_in_funding_cycle == 0,
        0,
        8 * 60 - minute_in_funding_cycle,
    )
    expected_binance_funding_bps = np.where(
        minutes_to_next_funding <= max(HORIZONS),
        np.maximum(
            0.0,
            sides * signal["perp_funding_rate"].to_numpy(float) * 10_000,
        ),
        0.0,
    )
    recent_low = rows["recent_low_5m"].to_numpy(float)[candidate_index]
    recent_high = rows["recent_high_5m"].to_numpy(float)[candidate_index]
    structural_stop = np.where(
        sides > 0,
        (entry_price / recent_low - 1) * 10_000,
        (recent_high / entry_price - 1) * 10_000,
    )
    stop_bps = np.clip(
        np.maximum(
            STOP_RANGE_MULTIPLIER * signal["range_60s_bps"].to_numpy(float),
            structural_stop + 2,
        ),
        MIN_STOP_BPS,
        MAX_TECHNICAL_STOP_BPS,
    )
    matrix = pd.DataFrame(
        {
            "signal_at": pd.to_datetime(signal["feature_available_at"], utc=True).to_numpy(),
            "entry_at": pd.to_datetime(
                rows.iloc[candidate_index + 1]["timestamp"], utc=True
            ).to_numpy(),
            "label_available_at": pd.to_datetime(
                rows.iloc[candidate_index + 60]["available_at"], utc=True
            ).to_numpy(),
            "family": families,
            "side": sides.astype(int),
            "entry_price": entry_price,
            "stop_bps": stop_bps,
            "expected_binance_funding_bps": expected_binance_funding_bps,
            "minutes_to_next_funding": minutes_to_next_funding.astype(float),
            "gross_60m_bps": sides * (close_60 / entry_price - 1) * 10_000,
            "protocol_hash": PROTOCOL_HASH,
            "directional_return_1m_bps": sides * signal["return_1m_bps"].to_numpy(float),
            "directional_return_2m_bps": sides * signal["return_2m_bps"].to_numpy(float),
            "directional_return_3m_bps": sides * signal["return_3m_bps"].to_numpy(float),
            "directional_return_5m_bps": sides * signal["return_5m_bps"].to_numpy(float),
            "directional_return_10m_bps": sides
            * signal["return_10m_bps"].to_numpy(float),
            "directional_return_15m_bps": sides
            * signal["return_15m_bps"].to_numpy(float),
            "directional_return_30m_bps": sides
            * signal["return_30m_bps"].to_numpy(float),
            "directional_return_60m_bps": sides
            * signal["return_60m_bps"].to_numpy(float),
            "directional_vwap_distance_bps": sides * signal["vwap_distance_bps"].to_numpy(float),
            "absolute_vwap_distance_bps": signal["vwap_distance_bps"].abs().to_numpy(float),
            **{
                f"directional_vwap_distance_{window}m_bps": sides
                * signal[f"vwap_distance_{window}m_bps"].to_numpy(float)
                for window in (5, 15, 240)
            },
            "directional_vwap_slope_bps": sides * signal["vwap_slope_bps"].to_numpy(float),
            "directional_vwap_slope_change_bps": sides
            * signal["vwap_slope_change_bps"].to_numpy(float),
            "directional_vwap_distance_velocity_3m_bps": sides
            * signal["vwap_distance_velocity_3m_bps"].to_numpy(float),
            "vwap_tests_30m": signal["vwap_tests_30m"].to_numpy(float),
            "vwap_rejections_30m": signal["vwap_rejections_30m"].to_numpy(float),
            "time_since_vwap_cross_minutes": signal[
                "time_since_vwap_cross_minutes"
            ].to_numpy(float),
            "directional_vwap_rejection_strength_bps": sides
            * signal["vwap_rejection_strength_bps"].to_numpy(float),
            "directional_vwap_band_position": sides
            * signal["vwap_band_position"].to_numpy(float),
            "structural_stop_bps": stop_bps,
            "range_60s_bps": signal["range_60s_bps"].to_numpy(float),
            **{
                f"atr_{timeframe}m_bps": signal[f"atr_{timeframe}m_bps"].to_numpy(float)
                for timeframe in (1, 5, 15, 30)
            },
            "realized_volatility_30m_bps": signal[
                "realized_volatility_30m_bps"
            ].to_numpy(float),
            "volatility_percentile": signal["volatility_percentile"].to_numpy(float),
            "volume_percentile": signal["volume_percentile"].to_numpy(float),
            "directional_taker_imbalance_60s": sides
            * signal["taker_imbalance_60s"].to_numpy(float),
            "directional_taker_imbalance_15m": sides
            * signal["taker_imbalance_15m"].to_numpy(float),
            "directional_taker_imbalance_60m": sides
            * signal["taker_imbalance_60m"].to_numpy(float),
            "directional_taker_imbalance_change_5m": sides
            * signal["taker_imbalance_change_5m"].to_numpy(float),
            "trade_count_zscore": signal["trade_count_zscore"].to_numpy(float),
            "aggressive_volume_zscore": signal["aggressive_volume_zscore"].to_numpy(float),
            "directional_candle_body_bps": sides
            * signal["candle_body_bps"].to_numpy(float),
            "directional_wick_imbalance_bps": sides
            * signal["wick_imbalance_bps"].to_numpy(float),
            "efficiency_15m": signal["efficiency_15m"].to_numpy(float),
            "efficiency_60m": signal["efficiency_60m"].to_numpy(float),
            "directional_range_position_15m": sides
            * signal["range_position_15m"].to_numpy(float),
            "directional_range_position_60m": sides
            * signal["range_position_60m"].to_numpy(float),
            "directional_oi_change_1h": sides * signal["oi_change_1h"].to_numpy(float),
            "return_oi_interaction": signal["return_oi_interaction_raw"].to_numpy(float),
            "directional_basis_bps": sides * signal["basis_bps"].to_numpy(float),
            "directional_funding_z": sides * signal["funding_z"].to_numpy(float),
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
        }
    )
    for horizon in HORIZONS:
        matrix[f"mfe_{horizon}m_bps"] = favorable[:, :horizon].max(axis=1)
        matrix[f"mae_{horizon}m_bps"] = adverse[:, :horizon].max(axis=1)
        matrix[f"target_30bps_within_{horizon}m"] = (favorable[:, :horizon] >= 30).any(axis=1)
    for barrier in BARRIERS:
        (
            target_first,
            stop_first,
            timeout,
            realized,
            target_minutes,
            exit_minutes,
            same_minute_ambiguous,
        ) = _barrier_outcome(
            favorable,
            adverse,
            stop_bps,
            matrix["gross_60m_bps"].to_numpy(float),
            barrier,
        )
        matrix[f"target_{barrier}bps_before_stop"] = target_first
        matrix[f"stop_before_{barrier}bps"] = stop_first
        matrix[f"timeout_{barrier}bps"] = timeout
        matrix[f"timeout_return_{barrier}bps"] = matrix["gross_60m_bps"].where(timeout)
        matrix[f"time_to_{barrier}bps_minutes"] = target_minutes
        matrix[f"plan_return_{barrier}bps"] = realized
        matrix[f"exit_after_{barrier}bps_minutes"] = exit_minutes
        realized_funding_bps = _realized_funding_bps(
            funding_path, sides, exit_minutes
        )
        matrix[f"binance_funding_bps_{barrier}bps"] = realized_funding_bps
        matrix[f"plan_return_after_binance_funding_{barrier}bps"] = (
            realized - realized_funding_bps
        )
        matrix[f"exit_at_{barrier}bps"] = pd.to_datetime(
            matrix["entry_at"], utc=True
        ) + pd.to_timedelta(exit_minutes, unit="m")
        matrix[f"same_minute_target_stop_{barrier}bps"] = same_minute_ambiguous
    matrix = matrix.dropna(subset=list(FEATURES)).reset_index(drop=True)
    _status("alpha_dataset", f"{len(matrix):,} eventi etichettati", 30)
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    matrix.to_parquet(temporary, index=False)
    temporary.replace(MATRIX)
    return matrix


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Alpha features must be finite and fully covered")
    return values


def _state_weights(rows: pd.DataFrame) -> np.ndarray:
    """Give every market timestamp equal total weight across counterfactual actions."""
    counts = rows.groupby("signal_at")["signal_at"].transform("size").to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.mean()


def _fit_weighted(model: Any, rows: pd.DataFrame, target: str) -> Any:
    weights = _state_weights(rows)
    if hasattr(model, "named_steps"):
        final_step = next(reversed(model.named_steps))
        return model.fit(_x(rows), rows[target], **{f"{final_step}__sample_weight": weights})
    return model.fit(_x(rows), rows[target], sample_weight=weights)


def _barrier_outcome(
    favorable: np.ndarray,
    adverse: np.ndarray,
    stop_bps: np.ndarray,
    terminal_gross_bps: np.ndarray,
    barrier_bps: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Resolve target/stop/timeout causally; stop wins an ambiguous same minute."""
    stop_hit = adverse >= stop_bps[:, None]
    target_hit = favorable >= barrier_bps
    has_stop = stop_hit.any(axis=1)
    has_target = target_hit.any(axis=1)
    horizon = favorable.shape[1]
    first_stop = np.where(has_stop, stop_hit.argmax(axis=1), horizon)
    first_target = np.where(has_target, target_hit.argmax(axis=1), horizon)
    target_first = has_target & (first_target < first_stop)
    stop_first = has_stop & (first_stop <= first_target)
    same_minute_ambiguous = has_target & has_stop & (first_target == first_stop)
    timeout = ~(target_first | stop_first)
    realized = np.select(
        (target_first, stop_first),
        (np.full(len(favorable), float(barrier_bps)), -stop_bps),
        default=terminal_gross_bps,
    )
    target_minutes = np.where(target_first, first_target + 1.0, np.nan)
    exit_minutes = np.select(
        (target_first, stop_first),
        (first_target + 1.0, first_stop + 1.0),
        default=float(horizon),
    )
    return (
        target_first,
        stop_first,
        timeout,
        realized,
        target_minutes,
        exit_minutes,
        same_minute_ambiguous,
    )


def _realized_funding_bps(
    funding_path: np.ndarray, sides: np.ndarray, exit_minutes: np.ndarray
) -> np.ndarray:
    """Charge only settlements observed while the position was actually open."""
    cumulative = funding_path.cumsum(axis=1)
    positions = np.clip(exit_minutes.astype(int) - 1, 0, funding_path.shape[1] - 1)
    return np.asarray(
        sides * cumulative[np.arange(len(sides)), positions] * 10_000,
        dtype=float,
    )


def _one_position_at_a_time(candidates: pd.DataFrame) -> pd.DataFrame:
    """Replay candidates chronologically and release capital at the observed exit."""
    if candidates.empty:
        return candidates
    chosen: list[Any] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    ordering = ["signal_at"]
    ascending = [True]
    if "predicted_ev_bps" in candidates:
        ordering.append("predicted_ev_bps")
        ascending.append(False)
    for index, row in candidates.sort_values(ordering, ascending=ascending).iterrows():
        signal_at = pd.Timestamp(row["signal_at"])
        target = int(row["chosen_target_bps"])
        if signal_at >= free_at:
            chosen.append(index)
            free_at = pd.Timestamp(row[f"exit_at_{target}bps"])
    return candidates.loc[chosen].sort_values("signal_at")


def _bound_physical_predictions(
    predictions: dict[str, np.ndarray], stop_bps: np.ndarray
) -> None:
    """Constrain regression outputs to the support of their observed labels."""
    for horizon in HORIZONS:
        for excursion in ("mfe", "mae"):
            column = f"expected_{excursion}_{horizon}m_bps"
            if column in predictions:
                predictions[column] = np.maximum(predictions[column], 0.0)
    mfe = predictions.get("expected_mfe_60m_bps", np.full(len(stop_bps), np.inf))
    mae = predictions.get("expected_mae_60m_bps", stop_bps)
    for barrier in BARRIERS:
        plan = f"expected_plan_return_{barrier}bps"
        upper = np.minimum(float(barrier), mfe)
        lower = -np.minimum(stop_bps, mae)
        if plan in predictions:
            predictions[plan] = np.minimum(
                np.maximum(predictions[plan], lower), upper
            )
        target_time = f"expected_time_to_{barrier}bps_minutes"
        if target_time in predictions:
            predictions[target_time] = np.clip(predictions[target_time], 1.0, 60.0)
        timeout_return = f"expected_timeout_return_{barrier}bps"
        if timeout_return in predictions:
            predictions[timeout_return] = np.clip(
                predictions[timeout_return], -stop_bps, float(barrier)
            )


def _probability_plan_gross(
    predictions: dict[str, np.ndarray], stop_bps: np.ndarray, barrier: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_probability = np.asarray(
        predictions[f"p_target_{barrier}bps_before_stop"], dtype=float
    )
    stop_probability = np.asarray(
        predictions[f"p_stop_before_{barrier}bps"], dtype=float
    )
    probability_total = np.maximum(1.0, target_probability + stop_probability)
    target_probability = target_probability / probability_total
    stop_probability = stop_probability / probability_total
    timeout_probability = np.clip(1 - target_probability - stop_probability, 0, 1)
    timeout_return = np.asarray(
        predictions[f"expected_timeout_return_{barrier}bps"], dtype=float
    )
    gross = (
        target_probability * barrier
        - stop_probability * stop_bps
        + timeout_probability * timeout_return
    )
    return gross, target_probability, stop_probability, timeout_probability


def _fit_ev_calibrator(
    raw_gross: np.ndarray, realized_gross: np.ndarray
) -> IsotonicRegression:
    """Fit monotonic EV calibration on stable equal-count blocks, not tail points."""
    order = np.argsort(raw_gross)
    blocks = np.array_split(order, min(20, len(order)))
    x = np.asarray([raw_gross[block].mean() for block in blocks], dtype=float)
    y = np.asarray([realized_gross[block].mean() for block in blocks], dtype=float)
    weight = np.asarray([len(block) for block in blocks], dtype=float)
    return IsotonicRegression(out_of_bounds="clip").fit(x, y, sample_weight=weight)


def _calibrated_plan_gross(
    raw_gross: np.ndarray,
    calibrator: IsotonicRegression,
    stop_bps: np.ndarray,
    barrier: int,
) -> np.ndarray:
    calibrated = np.asarray(calibrator.predict(raw_gross), dtype=float)
    return np.clip(calibrated, -stop_bps, float(barrier))


def _outcome_classifier(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, random_state=seed),
        )
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=20,
        reg_alpha=1,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=seed,
    )


def _regressor(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:pseudohubererror",
        huber_slope=10.0,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=20,
        reg_alpha=1,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        random_state=seed,
    )


def _outcome_truth(rows: pd.DataFrame, barrier: int) -> np.ndarray:
    return np.select(
        (
            rows[f"target_{barrier}bps_before_stop"].to_numpy(bool),
            rows[f"stop_before_{barrier}bps"].to_numpy(bool),
        ),
        (1, 2),
        default=0,
    ).astype(np.int8)


def _fit_outcome_head(
    kind: str, fit: pd.DataFrame, calibration: pd.DataFrame, barrier: int
) -> dict[str, Any]:
    truth = _outcome_truth(fit, barrier)
    counts = np.bincount(truth, minlength=3).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Barrier {barrier} has an outcome class without training rows")
    weights = _state_weights(fit) * len(truth) / (3 * counts[truth])
    model = _outcome_classifier(kind, 42 + barrier)
    if hasattr(model, "named_steps"):
        final_step = next(reversed(model.named_steps))
        model.fit(_x(fit), truth, **{f"{final_step}__sample_weight": weights})
    else:
        model.fit(_x(fit), truth, sample_weight=weights)
    raw = np.clip(model.predict_proba(_x(calibration)), 1e-7, 1)
    calibration_truth = _outcome_truth(calibration, barrier)
    calibrator = LogisticRegression(C=1.0, max_iter=2_000, random_state=42).fit(
        np.log(raw), calibration_truth
    )
    return {"model": model, "calibrator": calibrator}


def _predict_outcome(head: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    raw = np.clip(head["model"].predict_proba(_x(rows)), 1e-7, 1)
    calibrated = np.asarray(head["calibrator"].predict_proba(np.log(raw)), dtype=float)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _outcome_predictions(
    outcome_heads: dict[int, dict[str, Any]], rows: pd.DataFrame
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for barrier, head in outcome_heads.items():
        probability = _predict_outcome(head, rows)
        predictions[f"p_target_{barrier}bps_before_stop"] = probability[:, 1]
        predictions[f"p_stop_before_{barrier}bps"] = probability[:, 2]
    return predictions


def _local_expert_keys(rows: pd.DataFrame) -> pd.Series:
    if "family" in rows:
        family = rows["family"].astype(str).str.upper()
    else:
        expert = rows["expert"].astype(str).str.upper()
        family = pd.Series("", index=rows.index, dtype="object")
        for name in FAMILIES:
            family.loc[expert.str.contains(name, regex=False)] = name
    if "side" not in rows:
        return pd.Series("", index=rows.index, dtype="object")
    side_text = rows["side"].astype(str).str.upper()
    side = np.where(
        side_text.eq("1") | side_text.eq("1.0") | side_text.eq("LONG"),
        "LONG",
        np.where(
            side_text.eq("-1") | side_text.eq("-1.0") | side_text.eq("SHORT"),
            "SHORT",
            "",
        ),
    )
    return pd.Series(
        [
            f"{family_name}:{side_name}"
            for family_name, side_name in zip(family.tolist(), side, strict=True)
        ],
        index=rows.index,
        dtype="object",
    )


def _local_predictions(
    experts: dict[str, dict[str, Any]], rows: pd.DataFrame
) -> dict[str, np.ndarray]:
    keys = _local_expert_keys(rows)
    unknown = sorted(set(keys) - set(experts))
    if unknown:
        raise ValueError(f"Rows do not map to a trained local expert: {unknown}")
    predictions: dict[str, np.ndarray] = {}
    for key, policy in experts.items():
        positions = np.flatnonzero(keys.eq(key).to_numpy())
        if not len(positions):
            continue
        local_rows = rows.iloc[positions]
        local = _outcome_predictions(policy["outcome_heads"], local_rows)
        local |= {
            f"expected_{target}": np.asarray(
                model.predict(_x(local_rows)), dtype=float
            )
            for target, model in policy["regressions"].items()
        }
        for name, values in local.items():
            if name not in predictions:
                predictions[name] = np.full(len(rows), np.nan)
            predictions[name][positions] = values
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Local expert predictions are incomplete")
    return predictions


def _local_calibrated_gross(
    rows: pd.DataFrame,
    experts: dict[str, dict[str, Any]],
    predictions: dict[str, np.ndarray],
    stop_bps: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    keys = _local_expert_keys(rows)
    gross = {barrier: np.full(len(rows), np.nan) for barrier in BARRIERS}
    lcb = {barrier: np.full(len(rows), np.nan) for barrier in BARRIERS}
    for key, policy in experts.items():
        positions = np.flatnonzero(keys.eq(key).to_numpy())
        if not len(positions):
            continue
        for barrier in BARRIERS:
            raw = _probability_plan_gross(
                {name: values[positions] for name, values in predictions.items()},
                stop_bps[positions],
                barrier,
            )[0]
            gross[barrier][positions] = _calibrated_plan_gross(
                raw,
                policy["ev_calibrators"][barrier],
                stop_bps[positions],
                barrier,
            )
            lcb[barrier][positions] = float(policy["residual_lcb_bps"][barrier])
    if any(not np.isfinite(values).all() for values in (*gross.values(), *lcb.values())):
        raise ValueError("Local expert EV calibration is incomplete")
    return gross, lcb


def _block_mean_lcb(residuals: np.ndarray, *, seed: int = 20260808) -> float:
    """One-sided 95% confidence bound for mean error, not a single-trade quantile."""
    if len(residuals) < 30:
        return float("-inf")
    rng = np.random.default_rng(seed)
    block = max(10, round(np.sqrt(len(residuals))))
    starts = np.arange(max(1, len(residuals) - block + 1))
    means = np.empty(2_000)
    for sample in range(len(means)):
        chunks: list[np.ndarray] = []
        while sum(len(chunk) for chunk in chunks) < len(residuals):
            start = int(rng.choice(starts))
            chunks.append(residuals[start : start + block])
        means[sample] = np.concatenate(chunks)[: len(residuals)].mean()
    return float(np.quantile(means, 0.05))


def _policy_metrics(
    rows: pd.DataFrame, returns: np.ndarray, cost_bps: float | np.ndarray
) -> dict[str, Any]:
    if not len(returns):
        return {
            "trades": 0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "positive_window_fraction": 0.0,
            "bootstrap_expectancy_lcb_95_bps": None,
        }
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    costs = np.broadcast_to(np.asarray(cost_bps, dtype=float), len(returns))
    risk = rows["stop_bps"].to_numpy(float) + costs
    account_returns = 0.01 * returns / risk
    equity = np.cumprod(1 + account_returns)
    drawdown = 1 - equity / np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    monthly = pd.Series(
        returns,
        index=pd.DatetimeIndex(pd.to_datetime(rows["signal_at"], utc=True)),
    ).resample("30D").sum()
    bootstrap_lcb = _block_mean_lcb(returns)
    return {
        "trades": len(returns),
        "expectancy_bps": float(returns.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "max_drawdown": float(drawdown.max(initial=0.0)),
        "positive_window_fraction": float(monthly.gt(0).mean()),
        "bootstrap_expectancy_lcb_95_bps": (
            bootstrap_lcb if np.isfinite(bootstrap_lcb) else None
        ),
        "_returns_bps": returns.tolist(),
        "_costs_bps": costs.tolist(),
        "_stop_bps": rows["stop_bps"].to_numpy(float).tolist(),
        "_signal_at": [
            pd.Timestamp(value).isoformat() for value in rows["signal_at"]
        ],
    }


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _selection_probability_quality(
    rows: pd.DataFrame,
    outcome_head: dict[str, Any],
    timeout_model: Any,
    ev_calibrator: IsotonicRegression,
    *,
    vip_level: int = 5,
) -> dict[str, Any]:
    barrier = 30
    cost = 2 * FUTURES_VIP_FEE_BPS[vip_level][1] + 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
    stop = rows["stop_bps"].to_numpy(float)
    probability = _predict_outcome(outcome_head, rows)
    predictions = {
        f"p_target_{barrier}bps_before_stop": probability[:, 1],
        f"p_stop_before_{barrier}bps": probability[:, 2],
        f"expected_timeout_return_{barrier}bps": np.asarray(
            timeout_model.predict(_x(rows)), float
        ),
    }
    _bound_physical_predictions(predictions, stop)
    raw_gross = _probability_plan_gross(predictions, stop, barrier)[0]
    gross = _calibrated_plan_gross(raw_gross, ev_calibrator, stop, barrier)
    predicted_net = gross - cost
    candidates = rows.loc[predicted_net > 0].copy()
    candidates["chosen_target_bps"] = barrier
    candidates["predicted_ev_bps"] = predicted_net[predicted_net > 0]
    selected = _one_position_at_a_time(candidates)
    realized = np.asarray(
        [
            float(row[f"plan_return_{int(row['chosen_target_bps'])}bps"]) - cost
            for _, row in selected.iterrows()
        ],
        dtype=float,
    )
    return _policy_metrics(selected, realized, cost)


def _evaluate_economic_policy(
    rows: pd.DataFrame,
    experts: dict[str, dict[str, Any]],
    *,
    vip_level: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed_cost = (
        2 * FUTURES_VIP_FEE_BPS[vip_level][1]
        + 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
    )
    expected_cost = np.full(len(rows), fixed_cost, dtype=float)
    stop = rows["stop_bps"].to_numpy(float)
    predictions = _local_predictions(experts, rows)
    _bound_physical_predictions(predictions, stop)
    predicted_mfe = predictions["expected_mfe_60m_bps"]
    calibrated_gross, local_lcb = _local_calibrated_gross(
        rows, experts, predictions, stop
    )
    predicted_plan_gross = np.column_stack(
        [calibrated_gross[barrier] for barrier in BARRIERS]
    )
    prudent_plan_net = np.column_stack(
        [
            predicted_plan_gross[:, position] - expected_cost + local_lcb[barrier]
            for position, barrier in enumerate(BARRIERS)
        ]
    )
    best = prudent_plan_net.argmax(axis=1)
    target = np.asarray(BARRIERS)[best]
    best_gross = predicted_plan_gross[np.arange(len(rows)), best]
    best_ev = best_gross - expected_cost
    best_prudent_ev = prudent_plan_net[np.arange(len(rows)), best]
    accepted = (
        (best_ev > 0)
        & (best_prudent_ev > 0)
        & (predicted_mfe >= 3 * expected_cost)
        & (target >= 3 * expected_cost)
    )
    candidates = rows.loc[accepted].copy()
    candidates["predicted_ev_bps"] = best_ev[accepted]
    candidates["expected_cost_bps"] = expected_cost[accepted]
    candidates["chosen_target_bps"] = target[accepted]
    candidates = candidates.sort_values(
        ["signal_at", "predicted_ev_bps"], ascending=[True, False]
    )
    selected = _one_position_at_a_time(candidates)
    if selected.empty:
        realized = np.asarray([], dtype=float)
        stress = realized
        selected_cost = np.asarray([], dtype=float)
    else:
        realized_gross = np.asarray(
            [
                float(row[f"plan_return_{int(target_bps)}bps"])
             for (_, row), target_bps in zip(
                 selected.iterrows(), selected["chosen_target_bps"], strict=True
             )
            ],
            dtype=float,
        )
        realized = realized_gross - fixed_cost
        stress = realized_gross - 2 * fixed_cost
        selected_cost = selected["expected_cost_bps"].to_numpy(float)
        selected["realized_return_bps"] = realized
        selected["stress_return_bps"] = stress
    metrics = _policy_metrics(selected, realized, selected_cost)
    stress_metrics = _policy_metrics(selected, stress, 2 * fixed_cost)
    coverage: dict[str, Any] = {}
    ranking = rows.assign(
        predicted_ev_bps=best_prudent_ev, chosen_target_bps=target
    ).sort_values(
        "predicted_ev_bps", ascending=False
    )
    for fraction in (0.005, 0.01, 0.02, 0.05, 0.10):
        sample = _one_position_at_a_time(
            ranking.head(max(1, round(len(ranking) * fraction)))
        )
        sample_target = sample["chosen_target_bps"].to_numpy(int)
        sample_return = np.asarray(
            [
                float(row[f"plan_return_{int(target_bps)}bps"]) - fixed_cost
             for (_, row), target_bps in zip(sample.iterrows(), sample_target, strict=True)]
        )
        coverage[f"{fraction:.3f}"] = _policy_metrics(
            sample, sample_return, fixed_cost
        )
    return metrics | {"stress_2x": stress_metrics}, coverage


LOCAL_REGRESSION_TARGETS = (
    *(f"plan_return_{barrier}bps" for barrier in BARRIERS),
    *(f"timeout_return_{barrier}bps" for barrier in BARRIERS),
    "mfe_60m_bps",
    "mae_60m_bps",
    *(f"time_to_{barrier}bps_minutes" for barrier in BARRIERS),
)


def _calibration_windows(
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if calibration.empty:
        raise ValueError("Calibration window is empty")
    signal = pd.to_datetime(calibration["signal_at"], utc=True)
    labels = pd.to_datetime(calibration["label_available_at"], utc=True)
    start = signal.min()
    end = signal.max() + pd.Timedelta(minutes=1)
    duration = end - start
    probability_boundary = start + duration * 0.50
    ev_boundary = start + duration * 0.75
    probability = calibration.loc[
        signal.lt(probability_boundary) & labels.lt(probability_boundary)
    ].copy()
    ev = calibration.loc[
        signal.ge(probability_boundary)
        & signal.lt(ev_boundary)
        & labels.lt(ev_boundary)
    ].copy()
    residual = calibration.loc[signal.ge(ev_boundary)].copy()
    return probability, ev, residual


def _fit_local_policy(
    kind: str,
    fit: pd.DataFrame,
    probability_calibration: pd.DataFrame,
    ev_calibration: pd.DataFrame,
    residual_calibration: pd.DataFrame,
) -> dict[str, Any]:
    if len(fit) < 100 or min(
        len(probability_calibration),
        len(ev_calibration),
        len(residual_calibration),
    ) < 30:
        raise ValueError("Insufficient chronological rows for a local expert")
    outcome_heads = {
        barrier: _fit_outcome_head(kind, fit, probability_calibration, barrier)
        for barrier in BARRIERS
    }
    regressions: dict[str, Any] = {}
    for number, target in enumerate(LOCAL_REGRESSION_TARGETS):
        available = fit[target].notna()
        if int(available.sum()) < 100:
            raise ValueError(f"Insufficient {target} rows for a local expert")
        regressions[target] = _fit_weighted(
            _regressor(kind, 100 + number), fit.loc[available], target
        )
    ev_predictions = _outcome_predictions(outcome_heads, ev_calibration)
    ev_predictions |= {
        f"expected_{target}": np.asarray(model.predict(_x(ev_calibration)), dtype=float)
        for target, model in regressions.items()
    }
    ev_stop = ev_calibration["stop_bps"].to_numpy(float)
    _bound_physical_predictions(ev_predictions, ev_stop)
    ev_calibrators = {
        barrier: _fit_ev_calibrator(
            _probability_plan_gross(ev_predictions, ev_stop, barrier)[0],
            ev_calibration[f"plan_return_{barrier}bps"].to_numpy(float),
        )
        for barrier in BARRIERS
    }
    residual_predictions = _outcome_predictions(outcome_heads, residual_calibration)
    residual_predictions |= {
        f"expected_{target}": np.asarray(
            model.predict(_x(residual_calibration)), dtype=float
        )
        for target, model in regressions.items()
    }
    residual_stop = residual_calibration["stop_bps"].to_numpy(float)
    _bound_physical_predictions(residual_predictions, residual_stop)
    residual_lcb = {
        barrier: _block_mean_lcb(
            residual_calibration[f"plan_return_{barrier}bps"].to_numpy(float)
            - _calibrated_plan_gross(
                _probability_plan_gross(
                    residual_predictions, residual_stop, barrier
                )[0],
                ev_calibrators[barrier],
                residual_stop,
                barrier,
            ),
            seed=20260808 + barrier,
        )
        for barrier in BARRIERS
    }
    return {
        "kind": kind,
        "outcome_heads": outcome_heads,
        "regressions": regressions,
        "ev_calibrators": ev_calibrators,
        "residual_lcb_bps": residual_lcb,
        "feature_support": _feature_support(fit),
    }


def _local_selection_metrics(
    key: str, policy: dict[str, Any], rows: pd.DataFrame
) -> dict[str, Any]:
    predictions = _local_predictions({key: policy}, rows)
    stop = rows["stop_bps"].to_numpy(float)
    _bound_physical_predictions(predictions, stop)
    plan_mae = float(
        np.mean(
            [
                mean_absolute_error(
                    rows[f"plan_return_{barrier}bps"],
                    predictions[f"expected_plan_return_{barrier}bps"],
                )
                for barrier in BARRIERS
            ]
        )
    )
    target_probability = predictions["p_target_30bps_before_stop"]
    stop_probability = predictions["p_stop_before_30bps"]
    decision, _ = _evaluate_economic_policy(rows, {key: policy}, vip_level=5)
    public_decision = _public_metrics(decision)
    return {
        "plan_return_mae_bps": plan_mae,
        "p30_target_brier": float(
            brier_score_loss(rows["target_30bps_before_stop"], target_probability)
        ),
        "p30_stop_brier": float(
            brier_score_loss(rows["stop_before_30bps"], stop_probability)
        ),
        **{
            f"decision_{name}": value
            for name, value in public_decision.items()
            if name != "stress_2x"
        },
    }


def _xgboost_wins(challengers: dict[str, dict[str, Any]]) -> bool:
    ridge = challengers["ridge"]
    xgboost = challengers["xgboost"]
    xgb_lcb = xgboost["decision_bootstrap_expectancy_lcb_95_bps"]
    ridge_lcb = ridge["decision_bootstrap_expectancy_lcb_95_bps"]
    return bool(
        xgboost["plan_return_mae_bps"] < ridge["plan_return_mae_bps"]
        and xgboost["p30_target_brier"] < ridge["p30_target_brier"]
        and xgboost["p30_stop_brier"] < ridge["p30_stop_brier"]
        and xgboost["decision_expectancy_bps"] > ridge["decision_expectancy_bps"]
        and xgboost["decision_expectancy_bps"] > 0
        and (xgb_lcb if xgb_lcb is not None else float("-inf"))
        > (ridge_lcb if ridge_lcb is not None else float("-inf"))
        and (xgb_lcb if xgb_lcb is not None else float("-inf")) > 0
    )


def _time_window(
    rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    signal = pd.to_datetime(rows["signal_at"], utc=True)
    labels = pd.to_datetime(rows["label_available_at"], utc=True)
    return rows.loc[signal.ge(start) & signal.lt(end) & labels.lt(end)].copy()


def _expert_rows(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    return rows.loc[_local_expert_keys(rows).eq(key)].copy()


def _fit_expert_set(
    kinds: dict[str, str],
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    progress: tuple[str, str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    probability, ev, residual = _calibration_windows(calibration)
    experts: dict[str, dict[str, Any]] = {}
    for number, key in enumerate(LOCAL_EXPERTS, start=1):
        if progress is not None:
            phase, prefix, percent = progress
            _status(
                phase,
                f"{prefix} - esperto {number}/6 {key} ({kinds[key]})",
                percent,
            )
        experts[key] = _fit_local_policy(
            kinds[key],
            _expert_rows(fit, key),
            _expert_rows(probability, key),
            _expert_rows(ev, key),
            _expert_rows(residual, key),
        )
    return experts


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray(
        [value for fold in folds for value in fold.get("_returns_bps", [])],
        dtype=float,
    )
    if not len(returns):
        empty = _policy_metrics(
            pd.DataFrame(columns=["signal_at", "stop_bps"]), returns, 0.0
        )
        return _public_metrics(empty) | {"stress_2x": _public_metrics(empty)}
    records = pd.DataFrame(
        {
            "signal_at": [
                value for fold in folds for value in fold.get("_signal_at", [])
            ],
            "stop_bps": [
                value for fold in folds for value in fold.get("_stop_bps", [])
            ],
        }
    ).sort_values("signal_at")
    order = records.index.to_numpy(int)
    records = records.reset_index(drop=True)
    costs = np.asarray(
        [value for fold in folds for value in fold.get("_costs_bps", [])],
        dtype=float,
    )[order]
    returns = returns[order]
    aggregate = _policy_metrics(records, returns, costs)
    stress_folds = [fold["stress_2x"] for fold in folds]
    stress_returns = np.asarray(
        [
            value
            for fold in stress_folds
            for value in fold.get("_returns_bps", [])
        ],
        dtype=float,
    )[order]
    stress_costs = np.asarray(
        [value for fold in stress_folds for value in fold.get("_costs_bps", [])],
        dtype=float,
    )[order]
    stress = _policy_metrics(records, stress_returns, stress_costs)
    positive_fold_fraction = float(
        np.mean([fold["expectancy_bps"] > 0 for fold in folds])
    )
    aggregate["positive_window_fraction"] = positive_fold_fraction
    stress["positive_window_fraction"] = float(
        np.mean([fold["stress_2x"]["expectancy_bps"] > 0 for fold in folds])
    )
    return _public_metrics(aggregate) | {"stress_2x": _public_metrics(stress)}


def _economic_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    stress = metrics["stress_2x"]
    lcb = metrics["bootstrap_expectancy_lcb_95_bps"]
    return {
        "walk_forward_trades_300": metrics["trades"] >= 300,
        "expectancy_positive": metrics["expectancy_bps"] > 0,
        "bootstrap_lcb_positive": (
            lcb if lcb is not None else float("-inf")
        )
        > 0,
        "profit_factor_1_15": (metrics["profit_factor"] or 0) >= 1.15,
        "max_drawdown_8pct": metrics["trades"] > 0
        and metrics["max_drawdown"] <= 0.08,
        "majority_windows_positive": metrics["positive_window_fraction"] > 0.5,
        "stress_2x_nonnegative": stress["trades"] > 0
        and stress["expectancy_bps"] >= 0,
    }


def _obsolete_pooled_train(*, force_matrix: bool = False) -> dict[str, Any]:
    matrix = build_matrix(force=force_matrix).sort_values("signal_at").reset_index(drop=True)
    _status("alpha_training", f"{len(matrix):,} eventi BTC", 32)
    times = pd.to_datetime(matrix["signal_at"], utc=True)
    label_times = pd.to_datetime(matrix["label_available_at"], utc=True)
    holdout_start = times.max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    research = matrix.loc[times.lt(holdout_start) & label_times.lt(holdout_start)].copy()
    cut1, cut2, cut3 = (
        int(len(research) * 0.60),
        int(len(research) * 0.75),
        int(len(research) * 0.85),
    )
    boundary1 = pd.Timestamp(research.iloc[cut1]["signal_at"])
    boundary2 = pd.Timestamp(research.iloc[cut2]["signal_at"])
    boundary3 = pd.Timestamp(research.iloc[cut3]["signal_at"])
    signal = pd.to_datetime(research["signal_at"], utc=True)
    labels = pd.to_datetime(research["label_available_at"], utc=True)
    fit = research.loc[signal.lt(boundary1) & labels.lt(boundary1)].copy()
    calibration = research.loc[
        signal.ge(boundary1) & signal.lt(boundary2) & labels.lt(boundary2)
    ].copy()
    selection = research.loc[
        signal.ge(boundary2) & signal.lt(boundary3) & labels.lt(boundary3)
    ].copy()
    audit = research.loc[signal.ge(boundary3)].copy()
    calibration_signal = pd.to_datetime(calibration["signal_at"], utc=True)
    calibration_labels = pd.to_datetime(calibration["label_available_at"], utc=True)
    probability_boundary = pd.Timestamp(
        calibration.iloc[int(len(calibration) * 0.50)]["signal_at"]
    )
    ev_boundary = pd.Timestamp(
        calibration.iloc[int(len(calibration) * 0.75)]["signal_at"]
    )
    probability_calibration = calibration.loc[
        calibration_signal.lt(probability_boundary)
        & calibration_labels.lt(probability_boundary)
    ].copy()
    ev_calibration = calibration.loc[
        calibration_signal.ge(probability_boundary)
        & calibration_signal.lt(ev_boundary)
        & calibration_labels.lt(ev_boundary)
    ].copy()
    residual_calibration = calibration.loc[calibration_signal.ge(ev_boundary)].copy()
    challenger_selection: dict[str, dict[str, float]] = {}
    for position, kind in enumerate(("ridge", "xgboost"), start=1):
        _status("alpha_training", f"GPU challenger {kind} {position}/2", 35 + position * 15)
        plan_models = {
            barrier: _fit_weighted(
                _regressor(kind, 42 + barrier), fit, f"plan_return_{barrier}bps"
            )
            for barrier in BARRIERS
        }
        outcome_head = _fit_outcome_head(kind, fit, probability_calibration, 30)
        timeout_rows = fit.loc[fit["timeout_return_30bps"].notna()]
        timeout_model = _fit_weighted(
            _regressor(kind, 72), timeout_rows, "timeout_return_30bps"
        )
        ev_stop = ev_calibration["stop_bps"].to_numpy(float)
        ev_probability = _predict_outcome(outcome_head, ev_calibration)
        ev_predictions = {
            "p_target_30bps_before_stop": ev_probability[:, 1],
            "p_stop_before_30bps": ev_probability[:, 2],
            "expected_timeout_return_30bps": np.asarray(
                timeout_model.predict(_x(ev_calibration)), dtype=float
            ),
        }
        _bound_physical_predictions(ev_predictions, ev_stop)
        ev_calibrator = _fit_ev_calibrator(
            _probability_plan_gross(ev_predictions, ev_stop, 30)[0],
            ev_calibration["plan_return_30bps"].to_numpy(float),
        )
        plan_mae = np.mean(
            [
                mean_absolute_error(
                    selection[f"plan_return_{barrier}bps"],
                    np.clip(
                        plan_models[barrier].predict(_x(selection)),
                        -selection["stop_bps"].to_numpy(float),
                        float(barrier),
                    ),
                )
                for barrier in BARRIERS
            ]
        )
        outcome_probability = _predict_outcome(outcome_head, selection)
        target_probability = outcome_probability[:, 1]
        stop_probability = outcome_probability[:, 2]
        challenger_selection[kind] = {
            "plan_return_mae_bps": float(plan_mae),
            "p30_target_brier": float(
                brier_score_loss(
                    selection["target_30bps_before_stop"], target_probability
                )
            ),
            "p30_stop_brier": float(
                brier_score_loss(selection["stop_before_30bps"], stop_probability)
            ),
            **{
                f"decision_{key}": value
                for key, value in _selection_probability_quality(
                    selection, outcome_head, timeout_model, ev_calibrator
                ).items()
                if key != "stress_2x"
            },
        }
    xgb_wins = (
        challenger_selection["xgboost"]["plan_return_mae_bps"]
        < challenger_selection["ridge"]["plan_return_mae_bps"]
        and challenger_selection["xgboost"]["p30_target_brier"]
        < challenger_selection["ridge"]["p30_target_brier"]
        and challenger_selection["xgboost"]["p30_stop_brier"]
        < challenger_selection["ridge"]["p30_stop_brier"]
        and challenger_selection["xgboost"]["decision_expectancy_bps"]
        > challenger_selection["ridge"]["decision_expectancy_bps"]
        and challenger_selection["xgboost"]["decision_expectancy_bps"] > 0
        and (
            challenger_selection["xgboost"]["decision_bootstrap_expectancy_lcb_95_bps"]
            or float("-inf")
        )
        > (
            challenger_selection["ridge"]["decision_bootstrap_expectancy_lcb_95_bps"]
            or float("-inf")
        )
        and (
            challenger_selection["xgboost"]["decision_bootstrap_expectancy_lcb_95_bps"]
            or float("-inf")
        )
        > 0
    )
    champion = "xgboost" if xgb_wins else "ridge"
    _status("alpha_training", f"Champion {champion}; outcome heads", 70)
    regression_targets = [
        "gross_60m_bps",
        *(f"plan_return_{barrier}bps" for barrier in BARRIERS),
        *(f"timeout_return_{barrier}bps" for barrier in BARRIERS),
        *(f"mfe_{horizon}m_bps" for horizon in HORIZONS),
        *(f"mae_{horizon}m_bps" for horizon in HORIZONS),
        *(f"time_to_{barrier}bps_minutes" for barrier in BARRIERS),
    ]
    outcome_heads = {
        barrier: _fit_outcome_head(champion, fit, probability_calibration, barrier)
        for barrier in BARRIERS
    }
    regressions: dict[str, Any] = {}
    for number, target in enumerate(regression_targets):
        available = fit[target].notna()
        if int(available.sum()) >= 100:
            regressions[target] = _fit_weighted(
                _regressor(champion, 100 + number), fit.loc[available], target
            )
    ev_predictions = _outcome_predictions(outcome_heads, ev_calibration)
    ev_predictions |= {
        f"expected_{target}": np.asarray(model.predict(_x(ev_calibration)), float)
        for target, model in regressions.items()
    }
    ev_stop = ev_calibration["stop_bps"].to_numpy(float)
    _bound_physical_predictions(ev_predictions, ev_stop)
    ev_calibrators = {
        barrier: _fit_ev_calibrator(
            _probability_plan_gross(ev_predictions, ev_stop, barrier)[0],
            ev_calibration[f"plan_return_{barrier}bps"].to_numpy(float),
        )
        for barrier in BARRIERS
    }
    residual_predictions = _outcome_predictions(outcome_heads, residual_calibration)
    residual_predictions |= {
        f"expected_{target}": np.asarray(
            model.predict(_x(residual_calibration)), dtype=float
        )
        for target, model in regressions.items()
    }
    residual_stop = residual_calibration["stop_bps"].to_numpy(float)
    _bound_physical_predictions(residual_predictions, residual_stop)
    residual_lcb = {
        barrier: _block_mean_lcb(
            residual_calibration[f"plan_return_{barrier}bps"].to_numpy(float)
            - _calibrated_plan_gross(
                _probability_plan_gross(
                    residual_predictions, residual_stop, barrier
                )[0],
                ev_calibrators[barrier],
                residual_stop,
                barrier,
            ),
            seed=20260808 + barrier,
        )
        for barrier in BARRIERS
    }
    vip_policy_audit: dict[str, Any] = {}
    coverage_curves: dict[str, Any] = {}
    vip_gates: dict[str, dict[str, bool]] = {}
    for level in VIP_LEVELS:
        profile = f"VIP{level}"
        policy_audit, coverage_curve = _evaluate_economic_policy(  # type: ignore[call-arg,misc]
            audit,
            outcome_heads,  # type: ignore[arg-type]
            regressions,  # type: ignore[arg-type]
            ev_calibrators,
            residual_lcb,
            vip_level=level,
        )
        stress = policy_audit["stress_2x"]
        vip_policy_audit[profile] = policy_audit
        coverage_curves[profile] = coverage_curve
        vip_gates[profile] = {
            "audit_trades_300": policy_audit["trades"] >= 300,
            "expectancy_positive": policy_audit["expectancy_bps"] > 0,
            "bootstrap_lcb_positive": (
                policy_audit["bootstrap_expectancy_lcb_95_bps"] or float("-inf")
            )
            > 0,
            "profit_factor_1_15": (policy_audit["profit_factor"] or 0) >= 1.15,
            "max_drawdown_8pct": policy_audit["trades"] > 0
            and policy_audit["max_drawdown"] <= 0.08,
            "majority_windows_positive": policy_audit["positive_window_fraction"] > 0.5,
            "stress_2x_nonnegative": stress["trades"] > 0
            and stress["expectancy_bps"] >= 0,
        }
    deployable_profiles = [
        profile for profile, profile_gates in vip_gates.items() if all(profile_gates.values())
    ]
    verdict = "RESEARCH_ALPHA_READY" if deployable_profiles else "NO_ECONOMIC_ALPHA"
    support = _feature_support(fit)
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "champion": champion,
        "features": FEATURES,
        "feature_support": support,
        "outcome_heads": outcome_heads,
        "regressions": regressions,
        "ev_calibrators": ev_calibrators,
        "residual_lcb_bps": residual_lcb,
        "trained_through": pd.Timestamp(fit["signal_at"].max()).isoformat(),
        "calibrated_through": pd.Timestamp(
            residual_calibration["signal_at"].max()
        ).isoformat(),
        "selected_through": pd.Timestamp(selection["signal_at"].max()).isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "verdict": verdict,
        "deployable_profiles": deployable_profiles,
        "live_orders_enabled": False,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(BUNDLE)
    report = {
        "status": verdict,
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "source_rows": len(pd.read_parquet(SOURCE, columns=["timestamp"])),
        "events": len(matrix),
        "same_minute_target_stop_rate": {
            f"{barrier}bps": float(matrix[f"same_minute_target_stop_{barrier}bps"].mean())
            for barrier in BARRIERS
        },
        "fit_events": len(fit),
        "calibration_events": len(calibration),
        "probability_calibration_events": len(probability_calibration),
        "ev_calibration_events": len(ev_calibration),
        "residual_calibration_events": len(residual_calibration),
        "selection_events": len(selection),
        "audit_events": len(audit),
        "sealed_holdout_events": int(times.ge(holdout_start).sum()),
        "holdout_start": holdout_start.isoformat(),
        "holdout_opened": False,
        "champion": champion,
        "challenger_selection": challenger_selection,
        "audit_used_for_model_selection": False,
        "residual_lcb_bps": residual_lcb,
        "economic_policy_audit": vip_policy_audit["VIP0"],
        "vip_policy_audit": vip_policy_audit,
        "coverage_curve": coverage_curves["VIP0"],
        "coverage_curves": coverage_curves,
        "gates": vip_gates["VIP0"],
        "vip_gates": vip_gates,
        "deployable_profiles": deployable_profiles,
        "feature_support": support,
        "bundle": str(BUNDLE),
        "real_capital_allowed": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT, report)
    _status("alpha_complete", verdict, 100)
    return report


def train(*, force_matrix: bool = False) -> dict[str, Any]:
    """Train six causal local experts and audit only post-selection time windows."""
    matrix = build_matrix(force=force_matrix).sort_values("signal_at").reset_index(
        drop=True
    )
    _status("alpha_training", f"{len(matrix):,} eventi BTC locali", 32)
    times = pd.to_datetime(matrix["signal_at"], utc=True)
    labels = pd.to_datetime(matrix["label_available_at"], utc=True)
    holdout_start = times.max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    research = matrix.loc[times.lt(holdout_start) & labels.lt(holdout_start)].copy()
    research_start = pd.Timestamp(research["signal_at"].min())
    discovery_fit_end = research_start + pd.Timedelta(weeks=52)
    discovery_calibration_end = discovery_fit_end + pd.Timedelta(weeks=4)
    discovery_selection_end = discovery_calibration_end + pd.Timedelta(weeks=4)
    if discovery_selection_end + pd.Timedelta(weeks=4) > holdout_start:
        raise ValueError("Insufficient chronology for discovery plus untouched OOS folds")

    discovery_fit = _time_window(research, research_start, discovery_fit_end)
    discovery_calibration = _time_window(
        research, discovery_fit_end, discovery_calibration_end
    )
    discovery_selection = _time_window(
        research, discovery_calibration_end, discovery_selection_end
    )
    probability, ev, residual = _calibration_windows(discovery_calibration)
    challenger_selection: dict[str, dict[str, dict[str, Any]]] = {}
    champions: dict[str, str] = {}
    for expert_number, key in enumerate(LOCAL_EXPERTS, start=1):
        challenger_selection[key] = {}
        policies: dict[str, dict[str, Any]] = {}
        for kind_number, kind in enumerate(("ridge", "xgboost"), start=1):
            _status(
                "alpha_discovery",
                f"Esperto {expert_number}/6 {key} challenger {kind_number}/2 {kind}",
                33 + 17 * ((expert_number - 1) * 2 + kind_number) / 12,
            )
            policy = _fit_local_policy(
                kind,
                _expert_rows(discovery_fit, key),
                _expert_rows(probability, key),
                _expert_rows(ev, key),
                _expert_rows(residual, key),
            )
            policies[kind] = policy
            challenger_selection[key][kind] = _local_selection_metrics(
                key, policy, _expert_rows(discovery_selection, key)
            )
        champions[key] = (
            "xgboost"
            if _xgboost_wins(challenger_selection[key])
            else "ridge"
        )

    fold_start = discovery_selection_end
    fold_results: list[dict[str, Any]] = []
    fold_metrics: dict[str, list[dict[str, Any]]] = {
        f"VIP{level}": [] for level in VIP_LEVELS
    }
    possible_folds = max(
        1,
        int((holdout_start - fold_start) // pd.Timedelta(weeks=4)),
    )
    fold_number = 0
    while fold_start + pd.Timedelta(weeks=4) <= holdout_start:
        fold_number += 1
        test_end = fold_start + pd.Timedelta(weeks=4)
        calibration_start = fold_start - pd.Timedelta(weeks=4)
        fit_start = calibration_start - pd.Timedelta(weeks=52)
        fold_fit = _time_window(research, fit_start, calibration_start)
        fold_calibration = _time_window(research, calibration_start, fold_start)
        fold_test = _time_window(research, fold_start, test_end)
        _status(
            "alpha_walk_forward",
            f"Fold {fold_number}/{possible_folds}: 52w fit + 4w calibration + 4w OOS",
            52 + 28 * fold_number / possible_folds,
        )
        experts = _fit_expert_set(
            champions,
            fold_fit,
            fold_calibration,
            progress=(
                "alpha_walk_forward",
                f"Fold {fold_number}/{possible_folds}",
                52 + 28 * fold_number / possible_folds,
            ),
        )
        profiles: dict[str, Any] = {}
        for level in VIP_LEVELS:
            profile = f"VIP{level}"
            metrics, _ = _evaluate_economic_policy(
                fold_test, experts, vip_level=level
            )
            fold_metrics[profile].append(metrics)
            profiles[profile] = _public_metrics(metrics) | {
                "stress_2x": _public_metrics(metrics["stress_2x"])
            }
        fold_results.append(
            {
                "fold": fold_number,
                "fit_start": fit_start.isoformat(),
                "fit_end": calibration_start.isoformat(),
                "calibration_end": fold_start.isoformat(),
                "test_end": test_end.isoformat(),
                "test_events": len(fold_test),
                "profiles": profiles,
            }
        )
        fold_start = test_end
    if not fold_results:
        raise ValueError("No post-selection OOS walk-forward fold is available")

    vip_policy_audit = {
        profile: _aggregate_fold_metrics(metrics)
        for profile, metrics in fold_metrics.items()
    }
    vip_gates = {
        profile: _economic_gates(metrics)
        for profile, metrics in vip_policy_audit.items()
    }
    paper_eligible_profiles = [
        profile for profile, gates in vip_gates.items() if all(gates.values())
    ]
    verdict = (
        "RESEARCH_ALPHA_READY"
        if paper_eligible_profiles
        else "NO_ECONOMIC_ALPHA"
    )

    final_calibration_start = holdout_start - pd.Timedelta(weeks=4)
    final_fit_start = final_calibration_start - pd.Timedelta(weeks=52)
    final_fit = _time_window(research, final_fit_start, final_calibration_start)
    final_calibration = _time_window(
        research, final_calibration_start, holdout_start
    )
    _status("alpha_final_fit", "Refit recente senza aprire holdout", 88)
    final_experts = _fit_expert_set(
        champions,
        final_fit,
        final_calibration,
        progress=("alpha_final_fit", "Refit recente", 88),
    )
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "features": FEATURES,
        "experts": final_experts,
        "champions": champions,
        "trained_through": pd.Timestamp(final_fit["signal_at"].max()).isoformat(),
        "calibrated_through": pd.Timestamp(
            final_calibration["signal_at"].max()
        ).isoformat(),
        "selected_through": discovery_selection_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "verdict": verdict,
        "paper_eligible_profiles": paper_eligible_profiles,
        "deployable_profiles": paper_eligible_profiles,
        "research_only": True,
        "live_orders_enabled": False,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(BUNDLE)

    event_counts = {
        key: int(_local_expert_keys(matrix).eq(key).sum()) for key in LOCAL_EXPERTS
    }
    report = {
        "status": verdict,
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "source_rows": len(pd.read_parquet(SOURCE, columns=["timestamp"])),
        "events": len(matrix),
        "events_by_local_expert": event_counts,
        "same_minute_target_stop_rate": {
            f"{barrier}bps": float(
                matrix[f"same_minute_target_stop_{barrier}bps"].mean()
            )
            for barrier in BARRIERS
        },
        "discovery": {
            "fit_events": len(discovery_fit),
            "calibration_events": len(discovery_calibration),
            "selection_events": len(discovery_selection),
            "selection_end": discovery_selection_end.isoformat(),
        },
        "champions": champions,
        "challenger_selection": challenger_selection,
        "walk_forward": {
            "spec": "trailing 52w fit, next 4w calibration, next 4w OOS, step 4w",
            "folds": fold_results,
            "fold_count": len(fold_results),
            "post_selection_only": True,
        },
        "vip_policy_audit": vip_policy_audit,
        "economic_policy_audit": vip_policy_audit["VIP0"],
        "vip_gates": vip_gates,
        "gates": vip_gates["VIP0"],
        "paper_eligible_profiles": paper_eligible_profiles,
        "deployable_profiles": paper_eligible_profiles,
        "sealed_holdout_events": int(times.ge(holdout_start).sum()),
        "holdout_start": holdout_start.isoformat(),
        "holdout_opened": False,
        "audit_used_for_model_selection": False,
        "historical_independence": (
            "DISCOVERY_CONTAMINATED_BY_PRIOR_VERSIONS; only the sealed future holdout "
            "or subsequently collected paper data can provide final confirmation"
        ),
        "execution_cost_contract": {
            "historical": (
                "official Bitunix VIP taker fees plus preregistered slippage reserve; "
                "historical Bitunix funding and depth are unavailable and not imputed"
            ),
            "live_shadow": (
                "observed Bitunix bid/ask depth, book walk, VIP fee and funding"
            ),
            "binance_funding": "Alpha context/diagnostic only, never charged as Bitunix",
        },
        "final_feature_support": {
            key: policy["feature_support"] for key, policy in final_experts.items()
        },
        "bundle": str(BUNDLE),
        "research_only": True,
        "real_capital_allowed": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT, report)
    _status("alpha_complete", verdict, 100)
    return report


def live_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    side = np.where(rows["side"].astype(str).eq("LONG"), 1.0, -1.0)
    available = pd.to_datetime(rows["signal_at"], utc=True)
    result["directional_return_1m_bps"] = side * rows["alpha_return_1m_bps"].to_numpy(float)
    result["directional_return_2m_bps"] = side * rows["alpha_return_2m_bps"].to_numpy(float)
    result["directional_return_3m_bps"] = side * rows["alpha_return_3m_bps"].to_numpy(float)
    result["directional_return_5m_bps"] = side * rows["alpha_return_5m_bps"].to_numpy(float)
    result["directional_return_10m_bps"] = side * rows["alpha_return_10m_bps"].to_numpy(float)
    result["directional_return_15m_bps"] = side * rows["alpha_return_15m_bps"].to_numpy(float)
    result["directional_return_30m_bps"] = side * rows["alpha_return_30m_bps"].to_numpy(float)
    result["directional_return_60m_bps"] = side * rows["alpha_return_60m_bps"].to_numpy(float)
    result["directional_vwap_distance_bps"] = side * rows["alpha_vwap_distance_bps"].to_numpy(
        float
    )
    result["absolute_vwap_distance_bps"] = (
        rows["alpha_vwap_distance_bps"].abs().to_numpy(float)
    )
    for window in (5, 15, 240):
        result[f"directional_vwap_distance_{window}m_bps"] = side * rows[
            f"alpha_vwap_distance_{window}m_bps"
        ].to_numpy(float)
    result["directional_vwap_slope_bps"] = side * rows["alpha_vwap_slope_bps"].to_numpy(float)
    result["directional_vwap_slope_change_bps"] = side * rows[
        "alpha_vwap_slope_change_bps"
    ].to_numpy(float)
    result["directional_vwap_distance_velocity_3m_bps"] = side * rows[
        "alpha_vwap_distance_velocity_3m_bps"
    ].to_numpy(float)
    result["vwap_tests_30m"] = rows["alpha_vwap_tests_30m"].to_numpy(float)
    result["vwap_rejections_30m"] = rows["alpha_vwap_rejections_30m"].to_numpy(float)
    result["time_since_vwap_cross_minutes"] = rows[
        "alpha_time_since_vwap_cross_minutes"
    ].to_numpy(float)
    result["directional_vwap_rejection_strength_bps"] = side * rows[
        "alpha_vwap_rejection_strength_bps"
    ].to_numpy(float)
    result["directional_vwap_band_position"] = side * rows[
        "alpha_vwap_band_position"
    ].to_numpy(float)
    result["structural_stop_bps"] = rows["stop_bps"].to_numpy(float)
    result["range_60s_bps"] = rows["alpha_range_60s_bps"].to_numpy(float)
    for timeframe in (1, 5, 15, 30):
        result[f"atr_{timeframe}m_bps"] = rows[
            f"alpha_atr_{timeframe}m_bps"
        ].to_numpy(float)
    result["realized_volatility_30m_bps"] = rows[
        "alpha_realized_volatility_30m_bps"
    ].to_numpy(float)
    result["volatility_percentile"] = rows["alpha_volatility_percentile"].to_numpy(float)
    result["volume_percentile"] = rows["alpha_volume_percentile"].to_numpy(float)
    result["directional_taker_imbalance_60s"] = side * rows[
        "alpha_taker_imbalance_60s"
    ].to_numpy(
        float
    )
    for window in (15, 60):
        result[f"directional_taker_imbalance_{window}m"] = side * rows[
            f"alpha_taker_imbalance_{window}m"
        ].to_numpy(float)
    result["directional_taker_imbalance_change_5m"] = side * rows[
        "alpha_taker_imbalance_change_5m"
    ].to_numpy(float)
    result["trade_count_zscore"] = rows["alpha_trade_count_zscore"].to_numpy(float)
    result["aggressive_volume_zscore"] = rows["alpha_aggressive_volume_zscore"].to_numpy(float)
    result["directional_candle_body_bps"] = side * rows[
        "alpha_candle_body_bps"
    ].to_numpy(float)
    result["directional_wick_imbalance_bps"] = side * rows[
        "alpha_wick_imbalance_bps"
    ].to_numpy(float)
    for window in (15, 60):
        result[f"efficiency_{window}m"] = rows[f"alpha_efficiency_{window}m"].to_numpy(float)
        result[f"directional_range_position_{window}m"] = side * rows[
            f"alpha_range_position_{window}m"
        ].to_numpy(float)
    result["directional_oi_change_1h"] = side * rows["oi_change_1h_binance"].to_numpy(float)
    result["return_oi_interaction"] = (
        rows["alpha_return_5m_bps"].to_numpy(float)
        * rows["oi_change_1h_binance"].to_numpy(float)
    )
    result["directional_basis_bps"] = side * rows["basis_bps_binance"].to_numpy(float)
    result["directional_funding_z"] = side * rows["funding_z_binance"].to_numpy(float)
    result["hour_sin"] = np.sin(2 * np.pi * available.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * available.dt.hour / 24)
    return result


def _feature_support(rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        feature: (
            float(rows[feature].quantile(0.0001)),
            float(rows[feature].quantile(0.9999)),
        )
        for feature in FEATURES
    }


def _support_violations(
    rows: pd.DataFrame, support: dict[str, tuple[float, float]]
) -> pd.Series:
    def violations(row: pd.Series) -> str:
        return ",".join(
            feature
            for feature in FEATURES
            if feature not in support
            or float(row[feature]) < float(support[feature][0])
            or float(row[feature]) > float(support[feature][1])
        )

    return rows.apply(violations, axis=1)


@lru_cache(maxsize=2)
def _load_scoring_bundle(path: str, modified_ns: int, size: int) -> Any:
    """Load a frozen bundle once and invalidate it after an atomic replacement."""
    del modified_ns, size
    return joblib.load(path)


def score(rows: pd.DataFrame) -> pd.DataFrame:
    scored = rows.copy()
    if scored.empty or not BUNDLE.exists() or not LIVE_REQUIRED.issubset(scored.columns):
        scored["alpha_accepted"] = False
        scored["alpha_status"] = "MISSING_FEATURE_COVERAGE"
        return scored
    try:
        stat = BUNDLE.stat()
        bundle = _load_scoring_bundle(
            str(BUNDLE.resolve()), stat.st_mtime_ns, stat.st_size
        )
    except (OSError, ValueError, TypeError, AttributeError, ImportError, EOFError):
        scored["alpha_accepted"] = False
        scored["alpha_status"] = "MODEL_BUNDLE_INVALID"
        return scored
    if bundle.get("protocol_hash") != PROTOCOL_HASH:
        scored["alpha_accepted"] = False
        scored["alpha_status"] = "MODEL_PROTOCOL_MISMATCH"
        return scored
    if (
        tuple(bundle.get("features", ())) != FEATURES
        or "experts" not in bundle
        or set(bundle["experts"]) != set(LOCAL_EXPERTS)
    ):
        scored["alpha_accepted"] = False
        scored["alpha_status"] = "MODEL_FEATURE_CONTRACT_MISMATCH"
        return scored
    features = live_features(scored)
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    contract_valid = scored["alpha_feature_contract_valid"].fillna(False).astype(bool).to_numpy()
    valid = finite & contract_valid
    scored["alpha_accepted"] = False
    scored["alpha_status"] = "MISSING_FEATURE_COVERAGE"
    scored.loc[finite & ~contract_valid, "alpha_status"] = "FEATURE_CONTRACT_INCOMPLETE"
    if not valid.any():
        return scored
    valid_index = scored.index[valid]
    expert_keys = _local_expert_keys(scored.loc[valid_index])
    unknown = ~expert_keys.isin(bundle["experts"])
    if unknown.any():
        scored.loc[expert_keys.index[unknown], "alpha_status"] = (
            "MODEL_LOCAL_EXPERT_MISMATCH"
        )
    violations = pd.Series("", index=valid_index, dtype="object")
    for key, policy in bundle["experts"].items():
        local_index = expert_keys.index[expert_keys.eq(key)]
        if len(local_index):
            violations.loc[local_index] = _support_violations(
                features.loc[local_index], policy["feature_support"]
            )
    violations.loc[expert_keys.index[unknown]] = "local_expert"
    scored.loc[violations.index, "alpha_ood_features"] = violations
    supported_index = violations.index[violations.eq("")]
    rejected_index = violations.index[violations.ne("")]
    scored.loc[rejected_index, "alpha_status"] = "FEATURE_DISTRIBUTION_MISMATCH"
    if supported_index.empty:
        return scored
    subset = features.loc[supported_index].copy()
    subset["expert"] = scored.loc[supported_index, "expert"]
    subset["side"] = scored.loc[supported_index, "side"]
    predictions = _local_predictions(bundle["experts"], subset)
    valid_index = supported_index
    stop = scored.loc[valid_index, "stop_bps"].to_numpy(float)
    _bound_physical_predictions(predictions, stop)
    for column, values in predictions.items():
        scored.loc[valid_index, column] = values
    vip_evs: dict[int, np.ndarray] = {}
    target_evs: dict[int, pd.DataFrame] = {}
    expected_non_fee = scored.loc[valid_index].get("expected_non_fee_cost_bps")
    observed_non_fee: np.ndarray = (
        np.maximum(0.0, expected_non_fee.to_numpy(float))
        if expected_non_fee is not None
        else np.maximum(
            0.0,
            scored.loc[valid_index].get(
                "expected_total_cost_bps",
                pd.Series(13.0, index=valid_index),
            ).to_numpy(float)
            - 2 * FUTURES_VIP_FEE_BPS[0][1],
        )
    )
    probability_gross, local_lcb = _local_calibrated_gross(
        subset, bundle["experts"], predictions, stop
    )
    for level in VIP_LEVELS:
        cost = 2 * FUTURES_VIP_FEE_BPS[level][1] + observed_non_fee
        evs = pd.DataFrame(
            {
                barrier: probability_gross[barrier] - cost
                for barrier in BARRIERS
            },
            index=valid_index,
        )
        target_evs[level] = evs
        vip_evs[level] = evs.max(axis=1).to_numpy(float)
        scored.loc[valid_index, f"alpha_ev_vip{level}_bps"] = vip_evs[level]
    mfe = scored.loc[valid_index, "expected_mfe_60m_bps"].to_numpy(float)
    for level in VIP_LEVELS:
        targets = target_evs[level].idxmax(axis=1).astype(int)
        target_values = targets.to_numpy(dtype=int)
        target_probability = np.asarray(
            [
                predictions[f"p_target_{target}bps_before_stop"][position]
                for position, target in enumerate(target_values)
            ],
            dtype=float,
        )
        stop_probability = np.asarray(
            [
                predictions[f"p_stop_before_{target}bps"][position]
                for position, target in enumerate(target_values)
            ],
            dtype=float,
        )
        probability_total = np.maximum(1.0, target_probability + stop_probability)
        target_probability /= probability_total
        stop_probability /= probability_total
        timeout_probability = np.clip(1 - target_probability - stop_probability, 0, 1)
        expected_time = np.asarray(
            [
                predictions[f"expected_time_to_{target}bps_minutes"][position]
                for position, target in enumerate(target_values)
            ],
            dtype=float,
        )
        gross = np.asarray(
            [
                float(probability_gross[int(target)][position])
                for position, target in enumerate(target_values)
            ],
            dtype=np.float64,
        )
        cost = (
            2 * FUTURES_VIP_FEE_BPS[level][1] + observed_non_fee
        )
        net = gross - cost
        prudent = net + np.asarray(
            [
                float(local_lcb[int(target)][position])
                for position, target in enumerate(target_values)
            ],
            dtype=np.float64,
        )
        accepted = (
            (vip_evs[level] > 0)
            & (net > 0)
            & (prudent > 0)
            & (mfe >= 3 * cost)
            & (target_values >= 3 * cost)
        )
        suffix = f"vip{level}"
        scored.loc[valid_index, f"alpha_target_{suffix}_bps"] = targets
        scored.loc[valid_index, f"alpha_target_probability_{suffix}"] = target_probability
        scored.loc[valid_index, f"alpha_stop_probability_{suffix}"] = stop_probability
        scored.loc[valid_index, f"alpha_timeout_probability_{suffix}"] = timeout_probability
        scored.loc[valid_index, f"alpha_expected_time_to_target_{suffix}_minutes"] = expected_time
        scored.loc[valid_index, f"alpha_expected_total_cost_{suffix}_bps"] = cost
        scored.loc[valid_index, f"alpha_expected_net_{suffix}_bps"] = net
        scored.loc[valid_index, f"alpha_prudent_net_{suffix}_bps"] = prudent
        scored.loc[valid_index, f"alpha_accepted_{suffix}"] = accepted
        scored.loc[valid_index, f"alpha_status_{suffix}"] = np.where(
            accepted, "TRADE", "NO_TRADE_ECONOMIC_GATE"
        )

    # Backwards-compatible aliases keep every existing consumer on the conservative VIP0 path.
    vip0 = "vip0"
    for destination, source in (
        ("alpha_target_bps", f"alpha_target_{vip0}_bps"),
        ("alpha_target_probability", f"alpha_target_probability_{vip0}"),
        ("alpha_stop_probability", f"alpha_stop_probability_{vip0}"),
        ("alpha_timeout_probability", f"alpha_timeout_probability_{vip0}"),
        (
            "alpha_expected_time_to_target_minutes",
            f"alpha_expected_time_to_target_{vip0}_minutes",
        ),
        ("alpha_expected_total_cost_bps", f"alpha_expected_total_cost_{vip0}_bps"),
        ("alpha_expected_net_bps", f"alpha_expected_net_{vip0}_bps"),
        ("alpha_prudent_net_bps", f"alpha_prudent_net_{vip0}_bps"),
        ("alpha_accepted", f"alpha_accepted_{vip0}"),
        ("alpha_status", f"alpha_status_{vip0}"),
    ):
        scored.loc[valid_index, destination] = scored.loc[valid_index, source]
    return scored


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
