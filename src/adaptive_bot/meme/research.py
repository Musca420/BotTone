from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

import pandas as pd

from adaptive_bot.domain.enums import Side
from adaptive_bot.meme.config import MemeBotConfig
from adaptive_bot.meme.strategy import (
    MemeMomentumStrategy,
    MemeStrategyState,
    build_meme_features,
    triple_barrier_label,
)
from adaptive_bot.meme.universe import MarketQuality


@dataclass(frozen=True)
class ShadowFeatureSnapshot:
    timestamp: str
    symbol: str
    side: str
    return_5m: float
    return_15m: float
    return_1h: float
    atr_normalized: float
    volume_zscore: float
    range_atr: float
    three_bar_atr: float
    ema_spread_1h: float
    ema_slope_1h: float
    adx_1h: float
    spread_bps: float
    depth_half_percent: float
    funding_8h: float
    mark_divergence: float
    market_quality_risk: float
    label: int


def build_shadow_dataset(
    frames: dict[str, pd.DataFrame],
    qualities: dict[str, MarketQuality],
    config: MemeBotConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategy = MemeMomentumStrategy(config.strategy)
    for symbol, candles in sorted(frames.items()):
        quality = qualities.get(symbol)
        if quality is None or quality.funding_8h is None:
            continue
        features = build_meme_features(candles, config.strategy)
        state = MemeStrategyState()
        for index, (_, row) in enumerate(features.iterrows()):
            decision, state = strategy.evaluate(row, symbol, state)
            if decision.action not in {"enter_long", "enter_short"}:
                continue
            assert decision.stop_price is not None and decision.target_price is not None
            side = Side.BUY if decision.action == "enter_long" else Side.SELL
            label = triple_barrier_label(
                features,
                index,
                side,
                decision.reference_price,
                decision.stop_price,
                decision.target_price,
                config.strategy.time_stop_bars,
            )
            close = float(row["close"])
            fast = float(row["ema_fast_1h"])
            slow = float(row["ema_slow_1h"])
            snapshot = ShadowFeatureSnapshot(
                timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
                symbol=symbol,
                side=side.value,
                return_5m=float(features["close"].pct_change().iloc[index]),
                return_15m=float(features["close"].pct_change(3).iloc[index]),
                return_1h=float(features["close"].pct_change(12).iloc[index]),
                atr_normalized=float(row["atr"]) / close,
                volume_zscore=float(row["volume_zscore"]),
                range_atr=float(row["range_atr"]),
                three_bar_atr=float(row["three_bar_atr"]),
                ema_spread_1h=(fast - slow) / close,
                ema_slope_1h=float(row["ema_slope_1h"]),
                adx_1h=float(row["adx_1h"]),
                spread_bps=float(quality.spread_bps),
                depth_half_percent=float(quality.depth_half_percent),
                funding_8h=float(quality.funding_8h),
                mark_divergence=float(quality.mark_divergence),
                market_quality_risk=market_quality_risk(quality, config),
                label=label,
            )
            rows.append(asdict(snapshot))
    return pd.DataFrame(rows)


def market_quality_risk(quality: MarketQuality, config: MemeBotConfig) -> float:
    spread = min(1.0, float(quality.spread_bps / config.universe.maximum_spread_bps))
    funding = (
        1.0
        if quality.funding_8h is None
        else min(1.0, float(abs(quality.funding_8h) / config.universe.maximum_funding_8h))
    )
    divergence = min(
        1.0, float(abs(quality.mark_divergence) / config.universe.maximum_mark_divergence)
    )
    stale = min(1.0, float(quality.stream_age_seconds / Decimal("5")))
    return round((spread + funding + divergence + stale) / 4, 6)


def shadow_status(dataset: pd.DataFrame) -> dict[str, object]:
    if dataset.empty:
        return {
            "status": "collecting_data",
            "can_trade": False,
            "samples": 0,
            "reason": "no_labeled_setups",
        }
    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    weeks = max(0.0, (timestamps.max() - timestamps.min()).total_seconds() / 604800)
    ready = len(dataset) >= 1000 and weeks >= 20 and dataset["symbol"].nunique() >= 10
    return {
        "status": "ready_for_training" if ready else "collecting_data",
        "can_trade": False,
        "samples": len(dataset),
        "weeks": round(weeks, 2),
        "symbols": int(dataset["symbol"].nunique()),
        "reason": "offline_training_gate_only" if ready else "minimum_dataset_not_reached",
    }
