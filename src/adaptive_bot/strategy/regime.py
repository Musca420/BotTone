from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_bot.config import StrategyConfig
from adaptive_bot.domain.enums import MarketRegime


@dataclass(frozen=True)
class RegimeFeatures:
    adx: float
    atr_percentile: float
    ema_slope: float
    atr_change: float
    spread_bps: float
    missing_ratio: float
    cumulative_move: float


@dataclass(frozen=True)
class RegimeResult:
    regime: MarketRegime
    pending: MarketRegime | None
    pending_count: int


class RegimeClassifier:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def raw(self, features: RegimeFeatures) -> MarketRegime:
        values = (
            features.adx,
            features.atr_percentile,
            features.ema_slope,
            features.atr_change,
            features.spread_bps,
            features.missing_ratio,
            features.cumulative_move,
        )
        if not all(math.isfinite(value) for value in values):
            return MarketRegime.UNKNOWN
        if features.missing_ratio > 0 or features.spread_bps > self.config.max_spread_bps:
            return MarketRegime.UNKNOWN
        if (
            features.atr_percentile > 90
            or features.atr_change > self.config.atr_change_threshold
            or abs(features.cumulative_move) > self.config.cumulative_move_atr
        ):
            return MarketRegime.SHOCK
        if features.adx < self.config.range_adx_threshold:
            return MarketRegime.RANGE
        if features.adx > self.config.trend_adx_threshold:
            if features.ema_slope >= self.config.slope_threshold and features.cumulative_move >= 0:
                return MarketRegime.TREND_UP
            if features.ema_slope <= -self.config.slope_threshold and features.cumulative_move <= 0:
                return MarketRegime.TREND_DOWN
        return MarketRegime.UNKNOWN

    def update(
        self,
        features: RegimeFeatures,
        current: MarketRegime,
        pending: MarketRegime | None,
        pending_count: int,
    ) -> RegimeResult:
        candidate = self.raw(features)
        if candidate is MarketRegime.SHOCK:
            return RegimeResult(candidate, None, 0)
        if candidate is current:
            return RegimeResult(current, None, 0)
        count = pending_count + 1 if candidate is pending else 1
        if count >= self.config.confirmation_bars:
            return RegimeResult(candidate, None, 0)
        return RegimeResult(current, candidate, count)
