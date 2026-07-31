from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pandas as pd

from adaptive_bot.config import StrategyConfig
from adaptive_bot.domain.enums import AssetClass, MarketRegime, Side, SignalAction
from adaptive_bot.domain.models import Position, Signal, StrategyState
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import (
    atr_change,
    atr_percentile,
    cumulative_move,
    normalized_distance,
)
from adaptive_bot.indicators.vwap import rolling_vwap, session_vwap
from adaptive_bot.strategy.signals import MarketSnapshot, initial_stop


def build_features(
    candles: pd.DataFrame,
    sessions: pd.Series,
    config: StrategyConfig,
    asset_class: AssetClass = AssetClass.EQUITY,
) -> pd.DataFrame:
    frame = candles.copy()
    atr_values = atr(frame["high"], frame["low"], frame["close"], config.atr_period)
    adx_values = adx(frame["high"], frame["low"], frame["close"], config.adx_period)
    if asset_class is AssetClass.CRYPTO:
        center = rolling_vwap(
            frame["high"],
            frame["low"],
            frame["close"],
            frame["volume"],
            config.crypto_vwap_window,
        )
    else:
        center = session_vwap(
            frame["high"], frame["low"], frame["close"], frame["volume"], sessions
        )
    frame["atr"] = atr_values
    frame["adx"] = adx_values["adx"]
    frame["center"] = center
    frame["z"] = normalized_distance(frame["close"], center, atr_values)
    frame["atr_percentile"] = atr_percentile(atr_values, config.atr_percentile_window)
    frame["atr_change"] = atr_change(atr_values)
    frame["ema_slope"] = normalized_ema_slope(
        frame["close"], atr_values, config.ema_period, config.slope_lookback
    )
    frame["cumulative_move"] = cumulative_move(frame["close"], atr_values)
    return frame


class AdaptiveRangeStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def evaluate(
        self, snapshot: MarketSnapshot, state: StrategyState, position: Position | None
    ) -> Signal | None:
        candle = snapshot.candle
        if position is not None:
            reason = self._exit_reason(snapshot, state, position)
            if reason is None:
                return None
            action = (
                SignalAction.REDUCE
                if self.config.partial_exit_enabled and abs(snapshot.z_score) <= 0.5
                else SignalAction.EXIT
            )
            return self._signal(snapshot, action, reason)

        if not self._entry_window(snapshot) or state.cooldown_bars > 0:
            return None
        if not snapshot.data_reliable or snapshot.regime is not MarketRegime.RANGE:
            return None
        if snapshot.spread_bps > self.config.max_spread_bps:
            return None
        if snapshot.z_score <= -self.config.entry_z:
            stop = initial_stop(
                candle.close, snapshot.atr, Side.BUY, Decimal(str(self.config.stop_atr))
            )
            return self._signal(snapshot, SignalAction.ENTER_LONG, "range lower-band entry", stop)
        if self.config.short_enabled and snapshot.z_score >= self.config.entry_z:
            stop = initial_stop(
                candle.close, snapshot.atr, Side.SELL, Decimal(str(self.config.stop_atr))
            )
            return self._signal(snapshot, SignalAction.ENTER_SHORT, "range upper-band entry", stop)
        return None

    def _entry_window(self, snapshot: MarketSnapshot) -> bool:
        opened = snapshot.session_open + timedelta(minutes=self.config.no_entry_minutes_after_open)
        flatten = snapshot.session_close - timedelta(
            minutes=self.config.flatten_minutes_before_close
        )
        return opened <= snapshot.candle.exchange_timestamp < flatten

    def _exit_reason(
        self, snapshot: MarketSnapshot, state: StrategyState, position: Position
    ) -> str | None:
        flatten = snapshot.session_close - timedelta(
            minutes=self.config.flatten_minutes_before_close
        )
        if snapshot.force_exit_reason is not None:
            return snapshot.force_exit_reason
        if not snapshot.data_reliable:
            return "data integrity compromised"
        if snapshot.regime is MarketRegime.SHOCK:
            return "shock regime"
        if snapshot.candle.exchange_timestamp >= flatten:
            return "session flatten"
        if position.bars_held >= self.config.time_stop_bars:
            return "time stop"
        if position.side is Side.BUY and snapshot.z_score >= 0:
            return "center reached"
        if position.side is Side.SELL and snapshot.z_score <= 0:
            return "center reached"
        if (
            self.config.partial_exit_enabled
            and abs(snapshot.z_score) <= 0.5
            and state.last_z is not None
            and abs(state.last_z) > 0.5
        ):
            return "partial center approach"
        return None

    @staticmethod
    def _signal(
        snapshot: MarketSnapshot,
        action: SignalAction,
        reason: str,
        stop: Decimal | None = None,
    ) -> Signal:
        candle = snapshot.candle
        return Signal(
            exchange_timestamp=candle.exchange_timestamp,
            received_timestamp=candle.received_timestamp,
            source="adaptive_range",
            instrument=candle.instrument,
            sequence_number=candle.sequence_number,
            correlation_id=candle.correlation_id,
            action=action,
            reference_price=candle.close,
            stop_price=stop,
            target_price=snapshot.center,
            z_score=snapshot.z_score,
            regime=snapshot.regime,
            reason=reason,
        )
