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
        if not snapshot.data_reliable or snapshot.regime is MarketRegime.SHOCK:
            return None
        if snapshot.spread_bps > self.config.max_spread_bps:
            return None
        score = self.entry_score(snapshot, state)
        if self.config.entry_mode == "range" and snapshot.regime is not MarketRegime.RANGE:
            return None
        if self.config.entry_mode == "weighted_reversion_v11" and (
            (snapshot.z_score > 0 and snapshot.regime is MarketRegime.TREND_UP)
            or (snapshot.z_score < 0 and snapshot.regime is MarketRegime.TREND_DOWN)
        ):
            return None
        if self.config.entry_mode.startswith("weighted_reversion") and (
            score is None or score < self.config.weighted_entry_threshold
        ):
            return None
        if snapshot.z_score <= -self.config.entry_z:
            stop, target = self._entry_levels(snapshot, Side.BUY)
            return self._signal(
                snapshot,
                SignalAction.ENTER_LONG,
                self._entry_reason("lower-band", score),
                stop,
                target,
            )
        if self.config.short_enabled and snapshot.z_score >= self.config.entry_z:
            stop, target = self._entry_levels(snapshot, Side.SELL)
            return self._signal(
                snapshot,
                SignalAction.ENTER_SHORT,
                self._entry_reason("upper-band", score),
                stop,
                target,
            )
        return None

    def entry_score(self, snapshot: MarketSnapshot, state: StrategyState) -> float | None:
        if self.config.entry_mode == "weighted_reversion_v2":
            return self._v2_entry_score(snapshot, state)
        if self.config.entry_mode not in {"weighted_reversion", "weighted_reversion_v11"}:
            return None
        if state.last_z is None:
            return None
        mean_reversion = min(abs(snapshot.z_score) / 3, 1)
        momentum = min(max(abs(state.last_z) - abs(snapshot.z_score), 0) / 0.5, 1)
        volatility = max(0.0, 1 - snapshot.atr_percentile / 90)
        liquidity = max(0.0, 1 - snapshot.spread_bps / self.config.max_spread_bps)
        return 0.35 * mean_reversion + 0.30 * momentum + 0.20 * volatility + 0.15 * liquidity

    def _v2_entry_score(self, snapshot: MarketSnapshot, state: StrategyState) -> float | None:
        if not self._v2_entry_ready(snapshot, state):
            return None
        distance = abs(snapshot.z_score)
        extension = max(0.0, 1 - abs(distance - 2.5) / 1.5)
        assert state.previous_z is not None
        momentum = min((abs(state.previous_z) - distance) / 1.5, 1)
        volatility = max(0.0, 1 - abs(snapshot.atr_percentile - 45) / 45)
        liquidity = max(0.0, 1 - snapshot.spread_bps / self.config.max_spread_bps)
        return 0.35 * extension + 0.30 * momentum + 0.20 * volatility + 0.15 * liquidity

    def _v2_entry_ready(self, snapshot: MarketSnapshot, state: StrategyState) -> bool:
        if (
            state.previous_z is None
            or state.last_z is None
            or state.last_close is None
            or not self.config.entry_z <= abs(snapshot.z_score) <= 4
        ):
            return False
        if snapshot.z_score > 0:
            return (
                self.config.short_enabled
                and snapshot.regime is not MarketRegime.TREND_UP
                and state.previous_z > state.last_z > snapshot.z_score
                and snapshot.candle.close < state.last_close
            )
        return (
            snapshot.regime is not MarketRegime.TREND_DOWN
            and state.previous_z < state.last_z < snapshot.z_score
            and snapshot.candle.close > state.last_close
        )

    def _entry_reason(self, band: str, score: float | None) -> str:
        if self.config.entry_mode.startswith("weighted_reversion"):
            assert score is not None
            version = (
                " v1.1"
                if self.config.entry_mode.endswith("_v11")
                else " v2"
                if self.config.entry_mode.endswith("_v2")
                else ""
            )
            return f"weighted mean-reversion{version} {band} entry (score {score:.3f})"
        return f"range {band} entry"

    def _entry_levels(self, snapshot: MarketSnapshot, side: Side) -> tuple[Decimal, Decimal]:
        entry = snapshot.candle.close
        if self.config.fixed_stop_fraction is None:
            stop = initial_stop(entry, snapshot.atr, side, Decimal(str(self.config.stop_atr)))
            return stop, snapshot.center
        stop_distance = entry * self.config.fixed_stop_fraction
        assert self.config.fixed_target_fraction is not None
        target_distance = entry * self.config.fixed_target_fraction
        if side is Side.BUY:
            return entry - stop_distance, entry + target_distance
        return entry + stop_distance, entry - target_distance

    def _entry_window(self, snapshot: MarketSnapshot) -> bool:
        if not self.config.session_flatten_enabled:
            return True
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
        if (
            self.config.entry_mode in {"weighted_reversion_v11", "weighted_reversion_v2"}
            and state.previous_z is not None
            and state.last_z is not None
            and (
                (position.side is Side.SELL and state.previous_z < state.last_z < snapshot.z_score)
                or (
                    position.side is Side.BUY and state.previous_z > state.last_z > snapshot.z_score
                )
            )
        ):
            return "mean reversion invalidated"
        if self.config.entry_mode == "weighted_reversion_v11" and (
            (position.side is Side.SELL and snapshot.z_score <= self.config.v11_exit_z)
            or (position.side is Side.BUY and snapshot.z_score >= -self.config.v11_exit_z)
        ):
            return "VWAP approach"
        if self.config.session_flatten_enabled and snapshot.candle.exchange_timestamp >= flatten:
            return "session flatten"
        if position.bars_held >= self.config.time_stop_bars:
            return "time stop"
        if (
            self.config.fixed_target_fraction is None
            and position.side is Side.BUY
            and snapshot.z_score >= 0
        ):
            return "center reached"
        if (
            self.config.fixed_target_fraction is None
            and position.side is Side.SELL
            and snapshot.z_score <= 0
        ):
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
        target: Decimal | None = None,
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
            target_price=target or snapshot.center,
            z_score=snapshot.z_score,
            regime=snapshot.regime,
            reason=reason,
        )
