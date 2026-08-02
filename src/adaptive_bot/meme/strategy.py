from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

import pandas as pd

from adaptive_bot.domain.enums import MarketRegime, Side, SignalAction
from adaptive_bot.domain.models import Position, Signal
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.vwap import rolling_vwap
from adaptive_bot.meme.config import MemeStrategyConfig

MemeAction = Literal["hold", "watch", "reject", "enter_long", "enter_short", "reduce", "exit"]


class MemeRegime(StrEnum):
    BULLISH_EXPANSION = "bullish_expansion"
    EUPHORIC_PUMP = "euphoric_pump"
    SIDEWAYS = "sideways"
    DISTRIBUTION = "distribution"
    BEARISH_EXPANSION = "bearish_expansion"
    PANIC_CRASH = "panic_crash"
    ILLIQUID = "illiquid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemeDecision:
    timestamp: datetime
    symbol: str
    action: MemeAction
    reason: str
    reference_price: Decimal
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    strategy_name: str = "breakout_retest"
    meme_regime: MemeRegime = MemeRegime.UNKNOWN

    def signal(self) -> Signal | None:
        actions = {
            "enter_long": SignalAction.ENTER_LONG,
            "enter_short": SignalAction.ENTER_SHORT,
            "reduce": SignalAction.REDUCE,
            "exit": SignalAction.EXIT,
        }
        action = actions.get(self.action)
        if action is None:
            return None
        return Signal(
            exchange_timestamp=self.timestamp,
            received_timestamp=self.timestamp,
            source="meme_momentum",
            instrument=self.symbol,
            action=action,
            reference_price=self.reference_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            z_score=0,
            regime=self.regime,
            reason=self.reason,
        )


@dataclass(frozen=True)
class MemeStrategyState:
    pending_side: Side | None = None
    breakout_level: Decimal | None = None
    pending_bars: int = 0
    planned_stop: Decimal | None = None
    entry_price: Decimal | None = None
    initial_risk: Decimal | None = None
    one_r_hit: bool = False
    trailing_stop: Decimal | None = None
    best_price: Decimal | None = None


def build_meme_features(candles: pd.DataFrame, config: MemeStrategyConfig) -> pd.DataFrame:
    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column])
    atr_values = atr(frame["high"], frame["low"], frame["close"], config.atr_period)
    previous_volume = frame["volume"].shift(1)
    volume_mean = previous_volume.rolling(
        config.volume_window, min_periods=config.volume_window
    ).mean()
    volume_std = previous_volume.rolling(
        config.volume_window, min_periods=config.volume_window
    ).std(ddof=0)
    frame["atr"] = atr_values
    frame["breakout_high"] = frame["high"].shift(1).rolling(config.breakout_bars).max()
    frame["breakout_low"] = frame["low"].shift(1).rolling(config.breakout_bars).min()
    frame["volume_zscore"] = (frame["volume"] - volume_mean) / volume_std.replace(0, pd.NA)
    frame["range_atr"] = (frame["high"] - frame["low"]) / atr_values.replace(0, pd.NA)
    frame["three_bar_atr"] = (
        frame["close"].pct_change(3).abs() * frame["close"] / atr_values.replace(0, pd.NA)
    )
    frame["momentum_atr"] = (frame["close"] - frame["close"].shift(12)) / atr_values.replace(
        0, pd.NA
    )
    frame["adaptive_center"] = rolling_vwap(
        frame["high"],
        frame["low"],
        frame["close"],
        frame["volume"],
        config.adaptive_vwap_window,
    )
    frame["adaptive_z"] = (frame["close"] - frame["adaptive_center"]) / atr_values.replace(0, pd.NA)
    frame["ema_fast_5m"] = frame["close"].ewm(span=config.fast_ema, adjust=False).mean()
    frame["previous_close"] = frame["close"].shift(1)

    indexed = frame.set_index("timestamp")
    hourly = indexed.resample("1h", label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    hourly = hourly.dropna(subset=["open", "high", "low", "close"])
    hourly["ema_fast_1h"] = hourly["close"].ewm(span=config.fast_ema, adjust=False).mean()
    hourly["ema_slow_1h"] = hourly["close"].ewm(span=config.slow_ema, adjust=False).mean()
    hourly["ema_slope_1h"] = hourly["ema_fast_1h"].pct_change(3)
    hourly["adx_1h"] = adx(hourly["high"], hourly["low"], hourly["close"], config.adx_period)["adx"]
    context = hourly[["ema_fast_1h", "ema_slow_1h", "ema_slope_1h", "adx_1h"]]
    return pd.merge_asof(
        frame,
        context.reset_index(),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )


class MemeMomentumStrategy:
    def __init__(self, config: MemeStrategyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        row: pd.Series,
        symbol: str,
        state: MemeStrategyState,
        position: Position | None = None,
    ) -> tuple[MemeDecision, MemeStrategyState]:
        timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()
        close = Decimal(str(row["close"]))
        regime = self._regime(row)
        meme_regime = self._meme_regime(row)
        if position is not None:
            return self._manage_position(row, timestamp, symbol, close, regime, state, position)
        if self._missing(row):
            return MemeDecision(timestamp, symbol, "reject", "indicators_warming_up", close), state
        if self._shock(row):
            return MemeDecision(
                timestamp,
                symbol,
                "reject",
                "panic_or_vertical_candle",
                close,
                regime=MarketRegime.SHOCK,
                meme_regime=meme_regime,
            ), MemeStrategyState()

        atr_value = Decimal(str(row["atr"]))
        if state.pending_side is not None:
            return self._evaluate_retest(row, timestamp, symbol, close, atr_value, regime, state)

        adaptive_z = Decimal(str(row["adaptive_z"]))
        if (
            self.config.adaptive_range_enabled
            and meme_regime is MemeRegime.SIDEWAYS
            and abs(adaptive_z) >= self.config.adaptive_entry_z
        ):
            center = Decimal(str(row["adaptive_center"]))
            stop_distance = atr_value * self.config.adaptive_stop_atr
            side = Side.BUY if adaptive_z < 0 else Side.SELL
            stop = close - stop_distance if side is Side.BUY else close + stop_distance
            if abs(center - close) / stop_distance >= self.config.aggressive_reward_risk:
                return MemeDecision(
                    timestamp,
                    symbol,
                    "enter_long" if side is Side.BUY else "enter_short",
                    "adaptive_range_mean_reversion",
                    close,
                    stop,
                    center,
                    regime,
                    "adaptive_range",
                    meme_regime,
                ), MemeStrategyState(
                    planned_stop=stop,
                    entry_price=close,
                    initial_risk=stop_distance,
                    best_price=close,
                )

        volume_ok = Decimal(str(row["volume_zscore"])) >= self.config.minimum_volume_zscore
        momentum = Decimal(str(row["momentum_atr"]))
        if (
            volume_ok
            and meme_regime in {MemeRegime.BULLISH_EXPANSION, MemeRegime.EUPHORIC_PUMP}
            and momentum >= self.config.minimum_momentum_atr
            and close > Decimal(str(row["breakout_high"]))
        ):
            next_state = MemeStrategyState(Side.BUY, Decimal(str(row["breakout_high"])), 0)
            reason = (
                "aggressive_long_waiting_external_context_and_consolidation"
                if momentum >= self.config.aggressive_momentum_atr
                else "long_breakout_waiting_retest"
            )
            return MemeDecision(
                timestamp,
                symbol,
                "watch",
                reason,
                close,
                regime=regime,
                meme_regime=meme_regime,
            ), next_state
        if (
            self.config.short_enabled
            and volume_ok
            and meme_regime in {MemeRegime.DISTRIBUTION, MemeRegime.BEARISH_EXPANSION}
            and momentum <= -self.config.minimum_momentum_atr
            and close < Decimal(str(row["breakout_low"]))
        ):
            next_state = MemeStrategyState(Side.SELL, Decimal(str(row["breakout_low"])), 0)
            return MemeDecision(
                timestamp, symbol, "watch", "short_breakout_waiting_retest", close, regime=regime
            ), next_state
        if (
            meme_regime is MemeRegime.BULLISH_EXPANSION
            and Decimal(str(row["volume_zscore"])) >= self.config.minimum_volume_zscore
            and Decimal(str(row["low"])) <= Decimal(str(row["ema_fast_5m"]))
            and close > Decimal(str(row["ema_fast_5m"]))
            and close > Decimal(str(row["previous_close"]))
        ):
            atr_value = Decimal(str(row["atr"]))
            stop = Decimal(str(row["low"])) - atr_value * self.config.stop_buffer_atr
            risk = close - stop
            if 0 < risk <= atr_value * self.config.maximum_stop_atr:
                return MemeDecision(
                    timestamp,
                    symbol,
                    "enter_long",
                    "momentum_pullback_confirmed",
                    close,
                    stop,
                    close + risk * self.config.confirmed_reward_risk,
                    regime,
                    "momentum_pullback",
                    meme_regime,
                ), MemeStrategyState(
                    planned_stop=stop,
                    entry_price=close,
                    initial_risk=risk,
                    best_price=close,
                )
        return MemeDecision(timestamp, symbol, "hold", "no_breakout", close, regime=regime), state

    def _evaluate_retest(
        self,
        row: pd.Series,
        timestamp: datetime,
        symbol: str,
        close: Decimal,
        atr_value: Decimal,
        regime: MarketRegime,
        state: MemeStrategyState,
    ) -> tuple[MemeDecision, MemeStrategyState]:
        assert state.breakout_level is not None and state.pending_side is not None
        bars = state.pending_bars + 1
        if bars > self.config.retest_bars or self._shock(row):
            return MemeDecision(
                timestamp, symbol, "reject", "retest_expired_or_shock", close, regime=regime
            ), MemeStrategyState()
        tolerance = atr_value * self.config.retest_tolerance_atr
        low = Decimal(str(row["low"]))
        high = Decimal(str(row["high"]))
        if state.pending_side is Side.BUY:
            confirmed = (
                state.breakout_level - tolerance <= low <= state.breakout_level + tolerance
                and close > state.breakout_level
            )
            stop = low - atr_value * self.config.stop_buffer_atr
            action: MemeAction = "enter_long"
        else:
            confirmed = (
                state.breakout_level - tolerance <= high <= state.breakout_level + tolerance
                and close < state.breakout_level
            )
            stop = high + atr_value * self.config.stop_buffer_atr
            action = "enter_short"
        if not confirmed:
            return MemeDecision(
                timestamp, symbol, "watch", "waiting_retest_confirmation", close, regime=regime
            ), replace(state, pending_bars=bars)
        risk = abs(close - stop)
        if risk <= 0 or risk > atr_value * self.config.maximum_stop_atr:
            return MemeDecision(
                timestamp, symbol, "reject", "invalid_stop_distance", close, regime=regime
            ), MemeStrategyState()
        reward_ratio = (
            self.config.confirmed_reward_risk
            if state.pending_side is Side.BUY
            else self.config.short_reward_risk
        )
        reward = risk * reward_ratio
        target = close + reward if state.pending_side is Side.BUY else close - reward
        next_state = MemeStrategyState(
            planned_stop=stop,
            entry_price=close,
            initial_risk=risk,
            best_price=close,
        )
        return MemeDecision(
            timestamp,
            symbol,
            action,
            "breakout_retest_confirmed",
            close,
            stop,
            target,
            regime,
            meme_regime=self._meme_regime(row),
        ), next_state

    def _manage_position(
        self,
        row: pd.Series,
        timestamp: datetime,
        symbol: str,
        close: Decimal,
        regime: MarketRegime,
        state: MemeStrategyState,
        position: Position,
    ) -> tuple[MemeDecision, MemeStrategyState]:
        if self._shock(row):
            return MemeDecision(
                timestamp, symbol, "exit", "shock_exit", close, regime=MarketRegime.SHOCK
            ), state
        if position.bars_held >= self.config.time_stop_bars:
            return MemeDecision(timestamp, symbol, "exit", "time_stop", close, regime=regime), state
        atr_value = Decimal(str(row["atr"]))
        entry = state.entry_price or position.average_entry_price
        risk = state.initial_risk or abs(entry - (state.planned_stop or entry))
        if risk <= 0:
            return MemeDecision(
                timestamp, symbol, "exit", "missing_protective_stop", close, regime=regime
            ), state
        high = Decimal(str(row["high"]))
        low = Decimal(str(row["low"]))
        if position.side is Side.BUY:
            best = max(state.best_price or entry, high)
            hit = high >= entry + risk
            trailing = max(entry, best - atr_value * self.config.trailing_atr)
        else:
            best = min(state.best_price or entry, low)
            hit = low <= entry - risk
            trailing = min(entry, best + atr_value * self.config.trailing_atr)
        updated = replace(state, best_price=best)
        if not state.one_r_hit and hit:
            updated = replace(updated, one_r_hit=True, trailing_stop=entry)
            return MemeDecision(
                timestamp, symbol, "reduce", "one_r_partial", close, entry, regime=regime
            ), updated
        if state.one_r_hit:
            previous = state.trailing_stop or entry
            tightened = (
                max(previous, trailing) if position.side is Side.BUY else min(previous, trailing)
            )
            updated = replace(updated, trailing_stop=tightened)
        return MemeDecision(
            timestamp,
            symbol,
            "hold",
            "position_managed",
            close,
            updated.trailing_stop,
            regime=regime,
        ), updated

    def _regime(self, row: pd.Series) -> MarketRegime:
        meme_regime = self._meme_regime(row)
        if meme_regime in {MemeRegime.EUPHORIC_PUMP, MemeRegime.BULLISH_EXPANSION}:
            return MarketRegime.TREND_UP
        if meme_regime in {MemeRegime.DISTRIBUTION, MemeRegime.BEARISH_EXPANSION}:
            return MarketRegime.TREND_DOWN
        if meme_regime is MemeRegime.SIDEWAYS:
            return MarketRegime.RANGE
        if meme_regime is MemeRegime.PANIC_CRASH:
            return MarketRegime.SHOCK
        return MarketRegime.UNKNOWN

    def _meme_regime(self, row: pd.Series) -> MemeRegime:
        if self._missing(row):
            return MemeRegime.UNKNOWN
        momentum = Decimal(str(row["momentum_atr"]))
        if self._shock(row):
            return MemeRegime.PANIC_CRASH if momentum < 0 else MemeRegime.EUPHORIC_PUMP
        adx_value = Decimal(str(row["adx_1h"]))
        if adx_value < self.config.minimum_adx:
            return MemeRegime.SIDEWAYS
        fast = Decimal(str(row["ema_fast_1h"]))
        slow = Decimal(str(row["ema_slow_1h"]))
        slope = Decimal(str(row["ema_slope_1h"]))
        if fast > slow and slope > 0:
            return (
                MemeRegime.EUPHORIC_PUMP
                if momentum >= self.config.aggressive_momentum_atr
                else MemeRegime.BULLISH_EXPANSION
            )
        if fast > slow and slope <= 0:
            return MemeRegime.DISTRIBUTION
        if fast < slow and slope < 0:
            return MemeRegime.BEARISH_EXPANSION
        return MemeRegime.DISTRIBUTION

    def _shock(self, row: pd.Series) -> bool:
        return (
            Decimal(str(row["range_atr"])) > self.config.shock_candle_atr
            or Decimal(str(row["three_bar_atr"])) > self.config.shock_three_bar_atr
        )

    @staticmethod
    def _missing(row: pd.Series) -> bool:
        names = (
            "atr",
            "breakout_high",
            "breakout_low",
            "volume_zscore",
            "range_atr",
            "three_bar_atr",
            "momentum_atr",
            "adaptive_center",
            "adaptive_z",
            "ema_fast_1h",
            "ema_slow_1h",
            "ema_slope_1h",
            "adx_1h",
            "ema_fast_5m",
            "previous_close",
        )
        return bool(row[list(names)].isna().any())


def triple_barrier_label(
    candles: pd.DataFrame,
    entry_index: int,
    side: Side,
    entry: Decimal,
    stop: Decimal,
    target: Decimal,
    horizon: int,
) -> int:
    if horizon <= 0 or entry_index < 0 or entry_index >= len(candles):
        raise ValueError("invalid triple-barrier window")
    future = candles.iloc[entry_index + 1 : entry_index + 1 + horizon]
    for _, row in future.iterrows():
        high, low = Decimal(str(row["high"])), Decimal(str(row["low"]))
        stop_hit = low <= stop if side is Side.BUY else high >= stop
        target_hit = high >= target if side is Side.BUY else low <= target
        if stop_hit:
            return -1
        if target_hit:
            return 1
    return 0
