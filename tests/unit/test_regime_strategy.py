from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from adaptive_bot.config import StrategyConfig
from adaptive_bot.domain.enums import MarketRegime, Side, SignalAction
from adaptive_bot.domain.models import Position, StrategyState
from adaptive_bot.strategy.adaptive_range import AdaptiveRangeStrategy
from adaptive_bot.strategy.regime import RegimeClassifier, RegimeFeatures
from adaptive_bot.strategy.signals import MarketSnapshot, initial_stop, tighten_stop
from tests.conftest import candle


def features(**updates: float) -> RegimeFeatures:
    values = {
        "adx": 15.0,
        "atr_percentile": 50.0,
        "ema_slope": 0.0,
        "atr_change": 0.0,
        "spread_bps": 1.0,
        "missing_ratio": 0.0,
        "cumulative_move": 0.0,
    }
    values.update(updates)
    return RegimeFeatures(**values)


def test_regime_hysteresis_and_immediate_shock() -> None:
    classifier = RegimeClassifier(StrategyConfig(confirmation_bars=3))
    result = classifier.update(features(), MarketRegime.UNKNOWN, None, 0)
    assert result.regime is MarketRegime.UNKNOWN and result.pending_count == 1
    result = classifier.update(features(), result.regime, result.pending, result.pending_count)
    result = classifier.update(features(), result.regime, result.pending, result.pending_count)
    assert result.regime is MarketRegime.RANGE
    shock = classifier.update(
        features(atr_percentile=95), result.regime, result.pending, result.pending_count
    )
    assert shock.regime is MarketRegime.SHOCK


def test_range_adx_threshold_is_configurable() -> None:
    assert (
        RegimeClassifier(StrategyConfig(range_adx_threshold=23)).raw(features(adx=22.5))
        is MarketRegime.RANGE
    )
    assert (
        RegimeClassifier(StrategyConfig(range_adx_threshold=22)).raw(features(adx=22.5))
        is MarketRegime.UNKNOWN
    )
    with pytest.raises(ValueError, match="must not exceed"):
        StrategyConfig(range_adx_threshold=26, trend_adx_threshold=25)


def test_strategy_entry_time_stop_and_stop_direction() -> None:
    strategy = AdaptiveRangeStrategy(StrategyConfig())
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("103"),
        z_score=-1.6,
        atr_percentile=50,
        spread_bps=1,
        regime=MarketRegime.RANGE,
        session_open=bar.exchange_timestamp - timedelta(hours=1),
        session_close=bar.exchange_timestamp + timedelta(hours=5),
    )
    signal = strategy.evaluate(snapshot, StrategyState(), None)
    assert signal is not None and signal.stop_price == Decimal("95.0")
    position = Position(
        instrument="QQQ",
        quantity=Decimal("1"),
        side=Side.BUY,
        average_entry_price=Decimal("100"),
        opened_at=datetime(2026, 1, 5, 14, 0, tzinfo=UTC),
        bars_held=8,
    )
    assert strategy.evaluate(snapshot, StrategyState(), position).reason == "time stop"  # type: ignore[union-attr]


def test_stop_cannot_be_widened() -> None:
    assert initial_stop(Decimal("100"), Decimal("2"), Side.BUY, Decimal("2.5")) == 95
    assert tighten_stop(Decimal("95"), Decimal("97"), Side.BUY) == 97
    with pytest.raises(ValueError, match="widened"):
        tighten_stop(Decimal("95"), Decimal("94"), Side.BUY)


def test_fixed_roe_levels_are_symmetric_and_do_not_exit_at_center() -> None:
    strategy = AdaptiveRangeStrategy(
        StrategyConfig(
            session_flatten_enabled=False,
            fixed_stop_fraction=Decimal("0.01"),
            fixed_target_fraction=Decimal("0.01"),
        )
    )
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("101"),
        z_score=-1.6,
        atr_percentile=50,
        spread_bps=1,
        regime=MarketRegime.RANGE,
        session_open=bar.exchange_timestamp,
        session_close=bar.exchange_timestamp,
    )
    signal = strategy.evaluate(snapshot, StrategyState(), None)
    assert signal is not None
    assert signal.stop_price == Decimal("99.00")
    assert signal.target_price == Decimal("101.00")
    position = Position(
        instrument="BTCUSDT",
        quantity=Decimal("0.01"),
        side=Side.BUY,
        average_entry_price=Decimal("100"),
        opened_at=bar.exchange_timestamp,
    )
    assert strategy.evaluate(snapshot, StrategyState(), position) is None


def test_weighted_reversion_waits_for_exhaustion_and_blocks_shock() -> None:
    strategy = AdaptiveRangeStrategy(
        StrategyConfig(
            entry_mode="weighted_reversion",
            short_enabled=True,
            session_flatten_enabled=False,
        )
    )
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("97"),
        z_score=2.0,
        atr_percentile=20,
        spread_bps=1,
        regime=MarketRegime.TREND_UP,
        session_open=bar.exchange_timestamp,
        session_close=bar.exchange_timestamp,
    )
    assert strategy.evaluate(snapshot, StrategyState(last_z=1.8), None) is None
    signal = strategy.evaluate(snapshot, StrategyState(last_z=2.6), None)
    assert signal is not None and signal.action is SignalAction.ENTER_SHORT
    shock = MarketSnapshot(**{**snapshot.__dict__, "regime": MarketRegime.SHOCK})
    assert strategy.evaluate(shock, StrategyState(last_z=2.6), None) is None


def test_weighted_reversion_v2_requires_confirmed_reentry_and_follows_trend() -> None:
    strategy = AdaptiveRangeStrategy(
        StrategyConfig(
            entry_mode="weighted_reversion_v2",
            weighted_entry_threshold=0.70,
            short_enabled=True,
            session_flatten_enabled=False,
        )
    )
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("97"),
        z_score=2.2,
        atr_percentile=45,
        spread_bps=1,
        regime=MarketRegime.UNKNOWN,
        session_open=bar.exchange_timestamp,
        session_close=bar.exchange_timestamp,
    )
    state = StrategyState(previous_z=3.0, last_z=2.7, last_close=Decimal("101"))
    signal = strategy.evaluate(snapshot, state, None)
    assert signal is not None and signal.action is SignalAction.ENTER_SHORT
    assert (
        strategy.evaluate(
            MarketSnapshot(**{**snapshot.__dict__, "regime": MarketRegime.TREND_UP}), state, None
        )
        is None
    )
    assert (
        strategy.evaluate(MarketSnapshot(**{**snapshot.__dict__, "z_score": 4.1}), state, None)
        is None
    )

    position = Position(
        instrument="BTCUSDT",
        quantity=Decimal("0.01"),
        side=Side.SELL,
        average_entry_price=Decimal("100"),
        opened_at=bar.exchange_timestamp,
    )
    invalidation = StrategyState(previous_z=1.8, last_z=2.0)
    exit_signal = strategy.evaluate(snapshot, invalidation, position)
    assert exit_signal is not None and exit_signal.reason == "mean reversion invalidated"


def test_weighted_reversion_v11_keeps_fast_entry_but_blocks_countertrend() -> None:
    strategy = AdaptiveRangeStrategy(
        StrategyConfig(
            entry_mode="weighted_reversion_v11",
            short_enabled=True,
            session_flatten_enabled=False,
        )
    )
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("97"),
        z_score=2.0,
        atr_percentile=20,
        spread_bps=1,
        regime=MarketRegime.UNKNOWN,
        session_open=bar.exchange_timestamp,
        session_close=bar.exchange_timestamp,
    )
    state = StrategyState(last_z=2.6)
    assert strategy.evaluate(snapshot, state, None) is not None
    trend_up = MarketSnapshot(**{**snapshot.__dict__, "regime": MarketRegime.TREND_UP})
    assert strategy.evaluate(trend_up, state, None) is None

    position = Position(
        instrument="BTCUSDT",
        quantity=Decimal("0.01"),
        side=Side.SELL,
        average_entry_price=Decimal("100"),
        opened_at=bar.exchange_timestamp,
    )
    near_vwap = MarketSnapshot(**{**snapshot.__dict__, "z_score": 0.4})
    exit_signal = strategy.evaluate(near_vwap, StrategyState(), position)
    assert exit_signal is not None and exit_signal.reason == "VWAP approach"
