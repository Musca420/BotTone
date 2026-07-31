from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from adaptive_bot.config import StrategyConfig
from adaptive_bot.domain.enums import MarketRegime, Side
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


def test_strategy_entry_time_stop_and_stop_direction() -> None:
    strategy = AdaptiveRangeStrategy(StrategyConfig())
    bar = candle()
    snapshot = MarketSnapshot(
        candle=bar,
        atr=Decimal("2"),
        center=Decimal("103"),
        z_score=-1.6,
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
