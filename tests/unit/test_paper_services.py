from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from adaptive_bot.domain.models import Candle
from adaptive_bot.services.trading_service import CandleAggregator


def _minute(timestamp: datetime, price: str, volume: str = "10") -> Candle:
    value = Decimal(price)
    return Candle(
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        source="test",
        instrument="QQQ",
        open=value,
        high=value + Decimal("0.2"),
        low=value - Decimal("0.2"),
        close=value,
        volume=Decimal(volume),
        timeframe_minutes=1,
    )


def test_minute_bars_are_aggregated_without_future_data() -> None:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    aggregator = CandleAggregator(15)
    for offset in range(15):
        assert aggregator.add(_minute(start + timedelta(minutes=offset), str(100 + offset))) is None
    completed = aggregator.add(_minute(start + timedelta(minutes=15), "115"))
    assert completed is not None
    assert completed.exchange_timestamp == datetime(2026, 1, 5, 14, 45, tzinfo=UTC)
    assert completed.open == Decimal("100")
    assert completed.close == Decimal("114")
    assert completed.volume == Decimal("150")
