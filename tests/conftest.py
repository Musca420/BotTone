from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from adaptive_bot.domain.enums import AssetClass
from adaptive_bot.domain.models import Candle, Instrument


@pytest.fixture(scope="session")
def instrument() -> Instrument:
    return Instrument(
        symbol="QQQ",
        asset_class=AssetClass.EQUITY,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("1"),
        minimum_quantity=Decimal("1"),
        minimum_notional=Decimal("1"),
    )


def candle(
    timestamp: datetime | None = None,
    *,
    open_price: str = "100",
    high: str = "101",
    low: str = "99",
    close: str = "100",
    volume: str = "1000",
) -> Candle:
    timestamp = timestamp or datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    return Candle(
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        source="test",
        instrument="QQQ",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
    )


@pytest.fixture
def rth_frame() -> pd.DataFrame:
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule("2026-01-05", "2026-01-12")
    timestamps = mcal.date_range(schedule, frequency="15min")
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        base = 100 + (index % 12 - 6) * 0.15
        rows.append(
            {
                "timestamp": timestamp,
                "open": base,
                "high": base + 0.6,
                "low": base - 0.6,
                "close": base + (0.1 if index % 2 else -0.1),
                "volume": 100_000,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["split_adjusted"] = True
    return frame
