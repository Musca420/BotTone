from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from adaptive_bot.data.interfaces import MarketDataProvider
from adaptive_bot.data.repository import ParquetRepository
from adaptive_bot.data.resampling import resample_ohlcv
from adaptive_bot.data.validation import ValidationReport, validate_candles
from adaptive_bot.domain.models import Candle


async def download_history(
    provider: MarketDataProvider,
    instrument: str,
    start: datetime,
    end: datetime,
    output: str | Path,
    *,
    timeframe_minutes: int = 15,
) -> ValidationReport:
    events = await provider.historical(instrument, start, end)
    rows = [
        {
            "timestamp": event.exchange_timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }
        for event in events
        if isinstance(event, Candle)
    ]
    frame = pd.DataFrame(rows, columns=("timestamp", "open", "high", "low", "close", "volume"))
    if frame.empty:
        report = validate_candles(frame, timeframe_minutes=timeframe_minutes)
        report.require(1.0)
    frame = resample_ohlcv(frame, timeframe_minutes)
    frame = _regular_session(frame, timeframe_minutes)
    frame.attrs["split_adjusted"] = True
    report = validate_candles(frame, timeframe_minutes=timeframe_minutes)
    report.require(1.0)
    ParquetRepository.write(frame, output)
    return report


def _regular_session(frame: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(timestamps.min().date(), timestamps.max().date())
    expected = mcal.date_range(schedule, frequency=f"{timeframe_minutes}min")
    return frame[timestamps.isin(expected)].reset_index(drop=True)
