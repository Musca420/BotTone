from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig
from adaptive_bot.data.interfaces import MarketDataProvider
from adaptive_bot.data.repository import ParquetRepository
from adaptive_bot.domain.models import Candle


class CandleAggregator:
    def __init__(self, minutes: int) -> None:
        self.minutes = minutes
        self._current: Candle | None = None
        self._bucket_end: datetime | None = None

    def add(self, candle: Candle) -> Candle | None:
        minute = candle.exchange_timestamp.replace(second=0, microsecond=0)
        offset = self.minutes - minute.minute % self.minutes
        bucket_end = minute + timedelta(minutes=offset)
        if self._current is None or bucket_end == self._bucket_end:
            self._current = self._merge(self._current, candle, bucket_end)
            self._bucket_end = bucket_end
            return None
        completed = self._current
        self._current = self._merge(None, candle, bucket_end)
        self._bucket_end = bucket_end
        return completed

    def _merge(self, current: Candle | None, candle: Candle, bucket_end: datetime) -> Candle:
        if current is None:
            return candle.model_copy(
                update={"exchange_timestamp": bucket_end, "timeframe_minutes": self.minutes}
            )
        return current.model_copy(
            update={
                "received_timestamp": max(current.received_timestamp, candle.received_timestamp),
                "high": max(current.high, candle.high),
                "low": min(current.low, candle.low),
                "close": candle.close,
                "volume": current.volume + candle.volume,
            }
        )


async def run_shadow(
    config: AppConfig,
    provider: MarketDataProvider,
    input_path: str | Path,
    output_path: str | Path,
    *,
    stop_after_bars: int | None = None,
) -> int:
    frame = ParquetRepository.read(input_path)
    frame.attrs["split_adjusted"] = True
    aggregator = CandleAggregator(config.strategy.timeframe_minutes)
    completed_count = 0
    async for event in provider.stream(config.instrument.symbol):
        if not isinstance(event, Candle):
            continue
        completed = aggregator.add(event)
        if completed is None:
            continue
        frame = _append(frame, completed)
        # ponytail: replay is O(n²); replace with incremental runtime above sustained paper volume.
        result = await BacktestEngine(config).run(frame)
        result.write_json(output_path)
        completed_count += 1
        if stop_after_bars is not None and completed_count >= stop_after_bars:
            break
    return completed_count


def _append(frame: pd.DataFrame, candle: Candle) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "timestamp": candle.exchange_timestamp,
                "open": Decimal(candle.open),
                "high": Decimal(candle.high),
                "low": Decimal(candle.low),
                "close": Decimal(candle.close),
                "volume": Decimal(candle.volume),
            }
        ]
    )
    result = pd.concat([frame, row], ignore_index=True)
    result = result.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    result.attrs["split_adjusted"] = True
    return result.reset_index(drop=True)
