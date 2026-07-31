from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from typing import Any

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from adaptive_bot.adapters.alpaca.mapping import candle_from_alpaca, quote_from_alpaca
from adaptive_bot.domain.events import MarketEvent


class AlpacaMarketData:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
        adjustment: str = "all",
        historical_client: Any | None = None,
        stream_factory: Any | None = None,
    ) -> None:
        self.feed = DataFeed(feed)
        self.adjustment = Adjustment(adjustment)
        self._historical = historical_client or StockHistoricalDataClient(api_key, secret_key)
        self._stream_factory = stream_factory or (
            lambda: StockDataStream(api_key, secret_key, feed=self.feed)
        )

    async def historical(
        self, instrument: str, start: datetime, end: datetime
    ) -> Iterable[MarketEvent]:
        request = StockBarsRequest(
            symbol_or_symbols=instrument,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
            adjustment=self.adjustment,
            feed=self.feed,
        )
        result = await asyncio.to_thread(self._historical.get_stock_bars, request)
        return tuple(candle_from_alpaca(bar) for bar in result[instrument])

    async def stream(self, instrument: str) -> AsyncIterator[MarketEvent]:
        queue: asyncio.Queue[MarketEvent | BaseException] = asyncio.Queue()
        stream = self._stream_factory()

        async def on_bar(bar: Any) -> None:
            await queue.put(candle_from_alpaca(bar))

        async def on_quote(quote: Any) -> None:
            await queue.put(quote_from_alpaca(quote))

        stream.subscribe_bars(on_bar, instrument)
        stream.subscribe_quotes(on_quote, instrument)

        async def run() -> None:
            try:
                await asyncio.to_thread(stream.run)
            except BaseException as error:
                await queue.put(error)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            stream.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
