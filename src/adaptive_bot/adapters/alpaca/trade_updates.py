from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from alpaca.trading.stream import TradingStream

from adaptive_bot.adapters.alpaca.mapping import fill_from_trade_update, order_from_alpaca
from adaptive_bot.domain.models import Fill, Order


class AlpacaTradeUpdates:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        stream_factory: Any | None = None,
    ) -> None:
        self._stream_factory = stream_factory or (
            lambda: TradingStream(api_key, secret_key, paper=True)
        )

    async def stream(self) -> AsyncIterator[Order | Fill]:
        queue: asyncio.Queue[Order | Fill | BaseException] = asyncio.Queue()
        stream = self._stream_factory()

        async def on_update(update: Any) -> None:
            event = str(getattr(update.event, "value", update.event))
            await queue.put(order_from_alpaca(update.order, received_at=update.timestamp))
            if event in {"fill", "partial_fill"}:
                await queue.put(fill_from_trade_update(update, received_at=update.timestamp))

        stream.subscribe_trade_updates(on_update)

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
