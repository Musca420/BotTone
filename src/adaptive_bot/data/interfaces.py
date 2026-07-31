from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from typing import Protocol

from adaptive_bot.domain.events import MarketEvent
from adaptive_bot.domain.models import Instrument


class MarketDataProvider(Protocol):
    def stream(self, instrument: str) -> AsyncIterator[MarketEvent]: ...

    async def historical(
        self, instrument: str, start: datetime, end: datetime
    ) -> Iterable[MarketEvent]: ...


class InstrumentRepository(Protocol):
    async def get(self, symbol: str) -> Instrument | None: ...


class EventStore(Protocol):
    async def append(
        self, event_id: str, event_type: str, payload: str, timestamp: datetime
    ) -> bool: ...

    async def contains(self, event_id: str) -> bool: ...
