from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SimulatedClock:
    def __init__(self, current: datetime) -> None:
        if current.utcoffset() != timedelta(0):
            raise ValueError("simulated clock must start in UTC")
        self.current = current

    def now(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)

    def advance_to(self, timestamp: datetime) -> None:
        if timestamp < self.current:
            raise ValueError("clock cannot move backwards")
        self.current = timestamp
