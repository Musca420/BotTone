from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_bot.domain.events import MarketEvent
from adaptive_bot.domain.models import Candle

JsonGetter = Callable[[str], dict[str, Any]]


class BitunixMarketData:
    """Read-only Bitunix REST market data; never submits orders."""

    def __init__(
        self,
        market: Literal["spot", "futures"],
        *,
        timeframe_minutes: int = 15,
        get_json: JsonGetter | None = None,
    ) -> None:
        if timeframe_minutes not in {1, 5, 15, 30, 60, 120, 240}:
            raise ValueError("unsupported Bitunix timeframe")
        self.market = market
        self.timeframe_minutes = timeframe_minutes
        self._get_json = get_json or _get_json

    async def historical(
        self, instrument: str, start: datetime, end: datetime
    ) -> Iterable[MarketEvent]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Bitunix history boundaries must include a timezone")
        if start >= end:
            raise ValueError("history start must precede end")
        return await asyncio.to_thread(self._historical, instrument, start, end)

    async def stream(self, instrument: str) -> AsyncIterator[MarketEvent]:
        raise RuntimeError("Bitunix streaming is not enabled for simulated execution yet")
        yield  # pragma: no cover

    def _historical(self, instrument: str, start: datetime, end: datetime) -> tuple[Candle, ...]:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        cursor_ms = end_ms
        rows: dict[int, Candle] = {}
        while cursor_ms > start_ms:
            payload = self._get_json(self._history_url(instrument, cursor_ms))
            if payload.get("code") not in (0, "0"):
                raise RuntimeError(f"Bitunix market-data error: {payload.get('msg', 'unknown')}")
            batch = payload.get("data")
            if isinstance(batch, dict):
                batch = batch.get("list", batch.get("items", [batch]))
            if not isinstance(batch, list) or not batch:
                break
            timestamps: list[int] = []
            for raw in batch:
                if not isinstance(raw, dict):
                    raise ValueError("Bitunix candle payload must contain objects")
                candle, timestamp_ms = _candle(raw, instrument, self.market, self.timeframe_minutes)
                timestamps.append(timestamp_ms)
                if start_ms <= timestamp_ms < end_ms:
                    rows[timestamp_ms] = candle
            next_cursor = min(timestamps) - 1
            if next_cursor >= cursor_ms:
                raise RuntimeError("Bitunix history pagination did not advance")
            cursor_ms = next_cursor
            if min(timestamps) <= start_ms:
                break
        return tuple(rows[key] for key in sorted(rows))

    def _history_url(self, instrument: str, end_ms: int) -> str:
        if self.market == "futures":
            params = {
                "symbol": instrument,
                "endTime": end_ms,
                "interval": f"{self.timeframe_minutes}m",
                "limit": 200,
                "type": "LAST_PRICE",
            }
            base = "https://fapi.bitunix.com/api/v1/futures/market/kline"
        else:
            params = {
                "symbol": instrument,
                "interval": str(self.timeframe_minutes),
                "endTime": end_ms // 1000,
                "limit": 500,
            }
            base = "https://openapi.bitunix.com/api/spot/v1/market/kline/history"
        return f"{base}?{urlencode(params)}"


def _get_json(url: str) -> dict[str, Any]:
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "adaptive-range-bot/0.1"}
    )
    with urlopen(request, timeout=15) as response:
        return cast(dict[str, Any], json.load(response))


def _candle(
    raw: dict[str, Any],
    instrument: str,
    market: Literal["spot", "futures"],
    timeframe_minutes: int,
) -> tuple[Candle, int]:
    timestamp_ms = _timestamp_ms(raw.get("time", raw.get("ts")))
    exchange_timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    volume = raw.get("baseVol", raw.get("volume", "0"))
    return (
        Candle(
            exchange_timestamp=exchange_timestamp,
            received_timestamp=max(datetime.now(UTC), exchange_timestamp),
            source=f"bitunix-{market}",
            instrument=instrument,
            open=Decimal(str(raw["open"])),
            high=Decimal(str(raw["high"])),
            low=Decimal(str(raw["low"])),
            close=Decimal(str(raw["close"])),
            volume=Decimal(str(volume)),
            timeframe_minutes=timeframe_minutes,
        ),
        timestamp_ms,
    )


def _timestamp_ms(value: object) -> int:
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        timestamp = int(value)
        return timestamp if timestamp > 10_000_000_000 else timestamp * 1000
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Bitunix candle timestamp must include a timezone")
        return int(parsed.timestamp() * 1000)
    raise ValueError("Bitunix candle timestamp is missing")
