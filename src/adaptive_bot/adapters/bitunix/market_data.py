from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_bot.domain.events import MarketEvent
from adaptive_bot.domain.models import Candle

JsonGetter = Callable[[str], dict[str, Any]]
ProgressCallback = Callable[[int, float], None]


class BitunixMarketData:
    """Read-only Bitunix REST market data; never submits orders."""

    def __init__(
        self,
        market: Literal["spot", "futures"],
        *,
        timeframe_minutes: int = 15,
        futures_price_type: Literal["LAST_PRICE", "MARK_PRICE"] = "LAST_PRICE",
        progress_callback: ProgressCallback | None = None,
        get_json: JsonGetter | None = None,
    ) -> None:
        if timeframe_minutes not in {1, 5, 15, 30, 60, 120, 240}:
            raise ValueError("unsupported Bitunix timeframe")
        self.market = market
        self.timeframe_minutes = timeframe_minutes
        self.futures_price_type = futures_price_type
        self._progress_callback = progress_callback
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
            if self._progress_callback is not None:
                fraction = (end_ms - max(cursor_ms, start_ms)) / (end_ms - start_ms)
                self._progress_callback(len(rows), min(1.0, max(0.0, fraction)))
            if min(timestamps) <= start_ms:
                break
        interval_ms = self.timeframe_minutes * 60_000
        for missing_ms in _missing_timestamps(rows, interval_ms):
            payload = self._get_json(self._history_url(instrument, missing_ms + interval_ms))
            batch = payload.get("data")
            if isinstance(batch, dict):
                batch = batch.get("list", batch.get("items", [batch]))
            if not isinstance(batch, list):
                continue
            for raw in batch:
                if not isinstance(raw, dict):
                    continue
                candle, timestamp_ms = _candle(raw, instrument, self.market, self.timeframe_minutes)
                if start_ms <= timestamp_ms < end_ms:
                    rows[timestamp_ms] = candle
            time.sleep(0.11)
        if self._progress_callback is not None:
            self._progress_callback(len(rows), 1.0)
        return tuple(rows[key] for key in sorted(rows))

    def _history_url(self, instrument: str, end_ms: int) -> str:
        if self.market == "futures":
            params = {
                "symbol": instrument,
                "endTime": end_ms,
                "interval": f"{self.timeframe_minutes}m",
                "limit": 200,
                "type": self.futures_price_type,
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
    volume = (
        raw.get("quoteVol", raw.get("baseVol", raw.get("volume", "0")))
        if market == "futures"
        else raw.get("baseVol", raw.get("volume", "0"))
    )
    open_price = Decimal(str(raw["open"]))
    high_price = Decimal(str(raw["high"]))
    low_price = Decimal(str(raw["low"]))
    close_price = Decimal(str(raw["close"]))
    envelope_high = max(open_price, high_price, close_price)
    envelope_low = min(open_price, low_price, close_price)
    deviation_bps = (
        ((envelope_high - high_price) + (low_price - envelope_low)) / close_price * Decimal("10000")
    )
    if deviation_bps > Decimal("100"):
        raise ValueError(
            "Bitunix OHLC envelope deviation exceeds 100 bps: "
            f"time={timestamp_ms} open={open_price} high={high_price} "
            f"low={low_price} close={close_price} deviation_bps={deviation_bps}"
        )
    return (
        Candle(
            exchange_timestamp=exchange_timestamp,
            received_timestamp=max(datetime.now(UTC), exchange_timestamp),
            source=f"bitunix-{market}",
            instrument=instrument,
            open=open_price,
            high=envelope_high,
            low=envelope_low,
            close=close_price,
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


def _missing_timestamps(rows: dict[int, Candle], interval_ms: int) -> tuple[int, ...]:
    if len(rows) < 2:
        return ()
    first, last = min(rows), max(rows)
    return tuple(
        timestamp for timestamp in range(first, last, interval_ms) if timestamp not in rows
    )
