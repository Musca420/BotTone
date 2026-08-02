from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from adaptive_bot.adapters.bitunix.market_data import JsonGetter, _get_json

LOGGER = logging.getLogger(__name__)


async def collect_futures_candles(
    output: str | Path,
    *,
    duration_hours: float,
    poll_seconds: float = 60,
    timeframe_minutes: int = 5,
    get_json: JsonGetter = _get_json,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    if duration_hours <= 0 or poll_seconds <= 0:
        raise ValueError("collector duration and poll interval must be positive")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(target)
    deadline = monotonic() + duration_hours * 3600
    collected = 0
    depth_url = "https://fapi.bitunix.com/api/v1/futures/market/depth?" + urlencode(
        {"symbol": "BTCUSDT", "limit": 1}
    )
    with target.open("a", encoding="utf-8") as stream:
        while True:
            try:
                payload = await _candles_payload(get_json, timeframe_minutes, bootstrap=not seen)
                quote = _quote(await asyncio.to_thread(get_json, depth_url))
                _write_quote(target.with_suffix(".quote.json"), quote)
                cutoff = _bucket_start_ms(datetime.now(UTC), timeframe_minutes)
                for candle in _closed_candles(payload, cutoff, seen):
                    timestamp = int(candle["time"])
                    stream.write(
                        json.dumps(
                            {
                                "collected_at": datetime.now(UTC).isoformat(),
                                "source": "bitunix-futures-mark-public",
                                "symbol": "BTCUSDT",
                                "timeframe_minutes": timeframe_minutes,
                                "spread_bps": quote["spread_bps"],
                                "candle": candle,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    seen.add(timestamp)
                    collected += 1
                stream.flush()
                os.fsync(stream.fileno())
                LOGGER.info("Bitunix collector saved %s closed candles", collected)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("Bitunix collector retrying after error: %s", error)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_seconds, remaining))
    return collected


async def _candles_payload(
    get_json: JsonGetter, timeframe_minutes: int, *, bootstrap: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    end_time: int | None = None
    for _ in range(2 if bootstrap else 1):
        parameters: dict[str, object] = {
            "symbol": "BTCUSDT",
            "interval": f"{timeframe_minutes}m",
            "limit": 200 if bootstrap else 3,
            "type": "MARK_PRICE",
        }
        if end_time is not None:
            parameters["endTime"] = end_time
        payload = await asyncio.to_thread(
            get_json,
            "https://fapi.bitunix.com/api/v1/futures/market/kline?" + urlencode(parameters),
        )
        batch = payload.get("data")
        if payload.get("code") not in (0, "0") or not isinstance(batch, list) or not batch:
            raise ValueError("Bitunix mark-price history is unavailable")
        rows.extend(candle for candle in batch if isinstance(candle, dict))
        end_time = min(int(candle["time"]) for candle in batch) - 1
    return {"code": 0, "data": rows}


def _closed_candles(
    payload: dict[str, Any], cutoff_ms: int, seen: set[int]
) -> tuple[dict[str, Any], ...]:
    if payload.get("code") not in (0, "0"):
        raise ValueError(f"Bitunix market-data error: {payload.get('msg', 'unknown')}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Bitunix collector expected a candle list")
    closed = [
        candle
        for candle in data
        if isinstance(candle, dict)
        and str(candle.get("time", "")).isdigit()
        and int(candle["time"]) < cutoff_ms
        and int(candle["time"]) not in seen
    ]
    return tuple(sorted(closed, key=lambda candle: int(candle["time"])))


def _bucket_start_ms(now: datetime, timeframe_minutes: int) -> int:
    seconds = int(now.timestamp())
    width = timeframe_minutes * 60
    return (seconds // width) * width * 1000


def _quote(payload: dict[str, Any]) -> dict[str, str]:
    if payload.get("code") not in (0, "0") or not isinstance(payload.get("data"), dict):
        raise ValueError("Bitunix depth response is invalid")
    data = payload["data"]
    try:
        ask = Decimal(str(data["asks"][0][0]))
        bid = Decimal(str(data["bids"][0][0]))
    except (IndexError, InvalidOperation, KeyError, TypeError, ValueError) as error:
        raise ValueError("Bitunix depth is missing best bid/ask") from error
    if not (bid.is_finite() and ask.is_finite() and 0 < bid < ask):
        raise ValueError("Bitunix best bid/ask is crossed or invalid")
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "best_bid": str(bid),
        "best_ask": str(ask),
        "spread_bps": str((ask - bid) / ((ask + bid) / 2) * Decimal("10000")),
    }


def _write_quote(path: Path, quote: dict[str, str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(quote, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _load_seen(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            timestamp = json.loads(line)["candle"]["time"]
            seen.add(int(timestamp))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return seen
