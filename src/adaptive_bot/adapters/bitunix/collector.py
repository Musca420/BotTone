from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from adaptive_bot.adapters.bitunix.market_data import JsonGetter, _get_json
from adaptive_bot.bitunix_fees import futures_fee_bps

LOGGER = logging.getLogger(__name__)
BITUNIX_PUBLIC_WS = "wss://fapi.bitunix.com/public/"
DepthState = dict[str, dict[Decimal, Decimal]]
ECONOMIC_QUOTE_PROTOCOL = {
    "name": "musca_v5_vwap_post_only_quote_shadow_v1",
    "fair_value": "observed_public_trades_5m_vwap",
    "quote_distance": "1.5_x_observed_profile_maker_entry_taker_exit_cost",
    "quote_lifetime_seconds": 30,
    "submission_latency_ms": 250,
    "initial_equity_usdt": "10000",
    "margin_fraction": "0.10",
    "leverage": "10",
    "order_quantity": "floor(10000_usdt_notional/fair_value/0.001_btc)*0.001_btc",
    "catastrophic_stop_bps": "100",
    "orders": "one_bid_and_one_ask_cancel_opposite_after_first_complete_proxy_fill",
    "queue": "visible_quantity_ahead_cancellations_never_improve_fill",
    "label_horizons_seconds": [5, 30, 60, 300],
    "funding": "observed_bitunix_fraction_if_position_crosses_settlement",
    "live_orders_enabled": False,
}
ECONOMIC_QUOTE_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(ECONOMIC_QUOTE_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


async def collect_futures_candles(
    output: str | Path,
    *,
    duration_hours: float,
    poll_seconds: float = 60,
    timeframe_minutes: int = 5,
    price_type: str = "LAST_PRICE",
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
                payload = await _candles_payload(
                    get_json, timeframe_minutes, bootstrap=not seen, price_type=price_type
                )
                quote = _quote(await asyncio.to_thread(get_json, depth_url))
                _write_quote(target.with_suffix(".quote.json"), quote)
                cutoff = _bucket_start_ms(datetime.now(UTC), timeframe_minutes)
                for candle in _closed_candles(payload, cutoff, seen):
                    timestamp = int(candle["time"])
                    stream.write(
                        json.dumps(
                            {
                                "collected_at": datetime.now(UTC).isoformat(),
                                "source": f"bitunix-futures-{price_type.lower()}-public",
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
    get_json: JsonGetter,
    timeframe_minutes: int,
    *,
    bootstrap: bool,
    price_type: str = "LAST_PRICE",
) -> dict[str, Any]:
    if price_type not in {"LAST_PRICE", "MARK_PRICE"}:
        raise ValueError("Bitunix price_type must be LAST_PRICE or MARK_PRICE")
    rows: list[dict[str, Any]] = []
    end_time: int | None = None
    for _ in range(2 if bootstrap else 1):
        parameters: dict[str, object] = {
            "symbol": "BTCUSDT",
            "interval": f"{timeframe_minutes}m",
            "limit": 200 if bootstrap else 3,
            "type": price_type,
        }
        if end_time is not None:
            parameters["endTime"] = end_time
        payload = await asyncio.to_thread(
            get_json,
            "https://fapi.bitunix.com/api/v1/futures/market/kline?" + urlencode(parameters),
        )
        batch = payload.get("data")
        if payload.get("code") not in (0, "0") or not isinstance(batch, list) or not batch:
            raise ValueError(f"Bitunix {price_type.lower()} history is unavailable")
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


def normalize_microstructure_message(
    message: dict[str, Any],
    *,
    received_at: datetime | None = None,
    notionals: tuple[Decimal, ...] = (Decimal("1000"), Decimal("5000"), Decimal("10000")),
) -> list[dict[str, Any]]:
    """Normalize the officially documented Bitunix depth and public-trade channels."""
    received = received_at or datetime.now(UTC)
    channel = str(message.get("ch", ""))
    symbol = str(message.get("symbol", "")).upper()
    data = message.get("data")
    if not symbol:
        return []
    if channel.startswith("depth_") and isinstance(data, dict):
        bids = _microstructure_levels(data.get("b"), reverse=True)
        asks = _microstructure_levels(data.get("a"), reverse=False)
        if not bids or not asks or bids[0][0] >= asks[0][0]:
            return []
        bid, ask = bids[0][0], asks[0][0]
        midpoint = (bid + ask) / 2
        return [
            {
                "schema_version": 1,
                "event_type": "book",
                "source": "bitunix-official-public-websocket",
                "symbol": symbol,
                "exchange_timestamp": _exchange_timestamp(message.get("ts"), received),
                "received_timestamp": received.isoformat(),
                "channel": channel,
                "best_bid": str(bid),
                "best_ask": str(ask),
                "midpoint": str(midpoint),
                "spread_bps": str((ask - bid) / midpoint * Decimal("10000")),
                "bids": [[str(price), str(quantity)] for price, quantity in bids],
                "asks": [[str(price), str(quantity)] for price, quantity in asks],
                "market_order_shortfall_bps": {
                    str(notional): {
                        "buy": _book_shortfall(asks, notional, midpoint, buy=True),
                        "sell": _book_shortfall(bids, notional, midpoint, buy=False),
                    }
                    for notional in notionals
                },
            }
        ]
    if channel == "trade" and isinstance(data, list):
        records: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                price = Decimal(str(item["p"]))
                quantity = Decimal(str(item["v"]))
            except (InvalidOperation, KeyError, ValueError):
                continue
            side = str(item.get("s", "")).lower()
            if not (price.is_finite() and quantity.is_finite() and price > 0 and quantity > 0):
                continue
            if side not in {"buy", "sell"}:
                continue
            records.append(
                {
                    "schema_version": 1,
                    "event_type": "trade",
                    "source": "bitunix-official-public-websocket",
                    "symbol": symbol,
                    "exchange_timestamp": _exchange_timestamp(item.get("t"), received),
                    "received_timestamp": received.isoformat(),
                    "price": str(price),
                    "quantity": str(quantity),
                    "aggressor_side": side,
                }
            )
        return records
    return []


def normalize_deep_depth_message(
    message: dict[str, Any],
    state: DepthState,
    *,
    snapshot: bool,
    received_at: datetime | None = None,
    notionals: tuple[Decimal, ...] = (Decimal("1000"), Decimal("5000"), Decimal("10000")),
) -> list[dict[str, Any]]:
    """Apply the official full-depth snapshot/delta stream without inventing levels."""
    received = received_at or datetime.now(UTC)
    data = message.get("data")
    symbol = str(message.get("symbol", "")).upper()
    if str(message.get("ch", "")) != "depth_books" or not symbol or not isinstance(data, dict):
        return []
    bid_changes = _depth_changes(data.get("b"))
    ask_changes = _depth_changes(data.get("a"))
    wire_payload_mode = "snapshot" if snapshot else "incremental"
    if snapshot:
        if not bid_changes or not ask_changes:
            raise ValueError("Bitunix full-depth snapshot is incomplete")
        state["bids"] = {}
        state["asks"] = {}
    elif not bid_changes and not ask_changes:
        return []
    full_refresh = bool(
        not snapshot
        and state["bids"]
        and state["asks"]
        and len(bid_changes) >= len(state["bids"]) * 0.8
        and len(ask_changes) >= len(state["asks"]) * 0.8
    )
    if full_refresh:
        wire_payload_mode = "full_refresh_diff"
        for name, changes in (("bids", bid_changes), ("asks", ask_changes)):
            previous = state[name]
            refreshed = {price: quantity for price, quantity in changes if quantity > 0}
            actual_changes = [
                (price, quantity)
                for price, quantity in refreshed.items()
                if previous.get(price) != quantity
            ]
            actual_changes.extend(
                (price, Decimal("0")) for price in previous.keys() - refreshed.keys()
            )
            state[name] = refreshed
            if name == "bids":
                bid_changes = actual_changes
            else:
                ask_changes = actual_changes
    else:
        for name, changes in (("bids", bid_changes), ("asks", ask_changes)):
            levels = state[name]
            for price, quantity in changes:
                if quantity == 0:
                    levels.pop(price, None)
                else:
                    levels[price] = quantity
    bids = sorted(state["bids"].items(), reverse=True)
    asks = sorted(state["asks"].items())
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        raise ValueError("Bitunix full-depth state is empty or crossed")
    midpoint = (bids[0][0] + asks[0][0]) / 2
    stored_bids = bids if snapshot else bid_changes
    stored_asks = asks if snapshot else ask_changes
    return [
        {
            "schema_version": 2,
            "event_type": "book_snapshot" if snapshot else "book_delta",
            "source": "bitunix-official-public-websocket",
            "symbol": symbol,
            "exchange_timestamp": _exchange_timestamp(message.get("ts"), received),
            "received_timestamp": received.isoformat(),
            "channel": "depth_books",
            "best_bid": str(bids[0][0]),
            "best_ask": str(asks[0][0]),
            "midpoint": str(midpoint),
            "spread_bps": str((asks[0][0] - bids[0][0]) / midpoint * Decimal("10000")),
            "bids": [[str(price), str(quantity)] for price, quantity in stored_bids],
            "asks": [[str(price), str(quantity)] for price, quantity in stored_asks],
            "market_order_shortfall_bps": {
                str(notional): {
                    "buy": _book_shortfall(asks, notional, midpoint, buy=True),
                    "sell": _book_shortfall(bids, notional, midpoint, buy=False),
                }
                for notional in notionals
            },
            "sequence_id_available": False,
            "wire_payload_mode": wire_payload_mode,
        }
    ]


async def collect_futures_microstructure(
    output_directory: str | Path,
    *,
    symbol: str = "BTCUSDT",
    duration_hours: float = 168,
    get_json: JsonGetter = _get_json,
    websocket_url: str = BITUNIX_PUBLIC_WS,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if duration_hours <= 0:
        raise ValueError("collector duration must be positive")
    symbol = symbol.upper()
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    status_path = target / f"status_{symbol.lower()}.json"
    deadline = monotonic() + duration_hours * 3600
    counts = {"book": 0, "trade": 0}
    reconnects = 0
    deep_snapshots = 0
    deep_deltas = 0
    deep_bid_levels = 0
    deep_ask_levels = 0
    depth_state: DepthState = {"bids": {}, "asks": {}}
    recent_trades: deque[tuple[datetime, Decimal, Decimal]] = deque()
    last_spread: str | None = None
    last_bid: str | None = None
    last_ask: str | None = None
    last_shortfall: dict[str, str | None] | None = None
    stream: Any = None
    stream_path: Path | None = None
    stream_day = ""
    quote_stream: Any = None
    quote_day = ""
    tape_stream: Any = None
    tape_day = ""
    last_flush = monotonic()

    def append_tape(day: str, payload: dict[str, Any]) -> None:
        nonlocal tape_stream, tape_day
        if day != tape_day:
            if tape_stream is not None:
                tape_stream.flush()
                os.fsync(tape_stream.fileno())
                tape_stream.close()
            tape_stream = (target / f"{symbol.lower()}_post_only_tape_{day}.jsonl").open(
                "a", encoding="utf-8"
            )
            tape_day = day
        tape_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def save(message: dict[str, Any], records: list[dict[str, Any]]) -> None:
        nonlocal stream, stream_path, stream_day, last_flush
        nonlocal last_spread, last_bid, last_ask, last_shortfall
        nonlocal deep_snapshots, deep_deltas, deep_bid_levels, deep_ask_levels
        if not records:
            return
        received_at = str(records[0]["received_timestamp"])
        day = received_at[:10]
        if day != stream_day:
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            stream_path = target / f"{symbol.lower()}_{day}.jsonl"
            stream = stream_path.open("a", encoding="utf-8")
            stream_day = day
        stored_message = message
        if any(int(record.get("schema_version", 1)) == 2 for record in records):
            stored_message = {
                "ch": message.get("ch"),
                "symbol": message.get("symbol"),
                "ts": message.get("ts"),
                "payload_omitted": "normalized_schema_v2_contains_required_state",
            }
        stream.write(
            json.dumps(
                {"received_at": received_at, "raw": stored_message, "normalized": records},
                separators=(",", ":"),
            )
            + "\n"
        )
        for record in records:
            event_type = str(record["event_type"])
            kind = "book" if event_type.startswith("book") else event_type
            counts[kind] += 1
            if event_type == "book_snapshot":
                deep_snapshots += 1
                deep_bid_levels = len(record.get("bids", []))
                deep_ask_levels = len(record.get("asks", []))
            elif event_type == "book_delta":
                deep_deltas += 1
            elif event_type == "trade":
                try:
                    observed = datetime.fromisoformat(
                        str(record["received_timestamp"]).replace("Z", "+00:00")
                    ).astimezone(UTC)
                    recent_trades.append(
                        (
                            observed,
                            Decimal(str(record["price"])),
                            Decimal(str(record["quantity"])),
                        )
                    )
                except (InvalidOperation, KeyError, ValueError):
                    pass
            if int(record.get("schema_version", 1)) == 1 and event_type == "book":
                append_tape(
                    day,
                    {
                        "type": "book",
                        "at": record["received_timestamp"],
                        "best_bid": record["best_bid"],
                        "best_ask": record["best_ask"],
                    },
                )
            elif int(record.get("schema_version", 1)) == 1 and event_type == "trade":
                append_tape(
                    day,
                    {
                        "type": "trade",
                        "at": record["received_timestamp"],
                        "exchange_at": record["exchange_timestamp"],
                        "price": record["price"],
                        "quantity": record["quantity"],
                        "aggressor_side": record["aggressor_side"],
                    },
                )
            if kind == "book":
                last_spread = str(record["spread_bps"])
                last_bid = str(record["best_bid"])
                last_ask = str(record["best_ask"])
                last_shortfall = record["market_order_shortfall_bps"]["10000"]
        if monotonic() - last_flush >= 1:
            stream.flush()
            last_flush = monotonic()

    def status(*, connected: bool, error: str | None = None) -> None:
        nonlocal quote_stream, quote_day
        now = datetime.now(UTC)
        cutoff = now.timestamp() - 300
        while recent_trades and recent_trades[0][0].timestamp() < cutoff:
            recent_trades.popleft()
        traded = sum((quantity for _, _, quantity in recent_trades), Decimal("0"))
        fair_value = (
            sum((price * quantity for _, price, quantity in recent_trades), Decimal("0"))
            / traded
            if traded > 0
            else None
        )
        economic_quotes = _economic_quote_levels(depth_state, fair_value)
        if stream is not None:
            stream.flush()
        payload = {
                "schema_version": 2,
                "connected": connected,
                "symbol": symbol,
                "book_events": counts["book"],
                "trade_events": counts["trade"],
                "depth_mode": "book15_plus_full_depth_snapshot_delta",
                "deep_book_synchronized": deep_snapshots > 0,
                "deep_snapshot_events": deep_snapshots,
                "deep_delta_events": deep_deltas,
                "deep_snapshot_bid_levels": deep_bid_levels,
                "deep_snapshot_ask_levels": deep_ask_levels,
                "fair_value_5m_vwap": None if fair_value is None else str(fair_value),
                "economic_quote_protocol": ECONOMIC_QUOTE_PROTOCOL,
                "economic_quote_protocol_hash": ECONOMIC_QUOTE_PROTOCOL_HASH,
                "economic_quote_levels": economic_quotes,
                "reconnects": reconnects,
                "last_spread_bps": last_spread,
                "best_bid": last_bid,
                "best_ask": last_ask,
                "shortfall_10000_bps": last_shortfall,
                "current_file": None if not stream_day else f"{symbol.lower()}_{stream_day}.jsonl",
                "raw_stream_offset": None if stream is None else stream.tell(),
                "error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        _write_microstructure_status(status_path, payload)
        if connected and economic_quotes:
            day = now.date().isoformat()
            if day != quote_day:
                if quote_stream is not None:
                    quote_stream.flush()
                    os.fsync(quote_stream.fileno())
                    quote_stream.close()
                quote_stream = (
                    target / f"{symbol.lower()}_economic_quote_shadow_{day}.jsonl"
                ).open("a", encoding="utf-8")
                quote_day = day
            quote_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            quote_stream.flush()
            append_tape(
                day,
                {
                    "type": "quote",
                    "at": payload["updated_at"],
                    "fair_value": payload["fair_value_5m_vwap"],
                    "protocol_hash": ECONOMIC_QUOTE_PROTOCOL_HASH,
                    "quotes": economic_quotes,
                },
            )
            tape_stream.flush()
        LOGGER.info(
            "MICROSTRUCTURE connected=%s book=%s trade=%s bid=%s ask=%s spread_bps=%s "
            "shortfall_10k=%s reconnects=%s file=%s",
            connected,
            counts["book"],
            counts["trade"],
            last_bid,
            last_ask,
            last_spread,
            last_shortfall,
            reconnects,
            stream_day or "waiting",
        )

    try:
        while monotonic() < deadline:
            try:
                received = datetime.now(UTC)
                depth_url = "https://fapi.bitunix.com/api/v1/futures/market/depth?" + urlencode(
                    {"symbol": symbol, "limit": "max"}
                )
                snapshot = await asyncio.to_thread(get_json, depth_url)
                depth = snapshot.get("data")
                if snapshot.get("code") not in (0, "0") or not isinstance(depth, dict):
                    raise ValueError("Bitunix REST depth bootstrap is invalid")
                deep_bootstrap = {
                    "ch": "depth_books",
                    "symbol": symbol,
                    "ts": int(received.timestamp() * 1000),
                    "data": {"a": depth.get("asks"), "b": depth.get("bids")},
                }
                depth_state["bids"].clear()
                depth_state["asks"].clear()
                records = normalize_deep_depth_message(
                    deep_bootstrap,
                    depth_state,
                    snapshot=True,
                    received_at=received,
                )
                for record in records:
                    record["source"] = "bitunix-official-rest-depth-snapshot"
                save(deep_bootstrap, records)
                bootstrap = {
                    "ch": "depth_book15_rest_snapshot",
                    "symbol": symbol,
                    "ts": int(received.timestamp() * 1000),
                    "data": {
                        "a": list(depth.get("asks") or [])[:15],
                        "b": list(depth.get("bids") or [])[:15],
                    },
                }
                records = normalize_microstructure_message(bootstrap, received_at=received)
                for record in records:
                    record["source"] = "bitunix-official-rest-depth-snapshot"
                save(bootstrap, records)
                async with connect(websocket_url, ping_interval=None, close_timeout=5) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "op": "subscribe",
                                "args": [
                                    {"symbol": symbol, "ch": "depth_book15"},
                                    {"symbol": symbol, "ch": "depth_books"},
                                    {"symbol": symbol, "ch": "trade"},
                                ],
                            }
                        )
                    )
                    status(connected=True)
                    last_ping = monotonic()
                    last_report = monotonic()
                    last_market_event = monotonic()
                    depth_stream_started = False
                    last_deep_checkpoint = monotonic()
                    deep_checkpoint_day = datetime.now(UTC).date()
                    while monotonic() < deadline:
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=5)
                        except TimeoutError:
                            raw = None
                        received = datetime.now(UTC)
                        if isinstance(raw, str):
                            message = json.loads(raw)
                            if message.get("ch") == "depth_books":
                                records = normalize_deep_depth_message(
                                    message,
                                    depth_state,
                                    snapshot=not depth_stream_started,
                                    received_at=received,
                                )
                                depth_stream_started = True
                            else:
                                records = normalize_microstructure_message(
                                    message, received_at=received
                                )
                            save(message, records)
                            if records:
                                last_market_event = monotonic()
                        now = monotonic()
                        if depth_stream_started and (
                            now - last_deep_checkpoint >= 900
                            or received.date() != deep_checkpoint_day
                        ):
                            checkpoint = _depth_state_message(depth_state, symbol, received)
                            records = normalize_deep_depth_message(
                                checkpoint,
                                depth_state,
                                snapshot=True,
                                received_at=received,
                            )
                            for record in records:
                                record["source"] = "bitunix-derived-causal-depth-checkpoint"
                            save(checkpoint, records)
                            last_deep_checkpoint = now
                            deep_checkpoint_day = received.date()
                        if now - last_report >= 5:
                            status(connected=True)
                            last_report = now
                        if now - last_ping >= 15:
                            await websocket.send(
                                json.dumps(
                                    {"op": "ping", "ping": int(datetime.now(UTC).timestamp())}
                                )
                            )
                            last_ping = now
                        if now - last_market_event > 45:
                            raise TimeoutError("Bitunix public market stream is stale")
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
                TimeoutError,
                ConnectionClosed,
            ) as error:
                reconnects += 1
                status(connected=False, error=type(error).__name__)
                LOGGER.warning("Bitunix microstructure reconnect %s: %s", reconnects, error)
                await asyncio.sleep(
                    min(30, 2 ** min(reconnects, 4), max(0, deadline - monotonic()))
                )
    finally:
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        if quote_stream is not None:
            quote_stream.flush()
            os.fsync(quote_stream.fileno())
            quote_stream.close()
        if tape_stream is not None:
            tape_stream.flush()
            os.fsync(tape_stream.fileno())
            tape_stream.close()
        status(connected=False)
    return {**counts, "reconnects": reconnects, "output": str(target)}


def compact_microstructure_jsonl(path: str | Path) -> Path:
    """Create an idempotent daily Parquet while preserving the append-only JSONL."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = payload.get("normalized", [])
            if not isinstance(normalized, list):
                continue
            for record in normalized:
                if not isinstance(record, dict):
                    continue
                canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
                flat = {
                    key: value
                    for key, value in record.items()
                    if not isinstance(value, (dict, list))
                }
                flat["event_id"] = hashlib.sha256(canonical.encode()).hexdigest()
                for key in ("bids", "asks", "market_order_shortfall_bps"):
                    if key in record:
                        flat[f"{key}_json"] = json.dumps(
                            record[key], sort_keys=True, separators=(",", ":")
                        )
                rows.append(flat)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("event_id").sort_values(
            ["exchange_timestamp", "event_type", "event_id"]
        )
    target = source.with_suffix(".parquet")
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return target


def _microstructure_levels(value: object, *, reverse: bool) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(value, list):
        return []
    levels: list[tuple[Decimal, Decimal]] = []
    for item in value:
        try:
            price, quantity = Decimal(str(item[0])), Decimal(str(item[1]))
        except (IndexError, InvalidOperation, TypeError, ValueError):
            continue
        if price.is_finite() and quantity.is_finite() and price > 0 and quantity > 0:
            levels.append((price, quantity))
    return sorted(levels, key=lambda item: item[0], reverse=reverse)


def _depth_changes(value: object) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(value, list):
        return []
    changes: list[tuple[Decimal, Decimal]] = []
    for item in value:
        try:
            price, quantity = Decimal(str(item[0])), Decimal(str(item[1]))
        except (IndexError, InvalidOperation, TypeError, ValueError):
            continue
        if price.is_finite() and quantity.is_finite() and price > 0 and quantity >= 0:
            changes.append((price, quantity))
    return changes


def _depth_state_message(
    state: DepthState, symbol: str, received: datetime
) -> dict[str, Any]:
    return {
        "ch": "depth_books",
        "symbol": symbol,
        "ts": int(received.timestamp() * 1000),
        "data": {
            "b": [
                [str(price), str(quantity)]
                for price, quantity in sorted(state["bids"].items(), reverse=True)
            ],
            "a": [
                [str(price), str(quantity)]
                for price, quantity in sorted(state["asks"].items())
            ],
        },
    }


def _economic_quote_levels(
    state: DepthState, fair_value: Decimal | None
) -> dict[str, dict[str, str | None]]:
    if fair_value is None or fair_value <= 0 or not state["bids"] or not state["asks"]:
        return {}
    bids = sorted(state["bids"].items(), reverse=True)
    asks = sorted(state["asks"].items())
    result: dict[str, dict[str, str | None]] = {}
    for level in range(6):
        maker_bps, taker_bps = futures_fee_bps(level)
        round_trip = Decimal(str(maker_bps + taker_bps))
        minimum_distance = round_trip * Decimal("1.5")
        bid_target = fair_value * (Decimal("1") - minimum_distance / Decimal("10000"))
        ask_target = fair_value * (Decimal("1") + minimum_distance / Decimal("10000"))
        bid = next(((price, quantity) for price, quantity in bids if price <= bid_target), None)
        ask = next(((price, quantity) for price, quantity in asks if price >= ask_target), None)
        bid_distance = (
            (fair_value - bid[0]) / fair_value * Decimal("10000") if bid else None
        )
        ask_distance = (
            (ask[0] - fair_value) / fair_value * Decimal("10000") if ask else None
        )
        result[f"VIP{level}"] = {
            "minimum_distance_bps": str(minimum_distance),
            "bid_price": None if bid is None else str(bid[0]),
            "bid_queue_btc": None if bid is None else str(bid[1]),
            "bid_distance_bps": None if bid_distance is None else str(bid_distance),
            "bid_net_to_fair_bps": (
                None if bid_distance is None else str(bid_distance - round_trip)
            ),
            "ask_price": None if ask is None else str(ask[0]),
            "ask_queue_btc": None if ask is None else str(ask[1]),
            "ask_distance_bps": None if ask_distance is None else str(ask_distance),
            "ask_net_to_fair_bps": (
                None if ask_distance is None else str(ask_distance - round_trip)
            ),
        }
    return result


def _book_shortfall(
    levels: list[tuple[Decimal, Decimal]],
    notional: Decimal,
    midpoint: Decimal,
    *,
    buy: bool,
) -> str | None:
    remaining = notional / midpoint
    filled = Decimal("0")
    cost = Decimal("0")
    for price, available in levels:
        quantity = min(remaining, available)
        filled += quantity
        cost += price * quantity
        remaining -= quantity
        if remaining <= 0:
            break
    if remaining > 0 or filled <= 0:
        return None
    vwap = cost / filled
    shortfall = (vwap - midpoint if buy else midpoint - vwap) / midpoint * Decimal("10000")
    return str(shortfall)


def _exchange_timestamp(value: object, fallback: datetime) -> str:
    if value is None:
        return fallback.isoformat()
    try:
        if isinstance(value, str) and not value.isdigit():
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).isoformat()
        timestamp = int(str(value))
        if timestamp < 10_000_000_000:
            timestamp *= 1000
        return datetime.fromtimestamp(timestamp / 1000, UTC).isoformat()
    except (OverflowError, ValueError):
        return fallback.isoformat()


def _write_microstructure_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
