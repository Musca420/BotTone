from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

BOOK_URL = (
    "wss://fstream.binance.com/public/stream?streams=btcusdt@depth20@100ms"
)
TRADE_URL = "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade"
ROOT = Path("data/research/binance_l2")
STATUS = Path("data/reports/binance_l2_collector.status.json")


def normalize(message: dict[str, Any]) -> tuple[Literal["book", "trade"], dict[str, Any]]:
    stream, data = str(message.get("stream", "")), message.get("data")
    stream_lower = stream.lower()
    if not isinstance(data, dict):
        raise ValueError("Binance combined stream payload has no data object")
    if "depth20" in stream_lower:
        bids, asks = data.get("b"), data.get("a")
        if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
            raise ValueError("Binance depth payload is empty")
        bid, ask = Decimal(str(bids[0][0])), Decimal(str(asks[0][0]))
        if not 0 < bid < ask:
            raise ValueError("Binance order book is crossed")
        return "book", {
            "exchange_timestamp_ms": int(data["E"] if "E" in data else data.get("T", 0)),
            "last_update_id": int(data.get("u", 0)),
            "bids": bids,
            "asks": asks,
        }
    if "aggtrade" in stream_lower:
        price, quantity = Decimal(str(data["p"])), Decimal(str(data["q"]))
        return "trade", {
            "exchange_timestamp_ms": int(data["T"]),
            "price": price,
            "quantity": quantity,
            "buyer_taker": not bool(data["m"]),
        }
    raise ValueError(f"unsupported Binance stream: {stream}")


def _write_status(phase: str, detail: str, records: int) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "phase": phase,
                "detail": detail,
                "records": records,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(STATUS)


async def collect(*, duration_seconds: float | None = None) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    started = asyncio.get_running_loop().time()
    records = 0

    async def pump(url: str, queue: asyncio.Queue[str | bytes]) -> None:
        while True:
            try:
                async with connect(url, ping_interval=20, close_timeout=5) as websocket:
                    async for raw in websocket:
                        await queue.put(raw)
            except (ConnectionClosed, OSError) as error:
                _write_status("reconnecting", str(error), records)
                await asyncio.sleep(2)

    queue: asyncio.Queue[str | bytes] = asyncio.Queue(maxsize=10_000)
    tasks = [
        asyncio.create_task(pump(BOOK_URL, queue)),
        asyncio.create_task(pump(TRADE_URL, queue)),
    ]
    books: dict[int, dict[str, Any]] = {}
    flows: dict[int, tuple[Decimal, Decimal, int]] = {}
    aggregate_trades: dict[int, list[list[str]]] = {}
    aggregate_trade_events: dict[int, list[dict[str, str | int]]] = {}
    latest_second = -1
    try:
        while (
            duration_seconds is None
            or asyncio.get_running_loop().time() - started < duration_seconds
        ):
            raw = await queue.get()
            received = datetime.now(UTC)
            kind, payload = normalize(json.loads(raw))
            current_second = int(payload["exchange_timestamp_ms"]) // 1000
            payload["received_at"] = received.isoformat()
            latest_second = max(latest_second, current_second)
            if kind == "book":
                books[current_second] = payload
            else:
                buy_quote, sell_quote, trades = flows.get(
                    current_second, (Decimal(0), Decimal(0), 0)
                )
                quote = payload["price"] * payload["quantity"]
                if payload["buyer_taker"]:
                    buy_quote += quote
                    side = "BUY"
                else:
                    sell_quote += quote
                    side = "SELL"
                flows[current_second] = buy_quote, sell_quote, trades + 1
                aggregate_trades.setdefault(current_second, []).append(
                    [str(payload["price"]), str(payload["quantity"]), side]
                )
                aggregate_trade_events.setdefault(current_second, []).append(
                    {
                        "exchange_timestamp_ms": int(payload["exchange_timestamp_ms"]),
                        "received_at": str(payload["received_at"]),
                        "price": str(payload["price"]),
                        "quantity": str(payload["quantity"]),
                        "aggressor": side,
                    }
                )

            for second in sorted(value for value in books if value <= latest_second - 2):
                book = books.pop(second)
                buy_quote, sell_quote, trades = flows.pop(
                    second, (Decimal(0), Decimal(0), 0)
                )
                bids, asks = book["bids"], book["asks"]
                bid_depth = sum(Decimal(str(level[1])) for level in bids)
                ask_depth = sum(Decimal(str(level[1])) for level in asks)
                record = {
                    "schema_version": 3,
                    "exchange_second": second,
                    "available_at": received.isoformat(),
                    "book_received_at": str(book["received_at"]),
                    "book_network_latency_ms": max(
                        0,
                        int(
                            datetime.fromisoformat(str(book["received_at"])).timestamp() * 1000
                            - int(book["exchange_timestamp_ms"])
                        ),
                    ),
                    "feature_aggregation_delay_ms": max(
                        0,
                        int(received.timestamp() * 1000 - (second + 1) * 1000),
                    ),
                    "last_update_id": book["last_update_id"],
                    "bids": bids,
                    "asks": asks,
                    "depth_imbalance": str(
                        (bid_depth - ask_depth)
                        / max(bid_depth + ask_depth, Decimal("1e-9"))
                    ),
                    "buy_quote": str(buy_quote),
                    "sell_quote": str(sell_quote),
                    "trade_count": trades,
                    "aggregate_trades": aggregate_trades.pop(second, []),
                    "aggregate_trade_events": aggregate_trade_events.pop(second, []),
                    "source": "binance-official-usdm-websocket-routed",
                }
                day = datetime.fromtimestamp(second, UTC)
                target = ROOT / f"btcusdt_{day:%Y-%m-%d}.jsonl"
                with target.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                records += 1
                if records % 60 == 0:
                    _write_status("collecting", f"latest second {second}", records)

            for stale in [value for value in flows if value <= latest_second - 3]:
                flows.pop(stale, None)
                aggregate_trades.pop(stale, None)
                aggregate_trade_events.pop(stale, None)

            if duration_seconds is not None:
                elapsed = asyncio.get_running_loop().time() - started
                if elapsed >= duration_seconds:
                    _write_status("complete", "requested duration reached", records)
                    return records
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance BTCUSDT L2 research collector")
    parser.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    print(asyncio.run(collect(duration_seconds=args.duration_seconds)))


if __name__ == "__main__":
    main()
