from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from websockets.asyncio.client import connect

from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.vwap import rolling_vwap
from adaptive_bot.meme.config import MemeBotConfig
from adaptive_bot.meme.universe import MemeContract, MemeUniverseClient

BITUNIX_WS = "wss://fapi.bitunix.com/public/"


@dataclass
class SymbolStreamState:
    symbol: str
    quote_volume_24h: str = "0"
    mark_price: str | None = None
    index_price: str | None = None
    funding_rate: str | None = None
    funding_interval_hours: str = "8"
    best_bid: str | None = None
    best_ask: str | None = None
    spread_bps: str | None = None
    depth_half_percent: str = "0"
    buy_volume: str = "0"
    sell_volume: str = "0"
    trade_count: int = 0
    latest_candle: dict[str, Any] | None = None
    updated_at: str | None = None


@dataclass
class MemeCollectorState:
    connected: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None
    symbols: dict[str, SymbolStreamState] = field(default_factory=dict)
    universe_scan: list[dict[str, object]] = field(default_factory=list)


async def collect_meme_market(
    config: MemeBotConfig,
    *,
    duration_hours: float,
    api_key: str | None = None,
    websocket_url: str = BITUNIX_WS,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if duration_hours <= 0:
        raise ValueError("collector duration must be positive")
    contracts = await asyncio.to_thread(_discover_cached, config, api_key)
    selected, ticker_volumes, universe_scan = await _select_adaptive_range(config, contracts)
    if not selected:
        raise RuntimeError("no unambiguous Bitunix meme contracts were discovered")
    await _bootstrap_history(config.storage.raw_directory, selected)
    state = MemeCollectorState(
        symbols={
            contract.symbol: SymbolStreamState(
                contract.symbol,
                quote_volume_24h=str(ticker_volumes.get(contract.symbol, Decimal("0"))),
            )
            for contract in selected
        },
        universe_scan=universe_scan,
    )
    snapshot_path = config.storage.raw_directory / "stream.json"
    events_path = config.storage.raw_directory / "events.jsonl"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_snapshot(snapshot_path, state, selected)
    deadline = monotonic() + duration_hours * 3600
    refresh_at = monotonic() + config.universe.rescan_minutes * 60
    reconnects = 0
    while monotonic() < deadline:
        try:
            async with connect(websocket_url, ping_interval=None, close_timeout=5) as websocket:
                subscriptions = [
                    {"symbol": contract.symbol, "ch": channel}
                    for contract in selected
                    for channel in (
                        "market_kline_5min",
                        "market_kline_60min",
                        "trade",
                        "depth_book15",
                        "price",
                    )
                ]
                await websocket.send(json.dumps({"op": "subscribe", "args": subscriptions}))
                state.connected = True
                state.error = None
                _write_snapshot(snapshot_path, state, selected)
                last_ping = monotonic()
                while monotonic() < deadline:
                    timeout = min(5.0, max(0.1, deadline - monotonic()))
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    except TimeoutError:
                        raw = None
                    if isinstance(raw, str):
                        message = json.loads(raw)
                        apply_ws_message(state, message)
                        await asyncio.to_thread(_append_event, events_path, message)
                        _write_snapshot(snapshot_path, state, selected)
                    if monotonic() - last_ping >= 15:
                        await websocket.send(
                            json.dumps({"op": "ping", "ping": int(datetime.now(UTC).timestamp())})
                        )
                        last_ping = monotonic()
                    if monotonic() >= refresh_at:
                        break
                if monotonic() < deadline:
                    contracts = await asyncio.to_thread(_discover_cached, config, api_key)
                    selected, ticker_volumes, universe_scan = await _select_adaptive_range(
                        config, contracts
                    )
                    await _bootstrap_history(config.storage.raw_directory, selected)
                    state = MemeCollectorState(
                        symbols={
                            contract.symbol: SymbolStreamState(
                                contract.symbol,
                                quote_volume_24h=str(
                                    ticker_volumes.get(contract.symbol, Decimal("0"))
                                ),
                            )
                            for contract in selected
                        },
                        universe_scan=universe_scan,
                    )
                    refresh_at = monotonic() + config.universe.rescan_minutes * 60
                    _write_snapshot(snapshot_path, state, selected)
        except (OSError, ValueError, json.JSONDecodeError, TimeoutError) as error:
            reconnects += 1
            state.connected = False
            state.error = type(error).__name__
            _write_snapshot(snapshot_path, state, selected)
            await asyncio.sleep(min(5, max(0, deadline - monotonic())))
    state.connected = False
    _write_snapshot(snapshot_path, state, selected)
    return {
        "symbols": len(selected),
        "reconnects": reconnects,
        "output": str(snapshot_path),
    }


async def download_meme_history(
    config: MemeBotConfig, *, weeks: int, api_key: str | None = None
) -> dict[str, object]:
    if weeks <= 0 or weeks > 104:
        raise ValueError("history window must be between 1 and 104 weeks")
    contracts = await asyncio.to_thread(_discover_cached, config, api_key)
    selected, _ = await asyncio.to_thread(
        _select_liquid, contracts, config.universe.detailed_symbols
    )
    pages = math.ceil(weeks * 7 * 24 * 12 / 200)
    await _bootstrap_history(config.storage.raw_directory, selected, pages=pages, force=True)
    return {"symbols": len(selected), "weeks": weeks, "pages_per_symbol": pages}


def apply_ws_message(state: MemeCollectorState, message: dict[str, Any]) -> None:
    symbol = str(message.get("symbol", "")).upper()
    target = state.symbols.get(symbol)
    data = message.get("data")
    if target is None or not isinstance(data, (dict, list)):
        return
    channel = str(message.get("ch", ""))
    now = datetime.now(UTC).isoformat()
    try:
        if channel.startswith("market_kline_") and isinstance(data, dict):
            timestamp_ms = int(str(data.get("t", message.get("ts"))))
            if timestamp_ms < 10_000_000_000:
                timestamp_ms *= 1000
            width_ms = 3_600_000 if "60min" in channel else 300_000
            timestamp_ms = timestamp_ms // width_ms * width_ms
            candle = {
                "timestamp": _timestamp(timestamp_ms),
                "timeframe": "1h" if "60min" in channel else "5m",
                "open": str(data["o"]),
                "high": str(data["h"]),
                "low": str(data["l"]),
                "close": str(data["c"]),
                "volume": str(data.get("v", data.get("b", "0"))),
            }
            target.latest_candle = candle
        elif channel == "price" and isinstance(data, dict):
            mark, index, funding = (_finite_decimal(data.get(name)) for name in ("mp", "ip", "fr"))
            if mark is None or index is None or mark <= 0 or index <= 0:
                return
            target.mark_price = str(mark)
            target.index_price = str(index)
            target.funding_rate = None if funding is None else str(funding)
            target.funding_interval_hours = _funding_interval_hours(data.get("ft"), data.get("nft"))
        elif channel.startswith("depth_") and isinstance(data, dict):
            _apply_depth(target, data)
        elif channel == "trade" and isinstance(data, list):
            buy = Decimal(target.buy_volume)
            sell = Decimal(target.sell_volume)
            for trade in data:
                if not isinstance(trade, dict):
                    continue
                volume = Decimal(str(trade["v"]))
                if str(trade.get("s", "")).lower() == "buy":
                    buy += volume
                else:
                    sell += volume
                target.trade_count += 1
            target.buy_volume, target.sell_volume = str(buy), str(sell)
        target.updated_at = now
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return


def _apply_depth(target: SymbolStreamState, data: dict[str, Any]) -> None:
    bids = _levels(data.get("b"))
    asks = _levels(data.get("a"))
    if not bids or not asks:
        return
    bid, ask = bids[0][0], asks[0][0]
    if bid <= 0 or ask <= bid:
        return
    midpoint = (bid + ask) / 2
    lower, upper = midpoint * Decimal("0.995"), midpoint * Decimal("1.005")
    depth = sum((price * quantity for price, quantity in bids if price >= lower), Decimal("0"))
    depth += sum((price * quantity for price, quantity in asks if price <= upper), Decimal("0"))
    target.best_bid, target.best_ask = str(bid), str(ask)
    target.spread_bps = str((ask - bid) / midpoint * Decimal("10000"))
    target.depth_half_percent = str(depth)


def _levels(value: object) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(value, list):
        return []
    levels: list[tuple[Decimal, Decimal]] = []
    for level in value:
        if isinstance(level, list) and len(level) >= 2:
            levels.append((Decimal(str(level[0])), Decimal(str(level[1]))))
    return levels


def _finite_decimal(value: object) -> Decimal | None:
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _timestamp(value: object) -> str:
    number = int(str(value))
    if number < 10_000_000_000:
        number *= 1000
    return datetime.fromtimestamp(number / 1000, UTC).isoformat()


def _funding_interval_hours(start: object, end: object) -> str:
    try:
        first = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        second = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        hours = Decimal(str((second - first).total_seconds())) / Decimal("3600")
        return str(hours) if hours > 0 else "8"
    except ValueError:
        return "8"


def _discover_cached(config: MemeBotConfig, api_key: str | None) -> tuple[MemeContract, ...]:
    cache = config.storage.raw_directory / "universe.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < (
        config.universe.catalog_cache_hours
    ):
        return _read_contracts(cache)
    try:
        contracts = MemeUniverseClient(api_key).discover(config.universe.category)
        _write_contracts(cache, contracts)
        return contracts
    except (OSError, RuntimeError, ValueError):
        if not cache.exists():
            raise
        age_hours = (time.time() - cache.stat().st_mtime) / 3600
        if age_hours > config.universe.fail_closed_after_hours:
            raise RuntimeError("meme catalog cache is stale") from None
        return _read_contracts(cache)


def _select_liquid(
    contracts: tuple[MemeContract, ...], maximum: int
) -> tuple[tuple[MemeContract, ...], dict[str, Decimal]]:
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "adaptive-range-bot/0.1"},
    )
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    rows = payload.get("data") if isinstance(payload, dict) else None
    volumes = {
        str(row.get("symbol", "")).upper(): Decimal(str(row.get("quoteVol", "0")))
        for row in rows or []
        if isinstance(row, dict)
    }
    ranked = sorted(
        contracts,
        key=lambda item: volumes.get(item.symbol, Decimal("0")),
        reverse=True,
    )
    return tuple(ranked[:maximum]), volumes


async def _select_adaptive_range(
    config: MemeBotConfig, contracts: tuple[MemeContract, ...]
) -> tuple[tuple[MemeContract, ...], dict[str, Decimal], list[dict[str, object]]]:
    _, volumes = await asyncio.to_thread(_select_liquid, contracts, len(contracts))
    scan: list[dict[str, object]] = []
    for contract in contracts:
        try:
            candles = await asyncio.to_thread(_latest_candles, contract.symbol)
            metrics = adaptive_range_metrics(candles, config)
        except (InvalidOperation, OSError, ValueError, KeyError, TypeError):
            metrics = {
                "range_favorable": False,
                "adx": None,
                "z": None,
                "score": "-Infinity",
                "reason": "market_data_unavailable",
            }
        scan.append(
            {
                "symbol": contract.symbol,
                "quote_volume_24h": str(volumes.get(contract.symbol, Decimal("0"))),
                "volume_eligible": volumes.get(contract.symbol, Decimal("0"))
                >= config.universe.minimum_quote_volume,
                **metrics,
            }
        )
        await asyncio.sleep(0.11)
    ranked = sorted(
        scan,
        key=lambda item: (
            not bool(item["volume_eligible"]),
            not bool(item["range_favorable"]),
            -Decimal(str(item["score"])),
            -volumes.get(str(item["symbol"]), Decimal("0")),
        ),
    )
    selected_symbols = [str(item["symbol"]) for item in ranked[: config.universe.detailed_symbols]]
    contracts_by_symbol = {contract.symbol: contract for contract in contracts}
    selected = tuple(
        contracts_by_symbol[symbol] for symbol in selected_symbols if symbol in contracts_by_symbol
    )
    selected_set = set(selected_symbols)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["selected"] = str(item["symbol"]) in selected_set
    _atomic_write(
        config.storage.raw_directory / "universe_scan.json",
        json.dumps(ranked, separators=(",", ":")),
    )
    return selected, volumes, ranked


def _latest_candles(symbol: str) -> list[dict[str, object]]:
    parameters = urlencode({"symbol": symbol, "interval": "5m", "limit": 200, "type": "LAST_PRICE"})
    payload = _public_json(f"https://fapi.bitunix.com/api/v1/futures/market/kline?{parameters}")
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Bitunix kline response is invalid")
    return [row for row in rows if isinstance(row, dict)]


def adaptive_range_metrics(
    candles: list[dict[str, object]], config: MemeBotConfig
) -> dict[str, object]:
    frame = pd.DataFrame(candles).rename(
        columns={
            "time": "timestamp",
            "baseVol": "volume",
        }
    )
    if len(frame) < config.strategy.adaptive_vwap_window:
        raise ValueError("insufficient adaptive-range candles")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    atr_values = atr(frame["high"], frame["low"], frame["close"], config.strategy.atr_period)
    adx_values = adx(frame["high"], frame["low"], frame["close"], config.strategy.adx_period)["adx"]
    center = rolling_vwap(
        frame["high"],
        frame["low"],
        frame["close"],
        frame["volume"],
        config.strategy.adaptive_vwap_window,
    )
    atr_value = Decimal(str(atr_values.iloc[-1]))
    adx_value = Decimal(str(adx_values.iloc[-1]))
    close = Decimal(str(frame["close"].iloc[-1]))
    center_value = Decimal(str(center.iloc[-1]))
    if (
        atr_value <= 0
        or not atr_value.is_finite()
        or not adx_value.is_finite()
        or not center_value.is_finite()
    ):
        raise ValueError("invalid adaptive-range indicators")
    z = (close - center_value) / atr_value
    range_atr = Decimal(str(frame["high"].iloc[-1] - frame["low"].iloc[-1])) / atr_value
    three_bar = abs(close - Decimal(str(frame["close"].iloc[-4]))) / atr_value
    favorable = (
        adx_value < Decimal("20")
        and range_atr <= config.strategy.shock_candle_atr
        and three_bar <= config.strategy.shock_three_bar_atr
    )
    score = abs(z) * Decimal("10") + max(Decimal("0"), Decimal("20") - adx_value)
    return {
        "range_favorable": favorable,
        "adx": str(adx_value),
        "z": str(z),
        "score": str(score),
        "reason": "range_favorable" if favorable else "regime_not_range",
    }


def _write_contracts(path: Path, contracts: tuple[MemeContract, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(item).items()
        }
        for item in contracts
    ]
    _atomic_write(path, json.dumps(payload, separators=(",", ":")))


def _read_contracts(path: Path) -> tuple[MemeContract, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cached meme catalog is invalid")
    return tuple(
        MemeContract(
            symbol=str(row["symbol"]),
            base=str(row["base"]),
            tick_size=Decimal(str(row["tick_size"])),
            lot_size=Decimal(str(row["lot_size"])),
            minimum_quantity=Decimal(str(row["minimum_quantity"])),
            minimum_notional=Decimal(str(row["minimum_notional"])),
            maximum_leverage=Decimal(str(row["maximum_leverage"])),
        )
        for row in payload
        if isinstance(row, dict)
    )


def _write_snapshot(
    path: Path, state: MemeCollectorState, contracts: tuple[MemeContract, ...]
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "connected": state.connected,
        "started_at": state.started_at,
        "error": state.error,
        "contracts": [contract.symbol for contract in contracts],
        "symbols": {symbol: asdict(value) for symbol, value in state.symbols.items()},
        "universe_scan": state.universe_scan,
    }
    _atomic_write(path, json.dumps(payload, separators=(",", ":")))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, message: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"received_at": datetime.now(UTC).isoformat(), "message": message}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(envelope, separators=(",", ":")) + "\n")


async def _bootstrap_history(
    raw_directory: Path,
    contracts: tuple[MemeContract, ...],
    *,
    pages: int = 13,
    force: bool = False,
) -> None:
    history_directory = raw_directory / "history"
    history_directory.mkdir(parents=True, exist_ok=True)
    for contract in contracts:
        target = history_directory / f"{contract.symbol}_5m.json"
        if not force and _history_is_current(target):
            continue
        rows: dict[int, dict[str, object]] = {}
        end_time: int | None = None
        for _ in range(pages):
            parameters: dict[str, object] = {
                "symbol": contract.symbol,
                "interval": "5m",
                "limit": 200,
                "type": "LAST_PRICE",
            }
            if end_time is not None:
                parameters["endTime"] = end_time
            payload = await asyncio.to_thread(
                _public_json,
                "https://fapi.bitunix.com/api/v1/futures/market/kline?" + urlencode(parameters),
            )
            batch = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(batch, list) or not batch:
                break
            timestamps: list[int] = []
            for candle in batch:
                if not isinstance(candle, dict):
                    continue
                timestamp = int(candle["time"])
                timestamps.append(timestamp)
                rows[timestamp] = {
                    "timestamp": datetime.fromtimestamp(timestamp / 1000, UTC).isoformat(),
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                    "volume": candle.get("baseVol", candle.get("volume", "0")),
                }
            if not timestamps:
                break
            end_time = min(timestamps) - 1
            await asyncio.sleep(0.11)
        ordered = [rows[key] for key in sorted(rows)]
        _atomic_write(
            target,
            json.dumps(ordered, separators=(",", ":")),
        )


def _public_json(url: str) -> object:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "adaptive-range-bot/0.1"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def _history_is_current(path: Path) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) < 2016:
            return False
        latest = datetime.fromisoformat(str(rows[-1]["timestamp"]))
        return (datetime.now(UTC) - latest.astimezone(UTC)).total_seconds() <= 86400
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
