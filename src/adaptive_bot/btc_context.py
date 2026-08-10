from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from adaptive_bot.adapters.bitunix.market_data import _get_json

JsonGetter = Callable[[str], dict[str, Any]]
DEFAULT_OUTPUT = Path("data/raw/btc_context")
READINESS_REPORT = Path("data/reports/btc_model_readiness.json")
ALPHA_ROOT = Path("data/ml/hybrid_v7/alpha_raw")
ORDERFLOW_PATH = Path("data/ml/hybrid_v14/binance_reference_orderflow_1m.parquet")
MICROSTRUCTURE_ROOT = Path("data/raw/bitunix_microstructure")


def context_urls(now: datetime) -> dict[str, str]:
    del now
    # Operational Musca V5 context: Binance supplies Alpha and Bitunix supplies
    # execution/funding. Historical Bybit, OKX and Deribit files are preserved,
    # but those venues are no longer polled or allowed to delay this feed.
    return {
        "binance_premium": "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
        "binance_oi": "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
        "binance_spot": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "bitunix": (
            "https://fapi.bitunix.com/api/v1/futures/market/funding_rate?"
            + urlencode({"symbol": "BTCUSDT"})
        ),
    }


async def fetch_context(
    *, get_json: JsonGetter = _get_json, received_at: datetime | None = None
) -> dict[str, Any]:
    received = received_at or datetime.now(UTC)
    urls = context_urls(received)
    values = await asyncio.gather(
        *(asyncio.to_thread(get_json, url) for url in urls.values()), return_exceptions=True
    )
    payloads: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name, value in zip(urls, values, strict=True):
        if isinstance(value, BaseException):
            errors[name] = type(value).__name__
        elif isinstance(value, dict):
            payloads[name] = value
        else:
            errors[name] = "invalid_payload"
    records = normalize_context(payloads, received)
    present = {str(record["exchange"]) for record in records if record["coverage"]}
    return {
        "schema_version": 1,
        "received_at": received.isoformat(),
        "records": records,
        "errors": errors,
        "coverage": {
            "binance": "binance" in present,
            "bybit": "bybit" in present,
            "okx": "okx" in present,
            "bitunix": "bitunix" in present,
            "deribit_dvol": "deribit" in present,
        },
    }


def normalize_context(
    payloads: dict[str, dict[str, Any]], received: datetime
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    premium = payloads.get("binance_premium", {})
    oi = payloads.get("binance_oi", {})
    spot = payloads.get("binance_spot", {})
    records.append(
        _record(
            "binance",
            received,
            timestamp=premium.get("time", oi.get("time")),
            mark=premium.get("markPrice"),
            index=premium.get("indexPrice"),
            funding=premium.get("lastFundingRate"),
            next_funding=premium.get("nextFundingTime"),
            open_interest=oi.get("openInterest"),
            spot=spot.get("price"),
            open_interest_unit="BTC",
            required=(
                "mark_price",
                "index_price",
                "spot_price",
                "funding_rate",
                "open_interest",
            ),
        )
    )

    bybit = _first(payloads.get("bybit", {}).get("result", {}).get("list"))
    records.append(
        _record(
            "bybit",
            received,
            timestamp=payloads.get("bybit", {}).get("time"),
            mark=bybit.get("markPrice"),
            index=bybit.get("indexPrice"),
            last=bybit.get("lastPrice"),
            funding=bybit.get("fundingRate"),
            next_funding=bybit.get("nextFundingTime"),
            open_interest=bybit.get("openInterest"),
            open_interest_usd=bybit.get("openInterestValue"),
            open_interest_unit="BTC",
            required=("mark_price", "index_price", "funding_rate", "open_interest"),
        )
    )

    okx_oi = _first(payloads.get("okx_oi", {}).get("data"))
    okx_funding = _first(payloads.get("okx_funding", {}).get("data"))
    okx_ticker = _first(payloads.get("okx_ticker", {}).get("data"))
    records.append(
        _record(
            "okx",
            received,
            timestamp=max(
                (_integer(item.get("ts")) or 0 for item in (okx_oi, okx_funding, okx_ticker)),
                default=0,
            ),
            last=okx_ticker.get("last"),
            funding=okx_funding.get("fundingRate"),
            next_funding=okx_funding.get("fundingTime"),
            open_interest=okx_oi.get("oiCcy"),
            open_interest_usd=okx_oi.get("oiUsd"),
            open_interest_unit="BTC",
            premium=okx_funding.get("premium"),
            required=("last_price", "funding_rate", "open_interest"),
        )
    )

    bitunix = payloads.get("bitunix", {}).get("data", {})
    if isinstance(bitunix, list):
        bitunix = _first(bitunix)
    if not isinstance(bitunix, dict):
        bitunix = {}
    records.append(
        _record(
            "bitunix",
            received,
            timestamp=received,
            mark=bitunix.get("markPrice"),
            index=bitunix.get("indexPrice"),
            last=bitunix.get("lastPrice"),
            funding=bitunix.get("fundingRate"),
            funding_source_unit="percent",
            next_funding=bitunix.get("nextFundingTime"),
            required=("mark_price", "index_price", "funding_rate"),
        )
    )

    dvol = payloads.get("deribit_dvol", {}).get("result", {}).get("data", [])
    point = dvol[-1] if isinstance(dvol, list) and dvol else []
    records.append(
        _record(
            "deribit",
            received,
            timestamp=point[0] if isinstance(point, list) and len(point) >= 5 else None,
            dvol=point[4] if isinstance(point, list) and len(point) >= 5 else None,
            required=("dvol_close",),
        )
    )
    return records


def _record(
    exchange: str,
    received: datetime,
    *,
    timestamp: object = None,
    mark: object = None,
    index: object = None,
    last: object = None,
    funding: object = None,
    funding_source_unit: str = "fraction",
    next_funding: object = None,
    open_interest: object = None,
    open_interest_usd: object = None,
    spot: object = None,
    open_interest_unit: str | None = None,
    premium: object = None,
    dvol: object = None,
    required: tuple[str, ...],
) -> dict[str, Any]:
    if funding_source_unit not in {"fraction", "percent"}:
        raise ValueError("funding source unit must be fraction or percent")
    funding_number = _number(funding)
    if funding_number is not None and funding_source_unit == "percent":
        funding_number = str(Decimal(funding_number) / Decimal("100"))
    values = {
        "mark_price": _number(mark),
        "index_price": _number(index),
        "last_price": _number(last),
        "funding_rate": funding_number,
        "open_interest": _number(open_interest),
        "open_interest_usd": _number(open_interest_usd),
        "spot_price": _number(spot),
        "premium_rate": _number(premium),
        "dvol_close": _number(dvol),
    }
    mark_number, index_number = values["mark_price"], values["index_price"]
    spot_number = values["spot_price"]
    basis = None
    if mark_number is not None and index_number not in (None, "0"):
        basis = str((Decimal(mark_number) / Decimal(index_number) - 1) * Decimal(10_000))
    spot_basis = None
    if mark_number is not None and spot_number not in (None, "0"):
        spot_basis = str((Decimal(mark_number) / Decimal(spot_number) - 1) * Decimal(10_000))
    missing = [name for name in required if values[name] is None]
    return {
        "schema_version": 1,
        "exchange": exchange,
        "symbol": "BTCUSDT" if exchange != "deribit" else "BTC",
        "exchange_timestamp": _timestamp(timestamp, received),
        "received_timestamp": received.isoformat(),
        "available_at": received.isoformat(),
        **values,
        "mark_index_basis_bps": basis,
        "mark_spot_basis_bps": spot_basis,
        "next_funding_timestamp": _timestamp(next_funding, received, missing_none=True),
        "funding_rate_unit": "fraction_of_notional",
        "funding_rate_source_unit": funding_source_unit,
        "open_interest_unit": open_interest_unit,
        "coverage": not missing,
        "missing_fields": missing,
    }


async def collect_context(
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    duration_hours: float = 24 * 84,
    poll_seconds: float = 60,
    get_json: JsonGetter = _get_json,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    if duration_hours <= 0 or poll_seconds <= 0:
        raise ValueError("duration and poll interval must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + duration_hours * 3600
    samples = 0
    while monotonic() < deadline:
        snapshot = await fetch_context(get_json=get_json)
        day = snapshot["received_at"][:10]
        path = output_directory / f"btc_context_{day}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        samples += 1
        _atomic_json(
            output_directory / "status.json",
            {
                "connected": not snapshot["errors"],
                "samples": samples,
                "coverage": snapshot["coverage"],
                "errors": snapshot["errors"],
                "current_file": path.name,
                "updated_at": snapshot["received_at"],
            },
        )
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_seconds, remaining))


def readiness_report() -> dict[str, Any]:
    archives: dict[str, Any] = {}
    for exchange in ("binance", "bybit", "okx", "bitunix"):
        path = ALPHA_ROOT / f"exchange={exchange}" / "symbol=BTCUSDT" / "data.parquet"
        if not path.exists():
            archives[exchange] = {"available": False}
            continue
        columns = ["timestamp", "data_valid", "funding_event_rate"]
        if exchange == "bitunix":
            columns += ["open", "close", "raw_high", "raw_low"]
        frame = pd.read_parquet(path, columns=columns)
        timestamps = pd.to_datetime(frame["timestamp"], format="mixed", utc=True)
        valid = frame["data_valid"].fillna(False).astype(bool)
        corrected_valid_rows: int | None = None
        if exchange == "bitunix":
            envelope = (
                pd.concat(
                    [
                        frame[["open", "close"]].max(axis=1) - frame["raw_high"],
                        frame["raw_low"] - frame[["open", "close"]].min(axis=1),
                    ],
                    axis=1,
                )
                .max(axis=1)
                .clip(lower=0)
            )
            corrected_valid_rows = int((envelope / frame["close"] * 10_000).le(5).sum())
        archives[exchange] = {
            "available": True,
            "path": str(path),
            "rows": len(frame),
            "days": int(timestamps.dt.floor("1D").nunique()),
            "start": timestamps.min().isoformat(),
            "end": timestamps.max().isoformat(),
            "stored_valid_rows": int(valid.sum()),
            "corrected_valid_rows_5bps": corrected_valid_rows,
            "funding_event_rows": int(
                pd.to_numeric(frame["funding_event_rate"], errors="coerce").fillna(0).ne(0).sum()
            ),
        }
    orderflow = _parquet_span(ORDERFLOW_PATH, "timestamp")
    micro_days = sorted(
        {path.stem.removeprefix("btcusdt_") for path in MICROSTRUCTURE_ROOT.glob("btcusdt_*.jsonl")}
    )
    context_days = sorted(
        path.stem.removeprefix("btc_context_")
        for path in DEFAULT_OUTPUT.glob("btc_context_*.jsonl")
    )
    execution = Path("data/ml/hybrid_v8/execution_matrix.parquet")
    execution_span = _parquet_span(execution, "signal_timestamp")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": "BTCUSDT",
        "historical_archives": archives,
        "binance_orderflow": orderflow,
        "bitunix_microstructure": {
            "days": len(micro_days),
            "distinct_dates": micro_days,
            "execution_training_ready_30_days": len(micro_days) >= 30,
        },
        "cross_venue_context": {
            "days": len(context_days),
            "distinct_dates": context_days,
            "sources": ["binance", "bybit", "okx", "bitunix", "deribit_dvol"],
        },
        "execution_matrix": execution_span
        | {
            "observed_private_orders_and_fills": False,
            "training_enabled": False,
            "reason": "no private Bitunix order/fill archive; reconstructed rows are not fills",
        },
        "known_failures": [
            "legacy Bitunix data_valid flag rejects 62k sub-5bps OHLC envelope repairs",
            "venue-by-venue one-position sampling can create unsynchronised apparent transfer edge",
            "VWAP, trend, funding-carry and Binance aggressor-flow screens fail net-cost transfer",
            "Bitunix execution matrix has one day and no private observed fills",
        ],
        "training_decision": "PRICE_ONLY_RESEARCH_AVAILABLE_CONTEXT_ENRICHMENT_NOT_READY",
    }
    _atomic_json(READINESS_REPORT, report)
    return report


def _parquet_span(path: Path, timestamp_column: str) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "rows": 0, "days": 0}
    frame = pd.read_parquet(path, columns=[timestamp_column])
    timestamps = pd.to_datetime(frame[timestamp_column], format="mixed", utc=True)
    return {
        "available": True,
        "path": str(path),
        "rows": len(frame),
        "days": int(timestamps.dt.floor("1D").nunique()),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
    }


def _first(value: object) -> dict[str, Any]:
    return value[0] if isinstance(value, list) and value and isinstance(value[0], dict) else {}


def _number(value: object) -> str | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return str(number) if number.is_finite() else None


def _integer(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _timestamp(value: object, fallback: datetime, *, missing_none: bool = False) -> str | None:
    if value is None or value == "":
        return None if missing_none else fallback.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    integer = _integer(value)
    if integer is not None:
        seconds = integer / 1000 if integer > 10_000_000_000 else integer
        return datetime.fromtimestamp(seconds, UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None if missing_none else fallback.isoformat()
    return parsed.astimezone(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BTC public context collector and readiness audit")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    collect.add_argument("--duration-hours", type=float, default=24 * 84)
    collect.add_argument("--poll-seconds", type=float, default=60)
    commands.add_parser("audit")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(readiness_report(), indent=2))
        return 0
    asyncio.run(
        collect_context(
            args.output_directory,
            duration_hours=args.duration_hours,
            poll_seconds=args.poll_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
