from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("data/ml/musca_v5/aggtrades")
STATUS = Path("data/reports/musca_v5_microstructure.status.json")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT"
MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 8))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(month: str, phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "month": month,
            "phase": phase,
            "percent": round(percent, 2),
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _download(month: str) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / f"BTCUSDT-aggTrades-{month}.zip"
    url = f"{BASE_URL}/{target.name}"
    with urllib.request.urlopen(f"{url}.CHECKSUM", timeout=60) as response:
        expected = response.read().decode("ascii").split()[0].lower()
    if target.exists() and _sha256(target) == expected:
        return target
    temporary = target.with_suffix(".zip.part")
    digest = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(url, headers={"User-Agent": "musca-v5-research/1"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            _status(
                month,
                "download",
                100 * downloaded / max(total, downloaded),
                f"{downloaded / 1024**3:.2f}/{total / 1024**3:.2f} GB",
            )
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Binance checksum mismatch for {month}")
    temporary.replace(target)
    return target


def _aggregate(path: Path, month: str, *, seconds: int = 5) -> pd.DataFrame:
    if seconds not in {1, 5}:
        raise ValueError("Supported aggregation intervals are one and five seconds")
    output = ROOT / f"BTCUSDT-aggTrades-{seconds}s-{month}.parquet"
    if output.exists():
        cached = pd.read_parquet(output)
        if {"base_volume", "open"}.issubset(cached.columns):
            return cached
    pieces: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise RuntimeError(f"unexpected Binance archive layout: {path}")
        reader = pd.read_csv(
            archive.open(members[0]),
            header=None,
            names=["id", "price", "quantity", "first", "last", "timestamp", "buyer_maker"],
            dtype=str,
            chunksize=1_000_000,
        )
        for number, chunk in enumerate(reader, start=1):
            timestamp_number = pd.to_numeric(chunk["timestamp"], errors="coerce")
            price = pd.to_numeric(chunk["price"], errors="coerce")
            quantity = pd.to_numeric(chunk["quantity"], errors="coerce")
            valid = timestamp_number.notna() & price.notna() & quantity.notna()
            chunk = chunk.loc[valid].copy()
            timestamp = pd.to_datetime(timestamp_number.loc[valid], unit="ms", utc=True)
            bucket = timestamp.dt.floor(f"{seconds}s")
            price_values = price.loc[valid].to_numpy(float)
            quote = price_values * quantity.loc[valid].to_numpy(float)
            buyer_maker = chunk["buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
            buyer_taker = ~buyer_maker
            frame = pd.DataFrame(
                {
                    "timestamp": bucket,
                    "base_volume": quantity.loc[valid].to_numpy(float),
                    "quote_volume": quote,
                    "signed_quote_volume": np.where(buyer_taker, quote, -quote),
                    "trade_count": 1,
                    "buy_count": buyer_taker.astype(int),
                    "price": price_values,
                }
            )
            pieces.append(
                frame.groupby("timestamp", as_index=False).agg(
                    quote_volume=("quote_volume", "sum"),
                    signed_quote_volume=("signed_quote_volume", "sum"),
                    trade_count=("trade_count", "sum"),
                    buy_count=("buy_count", "sum"),
                    base_volume=("base_volume", "sum"),
                    open=("price", "first"),
                    high=("price", "max"),
                    low=("price", "min"),
                    close=("price", "last"),
                )
            )
            _status(month, "aggregate", 0, f"chunk {number}")
    result = (
        pd.concat(pieces, ignore_index=True)
        .groupby("timestamp", as_index=False)
        .agg(
            quote_volume=("quote_volume", "sum"),
            base_volume=("base_volume", "sum"),
            signed_quote_volume=("signed_quote_volume", "sum"),
            trade_count=("trade_count", "sum"),
            buy_count=("buy_count", "sum"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .sort_values("timestamp")
    )
    result["available_at"] = result["timestamp"] + pd.Timedelta(seconds=seconds)
    temporary = output.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(output)
    return result


def build() -> dict[str, Any]:
    audit: dict[str, Any] = {"source": "Binance official aggTrades", "months": {}}
    for month in MONTHS:
        archive = _download(month)
        frame = _aggregate(archive, month)
        one_second = _aggregate(archive, month, seconds=1)
        audit["months"][month] = {
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "buckets_5s": len(frame),
            "buckets_1s": len(one_second),
            "start": frame["timestamp"].min().isoformat(),
            "end": frame["timestamp"].max().isoformat(),
        }
    _status(MONTHS[-1], "complete", 100, "verified 1s and 5s aggTrades ready")
    _atomic_json(Path("data/reports/musca_v5_microstructure_audit.json"), audit)
    return audit


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
