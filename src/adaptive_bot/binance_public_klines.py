from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path("data/research/binance_public_klines")
STATUS = Path("data/reports/binance_public_klines.status.json")
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m"
COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _months(start: str, end: str) -> list[str]:
    current, stop = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    return [str(value) for value in pd.period_range(current, stop, freq="M")]


def _write_status(month: str, completed: int, total: int, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "phase": "complete" if completed == total else "download",
                "detail": detail,
                "month": month,
                "completed_months": completed,
                "total_months": total,
                "percent": round(completed / total * 100, 2),
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        payload: bytes = response.read()
        return payload


def _verified_archive(month: str) -> bytes:
    name = f"BTCUSDT-1m-{month}.zip"
    archive = ROOT / name
    checksum = ROOT / f"{name}.CHECKSUM"
    ROOT.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        payload = _download(f"{BASE}/{name}")
        temporary = archive.with_suffix(".zip.part")
        temporary.write_bytes(payload)
        temporary.replace(archive)
    if not checksum.exists():
        checksum.write_bytes(_download(f"{BASE}/{name}.CHECKSUM"))
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {name}")
    return payload


def _parse(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.namelist()[0]
        frame = pd.read_csv(archive.open(member), header=None, names=COLUMNS)
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.dropna(subset=["open_time"]).copy()
    frame["timestamp"] = pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True)
    numeric = [column for column in COLUMNS[1:] if column != "ignore"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["taker_sell_volume"] = frame["volume"] - frame["taker_buy_volume"]
    frame["taker_imbalance"] = (
        (frame["taker_buy_volume"] - frame["taker_sell_volume"])
        / frame["volume"].replace(0, pd.NA)
    )
    return frame.drop(columns=["close_time", "ignore"])


def run(start: str = "2024-04", end: str = "2026-07") -> Path:
    months = _months(start, end)
    outputs: list[Path] = []
    for index, month in enumerate(months, 1):
        output = ROOT / f"BTCUSDT-1m-{month}.parquet"
        if not output.exists():
            frame = _parse(_verified_archive(month))
            temporary = output.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(output)
        outputs.append(output)
        _write_status(month, index, len(months), f"{month}: verified")
    combined = ROOT / "BTCUSDT-1m-2024-04_2026-07.parquet"
    frame = pd.concat((pd.read_parquet(path) for path in outputs), ignore_index=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    temporary = combined.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(combined)
    _write_status(end, len(months), len(months), f"complete: {len(frame):,} rows")
    return combined


if __name__ == "__main__":
    print(run())
