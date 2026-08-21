from __future__ import annotations

import argparse
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/raw/binance_book_depth/BTCUSDT")
FEATURES = Path("data/ml/musca_v5/binance_book_depth_features.parquet")
BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT"
PERCENTAGES = (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0)
OPTIONAL_PERCENTAGES = (-0.2, 0.2)


def _download_day(day: date) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"BTCUSDT-bookDepth-{day.isoformat()}.zip"
    archive = ROOT / stem
    checksum_path = ROOT / f"{stem}.CHECKSUM"
    for path in (archive, checksum_path):
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            urllib.request.urlretrieve(f"{BASE_URL}/{path.name}", temporary)
            temporary.replace(path)
    expected = checksum_path.read_text(encoding="utf-8").split()[0].lower()
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {archive.name}")
    return archive


def download_range(start: str, end: str, *, workers: int = 8) -> list[Path]:
    days = [value.date() for value in pd.date_range(start, end, inclusive="left")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_download_day, days))


def build_features_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "percentage", "depth", "notional"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Book-depth data misses {sorted(required - set(rows.columns))}")
    data = rows.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in ("percentage", "depth", "notional"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = (
        data["timestamp"].notna()
        & data["percentage"].isin((*PERCENTAGES, *OPTIONAL_PERCENTAGES))
        & data["depth"].gt(0)
        & data["notional"].gt(0)
    )
    if not valid.all():
        raise ValueError("Invalid official Binance book-depth row")
    required_rows = data.loc[data["percentage"].isin(PERCENTAGES)]
    counts = required_rows.groupby("timestamp")["percentage"].nunique()
    complete = counts[counts.eq(len(PERCENTAGES))].index
    data = data.loc[data["timestamp"].isin(complete)]
    depth = data.pivot(index="timestamp", columns="percentage", values="depth")
    notional = data.pivot(index="timestamp", columns="percentage", values="notional")
    output = pd.DataFrame(index=depth.index)
    for band in (1.0, 2.0, 5.0):
        bid = depth[-band]
        ask = depth[band]
        output[f"bid_depth_{band:g}pct"] = bid
        output[f"ask_depth_{band:g}pct"] = ask
        output[f"depth_imbalance_{band:g}pct"] = (bid - ask) / (bid + ask)
    implied_bid = notional[-1.0] / depth[-1.0]
    implied_ask = notional[1.0] / depth[1.0]
    output["implied_mid"] = (implied_bid + implied_ask) / 2
    output["depth_skew_change_1m"] = output["depth_imbalance_1pct"].diff(2)
    output["depth_skew_change_5m"] = output["depth_imbalance_1pct"].diff(10)
    output["snapshot_timestamp"] = output.index
    output["available_at"] = output.index + pd.Timedelta(seconds=1)
    feature_values = output.drop(columns=["snapshot_timestamp", "available_at"])
    output["coverage_valid"] = np.isfinite(feature_values).all(axis=1)
    return output.reset_index(drop=True)


def materialize(start: str, end: str, *, workers: int = 8) -> pd.DataFrame:
    archives = download_range(start, end, workers=workers)
    frames = [pd.read_csv(path) for path in sorted(archives)]
    features = build_features_from_rows(pd.concat(frames, ignore_index=True))
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEATURES.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False)
    temporary.replace(FEATURES)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Binance USD-M book-depth features")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive UTC date")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = materialize(args.start, args.end, workers=args.workers)
    print(f"{len(result):,} validated BTCUSDT depth snapshots -> {FEATURES}")


if __name__ == "__main__":
    main()
