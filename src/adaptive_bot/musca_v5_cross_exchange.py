from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/ml/hybrid_v7/alpha_raw")
OUTPUT = Path("data/ml/musca_v5/cross_exchange_features.parquet")
VENUES = ("binance", "bybit", "okx")
CROSS_FEATURES = (
    "median_return_1m_bps",
    "dispersion_return_1m_bps",
    "binance_lag_1m_bps",
    "median_return_5m_bps",
    "dispersion_return_5m_bps",
    "binance_lag_5m_bps",
    "median_return_15m_bps",
    "dispersion_return_15m_bps",
    "binance_lag_15m_bps",
    "bybit_basis_to_binance_bps",
    "okx_basis_to_binance_bps",
)


def _source(venue: str) -> Path:
    return ROOT / f"exchange={venue}" / "symbol=BTCUSDT" / "data.parquet"


def build_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for venue in VENUES:
        frame = frames[venue][["timestamp", "close"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.rename(columns={"close": f"close_{venue}"})
        merged = frame if merged is None else merged.merge(
            frame, on="timestamp", how="inner", validate="one_to_one"
        )
    if merged is None or merged.empty:
        raise ValueError("BTC cross-exchange source is empty")
    for venue in VENUES:
        for minutes in (1, 5, 15):
            merged[f"return_{minutes}m_{venue}_bps"] = (
                merged[f"close_{venue}"].pct_change(minutes) * 10_000
            )
    for minutes in (1, 5, 15):
        columns = [f"return_{minutes}m_{venue}_bps" for venue in VENUES]
        merged[f"median_return_{minutes}m_bps"] = merged[columns].median(axis=1)
        merged[f"dispersion_return_{minutes}m_bps"] = merged[columns].std(axis=1)
        merged[f"binance_lag_{minutes}m_bps"] = (
            merged[[f"return_{minutes}m_bybit_bps", f"return_{minutes}m_okx_bps"]]
            .median(axis=1)
            .sub(merged[f"return_{minutes}m_binance_bps"])
        )
    merged["bybit_basis_to_binance_bps"] = (
        merged["close_bybit"] / merged["close_binance"] - 1
    ) * 10_000
    merged["okx_basis_to_binance_bps"] = (
        merged["close_okx"] / merged["close_binance"] - 1
    ) * 10_000
    merged["available_at"] = merged["timestamp"] + pd.Timedelta(minutes=1)
    merged["coverage_valid"] = np.isfinite(merged[list(CROSS_FEATURES)]).all(axis=1)
    return merged[["timestamp", "available_at", *CROSS_FEATURES, "coverage_valid"]]


def materialize() -> pd.DataFrame:
    frames = {venue: pd.read_parquet(_source(venue)) for venue in VENUES}
    features = build_features(frames)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)
    return features


if __name__ == "__main__":
    result = materialize()
    print(f"{len(result):,} causal BTC cross-exchange minutes -> {OUTPUT}")
