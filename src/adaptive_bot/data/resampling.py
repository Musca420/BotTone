from __future__ import annotations

import pandas as pd


def resample_ohlcv(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    indexed = frame.set_index(pd.to_datetime(frame["timestamp"], utc=True))
    result = indexed.resample(f"{minutes}min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return result.dropna(subset=["open", "high", "low", "close"]).reset_index(names="timestamp")
