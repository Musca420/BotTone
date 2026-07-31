from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def wilder_average(values: pd.Series, period: int) -> pd.Series:
    if period < 2:
        raise ValueError("period must be at least 2")
    data = values.astype(float).to_numpy()
    output = np.full(len(data), np.nan)
    valid = np.flatnonzero(~np.isnan(data))
    if len(valid) < period:
        return pd.Series(output, index=values.index, dtype=float)
    seed_end = int(valid[period - 1])
    seed_values = data[valid[:period]]
    output[seed_end] = float(seed_values.mean())
    for index in range(seed_end + 1, len(data)):
        if np.isnan(data[index]):
            continue
        previous = output[index - 1]
        if np.isnan(previous):
            continue
        output[index] = (previous * (period - 1) + data[index]) / period
    return pd.Series(output, index=values.index, dtype=float)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_average(true_range(high, low, close), period)
