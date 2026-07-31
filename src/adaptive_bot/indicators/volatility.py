from __future__ import annotations

import numpy as np
import pandas as pd


def atr_percentile(values: pd.Series, window: int = 100) -> pd.Series:
    def percentile(sample: np.ndarray) -> float:
        current = sample[-1]
        return float(np.count_nonzero(sample <= current) * 100 / len(sample))

    return values.rolling(window, min_periods=window).apply(percentile, raw=True)


def atr_change(values: pd.Series) -> pd.Series:
    return values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def cumulative_move(close: pd.Series, atr_values: pd.Series, bars: int = 3) -> pd.Series:
    return (close.diff(bars) / atr_values.replace(0, np.nan)).astype(float)


def normalized_distance(close: pd.Series, center: pd.Series, atr_values: pd.Series) -> pd.Series:
    return ((close - center) / atr_values.replace(0, np.nan)).astype(float)
