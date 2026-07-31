from __future__ import annotations

import numpy as np
import pandas as pd


def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high + low + close) / 3


def session_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    sessions: pd.Series,
) -> pd.Series:
    price_volume = typical_price(high, low, close) * volume
    cumulative_volume = volume.groupby(sessions).cumsum()
    result = price_volume.groupby(sessions).cumsum() / cumulative_volume.replace(0, np.nan)
    return result.astype(float)


def rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 96,
) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least 2")
    price_volume = typical_price(high, low, close) * volume
    denominator = volume.rolling(window, min_periods=window).sum().replace(0, np.nan)
    return (price_volume.rolling(window, min_periods=window).sum() / denominator).astype(float)
