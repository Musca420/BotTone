from __future__ import annotations

import numpy as np
import pandas as pd


def normalized_ema_slope(
    close: pd.Series, atr_values: pd.Series, period: int = 50, lookback: int = 5
) -> pd.Series:
    ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
    denominator = atr_values.replace(0, np.nan) * lookback
    return ((ema - ema.shift(lookback)) / denominator).astype(float)
