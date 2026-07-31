from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot.indicators.atr import true_range, wilder_average


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up = high.diff().to_numpy(dtype=float)
    down = (-low.diff()).to_numpy(dtype=float)
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    smooth_tr = wilder_average(true_range(high, low, close), period)
    smooth_plus = wilder_average(plus_dm, period)
    smooth_minus = wilder_average(minus_dm, period)
    plus_di = 100 * smooth_plus / smooth_tr.replace(0, np.nan)
    minus_di = 100 * smooth_minus / smooth_tr.replace(0, np.nan)
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    return pd.DataFrame(
        {"adx": wilder_average(dx, period), "plus_di": plus_di, "minus_di": minus_di}
    )
