import math

import pandas as pd

from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.volatility import normalized_distance
from adaptive_bot.indicators.vwap import rolling_vwap, session_vwap


def test_atr_uses_wilder_seed_and_smoothing() -> None:
    high = pd.Series([10, 12, 13, 15], dtype=float)
    low = pd.Series([8, 9, 10, 12], dtype=float)
    close = pd.Series([9, 11, 12, 14], dtype=float)
    result = atr(high, low, close, period=3)
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(8 / 3)
    assert result.iloc[3] == pytest.approx(25 / 9)


def test_vwap_session_reset_and_rolling_window() -> None:
    high = pd.Series([11, 13, 21, 23], dtype=float)
    low = pd.Series([9, 11, 19, 21], dtype=float)
    close = pd.Series([10, 12, 20, 22], dtype=float)
    volume = pd.Series([1, 3, 2, 2], dtype=float)
    sessions = pd.Series(["a", "a", "b", "b"])
    session = session_vwap(high, low, close, volume, sessions)
    rolling = rolling_vwap(high, low, close, volume, window=2)
    assert session.tolist() == pytest.approx([10, 11.5, 20, 21])
    assert math.isnan(rolling.iloc[0])
    assert rolling.iloc[1] == pytest.approx(11.5)


def test_adx_detects_unidirectional_trend_without_lookahead() -> None:
    close = pd.Series(range(1, 31), dtype=float)
    result = adx(close + 1, close - 1, close, period=3)
    assert result["plus_di"].iloc[-1] > result["minus_di"].iloc[-1]
    assert result["adx"].iloc[-1] == pytest.approx(100)
    shortened = adx((close + 1)[:-1], (close - 1)[:-1], close[:-1], period=3)
    assert result["adx"].iloc[-2] == shortened["adx"].iloc[-1]


def test_normalized_distance_handles_zero_atr() -> None:
    result = normalized_distance(pd.Series([10.0]), pd.Series([9.0]), pd.Series([0.0]))
    assert math.isnan(result.iloc[0])


import pytest  # noqa: E402
