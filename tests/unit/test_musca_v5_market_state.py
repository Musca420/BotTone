from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot.musca_v5_market_state import TIMEFRAMES, _multi_timeframe


def test_multi_timeframe_state_is_available_only_after_each_bar_closes() -> None:
    timestamp = pd.date_range("2026-01-01", periods=1_000, freq="1min", tz="UTC")
    close = pd.Series(100 + np.linspace(0, 2, len(timestamp)))
    minutes = pd.DataFrame(
        {
            "timestamp": timestamp,
            "is_available": True,
            "perp_open": close,
            "perp_high": close + 0.1,
            "perp_low": close - 0.1,
            "perp_close": close,
            "perp_volume": 1.0,
            "perp_quote_volume": close,
        }
    )
    result = _multi_timeframe(minutes).dropna()
    assert not result.empty
    changed_30m = result["atr_30m_bps"].ne(result["atr_30m_bps"].shift())
    assert result.loc[changed_30m, "available_at"].dt.minute.mod(30).eq(0).all()
    for timeframe in TIMEFRAMES:
        assert np.isfinite(result[f"atr_{timeframe}m_bps"]).all()
