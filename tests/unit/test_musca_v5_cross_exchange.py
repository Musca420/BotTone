from __future__ import annotations

import pandas as pd
import pytest

from adaptive_bot.musca_v5_cross_exchange import VENUES, build_features


def test_cross_exchange_features_use_completed_minutes_and_only_btc_frames() -> None:
    timestamp = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    frames = {
        venue: pd.DataFrame(
            {"timestamp": timestamp, "close": [100 + index for index in range(20)]}
        )
        for venue in VENUES
    }
    result = build_features(frames)
    assert (result["available_at"] == result["timestamp"] + pd.Timedelta(minutes=1)).all()
    assert bool(result["coverage_valid"].iloc[-1])
    assert result["dispersion_return_5m_bps"].iloc[-1] == pytest.approx(0)
