from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_bot.musca_v4_research import anchored_vwap_band
from adaptive_bot.musca_v5_touch_entry_frontier import (
    _vwap_band_from_prefixes,
    _vwap_bands_from_prefixes,
    _vwap_prefixes,
    label_touch_events,
)


def test_prefix_vwap_band_matches_reference() -> None:
    volume = np.array([2.0, 0.0, 3.0, 4.0, np.nan])
    quote = np.array([20.0, 0.0, 36.0, 44.0, np.nan])

    expected = anchored_vwap_band(volume, quote, 0, 3)
    actual = _vwap_band_from_prefixes(_vwap_prefixes(volume, quote), 0, 3)

    assert actual == pytest.approx(expected)


def test_vector_prefix_vwap_bands_match_scalar_queries() -> None:
    volume = np.array([2.0, 1.0, 3.0, 4.0])
    quote = np.array([20.0, 11.0, 36.0, 44.0])
    prefixes = _vwap_prefixes(volume, quote)
    ends = np.array([1, 2, 3])

    centers, sigmas = _vwap_bands_from_prefixes(prefixes, 1, ends)
    expected = [_vwap_band_from_prefixes(prefixes, 1, int(end)) for end in ends]

    assert centers == pytest.approx([item[0] for item in expected])
    assert sigmas == pytest.approx([item[1] for item in expected])


def test_touch_entry_same_minute_stop_beats_target() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            {
                "available_at": at,
                "direction": 1,
                "stop_price": 99.0,
                "target_price": 101.0,
                "room_threshold_bps": 10.0,
            }
        ]
    )
    minutes = pd.DataFrame(
        [
            {
                "timestamp": at,
                "data_valid": True,
                "perp_open": 100.0,
                "perp_high": 101.5,
                "perp_low": 98.5,
                "perp_close": 100.5,
                "funding_event_rate": 0.0,
            }
        ]
    )

    result = label_touch_events(events, minutes)

    assert result.loc[0, "exit_reason"] == "STRUCTURAL_STOP"
    assert result.loc[0, "gross_market_return_bps"] == -100.0
