from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from adaptive_bot.musca_v4 import _execution_avwap, _position_quantity
from adaptive_bot.musca_v4_research import (
    PROTOCOL,
    PROTOCOL_HASH,
    anchored_vwap_band,
    build_events,
    build_features,
)


def test_official_bitunix_volume_units_produce_trade_vwap() -> None:
    center, sigma = anchored_vwap_band(np.array([1.0]), np.array([60_000.0]), 0, 0)
    assert center == 60_000
    assert sigma == 0


def test_execution_avwap_is_cumulative_from_anchor_without_future() -> None:
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "volume": [1.0, 2.0, 100.0],
            "quote_volume": [100.0, 220.0, 50_000.0],
        }
    )
    value = _execution_avwap(candles, candles.loc[0, "timestamp"], candles.loc[1, "timestamp"])
    assert value == Decimal("320") / Decimal("3")


def test_execution_avwap_rejects_partial_anchor_history() -> None:
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-06T10:05:00Z"]),
            "volume": [2.0],
            "quote_volume": [204.0],
        }
    )

    with pytest.raises(ValueError, match="anchor predates"):
        _execution_avwap(
            candles,
            "2026-08-06T10:00:00Z",
            pd.Timestamp("2026-08-06T10:05:00Z"),
        )


def test_features_are_available_only_after_bar_close() -> None:
    timestamp = pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(timestamp)))
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "perp_open": close,
            "perp_high": close + 1,
            "perp_low": close - 1,
            "perp_close": close,
            "perp_volume": 2.0,
            "perp_quote_volume": close * 2,
            "perp_taker_buy_quote": close * 1.2,
            "spot_open": close,
            "spot_high": close + 1,
            "spot_low": close - 1,
            "spot_close": close,
            "spot_volume": 1.0,
            "spot_quote_volume": close,
            "spot_taker_buy_quote": close * 0.6,
        }
    )
    features = build_features(frame)
    assert (features["available_at"] == features["timestamp"] + pd.Timedelta(minutes=5)).all()
    assert PROTOCOL_HASH and PROTOCOL["entry"].startswith("next_1m")


def test_risk_sizing_respects_ten_x_leverage_cap() -> None:
    quantity = _position_quantity(Decimal("10000"), Decimal("100"), Decimal("99.99"))
    assert quantity * Decimal("100") == Decimal("100000")


def test_no_vwap_event_is_a_valid_empty_result() -> None:
    timestamp = pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC")
    close = pd.Series(np.full(len(timestamp), 100.0))
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "perp_open": close,
            "perp_high": close,
            "perp_low": close,
            "perp_close": close,
            "perp_volume": 1.0,
            "perp_quote_volume": 100.0,
            "perp_taker_buy_quote": 50.0,
            "spot_open": close,
            "spot_high": close,
            "spot_low": close,
            "spot_close": close,
            "spot_volume": 1.0,
            "spot_quote_volume": 100.0,
            "spot_taker_buy_quote": 50.0,
        }
    )
    audit: dict[str, int] = {}
    events = build_events(build_features(frame), audit)
    assert events.empty
    assert audit["event"] == 0
    assert {"signal_timestamp", "available_at", "event_family"} <= set(events.columns)
