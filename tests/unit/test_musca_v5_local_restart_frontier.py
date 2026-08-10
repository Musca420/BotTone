from __future__ import annotations

import pandas as pd

from adaptive_bot.musca_v5_local_restart_frontier import (
    PROTOCOL_HASH,
    build_features,
    label_events,
)


def _minute(at: pd.Timestamp, **prices: float | bool) -> dict[str, object]:
    return {
        "timestamp": at,
        "data_valid": True,
        "perp_open": 100.0,
        "perp_high": 100.5,
        "perp_low": 99.5,
        "perp_close": 100.0,
        "perp_volume": 2.0,
        "perp_quote_volume": 200.0,
        "perp_trade_count": 2,
        "perp_taker_buy_quote": 100.0,
        "spot_open": 100.0,
        "spot_high": 100.5,
        "spot_low": 99.5,
        "spot_close": 100.0,
        "spot_volume": 2.0,
        "spot_quote_volume": 200.0,
        "spot_trade_count": 2,
        "spot_taker_buy_quote": 100.0,
        "perp_daily_vwap": 100.0,
        "funding_event_rate": 0.0,
    } | prices


def test_features_are_available_only_after_closed_minute() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    features = build_features(pd.DataFrame([_minute(at)]))

    assert features.loc[0, "available_at"] == at + pd.Timedelta(minutes=1)
    assert features.loc[0, "direction"] == 0
    assert len(PROTOCOL_HASH) == 64


def test_local_restart_same_minute_stop_beats_target() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            {
                "available_at": at,
                "direction": 1,
                "stop_price": 99.5,
                "target_price": 100.5,
                "operating_vwap": 100.0,
            }
        ]
    )
    minutes = pd.DataFrame([_minute(at, perp_open=100.0, perp_high=101.5, perp_low=98.5)])

    result = label_events(events, minutes)

    assert result.loc[0, "exit_reason"] == "STRUCTURAL_STOP"
    assert result.loc[0, "gross_market_return_bps"] == -50.0


def test_vwap_invalidation_executes_at_next_minute_open() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            {
                "available_at": at,
                "direction": 1,
                "stop_price": 99.5,
                "target_price": 103.0,
                "operating_vwap": 100.0,
            }
        ]
    )
    minutes = pd.DataFrame(
        [
            _minute(at, perp_low=99.6, perp_close=99.8),
            _minute(at + pd.Timedelta(minutes=1), perp_open=99.7, perp_low=99.6),
        ]
    )

    result = label_events(events, minutes)

    assert result.loc[0, "exit_reason"] == "VWAP_INVALIDATION"
    assert result.loc[0, "exit_timestamp"] == at + pd.Timedelta(minutes=1)
    assert result.loc[0, "exit_price"] == 99.7


def test_hold_exit_ignores_recross_and_can_reach_target() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            {
                "available_at": at,
                "direction": 1,
                "stop_price": 99.5,
                "target_price": 103.0,
                "operating_vwap": 100.0,
            }
        ]
    )
    minutes = pd.DataFrame(
        [
            _minute(at, perp_low=99.6, perp_close=99.8),
            _minute(
                at + pd.Timedelta(minutes=1),
                perp_open=99.7,
                perp_high=103.2,
                perp_low=99.6,
            ),
        ]
    )

    result = label_events(events, minutes, invalidate_on_vwap=False)

    assert result.loc[0, "exit_reason"] == "IMPULSE_EXTREME_TARGET"
    assert result.loc[0, "exit_price"] == 103.0
