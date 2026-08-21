from __future__ import annotations

import pandas as pd

from adaptive_bot.musca_v5_local_restart_model import (
    FEATURES,
    PROTOCOL_HASH,
    _selected,
    prepare,
)


def _candidate(entry_price: float) -> dict[str, object]:
    at = pd.Timestamp("2024-08-01T12:00:00Z")
    return {
        "signal_timestamp": at,
        "entry_timestamp": at + pd.Timedelta(minutes=1),
        "exit_timestamp": at + pd.Timedelta(minutes=5),
        "entry_price": entry_price,
        "direction": 1,
        "expert_breakout_bars": 5,
        "trend_score": 4.0,
        "return_15m": 0.001,
        "return_60m": 0.002,
        "spot_return_15m": 0.001,
        "vwap_60m_slope": 0.2,
        "relative_volume": 1.2,
        "perp_taker_1m": 0.1,
        "perp_taker_5m": 0.08,
        "spot_taker_5m": 0.04,
        "pullback_depth_atr": 0.4,
        "risk_bps_at_signal": 20.0,
        "room_bps": 100.0,
        "target_price": 101.0,
        "operating_vwap": 99.9,
        "exit_reason": "IMPULSE_EXTREME_TARGET",
        "gross_return_bps": 30.0,
    }


def test_model_features_do_not_use_next_minute_entry_price() -> None:
    first = prepare(pd.DataFrame([_candidate(100.0)]))
    changed_future_fill = prepare(pd.DataFrame([_candidate(120.0)]))

    assert first.loc[0, list(FEATURES)].equals(changed_future_fill.loc[0, list(FEATURES)])
    assert len(PROTOCOL_HASH) == 64


def test_flat_wins_when_predicted_gross_does_not_cover_cost() -> None:
    rows = pd.DataFrame(
        [
            _candidate(100.0) | {"predicted_gross_bps": 7.0},
            _candidate(100.0)
            | {
                "signal_timestamp": pd.Timestamp("2024-08-01T12:10:00Z"),
                "entry_timestamp": pd.Timestamp("2024-08-01T12:11:00Z"),
                "exit_timestamp": pd.Timestamp("2024-08-01T12:15:00Z"),
                "predicted_gross_bps": 10.0,
            },
        ]
    )

    selected = _selected(rows, cost=8.0, threshold=9.0)

    assert len(selected) == 1
    assert selected.iloc[0]["predicted_gross_bps"] == 10.0
