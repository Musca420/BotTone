from __future__ import annotations

import pandas as pd
import pytest

from adaptive_bot.musca_v5_room_frontier import (
    PROTOCOL,
    _select_expert_rows,
    _sized_equity_metrics,
)


def test_room_frontier_prefers_prior_robust_expert_and_locks_position() -> None:
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    rows = pd.DataFrame(
        [
            {
                "signal_timestamp": start,
                "entry_timestamp": start,
                "exit_timestamp": start + pd.Timedelta(minutes=10),
                "direction": 1,
                "event_family": "IMPULSE_PULLBACK",
                "expert_breakout_bars": 3,
            },
            {
                "signal_timestamp": start,
                "entry_timestamp": start,
                "exit_timestamp": start + pd.Timedelta(minutes=5),
                "direction": 1,
                "event_family": "IMPULSE_PULLBACK",
                "expert_breakout_bars": 6,
            },
            {
                "signal_timestamp": start + pd.Timedelta(minutes=5),
                "entry_timestamp": start + pd.Timedelta(minutes=5),
                "exit_timestamp": start + pd.Timedelta(minutes=6),
                "direction": 1,
                "event_family": "IMPULSE_PULLBACK",
                "expert_breakout_bars": 3,
            },
        ]
    )

    selected = _select_expert_rows(
        [
            ("IMPULSE_PULLBACK", 3, 2.0),
            ("IMPULSE_PULLBACK", 6, 1.0),
        ],
        rows,
    )

    assert selected["expert_key"].tolist() == ["IMPULSE_PULLBACK:H3"]
    assert PROTOCOL["changes_to_active_paper"] is False
    assert PROTOCOL["holdout_opened"] is False


def test_sizing_caps_notional_before_applying_one_percent_risk_budget() -> None:
    rows = pd.DataFrame(
        {
            "signal_timestamp": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "entry_timestamp": [pd.Timestamp("2025-01-01T00:01:00Z")],
            "direction": [1],
            "entry_price": [100.0],
            "stop_price": [99.5],
            "net_return_bps": [-50.0],
        }
    )

    metrics = _sized_equity_metrics(rows, "net_return_bps")

    assert metrics["maximum_notional_fraction"] == 1.0
    assert metrics["max_drawdown"] == pytest.approx(0.005)
