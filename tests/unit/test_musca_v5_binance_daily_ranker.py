from __future__ import annotations

import pandas as pd

from adaptive_bot import musca_v5_binance_daily_ranker as ranker


def test_execute_uses_fixed_threshold_and_one_position() -> None:
    rows = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "entry_timestamp": pd.Timestamp("2026-01-01T00:01:00Z"),
                "exit_timestamp": pd.Timestamp("2026-01-01T00:10:00Z"),
                "direction": 1,
                "score": 2.0,
                "plan": "A",
                "net_return_bps": -5.0,
            },
            {
                "signal_timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "entry_timestamp": pd.Timestamp("2026-01-01T00:01:00Z"),
                "exit_timestamp": pd.Timestamp("2026-01-01T00:05:00Z"),
                "direction": 1,
                "score": 1.0,
                "plan": "B",
                "net_return_bps": 20.0,
            },
            {
                "signal_timestamp": pd.Timestamp("2026-01-01T00:02:00Z"),
                "entry_timestamp": pd.Timestamp("2026-01-01T00:03:00Z"),
                "exit_timestamp": pd.Timestamp("2026-01-01T00:08:00Z"),
                "direction": -1,
                "score": 3.0,
                "plan": "A",
                "net_return_bps": 20.0,
            },
            {
                "signal_timestamp": pd.Timestamp("2026-01-01T00:11:00Z"),
                "entry_timestamp": pd.Timestamp("2026-01-01T00:12:00Z"),
                "exit_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "direction": 1,
                "score": 0.5,
                "plan": "A",
                "net_return_bps": 10.0,
            },
        ]
    )

    trades = ranker.execute(rows, threshold=0.75)

    assert trades["plan"].tolist() == ["A"]
    assert trades["net_return_bps"].tolist() == [-5.0]


def test_metrics_accept_losing_trades_when_portfolio_is_positive() -> None:
    trades = pd.DataFrame(
        {
            "day": [pd.Timestamp("2026-01-01T00:00:00Z")] * 2,
            "net_return_bps": [-10.0, 30.0],
        }
    )

    value = ranker.metrics(trades, calendar_days=1)

    assert value["trades"] == 2
    assert value["expectancy_bps"] == 10.0
    assert value["win_rate"] == 0.5
    assert value["positive_active_days"] == 1.0


def test_gates_fail_closed_without_trades() -> None:
    value = ranker.metrics(pd.DataFrame(), calendar_days=30)

    assert not any(ranker._gates(value).values())
