from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_v5_binance_micro_ranker as policy


def test_protocol_is_binance_only_and_keeps_july_sealed() -> None:
    assert policy.PROTOCOL["cross_exchange_features"] is False
    assert policy.RAW_MONTHS[-1] == "2026-06"
    assert pd.Timestamp("2026-07-01T00:00:00Z") == policy.HOLDOUT_START


def test_timestamp_units_are_normalized_to_nanoseconds() -> None:
    microseconds = pd.Series(pd.to_datetime(["2026-01-01"], utc=True).as_unit("us"))
    nanoseconds = pd.Series(pd.to_datetime(["2026-01-01"], utc=True).as_unit("ns"))

    assert policy._nanoseconds(microseconds).tolist() == policy._nanoseconds(
        nanoseconds
    ).tolist()


def test_dynamic_plan_keeps_target_above_cost_and_stop_bounded() -> None:
    atr = np.array([1.0, 10.0, 100.0])
    target, stop = policy._plan_levels(atr, policy.PLANS[0])

    assert (target >= 2 * policy.COST_BPS).all()
    assert (stop >= policy.PLANS[0].minimum_stop_bps).all()
    assert (stop <= policy.PLANS[0].maximum_stop_bps).all()


def test_same_bar_stop_wins_over_target() -> None:
    timestamp = pd.date_range("2026-01-01", periods=40, freq="5s", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": np.full(40, 100.0),
            "high": np.r_[101.0, np.full(39, 100.0)],
            "low": np.r_[99.0, np.full(39, 100.0)],
            "close": np.full(40, 100.0),
        }
    )

    outcome, gross, _ = policy._barrier_outcomes(
        entry=np.array([100.0]),
        entry_indexes=np.array([0]),
        side=1,
        target=np.array([20.0]),
        stop=np.array([12.0]),
        horizon_seconds=180,
        raw=raw,
    )

    assert outcome.tolist() == [1]
    assert gross.tolist() == [-12.0]


def test_execution_accepts_losses_and_enforces_one_position() -> None:
    rows = pd.DataFrame(
        [
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:00:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:05:00Z"),
                "score": 2.0,
                "net_bps": -5.0,
            },
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:01:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:02:00Z"),
                "score": 3.0,
                "net_bps": 20.0,
            },
        ]
    )

    trades = policy.execute(rows, threshold=0.0)

    assert trades["net_bps"].tolist() == [-5.0]
