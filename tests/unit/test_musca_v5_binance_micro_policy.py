from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_v5_binance_micro_policy as micro


def test_label_enters_after_availability_and_same_bar_stop_wins() -> None:
    index = pd.date_range("2026-01-01", periods=370, freq="5s", tz="UTC")
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        },
        index=index,
    )
    entry_position = 2
    data.iloc[entry_position, data.columns.get_loc("high")] = 101.0
    data.iloc[entry_position, data.columns.get_loc("low")] = 99.0
    available_at = index[1]
    candidates = pd.DataFrame(
        [
            {
                "available_at": available_at,
                "decision_at": available_at,
                "side": 1,
                "target_bps": 20.0,
                "stop_bps": 12.0,
                "family": "VWAP_RECLAIM",
            }
        ],
        index=[index[0]],
    )

    labeled = micro.label_candidates(candidates, data)

    assert len(labeled) == 1
    assert labeled.iloc[0]["entry_at"] == index[entry_position]
    assert labeled.iloc[0]["entry_at"] > available_at
    assert labeled.iloc[0]["outcome_class"] == 0
    assert labeled.iloc[0]["gross_return_bps"] == -12.0
    assert labeled.iloc[0]["net_return_bps"] == -21.0


def test_platt_probabilities_are_normalized() -> None:
    probabilities = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]] * 20)
    labels = np.array([0, 1, 2] * 20)
    heads = micro._platt_fit(probabilities, labels)
    calibrated = micro._platt_predict(probabilities, heads)

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.isfinite(calibrated).all()


def test_portfolio_accepts_losing_trades_when_predicted_ev_is_positive() -> None:
    scored = pd.DataFrame(
        [
            {
                "entry_at": pd.Timestamp("2026-04-01T00:00:00Z"),
                "exit_at": pd.Timestamp("2026-04-01T00:05:00Z"),
                "predicted_net_ev_bps": 2.0,
                "net_return_bps": -10.0,
            },
            {
                "entry_at": pd.Timestamp("2026-04-01T00:06:00Z"),
                "exit_at": pd.Timestamp("2026-04-01T00:10:00Z"),
                "predicted_net_ev_bps": 3.0,
                "net_return_bps": 20.0,
            },
        ]
    )

    trades, metrics = micro._simulate(scored)

    assert len(trades) == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["expectancy_bps"] == 5.0
    assert metrics["positive_active_days"] == 1.0
