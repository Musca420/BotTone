from __future__ import annotations

import numpy as np

from adaptive_bot.musca_v5_strategy_class_audit import _barrier_actions


def test_barrier_labels_start_after_decision_and_stop_wins_same_bar() -> None:
    open_price = np.full(5, 100.0)
    high = np.full(5, 100.0)
    low = np.full(5, 100.0)
    close = np.full(5, 100.0)
    high[0], low[0] = 200.0, 1.0  # Decision bar must never affect the label.
    high[1], low[1] = 101.0, 99.0  # Both barriers: worst case is the stop.

    long, short, _, _, valid = _barrier_actions(
        high,
        low,
        close,
        open_price,
        np.array([0]),
        target_bps=50.0,
        stop_bps=50.0,
        horizon_bars=2,
    )

    assert valid.tolist() == [True]
    assert long.tolist() == [-50.0]
    assert short.tolist() == [-50.0]
