from __future__ import annotations

import pandas as pd

from adaptive_bot.binance_public_book_depth import (
    PERCENTAGES,
    build_features_from_rows,
)


def test_depth_features_are_available_after_the_official_snapshot() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="30s", tz="UTC")
    rows = []
    for timestamp in timestamps:
        for percentage in PERCENTAGES:
            depth = 120.0 if percentage < 0 else 80.0
            price = 100.0 * (1.0 + percentage / 100)
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": percentage,
                    "depth": depth,
                    "notional": depth * price,
                }
            )
    result = build_features_from_rows(pd.DataFrame(rows))
    assert (result["available_at"] > result["snapshot_timestamp"]).all()
    assert result["depth_imbalance_1pct"].iloc[-1] == 0.2
    assert bool(result["coverage_valid"].iloc[-1])
