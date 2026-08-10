from __future__ import annotations

import pandas as pd

from adaptive_bot.musca_v5_frequency_audit import _deduplicate_horizons


def test_frequency_audit_deduplicates_the_same_signal_across_horizons() -> None:
    rows = pd.DataFrame(
        {
            "signal_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]
            ),
            "direction": [1, 1],
            "expert_breakout_bars": [6, 24],
        }
    )

    result = _deduplicate_horizons(rows)

    assert len(result) == 1
    assert result.iloc[0]["expert_breakout_bars"] == 24
