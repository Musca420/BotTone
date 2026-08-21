from __future__ import annotations

import pandas as pd

from adaptive_bot.musca_v5_local_action_model import label_actions


def test_action_matrix_has_twelve_actions_and_stop_wins() -> None:
    at = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            {
                "available_at": at,
                "direction": 1,
                "stop_price": 99.5,
                "operating_vwap": 100.0,
            }
        ]
    )
    minutes = pd.DataFrame(
        [
            {
                "timestamp": at,
                "data_valid": True,
                "perp_open": 100.0,
                "perp_high": 101.0,
                "perp_low": 99.0,
                "perp_close": 100.0,
                "funding_event_rate": 0.0,
            }
        ]
    )

    result = label_actions(events, minutes)

    assert len(result) == 12
    assert result["exit_reason"].eq("STRUCTURAL_STOP").all()
    assert result["gross_market_return_bps"].eq(-50.0).all()
