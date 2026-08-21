from __future__ import annotations

import pandas as pd
import pytest

from adaptive_bot.musca_v5_funding_vwap_frontier import add_funding_cycle_vwap


def test_funding_cycle_vwap_resets_without_future_data(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T07:55:00Z", periods=3, freq="5min"),
            "perp_volume": [1.0, 1.0, 1.0],
            "perp_quote_volume": [100.0, 200.0, 300.0],
        }
    )
    prepared = bars.copy()
    prepared["available_at"] = prepared["timestamp"] + pd.Timedelta(minutes=5)
    monkeypatch.setattr(
        "adaptive_bot.musca_v5_funding_vwap_frontier.build_features",
        lambda frame: prepared.copy(),
    )
    funding = pd.DataFrame(
        {"funding_timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T08:00:00Z"])}
    )

    result = add_funding_cycle_vwap(bars, funding)

    assert result["funding_vwap"].tolist() == [100.0, 200.0, 250.0]
    assert result.loc[1, "funding_anchor_at"] == pd.Timestamp("2026-01-01T08:00:00Z")
    assert (
        pd.to_datetime(result["funding_anchor_at"], utc=True)
        <= pd.to_datetime(result["funding_feature_available_at"], utc=True)
    ).all()
