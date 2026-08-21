from __future__ import annotations

import json

import pandas as pd

from adaptive_bot import musca_v8_multi_horizon as v8


def test_expert_identity_and_fee_profiles_are_deterministic() -> None:
    assert v8.expert_id(24) == v8.expert_id(24)
    assert v8.expert_id(24) != v8.expert_id(48)
    assert v8.profile_cost_bps("VIP0") > v8.profile_cost_bps("VIP5")
    assert v8.profile_cost_bps("VIP5") == 8.0


def test_live_candidate_uses_only_recent_profile_eligible_event(monkeypatch, tmp_path) -> None:
    now = pd.Timestamp("2026-08-09T12:00:00Z")
    report = tmp_path / "v8.json"
    report.write_text(
        json.dumps({"paper_profiles": {"VIP5": {"eligible_horizons": [24]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(v8, "REPORT", report)
    monkeypatch.setattr(
        v8,
        "_live_events",
        lambda _: [
            {
                "available_at": now - pd.Timedelta(minutes=1),
                "signal_timestamp": now - pd.Timedelta(minutes=6),
                "breakout_bars": 24,
                "expert_id": "h24",
                "direction": -1,
                "stop_price": 61_000.0,
                "robust_expected_gross_bps": 20.0,
                "robust_tp1_probability": 0.52,
                "return_1h": -0.01,
                "return_4h": -0.02,
                "spot_return_1h": -0.01,
                "relative_volume": 1.2,
                "taker_imbalance": -0.2,
                "confluence_score": 2,
                "impulse_anchor_at": now - pd.Timedelta(minutes=30),
                "swing_anchor_at": now - pd.Timedelta(hours=1),
                "operating_vwap": 60_500.0,
            },
            {
                "available_at": now - pd.Timedelta(minutes=10),
                "signal_timestamp": now - pd.Timedelta(minutes=15),
                "breakout_bars": 24,
                "expert_id": "old",
                "direction": 1,
                "stop_price": 59_000.0,
                "robust_expected_gross_bps": 99.0,
                "robust_tp1_probability": 0.99,
                "return_1h": 0.01,
                "return_4h": 0.02,
                "spot_return_1h": 0.01,
                "relative_volume": 1.2,
                "taker_imbalance": 0.2,
                "confluence_score": 2,
                "impulse_anchor_at": now - pd.Timedelta(minutes=45),
                "swing_anchor_at": now - pd.Timedelta(hours=2),
                "operating_vwap": 59_500.0,
            },
        ],
    )

    candidate = v8.live_candidate("VIP5", now)

    assert candidate is not None
    assert candidate["expert_id"] == "h24"
    assert candidate["direction"] == "SHORT"
    assert candidate["maximum_hold_minutes"] == 360
    assert v8.live_candidate("VIP0", now) is None
