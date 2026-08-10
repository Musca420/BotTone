from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_btc_auto_moe as auto
from adaptive_bot import musca_btc_moe as previous


def test_previous_125_component_protocol_remains_frozen() -> None:
    assert (
        previous.PROTOCOL_HASH == "73563d1aed16e4f796d18d446dc9033946473429e48f18fe15c1a52ffc193ddd"
    )
    assert previous.PROTOCOL["return_experts"]["final_components"] == 125
    assert auto.PROTOCOL["parent_protocol_hash"] == previous.PROTOCOL_HASH


def test_library_size_is_discovered_instead_of_configured() -> None:
    phase = auto.PROTOCOL["phase_1"]
    assert phase["final_expert_limit"] is None
    assert phase["emergency_tree_ceiling_per_action"] == 128
    assert auto.PROTOCOL["phase_2"]["decision"].endswith("neutral FLAT")


def test_expert_id_is_deterministic_and_identifies_the_rule() -> None:
    first = auto._expert_id(1, 300, 7, 12)
    assert first == auto._expert_id(1, 300, 7, 12)
    assert first != auto._expert_id(-1, 300, 7, 12)
    assert first != auto._expert_id(1, 900, 7, 12)


def test_diversity_has_no_final_count_cap() -> None:
    candidates = [
        {
            "expert_id": f"e{number}",
            "side": 1,
            "horizon_seconds": 300,
            "robust_score": 10 - number,
            "signal_signature": f"s{number}",
        }
        for number in range(3)
    ]
    signals = {
        "e0": np.array([True, False, False, False]),
        "e1": np.array([False, True, False, False]),
        "e2": np.array([False, False, True, False]),
    }
    selected, rejected = auto._select_diverse(candidates, signals)
    assert len(selected) == len(candidates)
    assert not rejected


def test_correlated_or_duplicate_rules_are_rejected() -> None:
    candidates = [
        {
            "expert_id": "best",
            "side": 1,
            "horizon_seconds": 60,
            "robust_score": 2.0,
            "signal_signature": "a",
        },
        {
            "expert_id": "same",
            "side": 1,
            "horizon_seconds": 60,
            "robust_score": 1.0,
            "signal_signature": "b",
        },
    ]
    signals = {
        "best": np.array([True, True, False]),
        "same": np.array([True, True, False]),
    }
    selected, rejected = auto._select_diverse(candidates, signals)
    assert [item["expert_id"] for item in selected] == ["best"]
    assert rejected["correlated_signal"] == 1


def test_gate_rows_are_weighted_equally_per_timestamp() -> None:
    rows = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 2 + ["2026-01-01T00:01:00Z"] * 4,
                utc=True,
            )
        }
    )
    weights = auto._timestamp_weights(rows)
    totals = pd.Series(weights).groupby(rows["entry_timestamp"].reset_index(drop=True)).sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_flat_is_neutral_when_every_expert_has_negative_ev() -> None:
    rows = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-06-01T00:05:00Z", "2026-06-01T00:05:00Z"], utc=True
            ),
            "expert_id": ["long", "short"],
            "calibrated_ev_bps": [-0.1, -2.0],
            "raw_ev_bps": [1.0, 2.0],
        }
    )
    assert auto._execute(rows).empty


def test_chronology_keeps_discovery_gate_and_audit_separate() -> None:
    assert auto.DISCOVERY_FIT_END < auto.LIBRARY_FREEZE_END
    assert auto.LIBRARY_FREEZE_END < auto.GATE_TUNE_END
    assert auto.GATE_TUNE_END < auto.GATE_FIT_END
    assert auto.GATE_FIT_END < auto.CALIBRATION_END
    assert auto.CALIBRATION_END < auto.HISTORICAL_AUDIT_END
    assert auto.HISTORICAL_AUDIT_END < auto.FUTURE_HOLDOUT_START
