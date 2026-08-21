from pathlib import Path

import pandas as pd

from adaptive_bot.musca_v5_maker_feasibility import observed_label_coverage, run
from adaptive_bot.musca_v5_maker_upper_bound import exact_threshold_ceiling


def test_maker_training_fails_closed_without_private_labels(tmp_path: Path) -> None:
    report = run(
        labels_path=tmp_path / "missing.parquet",
        report_path=tmp_path / "report.json",
        l2_path=tmp_path / "missing-l2.parquet",
        raw_root=tmp_path / "raw",
    )
    assert report["verdict"] == "COLLECTING_NO_OBSERVED_MAKER_MODEL"
    assert not report["maker_cost_allowed_in_alpha_or_paper"]


def test_observed_labels_require_causal_complete_private_outcomes() -> None:
    created = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "created_at": created,
            "available_at": created + pd.Timedelta(minutes=1),
            "fill_probability_target": [float(index % 2 == 0) for index in range(40)],
            "fill_fraction_target": [float(index % 2 == 0) for index in range(40)],
            "fill_latency_ms": 1_000.0,
            "maker_only": True,
            "observation_complete": True,
            "adverse_selection_bps_1s": 0.1,
            "adverse_selection_bps_5s": 0.2,
            "adverse_selection_bps_30s": 0.3,
        }
    )
    coverage = observed_label_coverage(frame)
    assert coverage["independent_days"] == 40
    assert coverage["filled_orders"] == 20
    assert coverage["unfilled_orders"] == 20
    assert coverage["fully_labeled_maker_fills"] == 20


def test_perfect_fill_diagnostic_uses_only_exact_conservative_thresholds() -> None:
    available = (12.0, 13.5, 16.0, 16.5)
    assert exact_threshold_ceiling(8.25, available) == 12.0
    assert exact_threshold_ceiling(16.5, available) == 16.5
