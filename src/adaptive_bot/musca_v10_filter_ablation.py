from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.musca_v4_research import (
    BARS,
    MINUTES,
    _metrics,
    _non_overlapping,
    build_events,
    label_events,
)

REPORT = Path("data/reports/musca_v10_filter_ablation.json")
HOLDOUT_START = pd.Timestamp("2026-05-11T11:30:00Z")
CONFIGS = (
    ("DEFAULT", True, True, 1),
    ("NO_QUIETER", False, True, 1),
    ("NO_RESTART_SPOT", True, False, 1),
    ("CONFLUENCE_2", True, True, 2),
    ("NO_QUIETER_CONFLUENCE_2", False, True, 2),
    ("NO_RESTART_SPOT_CONFLUENCE_2", True, False, 2),
    ("NO_RESTART_SPOT_NO_QUIETER", False, False, 1),
    ("NO_RESTART_SPOT_NO_QUIETER_CONFLUENCE_2", False, False, 2),
)


def _worker(config: tuple[str, bool, bool, int]) -> tuple[dict[str, Any], pd.DataFrame]:
    name, quieter, spot, confluence = config
    events = build_events(
        pd.read_parquet(BARS),
        breakout_bars=24,
        require_quieter_pullback=quieter,
        require_restart_spot=spot,
        max_restart_confluence=confluence,
    )
    rows = _non_overlapping(
        label_events(events, pd.read_parquet(MINUTES)).loc[
            lambda frame: frame["event_family"].eq("IMPULSE_PULLBACK")
        ]
    )
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    rows = rows.loc[time.lt(HOLDOUT_START)]
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    train, validation, test = (rows.loc[time.dt.year.eq(year)] for year in (2024, 2025, 2026))
    train_metrics, validation_metrics = _metrics(train), _metrics(validation)
    eligible = (
        train_metrics["expectancy_bps"] > 0
        and train_metrics["profit_factor"] >= 1.05
        and validation_metrics["expectancy_bps"] > 0
        and validation_metrics["profit_factor"] >= 1.05
    )
    return (
        {
            "name": name,
            "events": len(rows),
            "train": train_metrics,
            "validation": validation_metrics,
            "train_stress": _metrics(train, "stress_return_bps"),
            "validation_stress": _metrics(validation, "stress_return_bps"),
            "eligible": eligible,
            "test": _metrics(test) if eligible else None,
            "test_stress": _metrics(test, "stress_return_bps") if eligible else None,
        },
        rows,
    )


def run() -> dict[str, Any]:
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, CONFIGS))
    audits = [audit for audit, _ in results]
    eligible = [audit for audit in audits if audit["eligible"]]
    champion = max(
        eligible,
        key=lambda audit: audit["validation"]["expectancy_bps"],
        default=None,
    )
    gates = {
        "eligible_configuration": champion is not None,
        "test_trades_50": bool(champion and champion["test"]["trades"] >= 50),
        "test_expectancy_positive": bool(champion and champion["test"]["expectancy_bps"] > 0),
        "test_pf_1_15": bool(champion and champion["test"]["profit_factor"] >= 1.15),
        "test_stress_nonnegative": bool(
            champion and champion["test_stress"]["expectancy_bps"] >= 0
        ),
    }
    report = {
        "protocol": {
            "name": "musca_v10_v8_filter_ablation_v1",
            "fixed_breakout_bars": 24,
            "fixed_room_bps": 24,
            "selection": "2024_2025_only",
            "test": "2026_before_holdout",
            "holdout_opened": False,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "configurations": audits,
        "champion": champion,
        "gates": gates,
        "verdict": "FREQUENT_RESEARCH_CANDIDATE" if all(gates.values()) else "NO_FILTER_EDGE",
        "real_capital_allowed": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
