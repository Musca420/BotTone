from __future__ import annotations

import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.musca_v4_research import (
    BARS,
    COST_BPS,
    MINUTES,
    _metrics,
    _non_overlapping,
    build_events,
    label_events,
)

REPORT = Path("data/reports/musca_v7_frequency_audit.json")
STATUS = Path("data/reports/musca_v7_frequency_audit.status.json")
HOLDOUT_START = pd.Timestamp("2026-05-11T11:30:00Z")
CONFIGURATIONS = tuple(itertools.product((0.75, 1.0), (0.15, 0.25), (6, 12)))
PROTOCOL = {
    "name": "musca_v7_v4_frequency_plateau_v1",
    "base": "musca_v4_multi_anchor_v1",
    "fixed": {
        "breakout_bars": 3,
        "minimum_room_bps": 3 * COST_BPS,
        "cost_bps": COST_BPS,
        "management": "unchanged_v4",
    },
    "registered_grid": {
        "relative_volume_min": [0.75, 1.0],
        "pullback_depth_atr": [0.15, 0.25],
        "restart_expiry_bars": [6, 12],
    },
    "selection": "2024_and_2025_only",
    "test": "2026_before_sealed_holdout",
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(index: int, detail: str) -> None:
    _write(
        STATUS,
        {
            "phase": "frequency_plateau",
            "percent": round(100 * index / len(CONFIGURATIONS), 2),
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def run() -> dict[str, Any]:
    bars, minutes = pd.read_parquet(BARS), pd.read_parquet(MINUTES)
    rows: list[dict[str, Any]] = []
    for index, (volume, depth, expiry) in enumerate(CONFIGURATIONS, start=1):
        config_id = f"vol{volume}_depth{depth}_expiry{expiry}"
        _status(index - 1, config_id)
        events = build_events(
            bars,
            relative_volume_min=volume,
            pullback_depth_atr=depth,
            restart_expiry_bars=expiry,
        )
        labeled = label_events(events, minutes)
        primary = _non_overlapping(
            labeled.loc[labeled["event_family"].eq("IMPULSE_PULLBACK")]
        )
        time = pd.to_datetime(primary["entry_timestamp"], utc=True)
        primary = primary.loc[time.lt(HOLDOUT_START)].copy()
        time = pd.to_datetime(primary["entry_timestamp"], utc=True)
        train = primary.loc[time.dt.year.eq(2024)]
        validation = primary.loc[time.dt.year.eq(2025)]
        test = primary.loc[time.dt.year.eq(2026)]
        train_metrics, validation_metrics = _metrics(train), _metrics(validation)
        train_stress = _metrics(train, "stress_return_bps")
        validation_stress = _metrics(validation, "stress_return_bps")
        eligible = (
            train_metrics["expectancy_bps"] > 0
            and validation_metrics["expectancy_bps"] > 0
            and train_metrics["profit_factor"] >= 1.10
            and validation_metrics["profit_factor"] >= 1.10
            and train_stress["expectancy_bps"] >= 0
            and validation_stress["expectancy_bps"] >= 0
        )
        rows.append(
            {
                "config_id": config_id,
                "relative_volume_min": volume,
                "pullback_depth_atr": depth,
                "restart_expiry_bars": expiry,
                "events": len(primary),
                "train": train_metrics,
                "validation": validation_metrics,
                "train_stress": train_stress,
                "validation_stress": validation_stress,
                "eligible_train_validation": eligible,
                "test": _metrics(test) if eligible else None,
                "test_stress": _metrics(test, "stress_return_bps") if eligible else None,
            }
        )
    eligible_rows = [row for row in rows if row["eligible_train_validation"]]
    champion = max(
        eligible_rows,
        key=lambda row: row["validation"]["expectancy_bps"],
        default=None,
    )
    gates = {
        "stable_configuration_exists": champion is not None,
        "test_expectancy_positive": bool(champion and champion["test"]["expectancy_bps"] > 0),
        "test_pf_1_15": bool(champion and champion["test"]["profit_factor"] >= 1.15),
        "test_stress_nonnegative": bool(
            champion and champion["test_stress"]["expectancy_bps"] >= 0
        ),
        "test_trades_50": bool(champion and champion["test"]["trades"] >= 50),
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "configurations": rows,
        "champion": champion,
        "gates": gates,
        "verdict": "RESEARCH_SHADOW_CANDIDATE" if all(gates.values()) else "NO_STABLE_PLATEAU",
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _write(REPORT, report)
    _status(len(CONFIGURATIONS), str(report["verdict"]))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
