from __future__ import annotations

import hashlib
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
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    HORIZONS,
)
from adaptive_bot.musca_v8_multi_horizon import (
    PROTOCOL_HASH as BASE_PROTOCOL_HASH,
)

REPORT = Path("data/reports/musca_v5_frequency_audit.json")
CANDIDATES = Path("data/ml/musca_v5/fine_tuning_candidates.parquet")
PROTOCOL = {
    "name": "musca_v5_frequency_funnel_v1",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "horizons": list(HORIZONS),
    "parameter_changes": None,
    "purpose": "attribute_frequency_loss_before_fine_tuning",
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
STAGES = ("scanned", "armed", "restart", "confirmed", "room", "risk", "event")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _period(timestamp: pd.Series) -> pd.Series:
    time = pd.to_datetime(timestamp, utc=True)
    return time.dt.year.map(
        {2024: "2024_fit", 2025: "2025_validation", 2026: "2026_pre_holdout"}
    )


def _counts_by_period(rows: pd.DataFrame, timestamp: str) -> dict[str, int]:
    if rows.empty:
        return {}
    return {
        str(period): int(count)
        for period, count in _period(rows[timestamp]).value_counts().sort_index().items()
        if pd.notna(period)
    }


def _worker(horizon: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(BARS)
    funnel: dict[str, int] = {}
    events = build_events(frame, funnel, breakout_bars=horizon)
    minutes = pd.read_parquet(MINUTES)
    labeled = label_events(events, minutes)
    if not labeled.empty:
        entry_time = pd.to_datetime(labeled["entry_timestamp"], utc=True)
        labeled = labeled.loc[entry_time.lt(HOLDOUT_START)].copy()
        labeled["expert_breakout_bars"] = horizon
    stage_rows = []
    scanned = funnel.get("scanned", 0)
    previous = funnel.get(STAGES[0], 0)
    for stage in STAGES:
        current = funnel.get(stage, 0)
        stage_rows.append(
            {
                "stage": stage,
                "count": current,
                "lost_from_previous": max(0, previous - current),
                "survival_from_previous": current / previous if previous else 0.0,
                "survival_from_scanned": current / scanned if scanned else 0.0,
            }
        )
        previous = current
    families: dict[str, Any] = {}
    for family, rows in labeled.groupby("event_family"):
        non_overlapping = _non_overlapping(rows)
        families[str(family)] = {
            "labeled": len(rows),
            "non_overlapping": len(non_overlapping),
            "by_period": _counts_by_period(non_overlapping, "entry_timestamp"),
            "metrics": _metrics(non_overlapping),
        }
    broad = build_events(
        frame,
        breakout_bars=horizon,
        minimum_room_bps=None,
        risk_bps_range=None,
    )
    broad = broad.loc[
        broad["event_family"].isin(("IMPULSE_PULLBACK", "IMPULSE_REENTRY"))
        & pd.to_datetime(broad["available_at"], utc=True).lt(HOLDOUT_START)
    ].copy()
    broad["expert_breakout_bars"] = horizon
    broad["frequency_audit_protocol_hash"] = PROTOCOL_HASH
    return (
        {
            "breakout_bars": horizon,
            "funnel": stage_rows,
            "events_returned": len(events),
            "labeled_before_holdout": len(labeled),
            "families": families,
        },
        labeled,
        broad,
    )


def _deduplicate_horizons(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return (
        rows.sort_values(
            ["signal_timestamp", "expert_breakout_bars"], ascending=[True, False]
        )
        .drop_duplicates(["signal_timestamp", "direction"], keep="first")
        .reset_index(drop=True)
    )


def run() -> dict[str, Any]:
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, HORIZONS))
    horizons = [audit for audit, _, _ in results]
    labeled = pd.concat([rows for _, rows, _ in results], ignore_index=True)
    family_summary: dict[str, Any] = {}
    for family, rows in labeled.groupby("event_family"):
        unique = _deduplicate_horizons(rows)
        executable = _non_overlapping(unique)
        family_summary[str(family)] = {
            "across_horizons": len(rows),
            "unique_signals": len(unique),
            "duplicate_horizon_signals": len(rows) - len(unique),
            "executable_without_overlap": len(executable),
            "lost_to_open_position": len(unique) - len(executable),
            "by_period_before_overlap": _counts_by_period(unique, "entry_timestamp"),
            "by_period_executable": _counts_by_period(executable, "entry_timestamp"),
            "metrics_if_traded_alone": _metrics(executable),
        }
    broad = pd.concat([rows for _, _, rows in results], ignore_index=True)
    broad_unique = _deduplicate_horizons(broad)
    broad_time = pd.to_datetime(broad_unique["available_at"], utc=True)
    span_days = max(1, (broad_time.max() - broad_time.min()).days + 1)
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    temporary_candidates = CANDIDATES.with_suffix(".parquet.tmp")
    broad.to_parquet(temporary_candidates, index=False)
    temporary_candidates.replace(CANDIDATES)
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(BARS),
        "minutes": str(MINUTES),
        "horizons": horizons,
        "families": family_summary,
        "broad_candidate_pool": {
            "path": str(CANDIDATES),
            "action_rows_across_horizons": len(broad),
            "unique_timestamp_direction": len(broad_unique),
            "calendar_days": span_days,
            "unique_candidates_per_calendar_day": len(broad_unique) / span_days,
            "by_period": _counts_by_period(broad_unique, "available_at"),
            "by_family": {
                str(family): int(count)
                for family, count in broad_unique["event_family"].value_counts().items()
            },
            "by_horizon_after_deduplication": {
                str(int(horizon)): int(count)
                for horizon, count in broad_unique["expert_breakout_bars"]
                .value_counts()
                .sort_index()
                .items()
            },
            "outcomes_attached": False,
        },
        "holdout_opened": False,
        "changes_to_trading_policy": False,
    }
    _atomic_json(REPORT, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
