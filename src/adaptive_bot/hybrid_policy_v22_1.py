from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.hybrid_policy_v22.path_audit import _first_hit, atomic_parquet
from adaptive_bot.hybrid_policy_v22.protocol import sha256

PROTOCOL = "hybrid_v22_1_path_label_completeness"
SOURCE_ROOT = Path("data/ml/hybrid_v22")
ROOT = Path("data/ml/hybrid_v22_1")
MODEL_ROOT = Path("data/models/expert_policy/v22_1")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
REPORT_PATH = Path("data/reports/ml_hybrid_v22_1.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v22_1.status.json")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": percent,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _protocol() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "purpose": "metadata and path-label completeness only; V22 economics frozen",
        "source_sha256": sha256(Path(__file__)),
        "v22_protocol_sha256": sha256(Path("data/models/expert_policy/v22/protocol.json")),
        "v22_report_sha256": sha256(Path("data/reports/ml_hybrid_v22_audit.json")),
        "definitions": {
            "available_at": "max of every causal feature availability timestamp",
            "outer_band": "rolling VWAP plus deviation_side * 1 ATR",
            "fade_invalidation": "event extreme plus deviation_side * 0.5 ATR",
            "follow_invalidation": "close-side touch of inner band at 0.5 ATR",
        },
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    current = _protocol()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != current["protocol_sha256"]:
            raise RuntimeError("V22.1 protocol changed after freezing")
        return dict(existing)
    payload = current | {"registered_at": datetime.now(UTC).isoformat()}
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def complete_labels(
    states: pd.DataFrame, paths: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability = [
        column
        for column in states
        if str(column).endswith("_available_at")
        or column in {"feature_available_at", "context_available_at"}
    ]
    timestamps = pd.concat(
        [pd.to_datetime(states[column], utc=True) for column in availability], axis=1
    )
    states = states.copy()
    states["available_at"] = timestamps.max(axis=1)
    decision = pd.to_datetime(states["decision_timestamp"], utc=True)
    if states["available_at"].gt(decision).any():
        raise RuntimeError("V22.1 causal availability violation")

    indexed = {key: rows for key, rows in paths.groupby("event_id", sort=False)}
    additions: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        path = indexed[row.event_id]
        high, low = path["high"].to_numpy(float), path["low"].to_numpy(float)
        side = int(np.sign(float(cast(Any, row.deviation_side))))
        vwap, atr = float(cast(Any, row.vwap)), float(cast(Any, row.atr))
        outer = vwap + side * atr
        fade_invalid = (
            float(cast(Any, row.high)) + 0.5 * atr
            if side > 0
            else float(cast(Any, row.low)) - 0.5 * atr
        )
        follow_invalid = vwap + side * 0.5 * atr
        outer_hit = _first_hit(high >= outer if side > 0 else low <= outer)
        fade_hit = _first_hit(high >= fade_invalid if side > 0 else low <= fade_invalid)
        follow_hit = _first_hit(low <= follow_invalid if side > 0 else high >= follow_invalid)
        additions.append(
            {
                "event_id": row.event_id,
                "time_to_outer_band_seconds": outer_hit[0],
                "outer_band_censored": outer_hit[1],
                "time_to_fade_invalidation_seconds": fade_hit[0],
                "fade_invalidation_censored": fade_hit[1],
                "time_to_follow_invalidation_seconds": follow_hit[0],
                "follow_invalidation_censored": follow_hit[1],
            }
        )
    complete = labels.merge(pd.DataFrame(additions), on="event_id", validate="one_to_one")
    return states, complete


def run() -> dict[str, Any]:
    protocol = preregister()
    _status("metadata", "Correct maximum causal available_at", 10)
    states = pd.read_parquet(SOURCE_ROOT / "event_states.parquet")
    paths = pd.read_parquet(SOURCE_ROOT / "event_paths.parquet")
    labels = pd.read_parquet(SOURCE_ROOT / "path_labels.parquet")
    states, labels = complete_labels(states, paths, labels)
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_parquet(ROOT / "event_states.parquet", states)
    atomic_parquet(ROOT / "path_labels.parquet", labels)
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "artifact_class": "RESEARCH_ONLY",
        "economic_audit_reopened": False,
        "v22_verdict_preserved": "NO_ECONOMIC_VWAP_SETUP_FOUND",
        "events": len(states),
        "available_at_violations": int(
            states["available_at"].gt(pd.to_datetime(states["decision_timestamp"], utc=True)).sum()
        ),
        "censoring": {
            str(column): float(labels[column].mean())
            for column in labels
            if str(column).endswith("_censored")
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    _status("complete", "V22.1_PATH_LABELS_COMPLETE", 100)
    return report


def main() -> None:
    argparse.ArgumentParser(description="V22.1 path-label completeness").parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
