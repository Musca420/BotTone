from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.hybrid_policy_v25.bundle import build_bundle, load_bundle, score_candidate
from adaptive_bot.hybrid_policy_v25.data import build_all
from adaptive_bot.hybrid_policy_v25.events import EVENT_PATH, build_events, build_outcomes
from adaptive_bot.hybrid_policy_v25.models import train_walk_forward
from adaptive_bot.hybrid_policy_v25.protocol import (
    ROOT,
    atomic_json,
    finalize,
    freeze,
    status,
)
from adaptive_bot.hybrid_policy_v25.reporting import data_report, model_markdown, policy_report


def _implementation_paths() -> tuple[Path, ...]:
    return tuple(sorted(Path(__file__).parent.glob("*.py")))


def _data(*, resume: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = freeze(implementation_paths=_implementation_paths())
    datasets, audits = build_all(resume=resume)
    report = data_report(audits, protocol_hash=protocol["protocol_hash"])
    if report["verdict"] != "DATA_READY":
        raise RuntimeError("V25 mandatory data audit failed")
    return protocol, datasets, report


def run(command: str, *, resume: bool, smoke: bool) -> dict[str, Any]:
    protocol, datasets, audit = _data(resume=resume)
    if command == "data-audit":
        return audit
    events, rejections = build_events(
        datasets, protocol_hash=protocol["protocol_hash"], resume=resume, smoke=smoke
    )
    if command == "build-events":
        return {"events": len(events), "rejections": len(rejections)}
    outcomes = build_outcomes(events, datasets, resume=resume, smoke=smoke)
    if command == "build-outcomes":
        return {"events": len(events), "outcomes": len(outcomes)}
    predictions, decisions, model_audit, _ = train_walk_forward(
        outcomes, resume=resume, smoke=smoke
    )
    if smoke:
        return {
            "verdict": "SMOKE_COMPLETE",
            "events": len(outcomes),
            "oos_events": len(predictions),
            "oos_trades": len(decisions),
            "protocol_hash": protocol["protocol_hash"],
            "smoke": True,
        }
    if not smoke:
        model_markdown(model_audit)
    if command == "train":
        return model_audit
    policy_audit = policy_report(
        outcomes,
        predictions,
        decisions,
        model_audit,
        protocol_hash=protocol["protocol_hash"],
    )
    if command == "evaluate":
        return policy_audit
    manifest = build_bundle(outcomes, protocol, model_audit)
    if not smoke:
        protocol = finalize(protocol, implementation_paths=_implementation_paths())
    if command == "build-paper-bundle":
        return manifest
    return {
        "verdict": policy_audit["verdict"],
        "events": len(outcomes),
        "oos_events": len(predictions),
        "oos_trades": len(decisions),
        "bundle": manifest,
        "protocol_hash": protocol["protocol_hash"],
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def paper_once(mode: str) -> dict[str, Any]:
    bundle = load_bundle(mode)
    events = pd.read_parquet(EVENT_PATH)
    latest = events.tail(1)
    result = score_candidate(bundle, latest)
    record = {
        "mode": mode,
        "timestamp": datetime.now(UTC).isoformat(),
        "event_id": latest.iloc[0]["event_id"],
        **result,
    }
    path = ROOT / "paper" / f"{datetime.now(UTC):%Y-%m-%d}.json"
    atomic_json(path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="V25 trend/VWAP meta-label pipeline")
    parser.add_argument(
        "command",
        choices=[
            "data-audit",
            "build-events",
            "build-outcomes",
            "train",
            "evaluate",
            "build-paper-bundle",
            "paper",
            "shadow",
            "all",
        ],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--output-root")
    parser.add_argument("--as-of")
    parser.add_argument("--asset", choices=["BTCUSDT", "ETHUSDT"])
    args = parser.parse_args()
    try:
        result = (
            paper_once(args.command)
            if args.command in {"paper", "shadow"}
            else run(args.command, resume=args.resume, smoke=args.smoke)
        )
    except Exception as error:
        status("failed", f"{type(error).__name__}: {error}", 0)
        raise
    status("complete", str(result.get("verdict", args.command)), 100)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
