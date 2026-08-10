from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from adaptive_bot.hybrid_policy_v22.path_audit import (
    build_grid,
    build_path_labels,
    build_paths,
    walk_forward_exit_audit,
)
from adaptive_bot.hybrid_policy_v22.protocol import REPORT_PATH, atomic_json, preregister, status


def run(*, resume: bool, smoke: bool) -> dict[str, object]:
    protocol = preregister()
    status("path_builder", "Causal Binance 1m event paths", 2, block="1/4")
    states, paths = build_paths(str(protocol["protocol_sha256"]), resume=resume)
    status("path_labels", "MFE/MAE and censored time-to-event labels", 28, block="2/4")
    labels = build_path_labels(states, paths, resume=resume)
    status("exit_grid", "Preregistered FADE/FOLLOW exit geometry", 35, block="3/4")
    grid = build_grid(states, paths, resume=resume)
    oos, audit = walk_forward_exit_audit(grid, smoke=smoke)
    passed = any(value["gate_passed"] for value in audit["families"].values())
    verdict = "V22A_ECONOMIC_SETUP_FOUND" if passed else "NO_ECONOMIC_VWAP_SETUP_FOUND"
    report: dict[str, object] = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol["protocol_sha256"],
        "stage": "V22-A_PATH_AUDIT",
        "verdict": verdict,
        "artifact_class": "RESEARCH_ONLY",
        "paper_bundle_created": False,
        "deployable_bundle_created": False,
        "event_states": len(states),
        "event_path_rows": len(paths),
        "path_labels": len(labels),
        "exit_grid_rows": len(grid),
        "oos_rows": len(oos),
        "audit": audit,
        "next_stage_authorized": passed and not smoke,
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(REPORT_PATH, report)
    status("complete", verdict, 100, block="4/4")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V22 hierarchical VWAP path audit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(resume=args.resume, smoke=args.smoke)
    except Exception as error:
        status("failed", f"{type(error).__name__}: {error}", 0)
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
