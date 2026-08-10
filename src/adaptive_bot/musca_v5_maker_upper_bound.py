from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_bot.bitunix_fees import futures_fee_bps
from adaptive_bot.musca_v5_fine_tuning import _atomic_json
from adaptive_bot.musca_v5_maker_feasibility import REPORT as MAKER_FEASIBILITY_REPORT
from adaptive_bot.musca_v5_room_frontier import (
    OUTCOME_PROTOCOL_HASH,
    PAPER_PROFILES,
    ROOM_COST_MULTIPLIERS,
    _evaluate_cost_frontier,
    build_matrix,
)
from adaptive_bot.musca_v8_multi_horizon import HISTORICAL_NON_FEE_RESERVE_BPS

REPORT = Path("data/reports/musca_v5_maker_perfect_fill_upper_bound.json")
PROTOCOL = {
    "name": "musca_v5_maker_entry_perfect_fill_upper_bound_v1",
    "outcome_protocol_hash": OUTCOME_PROTOCOL_HASH,
    "entry_role": "maker_POST_ONLY_perfect_fill_counterfactual",
    "exit_role": "taker",
    "non_fee_reserve_bps": HISTORICAL_NON_FEE_RESERVE_BPS,
    "room_multipliers": list(ROOM_COST_MULTIPLIERS),
    "threshold_mapping": "smallest_available_exact_stateful_threshold_not_below_request",
    "selection": "prior_2024_2025_only_then_single_2026_preholdout_audit",
    "maker_fill_probability_assumed": 1.0,
    "adverse_selection_assumed_bps": 0.0,
    "operational_authority": False,
    "official_fee_source": "https://www.bitunix.com/service/handling-fee",
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def exact_threshold_ceiling(requested: float, available: tuple[float, ...]) -> float:
    try:
        return next(value for value in sorted(available) if value + 1e-9 >= requested)
    except StopIteration as error:
        raise ValueError(f"No exact stateful room threshold covers {requested:g} bps") from error


def run(*, report_path: Path = REPORT) -> dict[str, Any]:
    matrix = build_matrix()
    available = tuple(sorted(float(value) for value in matrix["room_threshold_bps"].unique()))
    profiles: dict[str, Any] = {}
    for profile in PAPER_PROFILES:
        level = int(profile.removeprefix("VIP"))
        maker_bps, taker_bps = futures_fee_bps(level)
        perfect_fill_cost = maker_bps + taker_bps + HISTORICAL_NON_FEE_RESERVE_BPS
        requested = {
            f"{multiplier:g}x": multiplier * perfect_fill_cost
            for multiplier in ROOM_COST_MULTIPLIERS
        }
        mapped = {
            label: exact_threshold_ceiling(threshold, available)
            for label, threshold in requested.items()
        }
        unique_thresholds = {
            f"room_{threshold:g}bps": threshold for threshold in mapped.values()
        }
        evaluation = _evaluate_cost_frontier(
            matrix,
            cost_bps=perfect_fill_cost,
            thresholds=unique_thresholds,
        )
        evaluation["fee_bps"] = {"maker_entry": maker_bps, "taker_exit": taker_bps}
        evaluation["requested_room_bps"] = requested
        evaluation["exact_stateful_room_bps"] = mapped
        profiles[profile] = evaluation
    maker_gate = False
    if MAKER_FEASIBILITY_REPORT.exists():
        maker_gate = bool(
            json.loads(MAKER_FEASIBILITY_REPORT.read_text(encoding="utf-8")).get(
                "maker_training_authorized", False
            )
        )
    research_profiles = [
        profile for profile, result in profiles.items() if result["research_signal"]
    ]
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "profiles": profiles,
        "perfect_fill_research_profiles": research_profiles,
        "observed_maker_training_gate": maker_gate,
        "verdict": (
            "PERFECT_FILL_RESEARCH_SIGNAL_ONLY"
            if research_profiles
            else "PERFECT_FILL_NO_FREQUENCY_SOLUTION"
        ),
        "maker_cost_allowed_in_alpha_or_paper": False,
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(report_path, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
