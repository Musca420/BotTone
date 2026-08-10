from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import (
    BARS,
    MINUTES,
    _metrics,
    _non_overlapping,
    build_events,
    build_features,
    label_events,
)
from adaptive_bot.musca_v5_fine_tuning import _atomic_json, _bootstrap_lcb
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    PAPER_PROFILES,
    _with_cost,
    profile_cost_bps,
)
from adaptive_bot.musca_v8_multi_horizon import HORIZONS as FROZEN_HORIZONS
from adaptive_bot.musca_v8_multi_horizon import (
    PROTOCOL_HASH as BASE_PROTOCOL_HASH,
)
from adaptive_bot.musca_v8_multi_horizon import REPORT as BASE_REPORT

MATRIX = Path("data/ml/musca_v5/v8_room_frontier_matrix.parquet")
PARTS = Path("data/ml/musca_v5/v8_room_frontier_stateful_parts")
REPORT = Path("data/reports/musca_v5_room_frontier.json")
STATUS = Path("data/reports/musca_v5_room_frontier.status.json")
ROOM_COST_MULTIPLIERS = (1.5, 2.0, 3.0)
FAMILIES = ("IMPULSE_PULLBACK", "IMPULSE_REENTRY")
HORIZONS = (1, 2, *FROZEN_HORIZONS)
ROOM_THRESHOLDS_BPS = tuple(
    sorted(
        {
            profile_cost_bps(profile) * multiplier
            for profile in PAPER_PROFILES
            for multiplier in ROOM_COST_MULTIPLIERS
        }
    )
)
OUTCOME_PROTOCOL = {
    "name": "musca_v5_v8_management_room_cost_outcomes",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "candidate_source": str(BARS),
    "candidate_generation": "independent_stateful_path_per_room_threshold",
    "families": list(FAMILIES),
    "horizons": list(HORIZONS),
    "room_cost_multipliers": list(ROOM_COST_MULTIPLIERS),
    "risk_bps": [12.0, 200.0],
    "management": "frozen_V8_half_at_1_5R_cost_protected_15m_trail_6h",
    "entry": "next_observed_1m_open",
    "intrabar": "stop_wins",
    "holdout_opened": False,
}
OUTCOME_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(OUTCOME_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
LEGACY_OUTCOME_PROTOCOL_HASHES = (
    "0ed5d8c17fb4362bfb752ddaefe565315c564eef511606efb3d9831aed77be0f",
    "45811bb490f451b125d1389b0da073cf92a7582d943736800491f11ee0756eb6",
)
PROTOCOL = {
    "name": "musca_v5_v8_management_room_cost_frontier",
    "outcome_protocol_hash": OUTCOME_PROTOCOL_HASH,
    "outcome_protocol": OUTCOME_PROTOCOL,
    "expert_selection": "positive_EV_and_PF_1_05_in_both_2024_and_2025",
    "policy_selection": (
        "maximum_prior_frequency_across_nested_robustness_prefixes_"
        "subject_to_aggregate_gates"
    ),
    "sizing": {
        "risk_budget_fraction": 0.01,
        "maximum_margin_fraction": 0.10,
        "maximum_leverage": 10.0,
        "maximum_notional_fraction": 1.0,
        "equity_curve": "compounded_realized_returns",
    },
    "economic_gate_costs": "observed_profile_costs_1x",
    "cost_stress": "2x_reported_as_diagnostic_only",
    "audit": "2026_before_sealed_holdout_discovery_only",
    "changes_to_active_paper": False,
    "holdout_opened": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _write_matrix(rows: pd.DataFrame) -> None:
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    rows.to_parquet(temporary, index=False)
    os.replace(temporary, MATRIX)


_WORKER_FEATURES: pd.DataFrame | None = None
_WORKER_MINUTES: pd.DataFrame | None = None


def _label_threshold_horizon(spec: tuple[float, int]) -> pd.DataFrame:
    global _WORKER_FEATURES, _WORKER_MINUTES

    threshold, horizon = spec
    threshold_name = f"{threshold:g}".replace(".", "p")
    part = PARTS / f"room{threshold_name}-h{horizon}.parquet"
    if part.exists():
        cached = pd.read_parquet(part)
        if cached["room_frontier_protocol_hash"].eq(OUTCOME_PROTOCOL_HASH).all():
            return cached
        if cached["room_frontier_protocol_hash"].isin(
            LEGACY_OUTCOME_PROTOCOL_HASHES
        ).all():
            cached["room_frontier_protocol_hash"] = OUTCOME_PROTOCOL_HASH
            PARTS.mkdir(parents=True, exist_ok=True)
            temporary = part.with_suffix(".parquet.tmp")
            cached.to_parquet(temporary, index=False)
            os.replace(temporary, part)
            return cached
    if _WORKER_FEATURES is None:
        _WORKER_FEATURES = build_features(pd.read_parquet(BARS))
    if _WORKER_MINUTES is None:
        _WORKER_MINUTES = pd.read_parquet(MINUTES)
    candidates = build_events(
        pd.DataFrame(),
        breakout_bars=horizon,
        minimum_room_bps=threshold,
        _prepared_features=_WORKER_FEATURES,
    )
    candidates = candidates.loc[
        candidates["event_family"].isin(FAMILIES)
        & pd.to_datetime(candidates["available_at"], utc=True).lt(HOLDOUT_START)
    ]
    labeled = label_events(candidates, _WORKER_MINUTES)
    if labeled.empty:
        return labeled
    labeled = labeled.loc[
        pd.to_datetime(labeled["exit_timestamp"], utc=True).lt(HOLDOUT_START)
    ].copy()
    labeled["expert_breakout_bars"] = horizon
    labeled["room_threshold_bps"] = threshold
    labeled["room_frontier_protocol_hash"] = OUTCOME_PROTOCOL_HASH
    PARTS.mkdir(parents=True, exist_ok=True)
    temporary = part.with_suffix(".parquet.tmp")
    labeled.to_parquet(temporary, index=False)
    os.replace(temporary, part)
    return labeled


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        cached = pd.read_parquet(MATRIX)
        complete = set(cached["expert_breakout_bars"].unique()) == set(HORIZONS)
        if (
            complete
            and cached["room_frontier_protocol_hash"].eq(OUTCOME_PROTOCOL_HASH).all()
        ):
            return cached
        if complete and cached["room_frontier_protocol_hash"].isin(
            LEGACY_OUTCOME_PROTOCOL_HASHES
        ).all():
            cached["room_frontier_protocol_hash"] = OUTCOME_PROTOCOL_HASH
            _write_matrix(cached)
            return cached
    _atomic_json(
        STATUS,
        {
            "phase": "v8_exact_labels",
            "percent": 5,
            "detail": (
                f"{len(ROOM_THRESHOLDS_BPS)} room thresholds x {len(HORIZONS)} horizons"
            ),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    specs = [
        (threshold, horizon)
        for threshold in ROOM_THRESHOLDS_BPS
        for horizon in HORIZONS
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_label_threshold_horizon, spec): spec for spec in specs}
        parts = []
        for completed, future in enumerate(as_completed(futures), start=1):
            parts.append(future.result())
            threshold, horizon = futures[future]
            _atomic_json(
                STATUS,
                {
                    "phase": "v8_exact_labels",
                    "percent": 5 + 55 * completed / len(specs),
                    "detail": (
                        f"room {threshold:g} bps H{horizon} "
                        f"({completed}/{len(specs)})"
                    ),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
    matrix = pd.concat(parts, ignore_index=True)
    _write_matrix(matrix)
    return matrix


def _period(rows: pd.DataFrame, start: str, end: pd.Timestamp) -> pd.DataFrame:
    entries = pd.to_datetime(rows["entry_timestamp"], utc=True)
    exits = pd.to_datetime(rows["exit_timestamp"], utc=True)
    return rows.loc[entries.ge(pd.Timestamp(start, tz="UTC")) & exits.lt(end)].copy()


def _sized_equity_metrics(rows: pd.DataFrame, return_column: str) -> dict[str, float]:
    if rows.empty:
        return {
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "mean_notional_fraction": 0.0,
            "maximum_notional_fraction": 0.0,
        }
    ordered = rows.sort_values(["entry_timestamp", "signal_timestamp"])
    risk_bps = (
        ordered["direction"].to_numpy(float)
        * (
            ordered["entry_price"].to_numpy(float)
            - ordered["stop_price"].to_numpy(float)
        )
        / ordered["entry_price"].to_numpy(float)
        * 10_000
    )
    if not np.isfinite(risk_bps).all() or (risk_bps <= 0).any():
        raise ValueError("Sized equity received an invalid structural stop")
    notional = np.minimum(1.0, 100.0 / risk_bps)
    returns = ordered[return_column].to_numpy(float) / 10_000 * notional
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("Sized equity received an invalid realized return")
    equity = np.cumprod(1.0 + returns)
    curve = np.r_[1.0, equity]
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    return {
        "max_drawdown": float(drawdown.max()),
        "total_return": float(equity[-1] - 1.0),
        "mean_notional_fraction": float(notional.mean()),
        "maximum_notional_fraction": float(notional.max()),
    }


def _summary(
    rows: pd.DataFrame, cost_bps: float, *, bootstrap: bool = True
) -> dict[str, Any]:
    adjusted = _with_cost(rows, cost_bps)
    metrics = _metrics(adjusted)
    stress = _metrics(adjusted, "stress_return_bps")
    sizing = _sized_equity_metrics(adjusted, "net_return_bps")
    stress_sizing = _sized_equity_metrics(adjusted, "stress_return_bps")
    metrics["max_drawdown"] = sizing["max_drawdown"]
    stress["max_drawdown"] = stress_sizing["max_drawdown"]
    if adjusted.empty:
        positive_days = 0.0
    else:
        days = pd.to_datetime(adjusted["exit_timestamp"], utc=True).dt.floor("D")
        daily = pd.Series(adjusted["net_return_bps"].to_numpy(float), index=days).groupby(
            level=0
        ).sum()
        positive_days = float(daily.gt(0).mean())
    return {
        "metrics": metrics,
        "stress_2x": stress,
        "sizing": sizing,
        "stress_2x_sizing": stress_sizing,
        "stress_2x_nonnegative_diagnostic": stress["expectancy_bps"] >= 0,
        "expectancy_lcb_95_bps": (
            _bootstrap_lcb(adjusted, cost_bps) if bootstrap else float("nan")
        ),
        "positive_active_days": positive_days,
    }


def _expert_key(family: str, horizon: int) -> str:
    return f"{family}:H{horizon}"


def _select_expert_rows(
    eligible: list[tuple[str, int, float]], rows: pd.DataFrame
) -> pd.DataFrame:
    if not eligible:
        return rows.iloc[0:0].copy()
    robustness = {
        _expert_key(family, horizon): score for family, horizon, score in eligible
    }
    keys = pd.Series(
        [
            _expert_key(str(family), int(horizon))
            for family, horizon in zip(
                rows["event_family"],
                rows["expert_breakout_bars"],
                strict=True,
            )
        ],
        index=rows.index,
    )
    selected = rows.loc[keys.isin(robustness)].copy()
    selected["expert_key"] = keys.loc[selected.index]
    selected["prior_robustness_bps"] = selected["expert_key"].map(robustness)
    selected = selected.sort_values(
        ["signal_timestamp", "prior_robustness_bps"], ascending=[True, False]
    ).drop_duplicates(["signal_timestamp", "direction"], keep="first")
    return _non_overlapping(selected)


def _prior_gates(summaries: dict[str, Any]) -> dict[str, bool]:
    return {
        "trades_50": summaries["prior"]["metrics"]["trades"] >= 50,
        "expectancy_positive_both_years": (
            summaries["2024"]["metrics"]["expectancy_bps"] > 0
            and summaries["2025"]["metrics"]["expectancy_bps"] > 0
        ),
        "pf_1_05_both_years": (
            summaries["2024"]["metrics"]["profit_factor"] >= 1.05
            and summaries["2025"]["metrics"]["profit_factor"] >= 1.05
        ),
        "combined_pf_1_15": summaries["prior"]["metrics"]["profit_factor"] >= 1.15,
        "combined_drawdown_8pct": summaries["prior"]["metrics"]["max_drawdown"]
        <= 0.08,
        "expectancy_lcb_positive": bool(
            np.isfinite(summaries["prior"]["expectancy_lcb_95_bps"])
            and summaries["prior"]["expectancy_lcb_95_bps"] > 0
        ),
        "majority_positive_active_days": summaries["prior"]["positive_active_days"]
        > 0.5,
    }


def _audit_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "trades_50": summary["metrics"]["trades"] >= 50,
        "expectancy_positive": summary["metrics"]["expectancy_bps"] > 0,
        "pf_1_15": summary["metrics"]["profit_factor"] >= 1.15,
        "drawdown_8pct": summary["metrics"]["max_drawdown"] <= 0.08,
        "expectancy_lcb_positive": bool(
            np.isfinite(summary["expectancy_lcb_95_bps"])
            and summary["expectancy_lcb_95_bps"] > 0
        ),
        "majority_positive_active_days": summary["positive_active_days"] > 0.5,
    }


def _evaluate_cost_frontier(
    matrix: pd.DataFrame,
    *,
    cost_bps: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    periods = {
        "2024": ("2024-01-01", pd.Timestamp("2025-01-01", tz="UTC")),
        "2025": ("2025-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "prior": ("2024-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "audit": ("2026-01-01", HOLDOUT_START),
    }
    policies: dict[str, Any] = {}
    prior_passing: list[tuple[str, float]] = []
    selected_rows: dict[str, pd.DataFrame] = {}
    for label, threshold in thresholds.items():
        available = matrix.loc[np.isclose(matrix["room_threshold_bps"], threshold)].copy()
        expert_audits: dict[str, Any] = {}
        eligible: list[tuple[str, int, float]] = []
        for family in FAMILIES:
            for horizon in HORIZONS:
                expert_rows = _non_overlapping(
                    available.loc[
                        available["event_family"].eq(family)
                        & available["expert_breakout_bars"].eq(horizon)
                    ]
                )
                train = _summary(
                    _period(expert_rows, *periods["2024"]), cost_bps, bootstrap=False
                )
                validation = _summary(
                    _period(expert_rows, *periods["2025"]), cost_bps, bootstrap=False
                )
                accepted = (
                    train["metrics"]["expectancy_bps"] > 0
                    and validation["metrics"]["expectancy_bps"] > 0
                    and train["metrics"]["profit_factor"] >= 1.05
                    and validation["metrics"]["profit_factor"] >= 1.05
                )
                key = _expert_key(family, horizon)
                expert_audits[key] = {
                    "train": train,
                    "validation": validation,
                    "eligible": accepted,
                }
                if accepted:
                    eligible.append(
                        (
                            family,
                            horizon,
                            min(
                                train["metrics"]["expectancy_bps"],
                                validation["metrics"]["expectancy_bps"],
                            ),
                        )
                    )
        ranked_eligible = sorted(eligible, key=lambda item: (-item[2], item[0], item[1]))
        prefix_evaluations: list[dict[str, Any]] = []
        chosen: tuple[dict[str, Any], pd.DataFrame] | None = None
        for prefix_size in range(1, len(ranked_eligible) + 1):
            prefix = ranked_eligible[:prefix_size]
            candidate_rows = _select_expert_rows(prefix, available)
            prior_summaries = {
                name: _summary(_period(candidate_rows, *periods[name]), cost_bps)
                for name in ("2024", "2025", "prior")
            }
            gates = _prior_gates(prior_summaries)
            evaluation = {
                "prefix_size": prefix_size,
                "experts": [
                    _expert_key(family, horizon) for family, horizon, _ in prefix
                ],
                "summaries": prior_summaries,
                "gates": gates,
                "pass": all(gates.values()),
            }
            prefix_evaluations.append(evaluation)
            if evaluation["pass"] and (
                chosen is None
                or prior_summaries["prior"]["metrics"]["trades"]
                > chosen[0]["summaries"]["prior"]["metrics"]["trades"]
            ):
                chosen = (evaluation, candidate_rows)
        if chosen is None:
            fallback_rows = _select_expert_rows(ranked_eligible, available)
            summaries = {
                name: _summary(_period(fallback_rows, *periods[name]), cost_bps)
                for name in ("2024", "2025", "prior")
            }
            prior_gates = _prior_gates(summaries)
            selected_experts: list[str] = []
            prior_pass = False
        else:
            evaluation, rows = chosen
            summaries = evaluation["summaries"]
            prior_gates = evaluation["gates"]
            selected_experts = evaluation["experts"]
            prior_pass = True
            selected_rows[label] = rows
        if prior_pass:
            prior_passing.append((label, summaries["prior"]["metrics"]["trades"]))
        policies[label] = {
            "minimum_room_bps": threshold,
            "eligible_experts": [
                _expert_key(family, horizon) for family, horizon, _ in eligible
            ],
            "selected_prior_experts": selected_experts,
            "expert_audits": expert_audits,
            "prefix_evaluations": prefix_evaluations,
            "candidate_policy_prefixes_evaluated": len(prefix_evaluations),
            "summaries": summaries,
            "prior_gates": prior_gates,
            "prior_pass": prior_pass,
            "audit_evaluated": False,
            "audit_gates": None,
            "audit_pass": None,
        }
    selected_label = max(prior_passing, key=lambda item: item[1])[0] if prior_passing else None
    selected_policy = policies[selected_label] if selected_label is not None else None
    if selected_label is not None and selected_policy is not None:
        audit_summary = _summary(
            _period(selected_rows[selected_label], *periods["audit"]), cost_bps
        )
        audit_gates = _audit_gates(audit_summary)
        selected_policy["summaries"]["audit"] = audit_summary
        selected_policy["audit_evaluated"] = True
        selected_policy["audit_gates"] = audit_gates
        selected_policy["audit_pass"] = all(audit_gates.values())
    return {
        "round_trip_cost_bps": cost_bps,
        "policies": policies,
        "selected_on_prior": selected_label,
        "research_signal": bool(selected_policy and selected_policy["audit_pass"]),
    }


def _frozen_control_audit(matrix: pd.DataFrame) -> dict[str, Any]:
    baseline = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    expected = {
        int(expert["breakout_bars"]): expert for expert in baseline["experts"]
    }
    horizons: dict[str, Any] = {}
    for horizon in FROZEN_HORIZONS:
        rows = _non_overlapping(
            matrix.loc[
                np.isclose(matrix["room_threshold_bps"], 24.0)
                & matrix["event_family"].eq("IMPULSE_PULLBACK")
                & matrix["expert_breakout_bars"].eq(horizon)
            ]
        )
        entries = pd.to_datetime(rows["entry_timestamp"], utc=True)
        train = _metrics(rows.loc[entries.dt.year.eq(2024)])
        validation = _metrics(rows.loc[entries.dt.year.eq(2025)])
        reference = expected[horizon]
        passed = (
            len(rows) == int(reference["events"])
            and np.isclose(
                train["expectancy_bps"], reference["train"]["expectancy_bps"]
            )
            and np.isclose(
                validation["expectancy_bps"],
                reference["validation"]["expectancy_bps"],
            )
        )
        horizons[str(horizon)] = {
            "events": len(rows),
            "expected_events": int(reference["events"]),
            "train_expectancy_bps": train["expectancy_bps"],
            "expected_train_expectancy_bps": reference["train"]["expectancy_bps"],
            "validation_expectancy_bps": validation["expectancy_bps"],
            "expected_validation_expectancy_bps": reference["validation"][
                "expectancy_bps"
            ],
            "passed": bool(passed),
        }
    audit = {
        "baseline_protocol_hash": baseline["protocol_hash"],
        "room_threshold_bps": 24.0,
        "horizons": horizons,
        "passed": all(value["passed"] for value in horizons.values()),
    }
    if not audit["passed"]:
        raise RuntimeError("Stateful room control does not reproduce frozen V8")
    return audit


def run() -> dict[str, Any]:
    matrix = build_matrix()
    control_audit = _frozen_control_audit(matrix)
    profiles: dict[str, Any] = {}
    research_profiles: list[str] = []
    for profile in PAPER_PROFILES:
        cost = profile_cost_bps(profile)
        profiles[profile] = _evaluate_cost_frontier(
            matrix,
            cost_bps=cost,
            thresholds={
                f"{multiplier:g}x": multiplier * cost
                for multiplier in ROOM_COST_MULTIPLIERS
            },
        )
        if profiles[profile]["research_signal"]:
            research_profiles.append(profile)
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "outcome_protocol_hash": OUTCOME_PROTOCOL_HASH,
        "frozen_v8_control_audit": control_audit,
        "unique_signals": int(
            matrix[["signal_timestamp", "direction"]].drop_duplicates().shape[0]
        ),
        "profiles": profiles,
        "candidate_policy_prefixes_evaluated": sum(
            policy["candidate_policy_prefixes_evaluated"]
            for profile in profiles.values()
            for policy in profile["policies"].values()
        ),
        "research_profiles": research_profiles,
        "verdict": (
            "ROOM_FRONTIER_RESEARCH_ONLY"
            if research_profiles
            else "BASE_ONLY_NO_ROOM_FRONTIER_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    _atomic_json(
        STATUS,
        {
            "phase": "complete",
            "percent": 100,
            "detail": report["verdict"],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
