from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.bitunix_fees import futures_fee_bps
from adaptive_bot.musca_v4_research import (
    BARS,
    COST_BPS,
    MINUTES,
    _metrics,
    _non_overlapping,
    build_events,
    label_events,
)

REPORT = Path("data/reports/musca_v8_multi_horizon.json")
STATUS = Path("data/reports/musca_v8_multi_horizon.status.json")
HORIZONS = (3, 6, 12, 24, 48)
HOLDOUT_START = pd.Timestamp("2026-05-11T11:30:00Z")
PROTOCOL = {
    "name": "musca_v8_multi_horizon_impulse_pullback_v1",
    "base": "musca_v4_multi_anchor_v1",
    "breakout_bars": list(HORIZONS),
    "fixed": {
        "relative_volume_min": 1.0,
        "pullback_depth_atr": 0.25,
        "restart_expiry_bars": 6,
        "minimum_room_bps": 3 * COST_BPS,
        "cost_bps": COST_BPS,
        "management": "unchanged_v4",
    },
    "expert_selection": "positive_EV_and_PF_1.05_in_both_2024_and_2025",
    "test": "2026_before_sealed_holdout",
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
PAPER_PROFILES = tuple(f"VIP{level}" for level in range(6))
HISTORICAL_NON_FEE_RESERVE_BPS = 1.0
LIVE_EVENT_MAX_AGE = pd.Timedelta(minutes=6)
_LIVE_EVENT_CACHE: tuple[float, pd.Timestamp, list[dict[str, Any]]] | None = None


def expert_id(breakout_bars: int) -> str:
    payload = {"protocol_hash": PROTOCOL_HASH, "breakout_bars": breakout_bars}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"musca-v8-h{breakout_bars}-{digest}"


def profile_cost_bps(profile: str) -> float:
    level = int(profile.removeprefix("VIP"))
    _, taker_bps = futures_fee_bps(level)
    return 2 * taker_bps + HISTORICAL_NON_FEE_RESERVE_BPS


def _with_cost(rows: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    adjusted = rows.copy()
    if adjusted.empty:
        return adjusted
    adjusted["net_return_bps"] = adjusted["gross_return_bps"] - cost_bps
    adjusted["stress_return_bps"] = adjusted["gross_return_bps"] - 2 * cost_bps
    side = adjusted["direction"].astype(float)
    risk_bps = (
        side * (adjusted["entry_price"] - adjusted["stop_price"]) / adjusted["entry_price"] * 10_000
    )
    adjusted["net_return_r"] = adjusted["net_return_bps"] / risk_bps
    return adjusted


def _select_rows(
    results: list[tuple[dict[str, Any], pd.DataFrame]],
    eligible_horizons: list[int],
    cost_bps: float,
) -> pd.DataFrame:
    selected = [rows for audit, rows in results if int(audit["breakout_bars"]) in eligible_horizons]
    if not selected:
        return results[0][1].iloc[0:0].copy() if results else pd.DataFrame()
    robustness = {
        int(audit["breakout_bars"]): min(
            audit["train"]["expectancy_bps"] + COST_BPS - cost_bps,
            audit["validation"]["expectancy_bps"] + COST_BPS - cost_bps,
        )
        for audit, _ in results
        if int(audit["breakout_bars"]) in eligible_horizons
    }
    combined = pd.concat(selected, ignore_index=True)
    combined["robustness"] = combined["expert_breakout_bars"].map(robustness)
    combined = combined.sort_values("robustness", ascending=False).drop_duplicates(
        "signal_timestamp"
    )
    return _non_overlapping(combined)


def _live_events(evaluated_at: pd.Timestamp) -> list[dict[str, Any]]:
    global _LIVE_EVENT_CACHE

    monotonic_now = time.monotonic()
    if _LIVE_EVENT_CACHE is not None and monotonic_now - _LIVE_EVENT_CACHE[0] < 240:
        return _LIVE_EVENT_CACHE[2]
    from adaptive_bot.musca_v4 import _market_frame

    frame = _market_frame()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expert_audits = {
        int(audit["breakout_bars"]): audit
        for audit in report.get("experts", [])
        if audit.get("eligible")
    }
    events: list[dict[str, Any]] = []
    for horizon, audit in expert_audits.items():
        candidates = build_events(frame, breakout_bars=horizon)
        candidates = candidates.loc[candidates["event_family"].eq("IMPULSE_PULLBACK")]
        robust_gross = (
            min(
                float(audit["train"]["expectancy_bps"]),
                float(audit["validation"]["expectancy_bps"]),
            )
            + COST_BPS
        )
        robust_tp1_probability = min(
            float(audit["train_tp1_rate"]), float(audit["validation_tp1_rate"])
        )
        for event in candidates.to_dict("records"):
            event_record = {str(key): value for key, value in event.items()}
            available_at = pd.Timestamp(event_record["available_at"])
            if available_at.tzinfo is None:
                available_at = available_at.tz_localize("UTC")
            else:
                available_at = available_at.tz_convert("UTC")
            if available_at <= evaluated_at:
                events.append(
                    event_record
                    | {
                        "available_at": available_at,
                        "breakout_bars": horizon,
                        "expert_id": str(audit["expert_id"]),
                        "robust_expected_gross_bps": robust_gross,
                        "robust_tp1_probability": robust_tp1_probability,
                    }
                )
    _LIVE_EVENT_CACHE = (monotonic_now, evaluated_at, events)
    return events


def live_candidate(
    profile: str | None,
    evaluated_at: pd.Timestamp,
    *,
    cost_bps: float | None = None,
) -> dict[str, Any] | None:
    """Return the newest frozen V8 event that is still executable for one fee profile."""
    if not REPORT.exists():
        return None
    now = (
        evaluated_at.tz_localize("UTC")
        if evaluated_at.tzinfo is None
        else evaluated_at.tz_convert("UTC")
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if cost_bps is None:
        profile_audit = report.get("paper_profiles", {}).get(profile, {})
        eligible = set(int(value) for value in profile_audit.get("eligible_horizons", []))
    else:
        eligible = {
            int(audit["breakout_bars"])
            for audit in report.get("experts", [])
            if audit.get("eligible")
            and min(
                float(audit["train"]["expectancy_bps"]) + COST_BPS - cost_bps,
                float(audit["validation"]["expectancy_bps"]) + COST_BPS - cost_bps,
            )
            > 0
        }
    candidates = [
        event
        for event in _live_events(now)
        if int(event["breakout_bars"]) in eligible
        and now - LIVE_EVENT_MAX_AGE <= event["available_at"] <= now
    ]
    if not candidates:
        return None
    latest_time = max(event["available_at"] for event in candidates)
    same_signal = [event for event in candidates if event["available_at"] == latest_time]
    selected = max(same_signal, key=lambda event: event["robust_expected_gross_bps"])
    direction = "LONG" if int(selected["direction"]) > 0 else "SHORT"
    side = int(selected["direction"])
    checks = [
        {
            "name": "trend_1h_4h",
            "passed": True,
            "actual": f"{float(selected['return_1h']):.4%} / {float(selected['return_4h']):.4%}",
            "requirement": f"both agree with {direction}",
        },
        {
            "name": "spot_confirmation",
            "passed": True,
            "actual": float(selected["spot_return_1h"]),
            "requirement": f"Binance spot agrees with {direction}",
        },
        {
            "name": "impulse_breakout",
            "passed": True,
            "actual": int(selected["breakout_bars"]),
            "requirement": "closed 5m breakout at a selected frozen horizon",
        },
        {
            "name": "relative_volume",
            "passed": True,
            "actual": float(selected["relative_volume"]),
            "requirement": ">= 1.0 at impulse",
        },
        {
            "name": "taker_flow_confirmation",
            "passed": True,
            "actual": float(selected["taker_imbalance"]),
            "requirement": f"signed taker flow agrees with {direction}",
        },
        {
            "name": "vwap_pullback",
            "passed": True,
            "actual": int(selected["confluence_score"]),
            "requirement": "pullback touches daily/impulse/swing VWAP zone",
        },
        {
            "name": "causal_restart",
            "passed": True,
            "actual": side,
            "requirement": "closed 5m restart with flow and spot confirmation",
        },
    ]
    return {
        "setup": f"IMPULSE_PULLBACK_H{selected['breakout_bars']}",
        "direction": direction,
        "candidate": True,
        "setup_active": True,
        "passed_checks": 7,
        "total_checks": 7,
        "first_failed_check": None,
        "checks": checks,
        "policy_source": "MUSCA_V8_FROZEN_BASE",
        "expert_id": selected["expert_id"],
        "breakout_bars": int(selected["breakout_bars"]),
        "signal_timestamp": pd.Timestamp(selected["signal_timestamp"]).isoformat(),
        "available_at": pd.Timestamp(selected["available_at"]).isoformat(),
        "impulse_anchor_at": pd.Timestamp(selected["impulse_anchor_at"]).isoformat(),
        "swing_anchor_at": pd.Timestamp(selected["swing_anchor_at"]).isoformat(),
        "operating_vwap": float(selected["operating_vwap"]),
        "stop_price": float(selected["stop_price"]),
        "robust_expected_gross_bps": float(selected["robust_expected_gross_bps"]),
        "target_probability": float(selected["robust_tp1_probability"]),
        "management_style": "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M",
        "partial_target_fraction": 0.5,
        "maximum_hold_minutes": 360,
        "assumed_round_trip_cost_bps": cost_bps,
        "protocol_hash": PROTOCOL_HASH,
    }


def live_candidate_for_cost(cost_bps: float, evaluated_at: pd.Timestamp) -> dict[str, Any] | None:
    if not 0 <= cost_bps < 100:
        raise ValueError("round-trip cost must be between 0 and 100 bps")
    return live_candidate(None, evaluated_at, cost_bps=cost_bps)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _expert(breakout_bars: int) -> tuple[dict[str, Any], pd.DataFrame]:
    events = build_events(pd.read_parquet(BARS), breakout_bars=breakout_bars)
    labeled = label_events(events, pd.read_parquet(MINUTES))
    rows = _non_overlapping(labeled.loc[labeled["event_family"].eq("IMPULSE_PULLBACK")])
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    rows = rows.loc[time.lt(HOLDOUT_START)].copy()
    rows["expert_breakout_bars"] = breakout_bars
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    train, validation = rows.loc[time.dt.year.eq(2024)], rows.loc[time.dt.year.eq(2025)]
    train_metrics, validation_metrics = _metrics(train), _metrics(validation)
    eligible = (
        train_metrics["expectancy_bps"] > 0
        and validation_metrics["expectancy_bps"] > 0
        and train_metrics["profit_factor"] >= 1.05
        and validation_metrics["profit_factor"] >= 1.05
    )
    return (
        {
            "expert_id": expert_id(breakout_bars),
            "breakout_bars": breakout_bars,
            "events": len(rows),
            "train": train_metrics,
            "validation": validation_metrics,
            "train_stress": _metrics(train, "stress_return_bps"),
            "validation_stress": _metrics(validation, "stress_return_bps"),
            "train_tp1_rate": float(train["tp1"].mean()) if len(train) else 0.0,
            "validation_tp1_rate": (float(validation["tp1"].mean()) if len(validation) else 0.0),
            "eligible": eligible,
        },
        rows,
    )


def run() -> dict[str, Any]:
    _write(
        STATUS,
        {
            "phase": "parallel_experts",
            "percent": 5,
            "detail": "5 horizons / 4 CPU workers",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_expert, HORIZONS))
    audits = [audit for audit, _ in results]
    eligible_horizons = [int(audit["breakout_bars"]) for audit in audits if audit["eligible"]]
    combined = _select_rows(results, eligible_horizons, COST_BPS)
    if not combined.empty:
        combined_time = pd.to_datetime(combined["entry_timestamp"], utc=True)
        prior = combined.loc[combined_time.dt.year.lt(2026)].copy()
        test = combined.loc[combined_time.dt.year.eq(2026)].copy()
        prior_metrics = _metrics(prior)
    else:
        test = combined
        prior_metrics = _metrics(combined)
    metrics, stress = _metrics(test), _metrics(test, "stress_return_bps")
    gates = {
        "eligible_expert_exists": bool(eligible_horizons),
        "test_trades_50": metrics["trades"] >= 50,
        "test_expectancy_positive": metrics["expectancy_bps"] > 0,
        "test_pf_1_15": metrics["profit_factor"] >= 1.15,
        "test_drawdown_8pct": metrics["max_drawdown"] <= 0.08,
        "test_stress_nonnegative": stress["expectancy_bps"] >= 0,
    }
    profile_audit: dict[str, Any] = {}
    for profile in PAPER_PROFILES:
        cost = profile_cost_bps(profile)
        eligible_horizons = [
            int(audit["breakout_bars"])
            for audit in audits
            if audit["eligible"]
            and min(
                audit["train"]["expectancy_bps"] + COST_BPS - cost,
                audit["validation"]["expectancy_bps"] + COST_BPS - cost,
            )
            > 0
        ]
        profile_rows = _select_rows(results, eligible_horizons, cost)
        if profile_rows.empty:
            profile_prior = profile_test = profile_rows
        else:
            profile_time = pd.to_datetime(profile_rows["entry_timestamp"], utc=True)
            profile_prior = _with_cost(profile_rows.loc[profile_time.dt.year.lt(2026)], cost)
            profile_test = _with_cost(profile_rows.loc[profile_time.dt.year.eq(2026)], cost)
        historical = _metrics(profile_prior)
        historical_stress = _metrics(profile_prior, "stress_return_bps")
        oos = _metrics(profile_test)
        oos_stress = _metrics(profile_test, "stress_return_bps")
        profile_audit[profile] = {
            "assumed_round_trip_cost_bps": cost,
            "eligible_horizons": eligible_horizons,
            "historical": historical,
            "historical_stress_2x": historical_stress,
            "oos_2026": oos,
            "oos_2026_stress_2x": oos_stress,
            "paper_eligible": bool(eligible_horizons)
            and historical["expectancy_bps"] > 0
            and oos["expectancy_bps"] > 0,
            "real_capital_eligible": False,
        }
    paper_eligible_profiles = [
        profile for profile, audit in profile_audit.items() if audit["paper_eligible"]
    ]
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "experts": audits,
        "prior_combined": prior_metrics,
        "test": metrics,
        "test_stress": stress,
        "paper_profiles": profile_audit,
        "paper_eligible_profiles": paper_eligible_profiles,
        "deployable_profiles": [],
        "gates": gates,
        "verdict": (
            "RESEARCH_BASE_ALPHA_READY"
            if paper_eligible_profiles
            and all(value for key, value in gates.items() if key != "test_trades_50")
            else "NO_MULTI_HORIZON_EDGE"
        ),
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _write(REPORT, report)
    _write(
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
    print(json.dumps(run(), indent=2))
