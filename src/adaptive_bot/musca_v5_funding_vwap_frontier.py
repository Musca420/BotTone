from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.adapters.bitunix.market_data import _get_json
from adaptive_bot.hybrid_policy import _binance_funding
from adaptive_bot.musca_v4_research import (
    BARS,
    MINUTES,
    _non_overlapping,
    build_features,
    label_events,
)
from adaptive_bot.musca_v5_fine_tuning import _atomic_json
from adaptive_bot.musca_v5_room_frontier import (
    MATRIX as BASE_MATRIX,
)
from adaptive_bot.musca_v5_room_frontier import (
    REPORT as BASE_FRONTIER_REPORT,
)
from adaptive_bot.musca_v5_room_frontier import (
    _audit_gates,
    _period,
    _prior_gates,
    _select_expert_rows,
    _summary,
)
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    PAPER_PROFILES,
    profile_cost_bps,
)
from adaptive_bot.musca_v8_multi_horizon import (
    PROTOCOL_HASH as BASE_PROTOCOL_HASH,
)

ROOT = Path("data/ml/musca_v5")
FUNDING_HISTORY = ROOT / "binance_btcusdt_funding_history.parquet"
MATRIX = ROOT / "funding_vwap_frontier_matrix.parquet"
REPORT = Path("data/reports/musca_v5_funding_vwap_frontier.json")
STATUS = Path("data/reports/musca_v5_funding_vwap_frontier.status.json")
OFFICIAL_FUNDING_URL = (
    "https://developers.binance.com/en/docs/catalog/"
    "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
    "#get-funding-rate-history"
)
ROOM_MULTIPLIERS = (1.5, 2.0, 3.0)
RISK_BPS_RANGE = (12.0, 200.0)
PROTOCOL = {
    "name": "musca_v5_funding_cycle_vwap_pullback_frontier",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "family": "FUNDING_CYCLE_VWAP_PULLBACK",
    "funding_source": "Binance BTCUSDT official /fapi/v1/fundingRate",
    "funding_documentation": OFFICIAL_FUNDING_URL,
    "direction": "frozen_V8_1h_4h_ema_spot_confirmation",
    "extension": {
        "outside_funding_vwap_band": True,
        "relative_volume_min": 1.0,
        "perp_taker_imbalance_directional_min": 0.05,
        "spot_15m_directional": True,
    },
    "pullback": {
        "minimum_depth_atr": 0.25,
        "quieter_than_extension": True,
        "restart_expiry_bars_5m": 6,
        "restart": "prior_bar_break_plus_perp_flow_plus_spot_15m",
    },
    "room_cost_multipliers": list(ROOM_MULTIPLIERS),
    "risk_bps": list(RISK_BPS_RANGE),
    "management": "frozen_V8_half_at_1_5R_cost_protected_15m_trail_6h",
    "selection": "2024_2025_only_then_single_2026_preholdout_audit",
    "economic_costs": "observed_profile_taker_costs_1x",
    "cost_stress": "2x_diagnostic_only",
    "holdout_opened": False,
    "changes_to_active_paper": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_parquet(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": round(percent, 2),
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def ensure_funding_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if FUNDING_HISTORY.exists():
        cached = pd.read_parquet(FUNDING_HISTORY)
        times = pd.to_datetime(cached["funding_timestamp"], utc=True)
        if (
            not cached.empty
            and times.min() <= start + pd.Timedelta(hours=12)
            and times.max() >= end - pd.Timedelta(hours=12)
        ):
            return cached
    _status("funding_history", 5, "Binance BTCUSDT official funding history")
    downloaded = _binance_funding(
        _get_json,
        "BTCUSDT",
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
    )
    if downloaded.empty:
        raise RuntimeError("official Binance funding history is unavailable")
    downloaded = downloaded.rename(columns={"timestamp": "funding_timestamp"})
    downloaded["funding_timestamp"] = pd.to_datetime(
        pd.to_numeric(downloaded["funding_timestamp"], errors="raise"),
        unit="ms",
        utc=True,
    )
    downloaded["funding_rate"] = pd.to_numeric(downloaded["funding_rate"], errors="raise")
    downloaded["source"] = "observed:binance-official-rest-fapi-v1-fundingRate"
    downloaded = downloaded.drop_duplicates("funding_timestamp", keep="last").sort_values(
        "funding_timestamp"
    )
    if downloaded["funding_timestamp"].min() > start + pd.Timedelta(hours=12) or downloaded[
        "funding_timestamp"
    ].max() < end - pd.Timedelta(hours=12):
        raise RuntimeError("official Binance funding history coverage is incomplete")
    _write_parquet(FUNDING_HISTORY, downloaded)
    return downloaded


def add_funding_cycle_vwap(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    data = build_features(bars)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    funding_times = pd.DatetimeIndex(
        pd.to_datetime(funding["funding_timestamp"], utc=True).sort_values().unique()
    )
    cycle = funding_times.searchsorted(timestamps, side="right") - 1
    data["funding_cycle_id"] = cycle
    valid = cycle >= 0
    data["funding_anchor_at"] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    if valid.any():
        data.loc[valid, "funding_anchor_at"] = funding_times.take(cycle[valid]).to_numpy()
    bar_vwap = data["perp_quote_volume"] / data["perp_volume"].replace(0, np.nan)
    group = data["funding_cycle_id"].where(valid)
    cumulative_volume = data["perp_volume"].where(valid).groupby(group).cumsum()
    cumulative_quote = data["perp_quote_volume"].where(valid).groupby(group).cumsum()
    cumulative_second = (data["perp_volume"].where(valid) * bar_vwap.pow(2)).groupby(group).cumsum()
    data["funding_vwap"] = cumulative_quote / cumulative_volume
    data["funding_sigma"] = np.sqrt(
        (cumulative_second / cumulative_volume - data["funding_vwap"].pow(2)).clip(lower=0)
    )
    data["funding_feature_available_at"] = data["available_at"]
    data.loc[~valid, ["funding_vwap", "funding_sigma"]] = np.nan
    return data


def build_funding_events(
    features: pd.DataFrame,
    *,
    minimum_room_bps: float,
    audit: dict[str, int] | None = None,
) -> pd.DataFrame:
    data = features.sort_values("timestamp").reset_index(drop=True)
    numeric = {
        column: data[column].to_numpy(float)
        for column in data.columns
        if column != "timestamp" and pd.api.types.is_numeric_dtype(data[column])
    }
    high = numeric["perp_high"]
    low = numeric["perp_low"]
    close = numeric["perp_close"]
    cycles = numeric["funding_cycle_id"].astype(int)
    directions = numeric["direction"].astype(int)
    valid_data = (
        data["data_valid"].to_numpy(bool)
        if "data_valid" in data
        else np.ones(len(data), dtype=bool)
    )
    stages = {name: 0 for name in ("extension", "armed", "restart", "room", "risk", "event")}
    events: list[dict[str, Any]] = []
    active_cycle = -1
    side = 0
    extension_index = -1
    extension_extreme = np.nan
    pullback_extreme = np.nan
    armed_at = -1

    def reset_setup() -> None:
        nonlocal side, extension_index, extension_extreme, pullback_extreme, armed_at
        side = 0
        extension_index = -1
        extension_extreme = np.nan
        pullback_extreme = np.nan
        armed_at = -1

    for current in range(1, len(data) - 1):
        cycle = int(cycles[current])
        if cycle < 0 or not valid_data[current]:
            reset_setup()
            continue
        if cycle != active_cycle:
            active_cycle = cycle
            reset_setup()
        current_direction = int(directions[current])
        if side and current_direction != side:
            reset_setup()
        center = numeric["funding_vwap"][current]
        sigma = numeric["funding_sigma"][current]
        current_atr = numeric["atr"][current]
        if not np.isfinite(center) or not np.isfinite(sigma) or not np.isfinite(current_atr):
            continue
        band = max(sigma, 0.25 * current_atr)
        if side == 0:
            extension_side = current_direction
            extended = (
                extension_side != 0
                and extension_side * (close[current] - center) >= band
                and numeric["return_15m"][current] * extension_side > 0
                and numeric["relative_volume"][current] >= 1.0
                and numeric["taker_imbalance"][current] * extension_side > 0.05
                and numeric["spot_return_15m"][current] * extension_side > 0
            )
            if not extended:
                continue
            side = extension_side
            extension_index = current
            extension_extreme = high[current] if side > 0 else low[current]
            pullback_extreme = extension_extreme
            stages["extension"] += 1
            continue
        extension_extreme = (
            max(extension_extreme, high[current])
            if side > 0
            else min(extension_extreme, low[current])
        )
        pullback_extreme = (
            min(pullback_extreme, low[current])
            if side > 0
            else max(pullback_extreme, high[current])
        )
        depth = side * (extension_extreme - pullback_extreme) / current_atr
        in_zone = abs(close[current] - center) <= band
        quieter = (
            numeric["perp_quote_volume"][current] < numeric["perp_quote_volume"][extension_index]
        )
        if armed_at < 0 and in_zone and depth >= 0.25 and quieter:
            armed_at = current
            stages["armed"] += 1
            continue
        if armed_at < 0:
            continue
        if current - armed_at > 6:
            reset_setup()
            continue
        restarted = (
            close[current] > high[current - 1] if side > 0 else close[current] < low[current - 1]
        )
        if restarted:
            stages["restart"] += 1
        confirmed = (
            restarted
            and numeric["taker_imbalance"][current] * side > 0
            and numeric["spot_return_15m"][current] * side > 0
            and (close[current] - center) * side > 0
        )
        if not confirmed:
            continue
        room = side * (extension_extreme - close[current]) / close[current] * 10_000
        if room < minimum_room_bps:
            continue
        stages["room"] += 1
        zone_edge = center - side * band
        stop = (
            min(pullback_extreme, zone_edge) - 0.1 * current_atr
            if side > 0
            else max(pullback_extreme, zone_edge) + 0.1 * current_atr
        )
        risk_bps = side * (close[current] - stop) / close[current] * 10_000
        if not (RISK_BPS_RANGE[0] <= risk_bps <= RISK_BPS_RANGE[1]):
            continue
        stages["risk"] += 1
        row = data.iloc[current]
        events.append(
            {
                "protocol_hash": PROTOCOL_HASH,
                "signal_index": current,
                "signal_timestamp": pd.Timestamp(row["timestamp"]),
                "available_at": pd.Timestamp(row["available_at"]),
                "direction": side,
                "event_family": "FUNDING_CYCLE_VWAP_PULLBACK",
                "event_family_code": 3,
                "funding_anchor_at": pd.Timestamp(row["funding_anchor_at"]),
                "anchor_available_at": pd.Timestamp(row["funding_anchor_at"]),
                "operating_vwap": center,
                "operating_sigma": sigma,
                "stop_price": stop,
                "return_1h": float(row["return_1h"]),
                "return_4h": float(row["return_4h"]),
                "ema_spread_atr": float(row["ema_spread_atr"]),
                "relative_volume": float(row["relative_volume"]),
                "taker_imbalance": float(row["taker_imbalance"]),
                "spot_return_1h": float(row["spot_return_1h"]),
                "spot_perp_divergence_bps": float(row["spot_perp_divergence_bps"]),
                "pullback_depth_atr": depth,
                "room_bps": room,
                "atr_percentile": float(row["atr_percentile"]),
                "hour_sin": float(row["hour_sin"]),
                "hour_cos": float(row["hour_cos"]),
            }
        )
        stages["event"] += 1
        reset_setup()
    if audit is not None:
        audit.update(stages)
    return pd.DataFrame(events)


def _base_rows(profile: str, base_matrix: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    profile_report = report["profiles"][profile]
    selected = profile_report.get("selected_on_prior")
    if selected is None:
        return base_matrix.iloc[0:0].copy()
    policy = profile_report["policies"][selected]
    threshold = float(policy["minimum_room_bps"])
    rows = base_matrix.loc[np.isclose(base_matrix["room_threshold_bps"], threshold)].copy()
    eligible: list[tuple[str, int, float]] = []
    for key in policy["selected_prior_experts"]:
        family, raw_horizon = key.split(":H", 1)
        expert = policy["expert_audits"][key]
        robustness = min(
            expert["train"]["metrics"]["expectancy_bps"],
            expert["validation"]["metrics"]["expectancy_bps"],
        )
        eligible.append((family, int(raw_horizon), float(robustness)))
    return _select_expert_rows(eligible, rows)


def _merge_rows(base: pd.DataFrame, funding: pd.DataFrame, robustness: float) -> pd.DataFrame:
    left = base.copy()
    if not left.empty and "prior_robustness_bps" not in left:
        left["prior_robustness_bps"] = np.inf
    right = funding.copy()
    right["prior_robustness_bps"] = robustness
    combined = pd.concat([left, right], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.sort_values(
        ["signal_timestamp", "prior_robustness_bps"], ascending=[True, False]
    ).drop_duplicates(["signal_timestamp", "direction"], keep="first")
    return _non_overlapping(combined)


def run() -> dict[str, Any]:
    bars = pd.read_parquet(BARS)
    start = pd.to_datetime(bars["timestamp"], utc=True).min()
    funding = ensure_funding_history(start, HOLDOUT_START)
    features = add_funding_cycle_vwap(bars, funding)
    minutes = pd.read_parquet(MINUTES)
    thresholds = sorted(
        {
            profile_cost_bps(profile) * multiplier
            for profile in PAPER_PROFILES
            for multiplier in ROOM_MULTIPLIERS
        }
    )
    parts: list[pd.DataFrame] = []
    funnels: dict[str, Any] = {}
    for number, threshold in enumerate(thresholds, start=1):
        audit: dict[str, int] = {}
        events = build_funding_events(features, minimum_room_bps=threshold, audit=audit)
        events = events.loc[pd.to_datetime(events["available_at"], utc=True).lt(HOLDOUT_START)]
        labeled = label_events(events, minutes)
        labeled = labeled.loc[
            pd.to_datetime(labeled["exit_timestamp"], utc=True).lt(HOLDOUT_START)
        ].copy()
        labeled["room_threshold_bps"] = threshold
        labeled["funding_vwap_protocol_hash"] = PROTOCOL_HASH
        parts.append(labeled)
        funnels[f"{threshold:g}"] = audit | {"labeled": len(labeled)}
        _status(
            "funding_vwap_matrix",
            10 + 50 * number / len(thresholds),
            f"room {threshold:g} bps ({number}/{len(thresholds)})",
        )
    matrix = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    _write_parquet(MATRIX, matrix)
    base_matrix = pd.read_parquet(BASE_MATRIX)
    base_report = json.loads(BASE_FRONTIER_REPORT.read_text(encoding="utf-8"))
    periods = {
        "2024": ("2024-01-01", pd.Timestamp("2025-01-01", tz="UTC")),
        "2025": ("2025-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "prior": ("2024-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "audit": ("2026-01-01", HOLDOUT_START),
    }
    profiles: dict[str, Any] = {}
    for number, profile in enumerate(PAPER_PROFILES, start=1):
        cost = profile_cost_bps(profile)
        base = _base_rows(profile, base_matrix, base_report)
        base_prior = _summary(_period(base, *periods["prior"]), cost)
        evaluations: list[dict[str, Any]] = []
        chosen: tuple[dict[str, Any], pd.DataFrame] | None = None
        for multiplier in ROOM_MULTIPLIERS:
            threshold = cost * multiplier
            family = _non_overlapping(
                matrix.loc[np.isclose(matrix["room_threshold_bps"], threshold)].copy()
            )
            family_years = {
                year: _summary(_period(family, *periods[year]), cost) for year in ("2024", "2025")
            }
            eligible = all(
                family_years[year]["metrics"]["expectancy_bps"] > 0
                and family_years[year]["metrics"]["profit_factor"] >= 1.05
                for year in ("2024", "2025")
            )
            robustness = min(
                family_years["2024"]["metrics"]["expectancy_bps"],
                family_years["2025"]["metrics"]["expectancy_bps"],
            )
            combined = _merge_rows(base, family if eligible else family.iloc[0:0], robustness)
            prior = {
                name: _summary(_period(combined, *periods[name]), cost)
                for name in ("2024", "2025", "prior")
            }
            gates = _prior_gates(prior)
            evaluation: dict[str, Any] = {
                "room_multiplier": multiplier,
                "minimum_room_bps": threshold,
                "family_eligible": eligible,
                "family_summaries": family_years,
                "combined_prior": prior,
                "prior_gates": gates,
                "prior_pass": all(gates.values()),
                "audit_evaluated": False,
            }
            evaluations.append(evaluation)
            if evaluation["prior_pass"] and (
                chosen is None
                or prior["prior"]["metrics"]["trades"]
                > chosen[0]["combined_prior"]["prior"]["metrics"]["trades"]
            ):
                chosen = (evaluation, combined)
        if chosen is not None:
            evaluation, selected_rows = chosen
            audit_summary = _summary(_period(selected_rows, *periods["audit"]), cost)
            evaluation["audit_evaluated"] = True
            evaluation["audit_summary"] = audit_summary
            evaluation["audit_gates"] = _audit_gates(audit_summary)
            evaluation["audit_pass"] = all(evaluation["audit_gates"].values())
        selected_label = f"{chosen[0]['room_multiplier']:g}x" if chosen is not None else None
        selected_audit = chosen[0].get("audit_summary") if chosen is not None else None
        profiles[profile] = {
            "round_trip_cost_bps_1x": cost,
            "base_prior": base_prior,
            "base_prior_trades": base_prior["metrics"]["trades"],
            "evaluations": evaluations,
            "selected_on_prior": selected_label,
            "research_signal": bool(
                chosen is not None
                and chosen[0].get("audit_pass", False)
                and selected_audit is not None
                and selected_audit["metrics"]["trades"]
                > _summary(_period(base, *periods["audit"]), cost)["metrics"]["trades"]
            ),
        }
        _status(
            "funding_vwap_audit",
            60 + 35 * number / len(PAPER_PROFILES),
            profile,
        )
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "funding_history": {
            "rows": len(funding),
            "start": pd.to_datetime(funding["funding_timestamp"], utc=True).min().isoformat(),
            "end": pd.to_datetime(funding["funding_timestamp"], utc=True).max().isoformat(),
            "sha256": _sha256(FUNDING_HISTORY),
            "source": OFFICIAL_FUNDING_URL,
        },
        "matrix_rows": len(matrix),
        "unique_signals": int(matrix["signal_timestamp"].nunique()),
        "funnels": funnels,
        "profiles": profiles,
        "candidate_policies_evaluated": len(PAPER_PROFILES) * len(ROOM_MULTIPLIERS),
        "verdict": (
            "FUNDING_VWAP_RESEARCH_FRONTIER"
            if any(value["research_signal"] for value in profiles.values())
            else "NO_FUNDING_VWAP_FREQUENCY_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, result)
    _status("complete", 100, result["verdict"])
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
