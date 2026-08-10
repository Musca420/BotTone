from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import (
    BARS,
    MINUTES,
    _non_overlapping,
    build_features,
)
from adaptive_bot.musca_v5_fine_tuning import _atomic_json
from adaptive_bot.musca_v5_funding_vwap_frontier import _base_rows
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
    _summary,
)
from adaptive_bot.musca_v8_multi_horizon import (
    HOLDOUT_START,
    HORIZONS,
    PAPER_PROFILES,
    profile_cost_bps,
)
from adaptive_bot.musca_v8_multi_horizon import (
    PROTOCOL_HASH as BASE_PROTOCOL_HASH,
)

ROOT = Path("data/ml/musca_v5")
MATRIX = ROOT / "touch_entry_frontier_matrix.parquet"
REPORT = Path("data/reports/musca_v5_touch_entry_frontier.json")
STATUS = Path("data/reports/musca_v5_touch_entry_frontier.status.json")
ROOM_MULTIPLIERS = (1.5, 2.0)
RISK_BPS_RANGE = (12.0, 200.0)
MAXIMUM_HOLDING_MINUTES = 60
FAMILY = "IMPULSE_VWAP_TOUCH"
PROTOCOL = {
    "name": "musca_v5_impulse_vwap_touch_entry_frontier",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "direction_and_impulse": "frozen_V8",
    "horizons": list(HORIZONS),
    "entry": "next_1m_open_after_first_causal_pullback_touch_without_restart_wait",
    "pullback": {
        "anchors": ["daily_vwap", "impulse_vwap", "swing_vwap"],
        "minimum_depth_atr": 0.25,
        "quieter_than_impulse": True,
    },
    "target": "full_exit_at_frozen_impulse_extreme",
    "stop": "beyond_observed_pullback_and_operating_vwap_band_never_widens",
    "room_cost_multipliers": list(ROOM_MULTIPLIERS),
    "risk_bps": list(RISK_BPS_RANGE),
    "maximum_holding_minutes": MAXIMUM_HOLDING_MINUTES,
    "intrabar": "stop_wins",
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


def _vwap_prefixes(volume: np.ndarray, quote: np.ndarray) -> tuple[np.ndarray, ...]:
    valid = np.isfinite(volume) & np.isfinite(quote) & (volume > 0)
    safe_volume = np.where(valid, volume, 0.0)
    safe_quote = np.where(valid, quote, 0.0)
    weighted_square = np.zeros_like(safe_quote)
    np.divide(quote * quote, volume, out=weighted_square, where=valid)
    return tuple(
        np.concatenate(([0.0], np.cumsum(values)))
        for values in (safe_volume, safe_quote, weighted_square)
    )


def _vwap_band_from_prefixes(
    prefixes: tuple[np.ndarray, ...], start: int, end: int
) -> tuple[float, float]:
    volume_prefix, quote_prefix, square_prefix = prefixes
    total = float(volume_prefix[end + 1] - volume_prefix[start])
    if total <= 0:
        return np.nan, np.nan
    quote = float(quote_prefix[end + 1] - quote_prefix[start])
    weighted_square = float(square_prefix[end + 1] - square_prefix[start])
    center = quote / total
    variance = max(0.0, weighted_square / total - center * center)
    return center, float(np.sqrt(variance))


def _vwap_bands_from_prefixes(
    prefixes: tuple[np.ndarray, ...], start: int, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    volume_prefix, quote_prefix, square_prefix = prefixes
    total = volume_prefix[ends + 1] - volume_prefix[start]
    quote = quote_prefix[ends + 1] - quote_prefix[start]
    weighted_square = square_prefix[ends + 1] - square_prefix[start]
    center = np.full_like(total, np.nan)
    second = np.full_like(total, np.nan)
    np.divide(quote, total, out=center, where=total > 0)
    np.divide(weighted_square, total, out=second, where=total > 0)
    sigma = np.sqrt(np.maximum(0.0, second - center * center))
    return center, sigma


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


def _impulse_mask(
    data: pd.DataFrame, numeric: dict[str, np.ndarray], breakout_bars: int
) -> np.ndarray:
    close = numeric["perp_close"]
    direction = numeric["direction"].astype(int)
    prior_high = (
        data["perp_high"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .max()
        .to_numpy(float)
    )
    prior_low = (
        data["perp_low"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .min()
        .to_numpy(float)
    )
    spot_high = (
        data["spot_high"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .max()
        .to_numpy(float)
    )
    spot_low = (
        data["spot_low"]
        .shift(1)
        .rolling(breakout_bars, min_periods=breakout_bars)
        .min()
        .to_numpy(float)
    )
    return (
        (direction > 0)
        & (close > prior_high)
        & (numeric["spot_close"] > spot_high)
        & (numeric["relative_volume"] >= 1.0)
        & (numeric["taker_imbalance"] > 0.05)
        & (numeric["spot_taker_imbalance"] > 0)
    ) | (
        (direction < 0)
        & (close < prior_low)
        & (numeric["spot_close"] < spot_low)
        & (numeric["relative_volume"] >= 1.0)
        & (numeric["taker_imbalance"] < -0.05)
        & (numeric["spot_taker_imbalance"] < 0)
    )


def build_touch_events(
    features: pd.DataFrame,
    *,
    breakout_bars: int,
    room_thresholds_bps: tuple[float, ...],
    audit: dict[str, int] | None = None,
) -> pd.DataFrame:
    data = features.sort_values("timestamp").reset_index(drop=True)
    numeric = {
        column: data[column].to_numpy(float)
        for column in data.columns
        if column != "timestamp" and pd.api.types.is_numeric_dtype(data[column])
    }
    high, low, close = numeric["perp_high"], numeric["perp_low"], numeric["perp_close"]
    volume, quote = numeric["perp_volume"], numeric["perp_quote_volume"]
    vwap_prefixes = _vwap_prefixes(volume, quote)
    direction = numeric["direction"].astype(int)
    impulse = _impulse_mask(data, numeric, breakout_bars)
    thresholds = tuple(sorted(set(room_thresholds_bps)))
    stages = {name: 0 for name in ("impulse", "touch", "room", "risk", "event")}
    events: list[dict[str, Any]] = []
    for raw_index in np.flatnonzero(impulse):
        index = int(raw_index)
        stages["impulse"] += 1
        side = int(direction[index])
        swing_start = max(0, index - 24)
        swing_slice = slice(swing_start, index + 1)
        swing_index = swing_start + int(
            np.nanargmin(low[swing_slice]) if side > 0 else np.nanargmax(high[swing_slice])
        )
        impulse_extreme = high[index] if side > 0 else low[index]
        current = np.arange(index + 1, min(index + 289, len(data) - 1))
        mismatch = np.flatnonzero(direction[current] != side)
        if mismatch.size:
            current = current[: int(mismatch[0])]
        if not current.size:
            continue
        pullback_extreme = (
            np.minimum.accumulate(low[current])
            if side > 0
            else np.maximum.accumulate(high[current])
        )
        impulse_center, impulse_sigma = _vwap_bands_from_prefixes(vwap_prefixes, index, current)
        swing_center, swing_sigma = _vwap_bands_from_prefixes(vwap_prefixes, swing_index, current)
        centers = np.column_stack((numeric["daily_vwap"][current], impulse_center, swing_center))
        sigmas = np.column_stack((numeric["daily_sigma"][current], impulse_sigma, swing_sigma))
        atr = numeric["atr"][current]
        widths = np.maximum(sigmas, 0.25 * atr[:, None])
        distances = np.abs(close[current, None] - centers)
        valid_zones = np.isfinite(centers) & np.isfinite(sigmas) & (distances <= widths)
        depth = side * (impulse_extreme - pullback_extreme) / atr
        quieter = numeric["perp_quote_volume"][current] < numeric["perp_quote_volume"][index]
        touch = valid_zones.any(axis=1) & (depth >= 0.25) & quieter
        closest_zone = np.argmin(np.where(valid_zones, distances, np.inf), axis=1)
        positions = np.arange(len(current))
        center = centers[positions, closest_zone]
        sigma = sigmas[positions, closest_zone]
        room = side * (impulse_extreme - close[current]) / close[current] * 10_000
        zone_edge = center - side * np.maximum(sigma, 0.25 * atr)
        stop = (
            np.minimum(pullback_extreme, zone_edge) - 0.1 * atr
            if side > 0
            else np.maximum(pullback_extreme, zone_edge) + 0.1 * atr
        )
        risk_bps = side * (close[current] - stop) / close[current] * 10_000
        valid_risk = (risk_bps >= RISK_BPS_RANGE[0]) & (risk_bps <= RISK_BPS_RANGE[1])
        accepted_positions: list[int] = []
        for threshold in thresholds:
            has_room = touch & (room >= threshold)
            accepted = np.flatnonzero(has_room & valid_risk)
            if not accepted.size:
                stages["room"] += int(has_room.sum())
                continue
            position = int(accepted[0])
            stages["room"] += int(has_room[: position + 1].sum())
            stages["risk"] += 1
            stages["event"] += 1
            accepted_positions.append(position)
            row = data.iloc[int(current[position])]
            events.append(
                {
                    "protocol_hash": PROTOCOL_HASH,
                    "signal_timestamp": pd.Timestamp(row["timestamp"]),
                    "available_at": pd.Timestamp(row["available_at"]),
                    "direction": side,
                    "event_family": FAMILY,
                    "event_family_code": 4,
                    "expert_breakout_bars": breakout_bars,
                    "impulse_anchor_at": pd.Timestamp(data.iloc[index]["timestamp"]),
                    "swing_anchor_at": pd.Timestamp(data.iloc[swing_index]["timestamp"]),
                    "anchor_available_at": pd.Timestamp(data.iloc[index]["available_at"]),
                    "operating_vwap": center[position],
                    "operating_sigma": sigma[position],
                    "stop_price": stop[position],
                    "target_price": impulse_extreme,
                    "return_1h": float(row["return_1h"]),
                    "return_4h": float(row["return_4h"]),
                    "relative_volume": float(row["relative_volume"]),
                    "taker_imbalance": float(row["taker_imbalance"]),
                    "spot_return_1h": float(row["spot_return_1h"]),
                    "pullback_depth_atr": depth[position],
                    "room_bps": room[position],
                    "atr_percentile": float(row["atr_percentile"]),
                    "room_threshold_bps": threshold,
                }
            )
        stop_at = (
            max(accepted_positions)
            if len(accepted_positions) == len(thresholds)
            else len(current) - 1
        )
        stages["touch"] += int(touch[: stop_at + 1].sum())
    if audit is not None:
        audit.update(stages)
    return pd.DataFrame(events)


def label_touch_events(events: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    data = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(data["timestamp"], utc=True)
    time_values = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    open_price = data["perp_open"].to_numpy(float)
    high = data["perp_high"].to_numpy(float)
    low = data["perp_low"].to_numpy(float)
    close = data["perp_close"].to_numpy(float)
    funding = (
        data["funding_event_rate"].fillna(0).to_numpy(float)
        if "funding_event_rate" in data
        else np.zeros(len(data))
    )
    keys = ["available_at", "direction", "stop_price", "target_price"]
    geometry = events.drop_duplicates(keys).copy()
    geometry["_label_id"] = np.arange(len(geometry))
    output: list[dict[str, Any]] = []
    for raw in geometry.to_dict("records"):
        event = cast(dict[str, Any], raw)
        entry_index = int(
            np.searchsorted(
                time_values,
                pd.Timestamp(event["available_at"]).value,
                side="left",
            )
        )
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(open_price[entry_index])
        stop = float(event["stop_price"])
        target = float(event["target_price"])
        risk_bps = side * (entry - stop) / entry * 10_000
        target_bps = side * (target - entry) / entry * 10_000
        if risk_bps < RISK_BPS_RANGE[0] or risk_bps > RISK_BPS_RANGE[1]:
            continue
        last = min(entry_index + MAXIMUM_HOLDING_MINUTES, len(data) - 1)
        path = slice(entry_index, last + 1)
        stopped = low[path] <= stop if side > 0 else high[path] >= stop
        targeted = high[path] >= target if side > 0 else low[path] <= target
        stop_hits = np.flatnonzero(stopped)
        target_hits = np.flatnonzero(targeted)
        stop_at = int(stop_hits[0]) if stop_hits.size else MAXIMUM_HOLDING_MINUTES + 1
        target_at = int(target_hits[0]) if target_hits.size else MAXIMUM_HOLDING_MINUTES + 1
        if stop_at <= target_at and stop_hits.size:
            exit_index = entry_index + stop_at
            exit_price = (
                min(float(open_price[exit_index]), stop)
                if side > 0
                else max(float(open_price[exit_index]), stop)
            )
            reason = "STRUCTURAL_STOP"
        elif target_hits.size:
            exit_index = entry_index + target_at
            exit_price = target
            reason = "IMPULSE_EXTREME_TARGET"
        else:
            exit_index = last
            exit_price = float(close[last])
            reason = "TIMEOUT_60M"
        observed = slice(entry_index, exit_index + 1)
        funding_bps = -side * float(np.nansum(funding[observed])) * 10_000
        favorable = high[observed] if side > 0 else low[observed]
        adverse = low[observed] if side > 0 else high[observed]
        mfe_bps = max(0.0, float(np.nanmax(side * (favorable - entry) / entry * 10_000)))
        mae_bps = min(0.0, float(np.nanmin(side * (adverse - entry) / entry * 10_000)))
        gross_market = side * (exit_price - entry) / entry * 10_000
        gross = gross_market + funding_bps
        output.append(
            {
                "_label_id": int(event["_label_id"]),
                "entry_timestamp": times.iat[entry_index],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "exit_price": exit_price,
                "target_bps_at_entry": target_bps,
                "gross_market_return_bps": gross_market,
                "funding_return_bps": funding_bps,
                "gross_return_bps": gross,
                "net_return_bps": gross,
                "stress_return_bps": gross,
                "net_return_r": gross / risk_bps,
                "mfe_bps": mfe_bps,
                "mae_bps": mae_bps,
                "duration_minutes": exit_index - entry_index + 1,
                "exit_reason": reason,
            }
        )
    labels = pd.DataFrame(output)
    indexed = events.merge(
        geometry[[*keys, "_label_id"]], on=keys, how="left", validate="many_to_one"
    )
    labeled = indexed.merge(labels, on="_label_id", how="inner", validate="many_to_one")
    labeled = labeled.loc[labeled["target_bps_at_entry"] >= labeled["room_threshold_bps"]].drop(
        columns="_label_id"
    )
    return labeled


def build_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    if MATRIX.exists():
        cached = pd.read_parquet(MATRIX)
        if cached["touch_entry_protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached, {}
    _status("touch_entry_features", 1, "Loading frozen Binance BTC feature bars")
    features = build_features(pd.read_parquet(BARS))
    _status("touch_entry_features", 3, "Loading frozen Binance BTC one-minute path")
    minutes = pd.read_parquet(MINUTES)
    thresholds = tuple(
        sorted(
            {
                profile_cost_bps(profile) * multiplier
                for profile in PAPER_PROFILES
                for multiplier in ROOM_MULTIPLIERS
            }
        )
    )
    parts: list[pd.DataFrame] = []
    funnels: dict[str, Any] = {}
    for number, horizon in enumerate(HORIZONS, start=1):
        audit: dict[str, int] = {}
        events = build_touch_events(
            features,
            breakout_bars=horizon,
            room_thresholds_bps=thresholds,
            audit=audit,
        )
        labeled = label_touch_events(events, minutes)
        labeled = labeled.loc[
            pd.to_datetime(labeled["exit_timestamp"], utc=True).lt(HOLDOUT_START)
        ].copy()
        labeled["touch_entry_protocol_hash"] = PROTOCOL_HASH
        parts.append(labeled)
        funnels[f"H{horizon}"] = audit | {"labeled": len(labeled)}
        _status(
            "touch_entry_matrix",
            5 + 55 * number / len(HORIZONS),
            f"H{horizon} ({number}/{len(HORIZONS)})",
        )
    matrix = pd.concat(parts, ignore_index=True)
    _write_parquet(MATRIX, matrix)
    return matrix, funnels


def _select_touch_rows(rows: pd.DataFrame, eligible: list[tuple[int, float]]) -> pd.DataFrame:
    robustness = {horizon: score for horizon, score in eligible}
    selected = rows.loc[rows["expert_breakout_bars"].isin(robustness)].copy()
    selected["prior_robustness_bps"] = selected["expert_breakout_bars"].map(robustness)
    selected = selected.sort_values(
        ["signal_timestamp", "prior_robustness_bps"], ascending=[True, False]
    ).drop_duplicates(["signal_timestamp", "direction"], keep="first")
    return _non_overlapping(selected)


def _merge_base_touch(base: pd.DataFrame, touch: pd.DataFrame) -> pd.DataFrame:
    left = base.copy()
    if not left.empty and "prior_robustness_bps" not in left:
        left["prior_robustness_bps"] = np.inf
    combined = pd.concat([left, touch], ignore_index=True)
    combined = combined.sort_values(
        ["signal_timestamp", "prior_robustness_bps"], ascending=[True, False]
    ).drop_duplicates(["signal_timestamp", "direction"], keep="first")
    return _non_overlapping(combined)


def run() -> dict[str, Any]:
    matrix, funnels = build_matrix()
    base_matrix = pd.read_parquet(BASE_MATRIX)
    base_report = json.loads(BASE_FRONTIER_REPORT.read_text(encoding="utf-8"))
    periods = {
        "2024": ("2024-01-01", pd.Timestamp("2025-01-01", tz="UTC")),
        "2025": ("2025-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "prior": ("2024-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "audit": ("2026-01-01", HOLDOUT_START),
    }
    profiles: dict[str, Any] = {}
    total_prefixes = 0
    for number, profile in enumerate(PAPER_PROFILES, start=1):
        cost = profile_cost_bps(profile)
        base = _base_rows(profile, base_matrix, base_report)
        base_prior = _summary(_period(base, *periods["prior"]), cost)
        evaluations: list[dict[str, Any]] = []
        chosen: tuple[dict[str, Any], pd.DataFrame] | None = None
        for multiplier in ROOM_MULTIPLIERS:
            threshold = cost * multiplier
            available = matrix.loc[np.isclose(matrix["room_threshold_bps"], threshold)].copy()
            expert_audits: dict[str, Any] = {}
            eligible: list[tuple[int, float]] = []
            for horizon in HORIZONS:
                expert = _non_overlapping(
                    available.loc[available["expert_breakout_bars"].eq(horizon)]
                )
                years = {
                    year: _summary(_period(expert, *periods[year]), cost)
                    for year in ("2024", "2025")
                }
                accepted = all(
                    years[year]["metrics"]["expectancy_bps"] > 0
                    and years[year]["metrics"]["profit_factor"] >= 1.05
                    for year in ("2024", "2025")
                )
                robustness = min(
                    years["2024"]["metrics"]["expectancy_bps"],
                    years["2025"]["metrics"]["expectancy_bps"],
                )
                expert_audits[f"H{horizon}"] = years | {"eligible": accepted}
                if accepted:
                    eligible.append((horizon, robustness))
            ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
            prefix_evaluations: list[dict[str, Any]] = []
            for prefix_size in range(1, len(ranked) + 1):
                touch = _select_touch_rows(available, ranked[:prefix_size])
                combined = _merge_base_touch(base, touch)
                prior = {
                    name: _summary(_period(combined, *periods[name]), cost)
                    for name in ("2024", "2025", "prior")
                }
                gates = _prior_gates(prior)
                evaluation: dict[str, Any] = {
                    "room_multiplier": multiplier,
                    "minimum_room_bps": threshold,
                    "prefix_size": prefix_size,
                    "touch_experts": [f"H{item[0]}" for item in ranked[:prefix_size]],
                    "summaries": prior,
                    "prior_gates": gates,
                    "prior_pass": all(gates.values()),
                    "audit_evaluated": False,
                }
                prefix_evaluations.append(evaluation)
                total_prefixes += 1
                if (
                    evaluation["prior_pass"]
                    and prior["prior"]["metrics"]["trades"] > base_prior["metrics"]["trades"]
                    and (
                        chosen is None
                        or prior["prior"]["metrics"]["trades"]
                        > chosen[0]["summaries"]["prior"]["metrics"]["trades"]
                    )
                ):
                    chosen = (evaluation, combined)
            evaluations.append(
                {
                    "room_multiplier": multiplier,
                    "minimum_room_bps": threshold,
                    "expert_audits": expert_audits,
                    "prefixes": prefix_evaluations,
                }
            )
        if chosen is not None:
            evaluation, selected_rows = chosen
            audit_summary = _summary(_period(selected_rows, *periods["audit"]), cost)
            evaluation["audit_evaluated"] = True
            evaluation["audit_summary"] = audit_summary
            evaluation["audit_gates"] = _audit_gates(audit_summary)
            evaluation["audit_pass"] = all(evaluation["audit_gates"].values())
        profiles[profile] = {
            "round_trip_cost_bps_1x": cost,
            "base_prior": base_prior,
            "evaluations": evaluations,
            "selected_on_prior": chosen[0] if chosen is not None else None,
            "research_signal": bool(chosen is not None and chosen[0].get("audit_pass", False)),
        }
        _status("touch_entry_audit", 60 + 35 * number / len(PAPER_PROFILES), profile)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "unique_signals": int(matrix["signal_timestamp"].nunique()),
        "funnels": funnels,
        "profiles": profiles,
        "candidate_policy_prefixes_evaluated": total_prefixes,
        "verdict": (
            "TOUCH_ENTRY_RESEARCH_FRONTIER"
            if any(value["research_signal"] for value in profiles.values())
            else "NO_TOUCH_ENTRY_FREQUENCY_GAIN"
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
