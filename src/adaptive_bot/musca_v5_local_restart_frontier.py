from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import MINUTES, _non_overlapping
from adaptive_bot.musca_v5_fine_tuning import _atomic_json
from adaptive_bot.musca_v5_room_frontier import (
    _audit_gates,
    _period,
    _prior_gates,
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
MATRIX = ROOT / "local_restart_frontier_matrix.parquet"
REPORT = Path("data/reports/musca_v5_local_restart_frontier.json")
STATUS = Path("data/reports/musca_v5_local_restart_frontier.status.json")
HOLD_EXIT_MATRIX = ROOT / "local_restart_hold_exit_matrix.parquet"
HOLD_EXIT_REPORT = Path("data/reports/musca_v5_local_restart_hold_exit.json")
BREAKOUT_MINUTES = (5, 15, 30)
ANCHOR_LIFETIME_MINUTES = 60
RESTART_EXPIRY_MINUTES = 5
MAXIMUM_HOLDING_MINUTES = 60
MINIMUM_ROOM_COST_MULTIPLIER = 2.0
MINIMUM_REWARD_RISK = 1.5
RISK_BPS_RANGE = (4.0, 80.0)
FAMILY = "LOCAL_VWAP_RESTART"
PROTOCOL = {
    "name": "musca_v5_local_vwap_restart_frontier_v1",
    "base_protocol_hash": BASE_PROTOCOL_HASH,
    "market": "BINANCE_BTCUSDT_PERPETUAL_AND_SPOT",
    "state": "closed_1m_klines_available_next_minute",
    "direction": (
        "majority(return_15m,return_60m,vwap60_slope,spot_return_15m,perp_taker_5m,spot_taker_5m)"
    ),
    "breakout_minutes": list(BREAKOUT_MINUTES),
    "impulse": "breakout_with_relative_volume_and_perp_spot_taker_confirmation",
    "anchors": ["rolling_vwap_15m", "rolling_vwap_60m", "daily_vwap", "impulse_vwap"],
    "pullback": "depth_0.25_atr_quieter_than_impulse",
    "entry": "next_1m_open_after_micro_swing_break_and_perp_taker_restart",
    "stop": "beyond_pullback_and_anchor_never_widens",
    "target": "frozen_impulse_extreme",
    "invalidation": "next_open_after_close_crosses_operating_vwap_against_side",
    "maximum_holding_minutes": MAXIMUM_HOLDING_MINUTES,
    "intrabar": "stop_wins",
    "minimum_room_cost_multiplier": MINIMUM_ROOM_COST_MULTIPLIER,
    "minimum_reward_risk": MINIMUM_REWARD_RISK,
    "risk_bps": list(RISK_BPS_RANGE),
    "selection": "2024_2025_only_then_single_reused_2026_preholdout_discovery_audit",
    "economic_costs": "observed_profile_taker_costs_1x",
    "cost_stress": "2x_diagnostic_only",
    "holdout_opened": False,
    "changes_to_active_paper": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
HOLD_EXIT_PROTOCOL = PROTOCOL | {
    "name": "musca_v5_local_vwap_restart_hold_exit_v1",
    "invalidation": "none_after_restart_target_stop_or_timeout_only",
}
HOLD_EXIT_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(HOLD_EXIT_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


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


def _write_parquet(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_features(minutes: pd.DataFrame) -> pd.DataFrame:
    data = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True).copy()
    time = pd.to_datetime(data["timestamp"], utc=True)
    data["available_at"] = time + pd.Timedelta(minutes=1)
    previous = data["perp_close"].shift(1)
    true_range = pd.concat(
        [
            data["perp_high"] - data["perp_low"],
            (data["perp_high"] - previous).abs(),
            (data["perp_low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["return_15m"] = data["perp_close"].pct_change(15)
    data["return_60m"] = data["perp_close"].pct_change(60)
    data["spot_return_15m"] = data["spot_close"].pct_change(15)
    data["perp_taker_1m"] = (
        2 * data["perp_taker_buy_quote"] / data["perp_quote_volume"].replace(0, np.nan) - 1
    )
    data["spot_taker_1m"] = (
        2 * data["spot_taker_buy_quote"] / data["spot_quote_volume"].replace(0, np.nan) - 1
    )
    data["perp_taker_5m"] = (
        2
        * data["perp_taker_buy_quote"].rolling(5).sum()
        / data["perp_quote_volume"].rolling(5).sum().replace(0, np.nan)
        - 1
    )
    data["spot_taker_5m"] = (
        2
        * data["spot_taker_buy_quote"].rolling(5).sum()
        / data["spot_quote_volume"].rolling(5).sum().replace(0, np.nan)
        - 1
    )
    data["rolling_vwap_15m"] = data["perp_quote_volume"].rolling(15).sum() / data[
        "perp_volume"
    ].rolling(15).sum().replace(0, np.nan)
    data["rolling_vwap_60m"] = data["perp_quote_volume"].rolling(60).sum() / data[
        "perp_volume"
    ].rolling(60).sum().replace(0, np.nan)
    data["vwap_60m_slope"] = data["rolling_vwap_60m"].diff(5) / data["atr"]
    baseline = data["perp_quote_volume"].shift(1).rolling(1_440, min_periods=480).median()
    data["relative_volume"] = data["perp_quote_volume"] / baseline.replace(0, np.nan)
    votes = np.column_stack(
        [
            np.sign(data["return_15m"]),
            np.sign(data["return_60m"]),
            np.sign(data["vwap_60m_slope"]),
            np.sign(data["spot_return_15m"]),
            np.sign(data["perp_taker_5m"]),
            np.sign(data["spot_taker_5m"]),
        ]
    )
    data["trend_score"] = np.nansum(votes, axis=1)
    data["direction"] = np.where(
        data["trend_score"].ge(2),
        1,
        np.where(data["trend_score"].le(-2), -1, 0),
    )
    return data


def _prefixes(volume: np.ndarray, quote: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(volume) & np.isfinite(quote) & (volume > 0)
    return (
        np.r_[0.0, np.cumsum(np.where(valid, volume, 0.0))],
        np.r_[0.0, np.cumsum(np.where(valid, quote, 0.0))],
    )


def _anchored_vwap(prefixes: tuple[np.ndarray, np.ndarray], start: int, end: int) -> float:
    volume, quote = prefixes
    total = float(volume[end + 1] - volume[start])
    return float((quote[end + 1] - quote[start]) / total) if total > 0 else np.nan


def build_events(features: pd.DataFrame, breakout_minutes: int) -> pd.DataFrame:
    data = features.sort_values("timestamp").reset_index(drop=True)
    arrays = {
        column: data[column].to_numpy(float)
        for column in data.columns
        if column != "timestamp" and pd.api.types.is_numeric_dtype(data[column])
    }
    high, low, close = arrays["perp_high"], arrays["perp_low"], arrays["perp_close"]
    volume, quote = arrays["perp_volume"], arrays["perp_quote_volume"]
    direction = arrays["direction"].astype(int)
    prefixes = _prefixes(volume, quote)
    prior_high = data["perp_high"].shift(1).rolling(breakout_minutes).max().to_numpy(float)
    prior_low = data["perp_low"].shift(1).rolling(breakout_minutes).min().to_numpy(float)
    impulse = (
        (direction > 0)
        & (close > prior_high)
        & (arrays["relative_volume"] >= 1.0)
        & (arrays["perp_taker_1m"] > 0.05)
        & (arrays["spot_taker_5m"] > 0)
    ) | (
        (direction < 0)
        & (close < prior_low)
        & (arrays["relative_volume"] >= 1.0)
        & (arrays["perp_taker_1m"] < -0.05)
        & (arrays["spot_taker_5m"] < 0)
    )
    rows: list[dict[str, Any]] = []
    busy_until = -1
    for raw_index in np.flatnonzero(impulse):
        index = int(raw_index)
        if index <= busy_until:
            continue
        side = int(direction[index])
        impulse_extreme = high[index] if side > 0 else low[index]
        pullback_extreme = impulse_extreme
        armed_at: int | None = None
        operating_vwap = np.nan
        for current in range(
            index + 1,
            min(index + ANCHOR_LIFETIME_MINUTES + 1, len(data) - 1),
        ):
            if int(direction[current]) == -side:
                break
            if armed_at is None:
                impulse_extreme = (
                    max(impulse_extreme, high[current])
                    if side > 0
                    else min(impulse_extreme, low[current])
                )
            pullback_extreme = (
                min(pullback_extreme, low[current])
                if side > 0
                else max(pullback_extreme, high[current])
            )
            impulse_vwap = _anchored_vwap(prefixes, index, current)
            centers = (
                arrays["rolling_vwap_15m"][current],
                arrays["rolling_vwap_60m"][current],
                arrays["perp_daily_vwap"][current],
                impulse_vwap,
            )
            atr = arrays["atr"][current]
            valid = np.isfinite(centers) & (
                np.abs(close[current] - np.asarray(centers)) <= 0.25 * atr
            )
            depth = side * (impulse_extreme - pullback_extreme) / atr
            quieter = quote[current] < quote[index]
            if armed_at is None and valid.any() and depth >= 0.25 and quieter:
                armed_at = current
                distances = np.where(valid, np.abs(close[current] - np.asarray(centers)), np.inf)
                operating_vwap = float(centers[int(np.argmin(distances))])
            if armed_at is None:
                continue
            if current - armed_at > RESTART_EXPIRY_MINUTES:
                break
            restarted = (
                close[current] > high[current - 1]
                if side > 0
                else close[current] < low[current - 1]
            )
            flow = arrays["perp_taker_1m"][current] * side > 0
            accepted = (close[current] - operating_vwap) * side > 0
            if not (restarted and flow and accepted):
                continue
            zone_edge = operating_vwap - side * 0.25 * atr
            stop = (
                min(pullback_extreme, zone_edge) - 0.1 * atr
                if side > 0
                else max(pullback_extreme, zone_edge) + 0.1 * atr
            )
            risk_bps = side * (close[current] - stop) / close[current] * 10_000
            room_bps = side * (impulse_extreme - close[current]) / close[current] * 10_000
            if not (RISK_BPS_RANGE[0] <= risk_bps <= RISK_BPS_RANGE[1]) or room_bps <= 0:
                continue
            row = data.iloc[current]
            rows.append(
                {
                    "protocol_hash": PROTOCOL_HASH,
                    "signal_timestamp": pd.Timestamp(row["timestamp"]),
                    "available_at": pd.Timestamp(row["available_at"]),
                    "direction": side,
                    "event_family": FAMILY,
                    "expert_breakout_bars": breakout_minutes,
                    "impulse_anchor_at": pd.Timestamp(data.iloc[index]["timestamp"]),
                    "operating_vwap": operating_vwap,
                    "stop_price": stop,
                    "target_price": impulse_extreme,
                    "risk_bps_at_signal": risk_bps,
                    "room_bps": room_bps,
                    "trend_score": float(row["trend_score"]),
                    "return_15m": float(row["return_15m"]),
                    "return_60m": float(row["return_60m"]),
                    "spot_return_15m": float(row["spot_return_15m"]),
                    "vwap_60m_slope": float(row["vwap_60m_slope"]),
                    "relative_volume": float(row["relative_volume"]),
                    "perp_taker_1m": float(row["perp_taker_1m"]),
                    "perp_taker_5m": float(row["perp_taker_5m"]),
                    "spot_taker_5m": float(row["spot_taker_5m"]),
                    "pullback_depth_atr": depth,
                }
            )
            busy_until = current
            break
    return pd.DataFrame(rows)


def label_events(
    events: pd.DataFrame,
    minutes: pd.DataFrame,
    *,
    invalidate_on_vwap: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    data = minutes.loc[minutes["data_valid"]].sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(data["timestamp"], utc=True)
    time_values = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    open_price = data["perp_open"].to_numpy(float)
    high = data["perp_high"].to_numpy(float)
    low = data["perp_low"].to_numpy(float)
    close = data["perp_close"].to_numpy(float)
    funding = data["funding_event_rate"].fillna(0).to_numpy(float)
    rows: list[dict[str, Any]] = []
    for raw_event in events.to_dict("records"):
        event = cast(dict[str, Any], raw_event)
        entry_index = int(
            np.searchsorted(time_values, pd.Timestamp(event["available_at"]).value, side="left")
        )
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(open_price[entry_index])
        stop = float(event["stop_price"])
        target = float(event["target_price"])
        risk_bps = side * (entry - stop) / entry * 10_000
        target_bps = side * (target - entry) / entry * 10_000
        if not (RISK_BPS_RANGE[0] <= risk_bps <= RISK_BPS_RANGE[1]) or target_bps <= 0:
            continue
        last = min(entry_index + MAXIMUM_HOLDING_MINUTES, len(data) - 1)
        exit_index = last
        exit_price = float(close[last])
        exit_reason = "TIMEOUT_60M"
        mfe_bps = 0.0
        mae_bps = 0.0
        funding_bps = 0.0
        for current in range(entry_index, last + 1):
            funding_bps -= side * funding[current] * 10_000
            favorable = high[current] if side > 0 else low[current]
            adverse = low[current] if side > 0 else high[current]
            mfe_bps = max(mfe_bps, side * (favorable - entry) / entry * 10_000)
            mae_bps = min(mae_bps, side * (adverse - entry) / entry * 10_000)
            stopped = low[current] <= stop if side > 0 else high[current] >= stop
            targeted = high[current] >= target if side > 0 else low[current] <= target
            if stopped:
                exit_index = current
                exit_price = (
                    min(float(open_price[current]), stop)
                    if side > 0
                    else max(float(open_price[current]), stop)
                )
                exit_reason = "STRUCTURAL_STOP"
                break
            if targeted:
                exit_index = current
                exit_price = target
                exit_reason = "IMPULSE_EXTREME_TARGET"
                break
            invalidated = (
                invalidate_on_vwap and (close[current] - float(event["operating_vwap"])) * side < 0
            )
            if invalidated and current < last:
                exit_index = current + 1
                exit_price = float(open_price[exit_index])
                if exit_price <= stop if side > 0 else exit_price >= stop:
                    exit_price = min(exit_price, stop) if side > 0 else max(exit_price, stop)
                    exit_reason = "STRUCTURAL_STOP_GAP"
                else:
                    exit_reason = "VWAP_INVALIDATION"
                break
        gross_market = side * (exit_price - entry) / entry * 10_000
        gross = gross_market + funding_bps
        rows.append(
            event
            | {
                "entry_timestamp": times.iat[entry_index],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "exit_price": exit_price,
                "risk_bps_at_entry": risk_bps,
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
                "exit_reason": exit_reason,
            }
        )
    return pd.DataFrame(rows)


def build_matrix() -> tuple[pd.DataFrame, dict[str, int]]:
    if MATRIX.exists():
        cached = pd.read_parquet(MATRIX)
        if cached["local_restart_protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached, {}
    _status("local_features", 2, "Loading Binance BTC one-minute perpetual and spot path")
    minutes = pd.read_parquet(MINUTES)
    features = build_features(minutes)
    parts: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for number, horizon in enumerate(BREAKOUT_MINUTES, start=1):
        events = build_events(features, horizon)
        labeled = label_events(events, minutes)
        labeled = labeled.loc[
            pd.to_datetime(labeled["exit_timestamp"], utc=True).lt(HOLDOUT_START)
        ].copy()
        labeled["local_restart_protocol_hash"] = PROTOCOL_HASH
        parts.append(labeled)
        counts[f"M{horizon}"] = len(labeled)
        _status(
            "local_matrix",
            5 + 50 * number / len(BREAKOUT_MINUTES),
            f"M{horizon} ({number}/{len(BREAKOUT_MINUTES)})",
        )
    matrix = pd.concat(parts, ignore_index=True)
    _write_parquet(MATRIX, matrix)
    return matrix, counts


def _eligible_rows(matrix: pd.DataFrame, cost: float, horizon: int) -> pd.DataFrame:
    rows = matrix.loc[
        matrix["expert_breakout_bars"].eq(horizon)
        & (matrix["target_bps_at_entry"] >= MINIMUM_ROOM_COST_MULTIPLIER * cost)
        & (matrix["target_bps_at_entry"] >= MINIMUM_REWARD_RISK * matrix["risk_bps_at_entry"])
    ].copy()
    return _non_overlapping(rows)


def _combine(matrix: pd.DataFrame, cost: float, ranked: list[tuple[int, float]]) -> pd.DataFrame:
    robustness = dict(ranked)
    rows = matrix.loc[matrix["expert_breakout_bars"].isin(robustness)].copy()
    rows = rows.loc[
        (rows["target_bps_at_entry"] >= MINIMUM_ROOM_COST_MULTIPLIER * cost)
        & (rows["target_bps_at_entry"] >= MINIMUM_REWARD_RISK * rows["risk_bps_at_entry"])
    ]
    rows["prior_robustness_bps"] = rows["expert_breakout_bars"].map(robustness)
    rows = rows.sort_values(
        ["signal_timestamp", "prior_robustness_bps"], ascending=[True, False]
    ).drop_duplicates(["signal_timestamp", "direction"], keep="first")
    return _non_overlapping(rows)


def _evaluate_profile(matrix: pd.DataFrame, cost: float) -> dict[str, Any]:
    periods = {
        "2024": ("2024-01-01", pd.Timestamp("2025-01-01", tz="UTC")),
        "2025": ("2025-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "prior": ("2024-01-01", pd.Timestamp("2026-01-01", tz="UTC")),
        "audit": ("2026-01-01", HOLDOUT_START),
    }
    experts: dict[str, Any] = {}
    eligible: list[tuple[int, float]] = []
    for horizon in BREAKOUT_MINUTES:
        rows = _eligible_rows(matrix, cost, horizon)
        years = {
            year: _summary(_period(rows, *periods[year]), cost, bootstrap=False)
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
        experts[f"M{horizon}"] = years | {"eligible": accepted}
        if accepted:
            eligible.append((horizon, robustness))
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
    prefixes: list[dict[str, Any]] = []
    chosen: tuple[dict[str, Any], pd.DataFrame] | None = None
    for size in range(1, len(ranked) + 1):
        rows = _combine(matrix, cost, ranked[:size])
        summaries = {
            name: _summary(_period(rows, *periods[name]), cost)
            for name in ("2024", "2025", "prior")
        }
        gates = _prior_gates(summaries)
        evaluation = {
            "experts": [f"M{item[0]}" for item in ranked[:size]],
            "summaries": summaries,
            "gates": gates,
            "pass": all(gates.values()),
        }
        prefixes.append(evaluation)
        if evaluation["pass"] and (
            chosen is None
            or summaries["prior"]["metrics"]["trades"]
            > chosen[0]["summaries"]["prior"]["metrics"]["trades"]
        ):
            chosen = (evaluation, rows)
    audit_summary = audit_gates = None
    audit_pass = False
    if chosen is not None:
        audit_summary = _summary(_period(chosen[1], *periods["audit"]), cost)
        audit_gates = _audit_gates(audit_summary)
        audit_pass = all(audit_gates.values())
    return {
        "round_trip_cost_bps_1x": cost,
        "minimum_room_bps": MINIMUM_ROOM_COST_MULTIPLIER * cost,
        "minimum_reward_risk": MINIMUM_REWARD_RISK,
        "expert_audits": experts,
        "prefixes": prefixes,
        "selected_on_prior": chosen[0] if chosen is not None else None,
        "audit_summary": audit_summary,
        "audit_gates": audit_gates,
        "audit_pass": audit_pass,
        "research_signal": bool(chosen is not None and audit_pass),
    }


def run() -> dict[str, Any]:
    matrix, counts = build_matrix()
    profiles: dict[str, Any] = {}
    for number, profile in enumerate(PAPER_PROFILES, start=1):
        profiles[profile] = _evaluate_profile(matrix, profile_cost_bps(profile))
        _status("local_audit", 55 + 40 * number / len(PAPER_PROFILES), profile)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_rows": len(matrix),
        "unique_signals": int(matrix["signal_timestamp"].nunique()),
        "horizon_rows": counts,
        "profiles": profiles,
        "candidate_policy_prefixes_evaluated": sum(
            len(value["prefixes"]) for value in profiles.values()
        ),
        "verdict": (
            "LOCAL_RESTART_RESEARCH_FRONTIER"
            if any(value["research_signal"] for value in profiles.values())
            else "NO_LOCAL_RESTART_FREQUENCY_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, result)
    _status("complete", 100, result["verdict"])
    return result


def run_hold_exit() -> dict[str, Any]:
    if HOLD_EXIT_MATRIX.exists():
        matrix = pd.read_parquet(HOLD_EXIT_MATRIX)
        if not matrix["local_restart_protocol_hash"].eq(HOLD_EXIT_PROTOCOL_HASH).all():
            matrix = pd.DataFrame()
    else:
        matrix = pd.DataFrame()
    if matrix.empty:
        _status("hold_exit_matrix", 5, "Relabeling frozen FT-016 candidates")
        events = pd.read_parquet(MATRIX).copy()
        events["protocol_hash"] = HOLD_EXIT_PROTOCOL_HASH
        minutes = pd.read_parquet(MINUTES)
        matrix = label_events(events, minutes, invalidate_on_vwap=False)
        matrix = matrix.loc[
            pd.to_datetime(matrix["exit_timestamp"], utc=True).lt(HOLDOUT_START)
        ].copy()
        matrix["local_restart_protocol_hash"] = HOLD_EXIT_PROTOCOL_HASH
        _write_parquet(HOLD_EXIT_MATRIX, matrix)
    profiles: dict[str, Any] = {}
    for number, profile in enumerate(PAPER_PROFILES, start=1):
        profiles[profile] = _evaluate_profile(matrix, profile_cost_bps(profile))
        _status("hold_exit_audit", 50 + 45 * number / len(PAPER_PROFILES), profile)
    result: dict[str, Any] = {
        "protocol": HOLD_EXIT_PROTOCOL,
        "protocol_hash": HOLD_EXIT_PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "source_protocol_hash": PROTOCOL_HASH,
        "matrix_rows": len(matrix),
        "unique_signals": int(matrix["signal_timestamp"].nunique()),
        "profiles": profiles,
        "candidate_policy_prefixes_evaluated": sum(
            len(value["prefixes"]) for value in profiles.values()
        ),
        "verdict": (
            "LOCAL_HOLD_EXIT_RESEARCH_FRONTIER"
            if any(value["research_signal"] for value in profiles.values())
            else "NO_LOCAL_HOLD_EXIT_FREQUENCY_GAIN"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(HOLD_EXIT_REPORT, result)
    _status("complete", 100, result["verdict"])
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
