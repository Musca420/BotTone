from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot import (
    binance_l2_dataset,
    bitunix_l2_dataset,
    btc_vwap_alpha,
    btc_vwap_forward_selector,
    musca_v5_paper,
    musca_v8_multi_horizon,
)
from adaptive_bot.adapters.bitunix.market_data import _get_json
from adaptive_bot.bitunix_fees import FUTURES_VIP_FEE_BPS, futures_fee_bps
from adaptive_bot.btc_cross_exchange_dataset import (
    HORIZONS,
    L2_JOIN_FEATURES,
    OUTPUT,
    build_features,
    load_snapshots,
)
from adaptive_bot.musca_v5_execution import (
    MarketIntegrity,
    quote_taker_round_trip,
    size_for_technical_stop,
)

REPORT = Path("data/reports/btc_cross_exchange_forward_audit.json")
COUNTERFACTUAL = Path("data/research/btc_vwap_forward_counterfactual_v4.parquet")
FEE_PROFILE_COUNTERFACTUAL = Path("data/research/btc_vwap_forward_fee_profiles_v5.parquet")
PROTOCOL_START = pd.Timestamp("2026-08-06T07:45:00Z")
ANCHOR_PROTOCOL_START = pd.Timestamp("2026-08-06T08:35:00Z")
EXECUTION_PROTOCOL_START = pd.Timestamp("2026-08-06T08:50:00Z")
DYNAMIC_PROTOCOL_START = pd.Timestamp("2026-08-08T21:57:00Z")
MINIMUM_EVENTS = 100
FAMILIES = btc_vwap_alpha.FAMILIES
BITUNIX_VIP_LEVEL = int(os.getenv("BITUNIX_VIP_LEVEL", "0"))
MAKER_FEE_BPS_PER_SIDE, TAKER_FEE_BPS_PER_SIDE = futures_fee_bps(BITUNIX_VIP_LEVEL)
SLIPPAGE_RESERVE_BPS_PER_SIDE = float(
    os.getenv(
        "BITUNIX_SLIPPAGE_RESERVE_BPS",
        str(btc_vwap_alpha.SLIPPAGE_RESERVE_BPS_PER_SIDE),
    )
)
ROUND_TRIP_COST_BPS = 2 * MAKER_FEE_BPS_PER_SIDE
STRESS_COST_BPS = 2 * (TAKER_FEE_BPS_PER_SIDE + SLIPPAGE_RESERVE_BPS_PER_SIDE)
MINIMUM_GROSS_MOVEMENT_TO_COST = 3.0
MINIMUM_PROFIT_FACTOR = 1.15
MINIMUM_POSITIVE_DAY_FRACTION = 0.5
MAX_L2_GAP_SECONDS = 5.0
MAX_ALPHA_AGE_SECONDS = 120.0
MAX_EXECUTION_AGE_SECONDS = 10.0
HISTORICAL_CACHE_REFRESH_SECONDS = 3_600.0
REPORT_LOCK = threading.RLock()
_LIVE_ALPHA_CACHE: tuple[float, pd.DataFrame, pd.DataFrame] | None = None
LIVE_ALPHA_MINUTES = 480
LEGACY_EMBEDDED_REPORT_KEYS = (
    "anchor_protocol",
    "anchor_protocol_hash",
    "dynamic_protocol",
    "dynamic_protocol_hash",
    "feature_rows_after_protocol_start",
    "anchored_vwap_rows_after_protocol_start",
    "dynamic_feature_rows_after_execution_start",
    "rejection_funnel",
    "anchor_rejection_funnel",
    "dynamic_rejection_funnel",
    "dynamic_anchor_rejection_funnel",
    "oracle",
    "results",
    "anchored_results",
    "dynamic_results",
    "eligible",
    "anchored_eligible",
    "dynamic_eligible",
    "counterfactual_rows",
    "counterfactual_path",
    "fee_profile_counterfactual_rows",
    "fee_profile_counterfactual_path",
)
DYNAMIC_MIN_HOLD_SECONDS = 300
DYNAMIC_MAX_HOLD_MINUTES = max(btc_vwap_alpha.HORIZONS)
DYNAMIC_MIN_STOP_BPS = btc_vwap_alpha.MIN_STOP_BPS
DYNAMIC_RANGE_MULTIPLIER = btc_vwap_alpha.STOP_RANGE_MULTIPLIER
DYNAMIC_TRAIL_BPS = 8.0
DYNAMIC_PROFIT_ACTIVATION_BPS = STRESS_COST_BPS + DYNAMIC_TRAIL_BPS + 2.0
DYNAMIC_FLOW_INVALIDATION_SECONDS = 30
LABEL_HORIZONS_MINUTES = btc_vwap_alpha.HORIZONS
BARRIER_BPS = tuple(float(value) for value in btc_vwap_alpha.BARRIERS)
TIME_TARGET_BPS = (10.0, 20.0, 30.0, 50.0)
PAPER_INITIAL_EQUITY = 10_000.0
PAPER_RISK_PER_TRADE = 0.01
PAPER_MAX_LEVERAGE = 10.0
PAPER_MARGIN_FRACTION = 0.10
PAPER_ACCOUNT_START = pd.Timestamp(os.getenv("MUSCA_V5_PAPER_START", "2026-08-08T21:57:00Z"))
LABEL_NOTIONAL_USDT = 20_000.0
HARD_CATASTROPHIC_STOP_BPS = btc_vwap_alpha.MAX_TECHNICAL_STOP_BPS
DECISION_SPEC = {
    "vwap_pullback_continuation": {
        "direction": "sign(sign(return_15m)+sign(return_30m)+sign(vwap_slope_1m))",
        "directional_vwap_distance_bps": [-3.0, 15.0],
        "directional_taker_flow_positive": True,
        "directional_1m_restart": True,
    },
    "vwap_reversion": {
        "vwap_extension_abs_min_bps": 5.0,
        "direction": "toward rolling VWAP",
        "opposing_taker_flow_positive": True,
        "opposing_1m_reversal": True,
    },
    "rolling_vwap_reentry": {
        "event": "completed-minute VWAP side cross",
        "directional_taker_flow_positive": True,
        "directional_1m_return_positive": True,
    },
    "flat_value_bps": 0.0,
    "events_overlap": False,
}
PROTOCOL = {
    "name": "btc_binance_vwap_timing_bitunix_shadow_v1",
    "protocol_start": PROTOCOL_START.isoformat(),
    "direction": "completed Binance BTCUSDT 1m trade bars; same rules as fitted Alpha",
    "timing": "completed-minute VWAP events; partial candles excluded",
    "execution": "Bitunix observed book, funding and VIP fees",
    "other_venues": "excluded from entry decisions and Alpha",
    "families": list(FAMILIES),
    "horizons_minutes": list(HORIZONS),
    "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    "stress_cost_bps": STRESS_COST_BPS,
    "minimum_gross_movement_to_cost": MINIMUM_GROSS_MOVEMENT_TO_COST,
    "minimum_events": MINIMUM_EVENTS,
    "minimum_profit_factor": MINIMUM_PROFIT_FACTOR,
    "minimum_positive_day_fraction": MINIMUM_POSITIVE_DAY_FRACTION,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
POLICY_HASH = hashlib.sha256(
    json.dumps({"protocol": PROTOCOL, "decision_spec": DECISION_SPEC}, sort_keys=True).encode()
).hexdigest()
ANCHOR_PROTOCOL = {
    "name": "btc_causal_avwap_acceptance_failure_forward_v1",
    "protocol_start": ANCHOR_PROTOCOL_START.isoformat(),
    "horizons_minutes": [15, 30],
    "anchor_age_seconds": [60, 1800],
    "continuation_distance_bps": [0.5, 10.0],
    "failure_cross_bps": -1.0,
    "minimum_directional_flow_vote": 1,
    "events_overlap": False,
    "costs_bps": [ROUND_TRIP_COST_BPS, STRESS_COST_BPS],
}
ANCHOR_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(ANCHOR_PROTOCOL, sort_keys=True).encode()
).hexdigest()
EXECUTION_PROTOCOL = {
    "name": "bitunix_l2_next_event_execution_v1",
    "protocol_start": EXECUTION_PROTOCOL_START.isoformat(),
    "entry": "first observed Bitunix 15-level book strictly after signal availability",
    "exit": "Bitunix 15-level execution VWAP for a preregistered 20,000 USDT notional",
    "normal_return": "Bitunix taker execution VWAP less VIP fees, reserve and funding",
    "stress_return": "same observed execution with doubled fees and reserve",
    "maximum_path_gap_seconds": MAX_L2_GAP_SECONDS,
    "incomplete_path": "excluded fail-closed",
}
EXECUTION_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(EXECUTION_PROTOCOL, sort_keys=True).encode()
).hexdigest()
DYNAMIC_PROTOCOL = {
    "name": "btc_vwap_net_ev_forward_v4_bitunix",
    "protocol_start": DYNAMIC_PROTOCOL_START.isoformat(),
    "discovery_replay_start": EXECUTION_PROTOCOL_START.isoformat(),
    "discovery_rows_before_protocol_start": "paper only; excluded from model fitting",
    "alpha_source": "Binance BTCUSDT perpetual and spot only",
    "execution_venue": "Bitunix shadow only",
    "other_venues": "diagnostic archives only; excluded from every entry decision",
    "minimum_hold_seconds": DYNAMIC_MIN_HOLD_SECONDS,
    "maximum_hold_minutes": DYNAMIC_MAX_HOLD_MINUTES,
    "label_horizons_minutes": list(LABEL_HORIZONS_MINUTES),
    "barriers_bps": list(BARRIER_BPS),
    "time_to_target_bps": list(TIME_TARGET_BPS),
    "initial_stop_bps": (
        f"max({DYNAMIC_MIN_STOP_BPS}, {DYNAMIC_RANGE_MULTIPLIER} x observed 60s range)"
    ),
    "target_bps": "volatility target 30/50, raised to at least 3x configured round-trip cost",
    "profit_activation_bps": DYNAMIC_PROFIT_ACTIVATION_BPS,
    "trailing_giveback_bps": DYNAMIC_TRAIL_BPS,
    "invalidation": "30 seconds persistent adverse 5s/30s/60s flow plus depth/microprice",
    "fees_bps_per_side": {
        "venue": "Bitunix futures",
        "vip_level": BITUNIX_VIP_LEVEL,
        "maker": MAKER_FEE_BPS_PER_SIDE,
        "taker": TAKER_FEE_BPS_PER_SIDE,
        "slippage_reserve": SLIPPAGE_RESERVE_BPS_PER_SIDE,
    },
    "funding": "official settled Bitunix history when holding crosses funding; otherwise 0",
    "stop_never_widens": True,
    "hard_catastrophic_stop_bps": HARD_CATASTROPHIC_STOP_BPS,
    "execution": EXECUTION_PROTOCOL["name"],
    "reentry": "after exit only on changed side/flow/anchor state or setup score +2",
}
DYNAMIC_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(DYNAMIC_PROTOCOL, sort_keys=True).encode()
).hexdigest()
COUNTERFACTUAL_FEATURES = (
    "price_binance",
    "median_return_1m_bps",
    "median_return_5m_bps",
    "dispersion_return_1m_bps",
    "dispersion_return_5m_bps",
    "rolling_vwap_5m_distance_bps",
    "spread_bps",
    "range_60s_bps",
    "anchored_vwap_distance_bps",
    "anchor_direction",
    "anchor_age_seconds",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_15s",
    "trade_arrival_rate_30s",
    "depth_imbalance_5bps",
    "depth_imbalance_10bps",
    "depth_imbalance_20bps",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
    "spread_change_bps",
    "funding_median",
    "funding_dispersion",
    "basis_bps_binance",
    "basis_bps_bybit",
    "basis_bps_okx",
    "oi_change_1h_binance",
    "oi_change_1h_bybit",
    "oi_change_1h_okx",
    *sorted(btc_vwap_alpha.LIVE_REQUIRED - {"signal_at", "side", "expert", "stop_bps"}),
    *L2_JOIN_FEATURES,
)


def _flow_vote(frame: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        np.sign(frame["aggressive_imbalance_60s"])
        + np.sign(frame["depth_imbalance_5"])
        + np.sign(frame["microprice_distance_bps"]),
    )


def _setup(frame: pd.DataFrame, family: str) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    return btc_vwap_alpha.setup_conditions(frame, family, prefix="alpha_")


def rejection_funnel(frame: pd.DataFrame, family: str) -> dict[str, int]:
    eligible = frame["minute"].ge(PROTOCOL_START)
    _, stages = _setup(frame, family)
    counts = {"feature_rows": int(eligible.sum())}
    for name, condition in stages:
        eligible &= condition
        counts[name] = int(eligible.sum())
    return counts


def select_events(frame: pd.DataFrame, horizon: int, family: str) -> pd.DataFrame:
    side, stages = _setup(frame, family)
    setup = pd.Series(True, index=frame.index)
    for _, condition in stages:
        setup &= condition
    start = max(PROTOCOL_START, EXECUTION_PROTOCOL_START)
    minute = pd.to_datetime(frame["minute"], utc=True)
    decision_positions = np.flatnonzero(
        minute.ge(start).to_numpy() & minute.dt.minute.mod(5).eq(0).to_numpy()
    )
    active = setup.fillna(False).to_numpy(bool)[decision_positions]
    decision_side = side.fillna(0).to_numpy(float)[decision_positions]
    rising = active & (
        ~np.r_[False, active[:-1]]
        | (decision_side != np.r_[0.0, decision_side[:-1]])
    )
    candidate_positions = decision_positions[np.flatnonzero(rising)]
    candidates = frame.iloc[candidate_positions].copy()
    candidates["side"] = side.iloc[candidate_positions].to_numpy(float)
    chosen: list[Any] = []
    next_minute = start
    for index, row in candidates.sort_values("minute").iterrows():
        candidate_minute = pd.Timestamp(row["minute"])
        if candidate_minute >= next_minute:
            chosen.append(index)
            next_minute = candidate_minute + pd.Timedelta(minutes=horizon)
    return candidates.loc[chosen].copy()


def _anchor_setup(
    frame: pd.DataFrame, family: str
) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    anchor_side = np.sign(frame["anchor_direction"])
    distance = frame["anchored_vwap_distance_bps"] * anchor_side
    vote = _flow_vote(frame) * anchor_side
    age = frame["anchor_age_seconds"]
    common = [
        ("anchor_available", anchor_side.ne(0) & age.between(60, 1800)),
    ]
    if family == "ANCHOR_CONTINUATION":
        return anchor_side, [
            *common,
            ("accepted_side", distance.between(0.5, 10.0)),
            ("binance_direction", frame["return_5m_binance_bps"] * anchor_side > 0),
            ("order_flow_direction", vote.ge(1)),
            ("price_restart", frame["return_1m_binance_bps"] * anchor_side > 0),
        ]
    if family == "ANCHOR_FAILURE":
        return -anchor_side, [
            *common,
            ("failed_anchor", distance.le(-1.0)),
            ("binance_reversal", frame["return_5m_binance_bps"] * anchor_side < 0),
            ("order_flow_reversal", vote.le(-1)),
            ("price_reversal", frame["return_1m_binance_bps"] * anchor_side < 0),
        ]
    raise ValueError(f"Unknown anchored family: {family}")


def select_anchor_events(frame: pd.DataFrame, horizon: int, family: str) -> pd.DataFrame:
    side, stages = _anchor_setup(frame, family)
    setup = pd.Series(True, index=frame.index)
    for _, condition in stages:
        setup &= condition
    start = max(ANCHOR_PROTOCOL_START, EXECUTION_PROTOCOL_START)
    candidates = frame.loc[frame["minute"].ge(start) & setup].copy()
    candidates["side"] = side.loc[candidates.index]
    chosen: list[Any] = []
    next_minute = start
    for index, row in candidates.sort_values("minute").iterrows():
        minute = pd.Timestamp(row["minute"])
        if minute >= next_minute:
            chosen.append(index)
            next_minute = minute + pd.Timedelta(minutes=horizon)
    return candidates.loc[chosen].copy()


def anchor_rejection_funnel(frame: pd.DataFrame, family: str) -> dict[str, int]:
    eligible = frame["minute"].ge(ANCHOR_PROTOCOL_START)
    _, stages = _anchor_setup(frame, family)
    counts = {"feature_rows": int(eligible.sum())}
    for name, condition in stages:
        eligible &= condition
        counts[name] = int(eligible.sum())
    return counts


def _bootstrap_lcb(values: pd.Series, seed: int = 24) -> float | None:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 20:
        return None
    block = min(8, max(2, int(np.sqrt(len(clean)))))
    starts = np.arange(len(clean) - block + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(2_000)
    for sample in range(len(means)):
        pieces = [
            clean[start : start + block]
            for start in rng.choice(starts, size=int(np.ceil(len(clean) / block)))
        ]
        means[sample] = np.concatenate(pieces)[: len(clean)].mean()
    return float(np.quantile(means, 0.05))


def _settled_bitunix_funding(get_json: Any = _get_json) -> pd.DataFrame:
    url = (
        "https://fapi.bitunix.com/api/v1/futures/market/"
        "get_funding_rate_history?symbol=BTCUSDT&limit=200"
    )
    payload = get_json(url)
    data = payload.get("data")
    if payload.get("code") not in (0, "0") or not isinstance(data, list):
        return pd.DataFrame(columns=["funding_settlement_at", "settled_funding_rate"])
    rows = pd.DataFrame(data)
    if rows.empty or not {"fundingTime", "fundingRate"}.issubset(rows):
        return pd.DataFrame(columns=["funding_settlement_at", "settled_funding_rate"])
    rows["funding_settlement_at"] = pd.to_datetime(
        pd.to_numeric(rows["fundingTime"], errors="coerce"), unit="ms", utc=True
    )
    rows["settled_funding_rate"] = pd.to_numeric(rows["fundingRate"], errors="coerce")
    rows = rows.dropna(subset=["funding_settlement_at", "settled_funding_rate"])
    return rows.loc[
        rows["settled_funding_rate"].abs().le(0.01),
        ["funding_settlement_at", "settled_funding_rate"],
    ].sort_values("funding_settlement_at")


def path_labels(
    events: pd.DataFrame,
    l2: pd.DataFrame,
    horizon: int,
    *,
    dynamic_early: bool = False,
) -> pd.DataFrame:
    labels: list[dict[str, Any]] = []
    available = pd.to_datetime(l2["available_at"], utc=True)
    available_ns = available.to_numpy(dtype="datetime64[ns]").astype("int64")
    for _, event in events.iterrows():
        signal_at = pd.Timestamp(event["available_at"])
        entry_index = int(np.searchsorted(available_ns, signal_at.value, side="right"))
        if entry_index >= len(l2):
            continue
        entry_row = l2.iloc[entry_index]
        entry_at = pd.Timestamp(entry_row["available_at"])
        if (entry_at - signal_at).total_seconds() > MAX_L2_GAP_SECONDS:
            continue
        end_at = entry_at + pd.Timedelta(minutes=horizon)
        end_index = int(np.searchsorted(available_ns, end_at.value, side="right"))
        path = l2.iloc[entry_index:end_index]
        complete_through = end_at - pd.Timedelta(seconds=2)
        path_times = pd.to_datetime(path["available_at"], utc=True)
        complete = (
            not path.empty and pd.Timestamp(path.iloc[-1]["available_at"]) >= complete_through
        )
        if (
            path.empty
            or path_times.diff().dt.total_seconds().dropna().gt(MAX_L2_GAP_SECONDS).any()
            or not np.isfinite(path[["mid", "best_bid", "best_ask"]]).all().all()
            or path[["mid", "best_bid", "best_ask"]].le(0).any().any()
            or not np.isfinite(float(entry_row["range_60s_bps"]))
        ):
            continue
        side = float(event["side"])
        quantity = LABEL_NOTIONAL_USDT / float(entry_row["mid"])
        entry_execution = _book_vwap(entry_row["asks" if side > 0 else "bids"], quantity)
        if entry_execution is None:
            continue
        exit_execution = path["bids" if side > 0 else "asks"].map(
            lambda levels, size=quantity: _book_vwap(levels, size)
        )
        if exit_execution.isna().any():
            continue
        mid_return = side * (path["mid"] / float(entry_row["mid"]) - 1) * 10_000
        executable_return = (
            (exit_execution / entry_execution - 1) * 10_000
            if side > 0
            else (entry_execution / exit_execution - 1) * 10_000
        )
        dynamic_stop = max(
            DYNAMIC_MIN_STOP_BPS,
            DYNAMIC_RANGE_MULTIPLIER * float(entry_row["range_60s_bps"]),
        )
        volatility_target = 30.0 if float(entry_row["range_60s_bps"]) < 40 / 3 else 50.0
        economic_target = next(
            barrier
            for barrier in BARRIER_BPS
            if barrier >= MINIMUM_GROSS_MOVEMENT_TO_COST * STRESS_COST_BPS
        )
        target_bps = max(volatility_target, economic_target)
        dynamic_position = len(path) - 1
        dynamic_reason = "TIME"
        peak = float("-inf")
        adverse_since: pd.Timestamp | None = None
        for path_position in range(len(path)):
            current_at = pd.Timestamp(path.iloc[path_position]["available_at"])
            elapsed = (current_at - entry_at).total_seconds()
            value = float(executable_return.iloc[path_position])
            peak = max(peak, value)
            row = path.iloc[path_position]
            flow_values = (
                row.get("aggressive_imbalance_5s"),
                row.get("aggressive_imbalance_30s"),
                row.get("aggressive_imbalance_60s"),
                row.get("depth_imbalance_5"),
                row.get("microprice_distance_bps"),
            )
            adverse_votes = sum(
                side * np.sign(float(item)) < 0 for item in flow_values if pd.notna(item)
            )
            anchor_failed = (
                pd.notna(row["anchored_vwap_distance_bps"])
                and side * float(row["anchored_vwap_distance_bps"]) <= -2
            )
            if value <= -min(dynamic_stop, HARD_CATASTROPHIC_STOP_BPS):
                dynamic_position, dynamic_reason = (
                    path_position,
                    (
                        "CATASTROPHIC_STOP"
                        if dynamic_stop > HARD_CATASTROPHIC_STOP_BPS
                        else "DYNAMIC_STOP"
                    ),
                )
                break
            if value >= target_bps:
                dynamic_position, dynamic_reason = path_position, "TARGET"
                break
            if peak >= DYNAMIC_PROFIT_ACTIVATION_BPS and peak - value >= DYNAMIC_TRAIL_BPS:
                dynamic_position, dynamic_reason = path_position, "TRAIL"
                break
            if elapsed < DYNAMIC_MIN_HOLD_SECONDS:
                continue
            adverse = adverse_votes >= 3 and (anchor_failed or adverse_votes >= 4)
            adverse_since = current_at if adverse and adverse_since is None else adverse_since
            if not adverse:
                adverse_since = None
            if (
                adverse_since is not None
                and (current_at - adverse_since).total_seconds()
                >= DYNAMIC_FLOW_INVALIDATION_SECONDS
            ):
                dynamic_position = path_position
                dynamic_reason = "AVWAP_FAILURE" if anchor_failed else "FLOW_INVALIDATION"
                break
        if not complete and not (dynamic_early and dynamic_reason != "TIME"):
            continue
        dynamic_path = executable_return.iloc[: dynamic_position + 1]
        extra_labels: dict[str, Any] = {}
        for label_horizon in LABEL_HORIZONS_MINUTES:
            horizon_path = executable_return.loc[
                path_times.le(entry_at + pd.Timedelta(minutes=label_horizon))
            ]
            extra_labels[f"mfe_{label_horizon}m_bps"] = (
                float(horizon_path.max()) if len(horizon_path) else np.nan
            )
            extra_labels[f"mae_{label_horizon}m_bps"] = (
                float(horizon_path.min()) if len(horizon_path) else np.nan
            )
        for target in TIME_TARGET_BPS:
            hits = np.flatnonzero(executable_return.to_numpy(float) >= target)
            extra_labels[f"time_to_{int(target)}bps_seconds"] = (
                float((path_times.iloc[int(hits[0])] - entry_at).total_seconds())
                if len(hits)
                else np.nan
            )
        for label_horizon in LABEL_HORIZONS_MINUTES:
            reached_at = extra_labels["time_to_30bps_seconds"]
            extra_labels[f"target_30bps_within_{label_horizon}m"] = bool(
                pd.notna(reached_at) and float(reached_at) <= label_horizon * 60
            )
        stop_hits = np.flatnonzero(executable_return.to_numpy(float) <= -dynamic_stop)
        first_stop = int(stop_hits[0]) if len(stop_hits) else None
        extra_labels["time_to_stop_seconds"] = (
            float((path_times.iloc[first_stop] - entry_at).total_seconds())
            if first_stop is not None
            else np.nan
        )
        for barrier in BARRIER_BPS:
            target_hits = np.flatnonzero(executable_return.to_numpy(float) >= barrier)
            first_target = int(target_hits[0]) if len(target_hits) else None
            extra_labels[f"target_{int(barrier)}bps_before_stop"] = bool(
                first_target is not None and (first_stop is None or first_target < first_stop)
            )
        funding_bps = 0.0
        next_funding = entry_row.get("next_funding_timestamp_bitunix")
        if pd.notna(next_funding):
            funding_at = pd.Timestamp(next_funding)
            exit_at = pd.Timestamp(path.iloc[dynamic_position]["available_at"])
            if entry_at < funding_at <= exit_at:
                settled = path.loc[
                    path["funding_settlement_at"].eq(funding_at),
                    "settled_funding_rate",
                ].dropna()
                if settled.empty:
                    continue
                funding_bps = side * float(settled.iloc[-1]) * 10_000
        labels.append(
            {
                "minute": event["minute"],
                "signal_available_at": signal_at,
                "side": side,
                "entry_available_at": entry_at,
                "entry_mid": float(entry_row["mid"]),
                "entry_bid": float(entry_row["best_bid"]),
                "entry_ask": float(entry_row["best_ask"]),
                "entry_execution_vwap": float(entry_execution),
                "label_available_at": pd.Timestamp(path.iloc[-1]["available_at"]),
                "exit_mid": float(path.iloc[-1]["mid"]),
                "exit_bid": float(path.iloc[-1]["best_bid"]),
                "exit_ask": float(path.iloc[-1]["best_ask"]),
                "mfe_bps": float(executable_return.max()),
                "mae_bps": float(executable_return.min()),
                "terminal_bps": float(mid_return.iloc[-1]),
                "executable_terminal_bps": float(executable_return.iloc[-1]),
                "dynamic_exit_available_at": pd.Timestamp(
                    path.iloc[dynamic_position]["available_at"]
                ),
                "dynamic_exit_mid": float(path.iloc[dynamic_position]["mid"]),
                "dynamic_exit_bid": float(path.iloc[dynamic_position]["best_bid"]),
                "dynamic_exit_ask": float(path.iloc[dynamic_position]["best_ask"]),
                "dynamic_exit_execution_vwap": float(exit_execution.iloc[dynamic_position]),
                "dynamic_exit_reason": dynamic_reason,
                "dynamic_stop_bps": dynamic_stop,
                "dynamic_target_bps": target_bps,
                "fee_bps": 2 * TAKER_FEE_BPS_PER_SIDE,
                "slippage_reserve_bps": 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE,
                "funding_bps": funding_bps,
                "dynamic_mfe_bps": float(dynamic_path.max()),
                "dynamic_mae_bps": float(dynamic_path.min()),
                "dynamic_terminal_bps": float(mid_return.iloc[dynamic_position]),
                "dynamic_executable_terminal_bps": float(executable_return.iloc[dynamic_position]),
                **extra_labels,
            }
        )
    return pd.DataFrame(
        labels,
        columns=[
            "minute",
            "signal_available_at",
            "side",
            "entry_available_at",
            "entry_mid",
            "entry_bid",
            "entry_ask",
            "entry_execution_vwap",
            "label_available_at",
            "exit_mid",
            "exit_bid",
            "exit_ask",
            "mfe_bps",
            "mae_bps",
            "terminal_bps",
            "executable_terminal_bps",
            "dynamic_exit_available_at",
            "dynamic_exit_mid",
            "dynamic_exit_bid",
            "dynamic_exit_ask",
            "dynamic_exit_execution_vwap",
            "dynamic_exit_reason",
            "dynamic_stop_bps",
            "dynamic_target_bps",
            "fee_bps",
            "slippage_reserve_bps",
            "funding_bps",
            "dynamic_mfe_bps",
            "dynamic_mae_bps",
            "dynamic_terminal_bps",
            "dynamic_executable_terminal_bps",
            *(f"mfe_{value}m_bps" for value in LABEL_HORIZONS_MINUTES),
            *(f"mae_{value}m_bps" for value in LABEL_HORIZONS_MINUTES),
            *(f"time_to_{int(value)}bps_seconds" for value in TIME_TARGET_BPS),
            "time_to_stop_seconds",
            *(f"target_{int(value)}bps_before_stop" for value in BARRIER_BPS),
            *(f"target_30bps_within_{value}m" for value in LABEL_HORIZONS_MINUTES),
        ],
    )


def select_dynamic_events(
    frame: pd.DataFrame,
    l2: pd.DataFrame,
    family: str,
    *,
    anchored: bool = False,
    start: pd.Timestamp = DYNAMIC_PROTOCOL_START,
) -> pd.DataFrame:
    side, stages = _anchor_setup(frame, family) if anchored else _setup(frame, family)
    setup = pd.Series(True, index=frame.index)
    for _, condition in stages:
        setup &= condition
    candidates = frame.loc[frame["minute"].ge(start) & setup].copy()
    candidates["side"] = side.loc[candidates.index]
    chosen: list[Any] = []
    free_at = start
    previous_state: tuple[Any, ...] | None = None
    previous_score = float("-inf")
    latest_l2 = pd.to_datetime(l2["available_at"], utc=True).max()
    for index, row in candidates.sort_values("available_at").iterrows():
        signal_at = pd.Timestamp(row["available_at"])
        if signal_at < free_at:
            continue
        side_value = float(row["side"])
        state = (
            int(np.sign(side_value)),
            int(np.sign(float(row["alpha_vwap_distance_bps"]))),
            int(np.sign(float(row["alpha_taker_imbalance_60s"]))),
            str(row.get("anchor_available_at")) if anchored else "rolling",
        )
        score = (
            abs(float(row["alpha_return_5m_bps"]))
            + abs(float(row["alpha_vwap_distance_bps"]))
            + abs(float(row["alpha_taker_imbalance_60s"]))
        )
        if state == previous_state and score <= previous_score + 2.0:
            continue
        event = candidates.loc[[index]]
        label = path_labels(
            event,
            l2,
            DYNAMIC_MAX_HOLD_MINUTES,
            dynamic_early=True,
        )
        chosen.append(index)
        previous_state, previous_score = state, score
        if len(label):
            free_at = pd.Timestamp(label.iloc[0]["dynamic_exit_available_at"])
        elif signal_at + pd.Timedelta(minutes=DYNAMIC_MAX_HOLD_MINUTES) > latest_l2:
            break
        else:
            free_at = signal_at + pd.Timedelta(minutes=DYNAMIC_MAX_HOLD_MINUTES)
    return candidates.loc[chosen].copy()


def metrics(
    events: pd.DataFrame,
    horizon: int,
    l2: pd.DataFrame | None = None,
    *,
    dynamic: bool = False,
) -> dict[str, Any]:
    paths = (
        path_labels(events, l2, horizon, dynamic_early=dynamic)
        if l2 is not None and not l2.empty
        else pd.DataFrame()
    )
    executed = paths if l2 is not None else events
    pending = 0
    if l2 is not None and not l2.empty and len(events):
        latest_l2 = pd.to_datetime(l2["available_at"], utc=True).max()
        completed_minutes = set(pd.to_datetime(paths["minute"], utc=True))
        unfinished = events.loc[~pd.to_datetime(events["minute"], utc=True).isin(completed_minutes)]
        pending = int(
            (
                pd.to_datetime(unfinished["available_at"], utc=True) + pd.Timedelta(minutes=horizon)
                > latest_l2
            ).sum()
        )
    terminal = "dynamic_terminal_bps" if dynamic else "terminal_bps"
    executable_terminal = (
        "dynamic_executable_terminal_bps" if dynamic else "executable_terminal_bps"
    )
    mfe_column = "dynamic_mfe_bps" if dynamic else "mfe_bps"
    mae_column = "dynamic_mae_bps" if dynamic else "mae_bps"
    gross = (
        paths[terminal]
        if l2 is not None
        else events["side"] * events[f"future_return_{horizon}m_bps"]
    )
    funding = paths["funding_bps"] if l2 is not None and len(paths) else 0.0
    executable = paths[executable_terminal] if l2 is not None else gross
    net = executable - STRESS_COST_BPS - funding
    stress = executable - 2 * STRESS_COST_BPS - funding
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    gross_gains = gross[gross > 0].sum()
    observed_cost = gross - net
    holding_minutes = (
        (
            pd.to_datetime(paths["dynamic_exit_available_at"], utc=True)
            - pd.to_datetime(paths["entry_available_at"], utc=True)
        ).dt.total_seconds()
        / 60
        if dynamic and len(paths)
        else pd.Series(dtype=float)
    )
    daily = (
        pd.Series(net.to_numpy(), index=pd.DatetimeIndex(executed["minute"])).resample("1D").sum()
    )
    trades = []
    for position, (_, row) in enumerate(paths.reset_index(drop=True).iterrows()):
        trades.append(
            {
                "signal_at": pd.Timestamp(row["signal_available_at"]).isoformat(),
                "entry_at": pd.Timestamp(row["entry_available_at"]).isoformat(),
                "exit_at": pd.Timestamp(
                    row["dynamic_exit_available_at"] if dynamic else row["label_available_at"]
                ).isoformat(),
                "side": "LONG" if float(row["side"]) > 0 else "SHORT",
                "entry_mid": float(row["entry_mid"]),
                "exit_mid": float(row["dynamic_exit_mid"] if dynamic else row["exit_mid"]),
                "entry_execution_price": float(row["entry_execution_vwap"]),
                "exit_execution_price": float(
                    row["dynamic_exit_execution_vwap"]
                    if dynamic
                    else row["exit_bid"]
                    if float(row["side"]) > 0
                    else row["exit_ask"]
                ),
                "entry_spread_bps": float(
                    (row["entry_ask"] - row["entry_bid"]) / row["entry_mid"] * 10_000
                ),
                "execution_type": "TAKER_MARKET_REPLAY",
                "realized_entry_slippage_bps": float(
                    abs(float(row["entry_execution_vwap"]) / float(row["entry_mid"]) - 1) * 10_000
                ),
                "stop_bps": float(row["dynamic_stop_bps"]) if dynamic else None,
                "target_bps": float(row["dynamic_target_bps"]) if dynamic else None,
                "fee_bps": float(row["fee_bps"]) if dynamic else 2 * TAKER_FEE_BPS_PER_SIDE,
                "slippage_reserve_bps": (
                    float(row["slippage_reserve_bps"])
                    if dynamic
                    else 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
                ),
                "funding_bps": float(row["funding_bps"]) if dynamic else 0.0,
                "gross_bps": float(gross.iloc[position]),
                "net_bps": float(net.iloc[position]),
                "stress_bps": float(stress.iloc[position]),
                "mfe_bps": float(row[mfe_column]),
                "mae_bps": float(row[mae_column]),
                "exit_reason": (str(row["dynamic_exit_reason"]) if dynamic else f"TIME_{horizon}M"),
                **(
                    {
                        column: row[column]
                        for column in row.index
                        if column.startswith(("mfe_", "mae_", "time_to_", "target_"))
                    }
                    if dynamic
                    else {}
                ),
            }
        )
    return {
        "signals": len(events),
        "events": len(executed),
        "pending": pending,
        "execution_rejected": max(0, len(events) - len(executed) - pending),
        "gross_expectancy_bps": float(gross.mean()) if len(executed) else None,
        "net_expectancy_bps": float(net.mean()) if len(executed) else None,
        "stress_expectancy_bps": float(stress.mean()) if len(executed) else None,
        "net_expectancy_lcb_95_bps": _bootstrap_lcb(net),
        "win_rate": float(net.gt(0).mean()) if len(executed) else None,
        "gross_win_rate": float(gross.gt(0).mean()) if len(executed) else None,
        "average_winner_net_bps": float(net[net > 0].mean()) if net.gt(0).any() else None,
        "average_loser_net_bps": float(net[net < 0].mean()) if net.lt(0).any() else None,
        "mean_cost_bps": float(observed_cost.mean()) if len(executed) else None,
        "cost_to_gross_profit_ratio": (
            float(observed_cost.sum() / gross_gains) if gross_gains > 0 else None
        ),
        "profit_factor": float(gains / losses) if losses else None,
        "positive_day_fraction": float(daily.gt(0).mean()) if len(daily) else None,
        "path_labels": len(paths),
        "mean_mfe_bps": float(paths[mfe_column].mean()) if len(paths) else None,
        "median_mfe_bps": float(paths[mfe_column].median()) if len(paths) else None,
        "mean_mae_bps": float(paths[mae_column].mean()) if len(paths) else None,
        "median_mae_bps": float(paths[mae_column].median()) if len(paths) else None,
        "mean_holding_minutes": float(holding_minutes.mean()) if len(holding_minutes) else None,
        "median_holding_minutes": (
            float(holding_minutes.median()) if len(holding_minutes) else None
        ),
        "tp_before_stop_probability": {
            str(int(barrier)): (
                float(paths[f"target_{int(barrier)}bps_before_stop"].mean()) if len(paths) else None
            )
            for barrier in BARRIER_BPS
        },
        "mean_time_to_target_seconds": {
            str(int(target)): (
                float(paths[f"time_to_{int(target)}bps_seconds"].mean())
                if len(paths) and paths[f"time_to_{int(target)}bps_seconds"].notna().any()
                else None
            )
            for target in TIME_TARGET_BPS
        },
        "mfe_above_taker_cost_rate": (
            float(paths[mfe_column].ge(STRESS_COST_BPS).mean()) if len(paths) else None
        ),
        "exit_reasons": (
            paths["dynamic_exit_reason"].value_counts().to_dict() if dynamic else None
        ),
        "trades": trades,
    }


def oracle_metrics(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    label = frame.loc[frame["minute"].ge(PROTOCOL_START), f"future_return_{horizon}m_bps"].dropna()
    absolute = label.abs()
    oracle = (absolute - ROUND_TRIP_COST_BPS).clip(lower=0)
    stress_oracle = (absolute - STRESS_COST_BPS).clip(lower=0)
    return {
        "timestamps": len(label),
        "positive_rate": float(oracle.gt(0).mean()) if len(oracle) else None,
        "expectancy_bps": float(oracle.mean()) if len(oracle) else None,
        "mean_absolute_move_bps": float(absolute.mean()) if len(absolute) else None,
        "taker_taker_positive_rate": (
            float(stress_oracle.gt(0).mean()) if len(stress_oracle) else None
        ),
        "taker_taker_oracle_expectancy_bps": (
            float(stress_oracle.mean()) if len(stress_oracle) else None
        ),
        "move_at_least_three_x_maker_cost_rate": (
            float(absolute.ge(ROUND_TRIP_COST_BPS * MINIMUM_GROSS_MOVEMENT_TO_COST).mean())
            if len(absolute)
            else None
        ),
    }


def eligible_names(results: dict[str, dict[str, Any]]) -> list[str]:
    return [
        name
        for name, result in results.items()
        if result["events"] >= MINIMUM_EVENTS
        and (result["net_expectancy_bps"] or 0) > 0
        and (result["stress_expectancy_bps"] or 0) >= 0
        and (result["net_expectancy_lcb_95_bps"] or 0) > 0
        and (result["profit_factor"] or 0) >= MINIMUM_PROFIT_FACTOR
        and (result["positive_day_fraction"] or 0) > MINIMUM_POSITIVE_DAY_FRACTION
        and (result["gross_expectancy_bps"] or 0)
        >= ROUND_TRIP_COST_BPS * MINIMUM_GROSS_MOVEMENT_TO_COST
    ]


def counterfactual_frame(frame: pd.DataFrame, results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    available = pd.to_datetime(frame["available_at"], utc=True)
    records: list[dict[str, Any]] = []
    for expert, result in results.items():
        expert_id = hashlib.sha256(f"{DYNAMIC_PROTOCOL_HASH}:{expert}".encode()).hexdigest()[:16]
        for trade in result["trades"]:
            signal_at = pd.Timestamp(trade["signal_at"])
            matches = frame.loc[available.eq(signal_at)]
            if matches.empty:
                continue
            state = matches.iloc[-1]
            feature_available_at = pd.Timestamp(state["available_at"])
            if feature_available_at > signal_at:
                continue
            records.append(
                {
                    "expert_id": expert_id,
                    "expert": expert,
                    "protocol_hash": DYNAMIC_PROTOCOL_HASH,
                    "signal_at": signal_at,
                    "feature_available_at": feature_available_at,
                    "entry_at": pd.Timestamp(trade["entry_at"]),
                    "exit_at": pd.Timestamp(trade["exit_at"]),
                    "side": trade["side"],
                    "entry_mid": trade["entry_mid"],
                    "exit_mid": trade.get("exit_mid"),
                    "entry_execution_price": trade.get("entry_execution_price"),
                    "exit_execution_price": trade.get("exit_execution_price"),
                    "entry_spread_bps": trade.get("entry_spread_bps"),
                    "execution_type": trade.get("execution_type"),
                    "realized_entry_slippage_bps": trade.get("realized_entry_slippage_bps"),
                    "stop_bps": trade.get("stop_bps"),
                    "target_bps": trade.get("target_bps"),
                    "fee_bps": trade.get("fee_bps"),
                    "slippage_reserve_bps": trade.get("slippage_reserve_bps"),
                    "funding_bps": trade.get("funding_bps"),
                    "expected_total_cost_bps": float(trade.get("fee_bps", 0.0) or 0.0)
                    + float(trade.get("slippage_reserve_bps", 0.0) or 0.0)
                    + max(0.0, float(trade.get("funding_bps", 0.0) or 0.0)),
                    "gross_bps": trade["gross_bps"],
                    "net_bps": trade["net_bps"],
                    "stress_bps": trade["stress_bps"],
                    "mfe_bps": trade["mfe_bps"],
                    "mae_bps": trade["mae_bps"],
                    "exit_reason": trade["exit_reason"],
                    **{
                        key: value
                        for key, value in trade.items()
                        if key.startswith(("mfe_", "mae_", "time_to_", "target_"))
                    },
                    **{feature: state.get(feature) for feature in COUNTERFACTUAL_FEATURES},
                }
            )
    return pd.DataFrame(records)


def apply_alpha_management(counterfactual: pd.DataFrame, l2: pd.DataFrame) -> pd.DataFrame:
    """Make the Alpha-selected target change the actual causal execution replay."""
    if counterfactual.empty or "alpha_target_bps" not in counterfactual:
        return counterfactual
    managed = counterfactual.copy()
    available = pd.to_datetime(l2["available_at"], utc=True)
    available_ns = available.to_numpy(dtype="datetime64[ns]").astype("int64")
    for index, row in managed.loc[managed["alpha_target_bps"].notna()].iterrows():
        target = int(row["alpha_target_bps"])
        target_seconds = row.get(f"time_to_{target}bps_seconds")
        stop_seconds = row.get("time_to_stop_seconds")
        entry_at = pd.Timestamp(row["entry_at"])
        legacy_exit = pd.Timestamp(row["exit_at"])
        target_first = bool(row.get(f"target_{target}bps_before_stop", False))
        event_seconds: float | None = None
        reason: str | None = None
        if target_first and pd.notna(target_seconds):
            event_seconds, reason = float(target_seconds), "ALPHA_DYNAMIC_TARGET"
        elif pd.notna(stop_seconds):
            event_seconds, reason = float(stop_seconds), "TECHNICAL_STOP"
        if event_seconds is None:
            continue
        proposed_at = entry_at + pd.Timedelta(seconds=event_seconds)
        if proposed_at >= legacy_exit:
            continue
        position = int(np.searchsorted(available_ns, proposed_at.value, side="left"))
        if position >= len(l2):
            continue
        exit_row = l2.iloc[position]
        side = 1 if str(row["side"]) == "LONG" else -1
        entry_execution = float(row["entry_execution_price"])
        quantity = LABEL_NOTIONAL_USDT / entry_execution
        exit_execution = _book_vwap(exit_row["bids" if side > 0 else "asks"], quantity)
        if exit_execution is None:
            continue
        gross = (
            (exit_execution / entry_execution - 1) * 10_000
            if side > 0
            else (entry_execution / exit_execution - 1) * 10_000
        )
        fee = float(row.get("fee_bps", 2 * TAKER_FEE_BPS_PER_SIDE) or 0.0)
        reserve = float(row.get("slippage_reserve_bps", 0.0) or 0.0)
        funding = float(row.get("funding_bps", 0.0) or 0.0)
        managed.loc[index, "exit_at"] = pd.Timestamp(exit_row["available_at"])
        managed.loc[index, "exit_mid"] = float(exit_row["mid"])
        managed.loc[index, "exit_execution_price"] = exit_execution
        managed.loc[index, "gross_bps"] = gross
        managed.loc[index, "net_bps"] = gross - fee - reserve - funding
        managed.loc[index, "stress_bps"] = gross - 2 * (fee + reserve) - funding
        managed.loc[index, "exit_reason"] = reason
        managed.loc[index, "target_bps"] = target
    return managed


def fee_profile_counterfactuals(scored: pd.DataFrame, l2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Replay the same Alpha candidates as six independent Bitunix fee policies."""
    profiles: dict[str, pd.DataFrame] = {}
    for level in range(6):
        profile = f"VIP{level}"
        candidate = scored.copy()
        suffix = f"vip{level}"
        aliases = {
            "alpha_target_bps": f"alpha_target_{suffix}_bps",
            "alpha_target_probability": f"alpha_target_probability_{suffix}",
            "alpha_stop_probability": f"alpha_stop_probability_{suffix}",
            "alpha_timeout_probability": f"alpha_timeout_probability_{suffix}",
            "alpha_expected_time_to_target_minutes": (
                f"alpha_expected_time_to_target_{suffix}_minutes"
            ),
            "alpha_expected_total_cost_bps": f"alpha_expected_total_cost_{suffix}_bps",
            "alpha_expected_net_bps": f"alpha_expected_net_{suffix}_bps",
            "alpha_prudent_net_bps": f"alpha_prudent_net_{suffix}_bps",
            "alpha_accepted": f"alpha_accepted_{suffix}",
            "alpha_status": f"alpha_status_{suffix}",
        }
        if not set(aliases.values()).issubset(candidate.columns):
            profiles[profile] = candidate.iloc[0:0].copy()
            continue
        for destination, source in aliases.items():
            candidate[destination] = candidate[source]
        _, taker_bps = futures_fee_bps(level)
        candidate["fee_profile"] = profile
        candidate["fee_bps"] = 2 * taker_bps
        reserve = candidate.get(
            "slippage_reserve_bps", pd.Series(0.0, index=candidate.index)
        ).fillna(0.0)
        funding = candidate.get("funding_bps", pd.Series(0.0, index=candidate.index)).fillna(0.0)
        candidate["expected_total_cost_bps"] = (
            candidate["fee_bps"] + reserve + funding.clip(lower=0.0)
        )
        candidate["net_bps"] = candidate["gross_bps"] - candidate["fee_bps"] - reserve - funding
        candidate["stress_bps"] = (
            candidate["gross_bps"] - 2 * (candidate["fee_bps"] + reserve) - funding
        )
        profiles[profile] = apply_alpha_management(candidate, l2)
    return profiles


def _book_vwap(levels: Any, quantity: float) -> float | None:
    remaining = quantity
    cost = 0.0
    if not isinstance(levels, (list, np.ndarray)) or quantity <= 0:
        return None
    for level in levels:
        price, available = float(level[0]), float(level[1])
        filled = min(remaining, available)
        cost += filled * price
        remaining -= filled
        if remaining <= 1e-12:
            return cost / quantity
    return None


def one_position_diagnostics(
    frame: pd.DataFrame,
    l2: pd.DataFrame | None = None,
    *,
    paper_start: pd.Timestamp = PAPER_ACCOUNT_START,
    vip_level: int = BITUNIX_VIP_LEVEL,
) -> dict[str, Any]:
    profile_maker_bps, profile_taker_bps = futures_fee_bps(vip_level)
    fee_profile = f"VIP{vip_level}"
    if frame.empty:
        return {
            "earliest_candidate": {"trades": 0, "net_bps": 0.0, "stress_bps": 0.0},
            "stress_oracle": {"trades": 0, "stress_bps": 0.0},
            "paper_account": {
                "status": "SHADOW_BASELINE_NOT_TRAINED",
                "initial_equity": PAPER_INITIAL_EQUITY,
                "paper_start": paper_start.isoformat(),
                "risk_per_trade": PAPER_RISK_PER_TRADE,
                "margin_fraction": PAPER_MARGIN_FRACTION,
                "max_leverage": PAPER_MAX_LEVERAGE,
                "fee_profile": fee_profile,
                "final_equity": PAPER_INITIAL_EQUITY,
                "net_pnl": 0.0,
                "trades": [],
            },
        }
    ordered = frame.sort_values(["signal_at", "expert_id"]).reset_index(drop=True)
    chosen: list[int] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    for position, (_, row) in enumerate(ordered.iterrows()):
        signal_at = pd.Timestamp(row["signal_at"])
        if signal_at >= free_at:
            chosen.append(position)
            free_at = pd.Timestamp(row["exit_at"])
    earliest = ordered.loc[chosen]
    balance = PAPER_INITIAL_EQUITY
    peak_balance = balance
    max_drawdown = 0.0
    paper_trades: list[dict[str, Any]] = []
    total_costs = 0.0
    depth_rejections = 0
    risk_rejections: dict[str, int] = {}
    current_day: object | None = None
    day_start_balance = balance
    l2_available = pd.to_datetime(l2["available_at"], utc=True) if l2 is not None else None
    paper_columns = {
        "stop_bps",
        "entry_mid",
        "exit_mid",
        "entry_execution_price",
        "exit_execution_price",
        "entry_spread_bps",
        "expert",
        "side",
        "entry_at",
        "exit_reason",
    }
    paper_candidates = ordered.loc[pd.to_datetime(ordered["signal_at"], utc=True).ge(paper_start)]
    if "alpha_accepted" in paper_candidates:
        paper_candidates = paper_candidates.loc[paper_candidates["alpha_accepted"].astype(bool)]
    paper_chosen: list[int] = []
    paper_free_at = paper_start
    for position, (_, row) in enumerate(paper_candidates.iterrows()):
        signal_at = pd.Timestamp(row["signal_at"])
        if signal_at >= paper_free_at:
            paper_chosen.append(position)
            paper_free_at = pd.Timestamp(row["exit_at"])
    paper_rows = (
        paper_candidates.iloc[paper_chosen]
        if paper_columns.issubset(paper_candidates.columns)
        else paper_candidates.iloc[0:0]
    )
    for _, row in paper_rows.iterrows():
        trade_day = pd.Timestamp(row["signal_at"]).date()
        if trade_day != current_day:
            current_day, day_start_balance = trade_day, balance
        rejection: str | None = None
        if balance <= day_start_balance * 0.98:
            rejection = "daily_drawdown_limit"
        elif max_drawdown >= 0.08:
            rejection = "max_strategy_drawdown"
        stop_bps = float(row["stop_bps"])
        entry_mid = float(row["entry_mid"])
        exit_mid = float(row["exit_mid"])
        entry_execution = float(row["entry_execution_price"])
        exit_execution = float(row["exit_execution_price"])
        entry_spread_bps = float(row["entry_spread_bps"])
        if not all(
            np.isfinite(value) and value > 0
            for value in (
                stop_bps,
                entry_mid,
                exit_mid,
                entry_execution,
                exit_execution,
            )
        ) or not np.isfinite(entry_spread_bps):
            continue
        if entry_spread_bps > 5.0:
            rejection = "spread_too_high"
        if str(row.get("market_regime", "")) == "STRESS":
            rejection = "abnormal_volatility"
        if stop_bps > HARD_CATASTROPHIC_STOP_BPS:
            rejection = "technical_stop_beyond_catastrophic_stop"
        if rejection is not None:
            risk_rejections[rejection] = risk_rejections.get(rejection, 0) + 1
            continue
        fee_bps = float(row.get("fee_bps", 2 * profile_taker_bps))
        reserve_bps = float(row.get("slippage_reserve_bps", 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE))
        funding_bps = float(row.get("funding_bps", 0.0))
        estimated_cost_bps = fee_bps + reserve_bps
        risk_budget = balance * PAPER_RISK_PER_TRADE
        maximum_notional = balance * PAPER_MARGIN_FRACTION * PAPER_MAX_LEVERAGE
        notional = min(
            maximum_notional,
            risk_budget * 10_000 / (stop_bps + estimated_cost_bps),
        )
        if notional <= 0 or notional / PAPER_MAX_LEVERAGE > balance:
            risk_rejections["position_limit"] = risk_rejections.get("position_limit", 0) + 1
            continue
        quantity = notional / entry_execution
        if l2 is not None and l2_available is not None:
            entry_match = l2.loc[l2_available.eq(pd.Timestamp(row["entry_at"]))]
            exit_match = l2.loc[l2_available.eq(pd.Timestamp(row["exit_at"]))]
            if entry_match.empty or exit_match.empty:
                depth_rejections += 1
                continue
            entry_book = entry_match.iloc[-1]["asks" if row["side"] == "LONG" else "bids"]
            exit_book = exit_match.iloc[-1]["bids" if row["side"] == "LONG" else "asks"]
            entry_fill = _book_vwap(entry_book, quantity)
            if entry_fill is not None:
                quantity = notional / entry_fill
                entry_fill = _book_vwap(entry_book, quantity)
            exit_fill = _book_vwap(exit_book, quantity)
            if entry_fill is None or exit_fill is None:
                depth_rejections += 1
                continue
            entry_execution, exit_execution = entry_fill, exit_fill
        gross_pnl = notional * float(row["gross_bps"]) / 10_000
        maker_pnl = notional * float(row["net_bps"]) / 10_000
        exit_notional = quantity * exit_execution
        trading_pnl = quantity * (
            exit_execution - entry_execution
            if row["side"] == "LONG"
            else entry_execution - exit_execution
        )
        fee_cost = profile_taker_bps / 10_000 * (notional + exit_notional)
        slippage_reserve = SLIPPAGE_RESERVE_BPS_PER_SIDE / 10_000 * (notional + exit_notional)
        funding_cost = notional * funding_bps / 10_000
        net_pnl = trading_pnl - fee_cost - slippage_reserve - funding_cost
        vip_scenarios = {
            f"VIP{level}": {
                "maker_bps_per_side": maker,
                "taker_bps_per_side": taker,
                "net_pnl": trading_pnl
                - taker / 10_000 * (notional + exit_notional)
                - slippage_reserve
                - funding_cost,
                "predicted_ev_bps": (
                    float(row[f"alpha_ev_vip{level}_bps"])
                    if pd.notna(row.get(f"alpha_ev_vip{level}_bps"))
                    else None
                ),
            }
            for level, (maker, taker) in FUTURES_VIP_FEE_BPS.items()
            if level <= 5
        }
        costs = gross_pnl - net_pnl
        balance += net_pnl
        total_costs += costs
        peak_balance = max(peak_balance, balance)
        max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
        paper_trades.append(
            {
                "signal_at": pd.Timestamp(row["signal_at"]).isoformat(),
                "entry_at": pd.Timestamp(row["entry_at"]).isoformat(),
                "exit_at": pd.Timestamp(row["exit_at"]).isoformat(),
                "expert": str(row["expert"]),
                "side": str(row["side"]),
                "entry_price": entry_mid,
                "exit_price": exit_mid,
                "entry_execution_price": entry_execution,
                "exit_execution_price": exit_execution,
                "stop_bps": stop_bps,
                "target_bps": float(row.get("target_bps", 30.0)),
                "alpha_target_bps": (
                    float(row["alpha_target_bps"])
                    if pd.notna(row.get("alpha_target_bps"))
                    else None
                ),
                "alpha_target_probability": (
                    float(row["alpha_target_probability"])
                    if pd.notna(row.get("alpha_target_probability"))
                    else None
                ),
                "alpha_expected_time_to_target_minutes": (
                    float(row["alpha_expected_time_to_target_minutes"])
                    if pd.notna(row.get("alpha_expected_time_to_target_minutes"))
                    else None
                ),
                "alpha_expected_mfe_60m_bps": (
                    float(row["expected_mfe_60m_bps"])
                    if pd.notna(row.get("expected_mfe_60m_bps"))
                    else None
                ),
                "alpha_expected_mae_60m_bps": (
                    float(row["expected_mae_60m_bps"])
                    if pd.notna(row.get("expected_mae_60m_bps"))
                    else None
                ),
                "estimated_cost_bps": estimated_cost_bps,
                "risk_budget": risk_budget,
                "estimated_stop_loss_with_costs": notional
                * (stop_bps + estimated_cost_bps)
                / 10_000,
                "execution_type": "TAKER_MARKET_REPLAY",
                "break_even_price": entry_execution * (1 + estimated_cost_bps / 10_000)
                if row["side"] == "LONG"
                else entry_execution * (1 - estimated_cost_bps / 10_000),
                "technical_stop_price": entry_execution * (1 - stop_bps / 10_000)
                if row["side"] == "LONG"
                else entry_execution * (1 + stop_bps / 10_000),
                "catastrophic_stop_price": entry_execution
                * (1 - HARD_CATASTROPHIC_STOP_BPS / 10_000)
                if row["side"] == "LONG"
                else entry_execution * (1 + HARD_CATASTROPHIC_STOP_BPS / 10_000),
                "notional": notional,
                "quantity_btc": quantity,
                "gross_pnl": gross_pnl,
                "maker_pnl": maker_pnl,
                "net_pnl": net_pnl,
                "stress_pnl": net_pnl,
                "fees": fee_cost,
                "slippage_reserve": slippage_reserve,
                "funding": funding_cost,
                "modeled_costs": costs,
                "balance": balance,
                "exit_reason": str(row["exit_reason"]),
                "vip_scenarios": vip_scenarios,
            }
        )
    net = earliest["net_bps"]
    stress = earliest["stress_bps"]
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    daily = (
        pd.Series(net.to_numpy(), index=pd.DatetimeIndex(earliest["signal_at"]))
        .resample("1D")
        .sum()
    )
    profit_factor = float(gains / losses) if losses else None
    net_lcb = _bootstrap_lcb(net)
    positive_days = float(daily.gt(0).mean()) if len(daily) else None
    gross_expectancy = float(earliest["gross_bps"].mean())
    eligible = (
        len(earliest) >= MINIMUM_EVENTS
        and float(net.mean()) > 0
        and float(stress.mean()) >= 0
        and (net_lcb or 0) > 0
        and (profit_factor or 0) >= MINIMUM_PROFIT_FACTOR
        and (positive_days or 0) > MINIMUM_POSITIVE_DAY_FRACTION
        and gross_expectancy >= ROUND_TRIP_COST_BPS * MINIMUM_GROSS_MOVEMENT_TO_COST
    )

    by_exit = frame.sort_values("exit_at").reset_index(drop=True)
    best = [0.0] * (len(by_exit) + 1)
    counts = [0] * (len(by_exit) + 1)
    for position, (_, row) in enumerate(by_exit.iterrows()):
        prior = by_exit.iloc[:position]
        compatible = prior.index[prior["exit_at"].le(row["signal_at"])]
        previous = int(compatible[-1]) + 1 if len(compatible) else 0
        take = best[previous] + float(row["stress_bps"])
        if take > best[position]:
            best[position + 1] = take
            counts[position + 1] = counts[previous] + 1
        else:
            best[position + 1] = best[position]
            counts[position + 1] = counts[position]
    return {
        "earliest_candidate": {
            "trades": len(earliest),
            "net_bps": float(earliest["net_bps"].sum()),
            "stress_bps": float(earliest["stress_bps"].sum()),
            "net_expectancy_bps": float(net.mean()),
            "stress_expectancy_bps": float(stress.mean()),
            "net_expectancy_lcb_95_bps": net_lcb,
            "profit_factor": profit_factor,
            "positive_day_fraction": positive_days,
            "gross_expectancy_bps": gross_expectancy,
            "eligible": eligible,
        },
        "stress_oracle": {
            "trades": counts[-1],
            "stress_bps": best[-1],
            "uses_future_information": True,
        },
        "paper_account": {
            "status": "ALPHA_SHADOW_ACTIVE"
            if btc_vwap_alpha.BUNDLE.exists()
            else "ALPHA_NOT_TRAINED",
            "selection_rule": (
                f"positive Binance Alpha {fee_profile} net EV; one Bitunix position at a time"
            ),
            "used_for_initial_alpha_training": False,
            "reserved_for_future_bitunix_calibration": True,
            "initial_equity": PAPER_INITIAL_EQUITY,
            "paper_start": paper_start.isoformat(),
            "risk_per_trade": PAPER_RISK_PER_TRADE,
            "max_leverage": PAPER_MAX_LEVERAGE,
            "margin_fraction": PAPER_MARGIN_FRACTION,
            "maximum_notional_fraction": PAPER_MARGIN_FRACTION * PAPER_MAX_LEVERAGE,
            "fee_profile": fee_profile,
            "final_equity": balance,
            "net_pnl": balance - PAPER_INITIAL_EQUITY,
            "modeled_costs": total_costs,
            "maker_fees_per_side_bps": profile_maker_bps,
            "fees_per_side_bps": profile_taker_bps,
            "slippage_reserve_per_side_bps": SLIPPAGE_RESERVE_BPS_PER_SIDE,
            "max_drawdown": max_drawdown,
            "depth_execution_rejections": depth_rejections,
            "risk_rejections": risk_rejections,
            "execution_model": "observed Bitunix 15-level book VWAP; fail closed if insufficient",
            "maker_status": "DISABLED_UNTIL_PRIVATE_FILL_LABELS",
            "trades": paper_trades,
            "vip_scenario_note": "Same fills, size and funding; only official taker fee changes",
        },
    }


def _finite(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) and np.isfinite(float(number)) else None


def _source_snapshot(
    venue: str,
    purpose: str,
    observed_at: Any,
    evaluated_at: pd.Timestamp,
    *,
    valid: bool,
    max_age_seconds: float,
) -> dict[str, Any]:
    observed = pd.Timestamp(observed_at) if pd.notna(observed_at) else None
    age = max(0.0, (evaluated_at - observed).total_seconds()) if observed is not None else None
    state = (
        "MISSING"
        if observed is None
        else "INVALID"
        if not valid
        else "STALE"
        if age is None or age > max_age_seconds
        else "FRESH"
    )
    return {
        "venue": venue,
        "purpose": purpose,
        "observed_at": observed.isoformat() if observed is not None else None,
        "age_seconds": age,
        "max_age_seconds": max_age_seconds,
        "valid": state == "FRESH",
        "state": state,
    }


def _check_snapshot(
    name: str, passed: bool, row: pd.Series, side_value: float
) -> dict[str, Any]:
    """Expose the exact canonical Alpha input used by each visible gate."""
    return_1m = _finite(row.get("alpha_return_1m_bps"))
    return_15m = _finite(row.get("alpha_return_15m_bps"))
    return_30m = _finite(row.get("alpha_return_30m_bps"))
    distance = _finite(row.get("alpha_vwap_distance_bps"))
    slope = _finite(row.get("alpha_vwap_slope_bps"))
    flow = _finite(row.get("alpha_taker_imbalance_60s"))
    side = float(np.sign(side_value)) if np.isfinite(side_value) else 0.0

    def directed(value: float | None) -> float | None:
        return value * side if value is not None and side != 0 else None

    trend_inputs = (return_15m, return_30m, slope)
    trend_score = (
        float(sum(np.sign(value) for value in trend_inputs if value is not None))
        if all(value is not None for value in trend_inputs)
        else None
    )
    values: dict[str, tuple[object, str]] = {
        "direction": (
            trend_score,
            "voto sign(15m) + sign(30m) + sign(slope VWAP) diverso da zero",
        ),
        "trend_15m": (directed(return_15m), "rendimento Alpha 15m x direzione > 0"),
        "trend_30m": (directed(return_30m), "rendimento Alpha 30m x direzione > 0"),
        "vwap_zone": (
            directed(distance),
            "distanza VWAP x direzione tra -3 e 15 bps",
        ),
        "price_restart": (directed(return_1m), "rendimento Alpha 1m x direzione > 0"),
        "taker_flow_restart": (
            directed(flow),
            "taker imbalance Alpha 60s x direzione > 0",
        ),
        "vwap_extension": (
            abs(distance) if distance is not None else None,
            "|distanza rolling VWAP| >= 5 bps",
        ),
        "price_reversal": (directed(return_1m), "rendimento Alpha 1m verso VWAP > 0"),
        "taker_flow_reversal": (
            directed(flow),
            "taker imbalance Alpha 60s verso VWAP > 0",
        ),
        "vwap_cross": (
            distance,
            "cambio di lato VWAP osservato negli ultimi 5 minuti Alpha",
        ),
    }
    actual, requirement = values.get(name, (None, name))
    return {"name": name, "passed": passed, "actual": actual, "requirement": requirement}


def _anchor_snapshot(row: pd.Series) -> dict[str, Any]:
    age = _finite(row.get("anchor_age_seconds"))
    side = _finite(row.get("anchor_direction"))
    detected = (
        pd.Timestamp(row["anchor_available_at"])
        if pd.notna(row.get("anchor_available_at"))
        else None
    )
    observed = (
        pd.Timestamp(row["alpha_l2_available_at"])
        if pd.notna(row.get("alpha_l2_available_at"))
        else None
    )
    measured_age = (
        (observed - detected).total_seconds()
        if detected is not None and observed is not None
        else None
    )
    consistent = age is not None and measured_age is not None and abs(age - measured_age) <= 10
    state = (
        "NONE"
        if age is None or side in (None, 0)
        else "INVALID"
        if not consistent
        else "WARMING"
        if age < 60
        else "VALID"
        if age <= 1_800
        else "EXPIRED"
    )
    return {
        "source": "Binance BTCUSDT perpetual L2",
        "state": state,
        "valid": state == "VALID",
        "detected_at": detected.isoformat() if detected is not None else None,
        "age_seconds": age,
        "measured_age_seconds": measured_age,
        "direction": "LONG" if (side or 0) > 0 else "SHORT" if (side or 0) < 0 else None,
        "price": _finite(row.get("anchored_vwap")),
        "distance_bps": _finite(row.get("anchored_vwap_distance_bps")),
        "valid_from_seconds": 60,
        "valid_until_seconds": 1_800,
    }


def current_market_assessment(
    frame: pd.DataFrame,
    selector: dict[str, Any],
    execution_l2: pd.DataFrame | None = None,
    *,
    equity: float = PAPER_INITIAL_EQUITY,
    vip_level: int = BITUNIX_VIP_LEVEL,
    evaluated_at: pd.Timestamp | None = None,
    base_candidate: dict[str, Any] | None = None,
    base_policy_ready: bool = False,
) -> dict[str, Any]:
    if frame.empty:
        return {"decision": "WAIT", "reason": "NO_BINANCE_ALPHA_OBSERVATION", "setups": []}
    raw_frame = frame
    alpha_minute_column = (
        "alpha_l2_available_at" if "alpha_l2_available_at" in frame else "available_at"
    )
    frame = (
        frame.assign(
            __alpha_minute=pd.to_datetime(frame[alpha_minute_column], utc=True).dt.floor("min")
        )
        .sort_values("available_at")
        .drop_duplicates("__alpha_minute", keep="last")
        .reset_index(drop=True)
    )
    alpha_report_status = "NOT_TRAINED"
    if btc_vwap_alpha.REPORT.exists():
        try:
            alpha_report_status = str(
                json.loads(btc_vwap_alpha.REPORT.read_text(encoding="utf-8")).get(
                    "status", "REPORT_INVALID"
                )
            )
        except (OSError, ValueError, TypeError):
            alpha_report_status = "REPORT_INVALID"
    latest_position = int(frame["available_at"].argmax())
    latest = frame.iloc[latest_position]
    execution_rows = execution_l2 if execution_l2 is not None else raw_frame
    if "feature_valid" in execution_rows:
        execution_rows = execution_rows.loc[execution_rows["feature_valid"].fillna(False)]
    execution_latest = (
        execution_rows.sort_values("available_at").iloc[-1]
        if not execution_rows.empty and "available_at" in execution_rows
        else pd.Series(dtype=object)
    )
    source_times = [pd.Timestamp(latest["available_at"])]
    if not execution_latest.empty:
        source_times.append(pd.Timestamp(execution_latest["available_at"]))
    evaluated = pd.Timestamp(evaluated_at) if evaluated_at is not None else max(source_times)
    if evaluated.tzinfo is None:
        evaluated = evaluated.tz_localize("UTC")
    else:
        evaluated = evaluated.tz_convert("UTC")

    families = list(FAMILIES)
    evaluations: list[dict[str, Any]] = []
    cadence = frame["__alpha_minute"].dt.minute.mod(5).eq(0).to_numpy()
    previous_decisions = np.flatnonzero(cadence[:latest_position])
    previous_decision = int(previous_decisions[-1]) if len(previous_decisions) else None
    for family in families:
        side, stages = _setup(frame, family)
        side_value = float(side.iloc[latest_position])
        checks = [
            _check_snapshot(name, bool(condition.iloc[latest_position]), latest, side_value)
            for name, condition in stages
        ]
        failed = next((check["name"] for check in checks if not check["passed"]), None)
        active = failed is None and side_value != 0
        previous_active = False
        if previous_decision is not None and active:
            previous_side = float(side.iloc[previous_decision])
            previous_active = previous_side == side_value and all(
                bool(condition.iloc[previous_decision]) for _, condition in stages
            )
        candidate_block = (
            failed
            if failed is not None
            else "decision_cadence"
            if not bool(cadence[latest_position])
            else "episode_already_evaluated"
            if previous_active
            else None
        )
        evaluations.append(
            {
                "setup": family,
                "direction": "LONG" if side_value > 0 else "SHORT" if side_value < 0 else None,
                "setup_active": active,
                "candidate": active and bool(cadence[latest_position]) and not previous_active,
                "passed_checks": sum(check["passed"] for check in checks),
                "total_checks": len(checks),
                "first_failed_check": candidate_block,
                "checks": checks,
            }
        )
    if base_candidate is not None:
        evaluations.append(dict(base_candidate))
    elif base_policy_ready:
        evaluations.insert(
            0,
            {
                "setup": "IMPULSE_PULLBACK_MULTI_HORIZON",
                "direction": None,
                "candidate": False,
                "setup_active": False,
                "passed_checks": 0,
                "total_checks": 1,
                "first_failed_check": "waiting_new_impulse_pullback_restart",
                "checks": [],
                "policy_source": "MUSCA_V8_FROZEN_BASE_MONITOR",
                "management_style": "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M",
                "partial_target_fraction": 0.5,
                "maximum_hold_minutes": 360,
            },
        )
    candidates = [
        item
        for item in evaluations
        if item["candidate"]
        and (
            not base_policy_ready
            or item.get("policy_source") == "MUSCA_V8_FROZEN_BASE"
        )
    ]
    winner = (
        candidates[0]
        if candidates
        else max(evaluations, key=lambda item: (item["passed_checks"], item["setup"]))
    )
    alpha_price = _finite(latest.get("mid")) or _finite(latest.get("price_binance"))
    execution_price = _finite(execution_latest.get("mid"))
    range_bps = _finite(latest.get("alpha_range_60s_bps"))
    base_target_bps = 50.0 if range_bps is not None and range_bps >= 40 / 3 else 30.0

    alpha_input_columns = btc_vwap_alpha.LIVE_REQUIRED - {"signal_at", "side", "expert", "stop_bps"}
    missing_alpha_features = sorted(
        column
        for column in alpha_input_columns
        if column not in latest or pd.isna(latest.get(column))
    )
    alpha_complete = not missing_alpha_features and bool(
        latest.get("alpha_feature_contract_valid", False)
    )
    alpha_source = _source_snapshot(
        "Binance",
        "Alpha: barre Binance complete, VWAP, momentum e taker flow",
        latest.get("alpha_l2_available_at", latest.get("available_at")),
        evaluated,
        valid=bool(latest.get("feature_valid", False)) and alpha_complete,
        max_age_seconds=MAX_ALPHA_AGE_SECONDS,
    )
    integrity = MarketIntegrity(
        float(execution_latest.get("last_trade_update_age_ms", np.inf)),
        float(execution_latest.get("last_book_update_age_ms", np.inf)),
        bool(execution_latest.get("sequence_gap_detected", True)),
        bool(execution_latest.get("book_is_synced", False)),
        bool(execution_latest.get("trade_feed_alive", False)),
        bool(execution_latest.get("orderbook_feed_alive", False)),
        float(execution_latest.get("clock_drift_ms", np.inf)),
    )
    integrity_rejection = integrity.rejection_reason()
    execution_source = _source_snapshot(
        "Bitunix",
        "Paper execution: bid/ask, profondità, fill, fee e funding",
        execution_latest.get("available_at"),
        evaluated,
        valid=not execution_latest.empty and integrity_rejection is None,
        max_age_seconds=MAX_EXECUTION_AGE_SECONDS,
    )
    data_rejection = (
        f"BINANCE_ALPHA_{alpha_source['state']}"
        if not alpha_source["valid"]
        else f"BITUNIX_EXECUTION_{execution_source['state']}"
        if not execution_source["valid"]
        else None
    )

    best_bid = _finite(execution_latest.get("best_bid"))
    best_ask = _finite(execution_latest.get("best_ask"))
    book_ready = best_bid is not None and best_ask is not None and best_ask > best_bid
    fee_profile = f"VIP{vip_level}"
    _, profile_taker_bps = futures_fee_bps(vip_level)
    profile_suffix = f"vip{vip_level}"
    alpha_model_present = btc_vwap_alpha.BUNDLE.exists()

    def evaluate_action(action: dict[str, Any]) -> dict[str, Any]:
        action_direction = str(action.get("direction"))
        action_sign = 1 if action_direction == "LONG" else -1
        is_candidate = bool(action.get("candidate"))
        is_frozen_base = action.get("policy_source") == "MUSCA_V8_FROZEN_BASE"
        next_funding_value = latest.get("next_funding_timestamp_bitunix")
        next_funding = (
            pd.Timestamp(next_funding_value)
            if next_funding_value is not None and pd.notna(next_funding_value)
            else pd.NaT
        )
        if pd.notna(next_funding):
            next_funding = (
                next_funding.tz_localize("UTC")
                if next_funding.tzinfo is None
                else next_funding.tz_convert("UTC")
            )
        funding_due = bool(
            pd.notna(next_funding)
            and evaluated
            <= next_funding
            <= evaluated
            + pd.Timedelta(minutes=int(action.get("maximum_hold_minutes", 60)))
        )
        action_funding_bps = (
            max(
                0.0,
                action_sign * float(latest.get("funding_bitunix", 0.0) or 0.0) * 10_000,
            )
            if funding_due
            else 0.0
        )
        action_execution = (
            quote_taker_round_trip(
                side=action_direction,
                quantity=Decimal("0.001"),
                bids=execution_latest.get("bids"),
                asks=execution_latest.get("asks"),
                best_bid=Decimal(str(best_bid)),
                best_ask=Decimal(str(best_ask)),
                vip_level=vip_level,
                expected_funding_bps=Decimal(str(action_funding_bps)),
            )
            if is_candidate
            and data_rejection is None
            and execution_price is not None
            and book_ready
            and best_bid is not None
            and best_ask is not None
            else None
        )
        recent_low = _finite(latest.get("alpha_recent_low_5m"))
        recent_high = _finite(latest.get("alpha_recent_high_5m"))
        structural_stop = None
        frozen_stop = _finite(action.get("stop_price"))
        if is_frozen_base and alpha_price is not None and frozen_stop is not None:
            structural_stop = action_sign * (alpha_price - frozen_stop) / alpha_price * 10_000
        elif alpha_price is not None and recent_low is not None and recent_high is not None:
            structural_stop = (
                (alpha_price / recent_low - 1) * 10_000
                if action_sign > 0
                else (recent_high / alpha_price - 1) * 10_000
            )
        action_stop_bps = (
            float(structural_stop)
            if is_frozen_base and structural_stop is not None and structural_stop > 0
            else float(
                np.clip(
                    max(
                        DYNAMIC_RANGE_MULTIPLIER * range_bps,
                        (structural_stop or 0.0) + 2.0,
                    ),
                    DYNAMIC_MIN_STOP_BPS,
                    btc_vwap_alpha.MAX_TECHNICAL_STOP_BPS,
                )
            )
            if range_bps is not None
            else None
        )
        action_cost_bps = (
            float(action_execution.expected_total_cost_bps)
            + 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
            if action_execution is not None
            else 2 * (profile_taker_bps + SLIPPAGE_RESERVE_BPS_PER_SIDE)
            + action_funding_bps
        )
        action_risk = None
        if action_execution is not None and action_stop_bps is not None:
            for _ in range(2):
                action_risk = size_for_technical_stop(
                    equity=Decimal(str(equity)),
                    entry_price=action_execution.entry.execution_vwap,
                    side=action_direction,
                    technical_stop_bps=Decimal(str(action_stop_bps)),
                    expected_cost_bps=Decimal(str(action_cost_bps)),
                    lot_size=Decimal("0.001"),
                    minimum_quantity=Decimal("0.001"),
                    minimum_notional=Decimal("5"),
                )
                if not action_risk.approved:
                    break
                sized = quote_taker_round_trip(
                    side=action_direction,
                    quantity=action_risk.quantity,
                    bids=execution_latest.get("bids"),
                    asks=execution_latest.get("asks"),
                    best_bid=Decimal(str(best_bid)),
                    best_ask=Decimal(str(best_ask)),
                    vip_level=vip_level,
                    expected_funding_bps=Decimal(str(action_funding_bps)),
                )
                if sized is None:
                    action_execution = None
                    break
                action_execution = sized
                action_cost_bps = (
                    float(action_execution.expected_total_cost_bps)
                    + 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
                )
        if is_frozen_base:
            robust_gross = float(action["robust_expected_gross_bps"])
            expected_net = robust_gross - action_cost_bps
            target_bps = 1.5 * float(action_stop_bps or 0.0)
            scored: dict[str, Any] | pd.Series = {
                "alpha_status": (
                    "BASE_ALPHA_NET_POSITIVE" if expected_net > 0 else "BASE_ALPHA_COST_BLOCKED"
                ),
                f"alpha_status_{profile_suffix}": (
                    "BASE_ALPHA_NET_POSITIVE" if expected_net > 0 else "BASE_ALPHA_COST_BLOCKED"
                ),
                f"alpha_ev_{profile_suffix}_bps": expected_net,
                f"alpha_target_{profile_suffix}_bps": target_bps,
                f"alpha_target_probability_{profile_suffix}": action.get(
                    "target_probability"
                ),
                f"alpha_stop_probability_{profile_suffix}": None,
                f"alpha_timeout_probability_{profile_suffix}": None,
                f"alpha_expected_time_to_target_{profile_suffix}_minutes": None,
                "expected_mfe_60m_bps": None,
                "expected_mae_60m_bps": None,
            }
            for level in range(6):
                _, scenario_taker = futures_fee_bps(level)
                scored[f"alpha_ev_vip{level}_bps"] = (
                    robust_gross
                    - action_cost_bps
                    + 2 * (profile_taker_bps - scenario_taker)
                )
            ready = action_stop_bps is not None and action_stop_bps > 0
            model_accepts = expected_net > 0
        else:
            alpha_row = latest.to_frame().T.copy()
            alpha_row["side"] = action_direction
            alpha_row["expert"] = str(action["setup"]).lower() + "_dynamic"
            alpha_row["signal_at"] = latest["available_at"]
            alpha_row["stop_bps"] = action_stop_bps
            alpha_row["expected_non_fee_cost_bps"] = max(
                0.0, action_cost_bps - 2 * profile_taker_bps
            )
            scored = btc_vwap_alpha.score(alpha_row).iloc[0]
            ready = alpha_model_present and pd.notna(
                scored.get(f"alpha_target_probability_{profile_suffix}")
            )
            model_accepts = bool(scored.get(f"alpha_accepted_{profile_suffix}", False))
        accepted = (
            model_accepts
            and is_candidate
            and data_rejection is None
            and integrity_rejection is None
            and action_execution is not None
            and action_risk is not None
            and action_risk.approved
        )
        return {
            "action": action,
            "direction": action_direction,
            "sign": action_sign,
            "execution": action_execution,
            "risk": action_risk,
            "stop_bps": action_stop_bps,
            "estimated_cost_bps": action_cost_bps,
            "expected_funding_bps": action_funding_bps,
            "alpha_scored": scored,
            "alpha_ready": ready,
            "alpha_accepted": accepted,
            "is_frozen_base": is_frozen_base,
            "rank_ev": float(scored.get(f"alpha_ev_{profile_suffix}_bps", -np.inf)),
        }

    action_results = [evaluate_action(action) for action in (candidates or [winner])]
    accepted_actions = [result for result in action_results if result["alpha_accepted"]]
    ranked_actions = [result for result in action_results if result["alpha_ready"]]
    chosen = max(
        accepted_actions or ranked_actions or action_results,
        key=lambda item: item["rank_ev"],
    )
    winner = chosen["action"]
    direction = chosen["direction"]
    sign = chosen["sign"]
    execution = chosen["execution"]
    risk = chosen["risk"]
    stop_bps = chosen["stop_bps"]
    estimated_cost_bps = chosen["estimated_cost_bps"]
    expected_funding_bps = chosen["expected_funding_bps"]
    alpha_scored = chosen["alpha_scored"]
    alpha_ready = chosen["alpha_ready"]
    alpha_accepted = chosen["alpha_accepted"]
    is_frozen_base = chosen["is_frozen_base"]
    proposed = bool(candidates) and alpha_price is not None and sign != 0
    if not is_frozen_base:
        base_target_bps = max(
            base_target_bps, MINIMUM_GROSS_MOVEMENT_TO_COST * estimated_cost_bps
        )
    result_by_setup = {result["action"]["setup"]: result for result in action_results}
    for evaluation in evaluations:
        result = result_by_setup.get(evaluation["setup"])
        if result is not None:
            scored = result["alpha_scored"]
            evaluation["alpha_status"] = scored.get(
                f"alpha_status_{profile_suffix}", scored.get("alpha_status")
            )
            evaluation["expected_net_ev_bps"] = (
                result["rank_ev"] if np.isfinite(result["rank_ev"]) else None
            )
    decision = (
        "WAIT"
        if not candidates or data_rejection is not None
        else "TRADE"
        if alpha_accepted
        else "FLAT"
        if alpha_model_present or alpha_ready
        else "CANDIDATE_ONLY"
    )
    reason = (
        "WAIT_NEW_IMPULSE_PULLBACK_RESTART"
        if base_policy_ready and not candidates
        else "WAIT_NEXT_5M_DECISION"
        if not candidates and winner["first_failed_check"] == "decision_cadence"
        else "SETUP_EPISODE_ALREADY_EVALUATED"
        if not candidates and winner["first_failed_check"] == "episode_already_evaluated"
        else f"Closest setup blocked at: {winner['first_failed_check']}"
        if not candidates
        else str(data_rejection)
        if data_rejection is not None
        else str(integrity_rejection)
        if integrity_rejection is not None
        else "INSUFFICIENT_BITUNIX_BOOK_DEPTH"
        if execution is None
        else risk.reason
        if risk is not None and not risk.approved
        else str(
            alpha_scored.get(
                f"alpha_status_{profile_suffix}",
                alpha_scored.get("alpha_status", "ALPHA_UNAVAILABLE"),
            )
        )
        if alpha_model_present or alpha_ready
        else "SETUP_COMPLETE_ALPHA_NOT_AVAILABLE"
    )
    chosen_target_bps = (
        float(alpha_scored[f"alpha_target_{profile_suffix}_bps"])
        if alpha_ready and proposed
        else base_target_bps
    )
    entry_reference = float(execution.entry.execution_vwap) if execution is not None else None
    risk_status = (
        "APPROVED"
        if risk is not None and risk.approved
        else "REJECTED"
        if risk is not None
        else "NOT_EVALUATED"
    )
    execution_status = (
        "QUOTED" if execution is not None else "NOT_EVALUATED" if not proposed else "BLOCKED"
    )
    anchor = _anchor_snapshot(latest)
    display_winner = (
        next(
            evaluation
            for evaluation in evaluations
            if evaluation.get("policy_source") == "MUSCA_V8_FROZEN_BASE_MONITOR"
        )
        if base_policy_ready and not candidates
        else winner
    )
    display_is_base = str(display_winner.get("policy_source", "")).startswith(
        "MUSCA_V8_FROZEN_BASE"
    )
    show_alpha_values = alpha_ready and bool(candidates)
    return {
        "evaluated_at": evaluated.isoformat(),
        "observed_at": pd.Timestamp(latest["available_at"]).isoformat(),
        "decision": decision,
        "reason": reason,
        "fee_profile": fee_profile,
        "setup": display_winner["setup"],
        "direction": display_winner.get("direction", direction),
        "policy_source": display_winner.get(
            "policy_source", "GENERIC_ML_CHALLENGER"
        ),
        "expert_id": display_winner.get("expert_id"),
        "alpha_signal_at": pd.Timestamp(
            str(display_winner.get("available_at") or latest.get("available_at"))
        ).isoformat(),
        "management_style": display_winner.get(
            "management_style", "FIXED_TARGET_STOP"
        ),
        "partial_target_fraction": display_winner.get(
            "partial_target_fraction", 0.0
        ),
        "maximum_hold_minutes": int(
            display_winner.get("maximum_hold_minutes", 60)
        ),
        "candidate_complete": bool(candidates),
        "model_context": "COMPLETE_CANDIDATE" if candidates else "NEAREST_SETUP_HYPOTHESIS",
        "model_feature_coverage": {
            "complete": alpha_complete,
            "missing": missing_alpha_features,
        },
        "probability_status": (
            "FROZEN_BASE_HISTORICAL_CALIBRATION"
            if display_is_base and bool(candidates)
            else "FROZEN_BASE_WAITING_EVENT"
            if display_is_base
            else "ALPHA_RESEARCH_ONLY"
            if alpha_report_status == "NO_ECONOMIC_ALPHA"
            else "ALPHA_MODEL_READY"
        )
        if alpha_model_present or display_is_base
        else selector.get("probability_status", "NOT_TRAINED"),
        "target_probability": _finite(
            alpha_scored.get(f"alpha_target_probability_{profile_suffix}")
        )
        if show_alpha_values
        else None,
        "stop_probability": _finite(
            alpha_scored.get(f"alpha_stop_probability_{profile_suffix}")
        )
        if show_alpha_values
        else None,
        "timeout_probability": _finite(
            alpha_scored.get(f"alpha_timeout_probability_{profile_suffix}")
        )
        if show_alpha_values
        else None,
        "expected_net_ev_bps": _finite(
            alpha_scored.get(f"alpha_ev_{profile_suffix}_bps")
        )
        if show_alpha_values
        else None,
        "expected_mfe_60m_bps": _finite(alpha_scored.get("expected_mfe_60m_bps"))
        if show_alpha_values
        else None,
        "expected_mae_60m_bps": _finite(alpha_scored.get("expected_mae_60m_bps"))
        if show_alpha_values
        else None,
        "expected_time_to_target_minutes": _finite(
            alpha_scored.get(
                f"alpha_expected_time_to_target_{profile_suffix}_minutes"
            )
        )
        if show_alpha_values
        else None,
        "vip_ev_bps": {
            f"VIP{level}": (
                float(alpha_scored[f"alpha_ev_vip{level}_bps"])
                if show_alpha_values
                and pd.notna(alpha_scored.get(f"alpha_ev_vip{level}_bps"))
                else None
            )
            for level in range(6)
        },
        "price": alpha_price,
        "execution_price": execution_price,
        "rolling_vwap": _finite(latest.get("alpha_rolling_vwap")),
        "anchored_vwap": _finite(latest.get("anchored_vwap")),
        "spread_bps": _finite(execution_latest.get("spread_bps")),
        "flow_vote": float(_flow_vote(frame.iloc[[latest_position]]).iloc[0]),
        "binance_return_1m_bps": _finite(latest.get("alpha_return_1m_bps")),
        "binance_return_5m_bps": _finite(latest.get("alpha_return_5m_bps")),
        "vwap_distance_bps": _finite(latest.get("alpha_vwap_distance_bps")),
        "stop_bps": stop_bps,
        "target_bps": chosen_target_bps if proposed else None,
        "expected_cost_bps": estimated_cost_bps,
        "expected_funding_bps": expected_funding_bps,
        "execution_status": execution_status,
        "execution_type": execution.execution_type if execution is not None else None,
        "entry_execution_vwap": entry_reference,
        "estimated_exit_execution_vwap": float(execution.estimated_exit.execution_vwap)
        if execution is not None
        else None,
        "execution_levels_entry": execution.entry.levels_consumed
        if execution is not None
        else None,
        "risk_status": risk_status,
        "risk_approved": risk.approved if risk is not None else None,
        "risk_reason": risk.reason
        if risk is not None
        else "NOT_EVALUATED_NO_COMPLETE_EXECUTABLE_CANDIDATE",
        "risk_budget": float(risk.risk_budget) if risk is not None else None,
        "notional": float(risk.notional) if risk is not None else None,
        "quantity_btc": float(risk.quantity) if risk is not None else None,
        "break_even_price": entry_reference * (1 + sign * estimated_cost_bps / 10_000)
        if entry_reference is not None
        else None,
        "stop_price": entry_reference * (1 - sign * float(stop_bps) / 10_000)
        if entry_reference is not None and stop_bps is not None
        else None,
        "target_price": entry_reference * (1 + sign * chosen_target_bps / 10_000)
        if entry_reference is not None
        else None,
        "anchor": anchor,
        "sources": {"alpha": alpha_source, "execution": execution_source},
        "market_inputs": {
            "binance": {
                "price": alpha_price,
                "return_1m_bps": _finite(latest.get("alpha_return_1m_bps")),
                "return_5m_bps": _finite(latest.get("alpha_return_5m_bps")),
                "return_15m_bps": _finite(latest.get("alpha_return_15m_bps")),
                "return_30m_bps": _finite(latest.get("alpha_return_30m_bps")),
                "rolling_vwap": _finite(latest.get("alpha_rolling_vwap")),
                "vwap_distance_bps": _finite(latest.get("alpha_vwap_distance_bps")),
                "vwap_slope_bps": _finite(latest.get("alpha_vwap_slope_bps")),
                "range_60s_bps": range_bps,
                "taker_imbalance_60s": _finite(
                    latest.get("alpha_taker_imbalance_60s")
                ),
                "trend_score": (
                    float(
                        np.sign(float(latest["alpha_return_15m_bps"]))
                        + np.sign(float(latest["alpha_return_30m_bps"]))
                        + np.sign(float(latest["alpha_vwap_slope_bps"]))
                    )
                    if all(
                        _finite(latest.get(column)) is not None
                        for column in (
                            "alpha_return_15m_bps",
                            "alpha_return_30m_bps",
                            "alpha_vwap_slope_bps",
                        )
                    )
                    else None
                ),
                "l2_aggressive_imbalance_60s": _finite(
                    latest.get("aggressive_imbalance_60s")
                ),
                "depth_imbalance_5": _finite(latest.get("depth_imbalance_5")),
                "microprice_distance_bps": _finite(latest.get("microprice_distance_bps")),
                "l2_flow_vote": float(_flow_vote(frame.iloc[[latest_position]]).iloc[0]),
            },
            "bitunix": {
                "mid": execution_price,
                "mark_price": _finite(latest.get("price_bitunix")),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread_bps": _finite(execution_latest.get("spread_bps")),
                "funding_rate": _finite(latest.get("funding_bitunix")),
                "next_funding_timestamp": (
                    pd.Timestamp(latest["next_funding_timestamp_bitunix"]).isoformat()
                    if pd.notna(latest.get("next_funding_timestamp_bitunix"))
                    else None
                ),
                "book_synced": bool(execution_latest.get("book_is_synced", False)),
                "clock_drift_ms": _finite(execution_latest.get("clock_drift_ms")),
            },
        },
        "setups": evaluations,
        "evaluation_frequency": (
            "causal Binance Alpha snapshot; Bitunix execution quote evaluated separately"
        ),
        "outcome_horizons_minutes": list(LABEL_HORIZONS_MINUTES),
    }


def market_chart(
    alpha_l2: pd.DataFrame,
    assessment: dict[str, Any],
    execution_l2: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    recent = alpha_l2.sort_values("available_at").tail(1_800)
    if recent.empty:
        return []
    recent = recent.iloc[:: max(1, len(recent) // 360)]
    execution_prices = pd.Series(np.nan, index=recent.index)
    if execution_l2 is not None and not execution_l2.empty:
        joined = pd.merge_asof(
            recent[["available_at"]].sort_values("available_at"),
            execution_l2[["available_at", "mid"]]
            .sort_values("available_at")
            .rename(columns={"mid": "execution_price"}),
            on="available_at",
            direction="backward",
            tolerance=pd.Timedelta(seconds=5),
        )
        execution_prices = joined["execution_price"]
    levels = {
        "break_even": assessment.get("break_even_price"),
        "stop": assessment.get("stop_price"),
        "target": assessment.get("target_price"),
    }
    return [
        {
            "timestamp": pd.Timestamp(row["available_at"]).isoformat(),
            "close": float(row["mid"]),
            "execution_price": _finite(execution_prices.iloc[position]),
            "center": float(row["rolling_vwap"]) if pd.notna(row["rolling_vwap"]) else None,
            "anchored_vwap": (
                float(row["anchored_vwap"]) if pd.notna(row["anchored_vwap"]) else None
            ),
            **levels,
        }
        for position, (_, row) in enumerate(recent.iterrows())
    ]


def _live_market_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _LIVE_ALPHA_CACHE

    now = time.monotonic()
    if _LIVE_ALPHA_CACHE is None or now - _LIVE_ALPHA_CACHE[0] >= 50:
        try:
            alpha_l2 = binance_l2_dataset.build_live_minute_features(
                binance_l2_dataset.load_recent_records(max_lines=30_000),
                binance_l2_dataset.load_recent_official_minutes(),
            )
            alpha_l2 = alpha_l2.tail(LIVE_ALPHA_MINUTES).reset_index(drop=True)
            frame = build_features(load_snapshots(), alpha_l2, alpha_clock=True)
            if not frame.empty:
                frame = frame.loc[
                    frame["feature_valid"]
                    & frame["minute"].ge(EXECUTION_PROTOCOL_START)
                ].copy()
            _LIVE_ALPHA_CACHE = (now, frame, alpha_l2)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            if _LIVE_ALPHA_CACHE is None:
                raise
    _, frame, alpha_l2 = _LIVE_ALPHA_CACHE
    execution_l2 = bitunix_l2_dataset.latest_execution_snapshot()
    return frame, alpha_l2, execution_l2


def _write_report(payload: dict[str, Any]) -> None:
    with REPORT_LOCK:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        temporary = REPORT.with_name(f"{REPORT.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(REPORT)


def _align_active_alpha_metadata(
    payload: dict[str, Any], alpha_status: str
) -> dict[str, Any]:
    base_report: dict[str, Any] = {}
    if musca_v8_multi_horizon.REPORT.exists():
        with contextlib.suppress(OSError, ValueError, TypeError, json.JSONDecodeError):
            base_report = json.loads(
                musca_v8_multi_horizon.REPORT.read_text(encoding="utf-8")
            )
    base_status = str(base_report.get("verdict", "BASE_NOT_READY"))
    base_ready = base_status == "RESEARCH_BASE_ALPHA_READY"
    active_hash = (
        musca_v8_multi_horizon.PROTOCOL_HASH
        if base_ready
        else btc_vwap_alpha.PROTOCOL_HASH
    )
    protocol_changed = payload.get("protocol_hash") != active_hash
    legacy_embedded = any(key in payload for key in LEGACY_EMBEDDED_REPORT_KEYS)
    for key in LEGACY_EMBEDDED_REPORT_KEYS:
        payload.pop(key, None)
    payload["validation_status"] = (
        "RESEARCH_BASE_ALPHA_READY"
        if base_ready
        else "RESEARCH_ALPHA_READY"
        if alpha_status == "RESEARCH_ALPHA_READY"
        else "RESEARCH_ONLY_NO_ECONOMIC_ALPHA"
        if alpha_status == "NO_ECONOMIC_ALPHA"
        else "COLLECTING_NO_POLICY"
    )
    payload["live_orders_enabled"] = False
    payload["protocol"] = (
        musca_v8_multi_horizon.PROTOCOL if base_ready else btc_vwap_alpha.PROTOCOL
    )
    payload["protocol_hash"] = active_hash
    payload["policy_hash"] = active_hash
    payload["decision_spec"] = {
        "primary_family": (
            "IMPULSE_PULLBACK_MULTI_HORIZON"
            if base_ready
            else "GENERIC_VWAP_ACTIONS"
        ),
        "generic_challenger_families": list(btc_vwap_alpha.FAMILIES),
        "cadence_minutes": 5,
        "alpha_venue": "Binance BTCUSDT perpetual/spot",
        "execution_venue": "Bitunix BTCUSDT observed paper execution",
        "decision": (
            "frozen 2024-2025 impulse-pullback expert with positive expected net "
            "EV after observed Bitunix costs; generic ML is challenger only"
        ),
    }
    payload["selector"] = {
        "status": base_status if base_ready else alpha_status,
        "probability_status": (
            "FROZEN_BASE_HISTORICAL_CALIBRATION"
            if base_ready
            else "ALPHA_RESEARCH_ONLY"
            if alpha_status == "NO_ECONOMIC_ALPHA"
            else "ALPHA_MODEL_READY"
            if alpha_status == "RESEARCH_ALPHA_READY"
            else "NOT_TRAINED"
        ),
        "protocol": {
            "name": payload["protocol"]["name"],
            "protocol_hash": active_hash,
        },
        "minimum_holdout_days": btc_vwap_forward_selector.MINIMUM_HOLDOUT_DAYS,
        "minimum_holdout_trades": btc_vwap_forward_selector.MINIMUM_HOLDOUT_TRADES,
    }
    alpha = payload.setdefault("alpha", {})
    alpha.update(
        {
            "status": base_status if base_ready else alpha_status,
            "report": str(
                musca_v8_multi_horizon.REPORT if base_ready else btc_vwap_alpha.REPORT
            ),
            "bundle": None if base_ready else str(btc_vwap_alpha.BUNDLE),
            "operating_profile": "VIP0",
            "scenario_profiles": [f"VIP{level}" for level in range(6)],
            "historical_counterfactual_status": "SEE_ACTIVE_ALPHA_REPORT",
            "real_capital_allowed": False,
            "generic_ml_challenger": {
                "status": alpha_status,
                "report": str(btc_vwap_alpha.REPORT),
                "order_authority": False,
            },
            "paper_profiles": base_report.get("paper_profiles", {}) if base_ready else {},
            "oos_2026": base_report.get("test", {}) if base_ready else {},
            "oos_2026_stress_2x": base_report.get("test_stress", {})
            if base_ready
            else {},
            "gates": base_report.get("gates", {}) if base_ready else {},
        }
    )
    if protocol_changed or legacy_embedded:
        alpha.update(
            {
                "scored_candidates": 0,
                "accepted_candidates": 0,
                "accepted_candidates_by_profile": {
                    f"VIP{level}": 0 for level in range(6)
                },
                "latest_candidate": None,
            }
        )
    payload["legacy_discovery"] = {
        "status": "EXCLUDED_FROM_ACTIVE_PROTOCOL",
        "detail": (
            "V21-V23 embedded metrics are not mixed with the active Alpha. "
            "Source artifacts remain preserved on disk."
        ),
    }
    return payload


def run() -> dict[str, Any]:
    alpha_cache = binance_l2_dataset.ROOT / "btcusdt_l2_features.parquet"
    alpha_cache_age = (
        time.time() - alpha_cache.stat().st_mtime if alpha_cache.exists() else float("inf")
    )
    if alpha_cache_age > HISTORICAL_CACHE_REFRESH_SECONDS:
        binance_l2_dataset.materialize()
    alpha_l2 = pd.read_parquet(alpha_cache)

    execution_cache = bitunix_l2_dataset.MATERIALIZED
    execution_cache_age = (
        time.time() - execution_cache.stat().st_mtime if execution_cache.exists() else float("inf")
    )
    if execution_cache_age > HISTORICAL_CACHE_REFRESH_SECONDS:
        bitunix_l2_dataset.materialize()
    l2 = pd.read_parquet(execution_cache)
    frame = build_features(load_snapshots(), alpha_l2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_features = OUTPUT.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary_features, index=False)
    temporary_features.replace(OUTPUT)
    funding = frame[["available_at", "funding_bitunix", "next_funding_timestamp_bitunix"]].copy()
    funding["available_at"] = pd.to_datetime(funding["available_at"], utc=True)
    funding["next_funding_timestamp_bitunix"] = pd.to_datetime(
        funding["next_funding_timestamp_bitunix"], utc=True
    )
    l2 = pd.merge_asof(
        l2.sort_values("available_at"),
        funding.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=90),
    )
    l2["funding_settlement_at"] = pd.Series(pd.NaT, index=l2.index, dtype="datetime64[ns, UTC]")
    l2["settled_funding_rate"] = np.nan
    try:
        settlements = _settled_bitunix_funding()
    except (OSError, ValueError):
        settlements = pd.DataFrame()
    available_ns = pd.to_datetime(l2["available_at"], utc=True).astype("int64").to_numpy()
    for settlement in settlements.to_dict("records"):
        settled_at = pd.Timestamp(settlement["funding_settlement_at"])
        position = int(np.searchsorted(available_ns, settled_at.value, side="left"))
        if position < len(l2):
            l2.loc[l2.index[position], "funding_settlement_at"] = settled_at
            l2.loc[l2.index[position], "settled_funding_rate"] = settlement["settled_funding_rate"]
    required = [
        "alpha_return_1m_bps",
        "alpha_return_15m_bps",
        "alpha_return_30m_bps",
        "alpha_vwap_distance_bps",
        "alpha_vwap_slope_bps",
        "alpha_taker_imbalance_60s",
    ]
    frame = frame.loc[
        frame["feature_valid"]
        & frame["alpha_feature_contract_valid"].fillna(False)
        & frame[required].notna().all(axis=1)
    ].copy()
    # V1/V2 remain archived discovery; V3 only recomputes its forward protocol.
    results: dict[str, dict[str, Any]] = {}
    anchored_results: dict[str, dict[str, Any]] = {}
    dynamic_frame = frame.loc[frame["minute"].ge(EXECUTION_PROTOCOL_START)]
    dynamic_results = {
        f"{family.lower()}_dynamic": metrics(
            select_dynamic_events(dynamic_frame, l2, family, start=EXECUTION_PROTOCOL_START),
            DYNAMIC_MAX_HOLD_MINUTES,
            l2,
            dynamic=True,
        )
        for family in FAMILIES
    }
    eligible: list[str] = []
    anchored_eligible: list[str] = []
    dynamic_eligible = eligible_names(dynamic_results)
    scored_counterfactual = btc_vwap_alpha.score(counterfactual_frame(frame, dynamic_results))
    profile_counterfactuals = fee_profile_counterfactuals(scored_counterfactual, l2)
    counterfactual = profile_counterfactuals["VIP0"]
    profile_diagnostics = {
        profile: one_position_diagnostics(rows, l2, vip_level=level)
        for level, (profile, rows) in enumerate(profile_counterfactuals.items())
    }
    one_position = profile_diagnostics["VIP0"]
    one_position["historical_replay_paper_accounts"] = {
        profile: diagnostics["paper_account"]
        for profile, diagnostics in profile_diagnostics.items()
    }
    live_paper_accounts = musca_v5_paper.load_accounts()
    one_position["paper_accounts"] = live_paper_accounts
    one_position["paper_account"] = live_paper_accounts["VIP0"]
    COUNTERFACTUAL.parent.mkdir(parents=True, exist_ok=True)
    temporary_counterfactual = COUNTERFACTUAL.with_suffix(".parquet.tmp")
    counterfactual.to_parquet(temporary_counterfactual, index=False)
    temporary_counterfactual.replace(COUNTERFACTUAL)
    fee_profiles = pd.concat(profile_counterfactuals.values(), ignore_index=True)
    temporary_profiles = FEE_PROFILE_COUNTERFACTUAL.with_suffix(".parquet.tmp")
    fee_profiles.to_parquet(temporary_profiles, index=False)
    temporary_profiles.replace(FEE_PROFILE_COUNTERFACTUAL)
    selector = btc_vwap_forward_selector.evaluate(COUNTERFACTUAL)
    elapsed_hours = max(
        1 / 60,
        (datetime.now(UTC) - DYNAMIC_PROTOCOL_START.to_pydatetime()).total_seconds() / 3_600,
    )
    training_counterfactual = (
        counterfactual.loc[
            pd.to_datetime(counterfactual["signal_at"], utc=True).ge(
                btc_vwap_forward_selector.PROTOCOL_START
            )
        ]
        if len(counterfactual)
        else counterfactual
    )
    observed_decisions = (
        int(training_counterfactual["signal_at"].nunique()) if len(training_counterfactual) else 0
    )
    observed_rate = observed_decisions / elapsed_hours
    remaining = int(selector.get("decisions_remaining", 0))
    until_selector_start = max(
        0.0,
        (
            btc_vwap_forward_selector.PROTOCOL_START.to_pydatetime() - datetime.now(UTC)
        ).total_seconds()
        / 3_600,
    )
    selector["observed_training_decisions_per_hour"] = observed_rate
    selector["estimated_hours_to_gpu_freeze"] = (
        until_selector_start + remaining / observed_rate
        if observed_rate > 0 and remaining
        else None
    )
    selector["minimum_holdout_days"] = btc_vwap_forward_selector.MINIMUM_HOLDOUT_DAYS
    selector["minimum_holdout_trades"] = btc_vwap_forward_selector.MINIMUM_HOLDOUT_TRADES
    discovery = (
        counterfactual.loc[counterfactual["signal_at"].lt(btc_vwap_forward_selector.PROTOCOL_START)]
        if len(counterfactual)
        else counterfactual
    )
    selector["discovery_rows_excluded"] = len(discovery)
    selector["discovery_decisions_excluded"] = (
        int(discovery["signal_at"].nunique()) if len(discovery) else 0
    )
    live_frame, live_alpha_l2, live_execution_l2 = _live_market_inputs()
    assessment_time = pd.Timestamp.now(tz="UTC")
    assessments = {
        profile: current_market_assessment(
            live_frame,
            selector,
            live_execution_l2,
            equity=float(live_paper_accounts[profile]["final_equity"]),
            vip_level=level,
            evaluated_at=assessment_time,
        )
        for level, (profile, diagnostics) in enumerate(profile_diagnostics.items())
    }
    assessment = assessments["VIP0"]
    alpha_latest: dict[str, Any] | None = None
    alpha_status = "NOT_TRAINED"
    if btc_vwap_alpha.REPORT.exists():
        try:
            alpha_status = str(
                json.loads(btc_vwap_alpha.REPORT.read_text(encoding="utf-8")).get(
                    "status", "REPORT_INVALID"
                )
            )
        except (OSError, ValueError, TypeError):
            alpha_status = "REPORT_INVALID"
    if len(counterfactual):
        latest_alpha = counterfactual.sort_values("signal_at").iloc[-1]
        alpha_latest = {
            "signal_at": pd.Timestamp(latest_alpha["signal_at"]).isoformat(),
            "setup": str(latest_alpha["expert"]),
            "direction": str(latest_alpha["side"]),
            "decision": "TRADE" if bool(latest_alpha["alpha_accepted"]) else "FLAT",
            "reason": str(latest_alpha.get("alpha_status", "NOT_SCORED")),
            "target_bps": (
                float(latest_alpha["alpha_target_bps"])
                if pd.notna(latest_alpha.get("alpha_target_bps"))
                else None
            ),
            "target_probability": (
                float(latest_alpha["alpha_target_probability"])
                if pd.notna(latest_alpha.get("alpha_target_probability"))
                else None
            ),
            "stop_probability": (
                float(latest_alpha["alpha_stop_probability"])
                if pd.notna(latest_alpha.get("alpha_stop_probability"))
                else None
            ),
            "timeout_probability": (
                float(latest_alpha["alpha_timeout_probability"])
                if pd.notna(latest_alpha.get("alpha_timeout_probability"))
                else None
            ),
            "expected_mfe_60m_bps": (
                float(latest_alpha["expected_mfe_60m_bps"])
                if pd.notna(latest_alpha.get("expected_mfe_60m_bps"))
                else None
            ),
            "expected_mae_60m_bps": (
                float(latest_alpha["expected_mae_60m_bps"])
                if pd.notna(latest_alpha.get("expected_mae_60m_bps"))
                else None
            ),
            "expected_time_to_target_minutes": (
                float(latest_alpha["alpha_expected_time_to_target_minutes"])
                if pd.notna(latest_alpha.get("alpha_expected_time_to_target_minutes"))
                else None
            ),
            "vip_ev_bps": {
                f"VIP{level}": (
                    float(latest_alpha[f"alpha_ev_vip{level}_bps"])
                    if pd.notna(latest_alpha.get(f"alpha_ev_vip{level}_bps"))
                    else None
                )
                for level in range(6)
            },
            "profiles": {
                f"VIP{level}": {
                    "decision": (
                        "TRADE"
                        if bool(latest_alpha.get(f"alpha_accepted_vip{level}", False))
                        else "FLAT"
                    ),
                    "target_bps": (
                        float(latest_alpha[f"alpha_target_vip{level}_bps"])
                        if pd.notna(latest_alpha.get(f"alpha_target_vip{level}_bps"))
                        else None
                    ),
                    "target_probability": (
                        float(latest_alpha[f"alpha_target_probability_vip{level}"])
                        if pd.notna(latest_alpha.get(f"alpha_target_probability_vip{level}"))
                        else None
                    ),
                    "expected_net_bps": (
                        float(latest_alpha[f"alpha_expected_net_vip{level}_bps"])
                        if pd.notna(latest_alpha.get(f"alpha_expected_net_vip{level}_bps"))
                        else None
                    ),
                }
                for level in range(6)
            },
        }
    payload = {
        "validation_status": (
            "RESEARCH_ALPHA_READY"
            if alpha_status == "RESEARCH_ALPHA_READY"
            else "RESEARCH_ONLY_NO_ECONOMIC_ALPHA"
            if alpha_status == "NO_ECONOMIC_ALPHA"
            else "COLLECTING_NO_POLICY"
        ),
        "live_orders_enabled": False,
        "protocol": btc_vwap_alpha.PROTOCOL,
        "protocol_hash": btc_vwap_alpha.PROTOCOL_HASH,
        "decision_spec": {
            "families": list(btc_vwap_alpha.FAMILIES),
            "cadence_minutes": 5,
            "alpha_venue": "Binance BTCUSDT perpetual/spot",
            "execution_venue": "Bitunix BTCUSDT observed paper execution",
            "decision": "highest positive net EV after costs; otherwise NO_TRADE",
        },
        "policy_hash": btc_vwap_alpha.PROTOCOL_HASH,
        "anchor_protocol": ANCHOR_PROTOCOL,
        "anchor_protocol_hash": ANCHOR_PROTOCOL_HASH,
        "execution_protocol": EXECUTION_PROTOCOL,
        "execution_protocol_hash": EXECUTION_PROTOCOL_HASH,
        "dynamic_protocol": DYNAMIC_PROTOCOL,
        "dynamic_protocol_hash": DYNAMIC_PROTOCOL_HASH,
        "cost_interpretation": {
            "fee_venue": "Bitunix futures",
            "vip_level": BITUNIX_VIP_LEVEL,
            "maker_bps_per_side": MAKER_FEE_BPS_PER_SIDE,
            "taker_bps_per_side": TAKER_FEE_BPS_PER_SIDE,
            "slippage_reserve_bps_per_side": SLIPPAGE_RESERVE_BPS_PER_SIDE,
            "normal": "observed Bitunix 15-level depth + Bitunix taker fees + reserve + funding",
            "stress": "same execution with fees and reserve doubled",
            "funding": "Bitunix charge included only when the observed holding crosses funding",
            "official_fee_source": "https://www.bitunix.com/service/handling-fee",
            "available_profiles_bps": {
                f"VIP {level}": {"maker": maker, "taker": taker}
                for level, (maker, taker) in FUTURES_VIP_FEE_BPS.items()
            },
            "execution_limitation": (
                "public depth proves taker replay; maker fills require private observations"
            ),
        },
        "feature_rows_after_protocol_start": int(frame["minute"].ge(DYNAMIC_PROTOCOL_START).sum()),
        "anchored_vwap_rows_after_protocol_start": int(
            frame.loc[
                frame["minute"].ge(DYNAMIC_PROTOCOL_START),
                "anchored_vwap_distance_bps",
            ]
            .notna()
            .sum()
        ),
        "dynamic_feature_rows_after_execution_start": len(dynamic_frame),
        "rejection_funnel": {},
        "anchor_rejection_funnel": {},
        "dynamic_rejection_funnel": {
            family: rejection_funnel(dynamic_frame, family) for family in FAMILIES
        },
        "dynamic_anchor_rejection_funnel": {
            family: anchor_rejection_funnel(dynamic_frame, family)
            for family in ("ANCHOR_CONTINUATION", "ANCHOR_FAILURE")
        },
        "oracle": {},
        "results": results,
        "anchored_results": anchored_results,
        "dynamic_results": dynamic_results,
        "eligible": eligible,
        "anchored_eligible": anchored_eligible,
        "dynamic_eligible": dynamic_eligible,
        "counterfactual_rows": len(counterfactual),
        "counterfactual_path": str(COUNTERFACTUAL),
        "fee_profile_counterfactual_rows": len(fee_profiles),
        "fee_profile_counterfactual_path": str(FEE_PROFILE_COUNTERFACTUAL),
        "one_position_diagnostics": one_position,
        "selector": selector,
        "alpha": {
            "status": alpha_status,
            "report": str(btc_vwap_alpha.REPORT),
            "bundle": str(btc_vwap_alpha.BUNDLE),
            "operating_profile": "VIP0",
            "scenario_profiles": [f"VIP{level}" for level in range(6)],
            "scored_candidates": len(counterfactual),
            "accepted_candidates": int(
                counterfactual.get("alpha_accepted", pd.Series(dtype=bool)).sum()
            ),
            "accepted_candidates_by_profile": {
                profile: int(rows.get("alpha_accepted", pd.Series(dtype=bool)).sum())
                for profile, rows in profile_counterfactuals.items()
            },
            "latest_candidate": alpha_latest,
        },
        "current_market_assessment": assessment,
        "current_market_assessments": assessments,
        "market_chart": market_chart(
            live_alpha_l2,
            assessment,
            pd.concat([l2.tail(1_800), live_execution_l2], ignore_index=True),
        ),
        "data_coverage": {
            "alpha_venue": "Binance BTCUSDT perpetual",
            "binance_l2_rows": len(alpha_l2),
            "binance_l2_start": pd.Timestamp(alpha_l2["available_at"].min()).isoformat(),
            "binance_l2_end": pd.Timestamp(alpha_l2["available_at"].max()).isoformat(),
            "binance_l2_utc_days": int(
                pd.to_datetime(alpha_l2["available_at"], utc=True).dt.floor("D").nunique()
            ),
            "execution_venue": "Bitunix BTCUSDT futures shadow",
            "bitunix_l2_rows": len(l2),
            "bitunix_l2_start": pd.Timestamp(l2["available_at"].min()).isoformat(),
            "bitunix_l2_end": pd.Timestamp(l2["available_at"].max()).isoformat(),
            "bitunix_l2_utc_days": int(
                pd.to_datetime(l2["available_at"], utc=True).dt.floor("D").nunique()
            ),
            "counterfactual_rows": len(counterfactual),
            "historical_l2_years_available": False,
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _align_active_alpha_metadata(payload, alpha_status)
    _write_report(payload)
    return payload


def refresh_current_report() -> dict[str, Any]:
    with REPORT_LOCK:
        payload: dict[str, Any] = (
            json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
        )
    live_frame, live_alpha_l2, live_execution_l2 = _live_market_inputs()
    selector = {"probability_status": "NOT_TRAINED"}
    accounts = musca_v5_paper.load_accounts()
    assessed_at = pd.Timestamp.now(tz="UTC")
    base_candidates: dict[str, dict[str, Any] | None] = {}
    base_policy_ready = False
    if musca_v8_multi_horizon.REPORT.exists():
        with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
            base_policy_ready = (
                json.loads(
                    musca_v8_multi_horizon.REPORT.read_text(encoding="utf-8")
                ).get("verdict")
                == "RESEARCH_BASE_ALPHA_READY"
            )
    for profile in (f"VIP{value}" for value in range(6)):
        try:
            base_candidates[profile] = musca_v8_multi_horizon.live_candidate(
                profile, assessed_at
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            base_candidates[profile] = None
    assessments = {
        profile: current_market_assessment(
            live_frame,
            selector,
            live_execution_l2,
            equity=float(accounts.get(profile, {}).get("final_equity", PAPER_INITIAL_EQUITY)),
            vip_level=level,
            evaluated_at=assessed_at,
            base_candidate=base_candidates[profile],
            base_policy_ready=base_policy_ready,
        )
        for level, profile in enumerate(f"VIP{value}" for value in range(6))
    }
    accounts = musca_v5_paper.advance_accounts(assessments, live_execution_l2)
    with REPORT_LOCK:
        payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    alpha_status = "NOT_TRAINED"
    if btc_vwap_alpha.REPORT.exists():
        with contextlib.suppress(OSError, ValueError, TypeError, json.JSONDecodeError):
            alpha_status = str(
                json.loads(btc_vwap_alpha.REPORT.read_text(encoding="utf-8")).get(
                    "status", "REPORT_INVALID"
                )
            )
    _align_active_alpha_metadata(payload, alpha_status)
    diagnostics = payload.setdefault("one_position_diagnostics", {})
    diagnostics["paper_accounts"] = accounts
    diagnostics["paper_account"] = accounts["VIP0"]
    payload["current_market_assessment"] = assessments["VIP0"]
    payload["current_market_assessments"] = assessments
    payload["market_chart"] = market_chart(
        live_alpha_l2,
        assessments["VIP0"],
        live_execution_l2,
    )
    payload["updated_at"] = datetime.now(UTC).isoformat()
    _write_report(payload)
    return payload


def _live_refresh_loop() -> None:
    while True:
        with contextlib.suppress(OSError, ValueError, KeyError):
            refresh_current_report()
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only Binance Alpha / Bitunix VWAP audit")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the expensive historical audit once, then exit",
    )
    args = parser.parse_args()
    if args.once or args.rebuild:
        print(json.dumps(run(), indent=2))
        return
    if not REPORT.exists():
        run()
    _live_refresh_loop()


if __name__ == "__main__":
    main()
