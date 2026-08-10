from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.bitunix_fees import futures_fee_bps
from adaptive_bot.musca_v5_fine_tuning import _atomic_json

DATASET = Path("data/research/bitunix_l2/btcusdt_l2_features.parquet")
REPORT = Path("data/reports/musca_v5_post_only_touch_upper_bound.json")
DYNAMIC_REPORT = Path("data/reports/musca_v5_post_only_dynamic_vwap_upper_bound.json")
FILTER_REPORT = Path("data/reports/musca_v5_post_only_l2_filter_audit.json")
MAKER_EXIT_REPORT = Path("data/reports/musca_v5_post_only_maker_exit_upper_bound.json")
INVENTORY_HAZARD_REPORT = Path(
    "data/reports/musca_v5_post_only_inventory_hazard.json"
)
INVENTORY_HORIZONS_SECONDS = (30, 60, 120, 180, 300)
MODEL_FEATURES = (
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_15s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "trade_arrival_rate_30s",
    "range_60s_bps",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
    "rolling_vwap_5m_distance_bps",
    "rolling_vwap_slope_bps_60s",
    "rolling_vwap_slope_change_bps",
    "realized_volatility",
    "volatility_percentile",
    "spread_bps",
    "time_since_last_vwap_cross_seconds",
)
PROTOCOL = {
    "name": "musca_v5_post_only_touch_upper_bound_v1",
    "venue": "Bitunix",
    "symbol": "BTCUSDT",
    "decision_cadence_seconds": 30,
    "submission_latency_ms": 250,
    "quote_lifetime_seconds": 30,
    "position_lifetime_seconds": 300,
    "fair_value": "causal_rolling_trade_vwap_5m_at_decision",
    "quote_distance": "1.5_x_vip_maker_entry_plus_taker_exit_fee",
    "fill": "optimistic_executable_book_touch_without_queue_upper_bound",
    "exit": "first_executable_return_to_frozen_fair_or_100bps_stop_or_timeout",
    "one_position_per_vip": True,
    "costs": "Bitunix_VIP_actual_1x_with_2x_diagnostic",
    "funding": "omitted_upper_bound_positions_max_5m",
    "parameter_search": False,
    "paper_authority": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
DYNAMIC_PROTOCOL = {
    **PROTOCOL,
    "name": "musca_v5_post_only_dynamic_vwap_upper_bound_v1",
    "exit": (
        "first_executable_current_causal_vwap_or_vwap_crosses_entry_invalidation_"
        "or_100bps_catastrophic_stop_or_timeout"
    ),
    "center": "causal_rolling_trade_vwap_5m_updated_on_each_book",
}
DYNAMIC_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(DYNAMIC_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
FILTER_PROTOCOL = {
    **PROTOCOL,
    "name": "musca_v5_post_only_preregistered_l2_confluence_v1",
    "side_filter": (
        "at_least_3_of_4_directional_sign_votes_depth5bps_microprice_"
        "aggressive_flow5s_vwap_slope60s"
    ),
    "missing_feature_behavior": "fail_closed",
    "shock_behavior": "fail_closed",
    "split": "2026-08-03_to_05_discovery_2026-08-06_to_08_audit",
}
FILTER_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(FILTER_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
MAKER_EXIT_PROTOCOL = {
    **PROTOCOL,
    "name": "musca_v5_post_only_maker_entry_maker_target_upper_bound_v1",
    "quote_distance": "1.5_x_vip_maker_entry_plus_maker_target_fee",
    "target_execution": "optimistic_post_only_touch_at_frozen_fair",
    "risk_exit_execution": "taker_on_timeout_invalidation_or_catastrophic_stop",
    "private_fill_truth": False,
}
MAKER_EXIT_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(MAKER_EXIT_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
INVENTORY_HAZARD_PROTOCOL = {
    **MAKER_EXIT_PROTOCOL,
    "name": "musca_v5_vip5_post_only_inventory_hazard_v1",
    "profile": "VIP5_best_fee_case",
    "position_lifetime_seconds": list(INVENTORY_HORIZONS_SECONDS),
    "purpose": "diagnose_when_adverse_selection_appears_not_choose_a_horizon",
    "economic_gate": "n_100_ev_positive_pf_1_15_majority_positive_days",
    "selection_from_this_report": False,
}
INVENTORY_HAZARD_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(INVENTORY_HAZARD_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def load_frame(path: Path = DATASET) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
        "event_type",
        "available_at",
        "best_bid",
        "best_ask",
        "mid",
        "rolling_vwap",
        "depth_imbalance_5bps",
        "microprice_distance_bps",
        "aggressive_imbalance_5s",
        "rolling_vwap_slope_bps_60s",
        "market_regime",
        "feature_valid",
                *MODEL_FEATURES,
            ]
        )
    )
    frame = pd.read_parquet(path, columns=columns)
    frame["available_at"] = pd.to_datetime(frame["available_at"], format="mixed", utc=True)
    return frame.sort_values("available_at").reset_index(drop=True)


def simulate(
    frame: pd.DataFrame,
    vip_level: int,
    *,
    dynamic_center: bool = False,
    l2_filter: bool = False,
    maker_target: bool = False,
    position_lifetime_seconds: int = 300,
) -> dict[str, Any]:
    required = {
        "event_type",
        "available_at",
        "best_bid",
        "best_ask",
        "mid",
        "rolling_vwap",
        "depth_imbalance_5bps",
        "microprice_distance_bps",
        "aggressive_imbalance_5s",
        "rolling_vwap_slope_bps_60s",
        "market_regime",
        "feature_valid",
    }
    if missing := required - set(frame):
        raise ValueError(f"Bitunix upper-bound frame missing columns: {sorted(missing)}")
    data = frame.sort_values("available_at").copy()
    data["available_at"] = pd.to_datetime(
        data["available_at"], format="mixed", utc=True
    ).astype("datetime64[ns, UTC]")
    books = data.loc[
        data["event_type"].eq("book")
        & pd.to_numeric(data["best_bid"], errors="coerce").gt(0)
        & pd.to_numeric(data["best_ask"], errors="coerce").gt(0)
    ].copy()
    valid_decisions = books.loc[
        books["feature_valid"].fillna(False).astype(bool)
        & pd.to_numeric(books["rolling_vwap"], errors="coerce").gt(0)
    ].copy()
    valid_decisions["bucket"] = valid_decisions["available_at"].dt.floor("30s")
    decisions = valid_decisions.drop_duplicates("bucket", keep="first")
    decision_ns = decisions["available_at"].astype("int64").to_numpy()
    calendar_days = max(1, int(decisions["available_at"].dt.floor("D").nunique()))
    fair_values = pd.to_numeric(decisions["rolling_vwap"]).to_numpy(float)
    votes = decisions[
        [
            "depth_imbalance_5bps",
            "microprice_distance_bps",
            "aggressive_imbalance_5s",
            "rolling_vwap_slope_bps_60s",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    regimes = decisions["market_regime"].astype(str).str.upper().to_numpy()
    model_features = {
        name: (
            pd.to_numeric(decisions[name], errors="coerce").to_numpy(float)
            if name in decisions
            else np.full(len(decisions), np.nan)
        )
        for name in MODEL_FEATURES
    }
    book_ns = books["available_at"].astype("int64").to_numpy()
    bids = pd.to_numeric(books["best_bid"]).to_numpy(float)
    asks = pd.to_numeric(books["best_ask"]).to_numpy(float)
    mids = pd.to_numeric(books["mid"]).to_numpy(float)
    current_vwaps = pd.to_numeric(books["rolling_vwap"], errors="coerce").to_numpy(float)
    maker, taker = futures_fee_bps(vip_level)
    risk_exit_cost = maker + taker
    target_cost = 2 * maker if maker_target else risk_exit_cost
    distance = 1.5 * target_cost
    latency_ns = 250_000_000
    quote_life_ns = 30_000_000_000
    if position_lifetime_seconds <= 0:
        raise ValueError("position_lifetime_seconds must be positive")
    position_life_ns = position_lifetime_seconds * 1_000_000_000
    decision_index = pair_cycles = touches = 0
    closed: list[dict[str, Any]] = []
    while decision_index < len(decision_ns):
        decision_at = decision_ns[decision_index]
        fair = fair_values[decision_index]
        active_at = decision_at + latency_ns
        expires_at = active_at + quote_life_ns
        bid_quote = fair * (1 - distance / 10_000)
        ask_quote = fair * (1 + distance / 10_000)
        allow_long = allow_short = True
        if l2_filter:
            available = np.isfinite(votes[decision_index]).all()
            allow_long = bool(
                available
                and regimes[decision_index] != "SHOCK"
                and (votes[decision_index] > 0).sum() >= 3
            )
            allow_short = bool(
                available
                and regimes[decision_index] != "SHOCK"
                and (votes[decision_index] < 0).sum() >= 3
            )
            if not allow_long and not allow_short:
                decision_index += 1
                continue
        pair_cycles += 1
        touch_index = int(np.searchsorted(book_ns, active_at, side="left"))
        fill_at: int | None = None
        side = ""
        while touch_index < len(book_ns) and book_ns[touch_index] <= expires_at:
            if allow_long and asks[touch_index] <= bid_quote:
                fill_at, side = book_ns[touch_index], "long"
                break
            if allow_short and bids[touch_index] >= ask_quote:
                fill_at, side = book_ns[touch_index], "short"
                break
            touch_index += 1
        if fill_at is None:
            decision_index = int(np.searchsorted(decision_ns, expires_at, side="right"))
            continue
        touches += 1
        entry_at = fill_at
        entry = bid_quote if side == "long" else ask_quote
        direction = 1.0 if side == "long" else -1.0
        deadline = entry_at + position_life_ns
        book_index = int(np.searchsorted(book_ns, entry_at, side="left"))
        mfe = mae = 0.0
        exit_at: int | None = None
        exit_price = reason = None
        while book_index < len(book_ns):
            now = book_ns[book_index]
            excursion = direction * (mids[book_index] - entry) / entry * 10_000
            mfe, mae = max(mfe, excursion), min(mae, excursion)
            stopped = excursion <= -100
            center = current_vwaps[book_index] if dynamic_center else fair
            if not np.isfinite(center) or center <= 0:
                book_index += 1
                continue
            targeted = bids[book_index] >= center if side == "long" else asks[book_index] <= center
            invalidated = (
                center <= entry if side == "long" else center >= entry
            ) and not targeted
            timed_out = now >= deadline
            if stopped or targeted or invalidated or timed_out:
                exit_at = now
                exit_price = bids[book_index] if side == "long" else asks[book_index]
                reason = (
                    "STOP"
                    if stopped
                    else "VWAP_TARGET"
                    if targeted
                    else "VWAP_EDGE_INVALIDATED"
                    if invalidated
                    else "TIMEOUT"
                )
                break
            book_index += 1
        if exit_at is None or exit_price is None:
            break
        if maker_target and reason == "VWAP_TARGET":
            exit_price = center
        gross = direction * (exit_price - entry) / entry * 10_000
        realized_cost = target_cost if reason == "VWAP_TARGET" else risk_exit_cost
        net = gross - realized_cost
        closed.append(
            {
                "entered_at": pd.Timestamp(entry_at, tz="UTC").isoformat(),
                "exited_at": pd.Timestamp(exit_at, tz="UTC").isoformat(),
                "decision_at": pd.Timestamp(decision_at, tz="UTC").isoformat(),
                "time_to_touch_seconds": (entry_at - decision_at) / 1_000_000_000,
                "side": side,
                "gross_bps": gross,
                "net_bps": net,
                "net_2x_cost_bps": gross - 2 * realized_cost,
                "exit_role": "maker" if maker_target and reason == "VWAP_TARGET" else "taker",
                "mfe_bps": mfe,
                "mae_bps": mae,
                "reason": reason,
                **{
                    f"decision_{name}": float(values[decision_index])
                    if np.isfinite(values[decision_index])
                    else None
                    for name, values in model_features.items()
                },
            }
        )
        decision_index = int(np.searchsorted(decision_ns, exit_at, side="right"))
    return _metrics(vip_level, pair_cycles, touches, closed, calendar_days)


def run(path: Path = DATASET, report_path: Path = REPORT) -> dict[str, Any]:
    frame = load_frame(path)
    profiles = {f"VIP{level}": simulate(frame, level) for level in range(6)}
    timestamps = frame["available_at"]
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "dataset": str(path),
        "rows": len(frame),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "observed_days": int(timestamps.dt.floor("D").nunique()),
        "profiles": profiles,
        "verdict": "DIAGNOSTIC_UPPER_BOUND_ONLY",
        "paper_change_authorized": False,
        "holdout_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def run_dynamic(
    path: Path = DATASET, report_path: Path = DYNAMIC_REPORT
) -> dict[str, Any]:
    frame = load_frame(path)
    profiles = {
        f"VIP{level}": simulate(frame, level, dynamic_center=True) for level in range(6)
    }
    timestamps = frame["available_at"]
    report = {
        "protocol": DYNAMIC_PROTOCOL,
        "protocol_hash": DYNAMIC_PROTOCOL_HASH,
        "dataset": str(path),
        "rows": len(frame),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "observed_days": int(timestamps.dt.floor("D").nunique()),
        "profiles": profiles,
        "verdict": "DYNAMIC_CENTER_DISCOVERY_UPPER_BOUND_ONLY",
        "paper_change_authorized": False,
        "holdout_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def run_l2_filter(
    path: Path = DATASET, report_path: Path = FILTER_REPORT
) -> dict[str, Any]:
    frame = load_frame(path)
    cutoff = pd.Timestamp("2026-08-06T00:00:00Z")
    periods = {
        "discovery": frame.loc[frame["available_at"].lt(cutoff)],
        "audit": frame.loc[frame["available_at"].ge(cutoff)],
    }
    results = {
        name: {
            f"VIP{level}": simulate(period, level, l2_filter=True) for level in range(6)
        }
        for name, period in periods.items()
    }
    gates = {
        f"VIP{level}": all(
            (metrics := results[period][f"VIP{level}"])["closed_trades"] >= 20
            and (metrics["expectancy_net_bps"] or -1) > 0
            and (metrics["profit_factor"] or 0) >= 1.15
            for period in periods
        )
        for level in range(6)
    }
    report = {
        "protocol": FILTER_PROTOCOL,
        "protocol_hash": FILTER_PROTOCOL_HASH,
        "dataset": str(path),
        "periods": results,
        "gates": gates,
        "verdict": "L2_FILTER_DISCOVERY_ONLY" if any(gates.values()) else "NO_L2_FILTER_EDGE",
        "paper_change_authorized": False,
        "holdout_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def run_maker_exit_upper_bound(
    path: Path = DATASET, report_path: Path = MAKER_EXIT_REPORT
) -> dict[str, Any]:
    frame = load_frame(path)
    profiles = {
        f"VIP{level}": simulate(frame, level, maker_target=True) for level in range(6)
    }
    report = {
        "protocol": MAKER_EXIT_PROTOCOL,
        "protocol_hash": MAKER_EXIT_PROTOCOL_HASH,
        "dataset": str(path),
        "rows": len(frame),
        "start": frame["available_at"].min().isoformat(),
        "end": frame["available_at"].max().isoformat(),
        "observed_days": int(frame["available_at"].dt.floor("D").nunique()),
        "profiles": profiles,
        "verdict": "MAKER_EXIT_UPPER_BOUND_ONLY",
        "paper_change_authorized": False,
        "private_fill_truth": False,
        "holdout_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def run_inventory_hazard(
    path: Path = DATASET, report_path: Path = INVENTORY_HAZARD_REPORT
) -> dict[str, Any]:
    frame = load_frame(path)
    horizons: dict[str, Any] = {}
    for seconds in INVENTORY_HORIZONS_SECONDS:
        metrics = simulate(
            frame,
            5,
            maker_target=True,
            position_lifetime_seconds=seconds,
        )
        trades = metrics.pop("trades")
        metrics["reasons"] = {
            reason: sum(trade["reason"] == reason for trade in trades)
            for reason in ("VWAP_TARGET", "TIMEOUT", "STOP")
        }
        horizons[str(seconds)] = metrics
    gates = {
        seconds: bool(
            metrics["closed_trades"] >= 100
            and (metrics["expectancy_net_bps"] or -1) > 0
            and (metrics["profit_factor"] or 0) >= 1.15
            and (metrics["positive_trade_days_fraction"] or 0) > 0.5
        )
        for seconds, metrics in horizons.items()
    }
    report = {
        "protocol": INVENTORY_HAZARD_PROTOCOL,
        "protocol_hash": INVENTORY_HAZARD_PROTOCOL_HASH,
        "source_protocol_hash": MAKER_EXIT_PROTOCOL_HASH,
        "dataset": str(path),
        "rows": len(frame),
        "observed_days": int(frame["available_at"].dt.floor("D").nunique()),
        "horizons": horizons,
        "gates": gates,
        "verdict": (
            "INVENTORY_HORIZON_UPPER_BOUND_EXISTS"
            if any(gates.values())
            else "NO_INVENTORY_HORIZON_UPPER_BOUND"
        ),
        "paper_change_authorized": False,
        "holdout_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def _metrics(
    vip_level: int,
    pair_cycles: int,
    touches: int,
    trades: list[dict[str, Any]],
    calendar_days: int,
) -> dict[str, Any]:
    net = [float(trade["net_bps"]) for trade in trades]
    stress = [float(trade["net_2x_cost_bps"]) for trade in trades]
    gains = sum(value for value in net if value > 0)
    losses = -sum(value for value in net if value < 0)
    days = {str(trade["entered_at"])[:10] for trade in trades}
    daily = {
        day: sum(
            float(trade["net_bps"])
            for trade in trades
            if str(trade["entered_at"])[:10] == day
        )
        for day in days
    }
    return {
        "vip_level": vip_level,
        "pair_cycles": pair_cycles,
        "touch_fills_upper_bound": touches,
        "closed_trades": len(trades),
        "trades_per_calendar_day": len(trades) / calendar_days,
        "expectancy_net_bps": sum(net) / len(net) if net else None,
        "expectancy_2x_cost_bps": sum(stress) / len(stress) if stress else None,
        "profit_factor": gains / losses if losses else None,
        "positive_fraction": sum(value > 0 for value in net) / len(net) if net else None,
        "positive_trade_days_fraction": (
            sum(value > 0 for value in daily.values()) / len(daily) if daily else None
        ),
        "target_fraction": (
            sum(trade["reason"] == "VWAP_TARGET" for trade in trades) / len(trades)
            if trades
            else None
        ),
        "trades": trades,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
