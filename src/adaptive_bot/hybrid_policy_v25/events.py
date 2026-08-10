from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.hybrid_policy_v22.path_audit import atomic_parquet
from adaptive_bot.hybrid_policy_v25.protocol import (
    BASE_COST_BPS,
    ROOT,
    STRESS_COST_BPS,
    atomic_json,
    status,
)

EVENT_PATH = ROOT / "events.parquet"
REJECTION_PATH = ROOT / "event_rejections.parquet"


def _score(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    strength = (features["adx_1h"] / 25).clip(0, 1)
    volatility = (1 - (features["volatility_percentile"] - 80).clip(0, 20) / 20).clip(0, 1)
    basis = (1 - features["basis_change_bps"].abs() / 20).clip(0, 1)
    funding = (1 - features["funding_z"].abs() / 4).clip(0, 1)
    common = [strength, volatility, basis, funding]
    long_components = [
        features["perp_close_1h"].gt(features["perp_weekly_vwap_1h"]),
        features["spot_close_1h"].gt(features["spot_weekly_vwap_1h"]),
        features["perp_close_1h"].gt(features["ema50_1h"]),
        features["ema_slope_atr"].gt(0),
        features["weekly_vwap_slope_atr"].gt(0),
        features["trend_4h"].gt(0),
        *common,
    ]
    short_components = [
        features["perp_close_1h"].lt(features["perp_weekly_vwap_1h"]),
        features["spot_close_1h"].lt(features["spot_weekly_vwap_1h"]),
        features["perp_close_1h"].lt(features["ema50_1h"]),
        features["ema_slope_atr"].lt(0),
        features["weekly_vwap_slope_atr"].lt(0),
        features["trend_4h"].lt(0),
        *common,
    ]
    long_score = np.mean(
        np.vstack([np.asarray(value, dtype=float) for value in long_components]), axis=0
    )
    short_score = np.mean(
        np.vstack([np.asarray(value, dtype=float) for value in short_components]), axis=0
    )
    return long_score, short_score


def _minute_vwap(frame: pd.DataFrame) -> float:
    volume = float(frame["perp_volume"].sum())
    return float(frame["perp_quote_volume"].sum() / volume) if volume > 0 else float("nan")


def _event_from_impulse(
    asset: str,
    feature_index: int,
    row: Any,
    direction: int,
    regime_score: float,
    opposite_score: float,
    features: pd.DataFrame,
    bars5: pd.DataFrame,
    minutes: pd.DataFrame,
    five_times: pd.DatetimeIndex,
    minute_times: pd.DatetimeIndex,
    protocol_hash: str,
) -> tuple[dict[str, Any] | None, str]:
    impulse_timestamp = pd.Timestamp(cast(Any, row.timestamp))
    impulse_available = pd.Timestamp(cast(Any, row.available_at))
    start5 = int(five_times.searchsorted(impulse_available))
    end5 = int(five_times.searchsorted(impulse_available + pd.Timedelta(minutes=120)))
    window = bars5.iloc[start5:end5]
    if len(window) < 3 or not window["is_available"].all():
        return None, "DATA_UNAVAILABLE"
    pullback_start_index: int | None = None
    confirmation_index: int | None = None
    for index in range(1, len(window)):
        previous = window.iloc[index - 1]
        current = window.iloc[index]
        against = direction * (float(current["perp_close"]) - float(previous["perp_close"])) < 0
        if pullback_start_index is None:
            if against:
                pullback_start_index = index
            continue
        confirms = (
            float(current["perp_close"]) > float(previous["perp_high"])
            if direction > 0
            else float(current["perp_close"]) < float(previous["perp_low"])
        )
        if confirms:
            confirmation_index = index
            break
    if pullback_start_index is None or confirmation_index is None:
        return None, "NO_PULLBACK_CONFIRMATION"
    pullback = window.iloc[pullback_start_index : confirmation_index + 1]
    confirmation = window.iloc[confirmation_index]
    confirmation_timestamp = pd.Timestamp(confirmation["timestamp"]) + pd.Timedelta(minutes=5)
    entry_index = int(minute_times.searchsorted(confirmation_timestamp))
    if entry_index + 240 > len(minutes):
        return None, "NO_EXECUTABLE_PRICE"
    entry_timestamp = pd.Timestamp(minutes.iloc[entry_index]["timestamp"])
    if entry_timestamp < confirmation_timestamp:
        raise RuntimeError("V25 entry precedes confirmation")
    entry = float(minutes.iloc[entry_index]["perp_open"])
    atr_value = float(row.atr_15m)
    pullback_extreme = (
        float(pullback["perp_low"].min()) if direction > 0 else float(pullback["perp_high"].max())
    )
    buffer = 0.10 * atr_value
    stop = pullback_extreme - direction * buffer
    risk_price = direction * (entry - stop)
    stop_bps = risk_price / entry * 10_000
    stop_atr = risk_price / atr_value
    cost_ratio = BASE_COST_BPS / stop_bps if stop_bps > 0 else float("inf")
    if stop_bps < 12 or stop_atr > 2.5 or cost_ratio > 0.33:
        return None, "INVALID_ECONOMIC_STOP"
    impulse_start = int(minute_times.searchsorted(impulse_timestamp))
    pullback_end = int(minute_times.searchsorted(confirmation_timestamp))
    impulse_minutes = minutes.iloc[impulse_start:pullback_end]
    if impulse_minutes.empty or not impulse_minutes["is_available"].all():
        return None, "DATA_UNAVAILABLE"
    anchored_vwap = _minute_vwap(impulse_minutes)
    history_start = max(0, feature_index - 20)
    history = features.iloc[history_start : feature_index + 1]
    swing_index = history["perp_low"].idxmin() if direction > 0 else history["perp_high"].idxmax()
    swing_row = cast(Any, history.loc[swing_index])
    swing_start = int(minute_times.searchsorted(pd.Timestamp(swing_row["timestamp"])))
    swing_vwap = _minute_vwap(minutes.iloc[swing_start:pullback_end])
    pullback_volume = float(pullback["perp_quote_volume"].sum())
    impulse_volume = float(row.perp_quote_volume)
    pullback_flow = float(
        (2 * pullback["perp_taker_buy_quote"].sum() - pullback_volume) / pullback_volume
    )
    impulse_move = abs(
        float(row.perp_close) - float(features.iloc[max(0, feature_index - 4)]["perp_close"])
    )
    pullback_depth = direction * (float(row.perp_close) - pullback_extreme)
    vwap_distances = np.column_stack(
        [
            (pullback["perp_close"] - pullback["perp_daily_vwap"]).abs().to_numpy(float),
            np.abs(pullback["perp_close"].to_numpy(float) - anchored_vwap),
        ]
    )
    touches = int((vwap_distances.min(axis=1) <= 0.25 * atr_value).sum())
    crossed = bool((direction * (pullback["perp_close"] - pullback["perp_daily_vwap"]) < 0).any())
    returns = pullback["perp_close"].diff().dropna().to_numpy(float)
    pullback_efficiency = abs(
        float(pullback["perp_close"].iloc[-1] - pullback["perp_close"].iloc[0])
    ) / max(float(np.abs(returns).sum()), 1e-12)
    flow_series = direction * pullback["perp_taker_imbalance"].to_numpy(float)
    flow_persistence = float((flow_series > 0).mean())
    spot_move = direction * (
        float(confirmation["spot_close"]) - float(pullback.iloc[0]["spot_open"])
    )
    perp_move = direction * (
        float(confirmation["perp_close"]) - float(pullback.iloc[0]["perp_open"])
    )
    flow_gate_pass = direction * float(confirmation["perp_taker_imbalance"]) > 0
    hour = entry_timestamp.hour + entry_timestamp.minute / 60
    weekday = entry_timestamp.dayofweek
    pullback_timestamp = pd.Timestamp(pullback.iloc[0]["timestamp"])
    event_key = (
        f"{asset}|{direction}|{impulse_timestamp.isoformat()}|{pullback_timestamp.isoformat()}"
    )
    return (
        {
            "event_id": hashlib.sha256(event_key.encode()).hexdigest()[:24],
            "event_cluster_id": entry_timestamp.floor("4h").isoformat(),
            "asset": asset,
            "asset_code": 0 if asset == "BTCUSDT" else 1,
            "contract_type": "perpetual",
            "tick_size": 0.1 if asset == "BTCUSDT" else 0.01,
            "volatility_scale": atr_value / entry,
            "liquidity_scale": float(row.perp_quote_volume),
            "direction": direction,
            "regime_start_timestamp": pd.Timestamp(
                features.iloc[max(0, feature_index - 1)]["available_at"]
            ),
            "impulse_timestamp": impulse_timestamp,
            "pullback_start_timestamp": pd.Timestamp(pullback.iloc[0]["timestamp"]),
            "confirmation_timestamp": confirmation_timestamp,
            "entry_timestamp": entry_timestamp,
            "entry_index": entry_index,
            "entry_price": entry,
            "stop_price": stop,
            "pullback_extreme_price": pullback_extreme,
            "anchored_vwap_price": anchored_vwap,
            "tp1_price": entry + direction * 1.5 * risk_price,
            "max_horizon_timestamp": entry_timestamp + pd.Timedelta(minutes=240),
            "feature_available_at": confirmation_timestamp,
            "source_timestamp": pd.Timestamp(confirmation["source_timestamp"]),
            "available_at": confirmation_timestamp,
            "max_input_available_at": confirmation_timestamp,
            "freshness_seconds": 0.0,
            "is_available": True,
            "dataset_version": 1,
            "protocol_hash": protocol_hash,
            "regime_score": regime_score,
            "regime_score_opposite": opposite_score,
            "adx_1h": float(row.adx_1h),
            "ema_slope_atr": direction * float(row.ema_slope_atr),
            "weekly_vwap_slope_atr": direction * float(row.weekly_vwap_slope_atr),
            "volatility_percentile": float(row.volatility_percentile),
            "funding_z": direction * float(row.funding_z),
            "basis_bps": direction * float(row.basis_bps),
            "basis_change_bps": direction * float(row.basis_change_bps),
            "return_15m_atr": direction * float(row.return_15m) * float(row.perp_close) / atr_value,
            "return_30m_atr": direction * float(row.return_30m) * float(row.perp_close) / atr_value,
            "return_60m_atr": direction * float(row.return_60m) * float(row.perp_close) / atr_value,
            "donchian_break_atr": direction
            * (
                float(row.perp_close)
                - (float(row.donchian_high) if direction > 0 else float(row.donchian_low))
            )
            / atr_value,
            "impulse_range_atr": float(row.range_atr),
            "relative_volume": float(row.relative_volume),
            "trade_intensity": float(row.trade_intensity),
            "taker_imbalance_15m": direction * float(row.perp_taker_imbalance),
            "spot_taker_imbalance_15m": direction * float(row.spot_taker_imbalance),
            "spot_perp_return_divergence": direction
            * (
                float(row.spot_close)
                / float(features.iloc[max(0, feature_index - 1)]["spot_close"])
                - float(row.perp_close)
                / float(features.iloc[max(0, feature_index - 1)]["perp_close"])
            ),
            "impulse_efficiency": float(row.impulse_efficiency),
            "pullback_depth_atr": pullback_depth / atr_value,
            "pullback_depth_pct_impulse": pullback_depth / max(impulse_move, 1e-12),
            "pullback_duration_minutes": float(
                (
                    confirmation_timestamp - pd.Timestamp(pullback.iloc[0]["timestamp"])
                ).total_seconds()
                / 60
            ),
            "pullback_volume_ratio": pullback_volume / max(impulse_volume, 1e-12),
            "pullback_taker_ratio": direction * pullback_flow,
            "distance_daily_vwap_atr": direction
            * (entry - float(confirmation["perp_daily_vwap"]))
            / atr_value,
            "distance_weekly_vwap_atr": direction
            * (entry - float(row.perp_weekly_vwap))
            / atr_value,
            "distance_anchored_vwap_atr": direction * (entry - anchored_vwap) / atr_value,
            "distance_swing_vwap_atr": direction * (entry - swing_vwap) / atr_value,
            "vwap_zone_touched": touches > 0,
            "vwap_zone_crossed": crossed,
            "number_of_touches": touches,
            "pullback_efficiency": pullback_efficiency,
            "flow_persistence": flow_persistence,
            "flow_price_divergence": float(
                np.sign(perp_move) != np.sign(direction * pullback_flow)
            ),
            "spot_confirmation": float(spot_move > 0),
            "flow_gate_pass": flow_gate_pass,
            "oi_change_1h": float(row.oi_change_1h) if pd.notna(row.oi_change_1h) else np.nan,
            "oi_is_available": bool(row.context_coverage)
            if pd.notna(row.context_coverage)
            else False,
            "stop_distance_bps": stop_bps,
            "stop_distance_atr": stop_atr,
            "cost_to_stop_ratio": cost_ratio,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
            "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        },
        "",
    )


def build_events(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    protocol_hash: str,
    resume: bool,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resume and EVENT_PATH.exists() and REJECTION_PATH.exists() and not smoke:
        cached = pd.read_parquet(EVENT_PATH)
        if {"anchored_vwap_price", "pullback_extreme_price"}.issubset(cached.columns):
            return cached, pd.read_parquet(REJECTION_PATH)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for asset_number, (asset, (minutes, features, bars5)) in enumerate(datasets.items(), start=1):
        long_score, short_score = _score(features)
        direction = np.where(long_score >= short_score, 1, -1)
        score = np.maximum(long_score, short_score)
        impulse = direction * features["return_60m"].to_numpy(float)
        eligible = np.flatnonzero(
            features["is_available"].to_numpy(bool)
            & (score >= 0.60)
            & (impulse > 0)
            & features["impulse_efficiency"].ge(0.20).to_numpy(bool)
        )
        if smoke and len(eligible) > 750:
            eligible = eligible[np.linspace(0, len(eligible) - 1, 750, dtype=int)]
        rows = list(features.itertuples(index=False))
        five_times = pd.DatetimeIndex(pd.to_datetime(bars5["timestamp"], utc=True))
        minute_times = pd.DatetimeIndex(pd.to_datetime(minutes["timestamp"], utc=True))
        blocked_until = pd.Timestamp("1900", tz="UTC")
        for position, index in enumerate(eligible, start=1):
            row = rows[int(index)]
            signal = pd.Timestamp(cast(Any, row.available_at))
            if signal < blocked_until:
                continue
            side = int(direction[int(index)])
            event, reason = _event_from_impulse(
                asset,
                int(index),
                row,
                side,
                float(score[int(index)]),
                float(min(long_score[int(index)], short_score[int(index)])),
                features,
                bars5,
                minutes,
                five_times,
                minute_times,
                protocol_hash,
            )
            if event is None:
                rejected.append(
                    {
                        "asset": asset,
                        "impulse_timestamp": signal,
                        "direction": side,
                        "reason": reason,
                    }
                )
                continue
            accepted.append(event)
            blocked_until = pd.Timestamp(event["max_horizon_timestamp"])
            if position % 500 == 0:
                status(
                    "events",
                    f"{asset} impulse {position}/{len(eligible)}",
                    45 + 15 * (asset_number - 1 + position / max(len(eligible), 1)) / 2,
                )
    events = pd.DataFrame(accepted).drop_duplicates("event_id").sort_values("entry_timestamp")
    rejections = pd.DataFrame(rejected)
    if not smoke:
        atomic_parquet(EVENT_PATH, events)
        atomic_parquet(REJECTION_PATH, rejections)
        atomic_json(
            Path("data/reports/ml_hybrid_v25_candidate_audit.json"),
            {
                "verdict": "EVENT_DATASET_READY" if len(events) >= 2_000 else "INSUFFICIENT_DATA",
                "independent_events": len(events),
                "rows_simulated": len(events),
                "by_asset": events["asset"].value_counts().to_dict(),
                "by_direction": events["direction"].value_counts().to_dict(),
                "rejections": rejections["reason"].value_counts().to_dict(),
                "duplicate_event_ids": int(events["event_id"].duplicated().sum()),
                "stop_distance_bps": events["stop_distance_bps"].describe().to_dict(),
                "cost_to_stop_ratio": events["cost_to_stop_ratio"].describe().to_dict(),
                "pullback_duration_minutes": events["pullback_duration_minutes"]
                .describe()
                .to_dict(),
                "coverage": float(events["is_available"].mean()),
            },
        )
    return events.reset_index(drop=True), rejections.reset_index(drop=True)


def _simulate(event: Any, minutes: pd.DataFrame, *, tp1_r: float) -> dict[str, Any] | None:
    start = int(event.entry_index)
    window = minutes.iloc[start : start + 240]
    if len(window) < 30 or not window["is_available"].all():
        return None
    direction = int(event.direction)
    entry, stop = float(event.entry_price), float(event.stop_price)
    risk = direction * (entry - stop)
    tp1 = entry + direction * tp1_r * risk
    remaining, gross_fraction = 1.0, 0.0
    tp1_done = False
    exit_price = float(window.iloc[-1]["perp_close"])
    exit_reason = "timeout"
    exit_timestamp = pd.Timestamp(window.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=1)
    mfe_bps = mae_bps = 0.0
    time_to_mfe = time_to_mae = 0
    time_to_tp1: float | None = None
    time_to_stop: float | None = None
    funding_bps = 0.0
    same_bar_ambiguous = False
    adverse_flow_bars = 0
    pending_flow_exit = False
    for offset, raw_bar in enumerate(window.itertuples(index=False)):
        bar = cast(Any, raw_bar)
        timestamp = pd.Timestamp(bar.timestamp)
        open_price, high, low = float(bar.perp_open), float(bar.perp_high), float(bar.perp_low)
        if pending_flow_exit:
            gross_fraction += remaining * direction * (open_price - entry) / entry
            exit_price, exit_reason, exit_timestamp = open_price, "flow_invalidation", timestamp
            remaining = 0.0
            break
        favorable = direction * ((high if direction > 0 else low) - entry) / entry * 10_000
        adverse = -direction * ((low if direction > 0 else high) - entry) / entry * 10_000
        if favorable > mfe_bps:
            mfe_bps, time_to_mfe = favorable, offset
        if adverse > mae_bps:
            mae_bps, time_to_mae = adverse, offset
        funding_bps += direction * float(bar.perp_funding_event_rate) * 10_000
        gap_stop = open_price <= stop if direction > 0 else open_price >= stop
        stop_hit = low <= stop if direction > 0 else high >= stop
        tp1_hit = high >= tp1 if direction > 0 else low <= tp1
        same_bar_ambiguous |= bool(stop_hit and tp1_hit)
        if gap_stop or stop_hit:
            fill = open_price if gap_stop else stop
            gross_fraction += remaining * direction * (fill - entry) / entry
            exit_price, exit_reason, exit_timestamp = (
                fill,
                "stop_gap" if gap_stop else "stop",
                timestamp + (pd.Timedelta(0) if gap_stop else pd.Timedelta(minutes=1)),
            )
            time_to_stop = float(offset)
            remaining = 0.0
            break
        if not tp1_done and tp1_hit:
            gross_fraction += 0.5 * direction * (tp1 - entry) / entry
            remaining = 0.5
            tp1_done = True
            time_to_tp1 = float(offset)
            stop = entry
        if tp1_done and (timestamp.minute + 1) % 15 == 0:
            history = window.iloc[max(0, offset - 44) : offset + 1]
            proposed = (
                float(history["perp_low"].min())
                if direction > 0
                else float(history["perp_high"].max())
            )
            stop = max(stop, proposed) if direction > 0 else min(stop, proposed)
        if (offset + 1) % 5 == 0:
            recent = window.iloc[offset - 4 : offset + 1]
            quote_volume = float(recent["perp_quote_volume"].sum())
            imbalance = (
                (2 * float(recent["perp_taker_buy_quote"].sum()) - quote_volume) / quote_volume
                if quote_volume > 0
                else 0.0
            )
            adverse_flow_bars = adverse_flow_bars + 1 if direction * imbalance < 0 else 0
            close_price = float(bar.perp_close)
            structure_lost = (
                close_price < float(event.pullback_extreme_price)
                if direction > 0
                else close_price > float(event.pullback_extreme_price)
            )
            vwap_lost = direction * (close_price - float(event.anchored_vwap_price)) < 0
            pending_flow_exit = adverse_flow_bars >= 3 and (structure_lost or vwap_lost)
    if remaining:
        gross_fraction += remaining * direction * (exit_price - entry) / entry
    gross_bps = gross_fraction * 10_000
    return {
        "tp1_r": tp1_r,
        "tp1_price_outcome": tp1,
        "gross_return_bps": gross_bps,
        "fee_bps": np.nan,
        "spread_bps": np.nan,
        "slippage_bps": np.nan,
        "funding_bps": funding_bps,
        "composite_cost_bps": BASE_COST_BPS,
        "net_return_bps_4": gross_bps - BASE_COST_BPS - funding_bps,
        "net_return_bps_8": gross_bps - STRESS_COST_BPS - funding_bps,
        "gross_return_r": gross_fraction * entry / risk,
        "net_return_r": (gross_bps - BASE_COST_BPS - funding_bps) / (risk / entry * 10_000),
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "time_to_mfe": time_to_mfe,
        "time_to_mae": time_to_mae,
        "time_to_tp1": time_to_tp1,
        "time_to_stop": time_to_stop,
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "is_censored": exit_reason == "timeout",
        "stop_and_target_same_bar": same_bar_ambiguous,
    }


def build_outcomes(
    events: pd.DataFrame,
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    resume: bool,
    smoke: bool,
) -> pd.DataFrame:
    output_path = ROOT / "events_with_outcomes.parquet"
    if resume and output_path.exists() and not smoke:
        cached = pd.read_parquet(output_path)
        if "stop_and_target_same_bar" in cached:
            return cached
    output: list[dict[str, Any]] = []
    for number, raw_event in enumerate(events.itertuples(index=False), start=1):
        event = cast(Any, raw_event)
        minutes = datasets[str(event.asset)][0]
        base = _simulate(event, minutes, tp1_r=1.5)
        variant = _simulate(event, minutes, tp1_r=1.0)
        if base is None or variant is None:
            continue
        row = event._asdict() | base
        row |= {f"flow_variant_{key}": value for key, value in variant.items()}
        row["y_positive_4bps"] = int(row["net_return_bps_4"] > 0)
        row["y_positive_8bps"] = int(row["net_return_bps_8"] > 0)
        output.append(row)
        if number % 250 == 0:
            status(
                "outcomes", f"Event {number}/{len(events)}", 62 + 12 * number / max(len(events), 1)
            )
    outcomes = pd.DataFrame(output).sort_values("entry_timestamp").reset_index(drop=True)
    if not smoke:
        atomic_parquet(output_path, outcomes)
        atomic_json(
            Path("data/reports/ml_hybrid_v25_outcome_audit.json"),
            {
                "verdict": "OUTCOMES_READY" if len(outcomes) == len(events) else "DATA_FAILURE",
                "events": len(events),
                "outcomes": len(outcomes),
                "cost_monotonic_violations": int(
                    outcomes["net_return_bps_8"].gt(outcomes["net_return_bps_4"]).sum()
                ),
                "future_exit_violations": int(
                    pd.to_datetime(outcomes["exit_timestamp"], utc=True)
                    .lt(pd.to_datetime(outcomes["entry_timestamp"], utc=True))
                    .sum()
                ),
                "censored": int(outcomes["is_censored"].sum()),
                "same_bar_worst_case": int(outcomes["stop_and_target_same_bar"].sum()),
                "exit_reasons": outcomes["exit_reason"].value_counts().to_dict(),
                "net_return_bps_4": outcomes["net_return_bps_4"].describe().to_dict(),
                "mfe_bps": outcomes["mfe_bps"].describe().to_dict(),
                "mae_bps": outcomes["mae_bps"].describe().to_dict(),
            },
        )
    return outcomes
