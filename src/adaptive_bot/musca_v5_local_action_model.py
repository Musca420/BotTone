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
from adaptive_bot.musca_v5_local_restart_frontier import (
    MATRIX as EVENT_MATRIX,
)
from adaptive_bot.musca_v5_local_restart_frontier import (
    PROTOCOL_HASH as EVENT_PROTOCOL_HASH,
)
from adaptive_bot.musca_v5_local_restart_model import (
    QUALITY_FEATURES,
    _run_model,
    attach_restart_quality,
    prepare,
)
from adaptive_bot.musca_v5_room_frontier import _summary
from adaptive_bot.musca_v8_multi_horizon import HOLDOUT_START, profile_cost_bps

ROOT = Path("data/ml/musca_v5")
MATRIX = ROOT / "local_restart_action_matrix.parquet"
REPORT = Path("data/reports/musca_v5_local_restart_action_model.json")
STATUS = Path("data/reports/musca_v5_local_restart_action_model.status.json")
BUNDLE = Path("data/models/musca_v5/local_restart_action_model_research.joblib")
VIP5_COST_BPS = profile_cost_bps("VIP5")
TARGET_COST_MULTIPLIERS = (1.5, 2.0, 3.0)
HOLDING_MINUTES = (5, 15, 30, 60)
ACTION_FEATURES = (
    *QUALITY_FEATURES,
    "target_cost_multiple",
    "action_target_bps",
    "maximum_holding_minutes",
)
EVENT_COLUMNS = (
    "signal_timestamp",
    "available_at",
    "direction",
    "event_family",
    "expert_breakout_bars",
    "impulse_anchor_at",
    "operating_vwap",
    "stop_price",
    "target_price",
    "risk_bps_at_signal",
    "room_bps",
    "trend_score",
    "return_15m",
    "return_60m",
    "spot_return_15m",
    "vwap_60m_slope",
    "relative_volume",
    "perp_taker_1m",
    "perp_taker_5m",
    "spot_taker_5m",
    "pullback_depth_atr",
)
PROTOCOL = {
    "name": "musca_v5_local_restart_action_value_v1",
    "event_protocol_hash": EVENT_PROTOCOL_HASH,
    "market": "BINANCE_BTCUSDT_PERPETUAL_AND_SPOT",
    "target_cost_multipliers_vip5": list(TARGET_COST_MULTIPLIERS),
    "target_bps": [VIP5_COST_BPS * value for value in TARGET_COST_MULTIPLIERS],
    "holding_minutes": list(HOLDING_MINUTES),
    "entry_stop_invalidation": "frozen_FT016",
    "intrabar": "stop_wins",
    "features": list(ACTION_FEATURES),
    "model": "same_conditional_probability_EV_Ridge_vs_XGBoost_as_FT019",
    "action": "target_multiple_x_holding_minutes_then_FLAT_zero",
    "selection": "2024_selection_only_then_2025_validation",
    "audit": "2026_preholdout_reused_discovery_only",
    "holdout_opened": False,
    "changes_to_active_paper": False,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
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


def _write_parquet(rows: pd.DataFrame) -> None:
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(MATRIX.suffix + ".tmp")
    rows.to_parquet(temporary, index=False)
    os.replace(temporary, MATRIX)


def label_actions(events: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
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
    output: list[dict[str, Any]] = []
    raw_events = events.to_dict("records")
    for event_number, raw_event in enumerate(raw_events, start=1):
        event = cast(dict[str, Any], raw_event)
        entry_index = int(
            np.searchsorted(time_values, pd.Timestamp(event["available_at"]).value, side="left")
        )
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(open_price[entry_index])
        stop = float(event["stop_price"])
        risk_bps = side * (entry - stop) / entry * 10_000
        if risk_bps <= 0:
            continue
        for multiplier in TARGET_COST_MULTIPLIERS:
            target_bps = VIP5_COST_BPS * multiplier
            target = entry * (1 + side * target_bps / 10_000)
            for maximum_holding in HOLDING_MINUTES:
                last = min(entry_index + maximum_holding, len(data) - 1)
                exit_index = last
                exit_price = float(close[last])
                reason = f"TIMEOUT_{maximum_holding}M"
                funding_bps = 0.0
                mfe_bps = 0.0
                mae_bps = 0.0
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
                        reason = "STRUCTURAL_STOP"
                        break
                    if targeted:
                        exit_index = current
                        exit_price = target
                        reason = "COST_TARGET"
                        break
                    invalidated = (close[current] - float(event["operating_vwap"])) * side < 0
                    if invalidated and current < last:
                        exit_index = current + 1
                        exit_price = float(open_price[exit_index])
                        if exit_price <= stop if side > 0 else exit_price >= stop:
                            exit_price = (
                                min(exit_price, stop) if side > 0 else max(exit_price, stop)
                            )
                            reason = "STRUCTURAL_STOP_GAP"
                        else:
                            reason = "VWAP_INVALIDATION"
                        break
                gross_market = side * (exit_price - entry) / entry * 10_000
                gross = gross_market + funding_bps
                output.append(
                    event
                    | {
                        "action_protocol_hash": PROTOCOL_HASH,
                        "entry_timestamp": times.iat[entry_index],
                        "exit_timestamp": times.iat[exit_index],
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "action_target_price": target,
                        "target_cost_multiple": multiplier,
                        "action_target_bps": target_bps,
                        "maximum_holding_minutes": maximum_holding,
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
                        "exit_reason": reason,
                    }
                )
        if len(raw_events) > 100 and event_number % 5_000 == 0:
            _status(
                "action_matrix",
                5 + 40 * event_number / len(raw_events),
                f"{event_number:,}/{len(raw_events):,} restart candidates",
            )
    return pd.DataFrame(output)


def build_matrix() -> pd.DataFrame:
    if MATRIX.exists():
        cached = pd.read_parquet(MATRIX)
        if cached["action_protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached
    events = pd.read_parquet(EVENT_MATRIX).loc[:, list(EVENT_COLUMNS)].drop_duplicates()
    matrix = label_actions(events, pd.read_parquet(MINUTES))
    matrix = matrix.loc[pd.to_datetime(matrix["exit_timestamp"], utc=True).lt(HOLDOUT_START)].copy()
    _write_parquet(matrix)
    return matrix


def _oracle(matrix: pd.DataFrame, year: int) -> dict[str, float]:
    rows = matrix.loc[pd.to_datetime(matrix["signal_timestamp"], utc=True).dt.year.eq(year)].copy()
    rows["net_bps"] = rows["gross_return_bps"] - VIP5_COST_BPS
    best = rows.sort_values("net_bps").groupby("signal_timestamp", as_index=False).tail(1)
    positive = best["net_bps"].clip(lower=0)
    return {
        "signals": float(len(best)),
        "positive_actions": float((best["net_bps"] > 0).sum()),
        "positive_rate": float((best["net_bps"] > 0).mean()),
        "mean_flat_inclusive_oracle_net_bps": float(positive.mean()),
    }


def _deterministic_actions(matrix: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    entries = pd.to_datetime(matrix["entry_timestamp"], utc=True)
    for multiplier in TARGET_COST_MULTIPLIERS:
        for horizon in HOLDING_MINUTES:
            key = f"target_{multiplier:g}x_h{horizon}"
            rows = matrix.loc[
                matrix["target_cost_multiple"].eq(multiplier)
                & matrix["maximum_holding_minutes"].eq(horizon)
            ]
            result[key] = {
                str(year): _summary(
                    _non_overlapping(rows.loc[entries[rows.index].dt.year.eq(year)]),
                    VIP5_COST_BPS,
                    bootstrap=False,
                )
                for year in (2024, 2025)
            }
    return result


def run() -> dict[str, Any]:
    _status("action_matrix", 3, "Building target and holding counterfactuals")
    matrix = build_matrix()
    _status("action_features", 48, "Attaching FT-019 restart quality features")
    data = attach_restart_quality(prepare(matrix), pd.read_parquet(MINUTES))
    result = _run_model(
        data,
        features=ACTION_FEATURES,
        protocol=PROTOCOL,
        protocol_hash=PROTOCOL_HASH,
        report_path=REPORT,
        bundle_path=BUNDLE,
        pass_verdict="LOCAL_ACTION_MODEL_RESEARCH_ONLY",
        fail_verdict="NO_LOCAL_ACTION_MODEL",
    )
    result["matrix_rows"] = len(matrix)
    result["unique_signals"] = int(matrix["signal_timestamp"].nunique())
    result["oracle_vip5"] = {str(year): _oracle(matrix, year) for year in (2024, 2025)}
    result["deterministic_actions_vip5"] = _deterministic_actions(matrix)
    _atomic_json(REPORT, result)
    _status("complete", 100, result["verdict"])
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
