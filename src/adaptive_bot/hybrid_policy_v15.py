from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.hybrid_policy_v11 import (
    EXCHANGES,
    V11Expert,
    btc_inventory,
    build_feature_frame,
    evaluate_expert,
    return_metrics,
)

PROTOCOL = "hybrid_v15_vwap_zone_cycle"
TIMEFRAME_MINUTES = 5
VWAP_HOURS = 24
ENTRY_Z_MIN = 0.75
ENTRY_Z_MAX = 2.5
TARGET_Z = 0.25
STOP_ATR = 2.0
MAX_HOLDING_BARS = 24
MIN_GROSS_EDGE_BPS = 12.0
COOLDOWN_MINUTES = 5
BITUNIX_PATH = Path("data/ml/hybrid_v7/alpha_raw/exchange=bitunix/symbol=BTCUSDT/data.parquet")
REPORT_PATH = Path("data/reports/ml_hybrid_v15_discovery.json")
MATRIX_PATH = Path("data/ml/hybrid_v15/zone_cycle_matrix.parquet")
PROTOCOL_PATH = Path("data/models/expert_policy/v15/protocol.json")


def zone_expert(side: str) -> V11Expert:
    return V11Expert(
        "vwap_zone_cycle_5m",
        "mean_reversion",
        cast(Any, side),
        "range_exhaustion",
        ENTRY_Z_MIN,
        0,
        STOP_ATR,
        None,
        TARGET_Z,
        None,
        MAX_HOLDING_BARS,
    )


def zone_entry_mask(features: pd.DataFrame, side: str) -> pd.Series:
    z = features["distance_vwap_atr"].astype(float)
    distance = z.abs()
    direction = z.lt(0) if side == "long" else z.gt(0)
    turning = distance.lt(distance.shift(1))
    stable_range = features["regime_code"].eq(0) & features["regime_code"].shift(1).eq(0)
    available_bps = (distance - TARGET_Z) * features["atr"] / features["close"] * 10_000
    return (
        direction
        & turning
        & distance.between(ENTRY_Z_MIN, ENTRY_Z_MAX)
        & stable_range
        & features["atr_percentile"].le(90)
        & available_bps.ge(MIN_GROSS_EDGE_BPS)
        & features["data_valid"].fillna(False).astype(bool)
        & features["local_feature_coverage"].fillna(False).astype(bool)
    )


def run_discovery(app: AppConfig) -> dict[str, Any]:
    protocol = preregister()
    inventory = btc_inventory()
    sources = {exchange: Path(item["path"]) for exchange, item in inventory.items()}
    sources["bitunix"] = BITUNIX_PATH
    outcomes: list[pd.DataFrame] = []
    for exchange, path in sources.items():
        print(f"V15 VWAP zone-cycle: {exchange}", flush=True)
        raw, features = build_feature_frame(
            path,
            app,
            exchange,
            timeframe_minutes=TIMEFRAME_MINUTES,
            vwap_hours=VWAP_HOURS,
        )
        for side in ("long", "short"):
            result = evaluate_expert(
                features,
                raw,
                zone_expert(side),
                cost_bps=4,
                entry_mask=zone_entry_mask(features, side),
                timeframe_minutes=TIMEFRAME_MINUTES,
                vwap_hours=VWAP_HOURS,
            )
            if not result.empty:
                outcomes.append(result)
    if not outcomes:
        raise RuntimeError("V15 produced no valid BTC opportunities")
    matrix = pd.concat(outcomes, ignore_index=True).sort_values("signal_timestamp")
    policies = {
        exchange: _one_position(rows) for exchange, rows in matrix.groupby("exchange", sort=True)
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "PREREGISTERED_DISCOVERY_DIAGNOSTIC",
        "deployable": False,
        "paper_enabled": False,
        "live_enabled": False,
        "symbols": ["BTCUSDT"],
        "exchanges": [*EXCHANGES, "bitunix"],
        "rule": protocol_rule(),
        "opportunities": len(matrix),
        "results": {exchange: _audit(rows) for exchange, rows in policies.items()},
        "pooled": _audit(pd.concat(policies.values(), ignore_index=True)),
        "warning": (
            "historical discovery only; POST_ONLY fill probability and adverse selection "
            "remain unproven until observed Bitunix execution labels"
        ),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(MATRIX_PATH, matrix)
    _atomic_json(REPORT_PATH, report)
    return report


def _one_position(rows: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    blocked_until = pd.Timestamp("1900-01-01", tz="UTC")
    for _, row in rows.sort_values(["signal_timestamp", "side"]).iterrows():
        signal = pd.Timestamp(row["signal_timestamp"])
        if signal <= blocked_until:
            continue
        chosen.append(row.to_frame().T)
        blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(minutes=COOLDOWN_MINUTES)
    return pd.concat(chosen, ignore_index=True) if chosen else rows.iloc[:0].copy()


def _audit(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"trades": 0}
    funding = pd.to_numeric(rows["funding_return_r"], errors="coerce")
    funding_complete = funding.notna()
    cost_4bps = rows["cost_r_1x"].astype(float)
    scenarios = rows.copy()
    scenarios["gross_without_funding"] = rows["gross_return_r"].astype(float)
    scenarios["gross_with_funding"] = scenarios["gross_without_funding"] + funding
    for bps in (4, 12, 19):
        scenarios[f"net_{bps}bps"] = scenarios["gross_without_funding"] - cost_4bps * (bps / 4)
    times = pd.to_datetime(rows["signal_timestamp"], utc=True)
    days = max((times.max() - times.min()).total_seconds() / 86_400, 1)
    return {
        "trades": len(rows),
        "trades_per_30_days": len(rows) / days * 30,
        "funding_coverage_fraction": float(funding_complete.mean()),
        "gross_without_funding": return_metrics(scenarios, "gross_without_funding"),
        "gross_with_observed_funding": return_metrics(
            scenarios.loc[funding_complete], "gross_with_funding"
        ),
        "maker_4bps": return_metrics(scenarios, "net_4bps"),
        "taker_12bps": return_metrics(scenarios, "net_12bps"),
        "conservative_19bps": return_metrics(scenarios, "net_19bps"),
        "long_trades": int(rows["side"].eq("long").sum()),
        "short_trades": int(rows["side"].eq("short").sum()),
    }


def protocol_rule() -> dict[str, Any]:
    return {
        "center": "rolling_vwap_24h",
        "decision_interval_minutes": TIMEFRAME_MINUTES,
        "entry_zone_atr": [ENTRY_Z_MIN, ENTRY_Z_MAX],
        "entry": "deviation turning toward VWAP in stable RANGE",
        "target_z": TARGET_Z,
        "stop_atr": STOP_ATR,
        "maximum_holding_bars": MAX_HOLDING_BARS,
        "minimum_gross_edge_bps": MIN_GROSS_EDGE_BPS,
        "one_position": True,
        "cooldown_minutes": COOLDOWN_MINUTES,
        "historical_entry_execution": "next_1m_open; alpha diagnostic, not POST_ONLY fill proof",
        "same_bar_stop_target": "worst_case_stop",
    }


def protocol_hash() -> str:
    payload = {
        "protocol": PROTOCOL,
        "rule": protocol_rule(),
        "expert": {side: asdict(zone_expert(side)) for side in ("long", "short")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preregister() -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "rule": protocol_rule(),
        "experts": {side: asdict(zone_expert(side)) for side in ("long", "short")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "registered_at": datetime.now(UTC).isoformat(),
    }
    if PROTOCOL_PATH.exists():
        registered = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if registered["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V15 protocol is already frozen with a different hash")
        return cast(dict[str, Any], registered)
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 BTC VWAP zone-cycle discovery")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    print(json.dumps(run_discovery(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
