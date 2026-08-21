from __future__ import annotations

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
from adaptive_bot.hybrid_policy_v15 import BITUNIX_PATH

PROTOCOL = "hybrid_v18_hourly_gross_first"
TIMEFRAME_MINUTES = 60
VWAP_HOURS = 96
PROTOCOL_PATH = Path("data/models/expert_policy/v18/protocol.json")
MATRIX_PATH = Path("data/ml/hybrid_v18/hourly_matrix.parquet")
REPORT_PATH = Path("data/reports/ml_hybrid_v18.json")


def experts() -> tuple[V11Expert, ...]:
    templates = (
        ("vwap_reentry", "mean_reversion", "confirmed_reentry", 2.0, 0, 2.5, None, 0.0, None, 16),
        ("tsmom_12_48h", "momentum", "multi_horizon", 0.0, 0, 2.0, None, None, 2.5, 48),
        ("donchian_48h", "momentum", "trend_continuation", 0.0, 48, 2.0, None, None, 2.0, 48),
    )
    return tuple(
        V11Expert(
            name,
            cast(Any, family),
            cast(Any, side),
            cast(Any, kind),
            z,
            breakout,
            stop,
            target,
            target_z,
            trail,
            hold,
        )
        for side in ("long", "short")
        for name, family, kind, z, breakout, stop, target, target_z, trail, hold in templates
    )


def _one_position(rows: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    blocked_until = pd.Timestamp("1900", tz="UTC")
    for _, row in rows.sort_values("signal_timestamp").iterrows():
        signal = pd.Timestamp(row["signal_timestamp"])
        if signal <= blocked_until:
            continue
        selected.append(row.to_frame().T)
        blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(hours=1)
    return pd.concat(selected, ignore_index=True) if selected else rows.iloc[:0].copy()


def run(app: AppConfig) -> dict[str, Any]:
    protocol = preregister()
    sources = {name: Path(item["path"]) for name, item in btc_inventory().items()}
    sources["bitunix"] = BITUNIX_PATH
    outcomes: list[pd.DataFrame] = []
    for exchange, path in sources.items():
        print(f"V18 hourly gross-first: {exchange}", flush=True)
        raw, features = build_feature_frame(
            path, app, exchange, timeframe_minutes=TIMEFRAME_MINUTES, vwap_hours=VWAP_HOURS
        )
        for expert in experts():
            result = evaluate_expert(
                features,
                raw,
                expert,
                cost_bps=4,
                timeframe_minutes=TIMEFRAME_MINUTES,
                vwap_hours=VWAP_HOURS,
            )
            if not result.empty:
                outcomes.append(result)
    matrix = pd.concat(outcomes, ignore_index=True)
    results: dict[str, Any] = {}
    for (name, side), candidate in matrix.groupby(["expert_name", "side"]):
        venues = {
            exchange: _metrics(_one_position(rows))
            for exchange, rows in candidate.groupby("exchange")
        }
        external = [venues[name] for name in EXCHANGES]
        admitted = all(
            item["trades"] >= 100
            and item["gross"]["expectancy_r"] > 0
            and item["net_4bps"]["expectancy_r"] > 0
            and item["net_4bps"]["profit_factor"] >= 1.15
            for item in external
        )
        results[f"{name}:{side}"] = {"venues": venues, "admitted_to_ml": admitted}
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "verdict": "EXPERTS_READY_FOR_GATING"
        if any(item["admitted_to_ml"] for item in results.values())
        else "NO_GROSS_FIRST_EXPERT",
        "deployable": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "experts": results,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(MATRIX_PATH, matrix)
    _atomic_json(REPORT_PATH, report)
    return report


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": len(rows),
        "gross": return_metrics(rows, "gross_return_r"),
        "net_4bps": return_metrics(rows, "net_return_r_1x"),
        "stress_8bps": return_metrics(rows, "net_return_r_2x"),
    }


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "vwap_hours": VWAP_HOURS,
        "experts": [asdict(expert) | {"expert_id": expert.expert_id} for expert in experts()],
        "admission": "100 trades, gross EV>0, net4 EV>0 and PF>=1.15 on every external venue",
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V18 protocol changed after freezing")
        return dict(existing)
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


if __name__ == "__main__":
    print(
        json.dumps(run(load_config(Path("configs/bitunix_btc_futures_simulated.yaml"))), indent=2)
    )
