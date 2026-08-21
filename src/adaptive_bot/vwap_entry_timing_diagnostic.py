from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.hybrid_policy_v11 import btc_inventory, build_feature_frame, evaluate_expert
from adaptive_bot.hybrid_policy_v15 import (
    BITUNIX_PATH,
    ENTRY_Z_MAX,
    ENTRY_Z_MIN,
    MIN_GROSS_EDGE_BPS,
    _audit,
    _one_position,
    zone_expert,
)

REPORT_PATH = Path("data/reports/vwap_entry_timing_diagnostic.json")
MATRIX_PATH = Path("data/ml/vwap_entry_timing/diagnostic.parquet")


def entry_masks(features: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    z = features["distance_vwap_atr"].astype(float)
    distance = z.abs()
    direction = z.lt(0) if side == "long" else z.gt(0)
    valid = (
        direction
        & features["regime_code"].eq(0)
        & features["regime_code"].shift(1).eq(0)
        & features["atr_percentile"].le(90)
        & features["data_valid"].fillna(False).astype(bool)
        & features["local_feature_coverage"].fillna(False).astype(bool)
    )
    edge = (distance - 0.25) * features["atr"] / features["close"] * 10_000
    zone = distance.between(ENTRY_Z_MIN, ENTRY_Z_MAX) & edge.ge(MIN_GROSS_EDGE_BPS)
    return {
        "first_touch_outward": valid
        & zone
        & distance.shift(1).lt(ENTRY_Z_MIN)
        & distance.ge(ENTRY_Z_MIN),
        "continuous_zone": valid & zone,
        "turning_inside_zone": valid & zone & distance.lt(distance.shift(1)),
        "confirmed_reentry_after_1_5": valid & zone & distance.shift(1).ge(1.5) & distance.le(1.25),
    }


def run(app: AppConfig) -> dict[str, Any]:
    sources = {name: Path(item["path"]) for name, item in btc_inventory().items()}
    sources["bitunix"] = BITUNIX_PATH
    outcomes: list[pd.DataFrame] = []
    for exchange, path in sources.items():
        raw, features = build_feature_frame(path, app, exchange, timeframe_minutes=5, vwap_hours=24)
        for side in ("long", "short"):
            for timing, mask in entry_masks(features, side).items():
                result = evaluate_expert(
                    features,
                    raw,
                    zone_expert(side),
                    cost_bps=4,
                    entry_mask=mask,
                    timeframe_minutes=5,
                    vwap_hours=24,
                )
                if not result.empty:
                    result["entry_timing"] = timing
                    outcomes.append(result)
    matrix = pd.concat(outcomes, ignore_index=True)
    policies = {
        f"{timing}:{exchange}": _one_position(rows)
        for (timing, exchange), rows in matrix.groupby(["entry_timing", "exchange"])
    }
    report = {
        "status": "DISCOVERY_TIMING_DIAGNOSTIC_NOT_CONFIRMATION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison": {
            timing: {
                exchange: _audit(policies[f"{timing}:{exchange}"])
                for exchange in sources
                if f"{timing}:{exchange}" in policies
            }
            for timing in sorted(matrix["entry_timing"].unique())
        },
        "fixed_exit": "target z=0.25, stop 2 ATR, maximum 2 hours",
        "fixed_costs": "4/12/19 bps round trip",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(MATRIX_PATH, matrix)
    _atomic_json(REPORT_PATH, report)
    return report


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
