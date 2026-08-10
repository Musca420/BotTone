from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.musca_vwap_trend_baselines import (
    FAMILIES,
    HOLDOUT_START,
    HORIZONS,
    SOURCE,
    _bars,
    _metrics,
    _signals,
    _trades,
)

REPORT = Path("data/reports/musca_vwap_trend_quality_audit.json")
FILTERS = ("LIQUID_TREND", "PERSISTENT_FLOW", "COMBINED")
PROTOCOL = {
    "name": "btc_binance_vwap_trend_quality_v1",
    "base": "btc_binance_vwap_trend_baselines_v1",
    "filters": {
        "LIQUID_TREND": "trend_strength_atr>=1, relative_volume>=1, trade_intensity>=1",
        "PERSISTENT_FLOW": "signed_flow_15m>=0.05, signed_flow_1h>=0.10",
        "COMBINED": "both filters",
    },
    "normal_cost_bps": 8.0,
    "stress_cost_bps": 16.0,
    "selection": "2024 train and 2025 validation",
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def run() -> dict[str, Any]:
    bars = _bars(pd.read_parquet(SOURCE))
    signals = _signals(bars)
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        side = signals[family]
        liquid = (
            bars["trend_strength_atr"].ge(1)
            & bars["relative_volume"].ge(1)
            & bars["trade_intensity"].ge(1)
        )
        persistent = (side * bars["flow_15m"] >= 0.05) & (
            side * bars["flow_1h"] >= 0.10
        )
        masks = {
            "LIQUID_TREND": liquid,
            "PERSISTENT_FLOW": persistent,
            "COMBINED": liquid & persistent,
        }
        for filter_name in FILTERS:
            filtered = side.where(masks[filter_name])
            for horizon in HORIZONS:
                trades = _trades(bars, filtered, horizon)
                time = pd.to_datetime(trades["entry_timestamp"], utc=True)
                train = _metrics(trades.loc[time.dt.year.eq(2024)])
                validation = _metrics(trades.loc[time.dt.year.eq(2025)])
                stable = (
                    train["trades"] >= 50
                    and validation["trades"] >= 50
                    and train["gross_expectancy_bps"] >= 24
                    and validation["gross_expectancy_bps"] >= 24
                    and train["expectancy_bps"] > 0
                    and validation["expectancy_bps"] > 0
                    and train["profit_factor"] >= 1.10
                    and validation["profit_factor"] >= 1.10
                    and train["stress_expectancy_bps"] >= 0
                    and validation["stress_expectancy_bps"] >= 0
                )
                test = trades.loc[time.dt.year.eq(2026) & time.lt(HOLDOUT_START)]
                rows.append(
                    {
                        "family": family,
                        "filter": filter_name,
                        "holding_bars": horizon,
                        "train": train,
                        "validation": validation,
                        "eligible_train_validation": stable,
                        "test": _metrics(test) if stable else None,
                    }
                )
    eligible = [row for row in rows if row["eligible_train_validation"]]
    payload = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "configurations": rows,
        "eligible": eligible,
        "verdict": "QUALITY_EDGE_FOUND" if eligible else "NO_QUALITY_EDGE",
        "holdout_opened": False,
        "live_orders_enabled": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
