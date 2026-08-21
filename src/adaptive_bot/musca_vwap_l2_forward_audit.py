from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.binance_l2_dataset import FEATURES, materialize

REPORT = Path("data/reports/musca_vwap_l2_forward_audit.json")
HORIZONS = (300, 900, 1800)
MINIMUM_EVENTS = 30
MINIMUM_PROFIT_FACTOR = 1.10
PROTOCOL = {
    "name": "vwap_l2_follow_fade_forward_v1",
    "minimum_abs_vwap_distance_bps": 2.0,
    "minimum_vote_strength": 2,
    "horizons_seconds": list(HORIZONS),
    "round_trip_cost_bps": 4.0,
    "stress_cost_bps": 8.0,
    "minimum_independent_events": MINIMUM_EVENTS,
    "minimum_profit_factor": MINIMUM_PROFIT_FACTOR,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _events(frame: pd.DataFrame, horizon: int, family: str) -> pd.DataFrame:
    distance = frame["rolling_vwap_5m_distance_bps"]
    distance_sign = np.sign(distance)
    vote = (
        np.sign(frame["aggressive_imbalance_60s"])
        + np.sign(frame["depth_imbalance_5"])
        + np.sign(frame["microprice_distance_bps"])
    )
    aligned = distance_sign * vote
    side = distance_sign if family == "FOLLOW" else -distance_sign
    setup = distance.abs().ge(2.0) & (
        aligned.ge(2) if family == "FOLLOW" else aligned.le(-2)
    )
    candidates = frame.loc[
        setup & frame[f"future_return_{horizon}s_bps"].notna()
    ].copy()
    candidates["side"] = side.loc[candidates.index]
    chosen: list[int] = []
    next_second = -1
    for index, row in candidates.iterrows():
        second = int(row["exchange_timestamp"].timestamp())
        if second >= next_second:
            chosen.append(index)
            next_second = second + horizon
    selected: pd.DataFrame = candidates.loc[chosen].copy()
    return selected


def _metrics(events: pd.DataFrame, horizon: int) -> dict[str, Any]:
    gross = events["side"] * events[f"future_return_{horizon}s_bps"]
    net = gross - 4.0
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {
        "independent_events": len(events),
        "gross_expectancy_bps": float(gross.mean()) if len(events) else None,
        "net_expectancy_bps": float(net.mean()) if len(events) else None,
        "stress_expectancy_bps": float((gross - 8.0).mean()) if len(events) else None,
        "win_rate": float(net.gt(0).mean()) if len(events) else None,
        "profit_factor": float(gains / losses) if losses else None,
    }


def run() -> dict[str, Any]:
    source = materialize()
    frame = pd.read_parquet(source)
    frame = frame.loc[frame["feature_valid"] & frame[list(FEATURES)].notna().all(axis=1)]
    results = {
        f"{family.lower()}_{horizon}s": _metrics(
            _events(frame, horizon, family), horizon
        )
        for horizon in HORIZONS
        for family in ("FOLLOW", "FADE")
    }
    eligible = [
        name
        for name, metrics in results.items()
        if metrics["independent_events"] >= MINIMUM_EVENTS
        and (metrics["net_expectancy_bps"] or 0) > 0
        and (metrics["stress_expectancy_bps"] or 0) >= 0
        and (metrics["profit_factor"] or 0) >= MINIMUM_PROFIT_FACTOR
    ]
    payload = {
        "validation_status": "FORWARD_CANDIDATE" if eligible else "COLLECTING_NO_POLICY",
        "live_orders_enabled": False,
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "feature_rows": len(frame),
        "results": results,
        "eligible": eligible,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only VWAP/L2 event audit")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        result = run()
        print(json.dumps(result, indent=2))
        if args.once:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
