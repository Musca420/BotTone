from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import _metrics, _non_overlapping
from adaptive_bot.musca_v5_micro_model import load_5s
from adaptive_bot.musca_v5_research import BARS
from adaptive_bot.musca_v6_flow_research import build_states, label_actions

ROOT = Path("data/ml/musca_v6")
REPORT = Path("data/reports/musca_v6_horizon_audit.json")
HORIZONS = (120, 240, 360)


def run() -> dict[str, Any]:
    output = ROOT / "actions_2h_6h.parquet"
    if output.exists():
        actions = pd.read_parquet(output)
    else:
        path = load_5s()
        states = build_states(path, pd.read_parquet(BARS))
        actions = label_actions(states, path, horizons=HORIZONS)
        actions.to_parquet(output, index=False)
    timestamp = pd.to_datetime(actions["entry_timestamp"], utc=True)
    rules = {
        "flow": np.sign(actions["ofi_1m"]),
        "trend": np.sign(actions["trend_vote"]),
        "flow_trend_agree": np.where(
            np.sign(actions["ofi_1m"]) == np.sign(actions["trend_vote"]),
            np.sign(actions["trend_vote"]),
            0,
        ),
        "fade_flow": -np.sign(actions["ofi_1m"]),
        "fade_velocity": -np.sign(actions["return_1m_bps"]),
        "vwap_reversion": -np.sign(actions["distance_daily_vwap_atr"]),
    }
    audits: list[dict[str, Any]] = []
    for name, direction in rules.items():
        for month in (1, 2, 3):
            for horizon in HORIZONS:
                chosen = actions.loc[
                    timestamp.dt.month.eq(month)
                    & actions["action_direction"].eq(pd.Series(direction, index=actions.index))
                    & actions["horizon_minutes"].eq(horizon)
                ]
                selected = _non_overlapping(chosen)
                audits.append(
                    {
                        "rule": name,
                        "month": month,
                        "horizon_minutes": horizon,
                        "metrics": _metrics(selected),
                        "stress": _metrics(selected, "stress_return_bps"),
                        "gross_expectancy_bps": float(selected["gross_return_bps"].mean())
                        if len(selected)
                        else 0.0,
                    }
                )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "states": int(actions["available_at"].nunique()),
        "actions": len(actions),
        "horizons_minutes": HORIZONS,
        "minimum_required_gross_bps": 24.0,
        "deterministic_baselines": audits,
        "model_authorized": False,
        "real_capital_allowed": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    summary = {
        key: value for key, value in result.items() if key != "deterministic_baselines"
    }
    print(json.dumps(summary, indent=2))
