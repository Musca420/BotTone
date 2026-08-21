from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from adaptive_bot.musca_v4_research import _metrics, _non_overlapping
from adaptive_bot.musca_v5_research import _models
from adaptive_bot.musca_v6_flow_research import FEATURES

ACTIONS = Path("data/ml/musca_v6/actions.parquet")
REPORT = Path("data/reports/musca_v6_corrected_walkforward.json")
COVERAGES = (0.005, 0.01, 0.02, 0.05)


def _select(rows: pd.DataFrame, scores: np.ndarray, coverage: float) -> pd.DataFrame:
    threshold = float(np.quantile(scores, 1 - coverage))
    candidates = rows.loc[scores >= threshold].copy()
    candidates["score"] = scores[scores >= threshold]
    best = candidates.sort_values("score").groupby("available_at", as_index=False).tail(1)
    return _non_overlapping(best)


def run() -> dict[str, Any]:
    actions = pd.read_parquet(ACTIONS)
    timestamp = pd.to_datetime(actions["entry_timestamp"], utc=True)
    fit = actions.loc[timestamp.dt.month.eq(1)]
    calibration = actions.loc[timestamp.dt.month.eq(2)]
    validation = actions.loc[timestamp.dt.month.eq(3)]
    models = {name: model for name, model in _models().items() if name in {"ridge", "xgboost_gpu"}}
    audits: list[dict[str, Any]] = []
    eligible: list[tuple[float, str, float]] = []
    for name, model in models.items():
        model.fit(fit[list(FEATURES)], fit["net_return_bps"])
        scores = np.asarray(model.predict(calibration[list(FEATURES)]), dtype=float)
        for coverage in COVERAGES:
            selected = _select(calibration, scores, coverage)
            metrics = _metrics(selected)
            stress = _metrics(selected, "stress_return_bps")
            audits.append(
                {"model": name, "coverage": coverage, "metrics": metrics, "stress": stress}
            )
            if (
                len(selected) / 28 >= 0.5
                and stress["expectancy_bps"] > 0
                and stress["profit_factor"] >= 1.10
            ):
                eligible.append((stress["expectancy_bps"], name, coverage))

    selected_validation = validation.iloc[:0].copy()
    champion = "FLAT"
    coverage = 0.0
    if eligible:
        _, champion, coverage = max(eligible)
        model = clone(models[champion])
        refit = pd.concat([fit, calibration], ignore_index=True)
        model.fit(refit[list(FEATURES)], refit["net_return_bps"])
        scores = np.asarray(model.predict(validation[list(FEATURES)]), dtype=float)
        selected_validation = _select(validation, scores, coverage)
    metrics = _metrics(selected_validation)
    stress = _metrics(selected_validation, "stress_return_bps")
    gates = {
        "validation_trades_15": len(selected_validation) >= 15,
        "frequency_0_5_day": len(selected_validation) / 31 >= 0.5,
        "expectancy_positive": metrics["expectancy_bps"] > 0,
        "profit_factor_1_10": metrics["profit_factor"] >= 1.10,
        "stress_nonnegative": stress["expectancy_bps"] >= 0,
        "drawdown_8pct": metrics["max_drawdown"] <= 0.08,
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "fit_Jan_select_Feb_refit_JanFeb_validate_Mar_by_fixed_coverage",
        "models": ["ridge", "xgboost_gpu"],
        "calibration_audit": audits,
        "champion": champion,
        "coverage": coverage,
        "validation_metrics": metrics,
        "validation_stress_metrics": stress,
        "gates": gates,
        "verdict": "READY_FOR_ONE_TIME_APRIL_AUDIT" if all(gates.values()) else "NO_EDGE",
        "april_opened": False,
        "real_capital_allowed": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
