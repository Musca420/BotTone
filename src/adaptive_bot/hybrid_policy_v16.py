from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v11 import return_metrics

PROTOCOL = "hybrid_v16_vwap_calibrated_ev"
EXCHANGES = ("binance", "okx", "bybit")
MATRIX_PATH = Path("data/ml/hybrid_v15/zone_cycle_matrix.parquet")
MODEL_ROOT = Path("data/models/expert_policy/v16")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
REPORT_PATH = Path("data/reports/ml_hybrid_v16.json")
OOS_PATH = Path("data/ml/hybrid_v16/oos_candidates.parquet")
STATUS_PATH = Path("data/reports/ml_hybrid_v16.status.json")
FEATURES = (
    "distance_vwap_atr",
    "vwap_zscore",
    "vwap_slope_3",
    "vwap_slope_6",
    "vwap_slope_12",
    "bars_above_vwap",
    "bars_below_vwap",
    "crossed_vwap_12",
    "atr_pct",
    "atr_percentile",
    "realized_volatility",
    "adx",
    "adx_slope",
    "ema20_slope",
    "ema50_slope",
    "relative_volume",
    "volume_zscore",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_48",
    "rsi",
    "close_position_in_candle",
    "upper_wick_pct",
    "lower_wick_pct",
    "mark_last_divergence",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "regime_code",
)


@dataclass(frozen=True)
class FittedSide:
    model: Any
    calibrator: IsotonicRegression
    champion: str
    admission_mse: float
    admission_ev_r: float


def _weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("exchange")["exchange"].transform("size")
    weights = 1.0 / counts
    return np.asarray((weights / weights.mean()).to_numpy(float), dtype=float)


def _model(kind: str) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        random_state=20260804,
    )


def _raw(model: Any, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(rows[list(FEATURES)]), dtype=float)


def _fit_candidate(kind: str, train: pd.DataFrame, calibrate: pd.DataFrame) -> tuple[Any, Any]:
    model = _model(kind)
    fit_parameters = (
        {"ridge__sample_weight": _weights(train)}
        if kind == "ridge"
        else {"sample_weight": _weights(train)}
    )
    model.fit(train[list(FEATURES)], train["target_net_r"], **fit_parameters)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        _raw(model, calibrate), calibrate["target_net_r"].to_numpy(float)
    )
    return model, calibrator


def _predict(model: Any, calibrator: IsotonicRegression, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(calibrator.predict(_raw(model, rows)), dtype=float)


def _select(rows: pd.DataFrame, *, prediction: str = "ev_net") -> pd.DataFrame:
    proposals = rows.loc[rows[prediction].gt(0)].sort_values(
        ["exchange", "signal_timestamp", prediction], ascending=[True, True, False]
    )
    chosen: list[pd.DataFrame] = []
    for _, venue in proposals.groupby("exchange", sort=True):
        blocked_until = pd.Timestamp("1900", tz="UTC")
        for _, row in venue.iterrows():
            signal = pd.Timestamp(row["signal_timestamp"])
            if signal <= blocked_until:
                continue
            chosen.append(row.to_frame().T)
            blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(minutes=5)
    return pd.concat(chosen, ignore_index=True) if chosen else rows.iloc[:0].copy()


def _fit_side(train: pd.DataFrame, calibration: pd.DataFrame) -> FittedSide | None:
    if len(train) < 500 or len(calibration) < 160:
        return None
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    calibrate = calibration.loc[times.lt(midpoint)]
    admission = calibration.loc[times.ge(midpoint)]
    if min(len(calibrate), len(admission)) < 80:
        return None
    candidates: dict[str, FittedSide] = {}
    for kind in ("ridge", "xgboost"):
        model, calibrator = _fit_candidate(kind, train, calibrate)
        predicted = admission.assign(ev_net=_predict(model, calibrator, admission))
        selected = _select(predicted)
        candidates[kind] = FittedSide(
            model,
            calibrator,
            kind,
            float(mean_squared_error(admission["target_net_r"], predicted["ev_net"])),
            float(selected["target_net_r"].mean()) if len(selected) >= 20 else float("-inf"),
        )
    ridge, challenger = candidates["ridge"], candidates["xgboost"]
    return (
        challenger
        if challenger.admission_mse < ridge.admission_mse
        and challenger.admission_ev_r > ridge.admission_ev_r
        else ridge
    )


def run() -> dict[str, Any]:
    protocol = preregister()
    rows = pd.read_parquet(MATRIX_PATH)
    rows = rows.loc[rows["exchange"].isin(EXCHANGES)].copy()
    rows["feature_coverage"] = rows["local_feature_coverage"].fillna(False).astype(bool)
    rows["target_net_r"] = rows["gross_return_r"].astype(float) - rows["cost_r_1x"].astype(float)
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, "target_net_r"])
    outputs: list[pd.DataFrame] = []
    champions: dict[str, int] = {}
    total = 0
    scenarios: list[tuple[str, tuple[Any, ...]]] = []
    for held_out in EXCHANGES:
        folds = purged_expert_folds(
            rows,
            train_weeks=52,
            calibration_weeks=4,
            test_weeks=4,
            step_weeks=4,
            embargo_hours=2,
        )
        scenarios.append((held_out, folds))
        total += len(folds)
    completed = 0
    for held_out, folds in scenarios:
        for number, fold in enumerate(folds, start=1):
            train = rows.iloc[fold.train]
            calibration = rows.iloc[fold.calibration]
            test = rows.iloc[fold.test]
            train = train.loc[train["exchange"].ne(held_out)]
            calibration = calibration.loc[calibration["exchange"].ne(held_out)]
            test = test.loc[test["exchange"].eq(held_out)].copy()
            predictions: list[pd.DataFrame] = []
            for side in ("long", "short"):
                fitted = _fit_side(
                    train.loc[train["side"].eq(side)],
                    calibration.loc[calibration["side"].eq(side)],
                )
                if fitted is None:
                    continue
                selected = test.loc[test["side"].eq(side)].copy()
                selected["ev_net"] = _predict(fitted.model, fitted.calibrator, selected)
                selected["champion"] = fitted.champion
                predictions.append(selected)
                key = f"{held_out}:{side}:{fitted.champion}"
                champions[key] = champions.get(key, 0) + 1
            if predictions:
                fold_rows = pd.concat(predictions, ignore_index=True)
                fold_rows["held_out_exchange"] = held_out
                fold_rows["outer_fold"] = number
                outputs.append(fold_rows)
            completed += 1
            _status(
                "walk_forward",
                f"LOEO {held_out.upper()} fold {number}/{len(folds)}",
                5 + 90 * completed / max(total, 1),
            )
    candidates = pd.concat(outputs, ignore_index=True) if outputs else rows.iloc[:0].copy()
    decisions = _select(candidates)
    audit = _audit(decisions)
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "verdict": "ALPHA_SHADOW_CANDIDATE" if audit["gates_passed"] else "FLAT",
        "deployable": False,
        "paper_enabled": False,
        "live_enabled": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "decision": "calibrated_EV_net > 0; FLAT otherwise",
        "individual_trade_residual_lcb": "removed_as_statistically_invalid",
        "policy_oos_bootstrap_lcb_r": audit["expectancy_lower_bound_r"],
        "champions": champions,
        "audit": audit,
        "warning": "V15/V16 are discovery; confirmation requires chronologically newer data",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(OOS_PATH, candidates)
    _atomic_json(REPORT_PATH, report)
    _status("complete", report["verdict"], 100)
    return report


def _audit(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"trades": 0, "expectancy_lower_bound_r": float("-inf"), "gates_passed": False}
    base = return_metrics(rows, "target_net_r")
    stressed = rows.assign(stress=rows["gross_return_r"] - 2 * rows["cost_r_1x"])
    stress = return_metrics(stressed, "stress")
    lower = moving_block_lower_bound(
        rows["target_net_r"].to_numpy(float), block_size=min(20, len(rows)), seed=20260804
    )
    by_exchange = {
        name: return_metrics(group, "target_net_r") for name, group in rows.groupby("exchange")
    }
    gates = {
        "trades_300": len(rows) >= 300,
        "expectancy_positive": base["expectancy_r"] > 0,
        "bootstrap_lower_positive": lower > 0,
        "profit_factor_1_15": base["profit_factor"] >= 1.15,
        "drawdown_8pct": base["max_drawdown"] <= 0.08,
        "costs_2x_nonnegative": stress["expectancy_r"] >= 0,
        "all_three_exchanges_positive": len(by_exchange) == 3
        and all(item["expectancy_r"] > 0 for item in by_exchange.values()),
    }
    return {
        "trades": len(rows),
        "net_4bps": base,
        "stress_8bps": stress,
        "expectancy_lower_bound_r": lower,
        "by_exchange": by_exchange,
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_v15_protocol_sha256": (
            "f9076275a85ec28608419378018cd727cf7a25ad3e4de652ef7714df911fe920"
        ),
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "features": list(FEATURES),
        "target": "net_return_r_at_4bps",
        "models": {"default": "ridge_alpha_100", "challenger": "xgboost_gpu_fixed"},
        "walk_forward_weeks": [52, 4, 4, 4],
        "decision": "calibrated_ev_net_gt_zero_else_flat",
        "uncertainty": "moving_block_bootstrap_on_complete_oos_policy",
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V16 protocol changed after freezing")
        return dict(existing)
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


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
    argparse.ArgumentParser(description="V16 calibrated BTC VWAP EV walk-forward").parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
