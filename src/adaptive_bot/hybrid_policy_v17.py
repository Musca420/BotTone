from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v11 import (
    EXCHANGES,
    V11Expert,
    btc_inventory,
    build_feature_frame,
    evaluate_expert,
    return_metrics,
)
from adaptive_bot.hybrid_policy_v15 import BITUNIX_PATH
from adaptive_bot.hybrid_policy_v16 import _model, _weights

PROTOCOL = "hybrid_v17_vwap_fade_follow"
TIMEFRAME_MINUTES = 5
VWAP_HOURS = 24
MIN_DISTANCE_ATR = 0.5
MAX_DISTANCE_ATR = 3.0
STOP_ATR = 2.0
FOLLOW_TARGET_ATR = 2.0
MAX_HOLDING_BARS = 24
MODEL_ROOT = Path("data/models/expert_policy/v17")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
MATRIX_ROOT = Path("data/ml/hybrid_v17/matrix")
OOS_PATH = Path("data/ml/hybrid_v17/oos_candidates.parquet")
REPORT_PATH = Path("data/reports/ml_hybrid_v17.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v17.status.json")
FEATURES = (
    "distance_vwap_atr",
    "gap_velocity_1",
    "gap_velocity_3",
    "vwap_chase_ratio_3",
    "vwap_chase_ratio_12",
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
    "side_code",
)


@dataclass(frozen=True)
class FittedAction:
    model: Any
    calibrator: IsotonicRegression
    champion: str
    admission_mse: float
    admission_ev_r: float


def action_expert(action: str, side: str) -> V11Expert:
    if action == "fade":
        return V11Expert(
            "v17_fade_to_frozen_vwap",
            "mean_reversion",
            cast(Any, side),
            "range_exhaustion",
            MIN_DISTANCE_ATR,
            0,
            STOP_ATR,
            None,
            0.0,
            None,
            MAX_HOLDING_BARS,
        )
    return V11Expert(
        "v17_follow_extension",
        "momentum",
        cast(Any, side),
        "multi_horizon",
        0.0,
        0,
        STOP_ATR,
        FOLLOW_TARGET_ATR,
        None,
        None,
        MAX_HOLDING_BARS,
    )


def add_relational_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    gap = result["close"] - result["vwap"]
    result["gap_velocity_1"] = result["distance_vwap_atr"].diff()
    result["gap_velocity_3"] = result["distance_vwap_atr"].diff(3)
    denominator = gap.abs().replace(0, np.nan)
    result["vwap_chase_ratio_3"] = result["vwap"].diff(3).abs() / denominator
    result["vwap_chase_ratio_12"] = result["vwap"].diff(12).abs() / denominator
    return result


def state_masks(features: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    z = features["distance_vwap_atr"].astype(float)
    common = (
        z.abs().between(MIN_DISTANCE_ATR, MAX_DISTANCE_ATR)
        & features["atr_percentile"].le(90)
        & features["regime_code"].ne(4)
        & features["regime_code"].ne(3)
        & features["data_valid"].fillna(False).astype(bool)
        & features["local_feature_coverage"].fillna(False).astype(bool)
        & features[list(FEATURES[:-1])].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    )
    above, below = z.gt(0), z.lt(0)
    return {
        ("fade", "short"): common & above,
        ("follow", "long"): common & above,
        ("fade", "long"): common & below,
        ("follow", "short"): common & below,
    }


def build_exchange_matrix(app: AppConfig, exchange: str, path: Path) -> pd.DataFrame:
    raw, features = build_feature_frame(
        path,
        app,
        exchange,
        timeframe_minutes=TIMEFRAME_MINUTES,
        vwap_hours=VWAP_HOURS,
    )
    features = add_relational_features(features)
    outcomes: list[pd.DataFrame] = []
    for (action, side), mask in state_masks(features).items():
        result = evaluate_expert(
            features,
            raw,
            action_expert(action, side),
            cost_bps=4,
            entry_mask=mask,
            timeframe_minutes=TIMEFRAME_MINUTES,
            vwap_hours=VWAP_HOURS,
        )
        if not result.empty:
            result["action"] = action
            result["side_code"] = 1.0 if side == "long" else -1.0
            outcomes.append(result)
    if not outcomes:
        raise RuntimeError(f"V17 produced no valid states for {exchange}")
    matrix = pd.concat(outcomes, ignore_index=True)
    keys = ["exchange", "signal_timestamp"]
    paired = matrix.groupby(keys)["action"].transform("nunique").eq(2)
    matrix = matrix.loc[paired].sort_values(["signal_timestamp", "action"]).reset_index(drop=True)
    if not matrix.groupby(keys)["action"].nunique().eq(2).all():
        raise RuntimeError(f"unpaired V17 counterfactual state: {exchange}")
    matrix["target_net_r"] = matrix["gross_return_r"] - matrix["cost_r_1x"]
    matrix["state_id"] = (
        matrix["exchange"].astype(str)
        + "|"
        + pd.to_datetime(matrix["signal_timestamp"], utc=True).astype(str)
    )
    return matrix


def build_matrices(app: AppConfig, *, resume: bool) -> pd.DataFrame:
    sources = {name: Path(item["path"]) for name, item in btc_inventory().items()}
    sources["bitunix"] = BITUNIX_PATH
    frames: list[pd.DataFrame] = []
    for number, (exchange, path) in enumerate(sources.items(), start=1):
        target = MATRIX_ROOT / f"{exchange}.parquet"
        if resume and target.exists():
            frame = pd.read_parquet(target)
        else:
            _status(
                "matrix",
                f"{exchange.upper()} {number}/{len(sources)}",
                2 + 18 * number / len(sources),
            )
            frame = build_exchange_matrix(app, exchange, path)
            _atomic_parquet(target, frame)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True).sort_values("signal_timestamp").reset_index(drop=True)
    )


def _raw(model: Any, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(rows[list(FEATURES)]), dtype=float)


def _fit_candidate(kind: str, train: pd.DataFrame, calibrate: pd.DataFrame) -> tuple[Any, Any]:
    model = _model(kind)
    parameters = (
        {"ridge__sample_weight": _weights(train)}
        if kind == "ridge"
        else {"sample_weight": _weights(train)}
    )
    model.fit(train[list(FEATURES)], train["target_net_r"], **parameters)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        _raw(model, calibrate), calibrate["target_net_r"].to_numpy(float)
    )
    return model, calibrator


def _predict(fitted: FittedAction, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(fitted.calibrator.predict(_raw(fitted.model, rows)), dtype=float)


def select_actions(rows: pd.DataFrame) -> pd.DataFrame:
    proposals = (
        rows.loc[rows["ev_net"].gt(0)]
        .sort_values(
            ["exchange", "signal_timestamp", "ev_net", "action"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["exchange", "signal_timestamp"])
    )
    selected: list[pd.DataFrame] = []
    for _, venue in proposals.groupby("exchange", sort=True):
        blocked_until = pd.Timestamp("1900", tz="UTC")
        for _, row in venue.iterrows():
            signal = pd.Timestamp(row["signal_timestamp"])
            if signal <= blocked_until:
                continue
            selected.append(row.to_frame().T)
            blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(minutes=5)
    return pd.concat(selected, ignore_index=True) if selected else rows.iloc[:0].copy()


def fit_action(train: pd.DataFrame, calibration: pd.DataFrame) -> FittedAction | None:
    if len(train) < 500 or len(calibration) < 160:
        return None
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    calibrate, admission = calibration.loc[times.lt(midpoint)], calibration.loc[times.ge(midpoint)]
    if min(len(calibrate), len(admission)) < 80:
        return None
    candidates: dict[str, FittedAction] = {}
    for kind in ("ridge", "xgboost"):
        model, calibrator = _fit_candidate(kind, train, calibrate)
        predicted = admission.assign(
            ev_net=np.asarray(calibrator.predict(_raw(model, admission)), dtype=float)
        )
        selected = select_actions(predicted)
        candidates[kind] = FittedAction(
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


def run(app: AppConfig, *, resume: bool = False, smoke: bool = False) -> dict[str, Any]:
    protocol = preregister()
    matrix = build_matrices(app, resume=resume)
    external = matrix.loc[matrix["exchange"].isin(EXCHANGES)].copy()
    scenarios: list[tuple[str, tuple[Any, ...]]] = []
    for held_out in EXCHANGES:
        folds = purged_expert_folds(
            external,
            train_weeks=52,
            calibration_weeks=4,
            test_weeks=4,
            step_weeks=4,
            embargo_hours=2,
        )
        scenarios.append((held_out, folds[-2:] if smoke else folds))
    total = sum(len(folds) for _, folds in scenarios)
    completed = 0
    outputs: list[pd.DataFrame] = []
    champions: dict[str, int] = {}
    for held_out, folds in scenarios:
        for number, fold in enumerate(folds, start=1):
            train = external.iloc[fold.train]
            calibration = external.iloc[fold.calibration]
            test = external.iloc[fold.test]
            train = train.loc[train["exchange"].ne(held_out)]
            calibration = calibration.loc[calibration["exchange"].ne(held_out)]
            test = test.loc[test["exchange"].eq(held_out)].copy()
            predicted: list[pd.DataFrame] = []
            for action in ("fade", "follow"):
                fitted = fit_action(
                    train.loc[train["action"].eq(action)],
                    calibration.loc[calibration["action"].eq(action)],
                )
                if fitted is None:
                    continue
                rows = test.loc[test["action"].eq(action)].copy()
                rows["ev_net"] = _predict(fitted, rows)
                rows["champion"] = fitted.champion
                predicted.append(rows)
                key = f"{held_out}:{action}:{fitted.champion}"
                champions[key] = champions.get(key, 0) + 1
            if predicted:
                rows = pd.concat(predicted, ignore_index=True)
                rows["held_out_exchange"] = held_out
                rows["outer_fold"] = number
                outputs.append(rows)
            completed += 1
            _status(
                "walk_forward",
                f"LOEO {held_out.upper()} fold {number}/{len(folds)}",
                20 + 75 * completed / max(total, 1),
            )
    candidates = pd.concat(outputs, ignore_index=True) if outputs else external.iloc[:0].copy()
    decisions = select_actions(candidates)
    audit = audit_policy(decisions)
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "verdict": "ALPHA_SHADOW_CANDIDATE" if audit["gates_passed"] else "FLAT",
        "deployable": False,
        "paper_enabled": False,
        "live_enabled": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "states": int(matrix["state_id"].nunique()),
        "paired_counterfactuals": len(matrix),
        "champions": champions,
        "audit": audit,
        "smoke": smoke,
        "warning": "discovery only; confirmation requires chronologically newer data",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(OOS_PATH, candidates)
    _atomic_json(REPORT_PATH, report)
    _status("complete", report["verdict"], 100)
    return report


def audit_policy(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"trades": 0, "gates_passed": False}
    base = return_metrics(rows, "target_net_r")
    stressed = rows.assign(stress=rows["gross_return_r"] - 2 * rows["cost_r_1x"])
    stress = return_metrics(stressed, "stress")
    lower = moving_block_lower_bound(
        rows["target_net_r"].to_numpy(float),
        block_size=min(20, len(rows)),
        seed=20260804,
    )
    by_exchange = {
        name: return_metrics(group, "target_net_r") for name, group in rows.groupby("exchange")
    }
    by_action = {
        name: return_metrics(group, "target_net_r") for name, group in rows.groupby("action")
    }
    months = pd.to_datetime(rows["signal_timestamp"], utc=True).dt.to_period("M")
    positive_months = float(rows["target_net_r"].groupby(months).sum().gt(0).mean())
    gates = {
        "trades_300": len(rows) >= 300,
        "expectancy_positive": base["expectancy_r"] > 0,
        "bootstrap_lower_positive": lower > 0,
        "profit_factor_1_15": base["profit_factor"] >= 1.15,
        "drawdown_8pct": base["max_drawdown"] <= 0.08,
        "costs_2x_nonnegative": stress["expectancy_r"] >= 0,
        "positive_month_majority": positive_months > 0.5,
        "all_three_exchanges_positive": len(by_exchange) == 3
        and all(item["expectancy_r"] > 0 for item in by_exchange.values()),
    }
    return {
        "trades": len(rows),
        "net_4bps": base,
        "stress_8bps": stress,
        "expectancy_lower_bound_r": lower,
        "positive_month_fraction": positive_months,
        "by_exchange": by_exchange,
        "by_action": by_action,
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "vwap_hours": VWAP_HOURS,
        "state_distance_atr": [MIN_DISTANCE_ATR, MAX_DISTANCE_ATR],
        "actions": {
            "fade": "opposite deviation; frozen VWAP target; 2 ATR stop",
            "follow": "with deviation; 2 ATR target; 2 ATR stop",
            "flat": "implicit zero EV",
        },
        "maximum_holding_bars": MAX_HOLDING_BARS,
        "features": list(FEATURES),
        "target": "net_return_r_at_4bps",
        "walk_forward_weeks": [52, 4, 4, 4],
        "model": "ridge_default_xgboost_gpu_challenger",
        "decision": "highest_calibrated_positive_ev_else_flat",
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V17 protocol changed after freezing")
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
    parser = argparse.ArgumentParser(description="V17 BTC VWAP FADE/FOLLOW/FLAT")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(load_config(arguments.config), resume=arguments.resume, smoke=arguments.smoke),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
