from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.config import load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v9 import spa_reality_check
from adaptive_bot.hybrid_policy_v19 import (
    BASE_COST_BPS,
    BINANCE_MATRIX_PATH,
    CONTROL_MATRIX_PATH,
    EXTERNAL_CONTROLS,
    FEATURES,
    _replace_with_retry,
    audit,
)

PROTOCOL = "hybrid_v20_binance_arm_specific_q"
ARMS = (("fade", "long"), ("fade", "short"), ("follow", "long"), ("follow", "short"))
ROOT = Path("data/ml/hybrid_v20")
OOS_PATH = ROOT / "oos_candidates.parquet"
DECISIONS_PATH = ROOT / "oos_decisions.parquet"
MODEL_ROOT = Path("data/models/expert_policy/v20")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
BUNDLE_PATH = MODEL_ROOT / "shadow_bundle.joblib"
REPORT_PATH = Path("data/reports/ml_hybrid_v20.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v20.status.json")


@dataclass(frozen=True)
class ArmModel:
    model: Any
    calibrator: IsotonicRegression
    champion: str
    admission_mse: float
    admission_ev_r: float
    uncertainty_margin_r: float


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    _replace_with_retry(temporary, path)


def _status(phase: str, detail: str, percent: float, **extra: Any) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def _xgboost() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:pseudohubererror",
        tree_method="hist",
        device="cuda",
        n_estimators=2_000,
        early_stopping_rounds=75,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        random_state=20260805,
        n_jobs=8,
    )


def _predict(model: Any, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(rows[list(FEATURES)]), dtype=float)


def _fit_raw(kind: str, train: pd.DataFrame) -> Any:
    if kind == "ridge":
        model = _ridge()
        model.fit(train[list(FEATURES)], train["target_net_r"])
        return model
    model = _xgboost()
    split = max(1, int(len(train) * 0.85))
    model.fit(
        train.iloc[:split][list(FEATURES)],
        train.iloc[:split]["target_net_r"],
        eval_set=[
            (
                train.iloc[split:][list(FEATURES)],
                train.iloc[split:]["target_net_r"],
            )
        ],
        verbose=False,
    )
    return model


def _positive_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[rows["ev_lcb"].gt(0)]


def fit_arm(train: pd.DataFrame, calibration: pd.DataFrame) -> ArmModel | None:
    if len(train) < 400 or len(calibration) < 120:
        return None
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    calibrate = calibration.loc[times.lt(midpoint)]
    admission = calibration.loc[times.ge(midpoint)]
    if min(len(calibrate), len(admission)) < 60:
        return None
    candidates: dict[str, ArmModel] = {}
    for kind in ("ridge", "xgboost"):
        model = _fit_raw(kind, train)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            _predict(model, calibrate), calibrate["target_net_r"].to_numpy(float)
        )
        prediction = np.asarray(calibrator.predict(_predict(model, admission)), dtype=float)
        residual = admission["target_net_r"].to_numpy(float) - prediction
        margin = max(
            0.0,
            -moving_block_lower_bound(residual, block_size=min(20, len(residual)), seed=20260805),
        )
        assessed = admission.assign(ev_net=prediction, ev_lcb=prediction - margin)
        selected = _positive_rows(assessed)
        candidates[kind] = ArmModel(
            model,
            calibrator,
            kind,
            float(mean_squared_error(admission["target_net_r"], prediction)),
            float(selected["target_net_r"].mean()) if len(selected) >= 20 else -np.inf,
            margin,
        )
    ridge, challenger = candidates["ridge"], candidates["xgboost"]
    if (
        challenger.admission_mse < ridge.admission_mse
        and challenger.admission_ev_r > ridge.admission_ev_r
    ):
        return challenger
    return ridge


def select_policy(rows: pd.DataFrame) -> pd.DataFrame:
    proposals = (
        rows.loc[rows["ev_lcb"].gt(0)]
        .sort_values(
            ["signal_timestamp", "ev_lcb", "action", "side"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates("signal_timestamp")
    )
    selected: list[pd.DataFrame] = []
    blocked_until = pd.Timestamp("1900", tz="UTC")
    for _, row in proposals.iterrows():
        signal = pd.Timestamp(row["signal_timestamp"])
        if signal <= blocked_until:
            continue
        selected.append(row.to_frame().T)
        blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(minutes=15)
    return pd.concat(selected, ignore_index=True) if selected else rows.iloc[:0].copy()


def walk_forward(matrix: pd.DataFrame, *, smoke: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    folds = purged_expert_folds(
        matrix,
        train_weeks=52,
        calibration_weeks=4,
        test_weeks=4,
        step_weeks=4,
        embargo_hours=6,
    )
    if smoke:
        folds = folds[-2:]
    outputs: list[pd.DataFrame] = []
    champions: dict[str, int] = {}
    total = len(folds) * len(ARMS)
    done = 0
    for fold_number, fold in enumerate(folds, start=1):
        for action, side in ARMS:
            train = matrix.iloc[fold.train]
            calibration = matrix.iloc[fold.calibration]
            test = matrix.iloc[fold.test]
            train = train.loc[train["action"].eq(action) & train["side"].eq(side)]
            calibration = calibration.loc[
                calibration["action"].eq(action) & calibration["side"].eq(side)
            ]
            rows = test.loc[test["action"].eq(action) & test["side"].eq(side)].copy()
            fitted = fit_arm(train, calibration)
            if fitted is not None and not rows.empty:
                rows["ev_net"] = fitted.calibrator.predict(_predict(fitted.model, rows))
                rows["ev_lcb"] = rows["ev_net"] - fitted.uncertainty_margin_r
                rows["champion"] = fitted.champion
                rows["outer_fold"] = fold_number
                outputs.append(rows)
                key = f"{action}_{side}:{fitted.champion}"
                champions[key] = champions.get(key, 0) + 1
            done += 1
            _status(
                "gpu_walk_forward",
                f"Fold {fold_number}/{len(folds)} {action.upper()} {side.upper()}",
                5 + 85 * done / max(total, 1),
                backend="cuda:0",
                completed_arms=done,
                total_arms=total,
            )
    return (pd.concat(outputs, ignore_index=True) if outputs else matrix.iloc[:0].copy()), champions


def multiple_comparison(candidates: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, float]:
    days = pd.date_range(
        pd.to_datetime(candidates["signal_timestamp"], utc=True).min().floor("D"),
        pd.to_datetime(candidates["signal_timestamp"], utc=True).max().floor("D"),
        freq="D",
    )

    def daily(rows: pd.DataFrame) -> pd.Series:
        if rows.empty:
            return pd.Series(0.0, index=days)
        index = pd.to_datetime(rows["signal_timestamp"], utc=True).dt.floor("D")
        return rows["target_net_r"].groupby(index).sum().reindex(days, fill_value=0.0)

    policies = {"model": daily(decisions), "flat": pd.Series(0.0, index=days)}
    for action, side in ARMS:
        arm = candidates.loc[candidates["action"].eq(action) & candidates["side"].eq(side)]
        policies[f"{action}_{side}"] = daily(select_policy(arm))
    return spa_reality_check(pd.DataFrame(policies), control_expert_id="fade_long")


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_exchange": "binance",
        "symbol": "BTCUSDT perpetual",
        "arms": [list(arm) for arm in ARMS],
        "flat_ev": 0,
        "features": list(FEATURES),
        "walk_forward_weeks": [52, 4, 4, 4],
        "model": "arm-specific Ridge champion; robust XGBoost CUDA challenger",
        "challenger_gate": "lower admission MSE and higher admission decision EV",
        "cost_bps": BASE_COST_BPS,
        "stress_cost_bps": BASE_COST_BPS * 2,
        "external_controls": list(EXTERNAL_CONTROLS),
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V20 protocol changed after freezing")
        return dict(existing)
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def run(*, smoke: bool = False) -> dict[str, Any]:
    protocol = preregister()
    matrix = pd.read_parquet(BINANCE_MATRIX_PATH)
    controls = pd.read_parquet(CONTROL_MATRIX_PATH)
    if not set(FEATURES).issubset(matrix):
        raise RuntimeError("V20 requires the complete causal V19 matrix")
    candidates, champions = walk_forward(matrix, smoke=smoke)
    decisions = select_policy(candidates)
    result = audit(decisions, controls)
    comparison = multiple_comparison(candidates, decisions)
    gates = dict(result["gates"])
    gates["spa_5pct"] = comparison["spa_pvalue"] <= 0.05
    gates["reality_check_5pct"] = comparison["reality_check_pvalue"] <= 0.05
    result["gates"] = gates
    result["gates_passed"] = all(gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "verdict": "BINANCE_ALPHA_SHADOW_CANDIDATE" if result["gates_passed"] else "FLAT",
        "deployable": False,
        "shadow_enabled": bool(result["gates_passed"]),
        "paper_enabled": False,
        "live_enabled": False,
        "primary_exchange": "binance",
        "candidate_count": len(ARMS),
        "champions": champions,
        "multiple_comparison": comparison,
        "audit": result,
        "smoke": smoke,
        "warning": "V19/V20 historical data are discovery; forward shadow is mandatory",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(OOS_PATH, candidates)
    _atomic_parquet(DECISIONS_PATH, decisions)
    _atomic_json(REPORT_PATH, report)
    if result["gates_passed"]:
        cutoff = pd.to_datetime(matrix["signal_timestamp"], utc=True).max() - pd.Timedelta(weeks=8)
        final_models = {}
        for action, side in ARMS:
            arm = matrix.loc[matrix["action"].eq(action) & matrix["side"].eq(side)]
            times = pd.to_datetime(arm["signal_timestamp"], utc=True)
            final_models[(action, side)] = fit_arm(
                arm.loc[times.lt(cutoff)], arm.loc[times.ge(cutoff)]
            )
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"protocol": protocol, "features": FEATURES, "models": final_models, "report": report},
            BUNDLE_PATH,
        )
    _status("complete", report["verdict"], 100, backend="cuda:0")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V20 Binance arm-specific BTC VWAP Q models")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    load_config(arguments.config)
    print(json.dumps(run(smoke=arguments.smoke), indent=2))


if __name__ == "__main__":
    main()
