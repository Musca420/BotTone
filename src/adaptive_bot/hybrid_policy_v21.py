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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v11 import btc_inventory, build_feature_frame, evaluate_expert
from adaptive_bot.hybrid_policy_v17 import action_expert, add_relational_features
from adaptive_bot.hybrid_policy_v19 import (
    BASE_COST_BPS,
    CONTEXT_PATH,
    EXTERNAL_CONTROLS,
    FEATURES,
    _outcomes,
    _replace_with_retry,
    attach_context,
    audit,
)
from adaptive_bot.hybrid_policy_v20 import multiple_comparison

PROTOCOL = "hybrid_v21_event_dueling_probabilistic_vwap"
TIMEFRAME_MINUTES = 15
VWAP_HOURS = 24
ROOT = Path("data/ml/hybrid_v21")
MATRIX_PATH = ROOT / "event_matrix.parquet"
STATE_PATH = ROOT / "event_states.parquet"
CONTROL_PATH = ROOT / "external_controls.parquet"
OOS_PATH = ROOT / "oos_candidates.parquet"
DECISIONS_PATH = ROOT / "oos_decisions.parquet"
MODEL_ROOT = Path("data/models/expert_policy/v21")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
BUNDLE_PATH = MODEL_ROOT / "shadow_bundle.joblib"
REPORT_PATH = Path("data/reports/ml_hybrid_v21.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v21.status.json")
EVENT_NAMES = {0: "rejection", 1: "extension", 2: "confirmed_reentry"}
ANCHOR_FEATURES = (
    "exact_rolling_vwap_distance_atr",
    "daily_vwap_distance_atr",
    "weekly_vwap_distance_atr",
    "session_vwap_distance_atr",
    "anchor_dispersion_bps",
    "anchor_confluence",
    "time_since_vwap_touch",
    "vwap_touches_24h",
    "event_code",
    "excursion_max_z",
    "excursion_bars",
    "deviation_side",
    "estimated_cost_r",
    "atr_pct",
)
MODEL_FEATURES = tuple(
    dict.fromkeys([name for name in FEATURES if name != "side_code"] + list(ANCHOR_FEATURES))
)


@dataclass(frozen=True)
class ProbabilityHead:
    model: Any | None
    constant: float | None
    calibrator: Any | None


@dataclass(frozen=True)
class DuelingModel:
    value_model: Any
    advantage_model: Any
    fade_probability: ProbabilityHead
    follow_probability: ProbabilityHead
    q_calibration: tuple[tuple[float, float], tuple[float, float]]
    margins: tuple[float, float]
    champion: str
    admission_mse: float
    admission_ev_r: float


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


def _consecutive(condition: pd.Series) -> pd.Series:
    groups = condition.ne(condition.shift()).cumsum()
    counts = condition.groupby(groups).cumcount() + 1
    return counts.where(condition, 0).astype(float)


def exact_vwap_anchors(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    timestamp = pd.to_datetime(result["timestamp"], utc=True)
    volume = pd.to_numeric(result["volume"], errors="coerce")
    quote = pd.to_numeric(result["quote_volume"], errors="coerce")
    rolling_bars = VWAP_HOURS * 60 // TIMEFRAME_MINUTES

    def cumulative(key: pd.Series) -> pd.Series:
        numerator = quote.groupby(key).cumsum()
        denominator = volume.groupby(key).cumsum().replace(0, np.nan)
        return numerator / denominator

    exact = quote.rolling(rolling_bars, min_periods=rolling_bars).sum() / volume.rolling(
        rolling_bars, min_periods=rolling_bars
    ).sum().replace(0, np.nan)
    daily = cumulative(timestamp.dt.floor("D"))
    week_start = timestamp.dt.floor("D") - pd.to_timedelta(timestamp.dt.weekday, unit="D")
    weekly = cumulative(week_start)
    session = cumulative(timestamp.dt.floor("8h"))
    old_vwap = result["vwap"].copy()
    result["vwap_approximation_error_bps"] = (exact - old_vwap) / exact * 10_000
    result["vwap"] = exact
    atr = result["atr"].replace(0, np.nan)
    for name, anchor in (
        ("exact_rolling_vwap", exact),
        ("daily_vwap", daily),
        ("weekly_vwap", weekly),
        ("session_vwap", session),
    ):
        result[name] = anchor
        result[f"{name}_distance_atr"] = (result["close"] - anchor) / atr
    anchors = result[["vwap", "daily_vwap", "weekly_vwap", "session_vwap"]]
    result["anchor_dispersion_bps"] = anchors.std(axis=1, ddof=0) / anchors.mean(axis=1) * 10_000
    result["anchor_confluence"] = anchors.lt(result["close"], axis=0).sum(axis=1) / 4
    result["distance_vwap_atr"] = (result["close"] - exact) / atr
    result["distance_vwap_pct"] = (result["close"] - exact) / exact
    result["vwap_zscore"] = (result["close"] - exact) / result["close"].rolling(
        rolling_bars, min_periods=rolling_bars
    ).std().replace(0, np.nan)
    for bars in (3, 6, 12):
        result[f"vwap_slope_{bars}"] = (exact - exact.shift(bars)) / (atr * max(bars, 1))
    result["bars_above_vwap"] = _consecutive(result["close"].gt(exact))
    result["bars_below_vwap"] = _consecutive(result["close"].lt(exact))
    crossing = result["close"].gt(exact).ne(result["close"].shift().gt(exact.shift()))
    result["crossed_vwap_12"] = crossing.rolling(12, min_periods=12).sum()
    touch = result["distance_vwap_atr"].abs().le(0.25)
    last_touch = np.full(len(result), np.nan)
    latest = -1
    for index, is_touch in enumerate(touch.fillna(False).to_numpy(bool)):
        if is_touch:
            latest = index
        if latest >= 0:
            last_touch[index] = index - latest
    result["time_since_vwap_touch"] = last_touch
    result["vwap_touches_24h"] = touch.rolling(rolling_bars, min_periods=rolling_bars).sum()
    result["atr_pct"] = result["atr"] / result["close"]
    result["estimated_cost_r"] = BASE_COST_BPS / (2 * result["atr_pct"] * 10_000).replace(0, np.nan)
    return add_relational_features(result)


def vwap_events(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    z = result["distance_vwap_atr"].to_numpy(float)
    safe = (
        result["data_valid"].fillna(False).astype(bool)
        & result["local_feature_coverage"].fillna(False).astype(bool)
        & result["context_coverage"].fillna(False).astype(bool)
        & ~result["regime_code"].isin((3.0, 4.0))
    ).to_numpy(bool)
    codes = np.full(len(result), -1, dtype=int)
    maxima = np.full(len(result), np.nan)
    bars = np.full(len(result), np.nan)
    active_side = 0
    maximum = 0.0
    duration = 0
    extension_emitted = False
    reentry_emitted = False
    last_touch = -1
    previous_abs = np.nan
    for index, value in enumerate(z):
        absolute = abs(value)
        side = 1 if value > 0 else -1
        if not safe[index] or not np.isfinite(value):
            active_side = 0
            previous_abs = np.nan
            continue
        if absolute <= 0.25:
            active_side = 0
            maximum = 0.0
            duration = 0
            extension_emitted = False
            reentry_emitted = False
            last_touch = index
            previous_abs = absolute
            continue
        if active_side == 0 and absolute >= 0.5:
            active_side = side
            maximum = absolute
            duration = 1
            if 0 < index - last_touch <= 4:
                codes[index] = 0
        elif active_side == side:
            maximum = max(maximum, absolute)
            duration += 1
        else:
            active_side = side
            maximum = absolute
            duration = 1
            extension_emitted = False
            reentry_emitted = False
        if codes[index] < 0 and not extension_emitted and absolute >= 1.0:
            codes[index] = 1
            extension_emitted = True
        if (
            codes[index] < 0
            and extension_emitted
            and not reentry_emitted
            and maximum >= 1.5
            and absolute <= 1.25
            and np.isfinite(previous_abs)
            and absolute < previous_abs
        ):
            codes[index] = 2
            reentry_emitted = True
        if codes[index] >= 0:
            maxima[index] = maximum
            bars[index] = duration
        previous_abs = absolute
    result["event_code"] = codes.astype(float)
    result["event_name"] = pd.Series(codes, index=result.index).map(EVENT_NAMES)
    result["event_signal"] = codes >= 0
    result["excursion_max_z"] = maxima
    result["excursion_bars"] = bars
    result["deviation_side"] = np.sign(result["distance_vwap_atr"])
    return result


def build_event_matrix(app: AppConfig, *, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resume and MATRIX_PATH.exists() and CONTROL_PATH.exists():
        matrix = pd.read_parquet(MATRIX_PATH)
        if set(MODEL_FEATURES).issubset(matrix):
            return matrix, pd.read_parquet(CONTROL_PATH)
    inventory = btc_inventory()
    source = Path(inventory["binance"]["path"])
    raw, features = build_feature_frame(
        source, app, "binance", timeframe_minutes=TIMEFRAME_MINUTES, vwap_hours=VWAP_HOURS
    )
    context = pd.read_parquet(CONTEXT_PATH)
    features = vwap_events(exact_vwap_anchors(attach_context(features, context)))
    coverage = features[list(MODEL_FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    event = features["event_signal"].astype(bool) & coverage
    outputs: list[pd.DataFrame] = []
    z = features["distance_vwap_atr"]
    masks = {
        ("fade", "short"): event & z.gt(0),
        ("follow", "long"): event & z.gt(0),
        ("fade", "long"): event & z.lt(0),
        ("follow", "short"): event & z.lt(0),
    }
    for (action, side), mask in masks.items():
        result = evaluate_expert(
            features,
            raw,
            action_expert(action, side),
            cost_bps=BASE_COST_BPS,
            entry_mask=mask,
            timeframe_minutes=TIMEFRAME_MINUTES,
            vwap_hours=VWAP_HOURS,
        )
        if not result.empty:
            result["action"] = action
            result["target_net_r"] = result["net_return_r_1x"]
            result["state_id"] = pd.to_datetime(result["signal_timestamp"], utc=True).astype(str)
            outputs.append(result)
    matrix = pd.concat(outputs, ignore_index=True)
    paired = matrix.groupby("state_id")["action"].transform("nunique").eq(2)
    matrix = matrix.loc[paired].sort_values(["signal_timestamp", "action"]).reset_index(drop=True)
    _atomic_parquet(MATRIX_PATH, matrix)
    requested = matrix[["signal_timestamp", "action", "side"]].drop_duplicates()
    controls = []
    for number, exchange in enumerate(EXTERNAL_CONTROLS, start=1):
        _status("controls", f"Synchronized {exchange.upper()} {number}/2", 25 + number * 2)
        controls.append(
            _outcomes(app, exchange, Path(inventory[exchange]["path"]), requested=requested)
        )
    control = pd.concat(controls, ignore_index=True)
    _atomic_parquet(CONTROL_PATH, control)
    return matrix, control


def event_states(matrix: pd.DataFrame) -> pd.DataFrame:
    first = matrix.sort_values(["signal_timestamp", "action"]).drop_duplicates("state_id")
    targets = matrix.pivot(index="state_id", columns="action", values="target_net_r").rename(
        columns={"fade": "y_fade", "follow": "y_follow"}
    )
    gross = matrix.pivot(index="state_id", columns="action", values="gross_return_r").rename(
        columns={"fade": "gross_fade", "follow": "gross_follow"}
    )
    exits = matrix.pivot(index="state_id", columns="action", values="exit_timestamp").rename(
        columns={"fade": "exit_fade", "follow": "exit_follow"}
    )
    sides = matrix.pivot(index="state_id", columns="action", values="side").rename(
        columns={"fade": "side_fade", "follow": "side_follow"}
    )
    costs = matrix.pivot(index="state_id", columns="action", values="cost_r_1x").rename(
        columns={"fade": "cost_fade", "follow": "cost_follow"}
    )
    result = first.set_index("state_id").join([targets, gross, exits, sides, costs]).reset_index()
    result["exit_timestamp"] = result[["exit_fade", "exit_follow"]].max(axis=1)
    result["target_value"] = (result["y_fade"] + result["y_follow"]) / 2
    result["target_advantage"] = (result["y_fade"] - result["y_follow"]) / 2
    return result.sort_values("signal_timestamp").reset_index(drop=True)


def _regressor(kind: str) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    return XGBRegressor(
        objective="reg:pseudohubererror",
        tree_method="hist",
        device="cuda",
        n_estimators=1_500,
        early_stopping_rounds=75,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=20260805,
        n_jobs=8,
    )


def _fit_regressor(kind: str, rows: pd.DataFrame, target: str) -> Any:
    model = _regressor(kind)
    if kind == "ridge":
        model.fit(rows[list(MODEL_FEATURES)], rows[target])
    else:
        split = int(len(rows) * 0.85)
        model.fit(
            rows.iloc[:split][list(MODEL_FEATURES)],
            rows.iloc[:split][target],
            eval_set=[(rows.iloc[split:][list(MODEL_FEATURES)], rows.iloc[split:][target])],
            verbose=False,
        )
    return model


def _fit_probability(kind: str, rows: pd.DataFrame, target: str) -> ProbabilityHead:
    values = rows[target].gt(0).astype(int)
    if values.nunique() < 2:
        return ProbabilityHead(None, float(values.iloc[0]), None)
    if kind == "ridge":
        model: Any = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1_000))
    else:
        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            random_state=20260805,
            n_jobs=8,
        )
    model.fit(rows[list(MODEL_FEATURES)], values)
    return ProbabilityHead(model, None, None)


def _probability(head: ProbabilityHead, rows: pd.DataFrame) -> np.ndarray:
    if head.constant is not None:
        raw = np.full(len(rows), head.constant)
    else:
        if head.model is None:
            raise RuntimeError("probability head has neither model nor constant")
        raw = np.asarray(head.model.predict_proba(rows[list(MODEL_FEATURES)])[:, 1])
    if head.calibrator is None:
        return raw
    return np.asarray(head.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1])


def _affine(raw: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    variance = float(np.var(raw))
    slope = 0.0 if variance <= 1e-12 else float(np.cov(raw, target, ddof=0)[0, 1] / variance)
    slope = float(np.clip(slope, 0.0, 1.0))
    return float(np.mean(target - slope * raw)), slope


def _raw_q(
    value_model: Any, advantage_model: Any, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value_model.predict(rows[list(MODEL_FEATURES)]), dtype=float)
    advantage = np.asarray(advantage_model.predict(rows[list(MODEL_FEATURES)]), dtype=float)
    return value + advantage, value - advantage


def prediction_rows(
    states: pd.DataFrame,
    q: tuple[np.ndarray, np.ndarray],
    probability: tuple[np.ndarray, np.ndarray],
    calibration: tuple[tuple[float, float], tuple[float, float]],
    margins: tuple[float, float],
) -> pd.DataFrame:
    output = []
    for index, action in enumerate(("fade", "follow")):
        intercept, slope = calibration[index]
        rows = states.copy()
        rows["action"] = action
        rows["ev_net"] = intercept + slope * q[index]
        rows["ev_lcb"] = rows["ev_net"] - margins[index]
        rows["p_positive"] = probability[index]
        rows["target_net_r"] = rows[f"y_{action}"]
        rows["gross_return_r"] = rows[f"gross_{action}"]
        rows["exit_timestamp"] = rows[f"exit_{action}"]
        rows["side"] = rows[f"side_{action}"]
        rows["cost_r_1x"] = rows[f"cost_{action}"]
        output.append(rows)
    return pd.concat(output, ignore_index=True)


def select_policy(rows: pd.DataFrame) -> pd.DataFrame:
    proposals = (
        rows.loc[rows["ev_lcb"].gt(0) & rows["p_positive"].gt(0.5)]
        .sort_values(["signal_timestamp", "ev_lcb", "action"], ascending=[True, False, True])
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


def _fit_candidate(
    kind: str, train: pd.DataFrame, calibration: pd.DataFrame
) -> DuelingModel | None:
    if len(train) < 500 or len(calibration) < 100:
        return None
    value_model = _fit_regressor(kind, train, "target_value")
    advantage_model = _fit_regressor(kind, train, "target_advantage")
    heads = [
        _fit_probability(kind, train, "y_fade"),
        _fit_probability(kind, train, "y_follow"),
    ]
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    fit_rows = calibration.loc[times.lt(midpoint)]
    admission = calibration.loc[times.ge(midpoint)]
    if min(len(fit_rows), len(admission)) < 40:
        return None
    raw_fit = _raw_q(value_model, advantage_model, fit_rows)
    raw_admission = _raw_q(value_model, advantage_model, admission)
    q_calibration = (
        _affine(raw_fit[0], fit_rows["y_fade"].to_numpy(float)),
        _affine(raw_fit[1], fit_rows["y_follow"].to_numpy(float)),
    )
    calibrated_heads = []
    for head, target in zip(heads, ("y_fade", "y_follow"), strict=True):
        raw_probability = _probability(head, fit_rows)
        binary = fit_rows[target].gt(0).astype(int)
        calibrator = None
        if binary.nunique() == 2:
            calibrator = LogisticRegression(C=1.0).fit(raw_probability.reshape(-1, 1), binary)
        calibrated_heads.append(ProbabilityHead(head.model, head.constant, calibrator))
    q_admission = []
    margins = []
    for index, target in enumerate(("y_fade", "y_follow")):
        intercept, slope = q_calibration[index]
        prediction = intercept + slope * raw_admission[index]
        q_admission.append(prediction)
        residual = admission[target].to_numpy(float) - prediction
        margins.append(
            max(
                0.0,
                -moving_block_lower_bound(
                    residual, block_size=min(20, len(residual)), seed=20260805
                ),
            )
        )
    probabilities = (
        _probability(calibrated_heads[0], admission),
        _probability(calibrated_heads[1], admission),
    )
    assessed = prediction_rows(
        admission,
        raw_admission,
        probabilities,
        q_calibration,
        (margins[0], margins[1]),
    )
    selected = select_policy(assessed)
    mse = float(
        np.mean(
            [
                mean_squared_error(admission["y_fade"], q_admission[0]),
                mean_squared_error(admission["y_follow"], q_admission[1]),
            ]
        )
    )
    return DuelingModel(
        value_model,
        advantage_model,
        calibrated_heads[0],
        calibrated_heads[1],
        q_calibration,
        (margins[0], margins[1]),
        kind,
        mse,
        float(selected["target_net_r"].mean()) if len(selected) >= 20 else -np.inf,
    )


def fit_model(train: pd.DataFrame, calibration: pd.DataFrame) -> DuelingModel | None:
    ridge = _fit_candidate("ridge", train, calibration)
    challenger = _fit_candidate("xgboost", train, calibration)
    if ridge is None:
        return challenger
    if challenger is not None and (
        challenger.admission_mse < ridge.admission_mse
        and challenger.admission_ev_r > ridge.admission_ev_r
    ):
        return challenger
    return ridge


def walk_forward(states: pd.DataFrame, *, smoke: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    folds = purged_expert_folds(
        states,
        train_weeks=52,
        calibration_weeks=4,
        test_weeks=4,
        step_weeks=4,
        embargo_hours=6,
    )
    if smoke:
        folds = folds[-2:]
    outputs = []
    champions: dict[str, int] = {}
    for number, fold in enumerate(folds, start=1):
        fitted = fit_model(states.iloc[fold.train], states.iloc[fold.calibration])
        if fitted is not None:
            test = states.iloc[fold.test].copy()
            q = _raw_q(fitted.value_model, fitted.advantage_model, test)
            probabilities = (
                _probability(fitted.fade_probability, test),
                _probability(fitted.follow_probability, test),
            )
            rows = prediction_rows(test, q, probabilities, fitted.q_calibration, fitted.margins)
            rows["outer_fold"] = number
            rows["champion"] = fitted.champion
            outputs.append(rows)
            champions[fitted.champion] = champions.get(fitted.champion, 0) + 1
        _status(
            "gpu_walk_forward",
            f"Fold {number}/{len(folds)} dueling EV + probability",
            30 + 60 * number / max(len(folds), 1),
            backend="cuda:0",
        )
    return (pd.concat(outputs, ignore_index=True) if outputs else states.iloc[:0].copy()), champions


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_exchange": "binance",
        "symbol": "BTCUSDT perpetual",
        "primary_vwap": "exact rolling quote_volume/base_volume 24h",
        "context_vwaps": ["UTC daily", "UTC weekly", "8h session"],
        "events": list(EVENT_NAMES.values()),
        "actions": ["fade", "follow", "flat"],
        "model": "dueling value/advantage EV plus P(net>0)",
        "baseline": "Ridge/LogisticRegression",
        "challenger": "XGBoost Pseudo-Huber/Logistic GPU",
        "features": list(MODEL_FEATURES),
        "walk_forward_weeks": [52, 4, 4, 4],
        "cost_bps": BASE_COST_BPS,
        "stress_cost_bps": BASE_COST_BPS * 2,
        "decision": "max LCB with P(net>0)>0.5, otherwise FLAT",
        "external_controls": list(EXTERNAL_CONTROLS),
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V21 protocol changed after freezing")
        return dict(existing)
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def run(app: AppConfig, *, resume: bool, smoke: bool) -> dict[str, Any]:
    protocol = preregister()
    _status("events", "Exact multi-anchor VWAP event matrix", 5)
    matrix, controls = build_event_matrix(app, resume=resume)
    states = event_states(matrix)
    _atomic_parquet(STATE_PATH, states)
    candidates, champions = walk_forward(states, smoke=smoke)
    decisions = select_policy(candidates)
    result = audit(decisions, controls)
    comparison = (
        multiple_comparison(candidates, decisions)
        if not candidates.empty
        else {
            "spa_pvalue": 1.0,
            "reality_check_pvalue": 1.0,
        }
    )
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
        "event_states": len(states),
        "event_types": states["event_name"].value_counts().to_dict(),
        "champions": champions,
        "multiple_comparison": comparison,
        "audit": result,
        "smoke": smoke,
        "warning": "historical discovery; forward Binance shadow required",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(OOS_PATH, candidates)
    _atomic_parquet(DECISIONS_PATH, decisions)
    _atomic_json(REPORT_PATH, report)
    if result["gates_passed"]:
        cutoff = pd.to_datetime(states["signal_timestamp"], utc=True).max() - pd.Timedelta(weeks=8)
        times = pd.to_datetime(states["signal_timestamp"], utc=True)
        final = fit_model(states.loc[times.lt(cutoff)], states.loc[times.ge(cutoff)])
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"protocol": protocol, "features": MODEL_FEATURES, "model": final, "report": report},
            BUNDLE_PATH,
        )
    _status("complete", report["verdict"], 100, backend="cuda:0")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V21 event-driven multi-VWAP dueling policy")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    try:
        report = run(load_config(arguments.config), resume=arguments.resume, smoke=arguments.smoke)
    except Exception as error:
        _status("failed", f"{type(error).__name__}: {error}", 0)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
