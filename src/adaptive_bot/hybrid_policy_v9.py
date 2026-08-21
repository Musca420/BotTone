from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import optuna
import pandas as pd
from arch.bootstrap import SPA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.config import AppConfig, MachineLearningConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy import _resample_alpha
from adaptive_bot.ml_research import FEATURE_COLUMNS, build_ml_features

PROTOCOL_VERSION = "adaptive_range_hybrid_v9_btc"
EXCHANGES = ("binance", "okx", "bybit", "bitunix")
SIDES = ("long", "short")
V8_REPORT = Path("data/reports/ml_hybrid_v8.json")
V8_MANIFEST = Path("data/ml/hybrid_v7/alpha_manifest.json")
ROOT = Path("data/ml/hybrid_v9")
PROTOCOL_PATH = Path("data/models/expert_policy/v9/protocol.json")
REPORT_PATH = Path("data/reports/ml_hybrid_v9.json")
CPU_GPU_PREDICTION_ATOL = 1e-4
MINIMUM_CONFIRMATION_DAYS = 28
TEMPORAL_ENSEMBLE_MODELS = 10
NEW_FEATURES = (
    "funding_zscore",
    "cross_exchange_return_median",
    "cross_exchange_return_dispersion",
)
UNAVAILABLE_HISTORICAL_FEATURES = {
    "taker_imbalance_15m": "not archived on all four BTC venues",
    "taker_imbalance_1h": "not archived on all four BTC venues",
    "trade_count_zscore": "trade count was discarded by V8 archive normalization",
    "aggressive_volume_zscore": "aggressor volume was not archived on all four venues",
    "open_interest_change_1h": "no common official OI history was archived",
    "open_interest_change_4h": "no common official OI history was archived",
    "open_interest_change_24h": "no common official OI history was archived",
    "return_open_interest_interaction": "open-interest history is unavailable",
    "mark_index_basis_bps": "index price was not archived; mark-last is not a substitute",
    "basis_change_1h": "index price was not archived",
    "time_to_next_funding": "next funding timestamp was not archived historically",
    "cross_exchange_taker_imbalance_median": "taker imbalance is unavailable",
    "cross_exchange_taker_imbalance_dispersion": "taker imbalance is unavailable",
}
EXPERT_PARAMETERS = (
    "side_code",
    "breakout_bars",
    "stop_atr_parameter",
    "target_atr_parameter",
    "trailing_atr_parameter",
    "maximum_holding_bars",
)
MODEL_FEATURES = (*FEATURE_COLUMNS, *NEW_FEATURES, *EXPERT_PARAMETERS)


@dataclass(frozen=True)
class V9Expert:
    name: str
    side: Literal["long", "short"]
    entry_kind: Literal["breakout", "trend_continuation", "volatility", "multi_horizon"]
    breakout_bars: int
    stop_atr: float
    target_atr: float | None
    trailing_atr: float | None
    maximum_holding_bars: int

    @property
    def expert_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"v9-{self.side[0]}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class BarrierOutcome:
    return_r: float
    exit_index: int
    exit_price: float
    reason: Literal["stop", "target", "time"]
    stop_path: tuple[float, ...]


def v9_experts() -> tuple[V9Expert, ...]:
    templates = (
        ("v8_control", "breakout", 24, 2.0, 2.0, None, 32),
        ("trend_continuation", "trend_continuation", 48, 2.0, None, 2.0, 64),
        ("volatility_confirmed", "volatility", 24, 1.5, 3.0, None, 32),
        ("multi_horizon", "multi_horizon", 0, 2.0, None, 2.5, 48),
    )
    return tuple(
        V9Expert(name, cast(Any, side), cast(Any, kind), breakout, stop, target, trail, hold)
        for side in SIDES
        for name, kind, breakout, stop, target, trail, hold in templates
    )


def expert_universe_hash(experts: tuple[V9Expert, ...] | None = None) -> str:
    payload = [
        asdict(expert) | {"expert_id": expert.expert_id} for expert in experts or v9_experts()
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preregister_v9(
    *,
    now: datetime | None = None,
    config_path: Path = Path("configs/bitunix_btc_futures_simulated.yaml"),
) -> dict[str, Any]:
    if not V8_REPORT.exists() or not V8_MANIFEST.exists():
        raise RuntimeError("V9 requires frozen V8 report and BTC source manifest")
    created = (now or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    v8 = json.loads(V8_REPORT.read_text(encoding="utf-8"))
    payload = {
        "protocol": PROTOCOL_VERSION,
        "run_id": created.strftime("hybrid-v9-%Y%m%dT%H%M%SZ"),
        "created_at": created.isoformat(),
        "discovery_cutoff": v8["updated_at"],
        "v8_report_sha256": _file_sha256(V8_REPORT),
        "source_manifest_sha256": _file_sha256(V8_MANIFEST),
        "training_source_sha256": _file_sha256(Path(__file__)),
        "runtime_config_sha256": _file_sha256(config_path),
        "symbols": ["BTCUSDT"],
        "exchanges": list(EXCHANGES),
        "candidate_count": len(v9_experts()),
        "experts": [asdict(expert) | {"expert_id": expert.expert_id} for expert in v9_experts()],
        "expert_universe_sha256": expert_universe_hash(),
        "features": {
            "model": list(MODEL_FEATURES),
            "available_new": list(NEW_FEATURES),
            "shadow_only": UNAVAILABLE_HISTORICAL_FEATURES,
            "missing_policy": "drop_observation_fail_closed_never_zero_impute",
            "availability": "bar_close_timestamp",
        },
        "audit": {
            "walk_forward_weeks": [52, 4, 4, 4],
            "leave_one_exchange_out": True,
            "ridge_default": True,
            "xgboost": {"device": "cuda", "tree_method": "hist", "trials_per_side": 4},
            "temporal_bootstrap_ensemble_models": TEMPORAL_ENSEMBLE_MODELS,
            "bootstrap": "stationary_temporal_blocks",
            "multiple_comparison": "SPA_and_Reality_Check_vs_FLAT_and_v8_control",
            "post_v8_confirmation_days": MINIMUM_CONFIRMATION_DAYS,
        },
        "gates": {
            "minimum_bitunix_trades": 100,
            "expectancy_positive": True,
            "expectancy_lower_bound_positive": True,
            "minimum_profit_factor": 1.15,
            "maximum_drawdown": 0.08,
            "stress_2x_non_negative": True,
            "positive_window_majority": True,
            "positive_exchanges": 3,
            "bitunix_required": True,
            "spa_pvalue_maximum": 0.05,
        },
        "execution": {"minimum_distinct_days": 30, "enabled": False},
        "holdout": {"status": "sealed", "opened": False},
        "automatic_live": False,
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        immutable = (
            "protocol",
            "discovery_cutoff",
            "symbols",
            "exchanges",
            "experts",
            "audit",
            "gates",
            "runtime_config_sha256",
        )
        if any(existing.get(key) != payload.get(key) for key in immutable):
            raise RuntimeError("V9 protocol is already frozen with a different specification")
        return cast(dict[str, Any], existing)
    _exclusive_json(PROTOCOL_PATH, payload)
    return payload


def btc_data_inventory() -> dict[str, Any]:
    manifest_path = ROOT / "alpha_manifest.json"
    manifest = json.loads(
        (manifest_path if manifest_path.exists() else V8_MANIFEST).read_text(encoding="utf-8")
    )
    rows: dict[str, Any] = {}
    for item in manifest["files"]:
        if item["symbol"] != "BTCUSDT":
            continue
        path = Path(item["path"])
        frame = pd.read_parquet(path, columns=["timestamp"])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        rows[item["exchange"]] = {
            "path": str(path),
            "rows": len(frame),
            "start": timestamps.min().isoformat(),
            "end": timestamps.max().isoformat(),
            "sha256": _file_sha256(path),
            "fields": _parquet_columns(path),
        }
    return rows


def confirmation_readiness(protocol: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    cutoff = pd.Timestamp(protocol["discovery_cutoff"])
    post_days = {
        exchange: max(0, (pd.Timestamp(item["end"]) - cutoff).days + 1)
        for exchange, item in inventory.items()
    }
    ready = set(inventory) == set(EXCHANGES) and all(
        days >= MINIMUM_CONFIRMATION_DAYS for days in post_days.values()
    )
    return {
        "ready": ready,
        "required_days": MINIMUM_CONFIRMATION_DAYS,
        "post_v8_days": post_days,
        "reason": None if ready else "insufficient_chronologically_post_v8_data",
    }


def build_local_feature_frames(app: AppConfig) -> dict[str, pd.DataFrame]:
    inventory = btc_data_inventory()
    feature_app = app.model_copy(
        update={"strategy": app.strategy.model_copy(update={"timeframe_minutes": 15})}
    )
    frames: dict[str, pd.DataFrame] = {}
    for exchange in EXCHANGES:
        raw = pd.read_parquet(inventory[exchange]["path"])
        bars = _resample_alpha(raw, 15)
        features, _ = build_ml_features(bars, feature_app, alpha_gross=True)
        features["return_48"] = features["close"].pct_change(48, fill_method=None)
        funding = features["funding_rate"].astype(float)
        mean = funding.shift(1).rolling(96 * 30, min_periods=96).mean()
        deviation = funding.shift(1).rolling(96 * 30, min_periods=96).std()
        features["funding_zscore"] = (funding - mean) / deviation.replace(0, np.nan)
        features["feature_available_at"] = pd.to_datetime(
            features["timestamp"], utc=True
        ) + pd.Timedelta(minutes=15)
        features["signal_timestamp"] = features["feature_available_at"]
        features["funding_coverage"] = features["funding_zscore"].notna()
        features["exchange"] = exchange
        features["symbol"] = "BTCUSDT"
        frames[exchange] = features
    return frames


def add_cross_exchange_features(
    frames: dict[str, pd.DataFrame], *, allowed_exchanges: tuple[str, ...] = EXCHANGES
) -> dict[str, pd.DataFrame]:
    if len(allowed_exchanges) < 3 or not set(allowed_exchanges).issubset(frames):
        raise ValueError("cross-exchange features require at least three available BTC venues")
    returns = pd.concat(
        [
            frame[["timestamp", "return_1"]].assign(exchange=exchange)
            for exchange, frame in frames.items()
            if exchange in allowed_exchanges
        ],
        ignore_index=True,
    ).pivot(index="timestamp", columns="exchange", values="return_1")
    result: dict[str, pd.DataFrame] = {}
    for exchange, frame in frames.items():
        peers = [name for name in allowed_exchanges if name != exchange]
        source = (
            returns[peers] if exchange in allowed_exchanges else returns[list(allowed_exchanges)]
        )
        stats = pd.DataFrame(
            {
                "timestamp": returns.index.to_numpy(),
                "cross_exchange_return_median": source.median(axis=1, skipna=False),
                "cross_exchange_return_dispersion": source.std(axis=1, ddof=0, skipna=False),
            }
        ).reset_index(drop=True)
        joined = frame.merge(stats, on="timestamp", how="left", validate="one_to_one")
        joined["cross_exchange_coverage"] = joined[list(NEW_FEATURES[1:])].notna().all(axis=1)
        joined["lookahead_valid"] = pd.to_datetime(joined["feature_available_at"], utc=True).le(
            pd.to_datetime(joined["signal_timestamp"], utc=True)
        )
        joined["feature_coverage"] = (
            joined["funding_coverage"]
            & joined["cross_exchange_coverage"]
            & joined["lookahead_valid"]
        )
        result[exchange] = joined
    return result


def build_expert_matrix(frames: dict[str, pd.DataFrame], *, base_cost_bps: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for exchange, features in frames.items():
        if exchange not in EXCHANGES:
            continue
        for expert in v9_experts():
            eligible = _expert_entry_mask(features, expert)
            for index_value in np.flatnonzero(eligible.to_numpy(dtype=bool)):
                index = int(index_value)
                outcome = short_long_barrier_outcome(features, index, expert)
                if outcome is None:
                    continue
                row = features.iloc[[index]].copy()
                risk_bps = (
                    expert.stop_atr
                    * float(features.iloc[index]["atr"])
                    / float(features.iloc[index + 1]["open"])
                    * 10_000
                )
                if risk_bps <= 0:
                    continue
                row["expert_id"] = expert.expert_id
                row["family"] = expert.name
                row["side"] = expert.side
                row["timeframe_minutes"] = 15
                row["exit_timestamp"] = pd.Timestamp(
                    features.iloc[outcome.exit_index]["timestamp"]
                ) + pd.Timedelta(minutes=15)
                row["net_return_r"] = outcome.return_r - base_cost_bps / risk_bps
                row["net_return_r_2x"] = outcome.return_r - 2 * base_cost_bps / risk_bps
                row["side_code"] = 1.0 if expert.side == "long" else -1.0
                row["breakout_bars"] = expert.breakout_bars
                row["stop_atr_parameter"] = expert.stop_atr
                row["target_atr_parameter"] = expert.target_atr or 0.0
                row["trailing_atr_parameter"] = expert.trailing_atr or 0.0
                row["maximum_holding_bars"] = expert.maximum_holding_bars
                rows.append(row)
    matrix = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not matrix.empty:
        required = [*MODEL_FEATURES, "feature_available_at", "signal_timestamp"]
        finite = matrix[list(MODEL_FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        matrix = matrix.loc[
            matrix["feature_coverage"].astype(bool)
            & matrix["lookahead_valid"].astype(bool)
            & finite
        ].dropna(subset=required)
    return matrix.reset_index(drop=True)


def _expert_entry_mask(features: pd.DataFrame, expert: V9Expert) -> pd.Series:
    direction = 1 if expert.side == "long" else -1
    close = features["close"].astype(float)
    prior = (
        features["high"].shift(1).rolling(expert.breakout_bars).max()
        if direction > 0 and expert.breakout_bars
        else features["low"].shift(1).rolling(expert.breakout_bars).min()
        if expert.breakout_bars
        else None
    )
    breakout = (
        close.gt(prior)
        if direction > 0 and prior is not None
        else close.lt(prior)
        if prior is not None
        else pd.Series(True, index=features.index)
    )
    if expert.entry_kind == "trend_continuation":
        eligible = breakout & features["adx"].ge(20)
        eligible &= features["ema20_slope"].mul(direction).gt(0)
        eligible &= features["ema50_slope"].mul(direction).gt(0)
    elif expert.entry_kind == "volatility":
        eligible = breakout & features["adx"].ge(20) & features["adx_slope"].gt(0)
        eligible &= features["atr_percentile"].between(40, 90, inclusive="both")
        eligible &= features["relative_volume"].gt(1)
    elif expert.entry_kind == "multi_horizon":
        eligible = features["return_12"].mul(direction).gt(0)
        eligible &= features["return_48"].mul(direction).gt(0)
        eligible &= features["ema50_slope"].mul(direction).gt(0)
        eligible &= features["atr_percentile"].le(90)
    else:
        eligible = breakout & features["adx"].ge(20)
    return (
        eligible
        & features["data_valid"].fillna(False).astype(bool)
        & features["feature_coverage"].fillna(False).astype(bool)
    )


def short_long_barrier_outcome(
    features: pd.DataFrame, signal_index: int, expert: V9Expert
) -> BarrierOutcome | None:
    entry_index = signal_index + 1
    final_index = signal_index + expert.maximum_holding_bars
    if entry_index >= len(features) or final_index >= len(features):
        return None
    atr = float(features.iloc[signal_index]["atr"])
    entry = float(features.iloc[entry_index]["open"])
    if not math.isfinite(atr) or atr <= 0 or not math.isfinite(entry) or entry <= 0:
        return None
    direction = 1 if expert.side == "long" else -1
    stop = entry - direction * expert.stop_atr * atr
    target = entry + direction * expert.target_atr * atr if expert.target_atr else None
    stop_path = [stop]
    for future in range(entry_index, final_index + 1):
        bar = features.iloc[future]
        hit_stop = float(bar["low"]) <= stop if direction > 0 else float(bar["high"]) >= stop
        hit_target = target is not None and (
            float(bar["high"]) >= target if direction > 0 else float(bar["low"]) <= target
        )
        if hit_stop or hit_target:
            reason: Literal["stop", "target"] = "stop" if hit_stop else "target"
            price = (
                max(stop, float(bar["open"]))
                if direction < 0 and hit_stop
                else min(stop, float(bar["open"]))
                if direction > 0 and hit_stop
                else cast(float, target)
            )
            return BarrierOutcome(
                direction * (price - entry) / (expert.stop_atr * atr),
                future,
                price,
                reason,
                tuple(stop_path),
            )
        if expert.trailing_atr is not None:
            current_atr = float(bar["atr"])
            proposed = float(bar["close"]) - direction * expert.trailing_atr * current_atr
            stop = max(stop, proposed) if direction > 0 else min(stop, proposed)
            stop_path.append(stop)
    price = float(features.iloc[final_index]["close"])
    return BarrierOutcome(
        direction * (price - entry) / (expert.stop_atr * atr),
        final_index,
        price,
        "time",
        tuple(stop_path),
    )


def fit_v9_fold(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    side: Literal["long", "short"],
    config: MachineLearningConfig,
) -> dict[str, Any]:
    train = _complete_side(training, side)
    calibrate = _complete_side(calibration, side)
    if len(train) < 300 or len(calibrate) < 60:
        return {"enabled": False, "reason": "insufficient_complete_rows"}
    x_train = train[list(MODEL_FEATURES)].to_numpy(dtype=float)
    y_train = train["net_return_r"].to_numpy(dtype=float)
    x_cal = calibrate[list(MODEL_FEATURES)].to_numpy(dtype=float)
    y_cal = calibrate["net_return_r"].to_numpy(dtype=float)
    weights = _timestamp_weights(train)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(x_train, y_train, ridge__sample_weight=weights)
    split = max(100, int(len(train) * 0.8))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 2, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True),
        }
        model = _xgb_model(
            params,
            config.random_seed + trial.number,
            "cuda",
        )
        model.fit(x_train[:split], y_train[:split], sample_weight=weights[:split], verbose=False)
        return float(mean_squared_error(y_train[split:], model.predict(x_train[split:])))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.random_seed)
    )
    study.optimize(objective, n_trials=4, show_progress_bar=False)
    challenger = []
    for member in range(TEMPORAL_ENSEMBLE_MODELS):
        indexes = _temporal_block_indexes(
            train["signal_timestamp"], seed=config.random_seed + member
        )
        model = _xgb_model(study.best_params, config.random_seed + member, "cuda")
        model.fit(x_train[indexes], y_train[indexes], sample_weight=weights[indexes], verbose=False)
        challenger.append(model)
    map_mask = np.arange(len(calibrate)) < len(calibrate) // 2
    audit_mask = ~map_mask
    candidates = {"ridge": [ridge], "xgboost": challenger}
    audited: dict[str, Any] = {}
    for name, models in candidates.items():
        member_predictions = np.vstack(
            [np.asarray(model.predict(x_cal), dtype=float) for model in models]
        )
        raw = member_predictions.mean(axis=0)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw[map_mask], y_cal[map_mask])
        prediction = np.asarray(calibrator.predict(raw), dtype=float)
        residual = y_cal[audit_mask] - prediction[audit_mask]
        residual_lower = moving_block_lower_bound(
            residual, block_size=min(7, len(residual)), seed=config.random_seed
        )
        dispersion = member_predictions[:, audit_mask].std(axis=0, ddof=0)
        lower = residual_lower
        audited[name] = {
            "models": models,
            "calibrator": calibrator,
            "lower": lower,
            "mse": float(mean_squared_error(y_cal[audit_mask], prediction[audit_mask])),
            "calibration_error": float(abs(residual.mean())),
            "decision_regret": _decision_regret(
                calibrate.iloc[np.flatnonzero(audit_mask)], prediction[audit_mask]
            ),
            "lcb_regret": _decision_regret(
                calibrate.iloc[np.flatnonzero(audit_mask)],
                prediction[audit_mask] + lower - 1.645 * dispersion,
            ),
        }
    ridge_audit, xgb_audit = audited["ridge"], audited["xgboost"]
    xgb_wins = all(
        xgb_audit[key] < ridge_audit[key]
        for key in ("mse", "calibration_error", "decision_regret", "lcb_regret")
    )
    champion = "xgboost" if xgb_wins else "ridge"
    return {
        "enabled": True,
        "side": side,
        "features": list(MODEL_FEATURES),
        "champion": champion,
        **audited[champion],
        "audit": {
            name: {
                key: value for key, value in result.items() if key not in {"models", "calibrator"}
            }
            for name, result in audited.items()
        },
        "trials": len(study.trials),
    }


def run_v9_gpu_research(app: AppConfig, config_path: Path) -> dict[str, Any]:
    protocol = preregister_v9(config_path=config_path)
    inventory = btc_data_inventory()
    readiness = confirmation_readiness(protocol, inventory)
    if not readiness["ready"]:
        return run_v9_readiness(config_path)
    if app.machine_learning is None:
        raise ValueError("machine learning configuration is required")
    base_cost_bps = (
        2 * app.machine_learning.taker_fee_bps
        + 2 * float(app.backtest.slippage_bps)
        + float(app.backtest.spread_bps)
    )
    local = build_local_feature_frames(app)
    oos: list[pd.DataFrame] = []
    candidate_oos: list[pd.DataFrame] = []
    champion_counts: dict[str, int] = {}
    for held_out in EXCHANGES:
        augmented = add_cross_exchange_features(
            local,
            allowed_exchanges=tuple(exchange for exchange in EXCHANGES if exchange != held_out),
        )
        matrix = build_expert_matrix(augmented, base_cost_bps=base_cost_bps)
        folds = purged_expert_folds(
            matrix,
            train_weeks=52,
            calibration_weeks=4,
            test_weeks=4,
            step_weeks=4,
        )
        for fold_number, fold in enumerate(folds, start=1):
            fitting = matrix.iloc[fold.train].loc[matrix.iloc[fold.train]["exchange"].ne(held_out)]
            calibration = matrix.iloc[fold.calibration].loc[
                matrix.iloc[fold.calibration]["exchange"].ne(held_out)
            ]
            testing = matrix.iloc[fold.test].loc[matrix.iloc[fold.test]["exchange"].eq(held_out)]
            predicted: list[pd.DataFrame] = []
            for side in SIDES:
                model = fit_v9_fold(fitting, calibration, cast(Any, side), app.machine_learning)
                champion = str(model.get("champion", "disabled"))
                key = f"{held_out}:{side}:{champion}"
                champion_counts[key] = champion_counts.get(key, 0) + 1
                side_rows = predict_v9(testing, model)
                if not side_rows.empty:
                    predicted.append(side_rows)
            if predicted:
                candidates = pd.concat(predicted, ignore_index=True)
                candidates["outer_fold"] = fold_number
                candidates["held_out_exchange"] = held_out
                candidate_oos.append(candidates)
                selected = pd.concat(
                    [
                        select_v9_actions(candidates.loc[candidates["side"].eq(side)])
                        for side in SIDES
                    ],
                    ignore_index=True,
                )
                selected["outer_fold"] = fold_number
                selected["held_out_exchange"] = held_out
                oos.append(selected)
    decisions = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    candidates = pd.concat(candidate_oos, ignore_index=True) if candidate_oos else pd.DataFrame()
    audit = audit_v9_oos(decisions, candidate_returns=candidates, candidate_count=len(v9_experts()))
    report = {
        "protocol": PROTOCOL_VERSION,
        "run_id": protocol["run_id"],
        "verdict": "ALPHA_SHADOW_READY" if audit["ready"] else "NO_ALPHA_POLICY",
        "action": "SHADOW" if audit["ready"] else "FLAT",
        "deployable": False,
        "paper_only": audit["ready"],
        "candidate_count": len(v9_experts()),
        "champion_counts": champion_counts,
        "audit": audit,
        "execution": {"ready": False, "reason": "separate_30_day_gate_not_met"},
        "holdout": protocol["holdout"],
        "v8_report_sha256": _file_sha256(V8_REPORT),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    return report


def predict_v9(rows: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    if not fitted.get("enabled"):
        return rows.iloc[:0].copy()
    complete = _complete_side(rows, cast(Any, fitted["side"])).copy()
    values = complete[list(MODEL_FEATURES)].to_numpy(dtype=float)
    predictions = np.vstack([model.predict(values) for model in fitted["models"]])
    raw = predictions.mean(axis=0)
    complete["ev_mean"] = fitted["calibrator"].predict(raw)
    complete["ensemble_dispersion"] = predictions.std(axis=0, ddof=0)
    complete["lower_confidence_bound"] = np.minimum(
        complete["ev_mean"],
        complete["ev_mean"] + float(fitted["lower"]) - 1.645 * complete["ensemble_dispersion"],
    )
    return complete


def select_v9_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    valid = actions.loc[
        actions["ev_mean"].gt(0)
        & actions["lower_confidence_bound"].gt(0)
        & actions["feature_coverage"].astype(bool)
        & actions["lookahead_valid"].astype(bool)
    ].copy()
    if valid.empty:
        return valid
    return (
        valid.sort_values(
            ["signal_timestamp", "lower_confidence_bound", "expert_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("signal_timestamp")
        .reset_index(drop=True)
    )


def audit_v9_oos(
    decisions: pd.DataFrame,
    *,
    candidate_returns: pd.DataFrame | None = None,
    candidate_count: int = 8,
) -> dict[str, Any]:
    if decisions.empty or candidate_returns is None or candidate_returns.empty:
        return {"ready": False, "action": "FLAT", "reason": "no_positive_oos_decisions"}
    side_audits = {
        side: _audit_side(
            decisions.loc[decisions["side"].eq(side)],
            candidate_returns.loc[candidate_returns["side"].eq(side)],
            cast(Any, side),
        )
        for side in SIDES
    }
    enabled_sides = [side for side, result in side_audits.items() if result["ready"]]
    return {
        "ready": bool(enabled_sides),
        "action": "MODEL" if enabled_sides else "FLAT",
        "enabled_sides": enabled_sides,
        "candidate_count": candidate_count,
        "sides": side_audits,
    }


def _audit_side(
    decisions: pd.DataFrame, candidate_returns: pd.DataFrame, side: Literal["long", "short"]
) -> dict[str, Any]:
    if decisions.empty or candidate_returns.empty:
        return {"ready": False, "reason": "no_oos_decisions_for_side", "gates": {}}
    metrics = {exchange: _return_metrics(rows) for exchange, rows in decisions.groupby("exchange")}
    bitunix = metrics.get("bitunix", _return_metrics(decisions.iloc[:0]))
    daily = candidate_returns.assign(
        day=pd.to_datetime(candidate_returns["signal_timestamp"], utc=True).dt.floor("1D")
    ).pivot_table(
        index="day", columns="expert_id", values="net_return_r", aggfunc="sum", fill_value=0
    )
    control = next(
        expert.expert_id
        for expert in v9_experts()
        if expert.name == "v8_control" and expert.side == side
    )
    spa = spa_reality_check(daily, control_expert_id=control)
    exchange_positive = sum(result["expectancy_r"] > 0 for result in metrics.values())
    bitunix_lower = (
        moving_block_lower_bound(
            decisions.loc[decisions["exchange"].eq("bitunix"), "net_return_r"].to_numpy(
                dtype=float
            ),
            block_size=7,
            seed=20260804,
        )
        if bitunix["trades"]
        else float("-inf")
    )
    gates = {
        "minimum_100_bitunix_trades": bitunix["trades"] >= 100,
        "bitunix_expectancy_positive": bitunix["expectancy_r"] > 0,
        "bitunix_lower_bound_positive": bitunix_lower > 0,
        "bitunix_profit_factor": bitunix["profit_factor"] >= 1.15,
        "bitunix_drawdown": bitunix["max_drawdown"] <= 0.08,
        "bitunix_stress_2x": bitunix["stress_expectancy_r"] >= 0,
        "positive_window_majority": _positive_window_fraction(decisions) > 0.5,
        "three_exchanges_including_bitunix": exchange_positive >= 3 and bitunix["expectancy_r"] > 0,
        "spa_vs_flat": spa["spa_pvalue"] <= 0.05,
        "reality_check_vs_control": spa["reality_check_pvalue"] <= 0.05,
    }
    return {
        "ready": all(gates.values()),
        "metrics": metrics,
        "bitunix_expectancy_lower_bound_r": bitunix_lower,
        "multiple_comparison": spa,
        "gates": gates,
    }


def spa_reality_check(daily_returns: pd.DataFrame, *, control_expert_id: str) -> dict[str, float]:
    if len(daily_returns) < 20 or daily_returns.shape[1] < 2:
        return {"spa_pvalue": 1.0, "reality_check_pvalue": 1.0}
    losses = -daily_returns.to_numpy(dtype=float)
    flat = np.zeros(len(daily_returns))
    spa = SPA(
        flat,
        losses,
        reps=1000,
        block_size=max(2, int(math.sqrt(len(flat)))),
        bootstrap="stationary",
        seed=20260804,
    )
    spa.compute()
    control = -daily_returns.get(
        control_expert_id, pd.Series(0.0, index=daily_returns.index)
    ).to_numpy(dtype=float)
    reality = SPA(
        control,
        losses,
        reps=1000,
        block_size=max(2, int(math.sqrt(len(flat)))),
        bootstrap="stationary",
        seed=20260804,
    )
    reality.compute()
    return {
        "spa_pvalue": float(spa.pvalues["consistent"]),
        "reality_check_pvalue": float(reality.pvalues["consistent"]),
    }


def run_v9_readiness(config_path: Path) -> dict[str, Any]:
    protocol = preregister_v9(config_path=config_path)
    inventory = btc_data_inventory()
    readiness = confirmation_readiness(protocol, inventory)
    execution_days = _execution_days(Path("data/raw/bitunix_microstructure"))
    report = {
        "protocol": PROTOCOL_VERSION,
        "run_id": protocol["run_id"],
        "verdict": "WAITING_FOR_POST_V8_CONFIRMATION_DATA"
        if not readiness["ready"]
        else "READY_FOR_V9_GPU_RESEARCH",
        "action": "FLAT",
        "deployable": False,
        "paper_only": False,
        "training_started": False,
        "candidate_count": len(v9_experts()),
        "data": inventory,
        "confirmation": readiness,
        "features": {
            "included": list(NEW_FEATURES),
            "excluded": UNAVAILABLE_HISTORICAL_FEATURES,
        },
        "execution": {
            "ready": execution_days >= 30,
            "distinct_days": execution_days,
            "required_days": 30,
            "reason": None if execution_days >= 30 else "minimum_30_distinct_execution_days",
        },
        "holdout": protocol["holdout"],
        "v8_report_sha256": _file_sha256(V8_REPORT),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    return report


def _complete_side(rows: pd.DataFrame, side: Literal["long", "short"]) -> pd.DataFrame:
    selected = rows.loc[rows["side"].eq(side)].replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(subset=list(MODEL_FEATURES))
    return (
        selected.loc[selected["feature_coverage"].astype(bool)]
        .sort_values("signal_timestamp")
        .reset_index(drop=True)
    )


def _timestamp_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby(["exchange", "signal_timestamp"])["expert_id"].transform("count")
    weights = pd.Series(1.0 / counts.to_numpy(dtype=float), index=rows.index)
    exchange_total = weights.groupby(rows["exchange"]).transform("sum")
    return (weights / exchange_total * len(rows) / rows["exchange"].nunique()).to_numpy()


def _temporal_block_indexes(timestamps: pd.Series, *, seed: int) -> np.ndarray:
    days = pd.to_datetime(timestamps, utc=True).dt.floor("1D")
    unique_days = days.drop_duplicates().to_numpy()
    if len(unique_days) < 7:
        return np.arange(len(days))
    randomizer = np.random.default_rng(seed)
    starts = randomizer.integers(0, len(unique_days), size=math.ceil(len(unique_days) / 7))
    sampled_days = np.concatenate(
        [unique_days[(start + np.arange(7)) % len(unique_days)] for start in starts]
    )[: len(unique_days)]
    positions = [np.flatnonzero(days.to_numpy() == day) for day in sampled_days]
    return np.concatenate(positions).astype(int)


def _decision_regret(rows: pd.DataFrame, predictions: np.ndarray) -> float:
    truth = rows["net_return_r"].to_numpy(dtype=float)
    regrets = []
    for indexes in rows.groupby("signal_timestamp", sort=False).indices.values():
        positions = np.asarray(indexes, dtype=int)
        choice = positions[int(np.argmax(predictions[positions]))]
        chosen = truth[choice] if predictions[choice] > 0 else 0.0
        regrets.append(max(0.0, float(truth[positions].max())) - chosen)
    return float(np.mean(regrets)) if regrets else float("inf")


def _xgb_model(params: dict[str, Any], seed: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_estimators=300,
        random_state=seed,
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
        **params,
    )


def _return_metrics(rows: pd.DataFrame) -> dict[str, float]:
    values = rows.get("net_return_r", pd.Series(dtype=float)).to_numpy(dtype=float)
    stress = rows.get("net_return_r_2x", pd.Series(dtype=float)).to_numpy(dtype=float)
    if not len(values):
        return {
            "trades": 0.0,
            "expectancy_r": 0.0,
            "stress_expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }
    equity = np.cumprod(1 + 0.01 * values)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    losses = abs(float(values[values < 0].sum()))
    return {
        "trades": float(len(values)),
        "expectancy_r": float(values.mean()),
        "stress_expectancy_r": float(stress.mean()),
        "profit_factor": float(values[values > 0].sum()) / losses if losses else 999.0,
        "max_drawdown": float(np.max((peaks - equity) / peaks)),
    }


def _positive_window_fraction(rows: pd.DataFrame) -> float:
    values = (
        rows.assign(
            window=pd.to_datetime(rows["signal_timestamp"], utc=True).dt.to_period("M").astype(str)
        )
        .groupby("window")["net_return_r"]
        .mean()
    )
    return float(values.gt(0).mean()) if len(values) else 0.0


def _execution_days(directory: Path) -> int:
    days: set[object] = set()
    for path in directory.glob("btcusdt_*.parquet"):
        timestamps = pd.read_parquet(path, columns=["exchange_timestamp"])["exchange_timestamp"]
        days.update(pd.to_datetime(timestamps, format="mixed", utc=True).dt.date)
    return len(days)


def _parquet_columns(path: Path) -> list[str]:
    from pyarrow.parquet import ParquetFile  # type: ignore[import-untyped]

    return list(ParquetFile(path).schema.names)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC-only Hybrid V9 readiness and research")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    app = load_config(arguments.config)
    report = run_v9_gpu_research(app, arguments.config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
