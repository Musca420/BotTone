from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v11 import (
    btc_inventory,
    build_feature_frame,
    evaluate_expert,
    return_metrics,
)
from adaptive_bot.hybrid_policy_v14 import (
    _download_checked,
    build_orderflow_context,
)
from adaptive_bot.hybrid_policy_v17 import action_expert, add_relational_features

PROTOCOL = "hybrid_v19_binance_primary_vwap_actions"
TIMEFRAME_MINUTES = 15
VWAP_HOURS = 24
ROOT = Path("data/ml/hybrid_v19")
PUBLIC_ROOT = ROOT / "binance_public"
CONTEXT_PATH = ROOT / "binance_context_5m.parquet"
BINANCE_MATRIX_PATH = ROOT / "binance_matrix.parquet"
CONTROL_MATRIX_PATH = ROOT / "external_controls.parquet"
OOS_PATH = ROOT / "oos_candidates.parquet"
MODEL_ROOT = Path("data/models/expert_policy/v19")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
BUNDLE_PATH = MODEL_ROOT / "shadow_bundle.joblib"
REPORT_PATH = Path("data/reports/ml_hybrid_v19.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v19.status.json")
ORDERFLOW_PATH = Path("data/ml/hybrid_v14/binance_reference_orderflow_1m.parquet")
EXTERNAL_CONTROLS = ("okx", "bybit")
BASE_COST_BPS = 4.0

BASE_FEATURES = (
    "distance_vwap_atr",
    "gap_velocity_1",
    "gap_velocity_3",
    "vwap_chase_ratio_3",
    "vwap_chase_ratio_12",
    "vwap_zscore",
    "vwap_slope_3",
    "vwap_slope_12",
    "atr_percentile",
    "realized_volatility",
    "adx",
    "adx_slope",
    "ema20_slope",
    "ema50_slope",
    "relative_volume",
    "return_1",
    "return_3",
    "return_12",
    "return_48",
    "mark_last_divergence",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "regime_code",
    "side_code",
)
CONTEXT_FEATURES = (
    "binance_taker_imbalance_15m",
    "binance_taker_imbalance_1h",
    "binance_trade_count_z",
    "binance_aggressive_volume_z",
    "oi_change_1h",
    "oi_change_4h",
    "oi_change_24h",
    "oi_value_change_1h",
    "top_position_ratio",
    "global_long_short_ratio",
    "taker_long_short_ratio",
    "metrics_taker_ratio_z",
    "book_imbalance_1pct",
    "book_depth_log_z",
)
FEATURES = BASE_FEATURES + CONTEXT_FEATURES
STATE_FEATURES = tuple(name for name in FEATURES if name != "side_code")


@dataclass(frozen=True)
class FittedAction:
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


def _replace_with_retry(temporary: Path, target: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


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


def _daily_relative(dataset: str, day: pd.Timestamp) -> str:
    date = day.strftime("%Y-%m-%d")
    return f"daily/{dataset}/BTCUSDT/BTCUSDT-{dataset}-{date}.zip"


def _cache_archive(dataset: str, day: pd.Timestamp) -> Path | None:
    date = day.strftime("%Y-%m-%d")
    target = PUBLIC_ROOT / dataset / f"{date}.zip"
    if target.exists() and target.stat().st_size:
        return target
    try:
        payload = _download_checked(_daily_relative(dataset, day))
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def download_public_archives(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, list[Path]]:
    days = list(pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC"))
    tasks = [(dataset, day) for dataset in ("metrics", "bookDepth") for day in days]
    found: dict[str, list[Path]] = {"metrics": [], "bookDepth": []}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_cache_archive, dataset, day): dataset for dataset, day in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            path = future.result()
            if path is not None:
                found[dataset].append(path)
            _status(
                "official_download",
                f"Binance metrics/bookDepth {number}/{len(tasks)}",
                2 + 18 * number / max(len(tasks), 1),
                files_downloaded=sum(map(len, found.values())),
                files_total=len(tasks),
            )
    minimum = max(30, int(len(days) * 0.9))
    if len(found["metrics"]) < minimum or len(found["bookDepth"]) < minimum:
        raise RuntimeError(
            "insufficient official Binance metrics/bookDepth coverage: "
            f"{len(found['metrics'])}/{len(days)} metrics, "
            f"{len(found['bookDepth'])}/{len(days)} bookDepth"
        )
    return {name: sorted(paths) for name, paths in found.items()}


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"unexpected Binance archive layout: {path}")
        return pd.read_csv(archive.open(names[0]))


def _parse_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_zip_csv(path) for path in paths]
    rows = pd.concat(frames, ignore_index=True)
    rows["timestamp"] = pd.to_datetime(rows["create_time"], format="mixed", utc=True)
    columns = {
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "sum_toptrader_long_short_ratio": "top_position_ratio",
        "count_long_short_ratio": "global_long_short_ratio",
        "sum_taker_long_short_vol_ratio": "taker_long_short_ratio",
    }
    rows = rows.rename(columns=columns).drop_duplicates("timestamp").sort_values("timestamp")
    for column in columns.values():
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for hours in (1, 4, 24):
        rows[f"oi_change_{hours}h"] = rows["open_interest"].pct_change(hours * 12, fill_method=None)
    rows["oi_value_change_1h"] = rows["open_interest_value"].pct_change(12, fill_method=None)
    history = rows["taker_long_short_ratio"].shift(1).rolling(2_016, min_periods=576)
    rows["metrics_taker_ratio_z"] = (rows["taker_long_short_ratio"] - history.mean()) / history.std(
        ddof=0
    ).replace(0, np.nan)
    rows["metrics_available_at"] = rows["timestamp"] + pd.Timedelta(minutes=5)
    keep = ["metrics_available_at", *CONTEXT_FEATURES[4:12]]
    return rows[keep].replace([np.inf, -np.inf], np.nan)


def _parse_book_depth(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        rows = _read_zip_csv(path)
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], format="mixed", utc=True)
        rows["percentage"] = pd.to_numeric(rows["percentage"], errors="coerce")
        rows["notional"] = pd.to_numeric(rows["notional"], errors="coerce")
        pivot = rows.pivot_table(index="timestamp", columns="percentage", values="notional")
        if -1.0 not in pivot or 1.0 not in pivot:
            continue
        bid, ask = pivot[-1.0].abs(), pivot[1.0].abs()
        total = bid + ask
        frame = pd.DataFrame(
            {
                "timestamp": pivot.index,
                "book_imbalance_1pct": (bid - ask) / total.replace(0, np.nan),
                "book_depth_log": np.log1p(total),
            }
        )
        frames.append(frame.reset_index(drop=True))
    if not frames:
        raise RuntimeError("official Binance bookDepth archives contained no +/-1% depth")
    result = (
        pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    )
    history = result["book_depth_log"].shift(1).rolling(2_016, min_periods=576)
    result["book_depth_log_z"] = (result["book_depth_log"] - history.mean()) / history.std(
        ddof=0
    ).replace(0, np.nan)
    result["book_available_at"] = result["timestamp"] + pd.Timedelta(minutes=5)
    return result[["book_available_at", "book_imbalance_1pct", "book_depth_log_z"]]


def build_context(start: pd.Timestamp, end: pd.Timestamp, *, resume: bool) -> pd.DataFrame:
    if resume and CONTEXT_PATH.exists():
        return pd.read_parquet(CONTEXT_PATH)
    archives = download_public_archives(start, end)
    _status("context", "Parsing official Binance metrics", 22)
    metrics = _parse_metrics(archives["metrics"])
    _status("context", "Parsing official Binance bookDepth", 25)
    book = _parse_book_depth(archives["bookDepth"])
    flow_minutes = pd.read_parquet(ORDERFLOW_PATH)
    flow_times = pd.to_datetime(flow_minutes["timestamp"], format="mixed", utc=True)
    flow = build_orderflow_context(flow_minutes.loc[flow_times.between(start, end)])
    context = pd.merge_asof(
        metrics.sort_values("metrics_available_at"),
        book.sort_values("book_available_at"),
        left_on="metrics_available_at",
        right_on="book_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    context = pd.merge_asof(
        context.sort_values("metrics_available_at"),
        flow.sort_values("orderflow_available_at"),
        left_on="metrics_available_at",
        right_on="orderflow_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    context["context_available_at"] = context[
        ["metrics_available_at", "book_available_at", "orderflow_available_at"]
    ].max(axis=1)
    context["context_coverage"] = context[list(CONTEXT_FEATURES)].notna().all(axis=1)
    _atomic_parquet(CONTEXT_PATH, context)
    return context


def attach_context(features: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge_asof(
        features.sort_values("signal_timestamp"),
        context.sort_values("context_available_at"),
        left_on="signal_timestamp",
        right_on="context_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    result["context_lookahead_valid"] = pd.to_datetime(result["context_available_at"], utc=True).le(
        pd.to_datetime(result["signal_timestamp"], utc=True)
    )
    result["context_coverage"] = (
        result["context_coverage"].fillna(False).astype(bool)
        & result["context_lookahead_valid"]
        & result[list(CONTEXT_FEATURES)].notna().all(axis=1)
    )
    return result.sort_values("timestamp").reset_index(drop=True)


def state_masks(features: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    z = features["distance_vwap_atr"].astype(float)
    common = (
        z.abs().between(0.5, 3.0)
        & features["atr_percentile"].le(90)
        & ~features["regime_code"].isin((3.0, 4.0))
        & features["data_valid"].fillna(False).astype(bool)
        & features["local_feature_coverage"].fillna(False).astype(bool)
        & features["context_coverage"].fillna(False).astype(bool)
        & features[list(STATE_FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    )
    return {
        ("fade", "short"): common & z.gt(0),
        ("follow", "long"): common & z.gt(0),
        ("fade", "long"): common & z.lt(0),
        ("follow", "short"): common & z.lt(0),
    }


def _outcomes(
    app: AppConfig,
    exchange: str,
    path: Path,
    requested: pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw, features = build_feature_frame(
        path, app, exchange, timeframe_minutes=TIMEFRAME_MINUTES, vwap_hours=VWAP_HOURS
    )
    features = add_relational_features(features)
    if context is not None:
        features = attach_context(features, context)
    masks = state_masks(features) if context is not None else {}
    outcomes: list[pd.DataFrame] = []
    pairs: list[tuple[tuple[str, str], pd.Series]] = []
    if requested is None:
        pairs = list(masks.items())
    else:
        for (action, side), rows in requested.groupby(["action", "side"]):
            timestamps = set(pd.to_datetime(rows["signal_timestamp"], utc=True))
            mask = pd.Series(
                pd.to_datetime(features["signal_timestamp"], utc=True).isin(timestamps),
                index=features.index,
            )
            mask &= features["data_valid"].fillna(False).astype(bool)
            pairs.append(((str(action), str(side)), mask))
    for key, value in pairs:
        action, side = key
        result = evaluate_expert(
            features,
            raw,
            action_expert(str(action), str(side)),
            cost_bps=BASE_COST_BPS,
            entry_mask=value,
            timeframe_minutes=TIMEFRAME_MINUTES,
            vwap_hours=VWAP_HOURS,
        )
        if not result.empty:
            result["action"] = action
            result["side_code"] = 1.0 if side == "long" else -1.0
            outcomes.append(result)
    if not outcomes:
        raise RuntimeError(f"V19 produced no outcomes for {exchange}")
    return pd.concat(outcomes, ignore_index=True)


def build_matrices(app: AppConfig, *, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = btc_inventory()
    source = Path(inventory["binance"]["path"])
    raw = pd.read_parquet(source, columns=["timestamp"])
    times = pd.to_datetime(raw["timestamp"], utc=True)
    context = build_context(times.min(), times.max(), resume=resume)
    if resume and BINANCE_MATRIX_PATH.exists():
        primary = pd.read_parquet(BINANCE_MATRIX_PATH)
        if not set(FEATURES).issubset(primary):
            primary = pd.DataFrame()
    else:
        primary = pd.DataFrame()
    if primary.empty:
        _status("matrix", "Binance paired VWAP FADE/FOLLOW counterfactuals", 28)
        primary = _outcomes(app, "binance", source, context=context)
        paired = primary.groupby("signal_timestamp")["action"].transform("nunique").eq(2)
        primary = primary.loc[paired].copy()
        primary["target_net_r"] = primary["gross_return_r"] - primary["cost_r_1x"]
        primary["state_id"] = pd.to_datetime(primary["signal_timestamp"], utc=True).astype(str)
        _atomic_parquet(BINANCE_MATRIX_PATH, primary)
    if resume and CONTROL_MATRIX_PATH.exists():
        controls = pd.read_parquet(CONTROL_MATRIX_PATH)
    else:
        frames = []
        requested = primary[["signal_timestamp", "action", "side"]].drop_duplicates()
        for number, exchange in enumerate(EXTERNAL_CONTROLS, start=1):
            _status(
                "matrix", f"Synchronized control {exchange.upper()} {number}/2", 30 + number * 2
            )
            frames.append(_outcomes(app, exchange, Path(inventory[exchange]["path"]), requested))
        controls = pd.concat(frames, ignore_index=True)
        _atomic_parquet(CONTROL_MATRIX_PATH, controls)
    return primary.sort_values("signal_timestamp").reset_index(drop=True), controls


def _model(kind: str) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=500,
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


def _raw(model: Any, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(rows[list(FEATURES)]), dtype=float)


def _fit(kind: str, train: pd.DataFrame, calibration: pd.DataFrame) -> tuple[Any, Any]:
    model = _model(kind)
    model.fit(train[list(FEATURES)], train["target_net_r"])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        _raw(model, calibration), calibration["target_net_r"].to_numpy(float)
    )
    return model, calibrator


def _select(rows: pd.DataFrame) -> pd.DataFrame:
    proposals = (
        rows.loc[rows["ev_lcb"].gt(0)]
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


def fit_action(
    train: pd.DataFrame, calibration: pd.DataFrame, *, allow_gpu: bool
) -> FittedAction | None:
    if len(train) < 500 or len(calibration) < 160:
        return None
    times = pd.to_datetime(calibration["signal_timestamp"], utc=True)
    midpoint = times.min() + (times.max() - times.min()) / 2
    calibrate, admission = calibration.loc[times.lt(midpoint)], calibration.loc[times.ge(midpoint)]
    if min(len(calibrate), len(admission)) < 80:
        return None
    kinds = ("ridge", "xgboost") if allow_gpu else ("ridge",)
    candidates: dict[str, FittedAction] = {}
    for kind in kinds:
        model, calibrator = _fit(kind, train, calibrate)
        prediction = np.asarray(calibrator.predict(_raw(model, admission)), dtype=float)
        residuals = admission["target_net_r"].to_numpy(float) - prediction
        residual_lower = moving_block_lower_bound(
            residuals, block_size=min(20, len(residuals)), seed=20260805
        )
        margin = max(0.0, -residual_lower)
        assessed = admission.assign(ev_net=prediction, ev_lcb=prediction - margin)
        chosen = _select(assessed)
        candidates[kind] = FittedAction(
            model=model,
            calibrator=calibrator,
            champion=kind,
            admission_mse=float(mean_squared_error(admission["target_net_r"], prediction)),
            admission_ev_r=float(chosen["target_net_r"].mean()) if len(chosen) >= 20 else -np.inf,
            uncertainty_margin_r=margin,
        )
    ridge = candidates["ridge"]
    challenger = candidates.get("xgboost")
    if (
        challenger
        and challenger.admission_mse < ridge.admission_mse
        and challenger.admission_ev_r > ridge.admission_ev_r
    ):
        return challenger
    return ridge


def walk_forward(
    matrix: pd.DataFrame, *, allow_gpu: bool, smoke: bool
) -> tuple[pd.DataFrame, dict[str, int]]:
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
    for number, fold in enumerate(folds, start=1):
        predicted: list[pd.DataFrame] = []
        for action in ("fade", "follow"):
            train_rows = matrix.iloc[fold.train]
            calibration_rows = matrix.iloc[fold.calibration]
            test_rows = matrix.iloc[fold.test]
            fitted = fit_action(
                train_rows.loc[train_rows["action"].eq(action)],
                calibration_rows.loc[calibration_rows["action"].eq(action)],
                allow_gpu=allow_gpu,
            )
            if fitted is None:
                continue
            rows = test_rows.loc[test_rows["action"].eq(action)].copy()
            rows["ev_net"] = np.asarray(fitted.calibrator.predict(_raw(fitted.model, rows)))
            rows["ev_lcb"] = rows["ev_net"] - fitted.uncertainty_margin_r
            rows["champion"] = fitted.champion
            rows["outer_fold"] = number
            predicted.append(rows)
            key = f"{action}:{fitted.champion}"
            champions[key] = champions.get(key, 0) + 1
        if predicted:
            outputs.append(pd.concat(predicted, ignore_index=True))
        _status(
            "gpu_walk_forward" if allow_gpu else "ridge_screen",
            f"Fold {number}/{len(folds)}",
            37 + 55 * number / len(folds),
            backend="cuda" if allow_gpu else "cpu",
        )
    return (pd.concat(outputs, ignore_index=True) if outputs else matrix.iloc[:0].copy()), champions


def audit(decisions: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {"trades": 0, "gates": {"nonempty": False}, "gates_passed": False}
    base = return_metrics(decisions, "target_net_r")
    stressed = decisions.assign(stress=decisions["gross_return_r"] - 2 * decisions["cost_r_1x"])
    stress = return_metrics(stressed, "stress")
    lower = moving_block_lower_bound(
        decisions["target_net_r"].to_numpy(float),
        block_size=min(20, len(decisions)),
        seed=20260805,
    )
    external = synchronized_control_rows(decisions, controls)
    external_metrics = {
        name: return_metrics(rows.assign(target_net_r=rows["net_return_r_1x"]), "target_net_r")
        for name, rows in external.groupby("exchange")
    }
    months = pd.to_datetime(decisions["signal_timestamp"], utc=True).dt.to_period("M")
    positive_months = float(decisions["target_net_r"].groupby(months).sum().gt(0).mean())
    gates = {
        "trades_300": len(decisions) >= 300,
        "expectancy_positive": base["expectancy_r"] > 0,
        "bootstrap_lower_positive": lower > 0,
        "profit_factor_1_15": base["profit_factor"] >= 1.15,
        "drawdown_8pct": base["max_drawdown"] <= 0.08,
        "costs_2x_nonnegative": stress["expectancy_r"] >= 0,
        "positive_month_majority": positive_months > 0.5,
        "external_controls_positive": set(external_metrics) == set(EXTERNAL_CONTROLS)
        and all(item["expectancy_r"] > 0 for item in external_metrics.values()),
    }
    return {
        "trades": len(decisions),
        "binance_net_4bps": base,
        "binance_stress_8bps": stress,
        "expectancy_lower_bound_r": lower,
        "positive_month_fraction": positive_months,
        "external_synchronized_controls": external_metrics,
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def synchronized_control_rows(decisions: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    selected = decisions[["signal_timestamp", "action", "side"]].drop_duplicates().copy()
    external = controls.copy()
    selected["signal_timestamp"] = pd.to_datetime(selected["signal_timestamp"], utc=True)
    external["signal_timestamp"] = pd.to_datetime(external["signal_timestamp"], utc=True)
    return external.merge(selected, on=["signal_timestamp", "action", "side"], how="inner")


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_exchange": "binance",
        "symbol": "BTCUSDT perpetual",
        "external_controls": list(EXTERNAL_CONTROLS),
        "bitunix_role": "collector_only_future_transfer_audit",
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "vwap_hours": VWAP_HOURS,
        "actions": ["fade", "follow", "flat"],
        "features": list(FEATURES),
        "official_datasets": ["futures/um daily metrics", "futures/um daily bookDepth"],
        "walk_forward_weeks": [52, 4, 4, 4],
        "model": "ridge_default_xgboost_cuda_challenger",
        "decision": "maximum calibrated LCB above zero, otherwise FLAT",
        "cost_bps": BASE_COST_BPS,
        "stress_cost_bps": BASE_COST_BPS * 2,
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    payload = protocol_payload() | {"registered_at": datetime.now(UTC).isoformat()}
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != payload["protocol_sha256"]:
            raise RuntimeError("V19 protocol changed after freezing")
        return dict(existing)
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def run(app: AppConfig, *, resume: bool = False, smoke: bool = False) -> dict[str, Any]:
    protocol = preregister()
    primary, controls = build_matrices(app, resume=resume)
    ridge_candidates, ridge_champions = walk_forward(primary, allow_gpu=False, smoke=smoke)
    ridge_decisions = _select(ridge_candidates)
    ridge_metrics = return_metrics(ridge_decisions, "target_net_r")
    allow_gpu = len(ridge_decisions) >= 100 and ridge_metrics["expectancy_r"] > 0
    if allow_gpu:
        candidates, champions = walk_forward(primary, allow_gpu=True, smoke=smoke)
        decisions = _select(candidates)
    else:
        candidates, champions, decisions = ridge_candidates, ridge_champions, ridge_decisions
    result = audit(decisions, controls)
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "verdict": "BINANCE_ALPHA_SHADOW_CANDIDATE" if result["gates_passed"] else "FLAT",
        "deployable": False,
        "shadow_enabled": bool(result["gates_passed"]),
        "paper_enabled": False,
        "live_enabled": False,
        "primary_exchange": "binance",
        "external_controls": list(EXTERNAL_CONTROLS),
        "bitunix_execution_enabled": False,
        "states": int(primary["state_id"].nunique()),
        "paired_counterfactuals": len(primary),
        "ridge_screen": ridge_metrics,
        "gpu_admitted": allow_gpu,
        "champions": champions,
        "audit": result,
        "smoke": smoke,
        "warning": "historical discovery only; paper/shadow required before any live use",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(OOS_PATH, candidates)
    _atomic_json(REPORT_PATH, report)
    if result["gates_passed"]:
        timestamps = pd.to_datetime(primary["signal_timestamp"], utc=True)
        calibration_start = timestamps.max() - pd.Timedelta(weeks=8)
        final_models = {
            action: fit_action(
                rows.loc[timestamps.loc[rows.index].lt(calibration_start)],
                rows.loc[timestamps.loc[rows.index].ge(calibration_start)],
                allow_gpu=allow_gpu,
            )
            for action in ("fade", "follow")
            if not (rows := primary.loc[primary["action"].eq(action)]).empty
        }
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"protocol": protocol, "features": FEATURES, "models": final_models, "report": report},
            BUNDLE_PATH,
        )
    _status("complete", report["verdict"], 100, backend="cuda" if allow_gpu else "ridge_cpu")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V19 Binance-primary BTC VWAP Alpha")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), resume=args.resume, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
