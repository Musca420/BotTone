from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd

from adaptive_bot.config import AppConfig, MachineLearningConfig, load_config
from adaptive_bot.expert_policy import (
    _fit_side,
    _xgb_predict,
    moving_block_lower_bound,
    purged_expert_folds,
)
from adaptive_bot.hybrid_policy import _resample_alpha, balanced_exchange_symbol_weights
from adaptive_bot.hybrid_policy_v9 import spa_reality_check
from adaptive_bot.ml_research import FEATURE_COLUMNS, build_ml_features
from adaptive_bot.research import combinatorial_pbo, deflated_sharpe_probability

PROTOCOL = "hybrid_v11_btc_external_alpha"
EXCHANGES = ("binance", "okx", "bybit")
SIDES = ("long", "short")
MANIFEST_PATH = Path("data/ml/hybrid_v7/alpha_manifest.json")
ROOT = Path("data/ml/hybrid_v11")
MATRIX_ROOT = ROOT / "base_matrix"
OOS_ROOT = ROOT / "oos_folds"
MODEL_ROOT = Path("data/models/expert_policy/v11")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
BUNDLE_PATH = MODEL_ROOT / "bundle.joblib"
REPORT_PATH = Path("data/reports/ml_hybrid_v11.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v11.status.json")
V10_INVALIDATION_PATH = Path("data/reports/ml_hybrid_v10_external.invalidated.json")
VARIANT_SOURCE_PATHS: tuple[Path, ...] = ()
BASE_COST_BPS = 4.0
SHADOW_COST_BPS = 19.0
MODEL_TRIALS = 8
ENSEMBLE_MODELS = 10
COOLDOWN_MINUTES = 60
EXTRA_FEATURES = (
    "return_48",
    "cross_exchange_return_median",
    "cross_exchange_return_dispersion",
)
MODEL_COLUMNS = (
    *(name for name in FEATURE_COLUMNS if name != "funding_rate"),
    *EXTRA_FEATURES,
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "regime_code",
    "timeframe_minutes",
    "vwap_hours",
    "entry_z",
    "entry_rule_code",
    "stop_atr",
    "exit_z",
    "time_stop_hours",
    "range_adx_threshold",
    "regime_policy_code",
    "confirmation_bars",
)


@dataclass(frozen=True)
class V11Expert:
    name: str
    family: Literal["mean_reversion", "momentum"]
    side: Literal["long", "short"]
    entry_kind: Literal[
        "range_exhaustion",
        "confirmed_reentry",
        "breakout",
        "trend_continuation",
        "volatility",
        "multi_horizon",
    ]
    entry_z: float
    breakout_bars: int
    stop_atr: float
    target_atr: float | None
    target_z: float | None
    trailing_atr: float | None
    maximum_holding_bars: int

    @property
    def expert_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"v11-{self.side[0]}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def v11_experts() -> tuple[V11Expert, ...]:
    templates: tuple[tuple[Any, ...], ...] = (
        ("range_exhaustion", "mean_reversion", "range_exhaustion", 1.5, 0, 2.5, None, 0.0, None, 8),
        (
            "confirmed_reentry",
            "mean_reversion",
            "confirmed_reentry",
            2.0,
            0,
            2.5,
            None,
            0.5,
            None,
            16,
        ),
        ("v8_control", "momentum", "breakout", 0.0, 24, 2.0, 2.0, None, None, 32),
        ("trend_continuation", "momentum", "trend_continuation", 0.0, 48, 2.0, None, None, 2.0, 64),
        ("volatility_confirmed", "momentum", "volatility", 0.0, 24, 1.5, 3.0, None, None, 32),
        ("multi_horizon", "momentum", "multi_horizon", 0.0, 0, 2.0, None, None, 2.5, 48),
    )
    return tuple(
        V11Expert(
            name,
            cast(Any, family),
            cast(Any, side),
            cast(Any, kind),
            z,
            breakout,
            stop,
            target,
            target_z,
            trail,
            hold,
        )
        for side in SIDES
        for name, family, kind, z, breakout, stop, target, target_z, trail, hold in templates
    )


def run_v11(
    app: AppConfig, config_path: Path, *, resume: bool = False, smoke: bool = False
) -> dict[str, Any]:
    protocol = preregister(config_path, persist=not smoke)
    invalidate_v10()
    inventory = btc_inventory()
    _status("data_audit", "BTC-only manifest and causal coverage", 1, block="1/5")
    matrices = build_base_matrices(config_path, inventory, protocol, resume=resume)
    local = {
        exchange: build_feature_frame(Path(inventory[exchange]["path"]), app, exchange)[1]
        for exchange in EXCHANGES
    }
    candidate_frames: list[pd.DataFrame] = []
    champion_counts: dict[str, int] = {}
    total_outer_folds = 0
    completed_outer_folds = 0
    scenarios: list[tuple[str, pd.DataFrame, tuple[Any, ...]]] = []
    for held_out in EXCHANGES:
        allowed = tuple(exchange for exchange in EXCHANGES if exchange != held_out)
        matrix = assemble_matrix(matrices, cross_exchange_features(local, allowed))
        folds = purged_expert_folds(
            matrix,
            train_weeks=52,
            calibration_weeks=4,
            test_weeks=4,
            step_weeks=4,
            embargo_hours=16,
        )
        if smoke:
            folds = folds[:1]
        scenarios.append((held_out, matrix, folds))
        total_outer_folds += len(folds)
    if not total_outer_folds:
        return finish_no_policy(protocol, inventory, "no_complete_walk_forward_folds")
    OOS_ROOT.mkdir(parents=True, exist_ok=True)
    for held_out_number, (held_out, matrix, folds) in enumerate(scenarios, start=1):
        for fold_number, fold in enumerate(folds, start=1):
            checkpoint = OOS_ROOT / f"{held_out}_fold_{fold_number:02d}.joblib"
            if resume and checkpoint.exists():
                payload = cast(dict[str, Any], joblib.load(checkpoint))
                if payload.get("protocol_sha256") == protocol["protocol_sha256"]:
                    candidate_frames.append(cast(pd.DataFrame, payload["candidates"]))
                    completed_outer_folds += 1
                    _fold_status(
                        held_out_number,
                        held_out,
                        fold_number,
                        len(folds),
                        completed_outer_folds,
                        total_outer_folds,
                        "RESUME",
                    )
                    continue
            train = matrix.iloc[fold.train].loc[matrix.iloc[fold.train]["exchange"].ne(held_out)]
            calibration = matrix.iloc[fold.calibration].loc[
                matrix.iloc[fold.calibration]["exchange"].ne(held_out)
            ]
            testing = matrix.iloc[fold.test].loc[matrix.iloc[fold.test]["exchange"].eq(held_out)]
            previous = pd.concat([train, calibration], ignore_index=True).sort_values(
                "signal_timestamp"
            )
            global_fold = completed_outer_folds + 1
            fitted = fit_policy(
                previous,
                app.machine_learning,
                progress=(global_fold, total_outer_folds),
                admission_start=pd.to_datetime(calibration["signal_timestamp"], utc=True).min(),
            )
            predicted = predict_policy(testing, fitted)
            candidates = attach_predictions(testing, predicted)
            candidates["held_out_exchange"] = held_out
            candidates["outer_fold"] = fold_number
            for side, model in fitted.items():
                champion = str(model.get("champion", "disabled"))
                key = f"{held_out}:{side}:{champion}"
                champion_counts[key] = champion_counts.get(key, 0) + 1
            _atomic_joblib(
                checkpoint,
                {
                    "protocol_sha256": protocol["protocol_sha256"],
                    "candidates": candidates,
                    "model_diagnostics": {
                        side: {
                            "champion": model.get("champion"),
                            "enabled": model.get("enabled", False),
                            "reason": model.get("reason"),
                            "absolute_admission": model.get("absolute_admission"),
                        }
                        for side, model in fitted.items()
                    },
                },
            )
            candidate_frames.append(candidates)
            completed_outer_folds += 1
            _fold_status(
                held_out_number,
                held_out,
                fold_number,
                len(folds),
                completed_outer_folds,
                total_outer_folds,
                "DONE",
            )
    candidates = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    decisions = replay_policy(candidates, value_prefix="net")
    audit = audit_oos(decisions, candidates)
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "run_id": protocol["run_id"],
        "verdict": "ALPHA_DISCOVERY_SHADOW_ONLY" if audit["ready"] else "NO_ALPHA_POLICY",
        "action": "BITUNIX_SHADOW_DIAGNOSTIC" if audit["ready"] else "FLAT",
        "deployable": False,
        "paper_enabled": False,
        "live_enabled": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "symbols": ["BTCUSDT"],
        "experts": len(v11_experts()),
        "audit": audit,
        "champion_counts": champion_counts,
        "data": inventory,
        "excluded_features": {
            "funding_rate": "incomplete common official history; OKX begins 2026-05",
            "funding_zscore": "incomplete common official history",
            "execution_microstructure": "Bitunix live-only and below 30 distinct days",
        },
        "costs": {
            "alpha_target": "gross_return_r",
            "alpha_maker_round_trip_bps": BASE_COST_BPS,
            "alpha_maker_stress_bps": 2 * BASE_COST_BPS,
            "bitunix_transfer_diagnostic_bps": SHADOW_COST_BPS,
            "bitunix_transfer_stress_bps": 2 * SHADOW_COST_BPS,
        },
        "holdout": {
            "status": "sealed",
            "opened": False,
            "discovery_cutoff": protocol["discovery_cutoff"],
        },
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if audit["ready"] and not smoke:
        _status("final_fit", "Frozen BTC external shadow models", 94, block="4/5")
        final_matrix = assemble_matrix(matrices, cross_exchange_features(local, EXCHANGES))
        times = pd.to_datetime(final_matrix["signal_timestamp"], utc=True)
        end = times.max()
        previous = final_matrix.loc[times.ge(end - pd.Timedelta(weeks=56))].copy()
        final_models = fit_policy(previous, app.machine_learning)
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        _atomic_joblib(
            BUNDLE_PATH,
            {
                "protocol": PROTOCOL,
                "protocol_sha256": protocol["protocol_sha256"],
                "models": final_models,
                "experts": v11_experts(),
                "features": MODEL_COLUMNS,
                "training_exchanges": EXCHANGES,
                "target_exchange": "bitunix",
                "shadow_only": True,
                "deployable": False,
                "base_cost_bps": BASE_COST_BPS,
            },
        )
        report["bundle"] = str(BUNDLE_PATH)
    _atomic_json(REPORT_PATH, report)
    _status("complete", report["verdict"], 100, block="5/5")
    return report


def btc_inventory() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        if item["symbol"] != "BTCUSDT" or item["exchange"] not in EXCHANGES:
            continue
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"source hash mismatch: {path}")
        result[item["exchange"]] = cast(dict[str, Any], item)
    if set(result) != set(EXCHANGES):
        raise RuntimeError("V11 requires BTC Binance, OKX and Bybit archives")
    return result


def build_feature_frame(
    path: Path,
    app: AppConfig,
    exchange: str,
    *,
    timeframe_minutes: int = 15,
    vwap_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(path).copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed", utc=True)
    raw = raw.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if "funding_coverage" in raw:
        funding_coverage = raw["funding_coverage"].fillna(False).astype(bool)
    else:
        events = pd.to_numeric(raw["funding_event_rate"], errors="coerce").ne(0)
        funding_coverage = (
            raw["timestamp"].ge(raw.loc[events, "timestamp"].min())
            if events.any()
            else pd.Series(False, index=raw.index)
        )
    raw["funding_coverage"] = funding_coverage
    raw.loc[~funding_coverage, ["funding_rate", "funding_event_rate"]] = np.nan
    bars = _resample_alpha(raw, timeframe_minutes)
    vwap_window = vwap_hours * 60 // timeframe_minutes
    feature_app = app.model_copy(
        update={
            "strategy": app.strategy.model_copy(
                update={
                    "timeframe_minutes": timeframe_minutes,
                    "crypto_vwap_window": vwap_window,
                }
            )
        }
    )
    features, _ = build_ml_features(bars, feature_app, alpha_gross=True)
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    features["return_48"] = features["close"].pct_change(48, fill_method=None)
    features["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * timestamps.dt.weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * timestamps.dt.weekday / 7)
    features["regime_code"] = causal_regime_code(features)
    features["feature_available_at"] = timestamps + pd.Timedelta(minutes=timeframe_minutes)
    features["signal_timestamp"] = features["feature_available_at"]
    features["exchange"] = exchange
    features["symbol"] = "BTCUSDT"
    features["local_feature_coverage"] = (
        features[[name for name in MODEL_COLUMNS if name in features]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    return raw, features


def causal_regime_code(features: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=features.index, dtype=float)
    complete = features[["adx", "atr_percentile", "ema50_slope"]].notna().all(axis=1)
    result.loc[complete & features["atr_percentile"].gt(90)] = 4.0
    result.loc[complete & features["atr_percentile"].le(90) & features["adx"].lt(20)] = 0.0
    result.loc[
        complete
        & features["atr_percentile"].le(90)
        & features["adx"].gt(25)
        & features["ema50_slope"].gt(0)
    ] = 1.0
    result.loc[
        complete
        & features["atr_percentile"].le(90)
        & features["adx"].gt(25)
        & features["ema50_slope"].lt(0)
    ] = 2.0
    result.loc[complete & result.isna()] = 3.0
    return result


def cross_exchange_features(
    frames: dict[str, pd.DataFrame], allowed: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    if len(allowed) < 2 or not set(allowed).issubset(frames):
        raise ValueError("cross-exchange context requires at least two source venues")
    returns = pd.concat(
        [
            frames[name][["timestamp", "return_1", "feature_available_at"]].assign(source=name)
            for name in allowed
        ],
        ignore_index=True,
    )
    values = returns.pivot(index="timestamp", columns="source", values="return_1")[list(allowed)]
    availability = returns.pivot(
        index="timestamp", columns="source", values="feature_available_at"
    )[list(allowed)]
    stats = pd.DataFrame(
        {
            "timestamp": values.index,
            "cross_exchange_return_median": values.median(axis=1, skipna=False),
            "cross_exchange_return_dispersion": values.std(axis=1, ddof=0, skipna=False),
            "cross_feature_available_at": availability.max(axis=1),
        }
    ).reset_index(drop=True)
    result: dict[str, pd.DataFrame] = {}
    for exchange in EXCHANGES:
        joined = frames[exchange].merge(stats, on="timestamp", how="left", validate="one_to_one")
        joined["lookahead_valid"] = pd.to_datetime(
            joined["cross_feature_available_at"], utc=True
        ).le(pd.to_datetime(joined["signal_timestamp"], utc=True))
        joined["feature_coverage"] = (
            joined["local_feature_coverage"].astype(bool)
            & joined[["cross_exchange_return_median", "cross_exchange_return_dispersion"]]
            .notna()
            .all(axis=1)
            & joined["lookahead_valid"]
        )
        result[exchange] = joined
    return result


def expert_entry_mask(features: pd.DataFrame, expert: V11Expert) -> pd.Series:
    direction = 1 if expert.side == "long" else -1
    z = features["distance_vwap_atr"].astype(float) * direction
    close = features["close"].astype(float)
    safe = features["atr_percentile"].le(90) & features["data_valid"].fillna(False).astype(bool)
    if expert.entry_kind == "range_exhaustion":
        eligible = z.le(-expert.entry_z) & features["adx"].lt(20)
    elif expert.entry_kind == "confirmed_reentry":
        eligible = (
            z.shift(1).le(-expert.entry_z)
            & z.gt(-expert.entry_z)
            & z.lt(-0.5)
            & features["adx"].lt(25)
        )
    elif expert.entry_kind == "multi_horizon":
        eligible = features["return_12"].mul(direction).gt(0) & features["return_48"].mul(
            direction
        ).gt(0)
        eligible &= features["ema50_slope"].mul(direction).gt(0)
    else:
        prior = (
            features["high"].shift(1).rolling(expert.breakout_bars).max()
            if direction > 0
            else features["low"].shift(1).rolling(expert.breakout_bars).min()
        )
        eligible = close.gt(prior) if direction > 0 else close.lt(prior)
        eligible &= features["adx"].ge(20)
        if expert.entry_kind == "trend_continuation":
            eligible &= features["ema20_slope"].mul(direction).gt(0) & features["ema50_slope"].mul(
                direction
            ).gt(0)
        elif expert.entry_kind == "volatility":
            eligible &= (
                features["adx_slope"].gt(0)
                & features["atr_percentile"].between(40, 90)
                & features["relative_volume"].gt(1)
            )
    return eligible & safe & features["local_feature_coverage"].fillna(False).astype(bool)


def evaluate_expert(
    features: pd.DataFrame,
    raw: pd.DataFrame,
    expert: V11Expert,
    *,
    cost_bps: float = BASE_COST_BPS,
    entry_mask: pd.Series | None = None,
    stop_prices: pd.Series | None = None,
    timeframe_minutes: int = 15,
    vwap_hours: float = 24.0,
) -> pd.DataFrame:
    signals = expert_entry_mask(features, expert) if entry_mask is None else entry_mask
    if not signals.index.equals(features.index):
        raise ValueError("entry mask must share the feature index")
    if stop_prices is not None and not stop_prices.index.equals(features.index):
        raise ValueError("stop prices must share the feature index")
    signal_indexes = np.flatnonzero(signals.fillna(False).to_numpy(dtype=bool))
    if not len(signal_indexes):
        return pd.DataFrame()
    raw_times = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], utc=True))
    signal_times = pd.DatetimeIndex(
        pd.to_datetime(features.iloc[signal_indexes]["signal_timestamp"], utc=True)
    )
    entry_indexes = raw_times.get_indexer(signal_times)
    count = len(signal_indexes)
    direction = 1.0 if expert.side == "long" else -1.0
    valid = entry_indexes >= 0
    entry = np.full(count, np.nan)
    atr = features.iloc[signal_indexes]["atr"].to_numpy(dtype=float)
    entry[valid] = raw["open"].to_numpy(dtype=float)[entry_indexes[valid]]
    if stop_prices is None:
        risk = expert.stop_atr * atr
        stop = entry - direction * risk
    else:
        stop = stop_prices.iloc[signal_indexes].to_numpy(dtype=float)
        risk = direction * (entry - stop)
    valid &= np.isfinite(entry) & np.isfinite(risk) & (entry > 0) & (risk > 0)
    valid &= risk <= expert.stop_atr * atr
    initial_stop = stop.copy()
    center = features.iloc[signal_indexes]["vwap"].to_numpy(dtype=float)
    if expert.target_atr is not None:
        target = entry + direction * expert.target_atr * atr
    elif expert.target_z is not None:
        target = center - direction * expert.target_z * atr
    else:
        target = np.full(count, np.nan)
    crossed_target = (
        valid & np.isfinite(target) & ((target <= entry) if direction > 0 else (target >= entry))
    )
    active = valid & ~crossed_target
    completed = crossed_target.copy()
    executed = valid & ~crossed_target
    path_valid = valid.copy()
    gross = np.full(count, np.nan)
    exit_price = np.full(count, np.nan)
    exit_time = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    exit_time[crossed_target] = (
        signal_times[crossed_target].tz_localize(None).to_numpy(dtype="datetime64[ns]")
    )
    reason = np.full(count, "", dtype=object)
    reason[crossed_target] = "entry_target_crossed"
    gross[crossed_target] = 0.0
    exit_price[crossed_target] = entry[crossed_target]
    mae = np.zeros(count)
    mfe = np.zeros(count)
    funding_sum = np.zeros(count)
    funding_complete = valid.copy()
    opens = raw["open"].to_numpy(dtype=float)
    highs = raw["high"].to_numpy(dtype=float)
    lows = raw["low"].to_numpy(dtype=float)
    closes = raw["close"].to_numpy(dtype=float)
    marks_high = raw["mark_high"].to_numpy(dtype=float)
    marks_low = raw["mark_low"].to_numpy(dtype=float)
    raw_valid = raw["data_valid"].fillna(False).to_numpy(dtype=bool)
    funding = pd.to_numeric(raw["funding_event_rate"], errors="coerce").to_numpy(dtype=float)
    holding_minutes = expert.maximum_holding_bars * timeframe_minutes
    for offset in range(holding_minutes):
        indexes = entry_indexes + offset
        in_range = active & (indexes >= 0) & (indexes < len(raw))
        positions = np.flatnonzero(in_range)
        if len(positions):
            expected = signal_times[positions] + pd.to_timedelta(offset, unit="min")
            contiguous = raw_times[indexes[positions]].to_numpy(
                dtype="datetime64[ns]"
            ) == expected.to_numpy(dtype="datetime64[ns]")
            finite = (
                np.isfinite(opens[indexes[positions]])
                & np.isfinite(highs[indexes[positions]])
                & np.isfinite(lows[indexes[positions]])
                & np.isfinite(closes[indexes[positions]])
                & np.isfinite(marks_high[indexes[positions]])
                & np.isfinite(marks_low[indexes[positions]])
            )
            good = contiguous & finite & raw_valid[indexes[positions]]
            bad_positions = positions[~good]
            active[bad_positions] = False
            path_valid[bad_positions] = False
            positions = positions[good]
        missing = active & ~in_range
        active[missing] = False
        path_valid[missing] = False
        if not len(positions):
            continue
        indexes = entry_indexes[positions] + offset
        adverse = np.where(
            direction > 0, lows[indexes] - entry[positions], entry[positions] - highs[indexes]
        )
        favorable = np.where(
            direction > 0, highs[indexes] - entry[positions], entry[positions] - lows[indexes]
        )
        mae[positions] = np.minimum(mae[positions], adverse / risk[positions])
        mfe[positions] = np.maximum(mfe[positions], favorable / risk[positions])
        observed_funding = funding[indexes]
        funding_complete[positions] &= np.isfinite(observed_funding)
        funding_sum[positions] += np.nan_to_num(observed_funding)
        hit_stop = (
            lows[indexes] <= stop[positions] if direction > 0 else highs[indexes] >= stop[positions]
        )
        hit_target = np.isfinite(target[positions]) & (
            highs[indexes] >= target[positions]
            if direction > 0
            else lows[indexes] <= target[positions]
        )
        hit = hit_stop | hit_target
        if hit.any():
            hit_positions = positions[hit]
            hit_indexes = indexes[hit]
            stopped = hit_stop[hit]
            prices = np.where(
                stopped,
                np.minimum(stop[hit_positions], opens[hit_indexes])
                if direction > 0
                else np.maximum(stop[hit_positions], opens[hit_indexes]),
                target[hit_positions],
            )
            exit_price[hit_positions] = prices
            gross[hit_positions] = direction * (prices - entry[hit_positions]) / risk[hit_positions]
            exit_time[hit_positions] = (
                (raw_times[hit_indexes] + pd.Timedelta(minutes=1))
                .tz_localize(None)
                .to_numpy(dtype="datetime64[ns]")
            )
            reason[hit_positions] = np.where(stopped, "stop", "target")
            completed[hit_positions] = True
            active[hit_positions] = False
        if expert.trailing_atr is not None and (offset + 1) % timeframe_minutes == 0:
            trailing_positions = np.flatnonzero(active)
            if len(trailing_positions):
                bars_ahead = (offset + 1) // timeframe_minutes
                feature_indexes = signal_indexes[trailing_positions] + bars_ahead
                inside = feature_indexes < len(features)
                trailing_positions = trailing_positions[inside]
                feature_indexes = feature_indexes[inside]
                proposed = features.iloc[feature_indexes]["close"].to_numpy(
                    dtype=float
                ) - direction * expert.trailing_atr * features.iloc[feature_indexes][
                    "atr"
                ].to_numpy(dtype=float)
                stop[trailing_positions] = (
                    np.maximum(stop[trailing_positions], proposed)
                    if direction > 0
                    else np.minimum(stop[trailing_positions], proposed)
                )
        if offset == holding_minutes - 1:
            timed = np.flatnonzero(active)
            if len(timed):
                indexes = entry_indexes[timed] + offset
                exit_price[timed] = closes[indexes]
                gross[timed] = direction * (exit_price[timed] - entry[timed]) / risk[timed]
                exit_time[timed] = (
                    (raw_times[indexes] + pd.Timedelta(minutes=1))
                    .tz_localize(None)
                    .to_numpy(dtype="datetime64[ns]")
                )
                reason[timed] = "time"
                completed[timed] = True
                active[timed] = False
    keep = completed & path_valid & np.isfinite(gross)
    if not keep.any():
        return pd.DataFrame()
    rows = features.iloc[signal_indexes[keep]].copy().reset_index(drop=True)
    cost_r = np.where(executed[keep], cost_bps / (risk[keep] / entry[keep] * 10_000), 0.0)
    shadow_cost_r = np.where(
        executed[keep], SHADOW_COST_BPS / (risk[keep] / entry[keep] * 10_000), 0.0
    )
    rows["expert_id"] = expert.expert_id
    rows["expert_name"] = expert.name
    rows["family"] = expert.family
    rows["side"] = expert.side
    rows["entry_price"] = entry[keep]
    rows["exit_price"] = exit_price[keep]
    rows["exit_timestamp"] = pd.to_datetime(exit_time[keep], utc=True)
    rows["exit_reason"] = reason[keep]
    rows["initial_stop_price"] = initial_stop[keep]
    rows["final_stop_price"] = stop[keep]
    rows["gross_return_r"] = gross[keep]
    rows["net_return_r"] = gross[keep]
    rows["net_return_r_1x"] = gross[keep] - cost_r
    rows["net_return_r_2x"] = gross[keep] - 2 * cost_r
    rows["net_return_r_shadow_19bps"] = gross[keep] - shadow_cost_r
    rows["net_return_r_shadow_38bps"] = gross[keep] - 2 * shadow_cost_r
    rows["cost_r_1x"] = cost_r
    rows["funding_return_r"] = np.where(
        funding_complete[keep], -direction * funding_sum[keep] / (risk[keep] / entry[keep]), np.nan
    )
    rows["funding_coverage"] = funding_complete[keep]
    rows["mae_r"] = mae[keep]
    rows["mfe_r"] = mfe[keep]
    rows["timeframe_minutes"] = timeframe_minutes
    rows["vwap_hours"] = vwap_hours
    rows["entry_z"] = expert.entry_z
    rows["entry_rule_code"] = float(
        (
            "range_exhaustion",
            "confirmed_reentry",
            "breakout",
            "trend_continuation",
            "volatility",
            "multi_horizon",
        ).index(expert.entry_kind)
    )
    rows["stop_atr"] = expert.stop_atr
    rows["exit_z"] = expert.target_z or 0.0
    rows["time_stop_hours"] = expert.maximum_holding_bars * timeframe_minutes / 60
    rows["range_adx_threshold"] = 25.0 if expert.entry_kind == "confirmed_reentry" else 20.0
    rows["regime_policy_code"] = 0.0 if expert.family == "mean_reversion" else 1.0
    rows["confirmation_bars"] = 1.0 if expert.entry_kind == "confirmed_reentry" else 0.0
    rows["execution_valid"] = executed[keep]
    return rows


def build_base_matrices(
    config_path: Path,
    inventory: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, pd.DataFrame]:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    output: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for exchange in EXCHANGES:
        path = MATRIX_ROOT / f"{exchange}.parquet"
        if resume and path.exists():
            frame = pd.read_parquet(path)
            if (
                frame.get("protocol_sha256", pd.Series(dtype=str))
                .eq(protocol["protocol_sha256"])
                .all()
            ):
                output[exchange] = frame
                continue
        missing.append(exchange)
    if missing:
        with ProcessPoolExecutor(max_workers=len(missing)) as pool:
            jobs = {
                pool.submit(
                    _matrix_worker,
                    exchange,
                    inventory[exchange]["path"],
                    str(config_path),
                    protocol["protocol_sha256"],
                ): exchange
                for exchange in missing
            }
            for completed, future in enumerate(as_completed(jobs), start=1):
                exchange = jobs[future]
                matrix = future.result()
                _atomic_parquet(MATRIX_ROOT / f"{exchange}.parquet", matrix)
                output[exchange] = matrix
                _status(
                    "matrix",
                    f"{completed}/{len(missing)} {exchange.upper()} 1m outcomes",
                    5 + 15 * completed / len(missing),
                    block="1/5",
                )
    return output


def _matrix_worker(
    exchange: str, source: str, config_path: str, protocol_sha256: str
) -> pd.DataFrame:
    app = load_config(Path(config_path))
    raw, features = build_feature_frame(Path(source), app, exchange)
    rows = [evaluate_expert(features, raw, expert) for expert in v11_experts()]
    matrix = (
        pd.concat([row for row in rows if not row.empty], ignore_index=True)
        if any(not row.empty for row in rows)
        else pd.DataFrame()
    )
    matrix["protocol_sha256"] = protocol_sha256
    return matrix


def assemble_matrix(
    base: dict[str, pd.DataFrame], augmented: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    cross = [
        "cross_exchange_return_median",
        "cross_exchange_return_dispersion",
        "cross_feature_available_at",
        "lookahead_valid",
        "feature_coverage",
    ]
    for exchange in EXCHANGES:
        features = augmented[exchange][["signal_timestamp", *cross]].drop_duplicates(
            "signal_timestamp"
        )
        rows = (
            base[exchange]
            .drop(columns=cross, errors="ignore")
            .merge(features, on="signal_timestamp", how="left", validate="many_to_one")
        )
        finite = rows[list(MODEL_COLUMNS)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        output.append(rows.loc[rows["feature_coverage"].fillna(False).astype(bool) & finite])
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["signal_timestamp", "exchange", "expert_id"])
        .reset_index(drop=True)
    )


def fit_policy(
    previous: pd.DataFrame,
    ml: MachineLearningConfig | None,
    *,
    progress: tuple[int, int] | None = None,
    admission_start: pd.Timestamp | None = None,
) -> dict[str, dict[str, Any]]:
    if ml is None:
        raise ValueError("machine learning configuration is required")
    rows = previous.copy()
    times = pd.to_datetime(rows["signal_timestamp"], utc=True)
    audit_start = admission_start or times.max() - pd.Timedelta(weeks=4)
    development = rows.loc[times.lt(audit_start)].copy()
    audit = rows.loc[times.ge(audit_start)].copy()
    development["_training_weight"] = balanced_exchange_symbol_weights(development)
    config = ml.model_copy(
        update={"model_trials_per_side": MODEL_TRIALS, "status_path": STATUS_PATH}
    )
    fitted = {
        side: _fit_side(
            development,
            cast(Any, side),
            config,
            progress=progress,
            objective_kind="decision_regret",
            excluded_features=("exchange_code", "funding_rate"),
            benchmark_gate=True,
            ensemble_models=ENSEMBLE_MODELS,
            temporal_bootstrap=True,
            extra_features=EXTRA_FEATURES,
            decision_actual_column="net_return_r_1x",
            decision_cost_column="cost_r_1x",
            decision_group_columns=("exchange", "signal_timestamp"),
            calibration_kind="bias",
        )
        for side in SIDES
    }
    if any(model.get("champion") == "xgboost" for model in fitted.values()):
        selected = predict_policy(audit, fitted)
        ridge_models = {side: force_ridge(model) for side, model in fitted.items()}
        ridge = predict_policy(audit, ridge_models)
        selected_regret = decision_regret(selected, "ev_gross")
        ridge_regret = decision_regret(ridge, "ev_gross")
        selected_lcb = decision_regret(selected, "lcb_gross")
        ridge_lcb = decision_regret(ridge, "lcb_gross")
        if selected_regret >= ridge_regret or selected_lcb > ridge_lcb:
            fitted = ridge_models
    admission_candidates = predict_policy(audit, fitted)
    for side, model in fitted.items():
        admission = replay_policy(
            admission_candidates.loc[admission_candidates["side"].eq(side)], require_lcb=False
        )
        values = admission.get("net_return_r_1x", pd.Series(dtype=float)).to_numpy(dtype=float)
        lower = (
            moving_block_lower_bound(values, block_size=min(7, len(values)), seed=20260804)
            if len(values)
            else float("-inf")
        )
        residuals = (
            admission["net_return_r_1x"].to_numpy(dtype=float)
            - admission["ev_net"].to_numpy(dtype=float)
            if len(admission)
            else np.asarray([], dtype=float)
        )
        residual_lower = (
            moving_block_lower_bound(residuals, block_size=min(7, len(residuals)), seed=20260804)
            if len(residuals)
            else float("-inf")
        )
        admission_weeks = (
            pd.to_datetime(admission["signal_timestamp"], utc=True)
            .dt.tz_localize(None)
            .dt.to_period("W")
            if len(admission)
            else pd.Series(dtype="period[W]")
        )
        weekly = admission["net_return_r_1x"].groupby(admission_weeks).sum()
        positive_week_fraction = float((weekly > 0).mean()) if len(weekly) else 0.0
        admitted = (
            len(values) >= 50
            and float(values.mean()) > 0
            and len(weekly) >= 3
            and positive_week_fraction > 0.5
        )
        model["absolute_admission"] = {
            "trades": len(values),
            "expectancy_net_r": float(values.mean()) if len(values) else 0.0,
            "lower_bound_net_r": lower,
            "lower_bound_is_final_oos_gate_only": True,
            "selected_residual_lower_r": residual_lower,
            "weeks": len(weekly),
            "positive_week_fraction": positive_week_fraction,
            "passed": admitted,
        }
        if admitted:
            model["residual_lower"] = min(float(model["residual_lower"]), residual_lower)
        else:
            model["enabled"] = False
            model["reason"] = "no_positive_net_edge_on_chronological_admission_audit"
    return fitted


def force_ridge(model: dict[str, Any]) -> dict[str, Any]:
    if not model.get("enabled") or "benchmark" not in model:
        return model
    result = dict(model)
    result.update(
        models=[model["benchmark"]],
        calibrator=model["benchmark_calibrator"],
        residual_lower=model["benchmark_residual_lower"],
        champion="ridge_global_default",
    )
    return result


def predict_policy(rows: pd.DataFrame, fitted: dict[str, dict[str, Any]]) -> pd.DataFrame:
    predicted: list[pd.DataFrame] = []
    for side, model in fitted.items():
        if not model.get("enabled"):
            continue
        selected = rows.loc[rows["side"].eq(side)].copy()
        if selected.empty:
            continue
        complete = (
            selected[model["features"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        )
        selected = selected.loc[complete].copy()
        if selected.empty:
            continue
        values = selected[model["features"]].to_numpy(dtype=float)
        ensemble = np.vstack([_xgb_predict(member, values) for member in model["models"]])
        selected["ev_gross"] = model["calibrator"].predict(ensemble.mean(axis=0))
        selected["ensemble_dispersion"] = ensemble.std(axis=0, ddof=0)
        selected["lcb_gross"] = np.minimum(
            selected["ev_gross"],
            selected["ev_gross"]
            + float(model["residual_lower"])
            - 1.645 * selected["ensemble_dispersion"],
        )
        selected["ev_net"] = selected["ev_gross"] - selected["cost_r_1x"]
        selected["lcb_net"] = selected["lcb_gross"] - selected["cost_r_1x"]
        predicted.append(selected)
    return pd.concat(predicted, ignore_index=True) if predicted else rows.iloc[:0].copy()


def attach_predictions(testing: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    keys = ["exchange", "signal_timestamp", "expert_id"]
    columns = ["ev_gross", "ensemble_dispersion", "lcb_gross", "ev_net", "lcb_net"]
    values = predicted.copy()
    for column in columns:
        if column not in values:
            values[column] = np.nan
    return testing.merge(values[[*keys, *columns]], on=keys, how="left", validate="one_to_one")


def decision_regret(rows: pd.DataFrame, prediction: str) -> float:
    if rows.empty:
        return float("inf")
    regrets: list[float] = []
    for _, choices in rows.groupby(["exchange", "signal_timestamp"], sort=False):
        choice = choices.sort_values([prediction, "expert_id"], ascending=[False, True]).iloc[0]
        predicted_net = float(choice[prediction]) - float(choice["cost_r_1x"])
        chosen = float(choice["net_return_r_1x"]) if predicted_net > 0 else 0.0
        regrets.append(max(0.0, float(choices["net_return_r_1x"].max())) - chosen)
    return statistics.fmean(regrets) if regrets else float("inf")


def replay_policy(
    candidates: pd.DataFrame,
    *,
    value_prefix: Literal["net", "gross"] = "net",
    cooldown_minutes: int = COOLDOWN_MINUTES,
    require_lcb: bool = True,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ev = f"ev_{value_prefix}"
    lcb = f"lcb_{value_prefix}"
    eligible = candidates[ev].gt(0) & candidates["feature_coverage"].astype(bool)
    eligible &= candidates["execution_valid"].astype(bool)
    if require_lcb:
        eligible &= candidates[lcb].gt(0)
    valid = candidates.loc[eligible].copy()
    if valid.empty:
        return valid
    score = lcb if require_lcb else ev
    proposals = valid.sort_values(
        ["exchange", "signal_timestamp", score, "expert_id"], ascending=[True, True, False, True]
    ).drop_duplicates(["exchange", "signal_timestamp"])
    selected: list[pd.DataFrame] = []
    for _, rows in proposals.groupby("exchange", sort=True):
        blocked_until = pd.Timestamp("1900-01-01", tz="UTC")
        for index in range(len(rows)):
            row = rows.iloc[[index]]
            signal = pd.to_datetime(row.iloc[0]["signal_timestamp"], utc=True)
            if signal <= blocked_until:
                continue
            selected.append(row)
            blocked_until = pd.to_datetime(row.iloc[0]["exit_timestamp"], utc=True) + pd.Timedelta(
                minutes=cooldown_minutes
            )
    return pd.concat(selected, ignore_index=True) if selected else valid.iloc[:0].copy()


def replay_static(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    data = rows.loc[rows["execution_valid"].astype(bool)].sort_values(
        ["exchange", "signal_timestamp", "expert_id"]
    )
    selected: list[pd.DataFrame] = []
    for _, exchange_rows in data.groupby("exchange", sort=True):
        blocked_until = pd.Timestamp("1900-01-01", tz="UTC")
        for index in range(len(exchange_rows)):
            row = exchange_rows.iloc[[index]]
            signal = pd.to_datetime(row.iloc[0]["signal_timestamp"], utc=True)
            if signal <= blocked_until:
                continue
            selected.append(row)
            blocked_until = pd.to_datetime(row.iloc[0]["exit_timestamp"], utc=True) + pd.Timedelta(
                minutes=COOLDOWN_MINUTES
            )
    return pd.concat(selected, ignore_index=True) if selected else data.iloc[:0].copy()


def return_metrics(rows: pd.DataFrame, column: str) -> dict[str, float]:
    values = rows.get(column, pd.Series(dtype=float)).to_numpy(dtype=float)
    if not len(values):
        return {
            "trades": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    equity = np.cumprod(1 + 0.01 * values)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    losses = abs(float(values[values < 0].sum()))
    wins = float(values[values > 0].sum())
    return {
        "trades": float(len(values)),
        "expectancy_r": float(values.mean()),
        "profit_factor": wins / losses if losses else (999.0 if wins else 0.0),
        "max_drawdown": float(np.max((peaks - equity) / peaks)),
        "win_rate": float((values > 0).mean()),
    }


def daily_market_returns(
    rows: pd.DataFrame, column: str, calendar: pd.DatetimeIndex | None = None
) -> pd.Series:
    if rows.empty:
        return (
            pd.Series(0.0, index=calendar, dtype=float)
            if calendar is not None
            else pd.Series(dtype=float)
        )
    data = rows.assign(day=pd.to_datetime(rows["signal_timestamp"], utc=True).dt.floor("1D"))
    venue = data.groupby(["day", "exchange"])[column].sum().unstack("exchange")
    result = venue.reindex(columns=EXCHANGES, fill_value=0.0).fillna(0.0).median(axis=1)
    return (
        result.reindex(calendar, fill_value=0.0).sort_index()
        if calendar is not None
        else result.sort_index()
    )


def audit_oos(decisions: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(candidates["signal_timestamp"], utc=True).dt.floor("1D").unique())
    )
    metrics = {
        exchange: return_metrics(rows, "net_return_r_1x")
        for exchange, rows in decisions.groupby("exchange")
    }
    gross_metrics = {
        exchange: return_metrics(rows, "gross_return_r")
        for exchange, rows in decisions.groupby("exchange")
    }
    stress_metrics = {
        exchange: return_metrics(rows, "net_return_r_2x")
        for exchange, rows in decisions.groupby("exchange")
    }
    transfer_metrics = {
        exchange: return_metrics(rows, "net_return_r_shadow_19bps")
        for exchange, rows in decisions.groupby("exchange")
    }
    transfer_stress_metrics = {
        exchange: return_metrics(rows, "net_return_r_shadow_38bps")
        for exchange, rows in decisions.groupby("exchange")
    }
    lower = {
        exchange: moving_block_lower_bound(
            values,
            block_size=min(7, len(values)),
            seed=20260804,
        )
        if len(values := rows["net_return_r_1x"].to_numpy(dtype=float))
        else float("-inf")
        for exchange, rows in decisions.groupby("exchange")
    }
    policies: dict[str, pd.Series] = {
        "model": daily_market_returns(decisions, "net_return_r_1x", calendar)
    }
    for expert in v11_experts():
        policies[expert.expert_id] = daily_market_returns(
            replay_static(candidates.loc[candidates["expert_id"].eq(expert.expert_id)]),
            "net_return_r_1x",
            calendar,
        )
    control_rows = candidates.loc[candidates["expert_name"].eq("v8_control")]
    policies["v8_control"] = daily_market_returns(
        replay_static(control_rows), "net_return_r_1x", calendar
    )
    daily_frame = pd.concat(policies, axis=1).fillna(0.0).sort_index()
    day_index = pd.DatetimeIndex(daily_frame.index)
    spa = spa_reality_check(daily_frame, control_expert_id="v8_control")
    month_index = day_index.tz_localize(None).to_period("M")
    months = sorted(month_index.unique())
    blocks = [
        [daily_frame.loc[month_index == month, column].tolist() for month in months]
        for column in daily_frame.columns
    ]
    pbo = combinatorial_pbo(blocks)
    sharpes = [
        float(policy_returns.mean() / policy_returns.std(ddof=1))
        for column in daily_frame.columns
        if len(policy_returns := daily_frame[column]) > 1 and policy_returns.std(ddof=1) > 0
    ]
    model_daily = (
        daily_frame["model"].to_numpy(dtype=float) if "model" in daily_frame else np.asarray([])
    )
    dsr_raw = deflated_sharpe_probability(model_daily, sharpes)
    dsr = float(dsr_raw) if np.isfinite(dsr_raw) else 0.0
    positive_months = (
        float((daily_frame["model"].groupby(month_index).sum() > 0).mean())
        if len(daily_frame)
        else 0.0
    )
    gates = {
        "three_exchanges_present": len(metrics) == 3,
        "minimum_100_trades_each_exchange": len(metrics) == 3
        and all(value["trades"] >= 100 for value in metrics.values()),
        "expectancy_positive_each_exchange": len(metrics) == 3
        and all(value["expectancy_r"] > 0 for value in metrics.values()),
        "lower_bound_positive_each_exchange": len(lower) == 3
        and all(value > 0 for value in lower.values()),
        "profit_factor_each_exchange": len(metrics) == 3
        and all(value["profit_factor"] >= 1.15 for value in metrics.values()),
        "drawdown_each_exchange": len(metrics) == 3
        and all(value["max_drawdown"] <= 0.08 for value in metrics.values()),
        "stress_2x_each_exchange": len(stress_metrics) == 3
        and all(value["expectancy_r"] >= 0 for value in stress_metrics.values()),
        "positive_window_majority": positive_months > 0.5,
        "pbo": pbo["pbo"] is not None and float(pbo["pbo"]) <= 0.20,
        "dsr": dsr >= 0.95,
        "spa": spa["spa_pvalue"] <= 0.05,
        "reality_check": spa["reality_check_pvalue"] <= 0.05,
    }
    return {
        "ready": all(gates.values()),
        "policy": "GLOBAL_LONG_SHORT_FLAT_ONE_POSITION_60M_COOLDOWN",
        "metrics_net_1x": metrics,
        "metrics_gross": gross_metrics,
        "metrics_stress_2x": stress_metrics,
        "metrics_bitunix_transfer_19bps": transfer_metrics,
        "metrics_bitunix_transfer_38bps": transfer_stress_metrics,
        "expectancy_lower_bound_r": lower,
        "positive_month_fraction": positive_months,
        "pbo": pbo,
        "dsr_probability": dsr,
        "multiple_comparison": spa,
        "bitunix_execution_ready": False,
        "bitunix_execution_reason": "requires at least 30 distinct observed execution days",
        "total_candidates_evaluated": len(policies) + MODEL_TRIALS * len(SIDES),
        "gates": gates,
    }


def preregister(config_path: Path, *, persist: bool = True) -> dict[str, Any]:
    inventory = btc_inventory()
    source_hash = hashlib.sha256(
        "".join(
            _sha256(path)
            for path in [
                Path(__file__),
                Path("src/adaptive_bot/expert_policy.py"),
                Path("src/adaptive_bot/hybrid_policy.py"),
                Path("src/adaptive_bot/ml_research.py"),
                *VARIANT_SOURCE_PATHS,
            ]
        ).encode()
    ).hexdigest()
    cutoff = min(pd.Timestamp(item["end"]) for item in inventory.values()).isoformat()
    immutable = {
        "protocol": PROTOCOL,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "symbols": ["BTCUSDT"],
        "experts": [asdict(expert) | {"expert_id": expert.expert_id} for expert in v11_experts()],
        "expert_universe_sha256": hashlib.sha256(
            json.dumps([expert.expert_id for expert in v11_experts()]).encode()
        ).hexdigest(),
        "source_sha256": source_hash,
        "config_sha256": _sha256(config_path),
        "source_manifest_sha256": _sha256(MANIFEST_PATH),
        "discovery_cutoff": cutoff,
        "alpha_target": "gross_return_r",
        "decision_policy": "GLOBAL_LONG_SHORT_FLAT_ONE_POSITION_60M_COOLDOWN",
        "walk_forward": [52, 4, 4, 4],
        "cost_bps": {
            "alpha_maker": [BASE_COST_BPS, 2 * BASE_COST_BPS],
            "bitunix_transfer_diagnostic": [SHADOW_COST_BPS, 2 * SHADOW_COST_BPS],
        },
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    payload = immutable | {
        "protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "run_id": datetime.now(UTC).strftime("hybrid-v11-%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "holdout": {"status": "sealed", "opened": False},
        "automatic_live": False,
    }
    if not persist:
        return payload
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != payload["protocol_sha256"]:
            raise RuntimeError("V11 protocol changed after freezing")
        return cast(dict[str, Any], existing)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PROTOCOL_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return payload


def invalidate_v10() -> None:
    if V10_INVALIDATION_PATH.exists():
        return
    _atomic_json(
        V10_INVALIDATION_PATH,
        {
            "protocol": "hybrid_transfer_v10_btc_external_alpha",
            "status": "INVALIDATED_METHODOLOGY",
            "reasons": [
                "Alpha and execution costs mixed",
                "incomplete OKX funding truncated history",
                "15m outcome path used instead of 1m",
                "side-specific selection instead of one global chronological policy",
            ],
            "invalidated_at": datetime.now(UTC).isoformat(),
        },
    )


def finish_no_policy(
    protocol: dict[str, Any], inventory: dict[str, Any], reason: str
) -> dict[str, Any]:
    report = {
        "protocol": PROTOCOL,
        "run_id": protocol["run_id"],
        "verdict": "NO_ALPHA_POLICY",
        "action": "FLAT",
        "deployable": False,
        "reason": reason,
        "data": inventory,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    _status("complete", "NO_ALPHA_POLICY", 100, block="5/5")
    return report


def _fold_status(
    held_out_number: int,
    held_out: str,
    fold: int,
    folds: int,
    completed: int,
    total: int,
    state: str,
) -> None:
    _status(
        "outer_folds",
        f"{state} LOEO {held_out_number}/3 {held_out.upper()} fold {fold}/{folds}",
        20 + 70 * completed / max(1, total),
        block="2/5",
        exchange=held_out,
        fold=fold,
        folds=folds,
        completed_units=completed,
        total_units=total,
    )


def _status(phase: str, detail: str, percent: float, **extra: Any) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "heartbeat": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.joblib")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="V11 BTC external Alpha research")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    try:
        report = run_v11(
            load_config(arguments.config),
            arguments.config,
            resume=arguments.resume,
            smoke=arguments.smoke,
        )
    except Exception as error:
        _status("failed", f"{type(error).__name__}: {error}", 0, block="FAILED")
        raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
