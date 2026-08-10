from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.expert_policy import moving_block_lower_bound, purged_expert_folds
from adaptive_bot.hybrid_policy_v9 import (
    NEW_FEATURES,
    SIDES,
    _positive_window_fraction,
    _return_metrics,
    build_expert_matrix,
    build_local_feature_frames,
    fit_v9_fold,
    predict_v9,
    select_v9_actions,
    spa_reality_check,
    v9_experts,
)

PROTOCOL = "hybrid_transfer_v10_btc_external_alpha"
EXCHANGES = ("binance", "okx", "bybit")
ROOT = Path("data/models/expert_policy/v10_external_global_v2")
PROTOCOL_PATH = ROOT / "protocol.json"
BUNDLE_PATH = ROOT / "bundle.joblib"
REPORT_PATH = Path("data/reports/ml_hybrid_v10_external.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v10_external.status.json")
MATRIX_ROOT = Path("data/ml/hybrid_v10_external/base_matrix")
# The previous directory contains checkpoints produced before LONG/SHORT/FLAT
# were compared as one chronological policy. Never resume those decisions.
OOS_ROOT = Path("data/ml/hybrid_v10_external/oos_folds_global_v2")


def external_cross_features(
    frames: dict[str, pd.DataFrame], allowed: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    if len(allowed) < 2 or not set(allowed).issubset(frames):
        raise ValueError("external transfer requires at least two BTC source exchanges")
    returns = pd.concat(
        [frames[name][["timestamp", "return_1"]].assign(exchange=name) for name in allowed],
        ignore_index=True,
    ).pivot(index="timestamp", columns="exchange", values="return_1")
    output: dict[str, pd.DataFrame] = {}
    for exchange in EXCHANGES:
        # Every target sees the same causal source-market context. This avoids
        # a train/held-out distribution shift in leave-one-exchange-out.
        source = returns[list(allowed)]
        stats = pd.DataFrame(
            {
                "timestamp": returns.index.to_numpy(),
                "cross_exchange_return_median": source.median(axis=1, skipna=False),
                "cross_exchange_return_dispersion": source.std(axis=1, ddof=0, skipna=False),
            }
        ).reset_index(drop=True)
        joined = frames[exchange].merge(stats, on="timestamp", how="left", validate="one_to_one")
        joined["cross_exchange_coverage"] = joined[list(NEW_FEATURES[1:])].notna().all(axis=1)
        joined["lookahead_valid"] = pd.to_datetime(joined["feature_available_at"], utc=True).le(
            pd.to_datetime(joined["signal_timestamp"], utc=True)
        )
        joined["feature_coverage"] = (
            joined["funding_coverage"]
            & joined["cross_exchange_coverage"]
            & joined["lookahead_valid"]
        )
        output[exchange] = joined
    return output


def run_external_transfer(app: AppConfig, config_path: Path) -> dict[str, Any]:
    protocol = preregister(config_path)
    if app.machine_learning is None:
        raise ValueError("machine learning configuration is required")
    cost = (
        2 * app.machine_learning.taker_fee_bps
        + 2 * float(app.backtest.slippage_bps)
        + float(app.backtest.spread_bps)
    )
    _status("external_features", "BTC Binance/OKX/Bybit", 2)
    local = build_local_feature_frames(app)
    base = build_base_matrices(local, cost)
    candidates: list[pd.DataFrame] = []
    champion_counts: dict[str, int] = {}
    OOS_ROOT.mkdir(parents=True, exist_ok=True)
    for held_out_number, held_out in enumerate(EXCHANGES, start=1):
        allowed = tuple(exchange for exchange in EXCHANGES if exchange != held_out)
        augmented = external_cross_features(local, allowed)
        matrix = assemble_scenario_matrix(base, augmented)
        folds = purged_expert_folds(
            matrix,
            train_weeks=52,
            calibration_weeks=4,
            test_weeks=4,
            step_weeks=4,
        )
        for fold_number, fold in enumerate(folds, start=1):
            checkpoint = OOS_ROOT / f"{held_out}_fold_{fold_number:02d}.parquet"
            if checkpoint.exists():
                candidate_rows = pd.read_parquet(checkpoint)
                candidates.append(candidate_rows)
                _status(
                    "external_oos",
                    f"RESUME LOEO {held_out_number}/3 {held_out.upper()} "
                    f"fold {fold_number}/{len(folds)}",
                    20 + 65 * ((held_out_number - 1) + fold_number / max(1, len(folds))) / 3,
                )
                continue
            fitting = matrix.iloc[fold.train].loc[matrix.iloc[fold.train]["exchange"].ne(held_out)]
            calibration = matrix.iloc[fold.calibration].loc[
                matrix.iloc[fold.calibration]["exchange"].ne(held_out)
            ]
            testing = matrix.iloc[fold.test].loc[matrix.iloc[fold.test]["exchange"].eq(held_out)]
            predicted = []
            for side in SIDES:
                if not testing["side"].eq(side).any():
                    continue
                _status(
                    "external_oos",
                    f"LOEO {held_out_number}/3 {held_out.upper()} "
                    f"fold {fold_number}/{len(folds)} {side.upper()}",
                    20 + 65 * ((held_out_number - 1) + (fold_number - 1) / max(1, len(folds))) / 3,
                )
                fitted = fit_v9_fold(fitting, calibration, cast(Any, side), app.machine_learning)
                champion = str(fitted.get("champion", "disabled"))
                champion_counts[f"{held_out}:{side}:{champion}"] = (
                    champion_counts.get(f"{held_out}:{side}:{champion}", 0) + 1
                )
                rows = predict_v9(testing, fitted)
                if not rows.empty:
                    predicted.append(rows)
            if predicted:
                candidate_rows = pd.concat(predicted, ignore_index=True)
                candidate_rows["held_out_exchange"] = held_out
                temporary = checkpoint.with_suffix(".tmp.parquet")
                candidate_rows.to_parquet(temporary, index=False)
                os.replace(temporary, checkpoint)
                candidates.append(candidate_rows)
    candidate_oos = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    oos = (
        pd.concat(
            [select_policy_replay(rows) for _, rows in candidate_oos.groupby("exchange")],
            ignore_index=True,
        )
        if not candidate_oos.empty
        else pd.DataFrame()
    )
    audit = external_audit(oos, candidate_oos)
    _status("external_final_fit", "Fitting frozen shadow models", 88)
    final_matrix = assemble_scenario_matrix(base, external_cross_features(local, EXCHANGES))
    end = pd.to_datetime(final_matrix["signal_timestamp"], utc=True).max()
    calibration_start = end - pd.Timedelta(weeks=4)
    training_start = calibration_start - pd.Timedelta(weeks=52)
    fitting = final_matrix.loc[
        pd.to_datetime(final_matrix["signal_timestamp"], utc=True).between(
            training_start, calibration_start, inclusive="left"
        )
        & pd.to_datetime(final_matrix["exit_timestamp"], utc=True).lt(calibration_start)
    ]
    calibration = final_matrix.loc[
        pd.to_datetime(final_matrix["signal_timestamp"], utc=True).ge(calibration_start)
    ]
    models = {
        side: fit_v9_fold(fitting, calibration, cast(Any, side), app.machine_learning)
        for side in SIDES
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE_PATH.with_suffix(".tmp.joblib")
    joblib.dump(
        {
            "protocol": PROTOCOL,
            "run_id": protocol["run_id"],
            "models": models,
            "training_exchanges": EXCHANGES,
            "target_exchange": "bitunix",
            "shadow_only": True,
            "deployable": False,
            "external_audit_passed": audit["ready"],
        },
        temporary,
    )
    os.replace(temporary, BUNDLE_PATH)
    report = {
        "protocol": PROTOCOL,
        "run_id": protocol["run_id"],
        "verdict": "EXTERNAL_ALPHA_SHADOW_READY" if audit["ready"] else "EXTERNAL_DISCOVERY_ONLY",
        "action": "BITUNIX_SHADOW_ONLY",
        "deployable": False,
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix",
        "candidate_count": len(v9_experts()),
        "decision_policy": "GLOBAL_LONG_SHORT_FLAT_ONE_POSITION",
        "audit": audit,
        "champion_counts": champion_counts,
        "final_champions": {side: model.get("champion") for side, model in models.items()},
        "bundle": str(BUNDLE_PATH),
        "bitunix_in_training": False,
        "execution_model_enabled": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    _status("complete", report["verdict"], 100)
    return report


def build_base_matrices(local: dict[str, pd.DataFrame], cost: float) -> dict[str, pd.DataFrame]:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    output: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for exchange in EXCHANGES:
        path = MATRIX_ROOT / f"{exchange}.parquet"
        if path.exists():
            output[exchange] = pd.read_parquet(path)
        else:
            missing.append(exchange)
    if missing:
        with ProcessPoolExecutor(max_workers=len(missing)) as pool:
            jobs = {
                pool.submit(_build_base_matrix, exchange, local[exchange], cost): exchange
                for exchange in missing
            }
            complete = len(output)
            for future in as_completed(jobs):
                exchange = jobs[future]
                matrix = future.result()
                path = MATRIX_ROOT / f"{exchange}.parquet"
                temporary = path.with_suffix(".tmp.parquet")
                matrix.to_parquet(temporary, index=False)
                os.replace(temporary, path)
                output[exchange] = matrix
                complete += 1
                _status(
                    "external_matrix",
                    f"Base matrix {complete}/3 {exchange.upper()}",
                    5 + complete * 5,
                )
    return output


def _build_base_matrix(exchange: str, features: pd.DataFrame, cost: float) -> pd.DataFrame:
    values = features.copy()
    values["cross_exchange_return_median"] = 0.0
    values["cross_exchange_return_dispersion"] = 0.0
    values["cross_exchange_coverage"] = True
    values["lookahead_valid"] = True
    values["feature_coverage"] = values["funding_coverage"].astype(bool)
    return build_expert_matrix({exchange: values}, base_cost_bps=cost)


def select_policy_replay(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    proposals = select_v9_actions(candidate_rows).sort_values("signal_timestamp")
    selected: list[pd.DataFrame] = []
    available_at = pd.Timestamp("1900-01-01", tz="UTC")
    for index in range(len(proposals)):
        row = proposals.iloc[[index]]
        signal = pd.Timestamp(row.iloc[0]["signal_timestamp"])
        if signal < available_at:
            continue
        selected.append(row)
        available_at = pd.Timestamp(row.iloc[0]["exit_timestamp"])
    return pd.concat(selected, ignore_index=True) if selected else proposals.iloc[:0].copy()


def assemble_scenario_matrix(
    base: dict[str, pd.DataFrame], augmented: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    replace = [
        "cross_exchange_return_median",
        "cross_exchange_return_dispersion",
        "cross_exchange_coverage",
        "lookahead_valid",
        "feature_coverage",
    ]
    for exchange in EXCHANGES:
        features = augmented[exchange][["signal_timestamp", *replace]].drop_duplicates(
            "signal_timestamp"
        )
        rows = (
            base[exchange]
            .drop(columns=replace, errors="ignore")
            .merge(features, on="signal_timestamp", how="left", validate="many_to_one")
        )
        rows = rows.loc[
            rows["funding_coverage"].astype(bool)
            & rows["cross_exchange_coverage"].fillna(False).astype(bool)
            & rows["lookahead_valid"].fillna(False).astype(bool)
        ]
        output.append(rows)
    return (
        pd.concat(output, ignore_index=True).sort_values("signal_timestamp").reset_index(drop=True)
    )


def external_audit(decisions: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        exchange: _return_metrics(rows)
        for exchange, rows in decisions.groupby("exchange", sort=True)
    }
    side_metrics = (
        {
            side: {
                exchange: _return_metrics(rows)
                for exchange, rows in decisions.loc[decisions["side"].eq(side)].groupby("exchange")
            }
            for side in SIDES
        }
        if not decisions.empty
        else {side: {} for side in SIDES}
    )
    control = next(
        expert.expert_id
        for expert in v9_experts()
        if expert.side == "short" and expert.name == "v8_control"
    )
    daily = (
        candidates.assign(
            day=pd.to_datetime(candidates["signal_timestamp"], utc=True).dt.floor("1D")
        ).pivot_table(
            index="day", columns="expert_id", values="net_return_r", aggfunc="sum", fill_value=0
        )
        if not candidates.empty
        else pd.DataFrame()
    )
    returns = decisions.get("net_return_r", pd.Series(dtype=float)).to_numpy(dtype=float)
    lower = (
        moving_block_lower_bound(returns, block_size=7, seed=20260804)
        if len(returns)
        else float("-inf")
    )
    spa = spa_reality_check(daily, control_expert_id=control)
    gates = {
        "minimum_100_trades_each_exchange": len(metrics) == 3
        and all(value["trades"] >= 100 for value in metrics.values()),
        "all_three_positive": len(metrics) == 3
        and all(value["expectancy_r"] > 0 for value in metrics.values()),
        "all_three_profit_factor": len(metrics) == 3
        and all(value["profit_factor"] >= 1.15 for value in metrics.values()),
        "all_three_drawdown": len(metrics) == 3
        and all(value["max_drawdown"] <= 0.08 for value in metrics.values()),
        "stress_2x_non_negative": len(metrics) == 3
        and all(value["stress_expectancy_r"] >= 0 for value in metrics.values()),
        "pooled_lower_bound_positive": lower > 0,
        "positive_window_majority": _positive_window_fraction(decisions) > 0.5
        if not decisions.empty
        else False,
        "spa": spa["spa_pvalue"] <= 0.05,
        "reality_check": spa["reality_check_pvalue"] <= 0.05,
    }
    return {
        "ready": all(gates.values()),
        "policy": "GLOBAL_LONG_SHORT_FLAT_ONE_POSITION",
        "metrics": metrics,
        "side_diagnostics": side_metrics,
        "lower_bound_r": lower,
        "multiple_comparison": spa,
        "gates": gates,
    }


def preregister(config_path: Path) -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL,
        "run_id": datetime.now(UTC).strftime("v10-external-%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "training_exchanges": list(EXCHANGES),
        "target_exchange": "bitunix_shadow_only",
        "symbols": ["BTCUSDT"],
        "candidate_count": len(v9_experts()),
        "decision_policy": "GLOBAL_LONG_SHORT_FLAT_ONE_POSITION",
        "source_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "v8_discovery_reused": True,
        "independent_confirmation": False,
        "automatic_live": False,
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        for key in (
            "training_exchanges",
            "target_exchange",
            "symbols",
            "decision_policy",
            "source_sha256",
            "config_sha256",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError("V10 external protocol changed after freezing")
        return cast(dict[str, Any], existing)
    ROOT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PROTOCOL_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = run_external_transfer(load_config(arguments.config), arguments.config)
    except Exception as error:
        _status("failed", f"{type(error).__name__}: {error}", 0)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
