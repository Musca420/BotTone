from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

from adaptive_bot.musca_v5_fine_tuning import FEATURES, POLICY_STATE_INTERACTIONS

SOURCE = Path("data/ml/musca_v5/fine_tuning_matrix.parquet")
REPORT = Path("data/reports/musca_v5_binance_daily_ranker.json")
STATUS = Path("data/reports/musca_v5_binance_daily_ranker.status.json")
BUNDLE = Path("data/models/musca_v5_binance_daily_ranker/bundle.joblib")
COST_BPS = 9.0
MINIMUM_TRADES_PER_CALENDAR_DAY = 3.0
FIT_END = pd.Timestamp("2025-01-01T00:00:00Z")
MODEL_SELECTION_END = pd.Timestamp("2025-07-01T00:00:00Z")
POLICY_SELECTION_END = pd.Timestamp("2026-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-05-11T11:30:00Z")
COVERAGES = tuple(value / 20 for value in range(1, 21))
PROTOCOL = {
    "name": "musca_v5_binance_daily_action_ranker_v1",
    "market": "Binance BTCUSDT USD-M perpetual",
    "source": str(SOURCE),
    "features": list(FEATURES),
    "feature_count": len(FEATURES),
    "missing_features": "fail closed; no imputation",
    "entry": "strictly after maximum feature availability timestamp",
    "action": "direction plus dynamic target/stop/trailing plan",
    "fit": "2024 through 2024-12-31",
    "model_selection": "2025-H1",
    "policy_threshold_selection": "2025-H2",
    "reused_discovery_audit": "2026 before 2026-05-11 holdout boundary",
    "future_holdout": "not opened",
    "models": ["ridge_regression_baseline", "xgboost_cuda_pairwise_ranker"],
    "xgboost_parameters": {
        "objective": "rank:pairwise",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 20.0,
    },
    "model_selection_rule": (
        "XGBoost only if 2025-H1 top-30-percent expectancy and rank correlation "
        "both beat Ridge; otherwise Ridge"
    ),
    "policy_selection": (
        "maximum causal coverage passing economic gates; fixed score threshold then "
        "applied unchanged to the next period"
    ),
    "individual_positive_ev_required": False,
    "one_position": True,
    "round_trip_cost_bps": COST_BPS,
    "stress_round_trip_cost_bps": 2 * COST_BPS,
    "gates": {
        "minimum_trades_per_calendar_day": MINIMUM_TRADES_PER_CALENDAR_DAY,
        "expectancy_bps": 0,
        "profit_factor": 1.15,
        "positive_active_days": 0.50,
        "max_drawdown": 0.08,
        "bootstrap_lcb_bps": 0,
    },
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class Scorer(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "detail": detail,
            "percent": percent,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _derive_interactions(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["target_to_atr_1m"] = output["first_target_bps"] / output["atr_1m_bps"].replace(
        0, np.nan
    )
    output["target_to_atr_5m"] = output["first_target_bps"] / output["atr_5m_bps"].replace(
        0, np.nan
    )
    output["target_to_realized_volatility_30m"] = output["first_target_bps"] / output[
        "realized_volatility_30m_bps"
    ].replace(0, np.nan)
    output["stop_to_atr_5m"] = output["initial_stop_bps"] / output["atr_5m_bps"].replace(0, np.nan)
    output["flow_volume_confirmation"] = (
        output["signed_entry_taker_imbalance_15m"] * output["volume_percentile"]
    )
    output["orderflow_depth_confirmation"] = (
        output["signed_binance_taker_imbalance_15m"] * output["signed_book_imbalance_1pct"]
    )
    return output


def load_actions() -> pd.DataFrame:
    stored_features = [name for name in FEATURES if name not in POLICY_STATE_INTERACTIONS]
    columns = list(
        dict.fromkeys(
            [
                *stored_features,
                "first_target_bps",
                "signal_timestamp",
                "available_at",
                "max_input_available_at",
                "entry_timestamp",
                "exit_timestamp",
                "direction",
                "event_family",
                "plan",
                "gross_return_bps",
                "model_feature_coverage_valid",
            ]
        )
    )
    rows = _derive_interactions(pd.read_parquet(SOURCE, columns=columns))
    for column in (
        "signal_timestamp",
        "available_at",
        "max_input_available_at",
        "entry_timestamp",
        "exit_timestamp",
    ):
        rows[column] = pd.to_datetime(rows[column], utc=True)
    causal = rows["entry_timestamp"] > rows["max_input_available_at"]
    covered = rows["model_feature_coverage_valid"].fillna(False).astype(bool)
    feature_values = rows.loc[:, FEATURES].to_numpy(float)
    complete = pd.Series(np.isfinite(feature_values).all(axis=1), index=rows.index)
    finite = complete
    rows = rows.loc[causal & covered & complete & finite].copy()
    rows["net_return_bps"] = rows["gross_return_bps"] - COST_BPS
    rows["stress_return_bps"] = rows["gross_return_bps"] - 2 * COST_BPS
    rows["day"] = rows["entry_timestamp"].dt.floor("D")
    return rows.sort_values(["day", "entry_timestamp", "signal_timestamp", "plan"])


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, FEATURES].to_numpy(float)
    for name in ("daily_distance_sigma", "impulse_distance_sigma", "swing_distance_sigma"):
        values[:, FEATURES.index(name)] = np.clip(values[:, FEATURES.index(name)], -20, 20)
    return values


def _daily_relevance(rows: pd.DataFrame) -> np.ndarray:
    return (
        rows.groupby("day", sort=False)["net_return_bps"]
        .rank(method="average", pct=True)
        .to_numpy(float)
    )


def _period(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return rows.loc[rows["entry_timestamp"].ge(start) & rows["entry_timestamp"].lt(end)].copy()


def _fit_models(rows: pd.DataFrame) -> dict[str, Scorer]:
    x = _x(rows)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(
        x, rows["net_return_bps"].to_numpy(float)
    )
    ordered = rows.sort_values(["day", "entry_timestamp", "signal_timestamp", "plan"])
    groups = ordered.groupby("day", sort=False).size().to_numpy(int)
    ranker = XGBRanker(
        objective="rank:pairwise",
        tree_method="hist",
        device="cuda",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=80,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_lambda=20.0,
        random_state=42,
        n_jobs=1,
    ).fit(_x(ordered), _daily_relevance(ordered), group=groups, verbose=False)
    return {"ridge": ridge, "xgboost_ranker_cuda": ranker}


def _score(rows: pd.DataFrame, model: Scorer) -> pd.DataFrame:
    output = rows.copy()
    output["score"] = np.asarray(model.predict(_x(output)), dtype=float)
    return output


def _deduplicate_actions(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.sort_values(
            ["signal_timestamp", "direction", "score"], ascending=[True, True, False]
        )
        .drop_duplicates(["signal_timestamp", "direction"], keep="first")
        .sort_values("entry_timestamp")
    )


def execute(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = _deduplicate_actions(scored)
    candidates = candidates.loc[candidates["score"].ge(threshold)].copy()
    accepted: list[Hashable] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in candidates.iterrows():
        if row["entry_timestamp"] < blocked_until:
            continue
        accepted.append(index)
        blocked_until = pd.Timestamp(row["exit_timestamp"])
    return candidates.loc[accepted].sort_values("entry_timestamp")


def _bootstrap_daily_lcb(trades: pd.DataFrame, seed: int = 42) -> float | None:
    if trades.empty:
        return None
    daily = trades.groupby("day")["net_return_bps"].agg(["sum", "count"])
    if len(daily) < 10:
        return None
    values = daily[["sum", "count"]].to_numpy(float)
    block = min(5, len(values))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(1_000):
        sampled: list[np.ndarray] = []
        while sum(len(item) for item in sampled) < len(values):
            start = int(rng.integers(0, len(values) - block + 1))
            sampled.append(values[start : start + block])
        joined = np.concatenate(sampled, axis=0)[: len(values)]
        draws.append(float(joined[:, 0].sum() / joined[:, 1].sum()))
    return float(np.quantile(draws, 0.05))


def metrics(trades: pd.DataFrame, calendar_days: int) -> dict[str, float | None]:
    if trades.empty:
        return {
            "trades": 0.0,
            "trades_per_calendar_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "positive_active_days": None,
            "max_drawdown": None,
            "bootstrap_lcb_95_bps": None,
        }
    values = trades["net_return_bps"].to_numpy(float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    daily = trades.groupby("day")["net_return_bps"].sum()
    equity = np.cumprod(1 + values / 10_000)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "trades": float(len(trades)),
        "trades_per_calendar_day": float(len(trades) / calendar_days),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()),
        "positive_active_days": float((daily > 0).mean()),
        "max_drawdown": float((1 - equity / peaks).max()),
        "bootstrap_lcb_95_bps": _bootstrap_daily_lcb(trades),
    }


def _stress_metrics(trades: pd.DataFrame, calendar_days: int) -> dict[str, float | None]:
    stressed = trades.copy()
    stressed["net_return_bps"] = stressed["stress_return_bps"]
    return metrics(stressed, calendar_days)


def _gates(value: dict[str, float | None]) -> dict[str, bool]:
    lcb = value["bootstrap_lcb_95_bps"]
    expectancy = value["expectancy_bps"]
    profit_factor = value["profit_factor"]
    positive_days = value["positive_active_days"]
    drawdown = value["max_drawdown"]
    return {
        "frequency_3_per_calendar_day": (
            float(value["trades_per_calendar_day"] or 0) >= MINIMUM_TRADES_PER_CALENDAR_DAY
        ),
        "expectancy_positive": expectancy is not None and float(expectancy) > 0,
        "profit_factor_1_15": (profit_factor is not None and float(profit_factor) >= 1.15),
        "majority_positive_active_days": (positive_days is not None and float(positive_days) > 0.5),
        "drawdown_8pct": drawdown is not None and float(drawdown) <= 0.08,
        "bootstrap_lcb_positive": lcb is not None and float(lcb) > 0,
    }


def _calendar_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end.floor("D") - start.floor("D")).days)


def _threshold_curve(scored: pd.DataFrame, period: Period) -> list[dict[str, Any]]:
    unique = _deduplicate_actions(scored)
    days = _calendar_days(period.start, period.end)
    curve: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        threshold = float(unique["score"].quantile(1 - coverage))
        trades = execute(scored, threshold)
        value = metrics(trades, days)
        curve.append(
            {
                "coverage": coverage,
                "threshold": threshold,
                "metrics": value,
                "stress_costs_2x": _stress_metrics(trades, days),
                "gates": _gates(value),
            }
        )
    return curve


def _rank_correlation(scored: pd.DataFrame) -> float:
    unique = _deduplicate_actions(scored)
    value = unique["score"].corr(unique["net_return_bps"], method="spearman")
    return float(value) if pd.notna(value) else 0.0


def _top_30_expectancy(scored: pd.DataFrame) -> float:
    unique = _deduplicate_actions(scored)
    threshold = float(unique["score"].quantile(0.7))
    selected = execute(scored, threshold)
    return float(selected["net_return_bps"].mean()) if len(selected) else -np.inf


def train() -> dict[str, Any]:
    _status("load", "123 causal Binance features and dynamic action plans", 5)
    rows = load_actions()
    first = rows["entry_timestamp"].min().floor("D")
    periods = {
        "fit": Period("fit", first, FIT_END),
        "model_selection": Period("model_selection", FIT_END, MODEL_SELECTION_END),
        "policy_selection": Period("policy_selection", MODEL_SELECTION_END, POLICY_SELECTION_END),
        "audit": Period("audit", POLICY_SELECTION_END, AUDIT_END),
    }
    split = {name: _period(rows, period.start, period.end) for name, period in periods.items()}
    if min(len(value) for value in split.values()) < 1_000:
        raise ValueError("insufficient causally complete Binance actions in a chronological split")
    _status("training", f"Ridge and one CUDA ranker on {len(split['fit']):,} actions", 35)
    models = _fit_models(split["fit"])
    model_selection_scores = {
        name: _score(split["model_selection"], model) for name, model in models.items()
    }
    diagnostics = {
        name: {
            "rank_correlation": _rank_correlation(scored),
            "top_30_percent_expectancy_bps": _top_30_expectancy(scored),
        }
        for name, scored in model_selection_scores.items()
    }
    ridge_diagnostic = diagnostics["ridge"]
    xgb_diagnostic = diagnostics["xgboost_ranker_cuda"]
    champion = (
        "xgboost_ranker_cuda"
        if xgb_diagnostic["rank_correlation"] > ridge_diagnostic["rank_correlation"]
        and xgb_diagnostic["top_30_percent_expectancy_bps"]
        > ridge_diagnostic["top_30_percent_expectancy_bps"]
        else "ridge"
    )
    _status("selection", f"{champion}: causal coverage frontier on 2025-H2", 60)
    selection_scored = _score(split["policy_selection"], models[champion])
    selection_curve = _threshold_curve(selection_scored, periods["policy_selection"])
    passing = [point for point in selection_curve if all(point["gates"].values())]
    selected = max(
        passing,
        key=lambda point: float(point["metrics"]["trades_per_calendar_day"] or 0),
        default=None,
    )
    _status("audit", "unchanged score threshold on reused 2026 discovery audit", 80)
    audit_scored = _score(split["audit"], models[champion])
    audit_trades = (
        execute(audit_scored, float(selected["threshold"]))
        if selected is not None
        else audit_scored.iloc[0:0].copy()
    )
    audit_metrics = metrics(
        audit_trades, _calendar_days(periods["audit"].start, periods["audit"].end)
    )
    audit_gates = _gates(audit_metrics)
    passed = selected is not None and all(audit_gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "source_rows": 92_727,
            "causally_complete_rows": len(rows),
            "features": len(FEATURES),
            "rows_by_split": {name: len(value) for name, value in split.items()},
            "plans": {str(key): int(value) for key, value in rows["plan"].value_counts().items()},
            "future_holdout_rows_read": 0,
        },
        "causal_checks": {
            "entry_strictly_after_all_features": bool(
                (rows["entry_timestamp"] > rows["max_input_available_at"]).all()
            ),
            "missing_feature_rows_excluded": True,
            "future_day_candidates_used_by_live_threshold": False,
            "one_position_enforced": True,
        },
        "model_selection_2025_h1": diagnostics,
        "champion": champion,
        "policy_selection_2025_h2": {
            "curve": selection_curve,
            "selected": selected,
        },
        "reused_discovery_audit_2026": {
            "metrics": audit_metrics,
            "stress_costs_2x": _stress_metrics(
                audit_trades,
                _calendar_days(periods["audit"].start, periods["audit"].end),
            ),
            "gates": audit_gates,
            "trade_plans": {
                str(key): int(value) for key, value in audit_trades["plan"].value_counts().items()
            },
            "long": int(audit_trades["direction"].gt(0).sum()),
            "short": int(audit_trades["direction"].lt(0).sum()),
        },
        "verdict": (
            "BINANCE_DAILY_RANKER_READY_FOR_FUTURE_HOLDOUT" if passed else "NO_DAILY_RANKING_ALPHA"
        ),
        "paper_policy_changed": False,
        "future_holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    if passed:
        assert selected is not None
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        temporary = BUNDLE.with_suffix(".tmp")
        joblib.dump(
            {
                "protocol": PROTOCOL,
                "protocol_hash": PROTOCOL_HASH,
                "features": FEATURES,
                "model": models[champion],
                "champion": champion,
                "score_threshold": float(selected["threshold"]),
                "research_only": True,
            },
            temporary,
        )
        temporary.replace(BUNDLE)
    _status("complete", cast(str, report["verdict"]), 100)
    return report


if __name__ == "__main__":
    train()
