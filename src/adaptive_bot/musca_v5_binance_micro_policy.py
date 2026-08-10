from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from adaptive_bot.musca_v5_strategy_class_audit import _feature_frame

SOURCE = Path("data/ml/musca_v5/aggtrades")
FUNDING_SOURCE = Path("data/ml/hybrid_v24/joined_minutes.parquet")
REPORT = Path("data/reports/musca_v5_binance_micro_policy.json")
STATUS = Path("data/reports/musca_v5_binance_micro_policy.status.json")
BUNDLE = Path("data/models/musca_v5_binance_micro/bundle.joblib")
FIT_MONTHS = ("2026-01", "2026-02")
CALIBRATION_MONTHS = ("2026-03",)
VALIDATION_MONTHS = ("2026-04",)
SEALED_MONTHS = ("2026-05", "2026-06", "2026-07")
COST_BPS = 9.0
MAXIMUM_HOLD_MINUTES = 30
FEATURES = (
    "return_30s_bps",
    "return_1m_bps",
    "return_5m_bps",
    "return_15m_bps",
    "flow_30s",
    "flow_1m",
    "flow_5m",
    "last_5s_return_bps",
    "last_5s_flow",
    "rolling_vwap_distance_bps",
    "daily_vwap_distance_bps",
    "rolling_vwap_slope_1m_bps",
    "distance_change_1m_bps",
    "volatility_1m_bps",
    "volatility_5m_bps",
    "volume_z",
    "trade_count_z",
    "family_fade",
    "family_follow",
    "family_reclaim",
    "side",
    "target_bps",
    "stop_bps",
)
PROTOCOL = {
    "name": "musca_v5_binance_event_probability_v1",
    "market": "Binance BTCUSDT USD-M perpetual",
    "source": "official aggregate trades aggregated to causal 5-second bars",
    "decision_clock": "completed one-minute observations",
    "events": ["VWAP_FADE", "VWAP_TREND_RESTART", "VWAP_RECLAIM"],
    "fit": list(FIT_MONTHS),
    "calibration": list(CALIBRATION_MONTHS),
    "validation": list(VALIDATION_MONTHS),
    "sealed_holdout": list(SEALED_MONTHS),
    "round_trip_cost_bps": COST_BPS,
    "minimum_target_bps": 18.0,
    "maximum_target_bps": 40.0,
    "maximum_hold_minutes": MAXIMUM_HOLD_MINUTES,
    "same_5s_bar": "stop_wins",
    "one_position": True,
    "models": ["multinomial_logistic_baseline", "xgboost_cuda_challenger"],
    "calibration_method": "one-vs-rest Platt on March then renormalized",
    "selection": "March Brier; XGBoost only if it beats the linear baseline",
    "validation_gates": {
        "minimum_trades_per_calendar_day": 3.0,
        "expectancy_bps": 0,
        "profit_factor": 1.15,
        "positive_active_days": 0.50,
        "stress_2x_expectancy_bps": 0,
    },
    "live_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PlattHead:
    slope: float
    intercept: float


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _write_json(
        STATUS,
        {
            "phase": phase,
            "detail": detail,
            "percent": percent,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _load_months(months: tuple[str, ...]) -> pd.DataFrame:
    paths = [SOURCE / f"BTCUSDT-aggTrades-5s-{month}.parquet" for month in months]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing official Binance files: {missing}")
    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    index = pd.date_range(data.index.min(), data.index.max(), freq="5s", tz="UTC")
    return data.reindex(index)


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    features = _feature_frame(data)
    close = data["close"]
    quote = data["quote_volume"]
    signed = data["signed_quote_volume"]
    log_count = np.log1p(data["trade_count"])
    features["return_30s_bps"] = close.pct_change(6, fill_method=None) * 10_000
    features["flow_30s"] = (
        signed.rolling(6, min_periods=6).sum() / quote.rolling(6, min_periods=6).sum()
    )
    rolling_vwap = quote.rolling(60).sum() / data["base_volume"].rolling(60).sum()
    features["rolling_vwap_slope_1m_bps"] = rolling_vwap.pct_change(12, fill_method=None) * 10_000
    features["distance_change_1m_bps"] = features["rolling_vwap_distance_bps"].diff(12)
    features["volatility_5m_bps"] = (
        close.pct_change(fill_method=None).rolling(60, min_periods=60).std() * 10_000
    )
    count_mean = log_count.shift(1).rolling(720, min_periods=240).mean()
    count_std = log_count.shift(1).rolling(720, min_periods=240).std()
    features["trade_count_z"] = (log_count - count_mean) / count_std.replace(0, np.nan)
    return features


def event_candidates(data: pd.DataFrame) -> pd.DataFrame:
    features = feature_frame(data).iloc[11::12].copy()
    distance = features["rolling_vwap_distance_bps"]
    previous_distance = features["rolling_vwap_distance_bps"].shift(1)
    volatility = features["volatility_1m_bps"].clip(lower=1.0)
    band = pd.Series(np.maximum(8.0, 1.5 * volatility), index=features.index)
    fade_threshold = band.clip(lower=18.0)

    fade_side = -np.sign(distance)
    fade = (
        distance.abs().ge(fade_threshold)
        & (fade_side * features["return_1m_bps"] > 0)
        & (fade_side * features["flow_1m"] > 0)
    )
    trend_side = np.sign(features["return_5m_bps"])
    follow = (
        (trend_side != 0)
        & (trend_side * features["return_5m_bps"] >= np.maximum(10.0, 2 * volatility))
        & (trend_side * features["return_1m_bps"] > 0)
        & (trend_side * features["flow_1m"] > 0)
        & (trend_side * distance > 0)
        & (trend_side * features["distance_change_1m_bps"] > 0)
    )
    reclaim_side = np.sign(distance)
    reclaim = (
        (reclaim_side != 0)
        & (reclaim_side != np.sign(previous_distance))
        & (reclaim_side * features["return_5m_bps"] > 0)
        & (reclaim_side * features["flow_1m"] > 0)
    )

    family = np.select(
        [reclaim, fade, follow],
        ["VWAP_RECLAIM", "VWAP_FADE", "VWAP_TREND_RESTART"],
        default="",
    )
    side = np.select([reclaim, fade, follow], [reclaim_side, fade_side, trend_side], default=0)
    features["family"] = family
    features["side"] = side.astype(np.int8)
    features = features.loc[features["family"].ne("") & features["side"].ne(0)].copy()

    features["family_fade"] = features["family"].eq("VWAP_FADE").astype(float)
    features["family_follow"] = features["family"].eq("VWAP_TREND_RESTART").astype(float)
    features["family_reclaim"] = features["family"].eq("VWAP_RECLAIM").astype(float)
    features["target_bps"] = np.clip(
        np.maximum(18.0, 3.0 * features["volatility_1m_bps"]), 18.0, 40.0
    )
    fade_room = features["rolling_vwap_distance_bps"].abs()
    features.loc[features["family"].eq("VWAP_FADE"), "target_bps"] = np.clip(fade_room, 18.0, 40.0)
    features["stop_bps"] = np.clip(
        np.maximum(12.0, 1.5 * features["volatility_1m_bps"]), 12.0, 24.0
    )
    features = features.dropna(subset=[*FEATURES, "available_at"])
    features["available_at"] = pd.to_datetime(features["available_at"], utc=True)
    features["decision_at"] = features["available_at"]
    features = features.loc[features["available_at"] >= features["decision_at"]]
    return features


def label_candidates(
    candidates: pd.DataFrame,
    data: pd.DataFrame,
    funding: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index = pd.DatetimeIndex(data.index)
    high = data["high"].to_numpy(float)
    low = data["low"].to_numpy(float)
    close = data["close"].to_numpy(float)
    open_price = data["open"].to_numpy(float)
    horizon = MAXIMUM_HOLD_MINUTES * 12
    positions = index.searchsorted(pd.DatetimeIndex(candidates["available_at"]), side="right")
    for (_, candidate), position in zip(candidates.iterrows(), positions, strict=True):
        entry_position = int(position)
        if entry_position + horizon >= len(data) or not np.isfinite(open_price[entry_position]):
            continue
        entry = open_price[entry_position]
        side = int(candidate["side"])
        target_bps = float(candidate["target_bps"])
        stop_bps = float(candidate["stop_bps"])
        target = entry * (1 + side * target_bps / 10_000)
        stop = entry * (1 - side * stop_bps / 10_000)
        gross = np.nan
        outcome_class = 1
        exit_position = entry_position + horizon
        for offset in range(horizon + 1):
            cursor = entry_position + offset
            if not np.isfinite(high[cursor]) or not np.isfinite(low[cursor]):
                break
            stop_hit = low[cursor] <= stop if side > 0 else high[cursor] >= stop
            target_hit = high[cursor] >= target if side > 0 else low[cursor] <= target
            if stop_hit:
                gross, outcome_class, exit_position = -stop_bps, 0, cursor
                break
            if target_hit:
                gross, outcome_class, exit_position = target_bps, 2, cursor
                break
        if not np.isfinite(gross):
            final = close[exit_position]
            if not np.isfinite(final):
                continue
            gross = side * (final / entry - 1) * 10_000
        row = {str(key): value for key, value in candidate.to_dict().items()}
        funding_return_bps = 0.0
        if funding is not None:
            entry_minute = index[entry_position].floor("min")
            exit_minute = index[exit_position].floor("min")
            observed = funding.loc[entry_minute:exit_minute]
            funding_return_bps = -side * float(observed.sum()) * 10_000
        row.update(
            {
                "entry_at": index[entry_position],
                "exit_at": index[exit_position] + pd.Timedelta(seconds=5),
                "entry_price": entry,
                "gross_return_bps": gross,
                "funding_return_bps": funding_return_bps,
                "net_return_bps": gross + funding_return_bps - COST_BPS,
                "outcome_class": outcome_class,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _platt_fit(probabilities: np.ndarray, labels: np.ndarray) -> list[PlattHead]:
    heads: list[PlattHead] = []
    for label in range(3):
        probability = np.clip(probabilities[:, label], 1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, random_state=42).fit(logit, (labels == label).astype(int))
        heads.append(PlattHead(float(model.coef_[0, 0]), float(model.intercept_[0])))
    return heads


def _platt_predict(probabilities: np.ndarray, heads: list[PlattHead]) -> np.ndarray:
    calibrated = np.empty_like(probabilities)
    for label, head in enumerate(heads):
        probability = np.clip(probabilities[:, label], 1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability))
        calibrated[:, label] = 1 / (1 + np.exp(-(head.slope * logit + head.intercept)))
    return np.asarray(calibrated / calibrated.sum(axis=1, keepdims=True))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    target = np.eye(3)[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _timeout_values(rows: pd.DataFrame) -> dict[str, float]:
    timeouts = rows.loc[rows["outcome_class"].eq(1)]
    global_value = float(timeouts["gross_return_bps"].median()) if len(timeouts) else 0.0
    return {
        str(family): float(group["gross_return_bps"].median())
        for family, group in timeouts.groupby("family")
    } | {"__default__": global_value}


def _score(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    timeout_values: dict[str, float],
) -> pd.DataFrame:
    scored = rows.copy()
    timeout = scored["family"].map(timeout_values).fillna(timeout_values["__default__"])
    scored["predicted_loss_probability"] = probabilities[:, 0]
    scored["predicted_timeout_probability"] = probabilities[:, 1]
    scored["predicted_win_probability"] = probabilities[:, 2]
    scored["predicted_net_ev_bps"] = (
        probabilities[:, 2] * scored["target_bps"].to_numpy(float)
        - probabilities[:, 0] * scored["stop_bps"].to_numpy(float)
        + probabilities[:, 1] * timeout.to_numpy(float)
        - COST_BPS
    )
    return scored


def _simulate(
    scored: pd.DataFrame,
    *,
    additional_cost_bps: float = 0.0,
    calendar_days: int | None = None,
    require_positive_ev: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    accepted = (
        scored.loc[scored["predicted_net_ev_bps"] > 0] if require_positive_ev else scored
    ).sort_values("entry_at")
    trades: list[dict[str, Any]] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for row in accepted.to_dict("records"):
        entry_at = pd.Timestamp(row["entry_at"])
        if entry_at < blocked_until:
            continue
        trades.append({str(key): value for key, value in row.items()})
        blocked_until = pd.Timestamp(row["exit_at"])
    result = pd.DataFrame(trades)
    if result.empty:
        return result, {
            "trades": 0.0,
            "trades_per_active_day": 0.0,
            "trades_per_calendar_day": 0.0,
            "expectancy_bps": None,
            "profit_factor": None,
            "win_rate": None,
            "positive_active_days": None,
            "max_drawdown": None,
        }
    values = result["net_return_bps"].to_numpy(float) - additional_cost_bps
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    daily = (
        result.assign(day=pd.to_datetime(result["entry_at"], utc=True).dt.date)
        .groupby("day")["net_return_bps"]
        .sum()
    )
    equity = np.cumprod(1 + values / 10_000)
    drawdown = 1 - equity / np.maximum.accumulate(np.r_[1.0, equity])[-len(equity) :]
    return result, {
        "trades": float(len(result)),
        "trades_per_active_day": float(len(result) / len(daily)),
        "trades_per_calendar_day": float(len(result) / (calendar_days or len(daily))),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()),
        "positive_active_days": float((daily > 0).mean()),
        "max_drawdown": float(drawdown.max()),
    }


def _economic_summary(rows: pd.DataFrame) -> dict[str, float | int | None]:
    values = rows["net_return_bps"].to_numpy(float)
    gross = rows["gross_return_bps"].to_numpy(float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return {
        "rows": len(rows),
        "gross_expectancy_bps": float(gross.mean()) if len(gross) else None,
        "net_expectancy_bps": float(values.mean()) if len(values) else None,
        "net_median_bps": float(np.median(values)) if len(values) else None,
        "net_win_rate": float((values > 0).mean()) if len(values) else None,
        "profit_factor": float(gains / losses) if losses else None,
    }


def _ranking_curve(scored: pd.DataFrame, calendar_days: int) -> dict[str, Any]:
    ranked = scored.assign(
        day=pd.to_datetime(scored["entry_at"], utc=True).dt.floor("D")
    ).sort_values(["day", "predicted_net_ev_bps"], ascending=[True, False])
    curve: dict[str, Any] = {}
    for daily_limit in (1, 3, 5, 10):
        selected = ranked.groupby("day", sort=False).head(daily_limit).drop(columns="day")
        _, metrics = _simulate(
            selected,
            calendar_days=calendar_days,
            require_positive_ev=False,
        )
        curve[str(daily_limit)] = metrics
    return curve


def _validation_gates(
    metrics: dict[str, float | None], stress_metrics: dict[str, float | None]
) -> dict[str, bool]:
    stress_expectancy = stress_metrics["expectancy_bps"]
    return {
        "at_least_3_trades_per_calendar_day": (
            float(metrics["trades_per_calendar_day"] or 0) >= 3.0
        ),
        "expectancy_positive": float(metrics["expectancy_bps"] or 0) > 0,
        "profit_factor_1_15": float(metrics["profit_factor"] or 0) >= 1.15,
        "majority_positive_active_days": float(metrics["positive_active_days"] or 0) > 0.5,
        "drawdown_8pct": float(metrics["max_drawdown"] or 1) <= 0.08,
        "stress_2x_nonnegative": (stress_expectancy is not None and float(stress_expectancy) >= 0),
    }


def train() -> dict[str, Any]:
    if any(
        (SOURCE / f"BTCUSDT-aggTrades-5s-{month}.parquet").stat().st_size <= 0
        for month in SEALED_MONTHS
    ):
        raise ValueError("sealed Binance holdout files are missing or empty")
    _status("load", "Binance 5s development months January-April", 5)
    development = _load_months(FIT_MONTHS + CALIBRATION_MONTHS + VALIDATION_MONTHS)
    funding_frame = pd.read_parquet(FUNDING_SOURCE, columns=["timestamp", "funding_event_rate"])
    funding_frame["timestamp"] = pd.to_datetime(funding_frame["timestamp"], utc=True)
    funding = funding_frame.set_index("timestamp")["funding_event_rate"].sort_index()
    funding = funding.loc[development.index.min().floor("min") : development.index.max()]
    if funding.empty or funding.isna().any():
        raise ValueError("official Binance funding coverage is incomplete")
    _status("features", f"{len(development):,} causal 5-second rows", 20)
    candidates = event_candidates(development)
    _status("labels", f"{len(candidates):,} preregistered VWAP events", 35)
    labeled = label_candidates(candidates, development, funding)
    month = pd.to_datetime(labeled["entry_at"], utc=True).dt.strftime("%Y-%m")
    fit = labeled.loc[month.isin(FIT_MONTHS)].copy()
    calibration = labeled.loc[month.isin(CALIBRATION_MONTHS)].copy()
    validation = labeled.loc[month.isin(VALIDATION_MONTHS)].copy()
    if min(len(fit), len(calibration), len(validation)) < 100:
        raise ValueError("event generator did not produce enough independent development rows")

    x_fit, y_fit = fit.loc[:, FEATURES].to_numpy(float), fit["outcome_class"].to_numpy(int)
    x_cal = calibration.loc[:, FEATURES].to_numpy(float)
    y_cal = calibration["outcome_class"].to_numpy(int)
    x_validation = validation.loc[:, FEATURES].to_numpy(float)
    y_validation = validation["outcome_class"].to_numpy(int)
    for split_name, labels in (
        ("fit", y_fit),
        ("calibration", y_cal),
        ("validation", y_validation),
    ):
        if set(labels.tolist()) != {0, 1, 2}:
            raise ValueError(f"{split_name} does not contain stop, timeout and target outcomes")
    _status("training", f"linear baseline on {len(fit):,} events", 55)
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1_000, random_state=42),
    ).fit(x_fit, y_fit)
    _status("training", "XGBoost CUDA challenger", 70)
    challenger = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda",
        n_estimators=350,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=1,
    ).fit(x_fit, y_fit)
    models = {"linear": linear, "xgboost_cuda": challenger}
    timeout_values = _timeout_values(fit)
    audits: dict[str, Any] = {}
    fitted_heads: dict[str, list[PlattHead]] = {}
    validation_scored: dict[str, pd.DataFrame] = {}
    _status("audit", "Platt calibration and April validation", 85)
    for name, model in models.items():
        raw_calibration = model.predict_proba(x_cal)
        heads = _platt_fit(raw_calibration, y_cal)
        fitted_heads[name] = heads
        calibrated_calibration = _platt_predict(raw_calibration, heads)
        calibration_trades, calibration_metrics = _simulate(
            _score(calibration, calibrated_calibration, timeout_values), calendar_days=31
        )
        calibrated_validation = _platt_predict(model.predict_proba(x_validation), heads)
        scored = _score(validation, calibrated_validation, timeout_values)
        validation_scored[name] = scored
        validation_trades, validation_metrics = _simulate(scored, calendar_days=30)
        _, validation_stress_metrics = _simulate(
            scored, additional_cost_bps=COST_BPS, calendar_days=30
        )
        validation_gates = _validation_gates(validation_metrics, validation_stress_metrics)
        audits[name] = {
            "calibration_brier": _brier(calibrated_calibration, y_cal),
            "validation_brier": _brier(calibrated_validation, y_validation),
            "calibration": calibration_metrics,
            "validation": validation_metrics,
            "validation_stress_costs_2x": validation_stress_metrics,
            "validation_gates": validation_gates,
            "candidate_acceptance": {
                "calibration": len(calibration_trades),
                "validation": len(validation_trades),
            },
            "predicted_net_ev_quantiles_bps": {
                str(quantile): float(scored["predicted_net_ev_bps"].quantile(quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "validation_top_ranked_per_day": _ranking_curve(scored, 30),
        }

    linear_brier = float(audits["linear"]["calibration_brier"])
    xgb_brier = float(audits["xgboost_cuda"]["calibration_brier"])
    champion = "xgboost_cuda" if xgb_brier < linear_brier else "linear"
    champion_gates = audits[champion]["validation_gates"]
    passed = all(champion_gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "data": {
            "fit_rows": len(fit),
            "calibration_rows": len(calibration),
            "validation_rows": len(validation),
            "sealed_rows_read": 0,
            "candidate_days": int(pd.to_datetime(labeled["entry_at"], utc=True).dt.date.nunique()),
            "candidate_rate_per_day": float(
                len(labeled) / pd.to_datetime(labeled["entry_at"], utc=True).dt.date.nunique()
            ),
            "candidate_families": {
                str(key): int(value) for key, value in labeled["family"].value_counts().items()
            },
            "outcome_classes": {
                split_name: {
                    str(key): int(value)
                    for key, value in split["outcome_class"].value_counts().sort_index().items()
                }
                for split_name, split in (
                    ("fit", fit),
                    ("calibration", calibration),
                    ("validation", validation),
                )
            },
            "candidate_economics": {
                split_name: {
                    "all": _economic_summary(split),
                    "by_family": {
                        str(family): _economic_summary(group)
                        for family, group in split.groupby("family")
                    },
                }
                for split_name, split in (
                    ("fit", fit),
                    ("calibration", calibration),
                    ("validation", validation),
                )
            },
        },
        "causal_checks": {
            "feature_available_at_not_before_decision": bool(
                (pd.to_datetime(labeled["available_at"], utc=True) >= labeled["decision_at"]).all()
            ),
            "entry_strictly_after_decision": bool(
                (
                    pd.to_datetime(labeled["entry_at"], utc=True)
                    > pd.to_datetime(labeled["available_at"], utc=True)
                ).all()
            ),
            "sealed_holdout_read": False,
            "same_bar_stop_wins": True,
        },
        "models": audits,
        "champion": champion,
        "validation_gates": champion_gates,
        "verdict": (
            "BINANCE_MICRO_ALPHA_READY_FOR_MANUAL_HOLDOUT"
            if passed
            else "NO_FREQUENT_BINANCE_ALPHA"
        ),
        "paper_policy_changed": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _write_json(REPORT, report)
    if passed:
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        temporary = BUNDLE.with_suffix(".tmp")
        joblib.dump(
            {
                "protocol": PROTOCOL,
                "protocol_hash": PROTOCOL_HASH,
                "features": FEATURES,
                "champion": champion,
                "model": models[champion],
                "platt_heads": [asdict(head) for head in fitted_heads[champion]],
                "timeout_values": timeout_values,
                "validation": validation_scored[champion],
                "research_only": True,
            },
            temporary,
        )
        temporary.replace(BUNDLE)
    _status("complete", report["verdict"], 100)
    return report


if __name__ == "__main__":
    train()
