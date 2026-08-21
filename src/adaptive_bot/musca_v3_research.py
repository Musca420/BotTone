from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot.musca_v2_research import DATASET, build_features
from adaptive_bot.musca_v3 import (
    ANCHOR_MAX_DISTANCE_ATR,
    ANCHOR_MIN_SLOPE_1H,
    ANCHOR_MIN_SLOPE_15M,
    ARMED_MAX_BARS,
)

ROOT = Path("data/ml/musca_v3")
EVENTS = ROOT / "events.parquet"
OOS = ROOT / "oos.parquet"
REPORT = Path("data/reports/musca_v3_research.json")
STATUS = Path("data/reports/musca_v3_research.status.json")
BUNDLE = Path("data/models/musca_v3/bundle.joblib")
FEATURES = (
    "direction",
    "trend_strength",
    "distance_daily_vwap_atr",
    "return_15m",
    "return_1h",
    "anchor_distance_atr",
    "anchor_slope_15m",
    "anchor_slope_1h",
    "relative_volume",
    "atr_percentile",
    "hour_sin",
    "hour_cos",
    "anchored_setup",
    "anchor_source_impulse",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    data = build_features(frame)
    data["return_15m"] = data["perp_close"].pct_change(3)
    data["return_1h"] = data["perp_close"].pct_change(12)
    data["local_low"] = (
        data["perp_low"].shift(1).rolling(12, min_periods=12).min() - 0.1 * data["atr"]
    )
    data["local_high"] = (
        data["perp_high"].shift(1).rolling(12, min_periods=12).max() + 0.1 * data["atr"]
    )
    return data


def _base_row(
    data: pd.DataFrame, index: int, *, anchored: bool, impulse_anchor: bool = False
) -> dict[str, Any]:
    row = data.iloc[index]
    return {
        "signal_index": index,
        "signal_timestamp": pd.Timestamp(row["timestamp"]),
        "direction": int(row["direction"]),
        "trend_strength": float(row["trend_strength"]),
        "distance_daily_vwap_atr": float(row["distance_daily_vwap_atr"]),
        "return_15m": float(row["return_15m"]),
        "return_1h": float(row["return_1h"]),
        "relative_volume": float(row["relative_volume"]),
        "atr_percentile": float(row["atr_percentile"]),
        "hour_sin": float(row["hour_sin"]),
        "hour_cos": float(row["hour_cos"]),
        "anchored_setup": int(anchored),
        "anchor_source_impulse": int(impulse_anchor),
    }


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    data = _features(frame)
    close = data["perp_close"].to_numpy(float)
    high = data["perp_high"].to_numpy(float)
    low = data["perp_low"].to_numpy(float)
    volume = data["perp_volume"].to_numpy(float)
    quote = data["perp_quote_volume"].to_numpy(float)
    daily = data["daily_vwap"].to_numpy(float)
    atr = data["atr"].to_numpy(float)
    r15 = data["return_15m"].to_numpy(float)
    r1h = data["return_1h"].to_numpy(float)
    direction = data["direction"].to_numpy(float)
    reclaim = ((direction > 0) & (close > daily) & (np.roll(close, 1) <= np.roll(daily, 1))) | (
        (direction < 0) & (close < daily) & (np.roll(close, 1) >= np.roll(daily, 1))
    )
    reclaim[0] = False
    prior_high = data["perp_high"].shift(1).rolling(12, min_periods=12).max().to_numpy(float)
    prior_low = data["perp_low"].shift(1).rolling(12, min_periods=12).min().to_numpy(float)
    relative_volume = data["relative_volume"].to_numpy(float)
    impulse = ((direction > 0) & (close > prior_high) & (r15 > 0) & (relative_volume > 1.25)) | (
        (direction < 0) & (close < prior_low) & (r15 < 0) & (relative_volume > 1.25)
    )
    events: list[dict[str, Any]] = []
    for raw_index in np.flatnonzero(reclaim | impulse):
        index = int(raw_index)
        side = int(direction[index])
        price_confirmed = (
            close[index] > high[index - 1] if side > 0 else close[index] < low[index - 1]
        )
        if reclaim[index] and r15[index] * side > 0 and r1h[index] * side > 0 and price_confirmed:
            events.append(
                _base_row(data, index, anchored=False)
                | {
                    "anchor_distance_atr": np.nan,
                    "anchor_slope_15m": np.nan,
                    "anchor_slope_1h": np.nan,
                }
            )
        weighted = 0.0
        total_volume = 0.0
        anchors: list[float] = []
        armed = False
        armed_bars = 0
        for current in range(index, min(index + 97, len(data) - 1)):
            weighted += quote[current]
            total_volume += volume[current]
            anchor = weighted / total_volume if total_volume > 0 else np.nan
            anchors.append(anchor)
            if not np.isfinite(anchor) or not np.isfinite(atr[current]):
                continue
            touched = (
                low[current] <= anchor + 0.25 * atr[current]
                if side > 0
                else high[current] >= anchor - 0.25 * atr[current]
            )
            if touched and not armed:
                armed, armed_bars = True, 0
            elif armed:
                armed_bars += 1
            if armed_bars > ARMED_MAX_BARS:
                armed = False
                continue
            elapsed = current - index
            if elapsed < 12 or not armed:
                continue
            slope15 = (anchor - anchors[-4]) / atr[current]
            slope1h = (anchor - anchors[-13]) / atr[current]
            near = abs(close[current] - anchor) <= ANCHOR_MAX_DISTANCE_ATR * atr[current]
            confirmed = (
                close[current] > high[current - 1]
                if side > 0
                else close[current] < low[current - 1]
            )
            if (
                near
                and (close[current] - anchor) * side > 0
                and (close[current] - daily[current]) * side > 0
                and r15[current] * side > 0
                and r1h[current] * side > 0
                and slope15 * side >= ANCHOR_MIN_SLOPE_15M
                and slope1h * side >= ANCHOR_MIN_SLOPE_1H
                and confirmed
            ):
                events.append(
                    _base_row(data, current, anchored=True, impulse_anchor=bool(impulse[index]))
                    | {
                        "anchor_distance_atr": (close[current] - anchor) / atr[current],
                        "anchor_slope_15m": slope15,
                        "anchor_slope_1h": slope1h,
                    }
                )
                break
    candidates = pd.DataFrame(events).drop_duplicates(["signal_timestamp", "direction"])
    return _label(data, candidates).sort_values("signal_timestamp").reset_index(drop=True)


def _label(data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for raw_event in events.itertuples(index=False):
        event = cast(Any, raw_event)
        signal = int(event.signal_index)
        entry_index = signal + 1
        side = int(event.direction)
        entry = float(data.iloc[entry_index]["perp_open"])
        stop = float(data.iloc[signal]["local_low" if side > 0 else "local_high"])
        risk = side * (entry - stop)
        if not np.isfinite(risk) or risk <= 0:
            continue
        extreme = entry
        exit_index = min(entry_index + 96, len(data) - 1)
        exit_price = float(data.iloc[exit_index]["perp_close"])
        reason = "TIMEOUT_8H"
        mfe = mae = 0.0
        for current in range(entry_index, exit_index + 1):
            row = data.iloc[current]
            favorable = side * (float(row["perp_high" if side > 0 else "perp_low"]) - entry)
            adverse = side * (float(row["perp_low" if side > 0 else "perp_high"]) - entry)
            mfe, mae = max(mfe, favorable), min(mae, adverse)
            stopped = (
                float(row["perp_low"]) <= stop if side > 0 else float(row["perp_high"]) >= stop
            )
            if stopped:
                market = float(row["perp_open"])
                exit_price = min(market, stop) if side > 0 else max(market, stop)
                exit_index, reason = current, "STOP"
                break
            extreme = (
                max(extreme, float(row["perp_high"]))
                if side > 0
                else min(extreme, float(row["perp_low"]))
            )
            if side * (extreme - entry) >= float(row["atr"]):
                trail = extreme - side * 2 * float(row["atr"])
                cost_floor = entry + side * entry * 8 / 10_000
                stop = max(stop, trail, cost_floor) if side > 0 else min(stop, trail, cost_floor)
            if current > entry_index:
                previous = data.iloc[current - 1]
                reversal = (
                    float(row["return_15m"]) * side < 0 and float(row["return_1h"]) * side < 0
                )
                vwap_failed = (
                    float(row["perp_close"] - row["daily_vwap"]) * side < 0
                    and float(previous["perp_close"] - previous["daily_vwap"]) * side < 0
                )
                if (reversal or vwap_failed) and current + 1 < len(data):
                    exit_index, exit_price = current + 1, float(data.iloc[current + 1]["perp_open"])
                    reason = "LOCAL_REVERSAL" if reversal else "VWAP_FAILURE"
                    break
        gross = side * (exit_price - entry) / entry * 10_000
        output.append(
            event._asdict()
            | {
                "entry_timestamp": pd.Timestamp(data.iloc[entry_index]["timestamp"]),
                "exit_timestamp": pd.Timestamp(data.iloc[exit_index]["timestamp"]),
                "gross_return_bps": gross,
                "net_return_bps": gross - 8,
                "stress_return_bps": gross - 16,
                "mfe_bps": mfe / entry * 10_000,
                "mae_bps": mae / entry * 10_000,
                "risk_bps": risk / entry * 10_000,
                "net_return_r": (gross - 8) / (risk / entry * 10_000),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(output)


def _non_overlapping(rows: pd.DataFrame) -> pd.DataFrame:
    accepted: list[Any] = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for index, row in rows.sort_values("entry_timestamp").iterrows():
        if pd.Timestamp(row["entry_timestamp"]) >= busy_until:
            accepted.append(index)
            busy_until = pd.Timestamp(row["exit_timestamp"])
    return rows.loc[accepted].sort_values("entry_timestamp")


def _metrics(rows: pd.DataFrame, target: str = "net_return_bps") -> dict[str, float]:
    values = rows[target].to_numpy(float)
    if not len(values):
        return {"trades": 0.0, "expectancy_bps": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    curve = np.cumsum(rows["net_return_r"].to_numpy(float) * 0.01)
    drawdown = np.max(np.maximum.accumulate(np.r_[0.0, curve]) - np.r_[0.0, curve])
    return {
        "trades": float(len(rows)),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else float("inf"),
        "max_drawdown": float(drawdown),
    }


def _models() -> dict[str, tuple[Any, Any]]:
    linear = (
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, random_state=20260805),
        ),
    )
    common = dict(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        tree_method="hist",
        device="cuda",
        random_state=20260805,
        n_jobs=4,
    )
    return {
        "ridge": linear,
        "xgboost_gpu": (
            XGBRegressor(objective="reg:squarederror", **common),
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common),
        ),
    }


def train() -> dict[str, Any]:
    _atomic_json(STATUS, {"phase": "events", "percent": 5, "detail": "BTC 5m event scan"})
    events = build_events(pd.read_parquet(DATASET))
    ROOT.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS, index=False)
    _atomic_json(
        STATUS,
        {"phase": "base_audit", "percent": 45, "detail": f"{len(events):,} causal events"},
    )
    tradeable = events.loc[events["anchored_setup"].eq(1)].copy()
    base = _non_overlapping(tradeable)
    base_metrics = _metrics(base)
    base_stable = (
        base_metrics["expectancy_bps"] > 0
        and base_metrics["profit_factor"] >= 1.10
        and base.groupby(pd.to_datetime(base["entry_timestamp"], utc=True).dt.year)[
            "net_return_bps"
        ]
        .mean()
        .gt(0)
        .mean()
        > 0.5
    )
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    times = pd.to_datetime(tradeable["entry_timestamp"], utc=True)
    for year in range(2022, 2027):
        prior = tradeable.loc[times.lt(pd.Timestamp(f"{year}-01-01", tz="UTC"))]
        test = tradeable.loc[times.dt.year.eq(year)]
        if len(prior) < 150 or len(test) < 30:
            continue
        split = int(len(prior) * 0.8)
        fit, calibration = prior.iloc[:split], prior.iloc[split:]
        results: list[tuple[float, str, Any, Any]] = []
        for name, (regressor, classifier) in _models().items():
            regressor.fit(fit[list(FEATURES)], fit["net_return_bps"])
            classifier.fit(fit[list(FEATURES)], fit["net_return_bps"].gt(0))
            ev = regressor.predict(calibration[list(FEATURES)])
            probability = classifier.predict_proba(calibration[list(FEATURES)])[:, 1]
            selected = _non_overlapping(calibration.loc[(ev > 0) & (probability > 0.5)])
            score = _metrics(selected)["expectancy_bps"] if len(selected) >= 30 else -np.inf
            results.append((score, name, regressor, classifier))
        ridge = next(result for result in results if result[1] == "ridge")
        challenger = max(results, key=lambda item: item[0])
        champion = (
            challenger if challenger[1] != "xgboost_gpu" or challenger[0] > ridge[0] else ridge
        )
        _, name, regressor, classifier = champion
        regressor.fit(prior[list(FEATURES)], prior["net_return_bps"])
        classifier.fit(prior[list(FEATURES)], prior["net_return_bps"].gt(0))
        ev = regressor.predict(test[list(FEATURES)])
        probability = classifier.predict_proba(test[list(FEATURES)])[:, 1]
        selected = _non_overlapping(test.loc[(ev > 0) & (probability > 0.5)].copy())
        selected["outer_year"] = year
        selected["champion"] = name
        predictions.append(selected)
        folds.append({"year": year, "champion": name, "metrics": _metrics(selected)})
        _atomic_json(
            STATUS,
            {
                "phase": "walk_forward_gpu",
                "percent": 45 + 9 * (year - 2021),
                "detail": f"OOS {year}: {name}",
            },
        )
    oos = pd.concat(predictions, ignore_index=True) if predictions else tradeable.iloc[:0].copy()
    oos.to_parquet(OOS, index=False)
    oos_metrics = _metrics(oos)
    gates = {
        "base_edge": bool(base_stable),
        "oos_trades_300": len(oos) >= 300,
        "oos_expectancy_positive": oos_metrics["expectancy_bps"] > 0,
        "oos_profit_factor_1_15": oos_metrics["profit_factor"] >= 1.15,
        "oos_drawdown_8pct": oos_metrics["max_drawdown"] <= 0.08,
        "stress_nonnegative": _metrics(oos, "stress_return_bps")["expectancy_bps"] >= 0,
    }
    report = {
        "protocol": "musca_v3_event_driven_meta_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "data_rows": len(pd.read_parquet(DATASET, columns=["timestamp"])),
        "candidate_events": len(events),
        "tradeable_anchored_events": len(tradeable),
        "base_metrics": base_metrics,
        "oos_metrics": oos_metrics,
        "folds": folds,
        "gates": gates,
        "meta_authorized": all(gates.values()),
        "verdict": "RESEARCH_ONLY" if all(gates.values()) else "FLAT_META_DISABLED",
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    if base_stable and len(tradeable) >= 150:
        regressor, classifier = _models()["ridge"]
        regressor.fit(tradeable[list(FEATURES)], tradeable["net_return_bps"])
        classifier.fit(tradeable[list(FEATURES)], tradeable["net_return_bps"].gt(0))
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "bundle_type": "RESEARCH_ONLY",
                "features": FEATURES,
                "ev_model": regressor,
                "probability_model": classifier,
                "meta_authorized": all(gates.values()),
                "real_capital_allowed": False,
            },
            BUNDLE,
        )
    _atomic_json(
        STATUS,
        {"phase": "complete", "percent": 100, "detail": report["verdict"]},
    )
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
