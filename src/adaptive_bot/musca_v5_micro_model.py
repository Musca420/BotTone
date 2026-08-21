from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import COST_BPS, _metrics, _non_overlapping
from adaptive_bot.musca_v5_microstructure import MONTHS, ROOT
from adaptive_bot.musca_v5_research import (
    EVENTS,
    ML_FEATURES,
    _model_rows,
    _models,
)

REPORT = Path("data/reports/musca_v5_micro_model.json")
STATUS = Path("data/reports/musca_v5_micro_model.status.json")
TICK_LABELED = ROOT.parent / "events_with_5s_outcomes.parquet"
MICRO_FEATURES = (
    "ofi_15s",
    "ofi_1m",
    "ofi_5m",
    "ofi_persistence_1m",
    "trade_intensity_15s",
    "trade_intensity_1m",
    "absorption_1m",
    "price_velocity_15s",
    "price_velocity_1m",
)
DIRECTIONAL_MICRO_FEATURES = (
    "ofi_15s",
    "ofi_1m",
    "ofi_5m",
    "ofi_persistence_1m",
    "price_velocity_15s",
    "price_velocity_1m",
)
ONE_SECOND_MICRO_FEATURES = (
    "ofi_1s",
    "ofi_3s",
    "ofi_5s",
    "ofi_15s",
    "ofi_30s",
    "ofi_persistence_5s",
    "ofi_persistence_15s",
    "ofi_persistence_30s",
    "trade_intensity_3s",
    "trade_intensity_15s",
    "trade_intensity_30s",
    "price_velocity_1s",
    "price_velocity_3s",
    "price_velocity_5s",
    "price_velocity_15s",
    "price_velocity_30s",
    "flow_acceleration_5s_30s",
    "absorption_5s",
    "flow_price_alignment_5s",
    "realized_volatility_30s_bps",
    "buy_trade_ratio_5s",
)
DIRECTIONAL_ONE_SECOND_FEATURES = (
    "ofi_1s",
    "ofi_3s",
    "ofi_5s",
    "ofi_15s",
    "ofi_30s",
    "ofi_persistence_5s",
    "ofi_persistence_15s",
    "ofi_persistence_30s",
    "price_velocity_1s",
    "price_velocity_3s",
    "price_velocity_5s",
    "price_velocity_15s",
    "price_velocity_30s",
    "flow_acceleration_5s_30s",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": percent,
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def load_5s() -> pd.DataFrame:
    frames = [pd.read_parquet(ROOT / f"BTCUSDT-aggTrades-5s-{month}.parquet") for month in MONTHS]
    return pd.concat(frames, ignore_index=True).sort_values("available_at").reset_index(drop=True)


def load_1s(months: tuple[str, ...] = MONTHS) -> pd.DataFrame:
    frames = [
        pd.read_parquet(ROOT / f"BTCUSDT-aggTrades-1s-{month}.parquet")
        for month in months
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("available_at")
        .reset_index(drop=True)
    )


def build_one_second_features(data: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_1s() if data is None else data.copy()
    signed = data["signed_quote_volume"]
    quote = data["quote_volume"].replace(0, np.nan)
    for seconds in (1, 3, 5, 15, 30):
        data[f"ofi_{seconds}s"] = (
            signed.rolling(seconds, min_periods=seconds).sum()
            / quote.rolling(seconds, min_periods=seconds).sum()
        )
    sign = np.sign(signed)
    for seconds in (5, 15, 30):
        data[f"ofi_persistence_{seconds}s"] = sign.rolling(
            seconds, min_periods=seconds
        ).mean()
    baseline = data["trade_count"].shift(1).rolling(3_600, min_periods=600).median()
    for seconds in (3, 15, 30):
        data[f"trade_intensity_{seconds}s"] = (
            data["trade_count"].rolling(seconds, min_periods=seconds).sum()
            / (seconds * baseline).replace(0, np.nan)
        )
    for seconds in (1, 3, 5, 15, 30):
        data[f"price_velocity_{seconds}s"] = data["close"].pct_change(seconds) * 10_000
    data["flow_acceleration_5s_30s"] = data["ofi_5s"] - data["ofi_30s"]
    data["absorption_5s"] = data["ofi_5s"].abs() / (
        data["price_velocity_5s"].abs() + 0.1
    )
    data["flow_price_alignment_5s"] = data["ofi_5s"] * data["price_velocity_5s"]
    log_return = np.log(data["close"]).diff()
    data["realized_volatility_30s_bps"] = (
        log_return.rolling(30, min_periods=30).std() * np.sqrt(30) * 10_000
    )
    data["buy_trade_ratio_5s"] = (
        data["buy_count"].rolling(5, min_periods=5).sum()
        / data["trade_count"].rolling(5, min_periods=5).sum().replace(0, np.nan)
    )
    return data[["available_at", *ONE_SECOND_MICRO_FEATURES]].dropna().reset_index(
        drop=True
    )


def build_micro_features(data: pd.DataFrame | None = None) -> pd.DataFrame:
    data = load_5s() if data is None else data.copy()
    signed = data["signed_quote_volume"]
    quote = data["quote_volume"].replace(0, np.nan)
    for bars, name in ((3, "15s"), (12, "1m"), (60, "5m")):
        data[f"ofi_{name}"] = signed.rolling(bars, min_periods=bars).sum() / quote.rolling(
            bars, min_periods=bars
        ).sum()
    sign = np.sign(signed)
    data["ofi_persistence_1m"] = sign.rolling(12, min_periods=12).mean()
    baseline = data["trade_count"].shift(1).rolling(720, min_periods=120).median()
    data["trade_intensity_15s"] = (
        data["trade_count"].rolling(3, min_periods=3).sum() / baseline.replace(0, np.nan)
    )
    data["trade_intensity_1m"] = (
        data["trade_count"].rolling(12, min_periods=12).sum()
        / (12 * baseline).replace(0, np.nan)
    )
    data["price_velocity_15s"] = data["close"].pct_change(3) * 10_000
    data["price_velocity_1m"] = data["close"].pct_change(12) * 10_000
    data["absorption_1m"] = data["ofi_1m"].abs() / (
        data["price_velocity_1m"].abs() + 0.1
    )
    return data[["available_at", *MICRO_FEATURES]].dropna().reset_index(drop=True)


def label_events_5s(events: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Label events on the first executable 5-second close, never before availability."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["available_at"] = pd.to_datetime(data["available_at"], utc=True)
    timestamp_ns: np.ndarray = data["timestamp"].to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    high: np.ndarray = data["high"].to_numpy(float)
    low: np.ndarray = data["low"].to_numpy(float)
    close: np.ndarray = data["close"].to_numpy(float)
    trailing_low = pd.Series(low).rolling(360, min_periods=360).min().to_numpy()
    trailing_high = pd.Series(high).rolling(360, min_periods=360).max().to_numpy()
    rows: list[dict[Any, Any]] = []
    for event in events.to_dict("records"):
        available = pd.Timestamp(event["available_at"])
        entry_index = int(np.searchsorted(timestamp_ns, available.value, side="left"))
        if entry_index >= len(data):
            continue
        side = int(event["direction"])
        entry = float(close[entry_index])
        stop = float(event["stop_price"])
        risk = side * (entry - stop)
        if risk <= 0 or risk / entry * 10_000 < 12:
            continue
        trend = event["exit_style"] == "TREND"
        horizon = 8_640 if trend else 1_440
        exit_index = min(entry_index + horizon, len(data) - 1)
        reason = "TIMEOUT_12H" if trend else "TIMEOUT_2H"
        exit_price = float(close[exit_index])
        mfe = mae = 0.0
        armed = False
        target = float(event.get("target_price", np.nan))
        for current in range(entry_index + 1, exit_index + 1):
            favorable = float(high[current] if side > 0 else low[current])
            adverse = float(low[current] if side > 0 else high[current])
            mfe = max(mfe, side * (favorable - entry))
            mae = min(mae, side * (adverse - entry))
            stopped = low[current] <= stop if side > 0 else high[current] >= stop
            target_hit = not trend and (
                high[current] >= target if side > 0 else low[current] <= target
            )
            # Worst case remains only where the 5-second path is still ambiguous.
            if stopped:
                exit_index, exit_price = current, stop
                reason = "STRUCTURAL_OR_TRAILING_STOP" if trend else "STRUCTURAL_STOP"
                break
            if target_hit:
                exit_index, exit_price, reason = current, target, "VWAP_TARGET"
                break
            if trend and not armed and mfe >= risk:
                armed = True
                stop = max(stop, entry + entry * COST_BPS / 10_000) if side > 0 else min(
                    stop, entry - entry * COST_BPS / 10_000
                )
            if trend and armed and current - entry_index >= 360:
                proposal = float(
                    trailing_low[current] if side > 0 else trailing_high[current]
                )
                stop = max(stop, proposal) if side > 0 else min(stop, proposal)
        gross = side * (exit_price - entry) / entry * 10_000
        rows.append(
            event
            | {
                "entry_timestamp": data["available_at"].iat[entry_index],
                "exit_timestamp": data["available_at"].iat[exit_index],
                "entry_price": entry,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps": gross - COST_BPS,
                "stress_return_bps": gross - 2 * COST_BPS,
                "net_return_r": (gross - COST_BPS) / (risk / entry * 10_000),
                "mfe_bps": mfe / entry * 10_000,
                "mae_bps": mae / entry * 10_000,
                "tp1": armed if trend else reason == "VWAP_TARGET",
                "exit_reason": reason,
                "entry_order_type": "NEXT_5S_CLOSE_CONSERVATIVE",
            }
        )
    return pd.DataFrame(rows)


def train() -> dict[str, Any]:
    _status("labels", 5, "5-second causal execution paths")
    five_seconds = load_5s()
    raw_events = pd.read_parquet(EVENTS)
    event_time = pd.to_datetime(raw_events["available_at"], utc=True)
    raw_events = raw_events.loc[event_time.ge("2026-01-01") & event_time.lt("2026-04-01")]
    labeled = label_events_5s(raw_events, five_seconds)
    labeled.to_parquet(TICK_LABELED, index=False)
    _status("features", 20, "causal 5s order-flow features")
    micro = build_micro_features(five_seconds)
    events = _model_rows(labeled)
    events["available_at"] = pd.to_datetime(events["available_at"], utc=True)
    rows = pd.merge_asof(
        events.sort_values("available_at"),
        micro.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(seconds=5),
    ).dropna(subset=list(MICRO_FEATURES))
    for feature in DIRECTIONAL_MICRO_FEATURES:
        rows[feature] = rows[feature] * rows["direction"]
    features = [*ML_FEATURES, *MICRO_FEATURES]
    timestamp = pd.to_datetime(rows["entry_timestamp"], utc=True)
    fit = rows.loc[timestamp.lt("2026-02-01")]
    calibration = rows.loc[timestamp.ge("2026-02-01") & timestamp.lt("2026-03-01")]
    test = rows.loc[timestamp.ge("2026-03-01") & timestamp.lt("2026-04-01")]
    _status("gpu", 55, f"fit={len(fit)} calibration={len(calibration)} test={len(test)}")
    choices: list[tuple[float, str, float, Any, dict[str, float]]] = []
    configurations = 0
    for name, model in _models().items():
        classification = hasattr(model, "predict_proba")
        target = fit["stress_return_bps"].gt(0) if classification else fit["net_return_bps"]
        model.fit(fit[features], target)
        scores = (
            model.predict_proba(calibration[features])[:, 1]
            if classification
            else model.predict(calibration[features])
        )
        for coverage in (0.05, 0.10, 0.20, 0.30):
            configurations += 1
            threshold = float(np.quantile(scores, 1 - coverage))
            selected = _non_overlapping(calibration.loc[scores >= threshold])
            metrics = _metrics(selected, "stress_return_bps")
            if (
                len(selected) / 28 >= 0.5
                and metrics["expectancy_bps"] > 0
                and metrics["profit_factor"] >= 1.05
            ):
                choices.append((metrics["expectancy_bps"], name, threshold, model, metrics))
    selected_test = test.iloc[:0].copy()
    champion = "FLAT"
    calibration_metrics: dict[str, float] = _metrics(selected_test)
    if choices:
        _, champion, threshold, model, calibration_metrics = max(
            choices, key=lambda choice: choice[0]
        )
        scores = (
            model.predict_proba(test[features])[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(test[features])
        )
        selected_test = _non_overlapping(test.loc[scores >= threshold])
    metrics = _metrics(selected_test)
    stress = _metrics(selected_test, "stress_return_bps")
    gates = {
        "test_trades_15": len(selected_test) >= 15,
        "frequency_0_5_day": len(selected_test) / 31 >= 0.5,
        "expectancy_positive": metrics["expectancy_bps"] > 0,
        "profit_factor_1_10": metrics["profit_factor"] >= 1.10,
        "stress_nonnegative": stress["expectancy_bps"] >= 0,
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "Binance official USD-M aggTrades",
        "rows": {
            "micro_5s": len(micro),
            "fit": len(fit),
            "calibration": len(calibration),
            "test": len(test),
        },
        "features": features,
        "configurations_evaluated": configurations,
        "champion": champion,
        "calibration_metrics": calibration_metrics,
        "test_metrics": metrics,
        "test_stress_metrics": stress,
        "gates": gates,
        "verdict": "MICROSTRUCTURE_EDGE_FOUND" if all(gates.values()) else "NO_MICROSTRUCTURE_EDGE",
        "real_capital_allowed": False,
    }
    _atomic_json(REPORT, report)
    _status("complete", 100, str(report["verdict"]))
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
