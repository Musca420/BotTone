from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.indicators.atr import atr

ROOT = Path("data/ml/musca_v2")
STATUS = Path("data/reports/musca_v2_train.status.json")
ARCHIVE = "https://data.binance.vision/data"
DATASET = ROOT / "btc_spot_perp_5m.parquet"
TRADES = ROOT / "trades.parquet"
OOS_TRADES = ROOT / "oos_trades.parquet"
BUNDLE = Path("data/models/musca_v2/bundle.joblib")
REPORT = Path("data/reports/musca_v2_research.json")
FEATURES = (
    "direction",
    "trend_7d",
    "trend_30d",
    "trend_strength",
    "distance_daily_vwap_atr",
    "distance_weekly_vwap_atr",
    "relative_volume",
    "taker_imbalance_1h",
    "taker_imbalance_4h",
    "spot_confirmation",
    "spot_perp_divergence_bps",
    "basis_bps",
    "atr_percentile",
    "hour_sin",
    "hour_cos",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _status(detail: str, done: int, total: int) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": "official_btc_history",
            "detail": detail,
            "completed_archives": done,
            "total_archives": total,
            "percent": round(100 * done / total, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _download(relative: str) -> Path:
    target = ROOT / "official" / relative
    if target.exists():
        return target
    url = f"{ARCHIVE}/{relative}"
    headers = {"User-Agent": "adaptive-range-musca-v2/1"}
    with urllib.request.urlopen(
        urllib.request.Request(f"{url}.CHECKSUM", headers=headers), timeout=60
    ) as response:
        expected = response.read().decode("ascii").split()[0].lower()
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=180
    ) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"Binance checksum mismatch: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def _parse(path: Path, prefix: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        frame = pd.read_csv(archive.open(archive.namelist()[0]), header=None)
    frame = frame.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]].copy()
    frame.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote",
    ]
    raw_time = pd.to_numeric(frame.pop("timestamp"), errors="coerce")
    unit = "us" if raw_time.dropna().median() > 10**14 else "ms"
    frame.insert(
        0,
        "timestamp",
        pd.to_datetime(raw_time.to_numpy(), unit=unit, utc=True),  # type: ignore[call-overload]
    )
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.rename(columns={column: f"{prefix}_{column}" for column in frame.columns[1:]})


def download_history(start: str = "2020-01", end: str = "2026-07") -> pd.DataFrame:
    output = DATASET
    months = pd.period_range(start, end, freq="M")
    names = [
        f"{market}/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{month}.zip"
        for market in ("spot", "futures/um")
        for month in months
    ]
    completed = 0
    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, name): name for name in names}
        for future, name in [(future, futures[future]) for future in futures]:
            paths[name] = future.result()
            completed += 1
            _status(name, completed, len(names))
    frames: dict[str, list[pd.DataFrame]] = {"spot": [], "perp": []}
    for name in names:
        market = "spot" if name.startswith("spot/") else "perp"
        frames[market].append(_parse(paths[name], market))
    spot = pd.concat(frames["spot"], ignore_index=True).drop_duplicates("timestamp")
    perpetual = pd.concat(frames["perp"], ignore_index=True).drop_duplicates("timestamp")
    joined = perpetual.merge(spot, on="timestamp", how="inner", validate="one_to_one")
    joined = joined.sort_values("timestamp").dropna().reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    joined.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    _status(f"BTC 5m ready: {len(joined):,} rows", len(names), len(names))
    return joined


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    time = pd.to_datetime(data["timestamp"], utc=True)
    day = time.dt.floor("D")
    week = time.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    data["atr"] = atr(data["perp_high"], data["perp_low"], data["perp_close"], 14)
    data["daily_vwap"] = (
        data["perp_quote_volume"].groupby(day).cumsum() / data["perp_volume"].groupby(day).cumsum()
    )
    data["weekly_vwap"] = (
        data["perp_quote_volume"].groupby(week).cumsum()
        / data["perp_volume"].groupby(week).cumsum()
    )
    data["trend_7d"] = data["perp_close"].pct_change(2016)
    data["trend_30d"] = data["perp_close"].pct_change(8640)
    data["trend_strength"] = 0.6 * data["trend_7d"] + 0.4 * data["trend_30d"]
    data["direction"] = np.sign(data["trend_strength"])
    data["distance_daily_vwap_atr"] = (data["perp_close"] - data["daily_vwap"]) / data["atr"]
    data["distance_weekly_vwap_atr"] = (data["perp_close"] - data["weekly_vwap"]) / data["atr"]
    data["relative_volume"] = (
        data["perp_quote_volume"]
        / data["perp_quote_volume"].shift(1).rolling(2016, min_periods=2016).median()
    )
    imbalance = (2 * data["perp_taker_buy_quote"] - data["perp_quote_volume"]) / data[
        "perp_quote_volume"
    ].replace(0, np.nan)
    data["taker_imbalance_1h"] = imbalance.rolling(12, min_periods=12).mean()
    data["taker_imbalance_4h"] = imbalance.rolling(48, min_periods=48).mean()
    data["spot_confirmation"] = pd.Series(
        np.sign(data["spot_close"].pct_change(2016)), index=data.index
    ).eq(data["direction"])
    data["spot_perp_divergence_bps"] = (
        data["spot_close"].pct_change(12) - data["perp_close"].pct_change(12)
    ) * 10_000
    data["basis_bps"] = (data["perp_close"] / data["spot_close"] - 1) * 10_000
    data["atr_percentile"] = (
        data["atr"].shift(1).rolling(8640, min_periods=2016).rank(pct=True) * 100
    )
    hour = time.dt.hour + time.dt.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return data


def replay(frame: pd.DataFrame) -> pd.DataFrame:
    data = build_features(frame)
    side = data["direction"]
    side_values = side.to_numpy(dtype=float)
    reclaim = (
        side.gt(0)
        & data["perp_close"].gt(data["daily_vwap"])
        & data["perp_close"].shift(1).le(data["daily_vwap"].shift(1))
    ) | (
        side.lt(0)
        & data["perp_close"].lt(data["daily_vwap"])
        & data["perp_close"].shift(1).ge(data["daily_vwap"].shift(1))
    )
    low = data["perp_low"].shift(1).rolling(2016, min_periods=2016).min().to_numpy()
    high = data["perp_high"].shift(1).rolling(2016, min_periods=2016).max().to_numpy()
    arrays = {
        key: data[key].to_numpy()
        for key in ("perp_open", "perp_high", "perp_low", "perp_close", "atr")
    }
    trend = data["trend_strength"].to_numpy()
    times = pd.to_datetime(data["timestamp"], utc=True)
    trades: list[dict[str, Any]] = []
    busy = -1
    for raw_index in np.flatnonzero(reclaim.to_numpy(bool)):
        index = int(raw_index)
        if index <= busy or index + 1 >= len(data):
            continue
        direction = int(side_values[index])
        entry = float(arrays["perp_open"][index + 1])
        stop = (
            float(low[index] - 0.1 * arrays["atr"][index])
            if direction > 0
            else float(high[index] + 0.1 * arrays["atr"][index])
        )
        risk = direction * (entry - stop)
        if not np.isfinite(risk) or risk / entry * 10_000 < 12:
            continue
        exit_index = min(index + 4032, len(data) - 1)
        exit_price = float(arrays["perp_close"][exit_index])
        exit_reason = "TIMEOUT_14D"
        initial_stop = stop
        for current in range(index + 1, exit_index + 1):
            stop_hit = (
                arrays["perp_low"][current] <= stop
                if direction > 0
                else arrays["perp_high"][current] >= stop
            )
            if stop_hit:
                open_price = float(arrays["perp_open"][current])
                exit_price = min(open_price, stop) if direction > 0 else max(open_price, stop)
                exit_index, exit_reason = current, "STRUCTURAL_TRAIL"
                break
            if current > index + 288 and np.sign(trend[current]) == -direction:
                exit_price = float(arrays["perp_close"][current])
                exit_index, exit_reason = current, "TREND_REVERSAL"
                break
            if current - index >= 576 and current % 288 == 0:
                proposal = (
                    float(low[current] - 0.1 * arrays["atr"][current])
                    if direction > 0
                    else float(high[current] + 0.1 * arrays["atr"][current])
                )
                stop = max(stop, proposal) if direction > 0 else min(stop, proposal)
                if (direction > 0 and stop < initial_stop) or (
                    direction < 0 and stop > initial_stop
                ):
                    raise RuntimeError("Musca v2 trailing stop widened")
        gross = direction * (exit_price - entry) / entry * 10_000
        row = {column: data.iloc[index][column] for column in FEATURES}
        row |= {
            "atr": float(data.iloc[index]["atr"]),
            "daily_vwap": float(data.iloc[index]["daily_vwap"]),
            "weekly_vwap": float(data.iloc[index]["weekly_vwap"]),
        }
        trades.append(
            row
            | {
                "signal_timestamp": times.iat[index],
                "entry_timestamp": times.iat[index + 1],
                "exit_timestamp": times.iat[exit_index],
                "entry_price": entry,
                "exit_price": exit_price,
                "initial_stop": initial_stop,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps_8": gross - 8,
                "net_return_bps_16": gross - 16,
                "risk_bps": risk / entry * 10_000,
                "net_return_r": (gross - 8) / (risk / entry * 10_000),
                "exit_reason": exit_reason,
            }
        )
        busy = exit_index
    return pd.DataFrame(trades)


def _metrics(rows: pd.DataFrame, target: str = "net_return_bps_8") -> dict[str, float]:
    values = rows[target].to_numpy(float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    risk_curve = np.cumsum(rows["net_return_r"].to_numpy(float) * 0.0025)
    drawdown = np.max(np.maximum.accumulate(np.r_[0.0, risk_curve]) - np.r_[0.0, risk_curve])
    return {
        "trades": float(len(rows)),
        "expectancy_bps": float(values.mean()),
        "profit_factor": float(gains / losses),
        "win_rate": float((values > 0).mean()),
        "max_drawdown_fraction": float(drawdown),
    }


def train() -> dict[str, Any]:
    data = pd.read_parquet(DATASET)
    trades = replay(data)
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADES, index=False)
    times = pd.to_datetime(trades["entry_timestamp"], utc=True)
    fold_reports: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for year in range(2023, 2027):
        train_rows = trades.loc[times.lt(pd.Timestamp(f"{year}-01-01", tz="UTC"))]
        test_rows = trades.loc[times.dt.year.eq(year)]
        if len(train_rows) < 100 or len(test_rows) < 20:
            continue
        models = {
            "ridge": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
            ),
            "xgboost_gpu": XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                random_state=20260805,
                n_jobs=4,
            ),
        }
        candidates: list[tuple[float, str, Any]] = []
        split = pd.to_datetime(train_rows["entry_timestamp"], utc=True).quantile(0.8)
        fit_rows = train_rows.loc[pd.to_datetime(train_rows["entry_timestamp"], utc=True).lt(split)]
        calibration = train_rows.drop(fit_rows.index)
        for name, model in models.items():
            model.fit(fit_rows[list(FEATURES)], fit_rows["net_return_bps_8"])
            predicted = model.predict(calibration[list(FEATURES)])
            selected = calibration.loc[predicted > 0]
            metric = _metrics(selected)["expectancy_bps"] if len(selected) >= 20 else -np.inf
            candidates.append((metric, name, model))
        baseline = _metrics(calibration)["expectancy_bps"]
        best_metric, champion, model = max(candidates, key=lambda value: value[0])
        if best_metric <= baseline:
            champion, model = "deterministic", None
            selected_test = test_rows.copy()
        else:
            model.fit(train_rows[list(FEATURES)], train_rows["net_return_bps_8"])
            selected_test = test_rows.loc[model.predict(test_rows[list(FEATURES)]) > 0].copy()
        selected_test["outer_year"] = year
        selected_test["champion"] = champion
        predictions.append(selected_test)
        fold_reports.append(
            {
                "year": year,
                "champion": champion,
                "train_trades": len(train_rows),
                "test_trades": len(selected_test),
                "metrics": _metrics(selected_test),
            }
        )
        _status(f"GPU audit year {year}: {champion}", year - 2022, 4)
    oos = pd.concat(predictions, ignore_index=True)
    oos.to_parquet(OOS_TRADES, index=False)
    base_metrics = _metrics(trades)
    oos_metrics = _metrics(oos)
    annual = trades.groupby(times.dt.year)["net_return_bps_8"].mean()
    gates = {
        "trades_300": len(oos) >= 300,
        "expectancy_positive": oos_metrics["expectancy_bps"] > 0,
        "profit_factor_1_15": oos_metrics["profit_factor"] >= 1.15,
        "drawdown_8pct": oos_metrics["max_drawdown_fraction"] <= 0.08,
        "stress_2x_nonnegative": _metrics(oos, "net_return_bps_16")["expectancy_bps"] >= 0,
        "majority_years_positive": float(annual.gt(0).mean()) > 0.5,
    }
    report = {
        "verdict": "RESEARCH_ONLY_DISCOVERY_EDGE",
        "protocol": (
            "BTC trend(0.6*7d+0.4*30d), daily VWAP reclaim, 7d structural trailing, 14d timeout"
        ),
        "data_rows": len(data),
        "data_start": pd.Timestamp(data["timestamp"].min()).isoformat(),
        "data_end": pd.Timestamp(data["timestamp"].max()).isoformat(),
        "base_metrics": base_metrics,
        "oos_metrics": oos_metrics,
        "stress_metrics": _metrics(oos, "net_return_bps_16"),
        "annual_expectancy_bps": {str(k): float(v) for k, v in annual.items()},
        "folds": fold_reports,
        "gates": gates,
        "gates_passed": all(gates.values()),
        "holdout_status": "FUTURE_SHADOW_REQUIRED; discovery years already observed",
        "real_capital_allowed": False,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    final_meta = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    final_meta.fit(trades[list(FEATURES)], trades["net_return_bps_8"])
    joblib.dump(
        {
            "bundle_type": "RESEARCH_ONLY",
            "strategy_profile": "musca_v2_long_horizon_vwap",
            "protocol": report["protocol"],
            "features": FEATURES,
            "meta_model": final_meta,
            "meta_model_role": "challenger_filter; deterministic base remains champion",
            "real_capital_allowed": False,
            "live_orders_enabled": False,
        },
        BUNDLE,
    )
    _atomic_json(REPORT, report)
    return report


if __name__ == "__main__":
    download_history() if not DATASET.exists() else train()
