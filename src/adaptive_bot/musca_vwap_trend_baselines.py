from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE = Path(
    "data/research/binance_public_klines/BTCUSDT-1m-2024-04_2026-07.parquet"
)
REPORT = Path("data/reports/musca_vwap_trend_baselines.json")
HOLDOUT_START = pd.Timestamp("2026-05-11T11:30:00Z")
FAMILIES = ("PULLBACK_RESTART", "VWAP_CROSS", "BREAKOUT")
HORIZONS = (8, 16, 24)
PROTOCOL = {
    "name": "btc_binance_vwap_trend_baselines_v1",
    "bar_minutes": 15,
    "families": list(FAMILIES),
    "holding_bars": list(HORIZONS),
    "direction": "aligned 1h/4h return and weekly VWAP",
    "execution": "next 15m open",
    "normal_cost_bps": 8.0,
    "stress_cost_bps": 16.0,
    "minimum_gross_movement_bps": 24.0,
    "selection": "2024 train and 2025 validation",
    "test": "2026 before sealed holdout",
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _bars(minutes: pd.DataFrame) -> pd.DataFrame:
    source = minutes.set_index("timestamp")
    bars = source.resample("15min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        taker_buy_quote=("taker_buy_quote_volume", "sum"),
        trade_count=("trade_count", "sum"),
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index()
    bars["available_at"] = bars["timestamp"] + pd.Timedelta(minutes=15)
    typical_quote = bars["quote_volume"]
    day = bars["timestamp"].dt.floor("D")
    week = bars["timestamp"].dt.to_period("W-SUN").astype(str)
    bars["daily_vwap"] = typical_quote.groupby(day).cumsum() / bars["volume"].groupby(
        day
    ).cumsum()
    bars["weekly_vwap"] = typical_quote.groupby(week).cumsum() / bars["volume"].groupby(
        week
    ).cumsum()
    bars["rolling_vwap_4h"] = typical_quote.rolling(16).sum() / bars["volume"].rolling(
        16
    ).sum()
    previous = bars["close"].shift()
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous).abs(),
            (bars["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr"] = true_range.rolling(14).mean()
    bars["return_15m"] = bars["close"].pct_change()
    bars["return_1h"] = bars["close"].pct_change(4)
    bars["return_4h"] = bars["close"].pct_change(16)
    sell_quote = bars["quote_volume"] - bars["taker_buy_quote"]
    bars["flow_15m"] = (bars["taker_buy_quote"] - sell_quote) / bars[
        "quote_volume"
    ].replace(0, np.nan)
    buy_1h = bars["taker_buy_quote"].rolling(4).sum()
    quote_1h = bars["quote_volume"].rolling(4).sum()
    bars["flow_1h"] = (2 * buy_1h - quote_1h) / quote_1h.replace(0, np.nan)
    bars["trend_strength_atr"] = (
        bars["return_4h"].abs() * bars["close"] / bars["atr"]
    )
    bars["relative_volume"] = bars["volume"] / bars["volume"].rolling(96).median()
    bars["trade_intensity"] = bars["trade_count"] / bars["trade_count"].rolling(
        96
    ).median()
    bars["previous_rolling_vwap_4h"] = bars["rolling_vwap_4h"].shift()
    return bars


def _signals(bars: pd.DataFrame) -> dict[str, pd.Series]:
    trend = np.sign(bars["return_1h"])
    direction = trend.where(
        trend.eq(np.sign(bars["return_4h"]))
        & trend.eq(np.sign(bars["close"] - bars["weekly_vwap"]))
    )
    restart = direction * bars["return_15m"] > 0
    flow = (direction * bars["flow_15m"] > 0) & (direction * bars["flow_1h"] > 0)
    distance = (bars["close"] - bars["rolling_vwap_4h"]) / bars["atr"]
    pullback = distance.abs().le(0.35) & (direction * distance).ge(-0.10)
    cross = direction * (bars["close"] - bars["rolling_vwap_4h"]) > 0
    cross &= direction * (bars["open"] - bars["previous_rolling_vwap_4h"]) <= 0
    previous_high = bars["high"].rolling(24).max().shift()
    previous_low = bars["low"].rolling(24).min().shift()
    breakout = ((direction > 0) & bars["close"].gt(previous_high)) | (
        (direction < 0) & bars["close"].lt(previous_low)
    )
    valid = direction.notna() & flow & bars["atr"].gt(0)
    return {
        "PULLBACK_RESTART": direction.where(valid & restart & pullback),
        "VWAP_CROSS": direction.where(valid & restart & cross),
        "BREAKOUT": direction.where(valid & breakout),
    }


def _trades(bars: pd.DataFrame, side: pd.Series, horizon: int) -> pd.DataFrame:
    candidates = bars.loc[side.notna()].copy()
    candidates["side"] = side.loc[candidates.index]
    candidates["entry"] = bars["open"].shift(-1).loc[candidates.index]
    candidates["exit"] = bars["close"].shift(-(horizon + 1)).loc[candidates.index]
    candidates["entry_timestamp"] = bars["timestamp"].shift(-1).loc[candidates.index]
    candidates["exit_timestamp"] = bars["timestamp"].shift(-(horizon + 1)).loc[
        candidates.index
    ]
    candidates = candidates.dropna(subset=["entry", "exit", "entry_timestamp"])
    chosen: list[Any] = []
    next_entry = pd.Timestamp.min.tz_localize("UTC")
    for index, row in candidates.iterrows():
        entry_time = pd.Timestamp(row["entry_timestamp"])
        if entry_time > next_entry:
            chosen.append(index)
            next_entry = pd.Timestamp(row["exit_timestamp"])
    result = candidates.loc[chosen].copy()
    result["gross_bps"] = result["side"] * (result["exit"] / result["entry"] - 1) * 10_000
    result["net_bps"] = result["gross_bps"] - 8.0
    result["stress_bps"] = result["gross_bps"] - 16.0
    return result


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["net_bps"]
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "trades": float(len(frame)),
        "gross_expectancy_bps": float(frame["gross_bps"].mean()) if len(frame) else 0.0,
        "expectancy_bps": float(values.mean()) if len(frame) else 0.0,
        "stress_expectancy_bps": float(frame["stress_bps"].mean()) if len(frame) else 0.0,
        "profit_factor": float(gains / losses) if losses else 0.0,
        "win_rate": float(values.gt(0).mean()) if len(frame) else 0.0,
    }


def run() -> dict[str, Any]:
    bars = _bars(pd.read_parquet(SOURCE))
    signals = _signals(bars)
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for horizon in HORIZONS:
            trades = _trades(bars, signals[family], horizon)
            time = pd.to_datetime(trades["entry_timestamp"], utc=True)
            train = _metrics(trades.loc[time.dt.year.eq(2024)])
            validation = _metrics(trades.loc[time.dt.year.eq(2025)])
            stable = (
                train["trades"] >= 50
                and validation["trades"] >= 50
                and train["gross_expectancy_bps"] >= 24
                and validation["gross_expectancy_bps"] >= 24
                and train["expectancy_bps"] > 0
                and validation["expectancy_bps"] > 0
                and train["profit_factor"] >= 1.10
                and validation["profit_factor"] >= 1.10
                and train["stress_expectancy_bps"] >= 0
                and validation["stress_expectancy_bps"] >= 0
            )
            test_frame = trades.loc[
                time.dt.year.eq(2026) & time.lt(HOLDOUT_START)
            ]
            rows.append(
                {
                    "family": family,
                    "holding_bars": horizon,
                    "train": train,
                    "validation": validation,
                    "eligible_train_validation": stable,
                    "test": _metrics(test_frame) if stable else None,
                }
            )
    eligible = [row for row in rows if row["eligible_train_validation"]]
    payload = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "data_rows": len(bars),
        "configurations": rows,
        "eligible": eligible,
        "verdict": "BASE_EDGE_FOUND" if eligible else "NO_BASE_EDGE",
        "holdout_opened": False,
        "live_orders_enabled": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
