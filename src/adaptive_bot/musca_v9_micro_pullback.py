from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.musca_v4_research import COST_BPS, _metrics, _non_overlapping
from adaptive_bot.musca_v5_micro_model import load_5s
from adaptive_bot.musca_v5_research import BARS, build_features

REPORT = Path("data/reports/musca_v9_micro_pullback.json")
STATUS = Path("data/reports/musca_v9_micro_pullback.status.json")
HORIZONS = (5, 15, 30, 60)
PROTOCOL = {
    "name": "musca_v9_exact_micro_vwap_pullback_v1",
    "market": "BINANCE_BTCUSDT_PERPETUAL",
    "timeframe": "1m_state_5s_execution",
    "breakout_minutes": list(HORIZONS),
    "centers": ["ROLLING_15M_VWAP", "ROLLING_60M_VWAP", "DAILY_VWAP", "IMPULSE_AVWAP"],
    "entry": "pullback_to_vwap_then_causal_restart_with_order_flow",
    "room_bps": 3 * COST_BPS,
    "management": "half_at_1.5R_then_cost_protection_and_15m_trailing_timeout_6h",
    "fit": "2026-01",
    "validation": "2026-02",
    "development_test": "2026-03",
    "unopened_audit": "2026-04",
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_minute_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = load_5s()
    minute = (
        path.set_index("timestamp")
        .resample("1min")
        .agg(
            base_volume=("base_volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            signed_quote_volume=("signed_quote_volume", "sum"),
            trade_count=("trade_count", "sum"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .dropna()
        .reset_index()
    )
    minute["available_at"] = minute["timestamp"] + pd.Timedelta(minutes=1)
    previous = minute["close"].shift(1)
    true_range = pd.concat(
        [
            minute["high"] - minute["low"],
            (minute["high"] - previous).abs(),
            (minute["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    minute["atr"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    minute["ofi_1m"] = minute["signed_quote_volume"] / minute["quote_volume"].replace(
        0, np.nan
    )
    minute["ofi_5m"] = minute["signed_quote_volume"].rolling(5).sum() / minute[
        "quote_volume"
    ].rolling(5).sum().replace(0, np.nan)
    minute["return_15m"] = minute["close"].pct_change(15)
    minute["return_60m"] = minute["close"].pct_change(60)
    minute["rolling_vwap_15m"] = minute["quote_volume"].rolling(15).sum() / minute[
        "base_volume"
    ].rolling(15).sum().replace(0, np.nan)
    minute["rolling_vwap_60m"] = minute["quote_volume"].rolling(60).sum() / minute[
        "base_volume"
    ].rolling(60).sum().replace(0, np.nan)
    day = minute["timestamp"].dt.floor("D")
    minute["daily_vwap"] = minute["quote_volume"].groupby(day).cumsum() / minute[
        "base_volume"
    ].groupby(day).cumsum().replace(0, np.nan)
    minute["vwap_slope"] = minute["rolling_vwap_60m"].diff(15) / minute["atr"]
    baseline = minute["quote_volume"].shift(1).rolling(1_440, min_periods=480).median()
    minute["relative_volume"] = minute["quote_volume"] / baseline.replace(0, np.nan)
    context = build_features(pd.read_parquet(BARS))
    context = context[
        ["available_at", "spot_return_1h", "spot_taker_imbalance_15m", "basis_bps"]
    ].sort_values("available_at")
    minute = pd.merge_asof(
        minute.sort_values("available_at"),
        context,
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    votes = np.column_stack(
        [
            np.sign(minute["return_15m"]),
            np.sign(minute["return_60m"]),
            np.sign(minute["vwap_slope"]),
            np.sign(minute["spot_return_1h"]),
            np.sign(minute["ofi_5m"]),
        ]
    )
    minute["trend_vote"] = np.nansum(votes, axis=1)
    minute["direction"] = np.where(
        minute["trend_vote"].ge(2), 1, np.where(minute["trend_vote"].le(-2), -1, 0)
    )
    return minute.dropna().reset_index(drop=True), path


def build_events(data: pd.DataFrame, breakout: int) -> pd.DataFrame:
    high, low, close = (data[column].to_numpy(float) for column in ("high", "low", "close"))
    base, quote = data["base_volume"].to_numpy(float), data["quote_volume"].to_numpy(float)
    direction = data["direction"].to_numpy(int)
    atr_values = data["atr"].to_numpy(float)
    rolling_15 = data["rolling_vwap_15m"].to_numpy(float)
    rolling_60 = data["rolling_vwap_60m"].to_numpy(float)
    daily = data["daily_vwap"].to_numpy(float)
    ofi = data["ofi_1m"].to_numpy(float)
    prior_high = data["high"].shift(1).rolling(breakout).max().to_numpy(float)
    prior_low = data["low"].shift(1).rolling(breakout).min().to_numpy(float)
    impulse = (
        (direction > 0)
        & (close > prior_high)
        & data["relative_volume"].ge(1).to_numpy()
        & data["ofi_1m"].gt(0.05).to_numpy()
        & data["spot_return_1h"].gt(0).to_numpy()
    ) | (
        (direction < 0)
        & (close < prior_low)
        & data["relative_volume"].ge(1).to_numpy()
        & data["ofi_1m"].lt(-0.05).to_numpy()
        & data["spot_return_1h"].lt(0).to_numpy()
    )
    rows: list[dict[str, Any]] = []
    busy_until = -1
    for index_value in np.flatnonzero(impulse):
        index = int(index_value)
        if index <= busy_until:
            continue
        side = direction[index]
        impulse_extreme = high[index] if side > 0 else low[index]
        pullback_extreme = impulse_extreme
        armed_at: int | None = None
        center = np.nan
        for current in range(index + 1, min(index + 121, len(data) - 1)):
            pullback_extreme = (
                min(pullback_extreme, low[current])
                if side > 0
                else max(pullback_extreme, high[current])
            )
            impulse_vwap = quote[index : current + 1].sum() / base[
                index : current + 1
            ].sum()
            centers = (
                rolling_15[current],
                rolling_60[current],
                daily[current],
                impulse_vwap,
            )
            atr = atr_values[current]
            touched = [abs(close[current] - value) <= 0.25 * atr for value in centers]
            depth = side * (impulse_extreme - pullback_extreme) / atr
            quieter = quote[current] < quote[index]
            if armed_at is None and any(touched) and depth >= 0.25 and quieter:
                armed_at = current
                center = centers[int(np.argmin([abs(close[current] - value) for value in centers]))]
            if armed_at is None:
                continue
            if current - armed_at > 6:
                break
            restarted = (
                close[current] > high[current - 1]
                if side > 0
                else close[current] < low[current - 1]
            )
            flow = ofi[current] * side > 0
            accepted = (close[current] - center) * side > 0
            if not (restarted and flow and accepted):
                continue
            room = side * (impulse_extreme - close[current]) / close[current] * 10_000
            if room < 3 * COST_BPS:
                continue
            stop = (
                min(pullback_extreme, center - 0.25 * atr) - 0.1 * atr
                if side > 0
                else max(pullback_extreme, center + 0.25 * atr) + 0.1 * atr
            )
            risk_bps = side * (close[current] - stop) / close[current] * 10_000
            if not 12 <= risk_bps <= 200:
                continue
            row = data.iloc[current]
            rows.append(
                {
                    "protocol_hash": PROTOCOL_HASH,
                    "signal_timestamp": row["timestamp"],
                    "available_at": row["available_at"],
                    "direction": side,
                    "breakout_minutes": breakout,
                    "operating_vwap": center,
                    "stop_price": stop,
                    "room_bps": room,
                    "risk_bps": risk_bps,
                    "trend_vote": row["trend_vote"],
                    "ofi_1m": row["ofi_1m"],
                    "ofi_5m": row["ofi_5m"],
                    "relative_volume": row["relative_volume"],
                    "basis_bps": row["basis_bps"],
                }
            )
            busy_until = current
            break
    return pd.DataFrame(rows)


def label_events(events: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    path = path.sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(path["timestamp"], utc=True).to_numpy(dtype="datetime64[ns]").astype(
        "int64"
    )
    available = pd.to_datetime(path["available_at"], utc=True)
    high, low, close = (path[column].to_numpy(float) for column in ("high", "low", "close"))
    trailing_low = pd.Series(low).rolling(180).min().to_numpy()
    trailing_high = pd.Series(high).rolling(180).max().to_numpy()
    rows: list[dict[Any, Any]] = []
    for event in events.to_dict("records"):
        entry_index = int(
            np.searchsorted(times, pd.Timestamp(event["available_at"]).value, side="left")
        )
        if entry_index >= len(path):
            continue
        side, entry, stop = int(event["direction"]), close[entry_index], float(event["stop_price"])
        risk = side * (entry - stop)
        if risk <= 0 or risk / entry * 10_000 < 12:
            continue
        target, remaining, gross = entry + side * 1.5 * risk, 1.0, 0.0
        exit_index, reason, tp1 = min(entry_index + 4_320, len(path) - 1), "TIMEOUT_6H", False
        for current in range(entry_index + 1, exit_index + 1):
            stopped = low[current] <= stop if side > 0 else high[current] >= stop
            target_hit = high[current] >= target if side > 0 else low[current] <= target
            if stopped:
                gross += remaining * side * (stop - entry) / entry * 10_000
                exit_index, reason, remaining = current, "STRUCTURAL_STOP", 0
                break
            if not tp1 and target_hit:
                gross += 0.5 * side * (target - entry) / entry * 10_000
                remaining, tp1 = 0.5, True
                stop = max(stop, entry + entry * COST_BPS / 10_000) if side > 0 else min(
                    stop, entry - entry * COST_BPS / 10_000
                )
            if tp1 and current - entry_index >= 180:
                proposal = trailing_low[current] if side > 0 else trailing_high[current]
                stop = max(stop, proposal) if side > 0 else min(stop, proposal)
        if remaining:
            gross += remaining * side * (close[exit_index] - entry) / entry * 10_000
        rows.append(
            event
            | {
                "entry_timestamp": available.iat[entry_index],
                "exit_timestamp": available.iat[exit_index],
                "entry_price": entry,
                "final_stop": stop,
                "gross_return_bps": gross,
                "net_return_bps": gross - COST_BPS,
                "stress_return_bps": gross - 2 * COST_BPS,
                "net_return_r": (gross - COST_BPS) / (risk / entry * 10_000),
                "tp1": tp1,
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _expert(breakout: int) -> tuple[dict[str, Any], pd.DataFrame]:
    minute, path = build_minute_features()
    rows = _non_overlapping(label_events(build_events(minute, breakout), path))
    time = pd.to_datetime(rows["entry_timestamp"], utc=True)
    fit, validation, test = (
        rows.loc[time.dt.month.eq(month)] for month in (1, 2, 3)
    )
    fit_metrics, validation_metrics = _metrics(fit), _metrics(validation)
    eligible = (
        fit_metrics["expectancy_bps"] > 0
        and validation_metrics["expectancy_bps"] > 0
        and fit_metrics["profit_factor"] >= 1.05
        and validation_metrics["profit_factor"] >= 1.05
    )
    return (
        {
            "breakout_minutes": breakout,
            "events": len(rows),
            "fit": fit_metrics,
            "validation": validation_metrics,
            "fit_stress": _metrics(fit, "stress_return_bps"),
            "validation_stress": _metrics(validation, "stress_return_bps"),
            "eligible": eligible,
            "test": _metrics(test) if eligible else None,
            "test_stress": _metrics(test, "stress_return_bps") if eligible else None,
        },
        rows,
    )


def run() -> dict[str, Any]:
    _write(
        STATUS,
        {
            "phase": "micro_experts",
            "percent": 5,
            "detail": "4 horizons / 4 CPU workers",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_expert, HORIZONS))
    audits = [audit for audit, _ in results]
    selected = [rows for audit, rows in results if audit["eligible"]]
    combined = (
        pd.concat(selected, ignore_index=True) if selected else results[0][1].iloc[:0].copy()
    )
    if not combined.empty:
        time = pd.to_datetime(combined["entry_timestamp"], utc=True)
        combined = combined.loc[time.dt.month.eq(3)].sort_values("room_bps", ascending=False)
        combined = combined.drop_duplicates("signal_timestamp")
        combined = _non_overlapping(combined)
    metrics, stress = _metrics(combined), _metrics(combined, "stress_return_bps")
    gates = {
        "eligible_expert_exists": bool(selected),
        "test_trades_15": metrics["trades"] >= 15,
        "frequency_0_5_day": metrics["trades"] / 31 >= 0.5,
        "test_expectancy_positive": metrics["expectancy_bps"] > 0,
        "test_pf_1_10": metrics["profit_factor"] >= 1.10,
        "test_stress_nonnegative": stress["expectancy_bps"] >= 0,
        "test_drawdown_8pct": metrics["max_drawdown"] <= 0.08,
    }
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "experts": audits,
        "test_combined": metrics,
        "test_combined_stress": stress,
        "gates": gates,
        "verdict": "READY_FOR_NEW_APRIL_AUDIT" if all(gates.values()) else "NO_MICRO_PULLBACK_EDGE",
        "april_opened": False,
        "real_capital_allowed": False,
    }
    _write(REPORT, report)
    _write(
        STATUS,
        {
            "phase": "complete",
            "percent": 100,
            "detail": report["verdict"],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
