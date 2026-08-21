from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from adaptive_bot.expert_policy import moving_block_lower_bound
from adaptive_bot.hybrid_policy_v11 import btc_inventory, return_metrics
from adaptive_bot.hybrid_policy_v22.path_audit import atomic_parquet, non_overlapping
from adaptive_bot.hybrid_policy_v22.protocol import sha256
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr

PROTOCOL = "hybrid_v24_trend_vwap_pullback_continuation"
ROOT = Path("data/ml/hybrid_v24")
ARCHIVE_ROOT = ROOT / "binance_spot_archives"
SPOT_PATH = ROOT / "binance_spot_btcusdt_1m.parquet"
FEATURE_PATH = ROOT / "causal_features_15m.parquet"
CANDIDATE_PATH = ROOT / "candidates.parquet"
MATRIX_PATH = ROOT / "base_matrix.parquet"
TRADE_PATH = ROOT / "base_trades.parquet"
MODEL_ROOT = Path("data/models/expert_policy/v24")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
RESEARCH_BUNDLE_ROOT = Path("data/models/expert_policy/v24_research_bundle")
RESEARCH_BUNDLE_PATH = RESEARCH_BUNDLE_ROOT / "bundle.joblib"
DATA_REPORT_PATH = Path("data/reports/ml_hybrid_v24_data_audit.json")
BASE_REPORT_PATH = Path("data/reports/ml_hybrid_v24_base_audit.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v24.status.json")
ORDERFLOW_PATH = Path("data/ml/hybrid_v14/binance_reference_orderflow_1m.parquet")
CONTEXT_PATH = Path("data/ml/hybrid_v19/binance_context_5m.parquet")
SPEC_PATH = Path("C:/Users/david/Downloads/V24_VWAP_TREND_PULLBACK_SPEC.md")
BASE_COST_BPS = 4.0
STRESS_COST_BPS = 8.0
SPOT_ARCHIVE_URL = "https://data.binance.vision/data/spot"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def strategy_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for adx_threshold in (18, 22, 26):
        for ema_lookback in (6, 12):
            for weekly_lookback in (6, 12):
                for shock_percentile in (95, 99):
                    for donchian_bars in (20, 40):
                        row: dict[str, Any] = {
                            "adx_threshold": adx_threshold,
                            "ema_slope_lookback": ema_lookback,
                            "weekly_vwap_slope_lookback": weekly_lookback,
                            "shock_percentile": shock_percentile,
                            "donchian_bars": donchian_bars,
                            "relative_volume_min": 1.2,
                            "taker_imbalance_min": 0.05,
                            "oi_change_1h_min": -0.005,
                            "basis_change_abs_max_bps": 10.0,
                            "maximum_breakout_vwap_distance_atr": 3.0,
                            "pullback_zone_atr": 0.5,
                            "pullback_timeout_minutes": 60,
                            "confirmation_timeout_minutes": 15,
                            "maximum_stop_atr": 2.0,
                        }
                        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
                        row["config_id"] = (
                            f"v24-s-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"
                        )
                        rows.append(row)
    return rows


def exit_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tp1 in (1.0, 1.5):
        for tp2 in (2.0, 3.0):
            for partial in (0.33, 0.5):
                for trailing in ("atr_1", "swing_5m"):
                    for timeout in (240, 360):
                        row: dict[str, Any] = {
                            "tp1_r": tp1,
                            "tp2_r": tp2,
                            "partial_tp1": partial,
                            "trailing": trailing,
                            "timeout_minutes": timeout,
                            "break_even": "after_tp1",
                        }
                        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
                        row["exit_config_id"] = (
                            f"v24-x-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"
                        )
                        rows.append(row)
    return rows


def protocol_payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "spec_sha256": sha256(SPEC_PATH),
        "strategy": "TREND_VWAP_PULLBACK_CONTINUATION",
        "symbol": "BTCUSDT",
        "direction_source": "4h/1h trend plus 15m breakout and spot/perpetual confirmation",
        "vwap_role": "pullback zone, anchored invalidation and trade management",
        "timeframes_minutes": [240, 60, 15, 5, 1],
        "costs_round_trip_bps": [BASE_COST_BPS, STRESS_COST_BPS],
        "strategy_configs": strategy_configs(),
        "exit_configs": exit_configs(),
        "walk_forward_weeks": [52, 8, 8, 8],
        "early_exit": {
            "regime_invalidation": "next executable minute",
            "vwap_acceptance_bars_5m": 2,
            "taker_flow_reversal_bars_5m": 2,
            "spot_divergence_bars_5m": 2,
        },
        "meta_gate": {
            "oos_trades": ">=200",
            "expectancy_4bps": ">0",
            "profit_factor": ">=1.10",
            "positive_fold_fraction": ">=0.50",
            "maximum_fold_pnl_share": "<=0.35",
            "mean_gross_edge_bps": ">=12",
        },
        "bundle": {
            "bundle_type": "RESEARCH_ONLY",
            "real_capital_allowed": False,
            "auto_promotion": False,
        },
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {
        "protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "implementation_sha256": sha256(Path(__file__)),
    }


def preregister() -> dict[str, Any]:
    current = protocol_payload()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != current["protocol_sha256"]:
            raise RuntimeError("V24 protocol changed after freezing")
        return dict(existing)
    payload = current | {"registered_at": datetime.now(UTC).isoformat()}
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def finalize_implementation(protocol: dict[str, Any]) -> dict[str, Any]:
    current = sha256(Path(__file__))
    if protocol.get("implementation_finalized"):
        if protocol.get("implementation_sha256") != current:
            raise RuntimeError("V24 implementation changed after finalization")
        return protocol
    finalized = protocol | {
        "implementation_sha256": current,
        "implementation_finalized": True,
        "implementation_finalized_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(PROTOCOL_PATH, finalized)
    return finalized


def _archive_names(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    naive_start, naive_end = start.tz_localize(None), end.tz_localize(None)
    months = pd.period_range(naive_start, naive_end, freq="M")
    current = naive_end.to_period("M")
    names = [
        f"monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip" for month in months if month < current
    ]
    names.extend(
        f"daily/klines/BTCUSDT/1m/BTCUSDT-1m-{day:%Y-%m-%d}.zip"
        for day in pd.date_range(current.start_time, naive_end.floor("D"), freq="D")
    )
    return names


def _download_archive(relative: str) -> Path:
    target = ARCHIVE_ROOT / relative
    if target.exists():
        return target
    url = f"{SPOT_ARCHIVE_URL}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-range-research/24"})
    try:
        with urllib.request.urlopen(f"{url}.CHECKSUM", timeout=60) as response:
            expected = response.read().decode("ascii").split()[0].lower()
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"missing official Binance spot archive: {relative}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Binance spot checksum mismatch: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


def _parse_spot_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise RuntimeError(f"unexpected Binance spot archive layout: {path}")
        frame = pd.read_csv(archive.open(members[0]), header=None)
    frame = frame.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 10]].copy()
    frame.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_quote",
    ]
    numeric_time = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["timestamp"] = (
        pd.to_datetime(numeric_time, unit="us", utc=True)
        if numeric_time.dropna().median() > 10**14
        else pd.to_datetime(numeric_time, unit="ms", utc=True)
    )
    for column in frame.columns.drop("timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def ensure_spot(start: pd.Timestamp, end: pd.Timestamp, *, resume: bool) -> pd.DataFrame:
    if resume and SPOT_PATH.exists():
        frame = pd.read_parquet(SPOT_PATH)
        times = pd.to_datetime(frame["timestamp"], utc=True)
        if times.min() <= start.floor("min") and times.max() >= end.floor("min"):
            return frame.loc[times.between(start.floor("min"), end.floor("min"))].reset_index(
                drop=True
            )
    names = _archive_names(start, end)
    frames: list[pd.DataFrame] = []
    for number, relative in enumerate(names, start=1):
        _status(
            "spot_download",
            f"Binance spot archive {number}/{len(names)}",
            2 + 16 * number / len(names),
        )
        frames.append(_parse_spot_archive(_download_archive(relative)))
    result = pd.concat(frames, ignore_index=True)
    times = pd.to_datetime(result["timestamp"], utc=True)
    result = result.loc[times.between(start.floor("min"), end.floor("min"))]
    result = result.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    atomic_parquet(SPOT_PATH, result)
    return result


def _bars(minutes: pd.DataFrame, interval: int) -> pd.DataFrame:
    data = minutes.set_index("timestamp")
    grouped = data.resample(f"{interval}min", origin="epoch", closed="left", label="left")
    aggregations: dict[str, Any] = {
        column: operation
        for prefix in ("perp", "spot")
        for column, operation in (
            (f"{prefix}_open", "first"),
            (f"{prefix}_high", "max"),
            (f"{prefix}_low", "min"),
            (f"{prefix}_close", "last"),
            (f"{prefix}_volume", "sum"),
            (f"{prefix}_quote_volume", "sum"),
            (f"{prefix}_trade_count", "sum"),
            (f"{prefix}_taker_buy_quote", "sum"),
        )
    }
    aggregations["funding_rate"] = "last"
    result = grouped.agg(cast(Any, aggregations))
    result["minute_count"] = grouped["perp_close"].count()
    result["data_valid"] = result["minute_count"].eq(interval) & result.notna().all(axis=1)
    result["available_at"] = result.index + pd.Timedelta(minutes=interval)
    return result.reset_index()


def _group_vwap(frame: pd.DataFrame, prefix: str, key: pd.Series) -> pd.Series:
    quote = frame[f"{prefix}_quote_volume"]
    volume = frame[f"{prefix}_volume"]
    return quote.groupby(key).cumsum() / volume.groupby(key).cumsum().replace(0, np.nan)


def build_features(*, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if resume and FEATURE_PATH.exists() and (ROOT / "joined_minutes.parquet").exists():
        features = pd.read_parquet(FEATURE_PATH)
        minutes = pd.read_parquet(ROOT / "joined_minutes.parquet")
        return features, minutes, json.loads(DATA_REPORT_PATH.read_text(encoding="utf-8"))
    inventory = btc_inventory()["binance"]
    perp = pd.read_parquet(Path(inventory["path"])).copy()
    perp["timestamp"] = pd.to_datetime(perp["timestamp"], utc=True)
    start, end = perp["timestamp"].min(), perp["timestamp"].max()
    spot = ensure_spot(start, end, resume=resume)
    flow = pd.read_parquet(ORDERFLOW_PATH).copy()
    flow["timestamp"] = pd.to_datetime(flow["timestamp"], utc=True)
    perp = perp.merge(
        flow[["timestamp", "trade_count", "taker_buy_quote"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    perp = perp.rename(
        columns={
            column: f"perp_{column}"
            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trade_count",
                "taker_buy_quote",
                "data_valid",
            ]
        }
    )
    spot = spot.rename(
        columns={column: f"spot_{column}" for column in spot.columns if column != "timestamp"}
    )
    minutes = perp.merge(spot, on="timestamp", how="inner", validate="one_to_one")
    required = [
        str(column)
        for column in minutes.columns
        if str(column).startswith(("perp_", "spot_"))
        and any(
            name in str(column)
            for name in ("open", "high", "low", "close", "volume", "trade_count", "taker")
        )
    ]
    minutes["data_valid"] = minutes[required].notna().all(axis=1) & minutes[
        "perp_data_valid"
    ].astype(bool)
    minutes = minutes.sort_values("timestamp").reset_index(drop=True)
    minute_time = pd.to_datetime(minutes["timestamp"], utc=True)
    minute_day = minute_time.dt.floor("D")
    minutes["perp_daily_vwap"] = _group_vwap(minutes, "perp", minute_day)
    minutes["spot_daily_vwap"] = _group_vwap(minutes, "spot", minute_day)
    atomic_parquet(ROOT / "joined_minutes.parquet", minutes)

    bars15, bars5, bars1h, bars4h = (_bars(minutes, interval) for interval in (15, 5, 60, 240))
    for frame in (bars15, bars5, bars1h, bars4h):
        for prefix in ("perp", "spot"):
            sell = frame[f"{prefix}_quote_volume"] - frame[f"{prefix}_taker_buy_quote"]
            frame[f"{prefix}_taker_imbalance"] = (
                frame[f"{prefix}_taker_buy_quote"] - sell
            ) / frame[f"{prefix}_quote_volume"].replace(0, np.nan)

    time15 = pd.to_datetime(bars15["timestamp"], utc=True)
    day15 = time15.dt.floor("D")
    bars15["perp_daily_vwap"] = _group_vwap(bars15, "perp", day15)
    bars15["spot_daily_vwap"] = _group_vwap(bars15, "spot", day15)
    for prefix in ("perp", "spot"):
        average_price = bars15[f"{prefix}_quote_volume"] / bars15[
            f"{prefix}_volume"
        ].replace(0, np.nan)
        weighted_square = average_price.pow(2) * bars15[f"{prefix}_volume"]
        cumulative_square = weighted_square.groupby(day15).cumsum()
        cumulative_volume = bars15[f"{prefix}_volume"].groupby(day15).cumsum()
        variance = cumulative_square / cumulative_volume.replace(0, np.nan) - bars15[
            f"{prefix}_daily_vwap"
        ].pow(2)
        deviation = np.sqrt(variance.clip(lower=0))
        bars15[f"{prefix}_daily_vwap_deviation"] = deviation
        bars15[f"{prefix}_daily_vwap_upper"] = bars15[f"{prefix}_daily_vwap"] + deviation
        bars15[f"{prefix}_daily_vwap_lower"] = bars15[f"{prefix}_daily_vwap"] - deviation
    rolling = 96
    bars15["perp_rolling_vwap"] = bars15["perp_quote_volume"].rolling(
        rolling, min_periods=rolling
    ).sum() / bars15["perp_volume"].rolling(rolling, min_periods=rolling).sum().replace(0, np.nan)
    bars15["atr_15m"] = atr(bars15["perp_high"], bars15["perp_low"], bars15["perp_close"], 14)
    bars15["daily_vwap_slope_15m"] = bars15["perp_daily_vwap"].diff() / bars15[
        "atr_15m"
    ].replace(0, np.nan)
    bars15["daily_vwap_curvature_15m"] = bars15["daily_vwap_slope_15m"].diff()
    bars15["spot_perp_vwap_distance_bps"] = (
        (bars15["perp_daily_vwap"] - bars15["spot_daily_vwap"])
        / bars15["spot_daily_vwap"]
        * 10_000
    )
    for prefix in ("perp", "spot"):
        bars15[f"{prefix}_return_15m"] = bars15[f"{prefix}_close"].pct_change(fill_method=None)
    bars15["taker_imbalance_difference_15m"] = (
        bars15["perp_taker_imbalance"] - bars15["spot_taker_imbalance"]
    )
    bars15["relative_volume_15m"] = bars15["perp_quote_volume"] / bars15["perp_quote_volume"].shift(
        1
    ).rolling(96, min_periods=96).median().replace(0, np.nan)
    bars15["range_atr_15m"] = (bars15["perp_high"] - bars15["perp_low"]) / bars15[
        "atr_15m"
    ].replace(0, np.nan)
    bars15["range_percentile_15m"] = (
        bars15["range_atr_15m"].shift(1).rolling(2_688, min_periods=672).rank(pct=True) * 100
    )
    for lookback in (20, 40):
        bars15[f"perp_high_{lookback}"] = (
            bars15["perp_high"].shift(1).rolling(lookback, min_periods=lookback).max()
        )
        bars15[f"perp_low_{lookback}"] = (
            bars15["perp_low"].shift(1).rolling(lookback, min_periods=lookback).min()
        )
        bars15[f"spot_high_{lookback}"] = (
            bars15["spot_high"].shift(1).rolling(lookback, min_periods=lookback).max()
        )
        bars15[f"spot_low_{lookback}"] = (
            bars15["spot_low"].shift(1).rolling(lookback, min_periods=lookback).min()
        )

    time1h = pd.to_datetime(bars1h["timestamp"], utc=True)
    week = time1h.dt.floor("D") - pd.to_timedelta(time1h.dt.weekday, unit="D")
    bars1h["perp_weekly_vwap"] = _group_vwap(bars1h, "perp", week)
    bars1h["spot_weekly_vwap"] = _group_vwap(bars1h, "spot", week)
    bars1h["ema50_1h"] = bars1h["perp_close"].ewm(span=50, adjust=False, min_periods=50).mean()
    bars1h["atr_1h"] = atr(bars1h["perp_high"], bars1h["perp_low"], bars1h["perp_close"], 14)
    bars1h["adx_1h"] = adx(bars1h["perp_high"], bars1h["perp_low"], bars1h["perp_close"], 14)["adx"]
    bars1h["volatility_percentile_1h"] = (
        bars1h["atr_1h"].shift(1).rolling(2_160, min_periods=720).rank(pct=True) * 100
    )
    bars1h["basis_bps_1h"] = (
        (bars1h["perp_close"] - bars1h["spot_close"]) / bars1h["spot_close"] * 10_000
    )
    bars1h["basis_change_1h"] = bars1h["basis_bps_1h"].diff()
    funding_history = bars1h["funding_rate"].shift(1).rolling(2_160, min_periods=720)
    funding_mean = funding_history.mean()
    funding_std = funding_history.std(ddof=0)
    bars1h["funding_z_1h"] = (
        (bars1h["funding_rate"] - funding_mean) / funding_std.replace(0, np.nan)
    ).where(funding_std.ne(0), 0.0)
    for lookback in (6, 12):
        bars1h[f"ema50_slope_{lookback}h"] = (
            bars1h["ema50_1h"] - bars1h["ema50_1h"].shift(lookback)
        ) / bars1h["atr_1h"].replace(0, np.nan)
        bars1h[f"weekly_vwap_slope_{lookback}h"] = (
            bars1h["perp_weekly_vwap"] - bars1h["perp_weekly_vwap"].shift(lookback)
        ) / bars1h["atr_1h"].replace(0, np.nan)
    bars1h["perp_return_1h"] = bars1h["perp_close"].pct_change(fill_method=None)
    bars1h["spot_return_1h"] = bars1h["spot_close"].pct_change(fill_method=None)

    bars4h["ema20_4h"] = bars4h["perp_close"].ewm(span=20, adjust=False, min_periods=20).mean()
    bars4h["ema20_slope_12h"] = bars4h["ema20_4h"].diff(3)
    bars4h = bars4h[
        ["available_at", "perp_close", "ema20_4h", "ema20_slope_12h", "data_valid"]
    ].rename(columns={"perp_close": "perp_close_4h", "data_valid": "data_valid_4h"})
    one_hour_columns = [
        "available_at",
        "perp_close",
        "spot_close",
        "perp_weekly_vwap",
        "spot_weekly_vwap",
        "ema50_1h",
        "atr_1h",
        "adx_1h",
        "volatility_percentile_1h",
        "basis_bps_1h",
        "basis_change_1h",
        "funding_rate",
        "funding_z_1h",
        "perp_return_1h",
        "spot_return_1h",
        "data_valid",
        *[f"ema50_slope_{value}h" for value in (6, 12)],
        *[f"weekly_vwap_slope_{value}h" for value in (6, 12)],
    ]
    one_hour = bars1h[one_hour_columns].rename(
        columns={
            "perp_close": "perp_close_1h",
            "spot_close": "spot_close_1h",
            "data_valid": "data_valid_1h",
        }
    )
    features = pd.merge_asof(
        bars15.sort_values("available_at"),
        one_hour.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    features = pd.merge_asof(
        features.sort_values("available_at"),
        bars4h.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=4),
    )
    context = pd.read_parquet(CONTEXT_PATH).copy()
    context["context_available_at"] = pd.to_datetime(context["context_available_at"], utc=True)
    context_columns = ["context_available_at", "oi_change_1h", "context_coverage"]
    features = pd.merge_asof(
        features.sort_values("available_at"),
        context[context_columns].sort_values("context_available_at"),
        left_on="available_at",
        right_on="context_available_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    features["features_available_at"] = features[["available_at", "context_available_at"]].max(
        axis=1
    )
    features["coverage"] = (
        features["data_valid"]
        & features["data_valid_1h"].fillna(False)
        & features["data_valid_4h"].fillna(False)
        & features["context_coverage"].fillna(False)
        & features["features_available_at"].le(features["available_at"])
    )
    atomic_parquet(FEATURE_PATH, features)
    atomic_parquet(ROOT / "bars_5m.parquet", bars5)
    audit = {
        "protocol": PROTOCOL,
        "perpetual_rows": len(perp),
        "spot_rows": len(spot),
        "joined_minute_rows": len(minutes),
        "joined_start": minutes["timestamp"].min().isoformat(),
        "joined_end": minutes["timestamp"].max().isoformat(),
        "minute_coverage": float(len(minutes) / len(perp)),
        "feature_rows_15m": len(features),
        "causal_feature_coverage": float(features["coverage"].mean()),
        "spot_source": "Binance official public archive with SHA-256 checksums",
        "perpetual_source": inventory["path"],
        "dataset_sha256": {
            "perpetual": sha256(Path(inventory["path"])),
            "spot": sha256(SPOT_PATH),
            "orderflow": sha256(ORDERFLOW_PATH),
            "open_interest_context": sha256(CONTEXT_PATH),
        },
        "missing_execution_data": [
            "historical bid/ask",
            "historical L2",
            "historical aggregate trades (kline taker flow used)",
            "account latency/fills",
        ],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(DATA_REPORT_PATH, audit)
    return features, minutes, audit


def _regime_direction(row: Any, config: dict[str, Any]) -> int:
    if not bool(row.coverage):
        return 0
    ema_slope = float(getattr(row, f"ema50_slope_{config['ema_slope_lookback']}h"))
    weekly_slope = float(getattr(row, f"weekly_vwap_slope_{config['weekly_vwap_slope_lookback']}h"))
    common = (
        float(row.adx_1h) >= float(config["adx_threshold"])
        and float(row.volatility_percentile_1h) < float(config["shock_percentile"])
        and abs(float(row.basis_change_1h)) <= float(config["basis_change_abs_max_bps"])
        and abs(float(row.funding_z_1h)) <= 3.0
        and float(row.oi_change_1h) >= float(config["oi_change_1h_min"])
    )
    if not common:
        return 0
    long = (
        float(row.perp_close_1h) > float(row.perp_weekly_vwap)
        and float(row.spot_close_1h) > float(row.spot_weekly_vwap)
        and float(row.perp_close_1h) > float(row.ema50_1h)
        and ema_slope > 0
        and weekly_slope > 0
        and float(row.perp_close_4h) > float(row.ema20_4h)
        and float(row.ema20_slope_12h) > 0
    )
    short = (
        float(row.perp_close_1h) < float(row.perp_weekly_vwap)
        and float(row.spot_close_1h) < float(row.spot_weekly_vwap)
        and float(row.perp_close_1h) < float(row.ema50_1h)
        and ema_slope < 0
        and weekly_slope < 0
        and float(row.perp_close_4h) < float(row.ema20_4h)
        and float(row.ema20_slope_12h) < 0
    )
    return 1 if long and not short else -1 if short and not long else 0


def _breakout(row: Any, direction: int, config: dict[str, Any]) -> bool:
    lookback = int(config["donchian_bars"])
    perp_level = float(
        getattr(row, f"perp_high_{lookback}" if direction > 0 else f"perp_low_{lookback}")
    )
    spot_level = float(
        getattr(row, f"spot_high_{lookback}" if direction > 0 else f"spot_low_{lookback}")
    )
    price_break = (
        float(row.perp_close) > perp_level if direction > 0 else float(row.perp_close) < perp_level
    )
    spot_break = (
        float(row.spot_close) > spot_level if direction > 0 else float(row.spot_close) < spot_level
    )
    distance = abs(float(row.perp_close) - float(row.perp_daily_vwap)) / float(row.atr_15m)
    return bool(
        price_break
        and spot_break
        and float(row.relative_volume_15m) >= float(config["relative_volume_min"])
        and direction * float(row.perp_taker_imbalance) >= float(config["taker_imbalance_min"])
        and float(row.oi_change_1h) >= float(config["oi_change_1h_min"])
        and float(row.range_percentile_15m) < float(config["shock_percentile"])
        and distance <= float(config["maximum_breakout_vwap_distance_atr"])
    )


def _eligible_breakouts(
    features: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    ema_slope = features[f"ema50_slope_{config['ema_slope_lookback']}h"]
    weekly_slope = features[f"weekly_vwap_slope_{config['weekly_vwap_slope_lookback']}h"]
    common = (
        features["coverage"].astype(bool)
        & features["adx_1h"].ge(config["adx_threshold"])
        & features["volatility_percentile_1h"].lt(config["shock_percentile"])
        & features["basis_change_1h"].abs().le(config["basis_change_abs_max_bps"])
        & features["funding_z_1h"].abs().le(3.0)
        & features["oi_change_1h"].ge(config["oi_change_1h_min"])
    )
    long = (
        common
        & features["perp_close_1h"].gt(features["perp_weekly_vwap"])
        & features["spot_close_1h"].gt(features["spot_weekly_vwap"])
        & features["perp_close_1h"].gt(features["ema50_1h"])
        & ema_slope.gt(0)
        & weekly_slope.gt(0)
        & features["perp_close_4h"].gt(features["ema20_4h"])
        & features["ema20_slope_12h"].gt(0)
    )
    short = (
        common
        & features["perp_close_1h"].lt(features["perp_weekly_vwap"])
        & features["spot_close_1h"].lt(features["spot_weekly_vwap"])
        & features["perp_close_1h"].lt(features["ema50_1h"])
        & ema_slope.lt(0)
        & weekly_slope.lt(0)
        & features["perp_close_4h"].lt(features["ema20_4h"])
        & features["ema20_slope_12h"].lt(0)
    )
    directions = np.select([long, short], [1, -1], default=0).astype(np.int8)
    lookback = int(config["donchian_bars"])
    price_break = np.where(
        directions > 0,
        features["perp_close"].gt(features[f"perp_high_{lookback}"]),
        features["perp_close"].lt(features[f"perp_low_{lookback}"]),
    )
    spot_break = np.where(
        directions > 0,
        features["spot_close"].gt(features[f"spot_high_{lookback}"]),
        features["spot_close"].lt(features[f"spot_low_{lookback}"]),
    )
    distance = (features["perp_close"] - features["perp_daily_vwap"]).abs() / features[
        "atr_15m"
    ].replace(0, np.nan)
    eligible = (
        (directions != 0)
        & price_break
        & spot_break
        & features["relative_volume_15m"].ge(config["relative_volume_min"])
        & (directions * features["perp_taker_imbalance"]).ge(config["taker_imbalance_min"])
        & features["range_percentile_15m"].lt(config["shock_percentile"])
        & distance.le(config["maximum_breakout_vwap_distance_atr"])
    )
    return np.flatnonzero(np.asarray(eligible, dtype=bool)), directions


def _candidate_from_breakout(
    breakout: Any,
    direction: int,
    config: dict[str, Any],
    features: pd.DataFrame,
    bars5: pd.DataFrame,
    minutes: pd.DataFrame,
    feature_times: pd.DatetimeIndex,
    five_times: pd.DatetimeIndex,
    minute_times: pd.DatetimeIndex,
    regime_directions: np.ndarray,
) -> tuple[dict[str, Any] | None, str]:
    breakout_available = pd.Timestamp(cast(Any, breakout.available_at))
    impulse_start = pd.Timestamp(cast(Any, breakout.timestamp))
    five_start = int(five_times.searchsorted(breakout_available))
    five_end = int(
        five_times.searchsorted(
            breakout_available + pd.Timedelta(minutes=int(config["pullback_timeout_minutes"])),
            side="right",
        )
    )
    window = bars5.iloc[five_start:five_end].copy()
    if len(window) < 3 or not window["data_valid"].all():
        return None, "DATA_UNAVAILABLE"
    impulse_start_index = int(minute_times.searchsorted(impulse_start))
    impulse_end_index = int(minute_times.searchsorted(breakout_available))
    impulse_minutes = minutes.iloc[impulse_start_index:impulse_end_index]
    if len(impulse_minutes) != 15:
        return None, "DATA_UNAVAILABLE"
    anchored_quote = impulse_minutes["perp_quote_volume"].sum()
    anchored_volume = impulse_minutes["perp_volume"].sum()
    atr_value = float(breakout.atr_15m)
    pullback_row: Any | None = None
    anchored_vwap = np.nan
    prior_signed_flow = np.nan
    for index, raw_row in enumerate(window.itertuples(index=False)):
        row = cast(Any, raw_row)
        anchored_quote += float(row.perp_quote_volume)
        anchored_volume += float(row.perp_volume)
        anchored_vwap = anchored_quote / anchored_volume
        signed_flow = direction * float(row.perp_taker_imbalance)
        if index < 2:
            prior_signed_flow = signed_flow
            continue
        closes = window["perp_close"].iloc[index - 2 : index + 1].to_numpy(float)
        two_bar_pullback = (
            direction * (closes[-1] - closes[-2]) < 0 and direction * (closes[-2] - closes[-3]) < 0
        )
        zone_distance = min(
            abs(float(row.perp_close) - float(row.perp_daily_vwap)),
            abs(float(row.perp_close) - anchored_vwap),
        )
        zone = zone_distance <= float(config["pullback_zone_atr"]) * atr_value
        lighter_volume = float(row.perp_quote_volume) < float(breakout.perp_quote_volume) / 3
        slowing_flow = (
            signed_flow < 0 and np.isfinite(prior_signed_flow) and signed_flow > prior_signed_flow
        )
        structure = (
            float(row.perp_low) > float(breakout.perp_low) - atr_value
            if direction > 0
            else float(row.perp_high) < float(breakout.perp_high) + atr_value
        )
        feature_index = int(
            feature_times.searchsorted(pd.Timestamp(row.available_at), side="right") - 1
        )
        regime_held = (
            feature_index >= 0
            and _regime_direction(
                next(features.iloc[[feature_index]].itertuples(index=False)), config
            )
            == direction
        )
        if (
            two_bar_pullback
            and zone
            and lighter_volume
            and slowing_flow
            and structure
            and regime_held
        ):
            pullback_row = row
            break
        prior_signed_flow = signed_flow
    if pullback_row is None:
        return None, "PULLBACK_TOO_SHALLOW"

    pullback_available = pd.Timestamp(cast(Any, pullback_row.available_at))
    confirmation_start = int(minute_times.searchsorted(pullback_available))
    confirmation_end = min(
        confirmation_start + int(config["confirmation_timeout_minutes"]), len(minutes) - 1
    )
    confirmation_index: int | None = None
    for index in range(confirmation_start + 1, confirmation_end):
        previous, current = minutes.iloc[index - 1], minutes.iloc[index]
        current_flow = (
            2 * float(current["perp_taker_buy_quote"]) - float(current["perp_quote_volume"])
        ) / float(current["perp_quote_volume"])
        price_action = (
            float(current["perp_low"]) >= float(previous["perp_low"])
            and float(current["perp_close"]) > float(previous["perp_high"])
            if direction > 0
            else float(current["perp_high"]) <= float(previous["perp_high"])
            and float(current["perp_close"]) < float(previous["perp_low"])
        )
        spot_confirmation = (
            direction * (float(current["spot_close"]) - float(previous["spot_close"])) > 0
        )
        if price_action and direction * current_flow > 0 and spot_confirmation:
            confirmation_index = index
            break
    if confirmation_index is None or confirmation_index + 1 >= len(minutes):
        return None, "NO_FLOW_CONFIRMATION"
    entry_index = confirmation_index + 1
    entry = float(minutes.iloc[entry_index]["perp_open"])
    pullback_start_index = int(five_times.searchsorted(breakout_available))
    pullback_end_index = int(five_times.searchsorted(pullback_available))
    pullback_slice = bars5.iloc[pullback_start_index:pullback_end_index]
    if pullback_slice.empty:
        return None, "DATA_UNAVAILABLE"
    pullback_extreme = (
        float(pullback_slice["perp_low"].min())
        if direction > 0
        else float(pullback_slice["perp_high"].max())
    )
    anchored_stop = anchored_vwap - direction * 0.25 * atr_value
    stop = (
        min(pullback_extreme, anchored_stop)
        if direction > 0
        else max(pullback_extreme, anchored_stop)
    )
    risk = direction * (entry - stop)
    if risk <= 0 or risk > float(config["maximum_stop_atr"]) * atr_value:
        return None, "PULLBACK_TOO_DEEP"
    entry_timestamp = pd.Timestamp(minutes.iloc[entry_index]["timestamp"])
    confirmation_timestamp = pd.Timestamp(
        minutes.iloc[confirmation_index]["timestamp"]
    ) + pd.Timedelta(minutes=1)
    if entry_timestamp < confirmation_timestamp:
        raise RuntimeError("V24 non-causal entry")
    future_start = int(feature_times.searchsorted(entry_timestamp, side="right") - 1)
    future_end = int(feature_times.searchsorted(entry_timestamp + pd.Timedelta(hours=6)))
    regime_invalidation_at: Any = pd.NaT
    future_directions = regime_directions[max(0, future_start) : future_end]
    invalid = np.flatnonzero(future_directions != direction)
    if len(invalid):
        future_index = max(0, future_start) + int(invalid[0])
        regime_invalidation_at = pd.Timestamp(features.iloc[future_index]["available_at"])
    if pd.notna(regime_invalidation_at) and regime_invalidation_at <= entry_timestamp:
        return None, "NO_TREND_REGIME"
    candidate_key = (
        f"{config['config_id']}|{breakout_available.isoformat()}|{entry_timestamp.isoformat()}"
    )
    return (
        {
            "candidate_id": hashlib.sha256(candidate_key.encode()).hexdigest()[:24],
            "config_id": config["config_id"],
            "symbol": "BTCUSDT",
            "direction": direction,
            "regime_timestamp": breakout_available,
            "breakout_timestamp": breakout_available,
            "pullback_start": pd.Timestamp(window.iloc[0]["timestamp"]),
            "vwap_zone_reached": pullback_available,
            "confirmation_timestamp": confirmation_timestamp,
            "entry_timestamp": entry_timestamp,
            "entry_index": entry_index,
            "impulse_start_index": impulse_start_index,
            "entry_price": entry,
            "stop_price": stop,
            "risk_price": risk,
            "atr_15m": atr_value,
            "anchored_vwap": anchored_vwap,
            "daily_vwap": float(pullback_row.perp_daily_vwap),
            "features_available_at": confirmation_timestamp,
            "regime_invalidation_at": regime_invalidation_at,
            "adx_1h": float(breakout.adx_1h),
            "volatility_percentile_1h": float(breakout.volatility_percentile_1h),
            "relative_volume_15m": float(breakout.relative_volume_15m),
            "breakout_taker_imbalance": float(breakout.perp_taker_imbalance),
            "oi_change_1h": float(breakout.oi_change_1h),
            "basis_change_1h": float(breakout.basis_change_1h),
            "pullback_depth_atr": float(
                direction * (float(breakout.perp_close) - float(pullback_row.perp_close))
                / atr_value
            ),
            "pullback_duration_minutes": float(
                (pullback_available - breakout_available).total_seconds() / 60
            ),
            "decision": "ACCEPTED_BASE",
            "rejection_reason": "",
        },
        "",
    )


def build_candidates(
    features: pd.DataFrame, minutes: pd.DataFrame, *, resume: bool, smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rejection_path = ROOT / "candidate_rejections.parquet"
    if resume and CANDIDATE_PATH.exists() and rejection_path.exists():
        return pd.read_parquet(CANDIDATE_PATH), pd.read_parquet(rejection_path)
    bars5 = pd.read_parquet(ROOT / "bars_5m.parquet").copy()
    time5 = pd.to_datetime(bars5["timestamp"], utc=True)
    bars5["perp_daily_vwap"] = _group_vwap(bars5, "perp", time5.dt.floor("D"))
    feature_times = pd.DatetimeIndex(pd.to_datetime(features["available_at"], utc=True))
    five_times = pd.DatetimeIndex(pd.to_datetime(bars5["timestamp"], utc=True))
    minute_times = pd.DatetimeIndex(pd.to_datetime(minutes["timestamp"], utc=True))
    output: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    configs = strategy_configs()[:1] if smoke else strategy_configs()
    feature_rows = list(features.itertuples(index=False))
    for config_number, config in enumerate(configs, start=1):
        blocked_until = pd.Timestamp("1900", tz="UTC")
        eligible_indices, regime_directions = _eligible_breakouts(features, config)
        for row_index in eligible_indices:
            row = feature_rows[int(row_index)]
            direction = int(regime_directions[int(row_index)])
            signal = pd.Timestamp(cast(Any, row.available_at))
            if signal <= blocked_until:
                continue
            candidate, reason = _candidate_from_breakout(
                row,
                direction,
                config,
                features,
                bars5,
                minutes,
                feature_times,
                five_times,
                minute_times,
                regime_directions,
            )
            if candidate is None:
                rejections.append(
                    {
                        "config_id": config["config_id"],
                        "breakout_timestamp": signal,
                        "direction": direction,
                        "rejection_reason": reason,
                    }
                )
                continue
            output.append(candidate)
            blocked_until = pd.Timestamp(candidate["entry_timestamp"]) + pd.Timedelta(hours=1)
        _status(
            "candidates",
            f"Strategy configuration {config_number}/{len(configs)}",
            35 + 25 * config_number / len(configs),
        )
    candidates = pd.DataFrame(output)
    rejected = pd.DataFrame(rejections)
    if not smoke:
        atomic_parquet(CANDIDATE_PATH, candidates)
        atomic_parquet(rejection_path, rejected)
    return candidates, rejected


def _simulate_exit(
    candidate: Any, minutes: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any] | None:
    entry_index = int(candidate.entry_index)
    direction = int(candidate.direction)
    entry = float(candidate.entry_price)
    initial_stop = float(candidate.stop_price)
    risk = float(candidate.risk_price)
    if risk <= 0:
        return None
    tp1 = entry + direction * float(config["tp1_r"]) * risk
    tp2 = entry + direction * float(config["tp2_r"]) * risk
    partial = float(config["partial_tp1"])
    window = minutes.iloc[entry_index : entry_index + int(config["timeout_minutes"])].copy()
    if window.empty or not window["data_valid"].all():
        return None
    stop = initial_stop
    tp1_done = False
    realized = 0.0
    remaining = 1.0
    mfe = 0.0
    mae = 0.0
    best_price = entry
    exit_price = entry
    exit_reason = "timeout"
    exit_timestamp = pd.Timestamp(window.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=1)
    impulse_start_index = int(getattr(candidate, "impulse_start_index", entry_index))
    anchored_history = minutes.iloc[impulse_start_index:entry_index]
    anchored_quote = float(anchored_history["perp_quote_volume"].sum())
    anchored_volume = float(anchored_history["perp_volume"].sum())
    invalidation = pd.Timestamp(cast(Any, candidate.regime_invalidation_at))
    pending_exit: str | None = None
    vwap_loss_bars = flow_reversal_bars = spot_divergence_bars = 0
    block_quote = block_taker_buy = 0.0
    block_spot_open = block_perp_open = np.nan
    for offset, bar in enumerate(window.itertuples(index=False)):
        raw_bar = cast(Any, bar)
        timestamp = pd.Timestamp(raw_bar.timestamp)
        open_price = float(raw_bar.perp_open)
        low, high = float(raw_bar.perp_low), float(raw_bar.perp_high)
        gap_stop = open_price <= stop if direction > 0 else open_price >= stop
        if gap_stop:
            realized += remaining * direction * (open_price - entry) / risk
            exit_price, exit_reason, exit_timestamp = open_price, "stop_gap", timestamp
            remaining = 0.0
            break
        if pending_exit is not None or (pd.notna(invalidation) and invalidation <= timestamp):
            realized += remaining * direction * (open_price - entry) / risk
            exit_price = open_price
            exit_reason = pending_exit or "regime_invalidation"
            exit_timestamp = timestamp
            remaining = 0.0
            break
        favorable = high - entry if direction > 0 else entry - low
        adverse = entry - low if direction > 0 else high - entry
        mfe, mae = max(mfe, favorable / risk), max(mae, adverse / risk)
        best_price = max(best_price, high) if direction > 0 else min(best_price, low)
        stop_hit = low <= stop if direction > 0 else high >= stop
        tp1_hit = high >= tp1 if direction > 0 else low <= tp1
        tp2_hit = high >= tp2 if direction > 0 else low <= tp2
        if stop_hit:  # conservative when stop and target share the minute
            realized += remaining * direction * (stop - entry) / risk
            exit_price, exit_reason = stop, "stop"
            exit_timestamp = pd.Timestamp(cast(Any, bar.timestamp)) + pd.Timedelta(minutes=1)
            remaining = 0.0
            break
        if not tp1_done and tp1_hit:
            realized += partial * float(config["tp1_r"])
            remaining -= partial
            tp1_done = True
            stop = entry
        if tp1_done and tp2_hit:
            realized += remaining * float(config["tp2_r"])
            exit_price, exit_reason = tp2, "tp2"
            exit_timestamp = pd.Timestamp(cast(Any, bar.timestamp)) + pd.Timedelta(minutes=1)
            remaining = 0.0
            break
        if tp1_done:
            if config["trailing"] == "atr_1":
                proposed = best_price - direction * float(candidate.atr_15m)
            else:
                history = window.iloc[max(0, offset - 4) : offset + 1]
                proposed = (
                    float(history["perp_low"].min())
                    if direction > 0
                    else float(history["perp_high"].max())
                )
            stop = max(stop, proposed) if direction > 0 else min(stop, proposed)
        anchored_quote += float(raw_bar.perp_quote_volume)
        anchored_volume += float(raw_bar.perp_volume)
        anchored_vwap = anchored_quote / anchored_volume
        if not np.isfinite(block_spot_open):
            block_spot_open = float(raw_bar.spot_open)
            block_perp_open = open_price
        block_quote += float(raw_bar.perp_quote_volume)
        block_taker_buy += float(raw_bar.perp_taker_buy_quote)
        if (timestamp.minute + 1) % 5 == 0:
            close = float(raw_bar.perp_close)
            daily_vwap = float(raw_bar.perp_daily_vwap)
            lost_vwap = direction * (close - anchored_vwap) < 0 and direction * (
                close - daily_vwap
            ) < 0
            signed_flow = direction * (2 * block_taker_buy - block_quote) / block_quote
            spot_move = direction * (float(raw_bar.spot_close) - block_spot_open)
            perp_move = direction * (close - block_perp_open)
            vwap_loss_bars = vwap_loss_bars + 1 if lost_vwap else 0
            flow_reversal_bars = flow_reversal_bars + 1 if signed_flow <= -0.05 else 0
            spot_divergence_bars = (
                spot_divergence_bars + 1 if spot_move < 0 <= perp_move else 0
            )
            if vwap_loss_bars >= 2:
                pending_exit = "vwap_invalidation"
            elif flow_reversal_bars >= 2:
                pending_exit = "flow_invalidation"
            elif spot_divergence_bars >= 2:
                pending_exit = "spot_perp_divergence"
            block_quote = block_taker_buy = 0.0
            block_spot_open = block_perp_open = np.nan
    if remaining:
        exit_price = float(window.iloc[-1]["perp_close"])
        realized += remaining * direction * (exit_price - entry) / risk
    gross_bps = realized * risk / entry * 10_000
    cost_4_r = BASE_COST_BPS / (risk / entry * 10_000)
    cost_8_r = STRESS_COST_BPS / (risk / entry * 10_000)
    return {
        "exit_config_id": config["exit_config_id"],
        "tp1": tp1,
        "tp2": tp2,
        "timeout": int(config["timeout_minutes"]),
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return_r": realized,
        "net_return_r_4bps": realized - cost_4_r,
        "net_return_r_8bps": realized - cost_8_r,
        "gross_edge_bps": gross_bps,
        "fees_and_execution_bps": BASE_COST_BPS,
        "mfe_r": mfe,
        "mae_r": mae,
        "duration_minutes": float(
            (exit_timestamp - pd.Timestamp(candidate.entry_timestamp)).total_seconds() / 60
        ),
    }


def build_matrix(
    candidates: pd.DataFrame, minutes: pd.DataFrame, *, resume: bool, smoke: bool = False
) -> pd.DataFrame:
    if resume and MATRIX_PATH.exists():
        return pd.read_parquet(MATRIX_PATH)
    output: list[dict[str, Any]] = []
    exits = exit_configs()
    for number, candidate in enumerate(candidates.itertuples(index=False), start=1):
        for config in exits:
            outcome = _simulate_exit(candidate, minutes, config)
            if outcome is not None:
                output.append(cast(Any, candidate)._asdict() | outcome)
        if number % 250 == 0:
            _status(
                "base_matrix",
                f"Candidate {number}/{len(candidates)} x {len(exits)} exits",
                62 + 12 * number / max(len(candidates), 1),
            )
    matrix = pd.DataFrame(output)
    if not smoke:
        atomic_parquet(MATRIX_PATH, matrix)
    return matrix


def _default_exit_id() -> str:
    return str(
        next(
            config["exit_config_id"]
            for config in exit_configs()
            if config["tp1_r"] == 1.0
            and config["tp2_r"] == 2.0
            and config["partial_tp1"] == 0.5
            and config["trailing"] == "atr_1"
            and config["timeout_minutes"] == 240
        )
    )


def _metrics(rows: pd.DataFrame) -> dict[str, float]:
    base = return_metrics(rows, "net_return_r_4bps")
    base["max_drawdown"] = max(0.0, float(base["max_drawdown"]))
    values = rows.get("net_return_r_4bps", pd.Series(dtype=float)).to_numpy(float)
    gross = rows.get("gross_return_r", pd.Series(dtype=float)).to_numpy(float)
    return base | {
        "gross_expectancy_r": float(gross.mean()) if len(gross) else 0.0,
        "stress_expectancy_r": float(rows["net_return_r_8bps"].mean()) if len(rows) else 0.0,
        "mean_gross_edge_bps": float(rows["gross_edge_bps"].mean()) if len(rows) else 0.0,
        "median_r": float(np.median(values)) if len(values) else 0.0,
        "payoff_ratio": float(values[values > 0].mean() / abs(values[values < 0].mean()))
        if np.any(values > 0) and np.any(values < 0)
        else 0.0,
        "mfe_r": float(rows["mfe_r"].mean()) if len(rows) else 0.0,
        "mae_r": float(rows["mae_r"].mean()) if len(rows) else 0.0,
        "duration_minutes": float(rows["duration_minutes"].mean()) if len(rows) else 0.0,
        "turnover": float(len(rows)),
        "cost_r": float((rows["gross_return_r"] - rows["net_return_r_4bps"]).mean())
        if len(rows)
        else 0.0,
    }


def walk_forward(matrix: pd.DataFrame, *, smoke: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if matrix.empty:
        return matrix, {
            "selections": [],
            "metrics": {},
            "gates": {"nonempty": False},
            "gates_passed": False,
        }
    entry_times = pd.to_datetime(matrix["entry_timestamp"], utc=True)
    first = entry_times.min().floor("D") + pd.Timedelta(weeks=52)
    last = entry_times.max().floor("D")
    starts = list(pd.date_range(first, last - pd.Timedelta(weeks=16), freq="8W", tz="UTC"))
    if smoke:
        starts = starts[-1:]
    default_exit = _default_exit_id()
    output: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for fold, start in enumerate(starts, start=1):
        train = matrix.loc[
            entry_times.ge(start - pd.Timedelta(weeks=52)) & entry_times.lt(start)
        ].copy()
        train = train.loc[
            pd.to_datetime(train["exit_timestamp"], utc=True).lt(start - pd.Timedelta(hours=6))
        ]
        test_start, test_end = start + pd.Timedelta(weeks=8), start + pd.Timedelta(weeks=16)
        test = matrix.loc[entry_times.ge(test_start) & entry_times.lt(test_end)]
        strategy_scores = []
        central = train.loc[train["exit_config_id"].eq(default_exit)]
        for config_id, rows in central.groupby("config_id"):
            chosen = non_overlapping(rows.rename(columns={"entry_timestamp": "decision_timestamp"}))
            strategy_scores.append({"config_id": config_id, **_metrics(chosen)})
        if not strategy_scores:
            continue
        strategy_id = str(
            pd.DataFrame(strategy_scores)
            .sort_values(
                ["stress_expectancy_r", "profit_factor", "config_id"],
                ascending=[False, False, True],
            )
            .iloc[0]["config_id"]
        )
        exit_scores = []
        for exit_id, rows in train.loc[train["config_id"].eq(strategy_id)].groupby(
            "exit_config_id"
        ):
            chosen = non_overlapping(rows.rename(columns={"entry_timestamp": "decision_timestamp"}))
            exit_scores.append({"exit_config_id": exit_id, **_metrics(chosen)})
        exit_id = str(
            pd.DataFrame(exit_scores)
            .sort_values(
                ["stress_expectancy_r", "profit_factor", "exit_config_id"],
                ascending=[False, False, True],
            )
            .iloc[0]["exit_config_id"]
        )
        audited = test.loc[
            test["config_id"].eq(strategy_id) & test["exit_config_id"].eq(exit_id)
        ].copy()
        audited = non_overlapping(
            audited.rename(columns={"entry_timestamp": "decision_timestamp"})
        ).rename(columns={"decision_timestamp": "entry_timestamp"})
        audited["outer_fold"] = fold
        output.append(audited)
        selections.append(
            {
                "outer_fold": fold,
                "strategy_config_id": strategy_id,
                "exit_config_id": exit_id,
                "test": _metrics(audited),
            }
        )
        _status("base_oos", f"Fold {fold}/{len(starts)}", 76 + 18 * fold / len(starts))
    trades = pd.concat(output, ignore_index=True) if output else matrix.iloc[:0].copy()
    if not smoke:
        atomic_parquet(TRADE_PATH, trades)
    if trades.empty:
        return trades, {"selections": selections, "metrics": {}, "gates": {"nonempty": False}}
    result = _metrics(trades)
    fold_pnl = trades.groupby("outer_fold")["net_return_r_4bps"].sum()
    positive_total = fold_pnl.clip(lower=0).sum()
    maximum_share = (
        float(fold_pnl.clip(lower=0).max() / positive_total) if positive_total > 0 else 1.0
    )
    fold_ev = trades.groupby("outer_fold")["net_return_r_4bps"].mean()
    lower = moving_block_lower_bound(
        trades["net_return_r_4bps"].to_numpy(float),
        block_size=min(20, len(trades)),
        seed=20260805,
    )
    gates = {
        "trades_200": len(trades) >= 200,
        "expectancy_positive": result["expectancy_r"] > 0,
        "profit_factor_1_10": result["profit_factor"] >= 1.10,
        "positive_fold_half": float(fold_ev.gt(0).mean()) >= 0.5,
        "fold_concentration_35pct": maximum_share <= 0.35,
        "gross_edge_three_times_cost": result["mean_gross_edge_bps"] >= 3 * BASE_COST_BPS,
    }
    breakdown = {
        "long": _metrics(trades.loc[trades["direction"].eq(1)]),
        "short": _metrics(trades.loc[trades["direction"].eq(-1)]),
        "by_year": {
            str(year): _metrics(rows)
            for year, rows in trades.groupby(
                pd.to_datetime(trades["entry_timestamp"], utc=True).dt.year
            )
        },
    }
    trade_times = pd.to_datetime(trades["entry_timestamp"], utc=True)
    session = pd.cut(
        trade_times.dt.hour,
        bins=[-1, 7, 12, 20, 23],
        labels=["ASIA", "EUROPE", "US", "OTHER"],
    )
    volatility = pd.cut(
        trades["volatility_percentile_1h"],
        bins=[-np.inf, 50, 90, np.inf],
        labels=["LOW", "NORMAL", "HIGH"],
    )
    month_labels = trade_times.dt.strftime("%Y-%m")
    breakdown["by_month"] = {
        str(month): _metrics(rows) for month, rows in trades.groupby(month_labels)
    }
    breakdown["by_session_utc"] = {
        str(name): _metrics(rows) for name, rows in trades.groupby(session, observed=True)
    }
    breakdown["by_volatility"] = {
        str(name): _metrics(rows)
        for name, rows in trades.groupby(volatility, observed=True)
    }
    breakdown["by_exit_reason"] = trades["exit_reason"].value_counts().to_dict()
    monthly_ev = trades.groupby(month_labels)["net_return_r_4bps"].mean()
    deployment_gates = {
        "trades_300": len(trades) >= 300,
        "expectancy_4bps_positive": result["expectancy_r"] > 0,
        "expectancy_8bps_nonnegative": result["stress_expectancy_r"] >= 0,
        "profit_factor_1_20": result["profit_factor"] >= 1.20,
        "bootstrap_lcb_positive": lower > 0,
        "max_drawdown_10pct": result["max_drawdown"] <= 0.10,
        "majority_months_positive": float(monthly_ev.gt(0).mean()) > 0.5,
        "future_holdout_confirmed": False,
    }
    return trades, {
        "selections": selections,
        "metrics": result,
        "bootstrap_lcb": lower,
        "positive_fold_fraction": float(fold_ev.gt(0).mean()),
        "maximum_positive_fold_pnl_share": maximum_share,
        "breakdown": breakdown,
        "gates": gates,
        "gates_passed": all(gates.values()),
        "deployment_gates": deployment_gates,
        "deployment_gates_passed": all(deployment_gates.values()),
    }


def build_research_bundle(
    protocol: dict[str, Any], audit: dict[str, Any], *, data_end: pd.Timestamp
) -> None:
    selections = cast(list[dict[str, Any]], audit.get("selections", []))
    latest = (
        selections[-1]
        if selections
        else {
            "strategy_config_id": strategy_configs()[0]["config_id"],
            "exit_config_id": _default_exit_id(),
        }
    )
    RESEARCH_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_type": "RESEARCH_ONLY",
        "research_only": True,
        "real_capital_allowed": False,
        "auto_promotion": False,
        "protocol": protocol,
        "strategy_config_id": latest["strategy_config_id"],
        "exit_config_id": latest["exit_config_id"],
        "training_end_timestamp": data_end.isoformat(),
        "decision_reasons": [
            "NO_TREND_REGIME",
            "NO_BREAKOUT",
            "PULLBACK_TOO_DEEP",
            "PULLBACK_TOO_SHALLOW",
            "NO_FLOW_CONFIRMATION",
            "SPOT_PERP_DIVERGENCE",
            "VOLATILITY_SHOCK",
            "DATA_UNAVAILABLE",
            "POSITION_ALREADY_OPEN",
        ],
    }
    joblib.dump(payload, RESEARCH_BUNDLE_PATH)
    _atomic_json(
        RESEARCH_BUNDLE_ROOT / "manifest.json",
        {key: value for key, value in payload.items() if key not in {"protocol"}}
        | {"protocol_sha256": protocol["protocol_sha256"]},
    )


def load_research_bundle(mode: str) -> dict[str, Any]:
    if mode not in {"paper", "shadow"}:
        raise PermissionError("V24 RESEARCH_ONLY bundle refuses live capital")
    return cast(dict[str, Any], joblib.load(RESEARCH_BUNDLE_PATH))


def run(*, resume: bool, smoke: bool) -> dict[str, Any]:
    protocol = finalize_implementation(preregister())
    _status("data_audit", "Binance spot/perpetual causal synchronization", 1)
    features, minutes, data_audit = build_features(resume=resume)
    _status("candidates", "Deterministic trend-breakout VWAP pullbacks", 35)
    candidates, rejected = build_candidates(features, minutes, resume=resume, smoke=smoke)
    _status("base_matrix", "Structural stops, partial targets and trailing", 62)
    matrix = build_matrix(candidates, minutes, resume=resume, smoke=smoke)
    trades, audit = walk_forward(matrix, smoke=smoke)
    base_edge = bool(audit.get("gates_passed", False)) and not smoke
    if not smoke:
        build_research_bundle(
            protocol,
            audit,
            data_end=pd.to_datetime(minutes["timestamp"], utc=True).max(),
        )
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_sha256": sha256(Path(__file__)),
        "strategy": "TREND_VWAP_PULLBACK_CONTINUATION",
        "verdict": "BASE_EDGE_FOUND" if base_edge else "BASE_STRATEGY_NO_EDGE",
        "artifact_class": "RESEARCH_ONLY",
        "meta_model_authorized": base_edge,
        "ml_status": "AUTHORIZED_NOT_RUN" if base_edge else "STOP_ML",
        "deployable": False,
        "live_enabled": False,
        "cost_model": {
            "execution": "TAKER_ONLY",
            "base_round_trip_bps": BASE_COST_BPS,
            "stress_round_trip_bps": STRESS_COST_BPS,
            "component_breakdown": "unavailable without historical bid/ask; no invented split",
            "maker_fill_assumed": False,
        },
        "research_bundle": str(RESEARCH_BUNDLE_PATH),
        "events": len(candidates),
        "rejected_candidates": len(rejected),
        "matrix_rows": len(matrix),
        "oos_trades": len(trades),
        "data_audit": data_audit,
        "base_audit": audit,
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(BASE_REPORT_PATH, report)
    _status("complete", report["verdict"], 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V24 trend VWAP pullback continuation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        report = run(resume=args.resume, smoke=args.smoke)
    except Exception as error:
        _status("failed", f"{type(error).__name__}: {error}", 0)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
