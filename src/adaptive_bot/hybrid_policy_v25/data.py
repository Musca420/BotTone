from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from adaptive_bot.hybrid_policy_v22.path_audit import atomic_parquet
from adaptive_bot.hybrid_policy_v22.protocol import sha256
from adaptive_bot.hybrid_policy_v25.protocol import ASSETS, ROOT, status
from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr

ARCHIVE_URL = "https://data.binance.vision/data"
RAW_PERP_ROOT = Path("data/ml/hybrid_v7/alpha_raw/exchange=binance")
BTC_CONTEXT_PATH = Path("data/ml/hybrid_v19/binance_context_5m.parquet")


def _archive_names(asset: str, market: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    start_naive, end_naive = start.tz_localize(None), end.tz_localize(None)
    current = end_naive.to_period("M")
    base = "spot" if market == "spot" else "futures/um"
    names = [
        f"{base}/monthly/klines/{asset}/1m/{asset}-1m-{month}.zip"
        for month in pd.period_range(start_naive, end_naive, freq="M")
        if month < current
    ]
    names.extend(
        f"{base}/daily/klines/{asset}/1m/{asset}-1m-{day:%Y-%m-%d}.zip"
        for day in pd.date_range(current.start_time, end_naive.floor("D"), freq="D")
    )
    return names


def _checked_archive(relative: str) -> Path:
    target = ROOT / "official_archives" / relative
    legacy = Path("data/ml/hybrid_v24/binance_spot_archives") / relative.removeprefix("spot/")
    if target.exists():
        return target
    if relative.startswith("spot/") and legacy.exists():
        return legacy
    url = f"{ARCHIVE_URL}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-range-research/25"})
    try:
        with urllib.request.urlopen(f"{url}.CHECKSUM", timeout=60) as response:
            expected = response.read().decode("ascii").split()[0].lower()
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"missing official Binance archive: {relative}") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"Binance checksum mismatch: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


def _parse_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise RuntimeError(f"unexpected Binance archive layout: {path}")
        frame = pd.read_csv(archive.open(members[0]), header=None)
    frame = frame.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].copy()
    frame.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_timestamp",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote",
    ]
    numeric_time = pd.to_numeric(frame["timestamp"], errors="coerce")
    microseconds = numeric_time.dropna().median() > 10**14
    unit: Literal["us", "ms"] = "us" if microseconds else "ms"
    frame["timestamp"] = pd.to_datetime(numeric_time, unit=unit, utc=True)
    close_time = pd.to_numeric(frame["close_timestamp"], errors="coerce")
    frame["source_timestamp"] = pd.to_datetime(close_time, unit=unit, utc=True)
    frame["available_at"] = frame["timestamp"] + pd.Timedelta(minutes=1)
    for column in frame.columns:
        if column not in {"timestamp", "source_timestamp", "available_at"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns="close_timestamp").dropna().reset_index(drop=True)


def _official_klines(
    asset: str, market: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    names = _archive_names(asset, market, start, end)
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for number, relative in enumerate(names, start=1):
        path = _checked_archive(relative)
        frames.append(_parse_archive(path))
        provenance.append(
            {"relative": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
        )
        status(
            "data_download",
            f"{asset} {market} archive {number}/{len(names)}",
            2 + 18 * number / max(len(names), 1),
            asset=asset,
            market=market,
        )
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp", keep="last")
    times = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[times.between(start.floor("min"), end.floor("min"))].reset_index(
        drop=True
    ), provenance


def _prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns if column != "timestamp"}
    )


def build_minutes(asset: str, *, resume: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = ROOT / f"asset={asset}" / "minutes.parquet"
    report_path = ROOT / f"asset={asset}" / "data_audit.json"
    if resume and output.exists() and report_path.exists():
        import json

        return pd.read_parquet(output), json.loads(report_path.read_text(encoding="utf-8"))
    context_path = RAW_PERP_ROOT / f"symbol={asset}" / "data.parquet"
    context = pd.read_parquet(context_path).copy()
    context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
    start, end = context["timestamp"].min(), context["timestamp"].max()
    perpetual, perp_archives = _official_klines(asset, "perpetual", start, end)
    spot, spot_archives = _official_klines(asset, "spot", start, end)
    context_columns = [
        "timestamp",
        "mark_open",
        "mark_high",
        "mark_low",
        "mark_close",
        "funding_event_rate",
        "funding_rate",
        "data_valid",
    ]
    perpetual = perpetual.merge(
        context[context_columns], on="timestamp", how="inner", validate="one_to_one"
    )
    minutes = _prefix(perpetual, "perp").merge(
        _prefix(spot, "spot"), on="timestamp", how="inner", validate="one_to_one"
    )
    minutes = minutes.sort_values("timestamp").reset_index(drop=True)
    mandatory = [
        column
        for column in minutes.columns
        if column.startswith(("perp_", "spot_"))
        and any(
            token in column
            for token in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "taker_buy",
                "available_at",
            )
        )
    ]
    minutes["max_input_available_at"] = minutes[["perp_available_at", "spot_available_at"]].max(
        axis=1
    )
    minutes["available_at"] = minutes["max_input_available_at"]
    minutes["freshness_seconds"] = (
        minutes["available_at"] - minutes["max_input_available_at"]
    ).dt.total_seconds()
    minutes["is_available"] = (
        minutes[mandatory].notna().all(axis=1)
        & minutes["perp_data_valid"].astype(bool)
        & minutes["max_input_available_at"].le(minutes["available_at"])
    )
    if asset == "BTCUSDT" and BTC_CONTEXT_PATH.exists():
        oi = pd.read_parquet(BTC_CONTEXT_PATH)[
            ["context_available_at", "oi_change_1h", "context_coverage"]
        ].copy()
        oi["context_available_at"] = pd.to_datetime(oi["context_available_at"], utc=True)
        minutes = pd.merge_asof(
            minutes.sort_values("available_at"),
            oi.sort_values("context_available_at"),
            left_on="available_at",
            right_on="context_available_at",
            direction="backward",
            tolerance=pd.Timedelta(minutes=15),
        )
    else:
        minutes["context_available_at"] = pd.NaT
        minutes["oi_change_1h"] = np.nan
        minutes["context_coverage"] = False
    atomic_parquet(output, minutes)
    audit = {
        "asset": asset,
        "rows": len(minutes),
        "start": minutes["timestamp"].min().isoformat(),
        "end": minutes["timestamp"].max().isoformat(),
        "mandatory_coverage": float(minutes["is_available"].mean()),
        "mark_coverage": float(minutes["perp_mark_close"].notna().mean()),
        "funding_coverage": float(minutes["perp_funding_rate"].notna().mean()),
        "open_interest_optional_coverage": float(minutes["context_coverage"].fillna(False).mean()),
        "perpetual_archives": perp_archives,
        "spot_archives": spot_archives,
        "source_sha256": {"perpetual_context": sha256(context_path)},
    }
    from adaptive_bot.hybrid_policy_v25.protocol import atomic_json

    atomic_json(report_path, audit)
    return minutes, audit


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
            (f"{prefix}_taker_buy_volume", "sum"),
            (f"{prefix}_taker_buy_quote", "sum"),
        )
    }
    aggregations |= {
        "perp_mark_close": "last",
        "perp_funding_rate": "last",
        "perp_funding_event_rate": "sum",
        "oi_change_1h": "last",
        "context_coverage": "last",
    }
    result = grouped.agg(aggregations)  # type: ignore[arg-type]
    result["minute_count"] = grouped["perp_close"].count()
    result["is_available"] = result["minute_count"].eq(interval) & result[
        [column for column in result if column not in {"oi_change_1h", "context_coverage"}]
    ].notna().all(axis=1)
    result_index = pd.DatetimeIndex(result.index)
    result["source_timestamp"] = (
        result_index + pd.Timedelta(minutes=interval) - pd.Timedelta(milliseconds=1)
    )
    result["available_at"] = result_index + pd.Timedelta(minutes=interval)
    result["max_input_available_at"] = result["available_at"]
    result["freshness_seconds"] = 0.0
    return result.reset_index()


def _vwap(frame: pd.DataFrame, prefix: str, groups: pd.Series) -> pd.Series:
    return frame[f"{prefix}_quote_volume"].groupby(groups).cumsum() / frame[
        f"{prefix}_volume"
    ].groupby(groups).cumsum().replace(0, np.nan)


def build_features(
    asset: str, minutes: pd.DataFrame, *, resume: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = ROOT / f"asset={asset}" / "features_15m.parquet"
    bars5_path = ROOT / f"asset={asset}" / "bars_5m.parquet"
    if resume and feature_path.exists() and bars5_path.exists():
        cached = pd.read_parquet(feature_path)
        if {"oi_change_1h", "context_coverage"}.issubset(cached.columns):
            return cached, pd.read_parquet(bars5_path)
    bars15, bars5, bars1h, bars4h = (_bars(minutes, value) for value in (15, 5, 60, 240))
    for frame in (bars15, bars5, bars1h, bars4h):
        for prefix in ("perp", "spot"):
            quote = frame[f"{prefix}_quote_volume"]
            buy = frame[f"{prefix}_taker_buy_quote"]
            frame[f"{prefix}_taker_imbalance"] = (2 * buy - quote) / quote.replace(0, np.nan)
    time15 = pd.to_datetime(bars15["timestamp"], utc=True)
    day15, week15 = (
        time15.dt.floor("D"),
        time15.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC"),
    )
    for prefix in ("perp", "spot"):
        bars15[f"{prefix}_daily_vwap"] = _vwap(bars15, prefix, day15)
        bars15[f"{prefix}_weekly_vwap"] = _vwap(bars15, prefix, week15)
        bars15[f"{prefix}_rolling_vwap_24h"] = bars15[f"{prefix}_quote_volume"].rolling(
            96, min_periods=96
        ).sum() / bars15[f"{prefix}_volume"].rolling(96, min_periods=96).sum().replace(0, np.nan)
        average = bars15[f"{prefix}_quote_volume"] / bars15[f"{prefix}_volume"].replace(0, np.nan)
        square = (average.pow(2) * bars15[f"{prefix}_volume"]).groupby(day15).cumsum()
        cumulative_volume = bars15[f"{prefix}_volume"].groupby(day15).cumsum()
        deviation = np.sqrt(
            (
                square / cumulative_volume.replace(0, np.nan)
                - bars15[f"{prefix}_daily_vwap"].pow(2)
            ).clip(lower=0)
        )
        bars15[f"{prefix}_vwap_deviation"] = deviation
    time5 = pd.to_datetime(bars5["timestamp"], utc=True)
    day5 = time5.dt.floor("D")
    for prefix in ("perp", "spot"):
        bars5[f"{prefix}_daily_vwap"] = _vwap(bars5, prefix, day5)
    bars15["atr_15m"] = atr(bars15["perp_high"], bars15["perp_low"], bars15["perp_close"], 14)
    bars15["return_15m"] = bars15["perp_close"].pct_change(fill_method=None)
    bars15["return_30m"] = bars15["perp_close"].pct_change(2, fill_method=None)
    bars15["return_60m"] = bars15["perp_close"].pct_change(4, fill_method=None)
    bars15["relative_volume"] = bars15["perp_quote_volume"] / bars15["perp_quote_volume"].shift(
        1
    ).rolling(96, min_periods=96).median().replace(0, np.nan)
    bars15["trade_intensity"] = bars15["perp_trade_count"] / bars15["perp_trade_count"].shift(
        1
    ).rolling(96, min_periods=96).median().replace(0, np.nan)
    bars15["donchian_high"] = bars15["perp_high"].shift(1).rolling(20, min_periods=20).max()
    bars15["donchian_low"] = bars15["perp_low"].shift(1).rolling(20, min_periods=20).min()
    bars15["range_atr"] = (bars15["perp_high"] - bars15["perp_low"]) / bars15["atr_15m"].replace(
        0, np.nan
    )
    bars15["impulse_efficiency"] = bars15["return_60m"].abs() / bars15["return_15m"].abs().rolling(
        4, min_periods=4
    ).sum().replace(0, np.nan)

    time1h = pd.to_datetime(bars1h["timestamp"], utc=True)
    week1h = time1h.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    bars1h["perp_weekly_vwap_1h"] = _vwap(bars1h, "perp", week1h)
    bars1h["spot_weekly_vwap_1h"] = _vwap(bars1h, "spot", week1h)
    bars1h["ema50_1h"] = bars1h["perp_close"].ewm(span=50, adjust=False, min_periods=50).mean()
    bars1h["atr_1h"] = atr(bars1h["perp_high"], bars1h["perp_low"], bars1h["perp_close"], 14)
    bars1h["adx_1h"] = adx(bars1h["perp_high"], bars1h["perp_low"], bars1h["perp_close"], 14)["adx"]
    bars1h["ema_slope_atr"] = (bars1h["ema50_1h"] - bars1h["ema50_1h"].shift(6)) / bars1h[
        "atr_1h"
    ].replace(0, np.nan)
    bars1h["weekly_vwap_slope_atr"] = (
        bars1h["perp_weekly_vwap_1h"] - bars1h["perp_weekly_vwap_1h"].shift(6)
    ) / bars1h["atr_1h"].replace(0, np.nan)
    bars1h["basis_bps"] = (
        (bars1h["perp_close"] - bars1h["spot_close"]) / bars1h["spot_close"] * 10_000
    )
    bars1h["basis_change_bps"] = bars1h["basis_bps"].diff()
    funding_window = bars1h["perp_funding_rate"].shift(1).rolling(720, min_periods=240)
    funding_std = funding_window.std(ddof=0)
    bars1h["funding_z"] = (
        (bars1h["perp_funding_rate"] - funding_window.mean()) / funding_std.replace(0, np.nan)
    ).where(funding_std.ne(0), 0.0)
    bars1h["volatility_percentile"] = (
        bars1h["atr_1h"].shift(1).rolling(2160, min_periods=720).rank(pct=True) * 100
    )

    bars4h["ema20_4h"] = bars4h["perp_close"].ewm(span=20, adjust=False, min_periods=20).mean()
    bars4h["trend_4h"] = (bars4h["perp_close"] - bars4h["ema20_4h"]) / bars4h["perp_close"]
    one_hour = bars1h[
        [
            "available_at",
            "perp_close",
            "spot_close",
            "perp_weekly_vwap_1h",
            "spot_weekly_vwap_1h",
            "ema50_1h",
            "adx_1h",
            "ema_slope_atr",
            "weekly_vwap_slope_atr",
            "basis_bps",
            "basis_change_bps",
            "funding_z",
            "volatility_percentile",
            "oi_change_1h",
            "context_coverage",
            "is_available",
        ]
    ].rename(
        columns={
            "perp_close": "perp_close_1h",
            "spot_close": "spot_close_1h",
            "is_available": "is_available_1h",
        }
    )
    four_hour = bars4h[["available_at", "trend_4h", "is_available"]].rename(
        columns={"is_available": "is_available_4h"}
    )
    bars15 = bars15.drop(columns=["oi_change_1h", "context_coverage"])
    features = pd.merge_asof(
        bars15.sort_values("available_at"),
        one_hour.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    features = pd.merge_asof(
        features.sort_values("available_at"),
        four_hour.sort_values("available_at"),
        on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=4),
    )
    features["max_input_available_at"] = features["available_at"]
    features["freshness_seconds"] = 0.0
    features["is_available"] = (
        features["is_available"].astype(bool)
        & features["is_available_1h"].fillna(False)
        & features["is_available_4h"].fillna(False)
        & features["max_input_available_at"].le(features["available_at"])
    )
    atomic_parquet(feature_path, features)
    atomic_parquet(bars5_path, bars5)
    return features, bars5


def build_all(
    *, resume: bool
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], dict[str, Any]]:
    output: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    audits: dict[str, Any] = {}
    for number, asset in enumerate(ASSETS, start=1):
        status("data_audit", f"{asset} official spot/perpetual", 5 + 20 * (number - 1))
        minutes, audit = build_minutes(asset, resume=resume)
        features, bars5 = build_features(asset, minutes, resume=resume)
        output[asset] = (minutes, features, bars5)
        audits[asset] = audit | {
            "feature_rows": len(features),
            "feature_coverage": float(features["is_available"].mean()),
            "future_feature_violations": int(
                (features["max_input_available_at"] > features["available_at"]).sum()
            ),
        }
    return output, audits
