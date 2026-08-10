from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from adaptive_bot.adapters.bitunix.market_data import _get_json
from adaptive_bot.config import AppConfig, MachineLearningConfig
from adaptive_bot.expert_policy import (
    _fit_side,
    _xgb_predict,
    moving_block_lower_bound,
    purged_expert_folds,
)
from adaptive_bot.expert_policy_v6 import (
    maker_ev_scenarios,
    reconstruct_passive_fill,
    select_v6_decisions,
    summarize_v6_groups,
)
from adaptive_bot.ml_research import FEATURE_COLUMNS, add_triple_barrier_labels, build_ml_features
from adaptive_bot.scientific_ml import gpu_preflight

PROTOCOL_VERSION = "adaptive_range_hybrid_v8"
ALPHA_EXCHANGES = ("binance", "okx", "bybit", "bitunix")
ALPHA_FAMILIES = ("mean_reversion", "momentum")
ALPHA_SIDES = ("long", "short")
ALPHA_DATA_ROOT = Path("data/ml/hybrid_v7")
ROOT = Path("data/ml/hybrid_v8")
STATUS_PATH = Path("data/reports/ml_hybrid_v8.status.json")
REPORT_PATH = Path("data/reports/ml_hybrid_v8.json")
PROTOCOL_PATH = Path("data/models/expert_policy/v8/protocol.json")
BUNDLE_PATH = Path("data/models/expert_policy/v8/bundle.joblib")
JsonFetcher = Callable[[str], object]


@dataclass(frozen=True)
class HybridAction:
    family: Literal["mean_reversion", "momentum"]
    side: Literal["long", "short"]
    timeframe_minutes: int = 15

    @property
    def expert_id(self) -> str:
        return f"v8-{self.family}-{self.side}-{self.timeframe_minutes}m"


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1], dtype=float
        )


def hybrid_actions() -> tuple[HybridAction, ...]:
    return tuple(
        HybridAction(cast(Any, family), cast(Any, side))
        for family in ALPHA_FAMILIES
        for side in ALPHA_SIDES
    )


def preregister_hybrid(app: AppConfig, *, now: datetime | None = None) -> dict[str, Any]:
    ml = _ml(app)
    created = (now or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "run_id": created.strftime("hybrid-v8-%Y%m%dT%H%M%SZ"),
        "created_at": created.isoformat(),
        "actions": [
            action.__dict__ | {"expert_id": action.expert_id} for action in hybrid_actions()
        ],
        "alpha": {
            "exchanges": list(ALPHA_EXCHANGES),
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe_minutes": 15,
            "features": list(FEATURE_COLUMNS),
            "outer_split_weeks": [52, 4, 4, 4],
            "models": ["ridge", "xgboost_cuda_ten_week_blocks"],
            "champion_rule": "ridge_default_xgboost_four_metric_audit",
            "exchange_feature": False,
            "training_weights": "capped_equal_exchange_symbol",
            "validation": "purged_temporal_leave_one_exchange_out",
            "uncertainty": "ten_week_block_models_plus_oos_residual_bootstrap",
            "source_manifest": str(ALPHA_DATA_ROOT / "alpha_manifest.json"),
            "source_manifest_sha256": (
                _file_sha256(ALPHA_DATA_ROOT / "alpha_manifest.json")
                if (ALPHA_DATA_ROOT / "alpha_manifest.json").exists()
                else None
            ),
        },
        "execution": {
            "exchange": "bitunix",
            "entry": "post_only_best_quote_no_reprice_60s",
            "labels": ["fill_60s", "fill_fraction", "adverse_selection_5s"],
            "minimum_distinct_days": 30,
            "quantity_depth_ratios": [0.01, 0.05, 0.1, 0.25],
            "split": "whole_utc_days_fit_calibration_audit",
            "calibration": "platt_below_60_days_then_isotonic",
            "gate": "compound_execution_ev_beats_best_baseline",
            "maker_fee_bps": ml.maker_fee_bps,
            "taker_fee_bps": ml.maker_taker_fee_bps,
        },
        "promotion": {
            "alpha_retrain": "weekly",
            "execution_challenger": "daily_00:15_UTC",
            "paper_promotion": "weekly_sunday_00:30_UTC",
            "automatic_live": False,
        },
        "config_sha256": _json_sha256(app.model_dump(mode="json")),
        "source_sha256": _training_source_sha256(),
        "holdout": {"status": "sealed", "opened": False},
    }
    if PROTOCOL_PATH.exists():
        existing = cast(dict[str, Any], json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
        immutable = ("protocol", "actions", "alpha", "execution", "promotion")
        if any(existing.get(key) != payload.get(key) for key in immutable):
            raise RuntimeError("V8 protocol is already frozen with a different specification")
        return existing
    _exclusive_json(PROTOCOL_PATH, payload)
    _retire_v7(created)
    return payload


def download_alpha_archives(
    app: AppConfig,
    start: datetime,
    end: datetime,
    *,
    fetch: JsonFetcher = _get_json,
) -> dict[str, Any]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("alpha archive boundaries must be timezone-aware and ordered")
    targets = (
        (exchange, symbol) for exchange in ALPHA_EXCHANGES for symbol in ("BTCUSDT", "ETHUSDT")
    )
    files: list[dict[str, Any]] = []
    for number, (exchange, symbol) in enumerate(targets, start=1):
        _status("alpha_download", f"{exchange} {symbol}", number / 8 * 20, exchange=exchange)
        target = ROOT / "alpha_raw" / f"exchange={exchange}" / f"symbol={symbol}" / "data.parquet"
        if target.exists():
            saved = pd.read_parquet(target, columns=["timestamp"])
            saved_times = pd.to_datetime(saved["timestamp"], utc=True)
            if (
                saved_times.max() - saved_times.min() >= pd.Timedelta(weeks=60)
                and saved_times.min() <= pd.Timestamp(start) + pd.Timedelta(days=1)
                and saved_times.max() >= pd.Timestamp(end) - pd.Timedelta(days=1)
            ):
                files.append(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "path": str(target),
                        "rows": len(saved),
                        "start": saved_times.min().isoformat(),
                        "end": saved_times.max().isoformat(),
                        "sha256": _file_sha256(target),
                    }
                )
                continue
        frame = (
            _download_bitunix_existing(app, symbol, start, end)
            if exchange == "bitunix"
            else _download_external(exchange, symbol, start, end, fetch)
        )
        if frame.empty:
            raise RuntimeError(f"official alpha archive unavailable: {exchange} {symbol}")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if timestamps.max() - timestamps.min() < pd.Timedelta(weeks=60):
            raise RuntimeError(f"alpha archive shorter than 60 weeks: {exchange} {symbol}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        files.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "path": str(target),
                "rows": len(frame),
                "start": frame["timestamp"].min().isoformat(),
                "end": frame["timestamp"].max().isoformat(),
                "sha256": _file_sha256(target),
            }
        )
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "requested_start": start.astimezone(UTC).isoformat(),
        "requested_end": end.astimezone(UTC).isoformat(),
        "files": files,
    }
    _atomic_json(ROOT / "alpha_manifest.json", manifest)
    return manifest


def _download_external(
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
    fetch: JsonFetcher,
) -> pd.DataFrame:
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if exchange == "binance":
        price = _binance_klines(fetch, symbol, start_ms, end_ms, mark=False)
        mark = _binance_klines(fetch, symbol, start_ms, end_ms, mark=True)
        funding = _binance_funding(fetch, symbol, start_ms, end_ms)
    elif exchange == "okx":
        instrument = symbol.replace("USDT", "-USDT-SWAP")
        price = _okx_klines(fetch, instrument, start_ms, end_ms, mark=False)
        mark = _okx_klines(fetch, instrument, start_ms, end_ms, mark=True)
        funding = _okx_funding(fetch, instrument, start_ms, end_ms)
    elif exchange == "bybit":
        price = _bybit_klines(fetch, symbol, start_ms, end_ms, mark=False)
        mark = _bybit_klines(fetch, symbol, start_ms, end_ms, mark=True)
        funding = _bybit_funding(fetch, symbol, start_ms, end_ms)
    else:
        raise ValueError(f"unsupported alpha exchange: {exchange}")
    return _normalize_alpha_market(price, mark, funding, exchange)


def _binance_klines(
    fetch: JsonFetcher, symbol: str, start_ms: int, end_ms: int, *, mark: bool
) -> pd.DataFrame:
    endpoint = "markPriceKlines" if mark else "klines"
    rows: list[list[Any]] = []
    cursor = start_ms
    pages = 0
    while cursor < end_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            }
        )
        batch = _fetch_retry(fetch, f"https://fapi.binance.com/fapi/v1/{endpoint}?{query}")
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[list[Any]], batch))
        pages += 1
        next_cursor = int(batch[-1][0]) + 60_000
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if pages % 50 == 0:
            _status(
                "alpha_download",
                f"binance {symbol} {endpoint}: {len(rows):,} rows",
                5,
            )
        time.sleep(0.03)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return pd.DataFrame([row[:6] for row in rows], columns=columns)


def _binance_funding(fetch: JsonFetcher, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        batch = _fetch_retry(fetch, f"https://fapi.binance.com/fapi/v1/fundingRate?{query}")
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[dict[str, Any]], batch))
        cursor = int(rows[-1]["fundingTime"]) + 1
        time.sleep(0.03)
    return pd.DataFrame(
        [(row["fundingTime"], row["fundingRate"]) for row in rows],
        columns=["timestamp", "funding_rate"],
    )


def _okx_klines(
    fetch: JsonFetcher, instrument: str, start_ms: int, end_ms: int, *, mark: bool
) -> pd.DataFrame:
    endpoint = "history-mark-price-candles" if mark else "history-candles"
    rows: list[list[Any]] = []
    cursor = end_ms
    pages = 0
    while cursor > start_ms:
        query = urlencode({"instId": instrument, "bar": "1m", "after": cursor, "limit": 100})
        payload = _fetch_retry(fetch, f"https://www.okx.com/api/v5/market/{endpoint}?{query}")
        batch = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[list[Any]], batch))
        pages += 1
        next_cursor = min(int(row[0]) for row in batch) - 1
        if next_cursor >= cursor:
            raise RuntimeError("OKX pagination did not retreat")
        cursor = next_cursor
        if pages % 50 == 0:
            _status(
                "alpha_download",
                f"okx {instrument} {endpoint}: {len(rows):,} rows",
                10,
            )
        time.sleep(0.06)
    selected = [row for row in rows if start_ms <= int(row[0]) < end_ms]
    values = [[*row[:5], 0 if mark else row[5]] for row in selected]
    return pd.DataFrame(values, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _okx_funding(fetch: JsonFetcher, instrument: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = end_ms
    while cursor > start_ms:
        query = urlencode({"instId": instrument, "after": cursor, "limit": 100})
        payload = _fetch_retry(
            fetch, f"https://www.okx.com/api/v5/public/funding-rate-history?{query}"
        )
        batch = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[dict[str, Any]], batch))
        cursor = min(int(row["fundingTime"]) for row in batch) - 1
        time.sleep(0.06)
    return pd.DataFrame(
        [
            (row["fundingTime"], row.get("realizedRate", row.get("fundingRate", 0)))
            for row in rows
            if start_ms <= int(row["fundingTime"]) < end_ms
        ],
        columns=["timestamp", "funding_rate"],
    )


def _bybit_klines(
    fetch: JsonFetcher, symbol: str, start_ms: int, end_ms: int, *, mark: bool
) -> pd.DataFrame:
    endpoint = "mark-price-kline" if mark else "kline"
    rows: list[list[Any]] = []
    cursor = end_ms
    pages = 0
    while cursor > start_ms:
        query = urlencode(
            {
                "category": "linear",
                "symbol": symbol,
                "interval": "1",
                "start": start_ms,
                "end": cursor,
                "limit": 1000,
            }
        )
        payload = _fetch_retry(fetch, f"https://api.bybit.com/v5/market/{endpoint}?{query}")
        result = payload.get("result") if isinstance(payload, dict) else None
        batch = result.get("list") if isinstance(result, dict) else None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[list[Any]], batch))
        pages += 1
        next_cursor = min(int(row[0]) for row in batch) - 1
        if next_cursor >= cursor:
            raise RuntimeError("Bybit pagination did not retreat")
        cursor = next_cursor
        if pages % 50 == 0:
            _status(
                "alpha_download",
                f"bybit {symbol} {endpoint}: {len(rows):,} rows",
                15,
            )
        time.sleep(0.03)
    selected = [row for row in rows if start_ms <= int(row[0]) < end_ms]
    values = [[*row[:5], 0 if mark else row[5]] for row in selected]
    return pd.DataFrame(values, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _bybit_funding(fetch: JsonFetcher, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = end_ms
    while cursor > start_ms:
        query = urlencode(
            {
                "category": "linear",
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": cursor,
                "limit": 200,
            }
        )
        payload = _fetch_retry(fetch, f"https://api.bybit.com/v5/market/funding/history?{query}")
        result = payload.get("result") if isinstance(payload, dict) else None
        batch = result.get("list") if isinstance(result, dict) else None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(cast(list[dict[str, Any]], batch))
        cursor = min(int(row["fundingRateTimestamp"]) for row in batch) - 1
        time.sleep(0.03)
    return pd.DataFrame(
        [
            (row["fundingRateTimestamp"], row["fundingRate"])
            for row in rows
            if start_ms <= int(row["fundingRateTimestamp"]) < end_ms
        ],
        columns=["timestamp", "funding_rate"],
    )


def _fetch_retry(fetch: JsonFetcher, url: str) -> object:
    for attempt in range(6):
        try:
            return fetch(url)
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise
            retry_after = float(error.headers.get("Retry-After", 0) or 0)
            time.sleep(max(retry_after, min(30.0, 0.5 * 2**attempt)))
        except (TimeoutError, URLError):
            time.sleep(min(30.0, 0.5 * 2**attempt))
    raise RuntimeError("official exchange REST retries exhausted")


def _normalize_alpha_market(
    price: pd.DataFrame, mark: pd.DataFrame, funding: pd.DataFrame, exchange: str
) -> pd.DataFrame:
    if price.empty or mark.empty:
        return pd.DataFrame()
    for frame in (price, mark, funding):
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(
                pd.to_numeric(frame["timestamp"], errors="raise"), unit="ms", utc=True
            )
    numeric = ("open", "high", "low", "close", "volume")
    for name in numeric:
        price[name] = pd.to_numeric(price[name], errors="coerce")
    for name in ("open", "high", "low", "close"):
        mark[name] = pd.to_numeric(mark[name], errors="coerce")
    mark = mark.rename(columns={name: f"mark_{name}" for name in ("open", "high", "low", "close")})
    data = price.merge(
        mark[["timestamp", "mark_open", "mark_high", "mark_low", "mark_close"]], on="timestamp"
    )
    data["funding_event_rate"] = np.nan
    data["funding_rate"] = np.nan
    data["funding_coverage"] = False
    if not funding.empty:
        funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
        events = funding.dropna().drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        data = pd.merge_asof(
            data.sort_values("timestamp"),
            events,
            on="timestamp",
            direction="backward",
            suffixes=("", "_observed"),
        )
        if "funding_rate_observed" in data:
            data["funding_rate"] = data.pop("funding_rate_observed")
            data["funding_coverage"] = data["funding_rate"].notna()
        event_map = events.set_index("timestamp")["funding_rate"]
        data["funding_event_rate"] = data["timestamp"].map(event_map)
        data.loc[data["funding_coverage"], "funding_event_rate"] = data.loc[
            data["funding_coverage"], "funding_event_rate"
        ].fillna(0.0)
    data["quote_volume"] = pd.to_numeric(data["volume"]) * pd.to_numeric(data["close"])
    data["data_valid"] = True
    data["round_trip_cost_bps"] = 0.0
    data["market_data_source"] = f"observed:{exchange}-official-rest"
    data["price_source"] = "observed"
    data["volume_source"] = "observed"
    data["funding_source"] = "observed"
    data["spread_source"] = "unavailable"
    data["cost_source"] = "alpha_gross_no_execution_cost"
    data["exchange"] = exchange
    return data.dropna(subset=list(numeric)).drop_duplicates("timestamp").sort_values("timestamp")


def _download_bitunix_existing(
    app: AppConfig, symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    ml = _ml(app)
    path = ml.archive_directory / f"bitunix_{symbol.lower()}_observed_1m.parquet"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_parquet(path)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    data = data.loc[timestamps.between(start, end, inclusive="left")].copy()
    data["round_trip_cost_bps"] = 0.0
    data["cost_source"] = "alpha_gross_no_execution_cost"
    data["exchange"] = "bitunix"
    return data


def build_alpha_matrix(app: AppConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_app = app.model_copy(
        update={
            "strategy": app.strategy.model_copy(
                update={"timeframe_minutes": 15, "crypto_vwap_window": 96}
            )
        }
    )
    paths = sorted((_alpha_archive_root() / "alpha_raw").glob("exchange=*/symbol=*/data.parquet"))
    for number, path in enumerate(paths, start=1):
        exchange = path.parts[-3].split("=", 1)[1]
        symbol = path.parts[-2].split("=", 1)[1]
        raw = pd.read_parquet(path)
        bars = _resample_alpha(raw, 15)
        bars["funding_rate"] = bars["funding_rate"].fillna(0.0)
        bars["funding_event_rate"] = bars["funding_event_rate"].fillna(0.0)
        features, _ = build_ml_features(bars, feature_app, alpha_gross=True)
        features["exchange"] = exchange
        features["symbol"] = symbol
        rows.extend(_alpha_family_rows(features, "mean_reversion"))
        rows.extend(_alpha_family_rows(features, "momentum"))
        _status("alpha_matrix", f"{exchange} {symbol}", 20 + 20 * number / max(1, len(paths)))
    matrix = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    target = ROOT / "alpha_matrix.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    matrix.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return matrix


def _resample_alpha(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    grouped = data.resample(f"{minutes}min", origin="epoch", closed="left", label="left")
    result = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "mark_open": "first",
            "mark_high": "max",
            "mark_low": "min",
            "mark_close": "last",
            "funding_rate": "last",
            "funding_event_rate": "sum",
            "data_valid": "all",
            "market_data_source": "last",
            "price_source": "last",
            "volume_source": "last",
            "funding_source": "last",
            "spread_source": "last",
            "cost_source": "last",
            "round_trip_cost_bps": "last",
        }
    )
    result["minute_count"] = grouped["close"].count()
    result["data_valid"] &= result["minute_count"].eq(minutes)
    return result.dropna(subset=["open", "close"]).reset_index()


def _alpha_family_rows(features: pd.DataFrame, family: str) -> list[pd.DataFrame]:
    labelled = (
        add_triple_barrier_labels(features, max_holding_bars=32, stop_atr=2.5, target_z=0.5)
        if family == "mean_reversion"
        else _momentum_labels(features, holding_bars=32, stop_atr=2.0, target_atr=2.0)
    )
    output: list[pd.DataFrame] = []
    for side in ALPHA_SIDES:
        z = labelled["distance_vwap_atr"]
        if family == "mean_reversion":
            eligible = z.le(-1.0) if side == "long" else z.ge(1.0)
            eligible &= labelled["adx"].lt(25)
        else:
            prior = (
                labelled["high"].shift(1).rolling(24).max()
                if side == "long"
                else labelled["low"].shift(1).rolling(24).min()
            )
            eligible = (
                labelled["close"].gt(prior) if side == "long" else labelled["close"].lt(prior)
            )
            eligible &= labelled["adx"].ge(20)
        eligible &= labelled["data_valid"].fillna(False).astype(bool)
        target = labelled.loc[eligible & labelled[f"net_return_{side}"].notna()].copy()
        if target.empty:
            continue
        exits = target[f"exit_index_{side}"].to_numpy(dtype=int)
        valid = (exits >= 0) & (exits < len(labelled))
        target = target.loc[valid].copy()
        exits = exits[valid]
        indexes = target.index.to_numpy(dtype=int)
        invalid_prefix = np.concatenate(
            ([0], np.cumsum(~labelled["data_valid"].fillna(False).to_numpy(dtype=bool)))
        )
        path_valid = invalid_prefix[exits + 1] == invalid_prefix[indexes + 1]
        target = target.loc[path_valid].copy()
        exits = exits[path_valid]
        target["signal_timestamp"] = pd.to_datetime(target["timestamp"], utc=True) + pd.Timedelta(
            minutes=15
        )
        target["exit_timestamp"] = pd.to_datetime(
            labelled.iloc[exits]["timestamp"].to_numpy(), utc=True
        ) + pd.Timedelta(minutes=15)
        target["family"] = family
        target["side"] = side
        target["expert_id"] = f"v8-{family}-{side}-15m"
        target["timeframe_minutes"] = 15
        raw_return = target[f"net_return_{side}"].to_numpy(dtype=float)
        if family == "mean_reversion":
            entry = labelled["open"].shift(-1).loc[target.index].to_numpy(dtype=float)
            risk = 2.5 * target["atr"].to_numpy(dtype=float) / entry
            raw_return = np.divide(
                raw_return,
                risk,
                out=np.full(len(target), np.nan),
                where=risk > 0,
            )
        target["net_return_r"] = raw_return
        target["net_return_r_2x"] = target["net_return_r"]
        target["execution_valid"] = target["data_valid"].astype(bool)
        output.append(target)
    return output


def _momentum_labels(
    features: pd.DataFrame, *, holding_bars: int, stop_atr: float, target_atr: float
) -> pd.DataFrame:
    result = features.copy()
    size = len(result)
    for side in ALPHA_SIDES:
        returns = np.full(size, np.nan)
        exits = np.full(size, -1, dtype=int)
        for index in range(size - holding_bars - 1):
            atr_value = float(cast(Any, result.at[index, "atr"]))
            if not math.isfinite(atr_value) or atr_value <= 0:
                continue
            entry = float(cast(Any, result.at[index + 1, "open"]))
            direction = 1 if side == "long" else -1
            stop = entry - direction * stop_atr * atr_value
            target = entry + direction * target_atr * atr_value
            exit_price = float(cast(Any, result.at[index + holding_bars, "close"]))
            exit_at = index + holding_bars
            for future in range(index + 1, index + holding_bars + 1):
                hit_stop = (
                    float(cast(Any, result.at[future, "low"])) <= stop
                    if side == "long"
                    else float(cast(Any, result.at[future, "high"])) >= stop
                )
                hit_target = (
                    float(cast(Any, result.at[future, "high"])) >= target
                    if side == "long"
                    else float(cast(Any, result.at[future, "low"])) <= target
                )
                if hit_stop or hit_target:
                    exit_price = stop if hit_stop else target
                    exit_at = future
                    break
            returns[index] = direction * (exit_price - entry) / (stop_atr * atr_value)
            exits[index] = exit_at
        result[f"net_return_{side}"] = returns
        result[f"exit_index_{side}"] = exits
    return result


def train_alpha(app: AppConfig, matrix: pd.DataFrame) -> dict[str, Any]:
    ml = _ml(app).model_copy(update={"status_path": STATUS_PATH, "model_trials_per_side": 4})
    if matrix.empty:
        return {"ready": False, "reason": "alpha_matrix_empty"}
    gpu = gpu_preflight(required=ml.gpu_required)
    matrix = (
        matrix.loc[
            matrix["data_valid"].fillna(False).astype(bool)
            & matrix["execution_valid"].fillna(False).astype(bool)
        ]
        .assign(timeframe_minutes=15)
        .sort_values(["signal_timestamp", "family", "side", "exchange"])
    )
    folds = purged_expert_folds(
        matrix, train_weeks=52, calibration_weeks=4, test_weeks=4, step_weeks=4
    )
    decisions: list[pd.DataFrame] = []
    champion_counts: dict[str, int] = {}
    for fold_number, fold in enumerate(folds, start=1):
        previous = matrix.iloc[np.concatenate((fold.train, fold.calibration))]
        future = matrix.iloc[fold.test]
        for held_out in ALPHA_EXCHANGES:
            fitting = previous.loc[previous["exchange"].ne(held_out)].copy()
            fitting["_training_weight"] = balanced_exchange_symbol_weights(fitting)
            test = future.loc[future["exchange"].eq(held_out)]
            fitted: dict[tuple[str, str], dict[str, Any]] = {}
            for family in ALPHA_FAMILIES:
                for side in ALPHA_SIDES:
                    _status(
                        "alpha_oos",
                        f"Fold {fold_number}/{len(folds)} held-out {held_out} {family} {side}",
                        40 + 40 * (fold_number - 1) / max(1, len(folds)),
                        backend=str(gpu),
                    )
                    group = fitting.loc[fitting["family"].eq(family) & fitting["side"].eq(side)]
                    fitted[(family, side)] = _fit_side(
                        group,
                        cast(Any, side),
                        ml,
                        objective_kind="decision_regret",
                        excluded_features=("exchange_code",),
                        benchmark_gate=True,
                        ensemble_models=10,
                        temporal_bootstrap=True,
                    )
                    champion = str(fitted[(family, side)].get("champion", "disabled"))
                    key = f"{held_out}:{family}:{side}:{champion}"
                    champion_counts[key] = champion_counts.get(key, 0) + 1
            predicted = _predict_alpha(test, fitted)
            for symbol, symbol_rows in predicted.groupby("symbol", sort=True):
                chosen = select_v6_decisions(symbol_rows)
                if not chosen.empty:
                    chosen["outer_fold"] = fold_number
                    chosen["exchange"] = held_out
                    chosen["symbol"] = symbol
                    decisions.append(chosen)
        _status(
            "alpha_oos",
            f"Fold {fold_number}/{len(folds)}",
            40 + 40 * fold_number / max(1, len(folds)),
            backend=str(gpu),
        )
    oos = pd.concat(decisions, ignore_index=True) if decisions else matrix.iloc[:0].copy()
    btc_oos = oos.loc[oos["symbol"].eq("BTCUSDT")] if not oos.empty else oos
    groups = summarize_v6_groups(btc_oos) if not btc_oos.empty else {}
    exchange_groups = {
        str(exchange): summarize_v6_groups(rows)
        for exchange, rows in btc_oos.groupby("exchange", sort=True)
    }
    symbol_groups = {
        f"{exchange}:{symbol}": summarize_v6_groups(rows)
        for (exchange, symbol), rows in oos.groupby(["exchange", "symbol"], sort=True)
    }
    final_training = matrix.loc[matrix["exchange"].ne("bitunix")].copy()
    final_training["_training_weight"] = balanced_exchange_symbol_weights(final_training)
    final_models = {
        (family, side): _fit_side(
            final_training.loc[
                final_training["family"].eq(family) & final_training["side"].eq(side)
            ],
            cast(Any, side),
            ml,
            objective_kind="decision_regret",
            excluded_features=("exchange_code",),
            benchmark_gate=True,
            ensemble_models=10,
            temporal_bootstrap=True,
        )
        for family in ALPHA_FAMILIES
        for side in ALPHA_SIDES
    }
    positive_exchanges = sum(
        any(group["enabled"] for group in exchange.values())
        for exchange in exchange_groups.values()
    )
    robust_groups = _robust_alpha_groups(exchange_groups)
    final_groups = {f"{family}:{side}": model for (family, side), model in final_models.items()}
    ready = bool(robust_groups) and any(
        final_groups.get(group, {}).get("enabled") for group in robust_groups
    )
    return {
        "ready": ready,
        "gpu": gpu,
        "models": final_models,
        "oos": oos,
        "groups": groups,
        "exchange_groups": exchange_groups,
        "symbol_groups": symbol_groups,
        "positive_exchanges": positive_exchanges,
        "robust_groups": robust_groups,
        "champion_counts": champion_counts,
        "final_model_audit": {
            f"{family}:{side}": {
                "champion": model.get("champion"),
                "candidate_audit": model.get("candidate_audit"),
            }
            for (family, side), model in final_models.items()
        },
    }


def balanced_exchange_symbol_weights(frame: pd.DataFrame, *, cap_ratio: float = 10.0) -> np.ndarray:
    required = {"exchange", "symbol", "signal_timestamp", "expert_id"}
    if missing := required - set(frame):
        raise ValueError(f"balanced weights missing columns: {sorted(missing)}")
    if frame.empty or cap_ratio < 1:
        raise ValueError("balanced weights require rows and cap_ratio >= 1")
    timestamp_count = frame.groupby(["exchange", "symbol", "signal_timestamp"])[
        "expert_id"
    ].transform("count")
    weights = 1.0 / timestamp_count.to_numpy(dtype=float)
    group = frame[["exchange", "symbol"]].astype(str).agg(":".join, axis=1)
    totals = pd.Series(weights).groupby(group.to_numpy()).transform("sum").to_numpy()
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    positive = weights[weights > 0]
    cap = float(np.median(positive)) * cap_ratio
    weights = np.minimum(weights, cap)
    return np.asarray(weights / weights.mean(), dtype=float)


def _robust_alpha_groups(exchange_groups: dict[str, dict[str, Any]]) -> list[str]:
    bitunix = exchange_groups.get("bitunix", {})
    return sorted(
        group
        for group, result in bitunix.items()
        if result.get("enabled")
        and sum(
            exchange.get(group, {}).get("enabled", False) for exchange in exchange_groups.values()
        )
        >= 3
    )


def _predict_alpha(
    rows: pd.DataFrame, models: dict[tuple[str, str], dict[str, Any]]
) -> pd.DataFrame:
    predicted: list[pd.DataFrame] = []
    for (family, side), fitted in models.items():
        if not fitted.get("enabled"):
            continue
        group = rows.loc[rows["family"].eq(family) & rows["side"].eq(side)].copy()
        if group.empty:
            continue
        values = (
            group[fitted["features"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=float)
        )
        ensemble = np.vstack([_xgb_predict(model, values) for model in fitted["models"]])
        group["ev_mean"] = fitted["calibrator"].predict(ensemble.mean(axis=0))
        lower = group["ev_mean"] + float(fitted["residual_lower"]) - 1.645 * ensemble.std(axis=0)
        group["lower_confidence_bound"] = np.minimum(group["ev_mean"], lower)
        predicted.append(group)
    return pd.concat(predicted, ignore_index=True) if predicted else rows.iloc[:0].copy()


QUANTITY_DEPTH_RATIOS = (0.01, 0.05, 0.10, 0.25)
EXECUTION_FEATURES = (
    "side_code",
    "spread_bps",
    "spread_ticks",
    "bid_depth_5",
    "ask_depth_5",
    "book_imbalance",
    "queue_ahead",
    "requested_depth_ratio",
    "microprice_mid_bps",
    "trade_imbalance_5s",
    "trade_imbalance_30s",
    "aggressive_volume_5s",
    "aggressive_volume_30s",
    "trade_volatility_30s",
    "best_price_velocity_bps_5s",
    "front_depth_change",
    "book_latency_ms",
    "hour_sin",
    "hour_cos",
)


def build_execution_dataset(
    directory: Path, *, sample_seconds: int = 60, maker_fee_bps: float = 2.0
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("btcusdt_*.parquet")):
        events = pd.read_parquet(path)
        if events.empty:
            continue
        required = {
            "event_type",
            "exchange_timestamp",
            "received_timestamp",
            "midpoint",
            "spread_bps",
            "bids_json",
            "asks_json",
            "price",
            "quantity",
            "aggressor_side",
        }
        if missing := required - set(events):
            raise ValueError(f"execution events missing columns: {sorted(missing)}")
        events["exchange_timestamp"] = pd.to_datetime(
            events["exchange_timestamp"], format="mixed", utc=True
        )
        events = events.sort_values("exchange_timestamp").reset_index(drop=True)
        timestamps = events["exchange_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        books = events.loc[events["event_type"].eq("book")].copy()
        books["bucket"] = books["exchange_timestamp"].dt.floor(f"{sample_seconds}s")
        for _, book in books.drop_duplicates("bucket").iterrows():
            timestamp = pd.Timestamp(book["exchange_timestamp"])
            left = int(np.searchsorted(timestamps, (timestamp - pd.Timedelta(seconds=30)).value))
            right = int(np.searchsorted(timestamps, (timestamp + pd.Timedelta(seconds=95)).value))
            context = events.iloc[left:right]
            for side in ALPHA_SIDES:
                levels = _book_levels(book, "bids" if side == "long" else "asks")
                if not levels:
                    continue
                queue_ahead = levels[0][1]
                for ratio in QUANTITY_DEPTH_RATIOS:
                    quantity = Decimal(str(queue_ahead * ratio))
                    outcome = reconstruct_passive_fill(
                        context,
                        signal_timestamp=timestamp - pd.Timedelta(milliseconds=250),
                        side=cast(Any, side),
                        quantity=quantity,
                    )
                    rows.append(
                        _execution_row(
                            book,
                            context,
                            side,
                            ratio,
                            queue_ahead,
                            outcome,
                            maker_fee_bps,
                            f"{path.stem}:{timestamp.isoformat()}",
                        )
                    )
    return pd.DataFrame(rows)


def _book_levels(book: pd.Series, side: str) -> list[tuple[float, float]]:
    raw = json.loads(str(book.get(f"{side}_json", "[]")))
    return [
        (float(level[0]), float(level[1]))
        for level in raw
        if len(level) >= 2 and float(level[0]) > 0 and float(level[1]) > 0
    ]


def _execution_row(
    book: pd.Series,
    context: pd.DataFrame,
    side: str,
    ratio: float,
    queue_ahead: float,
    outcome: Any,
    maker_fee_bps: float,
    snapshot_id: str,
) -> dict[str, Any]:
    bids, asks = _book_levels(book, "bids"), _book_levels(book, "asks")
    bid_depth = sum(level[1] for level in bids[:5])
    ask_depth = sum(level[1] for level in asks[:5])
    total = bid_depth + ask_depth
    timestamp = pd.Timestamp(book["exchange_timestamp"])
    bid, ask = bids[0][0], asks[0][0]
    midpoint = (bid + ask) / 2
    microprice = (ask * bids[0][1] + bid * asks[0][1]) / (bids[0][1] + asks[0][1])
    ticks = [
        abs(left[0] - right[0])
        for levels in (bids, asks)
        for left, right in pairwise(levels)
        if left[0] != right[0]
    ]
    tick = min(ticks) if ticks else ask - bid
    past = context.loc[context["exchange_timestamp"].lt(timestamp)]
    trades = past.loc[past["event_type"].eq("trade")].copy()
    recent_5 = trades.loc[trades["exchange_timestamp"].ge(timestamp - pd.Timedelta(seconds=5))]
    recent_30 = trades.loc[trades["exchange_timestamp"].ge(timestamp - pd.Timedelta(seconds=30))]
    previous_books = past.loc[past["event_type"].eq("book")]
    previous = previous_books.iloc[-1] if not previous_books.empty else None
    latency = pd.Timestamp(str(book["received_timestamp"])) - timestamp
    fill_fraction = float(outcome.filled_quantity / outcome.requested_quantity)
    adverse = outcome.adverse_selection_bps_5s
    return {
        "snapshot_id": snapshot_id,
        "signal_timestamp": timestamp,
        "utc_day": timestamp.date().isoformat(),
        "side": side,
        "side_code": 1.0 if side == "long" else -1.0,
        "spread_bps": float(book["spread_bps"]),
        "spread_ticks": (ask - bid) / tick if tick > 0 else 0.0,
        "spread_bucket": min(5, round((ask - bid) / tick)) if tick > 0 else 0,
        "bid_depth_5": bid_depth,
        "ask_depth_5": ask_depth,
        "book_imbalance": (bid_depth - ask_depth) / total if total else 0.0,
        "queue_ahead": queue_ahead,
        "requested_quantity": float(outcome.requested_quantity),
        "requested_depth_ratio": ratio,
        "microprice_mid_bps": (microprice - midpoint) / midpoint * 10_000,
        "trade_imbalance_5s": _trade_imbalance(recent_5),
        "trade_imbalance_30s": _trade_imbalance(recent_30),
        "aggressive_volume_5s": _trade_volume(recent_5),
        "aggressive_volume_30s": _trade_volume(recent_30),
        "trade_volatility_30s": _trade_volatility(recent_30),
        "best_price_velocity_bps_5s": _best_velocity(previous, midpoint, timestamp),
        "front_depth_change": _front_depth_change(previous, book, side, queue_ahead),
        "book_latency_ms": max(0.0, latency.total_seconds() * 1000),
        "hour": timestamp.hour,
        "hour_sin": math.sin(2 * math.pi * timestamp.hour / 24),
        "hour_cos": math.cos(2 * math.pi * timestamp.hour / 24),
        "fill_60s": float(outcome.status in {"full", "partial"}),
        "fill_fraction": fill_fraction,
        "adverse_selection_bps_5s": adverse,
        "execution_cost_bps": fill_fraction * (maker_fee_bps + float(adverse or 0.0)),
        "_training_weight": 1.0 / (2 * len(QUANTITY_DEPTH_RATIOS)),
    }


def _trade_volume(trades: pd.DataFrame) -> float:
    return float(pd.to_numeric(trades["quantity"], errors="coerce").fillna(0).sum())


def _trade_imbalance(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    volume = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0)
    signed = np.where(trades["aggressor_side"].astype(str).str.lower().eq("buy"), volume, -volume)
    total = float(volume.sum())
    return float(np.sum(signed) / total) if total else 0.0


def _trade_volatility(trades: pd.DataFrame) -> float:
    price = pd.to_numeric(trades["price"], errors="coerce").dropna()
    value = float(price.pct_change(fill_method=None).std()) if len(price) > 1 else 0.0
    return value if math.isfinite(value) else 0.0


def _best_velocity(previous: pd.Series | None, midpoint: float, timestamp: pd.Timestamp) -> float:
    if previous is None:
        return 0.0
    previous_mid = float(previous["midpoint"])
    seconds = (timestamp - pd.Timestamp(previous["exchange_timestamp"])).total_seconds()
    return (midpoint - previous_mid) / previous_mid * 10_000 / seconds if seconds > 0 else 0.0


def _front_depth_change(
    previous: pd.Series | None, current: pd.Series, side: str, current_depth: float
) -> float:
    if previous is None or current_depth <= 0:
        return 0.0
    old = _book_levels(previous, "bids" if side == "long" else "asks")
    new = _book_levels(current, "bids" if side == "long" else "asks")
    if not old or not new or old[0][0] != new[0][0]:
        return 0.0
    return (new[0][1] - old[0][1]) / current_depth


def train_execution(dataset: pd.DataFrame, *, minimum_days: int = 30) -> dict[str, Any]:
    features = list(EXECUTION_FEATURES)
    if dataset.empty:
        return {"ready": False, "reason": "execution_dataset_empty", "rows": 0}
    required = {
        *features,
        "signal_timestamp",
        "utc_day",
        "snapshot_id",
        "side",
        "hour",
        "spread_bucket",
        "fill_60s",
        "fill_fraction",
        "adverse_selection_bps_5s",
        "execution_cost_bps",
        "_training_weight",
    }
    if missing := required - set(dataset):
        raise ValueError(f"execution dataset missing columns: {sorted(missing)}")
    data = dataset.sort_values("signal_timestamp").dropna(subset=[*features, "fill_60s"])
    days = sorted(data["utc_day"].astype(str).unique())
    summary = {
        "rows": len(data),
        "distinct_snapshots": int(data["snapshot_id"].nunique()),
        "distinct_days": len(days),
    }
    if len(days) < minimum_days:
        return {
            "ready": False,
            "reason": "minimum_30_distinct_execution_days",
            "features": features,
            "metrics": summary,
        }
    fit_end, calibration_end = int(len(days) * 0.6), int(len(days) * 0.8)
    fit_days, calibration_days, audit_days = (
        days[:fit_end],
        days[fit_end:calibration_end],
        days[calibration_end:],
    )
    if min(map(len, (fit_days, calibration_days, audit_days))) < 5:
        return {"ready": False, "reason": "insufficient_day_blocks", "metrics": summary}
    fit = data.loc[data["utc_day"].isin(fit_days)]
    calibration = data.loc[data["utc_day"].isin(calibration_days)]
    audit = data.loc[data["utc_day"].isin(audit_days)]
    if fit["fill_60s"].nunique() < 2 or calibration["fill_60s"].nunique() < 2:
        return {"ready": False, "reason": "fill_class_missing_in_day_split", "metrics": summary}
    classifier = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300, random_state=20260803
    ).fit(fit[features], fit["fill_60s"], sample_weight=fit["_training_weight"])
    calibration_raw = classifier.predict_proba(calibration[features])[:, 1]
    if len(days) >= 60:
        calibrator: IsotonicRegression | PlattCalibrator = IsotonicRegression(
            out_of_bounds="clip"
        ).fit(
            calibration_raw,
            calibration["fill_60s"],
            sample_weight=calibration["_training_weight"],
        )
        calibration_method = "isotonic"
    else:
        platt = LogisticRegression(random_state=20260803).fit(
            calibration_raw.reshape(-1, 1),
            calibration["fill_60s"],
            sample_weight=calibration["_training_weight"],
        )
        calibrator = PlattCalibrator(platt)
        calibration_method = "platt"
    probability = calibrator.predict(classifier.predict_proba(audit[features])[:, 1])
    filled_fit = fit.loc[fit["fill_60s"].gt(0)]
    adverse_fit = fit.loc[fit["adverse_selection_bps_5s"].notna()]
    if len(filled_fit) < 200 or len(adverse_fit) < 200:
        return {"ready": False, "reason": "insufficient_conditional_fill_rows", "metrics": summary}
    fill_model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=20260803
    ).fit(
        filled_fit[features],
        filled_fit["fill_fraction"],
        sample_weight=filled_fit["_training_weight"],
    )
    adverse_model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=20260803
    ).fit(
        adverse_fit[features],
        adverse_fit["adverse_selection_bps_5s"],
        sample_weight=adverse_fit["_training_weight"],
    )
    fraction = np.clip(fill_model.predict(audit[features]), 0, 1)
    adverse = adverse_model.predict(audit[features])
    predicted_cost = probability * fraction * (2.0 + adverse)
    actual_cost = audit["execution_cost_bps"].to_numpy(dtype=float)
    baseline_specs = (
        (),
        ("side",),
        ("side", "hour"),
        ("side", "spread_bucket", "requested_depth_ratio"),
    )
    compound_baselines = {
        ":".join(keys) or "global": mean_absolute_error(
            actual_cost, _group_baseline(fit, audit, "execution_cost_bps", keys)
        )
        for keys in baseline_specs
    }
    probability_baselines = {
        ":".join(keys) or "global": brier_score_loss(
            audit["fill_60s"], _group_baseline(fit, audit, "fill_60s", keys)
        )
        for keys in baseline_specs
    }
    compound_mae = mean_absolute_error(actual_cost, predicted_cost)
    brier = brier_score_loss(audit["fill_60s"], probability)
    best_compound = min(compound_baselines.values())
    best_brier = min(probability_baselines.values())
    ready = compound_mae <= best_compound * 0.9 and brier <= best_brier
    return {
        "ready": ready,
        "reason": None if ready else "compound_execution_model_did_not_beat_baseline",
        "features": features,
        "classifier": classifier,
        "calibrator": calibrator,
        "fill_model": fill_model,
        "adverse_model": adverse_model,
        "metrics": {
            **summary,
            "fit_days": len(fit_days),
            "calibration_days": len(calibration_days),
            "audit_days": len(audit_days),
            "calibration_method": calibration_method,
            "brier": brier,
            "best_baseline_brier": best_brier,
            "compound_cost_mae": compound_mae,
            "best_baseline_compound_cost_mae": best_compound,
            "compound_baselines": compound_baselines,
            "probability_baselines": probability_baselines,
        },
    }


def _group_baseline(
    fit: pd.DataFrame, audit: pd.DataFrame, target: str, keys: tuple[str, ...]
) -> np.ndarray:
    default = float(fit[target].mean())
    if not keys:
        return np.full(len(audit), default)
    lookup = fit.groupby(list(keys), observed=True)[target].mean().to_dict()
    return np.asarray(
        [
            lookup.get(
                row[keys[0]] if len(keys) == 1 else tuple(row[key] for key in keys),
                default,
            )
            for _, row in audit.iterrows()
        ],
        dtype=float,
    )


def audit_hybrid_shadow(path: Path = ROOT / "end_to_end_shadow.parquet") -> dict[str, Any]:
    if not path.exists():
        return {"ready": False, "reason": "end_to_end_shadow_missing"}
    data = pd.read_parquet(path)
    required = {
        "signal_timestamp",
        "predicted_net_ev_r",
        "realized_net_return_r",
        "baseline_global_ev_r",
        "baseline_side_ev_r",
        "baseline_hour_ev_r",
        "baseline_spread_depth_ev_r",
    }
    if missing := required - set(data):
        raise ValueError(f"end-to-end audit missing columns: {sorted(missing)}")
    data["signal_timestamp"] = pd.to_datetime(data["signal_timestamp"], utc=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    distinct_days = int(data["signal_timestamp"].dt.date.nunique())
    if distinct_days < 30:
        return {
            "ready": False,
            "reason": "minimum_30_end_to_end_shadow_days",
            "rows": len(data),
            "distinct_days": distinct_days,
        }
    truth = data["realized_net_return_r"].to_numpy(dtype=float)
    model_mae = mean_absolute_error(truth, data["predicted_net_ev_r"])
    baselines = {
        column: mean_absolute_error(truth, data[column])
        for column in (
            "baseline_global_ev_r",
            "baseline_side_ev_r",
            "baseline_hour_ev_r",
            "baseline_spread_depth_ev_r",
        )
    }
    selected = data.loc[data["predicted_net_ev_r"].gt(0)].copy()
    if selected.empty:
        return {"ready": False, "reason": "end_to_end_always_flat", "rows": len(data)}
    daily = selected.groupby(selected["signal_timestamp"].dt.date)["realized_net_return_r"].mean()
    lower = moving_block_lower_bound(
        daily.to_numpy(dtype=float), block_size=min(7, len(daily)), seed=20260803
    )
    best_baseline = min(baselines.values())
    return {
        "ready": model_mae <= best_baseline * 0.9 and lower > 0,
        "rows": len(data),
        "distinct_days": distinct_days,
        "model_mae": model_mae,
        "best_baseline_mae": best_baseline,
        "baseline_mae": baselines,
        "selected_rows": len(selected),
        "selected_daily_lower_bound_r": lower,
    }


def combine_hybrid_ev(
    actions: pd.DataFrame,
    execution: dict[str, Any],
    config: MachineLearningConfig,
) -> pd.DataFrame:
    """Apply Bitunix fill and cost estimates to gross Alpha predictions."""
    execution_features = cast(list[str], execution.get("features", []))
    required = {
        "alpha_ev_mean",
        "alpha_lower_confidence_bound",
        "risk_distance_bps",
        "stop_probability",
        "expected_funding_r",
        *execution_features,
    }
    if missing := required - set(actions):
        raise ValueError(f"hybrid actions missing columns: {sorted(missing)}")
    if not execution.get("ready"):
        result = actions.copy()
        result["ev_mean"] = 0.0
        result["lower_confidence_bound"] = 0.0
        result["execution_valid"] = False
        return result
    values = actions[execution_features].to_numpy(dtype=float)
    raw_fill = execution["classifier"].predict_proba(values)[:, 1]
    probability = np.asarray(execution["calibrator"].predict(raw_fill), dtype=float)
    fraction = np.clip(execution["fill_model"].predict(values), 0, 1)
    adverse = np.maximum(0, execution["adverse_model"].predict(values))
    effective_fill = probability * fraction
    gross = actions["alpha_ev_mean"].to_numpy(dtype=float) - actions["expected_funding_r"].to_numpy(
        dtype=float
    )
    lower_gross = actions["alpha_lower_confidence_bound"].to_numpy(dtype=float) - actions[
        "expected_funding_r"
    ].to_numpy(dtype=float)
    arguments = (
        effective_fill,
        actions["stop_probability"].to_numpy(dtype=float),
        actions["risk_distance_bps"].to_numpy(dtype=float),
        adverse,
    )
    mean, stressed, _ = maker_ev_scenarios(
        gross,
        *arguments,
        maker_fee_bps=config.maker_fee_bps,
        taker_fee_bps=config.maker_taker_fee_bps,
    )
    lower, _, _ = maker_ev_scenarios(
        lower_gross,
        *arguments,
        maker_fee_bps=config.maker_fee_bps,
        taker_fee_bps=config.maker_taker_fee_bps,
    )
    result = actions.copy()
    result["fill_probability"] = probability
    result["expected_fill_fraction"] = fraction
    result["expected_adverse_selection_bps"] = adverse
    result["ev_mean"] = mean
    result["lower_confidence_bound"] = np.minimum(mean, lower)
    result["net_return_r_2x"] = stressed
    result["execution_valid"] = True
    return result


def run_hybrid_training(app: AppConfig, *, download: bool = False) -> dict[str, Any]:
    protocol = preregister_hybrid(app)
    if protocol["source_sha256"] != _training_source_sha256():
        raise RuntimeError("V8 source changed after preregistration")
    if protocol["config_sha256"] != _json_sha256(app.model_dump(mode="json")):
        raise RuntimeError("V8 configuration changed after preregistration")
    if download:
        end = datetime.now(UTC).replace(second=0, microsecond=0)
        start = end - pd.Timedelta(weeks=120)
        download_alpha_archives(app, start, end)
    manifest = _alpha_archive_root() / "alpha_manifest.json"
    if not manifest.exists():
        report = _waiting_report(protocol, "alpha_archives_missing")
        _write_report(report)
        return report
    _verify_alpha_manifest(manifest, protocol)
    matrix_path = ROOT / "alpha_matrix.parquet"
    matrix = pd.read_parquet(matrix_path) if matrix_path.exists() else build_alpha_matrix(app)
    alpha = train_alpha(app, matrix)
    execution_data = build_execution_dataset(_ml(app).v6_microstructure_directory)
    execution_path = ROOT / "execution_matrix.parquet"
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    execution_temporary = execution_path.with_suffix(".tmp.parquet")
    execution_data.to_parquet(execution_temporary, index=False)
    os.replace(execution_temporary, execution_path)
    execution = train_execution(execution_data)
    end_to_end = audit_hybrid_shadow()
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle_temporary = BUNDLE_PATH.with_suffix(".tmp.joblib")
    joblib.dump(
        {
            "protocol": PROTOCOL_VERSION,
            "run_id": protocol["run_id"],
            "alpha_models": alpha.get("models", {}) if alpha.get("ready") else None,
            "execution": (
                {key: value for key, value in execution.items() if key not in {"metrics"}}
                if execution.get("ready")
                else None
            ),
            "paper_only": True,
            "deployable": False,
        },
        bundle_temporary,
    )
    os.replace(bundle_temporary, BUNDLE_PATH)
    verdict = (
        "PAPER_CHALLENGER_READY"
        if alpha.get("ready") and execution.get("ready") and end_to_end.get("ready")
        else "ALPHA_SHADOW_READY"
        if alpha.get("ready")
        else "NO_ALPHA_POLICY"
    )
    report = {
        "protocol": PROTOCOL_VERSION,
        "run_id": protocol["run_id"],
        "verdict": verdict,
        "deployable": False,
        "paper_only": True,
        "alpha": {key: value for key, value in alpha.items() if key not in {"models", "oos"}},
        "execution": {
            key: value
            for key, value in execution.items()
            if key not in {"classifier", "calibrator", "fill_model", "adverse_model"}
        },
        "end_to_end": end_to_end,
        "bundle": str(BUNDLE_PATH),
        "holdout": protocol["holdout"],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _write_report(report)
    _status("hybrid_complete", report["verdict"], 100)
    return report


def _waiting_report(protocol: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "run_id": protocol["run_id"],
        "verdict": "WAITING_FOR_DATA",
        "reason": reason,
        "deployable": False,
        "paper_only": True,
        "holdout": protocol["holdout"],
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _write_report(report: dict[str, Any]) -> None:
    _atomic_json(REPORT_PATH, report)


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


def write_hybrid_failure(message: str) -> None:
    _status("failed", message, 0)


def _ml(app: AppConfig) -> MachineLearningConfig:
    if app.machine_learning is None or not app.machine_learning.enabled:
        raise ValueError("machine learning is disabled")
    return app.machine_learning


def _alpha_archive_root() -> Path:
    return ROOT if (ROOT / "alpha_manifest.json").exists() else ALPHA_DATA_ROOT


def _verify_alpha_manifest(path: Path, protocol: dict[str, Any]) -> None:
    expected = protocol["alpha"].get("source_manifest_sha256")
    if expected and _file_sha256(path) != expected:
        raise RuntimeError("frozen Alpha manifest hash mismatch")
    manifest = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for item in manifest.get("files", []):
        source = Path(str(item["path"]))
        if not source.exists() or _file_sha256(source) != item.get("sha256"):
            raise RuntimeError(f"frozen Alpha archive hash mismatch: {source}")


def _retire_v7(retired_at: datetime) -> None:
    if not PROTOCOL_PATH.parent.parent.joinpath("v7/protocol.json").exists():
        return
    marker = PROTOCOL_PATH.parent / "v7_diagnostic_retired.json"
    if not marker.exists():
        _exclusive_json(
            marker,
            {
                "status": "RETIRED_DIAGNOSTIC",
                "retired_at": retired_at.isoformat(),
                "replacement": PROTOCOL_VERSION,
            },
        )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "hybrid_policy.py",
        "expert_policy.py",
        "expert_policy_v6.py",
        "ml_research.py",
        "scientific_ml.py",
    ):
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str), encoding="utf-8"
    )
    os.replace(temporary, path)
