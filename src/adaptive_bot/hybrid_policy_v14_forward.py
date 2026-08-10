from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from adaptive_bot.adapters.bitunix.market_data import _get_json
from adaptive_bot.config import AppConfig
from adaptive_bot.hybrid_policy import _download_external
from adaptive_bot.hybrid_policy_v11 import EXCHANGES, build_feature_frame, cross_exchange_features
from adaptive_bot.hybrid_policy_v14 import (
    ORDERFLOW_PATH,
    _archive_names,
    _download_checked,
    _parse_binance_kline_zip,
    _sha256,
    build_orderflow_context,
    first_reentry_events,
)
from adaptive_bot.ml_research import _download_klines

FORWARD_MARK_PATH = Path("data/raw/bitunix_btcusdt_mark_futures_1m_forward.jsonl")
FORWARD_MICROSTRUCTURE_ROOT = Path("data/raw/bitunix_microstructure")
FORWARD_MINUTES_PATH = Path("data/ml/hybrid_v14/forward/bitunix_observed_1m.parquet")
FORWARD_ALPHA_VIEW_PATH = Path("data/ml/hybrid_v14/forward/bitunix_alpha_feature_view.parquet")
FORWARD_WARMUP_PATH = Path("data/ml/hybrid_v14/forward/bitunix_official_warmup_1m.parquet")
FORWARD_WARMUP_META_PATH = FORWARD_WARMUP_PATH.with_suffix(".json")
FORWARD_EXTERNAL_ROOT = Path("data/ml/hybrid_v14/forward/external")
FORWARD_ORDERFLOW_PATH = Path("data/ml/hybrid_v14/forward/binance_orderflow_1m.parquet")
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
JsonFetcher = Callable[[str], object]


def download_bitunix_warmup(
    start: datetime,
    end: datetime,
    *,
    fetch: Any = _get_json,
    output_path: Path = FORWARD_WARMUP_PATH,
    metadata_path: Path = FORWARD_WARMUP_META_PATH,
) -> pd.DataFrame:
    """Download past observations for feature warm-up, never for forward outcomes."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Bitunix warm-up boundaries must be timezone-aware and ordered")
    last = _download_warmup_klines(fetch, start, end, "LAST_PRICE")
    mark = _download_warmup_klines(fetch, start, end, "MARK_PRICE").rename(
        columns={name: f"mark_{name}" for name in ("open", "high", "low", "close")}
    )
    if last.empty or mark.empty:
        raise RuntimeError("Bitunix official warm-up candles are unavailable")
    result = last.merge(
        mark[["timestamp", "mark_open", "mark_high", "mark_low", "mark_close"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp")
    result["raw_high"] = result["high"]
    result["raw_low"] = result["low"]
    result["raw_mark_high"] = result["mark_high"]
    result["raw_mark_low"] = result["mark_low"]
    result["high"] = result[["open", "high", "close"]].max(axis=1)
    result["low"] = result[["open", "low", "close"]].min(axis=1)
    result["mark_high"] = result[["mark_open", "mark_high", "mark_close"]].max(axis=1)
    result["mark_low"] = result[["mark_open", "mark_low", "mark_close"]].min(axis=1)
    times = pd.to_datetime(result["timestamp"], utc=True)
    result["data_valid"] = times.diff().eq(pd.Timedelta(minutes=1))
    if len(result):
        result.loc[result.index[0], "data_valid"] = True
    result["funding_rate"] = np.nan
    result["funding_event_rate"] = np.nan
    result["funding_coverage"] = False
    result["round_trip_cost_bps"] = 0.0
    result["market_data_source"] = "observed:bitunix-official-rest-warmup"
    result["price_source"] = "observed"
    result["volume_source"] = "observed"
    result["funding_source"] = "unavailable"
    result["spread_source"] = "unavailable"
    result["cost_source"] = "alpha_gross_no_execution_cost"
    result["exchange"] = "bitunix"
    available_at = datetime.now(UTC)
    result["available_at"] = available_at
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    metadata_temporary = metadata_path.with_suffix(".tmp")
    metadata_temporary.write_text(
        json.dumps(
            {
                "purpose": "FEATURE_WARMUP_ONLY_NOT_FORWARD_EVIDENCE",
                "available_at": available_at.isoformat(),
                "start": times.min().isoformat(),
                "end": times.max().isoformat(),
                "rows": len(result),
                "sha256": _sha256(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    return result.reset_index(drop=True)


def _download_warmup_klines(
    fetch: Any,
    start: datetime,
    end: datetime,
    price_type: Literal["LAST_PRICE", "MARK_PRICE"],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        boundary = min(cursor + pd.Timedelta(minutes=180).to_pytimedelta(), end)
        frames.append(_download_klines(fetch, cursor, boundary, price_type, interval="1m"))
        cursor = boundary
    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )
    times = pd.to_datetime(result["timestamp"], utc=True)
    expected = int((end - start).total_seconds() // 60)
    if len(result) != expected or times.diff().dropna().max() > pd.Timedelta(minutes=1):
        raise RuntimeError(f"incomplete Bitunix {price_type} 1m warm-up history")
    return result.reset_index(drop=True)


def combined_bitunix_alpha_minutes(
    forward: pd.DataFrame, *, warmup_path: Path = FORWARD_WARMUP_PATH
) -> pd.DataFrame:
    if not warmup_path.exists():
        return forward.copy()
    warmup = pd.read_parquet(warmup_path)
    observed = forward.loc[forward["data_valid"].fillna(False).astype(bool)].copy()
    combined = (
        pd.concat([warmup, observed], ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], format="mixed", utc=True)
    return combined


def download_external_forward(
    start: datetime,
    end: datetime,
    *,
    fetch: JsonFetcher = _get_json,
    output_root: Path = FORWARD_EXTERNAL_ROOT,
) -> dict[str, dict[str, Any]]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("forward archive boundaries must be timezone-aware and ordered")
    expected = (end - start).total_seconds() / 60
    manifest: dict[str, dict[str, Any]] = {}
    for exchange in EXCHANGES:
        frame = _download_external(exchange, "BTCUSDT", start, end, fetch)
        if frame.empty or "timestamp" not in frame:
            raise RuntimeError(f"empty forward archive: {exchange}")
        times = pd.to_datetime(frame["timestamp"], format="mixed", utc=True)
        frame = frame.loc[times.ge(pd.Timestamp(start)) & times.lt(pd.Timestamp(end))].copy()
        times = pd.to_datetime(frame["timestamp"], format="mixed", utc=True)
        if frame.empty:
            raise RuntimeError(f"no closed forward candles: {exchange}")
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise RuntimeError(f"invalid forward archive: {exchange}")
        coverage = len(frame) / expected
        if coverage < 0.95 or times.min() > pd.Timestamp(start) + pd.Timedelta(minutes=2):
            raise RuntimeError(f"insufficient forward coverage: {exchange} {coverage:.3%}")
        if times.max() < pd.Timestamp(end) - pd.Timedelta(minutes=2):
            raise RuntimeError(f"forward archive ends early: {exchange}")
        target = output_root / f"{exchange}_btcusdt_1m.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        manifest[exchange] = {
            "path": str(target),
            "rows": len(frame),
            "start": times.min().isoformat(),
            "end": times.max().isoformat(),
            "coverage": coverage,
            "sha256": _sha256(target),
        }
    return manifest


def download_orderflow_forward(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    output_path: Path = FORWARD_ORDERFLOW_PATH,
) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("forward order-flow boundaries must be ordered")
    frames = [
        _parse_binance_kline_zip(_download_checked(name)) for name in _archive_names(start, end)
    ]
    new = pd.concat(frames, ignore_index=True)
    history = pd.read_parquet(ORDERFLOW_PATH)
    combined = pd.concat([history, new], ignore_index=True).drop_duplicates(
        "timestamp", keep="last"
    )
    times = pd.to_datetime(combined["timestamp"], format="mixed", utc=True)
    combined = combined.loc[times.le(end)].sort_values("timestamp")
    context = build_orderflow_context(combined)
    selected = context.loc[
        pd.to_datetime(context["orderflow_available_at"], utc=True).gt(start)
        & pd.to_datetime(context["orderflow_available_at"], utc=True).le(end)
    ].copy()
    expected = (end - start).total_seconds() / (15 * 60)
    if len(selected.loc[selected["orderflow_coverage"]]) / expected < 0.95:
        raise RuntimeError("insufficient official Binance forward order-flow coverage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    selected.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    return selected


def _parse_binance_live_klines(payload: object) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise ValueError("Binance futures kline response must be a list")
    rows = cast(list[list[Any]], payload)
    if any(not isinstance(row, list) or len(row) < 11 for row in rows):
        raise ValueError("Binance futures kline row is malformed")
    frame = pd.DataFrame(
        [(row[0], row[7], row[8], row[10]) for row in rows],
        columns=["timestamp", "quote_volume", "trade_count", "taker_buy_quote"],
    )
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="raise"), unit="ms", utc=True
    )
    for column in ("quote_volume", "trade_count", "taker_buy_quote"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def download_binance_live_orderflow(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    fetch: JsonFetcher = _get_json,
    historical_path: Path = ORDERFLOW_PATH,
    output_path: Path = FORWARD_ORDERFLOW_PATH,
) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("live order-flow boundaries must be ordered")
    cursor = start.floor("min")
    boundary = end.floor("min")
    frames: list[pd.DataFrame] = []
    while cursor < boundary:
        url = (
            BINANCE_FUTURES_KLINES
            + "?"
            + urlencode(
                {
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": int(cursor.timestamp() * 1000),
                    "endTime": int(boundary.timestamp() * 1000) - 1,
                    "limit": 1500,
                }
            )
        )
        frame = _parse_binance_live_klines(fetch(url))
        frame = frame.loc[frame["timestamp"].ge(cursor) & frame["timestamp"].lt(boundary)]
        if frame.empty:
            raise RuntimeError("Binance live order-flow returned no closed minutes")
        frames.append(frame)
        advanced = pd.Timestamp(frame["timestamp"].max()) + pd.Timedelta(minutes=1)
        if advanced <= cursor:
            raise RuntimeError("Binance live order-flow pagination did not advance")
        cursor = advanced
    live = (
        pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    )
    expected = int((boundary - start.floor("min")).total_seconds() // 60)
    if len(live) != expected or live["timestamp"].diff().dropna().max() > pd.Timedelta(minutes=1):
        raise RuntimeError("Binance live order-flow coverage is incomplete")
    history_start = start.floor("min") - pd.Timedelta(days=8)
    history = pd.read_parquet(
        historical_path,
        filters=[("timestamp", ">=", history_start.to_pydatetime())],
    )
    combined = (
        pd.concat([history, live], ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )
    context = build_orderflow_context(combined)
    available = pd.to_datetime(context["orderflow_available_at"], utc=True)
    selected = context.loc[available.gt(start) & available.le(boundary)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    selected.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    return selected


def read_mark_minutes(path: Path = FORWARD_MARK_PATH) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
                candle = payload["candle"]
                rows.append(
                    {
                        "timestamp": pd.Timestamp(int(candle["time"]), unit="ms", tz="UTC"),
                        "mark_open": float(candle["open"]),
                        "mark_high": float(candle["high"]),
                        "mark_low": float(candle["low"]),
                        "mark_close": float(candle["close"]),
                        "mark_received_at": pd.Timestamp(payload["collected_at"]),
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def read_trade_minutes(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        events = pd.read_parquet(
            path,
            columns=[
                "event_id",
                "event_type",
                "exchange_timestamp",
                "received_timestamp",
                "price",
                "quantity",
            ],
        )
        frames.append(events.loc[events["event_type"].eq("trade")])
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True).drop_duplicates("event_id")
    return _aggregate_trades(trades)


def read_trade_json_minutes(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                    for event in payload.get("normalized", []):
                        if event.get("event_type") == "trade":
                            rows.append(event)
                except (json.JSONDecodeError, TypeError):
                    continue
    return _aggregate_trades(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    trades["exchange_timestamp"] = pd.to_datetime(
        trades["exchange_timestamp"], format="mixed", utc=True
    )
    trades["received_timestamp"] = pd.to_datetime(
        trades["received_timestamp"], format="mixed", utc=True
    )
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    trades = trades.dropna(subset=["exchange_timestamp", "received_timestamp", "price", "quantity"])
    order = ["exchange_timestamp", "received_timestamp"]
    if "event_id" in trades:
        order.append("event_id")
    trades = trades.loc[trades["price"].gt(0) & trades["quantity"].gt(0)].sort_values(
        order, kind="stable"
    )
    trades["timestamp"] = trades["exchange_timestamp"].dt.floor("min")
    trades["quote"] = trades["price"] * trades["quantity"]
    return (
        trades.groupby("timestamp", as_index=False)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            quote_volume=("quote", "sum"),
            last_received_at=("received_timestamp", "max"),
            trade_count=("price", "size"),
        )
        .sort_values("timestamp")
    )


def observed_trade_candles(
    microstructure_root: Path = FORWARD_MICROSTRUCTURE_ROOT,
    *,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    if timeframe_minutes <= 0:
        raise ValueError("timeframe must be positive")
    minutes = _observed_trade_minutes(microstructure_root)
    return resample_trade_minutes(minutes, timeframe_minutes=timeframe_minutes)


def resample_trade_minutes(minutes: pd.DataFrame, *, timeframe_minutes: int) -> pd.DataFrame:
    if timeframe_minutes <= 0:
        raise ValueError("timeframe must be positive")
    if minutes.empty:
        return minutes
    received_column = "last_received_at" if "last_received_at" in minutes else "received_at"
    grouped = minutes.set_index("timestamp").resample(
        f"{timeframe_minutes}min", origin="epoch", closed="left", label="left"
    )
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        minute_count=("close", "count"),
        received_at=(received_column, "max"),
    ).dropna(subset=["open", "close"])
    bars["data_valid"] = bars["minute_count"].eq(timeframe_minutes)
    bars["source"] = "bitunix-official-public-websocket-trades"
    return bars.reset_index()


def _observed_trade_minutes(microstructure_root: Path) -> pd.DataFrame:
    paths = sorted(microstructure_root.glob("btcusdt_*.parquet"))
    finalized = {path.stem for path in paths}
    json_paths = [
        path
        for path in sorted(microstructure_root.glob("btcusdt_*.jsonl"))
        if path.stem not in finalized
    ]
    archived = read_trade_minutes(paths)
    live = read_trade_json_minutes(json_paths)
    if not archived.empty and not live.empty:
        live = live.loc[~live["timestamp"].isin(archived["timestamp"])]
    minutes = pd.concat([archived, live], ignore_index=True).sort_values("timestamp")
    return minutes


def build_bitunix_forward_minutes(
    *,
    cutoff: pd.Timestamp,
    mark_path: Path = FORWARD_MARK_PATH,
    microstructure_root: Path = FORWARD_MICROSTRUCTURE_ROOT,
    output_path: Path = FORWARD_MINUTES_PATH,
    trade_minutes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    trades = (
        observed_trade_candles(microstructure_root, timeframe_minutes=1)
        if trade_minutes is None
        else trade_minutes.copy()
    )
    if not trades.empty:
        trades = trades.rename(columns={"received_at": "last_received_at"})
    marks = read_mark_minutes(mark_path)
    if trades.empty or marks.empty:
        return pd.DataFrame()
    result = trades.merge(marks, on="timestamp", how="inner", validate="one_to_one")
    result = result.loc[result["timestamp"].gt(cutoff)].copy()
    result["raw_mark_high"] = result["mark_high"]
    result["raw_mark_low"] = result["mark_low"]
    mark_high = result[["mark_open", "mark_high", "mark_close"]].max(axis=1)
    mark_low = result[["mark_open", "mark_low", "mark_close"]].min(axis=1)
    mark_envelope_deviation_bps = (
        (mark_high - result["mark_high"] + result["mark_low"] - mark_low)
        / result["mark_close"]
        * 10_000
    )
    result["mark_high"] = mark_high
    result["mark_low"] = mark_low
    result["available_at"] = result[["last_received_at", "mark_received_at"]].max(axis=1)
    result["data_valid"] = (
        result["data_valid"].fillna(False).astype(bool)
        & result["available_at"].le(result["timestamp"] + pd.Timedelta(minutes=2))
        & result["high"].ge(result[["open", "close"]].max(axis=1))
        & result["low"].le(result[["open", "close"]].min(axis=1))
        & mark_envelope_deviation_bps.le(100)
    )
    result["funding_rate"] = np.nan
    result["funding_event_rate"] = np.nan
    result["funding_coverage"] = False
    result["market_data_source"] = "bitunix-official-public-websocket-trades+rest-mark"
    result["price_source"] = "observed_trade"
    result["volume_source"] = "observed_trade"
    result["funding_source"] = "unavailable_until_official_history_audit"
    result["spread_source"] = "observed_separate_microstructure"
    result["cost_source"] = "execution_model_pending"
    result["round_trip_cost_bps"] = np.nan
    result["exchange"] = "bitunix"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    return result.reset_index(drop=True)


def build_forward_policy_events(
    app: AppConfig,
    *,
    bitunix_path: Path = FORWARD_MINUTES_PATH,
    external_root: Path = FORWARD_EXTERNAL_ROOT,
    orderflow_path: Path = FORWARD_ORDERFLOW_PATH,
    alpha_view_path: Path = FORWARD_ALPHA_VIEW_PATH,
) -> pd.DataFrame:
    """Build only causal, fully-covered Bitunix candidates for the frozen V14 model."""
    required = [external_root / f"{name}_btcusdt_1m.parquet" for name in EXCHANGES]
    if (
        not bitunix_path.exists()
        or not orderflow_path.exists()
        or not all(path.exists() for path in required)
    ):
        return pd.DataFrame()
    orderflow = pd.read_parquet(orderflow_path).sort_values("orderflow_available_at")
    alpha = combined_bitunix_alpha_minutes(pd.read_parquet(bitunix_path))
    alpha["round_trip_cost_bps"] = 0.0
    alpha["market_data_source"] = "observed:bitunix-official-public-websocket-trades+rest-mark"
    alpha["price_source"] = "observed"
    alpha["volume_source"] = "observed"
    alpha["funding_source"] = "unavailable"
    alpha["spread_source"] = "unavailable"
    alpha["cost_source"] = "alpha_gross_no_execution_cost"
    alpha_view_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = alpha_view_path.with_suffix(".tmp.parquet")
    alpha.to_parquet(temporary, index=False)
    os.replace(temporary, alpha_view_path)
    output: list[pd.DataFrame] = []
    for timeframe in (15, 30, 60):
        external = {
            name: build_feature_frame(path, app, name, timeframe_minutes=timeframe)[1]
            for name, path in zip(EXCHANGES, required, strict=True)
        }
        context = cross_exchange_features(external, EXCHANGES)[EXCHANGES[0]][
            [
                "timestamp",
                "cross_exchange_return_median",
                "cross_exchange_return_dispersion",
                "cross_feature_available_at",
            ]
        ]
        _, local = build_feature_frame(
            alpha_view_path, app, "bitunix", timeframe_minutes=timeframe
        )
        local["local_feature_coverage"] &= (
            local["data_valid"].fillna(False).astype(bool).rolling(100, min_periods=100).min().eq(1)
        )
        local = local.merge(context, on="timestamp", how="left", validate="one_to_one")
        local["lookahead_valid"] = pd.to_datetime(
            local["cross_feature_available_at"], utc=True
        ).le(pd.to_datetime(local["signal_timestamp"], utc=True))
        local = pd.merge_asof(
            local.sort_values("signal_timestamp"),
            orderflow,
            left_on="signal_timestamp",
            right_on="orderflow_available_at",
            direction="backward",
            tolerance=pd.Timedelta(minutes=15),
        )
        local["orderflow_lookahead_valid"] = pd.to_datetime(
            local["orderflow_available_at"], utc=True
        ).le(pd.to_datetime(local["signal_timestamp"], utc=True))
        local["feature_coverage"] = (
            local["local_feature_coverage"].fillna(False).astype(bool)
            & local["lookahead_valid"].fillna(False).astype(bool)
            & local["orderflow_coverage"].fillna(False).astype(bool)
            & local["orderflow_lookahead_valid"].fillna(False).astype(bool)
        )
        for side in ("long", "short"):
            events = first_reentry_events(local, side)
            selected = events.loc[events["event_signal"] & events["feature_coverage"]].copy()
            if selected.empty:
                continue
            selected["side"] = side
            selected["timeframe_minutes"] = timeframe
            selected["vwap_hours"] = 24.0
            selected["context_code"] = (
                selected["side_code"] * 1_000 + selected["regime_code"] * 100 + timeframe
            )
            output.append(selected)
    return (
        pd.concat(output, ignore_index=True).sort_values("signal_timestamp")
        if output
        else pd.DataFrame()
    )
