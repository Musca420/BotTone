from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.binance_l2_dataset import (
    _add_market_structure,
    _causal_anchor_features,
    _tail_daily_files,
)

ROOT = Path("data/raw/bitunix_microstructure")
MATERIALIZED = Path("data/research/bitunix_l2/btcusdt_l2_features.parquet")
FEATURES = (
    "spread_bps",
    "spread_change_bps",
    "microprice_distance_bps",
    "depth_imbalance_5bps",
    "depth_imbalance_10bps",
    "depth_imbalance_20bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_15s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "trade_arrival_rate_30s",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
    "mid_return_5s_bps",
    "mid_return_30s_bps",
    "range_60s_bps",
    "rolling_vwap_5m_distance_bps",
)


def _levels(value: object) -> list[list[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    result: list[list[float]] = []
    for level in value:
        try:
            price, quantity = float(level[0]), float(level[1])
        except (IndexError, TypeError, ValueError):
            continue
        if price > 0 and quantity >= 0 and np.isfinite(price + quantity):
            result.append([price, quantity])
    return result


def load_records(root: Path = ROOT) -> pd.DataFrame:
    """Load official Bitunix observations; availability is the local receive time."""
    frames: list[pd.DataFrame] = []
    parquet_columns = [
        "event_type",
        "exchange_timestamp",
        "received_timestamp",
        "best_bid",
        "best_ask",
        "midpoint",
        "bids_json",
        "asks_json",
        "price",
        "quantity",
        "aggressor_side",
    ]
    columns = [*parquet_columns, "bids", "asks"]
    parquet_days = {path.stem for path in root.glob("btcusdt_*.parquet")}
    for path in sorted(root.glob("btcusdt_*.parquet")):
        frame = pd.read_parquet(path, columns=parquet_columns)
        frame["bids"] = np.nan
        frame["asks"] = np.nan
        frames.append(frame)
    for path in sorted(root.glob("btcusdt_*.jsonl")):
        if path.stem in parquet_days:
            continue
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    normalized = json.loads(line).get("normalized", [])
                except json.JSONDecodeError:
                    continue
                rows.extend(item for item in normalized if isinstance(item, dict))
        if rows:
            frame = pd.DataFrame(rows)
            for column in columns:
                if column not in frame:
                    frame[column] = np.nan
            frames.append(frame[columns])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def latest_execution_snapshot(root: Path = ROOT, max_lines: int = 2_000) -> pd.DataFrame:
    """Build a fail-closed quote/integrity snapshot from recent official observations."""
    paths = sorted(root.glob("btcusdt_*.jsonl"))
    if not paths:
        return pd.DataFrame()
    events: list[dict[str, Any]] = []
    for line in _tail_daily_files(paths, max_lines):
        try:
            normalized = json.loads(line).get("normalized", [])
        except json.JSONDecodeError:
            continue
        events.extend(item for item in normalized if isinstance(item, dict))
    if not events:
        return pd.DataFrame()
    rows = pd.DataFrame(events)
    rows["received_timestamp"] = pd.to_datetime(
        rows["received_timestamp"], utc=True, format="mixed"
    )
    rows["exchange_timestamp"] = pd.to_datetime(
        rows["exchange_timestamp"], utc=True, format="mixed"
    )
    books = rows.loc[rows["event_type"].eq("book")].sort_values("received_timestamp")
    trades = rows.loc[rows["event_type"].eq("trade")].sort_values("received_timestamp")
    if books.empty:
        return pd.DataFrame()
    book = books.iloc[-1]
    observed_at = pd.Timestamp(book["received_timestamp"])
    recent_books = books.loc[books["received_timestamp"].ge(observed_at - pd.Timedelta(seconds=30))]
    book_gaps = recent_books["received_timestamp"].diff().dt.total_seconds()
    last_trade_at = (
        pd.Timestamp(trades.iloc[-1]["received_timestamp"]) if not trades.empty else pd.NaT
    )
    trade_age_ms = (
        max(0.0, (observed_at - last_trade_at).total_seconds() * 1_000)
        if pd.notna(last_trade_at)
        else float("inf")
    )
    bids = _levels(book.get("bids", book.get("bids_json")))
    asks = _levels(book.get("asks", book.get("asks_json")))
    best_bid = float(bids[0][0]) if bids else np.nan
    best_ask = float(asks[0][0]) if asks else np.nan
    mid = (best_bid + best_ask) / 2
    clock_drift_ms = (
        observed_at - pd.Timestamp(book["exchange_timestamp"])
    ).total_seconds() * 1_000
    sequence_gap = len(recent_books) < 2 or bool(book_gaps.dropna().gt(5).any())
    book_synced = bool(bids and asks and best_bid < best_ask)
    feed_alive = trade_age_ms <= 5_000
    valid = book_synced and feed_alive and not sequence_gap and abs(clock_drift_ms) <= 5_000
    return pd.DataFrame(
        [
            {
                "available_at": observed_at,
                "mid": mid,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bids": bids,
                "asks": asks,
                "spread_bps": (best_ask - best_bid) / mid * 10_000,
                "last_trade_update_age_ms": trade_age_ms,
                "last_book_update_age_ms": 0.0,
                "sequence_gap_detected": sequence_gap,
                "book_is_synced": book_synced,
                "trade_feed_alive": feed_alive,
                "orderbook_feed_alive": len(recent_books) >= 2,
                "clock_drift_ms": clock_drift_ms,
                "feature_valid": valid,
            }
        ]
    )


def _depth_in_band(levels: list[list[float]], mid: float, bps: float) -> float:
    return sum(quantity for price, quantity in levels if abs(price / mid - 1) * 10_000 <= bps)


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    rows = records.copy()
    for column in ("bids", "asks"):
        if column not in rows:
            rows[column] = np.nan
    rows["exchange_timestamp"] = pd.to_datetime(
        rows["exchange_timestamp"], format="mixed", utc=True, errors="coerce"
    )
    rows["available_at"] = pd.to_datetime(
        rows["received_timestamp"], format="mixed", utc=True, errors="coerce"
    )
    rows = rows.dropna(subset=["exchange_timestamp", "available_at"])
    rows = rows.loc[rows["available_at"].ge(rows["exchange_timestamp"] - pd.Timedelta(seconds=1))]
    rows["clock_drift_ms"] = (
        rows["available_at"] - rows["exchange_timestamp"]
    ).dt.total_seconds() * 1_000
    rows["second"] = rows["available_at"].dt.floor("s")

    books = rows.loc[rows["event_type"].eq("book")].sort_values("available_at")
    books = books.groupby("second", as_index=False).last()
    books["bids"] = books.apply(
        lambda row: _levels(row["bids"]) or _levels(row["bids_json"]), axis=1
    )
    books["asks"] = books.apply(
        lambda row: _levels(row["asks"]) or _levels(row["asks_json"]), axis=1
    )
    books["best_bid"] = pd.to_numeric(books["best_bid"], errors="coerce")
    books["best_ask"] = pd.to_numeric(books["best_ask"], errors="coerce")
    books["mid"] = (books["best_bid"] + books["best_ask"]) / 2

    trades = rows.loc[rows["event_type"].eq("trade")].copy()
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["quote"] = trades["price"] * trades["quantity"]
    trades["buy_quote"] = trades["quote"].where(trades["aggressor_side"].eq("buy"), 0.0)
    trades["sell_quote"] = trades["quote"].where(trades["aggressor_side"].eq("sell"), 0.0)
    trade_seconds = trades.groupby("second", as_index=False).agg(
        buy_quote=("buy_quote", "sum"),
        sell_quote=("sell_quote", "sum"),
        base_volume=("quantity", "sum"),
        quote_volume=("quote", "sum"),
        trade_count=("quantity", "size"),
        last_trade_available_at=("available_at", "max"),
    )
    data = books.merge(trade_seconds, on="second", how="left")
    for column in ("buy_quote", "sell_quote", "base_volume", "quote_volume", "trade_count"):
        data[column] = data[column].fillna(0.0)
    data["available_at"] = data[["available_at", "last_trade_available_at"]].max(axis=1)
    data["exchange_second"] = (
        data["second"].astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000
    )
    data = (
        data.sort_values("available_at")
        .drop_duplicates("second", keep="last")
        .reset_index(drop=True)
    )
    index = pd.DatetimeIndex(data["second"])

    bid_size = data["bids"].map(lambda levels: levels[0][1] if levels else np.nan)
    ask_size = data["asks"].map(lambda levels: levels[0][1] if levels else np.nan)
    microprice = (data["best_ask"] * bid_size + data["best_bid"] * ask_size) / (bid_size + ask_size)
    data["spread_bps"] = (data["best_ask"] - data["best_bid"]) / data["mid"] * 10_000
    data["spread_change_bps"] = data["spread_bps"].diff()
    data["microprice_distance_bps"] = (microprice / data["mid"] - 1) * 10_000
    for bps in (5, 10, 20):
        bid = data.apply(lambda row, n=bps: _depth_in_band(row["bids"], row["mid"], n), axis=1)
        ask = data.apply(lambda row, n=bps: _depth_in_band(row["asks"], row["mid"], n), axis=1)
        data[f"bid_depth_{bps}bps"] = bid
        data[f"ask_depth_{bps}bps"] = ask
        data[f"depth_imbalance_{bps}bps"] = (bid - ask) / (bid + ask).replace(0, np.nan)
    data["depth_imbalance_1"] = (bid_size - ask_size) / (bid_size + ask_size)
    data["depth_imbalance_5"] = data["depth_imbalance_5bps"]
    data["depth_imbalance_20"] = data["depth_imbalance_20bps"]

    buy = pd.Series(data["buy_quote"].to_numpy(), index=index)
    sell = pd.Series(data["sell_quote"].to_numpy(), index=index)
    for seconds in (1, 3, 5, 15, 30, 60):
        bought, sold = buy.rolling(f"{seconds}s").sum(), sell.rolling(f"{seconds}s").sum()
        data[f"aggressive_imbalance_{seconds}s"] = (
            (bought - sold) / (bought + sold).replace(0, np.nan)
        ).to_numpy()
    trade_count = pd.Series(data["trade_count"].to_numpy(), index=index)
    data["trade_count_30s"] = trade_count.rolling("30s").sum().to_numpy()
    data["trade_arrival_rate_30s"] = data["trade_count_30s"] / 30
    data["aggressive_buy_volume"] = data["buy_quote"]
    data["aggressive_sell_volume"] = data["sell_quote"]
    data["market_buy_sell_delta"] = data["buy_quote"] - data["sell_quote"]
    data["aggressor_ratio"] = data["buy_quote"] / data["sell_quote"].replace(0, np.nan)
    timed_mid = pd.Series(data["mid"].to_numpy(), index=index)
    data["mid_return_5s_bps"] = timed_mid.pct_change(freq="5s").to_numpy() * 10_000
    data["mid_return_30s_bps"] = timed_mid.pct_change(freq="30s").to_numpy() * 10_000
    data["mid_return_15m_bps"] = timed_mid.pct_change(freq="15min").to_numpy() * 10_000
    data["mid_return_30m_bps"] = timed_mid.pct_change(freq="30min").to_numpy() * 10_000
    data["range_60s_bps"] = (
        (timed_mid.rolling("60s").max() - timed_mid.rolling("60s").min()) / timed_mid * 10_000
    ).to_numpy()
    for side in ("bid", "ask"):
        depth = pd.Series(data[f"{side}_depth_5bps"].to_numpy(), index=index)
        removed = (-depth.diff()).clip(lower=0).rolling("5s").sum()
        data[f"{side}_cancel_rate_5s"] = (
            removed / depth.rolling("5s").mean().replace(0, np.nan)
        ).to_numpy()

    base = pd.Series(data["base_volume"].to_numpy(), index=index)
    quote = pd.Series(data["quote_volume"].to_numpy(), index=index)
    rolling_base, rolling_quote = base.rolling("300s").sum(), quote.rolling("300s").sum()
    rolling_vwap = rolling_quote / rolling_base.replace(0, np.nan)
    data["rolling_vwap"] = rolling_vwap.to_numpy()
    data["rolling_vwap_5m_distance_bps"] = (data["mid"] / data["rolling_vwap"] - 1) * 10_000
    anchor = _causal_anchor_features(data, data["base_volume"], data["quote_volume"], index)
    data[list(anchor.columns)] = anchor
    data["anchored_vwap_distance_bps"] = (data["mid"] / data["anchored_vwap"] - 1) * 10_000
    reset = data["anchor_age_seconds"].eq(0) | data["anchor_age_seconds"].lt(
        data["anchor_age_seconds"].shift(1)
    )
    data["anchor_type"] = "MAJOR_IMPULSE"
    data["anchor_timestamp"] = data["available_at"].where(reset).ffill()
    data["anchor_price"] = data["mid"].where(reset).ffill()
    data["anchor_age_bars"] = data["anchor_age_seconds"]
    anchor_group = reset.cumsum()
    data["volume_since_anchor"] = data["quote_volume"].groupby(anchor_group).cumsum()
    data["return_since_anchor_bps"] = (data["mid"] / data["anchor_price"] - 1) * 10_000
    data["avwap_slope_bps_60s"] = data["anchored_vwap"].pct_change(60) * 10_000
    data["avwap_slope_change_bps"] = data["avwap_slope_bps_60s"].diff(60)
    anchor_touch = data["anchored_vwap_distance_bps"].abs().le(1.0)
    anchor_rejection = anchor_touch.shift(1, fill_value=False) & data[
        "anchored_vwap_distance_bps"
    ].abs().gt(2.0)
    data["avwap_tests"] = anchor_touch.groupby(anchor_group).cumsum()
    data["avwap_rejections"] = anchor_rejection.groupby(anchor_group).cumsum()
    data["avwap_rejection_strength_bps"] = (
        data["anchored_vwap_distance_bps"].abs().where(anchor_rejection, 0.0)
    )
    data["rolling_vwap_vs_avwap_distance_bps"] = (
        data["rolling_vwap"] / data["anchored_vwap"] - 1
    ) * 10_000
    data["rolling_vwap_avwap_convergence_bps"] = (
        -data["rolling_vwap_vs_avwap_distance_bps"].abs().diff()
    )
    data = _add_market_structure(data)
    data["market_regime"] = np.select(
        [
            data["volatility_percentile"].ge(0.99),
            data["volatility_percentile"].ge(0.75),
            data["volatility_percentile"].le(0.25),
        ],
        ["STRESS", "HIGH_VOL", "LOW_VOL"],
        default="NORMAL",
    )

    seconds_since_book = data["available_at"].diff().dt.total_seconds()
    trade_age = (data["available_at"] - data["last_trade_available_at"]).dt.total_seconds() * 1_000
    data["last_book_update_age_ms"] = 0.0
    data["last_trade_update_age_ms"] = trade_age
    data["sequence_gap_detected"] = seconds_since_book.gt(5)
    data["book_is_synced"] = (
        data["bids"].map(bool) & data["asks"].map(bool) & data["best_bid"].lt(data["best_ask"])
    )
    data["trade_feed_alive"] = trade_age.le(5_000)
    data["orderbook_feed_alive"] = seconds_since_book.fillna(0).le(5)
    data["feature_valid"] = (
        data["book_is_synced"]
        & data["trade_feed_alive"]
        & data["orderbook_feed_alive"]
        & ~data["sequence_gap_detected"]
        & data["clock_drift_ms"].abs().le(5_000)
        & data[list(FEATURES)].notna().all(axis=1)
    )
    return data


def materialize(root: Path = ROOT, output: Path = MATERIALIZED) -> Path:
    frame = build_features(load_records(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output


if __name__ == "__main__":
    print(materialize())
