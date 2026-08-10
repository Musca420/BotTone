from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from adaptive_bot.btc_vwap_alpha import LIVE_REQUIRED, canonical_minute_market_features

ROOT = Path("data/research/binance_l2")
BINANCE_FUTURES_KLINES = (
    "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=500"
)
LABEL_HORIZONS_SECONDS = (5, 30, 60, 300, 900, 1800)
FEATURES = (
    "spread_bps",
    "microprice_distance_bps",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "trade_count_30s",
    "mid_return_5s_bps",
    "mid_return_30s_bps",
    "depth_change_5s",
    "rolling_vwap_5m_distance_bps",
)
ANCHOR_FEATURES = (
    "anchored_vwap",
    "anchored_vwap_distance_bps",
    "anchor_direction",
    "anchor_age_seconds",
    "anchor_available_at",
)


def _tail_text_lines(path: Path, max_lines: int) -> list[str]:
    """Read complete trailing lines without loading a growing collector file."""
    if max_lines <= 0:
        return []
    chunk_size = 1_048_576
    chunks: list[bytes] = []
    with path.open("rb") as source:
        source.seek(0, 2)
        position = source.tell()
        line_count = 0
        while position > 0 and line_count <= max_lines:
            size = min(chunk_size, position)
            position -= size
            source.seek(position)
            chunk = source.read(size)
            chunks.append(chunk)
            line_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    lines = data.splitlines()
    return [line.decode("utf-8") for line in lines[-max_lines:]]


def _tail_daily_files(paths: list[Path], max_lines: int) -> list[str]:
    """Read one continuous tail even when a UTC day just rolled over."""
    batches: list[list[str]] = []
    remaining = max_lines
    for path in reversed(paths):
        if remaining <= 0:
            break
        lines = _tail_text_lines(path, remaining)
        batches.append(lines)
        remaining -= len(lines)
    return [line for batch in reversed(batches) for line in batch]


def _add_market_structure(data: pd.DataFrame) -> pd.DataFrame:
    """Add causal VWAP, volatility and multi-timeframe state features."""
    index = pd.DatetimeIndex(data["second"])
    distance = data["rolling_vwap_5m_distance_bps"]
    data["rolling_vwap_slope_bps_60s"] = data["rolling_vwap"].pct_change(60) * 10_000
    data["rolling_vwap_slope_change_bps"] = data["rolling_vwap_slope_bps_60s"].diff(60)
    touch = distance.abs().le(1.0)
    cross = np.sign(distance).ne(np.sign(distance.shift(1)))
    rejection = touch.shift(1, fill_value=False) & distance.abs().gt(2.0)
    data["rolling_vwap_tests_30m"] = (
        pd.Series(touch.to_numpy(float), index=index).rolling("1800s").sum().to_numpy()
    )
    data["rolling_vwap_rejections_30m"] = (
        pd.Series(rejection.to_numpy(float), index=index).rolling("1800s").sum().to_numpy()
    )
    cross_time = data["available_at"].where(cross).ffill()
    data["time_since_last_vwap_cross_seconds"] = (
        data["available_at"] - cross_time
    ).dt.total_seconds()
    data["volume_on_vwap_test"] = data["quote_volume"].where(touch)
    data["vwap_rejection_strength_bps"] = distance.abs().where(rejection, 0.0)
    rolling_std = pd.Series(data["mid"].to_numpy(), index=index).rolling("300s").std()
    data["vwap_band_position"] = (
        data["mid"].to_numpy() - data["rolling_vwap"].to_numpy()
    ) / rolling_std.replace(0, np.nan).to_numpy()

    minute = (
        data.set_index("second")
        .resample("1min")
        .agg(
            open=("mid", "first"),
            high=("mid", "max"),
            low=("mid", "min"),
            close=("mid", "last"),
            volume=("quote_volume", "sum"),
            spread=("spread_bps", "mean"),
            depth=("bid_depth_10bps", "mean"),
        )
    )
    for timeframe in (1, 5, 15, 30):
        bars = minute.resample(f"{timeframe}min").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        prior = bars["close"].shift(1)
        true_range = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - prior).abs(),
                (bars["low"] - prior).abs(),
            ],
            axis=1,
        ).max(axis=1)
        available_index = bars.index + pd.Timedelta(minutes=timeframe)
        atr = pd.Series(
            true_range.rolling(14, min_periods=14).mean().to_numpy(),
            index=available_index,
        )
        minute[f"atr_{timeframe}m"] = atr.reindex(minute.index, method="ffill")
        trend = pd.Series(bars["close"].pct_change(3).to_numpy(), index=available_index).reindex(
            minute.index, method="ffill"
        )
        minute[f"trend_{timeframe}m"] = np.sign(trend)
        minute[f"vwap_state_{timeframe}m"] = np.sign(
            minute["close"] - minute["close"].rolling(timeframe, min_periods=timeframe).mean()
        )
    minute["realized_volatility"] = minute["close"].pct_change().rolling(30).std()
    minute["range_n_bps"] = (
        (minute["high"].rolling(30).max() - minute["low"].rolling(30).min())
        / minute["close"]
        * 10_000
    )
    for source, target in (
        ("realized_volatility", "volatility_percentile"),
        ("volume", "volume_percentile"),
        ("spread", "spread_percentile"),
        ("depth", "depth_percentile"),
    ):
        minute[target] = minute[source].rolling(360, min_periods=60).rank(pct=True)
    structure = minute.drop(columns=["open", "high", "low", "close", "volume", "spread", "depth"])
    structure.index = structure.index + pd.Timedelta(minutes=1)
    structure.index.name = "available_minute"
    merged = pd.merge_asof(
        data.sort_values("available_at"),
        structure.reset_index().sort_values("available_minute"),
        left_on="available_at",
        right_on="available_minute",
        direction="backward",
    ).drop(columns="available_minute")
    trade_bar_columns = {
        "trade_open",
        "trade_high",
        "trade_low",
        "trade_close",
    }
    if not trade_bar_columns.issubset(data.columns):
        # Bitunix reuses the venue-neutral structure builder but has only
        # aggregated trade volume here. Binance-only Alpha candles are added
        # exclusively when the four observed trade OHLC fields exist.
        return merged
    alpha_minutes = (
        data.set_index("second")
        .resample("1min")
        .agg(
            open=("trade_open", "first"),
            high=("trade_high", "max"),
            low=("trade_low", "min"),
            close=("trade_close", "last"),
            volume=("base_volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            taker_buy_quote=("buy_quote", "sum"),
            trade_count=("trade_count", "sum"),
        )
        .reset_index(names="timestamp")
    )
    alpha_minutes["available_at"] = alpha_minutes["timestamp"] + pd.Timedelta(minutes=1)
    canonical = canonical_minute_market_features(alpha_minutes)
    alpha_columns = (
        "return_1m_bps",
        "return_2m_bps",
        "return_3m_bps",
        "return_5m_bps",
        "return_10m_bps",
        "return_15m_bps",
        "return_30m_bps",
        "return_60m_bps",
        "rolling_vwap",
        "rolling_vwap_5m",
        "rolling_vwap_15m",
        "rolling_vwap_60m",
        "rolling_vwap_240m",
        "vwap_distance_bps",
        "vwap_distance_5m_bps",
        "vwap_distance_15m_bps",
        "vwap_distance_240m_bps",
        "vwap_slope_bps",
        "vwap_slope_change_bps",
        "vwap_distance_velocity_3m_bps",
        "vwap_tests_30m",
        "vwap_rejections_30m",
        "time_since_vwap_cross_minutes",
        "vwap_rejection_strength_bps",
        "vwap_band_position",
        "range_60s_bps",
        "recent_low_5m",
        "recent_high_5m",
        "atr_1m_bps",
        "atr_5m_bps",
        "atr_15m_bps",
        "atr_30m_bps",
        "realized_volatility_30m_bps",
        "volatility_percentile",
        "volume_percentile",
        "taker_imbalance_60s",
        "taker_imbalance_15m",
        "taker_imbalance_60m",
        "taker_imbalance_change_5m",
        "trade_count_zscore",
        "aggressive_volume_zscore",
        "candle_body_bps",
        "wick_imbalance_bps",
        "efficiency_15m",
        "efficiency_60m",
        "range_position_15m",
        "range_position_60m",
        "feature_contract_valid",
    )
    canonical = canonical[["available_at", *alpha_columns]].rename(
        columns={column: f"alpha_{column}" for column in alpha_columns}
    )
    return pd.merge_asof(
        merged.sort_values("available_at"),
        canonical.sort_values("available_at"),
        on="available_at",
        direction="backward",
    )


def load_records(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("btcusdt_*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if (
                    int(row.get("schema_version", 0)) >= 2
                    and row.get("source") == "binance-official-usdm-websocket-routed"
                ):
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["exchange_second", "available_at"])
        .drop_duplicates("exchange_second", keep="last")
        .reset_index(drop=True)
    )


def load_recent_records(root: Path = ROOT, max_lines: int = 7_200) -> pd.DataFrame:
    paths = sorted(root.glob("btcusdt_*.jsonl"))
    if not paths:
        return pd.DataFrame()
    rows = []
    for line in _tail_daily_files(paths, max_lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            int(row.get("schema_version", 0)) >= 2
            and row.get("source") == "binance-official-usdm-websocket-routed"
        ):
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["exchange_second", "available_at"])
        .drop_duplicates("exchange_second", keep="last")
        .reset_index(drop=True)
    )


def _fetch_json(url: str) -> object:
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "BotTone/1.0"}
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def load_recent_official_minutes(
    *,
    fetch: Callable[[str], object] = _fetch_json,
    retrieved_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load closed Binance USD-M one-minute bars for the live Alpha contract."""
    payload = fetch(BINANCE_FUTURES_KLINES)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Binance futures kline response must be a non-empty list")
    if any(not isinstance(row, list) or len(row) < 11 for row in payload):
        raise ValueError("Binance futures kline row is malformed")
    columns = (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote",
    )
    rows = pd.DataFrame(
        [row[: len(columns)] for row in payload], columns=columns
    )
    for column in columns:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows["timestamp"] = pd.to_datetime(rows.pop("open_time"), unit="ms", utc=True)
    close_time = pd.to_datetime(rows.pop("close_time"), unit="ms", utc=True)
    observed = retrieved_at if retrieved_at is not None else pd.Timestamp.now(tz="UTC")
    observed = (
        observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    )
    rows = rows.loc[close_time.lt(observed)].copy()
    rows["available_at"] = rows["timestamp"] + pd.Timedelta(minutes=1)
    valid = (
        rows[["open", "high", "low", "close", "volume", "quote_volume"]]
        .notna()
        .all(axis=1)
        & rows[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & rows["high"].ge(rows[["open", "close"]].max(axis=1))
        & rows["low"].le(rows[["open", "close"]].min(axis=1))
        & rows["high"].ge(rows["low"])
        & rows["volume"].ge(0)
        & rows["quote_volume"].ge(0)
        & rows["taker_buy_quote"].between(0, rows["quote_volume"])
    )
    if not valid.all():
        raise ValueError("Binance futures kline response contains invalid OHLCV")
    return (
        rows.drop(columns=["taker_buy_volume"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _depth(levels: list[list[str]], count: int) -> float:
    return sum(float(level[1]) for level in levels[:count])


def _depth_in_band(levels: list[list[str]], mid: float, bps: float) -> float:
    return sum(
        float(level[1]) for level in levels if abs(float(level[0]) / mid - 1) * 10_000 <= bps
    )


def _causal_anchor_features(
    rows: pd.DataFrame,
    base_volume: pd.Series,
    quote_volume: pd.Series,
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    intensity_baseline = pd.Series(
        pd.Series(rows["trade_count_30s"].to_numpy(), index=time_index)
        .rolling("900s", min_periods=300)
        .median()
        .to_numpy(),
        index=rows.index,
    )
    impulse = rows["mid_return_30s_bps"].abs().ge(5.0) & rows["trade_count_30s"].ge(
        intensity_baseline * 1.25
    )
    anchored = np.full(len(rows), np.nan)
    direction = np.full(len(rows), np.nan)
    age = np.full(len(rows), np.nan)
    available_at: list[object] = [pd.NaT] * len(rows)
    anchor_index: int | None = None
    anchor_second: int | None = None
    anchor_side = 0
    cumulative_base = 0.0
    cumulative_quote = 0.0
    seconds = rows["exchange_second"].to_numpy(dtype=np.int64)
    returns = rows["mid_return_30s_bps"].to_numpy(dtype=float)
    timestamps = rows["available_at"].to_numpy()
    base_values = base_volume.to_numpy(dtype=float)
    quote_values = quote_volume.to_numpy(dtype=float)
    for position in range(len(rows)):
        current_second = int(seconds[position])
        elapsed = None if anchor_second is None else current_second - anchor_second
        triggered = bool(impulse.iloc[position])
        proposed_side = int(np.sign(returns[position])) if triggered else anchor_side
        reset = triggered and (
            anchor_index is None
            or elapsed is None
            or elapsed >= 300
            or (proposed_side != anchor_side and elapsed >= 60)
        )
        if reset:
            anchor_index = position
            anchor_second = current_second
            anchor_side = proposed_side
            cumulative_base = 0.0
            cumulative_quote = 0.0
        if anchor_index is None or anchor_second is None:
            continue
        cumulative_base += base_values[position]
        cumulative_quote += quote_values[position]
        if cumulative_base > 0:
            anchored[position] = cumulative_quote / cumulative_base
        direction[position] = anchor_side
        age[position] = current_second - anchor_second
        available_at[position] = pd.Timestamp(timestamps[anchor_index])
    return pd.DataFrame(
        {
            "anchored_vwap": anchored,
            "anchor_direction": direction,
            "anchor_age_seconds": age,
            "anchor_available_at": available_at,
        },
        index=rows.index,
    )


def build_live_minute_features(
    records: pd.DataFrame, official_minutes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build only the completed-minute contract used by the live Alpha.

    Official Binance klines provide the canonical Alpha bars when supplied;
    WebSocket records remain the causal book/order-flow diagnostics. The research
    builder below keeps its wide per-second matrix for path labels.
    """
    if records.empty:
        return records.copy()
    rows = records.copy()
    rows["available_at"] = pd.to_datetime(
        rows["available_at"], utc=True, format="mixed"
    )
    rows["exchange_timestamp"] = pd.to_datetime(
        rows["exchange_second"], unit="s", utc=True
    )
    rows = (
        rows.loc[rows["available_at"].ge(rows["exchange_timestamp"])]
        .sort_values("exchange_timestamp")
        .drop_duplicates("exchange_second", keep="last")
        .reset_index(drop=True)
    )
    if rows.empty:
        return rows

    bid = rows["bids"].map(lambda levels: float(levels[0][0]))
    ask = rows["asks"].map(lambda levels: float(levels[0][0]))
    bid_size = rows["bids"].map(lambda levels: float(levels[0][1]))
    ask_size = rows["asks"].map(lambda levels: float(levels[0][1]))
    rows["mid"] = (bid + ask) / 2
    rows["best_bid"] = bid
    rows["best_ask"] = ask
    rows["spread_bps"] = (ask - bid) / rows["mid"] * 10_000
    microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    rows["microprice_distance_bps"] = (microprice - rows["mid"]) / rows["mid"] * 10_000

    buy = pd.to_numeric(rows["buy_quote"], errors="coerce")
    sell = pd.to_numeric(rows["sell_quote"], errors="coerce")
    trades = pd.to_numeric(rows["trade_count"], errors="coerce")
    time_index = pd.DatetimeIndex(rows["exchange_timestamp"])
    for seconds in (5, 30, 60):
        bought = pd.Series(buy.to_numpy(), index=time_index).rolling(f"{seconds}s").sum()
        sold = pd.Series(sell.to_numpy(), index=time_index).rolling(f"{seconds}s").sum()
        rows[f"aggressive_imbalance_{seconds}s"] = (
            (bought - sold) / (bought + sold).replace(0, np.nan)
        ).to_numpy()
    rows["trade_count_30s"] = (
        pd.Series(trades.to_numpy(), index=time_index).rolling("30s").sum().to_numpy()
    )
    rows["mid_return_30s_bps"] = (
        pd.Series(rows["mid"].to_numpy(), index=time_index)
        .pct_change(freq="30s")
        .to_numpy()
        * 10_000
    )

    trade_prices = rows["aggregate_trades"].map(
        lambda values: [
            float(trade[0]) for trade in (values if isinstance(values, list) else [])
        ]
    )
    rows["trade_open"] = trade_prices.map(
        lambda prices: prices[0] if prices else np.nan
    )
    rows["trade_high"] = trade_prices.map(
        lambda prices: max(prices) if prices else np.nan
    )
    rows["trade_low"] = trade_prices.map(
        lambda prices: min(prices) if prices else np.nan
    )
    rows["trade_close"] = trade_prices.map(
        lambda prices: prices[-1] if prices else np.nan
    )
    base_volume = rows["aggregate_trades"].map(
        lambda values: sum(
            float(trade[1]) for trade in (values if isinstance(values, list) else [])
        )
    )
    quote_volume = rows["aggregate_trades"].map(
        lambda values: sum(
            float(trade[0]) * float(trade[1])
            for trade in (values if isinstance(values, list) else [])
        )
    )
    anchor = _causal_anchor_features(rows, base_volume, quote_volume, time_index)
    rows[list(anchor.columns)] = anchor
    rows["minute"] = rows["exchange_timestamp"].dt.floor("min")
    rows["base_volume"] = base_volume
    rows["quote_volume"] = quote_volume
    rows["taker_buy_quote"] = buy

    grouped = rows.groupby("minute", sort=True)
    observed_minute = grouped.agg(
        available_at=("available_at", "max"),
        open=("trade_open", "first"),
        high=("trade_high", "max"),
        low=("trade_low", "min"),
        close=("trade_close", "last"),
        volume=("base_volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
        trade_count=("trade_count", "sum"),
        observed_seconds=("exchange_second", "nunique"),
        last_exchange_second=("exchange_second", "max"),
    ).reset_index(names="timestamp")
    # A minute is causal only once its closing boundary has passed.  The last
    # second may arrive a fraction before that boundary, so never expose the
    # aggregate earlier than timestamp + one minute.
    observed_minute["available_at"] = pd.concat(
        [
            observed_minute["available_at"],
            observed_minute["timestamp"] + pd.Timedelta(minutes=1),
        ],
        axis=1,
    ).max(axis=1)
    observed_minute = observed_minute.loc[
        observed_minute["observed_seconds"].ge(57)
        & observed_minute["last_exchange_second"].mod(60).eq(59)
    ].copy()
    minute = (
        official_minutes.copy()
        if official_minutes is not None
        else observed_minute
    )
    if minute.empty:
        return minute

    canonical = canonical_minute_market_features(minute)
    source_columns = sorted(
        {
            column.removeprefix("alpha_")
            for column in LIVE_REQUIRED
            if column.startswith("alpha_")
        }
        | {"rolling_vwap", "recent_low_5m", "recent_high_5m"}
    )
    alpha = canonical[["timestamp", "available_at", *source_columns]].rename(
        columns={column: f"alpha_{column}" for column in source_columns}
    )

    diagnostic = grouped.tail(1).copy()
    diagnostic = diagnostic.loc[
        diagnostic["minute"].isin(pd.DatetimeIndex(minute["timestamp"]))
    ]
    for levels in (1, 5, 20):
        bid_depth = diagnostic["bids"].map(lambda value, n=levels: _depth(value, n))
        ask_depth = diagnostic["asks"].map(lambda value, n=levels: _depth(value, n))
        diagnostic[f"depth_imbalance_{levels}"] = (bid_depth - ask_depth) / (
            bid_depth + ask_depth
        ).replace(0, np.nan)
    diagnostic["anchored_vwap_distance_bps"] = (
        diagnostic["mid"] / diagnostic["anchored_vwap"] - 1
    ) * 10_000
    diagnostic["clock_drift_ms"] = (
        diagnostic["available_at"] - diagnostic["exchange_timestamp"]
    ).dt.total_seconds() * 1_000
    diagnostic["last_book_update_age_ms"] = diagnostic["clock_drift_ms"]
    diagnostic["last_trade_update_age_ms"] = diagnostic["clock_drift_ms"]
    diagnostic["sequence_gap_detected"] = False
    diagnostic["book_is_synced"] = diagnostic["best_bid"].lt(diagnostic["best_ask"])
    diagnostic["trade_feed_alive"] = diagnostic["clock_drift_ms"].le(5_000)
    diagnostic["orderbook_feed_alive"] = True
    diagnostic = diagnostic.rename(columns={"minute": "timestamp"})
    result = pd.merge(
        alpha,
        diagnostic,
        on="timestamp",
        how="inner",
        suffixes=("", "_raw"),
        validate="one_to_one",
    )
    result["available_at"] = result["available_at"].combine(
        result.pop("available_at_raw"), max
    )
    result["rolling_vwap"] = result["alpha_rolling_vwap"]
    result["rolling_vwap_5m_distance_bps"] = result[
        "alpha_vwap_distance_5m_bps"
    ]
    result["range_60s_bps"] = result["alpha_range_60s_bps"]
    result["aggressive_imbalance_60s"] = result["alpha_taker_imbalance_60s"]
    canonical_required = [
        str(column)
        for column in result.columns
        if isinstance(column, str)
        and column.startswith("alpha_")
        and column != "alpha_feature_contract_valid"
    ]
    result["feature_valid"] = (
        result["alpha_feature_contract_valid"].fillna(False)
        & result[canonical_required].notna().all(axis=1)
        & result["book_is_synced"]
        & result["trade_feed_alive"]
    )
    return result.sort_values("available_at").reset_index(drop=True)


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    rows = records.copy()
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True, format="mixed")
    rows["exchange_timestamp"] = pd.to_datetime(rows["exchange_second"], unit="s", utc=True)
    rows = rows.loc[rows["available_at"].ge(rows["exchange_timestamp"])].copy()
    bid = rows["bids"].map(lambda levels: float(levels[0][0]))
    ask = rows["asks"].map(lambda levels: float(levels[0][0]))
    bid_size = rows["bids"].map(lambda levels: float(levels[0][1]))
    ask_size = rows["asks"].map(lambda levels: float(levels[0][1]))
    rows["mid"] = (bid + ask) / 2
    rows["best_bid"] = bid
    rows["best_ask"] = ask
    rows["spread_bps"] = (ask - bid) / rows["mid"] * 10_000
    rows["spread_change_bps"] = rows["spread_bps"].diff()
    microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    rows["microprice_distance_bps"] = (microprice - rows["mid"]) / rows["mid"] * 10_000
    for levels in (1, 5, 20):
        bid_depth = rows["bids"].map(lambda value, n=levels: _depth(value, n))
        ask_depth = rows["asks"].map(lambda value, n=levels: _depth(value, n))
        rows[f"depth_imbalance_{levels}"] = (bid_depth - ask_depth) / (
            bid_depth + ask_depth
        ).replace(0, np.nan)
        if levels == 5:
            rows["depth_5"] = bid_depth + ask_depth
    for bps in (5, 10, 20):
        bid_depth = rows.apply(
            lambda row, band=bps: _depth_in_band(row["bids"], row["mid"], band), axis=1
        )
        ask_depth = rows.apply(
            lambda row, band=bps: _depth_in_band(row["asks"], row["mid"], band), axis=1
        )
        rows[f"bid_depth_{bps}bps"] = bid_depth
        rows[f"ask_depth_{bps}bps"] = ask_depth
        rows[f"depth_imbalance_{bps}bps"] = (bid_depth - ask_depth) / (
            bid_depth + ask_depth
        ).replace(0, np.nan)
    buy = pd.to_numeric(rows["buy_quote"], errors="coerce")
    sell = pd.to_numeric(rows["sell_quote"], errors="coerce")
    rows["buy_quote"] = buy
    rows["sell_quote"] = sell
    time_index = pd.DatetimeIndex(rows["exchange_timestamp"])
    for seconds in (1, 3, 5, 15, 30, 60):
        bought = pd.Series(buy.to_numpy(), index=time_index).rolling(f"{seconds}s").sum()
        sold = pd.Series(sell.to_numpy(), index=time_index).rolling(f"{seconds}s").sum()
        rows[f"aggressive_imbalance_{seconds}s"] = (
            (bought - sold) / (bought + sold).replace(0, np.nan)
        ).to_numpy()
    trade_count = pd.to_numeric(rows["trade_count"], errors="coerce")
    rows["trade_count_30s"] = (
        pd.Series(trade_count.to_numpy(), index=time_index).rolling("30s").sum().to_numpy()
    )
    rows["trade_arrival_rate_30s"] = rows["trade_count_30s"] / 30
    rows["aggressive_buy_volume"] = buy
    rows["aggressive_sell_volume"] = sell
    rows["market_buy_sell_delta"] = buy - sell
    rows["aggressor_ratio"] = buy / sell.replace(0, np.nan)
    rows["mid_return_5s_bps"] = (
        pd.Series(rows["mid"].to_numpy(), index=time_index).pct_change(freq="5s").to_numpy()
        * 10_000
    )
    rows["mid_return_30s_bps"] = (
        pd.Series(rows["mid"].to_numpy(), index=time_index).pct_change(freq="30s").to_numpy()
        * 10_000
    )
    rows["mid_return_15m_bps"] = (
        pd.Series(rows["mid"].to_numpy(), index=time_index).pct_change(freq="15min").to_numpy()
        * 10_000
    )
    rows["mid_return_30m_bps"] = (
        pd.Series(rows["mid"].to_numpy(), index=time_index).pct_change(freq="30min").to_numpy()
        * 10_000
    )
    timed_mid = pd.Series(rows["mid"].to_numpy(), index=time_index)
    rows["range_60s_bps"] = (
        (timed_mid.rolling("60s").max() - timed_mid.rolling("60s").min()) / timed_mid * 10_000
    ).to_numpy()
    rows["depth_change_5s"] = rows["depth_5"].pct_change(5)
    for side in ("bid", "ask"):
        depth = pd.Series(rows[f"{side}_depth_5bps"].to_numpy(), index=time_index)
        removed = (-depth.diff()).clip(lower=0).rolling("5s").sum()
        rows[f"{side}_cancel_rate_5s"] = (
            removed / depth.rolling("5s").mean().replace(0, np.nan)
        ).to_numpy()
    trade_prices = rows["aggregate_trades"].map(
        lambda trades: [
            float(trade[0]) for trade in (trades if isinstance(trades, list) else [])
        ]
    )
    rows["trade_open"] = trade_prices.map(lambda prices: prices[0] if prices else np.nan)
    rows["trade_high"] = trade_prices.map(lambda prices: max(prices) if prices else np.nan)
    rows["trade_low"] = trade_prices.map(lambda prices: min(prices) if prices else np.nan)
    rows["trade_close"] = trade_prices.map(lambda prices: prices[-1] if prices else np.nan)
    base_volume = rows["aggregate_trades"].map(
        lambda trades: sum(
            float(trade[1]) for trade in (trades if isinstance(trades, list) else [])
        )
    )
    quote_volume = rows["aggregate_trades"].map(
        lambda trades: sum(
            float(trade[0]) * float(trade[1])
            for trade in (trades if isinstance(trades, list) else [])
        )
    )
    rolling_base = (
        pd.Series(base_volume.to_numpy(), index=time_index).rolling("300s", min_periods=285).sum()
    )
    rolling_quote = (
        pd.Series(quote_volume.to_numpy(), index=time_index).rolling("300s", min_periods=285).sum()
    )
    rolling_vwap = rolling_quote / rolling_base.replace(0, np.nan)
    rows["second"] = rows["exchange_timestamp"].dt.floor("s")
    rows["base_volume"] = base_volume
    rows["quote_volume"] = quote_volume
    rows["rolling_vwap"] = rolling_vwap.to_numpy()
    rows["rolling_vwap_5m_distance_bps"] = (
        rows["mid"].to_numpy() / rolling_vwap.to_numpy() - 1
    ) * 10_000
    anchored = _causal_anchor_features(rows, base_volume, quote_volume, time_index)
    rows[list(anchored.columns)] = anchored
    rows["anchored_vwap_distance_bps"] = (rows["mid"] / rows["anchored_vwap"] - 1) * 10_000
    reset = rows["anchor_age_seconds"].eq(0) | rows["anchor_age_seconds"].lt(
        rows["anchor_age_seconds"].shift(1)
    )
    rows["anchor_type"] = "MAJOR_IMPULSE"
    rows["anchor_timestamp"] = rows["available_at"].where(reset).ffill()
    rows["anchor_price"] = rows["mid"].where(reset).ffill()
    rows["anchor_age_bars"] = rows["anchor_age_seconds"]
    anchor_group = reset.cumsum()
    rows["volume_since_anchor"] = rows["quote_volume"].groupby(anchor_group).cumsum()
    rows["return_since_anchor_bps"] = (rows["mid"] / rows["anchor_price"] - 1) * 10_000
    rows["avwap_slope_bps_60s"] = rows["anchored_vwap"].pct_change(60) * 10_000
    rows["avwap_slope_change_bps"] = rows["avwap_slope_bps_60s"].diff(60)
    anchor_touch = rows["anchored_vwap_distance_bps"].abs().le(1.0)
    anchor_rejection = anchor_touch.shift(1, fill_value=False) & rows[
        "anchored_vwap_distance_bps"
    ].abs().gt(2.0)
    rows["avwap_tests"] = anchor_touch.groupby(anchor_group).cumsum()
    rows["avwap_rejections"] = anchor_rejection.groupby(anchor_group).cumsum()
    rows["avwap_rejection_strength_bps"] = (
        rows["anchored_vwap_distance_bps"].abs().where(anchor_rejection, 0.0)
    )
    rows["rolling_vwap_vs_avwap_distance_bps"] = (
        rows["rolling_vwap"] / rows["anchored_vwap"] - 1
    ) * 10_000
    rows["rolling_vwap_avwap_convergence_bps"] = (
        -rows["rolling_vwap_vs_avwap_distance_bps"].abs().diff()
    )
    rows = _add_market_structure(rows).copy()
    rows["market_regime"] = np.select(
        [
            rows["volatility_percentile"].ge(0.99),
            rows["volatility_percentile"].ge(0.75),
            rows["volatility_percentile"].le(0.25),
        ],
        ["STRESS", "HIGH_VOL", "LOW_VOL"],
        default="NORMAL",
    )
    clock_drift_ms = (rows["available_at"] - rows["exchange_timestamp"]).dt.total_seconds() * 1_000
    seconds_since_book = rows["exchange_timestamp"].diff().dt.total_seconds()
    rows["clock_drift_ms"] = clock_drift_ms
    rows["last_book_update_age_ms"] = clock_drift_ms
    rows["last_trade_update_age_ms"] = clock_drift_ms
    rows["sequence_gap_detected"] = seconds_since_book.gt(5)
    rows["book_is_synced"] = bid.lt(ask)
    rows["trade_feed_alive"] = clock_drift_ms.le(5_000)
    rows["orderbook_feed_alive"] = seconds_since_book.fillna(0).le(5)
    observed = pd.Series(1.0, index=time_index).rolling("60s").sum().to_numpy()
    rows["coverage_60s"] = observed / 60
    rows["feature_valid"] = (
        rows["coverage_60s"].ge(0.95)
        & rows["book_is_synced"]
        & rows["trade_feed_alive"]
        & rows["orderbook_feed_alive"]
        & ~rows["sequence_gap_detected"]
        & rows[list(FEATURES)].notna().all(axis=1)
    )
    for horizon in LABEL_HORIZONS_SECONDS:
        future_mid = rows["mid"].shift(-horizon)
        future_second = rows["exchange_second"].shift(-horizon)
        exact = future_second.sub(rows["exchange_second"]).eq(horizon)
        rows[f"future_return_{horizon}s_bps"] = ((future_mid / rows["mid"] - 1) * 10_000).where(
            exact
        )
        rows[f"label_available_at_{horizon}s"] = rows["available_at"].shift(-horizon).where(exact)
    return rows.drop(
        columns=["aggregate_trades", "trade_open", "trade_high", "trade_low", "trade_close"],
        errors="ignore",
    )


def materialize(root: Path = ROOT) -> Path:
    output = root / "btcusdt_l2_features.parquet"
    frame = build_features(load_records(root))
    temporary = output.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output


if __name__ == "__main__":
    print(materialize())
