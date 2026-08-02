from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig


async def run_bitunix_paper(
    config: AppConfig,
    input_path: str | Path,
    output_path: str | Path,
    *,
    duration_hours: float = 168,
    poll_seconds: float = 15,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    if duration_hours <= 0 or poll_seconds <= 0:
        raise ValueError("paper duration and poll interval must be positive")
    source = Path(input_path)
    output = Path(output_path)
    frame = read_collected_candles(source)
    baseline_path = output.with_suffix(".start")
    baseline = _baseline(baseline_path, frame)
    deadline = monotonic() + duration_hours * 3600
    last_written: pd.Timestamp | None = None
    while True:
        frame = read_collected_candles(source)
        latest = pd.Timestamp(frame["timestamp"].iloc[-1])
        future = frame[frame["timestamp"] > baseline]
        if last_written != latest and (future.empty or future["spread_bps"].notna().all()):
            warmup_start = baseline - pd.Timedelta(
                minutes=config.strategy.timeframe_minutes * (config.strategy.crypto_vwap_window - 1)
            )
            replay = frame[frame["timestamp"] >= warmup_start].reset_index(drop=True)
            result = await BacktestEngine(config).run(
                replay,
                trade_after=baseline.to_pydatetime(),
                mode="paper",
            )
            result.write_json(output)
            last_written = latest
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(poll_seconds, remaining))


def read_collected_candles(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        envelope = json.loads(line)
        candle = envelope["candle"]
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(int(candle["time"]) / 1000, UTC),
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle.get("baseVol", candle.get("volume", "0")),
                "spread_bps": envelope.get("spread_bps"),
            }
        )
    if not rows:
        raise ValueError("Bitunix paper mode requires collected candles")
    frame = pd.DataFrame(rows).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Bitunix OHLCV values must be positive")
    outside = (
        (frame["open"] < frame["low"])
        | (frame["open"] > frame["high"])
        | (frame["close"] < frame["low"])
        | (frame["close"] > frame["high"])
    )
    envelope_high = frame[["open", "high", "close"]].max(axis=1)
    envelope_low = frame[["open", "low", "close"]].min(axis=1)
    deviation_bps = (
        (
            (envelope_high - frame["high"]).clip(lower=0)
            + (frame["low"] - envelope_low).clip(lower=0)
        )
        / frame["close"]
        * 10_000
    )
    if (deviation_bps[outside] > 1).any():
        raise ValueError("Bitunix OHLC deviation exceeds the 1 bps quarantine boundary")
    frame.loc[outside, "high"] = envelope_high[outside]
    frame.loc[outside, "low"] = envelope_low[outside]
    frame["ohlc_adjusted"] = outside
    return frame.reset_index(drop=True)


def _baseline(path: Path, frame: pd.DataFrame) -> pd.Timestamp:
    if path.exists():
        baseline = pd.Timestamp(path.read_text(encoding="utf-8").strip())
        if baseline.tzinfo is None:
            raise ValueError("paper baseline must include a timezone")
        return baseline.tz_convert("UTC")
    baseline = pd.Timestamp(frame["timestamp"].iloc[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(baseline.isoformat(), encoding="utf-8")
    temporary.replace(path)
    return baseline
