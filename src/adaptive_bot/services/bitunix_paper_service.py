from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pandas as pd

from adaptive_bot.adapters.bitunix.market_data import _candle as map_bitunix_candle
from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig

ADX_VARIANTS = (20.0, 22.0, 23.0, 24.0, 25.0)
WEIGHTED_PROFILE = "mr_score"
WEIGHTED_V11_PROFILE = "mr_score_v11"
WEIGHTED_V2_PROFILE = "mr_score_v2"


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
            for threshold in ADX_VARIANTS:
                variant = config.model_copy(
                    update={
                        "strategy": config.strategy.model_copy(
                            update={"range_adx_threshold": threshold}
                        )
                    }
                )
                result = await BacktestEngine(variant).run(
                    replay,
                    trade_after=baseline.to_pydatetime(),
                    mode="paper",
                )
                result.write_json(variant_report_path(output, threshold))
                if threshold == ADX_VARIANTS[0]:
                    result.write_json(output)
            for profile, entry_mode, threshold in (
                (WEIGHTED_PROFILE, "weighted_reversion", 0.65),
                (WEIGHTED_V11_PROFILE, "weighted_reversion_v11", 0.65),
                (WEIGHTED_V2_PROFILE, "weighted_reversion_v2", 0.70),
            ):
                weighted = config.model_copy(
                    update={
                        "strategy": config.strategy.model_copy(
                            update={
                                "entry_mode": entry_mode,
                                "weighted_entry_threshold": threshold,
                            }
                        )
                    }
                )
                result = await BacktestEngine(weighted).run(
                    replay,
                    trade_after=baseline.to_pydatetime(),
                    mode="paper",
                )
                result.write_json(profile_report_path(output, profile))
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
        source = str(envelope.get("source", ""))
        normalized, _ = map_bitunix_candle(candle, "BTCUSDT", "futures", 5)
        rows.append(
            {
                "timestamp": normalized.exchange_timestamp,
                "open": normalized.open,
                "high": normalized.high,
                "low": normalized.low,
                "close": normalized.close,
                "volume": candle.get("quoteVol", candle.get("volume", "0")),
                "quote_volume": candle.get("baseVol"),
                "ohlc_adjusted": (
                    normalized.high != Decimal(str(candle["high"]))
                    or normalized.low != Decimal(str(candle["low"]))
                ),
                "price_type": (
                    "MARK_PRICE"
                    if "mark" in source
                    else str(candle.get("type", "LAST_PRICE"))
                ),
                "spread_bps": envelope.get("spread_bps"),
            }
        )
    if not rows:
        raise ValueError("Bitunix paper mode requires collected candles")
    frame = pd.DataFrame(rows).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Bitunix OHLCV values must be positive")
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


def variant_report_path(path: str | Path, threshold: float) -> Path:
    target = Path(path)
    label = f"{threshold:g}"
    return target.with_name(f"{target.stem}.adx{label}{target.suffix}")


def profile_report_path(path: str | Path, profile: str) -> Path:
    if profile.startswith("adx"):
        return variant_report_path(path, float(profile.removeprefix("adx")))
    target = Path(path)
    return target.with_name(f"{target.stem}.{profile}{target.suffix}")
