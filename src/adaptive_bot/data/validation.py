from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

from adaptive_bot.domain.exceptions import DataQualityError

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    score: float
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_candles: int = 0

    def require(self, minimum_score: float) -> None:
        if not self.passed or self.score < minimum_score:
            raise DataQualityError("; ".join(self.errors) or "data quality below threshold")


def validate_candles(
    frame: pd.DataFrame,
    *,
    timeframe_minutes: int = 15,
    calendar_name: str | None = "NYSE",
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        return ValidationReport(False, 0.0, (f"missing columns: {', '.join(missing_columns)}",), ())
    if frame.empty:
        return ValidationReport(False, 0.0, ("dataset is empty",), ())

    if not frame["timestamp"].map(_has_timezone).all():
        errors.append("timestamps must include a timezone")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        errors.append("invalid timestamps")
    if timestamps.duplicated().any():
        errors.append("duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        errors.append("timestamps are out of order")

    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        errors.append("null or non-numeric OHLCV values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("prices must be positive")
    if (numeric["volume"] < 0).any():
        errors.append("volume must be non-negative")
    if (numeric["high"] < numeric["low"]).any():
        errors.append("high below low")
    if (
        (numeric["open"] < numeric["low"])
        | (numeric["open"] > numeric["high"])
        | (numeric["close"] < numeric["low"])
        | (numeric["close"] > numeric["high"])
    ).any():
        errors.append("open or close outside high-low")

    jumps = numeric["close"].pct_change(fill_method=None).abs() > 0.20
    acknowledged = (
        frame["corporate_action"].fillna(False).astype(bool)
        if "corporate_action" in frame
        else pd.Series(False, index=frame.index)
    )
    if (jumps & ~acknowledged).any():
        errors.append("unexplained price jump above 20%")
    if calendar_name is not None and not bool(frame.attrs.get("split_adjusted", False)):
        warnings.append("split-adjustment metadata is absent")

    missing = (
        _missing_rth_candles(timestamps.dropna(), timeframe_minutes, calendar_name)
        if calendar_name is not None
        else _missing_continuous_candles(timestamps.dropna(), timeframe_minutes)
    )
    if missing:
        errors.append(f"{missing} candles are missing")
    denominator = max(len(frame) + missing, 1)
    score = max(0.0, 1.0 - (len(errors) + missing) / denominator)
    return ValidationReport(not errors, score, tuple(errors), tuple(warnings), missing)


def _has_timezone(value: Any) -> bool:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _missing_rth_candles(timestamps: pd.Series, timeframe_minutes: int, calendar_name: str) -> int:
    if timestamps.empty:
        return 0
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(
        start_date=timestamps.iloc[0].date(), end_date=timestamps.iloc[-1].date()
    )
    if schedule.empty:
        return len(timestamps)
    expected = mcal.date_range(schedule, frequency=f"{timeframe_minutes}min")
    observed = pd.DatetimeIndex(timestamps)
    expected_in_span = expected[(expected >= observed.min()) & (expected <= observed.max())]
    return len(expected_in_span.difference(observed))


def _missing_continuous_candles(timestamps: pd.Series, timeframe_minutes: int) -> int:
    if timestamps.empty:
        return 0
    observed = pd.DatetimeIndex(timestamps)
    expected = pd.date_range(
        observed.min(), observed.max(), freq=f"{timeframe_minutes}min", tz="UTC"
    )
    return len(expected.difference(observed))
