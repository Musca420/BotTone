from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_windows(
    frame: pd.DataFrame,
    *,
    train_weeks: int = 12,
    validation_weeks: int = 4,
    test_weeks: int = 4,
    step_weeks: int = 4,
    holdout_weeks: int = 4,
) -> tuple[WalkForwardWindow, ...]:
    """Return chronological windows, reserving the final holdout completely."""
    if frame.empty:
        return ()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = timestamps.min()
    usable_end = timestamps.max() - timedelta(weeks=holdout_weeks)
    windows: list[WalkForwardWindow] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(weeks=train_weeks)
        validation_end = train_end + timedelta(weeks=validation_weeks)
        test_end = validation_end + timedelta(weeks=test_weeks)
        if test_end > usable_end:
            break
        windows.append(WalkForwardWindow(cursor, train_end, validation_end, test_end))
        cursor += timedelta(weeks=step_weeks)
    return tuple(windows)
