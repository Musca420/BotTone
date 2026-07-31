from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from adaptive_bot.domain.enums import MarketRegime, Side
from adaptive_bot.domain.models import Candle


@dataclass(frozen=True)
class MarketSnapshot:
    candle: Candle
    atr: Decimal
    center: Decimal
    z_score: float
    spread_bps: float
    regime: MarketRegime
    session_open: datetime
    session_close: datetime
    data_reliable: bool = True
    force_exit_reason: str | None = None


def initial_stop(entry: Decimal, atr_value: Decimal, side: Side, multiple: Decimal) -> Decimal:
    distance = atr_value * multiple
    stop = entry - distance if side is Side.BUY else entry + distance
    if stop <= 0:
        raise ValueError("stop must be positive")
    return stop


def tighten_stop(current: Decimal, proposed: Decimal, side: Side) -> Decimal:
    if side is Side.BUY and proposed < current:
        raise ValueError("long stop cannot be widened")
    if side is Side.SELL and proposed > current:
        raise ValueError("short stop cannot be widened")
    return proposed
