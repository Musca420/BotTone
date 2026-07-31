from __future__ import annotations

from typing import Protocol

from adaptive_bot.domain.models import Position, Signal, StrategyState
from adaptive_bot.strategy.signals import MarketSnapshot


class Strategy(Protocol):
    def evaluate(
        self, snapshot: MarketSnapshot, state: StrategyState, position: Position | None
    ) -> Signal | None: ...
