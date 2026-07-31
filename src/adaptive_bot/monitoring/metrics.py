from __future__ import annotations

from decimal import Decimal
from typing import Protocol


class MetricsSink(Protocol):
    def gauge(self, name: str, value: Decimal | float | int) -> None: ...

    def increment(self, name: str, value: int = 1) -> None: ...
