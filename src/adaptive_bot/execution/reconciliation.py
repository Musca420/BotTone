from __future__ import annotations

from dataclasses import dataclass

from adaptive_bot.domain.models import Position


@dataclass(frozen=True)
class ReconciliationResult:
    reconciled: bool
    reason: str


def reconcile_position(local: Position | None, broker: Position | None) -> ReconciliationResult:
    if local == broker:
        return ReconciliationResult(True, "positions match")
    return ReconciliationResult(False, "local and broker positions differ")
