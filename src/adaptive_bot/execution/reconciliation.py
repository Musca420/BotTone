from __future__ import annotations

from dataclasses import dataclass

from adaptive_bot.domain.models import Position


@dataclass(frozen=True)
class ReconciliationResult:
    reconciled: bool
    reason: str


def reconcile_position(local: Position | None, broker: Position | None) -> ReconciliationResult:
    if local is None or broker is None:
        matched = local is broker
    else:
        matched = (
            local.instrument == broker.instrument
            and local.quantity == broker.quantity
            and local.side is broker.side
            and local.average_entry_price == broker.average_entry_price
        )
    if matched:
        return ReconciliationResult(True, "positions match")
    return ReconciliationResult(False, "local and broker positions differ")
