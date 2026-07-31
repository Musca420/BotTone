from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from adaptive_bot.domain.enums import KillSwitchCause
from adaptive_bot.domain.models import Position
from adaptive_bot.execution.interfaces import Broker
from adaptive_bot.execution.reconciliation import reconcile_position
from adaptive_bot.risk.kill_switch import KillSwitch


@dataclass(frozen=True)
class RecoveryReport:
    reconciled: bool
    reason: str
    broker_position: Position | None


async def reconcile_before_trading(
    broker: Broker,
    instrument: str,
    local_position: Position | None,
    kill_switch: KillSwitch,
    timestamp: datetime,
) -> RecoveryReport:
    positions = await broker.get_positions()
    unexpected = [position for position in positions if position.instrument != instrument]
    matching = [position for position in positions if position.instrument == instrument]
    if unexpected or len(matching) > 1:
        event = kill_switch.trigger(
            KillSwitchCause.UNKNOWN_POSITION,
            timestamp,
            "broker contains an unexpected or duplicate position",
        )
        return RecoveryReport(False, event.details, None)
    broker_position = matching[0] if matching else None
    result = reconcile_position(local_position, broker_position)
    if not result.reconciled:
        kill_switch.trigger(
            KillSwitchCause.STATE_DIVERGENCE,
            timestamp,
            result.reason,
        )
    return RecoveryReport(result.reconciled, result.reason, broker_position)
