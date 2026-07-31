from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from adaptive_bot.domain.enums import KillSwitchCause
from adaptive_bot.domain.models import OrderRequest


@dataclass(frozen=True)
class KillSwitchEvent:
    cause: KillSwitchCause
    timestamp: datetime
    details: str
    close_position: bool


class KillSwitch:
    def __init__(self) -> None:
        self._event: KillSwitchEvent | None = None

    @property
    def active(self) -> bool:
        return self._event is not None

    @property
    def event(self) -> KillSwitchEvent | None:
        return self._event

    def trigger(
        self,
        cause: KillSwitchCause,
        timestamp: datetime,
        details: str,
        *,
        close_position: bool = False,
    ) -> KillSwitchEvent:
        if self._event is None:
            self._event = KillSwitchEvent(cause, timestamp, details, close_position)
        return self._event

    def allows(self, request: OrderRequest) -> bool:
        return not self.active or request.protective or request.reduce_only

    def reset(self, *, actor: str, reason: str) -> None:
        if not actor.strip() or not reason.strip():
            raise ValueError("manual reset requires actor and reason")
        self._event = None
