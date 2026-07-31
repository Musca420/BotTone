from typing import Protocol


class AlertSink(Protocol):
    def critical(self, message: str) -> None: ...
