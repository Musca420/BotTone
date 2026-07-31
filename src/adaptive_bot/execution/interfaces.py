from __future__ import annotations

from typing import Protocol

from adaptive_bot.domain.models import AccountSnapshot, Order, OrderRequest, Position


class Broker(Protocol):
    async def submit_order(self, request: OrderRequest) -> Order: ...

    async def cancel_order(self, client_order_id: str) -> Order: ...

    async def get_order(self, client_order_id: str) -> Order | None: ...

    async def get_positions(self) -> tuple[Position, ...]: ...


class AccountProvider(Protocol):
    async def get_account(self) -> AccountSnapshot: ...
