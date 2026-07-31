from __future__ import annotations

from adaptive_bot.domain.models import Order, OrderRequest
from adaptive_bot.execution.interfaces import Broker
from adaptive_bot.risk.kill_switch import KillSwitch


class OrderManager:
    def __init__(self, broker: Broker, kill_switch: KillSwitch) -> None:
        self.broker = broker
        self.kill_switch = kill_switch

    async def submit(self, request: OrderRequest) -> Order:
        existing = await self.broker.get_order(request.client_order_id)
        if existing is not None:
            return existing
        if not self.kill_switch.allows(request):
            raise PermissionError("kill switch blocks new orders")
        return await self.broker.submit_order(request)
