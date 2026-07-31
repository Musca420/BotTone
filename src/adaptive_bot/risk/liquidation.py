from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class LiquidationCheck:
    leverage: Decimal
    mark_price: Decimal
    stop_price: Decimal
    broker_liquidation_price: Decimal | None
    estimated_liquidation_price: Decimal | None
    extra_buffer: Decimal = Decimal("0")
    minimum_multiple: Decimal = Decimal("3")
    maximum_estimate_difference: Decimal = Decimal("0.05")


def liquidation_buffer_valid(check: LiquidationCheck) -> bool:
    if check.leverage <= 1:
        return True
    if check.broker_liquidation_price is None or check.estimated_liquidation_price is None:
        return False
    broker = check.broker_liquidation_price
    estimate = check.estimated_liquidation_price
    if broker <= 0 or estimate <= 0 or check.mark_price <= 0:
        return False
    relative_difference = abs(broker - estimate) / broker
    if relative_difference > check.maximum_estimate_difference:
        return False
    liquidation_distance = abs(check.mark_price - broker) - check.extra_buffer
    stop_distance = abs(check.mark_price - check.stop_price)
    return liquidation_distance >= check.minimum_multiple * stop_distance
