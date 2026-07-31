from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Instrument, RiskDecision


@dataclass(frozen=True)
class SizingInput:
    equity: Decimal
    buying_power: Decimal
    entry_price: Decimal
    stop_price: Decimal
    estimated_cost_per_unit: Decimal
    risk_fraction: Decimal
    hard_notional_cap: Decimal
    side: Side


def floor_to_lot(quantity: Decimal, lot_size: Decimal) -> Decimal:
    if quantity <= 0:
        return Decimal("0")
    return (quantity / lot_size).to_integral_value(rounding=ROUND_FLOOR) * lot_size


def size_position(inputs: SizingInput, instrument: Instrument) -> RiskDecision:
    if inputs.equity <= 0 or inputs.buying_power <= 0:
        return RiskDecision(approved=False, reason="non-positive equity or buying power")
    if inputs.side is Side.BUY and inputs.stop_price >= inputs.entry_price:
        return RiskDecision(approved=False, reason="long stop must be below entry")
    if inputs.side is Side.SELL and inputs.stop_price <= inputs.entry_price:
        return RiskDecision(approved=False, reason="short stop must be above entry")

    budget = inputs.equity * inputs.risk_fraction
    risk_per_unit = (
        abs(inputs.entry_price - inputs.stop_price) * instrument.point_value
        + inputs.estimated_cost_per_unit
    )
    if budget <= 0 or risk_per_unit <= 0:
        return RiskDecision(approved=False, reason="invalid risk budget")

    raw_quantity = budget / risk_per_unit
    quantity_cap = min(
        raw_quantity,
        inputs.buying_power / inputs.entry_price,
        inputs.hard_notional_cap / inputs.entry_price,
    )
    quantity = floor_to_lot(quantity_cap, instrument.lot_size)
    effective = quantity * risk_per_unit
    notional = quantity * inputs.entry_price
    if quantity < instrument.minimum_quantity:
        return RiskDecision(
            approved=False,
            reason="quantity below instrument minimum",
            risk_budget=budget,
            effective_risk=effective,
        )
    if notional < instrument.minimum_notional:
        return RiskDecision(
            approved=False,
            reason="notional below instrument minimum",
            risk_budget=budget,
            effective_risk=effective,
        )
    if effective > budget:
        return RiskDecision(
            approved=False,
            reason="rounded risk exceeds budget",
            risk_budget=budget,
            effective_risk=effective,
        )
    return RiskDecision(
        approved=True,
        reason="approved",
        quantity=quantity,
        risk_budget=budget,
        effective_risk=effective,
    )
