from decimal import Decimal


def estimated_round_trip_cost_per_unit(
    price: Decimal,
    spread_bps: Decimal,
    slippage_bps: Decimal,
    commission_per_unit: Decimal,
    fee_bps: Decimal = Decimal("0"),
) -> Decimal:
    market_impact = price * (spread_bps + slippage_bps * 2 + fee_bps) / Decimal("10000")
    return market_impact + commission_per_unit * 2
