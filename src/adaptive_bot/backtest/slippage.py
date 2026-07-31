from decimal import Decimal


def apply_slippage(price: Decimal, bps: Decimal, *, adverse_sign: int) -> Decimal:
    return price * (Decimal("1") + Decimal(adverse_sign) * bps / Decimal("10000"))
