from decimal import Decimal


def funding_payment(notional: Decimal, rate: Decimal, *, long_position: bool) -> Decimal:
    payment = notional * rate
    return -payment if long_position else payment
