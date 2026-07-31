from decimal import Decimal


def loss_limit_breached(
    current_equity: Decimal, reference_equity: Decimal, maximum_loss: Decimal
) -> bool:
    if reference_equity <= 0:
        return True
    return current_equity <= reference_equity * (Decimal("1") - maximum_loss)


def drawdown(current_equity: Decimal, peak_equity: Decimal) -> Decimal:
    if peak_equity <= 0:
        return Decimal("1")
    return max(Decimal("0"), (peak_equity - current_equity) / peak_equity)
