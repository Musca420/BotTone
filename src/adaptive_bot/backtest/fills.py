from decimal import Decimal


def worst_case_stop_or_target(
    *,
    low: Decimal,
    high: Decimal,
    stop: Decimal,
    target: Decimal,
) -> str | None:
    stop_hit = low <= stop
    target_hit = high >= target
    if stop_hit:
        return "stop"
    if target_hit:
        return "target"
    return None
