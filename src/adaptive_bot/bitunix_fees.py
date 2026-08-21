from __future__ import annotations

# Official Bitunix futures maker/taker rates, converted from percent to basis points.
FUTURES_VIP_FEE_BPS = {
    0: (2.0, 6.0),
    1: (2.0, 5.0),
    2: (1.6, 5.0),
    3: (1.4, 4.0),
    4: (1.2, 3.75),
    5: (1.0, 3.5),
    6: (0.8, 3.15),
    7: (0.6, 3.0),
    8: (0.0, 2.6),
}


def futures_fee_bps(vip_level: int) -> tuple[float, float]:
    try:
        return FUTURES_VIP_FEE_BPS[vip_level]
    except KeyError as error:
        raise ValueError("Bitunix VIP level must be between 0 and 8") from error


def expected_round_trip_cost_bps(
    *,
    vip_level: int,
    entry_role: str,
    exit_role: str,
    spread_bps: float,
    entry_slippage_bps: float = 0.0,
    exit_slippage_bps: float = 0.0,
    funding_bps: float = 0.0,
) -> float:
    """Estimate an explicit round trip from observed spread and configured roles."""
    if entry_role not in {"maker", "taker"} or exit_role not in {"maker", "taker"}:
        raise ValueError("Liquidity role must be 'maker' or 'taker'")
    values = (spread_bps, entry_slippage_bps, exit_slippage_bps)
    if any(value < 0 for value in values):
        raise ValueError("Spread and slippage costs cannot be negative")
    maker, taker = futures_fee_bps(vip_level)
    entry_fee = maker if entry_role == "maker" else taker
    exit_fee = maker if exit_role == "maker" else taker
    crossed_spread = 0.5 * spread_bps * (
        int(entry_role == "taker") + int(exit_role == "taker")
    )
    return (
        entry_fee
        + exit_fee
        + crossed_spread
        + entry_slippage_bps
        + exit_slippage_bps
        + funding_bps
    )
