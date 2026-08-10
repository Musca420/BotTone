from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from adaptive_bot.bitunix_fees import futures_fee_bps

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")


@dataclass(frozen=True)
class MarketIntegrity:
    last_trade_update_age_ms: float
    last_book_update_age_ms: float
    sequence_gap_detected: bool
    book_is_synced: bool
    trade_feed_alive: bool
    orderbook_feed_alive: bool
    clock_drift_ms: float

    def rejection_reason(self) -> str | None:
        if self.sequence_gap_detected or not self.book_is_synced:
            return "DESYNCHRONIZED_ORDERBOOK"
        if not self.trade_feed_alive or self.last_trade_update_age_ms > 5_000:
            return "STALE_TRADE_FEED"
        if not self.orderbook_feed_alive or self.last_book_update_age_ms > 5_000:
            return "STALE_ORDERBOOK"
        if abs(self.clock_drift_ms) > 5_000:
            return "CLOCK_DRIFT"
        return None


@dataclass(frozen=True)
class BookFill:
    execution_vwap: Decimal
    quantity: Decimal
    notional: Decimal
    slippage_bps: Decimal
    levels_consumed: int


@dataclass(frozen=True)
class ExecutionQuote:
    side: str
    entry: BookFill
    estimated_exit: BookFill
    vip_level: int
    entry_fee_bps: Decimal
    exit_fee_bps: Decimal
    spread_bps: Decimal
    expected_funding_bps: Decimal
    expected_total_cost_bps: Decimal
    execution_type: str = "TAKER_MARKET_DEPTH_VWAP"
    maker_status: str = "DISABLED_UNTIL_PRIVATE_FILL_LABELS"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskQuote:
    approved: bool
    reason: str
    risk_budget: Decimal
    notional: Decimal
    quantity: Decimal
    margin_required: Decimal
    estimated_loss_at_stop: Decimal
    leverage: Decimal
    technical_stop_price: Decimal
    catastrophic_stop_price: Decimal


def _decimal_levels(levels: Any) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(levels, list):
        return []
    parsed: list[tuple[Decimal, Decimal]] = []
    for level in levels:
        try:
            price, quantity = Decimal(str(level[0])), Decimal(str(level[1]))
        except (IndexError, TypeError, ValueError):
            continue
        if price > 0 and quantity > 0:
            parsed.append((price, quantity))
    return parsed


def walk_book(levels: Any, quantity: Decimal, *, reference_mid: Decimal) -> BookFill | None:
    """Consume observed depth in order and fail if the requested quantity cannot fill."""
    if quantity <= 0 or reference_mid <= 0:
        return None
    remaining, value = quantity, ZERO
    for consumed, (price, available) in enumerate(_decimal_levels(levels), start=1):
        filled = min(remaining, available)
        value += filled * price
        remaining -= filled
        if remaining <= 0:
            execution = value / quantity
            return BookFill(
                execution_vwap=execution,
                quantity=quantity,
                notional=value,
                slippage_bps=abs(execution / reference_mid - 1) * TEN_THOUSAND,
                levels_consumed=consumed,
            )
    return None


def quote_taker_round_trip(
    *,
    side: str,
    quantity: Decimal,
    bids: Any,
    asks: Any,
    best_bid: Decimal,
    best_ask: Decimal,
    vip_level: int = 0,
    taker_fee_bps: Decimal | None = None,
    expected_funding_bps: Decimal = ZERO,
) -> ExecutionQuote | None:
    if side not in {"LONG", "SHORT"} or best_bid <= 0 or best_ask <= best_bid:
        return None
    mid = (best_bid + best_ask) / 2
    entry_levels = asks if side == "LONG" else bids
    exit_levels = bids if side == "LONG" else asks
    entry = walk_book(entry_levels, quantity, reference_mid=mid)
    estimated_exit = walk_book(exit_levels, quantity, reference_mid=mid)
    if entry is None or estimated_exit is None:
        return None
    if taker_fee_bps is None:
        _, taker = futures_fee_bps(vip_level)
        fee = Decimal(str(taker))
    else:
        if not taker_fee_bps.is_finite() or taker_fee_bps < 0:
            return None
        fee = taker_fee_bps
    spread = (best_ask - best_bid) / mid * TEN_THOUSAND
    total = (
        2 * fee + entry.slippage_bps + estimated_exit.slippage_bps + max(ZERO, expected_funding_bps)
    )
    return ExecutionQuote(
        side=side,
        entry=entry,
        estimated_exit=estimated_exit,
        vip_level=vip_level,
        entry_fee_bps=fee,
        exit_fee_bps=fee,
        spread_bps=spread,
        expected_funding_bps=max(ZERO, expected_funding_bps),
        expected_total_cost_bps=total,
    )


def size_for_technical_stop(
    *,
    equity: Decimal,
    entry_price: Decimal,
    side: str,
    technical_stop_bps: Decimal,
    expected_cost_bps: Decimal,
    lot_size: Decimal,
    minimum_quantity: Decimal,
    minimum_notional: Decimal,
    risk_fraction: Decimal = Decimal("0.01"),
    margin_fraction: Decimal = Decimal("0.10"),
    maximum_leverage: Decimal = Decimal("10"),
    catastrophic_stop_bps: Decimal = Decimal("200"),
) -> RiskQuote:
    empty = RiskQuote(False, "INVALID_RISK_INPUT", ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)
    if (
        equity <= 0
        or entry_price <= 0
        or side not in {"LONG", "SHORT"}
        or technical_stop_bps <= 0
        or expected_cost_bps < 0
        or lot_size <= 0
    ):
        return empty
    risk_budget = equity * risk_fraction
    loss_bps = technical_stop_bps + expected_cost_bps
    risk_notional = risk_budget * TEN_THOUSAND / loss_bps
    maximum_notional = equity * margin_fraction * maximum_leverage
    notional = min(risk_notional, maximum_notional)
    quantity = (notional / entry_price / lot_size).to_integral_value(rounding=ROUND_DOWN) * lot_size
    notional = quantity * entry_price
    margin = notional / maximum_leverage
    estimated_loss = notional * loss_bps / TEN_THOUSAND
    direction = Decimal("1") if side == "LONG" else Decimal("-1")
    technical_stop = entry_price * (1 - direction * technical_stop_bps / TEN_THOUSAND)
    catastrophic_stop = entry_price * (1 - direction * catastrophic_stop_bps / TEN_THOUSAND)
    reason = "APPROVED"
    approved = True
    if technical_stop_bps > catastrophic_stop_bps:
        approved, reason = False, "TECHNICAL_STOP_BEYOND_CATASTROPHIC_STOP"
    elif quantity < minimum_quantity or notional < minimum_notional:
        approved, reason = False, "MINIMUM_ORDER_NOT_MET"
    elif margin > equity * margin_fraction:
        approved, reason = False, "MARGIN_LIMIT"
    elif estimated_loss > risk_budget:
        approved, reason = False, "RISK_BUDGET_EXCEEDED"
    return RiskQuote(
        approved,
        reason,
        risk_budget,
        notional,
        quantity,
        margin,
        estimated_loss,
        maximum_leverage,
        technical_stop,
        catastrophic_stop,
    )


def protected_stop(
    *,
    side: str,
    current_stop: Decimal,
    proposed_stop: Decimal,
    entry_price: Decimal,
    total_cost_bps: Decimal,
    favorable_excursion_bps: Decimal,
    minimum_net_profit_bps: Decimal = Decimal("8"),
) -> tuple[Decimal, str]:
    """Activate profit protection only after costs plus a real minimum net gain."""
    activation = total_cost_bps + minimum_net_profit_bps
    if favorable_excursion_bps < activation:
        return current_stop, "ZONE_A_OR_B_NO_TRAIL"
    economic_break_even = entry_price * (
        1 + (total_cost_bps / TEN_THOUSAND if side == "LONG" else -total_cost_bps / TEN_THOUSAND)
    )
    candidate = (
        max(proposed_stop, economic_break_even)
        if side == "LONG"
        else min(proposed_stop, economic_break_even)
    )
    tightened = max(current_stop, candidate) if side == "LONG" else min(current_stop, candidate)
    return tightened, "ZONE_C_OR_D_TRAIL_ACTIVE"
