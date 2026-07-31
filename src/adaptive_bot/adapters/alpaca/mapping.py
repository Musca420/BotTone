from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from adaptive_bot.domain.enums import OrderStatus, OrderType, Side
from adaptive_bot.domain.models import AccountSnapshot, Candle, Order, Position, Quote

ORDER_STATUSES = {
    "new": OrderStatus.ACKNOWLEDGED,
    "accepted": OrderStatus.ACKNOWLEDGED,
    "pending_new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "pending_cancel": OrderStatus.CANCEL_REQUESTED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


def candle_from_alpaca(bar: Any, *, received_at: datetime | None = None) -> Candle:
    exchange = _utc(bar.timestamp)
    received = received_at or datetime.now(UTC)
    return Candle(
        exchange_timestamp=exchange,
        received_timestamp=max(exchange, _utc(received)),
        source="alpaca",
        instrument=str(bar.symbol),
        open=_decimal(bar.open),
        high=_decimal(bar.high),
        low=_decimal(bar.low),
        close=_decimal(bar.close),
        volume=_decimal(bar.volume),
        timeframe_minutes=1,
    )


def quote_from_alpaca(quote: Any, *, received_at: datetime | None = None) -> Quote:
    exchange = _utc(quote.timestamp)
    received = received_at or datetime.now(UTC)
    return Quote(
        exchange_timestamp=exchange,
        received_timestamp=max(exchange, _utc(received)),
        source="alpaca",
        instrument=str(quote.symbol),
        bid=_decimal(quote.bid_price),
        ask=_decimal(quote.ask_price),
        bid_size=_decimal(quote.bid_size),
        ask_size=_decimal(quote.ask_size),
    )


def account_from_alpaca(account: Any, *, received_at: datetime | None = None) -> AccountSnapshot:
    return AccountSnapshot(
        timestamp=_utc(received_at or datetime.now(UTC)),
        account_id=str(account.id),
        equity=_decimal(account.equity),
        cash=_decimal(account.cash),
        buying_power=_decimal(account.buying_power),
        unrealized_pnl=_decimal(getattr(account, "unrealized_pl", 0)),
    )


def order_from_alpaca(order: Any, *, received_at: datetime | None = None) -> Order:
    now = _utc(received_at or datetime.now(UTC))
    exchange = _utc(order.updated_at or order.submitted_at or order.created_at or now)
    status = ORDER_STATUSES.get(_value(order.status), OrderStatus.UNKNOWN)
    return Order(
        exchange_timestamp=exchange,
        received_timestamp=max(exchange, now),
        source="alpaca",
        instrument=str(order.symbol),
        client_order_id=str(order.client_order_id),
        side=Side(_value(order.side)),
        order_type=OrderType(_value(order.type)),
        quantity=_decimal(order.qty),
        filled_quantity=_decimal(order.filled_qty or 0),
        average_fill_price=(
            _decimal(order.filled_avg_price) if order.filled_avg_price is not None else None
        ),
        status=status,
        limit_price=_optional_decimal(order.limit_price),
        stop_price=_optional_decimal(order.stop_price),
        protective=_value(order.type) == "stop",
    )


def position_from_alpaca(position: Any, *, received_at: datetime | None = None) -> Position:
    timestamp = _utc(received_at or datetime.now(UTC))
    return Position(
        instrument=str(position.symbol),
        quantity=abs(_decimal(position.qty)),
        side=Side.BUY if _value(position.side) == "long" else Side.SELL,
        average_entry_price=_decimal(position.avg_entry_price),
        opened_at=timestamp,
    )


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _optional_decimal(value: Any) -> Decimal | None:
    return None if value is None else _decimal(value)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Alpaca timestamp must include a timezone")
    return value.astimezone(UTC)
