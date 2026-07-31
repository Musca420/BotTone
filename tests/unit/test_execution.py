from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from adaptive_bot.adapters.simulated.broker import SimulatedBroker
from adaptive_bot.domain.enums import OrderStatus, OrderType, Side
from adaptive_bot.domain.exceptions import InvalidOrderTransition
from adaptive_bot.domain.models import OrderRequest
from adaptive_bot.execution.state_machine import transition
from tests.conftest import candle


def request(identifier: str, order_type: OrderType, side: Side, **prices: Decimal) -> OrderRequest:
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    return OrderRequest(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        client_order_id=identifier,
        side=side,
        order_type=order_type,
        quantity=Decimal("10"),
        **prices,
    )


def test_state_machine_rejects_invalid_transition() -> None:
    assert transition(OrderStatus.CREATED, OrderStatus.VALIDATED) is OrderStatus.VALIDATED
    with pytest.raises(InvalidOrderTransition):
        transition(OrderStatus.CREATED, OrderStatus.FILLED)


@pytest.mark.asyncio
async def test_broker_is_idempotent_and_never_fills_signal_candle(instrument) -> None:  # type: ignore[no-untyped-def]
    broker = SimulatedBroker(instrument, Decimal("100000"))
    order_request = request("entry", OrderType.MARKET, Side.BUY)
    first = await broker.submit_order(order_request)
    assert await broker.submit_order(order_request) == first
    assert await broker.process_candle(candle(order_request.exchange_timestamp)) == ()
    fills = await broker.process_candle(
        candle(order_request.exchange_timestamp + timedelta(minutes=15))
    )
    assert len(fills) == 1 and broker.position is not None
    assert fills[0].price % instrument.tick_size == 0
    assert fills[0].commission == Decimal("0.05")
    assert fills[0].slippage > 0


@pytest.mark.asyncio
async def test_stop_wins_when_stop_and_target_touch_same_candle(instrument) -> None:  # type: ignore[no-untyped-def]
    broker = SimulatedBroker(instrument, Decimal("100000"))
    await broker.submit_order(request("entry", OrderType.MARKET, Side.BUY))
    await broker.submit_order(
        request(
            "stop",
            OrderType.STOP,
            Side.SELL,
            stop_price=Decimal("95"),
        ).model_copy(update={"reduce_only": True, "protective": True})
    )
    await broker.submit_order(
        request("target", OrderType.LIMIT, Side.SELL, limit_price=Decimal("105")).model_copy(
            update={"reduce_only": True}
        )
    )
    fills = await broker.process_candle(
        candle(
            datetime(2026, 1, 5, 15, 15, tzinfo=UTC),
            high="106",
            low="94",
        )
    )
    assert [fill.client_order_id for fill in fills] == ["entry", "stop"]
    assert broker.position is None
