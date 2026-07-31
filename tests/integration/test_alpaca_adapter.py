from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from requests import HTTPError, Response

from adaptive_bot.adapters.alpaca.broker import AlpacaPaperBroker
from adaptive_bot.adapters.alpaca.mapping import candle_from_alpaca, order_from_alpaca
from adaptive_bot.adapters.alpaca.market_data import AlpacaMarketData
from adaptive_bot.adapters.alpaca.trade_updates import AlpacaTradeUpdates
from adaptive_bot.domain.enums import OrderStatus, OrderType, Side
from adaptive_bot.domain.models import OrderRequest
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.services.recovery_service import reconcile_before_trading


def _account(account_id: str = "paper-account") -> SimpleNamespace:
    return SimpleNamespace(
        id=account_id,
        equity="100000",
        cash="100000",
        buying_power="100000",
        unrealized_pl="0",
    )


def _order(client_order_id: str = "order-1", status: str = "new") -> SimpleNamespace:
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    return SimpleNamespace(
        id="alpaca-order-id",
        symbol="QQQ",
        client_order_id=client_order_id,
        side=SimpleNamespace(value="buy"),
        type=SimpleNamespace(value="market"),
        qty="2",
        filled_qty="0",
        filled_avg_price=None,
        status=SimpleNamespace(value=status),
        limit_price=None,
        stop_price=None,
        updated_at=now,
        submitted_at=now,
        created_at=now,
    )


class FakeTradingClient:
    def __init__(self, *, existing: Any | None = None, account_id: str = "paper-account") -> None:
        self.existing = existing
        self.account = _account(account_id)
        self.submissions = 0
        self.last_request: Any | None = None

    def get_account(self) -> Any:
        return self.account

    def get_order_by_client_id(self, client_id: str) -> Any:
        del client_id
        if self.existing is None:
            response = Response()
            response.status_code = 404
            raise APIError('{"code":404,"message":"not found"}', HTTPError(response=response))
        return self.existing

    def submit_order(self, request: Any) -> Any:
        self.submissions += 1
        self.last_request = request
        self.existing = _order(str(request.client_order_id))
        return self.existing

    def get_all_positions(self) -> list[Any]:
        return []


class TimeoutAfterAcceptanceClient(FakeTradingClient):
    def submit_order(self, request: Any) -> Any:
        self.submissions += 1
        self.last_request = request
        self.existing = _order(str(request.client_order_id))
        raise TimeoutError("response lost")


def test_alpaca_mapping_preserves_decimal_and_utc() -> None:
    bar = SimpleNamespace(
        symbol="QQQ",
        timestamp=datetime(2026, 1, 5, 15, 0, tzinfo=UTC),
        open=500.1,
        high=501.2,
        low=499.4,
        close=500.8,
        volume=1234,
    )
    candle = candle_from_alpaca(bar, received_at=bar.timestamp)
    assert candle.close == Decimal("500.8")
    assert candle.exchange_timestamp.utcoffset().total_seconds() == 0
    assert order_from_alpaca(_order()).status is OrderStatus.ACKNOWLEDGED
    assert (
        order_from_alpaca(_order(status="partially_filled")).status is OrderStatus.PARTIALLY_FILLED
    )
    assert order_from_alpaca(_order(status="rejected")).status is OrderStatus.REJECTED


class DisconnectingStream:
    def subscribe_bars(self, handler: Any, *symbols: str) -> None:
        del handler, symbols

    def subscribe_quotes(self, handler: Any, *symbols: str) -> None:
        del handler, symbols

    def run(self) -> None:
        raise ConnectionError("websocket disconnected")

    def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_stream_disconnect_is_propagated() -> None:
    provider = AlpacaMarketData(
        "key",
        "secret",
        historical_client=SimpleNamespace(),
        stream_factory=DisconnectingStream,
    )
    with pytest.raises(ConnectionError, match="disconnected"):
        await anext(provider.stream("QQQ"))


class FillUpdateStream:
    def __init__(self) -> None:
        self.handler: Any | None = None

    def subscribe_trade_updates(self, handler: Any) -> None:
        self.handler = handler

    def run(self) -> None:
        assert self.handler is not None
        update = SimpleNamespace(
            event=SimpleNamespace(value="partial_fill"),
            order=_order(status="partially_filled"),
            timestamp=datetime(2026, 1, 5, 15, 1, tzinfo=UTC),
            price=500.25,
            qty=1,
        )
        asyncio.run(self.handler(update))

    def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_trade_updates_emit_order_then_fill() -> None:
    updates = AlpacaTradeUpdates("key", "secret", stream_factory=FillUpdateStream)
    stream = updates.stream()
    order = await anext(stream)
    fill = await anext(stream)
    await stream.aclose()
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert fill.quantity == Decimal("1")
    assert fill.price == Decimal("500.25")


@pytest.mark.asyncio
async def test_broker_returns_existing_order_without_duplicate() -> None:
    client = FakeTradingClient(existing=_order())
    broker = AlpacaPaperBroker("key", "secret", ("paper-account",), client=client)
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    request = OrderRequest(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        client_order_id="order-1",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("2"),
    )
    result = await broker.submit_order(request)
    assert result.client_order_id == "order-1"
    assert client.submissions == 0


@pytest.mark.asyncio
async def test_ambiguous_timeout_is_reconciled_without_retry() -> None:
    client = TimeoutAfterAcceptanceClient()
    broker = AlpacaPaperBroker("key", "secret", ("paper-account",), client=client)
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    request = OrderRequest(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        client_order_id="order-2",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("2"),
    )
    result = await broker.submit_order(request)
    assert result.client_order_id == "order-2"
    assert client.submissions == 1


@pytest.mark.asyncio
async def test_entry_is_submitted_as_atomic_bracket() -> None:
    client = FakeTradingClient()
    broker = AlpacaPaperBroker("key", "secret", ("paper-account",), client=client)
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    request = OrderRequest(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        client_order_id="bracket-1",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("2"),
        stop_price=Decimal("495"),
    )
    await broker.submit_bracket_entry(request, Decimal("505"))
    assert client.last_request.order_class.value == "bracket"
    assert client.last_request.stop_loss.stop_price == 495.0
    assert client.last_request.take_profit.limit_price == 505.0


@pytest.mark.asyncio
async def test_account_allowlist_and_clean_reconciliation() -> None:
    broker = AlpacaPaperBroker(
        "key", "secret", ("paper-account",), client=FakeTradingClient(existing=_order())
    )
    kill_switch = KillSwitch()
    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    report = await reconcile_before_trading(broker, "QQQ", None, kill_switch, now)
    assert report.reconciled
    assert not kill_switch.active

    blocked = AlpacaPaperBroker("key", "secret", ("different-account",), client=FakeTradingClient())
    with pytest.raises(PermissionError, match="allowlisted"):
        await blocked.verify_account()
