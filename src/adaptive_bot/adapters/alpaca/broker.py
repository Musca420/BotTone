from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout as RequestsTimeout

from adaptive_bot.adapters.alpaca.mapping import (
    account_from_alpaca,
    order_from_alpaca,
    position_from_alpaca,
)
from adaptive_bot.domain.enums import OrderType, Side
from adaptive_bot.domain.models import AccountSnapshot, Order, OrderRequest, Position


class AlpacaPaperBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        allowed_accounts: tuple[str, ...],
        allowed_instruments: tuple[str, ...] = ("QQQ",),
        *,
        client: Any | None = None,
    ) -> None:
        self._client = client or TradingClient(api_key, secret_key, paper=True)
        self._allowed_accounts = allowed_accounts
        self._allowed_instruments = allowed_instruments

    async def verify_account(self) -> AccountSnapshot:
        account = account_from_alpaca(await asyncio.to_thread(self._client.get_account))
        if account.account_id not in self._allowed_accounts:
            raise PermissionError("Alpaca account is not allowlisted")
        return account

    async def get_account(self) -> AccountSnapshot:
        return await self.verify_account()

    async def submit_order(self, request: OrderRequest) -> Order:
        await self.verify_account()
        existing = await self.get_order(request.client_order_id)
        if existing is not None:
            return existing
        if request.reduce_only:
            await self._verify_reduce_only(request)
        payload = self._request(request)
        try:
            raw = await asyncio.to_thread(self._client.submit_order, payload)
        except (TimeoutError, ConnectionError, RequestsTimeout, RequestsConnectionError):
            recovered = await self.get_order(request.client_order_id)
            if recovered is not None:
                return recovered
            raise
        return order_from_alpaca(raw)

    async def submit_bracket_entry(
        self, request: OrderRequest, take_profit_price: Decimal
    ) -> Order:
        if request.order_type is not OrderType.MARKET or request.stop_price is None:
            raise ValueError("bracket entry requires a market request and protective stop")
        await self.verify_account()
        existing = await self.get_order(request.client_order_id)
        if existing is not None:
            return existing
        payload = MarketOrderRequest(
            symbol=request.instrument,
            qty=str(request.quantity),
            side=OrderSide(request.side.value),
            time_in_force=TimeInForce(request.time_in_force.value),
            client_order_id=request.client_order_id,
            extended_hours=False,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=str(take_profit_price)),
            stop_loss=StopLossRequest(stop_price=str(request.stop_price)),
        )
        try:
            raw = await asyncio.to_thread(self._client.submit_order, payload)
        except (TimeoutError, ConnectionError, RequestsTimeout, RequestsConnectionError):
            recovered = await self.get_order(request.client_order_id)
            if recovered is not None:
                return recovered
            raise
        return order_from_alpaca(raw)

    async def cancel_order(self, client_order_id: str) -> Order:
        order = await self._get_raw_order(client_order_id)
        if order is None:
            raise KeyError(client_order_id)
        await asyncio.to_thread(self._client.cancel_order_by_id, order.id)
        refreshed = await self._get_raw_order(client_order_id)
        return order_from_alpaca(refreshed or order)

    async def get_order(self, client_order_id: str) -> Order | None:
        raw = await self._get_raw_order(client_order_id)
        return order_from_alpaca(raw) if raw is not None else None

    async def get_positions(self) -> tuple[Position, ...]:
        values = await asyncio.to_thread(self._client.get_all_positions)
        now = datetime.now(UTC)
        return tuple(position_from_alpaca(value, received_at=now) for value in values)

    async def get_open_orders(self, instrument: str) -> tuple[Order, ...]:
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
            symbols=[instrument],
        )
        values = await asyncio.to_thread(self._client.get_orders, request)
        flattened = []
        for value in values:
            flattened.append(value)
            flattened.extend(getattr(value, "legs", None) or ())
        return tuple(order_from_alpaca(value) for value in flattened)

    async def cancel_non_protective_orders(self) -> None:
        values = await asyncio.to_thread(self._client.get_orders)
        for value in values:
            if str(getattr(value.type, "value", value.type)) != "stop":
                await asyncio.to_thread(self._client.cancel_order_by_id, value.id)

    async def flatten_all(self) -> None:
        positions = await self.get_positions()
        if any(position.instrument not in self._allowed_instruments for position in positions):
            raise PermissionError("cannot flatten a position outside the instrument allowlist")
        if not positions:
            return
        try:
            await asyncio.to_thread(self._client.close_all_positions, True)
        except (TimeoutError, ConnectionError, RequestsTimeout, RequestsConnectionError):
            if not await self.get_positions():
                return
            raise

    async def _get_raw_order(self, client_order_id: str) -> Any | None:
        try:
            return await asyncio.to_thread(self._client.get_order_by_client_id, client_order_id)
        except APIError as error:
            if getattr(error, "status_code", None) == 404:
                return None
            raise

    async def _verify_reduce_only(self, request: OrderRequest) -> None:
        positions = await self.get_positions()
        position = next((item for item in positions if item.instrument == request.instrument), None)
        closing_side = Side.SELL if position and position.side is Side.BUY else Side.BUY
        if (
            position is None
            or request.side is not closing_side
            or request.quantity > position.quantity
        ):
            raise ValueError("reduce-only order does not match the broker position")

    @staticmethod
    def _request(request: OrderRequest) -> Any:
        common = {
            "symbol": request.instrument,
            "qty": str(request.quantity),
            "side": OrderSide(request.side.value),
            "time_in_force": TimeInForce(request.time_in_force.value),
            "client_order_id": request.client_order_id,
            "extended_hours": False,
        }
        if request.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if request.order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=str(request.limit_price), **common)
        return StopOrderRequest(stop_price=str(request.stop_price), **common)
