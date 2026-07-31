from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from adaptive_bot.domain.enums import OrderStatus, OrderType, Side
from adaptive_bot.domain.models import (
    AccountSnapshot,
    Candle,
    Fill,
    Instrument,
    Order,
    OrderRequest,
    Position,
)
from adaptive_bot.execution.state_machine import transition


class SimulatedBroker:
    def __init__(
        self,
        instrument: Instrument,
        initial_cash: Decimal,
        *,
        spread_bps: Decimal = Decimal("2"),
        slippage_bps: Decimal = Decimal("1"),
        commission_per_unit: Decimal = Decimal("0.005"),
        max_volume_participation: Decimal = Decimal("0.10"),
    ) -> None:
        self.instrument = instrument
        self.cash = initial_cash
        self.realized_pnl = Decimal("0")
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.commission_per_unit = commission_per_unit
        self.max_volume_participation = max_volume_participation
        self.orders: dict[str, Order] = {}
        self.position: Position | None = None
        self.last_price = Decimal("0")
        self.current_time = datetime.now(UTC)
        self.fills: list[Fill] = []

    async def submit_order(self, request: OrderRequest) -> Order:
        if request.client_order_id in self.orders:
            return self.orders[request.client_order_id]
        status = transition(OrderStatus.CREATED, OrderStatus.VALIDATED)
        status = transition(status, OrderStatus.SUBMITTED)
        status = transition(status, OrderStatus.ACKNOWLEDGED)
        order = Order(
            exchange_timestamp=request.exchange_timestamp,
            received_timestamp=request.received_timestamp,
            source="simulated",
            instrument=request.instrument,
            sequence_number=request.sequence_number,
            correlation_id=request.correlation_id,
            client_order_id=request.client_order_id,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            reduce_only=request.reduce_only,
            protective=request.protective,
            status=status,
        )
        self.orders[order.client_order_id] = order
        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        order = self.orders[client_order_id]
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}:
            return order
        status = transition(order.status, OrderStatus.CANCEL_REQUESTED)
        status = transition(status, OrderStatus.CANCELED)
        order = order.model_copy(update={"status": status})
        self.orders[client_order_id] = order
        return order

    async def get_order(self, client_order_id: str) -> Order | None:
        return self.orders.get(client_order_id)

    async def get_positions(self) -> tuple[Position, ...]:
        return (self.position,) if self.position is not None else ()

    async def get_account(self) -> AccountSnapshot:
        mark = self.last_price
        position_value = Decimal("0")
        unrealized = Decimal("0")
        if self.position is not None and mark > 0:
            sign = Decimal("1") if self.position.side is Side.BUY else Decimal("-1")
            position_value = sign * self.position.quantity * mark
            unrealized = sign * self.position.quantity * (mark - self.position.average_entry_price)
        return AccountSnapshot(
            timestamp=self.current_time,
            account_id="SIM-QQQ",
            equity=self.cash + position_value,
            cash=self.cash,
            buying_power=max(Decimal("0"), self.cash),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
        )

    async def process_candle(self, candle: Candle) -> tuple[Fill, ...]:
        self.last_price = candle.close
        self.current_time = candle.exchange_timestamp
        candidates = [
            order
            for order in self.orders.values()
            if order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}
            and order.exchange_timestamp < candle.exchange_timestamp
        ]
        candidates.sort(key=self._priority)
        new_fills: list[Fill] = []
        remaining_volume = self._floor_lot(candle.volume * self.max_volume_participation)
        for order in candidates:
            if remaining_volume <= 0:
                break
            if order.reduce_only and self.position is None:
                continue
            price = self._fill_price(order, candle)
            if price is None:
                continue
            remaining_order = order.quantity - order.filled_quantity
            position_cap = (
                self.position.quantity
                if order.reduce_only and self.position is not None
                else remaining_order
            )
            quantity = self._floor_lot(min(remaining_order, remaining_volume, position_cap))
            if quantity <= 0:
                continue
            theoretical = self._theoretical_price(order, candle)
            fill = Fill(
                exchange_timestamp=candle.exchange_timestamp,
                received_timestamp=candle.received_timestamp,
                source="simulated",
                instrument=candle.instrument,
                sequence_number=candle.sequence_number,
                correlation_id=order.correlation_id,
                client_order_id=order.client_order_id,
                side=order.side,
                price=price,
                quantity=quantity,
                commission=quantity * self.commission_per_unit,
                slippage=abs(price - theoretical) * quantity,
            )
            self._apply_fill(fill, order)
            filled = order.filled_quantity + quantity
            status = (
                OrderStatus.FILLED if filled == order.quantity else OrderStatus.PARTIALLY_FILLED
            )
            transition(order.status, status)
            previous_notional = (order.average_fill_price or Decimal("0")) * order.filled_quantity
            average = (previous_notional + price * quantity) / filled
            updated = order.model_copy(
                update={
                    "filled_quantity": filled,
                    "average_fill_price": average,
                    "status": status,
                }
            )
            self.orders[order.client_order_id] = updated
            self.fills.append(fill)
            new_fills.append(fill)
            remaining_volume -= quantity
        if self.position is not None:
            self.position = self.position.model_copy(
                update={"bars_held": self.position.bars_held + 1}
            )
        return tuple(new_fills)

    @staticmethod
    def _priority(order: Order) -> tuple[int, str]:
        if not order.reduce_only:
            return (0, order.client_order_id)
        if order.order_type is OrderType.STOP:
            return (1, order.client_order_id)
        return (2, order.client_order_id)

    def _fill_price(self, order: Order, candle: Candle) -> Decimal | None:
        if order.order_type is OrderType.MARKET:
            return self._round_price(self._market_price(candle.open, order.side), order.side)
        if order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            crossed = (
                candle.low <= order.limit_price
                if order.side is Side.BUY
                else candle.high >= order.limit_price
            )
            if not crossed:
                return None
            market = self._market_price(candle.open, order.side)
            price = (
                min(order.limit_price, market)
                if order.side is Side.BUY
                else max(order.limit_price, market)
            )
            return self._round_limit_price(price, order.side)
        assert order.stop_price is not None
        triggered = (
            candle.high >= order.stop_price
            if order.side is Side.BUY
            else candle.low <= order.stop_price
        )
        if not triggered:
            return None
        market = self._market_price(candle.open, order.side)
        price = (
            max(order.stop_price, market)
            if order.side is Side.BUY
            else min(order.stop_price, market)
        )
        return self._round_price(price, order.side)

    def _theoretical_price(self, order: Order, candle: Candle) -> Decimal:
        half_spread = self.spread_bps / Decimal("20000")
        if order.order_type is OrderType.STOP and order.stop_price is not None:
            return order.stop_price
        if order.order_type is OrderType.LIMIT and order.limit_price is not None:
            return order.limit_price
        multiplier = (
            Decimal("1") + half_spread if order.side is Side.BUY else Decimal("1") - half_spread
        )
        return candle.open * multiplier

    def _market_price(self, price: Decimal, side: Side) -> Decimal:
        total_bps = self.spread_bps / Decimal("2") + self.slippage_bps
        adjustment = total_bps / Decimal("10000")
        multiplier = Decimal("1") + adjustment if side is Side.BUY else Decimal("1") - adjustment
        return price * multiplier

    def _floor_lot(self, quantity: Decimal) -> Decimal:
        return (quantity / self.instrument.lot_size).to_integral_value(
            rounding=ROUND_FLOOR
        ) * self.instrument.lot_size

    def _round_price(self, price: Decimal, side: Side) -> Decimal:
        rounding = ROUND_CEILING if side is Side.BUY else ROUND_FLOOR
        ticks = (price / self.instrument.tick_size).to_integral_value(rounding=rounding)
        return ticks * self.instrument.tick_size

    def _round_limit_price(self, price: Decimal, side: Side) -> Decimal:
        rounding = ROUND_FLOOR if side is Side.BUY else ROUND_CEILING
        ticks = (price / self.instrument.tick_size).to_integral_value(rounding=rounding)
        return ticks * self.instrument.tick_size

    def _apply_fill(self, fill: Fill, order: Order) -> None:
        signed_cash = fill.price * fill.quantity
        if fill.side is Side.BUY:
            self.cash -= signed_cash + fill.commission
        else:
            self.cash += signed_cash - fill.commission

        if self.position is None:
            if order.reduce_only:
                return
            self.position = Position(
                instrument=fill.instrument,
                quantity=fill.quantity,
                side=fill.side,
                average_entry_price=fill.price,
                opened_at=fill.exchange_timestamp,
            )
            return
        if fill.side is self.position.side and not order.reduce_only:
            total = self.position.quantity + fill.quantity
            average = (
                self.position.average_entry_price * self.position.quantity
                + fill.price * fill.quantity
            ) / total
            self.position = self.position.model_copy(
                update={"quantity": total, "average_entry_price": average}
            )
            return

        closed = min(fill.quantity, self.position.quantity)
        sign = Decimal("1") if self.position.side is Side.BUY else Decimal("-1")
        self.realized_pnl += sign * closed * (fill.price - self.position.average_entry_price)
        remaining = self.position.quantity - closed
        self.position = (
            None if remaining == 0 else self.position.model_copy(update={"quantity": remaining})
        )
