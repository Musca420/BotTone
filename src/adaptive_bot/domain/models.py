from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_bot.domain.enums import (
    AssetClass,
    HealthLevel,
    LiquidityRole,
    MarketRegime,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
    TimeInForce,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_utc(value: datetime, field: str) -> None:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")


class Instrument(StrictModel):
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    tick_size: Decimal = Field(gt=0)
    lot_size: Decimal = Field(gt=0)
    minimum_quantity: Decimal = Field(gt=0)
    minimum_notional: Decimal = Field(ge=0)
    point_value: Decimal = Field(gt=0, default=Decimal("1"))
    shortable: bool = False
    max_leverage: Decimal = Field(ge=1, default=Decimal("1"))


class DomainEvent(StrictModel):
    exchange_timestamp: datetime
    received_timestamp: datetime
    source: str
    instrument: str
    sequence_number: int | None = Field(default=None, ge=0)
    correlation_id: UUID = Field(default_factory=uuid4)
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def timestamps_are_utc(self) -> DomainEvent:
        _require_utc(self.exchange_timestamp, "exchange_timestamp")
        _require_utc(self.received_timestamp, "received_timestamp")
        if self.received_timestamp < self.exchange_timestamp:
            raise ValueError("received_timestamp cannot precede exchange_timestamp")
        return self


class Candle(DomainEvent):
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    timeframe_minutes: int = Field(default=15, gt=0)

    @model_validator(mode="after")
    def valid_ohlc(self) -> Candle:
        if self.high < self.low:
            raise ValueError("high cannot be below low")
        if not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise ValueError("open and close must be within high-low")
        return self


class Quote(DomainEvent):
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    bid_size: Decimal = Field(ge=0)
    ask_size: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def valid_spread(self) -> Quote:
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        return self


class Trade(DomainEvent):
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)


class BookLevel(StrictModel):
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)


class OrderBookSnapshot(DomainEvent):
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]


class Signal(DomainEvent):
    action: SignalAction
    reference_price: Decimal = Field(gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    target_price: Decimal | None = Field(default=None, gt=0)
    z_score: float
    regime: MarketRegime
    reason: str
    entry_limit_price: Decimal | None = Field(default=None, gt=0)
    entry_post_only: bool = False

    @model_validator(mode="after")
    def post_only_entry_has_price(self) -> Signal:
        if self.entry_post_only and self.entry_limit_price is None:
            raise ValueError("post-only entry requires entry_limit_price")
        return self


class OrderRequest(DomainEvent):
    client_order_id: str
    side: Side
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.DAY
    reduce_only: bool = False
    protective: bool = False
    post_only: bool = False

    @model_validator(mode="after")
    def required_price(self) -> OrderRequest:
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("stop order requires stop_price")
        if self.post_only and self.order_type is not OrderType.LIMIT:
            raise ValueError("post-only order must be a limit order")
        return self


class Order(DomainEvent):
    client_order_id: str
    side: Side
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    filled_quantity: Decimal = Field(ge=0, default=Decimal("0"))
    average_fill_price: Decimal | None = Field(default=None, gt=0)
    status: OrderStatus = OrderStatus.CREATED
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    reduce_only: bool = False
    protective: bool = False
    post_only: bool = False

    @model_validator(mode="after")
    def fill_not_above_order(self) -> Order:
        if self.filled_quantity > self.quantity:
            raise ValueError("filled quantity exceeds order quantity")
        return self


class Fill(DomainEvent):
    client_order_id: str
    side: Side
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    commission: Decimal = Field(ge=0)
    slippage: Decimal = Field(ge=0)
    liquidity_role: LiquidityRole = LiquidityRole.UNKNOWN


class Position(StrictModel):
    instrument: str
    quantity: Decimal = Field(gt=0)
    side: Side
    average_entry_price: Decimal = Field(gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    opened_at: datetime
    bars_held: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def opened_at_utc(self) -> Position:
        _require_utc(self.opened_at, "opened_at")
        return self


class AccountSnapshot(StrictModel):
    timestamp: datetime
    account_id: str
    equity: Decimal = Field(ge=0)
    cash: Decimal
    buying_power: Decimal = Field(ge=0)
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    @model_validator(mode="after")
    def timestamp_utc(self) -> AccountSnapshot:
        _require_utc(self.timestamp, "timestamp")
        return self


class RiskDecision(StrictModel):
    approved: bool
    reason: str
    quantity: Decimal = Field(ge=0, default=Decimal("0"))
    risk_budget: Decimal = Field(ge=0, default=Decimal("0"))
    effective_risk: Decimal = Field(ge=0, default=Decimal("0"))


class StrategyState(StrictModel):
    regime: MarketRegime = MarketRegime.UNKNOWN
    pending_regime: MarketRegime | None = None
    pending_regime_count: int = Field(default=0, ge=0)
    cooldown_bars: int = Field(default=0, ge=0)
    previous_z: float | None = None
    last_z: float | None = None
    last_close: Decimal | None = None


class HealthStatus(StrictModel):
    timestamp: datetime
    level: HealthLevel
    checks: dict[str, bool]
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def timestamp_utc(self) -> HealthStatus:
        _require_utc(self.timestamp, "timestamp")
        return self
