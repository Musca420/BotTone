from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from adaptive_bot.clock import SimulatedClock
from adaptive_bot.config import load_config
from adaptive_bot.dashboard.server import build_dashboard_payload
from adaptive_bot.data.repository import SQLiteStateStore
from adaptive_bot.domain.enums import MarketRegime, OrderStatus, OrderType, Side, SignalAction
from adaptive_bot.domain.models import (
    AccountSnapshot,
    Candle,
    Order,
    OrderRequest,
    Position,
    Quote,
    Signal,
)
from adaptive_bot.services.paper_service import PaperRuntime
from adaptive_bot.services.trading_service import CandleAggregator


def _minute(timestamp: datetime, price: str, volume: str = "10") -> Candle:
    value = Decimal(price)
    return Candle(
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        source="test",
        instrument="QQQ",
        open=value,
        high=value + Decimal("0.2"),
        low=value - Decimal("0.2"),
        close=value,
        volume=Decimal(volume),
        timeframe_minutes=1,
    )


def test_minute_bars_are_aggregated_without_future_data() -> None:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    aggregator = CandleAggregator(15)
    for offset in range(15):
        assert aggregator.add(_minute(start + timedelta(minutes=offset), str(100 + offset))) is None
    completed = aggregator.add(_minute(start + timedelta(minutes=15), "115"))
    assert completed is not None
    assert completed.exchange_timestamp == datetime(2026, 1, 5, 14, 45, tzinfo=UTC)
    assert completed.open == Decimal("100")
    assert completed.close == Decimal("114")
    assert completed.volume == Decimal("150")


class FakePaperBroker:
    def __init__(self) -> None:
        self.account = AccountSnapshot(
            timestamp=datetime(2026, 1, 13, 14, 46, tzinfo=UTC),
            account_id="paper-account",
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )
        self.positions: tuple[Position, ...] = ()
        self.open_orders: tuple[Order, ...] = ()
        self.brackets: list[tuple[OrderRequest, Decimal]] = []
        self.flattened = False

    async def verify_account(self) -> AccountSnapshot:
        return self.account

    async def get_account(self) -> AccountSnapshot:
        return self.account

    async def get_positions(self) -> tuple[Position, ...]:
        return self.positions

    async def get_open_orders(self, instrument: str) -> tuple[Order, ...]:
        del instrument
        return self.open_orders

    async def submit_bracket_entry(
        self, request: OrderRequest, take_profit_price: Decimal
    ) -> Order:
        self.brackets.append((request, take_profit_price))
        return Order(
            **request.model_dump(exclude={"time_in_force"}),
            status=OrderStatus.ACKNOWLEDGED,
        )

    async def flatten_all(self) -> None:
        self.flattened = True

    async def submit_order(self, request: OrderRequest) -> Order:
        raise AssertionError(request)

    async def cancel_order(self, client_order_id: str) -> Order:
        raise AssertionError(client_order_id)

    async def get_order(self, client_order_id: str) -> Order | None:
        del client_order_id
        return None


class EntryStrategy:
    def evaluate(self, snapshot: Any, state: Any, position: Position | None) -> Signal | None:
        del state
        if position is not None:
            return None
        candle = snapshot.candle
        return Signal(
            exchange_timestamp=candle.exchange_timestamp,
            received_timestamp=candle.received_timestamp,
            source="test",
            instrument="QQQ",
            action=SignalAction.ENTER_LONG,
            reference_price=candle.close,
            stop_price=candle.close - Decimal("2.5"),
            target_price=candle.close + Decimal("1"),
            z_score=-2,
            regime=MarketRegime.RANGE,
            reason="test entry",
        )


def _paper_runtime(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
    broker: FakePaperBroker,
    state_store: SQLiteStateStore | None = None,
) -> tuple[PaperRuntime, datetime]:
    input_path = tmp_path / "seed.parquet"
    rth_frame.to_parquet(input_path, index=False)
    current = datetime(2026, 1, 13, 14, 46, tzinfo=UTC)
    runtime = PaperRuntime(
        load_config("configs/alpaca_qqq_paper.yaml"),
        broker,  # type: ignore[arg-type]
        input_path,
        tmp_path / "paper.json",
        clock=SimulatedClock(current),
        state_store=state_store,
    )
    runtime.strategy = EntryStrategy()  # type: ignore[assignment]
    return runtime, current


@pytest.mark.asyncio
async def test_paper_runtime_sizes_and_submits_protected_bracket(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    broker = FakePaperBroker()
    store = SQLiteStateStore(tmp_path / "state.db")
    runtime, current = _paper_runtime(rth_frame, tmp_path, broker, store)
    await runtime.initialize()
    runtime.on_quote(
        Quote(
            exchange_timestamp=current,
            received_timestamp=current,
            source="test",
            instrument="QQQ",
            bid=Decimal("99.99"),
            ask=Decimal("100.01"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
        )
    )
    await runtime.process(_minute(datetime(2026, 1, 13, 14, 45, tzinfo=UTC), "100"))
    assert len(broker.brackets) == 1
    request, target = broker.brackets[0]
    assert request.quantity > 0
    assert request.stop_price == Decimal("97.5")
    assert target == Decimal("101")
    dashboard = build_dashboard_payload(tmp_path / "paper.json")
    assert dashboard["summary"]["mode"] == "paper"
    assert "Paper bracket submitted" in dashboard["latest"]["activity"]
    risk_payload = await store.get("paper_risk_state")
    assert risk_payload is not None and "day_start_equity" in risk_payload


@pytest.mark.asyncio
async def test_stale_data_kill_switch_blocks_paper_entry(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    broker = FakePaperBroker()
    runtime, _ = _paper_runtime(rth_frame, tmp_path, broker)
    await runtime.initialize()
    await runtime.process(_minute(datetime(2026, 1, 13, 14, 45, tzinfo=UTC), "100"))
    assert runtime.kill_switch.active
    assert not broker.brackets


@pytest.mark.asyncio
async def test_missing_protective_stop_forces_paper_flatten(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    broker = FakePaperBroker()
    runtime, _ = _paper_runtime(rth_frame, tmp_path, broker)
    await runtime.initialize()
    broker.positions = (
        Position(
            instrument="QQQ",
            quantity=Decimal("2"),
            side=Side.BUY,
            average_entry_price=Decimal("100"),
            opened_at=datetime(2026, 1, 13, 14, 40, tzinfo=UTC),
        ),
    )
    await runtime.process(_minute(datetime(2026, 1, 13, 14, 45, tzinfo=UTC), "100"))
    assert broker.flattened
    assert runtime.kill_switch.event.cause.value == "missing_protective_stop"


@pytest.mark.asyncio
async def test_restart_restores_and_reconciles_protected_paper_position(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    broker = FakePaperBroker()
    position = Position(
        instrument="QQQ",
        quantity=Decimal("2"),
        side=Side.BUY,
        average_entry_price=Decimal("100"),
        opened_at=datetime(2026, 1, 13, 14, 40, tzinfo=UTC),
    )
    broker.positions = (position,)
    timestamp = datetime(2026, 1, 13, 14, 41, tzinfo=UTC)
    broker.open_orders = (
        Order(
            exchange_timestamp=timestamp,
            received_timestamp=timestamp,
            source="test",
            instrument="QQQ",
            client_order_id="protective-stop",
            side=Side.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("2"),
            stop_price=Decimal("97.5"),
            status=OrderStatus.ACKNOWLEDGED,
            protective=True,
        ),
    )
    store = SQLiteStateStore(tmp_path / "state.db")
    store.initialize()
    await store.set("paper_position", position.model_dump_json())
    runtime, _ = _paper_runtime(rth_frame, tmp_path, broker, store)
    await runtime.initialize()
    assert not runtime.kill_switch.active


@pytest.mark.asyncio
async def test_persisted_kill_switch_blocks_automatic_restart(
    rth_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    store.initialize()
    await store.set(
        "paper_kill_switch",
        '{"cause":"stale_data","timestamp":"2026-01-13T14:00:00+00:00"}',
    )
    runtime, _ = _paper_runtime(rth_frame, tmp_path, FakePaperBroker(), store)
    with pytest.raises(RuntimeError, match="latched"):
        await runtime.initialize()
