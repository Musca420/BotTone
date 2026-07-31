from datetime import UTC, datetime
from decimal import Decimal

from adaptive_bot.config import RiskConfig
from adaptive_bot.domain.enums import KillSwitchCause, MarketRegime, Side, SignalAction
from adaptive_bot.domain.models import AccountSnapshot, Signal
from adaptive_bot.risk.engine import DefaultRiskEngine, RiskState
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.risk.limits import drawdown, loss_limit_breached
from adaptive_bot.risk.liquidation import LiquidationCheck, liquidation_buffer_valid
from adaptive_bot.risk.position_sizing import SizingInput, floor_to_lot, size_position


def test_sizing_rounds_down_and_respects_budget(instrument) -> None:  # type: ignore[no-untyped-def]
    result = size_position(
        SizingInput(
            equity=Decimal("100000"),
            buying_power=Decimal("100000"),
            entry_price=Decimal("100"),
            stop_price=Decimal("95"),
            estimated_cost_per_unit=Decimal("0.10"),
            risk_fraction=Decimal("0.0025"),
            hard_notional_cap=Decimal("25000"),
            side=Side.BUY,
        ),
        instrument,
    )
    assert result.approved and result.quantity == 49
    assert result.effective_risk <= result.risk_budget
    assert floor_to_lot(Decimal("3.9"), Decimal("1")) == 3


def test_daily_loss_drawdown_and_liquidation_buffer() -> None:
    assert loss_limit_breached(Decimal("99000"), Decimal("100000"), Decimal("0.01"))
    assert drawdown(Decimal("90"), Decimal("100")) == Decimal("0.1")
    assert liquidation_buffer_valid(
        LiquidationCheck(
            leverage=Decimal("2"),
            mark_price=Decimal("100"),
            stop_price=Decimal("95"),
            broker_liquidation_price=Decimal("70"),
            estimated_liquidation_price=Decimal("71"),
            extra_buffer=Decimal("5"),
        )
    )
    assert not liquidation_buffer_valid(
        LiquidationCheck(
            leverage=Decimal("2"),
            mark_price=Decimal("100"),
            stop_price=Decimal("95"),
            broker_liquidation_price=None,
            estimated_liquidation_price=None,
        )
    )


def test_kill_switch_latches_and_blocks_entries(instrument) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    switch = KillSwitch()
    switch.trigger(KillSwitchCause.DAILY_LOSS, now, "test")
    engine = DefaultRiskEngine(RiskConfig(), switch)
    signal = Signal(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        action=SignalAction.ENTER_LONG,
        reference_price=Decimal("100"),
        stop_price=Decimal("95"),
        z_score=-2,
        regime=MarketRegime.RANGE,
        reason="test",
    )
    account = AccountSnapshot(
        timestamp=now,
        account_id="SIM-QQQ",
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
    )
    decision = engine.assess(
        signal,
        instrument,
        account,
        RiskState(Decimal("100000"), Decimal("100000"), Decimal("100000")),
        Decimal("0"),
    )
    assert not decision.approved
