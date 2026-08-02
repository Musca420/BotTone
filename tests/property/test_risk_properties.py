from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from adaptive_bot.domain.enums import KillSwitchCause, MarketRegime, OrderType, Side, SignalAction
from adaptive_bot.domain.models import AccountSnapshot, OrderRequest, Signal
from adaptive_bot.execution.idempotency import client_order_id
from adaptive_bot.meme.universe import choose_leverage
from adaptive_bot.risk.engine import DefaultRiskEngine, RiskState
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.risk.position_sizing import SizingInput, size_position
from adaptive_bot.strategy.signals import initial_stop, tighten_stop


@given(
    equity=st.decimals(min_value="1", max_value="1000000", places=2),
    entry=st.decimals(min_value="10", max_value="1000", places=2),
    distance=st.decimals(min_value="0.01", max_value="100", places=2),
)
def test_rounded_risk_never_exceeds_budget(instrument, equity, entry, distance) -> None:  # type: ignore[no-untyped-def]
    stop = entry - distance
    if stop <= 0:
        return
    result = size_position(
        SizingInput(
            equity=equity,
            buying_power=equity,
            entry_price=entry,
            stop_price=stop,
            estimated_cost_per_unit=Decimal("0.01"),
            risk_fraction=Decimal("0.01"),
            hard_notional_cap=equity,
            side=Side.BUY,
        ),
        instrument,
    )
    assert result.quantity >= 0
    assert result.effective_risk <= result.risk_budget


@given(
    entry=st.decimals(min_value="2", max_value="1000", places=2),
    atr=st.decimals(min_value="0.01", max_value="1", places=2),
)
def test_stop_direction_and_tightening_never_increase_risk(entry, atr) -> None:  # type: ignore[no-untyped-def]
    long_stop = initial_stop(entry, atr, Side.BUY, Decimal("1"))
    short_stop = initial_stop(entry, atr, Side.SELL, Decimal("1"))
    assert long_stop < entry < short_stop
    tightened = long_stop + atr / 2
    assert tighten_stop(long_stop, tightened, Side.BUY) >= long_stop


@given(cause=st.sampled_from(list(KillSwitchCause)))
def test_kill_switch_always_blocks_entry(cause: KillSwitchCause) -> None:
    now = datetime.now(UTC)
    switch = KillSwitch()
    switch.trigger(cause, now, "property")
    request = OrderRequest(
        exchange_timestamp=now,
        received_timestamp=now,
        source="test",
        instrument="QQQ",
        client_order_id="id",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
    )
    assert not switch.allows(request)


@given(reason=st.text(min_size=1, max_size=20))
def test_same_event_has_same_order_id(reason: str) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
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
        reason=reason,
    )
    assert client_order_id(signal, "entry") == client_order_id(signal, "entry")


def test_zero_equity_never_approved(instrument) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    switch = KillSwitch()
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
        equity=Decimal("0"),
        cash=Decimal("0"),
        buying_power=Decimal("0"),
    )
    decision = engine.assess(
        signal,
        instrument,
        account,
        RiskState(Decimal("0"), Decimal("0"), Decimal("0")),
        Decimal("0"),
    )
    assert not decision.approved and decision.quantity == 0


@given(
    notional=st.decimals(min_value="0.01", max_value="5000", places=2),
    equity=st.decimals(min_value="1", max_value="100000", places=2),
    ceiling=st.sampled_from([2, 3, 5]),
)
def test_meme_leverage_never_exceeds_manual_ceiling(
    notional: Decimal, equity: Decimal, ceiling: int
) -> None:
    leverage = choose_leverage(notional, equity, Decimal("0.10"), ceiling)
    if leverage is not None:
        assert leverage <= ceiling
        assert notional / leverage <= equity * Decimal("0.10")


from adaptive_bot.config import RiskConfig  # noqa: E402
