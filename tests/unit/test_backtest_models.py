from decimal import Decimal

from adaptive_bot.backtest.costs import estimated_round_trip_cost_per_unit
from adaptive_bot.backtest.fills import worst_case_stop_or_target
from adaptive_bot.backtest.funding import funding_payment
from adaptive_bot.backtest.slippage import apply_slippage


def test_cost_slippage_funding_and_worst_case() -> None:
    cost = estimated_round_trip_cost_per_unit(
        Decimal("100"), Decimal("2"), Decimal("1"), Decimal("0.005")
    )
    assert cost == Decimal("0.05")
    assert apply_slippage(Decimal("100"), Decimal("10"), adverse_sign=1) == Decimal("100.100")
    assert funding_payment(Decimal("1000"), Decimal("0.001"), long_position=True) == -1
    assert (
        worst_case_stop_or_target(
            low=Decimal("94"),
            high=Decimal("106"),
            stop=Decimal("95"),
            target=Decimal("105"),
        )
        == "stop"
    )
