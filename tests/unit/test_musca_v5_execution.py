from decimal import Decimal

from adaptive_bot.musca_v5_execution import (
    MarketIntegrity,
    protected_stop,
    quote_taker_round_trip,
    size_for_technical_stop,
    walk_book,
)


def test_book_execution_consumes_multiple_levels() -> None:
    fill = walk_book([[100, 1], [101, 2]], Decimal("2"), reference_mid=Decimal("99.5"))
    assert fill is not None
    assert fill.execution_vwap == Decimal("100.5")
    assert fill.levels_consumed == 2
    assert walk_book([[100, 1]], Decimal("2"), reference_mid=Decimal("99.5")) is None


def test_vip0_round_trip_uses_taker_fees_and_observed_depth() -> None:
    quote = quote_taker_round_trip(
        side="LONG",
        quantity=Decimal("1"),
        bids=[[99, 1]],
        asks=[[101, 1]],
        best_bid=Decimal("99"),
        best_ask=Decimal("101"),
    )
    assert quote is not None
    assert quote.entry.execution_vwap == Decimal("101")
    assert quote.estimated_exit.execution_vwap == Decimal("99")
    assert quote.expected_total_cost_bps == Decimal("212.0")
    assert quote.maker_status.startswith("DISABLED")


def test_position_sizing_respects_one_percent_and_ten_x_cap() -> None:
    quote = size_for_technical_stop(
        equity=Decimal("10000"),
        entry_price=Decimal("50000"),
        side="LONG",
        technical_stop_bps=Decimal("100"),
        expected_cost_bps=Decimal("13"),
        lot_size=Decimal("0.001"),
        minimum_quantity=Decimal("0.001"),
        minimum_notional=Decimal("5"),
    )
    assert quote.approved
    assert quote.estimated_loss_at_stop <= Decimal("100")
    assert quote.notional <= Decimal("10000")
    assert quote.margin_required <= Decimal("1000")


def test_trailing_never_widens_and_waits_for_net_profit() -> None:
    unchanged, zone = protected_stop(
        side="LONG",
        current_stop=Decimal("99"),
        proposed_stop=Decimal("98"),
        entry_price=Decimal("100"),
        total_cost_bps=Decimal("13"),
        favorable_excursion_bps=Decimal("20"),
    )
    assert unchanged == Decimal("99")
    assert zone == "ZONE_A_OR_B_NO_TRAIL"
    tightened, _ = protected_stop(
        side="LONG",
        current_stop=Decimal("99"),
        proposed_stop=Decimal("100.5"),
        entry_price=Decimal("100"),
        total_cost_bps=Decimal("13"),
        favorable_excursion_bps=Decimal("30"),
    )
    assert tightened >= Decimal("100.13")


def test_integrity_fails_closed() -> None:
    integrity = MarketIntegrity(100, 100, True, True, True, True, 0)
    assert integrity.rejection_reason() == "DESYNCHRONIZED_ORDERBOOK"
