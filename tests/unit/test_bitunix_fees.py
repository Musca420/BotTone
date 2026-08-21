import pytest

from adaptive_bot.bitunix_fees import expected_round_trip_cost_bps, futures_fee_bps


def test_official_bitunix_futures_vip_profiles() -> None:
    assert futures_fee_bps(0) == (2.0, 6.0)
    assert futures_fee_bps(8) == (0.0, 2.6)
    with pytest.raises(ValueError):
        futures_fee_bps(9)


def test_vip0_taker_round_trip_includes_observed_spread_and_slippage() -> None:
    cost = expected_round_trip_cost_bps(
        vip_level=0,
        entry_role="taker",
        exit_role="taker",
        spread_bps=0.02,
        entry_slippage_bps=0.4,
        exit_slippage_bps=0.6,
    )
    assert cost == pytest.approx(13.02)


def test_maker_order_does_not_assume_crossing_the_spread() -> None:
    cost = expected_round_trip_cost_bps(
        vip_level=0,
        entry_role="maker",
        exit_role="taker",
        spread_bps=2.0,
    )
    assert cost == 9.0
