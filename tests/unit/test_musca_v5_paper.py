from __future__ import annotations

from math import isclose
from pathlib import Path

import pandas as pd

from adaptive_bot.musca_v5_paper import (
    advance_account,
    advance_accounts,
    new_paper_account,
)


def _book(at: pd.Timestamp, mid: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "available_at": at,
                "mid": mid,
                "best_bid": mid - 1,
                "best_ask": mid + 1,
                "bids": [[mid - 1, 10.0], [mid - 2, 10.0]],
                "asks": [[mid + 1, 10.0], [mid + 2, 10.0]],
                "feature_valid": True,
            }
        ]
    )


def _assessment(at: pd.Timestamp, profile: str = "VIP0") -> dict[str, object]:
    return {
        "evaluated_at": (at + pd.Timedelta(milliseconds=100)).isoformat(),
        "observed_at": at.isoformat(),
        "decision": "TRADE",
        "reason": "TRADE",
        "candidate_complete": True,
        "fee_profile": profile,
        "setup": "PULLBACK_CONTINUATION",
        "direction": "LONG",
        "quantity_btc": 0.15,
        "target_bps": 30.0,
        "stop_bps": 20.0,
        "expected_cost_bps": 13.0,
        "expected_funding_bps": 0.0,
        "risk_budget": 100.0,
        "target_probability": 0.7,
        "stop_probability": 0.2,
        "timeout_probability": 0.1,
        "expected_net_ev_bps": 5.0,
        "expected_time_to_target_minutes": 10.0,
        "flow_vote": 2.0,
        "market_inputs": {
            "bitunix": {
                "mark_price": 60_000.0,
                "funding_rate": 0.0001,
                "next_funding_timestamp": (at + pd.Timedelta(hours=8)).isoformat(),
            }
        },
    }


def test_market_entry_waits_for_the_next_observed_bitunix_book(tmp_path: Path) -> None:
    state = tmp_path / "paper.json"
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    accounts = advance_accounts({"VIP0": _assessment(first)}, _book(first, 60_000), path=state)

    assert accounts["VIP0"]["pending_order"]["status"] == "ACKNOWLEDGED"
    assert accounts["VIP0"]["open_position"] is None

    second = first + pd.Timedelta(seconds=1)
    accounts = advance_accounts({"VIP0": _assessment(first)}, _book(second, 60_000), path=state)

    position = accounts["VIP0"]["open_position"]
    assert accounts["VIP0"]["pending_order"] is None
    assert position["status"] == "OPEN"
    assert position["entry_at"] == second.isoformat()
    assert position["entry_execution_price"] == 60_001.0
    assert accounts["VIP0"]["fees"] > 0


def test_open_position_closes_at_target_with_realized_fees_and_no_duplicate(
    tmp_path: Path,
) -> None:
    state = tmp_path / "paper.json"
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    assessment = _assessment(first)
    advance_accounts({"VIP0": assessment}, _book(first, 60_000), path=state)
    advance_accounts(
        {"VIP0": assessment}, _book(first + pd.Timedelta(seconds=1), 60_000), path=state
    )
    accounts = advance_accounts(
        {"VIP0": assessment}, _book(first + pd.Timedelta(seconds=2), 60_250), path=state
    )

    account = accounts["VIP0"]
    assert account["open_position"] is None
    assert account["pending_order"] is None
    assert len(account["trades"]) == 1
    assert account["trades"][0]["exit_reason"] == "DYNAMIC_TARGET"
    assert account["trades"][0]["fees"] > 0
    assert account["trades"][0]["net_pnl"] < account["trades"][0]["gross_pnl"]
    assert isclose(
        account["trades"][0]["gross_pnl"] - account["trades"][0]["modeled_costs"],
        account["trades"][0]["net_pnl"],
        abs_tol=1e-9,
    )
    assert account["final_equity"] > 10_000

    accounts = advance_accounts(
        {"VIP0": assessment}, _book(first + pd.Timedelta(seconds=3), 60_260), path=state
    )
    assert accounts["VIP0"]["pending_order"] is None
    assert len(accounts["VIP0"]["trades"]) == 1


def test_vip_profiles_trade_the_same_signal_with_independent_real_costs(tmp_path: Path) -> None:
    state = tmp_path / "paper.json"
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    assessments = {f"VIP{level}": _assessment(first, f"VIP{level}") for level in range(6)}
    advance_accounts(assessments, _book(first, 60_000), path=state)
    advance_accounts(assessments, _book(first + pd.Timedelta(seconds=1), 60_000), path=state)
    accounts = advance_accounts(
        assessments, _book(first + pd.Timedelta(seconds=2), 60_250), path=state
    )

    assert all(len(accounts[f"VIP{level}"]["trades"]) == 1 for level in range(6))
    assert accounts["VIP5"]["fees"] < accounts["VIP0"]["fees"]
    assert accounts["VIP5"]["final_equity"] > accounts["VIP0"]["final_equity"]


def test_live_decision_counts_are_separate_and_deduplicated(tmp_path: Path) -> None:
    state = tmp_path / "paper.json"
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    trade = _assessment(first, "VIP5")
    accounts = advance_accounts({"VIP5": trade}, _book(first, 60_000), path=state)
    accounts = advance_accounts(
        {"VIP5": trade}, _book(first + pd.Timedelta(seconds=1), 60_000), path=state
    )

    vip5 = accounts["VIP5"]
    assert vip5["assessment_count"] == 1
    assert vip5["complete_candidate_count"] == 1
    assert vip5["trade_signal_count"] == 1

    wait = {
        **trade,
        "observed_at": (first + pd.Timedelta(minutes=1)).isoformat(),
        "decision": "WAIT",
        "reason": "NO_SETUP",
        "candidate_complete": False,
    }
    accounts = advance_accounts(
        {"VIP5": wait}, _book(first + pd.Timedelta(minutes=1), 60_000), path=state
    )
    vip5 = accounts["VIP5"]
    assert vip5["assessment_count"] == 2
    assert vip5["wait_count"] == 1
    assert vip5["last_assessment"]["reason"] == "NO_SETUP"


def test_frozen_base_takes_half_then_protects_cost_and_trails(tmp_path: Path) -> None:
    state = tmp_path / "paper.json"
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    assessment = {
        **_assessment(first, "VIP5"),
        "policy_source": "MUSCA_V8_FROZEN_BASE",
        "expert_id": "musca-v8-h24-test",
        "alpha_signal_at": first.isoformat(),
        "management_style": "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M",
        "partial_target_fraction": 0.5,
        "maximum_hold_minutes": 360,
    }
    advance_accounts({"VIP5": assessment}, _book(first, 60_000), path=state)
    accounts = advance_accounts(
        {"VIP5": assessment}, _book(first + pd.Timedelta(seconds=1), 60_000), path=state
    )
    initial_quantity = accounts["VIP5"]["open_position"]["quantity_btc"]

    accounts = advance_accounts(
        {"VIP5": assessment}, _book(first + pd.Timedelta(seconds=2), 60_250), path=state
    )
    position = accounts["VIP5"]["open_position"]
    assert position["tp1_hit"]
    assert 0 < position["quantity_btc"] < initial_quantity
    assert position["current_stop_price"] >= position["break_even_price"]
    protected_stop = position["current_stop_price"]

    accounts = advance_accounts(
        {"VIP5": assessment},
        _book(first + pd.Timedelta(minutes=16), 60_260),
        path=state,
    )
    position = accounts["VIP5"]["open_position"]
    assert position["current_stop_price"] >= protected_stop
    assert position["maximum_hold_minutes"] == 360

    accounts = advance_accounts(
        {"VIP5": assessment},
        _book(first + pd.Timedelta(minutes=16, seconds=1), 60_200),
        path=state,
    )
    account = accounts["VIP5"]
    assert account["open_position"] is None
    assert account["trades"][0]["tp1_hit"]
    assert len(account["trades"][0]["exit_fills"]) == 2
    assert account["trades"][0]["policy_source"] == "MUSCA_V8_FROZEN_BASE"


def test_auto_moe_uses_second_target_and_never_widens_its_trail(tmp_path: Path) -> None:
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    assessment = {
        **_assessment(first, "BINANCE"),
        "policy_source": "BTC_AUTO_MOE_RESEARCH_PAPER",
        "expert_id": "btc-long-3600s-test",
        "alpha_signal_at": first.isoformat(),
        "management_style": "HALF_AT_Q50_Q75_NON_WIDENING_TRAIL",
        "partial_target_fraction": 0.5,
        "target_2_bps": 50.0,
        "trailing_bps": 10.0,
        "maximum_hold_minutes": 60,
    }
    account = new_paper_account(
        "BINANCE",
        first.isoformat(),
        execution_venue="BINANCE",
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
    )
    advance_account(account, assessment, _book(first, 60_000))
    advance_account(account, assessment, _book(first + pd.Timedelta(seconds=1), 60_000))

    advance_account(account, assessment, _book(first + pd.Timedelta(seconds=2), 60_200))
    position = account["open_position"]
    assert position["tp1_hit"] is True
    cost_protected = position["current_stop_price"]

    advance_account(account, assessment, _book(first + pd.Timedelta(seconds=3), 60_260))
    tightened = account["open_position"]["current_stop_price"]
    assert tightened >= cost_protected

    advance_account(account, assessment, _book(first + pd.Timedelta(seconds=4), 60_320))
    assert account["open_position"] is None
    assert account["trades"][0]["exit_reason"] == "DYNAMIC_TARGET_2"
    assert len(account["trades"][0]["exit_fills"]) == 2


def test_generic_binance_account_uses_its_own_fee_schedule() -> None:
    first = pd.Timestamp("2026-08-09T00:00:00Z")
    account = new_paper_account(
        "BINANCE",
        first.isoformat(),
        execution_venue="BINANCE",
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
    )
    assessment = {
        **_assessment(first, "BINANCE"),
        "market_inputs": {
            "binance": {
                "mark_price": 60_000.0,
                "funding_rate": 0.0001,
                "next_funding_timestamp": (first + pd.Timedelta(hours=8)).isoformat(),
            }
        },
    }

    advance_account(account, assessment, _book(first, 60_000))
    assert account["pending_order"]["status"] == "ACKNOWLEDGED"
    advance_account(account, assessment, _book(first + pd.Timedelta(seconds=1), 60_000))

    assert account["execution_venue"] == "BINANCE"
    assert account["open_position"] is not None
    assert account["fees_per_side_bps"] == 4.0
    assert account["fees"] > 0
