from __future__ import annotations

import pandas as pd

from adaptive_bot.musca_v5_event_policy import (
    ManagementPlan,
    _causal_micro_stop_bps,
    _economically_protected_stop,
    _multi_horizon_path_labels,
    calculate_net_ev_bps,
    select_economic_decision,
    simulate_management,
)


def _path(high: list[float], low: list[float], close: list[float]) -> pd.DataFrame:
    high = [100.05] * 12 + high
    low = [99.95] * 12 + low
    close = [100.0] * 12 + close
    timestamp = pd.date_range("2026-01-01", periods=len(high), freq="5s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(seconds=5),
            "open": [100.0, *close[:-1]],
            "high": high,
            "low": low,
            "close": close,
            "ofi_15s": 0.2,
            "ofi_1m": 0.1,
            "ofi_persistence_1m": 0.25,
            "trade_intensity_15s": 1.2,
            "price_velocity_15s": 2.0,
        }
    )


def _event(direction: int = 1) -> dict[str, object]:
    return {
        "available_at": pd.Timestamp("2026-01-01T00:01:00Z"),
        "direction": direction,
        "stop_price": 99.8 if direction > 0 else 100.2,
    }


def test_exit_happens_at_first_event_not_at_fixed_horizon() -> None:
    plan = ManagementPlan("TEST", 30.0, 50.0, 0.5, 10.0, 60)
    data = _path(
        [100.0, 100.31, 100.51, 100.10],
        [100.0, 100.15, 100.35, 99.90],
        [100.0, 100.30, 100.50, 100.00],
    )
    result = simulate_management(_event(), data, plan)
    assert result is not None
    assert result["exit_reason"] == "SECOND_TARGET"
    assert result["duration_seconds"] == 10.0
    assert result["gross_return_bps"] == 40.0


def test_same_bucket_stop_wins_over_first_target() -> None:
    plan = ManagementPlan("TEST", 30.0, 50.0, 0.5, 10.0, 60)
    data = _path(
        [100.0, 100.4, 100.0],
        [100.0, 99.7, 100.0],
        [100.0, 100.1, 100.0],
    )
    result = simulate_management(_event(), data, plan)
    assert result is not None
    assert result["exit_reason"] == "INITIAL_STOP"
    assert result["first_target_hit"] is False


def test_trailing_stop_never_widens_and_protects_net_result() -> None:
    plan = ManagementPlan("TEST", 30.0, 80.0, 0.5, 10.0, 60)
    data = _path(
        [100.0, 100.31, 100.50, 100.42],
        [100.0, 100.20, 100.35, 100.38],
        [100.0, 100.30, 100.45, 100.40],
    )
    result = simulate_management(_event(), data, plan)
    assert result is not None
    assert result["exit_reason"] == "PROTECTED_TRAIL"
    assert result["final_stop_bps"] >= _economically_protected_stop(plan)
    assert result["net_return_bps"] > 0


def test_timeout_uses_actual_early_plan_horizon() -> None:
    plan = ManagementPlan("TEST", 30.0, 60.0, 0.5, 10.0, 1)
    data = _path([100.0] * 20, [100.0] * 20, [100.0] * 20)
    result = simulate_management(_event(), data, plan)
    assert result is not None
    assert result["exit_reason"] == "TIME_STOP_1M"
    assert result["duration_seconds"] == 60.0


def test_micro_stop_uses_only_completed_pre_entry_buckets() -> None:
    data = _path(
        [100.05] * 12 + [110.0],
        [99.90] * 12 + [90.0],
        [100.0] * 13,
    )
    stop = _causal_micro_stop_bps(1, 100.0, data, 12)
    assert stop is not None
    assert -20.0 < stop <= -12.0


def test_net_ev_subtracts_observed_execution_cost_after_movement_model() -> None:
    low_spread = calculate_net_ev_bps(0.7, 40.0, 20.0, 13.0)
    high_spread = calculate_net_ev_bps(0.7, 40.0, 20.0, 25.0)
    assert float(low_spread) == 9.0
    assert float(high_spread) == -3.0


def test_plan_is_rejected_when_target_is_too_small_for_vip0_cost() -> None:
    plan = ManagementPlan("TOO_SMALL", 10.0, 20.0, 0.5, 5.0, 10)
    data = _path([100.0] * 130, [100.0] * 130, [100.0] * 130)
    assert simulate_management(_event(), data, plan) is None


def test_profile_cost_is_used_for_runner_net_result() -> None:
    plan = ManagementPlan("VIP5", 12.0, 24.0, 0.5, 6.0, 1)
    data = _path(
        [100.0, 100.13, 100.25],
        [100.0, 100.05, 100.15],
        [100.0, 100.12, 100.24],
    )

    result = simulate_management(
        _event(), data, plan, cost_bps=8.0, minimum_gross_to_cost=0.0
    )

    assert result is not None
    assert result["expected_round_trip_cost_bps"] == 8.0
    assert result["net_return_bps"] == result["gross_return_bps"] - 8.0


def test_immediate_entry_does_not_invent_a_micro_confirmation() -> None:
    plan = ManagementPlan("IMMEDIATE", 12.0, 24.0, 0.5, 6.0, 1)
    data = _path([100.0] * 20, [100.0] * 20, [100.0] * 20)
    data[["ofi_15s", "ofi_1m", "price_velocity_15s"]] = -1.0

    confirmed = simulate_management(
        _event(), data, plan, minimum_gross_to_cost=0.0
    )
    immediate = simulate_management(
        _event(),
        data,
        plan,
        minimum_gross_to_cost=0.0,
        entry_style="immediate_next_bucket",
    )

    assert confirmed is None
    assert immediate is not None


def test_multi_horizon_labels_preserve_time_and_barrier_order() -> None:
    data = _path(
        [100.0, 100.21, 100.31, *([100.31] * 720)],
        [100.0, 100.05, 99.70, *([99.90] * 720)],
        [100.0, 100.20, 100.00, *([100.00] * 720)],
    )
    labels = _multi_horizon_path_labels(
        data,
        entry_index=12,
        side=1,
        entry=100.0,
        stop_bps=-20.0,
        first_target_bps=30.0,
    )
    assert labels["hit_20bps_before_stop"] is True
    assert labels["hit_30bps_before_stop"] is False
    assert labels["time_to_20bps_seconds"] == 5.0
    assert labels["mfe_5m_bps"] >= 30.0
    assert labels["mae_5m_bps"] <= -30.0
    assert labels["label_available_at_5m"] > data.iloc[12]["available_at"]


def test_economic_gate_uses_flat_only_as_neutral_veto() -> None:
    scored = pd.DataFrame(
        [
            {
                "movement_covers_cost": True,
                "predicted_ev_bps": 4.0,
                "predicted_total_cost_bps": 13.0,
                "predicted_win_probability": 0.61,
                "predicted_gross_win_probability": 0.67,
                "predicted_target_probability": 0.58,
                "predicted_gross_ev_bps": 17.0,
                "predicted_mfe_bps": 42.0,
                "predicted_mae_bps": 18.0,
                "predicted_time_to_target_minutes": 14.0,
                "direction": 1,
                "event_family": "ROLLING_VWAP_REENTRY",
                "plan": "STANDARD_40_70",
                "entry_price": 100.0,
                "initial_stop_bps": -20.0,
                "plan_first_target_bps": 40.0,
                "plan_second_target_bps": 70.0,
                "plan_first_exit_fraction": 0.5,
                "plan_maximum_minutes": 60,
            }
        ]
    )
    decision = select_economic_decision(scored, entry_threshold_bps=2.0)
    assert decision["action"] == "LONG"
    assert decision["predicted_net_ev_bps"] == 4.0
    scored["predicted_ev_bps"] = -0.01
    decision = select_economic_decision(scored, entry_threshold_bps=0.0)
    assert decision["action"] == "NO_TRADE"
