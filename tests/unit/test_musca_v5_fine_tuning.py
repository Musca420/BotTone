from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot.musca_v5_fine_tuning import (
    FEATURES,
    _causal_context_coverage,
    _conditional_ev,
    _cost_runner_plan,
    _cost_target_plan,
    _policy_coverage_evaluations,
    _policy_coverage_frontier,
    _simulate_plan,
    _technical_stop_bps,
)
from adaptive_bot.musca_v5_micro_entry_audit import MODEL_FEATURES as MICRO_MODEL_FEATURES
from adaptive_bot.musca_v5_micro_entry_audit import _score as score_micro_conditional_ev
from adaptive_bot.musca_v5_micro_entry_audit import merge_causal_micro
from adaptive_bot.musca_v5_micro_model import ONE_SECOND_MICRO_FEATURES


def test_micro_stop_is_causal_and_inside_registered_risk_range() -> None:
    minutes = pd.DataFrame(
        {
            "perp_high": [101.0, 101.2, 101.1, 101.0, 100.9],
            "perp_low": [100.3, 100.35, 100.4, 100.45, 100.5],
            "perp_close": [100.4, 100.45, 100.5, 100.55, 100.6],
            "data_valid": True,
        }
    )
    event = {"direction": 1, "stop_price": 99.5}

    stop = _technical_stop_bps(event, minutes, len(minutes), 100.7)

    assert stop is not None
    assert -60.0 <= stop <= -12.0


def test_micro_stop_rejects_history_across_invalid_minute() -> None:
    minutes = pd.DataFrame(
        {
            "perp_high": [101.0] * 5,
            "perp_low": [100.0] * 5,
            "perp_close": [100.5] * 5,
            "data_valid": [True, True, False, True, True],
        }
    )

    assert _technical_stop_bps(
        {"direction": 1, "stop_price": 99.0}, minutes, 5, 100.7
    ) is None


def test_target_miss_keeps_its_signed_economic_outcome() -> None:
    result = _conditional_ev(
        np.array([0.25]), np.array([40.0]), np.array([8.0])
    )

    assert result.item() == 16.0


def test_micro_model_composes_probability_and_conditional_signed_outcomes() -> None:
    class ConstantRegressor:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), self.value)

    class ConstantClassifier:
        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile([0.75, 0.25], (len(values), 1))

    class IdentityCalibrator:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values)

    rows = pd.DataFrame([{name: 1.0 for name in MICRO_MODEL_FEATURES}])
    scored = score_micro_conditional_ev(
        {
            "classifier": ConstantClassifier(),
            "gain_regressor": ConstantRegressor(20.0),
            "miss_regressor": ConstantRegressor(-4.0),
            "probability_calibrator": IdentityCalibrator(),
            "ev_calibrator": IdentityCalibrator(),
        },
        rows,
    )

    assert "round_trip_cost_bps" in MICRO_MODEL_FEATURES
    assert scored["predicted_net_positive_probability"].iat[0] == 0.25
    assert scored["predicted_gross_bps"].iat[0] == 2.0


def test_historical_context_fails_closed_on_future_or_missing_data() -> None:
    now = pd.Timestamp("2025-01-01T12:00:00Z")
    rows = pd.DataFrame([{feature: 1.0 for feature in FEATURES} for _ in range(3)])
    rows["available_at"] = now
    rows["binance_context_available_at"] = [now, now + pd.Timedelta(minutes=1), now]
    rows["max_input_available_at"] = now
    rows["binance_context_coverage"] = True
    rows.loc[2, "oi_change_1h"] = np.nan

    assert _causal_context_coverage(rows).tolist() == [True, False, False]


def test_cost_linked_tp1_exits_immediately_and_stop_wins_same_minute() -> None:
    times = pd.Series(pd.date_range("2025-01-01", periods=7, freq="min", tz="UTC"))
    minutes = pd.DataFrame(
        {
            "perp_open": [100.0] * 7,
            "perp_high": [100.05] * 5 + [100.20, 100.05],
            "perp_low": [99.90] * 5 + [99.80, 99.95],
            "perp_close": [100.0] * 7,
            "funding_event_rate": 0.0,
            "data_valid": True,
        }
    )
    event = {
        "available_at": times.iat[5],
        "direction": 1,
        "stop_price": 99.90,
    }
    plan = _cost_target_plan(8.0, 1.5)

    stopped = _simulate_plan(event, minutes, times, plan)
    target_minutes = minutes.copy()
    target_minutes.loc[5, "perp_low"] = 99.95
    targeted = _simulate_plan(event, target_minutes, times, plan)

    assert stopped is not None and stopped["exit_reason"] == "DYNAMIC_STOP"
    assert stopped["gross_return_bps"] == -12.0
    assert targeted is not None and targeted["exit_reason"] == "FIRST_TARGET_FULL_EXIT"
    assert targeted["gross_return_bps"] == 12.0
    assert targeted["duration_minutes"] == 1


def test_runner_takes_half_then_protects_real_profile_cost() -> None:
    times = pd.Series(pd.date_range("2025-01-01", periods=8, freq="min", tz="UTC"))
    minutes = pd.DataFrame(
        {
            "perp_open": [100.0] * 6 + [100.08, 100.0],
            "perp_high": [100.05] * 5 + [100.121, 100.08, 100.05],
            "perp_low": [99.90] * 5 + [99.95, 100.05, 99.95],
            "perp_close": [100.0] * 8,
            "funding_event_rate": 0.0,
            "data_valid": True,
        }
    )
    event = {
        "available_at": times.iat[5],
        "direction": 1,
        "stop_price": 99.90,
    }

    outcome = _simulate_plan(
        event,
        minutes,
        times,
        _cost_runner_plan(8.0, 1.5),
        protection_cost_bps=8.0,
    )

    assert outcome is not None
    assert outcome["first_target_hit"] is True
    assert outcome["exit_reason"] == "DYNAMIC_STOP"
    assert outcome["gross_return_bps"] >= 9.0
    assert outcome["continuation_increment_bps"] < 0


def test_policy_gate_can_accept_losses_in_prediction_at_aggregate_level() -> None:
    entry = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    rows = pd.DataFrame(
        {
            "signal_timestamp": entry,
            "direction": 1,
            "entry_timestamp": entry,
            "exit_timestamp": entry + pd.Timedelta(minutes=5),
            "predicted_gross_bps": 0.0,
            "gross_return_bps": 5.0,
            "initial_stop_bps": 20.0,
        }
    )

    frontier = _policy_coverage_frontier(rows, cost_bps=1.0)

    assert frontier is not None
    assert frontier["coverage"] == 1.0
    assert frontier["threshold_bps"] == -1.0
    assert frontier["metrics"]["trades"] == 60


def test_cost_stress_is_diagnostic_not_an_operational_gate() -> None:
    entry = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    rows = pd.DataFrame(
        {
            "signal_timestamp": entry,
            "direction": 1,
            "entry_timestamp": entry,
            "exit_timestamp": entry + pd.Timedelta(minutes=5),
            "predicted_gross_bps": 1.5,
            "gross_return_bps": 1.5,
            "initial_stop_bps": 20.0,
        }
    )

    frontier = _policy_coverage_frontier(rows, cost_bps=1.0)

    assert frontier is not None
    assert frontier["metrics"]["expectancy_bps"] == 0.5
    assert frontier["stress_2x"]["expectancy_bps"] == -0.5
    assert frontier["stress_2x_nonnegative_diagnostic"] is False


def test_coverage_uses_continuous_rank_without_isotonic_tie_expansion() -> None:
    entry = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    rows = pd.DataFrame(
        {
            "signal_timestamp": entry,
            "direction": 1,
            "entry_timestamp": entry,
            "exit_timestamp": entry + pd.Timedelta(minutes=5),
            "predicted_gross_bps": 2.0,
            "predicted_uncalibrated_gross_bps": np.linspace(-1.0, 1.0, 60),
            "gross_return_bps": 5.0,
            "initial_stop_bps": 20.0,
        }
    )

    evaluations = _policy_coverage_evaluations(rows, cost_bps=1.0)

    smallest = evaluations[-1]
    assert smallest["coverage"] == 0.1
    assert smallest["selected_before_position_lock"] == 6
    assert smallest["metrics"]["trades"] == 6


def test_one_second_context_uses_only_information_already_available() -> None:
    now = pd.Timestamp("2026-01-01T12:00:05Z")
    rows = pd.DataFrame({"available_at": [now], "direction": [-1]})
    micro = pd.DataFrame(
        [
            {"available_at": now - pd.Timedelta(seconds=1)}
            | {name: 1.0 for name in ONE_SECOND_MICRO_FEATURES},
            {"available_at": now + pd.Timedelta(seconds=1)}
            | {name: 2.0 for name in ONE_SECOND_MICRO_FEATURES},
        ]
    )

    merged = merge_causal_micro(rows, micro)

    assert merged["micro_available_at"].iat[0] == now - pd.Timedelta(seconds=1)
    assert merged["micro_ofi_1s"].iat[0] == -1.0
    assert merged["micro_feature_coverage_valid"].iat[0]
