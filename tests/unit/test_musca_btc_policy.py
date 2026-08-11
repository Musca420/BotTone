from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from adaptive_bot import cli
from adaptive_bot import musca_btc_auto_moe as frozen
from adaptive_bot import musca_btc_policy as policy


def _source(values: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(values), freq="1s"),
            "available_at": pd.date_range(start, periods=len(values), freq="1s")
            + pd.Timedelta(seconds=1),
            "open": [value[0] for value in values],
            "high": [value[1] for value in values],
            "low": [value[2] for value in values],
            "close": [value[3] for value in values],
            "trade_count": np.ones(len(values)),
        }
    )


def test_single_challenger_does_not_replace_frozen_discovery_control() -> None:
    assert policy.PROTOCOL["frozen_discovery_control"]["protocol_hash"] == frozen.PROTOCOL_HASH
    assert policy.REPORT != frozen.REPORT
    assert policy.BUNDLE != frozen.BUNDLE
    assert policy.PROTOCOL["real_capital_allowed"] is False


def test_atomic_replace_retries_transient_windows_reader_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flaky_replace(source_path: object, destination_path: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary reader lock")

    monkeypatch.setattr(policy.os, "replace", flaky_replace)
    policy._atomic_replace(policy.Path("status.tmp"), policy.Path("status.json"))
    assert attempts == 3


def test_expert_id_is_deterministic() -> None:
    assert policy.expert_id(1, 300) == policy.expert_id(1, 300)
    assert policy.expert_id(1, 300) != policy.expert_id(-1, 300)


def _inherited_expert_rows() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    rows: list[dict[str, object]] = []
    for side in (-1, 1):
        for horizon_number, horizon in enumerate(policy.PREDICTION_HORIZONS_SECONDS, start=1):
            row: dict[str, object] = {
                "available_at": timestamp,
                "entry_timestamp": timestamp + pd.Timedelta(seconds=1),
                "decision_position": 10,
                "side": side,
                "horizon_seconds": horizon,
                "predicted_favorable_q50_bps": 8.0 + 5.0 * horizon_number,
                "predicted_favorable_q75_bps": 12.0 + 8.0 * horizon_number,
                "predicted_adverse_q75_bps": 6.0 + 4.0 * horizon_number,
            }
            row.update({name: 0.0 for name in policy.base.GATING_CONTEXT})
            row["volatility_percentile"] = 0.5
            for expert_horizon in policy.PREDICTION_HORIZONS_SECONDS:
                for view_number, view in enumerate(policy.base.VIEWS, start=1):
                    row[f"expert_{expert_horizon}s_{view}"] = side * (
                        expert_horizon / 900 + view_number
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def test_multi_expert_generator_replaces_ten_templates_with_parameterized_plans() -> None:
    inherited = _inherited_expert_rows()
    plans = policy.compose_parameterized_plans(inherited, round_trip_cost_bps=8.0)
    repeated = policy.compose_parameterized_plans(inherited, round_trip_cost_bps=8.0)
    assert len(inherited) == 10
    assert len(plans) == 2
    assert plans["plan_id"].tolist() == repeated["plan_id"].tolist()
    assert (~plans["horizon_seconds"].isin(policy.PREDICTION_HORIZONS_SECONDS)).any()
    assert plans["gate_effective_experts"].gt(1).all()
    assert plans["plan_contributors"].str.count(",").eq(2).all()
    assert plans["target_2_bps"].gt(plans["target_1_bps"]).all()
    assert plans["trailing_bps"].le(plans["stop_bps"]).all()
    assert all(f"view_{view}_prediction_bps" in plans for view in policy.base.VIEWS)
    assert "equal_weight_expert_prediction_bps" in plans
    assert policy.PROTOCOL["state_action"]["fixed_action_plans"] is False


def test_plan_identity_changes_when_expert_mixture_changes() -> None:
    inherited = _inherited_expert_rows()
    baseline = policy.compose_parameterized_plans(inherited, round_trip_cost_bps=8.0)
    changed = inherited.copy()
    changed.loc[changed["side"].eq(1), "expert_21600s_flow"] += 500.0
    challenger = policy.compose_parameterized_plans(changed, round_trip_cost_bps=8.0)
    baseline_long = baseline.loc[baseline["side"].eq(1), "plan_id"].item()
    challenger_long = challenger.loc[challenger["side"].eq(1), "plan_id"].item()
    assert baseline_long != challenger_long


def test_local_plan_neighborhood_is_bounded_and_preserves_management_constraints() -> None:
    plans = policy.compose_parameterized_plans(_inherited_expert_rows(), round_trip_cost_bps=8.0)
    plans["actual_entry_timestamp"] = plans["entry_timestamp"]
    variants = policy.local_plan_variants(plans)
    assert variants.groupby(["actual_entry_timestamp", "side"]).size().le(11).all()
    assert variants["target_2_bps"].gt(variants["target_1_bps"]).all()
    assert variants["trailing_bps"].le(variants["stop_bps"]).all()
    assert (
        variants["horizon_seconds"]
        .between(min(policy.PREDICTION_HORIZONS_SECONDS), policy.MAXIMUM_HORIZON_SECONDS)
        .all()
    )
    assert {"PARTIAL_SMALLER", "PARTIAL_LARGER"}.issubset(set(variants["local_variant"]))
    assert variants["first_exit_fraction"].between(0.10, 0.90).all()
    assert variants["plan_id"].is_unique


def test_local_plan_labels_use_the_same_observed_path_for_every_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.2, 99.9, 100.1),
            (100.1, 100.3, 100.0, 100.2),
            (100.2, 100.4, 100.1, 100.3),
            (100.3, 100.3, 100.2, 100.2),
        ]
        + [(100.2, 100.3, 100.1, 100.2)] * 100
    )
    source["observed_trade"] = True
    plans = policy.compose_parameterized_plans(_inherited_expert_rows(), round_trip_cost_bps=8.0)
    plans["actual_entry_timestamp"] = pd.Timestamp("2026-01-01T00:00:01Z")
    plans["source_month"] = "2026-01"
    plans["horizon_seconds"] = 60
    plans["horizon_fraction"] = 60 / policy.MAXIMUM_HORIZON_SECONDS
    plans["target_1_bps"] = 10.0
    plans["target_2_bps"] = 20.0
    plans["stop_bps"] = 20.0
    plans["trailing_bps"] = 10.0
    monkeypatch.setattr(policy, "_load_second_window", lambda month: source)
    monkeypatch.setattr(
        policy,
        "_raw_events_for_seconds",
        lambda seconds: {
            second: [
                (second * 1_000 + offset, offset, price)
                for offset, price in enumerate([100.0, 99.0, 101.0, 98.0, 102.0])
            ]
            for second in seconds
        },
    )
    monkeypatch.setattr(
        policy,
        "_funding_for_actions",
        lambda actions: np.zeros(len(actions), dtype=float),
    )
    labelled = policy.label_local_plan_variants(plans, policy.FeeContract(2.0, 4.0, 0.0, "test"))
    assert labelled["actual_entry_timestamp"].nunique() == 1
    assert labelled["entry_price"].eq(100.0).all()
    assert np.isfinite(labelled["log_utility"]).all()
    assert labelled["execution_quality"].eq("TRADE_PATH_PROXY_NO_HISTORICAL_L2").all()


def test_ordered_event_stop_uses_first_crossing_price_and_records_slippage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = pd.Timestamp("2026-01-01T00:00:05Z")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": [second - pd.Timedelta(seconds=5)],
            "exit_timestamp": [second],
            "management_code": [policy.OUTCOME_STOP],
            "time_to_target_seconds": [-1],
            "time_to_stop_seconds": [5],
            "gross_bps": [-10.0],
            "side": [1],
            "entry_price": [100.0],
            "target_1_bps": [20.0],
            "first_exit_fraction": [0.5],
            "stop_bps": [10.0],
        }
    )
    monkeypatch.setattr(
        policy,
        "_raw_events_for_seconds",
        lambda seconds: {
            int(second.timestamp()): [
                (int(second.timestamp() * 1_000), 1, 100.0),
                (int(second.timestamp() * 1_000) + 1, 2, 99.8),
            ]
        },
    )
    refined = policy.refine_stop_fills_with_ordered_events(rows)
    assert refined.loc[0, "gross_bps"] == pytest.approx(-20.0)
    assert refined.loc[0, "stop_slippage_bps"] == pytest.approx(10.0)
    assert refined.loc[0, "event_fill_price"] == pytest.approx(99.8)
    assert refined.loc[0, "exit_event_id"] == 2
    assert bool(refined.loc[0, "event_order_refined"])


def test_event_exit_bucket_is_not_shifted_to_the_next_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = pd.Timestamp("2026-01-01T00:00:05Z")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": [bucket - pd.Timedelta(seconds=5)],
            "entry_bucket_timestamp": [bucket - pd.Timedelta(seconds=5)],
            "exit_seconds": [6],
            "exit_timestamp": [bucket + pd.Timedelta(seconds=1)],
            "management_code": [policy.OUTCOME_STOP],
            "time_to_target_seconds": [-1],
            "time_to_stop_seconds": [6],
            "gross_bps": [-10.0],
            "side": [1],
            "entry_price": [100.0],
            "target_1_bps": [20.0],
            "target_2_bps": [30.0],
            "first_exit_fraction": [0.5],
            "stop_bps": [10.0],
        }
    )
    requested: list[set[int]] = []

    def events(seconds: set[int]) -> dict[int, list[tuple[int, int, float]]]:
        requested.append(seconds)
        return {int(bucket.timestamp()): [(int(bucket.timestamp() * 1_000) + 25, 7, 99.8)]}

    monkeypatch.setattr(policy, "_raw_events_for_seconds", events)
    refined = policy.refine_stop_fills_with_ordered_events(rows)
    assert requested == [{int(bucket.timestamp())}]
    assert refined.loc[0, "exit_timestamp"] == bucket + pd.Timedelta(milliseconds=25)


def test_entry_uses_first_ordered_event_after_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = pd.Timestamp("2026-01-01T00:00:05Z")
    source = pd.DataFrame({"timestamp": [bucket]})
    actions = pd.DataFrame({"entry_timestamp": [bucket + pd.Timedelta(milliseconds=10)]})
    monkeypatch.setattr(
        policy,
        "_raw_events_for_seconds",
        lambda seconds: {
            int(bucket.timestamp()): [
                (int(bucket.timestamp() * 1_000) + 5, 1, 99.9),
                (int(bucket.timestamp() * 1_000) + 15, 2, 100.1),
            ]
        },
    )
    refined = policy._refine_entry_events(actions, source, np.asarray([0]))
    assert refined.loc[0, "actual_entry_timestamp"] == bucket + pd.Timedelta(milliseconds=15)
    assert refined.loc[0, "entry_price"] == pytest.approx(100.1)
    assert refined.loc[0, "entry_event_id"] == 2
    assert refined.loc[0, "entry_delay_seconds"] == pytest.approx(0.005)


def test_constant_plan_uses_only_reference_medians() -> None:
    rows = policy.compose_parameterized_plans(_inherited_expert_rows(), round_trip_cost_bps=8.0)
    rows["actual_entry_timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z")
    reference = pd.concat([rows, rows], ignore_index=True)
    reference.loc[: len(rows) - 1, "horizon_seconds"] = 300
    reference.loc[len(rows) :, "horizon_seconds"] = 900
    control = policy.train_median_constant_plans(rows, reference)
    assert control["horizon_seconds"].eq(600).all()
    assert control["target_2_bps"].gt(control["target_1_bps"]).all()
    assert control["trailing_bps"].le(control["stop_bps"]).all()
    assert control["plan_contributors"].eq("TRAIN_MEDIAN_CONSTANT_PLAN").all()


def test_ordered_event_archive_preserves_timestamp_and_event_id_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: policy.Path
) -> None:
    archive = tmp_path / "BTCUSDT-aggTrades-2026-01.zip"
    with policy.zipfile.ZipFile(archive, "w") as target:
        target.writestr(
            "BTCUSDT-aggTrades-2026-01.csv",
            "2,99.8,0.1,2,2,1767225605001,true\n3,100.2,0.2,3,3,1767225605002,false\n",
        )
    monkeypatch.setattr(policy.base, "MICRO_ROOT", tmp_path)
    monkeypatch.setattr(policy, "ORDERED_EVENT_ROOT", tmp_path / "ordered")
    second = 1767225605
    prices = policy._raw_event_prices_for_seconds({second})
    assert prices[second] == [99.8, 100.2]
    stored = pd.read_parquet(policy._ordered_event_path("2026-01"))
    assert stored["event_id"].tolist() == [2, 3]
    assert stored["timestamp_ms"].tolist() == [1767225605001, 1767225605002]


def test_same_second_target_stop_conflict_is_excluded_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = pd.Timestamp("2026-01-01T00:00:05Z")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": [second - pd.Timedelta(seconds=5)],
            "exit_timestamp": [second],
            "management_code": [policy.OUTCOME_STOP],
            "time_to_target_seconds": [5],
            "time_to_stop_seconds": [5],
            "gross_bps": [-10.0],
            "side": [1],
            "entry_price": [100.0],
            "target_1_bps": [20.0],
            "first_exit_fraction": [0.5],
            "stop_bps": [10.0],
        }
    )
    monkeypatch.setattr(
        policy,
        "_raw_events_for_seconds",
        lambda seconds: {},
    )
    refined = policy.refine_stop_fills_with_ordered_events(rows)
    assert bool(refined.loc[0, "same_second_conflict"])
    assert not bool(refined.loc[0, "data_valid"])


def test_same_second_target_and_stop_uses_stop_event_and_stop_management() -> None:
    source = _source([(100.0, 100.2, 99.8, 100.0)])
    result = policy.simulate_management(
        source,
        np.asarray([0]),
        1,
        1,
        np.asarray([10.0]),
        np.asarray([20.0]),
        np.asarray([10.0]),
        np.asarray([10.0]),
        backend="cpu",
    )
    assert result["event_class"][0] == policy.OUTCOME_STOP
    assert result["management_code"][0] == policy.OUTCOME_STOP
    assert result["gross_bps"][0] == pytest.approx(-10.0)


def test_trailing_stop_tightens_after_first_target() -> None:
    source = _source(
        [
            (100.0, 100.2, 100.0, 100.15),
            (100.15, 100.4, 100.15, 100.35),
            (100.2, 100.25, 100.1, 100.15),
        ]
    )
    result = policy.simulate_management(
        source,
        np.asarray([0]),
        1,
        3,
        np.asarray([10.0]),
        np.asarray([100.0]),
        np.asarray([50.0]),
        np.asarray([10.0]),
        backend="cpu",
    )
    assert result["event_class"][0] == policy.OUTCOME_TARGET
    assert result["management_code"][0] == 4
    assert result["gross_bps"][0] == pytest.approx(15.0)


def test_partial_exit_fraction_is_an_executable_plan_parameter() -> None:
    source = _source(
        [
            (100.0, 100.2, 100.0, 100.15),
            (100.15, 100.4, 100.15, 100.35),
        ]
    )
    result = policy.simulate_management(
        source,
        np.asarray([0]),
        1,
        np.asarray([2]),
        np.asarray([10.0]),
        np.asarray([30.0]),
        np.asarray([100.0]),
        np.asarray([20.0]),
        np.asarray([0.25]),
        backend="cpu",
    )
    assert result["gross_bps"][0] == pytest.approx(25.0)


def test_first_observed_trade_is_used_instead_of_invented_no_trade_fill() -> None:
    source = _source([(100, 100, 100, 100), (101, 101, 101, 101), (102, 102, 102, 102)])
    source["observed_trade"] = [True, False, True]
    requested = pd.Series(pd.to_datetime(["2026-01-01T00:00:01Z"], utc=True))
    positions, delay = policy._first_observed_positions(source, requested)
    assert positions.tolist() == [2]
    assert delay.tolist() == [1.0]


def test_target_probability_means_target_before_stop() -> None:
    class Classifier:
        classes_ = np.asarray([0, 1, 2])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.7, 0.2, 0.1]]), (len(values), 1))

    class Regressor:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), self.value)

    class EV:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return values

    rows = pd.DataFrame({name: [0.0] for name in policy.MODEL_FEATURES})
    head: dict[str, Any] = {
        "classifier": Classifier(),
        "conditional": {0: Regressor(10), 1: Regressor(-10), 2: Regressor(0)},
        "conditional_utility": {
            0: Regressor(0.01),
            1: Regressor(-0.01),
            2: Regressor(0),
        },
        "direct": {"net_bps": Regressor(1), "log_utility": Regressor(0.001)},
        "aux": {
            "mfe_bps": Regressor(15),
            "mae_bps": Regressor(8),
            "time_to_target_seconds": Regressor(30),
            "exit_seconds": Regressor(60),
        },
    }
    calibration = {
        "probability": Classifier(),
        "value": {name: {"net_bps": EV(), "log_utility": EV()} for name in policy.VALUE_HEADS},
        "residual_quantiles": {"mfe_bps": [0, 0, 0], "mae_bps": [0, 0, 0]},
    }
    scored = policy.score_actions(rows, head, calibration)
    assert scored.loc[0, "target_probability"] == pytest.approx(0.7)
    assert scored.loc[0, "p_stop"] == pytest.approx(0.2)
    assert scored.loc[0, "p_timeout"] == pytest.approx(0.1)


def test_position_can_cross_midnight_without_forced_close() -> None:
    entry = pd.Timestamp("2026-01-01T23:59:30Z")
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": [entry],
            "exit_timestamp": [entry + pd.Timedelta(minutes=5)],
            "calibrated_ev_bps": [5.0],
            "p_target": [0.6],
            "expert_id": [policy.expert_id(1, 300)],
            "side": [1],
            "stop_bps": [50.0],
            "net_bps": [10.0],
            "funding_bps": [0.0],
            "stress_1_5x_bps": [5.5],
            "stress_2x_bps": [1.0],
            "time_to_target_seconds": [-1],
            "exit_seconds": [300],
            "outcome": ["TIMEOUT"],
        }
    )
    trades, decisions = policy.sequential_replay(scored, 0.0, 9.0)
    assert len(trades) == 1
    assert trades.loc[0, "exit_timestamp"].day == 2
    assert decisions["action"].tolist() == ["ENTER_LONG", "CLOSE"]
    close = decisions.loc[decisions["action"].eq("CLOSE")].iloc[0]
    assert close["position_side"] == 1
    assert close["time_in_position_seconds"] == 300


def test_parameterized_plan_exposes_partial_exit_and_trailing_actions() -> None:
    entry = pd.Timestamp("2026-01-01T12:00:00Z")
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": [entry],
            "exit_timestamp": [entry + pd.Timedelta(minutes=5)],
            "calibrated_ev_bps": [5.0],
            "p_target": [0.6],
            "expert_id": ["plan-long-example"],
            "side": [1],
            "stop_bps": [50.0],
            "trailing_bps": [18.0],
            "target_1_bps": [20.0],
            "first_exit_fraction": [0.4],
            "net_bps": [30.0],
            "funding_bps": [0.0],
            "stress_1_5x_bps": [26.0],
            "stress_2x_bps": [22.0],
            "time_to_target_seconds": [60],
            "exit_seconds": [300],
            "outcome": ["TARGET_2"],
        }
    )
    _, decisions = policy.sequential_replay(scored, 0.0, 8.0)
    actions = decisions["action"].tolist()
    assert actions == [
        "ENTER_LONG",
        "REDUCE",
        "TIGHTEN_STOP",
        "UPDATE_TRAIL",
        "CLOSE",
    ]
    reduction = decisions.loc[decisions["action"].eq("REDUCE")].iloc[0]
    assert reduction["reduce_fraction"] == pytest.approx(0.4)


def test_open_position_does_not_reveal_its_future_pnl_to_risk_state() -> None:
    start = pd.Timestamp("2026-01-01T10:00:00Z")
    entries = [start, start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=3)]
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": entries,
            "exit_timestamp": [
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=4),
            ],
            "calibrated_ev_bps": [5.0, 5.0, -1.0],
            "p_target": [0.6, 0.6, 0.4],
            "expert_id": [policy.expert_id(1, 300)] * 3,
            "side": [1, 1, 1],
            "stop_bps": [50.0] * 3,
            "net_bps": [-59.0, 10.0, 10.0],
            "funding_bps": [0.0] * 3,
            "stress_1_5x_bps": [-63.5, 5.5, 5.5],
            "stress_2x_bps": [-68.0, 1.0, 1.0],
            "time_to_target_seconds": [-1] * 3,
            "exit_seconds": [120, 60, 60],
            "outcome": ["STOP", "TARGET", "TARGET"],
        }
    )
    _, decisions = policy.sequential_replay(scored, 0.0, 9.0)
    hold = decisions.loc[decisions["action"].eq("HOLD")].iloc[0]
    wait_after_exit = decisions.loc[
        decisions["action"].eq("WAIT") & decisions["timestamp"].eq(entries[2])
    ].iloc[0]
    assert hold["daily_pnl_fraction"] == pytest.approx(0.0)
    assert hold["time_in_position_seconds"] == 60
    assert wait_after_exit["daily_pnl_fraction"] < 0


def test_negative_result_has_explicit_non_operational_verdict() -> None:
    economics = {"has_positive_unconditional_action": False, "oracle_mean_net_bps": -1.0}
    assert policy._verdict(economics, [], {}, {}) == "NO_ECONOMIC_ACTION_SET"


def test_every_chronological_boundary_can_purge_by_actual_exit() -> None:
    boundary = pd.Timestamp("2026-01-02T00:00:00Z")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": [
                boundary - pd.Timedelta(hours=2),
                boundary - pd.Timedelta(hours=1),
            ],
            "exit_timestamp": [
                boundary - pd.Timedelta(minutes=1),
                boundary + pd.Timedelta(minutes=1),
            ],
        }
    )
    purged = policy._period(rows, None, boundary, purge_exit=True)
    assert len(purged) == 1
    assert purged.iloc[0]["exit_timestamp"] < boundary


def test_flat_fold_is_not_mislabeled_as_calibration_failure() -> None:
    economics = {"has_positive_unconditional_action": True, "oracle_mean_net_bps": 1.0}
    metrics = {
        "trades": policy.MINIMUM_OOS_TRADES,
        "expectancy_bps": 1.0,
        "daily_lcb_95": 0.0001,
        "weekly_lcb_95": 0.0001,
        "profit_factor": 1.2,
        "maximum_drawdown": 0.01,
        "positive_active_days": 0.6,
        "risk_violations": 0,
    }
    folds = [
        {
            "selected_threshold_bps": None,
            "test_metrics": {"trades": 1},
            "candidate_metrics": {
                "ridge": {
                    "brier": 0.5,
                    "ev_calibration_error_bps": 1.0,
                    "ev_mae_bps": 2.0,
                    "decision_regret_bps": 3.0,
                }
            },
        }
    ]
    sides = {"LONG": {"gates": {"stable": True}}}
    assert policy._verdict(economics, folds, metrics, sides) == "RESEARCH_PAPER_READY"


def test_decision_cost_contains_no_invented_non_fee_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Schedule:
        maker_bps = 2.0
        taker_bps = 4.0
        source = "test"

    monkeypatch.setattr(policy, "load_config", lambda _: object())
    monkeypatch.setattr(policy, "fee_schedule", lambda _: Schedule())
    fee = policy.resolve_fee_contract()
    assert fee.reserve_round_trip_bps == 0.0
    assert fee.round_trip_bps == 8.0


def test_four_week_selection_does_not_require_twenty_week_bootstrap() -> None:
    metrics = {
        "expectancy_bps": 2.0,
        "daily_lcb_95": 0.0001,
        "weekly_lcb_95": None,
        "profit_factor": 1.2,
        "maximum_drawdown": 0.02,
        "positive_active_days": 0.6,
        "risk_violations": 0,
    }
    assert all(policy.selection_gates(metrics).values())
    assert not policy.policy_gates(metrics)["lower_confidence_bound_positive"]


def test_fold_experts_are_filtered_only_after_managed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "MINIMUM_EXPERT_OPPORTUNITIES", 2)
    leaves = np.asarray([[1], [1], [2], [2]])
    net = np.asarray([10.0, 5.0, -10.0, -5.0])
    event = np.asarray([policy.OUTCOME_TARGET, policy.OUTCOME_TARGET, policy.OUTCOME_STOP, 2])
    _, catalog = policy._leaf_statistics(
        leaves,
        net,
        event,
        fold_scope="fold-a",
        side=1,
        horizon=300,
    )
    losing = next(item for item in catalog if item["leaf"] == 2)
    assert losing["managed_outcomes_evaluated"] is True
    assert losing["managed_expectancy_bps"] < 0
    assert losing["eligible_after_managed_evaluation"] is True
    assert losing["elimination_reason"] is None
    assert all(item["terminal_return_prefilter"] is False for item in catalog)


def test_side_specific_threshold_does_not_let_blocked_long_hide_short() -> None:
    entry = pd.Timestamp("2026-01-01T10:00:00Z")
    common = {
        "actual_entry_timestamp": [entry, entry],
        "exit_timestamp": [entry + pd.Timedelta(minutes=1)] * 2,
        "p_target": [0.6, 0.6],
        "expert_id": [policy.expert_id(1, 60), policy.expert_id(-1, 60)],
        "horizon_seconds": [60, 60],
        "stop_bps": [50.0, 50.0],
        "net_bps": [20.0, 10.0],
        "funding_bps": [0.0, 0.0],
        "stress_1_5x_bps": [16.0, 6.0],
        "stress_2x_bps": [12.0, 2.0],
        "time_to_target_seconds": [-1, -1],
        "exit_seconds": [60, 60],
        "outcome": ["TARGET", "TARGET"],
    }
    scored = pd.DataFrame(common | {"calibrated_ev_bps": [5.0, 3.0], "side": [1, -1]})
    trades, _ = policy.sequential_replay(scored, {1: 10.0, -1: 0.0}, 8.0)
    assert len(trades) == 1
    assert trades.iloc[0]["side"] == -1


def test_model_features_exclude_legacy_terminal_expert_outputs() -> None:
    assert not set(policy.MODEL_FEATURES).intersection(policy.base.EXPERT_COLUMNS)
    assert set(policy.FOLD_EXPERT_FEATURES).issubset(policy.MODEL_FEATURES)


def test_fold_expert_application_does_not_read_future_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Generator:
        def apply(self, values: np.ndarray) -> np.ndarray:
            return np.where(values[:, :1] >= 0, 1, 2)

        def predict(self, values: np.ndarray) -> np.ndarray:
            return values[:, 0]

    monkeypatch.setattr(policy, "MINIMUM_EXPERT_OPPORTUNITIES", 2)
    leaves = np.asarray([[1], [1], [2], [2]])
    statistics, _ = policy._leaf_statistics(
        leaves,
        np.asarray([10.0, 5.0, -10.0, -5.0]),
        np.asarray([0, 0, 1, 2]),
        fold_scope="fold-a",
        side=1,
        horizon=300,
    )
    rows = pd.DataFrame({name: [1.0, -1.0] for name in policy.base.GATING_CONTEXT})
    rows["side"] = 1
    rows["horizon_seconds"] = 300
    rows["expert_id"] = policy.expert_id(1, 300)
    library = {
        "fold_scope": "fold-a",
        "groups": {
            (1, 300): {
                "model": Generator(),
                "statistics": statistics,
                "fallback": {
                    "mean": 0.0,
                    "q90": 0.0,
                    "lcb": 0.0,
                    "positive_fraction": 0.5,
                    "target_rate": 0.5,
                    "stop_rate": 0.5,
                    "count": 4,
                },
            }
        },
    }
    transformed = policy.apply_fold_expert_library(rows, library)
    assert transformed["managed_expert_mean_bps"].tolist() == [7.5, -7.5]
    assert np.isfinite(transformed.loc[:, policy.FOLD_EXPERT_FEATURES]).all().all()


def test_fold_expert_fit_encoding_is_strictly_past_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: policy.Path
) -> None:
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    timestamps = pd.date_range(start, periods=140, freq="1D")
    rows = pd.DataFrame(
        [
            {
                "actual_entry_timestamp": timestamp,
                "exit_timestamp": timestamp + pd.Timedelta(hours=1),
                "side": side,
                "plan_id": f"{timestamp.isoformat()}:{side}",
            }
            for timestamp in timestamps
            for side in (-1, 1)
        ]
    )
    fitted_histories: list[pd.Timestamp] = []

    def fake_fit(
        fit: pd.DataFrame,
        fold_number: int | str,
        *,
        resume: bool = False,
        progress: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        del resume, progress
        history_end = pd.to_datetime(fit["exit_timestamp"], utc=True).max()
        if "crossfit-" in str(fold_number):
            fitted_histories.append(history_end)
        return {"fold_scope": str(fold_number), "history_end": history_end}

    def fake_apply(rows: pd.DataFrame, library: dict[str, Any]) -> pd.DataFrame:
        output = rows.copy()
        encoded = float(pd.Timestamp(library["history_end"]).timestamp())
        for name in policy.FOLD_EXPERT_FEATURES:
            output[name] = encoded
        output["fold_expert_scope"] = str(library["fold_scope"])
        output["expert_tree_index"] = 0
        output["expert_leaf_id"] = 0
        return output

    monkeypatch.setattr(policy, "EXPERT_CATALOG_ROOT", tmp_path)
    monkeypatch.setattr(policy, "fit_fold_expert_library", fake_fit)
    monkeypatch.setattr(policy, "apply_fold_expert_library", fake_apply)
    transformed, _, diagnostics = policy.cross_fit_fold_expert_features(rows, 1)
    encoded_end = pd.to_datetime(transformed["managed_generator_score_bps"], unit="s", utc=True)
    entry = pd.to_datetime(transformed["actual_entry_timestamp"], utc=True)
    assert encoded_end.lt(entry).all()
    assert fitted_histories
    assert diagnostics["strictly_past_only"] is True
    assert diagnostics["warmup_rows_excluded"] > 0


def test_policy_selection_allows_losing_trades_when_net_equity_is_positive() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    entries = [start, start + pd.Timedelta(minutes=2)]
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": entries,
            "exit_timestamp": [value + pd.Timedelta(minutes=1) for value in entries],
            "calibrated_ev_bps": [5.0, 5.0],
            "p_target": [0.4, 0.7],
            "expert_id": [policy.expert_id(1, 60)] * 2,
            "side": [1, 1],
            "horizon_seconds": [60, 60],
            "stop_bps": [50.0, 50.0],
            "net_bps": [-10.0, 20.0],
            "funding_bps": [0.0, 0.0],
            "stress_1_5x_bps": [-14.0, 16.0],
            "stress_2x_bps": [-18.0, 12.0],
            "time_to_target_seconds": [-1, 30],
            "exit_seconds": [60, 60],
            "outcome": ["STOP", "TARGET"],
        }
    )
    fee = policy.FeeContract(2.0, 4.0, 0.0, "test")
    threshold, frontier = policy._choose_frequency_threshold(
        scored, fee, start, start + pd.Timedelta(days=1)
    )
    assert threshold == 0.0
    selected = next(item for item in frontier if item["threshold_bps"] == threshold)
    assert selected["metrics"]["trades"] == 2
    assert selected["metrics"]["win_rate"] == 0.5
    assert selected["final_statistical_gates_applied"] is False
    policy.json.dumps(frontier, allow_nan=False)


def test_equity_utility_ranks_same_bps_by_risk_sized_impact() -> None:
    entry = pd.Timestamp("2026-01-01T10:00:00Z")
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": [entry, entry],
            "exit_timestamp": [entry + pd.Timedelta(minutes=1)] * 2,
            "calibrated_ev_bps": [5.0, 5.0],
            "expected_log_utility": [0.001, 0.002],
            "p_target": [0.6, 0.6],
            "expert_id": ["wide-stop", "tight-stop"],
            "side": [1, -1],
            "stop_bps": [100.0, 40.0],
            "net_bps": [5.0, 5.0],
            "funding_bps": [0.0, 0.0],
            "stress_1_5x_bps": [1.0, 1.0],
            "stress_2x_bps": [-3.0, -3.0],
            "time_to_target_seconds": [-1, -1],
            "exit_seconds": [60, 60],
            "outcome": ["TIMEOUT", "TIMEOUT"],
        }
    )
    trades, _ = policy.sequential_replay(scored, 0.0, 8.0)
    assert len(trades) == 1
    assert trades.iloc[0]["expert_id"] == "tight-stop"


def test_outcome_permutation_keeps_event_path_targets_together() -> None:
    rows = pd.DataFrame(
        {
            "event_class": [0, 1, 2],
            "net_bps": [10.0, -20.0, 3.0],
            "log_utility": [0.01, -0.02, 0.003],
            "mfe_bps": [11.0, 1.0, 4.0],
            "mae_bps": [1.0, 21.0, 2.0],
            "time_to_target_seconds": [5, -1, -1],
            "exit_seconds": [5, 8, 10],
        }
    )
    permuted = policy._permute_outcomes(rows, 42)
    original_tuples = set(map(tuple, rows.to_numpy()))
    assert set(map(tuple, permuted.to_numpy())) == original_tuples


def test_wait_has_continuation_value_instead_of_constant_zero() -> None:
    class Constant:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), 0.001)

    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame({name: [0.0, 0.0] for name in policy.MODEL_FEATURES})
    rows["actual_entry_timestamp"] = [start, start + pd.Timedelta(minutes=10)]
    rows["exit_timestamp"] = [
        start + pd.Timedelta(hours=3),
        start + pd.Timedelta(minutes=11),
    ]
    rows["side"] = [1, 1]
    rows["log_utility"] = [0.0003, 0.0010]
    assert policy.continuation_targets(rows).loc[0, "target_q_wait_log_utility"] == pytest.approx(
        0.0
    )
    targets = policy.continuation_targets(rows, Constant())
    first = targets.iloc[0]
    assert first["target_q_wait_log_utility"] > 0
    assert first["target_q_wait_log_utility"] > first["target_q_enter_log_utility"]
    assert targets["continuation_target_source"].eq("SINGLE_FITTED_IMMEDIATE_VALUE").all()


def test_double_backup_does_not_value_the_action_selected_by_the_same_noisy_model() -> None:
    class Fixed:
        def __init__(self, prediction: list[float]) -> None:
            self.prediction = np.asarray(prediction, dtype=float)

        def predict(self, values: np.ndarray) -> np.ndarray:
            return self.prediction[: len(values)]

    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame({name: [0.0] * 4 for name in policy.MODEL_FEATURES})
    rows["actual_entry_timestamp"] = [
        start,
        start,
        start + pd.Timedelta(minutes=1),
        start + pd.Timedelta(minutes=1),
    ]
    value = policy._double_state_value(
        rows,
        (Fixed([10.0, 0.0, 10.0, 0.0]), Fixed([-10.0, 1.0, -10.0, 1.0])),
    )
    assert value.eq(0.0).all()


def test_predicted_utility_cannot_be_positive_when_predicted_net_ev_is_negative() -> None:
    class Classifier:
        classes_ = np.asarray([0, 1, 2])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.7, 0.2, 0.1]]), (len(values), 1))

    class Constant:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), self.value)

    class Identity:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return values

    rows = pd.DataFrame({name: [0.0] for name in policy.MODEL_FEATURES})
    rows["sized_leverage"] = 10.0
    rows["horizon_seconds"] = 30
    head: dict[str, Any] = {
        "classifier": Classifier(),
        "conditional": {event: Constant(-10.0) for event in range(3)},
        "conditional_utility": {event: Constant(0.001) for event in range(3)},
        "direct": {"net_bps": Constant(-10.0), "log_utility": Constant(0.001)},
        "aux": {
            "mfe_bps": Constant(1.0),
            "mae_bps": Constant(1.0),
            "time_to_target_seconds": Constant(60.0),
            "exit_seconds": Constant(60.0),
        },
    }
    calibration = {
        "probability": Classifier(),
        "value": {
            name: {"net_bps": Identity(), "log_utility": Identity()} for name in policy.VALUE_HEADS
        },
        "residual_quantiles": {"mfe_bps": [0, 0, 0], "mae_bps": [0, 0, 0]},
    }
    scored = policy.score_actions(rows, head, calibration)
    for name in policy.VALUE_HEADS:
        assert scored.loc[0, f"{name}_ev_bps"] < 0
        assert scored.loc[0, f"{name}_log_utility"] < 0
        assert bool(scored.loc[0, f"{name}_utility_consistency_clipped"])
    assert scored.loc[0, "expected_holding_seconds"] == 30
    assert scored.loc[0, "expected_time_to_target_seconds"] == 30


def test_controller_falls_back_to_positive_myopic_policy_when_continuation_is_unfit() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    entries = pd.date_range(start, periods=policy.MINIMUM_CONTROLLER_SELECTION_TRADES, freq="10min")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": entries,
            "exit_timestamp": entries + pd.Timedelta(minutes=1),
            "calibrated_ev_bps": 10.0,
            "immediate_expected_log_utility": 0.001,
            "action_advantage_log_utility": -0.001,
            "expected_log_utility": -0.001,
            "p_target": 0.6,
            "expert_id": "myopic-control",
            "side": 1,
            "stop_bps": 50.0,
            "net_bps": 10.0,
            "funding_bps": 0.0,
            "stress_1_5x_bps": 6.0,
            "stress_2x_bps": 2.0,
            "time_to_target_seconds": 30,
            "exit_seconds": 60,
            "outcome": "TARGET",
        }
    )
    selected, audit = policy.select_entry_controller(
        rows,
        policy.FeeContract(2.0, 4.0, 0.0, "test"),
        start,
        start + pd.Timedelta(days=28),
        policy.MINIMUM_CONTINUATION_CROSSFIT_BLOCKS,
    )
    assert selected == "MYOPIC"
    assert audit["MYOPIC"]["eligible"] is True
    assert audit["CONTINUATION"]["eligible"] is False


def test_continuation_cannot_rescue_negative_immediate_utility() -> None:
    class Constant:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.full(len(values), self.value)

    class Identity:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return values

    rows = pd.DataFrame({name: [0.0] for name in policy.MODEL_FEATURES})
    rows["actual_entry_timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z")
    rows["expected_log_utility"] = -0.001
    scored = policy.score_continuation(
        rows,
        {
            "enter": Constant(0.5),
            "wait": Constant(0.4),
            "advantage": Constant(0.1),
        },
        {"enter": Identity(), "wait": Identity(), "advantage": Identity()},
    )
    assert bool(scored.loc[0, "continuation_dominance_violation"])
    assert scored.loc[0, "expected_log_utility"] == pytest.approx(-0.001)


def test_continuation_fit_uses_strictly_past_fitted_backup() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamps = pd.date_range(start, periods=24 * 70, freq="1h")
    rows = pd.DataFrame(
        {name: np.tile([0.0, 0.1], len(timestamps)) for name in policy.MODEL_FEATURES}
    )
    rows["actual_entry_timestamp"] = np.repeat(timestamps, 2)
    rows["exit_timestamp"] = rows["actual_entry_timestamp"] + pd.Timedelta(minutes=30)
    rows["side"] = np.tile([1, -1], len(timestamps))
    rows["log_utility"] = np.tile([0.0002, -0.0001], len(timestamps))
    models = policy.fit_continuation_models(rows)
    assert models["crossfit"]["strictly_past_only"] is True
    assert models["crossfit"]["blocks"] >= 1
    assert models["crossfit"]["rows"] > 100


def test_past_stop_overrun_reserve_keeps_realized_risk_inside_budget() -> None:
    entry = pd.Timestamp("2026-01-01T10:00:00Z")
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": [entry],
            "exit_timestamp": [entry + pd.Timedelta(minutes=1)],
            "calibrated_ev_bps": [5.0],
            "expected_log_utility": [0.001],
            "p_target": [0.1],
            "expert_id": ["stop-overrun"],
            "side": [1],
            "stop_bps": [50.0],
            "net_bps": [-58.1],
            "funding_bps": [0.0],
            "stress_1_5x_bps": [-62.1],
            "stress_2x_bps": [-66.1],
            "time_to_target_seconds": [-1],
            "exit_seconds": [60],
            "outcome": ["STOP"],
        }
    )
    reserve = policy.fit_stop_loss_overrun_reserve(rows, 8.0)
    assert reserve == pytest.approx(0.1)
    sized = policy.apply_risk_sizing_contract(rows, 8.0, reserve)
    trades, _ = policy.sequential_replay(sized, 0.0, 8.0)
    assert len(trades) == 1
    assert not bool(trades.loc[0, "risk_violation"])
    assert -trades.loc[0, "portfolio_return"] <= policy.RISK_PER_TRADE + 1e-12


def test_replay_vetoes_new_risk_before_maximum_drawdown_can_be_exceeded() -> None:
    start = pd.Timestamp("2026-01-01T10:00:00Z")
    entries = [start + pd.Timedelta(days=index) for index in range(12)]
    rows = pd.DataFrame(
        {
            "actual_entry_timestamp": entries,
            "exit_timestamp": [value + pd.Timedelta(minutes=1) for value in entries],
            "calibrated_ev_bps": [5.0] * len(entries),
            "expected_log_utility": [0.001] * len(entries),
            "p_target": [0.1] * len(entries),
            "expert_id": ["drawdown-test"] * len(entries),
            "side": [1] * len(entries),
            "stop_bps": [50.0] * len(entries),
            "net_bps": [-58.0] * len(entries),
            "funding_bps": [0.0] * len(entries),
            "stress_1_5x_bps": [-62.0] * len(entries),
            "stress_2x_bps": [-66.0] * len(entries),
            "time_to_target_seconds": [-1] * len(entries),
            "exit_seconds": [60] * len(entries),
            "outcome": ["STOP"] * len(entries),
        }
    )
    trades, decisions = policy.sequential_replay(rows, 0.0, 8.0)
    equity = np.cumprod(1 + trades["portfolio_return"].to_numpy(float))
    drawdown = 1 - equity / np.maximum.accumulate(np.r_[1.0, equity])[1:]
    assert float(drawdown.max(initial=0.0)) <= policy.MAXIMUM_DRAWDOWN
    assert "MAXIMUM_DRAWDOWN_VETO" in decisions["reason"].tolist()


def test_empty_threshold_frontier_is_strict_json() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    scored = pd.DataFrame(
        {
            "actual_entry_timestamp": [start],
            "exit_timestamp": [start + pd.Timedelta(minutes=1)],
            "calibrated_ev_bps": [-1.0],
            "p_target": [0.4],
            "expert_id": [policy.expert_id(1, 60)],
            "side": [1],
            "horizon_seconds": [60],
            "stop_bps": [50.0],
            "net_bps": [-10.0],
            "funding_bps": [0.0],
            "stress_1_5x_bps": [-14.0],
            "stress_2x_bps": [-18.0],
            "time_to_target_seconds": [-1],
            "exit_seconds": [60],
            "outcome": ["STOP"],
        }
    )
    fee = policy.FeeContract(2.0, 4.0, 0.0, "test")
    threshold, frontier = policy._choose_frequency_threshold(
        scored, fee, start, start + pd.Timedelta(days=1)
    )
    assert not policy.math.isfinite(threshold)
    assert all(item["selection_utility"] is None for item in frontier)
    policy.json.dumps(frontier, allow_nan=False)
    _, decisions = policy.sequential_replay(scored, 0.0, fee.round_trip_bps)
    assert decisions.loc[0, "candidate_calibrated_ev_bps"] == pytest.approx(-1.0)
    assert decisions.loc[0, "candidate_side"] == 1


def test_full_training_is_forbidden_without_passing_same_protocol_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy, "PREFLIGHT_REPORT", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="preflight-only passes"):
        policy.train(resume=True)


def test_preflight_requires_proportional_economic_and_causal_gates() -> None:
    metrics = {
        "trades": 60,
        "expectancy_bps": 2.0,
        "daily_lcb_95": 0.0001,
        "profit_factor": 1.2,
        "maximum_drawdown": 0.02,
        "positive_active_days": 0.6,
        "risk_violations": 0,
    }
    fold = {
        "test_metrics": {"expectancy_bps": 1.0},
        "economic_calibration": {"utility_ev_consistency_violations": 0},
        "continuation_value_audit": {
            "target_source": "DOUBLE_TEMPORAL_FITTED_IMMEDIATE_VALUE",
            "crossfit": {
                "strictly_past_only": True,
                "blocks": policy.MINIMUM_CONTINUATION_CROSSFIT_BLOCKS,
            },
        },
        "entry_controller_selection": {
            "LONG": {
                "selection_period_only": True,
                "outer_test_read_for_selection": False,
            },
            "SHORT": {
                "selection_period_only": True,
                "outer_test_read_for_selection": False,
            },
        },
        "local_plan_variants_enabled": True,
        "local_plan_training_support": {"fit": {"added_rows": 1}},
    }
    gates = policy.preflight_gates(metrics, [fold, fold], 10)
    assert all(gates.values())
    inconsistent = [fold, fold | {"economic_calibration": {}}]
    assert not policy.preflight_gates(metrics, inconsistent, 10)["utility_ev_consistent"]


def test_gpu_and_cpu_paths_are_equivalent_when_cuda_is_available() -> None:
    if not policy._gpu_info().get("available"):
        pytest.skip("CUDA unavailable")
    source = _source(
        [
            (100.0, 100.1, 99.95, 100.05),
            (100.05, 100.3, 100.0, 100.25),
            (100.25, 100.4, 100.1, 100.2),
        ]
    )
    arguments = (
        source,
        np.asarray([0, 0]),
        1,
        np.asarray([2, 3]),
        np.asarray([10.0, 10.0]),
        np.asarray([30.0, 30.0]),
        np.asarray([20.0, 20.0]),
        np.asarray([15.0, 15.0]),
        np.asarray([0.25, 0.65]),
    )
    cpu = policy.simulate_management(*arguments, backend="cpu")
    gpu = policy.simulate_management(*arguments, backend="cuda")
    for name in cpu:
        if np.issubdtype(cpu[name].dtype, np.floating):
            assert np.allclose(cpu[name], gpu[name], atol=policy.GPU_CPU_TOLERANCE_BPS)
        else:
            assert np.array_equal(cpu[name], gpu[name])


def test_cli_exposes_only_the_single_policy_training_flow() -> None:
    parsed = cli._parser().parse_args(["musca-btc-policy-train", "--resume"])
    assert parsed.command == "musca-btc-policy-train"
    assert parsed.resume
    status = cli._parser().parse_args(["musca-btc-policy-status", "--watch"])
    assert status.command == "musca-btc-policy-status"
    assert status.watch
