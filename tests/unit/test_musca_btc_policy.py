from __future__ import annotations

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

    rows = pd.DataFrame({name: [0.0] for name in policy.ALPHA_FEATURES})
    head: dict[str, Any] = {
        "classifier": Classifier(),
        "conditional": {0: Regressor(10), 1: Regressor(-10), 2: Regressor(0)},
        "aux": {
            "mfe_bps": Regressor(15),
            "mae_bps": Regressor(8),
            "time_to_target_seconds": Regressor(30),
        },
    }
    calibration = {
        "probability": Classifier(),
        "ev": EV(),
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
        np.asarray([0]),
        1,
        3,
        np.asarray([10.0]),
        np.asarray([30.0]),
        np.asarray([20.0]),
        np.asarray([15.0]),
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
