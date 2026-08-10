from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_bot import musca_btc_auto_moe as auto
from adaptive_bot import musca_btc_moe as previous


def test_previous_125_component_protocol_remains_frozen() -> None:
    assert (
        previous.PROTOCOL_HASH == "73563d1aed16e4f796d18d446dc9033946473429e48f18fe15c1a52ffc193ddd"
    )
    assert previous.PROTOCOL["return_experts"]["final_components"] == 125
    assert auto.PROTOCOL["parent_protocol_hash"] == previous.PROTOCOL_HASH


def test_library_size_is_discovered_instead_of_configured() -> None:
    phase = auto.PROTOCOL["phase_1"]
    assert phase["final_expert_limit"] is None
    assert phase["emergency_tree_ceiling_per_action"] == 128
    assert auto.PROTOCOL["phase_2"]["decision"].endswith("neutral FLAT")
    assert auto.PROTOCOL["phase_1"]["expert_selection_target"].startswith("managed net PnL")
    assert auto.PROTOCOL["phase_2"]["adaptation"].startswith("monthly prequential")


def test_training_contract_keeps_only_features_recreated_by_live_paper() -> None:
    assert set(auto.MODEL_FEATURES) == set(previous.FEATURES)
    assert auto.LIVE_OFFICIAL_DERIVATIVE_FEATURES.issubset(auto.MODEL_FEATURES)
    assert auto.PROTOCOL["phase_1"]["model_features"] == list(auto.MODEL_FEATURES)


def test_expert_id_is_deterministic_and_identifies_the_rule() -> None:
    first = auto._expert_id(1, 300, 7, 12)
    assert first == auto._expert_id(1, 300, 7, 12)
    assert first != auto._expert_id(-1, 300, 7, 12)
    assert first != auto._expert_id(1, 900, 7, 12)


def test_diversity_has_no_final_count_cap() -> None:
    candidates = [
        {
            "expert_id": f"e{number}",
            "side": 1,
            "horizon_seconds": 300,
            "robust_score": 10 - number,
            "signal_signature": f"s{number}",
        }
        for number in range(3)
    ]
    signals = {
        "e0": np.array([True, False, False, False]),
        "e1": np.array([False, True, False, False]),
        "e2": np.array([False, False, True, False]),
    }
    selected, rejected = auto._select_diverse(candidates, signals)
    assert len(selected) == len(candidates)
    assert not rejected


def test_correlated_or_duplicate_rules_are_rejected() -> None:
    candidates = [
        {
            "expert_id": "best",
            "side": 1,
            "horizon_seconds": 60,
            "robust_score": 2.0,
            "signal_signature": "a",
        },
        {
            "expert_id": "same",
            "side": 1,
            "horizon_seconds": 60,
            "robust_score": 1.0,
            "signal_signature": "b",
        },
    ]
    signals = {
        "best": np.array([True, True, False]),
        "same": np.array([True, True, False]),
    }
    selected, rejected = auto._select_diverse(candidates, signals)
    assert [item["expert_id"] for item in selected] == ["best"]
    assert rejected["correlated_signal"] == 1


def test_gate_rows_are_weighted_equally_per_timestamp() -> None:
    rows = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z"] * 2 + ["2026-01-01T00:01:00Z"] * 4,
                utc=True,
            )
        }
    )
    weights = auto._timestamp_weights(rows)
    totals = pd.Series(weights).groupby(rows["entry_timestamp"].reset_index(drop=True)).sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_infinite_profit_factor_has_a_finite_model_representation() -> None:
    rows = pd.DataFrame({name: [0.0] for name in auto.GATE_FEATURES})
    rows["validation_profit_factor"] = np.inf
    values = auto._gate_x(rows)
    assert np.isfinite(values).all()
    assert values[0, auto.GATE_FEATURES.index("validation_profit_factor")] == 100.0


def test_flat_is_neutral_when_every_expert_has_negative_ev() -> None:
    rows = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-06-01T00:05:00Z", "2026-06-01T00:05:00Z"], utc=True
            ),
            "expert_id": ["long", "short"],
            "calibrated_ev_bps": [-0.1, -2.0],
            "raw_ev_bps": [1.0, 2.0],
        }
    )
    assert auto._execute(rows).empty


def test_live_evaluation_explains_why_ready_policy_is_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Generator:
        def apply(self, _values: np.ndarray) -> np.ndarray:
            return np.array([[7]])

        def predict(self, _values: np.ndarray) -> np.ndarray:
            return np.array([2.0])

    expert = {
        "expert_id": "short-60m",
        "side": -1,
        "horizon_seconds": 3_600,
        "tree_index": 0,
        "leaf_id": 7,
        "target_1_bps": 20.0,
        "target_2_bps": 35.0,
        "stop_bps": 18.0,
        "trailing_bps": 12.0,
        "fit_expectancy_bps": 1.0,
        "validation_expectancy_bps": 1.0,
        "validation_profit_factor": 1.1,
        "validation_activation_rate": 0.1,
    }
    monkeypatch.setattr(
        auto,
        "_paper_bundle",
        lambda: {
            "expert_library": {
                "experts": [expert],
                "generators": {"-1:3600": Generator()},
            },
            "meta_model": {},
            "calibrators": {},
        },
    )
    monkeypatch.setattr(
        auto,
        "_live_micro_features",
        lambda *_: {name: 0.1 for name in previous.MICRO_FEATURES},
    )

    def negative_score(
        rows: pd.DataFrame, _model: dict[str, object], _calibrators: dict[str, object]
    ) -> pd.DataFrame:
        scored = rows.copy()
        scored["raw_ev_bps"] = -0.5
        scored["calibrated_ev_bps"] = -1.25
        scored["probability_net_positive"] = 0.45
        return scored

    monkeypatch.setattr(auto, "_score", negative_score)
    now = pd.Timestamp("2026-08-10T12:01:05Z")
    context = pd.DataFrame(
        [
            {
                **{name: 0.1 for name in auto.MODEL_FEATURES},
                "available_at": now - pd.Timedelta(seconds=5),
                "rolling_vwap": 60_000.0,
            }
        ]
    )
    l2 = pd.DataFrame(
        [
            {
                "exchange_second": int((now - pd.Timedelta(seconds=6)).timestamp()),
                "available_at": now - pd.Timedelta(seconds=5),
            }
        ]
    )

    result = auto.evaluate_live_actions(context, l2, now)

    assert result["status"] == "READY_FLAT"
    assert result["reason"] == "NO_POSITIVE_CALIBRATED_EV"
    assert result["candidate"] is None
    assert result["active_expert_count"] == 1
    assert result["best_action"]["expert_id"] == "short-60m"
    assert result["best_action"]["calibrated_ev_bps"] == -1.25


def test_trade_breakdown_keeps_gross_costs_and_net_separate() -> None:
    trades = pd.DataFrame(
        {
            "net_bps": [10.0, -5.0],
            "gross_bps": [19.0, 4.0],
            "funding_bps": [0.0, 0.0],
            "side": [1, -1],
            "horizon_seconds": [300, 900],
            "outcome": ["TARGET_2", "STOP"],
        }
    )
    result = auto._trade_breakdown(trades)
    assert result["gross_expectancy_bps"] == 11.5
    assert result["round_trip_cost_bps"] == 9.0
    assert auto._simple_trade_metrics(trades)["profit_factor"] == 2.0


def test_managed_expert_label_uses_the_same_exit_path_as_replay() -> None:
    source = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": np.r_[100.0, 100.0, 102.0, np.full(17, 100.0)],
            "low": np.full(20, 100.0),
            "close": np.full(20, 100.0),
        }
    )
    rows = pd.DataFrame(
        {
            "decision_position": [0],
            "entry_timestamp": pd.to_datetime(["2026-01-01T00:00:05Z"], utc=True),
        }
    )
    value = auto._managed_net(
        rows,
        np.array([0]),
        source,
        (np.array([], dtype=np.int64), np.array([0.0])),
        side=1,
        horizon=60,
        target_1=100.0,
        target_2=200.0,
        stop=50.0,
        trailing=50.0,
    )
    assert value[0] == 141.0


def test_flat_calendar_days_are_neutral_not_losing_days() -> None:
    trades = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(["2026-06-01T12:00:00Z"], utc=True),
            "net_bps": [10.0],
            "stress_1_5x_bps": [5.5],
            "stress_2x_bps": [1.0],
            "stop_bps": [50.0],
            "funding_bps": [0.0],
            "side": [1],
            "horizon_seconds": [3_600],
        }
    )
    metrics = auto._policy_metrics(
        trades,
        pd.Timestamp("2026-06-01T00:00:00Z"),
        pd.Timestamp("2026-06-04T00:00:00Z"),
    )
    assert metrics["positive_active_days"] == 1.0
    assert metrics["flat_calendar_days"] == 2
    assert metrics["nonnegative_calendar_days"] == 1.0


def test_chronology_keeps_discovery_gate_and_audit_separate() -> None:
    assert auto.DISCOVERY_FIT_END < auto.LIBRARY_FREEZE_END
    assert auto.LIBRARY_FREEZE_END < auto.GATE_TUNE_END
    assert auto.GATE_TUNE_END < auto.GATE_FIT_END
    assert auto.GATE_FIT_END < auto.CALIBRATION_END
    assert auto.CALIBRATION_END < auto.HISTORICAL_AUDIT_END
    assert auto.HISTORICAL_AUDIT_END < auto.FUTURE_HOLDOUT_START


def test_live_micro_features_match_the_historical_feature_definition() -> None:
    start = pd.Timestamp("2026-08-10T00:00:00Z")
    records: list[dict[str, object]] = []
    buckets: list[dict[str, object]] = []
    for bucket_number in range(121):
        bucket_at = start + pd.Timedelta(seconds=5 * bucket_number)
        for second in range(5):
            price = 60_000 + bucket_number + second / 10
            records.append(
                {
                    "exchange_second": int((bucket_at + pd.Timedelta(seconds=second)).timestamp()),
                    "available_at": (bucket_at + pd.Timedelta(seconds=second, milliseconds=100)),
                    "buy_quote": 2.0,
                    "sell_quote": 1.0,
                    "trade_count": 1,
                    "aggregate_trades": [[str(price), "0.001", "BUY"]],
                }
            )
        buckets.append(
            {
                "available_at": bucket_at + pd.Timedelta(seconds=5),
                "close": 60_000 + bucket_number + 0.4,
                "quote_volume": 15.0,
                "signed_quote_volume": 5.0,
                "trade_count": 5,
            }
        )
    evaluated_at = start + pd.Timedelta(seconds=5 * 121)
    actual = auto._live_micro_features(pd.DataFrame(records), evaluated_at)
    expected = previous._build_moe_micro_features(pd.DataFrame(buckets)).iloc[-1]

    assert actual is not None
    for name in previous.MICRO_FEATURES:
        assert actual[name] == pytest.approx(float(expected[name]))


def test_live_micro_features_fail_closed_on_a_recent_missing_second() -> None:
    start = pd.Timestamp("2026-08-10T00:00:00Z")
    records = [
        {
            "exchange_second": int((start + pd.Timedelta(seconds=second)).timestamp()),
            "available_at": start + pd.Timedelta(seconds=second, milliseconds=100),
            "buy_quote": 2.0,
            "sell_quote": 1.0,
            "trade_count": 1,
            "aggregate_trades": [["60000", "0.001", "BUY"]],
        }
        for second in range(605)
        if second != 600
    ]
    assert (
        auto._live_micro_features(pd.DataFrame(records), start + pd.Timedelta(seconds=605)) is None
    )
