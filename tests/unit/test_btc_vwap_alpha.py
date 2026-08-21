from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import btc_vwap_alpha


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.value)


class _IdentityCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values


def test_protocol_is_btc_only_and_holdout_stays_closed() -> None:
    assert btc_vwap_alpha.PROTOCOL["asset"] == "BTCUSDT"
    assert "ETH" not in str(btc_vwap_alpha.PROTOCOL)
    assert btc_vwap_alpha.PROTOCOL["real_capital_allowed"] is False
    assert btc_vwap_alpha.HOLDOUT_WEEKS == 12


def test_candidate_sampling_emits_one_event_per_continuous_setup_episode() -> None:
    count = 90
    timestamp = pd.date_range("2026-01-01", periods=count, freq="1min", tz="UTC")
    rows = pd.DataFrame(
        {
            "feature_available_at": timestamp,
            "is_available": True,
            "feature_contract_valid": True,
            "oi_feature_available": True,
            "return_1m_bps": 1.0,
            "return_5m_bps": 3.0,
            "return_15m_bps": 4.0,
            "return_30m_bps": 5.0,
            "vwap_distance_bps": 2.0,
            "vwap_slope_bps": 0.2,
            "range_60s_bps": 4.0,
            "taker_imbalance_60s": 0.2,
            "avwap_distance_bps": 1.0,
            "avwap_slope_bps": 0.1,
            "volatility_percentile": 0.5,
            "volume_percentile": 0.5,
            "oi_change_1h": 0.001,
            "funding_z": 0.0,
            "basis_bps": 1.0,
        }
    )
    candidates = btc_vwap_alpha._candidate_indexes(rows)
    indexes = [
        index
        for index, side, family in candidates
        if side == 1 and family == "VWAP_PULLBACK_CONTINUATION"
    ]
    assert len(indexes) == 1
    assert all(index <= count - 62 for index in indexes)


def test_candidate_generation_fails_closed_for_future_oi_or_broken_warmup() -> None:
    count = 90
    timestamp = pd.date_range("2026-01-01", periods=count, freq="1min", tz="UTC")
    rows = pd.DataFrame(
        {
            "feature_available_at": timestamp,
            "is_available": True,
            "feature_contract_valid": True,
            "oi_feature_available": False,
            "return_1m_bps": 1.0,
            "return_5m_bps": 3.0,
            "return_15m_bps": 4.0,
            "return_30m_bps": 5.0,
            "vwap_distance_bps": 2.0,
            "vwap_slope_bps": 0.2,
            "range_60s_bps": 4.0,
            "taker_imbalance_60s": 0.2,
            "volatility_percentile": 0.5,
            "volume_percentile": 0.5,
            "oi_change_1h": 0.001,
            "funding_z": 0.0,
            "basis_bps": 1.0,
        }
    )

    assert btc_vwap_alpha._candidate_indexes(rows) == []
    rows["oi_feature_available"] = True
    rows["feature_contract_valid"] = False
    assert btc_vwap_alpha._candidate_indexes(rows) == []


def test_live_features_preserve_direction_and_availability_time() -> None:
    timestamp = pd.Timestamp("2026-08-08T12:00:00Z")
    row: dict[str, object] = {
        column: 1.0 for column in btc_vwap_alpha.LIVE_REQUIRED
    }
    row |= {
        "signal_at": timestamp,
        "side": "SHORT",
        "expert": "vwap_reversion_dynamic",
    }
    rows = pd.DataFrame([row])
    rows.loc[0, "alpha_return_1m_bps"] = 3.0
    rows.loc[0, "alpha_return_5m_bps"] = 5.0
    rows.loc[0, "alpha_vwap_distance_bps"] = 4.0
    rows.loc[0, "oi_change_1h_binance"] = 0.02
    rows.loc[0, "basis_bps_binance"] = 2.0
    rows.loc[0, "funding_z_binance"] = 0.5
    for column in btc_vwap_alpha.LIVE_REQUIRED:
        if column not in rows:
            rows[column] = 1.0
    features = btc_vwap_alpha.live_features(rows).iloc[0]
    assert features["directional_return_1m_bps"] == -3.0
    assert features["directional_return_5m_bps"] == -5.0
    assert features["directional_oi_change_1h"] == -0.02
    assert features["return_oi_interaction"] == 0.1
    assert features["directional_basis_bps"] == -2.0
    assert features["directional_funding_z"] == -0.5
    assert features["directional_vwap_distance_bps"] == -4.0
    assert btc_vwap_alpha._local_expert_keys(rows).iloc[0] == "VWAP_REVERSION:SHORT"
    assert np.isfinite(features.to_numpy(float)).all()


def test_barrier_outcome_separates_target_stop_and_timeout() -> None:
    favorable = np.asarray([[35.0, 40.0], [35.0, 40.0], [10.0, 15.0]])
    adverse = np.asarray([[5.0, 10.0], [25.0, 25.0], [5.0, 10.0]])
    target, stop, timeout, realized, target_minutes, exit_minutes, ambiguous = (
        btc_vwap_alpha._barrier_outcome(
        favorable,
        adverse,
        np.asarray([20.0, 20.0, 20.0]),
        np.asarray([7.0, -9.0, 4.0]),
        30,
        )
    )

    assert target.tolist() == [True, False, False]
    assert stop.tolist() == [False, True, False]
    assert timeout.tolist() == [False, False, True]
    assert realized.tolist() == [30.0, -20.0, 4.0]
    assert target_minutes[0] == 1.0
    assert np.isnan(target_minutes[1:]).all()
    assert exit_minutes.tolist() == [1.0, 1.0, 2.0]
    assert ambiguous.tolist() == [False, True, False]


def test_funding_is_charged_only_until_each_observed_exit() -> None:
    funding_path = np.asarray(
        [
            [0.0, 0.001, 0.0],
            [0.0, -0.002, 0.0],
            [0.0, 0.001, 0.0],
        ]
    )

    funding = btc_vwap_alpha._realized_funding_bps(
        funding_path,
        np.asarray([1.0, -1.0, -1.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert funding.tolist() == [0.0, 20.0, -10.0]


def test_reentry_cross_between_decision_ticks_remains_visible_causally() -> None:
    rows = pd.DataFrame(
        {
            "return_1m_bps": [1.0] * 5,
            "return_15m_bps": [1.0] * 5,
            "return_30m_bps": [1.0] * 5,
            "vwap_distance_bps": [-2.0, -1.0, 1.0, 2.0, 3.0],
            "vwap_slope_bps": [0.1] * 5,
            "taker_imbalance_60s": [0.2] * 5,
        }
    )

    side, stages = btc_vwap_alpha.setup_conditions(
        rows, "ROLLING_VWAP_REENTRY"
    )

    assert side.iloc[-1] == 1
    assert dict(stages)["vwap_cross"].iloc[-1]


def test_missing_trend_input_fails_closed_instead_of_creating_a_direction() -> None:
    rows = pd.DataFrame(
        {
            "return_1m_bps": [1.0],
            "return_15m_bps": [np.nan],
            "return_30m_bps": [1.0],
            "vwap_distance_bps": [1.0],
            "vwap_slope_bps": [0.1],
            "taker_imbalance_60s": [0.2],
        }
    )

    side, stages = btc_vwap_alpha.setup_conditions(
        rows, "VWAP_PULLBACK_CONTINUATION"
    )

    assert np.isnan(side.iloc[0])
    assert not bool(dict(stages)["direction"].iloc[0])


def test_chronological_window_purges_labels_crossing_the_boundary() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame(
        {
            "signal_at": [start, start + pd.Timedelta(minutes=1)],
            "label_available_at": [
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=3),
            ],
        }
    )

    selected = btc_vwap_alpha._time_window(
        rows, start, start + pd.Timedelta(minutes=2)
    )

    assert selected.index.tolist() == [0]


def test_position_lock_ends_at_observed_exit_not_label_horizon() -> None:
    start = pd.Timestamp("2026-08-08T12:00:00Z")
    candidates = pd.DataFrame(
        {
            "signal_at": [start, start + pd.Timedelta(minutes=2)],
            "chosen_target_bps": [20, 20],
            "exit_at_20bps": [
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=3),
            ],
        }
    )

    selected = btc_vwap_alpha._one_position_at_a_time(candidates)

    assert len(selected) == 2


def test_counterfactual_actions_have_equal_total_weight_per_market_state() -> None:
    first = pd.Timestamp("2026-08-08T12:00:00Z")
    second = first + pd.Timedelta(minutes=1)
    rows = pd.DataFrame({"signal_at": [first, first, first, second]})

    weights = btc_vwap_alpha._state_weights(rows)
    totals = pd.Series(weights, index=rows.index).groupby(rows["signal_at"]).sum()

    assert np.isclose(totals.iloc[0], totals.iloc[1])


def test_small_policy_reports_unavailable_bootstrap_as_null() -> None:
    rows = pd.DataFrame(
        {
            "signal_at": [pd.Timestamp("2026-08-08T12:00:00Z")],
            "stop_bps": [20.0],
        }
    )

    metrics = btc_vwap_alpha._policy_metrics(rows, np.asarray([1.0]), 8.0)

    assert metrics["bootstrap_expectancy_lcb_95_bps"] is None


def test_live_reentry_routes_to_the_same_local_expert_as_training() -> None:
    row = {column: 1.0 for column in btc_vwap_alpha.LIVE_REQUIRED}
    row.update(
        signal_at=pd.Timestamp("2026-08-08T12:00:00Z"),
        side="LONG",
        expert="rolling_vwap_reentry_dynamic",
    )
    rows = pd.DataFrame([row])
    assert (
        btc_vwap_alpha._local_expert_keys(rows).iloc[0]
        == "ROLLING_VWAP_REENTRY:LONG"
    )


def test_predictions_are_bounded_to_physical_label_support() -> None:
    predictions = {
        "expected_plan_return_20bps": np.asarray([50.0, -50.0]),
        "expected_plan_return_30bps": np.asarray([50.0, -50.0]),
        "expected_plan_return_50bps": np.asarray([80.0, -50.0]),
        "expected_time_to_20bps_minutes": np.asarray([793.0, -5.0]),
        "expected_mfe_60m_bps": np.asarray([-169.0, 20.0]),
        "expected_mae_60m_bps": np.asarray([-240.0, 15.0]),
    }
    btc_vwap_alpha._bound_physical_predictions(predictions, np.asarray([25.0, 30.0]))

    assert predictions["expected_plan_return_20bps"].tolist() == [0.0, -15.0]
    assert predictions["expected_plan_return_50bps"].tolist() == [0.0, -15.0]
    assert predictions["expected_time_to_20bps_minutes"].tolist() == [60.0, 1.0]
    assert predictions["expected_mfe_60m_bps"].tolist() == [0.0, 20.0]
    assert predictions["expected_mae_60m_bps"].tolist() == [0.0, 15.0]


def test_score_emits_independent_fee_profile_decisions(monkeypatch, tmp_path) -> None:
    bundle_path = tmp_path / "bundle.joblib"
    bundle_path.touch()
    outcome_heads = {barrier: {} for barrier in btc_vwap_alpha.BARRIERS}
    regressions = {
        **{
            f"plan_return_{barrier}bps": _ConstantRegressor(
                13.0 if barrier == 30 else 12.0
            )
            for barrier in btc_vwap_alpha.BARRIERS
        },
        **{
            f"time_to_{barrier}bps_minutes": _ConstantRegressor(15.0)
            for barrier in btc_vwap_alpha.BARRIERS
        },
        **{
            f"timeout_return_{barrier}bps": _ConstantRegressor(0.0)
            for barrier in btc_vwap_alpha.BARRIERS
        },
        "mfe_60m_bps": _ConstantRegressor(100.0),
        "mae_60m_bps": _ConstantRegressor(10.0),
    }
    policy = {
        "kind": "ridge",
        "feature_support": {
            feature: (-1_000_000.0, 1_000_000.0)
            for feature in btc_vwap_alpha.FEATURES
        },
        "outcome_heads": outcome_heads,
        "regressions": regressions,
        "ev_calibrators": {
            barrier: _IdentityCalibrator() for barrier in btc_vwap_alpha.BARRIERS
        },
        "residual_lcb_bps": {
            barrier: 0.0 for barrier in btc_vwap_alpha.BARRIERS
        },
    }
    bundle = {
        "protocol_hash": btc_vwap_alpha.PROTOCOL_HASH,
        "features": btc_vwap_alpha.FEATURES,
        "experts": {key: policy for key in btc_vwap_alpha.LOCAL_EXPERTS},
    }
    monkeypatch.setattr(btc_vwap_alpha, "BUNDLE", bundle_path)
    btc_vwap_alpha._load_scoring_bundle.cache_clear()
    loads = 0

    def load_bundle(_: str) -> dict:
        nonlocal loads
        loads += 1
        return bundle

    monkeypatch.setattr(btc_vwap_alpha.joblib, "load", load_bundle)
    monkeypatch.setattr(
        btc_vwap_alpha,
        "_predict_outcome",
        lambda _head, rows: np.tile(np.asarray([0.4, 0.3, 0.3]), (len(rows), 1)),
    )
    row = {column: 1.0 for column in btc_vwap_alpha.LIVE_REQUIRED}
    row.update(
        signal_at=pd.Timestamp("2026-08-08T12:00:00Z"),
        side="LONG",
        expert="vwap_reversion_dynamic",
        stop_bps=20.0,
        expected_non_fee_cost_bps=1.0,
        alpha_feature_contract_valid=True,
    )

    scored = btc_vwap_alpha.score(pd.DataFrame([row])).iloc[0]
    btc_vwap_alpha.score(pd.DataFrame([row]))

    assert not bool(scored["alpha_accepted_vip0"])
    assert bool(scored["alpha_accepted_vip5"])
    assert scored["alpha_expected_total_cost_vip0_bps"] == 13.0
    assert scored["alpha_expected_total_cost_vip5_bps"] == 8.0
    assert scored["alpha_accepted"] == scored["alpha_accepted_vip0"]
    assert loads == 1


def test_score_fails_closed_outside_training_feature_support(monkeypatch, tmp_path) -> None:
    bundle_path = tmp_path / "bundle.joblib"
    bundle_path.touch()
    policy = {
        "kind": "ridge",
        "feature_support": {
            feature: (-10.0, 10.0) for feature in btc_vwap_alpha.FEATURES
        },
        "outcome_heads": {},
        "regressions": {},
        "ev_calibrators": {
            barrier: _IdentityCalibrator() for barrier in btc_vwap_alpha.BARRIERS
        },
        "residual_lcb_bps": {},
    }
    bundle = {
        "protocol_hash": btc_vwap_alpha.PROTOCOL_HASH,
        "features": btc_vwap_alpha.FEATURES,
        "experts": {key: policy for key in btc_vwap_alpha.LOCAL_EXPERTS},
    }
    monkeypatch.setattr(btc_vwap_alpha, "BUNDLE", bundle_path)
    monkeypatch.setattr(btc_vwap_alpha.joblib, "load", lambda _: bundle)
    row = {column: 1.0 for column in btc_vwap_alpha.LIVE_REQUIRED}
    row.update(
        signal_at=pd.Timestamp("2026-08-08T12:00:00Z"),
        side="LONG",
        expert="vwap_reversion_dynamic",
        alpha_feature_contract_valid=True,
        alpha_vwap_tests_30m=600.0,
    )

    scored = btc_vwap_alpha.score(pd.DataFrame([row])).iloc[0]

    assert not bool(scored["alpha_accepted"])
    assert scored["alpha_status"] == "FEATURE_DISTRIBUTION_MISMATCH"
    assert "vwap_tests_30m" in scored["alpha_ood_features"]
