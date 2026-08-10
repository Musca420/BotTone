from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_bot import musca_v5_economic_alpha as alpha
from adaptive_bot.dashboard.server import build_musca_v5_economic_payload


class _ProbabilityModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.5, 0.25, 0.25]]), (len(values), 1))


class _Calibrator:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.5, 0.25, 0.25]]), (len(values), 1))


class _TimeoutModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), 10.0)


def test_vip0_uses_conservative_taker_round_trip() -> None:
    assert alpha._costs(0) == (13.0, 13.0)


def test_vip_profiles_reduce_cost_without_changing_alpha() -> None:
    costs = [alpha._costs(level) for level in range(6)]
    assert costs == [
        (13.0, 13.0),
        (11.0, 11.0),
        (11.0, 11.0),
        (9.0, 9.0),
        (8.5, 8.5),
        (8.0, 8.0),
    ]
    assert alpha._costs(0, multiplier=2) == (26.0, 26.0)


def test_broad_depth_is_excluded_when_live_definition_cannot_match() -> None:
    assert not any("depth" in feature for feature in alpha.FEATURES)
    assert alpha.PROTOCOL["historical_depth_in_alpha"] is False
    assert "diagnostic-only" in alpha.PROTOCOL["depth_transfer"]


def test_structured_ev_uses_target_stop_and_timeout_costs() -> None:
    rows = pd.DataFrame({feature: [0.0] for feature in alpha.FEATURES})
    rows["initial_stop_bps"] = -20.0
    plan = {
        "target_bps": 40,
        "horizon_minutes": 15,
        "outcome_model": _ProbabilityModel(),
        "calibrator": _Calibrator(),
        "timeout_model": _TimeoutModel(),
    }
    scored = alpha._score_plan(plan, rows)
    assert scored["predicted_net_ev_bps"].iat[0] == 4.5
    assert scored[["p_target", "p_stop", "p_timeout"]].sum(axis=1).iat[0] == 1.0


def test_one_position_uses_chosen_plan_horizon() -> None:
    rows = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:10:00Z", "2026-06-01T00:31:00Z"]
            ),
            "entry_timestamp": pd.to_datetime(
                ["2026-06-01T00:00:05Z", "2026-06-01T00:10:05Z", "2026-06-01T00:31:05Z"]
            ),
            "horizon_minutes": [30, 5, 5],
            "predicted_net_ev_bps": [2.0, 3.0, 1.0],
        }
    )
    selected = alpha._one_position(rows, threshold=0.0)
    assert selected.index.tolist() == [0, 2]


def test_one_position_releases_on_observed_exit() -> None:
    rows = pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:10:00Z"]),
            "entry_timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:10:00Z"]),
            "exit_timestamp": pd.to_datetime(["2026-06-01T00:05:00Z", "2026-06-01T00:15:00Z"]),
            "horizon_minutes": [60, 60],
            "predicted_net_ev_bps": [2.0, 1.0],
        }
    )
    selected = alpha._one_position(rows, threshold=0.0)
    assert selected.index.tolist() == [0, 1]


def test_causal_coverage_filter_adapts_without_test_outcomes() -> None:
    history = pd.DataFrame({"predicted_net_ev_bps": np.arange(100, dtype=float)})
    test = pd.DataFrame(
        {
            "available_at": pd.date_range("2026-06-01", periods=100, freq="min", tz="UTC"),
            "predicted_net_ev_bps": np.full(100, 200.0),
        }
    )
    accepted = alpha._causal_coverage_filter(history, test, coverage=0.01)
    assert 1 <= len(accepted) <= 3
    assert "causal_threshold_bps" in accepted


def test_protocol_keeps_final_holdout_closed() -> None:
    assert alpha.PROTOCOL["sealed_holdout_start"] == "2026-07-01T00:00:00+00:00"
    assert alpha.PROTOCOL["real_capital_allowed"] is False
    assert alpha.TRAINING_MONTHS[-1] == "2026-06"
    assert "2026-07" not in alpha.TRAINING_MONTHS


def test_one_second_paths_cover_the_declared_horizon() -> None:
    assert alpha.PATH_BUCKETS_PER_MINUTE == 60
    assert max(alpha.HORIZONS_MINUTES) * alpha.PATH_BUCKETS_PER_MINUTE == 3_600


def test_vip_plan_gate_requires_three_times_round_trip_cost() -> None:
    vip0_cost, _ = alpha._costs(0)
    vip3_cost, _ = alpha._costs(3)
    assert alpha.MINIMUM_TARGET_TO_COST * vip0_cost > 30
    assert alpha.MINIMUM_TARGET_TO_COST * vip3_cost <= 30


def test_action_context_is_side_relative_and_family_is_not_ordinal() -> None:
    record = {feature: 2.0 for feature in alpha.DIRECTIONAL_MARKET_FEATURES}
    alpha._encode_action_context(
        record,
        side=-1,
        event_direction=1,
        event_family="DAILY_FADE",
    )
    assert record["event_direction_alignment"] == -1.0
    assert all(record[feature] == -2.0 for feature in alpha.DIRECTIONAL_MARKET_FEATURES)
    assert record["event_family_daily_fade"] == 1.0
    assert sum(record[feature] for feature in alpha.EVENT_FAMILY_FEATURES) == 1.0
    assert "event_family_code" not in alpha.FEATURES


def test_probability_mapping_handles_absent_chronological_class() -> None:
    class TwoClassModel:
        classes_ = np.array([0, 2])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.25, 0.75]]), (len(values), 1))

    result = alpha._three_class_probabilities(TwoClassModel(), np.zeros((2, 1)))
    np.testing.assert_allclose(result, [[0.25, 0.0, 0.75], [0.25, 0.0, 0.75]])


def test_rolling_fit_uses_only_the_preceding_twelve_weeks() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=130, freq="D", tz="UTC"))
    rows = pd.DataFrame({"value": range(len(timestamps))})
    fit_end = pd.Timestamp("2026-05-01T00:00:00Z")
    selected = alpha._rolling_fit(rows, timestamps, fit_end)
    selected_times = timestamps.loc[selected.index]
    assert selected_times.min() >= fit_end - pd.Timedelta(days=84)
    assert selected_times.max() < fit_end


def test_dashboard_compacts_all_six_vip_profiles(tmp_path: Path) -> None:
    report = tmp_path / "economic.json"
    profile = {
        "audit_metrics": {
            "trades": 36,
            "expectancy_bps": 2.5,
            "expectancy_bootstrap_lcb_95_bps": 1.0,
            "profit_factor": 1.2,
            "max_account_drawdown": 0.04,
        },
        "audit_cost_stress_2x_metrics": {"expectancy_bps": -4.0},
    }
    gates = {
        "expectancy_positive": True,
        "expectancy_lcb_positive": True,
        "profit_factor_1_15": True,
        "max_drawdown_8pct": True,
        "majority_positive_weeks": True,
        "cost_stress_2x_nonnegative": False,
        "gross_winner_at_least_3x_cost": True,
    }
    report.write_text(
        json.dumps(
            {
                "status": "NO_DEPLOYABLE_POLICY_SHADOW_CHALLENGER_ONLY",
                "champion": "ridge",
                "shadow_challenger": {
                    "status": "DISCOVERY_ONLY_SHADOW_CHALLENGER",
                    "model": "xgboost",
                    "base_viable_profiles": ["VIP3", "VIP4", "VIP5"],
                    "vip_audit": {f"VIP{level}": profile for level in range(6)},
                    "profile_financial_gates": {f"VIP{level}": gates for level in range(6)},
                },
            }
        ),
        encoding="utf-8",
    )
    payload = build_musca_v5_economic_payload(report)
    assert len(payload["profiles"]) == 6
    assert payload["profiles"][3]["base_financial_gates_passed"] is True
    assert payload["profiles"][3]["stress_gate_passed"] is False
    assert payload["profiles"][3]["trade_count_gate_passed"] is False


def test_dashboard_reads_active_local_alpha_report(tmp_path: Path) -> None:
    report = tmp_path / "btc_vwap_alpha_v1.json"
    report.write_text(
        json.dumps(
            {
                "status": "NO_ECONOMIC_ALPHA",
                "protocol_hash": "abc",
                "holdout_opened": False,
                "paper_eligible_profiles": [],
                "champions": {"VWAP_REVERSION:LONG": "ridge"},
                "vip_policy_audit": {
                    "VIP5": {
                        "trades": 288,
                        "expectancy_bps": -5.0,
                        "profit_factor": 0.68,
                        "max_drawdown": 0.39,
                        "bootstrap_expectancy_lcb_95_bps": -7.8,
                        "stress_2x": {"expectancy_bps": -13.0},
                    }
                },
                "vip_gates": {
                    "VIP5": {
                        "walk_forward_trades_300": False,
                        "expectancy_positive": False,
                        "bootstrap_lcb_positive": False,
                        "profit_factor_1_15": False,
                        "max_drawdown_8pct": False,
                        "majority_windows_positive": False,
                        "stress_2x_nonnegative": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_musca_v5_economic_payload(report)

    assert payload["status"] == "NO_ECONOMIC_ALPHA"
    assert payload["protocol_hash"] == "abc"
    assert payload["profiles"][5]["trades"] == 288
    assert payload["profiles"][5]["expectancy_bps"] == -5.0
    assert not payload["profiles"][5]["trade_count_gate_passed"]


def test_dashboard_exposes_only_current_binance_and_archived_v2_profiles() -> None:
    html = Path("src/adaptive_bot/dashboard/static/index.html").read_text(encoding="utf-8")
    script = Path("src/adaptive_bot/dashboard/static/app.js").read_text(encoding="utf-8")
    assert 'id="v5-vip-profile-body"' in html
    assert "function renderEconomicAlpha" in script
    assert 'value="musca-v5-binance"' in html
    assert 'value="musca-v5-vip0"' not in html
    assert 'const allowedProfiles = ["musca-v5-binance", "musca-v2"]' in script
    assert "function selectedV5Paper" in script
    assert "paper_accounts?.[selectedV5Profile(audit)]" in script
