from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_altcoin_micro as policy


class _RawRegressor:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0]


class _FlatCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values))


class _RawClassifier:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack((np.full(len(values), 0.5), np.full(len(values), 0.5)))


class _ProbabilityCalibrator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack((np.full(len(values), 0.5), np.full(len(values), 0.5)))


def test_protocol_trades_only_altcoins_and_keeps_btc_as_context() -> None:
    assert set(policy.SYMBOLS) == {"ETHUSDT", "XRPUSDT", "DOGEUSDT"}
    assert "BTCUSDT" not in policy.SYMBOLS
    assert policy.PROTOCOL["context_symbol"] == "BTCUSDT"
    assert policy.MONTHS[-1] == "2026-06"
    assert pd.Timestamp("2026-07-01T00:00:00Z") == policy.HOLDOUT_START
    assert policy.PROTOCOL["event_cooldown_minutes"] == 3


def test_dynamic_levels_cover_cost_and_remain_bounded() -> None:
    plan = policy.PLANS[0]
    target, stop = policy.plan_levels(np.array([1.0, 10.0, 100.0]), plan, 11.5)

    assert (target >= 1.5 * 11.5).all()
    assert (stop >= plan.minimum_stop_bps).all()
    assert (stop <= plan.maximum_stop_bps).all()


def test_same_minute_stop_wins() -> None:
    raw = pd.DataFrame(
        {
            "open": np.full(70, 100.0),
            "high": np.r_[100.0, 101.0, np.full(68, 100.0)],
            "low": np.r_[100.0, 99.0, np.full(68, 100.0)],
            "close": np.full(70, 100.0),
        }
    )
    plan = policy.Plan("TEST", 5, 1.0, 10.0, 10.0, 1.0)

    outcome, gross, _, _, _ = policy.barrier_outcomes(
        raw, np.full(70, 10.0), plan, side=1, cost_bps=1.0
    )

    assert outcome[0] == 1
    assert gross[0] == -10.0


def test_execute_accepts_losses_but_never_overlaps_positions() -> None:
    rows = pd.DataFrame(
        [
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:00:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:05:00Z"),
                "score": 2.0,
                "net_bps": -5.0,
            },
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:01:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:02:00Z"),
                "score": 3.0,
                "net_bps": 20.0,
            },
        ]
    )

    trades = policy.execute(rows, threshold=0.0)

    assert trades["net_bps"].tolist() == [-5.0]


def test_frozen_negative_rank_threshold_can_be_evaluated_as_a_policy() -> None:
    rows = pd.DataFrame(
        [
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:00:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:01:00Z"),
                "score": -1.0,
            },
            {
                "entry_timestamp": pd.Timestamp("2026-04-01T00:02:00Z"),
                "exit_timestamp": pd.Timestamp("2026-04-01T00:03:00Z"),
                "score": -3.0,
            },
        ]
    )

    trades = policy.execute(rows, threshold=-2.0)

    assert trades["score"].tolist() == [-1.0]


def test_isotonic_plateau_does_not_destroy_raw_ranking() -> None:
    timestamp = pd.date_range("2026-04-01", periods=2, freq="1min", tz="UTC")
    rows = pd.DataFrame({feature: np.zeros(2) for feature in policy.FEATURES})
    rows[policy.FEATURES[0]] = [1.0, 2.0]
    rows["available_at"] = timestamp
    rows["entry_timestamp"] = timestamp
    rows["day"] = timestamp.floor("D")
    for _, _, stem in policy._action_stems():
        rows[f"{stem}_target_bps"] = 20.0
        rows[f"{stem}_stop_bps"] = 10.0
        rows[f"{stem}_gross_bps"] = 20.0
        rows[f"{stem}_exit_minutes"] = 1
    model = {
        "regressor": _RawRegressor(),
        "ev_calibrator": _FlatCalibrator(),
        "classifier": _RawClassifier(),
        "probability_calibrator": _ProbabilityCalibrator(),
    }

    scored = policy.score(rows, model, cost_bps=9.0)

    assert scored["score"].tolist() == [-8.0, -7.0]
    assert scored["expected_gross_bps"].tolist() == [0.0, 0.0]
