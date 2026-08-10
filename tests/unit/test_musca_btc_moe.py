from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_btc_moe as moe


def _source(
    opens: list[float], highs: list[float], lows: list[float], closes: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_protocol_contains_125_return_experts_and_only_btc() -> None:
    assert moe.SYMBOL == "BTCUSDT"
    assert moe.PROTOCOL["return_experts"]["final_components"] == 125
    assert moe.PROTOCOL["quantile_tools"]["components"] == 30
    assert len(moe.EXPERT_COLUMNS) == 25
    assert moe.PROTOCOL["horizons_seconds"] == [60, 300, 900, 3_600, 21_600]
    assert "ETHUSDT" not in str(moe.PROTOCOL)


def test_gating_must_listen_to_experts_instead_of_relearning_alpha() -> None:
    assert set(moe.EXPERT_COLUMNS).issubset(moe.META_FEATURES)
    assert "volatility_percentile" in moe.META_FEATURES
    assert "return_1m_bps" not in moe.META_FEATURES


def test_future_extrema_start_after_decision_bucket() -> None:
    values = np.array([1.0, 4.0, 2.0, 5.0])
    maximum = moe._forward_extreme(values, 2, "max")
    minimum = moe._forward_extreme(values, 2, "min")
    assert np.allclose(maximum[:2], [4.0, 5.0])
    assert np.allclose(minimum[:2], [2.0, 2.0])
    assert np.isnan(maximum[2:]).all()


def test_decision_cadence_does_not_depend_on_parquet_timestamp_resolution() -> None:
    for unit in ("ms", "us", "ns"):
        values = pd.Series(
            pd.date_range("2026-01-01", periods=24, freq="5s", tz="UTC").as_unit(unit)
        )
        mask = moe._decision_time_mask(values)
        assert mask.sum() == 2
        assert mask.iloc[0]
        assert mask.iloc[12]


def test_micro_features_have_explicit_availability_and_no_zero_fill() -> None:
    assert "available_at" not in moe.FEATURES
    assert set(moe.DIRECTIONAL_MICRO_FEATURES).issubset(moe.DIRECTIONAL_FEATURES)
    assert moe.PROTOCOL["entry"].startswith("next 5-second bucket")


def test_stop_and_target_same_5s_bucket_uses_stop() -> None:
    source = _source([100, 100], [100, 100.2], [100, 99.9], [100, 100.1])
    gross, minutes, outcome = moe._simulate_management(
        source,
        np.array([0]),
        1,
        5,
        np.array([10.0]),
        np.array([20.0]),
        np.array([10.0]),
        np.array([10.0]),
    )
    assert gross[0] == -10.0
    assert minutes[0] == 5
    assert outcome[0] == "STOP"


def test_partial_target_then_second_target() -> None:
    source = _source(
        [100, 100, 100.1],
        [100, 100.15, 100.4],
        [100, 99.95, 100.05],
        [100, 100.1, 100.35],
    )
    gross, minutes, outcome = moe._simulate_management(
        source,
        np.array([0]),
        1,
        10,
        np.array([10.0]),
        np.array([30.0]),
        np.array([20.0]),
        np.array([20.0]),
    )
    assert np.isclose(gross[0], 20.0)
    assert minutes[0] == 10
    assert outcome[0] == "TARGET_2"


def test_trailing_stop_never_widens() -> None:
    current = np.array([-20.0, 0.0, 15.0])
    candidate = np.array([-30.0, 10.0, 5.0])
    tightened = moe._tighten_stop(current, candidate)
    assert np.array_equal(tightened, np.array([-20.0, 10.0, 15.0]))
    assert np.all(tightened >= current)


def test_gate_accepts_normal_losing_trades_when_aggregate_is_positive() -> None:
    value = {
        "trades": 400,
        "trades_per_day": 4.0,
        "expectancy_bps": 2.0,
        "profit_factor": 1.2,
        "positive_calendar_days": 0.6,
        "max_drawdown": 0.08,
        "bootstrap_lcb_95_bps": 0.5,
        "spa_pvalue": 0.04,
        "win_rate": 0.45,
        "risk_budget_violations": 0,
    }
    assert value["win_rate"] < 0.5
    assert all(moe._audit_gates(value).values())


def test_final_holdout_starts_after_historical_audit() -> None:
    assert moe.HISTORICAL_AUDIT_END < moe.FUTURE_HOLDOUT_START
