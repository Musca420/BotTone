from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import adaptive_bot.btc_vwap_forward_selector as selector


def _rows(count: int = 3) -> pd.DataFrame:
    signal = pd.date_range(selector.PROTOCOL_START, periods=count, freq="min")
    frame = pd.DataFrame(
        {
            "protocol_hash": selector.EXPECTED_OUTCOME_PROTOCOL_HASH,
            "signal_at": signal,
            "feature_available_at": signal,
            "entry_at": signal + pd.Timedelta(seconds=1),
            "exit_at": signal + pd.Timedelta(seconds=30),
            "expert": "pullback_continuation_dynamic",
            "side": "LONG",
            "gross_bps": 12.0,
            "net_bps": 3.0,
            "stress_bps": 4.0,
            "stop_bps": 20.0,
            "fee_bps": 12.0,
            "slippage_reserve_bps": 1.0,
        }
    )
    for feature in selector.FEATURES:
        frame[feature] = 1.0
    for barrier in selector.BARRIERS:
        frame[f"target_{barrier}bps_before_stop"] = True
    for column in selector.HORIZON_PROBABILITY_COLUMNS:
        frame[column] = True
    for column in selector.OUTCOME_COLUMNS:
        frame[column] = 1.0
    return frame


def test_matrix_fails_closed_on_preprotocol_future_or_missing_features(tmp_path: Path) -> None:
    rows = _rows()
    rows.loc[0, "signal_at"] = selector.PROTOCOL_START - pd.Timedelta(seconds=1)
    rows.loc[1, "feature_available_at"] = rows.loc[1, "signal_at"] + pd.Timedelta(seconds=1)
    rows.loc[2, selector.FEATURES[0]] = np.nan
    path = tmp_path / "matrix.parquet"
    rows.to_parquet(path, index=False)
    assert selector.load_matrix(path).empty


def test_evaluate_does_not_start_gpu_without_frozen_rows(
    tmp_path: Path, monkeypatch: object
) -> None:
    path = tmp_path / "matrix.parquet"
    _rows(1).to_parquet(path, index=False)
    monkeypatch.setattr(selector, "BUNDLE", tmp_path / "bundle.joblib")  # type: ignore[attr-defined]
    monkeypatch.setattr(selector, "REPORT", tmp_path / "report.json")  # type: ignore[attr-defined]
    result = selector.evaluate(path)
    assert result["status"] == "INSUFFICIENT_FORWARD_DATA"
    assert not result["gpu_training_started"]
    assert not selector.BUNDLE.exists()


def test_negative_prudent_ev_selects_flat() -> None:
    rows = _rows()
    model = DummyRegressor(strategy="constant", constant=-1.0).fit(
        selector.design(rows), rows["stress_bps"]
    )
    result = selector._holdout_metrics(
        rows,
        {"models": [model], "bias": 0.0, "residual_lower": 0.0},
    )
    assert result["trades"] == 0
    assert result["flat_decisions"] == len(rows)
    assert not result["eligible"]


def test_barrier_probabilities_and_excursions_are_fitted_from_observed_labels() -> None:
    rows = _rows(120)
    rows[selector.FEATURES[0]] = np.linspace(-2, 2, len(rows))
    positive = pd.Series(np.arange(len(rows)) % 2 == 0, index=rows.index)
    for barrier in selector.BARRIERS:
        rows[f"target_{barrier}bps_before_stop"] = positive
    for column in selector.HORIZON_PROBABILITY_COLUMNS:
        rows[column] = positive
    outcomes = selector._fit_outcomes(rows.iloc[:80], rows.iloc[80:])
    bundle = {"outcomes": outcomes}
    predicted = selector._predict_outcomes(bundle, rows.iloc[-2:])
    assert set(outcomes["barriers"]) == set(selector.BARRIERS)
    assert predicted.filter(like="p_target_").notna().all().all()
    assert predicted.filter(like="expected_mfe_").notna().all().all()
