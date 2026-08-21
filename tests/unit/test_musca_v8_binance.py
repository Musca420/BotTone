from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot import musca_v8_binance as binance
from adaptive_bot.dashboard.server import build_binance_alpha_payload


def _config() -> binance.BinancePaperConfig:
    return binance.BinancePaperConfig(
        symbol="BTCUSDT",
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
        non_fee_reserve_bps=1.0,
        fee_source="official_example_fallback",
        max_book_age_seconds=5.0,
    )


def _context(at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "available_at": at - pd.Timedelta(seconds=1),
                "feature_contract_valid": True,
                "derivatives_feature_valid": True,
                "close": 60_000.0,
                "rolling_vwap": 59_950.0,
                "taker_imbalance_60s": 0.2,
                "return_1m_bps": 2.0,
                "return_5m_bps": 8.0,
                "return_15m_bps": 10.0,
                "return_30m_bps": 15.0,
                "vwap_distance_bps": 8.3,
                "vwap_slope_bps": 1.1,
                "range_60s_bps": 5.0,
                "oi_change_1h": 0.01,
                "return_oi_interaction_raw": 0.08,
                "basis_bps": 2.0,
                "funding_z": 0.5,
            }
        ]
    )


def _book(at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "available_at": at - pd.Timedelta(milliseconds=100),
                "mid": 60_000.0,
                "best_bid": 59_999.0,
                "best_ask": 60_001.0,
                "bids": [[59_999.0, 2.0]],
                "asks": [[60_001.0, 2.0]],
                "spread_bps": 1 / 60_000 * 20_000,
                "feature_valid": True,
            }
        ]
    )


def _funding(at: pd.Timestamp) -> dict[str, object]:
    return {
        "observed_at": (at - pd.Timedelta(seconds=1)).isoformat(),
        "mark_price": 60_000.0,
        "index_price": 59_999.5,
        "funding_rate": 0.0001,
        "next_funding_timestamp": (at + pd.Timedelta(hours=8)).isoformat(),
    }


def _candidate(at: pd.Timestamp) -> dict[str, object]:
    return {
        "setup": "IMPULSE_PULLBACK_H24",
        "direction": "LONG",
        "candidate": True,
        "setup_active": True,
        "passed_checks": 7,
        "total_checks": 7,
        "first_failed_check": None,
        "checks": [],
        "policy_source": "MUSCA_V8_FROZEN_BASE",
        "expert_id": "musca-v8-h24-test",
        "available_at": (at - pd.Timedelta(seconds=1)).isoformat(),
        "impulse_anchor_at": (at - pd.Timedelta(hours=1)).isoformat(),
        "operating_vwap": 59_980.0,
        "stop_price": 59_400.0,
        "robust_expected_gross_bps": 20.0,
        "target_probability": 0.6,
        "maximum_hold_minutes": 360,
    }


def test_config_is_paper_only(tmp_path: Path) -> None:
    invalid = tmp_path / "live.yaml"
    invalid.write_text(
        "mode: live\nlive_trading_enabled: true\nexecution: {}\nrisk: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paper"):
        binance.load_config(invalid)


def test_fee_schedule_is_explicit_fallback_or_signed_account_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.setattr(binance, "_FEE_CACHE", None)
    fallback = binance.fee_schedule(_config())
    assert fallback.taker_bps == 4.0
    assert fallback.source.startswith("CONFIG_FALLBACK")

    monkeypatch.setenv("BINANCE_API_KEY", "paper-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "paper-secret")
    monkeypatch.setattr(binance, "_FEE_CACHE", None)
    observed = binance.fee_schedule(
        _config(),
        fetch=lambda *_args, **_kwargs: {
            "symbol": "BTCUSDT",
            "makerCommissionRate": "0.00015",
            "takerCommissionRate": "0.00035",
        },
    )
    assert observed.maker_bps == pytest.approx(1.5)
    assert observed.taker_bps == pytest.approx(3.5)
    assert observed.source == "BINANCE_SIGNED_COMMISSION_RATE"


def test_public_derivatives_features_recreate_the_historical_contract() -> None:
    observed = pd.Timestamp("2026-08-10T12:01:00Z")
    candle = pd.Timestamp("2026-08-10T12:00:00Z")
    oi = [
        {
            "timestamp": int(
                (observed - pd.Timedelta(minutes=5 * (12 - index))).timestamp() * 1000
            ),
            "sumOpenInterest": str(100 + index),
        }
        for index in range(13)
    ]
    funding = [
        {
            "fundingTime": int(
                (observed - pd.Timedelta(hours=8 * (40 - index))).timestamp() * 1000
            ),
            "fundingRate": str(0.0001 + (index % 5) * 0.00001),
        }
        for index in range(41)
    ]
    close_time = int(
        (candle + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000
    )
    mark = [[int(candle.timestamp() * 1000), "0", "0", "0", "60012", "0", close_time]]
    spot = [[int(candle.timestamp() * 1000), "0", "0", "0", "60000", "0", close_time]]
    payloads = {
        binance.OPEN_INTEREST_HISTORY_URL: oi,
        binance.FUNDING_HISTORY_URL: funding,
        binance.MARK_KLINES_URL: mark,
        binance.SPOT_KLINES_URL: spot,
    }

    features = binance.derivatives_alpha_features(
        candle_timestamp=candle,
        observed_at=observed,
        return_5m_bps=8.0,
        fetch=lambda url: payloads[url],
    )

    assert features["oi_change_1h"] == pytest.approx(0.12)
    assert features["return_oi_interaction_raw"] == pytest.approx(0.96)
    assert features["basis_bps"] == pytest.approx(2.0)
    assert abs(float(features["funding_z"])) < 10
    assert features["derivatives_feature_valid"] is True


def test_public_derivatives_features_fail_closed_without_one_hour_of_oi() -> None:
    observed = pd.Timestamp("2026-08-10T12:01:00Z")
    candle = pd.Timestamp("2026-08-10T12:00:00Z")

    with pytest.raises(ValueError, match="open-interest"):
        binance.derivatives_alpha_features(
            candle_timestamp=candle,
            observed_at=observed,
            return_5m_bps=8.0,
            fetch=lambda url: [] if url == binance.OPEN_INTEREST_HISTORY_URL else {},
        )


def test_assessment_trades_only_when_binance_alpha_is_net_positive() -> None:
    now = pd.Timestamp("2026-08-09T12:00:00Z")
    assessment = binance.build_assessment(
        candidate=_candidate(now),
        context=_context(now),
        book=_book(now),
        funding=_funding(now),
        fees=binance.FeeSchedule(2.0, 4.0, "TEST"),
        config=_config(),
        equity=10_000,
        evaluated_at=now,
    )

    assert assessment["decision"] == "TRADE"
    assert assessment["expected_net_ev_bps"] > 0
    assert assessment["risk_approved"] is True
    assert assessment["market_inputs"]["binance"]["book_synced"] is True
    assert assessment["target_price"] > assessment["entry_execution_vwap"]
    assert assessment["stop_price"] < assessment["entry_execution_vwap"]

    expensive = binance.build_assessment(
        candidate=_candidate(now),
        context=_context(now),
        book=_book(now),
        funding=_funding(now),
        fees=binance.FeeSchedule(20.0, 20.0, "TEST"),
        config=_config(),
        equity=10_000,
        evaluated_at=now,
    )
    assert expensive["decision"] == "FLAT"
    assert expensive["reason"] == "BASE_ALPHA_COST_BLOCKED"


def test_missing_binance_book_fails_closed() -> None:
    now = pd.Timestamp("2026-08-09T12:00:00Z")
    assessment = binance.build_assessment(
        candidate=_candidate(now),
        context=_context(now),
        book=pd.DataFrame(),
        funding=_funding(now),
        fees=binance.FeeSchedule(2.0, 4.0, "TEST"),
        config=_config(),
        equity=10_000,
        evaluated_at=now,
    )
    assert assessment["decision"] == "WAIT"
    assert assessment["reason"] == "BINANCE_DATA_FAIL_CLOSED"


def test_fresh_binance_data_without_positive_expert_is_neutral_flat() -> None:
    now = pd.Timestamp("2026-08-09T12:00:00Z")
    evaluation = {
        "status": "READY_FLAT",
        "reason": "NO_POSITIVE_CALIBRATED_EV",
        "feature_coverage": {"complete": True, "missing": []},
        "frozen_expert_count": 45,
        "active_expert_count": 3,
        "horizons_minutes": [60, 360],
        "best_action": {
            "expert_id": "best-negative",
            "direction": "SHORT",
            "horizon_minutes": 60,
            "calibrated_ev_bps": -0.75,
            "probability_net_positive": 0.47,
        },
        "alternatives": [],
    }

    assessment = binance.build_assessment(
        candidate=None,
        context=_context(now),
        book=_book(now),
        funding=_funding(now),
        fees=binance.FeeSchedule(2.0, 4.0, "TEST"),
        config=_config(),
        equity=10_000,
        evaluated_at=now,
        model_evaluation=evaluation,
    )

    assert assessment["decision"] == "FLAT"
    assert assessment["reason"] == "FLAT_NO_POSITIVE_AUTO_MOE_ACTION"
    assert assessment["expected_net_ev_bps"] == pytest.approx(-0.75)
    assert assessment["direction"] == "SHORT"
    assert assessment["model_evaluation"]["active_expert_count"] == 3
    assert assessment["outcome_horizons_minutes"] == [60, 360]


def test_missing_model_inputs_waits_even_when_binance_sources_are_fresh() -> None:
    now = pd.Timestamp("2026-08-09T12:00:00Z")
    assessment = binance.build_assessment(
        candidate=None,
        context=_context(now),
        book=_book(now),
        funding=_funding(now),
        fees=binance.FeeSchedule(2.0, 4.0, "TEST"),
        config=_config(),
        equity=10_000,
        evaluated_at=now,
        model_evaluation={
            "status": "DATA_UNAVAILABLE",
            "reason": "INSUFFICIENT_CAUSAL_L2_HISTORY",
            "feature_coverage": {"complete": False, "missing": []},
            "horizons_minutes": [60, 360],
        },
    )

    assert assessment["decision"] == "WAIT"
    assert assessment["reason"] == "MODEL_INPUT_FAIL_CLOSED"
    assert assessment["probability_status"] == "AUTO_MOE_INPUT_FAIL_CLOSED"


def test_binance_chart_uses_closed_official_ohlcv_without_fake_execution_line() -> None:
    context = pd.DataFrame(
        [
            {
                "available_at": pd.Timestamp("2026-08-10T12:01:00Z"),
                "open": 60_000.0,
                "high": 60_020.0,
                "low": 59_990.0,
                "close": 60_010.0,
                "volume": 12.5,
                "rolling_vwap": 60_005.0,
            }
        ]
    )

    point = binance._chart(context, {"anchored_vwap": 60_001.0})[0]

    assert point["open"] == 60_000.0
    assert point["high"] == 60_020.0
    assert point["low"] == 59_990.0
    assert point["volume"] == 12.5
    assert point["execution_price"] is None
    assert point["anchored_vwap"] is None


def test_latest_book_ignores_records_received_after_the_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-08-10T12:00:00Z")
    rows = pd.DataFrame(
        [
            {
                "exchange_second": int((now - pd.Timedelta(seconds=2)).timestamp()),
                "available_at": now - pd.Timedelta(seconds=1),
                "bids": [["59999.9", "1"]],
                "asks": [["60000.1", "1"]],
                "source": "binance-official-usdm-websocket-routed",
            },
            {
                "exchange_second": int(now.timestamp()),
                "available_at": now + pd.Timedelta(seconds=1),
                "bids": [["60999.9", "1"]],
                "asks": [["61000.1", "1"]],
                "source": "binance-official-usdm-websocket-routed",
            },
        ]
    )
    monkeypatch.setattr(binance.binance_l2_dataset, "load_recent_records", lambda **_: rows)

    selected = binance.latest_book(now, _config())

    assert float(selected.iloc[-1]["mid"]) == pytest.approx(60_000)
    assert bool(selected.iloc[-1]["feature_valid"])


def test_dashboard_alpha_payload_reports_binance_oos_without_opening_holdout(
    tmp_path: Path,
) -> None:
    report = tmp_path / "binance.json"
    report.write_text(
        """
        {
          "alpha": {
            "status": "RESEARCH_BASE_ALPHA_READY",
            "protocol_hash": "abc",
            "paper_profiles": {
              "BINANCE": {
                "paper_eligible": true,
                "oos_2026": {
                  "trades": 24,
                  "expectancy_bps": 13.5,
                  "profit_factor": 1.39,
                  "max_drawdown": 0.032
                },
                "oos_2026_stress_2x": {"expectancy_bps": 4.5}
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    payload = build_binance_alpha_payload(report)

    assert payload["status"] == "RESEARCH_BASE_ALPHA_READY"
    assert payload["profiles"][0]["profile"] == "BINANCE"
    assert payload["profiles"][0]["expectancy_bps"] == 13.5
    assert payload["holdout_opened"] is False


def test_dashboard_alpha_payload_supports_auto_moe_audit(tmp_path: Path) -> None:
    report = tmp_path / "auto-moe.json"
    report.write_text(
        """
        {
          "verdict": "RESEARCH_PAPER",
          "protocol_hash": "auto",
          "gating_champion": "ridge",
          "future_holdout_rows_read": 0,
          "historical_audit": {
            "metrics": {
              "trades": 72,
              "expectancy_bps": 19.5,
              "profit_factor": 1.55,
              "max_drawdown": 0.024,
              "bootstrap_lcb_95_bps": 11.4,
              "stress_2x_expectancy_bps": 10.5
            },
            "paper_gates": {"minimum_research_trades_50": true, "expectancy": true},
            "live_gates": {
              "minimum_historical_trades_300": false,
              "stress_1_5x": true
            }
          }
        }
        """,
        encoding="utf-8",
    )

    payload = build_binance_alpha_payload(report)

    assert payload["status"] == "RESEARCH_PAPER"
    assert payload["champion"] == "ridge adaptive expert gate"
    assert payload["profiles"][0]["trades"] == 72
    assert payload["profiles"][0]["base_financial_gates_passed"] is True
    assert payload["profiles"][0]["trade_count_gate_passed"] is False
    assert payload["holdout_opened"] is False
