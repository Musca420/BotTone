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
