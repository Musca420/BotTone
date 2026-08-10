import json
from pathlib import Path

import pandas as pd

from adaptive_bot.btc_cross_exchange_dataset import build_features, load_snapshots


def test_bitunix_current_funding_percent_is_normalized_for_features(tmp_path: Path) -> None:
    payload = {
        "received_at": "2026-08-08T08:00:01+00:00",
        "records": [
            {
                "exchange": "bitunix",
                "coverage": True,
                "mark_price": "65000",
                "funding_rate": "-0.006488",
            }
        ],
    }
    (tmp_path / "btc_context_test.jsonl").write_text(json.dumps(payload) + "\n")
    row = load_snapshots(tmp_path).iloc[0]
    assert row["funding"] == -0.00006488


def test_cross_exchange_features_are_causal() -> None:
    minutes = pd.date_range("2026-08-06", periods=40, freq="min", tz="UTC")
    rows = pd.DataFrame(
        [
            {
                "minute": minute,
                "available_at": minute + pd.Timedelta(seconds=10),
                "exchange": exchange,
                "price": 100 + index,
                "basis_bps": 1,
                "funding": 0.0001,
                "open_interest": 1000 + index,
            }
            for index, minute in enumerate(minutes)
            for exchange in ("binance", "bybit", "okx", "bitunix")
        ]
    )
    l2 = pd.DataFrame(
        {
            "available_at": minutes + pd.Timedelta(seconds=5),
            "feature_valid": True,
            "rolling_vwap_5m_distance_bps": 2.0,
            "aggressive_imbalance_60s": 0.2,
            "depth_imbalance_5": 0.1,
            "microprice_distance_bps": 0.05,
        }
    )
    original = build_features(rows, l2)
    changed = rows.copy()
    changed.loc[
        changed["minute"].eq(minutes[15]) & changed["exchange"].eq("binance"), "price"
    ] = 999
    mutated = build_features(changed, l2)
    assert original.loc[10, "median_return_5m_bps"] == mutated.loc[10, "median_return_5m_bps"]
    assert original.loc[10, "future_return_5m_bps"] != mutated.loc[10, "future_return_5m_bps"]
    assert original.loc[10, "label_available_at_5m"] > original.loc[10, "available_at"]


def test_binance_alpha_does_not_wait_for_or_require_other_venues() -> None:
    minutes = pd.date_range("2026-08-08", periods=10, freq="min", tz="UTC")
    delays = {"binance": 3, "bybit": 50, "okx": 55, "bitunix": 40}
    rows = pd.DataFrame(
        [
            {
                "minute": minute,
                "available_at": minute + pd.Timedelta(seconds=delays[exchange]),
                "exchange": exchange,
                "price": 100 + index if exchange == "binance" else None,
                "basis_bps": 1 if exchange == "binance" else None,
                "funding": 0.0001 if exchange == "binance" else None,
                "open_interest": 1_000 + index if exchange == "binance" else None,
            }
            for index, minute in enumerate(minutes)
            for exchange in ("binance", "bybit", "okx", "bitunix")
        ]
    )
    l2 = pd.DataFrame(
        {
            "available_at": minutes + pd.Timedelta(seconds=2),
            "feature_valid": True,
            "rolling_vwap_5m_distance_bps": 2.0,
        }
    )

    result = build_features(rows, l2)

    assert result.loc[6, "available_at"] == minutes[6] + pd.Timedelta(seconds=3)
    assert bool(result.loc[6, "feature_valid"])
    assert pd.isna(result.loc[6, "return_5m_bybit_bps"])
    assert pd.isna(result.loc[6, "return_5m_okx_bps"])


def test_missing_binance_minute_fails_closed_without_breaking_l2_join() -> None:
    minutes = pd.date_range("2026-08-08", periods=10, freq="min", tz="UTC")
    rows = pd.DataFrame(
        [
            {
                "minute": minute,
                "available_at": minute + pd.Timedelta(seconds=3),
                "exchange": "binance",
                "price": 100 + index,
                "basis_bps": 1,
                "funding": 0.0001,
                "open_interest": 1_000 + index,
            }
            for index, minute in enumerate(minutes)
            if index != 5
        ]
    )
    l2 = pd.DataFrame(
        {
            "available_at": minutes + pd.Timedelta(seconds=2),
            "feature_valid": True,
            "rolling_vwap_5m_distance_bps": 2.0,
        }
    )

    result = build_features(rows, l2)

    missing = result.loc[result["minute"].eq(minutes[5])].iloc[0]
    assert pd.isna(missing["available_at"])
    assert pd.isna(missing["alpha_l2_available_at"])
    assert not bool(missing["feature_valid"])


def test_live_alpha_clock_uses_latest_context_known_before_bar_completion() -> None:
    minutes = pd.date_range("2026-08-09", periods=10, freq="min", tz="UTC")
    rows = pd.DataFrame(
        [
            {
                "minute": minute,
                "available_at": minute + pd.Timedelta(seconds=59),
                "exchange": "binance",
                "price": 100 + index,
                "basis_bps": 1,
                "funding": 0.0001,
                "open_interest": 1_000 + index,
            }
            for index, minute in enumerate(minutes)
        ]
    )
    l2 = pd.DataFrame(
        {
            "available_at": minutes + pd.Timedelta(minutes=1),
            "feature_valid": True,
            "rolling_vwap_5m_distance_bps": 2.0,
        }
    )

    result = build_features(rows, l2, alpha_clock=True)
    latest = result.iloc[-1]

    assert latest["available_at"] == minutes[-1] + pd.Timedelta(minutes=1)
    assert latest["context_available_at"] == minutes[-1] + pd.Timedelta(seconds=59)
    assert latest["price_binance"] == 109
    assert bool(latest["feature_valid"])
