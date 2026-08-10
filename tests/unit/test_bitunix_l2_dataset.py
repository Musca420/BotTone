import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from adaptive_bot.bitunix_l2_dataset import build_features, latest_execution_snapshot


def test_latest_execution_snapshot_uses_observed_book_and_trade(tmp_path: Path) -> None:
    start = datetime(2026, 8, 8, tzinfo=UTC)
    lines = []
    for second in range(3):
        timestamp = start + timedelta(seconds=second)
        lines.append(
            {
                "normalized": [
                    {
                        "event_type": "book",
                        "exchange_timestamp": timestamp.isoformat(),
                        "received_timestamp": (timestamp + timedelta(milliseconds=100)).isoformat(),
                        "bids": [["99.99", "2"]],
                        "asks": [["100.01", "2"]],
                    },
                    {
                        "event_type": "trade",
                        "exchange_timestamp": timestamp.isoformat(),
                        "received_timestamp": (timestamp + timedelta(milliseconds=100)).isoformat(),
                        "price": "100",
                        "quantity": "1",
                        "aggressor_side": "buy",
                    },
                ]
            }
        )
    path = tmp_path / "btcusdt_2026-08-08.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    result = latest_execution_snapshot(tmp_path)
    assert len(result) == 1
    assert result.loc[0, "feature_valid"]
    assert result.loc[0, "best_bid"] == 99.99
    assert result.loc[0, "best_ask"] == 100.01
    assert result.loc[0, "clock_drift_ms"] == 100


def test_latest_execution_snapshot_spans_utc_midnight(tmp_path: Path) -> None:
    before = datetime(2026, 8, 8, 23, 59, 59, tzinfo=UTC)
    after = before + timedelta(seconds=1)
    observations = []
    for timestamp in (before, after):
        observations.append(
            {
                "normalized": [
                    {
                        "event_type": "book",
                        "exchange_timestamp": timestamp.isoformat(),
                        "received_timestamp": (timestamp + timedelta(milliseconds=100)).isoformat(),
                        "bids": [["99.99", "2"]],
                        "asks": [["100.01", "2"]],
                    },
                    {
                        "event_type": "trade",
                        "exchange_timestamp": timestamp.isoformat(),
                        "received_timestamp": (timestamp + timedelta(milliseconds=100)).isoformat(),
                        "price": "100",
                        "quantity": "1",
                        "aggressor_side": "buy",
                    },
                ]
            }
        )
    (tmp_path / "btcusdt_2026-08-08.jsonl").write_text(
        json.dumps(observations[0]) + "\n", encoding="utf-8"
    )
    (tmp_path / "btcusdt_2026-08-09.jsonl").write_text(
        json.dumps(observations[1]) + "\n", encoding="utf-8"
    )

    result = latest_execution_snapshot(tmp_path)

    assert result.loc[0, "feature_valid"]
    assert result.loc[0, "orderbook_feed_alive"]


def test_bitunix_features_are_available_only_after_observation() -> None:
    start = datetime(2026, 8, 8, tzinfo=UTC)
    rows = []
    for second in range(400):
        exchange = start + timedelta(seconds=second)
        received = exchange + timedelta(milliseconds=100)
        rows.extend(
            [
                {
                    "event_type": "book",
                    "exchange_timestamp": exchange.isoformat(),
                    "received_timestamp": received.isoformat(),
                    "best_bid": "99.96",
                    "best_ask": "100.04",
                    "midpoint": "100",
                    "bids_json": '[["99.96","100"]]',
                    "asks_json": '[["100.04","100"]]',
                },
                {
                    "event_type": "trade",
                    "exchange_timestamp": exchange.isoformat(),
                    "received_timestamp": received.isoformat(),
                    "price": "100",
                    "quantity": "1",
                    "aggressor_side": "buy" if second % 2 else "sell",
                },
            ]
        )
    result = build_features(pd.DataFrame(rows))
    valid = result.loc[result["feature_valid"]]
    assert not valid.empty
    assert valid["available_at"].ge(valid["exchange_timestamp"] - pd.Timedelta(seconds=1)).all()
    assert valid["book_is_synced"].all()
    assert valid["last_trade_update_age_ms"].le(5_000).all()
    assert result["exchange_second"].diff().dropna().eq(1).all()
