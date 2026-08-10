import json
from pathlib import Path

import pandas as pd

from adaptive_bot.binance_l2_dataset import (
    ANCHOR_FEATURES,
    FEATURES,
    build_features,
    build_live_minute_features,
    load_recent_official_minutes,
    load_recent_records,
)


def _records(count: int = 370) -> pd.DataFrame:
    start = pd.Timestamp("2026-08-06T00:00:00Z")
    return pd.DataFrame(
        [
            {
                "exchange_second": int(start.timestamp()) + index,
                "available_at": (start + pd.Timedelta(seconds=index, milliseconds=200)).isoformat(),
                "bids": [[str(100 + index / 100), "2"]] * 20,
                "asks": [[str(100.1 + index / 100), "1"]] * 20,
                "buy_quote": "100",
                "sell_quote": "50",
                "trade_count": 2,
                "aggregate_trades": [[str(100 + index / 100), "1", "BUY"]],
            }
            for index in range(count)
        ]
    )


def test_recent_loader_reads_only_complete_trailing_observations(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_2026-08-08.jsonl"
    rows = [
        {
            "schema_version": 3,
            "source": "binance-official-usdm-websocket-routed",
            "exchange_second": second,
            "available_at": f"2026-08-08T00:00:0{second}+00:00",
        }
        for second in range(5)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = load_recent_records(tmp_path, max_lines=2)
    assert result["exchange_second"].tolist() == [3, 4]


def test_recent_loader_keeps_warmup_across_utc_midnight(tmp_path: Path) -> None:
    source = "binance-official-usdm-websocket-routed"
    first = tmp_path / "btcusdt_2026-08-08.jsonl"
    second = tmp_path / "btcusdt_2026-08-09.jsonl"
    first.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 3,
                    "source": source,
                    "exchange_second": value,
                    "available_at": f"2026-08-08T23:59:5{value}+00:00",
                }
            )
            for value in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 3,
                    "source": source,
                    "exchange_second": value + 5,
                    "available_at": f"2026-08-09T00:00:0{value}+00:00",
                }
            )
            for value in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_recent_records(tmp_path, max_lines=6)

    assert result["exchange_second"].tolist() == [1, 2, 3, 4, 5, 6]


def test_l2_features_are_causal_and_labels_wait_for_future() -> None:
    records = _records()
    original = build_features(records)
    changed = records.copy()
    changed.at[305, "bids"] = [["999", "2"]] * 20
    changed.at[305, "asks"] = [["1000", "1"]] * 20
    mutated = build_features(changed)
    pd.testing.assert_series_equal(
        original.loc[300, list(FEATURES)], mutated.loc[300, list(FEATURES)]
    )
    assert original.loc[300, "feature_valid"]
    assert not original.loc[0, "feature_valid"]
    assert original.loc[300, "label_available_at_5s"] > original.loc[300, "available_at"]
    assert original.loc[300, "future_return_5s_bps"] != mutated.loc[300, "future_return_5s_bps"]
    pd.testing.assert_series_equal(
        original.loc[300, list(ANCHOR_FEATURES)], mutated.loc[300, list(ANCHOR_FEATURES)]
    )


def test_long_horizon_label_is_unavailable_until_future_exists() -> None:
    frame = build_features(_records(310))
    assert pd.isna(frame.loc[20, "future_return_300s_bps"])
    assert frame.loc[5, "label_available_at_300s"] > frame.loc[5, "available_at"]


def test_anchored_vwap_starts_only_after_observed_impulse() -> None:
    records = _records()
    records.loc[300:, "trade_count"] = 10
    frame = build_features(records)
    first = frame["anchored_vwap"].first_valid_index()
    assert first is not None and first >= 300
    assert frame.loc[first, "anchor_available_at"] <= frame.loc[first, "available_at"]


def test_mixed_valid_iso_timestamp_precision_is_accepted() -> None:
    records = _records()
    records.loc[0, "available_at"] = "2026-08-06T00:00:00+00:00"
    assert len(build_features(records)) == len(records)


def test_live_builder_emits_causal_completed_minute_alpha_contract() -> None:
    records = _records(421 * 60)
    records["trade_count"] = 2 + records.index % 7

    result = build_live_minute_features(records)

    assert len(result) == 421
    final = result.iloc[-1]
    assert bool(final["feature_valid"])
    assert bool(final["alpha_feature_contract_valid"])
    assert final["available_at"] >= final["timestamp"] + pd.Timedelta(minutes=1)
    assert final["rolling_vwap"] == final["alpha_rolling_vwap"]
    assert (
        final["rolling_vwap_5m_distance_bps"]
        == final["alpha_vwap_distance_5m_bps"]
    )


def test_official_minute_loader_excludes_the_open_kline() -> None:
    start = pd.Timestamp("2026-08-09T00:00:00Z")

    def row(minute: int) -> list[object]:
        opened = start + pd.Timedelta(minutes=minute)
        return [
            int(opened.timestamp() * 1_000),
            "100",
            "102",
            "99",
            "101",
            "10",
            int((opened + pd.Timedelta(minutes=1, milliseconds=-1)).timestamp() * 1_000),
            "1010",
            7,
            "6",
            "606",
            "0",
        ]

    result = load_recent_official_minutes(
        fetch=lambda _: [row(0), row(1), row(2)],
        retrieved_at=pd.Timestamp("2026-08-09T00:02:30Z"),
    )

    assert result["timestamp"].tolist() == [start, start + pd.Timedelta(minutes=1)]
    assert result["available_at"].iloc[-1] == start + pd.Timedelta(minutes=2)


def test_official_minutes_keep_alpha_valid_across_an_l2_collector_gap() -> None:
    records = _records(421 * 60)
    records["trade_count"] = 2 + records.index % 7
    start_second = int(pd.Timestamp("2026-08-06T00:00:00Z").timestamp())
    gap_start = start_second + 300 * 60
    records = records.loc[
        ~records["exchange_second"].between(gap_start, gap_start + 2 * 60 - 1)
    ]
    timestamp = pd.date_range("2026-08-06T00:00:00Z", periods=421, freq="1min")
    close = pd.Series([100 + minute / 10 for minute in range(421)])
    volume = pd.Series([60 + minute % 11 for minute in range(421)])
    official = pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(minutes=1),
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": volume,
            "quote_volume": close * volume,
            "taker_buy_quote": close * volume * 0.55,
            "trade_count": [120 + minute % 7 for minute in range(421)],
        }
    )

    result = build_live_minute_features(records, official)

    assert bool(result.iloc[-1]["alpha_feature_contract_valid"])
    assert bool(result.iloc[-1]["feature_valid"])
