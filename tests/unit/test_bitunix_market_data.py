import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import pytest

from adaptive_bot.adapters.bitunix.collector import (
    _bucket_start_ms,
    _candles_payload,
    _closed_candles,
    _economic_quote_levels,
    _quote,
    compact_microstructure_jsonl,
    normalize_deep_depth_message,
    normalize_microstructure_message,
)
from adaptive_bot.adapters.bitunix.market_data import (
    BitunixMarketData,
    _missing_timestamps,
)
from adaptive_bot.adapters.bitunix.market_data import (
    _candle as map_bitunix_candle,
)
from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.domain.exceptions import LiveTradingDisabled
from adaptive_bot.domain.models import Candle
from adaptive_bot.services.bitunix_paper_service import read_collected_candles


@pytest.mark.parametrize("market", ["spot", "futures"])
async def test_bitunix_history_maps_and_paginates_without_credentials(market: str) -> None:
    calls: list[str] = []

    def get_json(url: str) -> dict[str, object]:
        calls.append(url)
        if len(calls) > 1:
            return {"code": 0, "data": []}
        return {
            "code": 0,
            "data": [
                {
                    "time" if market == "futures" else "ts": 1_700_000_000_000,
                    "open": "100",
                    "high": "102",
                    "low": "99",
                    "close": "101",
                    "baseVol": "12",
                }
            ],
        }

    provider = BitunixMarketData(
        cast(Literal["spot", "futures"], market), timeframe_minutes=5, get_json=get_json
    )
    events = await provider.historical(
        "BTCUSDT",
        datetime.fromtimestamp(1_699_999_000, tz=UTC),
        datetime.fromtimestamp(1_700_001_000, tz=UTC),
    )
    candle = next(iter(events))
    assert candle.source == f"bitunix-{market}"
    assert candle.close == Decimal("101")
    assert candle.timeframe_minutes == 5
    assert "interval=5m" in calls[0] if market == "futures" else "interval=5" in calls[0]
    assert len(calls) == 2


def test_collector_keeps_only_closed_unseen_candles() -> None:
    payload = {
        "code": 0,
        "data": [{"time": "1000"}, {"time": "2000"}, {"time": "3000"}],
    }
    assert _closed_candles(payload, 3000, {1000}) == ({"time": "2000"},)
    assert _bucket_start_ms(datetime(2026, 1, 1, 0, 7, tzinfo=UTC), 5) == 1767225900000


def test_collector_calculates_real_spread_and_paper_frame(tmp_path: Path) -> None:
    quote = _quote({"code": 0, "data": {"asks": [["101", "2"]], "bids": [["99", "3"]]}})
    assert Decimal(quote["spread_bps"]) == Decimal("200")
    path = tmp_path / "candles.jsonl"
    path.write_text(
        json.dumps(
            {
                "spread_bps": quote["spread_bps"],
                "candle": {
                    "time": "1767225600000",
                    "open": "101.005",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                    "quoteVol": "1",
                    "baseVol": "100",
                    "type": "LAST_PRICE",
                },
            }
        ),
        encoding="utf-8",
    )
    frame = read_collected_candles(path)
    assert Decimal(frame.iloc[0]["spread_bps"]) == Decimal("200")
    assert frame.iloc[0]["high"] == 101.005
    assert bool(frame.iloc[0]["ohlc_adjusted"])
    assert frame.iloc[0]["volume"] == 1
    assert frame.iloc[0]["quote_volume"] == 100
    assert frame.iloc[0]["price_type"] == "LAST_PRICE"


def test_microstructure_normalizes_official_book_and_trade_messages() -> None:
    received = datetime(2026, 8, 3, tzinfo=UTC)
    book = normalize_microstructure_message(
        {
            "ch": "depth_book15",
            "symbol": "BTCUSDT",
            "ts": 1_775_541_541_009,
            "data": {"b": [["99", "20"]], "a": [["101", "20"]]},
        },
        received_at=received,
        notionals=(Decimal("1000"),),
    )[0]
    assert Decimal(book["spread_bps"]) == Decimal("200")
    assert book["market_order_shortfall_bps"]["1000"] == {
        "buy": "100.00",
        "sell": "100.00",
    }
    trade = normalize_microstructure_message(
        {
            "ch": "trade",
            "symbol": "BTCUSDT",
            "data": [{"t": "2026-08-03T00:00:00Z", "p": "100", "v": "0.5", "s": "buy"}],
        },
        received_at=received,
    )[0]
    assert trade["aggressor_side"] == "buy"
    assert trade["price"] == "100"


def test_full_depth_stream_keeps_snapshot_and_causal_zero_quantity_deltas() -> None:
    received = datetime(2026, 8, 9, tzinfo=UTC)
    state: dict[str, dict[Decimal, Decimal]] = {"bids": {}, "asks": {}}
    snapshot = normalize_deep_depth_message(
        {
            "ch": "depth_books",
            "symbol": "BTCUSDT",
            "ts": 1_775_541_541_009,
            "data": {
                "b": [["99", "2"], ["98", "3"]],
                "a": [["101", "2"], ["102", "3"]],
            },
        },
        state,
        snapshot=True,
        received_at=received,
        notionals=(Decimal("100"),),
    )[0]
    assert snapshot["event_type"] == "book_snapshot"
    assert snapshot["best_bid"] == "99"
    delta = normalize_deep_depth_message(
        {
            "ch": "depth_books",
            "symbol": "BTCUSDT",
            "ts": 1_775_541_541_010,
            "data": {"b": [["99", "0"], ["100", "1"]], "a": []},
        },
        state,
        snapshot=False,
        received_at=received,
        notionals=(Decimal("100"),),
    )[0]
    assert delta["event_type"] == "book_delta"
    assert delta["best_bid"] == "100"
    assert delta["best_ask"] == "101"
    assert delta["bids"] == [["99", "0"], ["100", "1"]]
    assert Decimal("99") not in state["bids"]


def test_full_depth_refresh_is_compacted_to_real_differences() -> None:
    received = datetime(2026, 8, 9, tzinfo=UTC)
    state = {
        "bids": {Decimal(str(price)): Decimal("1") for price in range(1, 101)},
        "asks": {Decimal(str(price)): Decimal("1") for price in range(101, 201)},
    }
    refreshed_bids = [[str(price), "1"] for price in range(2, 101)]
    refreshed_asks = [[str(price), "1"] for price in range(101, 201)]
    refreshed_asks[-1][1] = "2"

    delta = normalize_deep_depth_message(
        {
            "ch": "depth_books",
            "symbol": "BTCUSDT",
            "data": {"b": refreshed_bids, "a": refreshed_asks},
        },
        state,
        snapshot=False,
        received_at=received,
    )[0]

    assert delta["wire_payload_mode"] == "full_refresh_diff"
    assert delta["bids"] == [["1", "0"]]
    assert delta["asks"] == [["200", "2"]]
    assert Decimal("1") not in state["bids"]


def test_economic_quote_levels_respect_each_vip_maker_cost() -> None:
    state = {
        "bids": {
            Decimal("99.99"): Decimal("1"),
            Decimal("99.97"): Decimal("2"),
            Decimal("99.94"): Decimal("3"),
            Decimal("99.93"): Decimal("4"),
            Decimal("99.88"): Decimal("5"),
        },
        "asks": {
            Decimal("100.01"): Decimal("1"),
            Decimal("100.03"): Decimal("2"),
            Decimal("100.06"): Decimal("3"),
            Decimal("100.07"): Decimal("4"),
            Decimal("100.12"): Decimal("5"),
        },
    }

    levels = _economic_quote_levels(state, Decimal("100"))

    assert levels["VIP5"]["minimum_distance_bps"] == "6.75"
    assert levels["VIP5"]["bid_price"] == "99.93"
    assert levels["VIP5"]["ask_price"] == "100.07"
    assert levels["VIP0"]["minimum_distance_bps"] == "12.00"
    assert levels["VIP0"]["bid_price"] == "99.88"
    assert levels["VIP0"]["ask_price"] == "100.12"


def test_microstructure_jsonl_compacts_atomically_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_2026-08-03.jsonl"
    record = {
        "event_type": "trade",
        "exchange_timestamp": "2026-08-03T00:00:00+00:00",
        "received_timestamp": "2026-08-03T00:00:00+00:00",
        "price": "100",
        "quantity": "1",
        "aggressor_side": "buy",
    }
    line = json.dumps({"normalized": [record]})
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    compacted = compact_microstructure_jsonl(path)
    frame = pd.read_parquet(compacted)
    assert len(frame) == 1
    assert path.exists()


async def test_collector_bootstraps_last_price_history() -> None:
    calls: list[str] = []

    def get_json(url: str) -> dict[str, object]:
        calls.append(url)
        start = 3000 if len(calls) == 1 else 1000
        return {"code": 0, "data": [{"time": start}, {"time": start + 1000}]}

    payload = await _candles_payload(get_json, 5, bootstrap=True)
    assert [row["time"] for row in payload["data"]] == [3000, 4000, 1000, 2000]
    assert all("type=LAST_PRICE" in call for call in calls)
    assert "endTime=2999" in calls[1]


def test_history_can_explicitly_request_mark_price() -> None:
    provider = BitunixMarketData("futures", timeframe_minutes=5, futures_price_type="MARK_PRICE")
    assert "type=MARK_PRICE" in provider._history_url("BTCUSDT", 1234)


def test_history_expands_small_ohlc_envelope_but_rejects_large_deviation() -> None:
    raw = {
        "time": 1_700_000_000_000,
        "open": "100.005",
        "high": "100",
        "low": "99",
        "close": "100",
        "baseVol": "1",
    }
    candle, _ = map_bitunix_candle(raw, "BTCUSDT", "futures", 5)
    assert candle.high == Decimal("100.005")
    raw["open"] = "102"
    with pytest.raises(ValueError, match="exceeds 100 bps"):
        map_bitunix_candle(raw, "BTCUSDT", "futures", 5)


def test_history_detects_only_internal_missing_candles() -> None:
    rows = {timestamp: cast(Candle, object()) for timestamp in (1_000, 2_000, 4_000)}
    assert _missing_timestamps(rows, 1_000) == (3_000,)


def test_bitunix_configs_are_simulation_only() -> None:
    config = load_config("configs/bitunix_btc_futures_simulated.yaml")
    assert config.bitunix is not None
    assert config.instrument.symbol == "BTCUSDT"
    assert config.bitunix.margin_coin == "USDT"
    assert config.bitunix.margin_mode == "isolated"
    assert config.bitunix.leverage == Decimal("10")
    assert config.risk.target_exposure_fraction == Decimal("0.20")
    assert config.risk.target_exposure_fraction / config.bitunix.leverage == Decimal("0.02")
    raw = config.model_dump()
    raw["broker"] = "bitunix"
    with pytest.raises(LiveTradingDisabled):
        AppConfig.model_validate(raw)
