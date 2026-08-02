import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

import pytest

from adaptive_bot.adapters.bitunix.collector import (
    _bucket_start_ms,
    _candles_payload,
    _closed_candles,
    _quote,
)
from adaptive_bot.adapters.bitunix.market_data import BitunixMarketData
from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.domain.exceptions import LiveTradingDisabled
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
                    "baseVol": "12",
                },
            }
        ),
        encoding="utf-8",
    )
    frame = read_collected_candles(path)
    assert Decimal(frame.iloc[0]["spread_bps"]) == Decimal("200")
    assert frame.iloc[0]["high"] == 101.005
    assert bool(frame.iloc[0]["ohlc_adjusted"])


async def test_collector_bootstraps_mark_price_history() -> None:
    calls: list[str] = []

    def get_json(url: str) -> dict[str, object]:
        calls.append(url)
        start = 3000 if len(calls) == 1 else 1000
        return {"code": 0, "data": [{"time": start}, {"time": start + 1000}]}

    payload = await _candles_payload(get_json, 5, bootstrap=True)
    assert [row["time"] for row in payload["data"]] == [3000, 4000, 1000, 2000]
    assert all("type=MARK_PRICE" in call for call in calls)
    assert "endTime=2999" in calls[1]


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
