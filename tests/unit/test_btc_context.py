from datetime import UTC, datetime

from adaptive_bot.btc_context import context_urls, normalize_context


def test_operational_context_polls_only_binance_and_bitunix() -> None:
    urls = context_urls(datetime(2026, 8, 8, tzinfo=UTC))

    assert set(urls) == {
        "binance_premium",
        "binance_oi",
        "binance_spot",
        "bitunix",
    }
    assert not any("bybit" in url or "okx" in url for url in urls.values())


def test_context_normalizes_official_payloads_and_fails_closed() -> None:
    received = datetime(2026, 8, 4, tzinfo=UTC)
    records = normalize_context(
        {
            "binance_premium": {
                "markPrice": "100",
                "indexPrice": "99",
                "lastFundingRate": "0.001",
                "nextFundingTime": 1_786_000_000_000,
                "time": 1_785_880_772_000,
            },
            "binance_oi": {"openInterest": "42", "time": 1_785_880_772_000},
            "binance_spot": {"symbol": "BTCUSDT", "price": "98"},
            "bybit": {
                "time": 1_785_880_772_000,
                "result": {
                    "list": [
                        {
                            "markPrice": "100",
                            "indexPrice": "100",
                            "lastPrice": "100",
                            "fundingRate": "0.001",
                            "nextFundingTime": "1786000000000",
                            "openInterest": "40",
                            "openInterestValue": "4000",
                        }
                    ]
                },
            },
            "okx_oi": {"data": [{"oiCcy": "39", "oiUsd": "3900", "ts": "1785880772000"}]},
            "okx_funding": {
                "data": [
                    {
                        "fundingRate": "0.001",
                        "fundingTime": "1786000000000",
                        "premium": "0.0002",
                        "ts": "1785880772000",
                    }
                ]
            },
            "okx_ticker": {"data": [{"last": "100", "ts": "1785880772000"}]},
            "bitunix": {
                "data": {
                    "markPrice": "100",
                    "indexPrice": "99",
                    "lastPrice": "100",
                    "fundingRate": "0.001",
                    "nextFundingTime": "1786000000000",
                }
            },
            "deribit_dvol": {"result": {"data": [[1_785_880_720_000, 30, 31, 29, 30.5]]}},
        },
        received,
    )
    by_exchange = {record["exchange"]: record for record in records}
    assert all(by_exchange[name]["coverage"] for name in by_exchange)
    assert float(by_exchange["binance"]["mark_index_basis_bps"]) > 100
    assert float(by_exchange["binance"]["mark_spot_basis_bps"]) > 200
    assert by_exchange["binance"]["funding_rate"] == "0.001"
    assert by_exchange["bitunix"]["funding_rate"] == "0.00001"
    assert by_exchange["bitunix"]["funding_rate_source_unit"] == "percent"
    assert by_exchange["deribit"]["dvol_close"] == "30.5"

    failed = normalize_context({}, received)
    assert not any(record["coverage"] for record in failed)
    assert "open_interest" in failed[0]["missing_fields"]
