from datetime import UTC, datetime, timedelta

from adaptive_bot.adapters.bitunix.collector import ECONOMIC_QUOTE_PROTOCOL_HASH
from adaptive_bot.musca_v5_post_only_replay import replay


def test_post_only_replay_requires_queue_then_exits_at_vwap() -> None:
    start = datetime(2026, 8, 9, tzinfo=UTC)
    rows = [
        {
            "type": "quote",
            "at": start.isoformat(),
            "fair_value": "100",
            "protocol_hash": ECONOMIC_QUOTE_PROTOCOL_HASH,
            "quotes": {
                "VIP5": {
                    "bid_price": "99",
                    "bid_queue_btc": "0.001",
                    "ask_price": "101",
                    "ask_queue_btc": "0.001",
                }
            },
        },
        {
            "type": "trade",
            "at": (start + timedelta(seconds=1)).isoformat(),
            "price": "99",
            "quantity": "0.001",
            "aggressor_side": "sell",
        },
        {
            "type": "trade",
            "at": (start + timedelta(seconds=2)).isoformat(),
            "price": "99",
            "quantity": "100.001",
            "aggressor_side": "sell",
        },
        {
            "type": "book",
            "at": (start + timedelta(seconds=3)).isoformat(),
            "best_bid": "100",
            "best_ask": "100.1",
        },
    ]

    result = replay(rows)["profiles"]["VIP5"]

    assert result["proxy_fills"] == 1
    assert result["closed_trades"] == 1
    assert result["trades"][0]["reason"] == "VWAP_TARGET"
    assert result["trades"][0]["net_bps"] > 0
