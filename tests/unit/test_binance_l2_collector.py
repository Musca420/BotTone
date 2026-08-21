from adaptive_bot.binance_l2_collector import normalize


def test_binance_l2_normalizes_book_and_aggressor_side() -> None:
    kind, book = normalize(
        {
            "stream": "btcusdt@depth20@100ms",
            "data": {"E": 1, "u": 2, "b": [["100", "3"]], "a": [["101", "4"]]},
        }
    )
    assert kind == "book" and book["last_update_id"] == 2
    kind, trade = normalize(
        {
            "stream": "btcusdt@aggtrade",
            "data": {"T": 3, "p": "100", "q": "0.1", "m": False},
        }
    )
    assert kind == "trade" and trade["buyer_taker"] is True
