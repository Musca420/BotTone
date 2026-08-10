import pandas as pd


def test_buyer_maker_flag_means_sell_aggressor() -> None:
    buyer_maker = pd.Series(["true", "false"]).astype(str).str.lower().eq("true")
    buyer_taker = ~buyer_maker
    assert buyer_taker.tolist() == [False, True]
