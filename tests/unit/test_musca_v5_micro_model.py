import numpy as np
import pandas as pd

from adaptive_bot.musca_v5_micro_model import build_one_second_features, label_events_5s


def test_5s_label_enters_after_availability_and_uses_worst_case() -> None:
    times = pd.date_range("2026-01-01", periods=3, freq="5s", tz="UTC")
    path = pd.DataFrame(
        {
            "timestamp": times,
            "available_at": times + pd.Timedelta(seconds=5),
            "high": [100.0, 100.0, 103.0],
            "low": [100.0, 100.0, 97.0],
            "close": [100.0, 100.0, 101.0],
        }
    )
    event = pd.DataFrame(
        [
            {
                "available_at": times[1],
                "direction": 1,
                "stop_price": 98.0,
                "target_price": 102.0,
                "exit_style": "CENTER",
            }
        ]
    )
    result = label_events_5s(event, path).iloc[0]
    assert result["entry_timestamp"] > event.iloc[0]["available_at"]
    assert result["exit_reason"] == "STRUCTURAL_STOP"
    assert result["gross_return_bps"] == -200.0


def test_one_second_features_do_not_use_future_rows() -> None:
    times = pd.date_range("2026-01-01", periods=700, freq="s", tz="UTC")
    close = 100 + np.arange(700) * 0.001
    rows = pd.DataFrame(
        {
            "timestamp": times,
            "available_at": times + pd.Timedelta(seconds=1),
            "close": close,
            "quote_volume": np.full(700, 1_000.0),
            "signed_quote_volume": np.where(np.arange(700) % 2, 200.0, -100.0),
            "trade_count": np.full(700, 10),
            "buy_count": np.full(700, 6),
        }
    )
    baseline = build_one_second_features(rows)
    changed = rows.copy()
    changed.loc[699, ["close", "signed_quote_volume", "trade_count"]] = [200.0, 999.0, 99]
    replay = build_one_second_features(changed)
    columns = [column for column in baseline if column != "available_at"]
    assert np.allclose(
        baseline.iloc[:-1][columns],
        replay.iloc[:-1][columns],
        equal_nan=True,
    )
