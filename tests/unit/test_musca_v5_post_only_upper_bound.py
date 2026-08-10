import pandas as pd

from adaptive_bot.musca_v5_post_only_upper_bound import simulate


def test_touch_upper_bound_enters_after_latency_and_exits_at_frozen_vwap() -> None:
    start = pd.Timestamp("2026-08-03T00:00:00Z")
    frame = pd.DataFrame(
        [
            {
                "event_type": "book",
                "available_at": start,
                "best_bid": 99.99,
                "best_ask": 100.01,
                "mid": 100.0,
                "rolling_vwap": 100.0,
                "depth_imbalance_5bps": 0.5,
                "microprice_distance_bps": 0.1,
                "aggressive_imbalance_5s": 0.5,
                "rolling_vwap_slope_bps_60s": 0.1,
                "market_regime": "NORMAL",
                "feature_valid": True,
            },
            {
                "event_type": "book",
                "available_at": start + pd.Timedelta(seconds=1),
                "best_bid": 99.89,
                "best_ask": 99.9,
                "mid": 99.895,
                "rolling_vwap": 100.0,
                "depth_imbalance_5bps": 0.5,
                "microprice_distance_bps": 0.1,
                "aggressive_imbalance_5s": 0.5,
                "rolling_vwap_slope_bps_60s": 0.1,
                "market_regime": "NORMAL",
                "feature_valid": True,
            },
            {
                "event_type": "book",
                "available_at": start + pd.Timedelta(seconds=2),
                "best_bid": 100.0,
                "best_ask": 100.01,
                "mid": 100.005,
                "rolling_vwap": 100.0,
                "depth_imbalance_5bps": 0.5,
                "microprice_distance_bps": 0.1,
                "aggressive_imbalance_5s": 0.5,
                "rolling_vwap_slope_bps_60s": 0.1,
                "market_regime": "NORMAL",
                "feature_valid": True,
            },
        ]
    )

    result = simulate(frame, 5)

    assert result["closed_trades"] == 1
    assert result["trades"][0]["reason"] == "VWAP_TARGET"
    assert result["trades"][0]["net_bps"] > 0

    dynamic = frame.copy()
    dynamic.loc[1, "rolling_vwap"] = 99.91
    invalidated = simulate(dynamic, 5, dynamic_center=True)
    assert invalidated["trades"][0]["reason"] == "VWAP_EDGE_INVALIDATED"

    filtered = simulate(frame, 5, l2_filter=True)
    assert filtered["closed_trades"] == 1

    maker_target = simulate(frame, 5, maker_target=True)
    assert maker_target["trades"][0]["exit_role"] == "maker"

    timeout_frame = frame.copy()
    timeout_frame.loc[2, ["best_bid", "best_ask", "mid"]] = [99.95, 99.96, 99.955]
    short_lived = simulate(
        timeout_frame,
        5,
        maker_target=True,
        position_lifetime_seconds=1,
    )
    assert short_lived["trades"][0]["reason"] == "TIMEOUT"
