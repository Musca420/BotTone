import numpy as np
import pandas as pd

from adaptive_bot import btc_vwap_alpha, musca_v8_multi_horizon
from adaptive_bot.btc_cross_exchange_forward_audit import (
    DYNAMIC_PROTOCOL_START,
    EXECUTION_PROTOCOL_START,
    POLICY_HASH,
    PROTOCOL_START,
    _align_active_alpha_metadata,
    _book_vwap,
    _check_snapshot,
    _settled_bitunix_funding,
    apply_alpha_management,
    counterfactual_frame,
    current_market_assessment,
    fee_profile_counterfactuals,
    metrics,
    one_position_diagnostics,
    oracle_metrics,
    path_labels,
    rejection_funnel,
    select_anchor_events,
    select_dynamic_events,
    select_events,
)


def test_visible_gate_values_are_the_exact_canonical_alpha_inputs() -> None:
    row = pd.Series(
        {
            "alpha_return_1m_bps": 2.0,
            "alpha_return_15m_bps": 3.0,
            "alpha_return_30m_bps": 4.0,
            "alpha_vwap_slope_bps": 1.0,
            "alpha_vwap_distance_bps": 5.0,
            "alpha_taker_imbalance_60s": 0.25,
            # Legacy live fields must not leak into the canonical gate display.
            "return_1m_binance_bps": -99.0,
            "aggressive_imbalance_60s": -1.0,
        }
    )

    assert _check_snapshot("direction", True, row, 1.0)["actual"] == 3.0
    assert _check_snapshot("trend_15m", True, row, 1.0)["actual"] == 3.0
    assert _check_snapshot("trend_30m", True, row, 1.0)["actual"] == 4.0
    assert _check_snapshot("vwap_zone", True, row, 1.0)["actual"] == 5.0
    assert _check_snapshot("price_restart", True, row, 1.0)["actual"] == 2.0
    assert _check_snapshot("taker_flow_restart", True, row, 1.0)["actual"] == 0.25


def test_visible_gate_values_preserve_missing_data_as_missing() -> None:
    row = pd.Series(dtype=object)

    snapshot = _check_snapshot("taker_flow_restart", False, row, 1.0)

    assert snapshot["passed"] is False
    assert snapshot["actual"] is None


def test_active_alpha_metadata_invalidates_legacy_historical_counters() -> None:
    payload = {
        "protocol_hash": "legacy",
        "dynamic_protocol": {"name": "legacy"},
        "dynamic_results": {"legacy": {"events": 99}},
        "dynamic_rejection_funnel": {"legacy": {"feature_rows": 99}},
        "alpha": {
            "scored_candidates": 99,
            "accepted_candidates": 10,
            "latest_candidate": {"decision": "TRADE"},
        },
    }

    aligned = _align_active_alpha_metadata(payload, "NO_ECONOMIC_ALPHA")

    assert aligned["protocol_hash"] == musca_v8_multi_horizon.PROTOCOL_HASH
    assert aligned["validation_status"] == "RESEARCH_BASE_ALPHA_READY"
    assert aligned["decision_spec"]["primary_family"] == (
        "IMPULSE_PULLBACK_MULTI_HORIZON"
    )
    assert aligned["alpha"]["scored_candidates"] == 0
    assert aligned["alpha"]["latest_candidate"] is None
    assert aligned["alpha"]["historical_counterfactual_status"] == (
        "SEE_ACTIVE_ALPHA_REPORT"
    )
    assert aligned["selector"]["protocol"]["protocol_hash"] == (
        musca_v8_multi_horizon.PROTOCOL_HASH
    )
    assert aligned["alpha"]["generic_ml_challenger"]["status"] == "NO_ECONOMIC_ALPHA"
    assert "dynamic_protocol" not in aligned
    assert "dynamic_results" not in aligned
    assert "dynamic_rejection_funnel" not in aligned
    assert aligned["legacy_discovery"]["status"] == "EXCLUDED_FROM_ACTIVE_PROTOCOL"
    assert aligned["live_orders_enabled"] is False


def test_book_vwap_uses_visible_depth_and_fails_closed() -> None:
    levels = [["100", "1"], ["101", "2"]]
    assert _book_vwap(levels, 2) == 100.5
    assert _book_vwap(levels, 4) is None


def test_settled_funding_uses_official_fraction_not_current_percent() -> None:
    result = _settled_bitunix_funding(
        lambda _: {
            "code": 0,
            "data": [
                {
                    "fundingTime": "1786176000000",
                    "fundingRate": "-0.00005192",
                }
            ],
        }
    )
    assert result.loc[0, "settled_funding_rate"] == -0.00005192


def _frame() -> pd.DataFrame:
    minute = pd.date_range(PROTOCOL_START, periods=40, freq="min")
    frame = pd.DataFrame(
        {
            "minute": minute,
            "median_return_5m_bps": 3.0,
            "dispersion_return_5m_bps": 1.0,
            "return_5m_binance_bps": 3.0,
            "return_5m_bybit_bps": 2.5,
            "return_5m_okx_bps": 3.5,
            "return_1m_binance_bps": 1.0,
            "rolling_vwap_5m_distance_bps": 2.0,
            "aggressive_imbalance_60s": 0.5,
            "depth_imbalance_5": 0.5,
            "microprice_distance_bps": 0.2,
            "future_return_5m_bps": 8.0,
            "future_return_15m_bps": 12.0,
            "future_return_30m_bps": 20.0,
            "alpha_feature_contract_valid": True,
            "alpha_return_1m_bps": 1.0,
            "alpha_return_5m_bps": 3.0,
            "alpha_return_15m_bps": 4.0,
            "alpha_return_30m_bps": 5.0,
            "alpha_vwap_distance_bps": 2.0,
            "alpha_vwap_slope_bps": 0.2,
            "alpha_vwap_slope_change_bps": 0.1,
            "alpha_vwap_tests_30m": 2.0,
            "alpha_vwap_rejections_30m": 1.0,
            "alpha_time_since_vwap_cross_minutes": 2.0,
            "alpha_vwap_rejection_strength_bps": 0.0,
            "alpha_vwap_band_position": 0.2,
            "alpha_taker_imbalance_60s": 0.5,
            "alpha_range_60s_bps": 10.0,
            "alpha_atr_1m_bps": 5.0,
            "alpha_atr_5m_bps": 5.0,
            "alpha_atr_15m_bps": 5.0,
            "alpha_atr_30m_bps": 5.0,
            "alpha_realized_volatility_30m_bps": 4.0,
            "alpha_volatility_percentile": 0.5,
            "alpha_volume_percentile": 0.5,
            "oi_change_1h_binance": 0.001,
            "basis_bps_binance": 1.0,
            "funding_z_binance": 0.0,
        }
    )
    for column in btc_vwap_alpha.LIVE_REQUIRED:
        if column.startswith("alpha_") and column not in frame:
            frame[column] = 1.0
    frame["alpha_recent_low_5m"] = 99.0
    frame["alpha_recent_high_5m"] = 101.0
    return frame


def _books(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bids"] = [[[str(price), "1000"]] for price in frame["best_bid"]]
    frame["asks"] = [[[str(price), "1000"]] for price in frame["best_ask"]]
    return frame


def test_forward_events_use_the_same_five_minute_episode_edge_as_live() -> None:
    frame = _frame()
    frame["minute"] = pd.date_range(EXECUTION_PROTOCOL_START, periods=40, freq="min")
    events = select_events(frame, 15, "VWAP_PULLBACK_CONTINUATION")
    assert events["minute"].tolist() == [EXECUTION_PROTOCOL_START]
    assert events["side"].eq(1).all()


def test_future_return_does_not_change_selected_action() -> None:
    frame = _frame()
    frame["minute"] = pd.date_range(EXECUTION_PROTOCOL_START, periods=40, freq="min")
    original = select_events(frame, 15, "VWAP_PULLBACK_CONTINUATION")[["minute", "side"]]
    frame["future_return_15m_bps"] *= -1
    changed = select_events(frame, 15, "VWAP_PULLBACK_CONTINUATION")[["minute", "side"]]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), changed.reset_index(drop=True))


def test_bybit_and_okx_cannot_approve_block_or_reverse_an_entry() -> None:
    frame = _frame()
    frame["minute"] = pd.date_range(EXECUTION_PROTOCOL_START, periods=40, freq="min")
    original = select_events(frame, 15, "VWAP_PULLBACK_CONTINUATION")[["minute", "side"]]
    frame["return_5m_bybit_bps"] = -10_000.0
    frame["return_5m_okx_bps"] = -10_000.0
    frame["median_return_5m_bps"] = -10_000.0
    frame["dispersion_return_5m_bps"] = 10_000.0
    changed = select_events(frame, 15, "VWAP_PULLBACK_CONTINUATION")[["minute", "side"]]

    pd.testing.assert_frame_equal(original.reset_index(drop=True), changed.reset_index(drop=True))
    assert changed["side"].eq(1).all()


def test_rejection_funnel_matches_the_frozen_setup() -> None:
    result = rejection_funnel(_frame(), "VWAP_PULLBACK_CONTINUATION")
    assert result["feature_rows"] == 40
    assert result["price_restart"] == 40
    assert len(POLICY_HASH) == 64


def test_oracle_flat_is_neutral_when_both_trades_lose_after_costs() -> None:
    frame = _frame().iloc[:1].copy()
    frame["future_return_5m_bps"] = 2.0
    result = oracle_metrics(frame, 5)
    assert result["positive_rate"] == 0
    assert result["expectancy_bps"] == 0
    assert result["taker_taker_oracle_expectancy_bps"] == 0


def test_path_labels_enter_after_signal_and_measure_mfe_mae() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = PROTOCOL_START + pd.Timedelta(seconds=10)
    event["side"] = 1.0
    times = pd.date_range(PROTOCOL_START, periods=313, freq="s")
    mid = [100.0] * 12 + [101.0] * 100 + [99.0] * 100 + [100.5] * 101
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": mid,
            "best_bid": [value - 0.01 for value in mid],
            "best_ask": [value + 0.01 for value in mid],
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": 1.0,
            "depth_imbalance_5": 1.0,
            "microprice_distance_bps": 1.0,
            "anchored_vwap_distance_bps": 1.0,
        }
    )
    labels = path_labels(event, _books(l2), 5)
    assert labels.loc[0, "entry_available_at"] > event.iloc[0]["available_at"]
    assert labels.loc[0, "mfe_bps"] > 0
    assert labels.loc[0, "mae_bps"] < 0


def test_anchored_challenger_is_forward_only_and_non_overlapping() -> None:
    frame = _frame()
    frame["minute"] = pd.date_range(EXECUTION_PROTOCOL_START, periods=40, freq="min")
    frame["anchor_direction"] = 1.0
    frame["anchor_age_seconds"] = 120.0
    frame["anchored_vwap_distance_bps"] = 2.0
    events = select_anchor_events(frame, 15, "ANCHOR_CONTINUATION")
    assert events["minute"].tolist() == [
        EXECUTION_PROTOCOL_START,
        EXECUTION_PROTOCOL_START + pd.Timedelta(minutes=15),
        EXECUTION_PROTOCOL_START + pd.Timedelta(minutes=30),
    ]
    assert events["side"].eq(1).all()


def test_metrics_use_next_event_l2_return_not_minute_mark_label() -> None:
    event = _frame().iloc[[0]].copy()
    event["minute"] = EXECUTION_PROTOCOL_START
    event["available_at"] = EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=10)
    event["side"] = 1.0
    event["future_return_5m_bps"] = 100.0
    times = pd.date_range(
        event.iloc[0]["available_at"] + pd.Timedelta(seconds=1), periods=301, freq="s"
    )
    mid = [100.0] * 300 + [99.9]
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": mid,
            "best_bid": [value - 0.01 for value in mid],
            "best_ask": [value + 0.01 for value in mid],
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": 1.0,
            "depth_imbalance_5": 1.0,
            "microprice_distance_bps": 1.0,
            "anchored_vwap_distance_bps": 1.0,
        }
    )
    result = metrics(event, 5, _books(l2))
    assert result["signals"] == 1
    assert result["events"] == 1
    assert result["gross_expectancy_bps"] < 0
    assert result["trades"][0]["side"] == "LONG"
    assert result["trades"][0]["entry_at"] > result["trades"][0]["signal_at"]


def test_metrics_report_unfinished_path_as_pending() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = EXECUTION_PROTOCOL_START
    event["side"] = 1.0
    times = pd.date_range(EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1), periods=60, freq="s")
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": 1.0,
            "depth_imbalance_5": 1.0,
            "microprice_distance_bps": 1.0,
            "anchored_vwap_distance_bps": 1.0,
        }
    )
    result = metrics(event, 5, _books(l2))
    assert result["events"] == 0
    assert result["pending"] == 1
    assert result["execution_rejected"] == 0


def test_path_labels_fail_closed_on_l2_gap() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = EXECUTION_PROTOCOL_START
    event["side"] = 1.0
    times = pd.date_range(EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1), periods=301, freq="s")
    kept = times.delete(slice(150, 156))
    l2 = pd.DataFrame(
        {
            "available_at": kept,
            "mid": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": 1.0,
            "depth_imbalance_5": 1.0,
            "microprice_distance_bps": 1.0,
            "anchored_vwap_distance_bps": 1.0,
        }
    )
    assert path_labels(event, _books(l2), 5).empty


def test_short_execution_crosses_bid_then_ask() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = EXECUTION_PROTOCOL_START
    event["side"] = -1.0
    times = pd.date_range(EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1), periods=301, freq="s")
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": -1.0,
            "depth_imbalance_5": -1.0,
            "microprice_distance_bps": -1.0,
            "anchored_vwap_distance_bps": -1.0,
        }
    )
    labels = path_labels(event, _books(l2), 5)
    assert labels.loc[0, "terminal_bps"] == 0
    assert labels.loc[0, "executable_terminal_bps"] < 0


def test_dynamic_exit_trails_only_after_cost_covering_move() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = EXECUTION_PROTOCOL_START
    event["side"] = 1.0
    times = pd.date_range(EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1), periods=301, freq="s")
    mid = [100.0] * 70 + [100.26] * 20 + [100.15] * 211
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": mid,
            "best_bid": [value - 0.01 for value in mid],
            "best_ask": [value + 0.01 for value in mid],
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": 1.0,
            "depth_imbalance_5": 1.0,
            "microprice_distance_bps": 1.0,
            "anchored_vwap_distance_bps": 1.0,
        }
    )
    labels = path_labels(event, _books(l2), 5)
    assert labels.loc[0, "dynamic_stop_bps"] == 20.0
    assert labels.loc[0, "dynamic_target_bps"] == 50.0
    assert labels.loc[0, "fee_bps"] == 12.0
    assert labels.loc[0, "slippage_reserve_bps"] == 1.0
    assert labels.loc[0, "dynamic_exit_reason"] == "TRAIL"
    assert labels.loc[0, "dynamic_exit_available_at"] < labels.loc[0, "label_available_at"]


def test_dynamic_exit_completes_before_max_horizon() -> None:
    event = _frame().iloc[[0]].copy()
    event["available_at"] = EXECUTION_PROTOCOL_START
    event["side"] = 1.0
    times = pd.date_range(EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1), periods=340, freq="s")
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": -1.0,
            "aggressive_imbalance_5s": -1.0,
            "aggressive_imbalance_30s": -1.0,
            "depth_imbalance_5": -1.0,
            "microprice_distance_bps": -1.0,
            "anchored_vwap_distance_bps": -3.0,
        }
    )
    result = metrics(event, 30, _books(l2), dynamic=True)
    assert result["events"] == 1
    assert result["pending"] == 0
    assert result["trades"][0]["exit_reason"] == "AVWAP_FAILURE"
    assert result["trades"][0]["exit_mid"] == 100.0
    assert result["trades"][0]["entry_execution_price"] == 100.01
    assert result["trades"][0]["exit_execution_price"] == 99.99
    assert result["trades"][0]["stop_bps"] == 20.0


def test_counterfactual_contains_only_signal_time_features() -> None:
    frame = _frame().iloc[[0]].copy()
    frame["available_at"] = EXECUTION_PROTOCOL_START
    frame["price_binance"] = 100.0
    frame["future_secret"] = 999.0
    results = {
        "expert": {
            "trades": [
                {
                    "signal_at": EXECUTION_PROTOCOL_START.isoformat(),
                    "entry_at": (EXECUTION_PROTOCOL_START + pd.Timedelta(seconds=1)).isoformat(),
                    "exit_at": (EXECUTION_PROTOCOL_START + pd.Timedelta(minutes=1)).isoformat(),
                    "side": "LONG",
                    "entry_mid": 100.0,
                    "gross_bps": 10.0,
                    "net_bps": 6.0,
                    "stress_bps": 2.0,
                    "mfe_bps": 12.0,
                    "mae_bps": -1.0,
                    "exit_reason": "TRAIL",
                }
            ]
        }
    }
    result = counterfactual_frame(frame, results)
    assert len(result) == 1
    assert result.loc[0, "feature_available_at"] <= result.loc[0, "signal_at"]
    assert "future_secret" not in result


def test_one_position_diagnostic_rejects_overlapping_trade() -> None:
    start = EXECUTION_PROTOCOL_START
    frame = pd.DataFrame(
        {
            "signal_at": [start, start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=4)],
            "entry_at": [
                start + pd.Timedelta(seconds=1),
                start + pd.Timedelta(minutes=1, seconds=1),
                start + pd.Timedelta(minutes=4, seconds=1),
            ],
            "exit_at": [
                start + pd.Timedelta(minutes=3),
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=5),
            ],
            "expert_id": ["a", "b", "c"],
            "expert": ["first", "overlap", "third"],
            "side": ["LONG", "SHORT", "LONG"],
            "entry_mid": [100.0, 100.0, 100.0],
            "exit_mid": [99.99, 99.98, 100.02],
            "entry_execution_price": [100.01, 99.99, 100.01],
            "exit_execution_price": [99.98, 100.0, 100.01],
            "entry_spread_bps": [2.0, 2.0, 2.0],
            "stop_bps": [20.0, 20.0, 20.0],
            "exit_reason": ["FLOW_INVALIDATION", "TRAIL", "TIME"],
            "net_bps": [-1.0, 6.0, 2.0],
            "gross_bps": [3.0, 10.0, 6.0],
            "stress_bps": [-5.0, 2.0, 1.0],
        }
    )
    result = one_position_diagnostics(frame, paper_start=start)
    assert result["earliest_candidate"]["trades"] == 2
    assert result["earliest_candidate"]["net_bps"] == 1.0
    assert result["earliest_candidate"]["stress_bps"] == -4.0
    assert not result["earliest_candidate"]["eligible"]
    assert result["stress_oracle"]["stress_bps"] == 3.0
    assert len(result["paper_account"]["trades"]) == 2
    assert result["paper_account"]["final_equity"] < 10_000
    assert result["paper_account"]["trades"][0]["estimated_stop_loss_with_costs"] <= 100
    assert result["paper_account"]["trades"][0]["notional"] <= 10_000
    assert result["paper_account"]["margin_fraction"] == 0.10


def test_current_assessment_explains_first_failed_gate_without_fake_probability(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(btc_vwap_alpha, "BUNDLE", tmp_path / "missing.joblib")
    frame = _frame().iloc[:2].copy()
    frame["available_at"] = frame["minute"] + pd.Timedelta(seconds=1)
    frame["mid"] = 100.0
    frame["rolling_vwap"] = 99.98
    frame["anchored_vwap"] = 99.97
    frame["spread_bps"] = 1.0
    frame["range_60s_bps"] = 10.0
    frame["anchor_direction"] = 0.0
    frame["anchor_age_seconds"] = 0.0
    frame["anchored_vwap_distance_bps"] = 3.0
    result = current_market_assessment(frame, {"probability_status": "NOT_TRAINED"})
    assert result["decision"] == "WAIT"
    assert result["sources"]["alpha"]["valid"] is False
    assert result["sources"]["execution"]["valid"] is False
    assert result["probability_status"] == "NOT_TRAINED"
    assert result["target_probability"] is None
    assert result["target_price"] is None
    assert result["risk_status"] == "NOT_EVALUATED"


def test_current_assessment_keeps_binance_alpha_separate_from_bitunix_execution() -> None:
    observed = EXECUTION_PROTOCOL_START + pd.Timedelta(hours=1)
    frame = _frame().iloc[:2].copy()
    frame["available_at"] = [observed - pd.Timedelta(minutes=1), observed]
    frame["mid"] = 100.0
    frame["price_binance"] = 100.0
    frame["rolling_vwap"] = 99.98
    frame["anchored_vwap"] = float("nan")
    frame["range_60s_bps"] = 10.0
    frame["alpha_l2_available_at"] = frame["available_at"]
    frame["feature_valid"] = True
    frame["anchor_direction"] = float("nan")
    frame["anchor_age_seconds"] = float("nan")
    frame["anchor_available_at"] = pd.NaT
    frame["anchored_vwap_distance_bps"] = float("nan")
    execution = _books(
        pd.DataFrame(
            {
                "available_at": [observed],
                "mid": [200.0],
                "best_bid": [199.99],
                "best_ask": [200.01],
                "feature_valid": [True],
                "last_trade_update_age_ms": [100.0],
                "last_book_update_age_ms": [0.0],
                "sequence_gap_detected": [False],
                "book_is_synced": [True],
                "trade_feed_alive": [True],
                "orderbook_feed_alive": [True],
                "clock_drift_ms": [100.0],
                "spread_bps": [1.0],
            }
        )
    )
    result = current_market_assessment(
        frame,
        {"probability_status": "NOT_TRAINED"},
        execution,
        evaluated_at=observed + pd.Timedelta(seconds=1),
    )
    assert result["price"] == 100.0
    assert result["execution_price"] == 200.0
    assert result["sources"]["alpha"]["venue"] == "Binance"
    assert result["sources"]["execution"]["venue"] == "Bitunix"
    assert result["sources"]["alpha"]["valid"]
    assert result["sources"]["execution"]["valid"]


def test_current_assessment_scores_all_simultaneous_actions(monkeypatch, tmp_path) -> None:
    observed = EXECUTION_PROTOCOL_START + pd.Timedelta(hours=1)
    frame = _frame().iloc[:2].copy()
    frame["available_at"] = [observed - pd.Timedelta(minutes=1), observed]
    frame["mid"] = 100.0
    frame["price_binance"] = 100.0
    frame["alpha_l2_available_at"] = frame["available_at"]
    frame["feature_valid"] = True
    frame["alpha_feature_contract_valid"] = True
    frame["alpha_return_1m_bps"] = [-1.0, 1.0]
    frame["alpha_return_15m_bps"] = [-1.0, 4.0]
    frame["alpha_return_30m_bps"] = [-1.0, 5.0]
    frame["alpha_vwap_slope_bps"] = [-0.2, 0.2]
    frame["alpha_vwap_distance_bps"] = [-1.0, 1.0]
    frame["alpha_taker_imbalance_60s"] = [-0.2, 0.2]
    frame["alpha_recent_low_5m"] = 99.9
    frame["alpha_recent_high_5m"] = 100.1
    execution = _books(
        pd.DataFrame(
            {
                "available_at": [observed],
                "mid": [100.0],
                "best_bid": [99.99],
                "best_ask": [100.01],
                "feature_valid": [True],
                "last_trade_update_age_ms": [100.0],
                "last_book_update_age_ms": [0.0],
                "sequence_gap_detected": [False],
                "book_is_synced": [True],
                "trade_feed_alive": [True],
                "orderbook_feed_alive": [True],
                "clock_drift_ms": [100.0],
                "spread_bps": [2.0],
            }
        )
    )
    bundle = tmp_path / "bundle.joblib"
    bundle.write_bytes(b"present")
    monkeypatch.setattr(btc_vwap_alpha, "BUNDLE", bundle)

    def fake_score(rows: pd.DataFrame) -> pd.DataFrame:
        scored = rows.copy()
        reentry = scored["expert"].astype(str).str.contains("reentry")
        scored["alpha_target_probability_vip5"] = 0.7
        scored["alpha_stop_probability_vip5"] = 0.2
        scored["alpha_timeout_probability_vip5"] = 0.1
        scored["alpha_expected_time_to_target_vip5_minutes"] = 10.0
        scored["alpha_target_vip5_bps"] = 20.0
        scored["alpha_ev_vip5_bps"] = np.where(reentry, 10.0, 1.0)
        scored["alpha_accepted_vip5"] = True
        scored["alpha_status_vip5"] = "TRADE"
        scored["expected_mfe_60m_bps"] = 30.0
        scored["expected_mae_60m_bps"] = 10.0
        return scored

    monkeypatch.setattr(btc_vwap_alpha, "score", fake_score)
    result = current_market_assessment(
        frame,
        {"probability_status": "TRAINED"},
        execution,
        vip_level=5,
        evaluated_at=observed + pd.Timedelta(seconds=1),
    )

    assert result["decision"] == "TRADE"
    assert result["setup"] == "ROLLING_VWAP_REENTRY"
    assert result["expected_net_ev_bps"] == 10.0


def test_frozen_v8_base_can_trade_while_generic_challenger_is_disabled(
    monkeypatch, tmp_path
) -> None:
    observed = EXECUTION_PROTOCOL_START + pd.Timedelta(hours=1)
    frame = _frame().iloc[:2].copy()
    frame["available_at"] = [observed - pd.Timedelta(minutes=1), observed]
    frame["mid"] = 100.0
    frame["price_binance"] = 100.0
    frame["alpha_l2_available_at"] = frame["available_at"]
    frame["feature_valid"] = True
    frame["alpha_feature_contract_valid"] = True
    execution = _books(
        pd.DataFrame(
            {
                "available_at": [observed],
                "mid": [100.0],
                "best_bid": [99.99],
                "best_ask": [100.01],
                "feature_valid": [True],
                "last_trade_update_age_ms": [100.0],
                "last_book_update_age_ms": [0.0],
                "sequence_gap_detected": [False],
                "book_is_synced": [True],
                "trade_feed_alive": [True],
                "orderbook_feed_alive": [True],
                "clock_drift_ms": [100.0],
                "spread_bps": [2.0],
            }
        )
    )
    monkeypatch.setattr(btc_vwap_alpha, "BUNDLE", tmp_path / "missing.joblib")
    base = {
        "setup": "IMPULSE_PULLBACK_H24",
        "direction": "LONG",
        "candidate": True,
        "setup_active": True,
        "passed_checks": 7,
        "total_checks": 7,
        "first_failed_check": None,
        "checks": [],
        "policy_source": "MUSCA_V8_FROZEN_BASE",
        "expert_id": "musca-v8-h24-test",
        "available_at": observed.isoformat(),
        "stop_price": 99.7,
        "robust_expected_gross_bps": 30.0,
        "target_probability": 0.55,
        "management_style": "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M",
        "partial_target_fraction": 0.5,
        "maximum_hold_minutes": 360,
    }
    result = current_market_assessment(
        frame,
        {"probability_status": "NOT_TRAINED"},
        execution,
        vip_level=5,
        evaluated_at=observed + pd.Timedelta(seconds=1),
        base_candidate=base,
    )

    assert result["decision"] == "TRADE"
    assert result["policy_source"] == "MUSCA_V8_FROZEN_BASE"
    assert result["expected_net_ev_bps"] > 0
    assert result["maximum_hold_minutes"] == 360
    assert result["target_probability"] == 0.55
    assert np.isclose(result["target_bps"], 45.0)


def test_dynamic_reentry_waits_for_observed_exit_not_max_horizon() -> None:
    start = DYNAMIC_PROTOCOL_START
    frame = _frame().iloc[:10].copy()
    frame["minute"] = pd.date_range(start, periods=10, freq="min")
    frame["available_at"] = frame["minute"] + pd.Timedelta(seconds=10)
    frame.loc[frame.index[6], "alpha_vwap_distance_bps"] = 5.0
    times = pd.date_range(start, periods=20 * 60, freq="s")
    l2 = pd.DataFrame(
        {
            "available_at": times,
            "mid": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "range_60s_bps": 10.0,
            "aggressive_imbalance_60s": -1.0,
            "aggressive_imbalance_5s": -1.0,
            "aggressive_imbalance_30s": -1.0,
            "depth_imbalance_5": -1.0,
            "microprice_distance_bps": -1.0,
            "anchored_vwap_distance_bps": -3.0,
        }
    )
    l2 = _books(l2)
    events = select_dynamic_events(frame, l2, "VWAP_PULLBACK_CONTINUATION")
    assert len(events) > 1
    assert events.iloc[1]["available_at"] - events.iloc[0]["available_at"] < pd.Timedelta(
        minutes=30
    )
    labels = path_labels(events, l2, 30, dynamic_early=True).sort_values("signal_available_at")
    assert all(
        labels.iloc[position]["signal_available_at"]
        >= labels.iloc[position - 1]["dynamic_exit_available_at"]
        for position in range(1, len(labels))
    )


def test_alpha_target_changes_the_actual_replay_exit() -> None:
    start = EXECUTION_PROTOCOL_START
    frame = pd.DataFrame(
        {
            "entry_at": [start],
            "exit_at": [start + pd.Timedelta(seconds=60)],
            "side": ["LONG"],
            "entry_execution_price": [100.01],
            "exit_execution_price": [100.0],
            "exit_mid": [100.0],
            "fee_bps": [12.0],
            "slippage_reserve_bps": [1.0],
            "funding_bps": [0.0],
            "alpha_target_bps": [20.0],
            "time_to_20bps_seconds": [10.0],
            "time_to_stop_seconds": [30.0],
            "target_20bps_before_stop": [True],
            "gross_bps": [0.0],
            "net_bps": [-13.0],
            "stress_bps": [-26.0],
            "exit_reason": ["TIME"],
            "target_bps": [50.0],
        }
    )
    times = pd.date_range(start, periods=61, freq="s")
    mid = [100.0] * 10 + [100.25] * 51
    l2 = _books(
        pd.DataFrame(
            {
                "available_at": times,
                "mid": mid,
                "best_bid": [value - 0.01 for value in mid],
                "best_ask": [value + 0.01 for value in mid],
            }
        )
    )
    managed = apply_alpha_management(frame, l2)
    assert managed.loc[0, "exit_at"] == start + pd.Timedelta(seconds=10)
    assert managed.loc[0, "exit_reason"] == "ALPHA_DYNAMIC_TARGET"
    assert managed.loc[0, "target_bps"] == 20
    assert managed.loc[0, "net_bps"] > 0


def test_fee_profiles_keep_independent_acceptance_targets_and_balances() -> None:
    start = EXECUTION_PROTOCOL_START
    row = {
        "signal_at": start,
        "entry_at": start,
        "exit_at": start + pd.Timedelta(seconds=60),
        "expert_id": "same-alpha",
        "expert": "vwap_reversion_dynamic",
        "side": "LONG",
        "entry_mid": 100.0,
        "exit_mid": 100.55,
        "entry_execution_price": 100.01,
        "exit_execution_price": 100.54,
        "entry_spread_bps": 2.0,
        "stop_bps": 20.0,
        "gross_bps": 53.0,
        "net_bps": 40.0,
        "stress_bps": 27.0,
        "slippage_reserve_bps": 1.0,
        "funding_bps": 0.0,
        "exit_reason": "TIME",
        "target_bps": 50.0,
        "time_to_20bps_seconds": 10.0,
        "time_to_50bps_seconds": 20.0,
        "time_to_stop_seconds": 30.0,
        "target_20bps_before_stop": True,
        "target_50bps_before_stop": True,
    }
    for level in range(6):
        suffix = f"vip{level}"
        row |= {
            f"alpha_target_{suffix}_bps": 50.0 if level == 5 else 20.0,
            f"alpha_target_probability_{suffix}": 0.6,
            f"alpha_stop_probability_{suffix}": 0.2,
            f"alpha_timeout_probability_{suffix}": 0.2,
            f"alpha_expected_time_to_target_{suffix}_minutes": 15.0,
            f"alpha_expected_total_cost_{suffix}_bps": 13.0 - level,
            f"alpha_expected_net_{suffix}_bps": 1.0,
            f"alpha_prudent_net_{suffix}_bps": 0.5,
            f"alpha_accepted_{suffix}": level >= 3,
            f"alpha_status_{suffix}": "TRADE" if level >= 3 else "NO_TRADE",
        }
    scored = pd.DataFrame([row])
    times = pd.date_range(start, periods=61, freq="s")
    mid = [100.0] * 10 + [100.25] * 10 + [100.55] * 41
    l2 = _books(
        pd.DataFrame(
            {
                "available_at": times,
                "mid": mid,
                "best_bid": [value - 0.01 for value in mid],
                "best_ask": [value + 0.01 for value in mid],
            }
        )
    )

    profiles = fee_profile_counterfactuals(scored, l2)
    vip0 = one_position_diagnostics(profiles["VIP0"], paper_start=start, vip_level=0)
    vip5 = one_position_diagnostics(profiles["VIP5"], paper_start=start, vip_level=5)

    assert set(profiles) == {f"VIP{level}" for level in range(6)}
    assert not bool(profiles["VIP0"].loc[0, "alpha_accepted"])
    assert bool(profiles["VIP5"].loc[0, "alpha_accepted"])
    assert profiles["VIP0"].loc[0, "target_bps"] == 20
    assert profiles["VIP5"].loc[0, "target_bps"] == 50
    assert vip0["paper_account"]["trades"] == []
    assert len(vip5["paper_account"]["trades"]) == 1
    assert vip5["paper_account"]["fee_profile"] == "VIP5"
    assert vip5["paper_account"]["fees_per_side_bps"] == 3.5
