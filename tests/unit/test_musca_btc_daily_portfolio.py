from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_bot import musca_btc_daily_portfolio as daily
from adaptive_bot import musca_btc_daily_portfolio_fqi as fqi
from adaptive_bot import musca_btc_moe as base


def _oracle_rows() -> pd.DataFrame:
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    start = pd.Timestamp("2025-04-01T00:00:00Z")
    for minute in range(3):
        timestamp = start + pd.Timedelta(minutes=minute)
        for side in base.SIDES:
            for horizon in base.HORIZONS:
                net = 0.0
                if minute == 1 and side == 1 and horizon == 60:
                    net = 100.0
                if minute == 0 and side == 1 and horizon == 300:
                    net = 20.0
                rows.append(
                    {
                        "entry_timestamp": timestamp,
                        "side": side,
                        "horizon_seconds": horizon,
                        "stop_bps": 100.0,
                        "net_bps": net,
                    }
                )
    return pd.DataFrame(rows)


def test_daily_oracle_prices_opportunities_blocked_by_a_long_action() -> None:
    targeted = daily._daily_advantage_targets(_oracle_rows())
    long_action = targeted.loc[
        targeted["entry_timestamp"].eq(pd.Timestamp("2025-04-01T00:00:00Z"))
        & targeted["side"].eq(1)
        & targeted["horizon_seconds"].eq(300),
        "daily_q_advantage_bps",
    ].iloc[0]
    assert long_action < 0


def test_state_features_are_known_at_the_decision() -> None:
    rows = pd.DataFrame(
        {
            "entry_timestamp": [pd.Timestamp("2025-04-01T18:00:00Z")],
            "horizon_seconds": [3_600],
            "stop_bps": [91.0],
        }
    )
    derived = daily._derive_state_features(rows)
    assert derived.loc[0, "minutes_to_utc_close"] == 360
    assert derived.loc[0, "horizon_to_remaining_day"] == 1 / 6
    assert 0 < derived.loc[0, "portfolio_leverage"] <= 10


def test_replay_allows_successive_trades_but_not_overlapping_positions() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame(
        {
            "entry_timestamp": [
                start,
                start + pd.Timedelta(seconds=30),
                start + pd.Timedelta(seconds=60),
            ],
            "side": [1, -1, 1],
            "horizon_seconds": [60, 60, 60],
            "stop_bps": [100.0, 100.0, 100.0],
            "net_bps": [10.0, 500.0, 10.0],
            "score": [1.0, 2.0, 1.0],
        }
    )
    trades = daily._terminal_replay(rows, "score")
    assert trades["entry_timestamp"].tolist() == [start, start + pd.Timedelta(seconds=60)]


def test_protocol_and_feature_contract_are_fail_closed() -> None:
    assert (
        daily.hashlib.sha256(
            daily.json.dumps(daily.PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == daily.PROTOCOL_HASH
    )
    assert "net_bps" not in daily.FEATURES
    assert "gross_bps" not in daily.FEATURES
    assert daily.PROTOCOL["future_holdout_opened"] is False
    assert daily.PROTOCOL["real_capital_allowed"] is False


def test_portfolio_leverage_respects_risk_and_leverage_caps() -> None:
    leverage = daily._portfolio_leverage(np.asarray([1.0, 100.0, 1_000.0]))
    assert np.all(leverage <= daily.MAXIMUM_LEVERAGE)
    assert leverage[0] == daily.MAXIMUM_LEVERAGE
    assert leverage[2] < leverage[1]


def test_fqi_transitions_keep_flat_and_actions_inside_the_same_day() -> None:
    transitions = fqi.build_transitions(_oracle_rows(), require_complete_days=False)
    assert len(transitions.states) == 3
    assert transitions.next_wait_state.tolist() == [1, 2, -1]
    assert transitions.valid_action.all()


def test_fqi_rejects_actions_that_cannot_finish_by_utc_close() -> None:
    rows = _oracle_rows().iloc[: fqi.ACTION_COUNT].copy()
    rows["entry_timestamp"] = pd.Timestamp("2025-04-01T23:59:00Z")
    transitions = fqi.build_transitions(rows, require_complete_days=False)
    assert transitions.valid_action.sum() == 2
    assert transitions.actions.loc[transitions.valid_action, "horizon_seconds"].eq(60).all()


def test_fqi_protocol_never_uses_realized_future_maximum() -> None:
    assert fqi.PROTOCOL["fitted_q"]["realized_future_maximum_used"] is False
    assert fqi.PROTOCOL["future_holdout_opened"] is False
    assert fqi.PROTOCOL["real_capital_allowed"] is False


def test_fqi_excludes_partial_utc_days_from_training() -> None:
    assert fqi._complete_utc_days(_oracle_rows()).empty
