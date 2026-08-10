import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot.musca_v5_l2_toxicity_model import (
    PREFILL_DIRECTIONAL,
    PREFILL_NONDIRECTIONAL,
    feature_frame,
    prefill_feature_frame,
)


def test_toxicity_features_are_signed_for_the_candidate_side() -> None:
    base = {
        "decision_at": "2026-08-03T00:00:00Z",
        "net_bps": 1,
        "reason": "VWAP_TARGET",
        "decision_depth_imbalance_1": 0.5,
        "decision_bid_cancel_rate_5s": 0.1,
        "decision_ask_cancel_rate_5s": 0.3,
    }
    features, returns, targets = feature_frame(
        [{**base, "side": "long"}, {**base, "side": "short"}]
    )

    assert features.loc[0, "signed_depth_imbalance_1"] == 0.5
    assert features.loc[1, "signed_depth_imbalance_1"] == -0.5
    assert features.loc[0, "signed_cancel_balance"] == pytest.approx(0.2)
    assert returns.tolist() == [1.0, 1.0]
    assert targets.tolist() == [1, 1]


def test_prefill_join_never_uses_a_feature_available_after_the_cancel_time(
    tmp_path: Path,
) -> None:
    entered_at = pd.Timestamp("2026-08-06T00:00:10Z")
    source = tmp_path / "trades.json"
    source.write_text(
        json.dumps(
            {
                "profiles": {
                    "VIP5": {
                        "trades": [
                            {
                                "entered_at": entered_at.isoformat(),
                                "side": "long",
                                "time_to_touch_seconds": 1.0,
                                "net_bps": 1.0,
                                "reason": "VWAP_TARGET",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def row(available_at: pd.Timestamp, value: float) -> dict[str, object]:
        result: dict[str, object] = {
            "available_at": available_at,
            "feature_valid": True,
            "mid": 100.0,
        }
        result.update({name: value for name in PREFILL_DIRECTIONAL})
        result.update({name: abs(value) for name in PREFILL_NONDIRECTIONAL})
        return result

    past = entered_at - pd.Timedelta(seconds=1)
    future = entered_at
    venue = pd.DataFrame([row(past, 0.1), row(future, 9.0)])
    binance = tmp_path / "binance.parquet"
    bitunix = tmp_path / "bitunix.parquet"
    venue.to_parquet(binance, index=False)
    venue.to_parquet(bitunix, index=False)

    features, _, _, _, coverage = prefill_feature_frame(source, binance, bitunix)

    assert coverage["causality_violations"] == 0
    assert features.loc[0, "signed_binance_mid_return_5s_bps"] == pytest.approx(0.1)
