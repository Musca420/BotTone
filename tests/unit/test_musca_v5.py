import numpy as np
import pandas as pd

from adaptive_bot.musca_v5_research import PROTOCOL, PROTOCOL_HASH, _swing_avwap


def test_swing_anchor_uses_only_confirmed_pivot() -> None:
    frame = pd.DataFrame(
        {
            "perp_low": [5, 4, 1, 4, 5],
            "perp_high": [6, 5, 2, 5, 6],
            "perp_volume": np.ones(5),
            "perp_quote_volume": [5, 4, 1, 4, 5],
        }
    )
    result = _swing_avwap(frame, long=True)
    assert result[3] != 1
    assert result[4] == (1 + 4 + 5) / 3
    assert PROTOCOL_HASH and PROTOCOL["holdout_opened"] is False
