from datetime import datetime

import pytest
from pydantic import ValidationError

from tests.conftest import candle


def test_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="must be UTC"):
        candle(datetime(2026, 1, 5, 15, 0))


def test_candle_rejects_invalid_ohlc() -> None:
    with pytest.raises(ValidationError, match="within high-low"):
        candle(open_price="102")
