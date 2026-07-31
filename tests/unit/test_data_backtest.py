from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import load_config
from adaptive_bot.data.repository import ParquetRepository
from adaptive_bot.data.validation import validate_candles


def test_data_validation_and_parquet_round_trip(rth_frame: pd.DataFrame, tmp_path: Path) -> None:
    report = validate_candles(rth_frame)
    assert report.passed and report.score == 1
    path = tmp_path / "candles.parquet"
    ParquetRepository.write(rth_frame, path)
    assert len(ParquetRepository.read(path)) == len(rth_frame)
    counts = ParquetRepository.query(path, "SELECT count(*) AS n FROM candles")
    assert counts.loc[0, "n"] == len(rth_frame)


def test_data_validation_rejects_naive_timestamps(rth_frame: pd.DataFrame) -> None:
    rth_frame["timestamp"] = rth_frame["timestamp"].dt.tz_localize(None)
    report = validate_candles(rth_frame)
    assert not report.passed
    assert "timestamps must include a timezone" in report.errors


@pytest.mark.asyncio
async def test_backtest_is_deterministic(rth_frame: pd.DataFrame) -> None:
    config = load_config("configs/backtest.yaml")
    first = await BacktestEngine(config).run(rth_frame)
    second = await BacktestEngine(config).run(rth_frame)
    assert first.model_dump() == second.model_dump()
    assert first.final_equity >= 0
