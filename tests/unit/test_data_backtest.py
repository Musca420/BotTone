from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import load_config
from adaptive_bot.dashboard.server import build_dashboard_payload, serve_dashboard
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
    assert len(first.telemetry) == len(rth_frame)
    assert first.telemetry[-1].activity


@pytest.mark.asyncio
async def test_dashboard_payload_uses_backtest_telemetry(
    rth_frame: pd.DataFrame, tmp_path: Path
) -> None:
    result = await BacktestEngine(load_config("configs/backtest.yaml")).run(rth_frame)
    report = tmp_path / "report.json"
    result.write_json(report)
    payload = build_dashboard_payload(report)
    assert payload["available"] is True
    assert payload["summary"]["instrument"] == "QQQ"
    assert payload["latest"]["activity"]


def test_dashboard_missing_report_and_remote_bind_are_safe(tmp_path: Path) -> None:
    assert build_dashboard_payload(tmp_path / "missing.json")["available"] is False
    with pytest.raises(ValueError, match="loopback-only"):
        serve_dashboard(tmp_path / "missing.json", host="0.0.0.0", port=0)
