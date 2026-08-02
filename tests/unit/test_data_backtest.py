import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import load_config
from adaptive_bot.dashboard.server import (
    build_dashboard_payload,
    build_live_market_payload,
    serve_dashboard,
)
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
async def test_crypto_backtest_uses_continuous_24_7_sessions() -> None:
    timestamps = pd.date_range("2026-01-01", periods=360, freq="5min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        price = 60000 + (index % 20 - 10) * 30
        rows.append(
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 50,
                "low": price - 50,
                "close": price + (10 if index % 2 else -10),
                "volume": 100,
            }
        )
    frame = pd.DataFrame(rows)
    frame["spread_bps"] = 1.25
    config = load_config("configs/bitunix_btc_futures_simulated.yaml")
    result = await BacktestEngine(config).run(frame)
    assert result.instrument == "BTCUSDT"
    assert len(result.telemetry) == len(rows)
    assert result.telemetry[-1].spread_bps == 1.25
    paper = await BacktestEngine(config).run(
        frame,
        trade_after=timestamps[-1].to_pydatetime(),
        mode="paper",
    )
    assert paper.mode == "paper"
    assert not paper.fills


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
    assert payload["safety"]["risk_per_trade"] == 0.01
    assert payload["safety"]["max_daily_loss"] == 0.02
    assert payload["safety"]["max_weekly_loss"] == 0.10
    assert payload["latest"]["activity"]
    assert payload["current_position"] == {"status": "FLAT", "quantity": "0"}
    assert payload["no_trade_reason"].startswith("No order was filled")


def test_dashboard_labels_open_and_close_operations(tmp_path: Path) -> None:
    report = tmp_path / "operations.json"
    fill = {
        "exchange_timestamp": "2026-01-05T15:00:00Z",
        "side": "buy",
        "quantity": "2",
        "price": "100",
        "commission": "1",
        "slippage": "0.1",
        "client_order_id": "entry-1",
    }
    report.write_text(
        json.dumps(
            {
                "instrument": "QQQ",
                "fills": [fill, {**fill, "side": "sell", "client_order_id": "exit-1"}],
            }
        ),
        encoding="utf-8",
    )
    payload = build_dashboard_payload(report)
    assert [event["event"] for event in payload["operations"]] == ["CLOSE", "OPEN LONG"]
    assert payload["current_position"] == {"status": "FLAT", "quantity": "0"}


def test_dashboard_missing_report_and_remote_bind_are_safe(tmp_path: Path) -> None:
    assert build_dashboard_payload(tmp_path / "missing.json")["available"] is False
    with pytest.raises(ValueError, match="loopback-only"):
        serve_dashboard(tmp_path / "missing.json", host="0.0.0.0", port=0)


def test_dashboard_reads_live_bitunix_candles_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "live.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "collected_at": "2026-01-01T00:06:00+00:00",
                        "candle": {
                            "time": "1767225600000",
                            "open": "100",
                            "high": "102",
                            "low": "99",
                            "close": "101",
                            "baseVol": "12",
                        },
                    }
                ),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )
    payload = build_live_market_payload(path)
    assert payload["available"] is True
    assert payload["bars"] == 1
    assert payload["invalid_rows"] == 1
    assert payload["latest"]["close"] == 101
    assert payload["latest"]["center"] is None
