import json
from pathlib import Path

import pandas as pd
import pytest

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import load_config
from adaptive_bot.dashboard import server as dashboard_server
from adaptive_bot.dashboard.server import (
    build_dashboard_payload,
    build_live_market_payload,
    build_ml_payload,
    serve_dashboard,
)
from adaptive_bot.data.repository import ParquetRepository
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.services.bitunix_paper_service import profile_report_path, variant_report_path


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


@pytest.mark.asyncio
async def test_dashboard_selects_independent_adx_variant(
    rth_frame: pd.DataFrame, tmp_path: Path
) -> None:
    result = await BacktestEngine(load_config("configs/backtest.yaml")).run(rth_frame)
    report = tmp_path / "paper.json"
    result.write_json(report)
    for threshold, pnl in ((20, "0"), (23, "7")):
        payload = result.model_dump(mode="json")
        payload["range_adx_threshold"] = threshold
        payload["net_pnl"] = pnl
        variant_report_path(report, threshold).write_text(json.dumps(payload), encoding="utf-8")

    selected = build_dashboard_payload(report, 23)
    assert selected["summary"]["range_adx_threshold"] == 23
    assert selected["summary"]["net_pnl"] == "7"
    assert [
        item["range_adx_threshold"]
        for item in selected["variants"]
        if item["profile_id"].startswith("adx")
    ] == [20, 23]


@pytest.mark.asyncio
async def test_dashboard_selects_weighted_profile(rth_frame: pd.DataFrame, tmp_path: Path) -> None:
    result = await BacktestEngine(load_config("configs/backtest.yaml")).run(rth_frame)
    report = tmp_path / "paper.json"
    result.write_json(report)
    payload = result.model_dump(mode="json")
    payload["strategy_profile"] = "weighted_reversion"
    profile_report_path(report, "mr_score").write_text(json.dumps(payload), encoding="utf-8")

    selected = build_dashboard_payload(report, "mr_score")
    assert selected["summary"]["profile_id"] == "mr_score"
    assert any(variant["profile_label"] == "MR SCORE v1" for variant in selected["variants"])

    payload["strategy_profile"] = "weighted_reversion_v11"
    profile_report_path(report, "mr_score_v11").write_text(json.dumps(payload), encoding="utf-8")
    selected = build_dashboard_payload(report, "mr_score_v11")
    assert selected["summary"]["profile_label"] == "MR SCORE v1.1"


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


def test_dashboard_selects_independent_musca_v5_fee_account(tmp_path: Path, monkeypatch) -> None:
    shadow = tmp_path / "musca_v5_shadow.json"
    audit = tmp_path / "btc_cross_exchange_forward_audit.json"
    shadow.write_text(
        json.dumps(
            {
                "mode": "shadow",
                "instrument": "BTCUSDT",
                "timeframe_minutes": 1,
                "strategy_profile": "musca_v5_stable_multi_horizon_vwap",
                "fills": [],
                "telemetry": [],
            }
        ),
        encoding="utf-8",
    )
    accounts = {
        f"VIP{level}": {
            "initial_equity": 10_000,
            "final_equity": 10_000 + level,
            "net_pnl": level,
            "max_drawdown": 0,
            "trades": [{"balance": 10_000 + level}] if level else [],
        }
        for level in range(6)
    }
    audit.write_text(
        json.dumps(
            {
                "one_position_diagnostics": {"paper_accounts": accounts},
                "alpha": {
                    "accepted_candidates_by_profile": {f"VIP{level}": level for level in range(6)}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_server, "MUSCA_V5_REPORT_PATH", shadow)
    monkeypatch.setattr(dashboard_server, "CROSS_EXCHANGE_AUDIT_PATH", audit)

    payload = build_dashboard_payload(tmp_path / "unused.json", "musca-v5-vip5")

    assert payload["summary"]["profile_id"] == "musca-v5-vip5"
    assert payload["summary"]["fee_profile"] == "VIP5"
    assert payload["summary"]["forward_audit"]["selected_fee_profile"] == "VIP5"
    vip_variants = [
        variant
        for variant in payload["variants"]
        if variant["profile_id"].startswith("musca-v5-vip")
    ]
    assert len(vip_variants) == 6
    assert vip_variants[-1]["final_equity"] == 10_005


def test_dashboard_missing_report_and_remote_bind_are_safe(tmp_path: Path) -> None:
    assert build_dashboard_payload(tmp_path / "missing.json")["available"] is False
    with pytest.raises(ValueError, match="loopback-only"):
        serve_dashboard(tmp_path / "missing.json", host="0.0.0.0", port=0)


def test_expert_dashboard_reads_shared_training_status(tmp_path: Path) -> None:
    report = tmp_path / "ml_expert_research_v5.json"
    report.write_text('{"protocol":"adaptive_range_multi_expert_v5"}', encoding="utf-8")
    report.with_name("ml_expert_research.status.json").write_text(
        '{"phase":"counterfactual","completed_chunks":12}', encoding="utf-8"
    )
    payload = build_ml_payload(report)
    assert payload["available"] is True
    assert payload["status"]["completed_chunks"] == 12


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
                            "quoteVol": "0.12",
                            "baseVol": "12.12",
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
    assert payload["latest"]["daily_vwap"] == pytest.approx(101)
