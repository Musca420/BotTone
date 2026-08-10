from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from adaptive_bot.adapters.alpaca.broker import AlpacaPaperBroker
from adaptive_bot.adapters.alpaca.market_data import AlpacaMarketData
from adaptive_bot.adapters.alpaca.trade_updates import AlpacaTradeUpdates
from adaptive_bot.adapters.bitunix.collector import (
    collect_futures_candles,
    collect_futures_microstructure,
)
from adaptive_bot.adapters.bitunix.market_data import BitunixMarketData
from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig, alpaca_credentials, load_config
from adaptive_bot.dashboard.meme_server import serve_meme_dashboard
from adaptive_bot.dashboard.server import serve_dashboard
from adaptive_bot.data.interfaces import MarketDataProvider
from adaptive_bot.data.repository import ParquetRepository, SQLiteStateStore
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Position
from adaptive_bot.expert_policy import finalize_expert_training, run_expert_training
from adaptive_bot.expert_policy_v6 import finalize_v6_training, run_v6_training
from adaptive_bot.hybrid_policy import (
    REPORT_PATH as HYBRID_REPORT_PATH,
)
from adaptive_bot.hybrid_policy import (
    STATUS_PATH as HYBRID_STATUS_PATH,
)
from adaptive_bot.hybrid_policy import (
    download_alpha_archives,
    preregister_hybrid,
    run_hybrid_training,
    write_hybrid_failure,
)
from adaptive_bot.hybrid_policy_v11 import REPORT_PATH as HYBRID_V11_REPORT_PATH
from adaptive_bot.hybrid_policy_v11 import STATUS_PATH as HYBRID_V11_STATUS_PATH
from adaptive_bot.hybrid_policy_v11 import run_v11
from adaptive_bot.hybrid_policy_v12 import REPORT_PATH as HYBRID_V12_REPORT_PATH
from adaptive_bot.hybrid_policy_v12 import STATUS_PATH as HYBRID_V12_STATUS_PATH
from adaptive_bot.hybrid_policy_v12 import run_v12, write_v12_failure
from adaptive_bot.hybrid_policy_v13 import REPORT_PATH as HYBRID_V13_REPORT_PATH
from adaptive_bot.hybrid_policy_v13 import STATUS_PATH as HYBRID_V13_STATUS_PATH
from adaptive_bot.hybrid_policy_v13 import run_v13, write_v13_failure
from adaptive_bot.hybrid_policy_v14 import FORWARD_LOCK_PATH as HYBRID_V14_FORWARD_LOCK_PATH
from adaptive_bot.hybrid_policy_v14 import REPORT_PATH as HYBRID_V14_REPORT_PATH
from adaptive_bot.hybrid_policy_v14 import STATUS_PATH as HYBRID_V14_STATUS_PATH
from adaptive_bot.hybrid_policy_v14 import (
    forward_readiness,
    freeze_research_candidate,
    run_v14,
    write_v14_failure,
)
from adaptive_bot.meme.collector import collect_meme_market, download_meme_history
from adaptive_bot.meme.config import MemeBotConfig, load_meme_config
from adaptive_bot.meme.luna import (
    LunaLowRequest,
    PolicyStore,
    codex_login_status,
    run_luna_low,
    run_luna_max,
    validate_policy,
)
from adaptive_bot.meme.research import build_shadow_dataset, shadow_status
from adaptive_bot.meme.runtime import (
    MemePaperEngine,
    load_cached_contracts,
    load_market_qualities,
    load_recorded_frames,
    run_meme_paper,
)
from adaptive_bot.ml_research import (
    download_official_ml_history,
    run_ml_research,
    write_ml_status,
)
from adaptive_bot.policy_discovery import policy_research_config, run_policy_discovery
from adaptive_bot.research import (
    ResearchRegistry,
    refresh_research_shadow,
    run_research,
    write_research_status,
)
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.scientific_ml import (
    download_scientific_archive,
    finalize_scientific_research,
    run_scientific_research,
    scientific_archive_path,
)
from adaptive_bot.services.bitunix_paper_service import run_bitunix_paper
from adaptive_bot.services.market_data_service import download_history
from adaptive_bot.services.paper_service import PaperRuntime, run_paper
from adaptive_bot.services.recovery_service import reconcile_before_trading
from adaptive_bot.services.trading_service import run_shadow


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-bot")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--timeframe-minutes", type=int, default=15)

    download = commands.add_parser("download-data")
    download.add_argument("--config", type=Path, required=True)
    download.add_argument("--start", type=_utc_datetime)
    download.add_argument("--end", type=_utc_datetime)
    download.add_argument("--output", type=Path, default=Path("data/raw/qqq_15m.parquet"))

    collect = commands.add_parser("collect-bitunix")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument(
        "--output", type=Path, default=Path("data/raw/bitunix_btcusdt_last_futures_5m.jsonl")
    )
    collect.add_argument("--duration-hours", type=float, default=168)
    collect.add_argument("--poll-seconds", type=float, default=60)
    collect.add_argument("--timeframe-minutes", type=int)
    collect.add_argument(
        "--price-type",
        choices=("LAST_PRICE", "MARK_PRICE"),
        default="LAST_PRICE",
    )

    microstructure = commands.add_parser("collect-bitunix-microstructure")
    microstructure.add_argument("--config", type=Path, required=True)
    microstructure.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/raw/bitunix_microstructure"),
    )
    microstructure.add_argument("--duration-hours", type=float, default=168)
    microstructure.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))

    bitunix_paper = commands.add_parser("paper-bitunix")
    bitunix_paper.add_argument("--config", type=Path, required=True)
    bitunix_paper.add_argument(
        "--input", type=Path, default=Path("data/raw/bitunix_btcusdt_last_futures_5m.jsonl")
    )
    bitunix_paper.add_argument(
        "--output", type=Path, default=Path("data/reports/bitunix_paper.json")
    )
    bitunix_paper.add_argument("--duration-hours", type=float, default=168)
    bitunix_paper.add_argument("--poll-seconds", type=float, default=15)

    meme_collect = commands.add_parser("meme-collect")
    meme_collect.add_argument("--config", type=Path, required=True)
    meme_collect.add_argument("--duration-hours", type=float, default=168)

    meme_history = commands.add_parser("meme-download-history")
    meme_history.add_argument("--config", type=Path, required=True)
    meme_history.add_argument("--weeks", type=int, default=52)

    meme_backtest = commands.add_parser("meme-backtest")
    meme_backtest.add_argument("--config", type=Path, required=True)
    meme_backtest.add_argument("--events", type=Path)
    meme_backtest.add_argument("--output", type=Path)

    meme_paper = commands.add_parser("meme-paper")
    meme_paper.add_argument("--config", type=Path, required=True)
    meme_paper.add_argument("--duration-hours", type=float, default=168)
    meme_paper.add_argument("--poll-seconds", type=float, default=15)

    meme_dashboard = commands.add_parser("meme-dashboard")
    meme_dashboard.add_argument("--config", type=Path, required=True)
    meme_dashboard.add_argument("--host")
    meme_dashboard.add_argument("--port", type=int)
    meme_dashboard.add_argument("--allow-non-loopback", action="store_true")

    meme_dataset = commands.add_parser("meme-build-dataset")
    meme_dataset.add_argument("--config", type=Path, required=True)
    meme_dataset.add_argument("--events", type=Path)
    meme_dataset.add_argument(
        "--output", type=Path, default=Path("data/meme/processed/shadow_features.parquet")
    )
    meme_luna = commands.add_parser("meme-luna-sidecar")
    meme_luna.add_argument("--config", type=Path, required=True)
    meme_luna.add_argument("--duration-hours", type=float, default=168)
    meme_luna.add_argument("--poll-seconds", type=float, default=5)
    meme_luna.add_argument("--once", action="store_true")
    meme_luna.add_argument("--refresh-max", action="store_true")

    meme_luna_status = commands.add_parser("meme-luna-status")
    meme_luna_status.add_argument("--config", type=Path, required=True)

    backtest = commands.add_parser("backtest")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.add_argument("--input", type=Path)
    backtest.add_argument("--output", type=Path, default=Path("data/reports/backtest.json"))
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--report", type=Path, default=Path("data/reports/backtest.json"))
    dashboard.add_argument(
        "--live-data", type=Path, default=Path("data/raw/bitunix_btcusdt_last_futures_5m.jsonl")
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8080)
    dashboard.add_argument(
        "--research-report", type=Path, default=Path("data/reports/research.json")
    )
    dashboard.add_argument(
        "--ml-report",
        type=Path,
        default=Path("data/reports/ml_expert_research_v5.json"),
    )
    research_init = commands.add_parser("research-init")
    research_init.add_argument("--config", type=Path, required=True)
    research_init.add_argument("--skip-download", action="store_true")
    research_init.add_argument("--core-only", action="store_true")
    research_init.add_argument("--candidate-limit", type=int)
    research_worker = commands.add_parser("research-worker")
    research_worker.add_argument("--config", type=Path, required=True)
    ml_research = commands.add_parser("ml-research")
    ml_research.add_argument("--config", type=Path, required=True)
    ml_research.add_argument("--input", type=Path)
    ml_research.add_argument("--resume", action="store_true")
    ml_policy = commands.add_parser("ml-policy-research")
    ml_policy.add_argument("--config", type=Path, required=True)
    ml_policy.add_argument("--input", type=Path)
    ml_status = commands.add_parser("ml-status")
    ml_status.add_argument("--config", type=Path, required=True)
    ml_status.add_argument("--watch", action="store_true")
    ml_status.add_argument("--interval", type=float, default=5.0)
    ml_download = commands.add_parser("ml-download-data")
    ml_download.add_argument("--config", type=Path, required=True)
    ml_download.add_argument("--start", type=_utc_datetime)
    ml_download.add_argument("--end", type=_utc_datetime)
    ml_download.add_argument("--all-available", action="store_true")
    ml_finalize = commands.add_parser("ml-finalize")
    ml_finalize.add_argument("--config", type=Path, required=True)
    ml_finalize.add_argument("--input", type=Path)
    ml_finalize.add_argument("--run-id", required=True)
    ml_finalize.add_argument("--open-holdout", action="store_true", required=True)
    expert_train = commands.add_parser("ml-expert-train")
    expert_train.add_argument("--config", type=Path, required=True)
    expert_train.add_argument("--input", type=Path)
    expert_train.add_argument("--resume", action="store_true")
    expert_train.add_argument("--protocol", choices=("v5", "v6"), default="v5")
    expert_train.add_argument("--preregister-only", action="store_true")
    expert_status = commands.add_parser("ml-expert-status")
    expert_status.add_argument("--config", type=Path, required=True)
    expert_status.add_argument("--watch", action="store_true")
    expert_status.add_argument("--interval", type=float, default=5.0)
    expert_status.add_argument("--protocol", choices=("v5", "v6"), default="v5")
    expert_finalize = commands.add_parser("ml-expert-finalize")
    expert_finalize.add_argument("--config", type=Path, required=True)
    expert_finalize.add_argument("--run-id", required=True)
    expert_finalize.add_argument("--open-holdout", action="store_true", required=True)
    expert_finalize.add_argument("--protocol", choices=("v5", "v6"), default="v5")
    hybrid_init = commands.add_parser("ml-hybrid-init")
    hybrid_init.add_argument("--config", type=Path, required=True)
    hybrid_init.add_argument("--start", type=_utc_datetime, required=True)
    hybrid_init.add_argument("--end", type=_utc_datetime, required=True)
    hybrid_train = commands.add_parser("ml-hybrid-train")
    hybrid_train.add_argument("--config", type=Path, required=True)
    hybrid_train.add_argument("--download", action="store_true")
    hybrid_status = commands.add_parser("ml-hybrid-status")
    hybrid_status.add_argument("--watch", action="store_true")
    hybrid_status.add_argument("--interval", type=float, default=5.0)
    hybrid_v11_train = commands.add_parser("ml-hybrid-v11-train")
    hybrid_v11_train.add_argument("--config", type=Path, required=True)
    hybrid_v11_train.add_argument("--resume", action="store_true")
    hybrid_v11_train.add_argument("--smoke", action="store_true")
    hybrid_v11_status = commands.add_parser("ml-hybrid-v11-status")
    hybrid_v11_status.add_argument("--watch", action="store_true")
    hybrid_v11_status.add_argument("--interval", type=float, default=5.0)
    hybrid_v12_train = commands.add_parser("ml-hybrid-v12-train")
    hybrid_v12_train.add_argument("--config", type=Path, required=True)
    hybrid_v12_train.add_argument("--resume", action="store_true")
    hybrid_v12_train.add_argument("--smoke", action="store_true")
    hybrid_v12_status = commands.add_parser("ml-hybrid-v12-status")
    hybrid_v12_status.add_argument("--watch", action="store_true")
    hybrid_v12_status.add_argument("--interval", type=float, default=5.0)
    hybrid_v13_train = commands.add_parser("ml-hybrid-v13-train")
    hybrid_v13_train.add_argument("--config", type=Path, required=True)
    hybrid_v13_train.add_argument("--resume", action="store_true")
    hybrid_v13_train.add_argument("--smoke", action="store_true")
    hybrid_v13_status = commands.add_parser("ml-hybrid-v13-status")
    hybrid_v13_status.add_argument("--watch", action="store_true")
    hybrid_v13_status.add_argument("--interval", type=float, default=5.0)
    hybrid_v14_train = commands.add_parser("ml-hybrid-v14-train")
    hybrid_v14_train.add_argument("--config", type=Path, required=True)
    hybrid_v14_train.add_argument("--resume", action="store_true")
    hybrid_v14_train.add_argument("--smoke", action="store_true")
    hybrid_v14_status = commands.add_parser("ml-hybrid-v14-status")
    hybrid_v14_status.add_argument("--watch", action="store_true")
    hybrid_v14_status.add_argument("--interval", type=float, default=5.0)
    hybrid_v14_freeze = commands.add_parser("ml-hybrid-v14-freeze-forward")
    hybrid_v14_freeze.add_argument("--config", type=Path, required=True)
    hybrid_v14_forward = commands.add_parser("ml-hybrid-v14-forward-status")
    hybrid_v14_forward.add_argument("--config", type=Path, required=True)
    hybrid_v14_forward.add_argument("--watch", action="store_true")
    hybrid_v14_forward.add_argument("--interval", type=float, default=3600.0)
    musca_btc_policy_train = commands.add_parser("musca-btc-policy-train")
    musca_btc_policy_train.add_argument("--resume", action="store_true")
    musca_btc_policy_status = commands.add_parser("musca-btc-policy-status")
    musca_btc_policy_status.add_argument("--watch", action="store_true")
    musca_btc_policy_status.add_argument("--interval", type=float, default=5.0)
    research_worker.add_argument("--once", action="store_true")
    research_status = commands.add_parser("research-status")
    research_status.add_argument("--config", type=Path, required=True)
    research_pin = commands.add_parser("research-pin")
    research_pin.add_argument("--config", type=Path, required=True)
    research_pin.add_argument("candidate_id")
    research_pin.add_argument("--unpin", action="store_true")
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--config", type=Path, required=True)
    reconcile.add_argument("--accept-broker-state", action="store_true")
    reconcile.add_argument("--reset-kill-switch", action="store_true")
    reconcile.add_argument("--actor")
    reconcile.add_argument("--reason")
    shadow = commands.add_parser("shadow")
    shadow.add_argument("--config", type=Path, required=True)
    shadow.add_argument("--input", type=Path)
    shadow.add_argument("--output", type=Path, default=Path("data/reports/shadow.json"))
    paper = commands.add_parser("paper")
    paper.add_argument("--config", type=Path, required=True)
    paper.add_argument("--input", type=Path)
    paper.add_argument("--output", type=Path, default=Path("data/reports/paper.json"))
    commands.add_parser("live")
    return parser


def _musca_btc_policy_status(arguments: argparse.Namespace) -> int:
    path = Path("data/reports/musca_btc_policy.status.json")
    last = ""
    while True:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content != last:
                print(content, flush=True)
                last = content
            payload = json.loads(content)
            if payload.get("phase") in {"complete", "failed"}:
                return 0 if payload.get("phase") == "complete" else 2
        else:
            print(json.dumps({"phase": "not_started", "percent": 0}), flush=True)
        if not arguments.watch:
            return 0
        time.sleep(max(0.5, float(arguments.interval)))


async def _backtest(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    input_path = arguments.input or config.backtest.input_path
    frame = ParquetRepository.read(input_path)
    result = await BacktestEngine(config).run(frame)
    result.write_json(arguments.output)
    print(result.model_dump_json(indent=2))
    return 0


async def _download(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    end = arguments.end or datetime.now(UTC)
    provider: MarketDataProvider
    if config.bitunix is not None:
        start = arguments.start or end - timedelta(days=30)
        provider = BitunixMarketData(
            config.bitunix.market,
            timeframe_minutes=config.strategy.timeframe_minutes,
        )
    else:
        if config.alpaca is None:
            raise ValueError("Alpaca or Bitunix market-data configuration is required")
        key, secret = alpaca_credentials()
        start = arguments.start or end - timedelta(days=config.alpaca.historical_days)
        provider = AlpacaMarketData(
            key,
            secret,
            feed=config.alpaca.feed,
            adjustment=config.alpaca.adjustment,
        )
    report = await download_history(
        provider,
        config.instrument.symbol,
        start,
        end,
        arguments.output,
        timeframe_minutes=config.strategy.timeframe_minutes,
        regular_session=config.instrument.asset_class.value == "equity",
    )
    print(json.dumps(asdict(report), indent=2))
    return 0


async def _research_init(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    research = config.research
    if research is None or not research.enabled:
        raise ValueError("research mode is disabled")
    write_research_status(config, "starting", detail="Preparing historical data")
    if not arguments.skip_download:
        end = datetime.now(UTC)
        start = end - timedelta(days=round(research.history_months * 365.25 / 12))
        provider = BitunixMarketData(
            "futures",
            timeframe_minutes=config.strategy.timeframe_minutes,
            futures_price_type="MARK_PRICE",
            progress_callback=lambda count, fraction: write_research_status(
                config,
                "downloading",
                round(fraction * 1000),
                1000,
                f"{count:,} MARK_PRICE candles",
            ),
        )
        await download_history(
            provider,
            config.instrument.symbol,
            start,
            end,
            research.history_path,
            timeframe_minutes=config.strategy.timeframe_minutes,
            regular_session=False,
        )
    write_research_status(config, "validating", detail="Checking historical dataset")
    frame = ParquetRepository.read(research.history_path)
    payload = await run_research(
        config,
        frame,
        include_broad=not arguments.core_only,
        candidate_limit=arguments.candidate_limit,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "evaluated": len(payload["evaluations"]),
                "report": str(research.report_path),
            },
            indent=2,
        )
    )
    return 0


async def _research_worker(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    research = config.research
    if research is None or not research.enabled:
        raise ValueError("research mode is disabled")
    while True:
        frame = await _update_research_history(config)
        now = datetime.now(UTC)
        payload = None
        if research.report_path.exists():
            payload = json.loads(research.report_path.read_text(encoding="utf-8"))
        last_full = (
            datetime.fromisoformat(payload["completed_at"]).astimezone(UTC)
            if payload is not None
            else None
        )
        full_due = payload is None or (
            (now.hour, now.minute) >= (0, 15)
            and (last_full is None or last_full.date() < now.date())
        )
        if full_due:
            await run_research(config, frame, include_broad=now.weekday() == 6)
        else:
            assert payload is not None
            await refresh_research_shadow(config, frame, payload)
        if arguments.once:
            return 0
        await asyncio.sleep(300)


async def _update_research_history(config: AppConfig) -> pd.DataFrame:
    if config.research is None:
        raise ValueError("research configuration is missing")
    research = config.research
    existing = ParquetRepository.read(research.history_path)
    timestamps = pd.to_datetime(existing["timestamp"], utc=True)
    start = timestamps.max().to_pydatetime() + timedelta(minutes=config.strategy.timeframe_minutes)
    end = datetime.now(UTC)
    if start >= end - timedelta(minutes=config.strategy.timeframe_minutes):
        return existing
    update_path = research.history_path.with_name(f"{research.history_path.stem}.update.parquet")
    await download_history(
        BitunixMarketData(
            "futures",
            timeframe_minutes=config.strategy.timeframe_minutes,
            futures_price_type="MARK_PRICE",
            progress_callback=lambda count, fraction: write_research_status(
                config,
                "updating_history",
                round(fraction * 1000),
                1000,
                f"{count:,} new candles scanned",
            ),
        ),
        config.instrument.symbol,
        start,
        end,
        update_path,
        timeframe_minutes=config.strategy.timeframe_minutes,
        regular_session=False,
    )
    combined = pd.concat((existing, ParquetRepository.read(update_path)), ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    validate_candles(
        combined, timeframe_minutes=config.strategy.timeframe_minutes, calendar_name=None
    ).require(1.0)
    merged_path = research.history_path.with_name(f"{research.history_path.stem}.merge.parquet")
    ParquetRepository.write(combined, merged_path)
    merged_path.replace(research.history_path)
    return combined


def _research_status(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    research = config.research
    if research is None:
        raise ValueError("research configuration is missing")
    if not research.report_path.exists():
        print(json.dumps({"available": False, "error": "No research run is available."}, indent=2))
        return 2
    payload = json.loads(research.report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "available": True,
                "run_id": payload["run_id"],
                "completed_at": payload["completed_at"],
                "evaluated": len(payload["evaluations"]),
                "champions": payload.get("champions", []),
            },
            indent=2,
        )
    )
    return 0


def _research_pin(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    research = config.research
    if research is None:
        raise ValueError("research configuration is missing")
    changed = ResearchRegistry(research.database_path).pin(
        arguments.candidate_id, not arguments.unpin
    )
    print(
        json.dumps(
            {
                "candidate_id": arguments.candidate_id,
                "pinned": not arguments.unpin,
                "updated": changed,
            },
            indent=2,
        )
    )
    return 0 if changed else 2


async def _collect_bitunix(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config.bitunix is None or config.bitunix.market != "futures":
        raise ValueError("collector requires the Bitunix futures configuration")
    count = await collect_futures_candles(
        arguments.output,
        duration_hours=arguments.duration_hours,
        poll_seconds=arguments.poll_seconds,
        timeframe_minutes=arguments.timeframe_minutes or config.strategy.timeframe_minutes,
        price_type=arguments.price_type,
    )
    print(json.dumps({"collected": count, "output": str(arguments.output)}, indent=2))
    return 0


async def _collect_bitunix_microstructure(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config.bitunix is None or config.bitunix.market != "futures":
        raise ValueError("microstructure collector requires Bitunix futures")
    result = await collect_futures_microstructure(
        arguments.output_directory,
        symbol=arguments.symbol or config.instrument.symbol,
        duration_hours=arguments.duration_hours,
    )
    print(json.dumps(result, indent=2))
    return 0


def _ml_research(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    ml = config.machine_learning
    if ml is None or not ml.enabled:
        raise ValueError("machine-learning research is disabled")
    source = arguments.input or (
        scientific_archive_path(ml, "BTCUSDT")
        if ml.protocol_version == "scientific_v2"
        else ml.history_path
    )
    frame = ParquetRepository.read(source)
    result = (
        run_scientific_research(config, frame, resume=arguments.resume)
        if ml.protocol_version == "scientific_v2"
        else run_ml_research(config, frame)
    )
    print(
        json.dumps(
            {
                "run_id": result.get("run_id"),
                "verdict": result.get("development_verdict", result.get("verdict")),
                "gate_passed": result.get("development_gate_passed", result.get("accepted")),
                "report": str(ml.report_path),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _ml_policy_research(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    ml = config.machine_learning
    if ml is None or not ml.enabled:
        raise ValueError("machine-learning research is disabled")
    source = arguments.input or scientific_archive_path(ml, "BTCUSDT")
    write_ml_status(
        policy_research_config(config),
        "archive_load",
        f"Loading existing real archive: {source}",
        1,
        backend="cuda",
    )
    result = run_policy_discovery(config, ParquetRepository.read(source))
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_download(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    end = arguments.end or datetime.now(UTC)
    ml = config.machine_learning
    if ml is None:
        raise AssertionError("machine-learning configuration disappeared")
    if ml.protocol_version == "scientific_v2":
        start = arguments.start or (
            datetime(2022, 4, 1, tzinfo=UTC)
            if arguments.all_available
            else end - timedelta(days=365)
        )
        manifest = download_scientific_archive(config, start, end)
        print(json.dumps(manifest, indent=2))
        return 0
    start = arguments.start or end - timedelta(days=365)
    frame = download_official_ml_history(config, start, end)
    print(json.dumps({"rows": len(frame), "output": str(ml.history_path)}, indent=2))
    return 0


def _ml_status(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    ml = config.machine_learning
    if ml is None:
        raise ValueError("machine-learning configuration is missing")
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not ml.status_path.exists()
            else json.loads(ml.status_path.read_text(encoding="utf-8"))
        )
        if arguments.watch:
            completed = payload.get("completed")
            total = payload.get("total")
            if completed is None and payload.get("completed_chunks") is not None:
                completed = payload["completed_chunks"]
                total = payload.get("total_chunks")
            count = f" | {completed}/{total}" if completed is not None and total else ""
            rows = (
                f" | righe {int(str(payload['downloaded_rows'])):,}"
                if payload.get("downloaded_rows") is not None
                else ""
            )
            cursor = f" | fino a {payload['cursor']}" if payload.get("cursor") else ""
            backend = f" | {payload['backend']}" if payload.get("backend") else ""
            workers = (
                f" | worker {payload['active_workers']}"
                if payload.get("active_workers") is not None
                else ""
            )
            eta = (
                f" | ETA {timedelta(seconds=int(str(payload['eta_seconds'])))}"
                if payload.get("eta_seconds") is not None
                else ""
            )
            finding = ""
            if payload.get("best_candidate"):
                finding = (
                    f" | BEST {payload['best_candidate']}"
                    f" E={payload.get('best_expectancy_r')}R"
                    f" PF={payload.get('best_profit_factor')}"
                    f" trade={payload.get('best_trades')}"
                )
            elif payload.get("selected_mode"):
                finding = (
                    f" | mode={payload['selected_mode']}"
                    f" det={payload.get('deterministic_score_r')}R"
                    f" ML={payload.get('meta_score_r')}R"
                )
            print(
                f"[{payload.get('updated_at', '-')}] {payload.get('percent', 0):>5}%"
                f" | {payload.get('phase', 'unknown')}{count}{rows}{cursor}{eta}"
                f"{workers}{backend}"
                f"{finding} | {payload.get('detail', '')}",
                flush=True,
            )
        else:
            print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {
            "complete",
            "development_complete",
            "failed",
        }:
            return 0 if payload.get("phase") != "failed" else 2
        time.sleep(arguments.interval)


def _ml_finalize(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    ml = config.machine_learning
    if ml is None:
        raise ValueError("machine-learning configuration is missing")
    source = arguments.input or scientific_archive_path(ml, "BTCUSDT")
    frame = ParquetRepository.read(source)
    eth_path = scientific_archive_path(ml, "ETHUSDT")
    external = ParquetRepository.read(eth_path) if eth_path.exists() else None
    result = finalize_scientific_research(
        config, frame, arguments.run_id, external_control=external
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["accepted"] else 2


def _ml_expert_train(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    ml = config.machine_learning
    if ml is None or not ml.enabled:
        raise ValueError("machine-learning research is disabled")
    if arguments.protocol == "v6":
        result = run_v6_training(config, preregister_only=arguments.preregister_only)
        print(json.dumps(result, indent=2, default=str))
        return 0
    source = arguments.input or scientific_archive_path(ml, "BTCUSDT")
    result = run_expert_training(config, ParquetRepository.read(source), resume=arguments.resume)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "ELIGIBLE_FOR_FINAL_HOLDOUT" else 2


def _ml_expert_status(arguments: argparse.Namespace) -> int:
    path = Path(
        "data/reports/ml_expert_research_v6.status.json"
        if arguments.protocol == "v6"
        else "data/reports/ml_expert_research.status.json"
    )
    while True:
        if arguments.protocol == "v6":
            run_v6_training(load_config(arguments.config))
        payload = (
            {"phase": "not_started", "percent": 0}
            if not path.exists()
            else json.loads(path.read_text(encoding="utf-8"))
        )
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {
            "development_complete",
            "final_holdout_complete",
            "complete",
            "failed",
        }:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_expert_finalize(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    result = (
        finalize_v6_training(config, arguments.run_id)
        if arguments.protocol == "v6"
        else finalize_expert_training(config, arguments.run_id)
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_hybrid_init(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    protocol = preregister_hybrid(config)
    manifest = download_alpha_archives(config, arguments.start, arguments.end)
    print(json.dumps({"protocol": protocol, "manifest": manifest}, indent=2, default=str))
    return 0


def _ml_hybrid_train(arguments: argparse.Namespace) -> int:
    try:
        result = run_hybrid_training(load_config(arguments.config), download=arguments.download)
    except Exception as error:
        write_hybrid_failure(str(error))
        raise
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("verdict") in {"ALPHA_SHADOW_READY", "PAPER_CHALLENGER_READY"} else 2


def _ml_hybrid_status(arguments: argparse.Namespace) -> int:
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not HYBRID_STATUS_PATH.exists()
            else json.loads(HYBRID_STATUS_PATH.read_text(encoding="utf-8"))
        )
        if HYBRID_REPORT_PATH.exists():
            payload["report"] = str(HYBRID_REPORT_PATH)
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {"hybrid_complete", "failed"}:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_hybrid_v11_train(arguments: argparse.Namespace) -> int:
    result = run_v11(
        load_config(arguments.config),
        arguments.config,
        resume=arguments.resume,
        smoke=arguments.smoke,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_hybrid_v11_status(arguments: argparse.Namespace) -> int:
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not HYBRID_V11_STATUS_PATH.exists()
            else json.loads(HYBRID_V11_STATUS_PATH.read_text(encoding="utf-8"))
        )
        if HYBRID_V11_REPORT_PATH.exists():
            payload["report"] = str(HYBRID_V11_REPORT_PATH)
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {"complete", "failed"}:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_hybrid_v12_train(arguments: argparse.Namespace) -> int:
    try:
        result = run_v12(
            load_config(arguments.config),
            arguments.config,
            resume=arguments.resume,
            smoke=arguments.smoke,
        )
    except Exception as error:
        write_v12_failure(error)
        raise
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_hybrid_v12_status(arguments: argparse.Namespace) -> int:
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not HYBRID_V12_STATUS_PATH.exists()
            else json.loads(HYBRID_V12_STATUS_PATH.read_text(encoding="utf-8"))
        )
        if HYBRID_V12_REPORT_PATH.exists():
            payload["report"] = str(HYBRID_V12_REPORT_PATH)
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {"complete", "failed"}:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_hybrid_v13_train(arguments: argparse.Namespace) -> int:
    try:
        result = run_v13(
            load_config(arguments.config),
            arguments.config,
            resume=arguments.resume,
            smoke=arguments.smoke,
        )
    except Exception as error:
        write_v13_failure(error)
        raise
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_hybrid_v13_status(arguments: argparse.Namespace) -> int:
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not HYBRID_V13_STATUS_PATH.exists()
            else json.loads(HYBRID_V13_STATUS_PATH.read_text(encoding="utf-8"))
        )
        if HYBRID_V13_REPORT_PATH.exists():
            payload["report"] = str(HYBRID_V13_REPORT_PATH)
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {"complete", "failed"}:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_hybrid_v14_train(arguments: argparse.Namespace) -> int:
    try:
        result = run_v14(
            load_config(arguments.config),
            arguments.config,
            resume=arguments.resume,
            smoke=arguments.smoke,
        )
    except Exception as error:
        write_v14_failure(error)
        raise
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ml_hybrid_v14_status(arguments: argparse.Namespace) -> int:
    while True:
        payload = (
            {"phase": "not_started", "percent": 0}
            if not HYBRID_V14_STATUS_PATH.exists()
            else json.loads(HYBRID_V14_STATUS_PATH.read_text(encoding="utf-8"))
        )
        if HYBRID_V14_REPORT_PATH.exists():
            payload["report"] = str(HYBRID_V14_REPORT_PATH)
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch or payload.get("phase") in {"complete", "failed"}:
            return 2 if payload.get("phase") == "failed" else 0
        time.sleep(arguments.interval)


def _ml_hybrid_v14_freeze(arguments: argparse.Namespace) -> int:
    result = freeze_research_candidate(load_config(arguments.config), arguments.config)
    print(json.dumps(result, indent=2))
    return 0


def _ml_hybrid_v14_forward_status(arguments: argparse.Namespace) -> int:
    load_config(arguments.config)
    if not HYBRID_V14_FORWARD_LOCK_PATH.exists():
        raise RuntimeError("V14 forward protocol is not frozen")
    lock = json.loads(HYBRID_V14_FORWARD_LOCK_PATH.read_text(encoding="utf-8"))
    while True:
        payload = forward_readiness(cutoff=pd.Timestamp(lock["cutoff"]))
        print(json.dumps(payload, indent=2), flush=True)
        if not arguments.watch:
            return 0
        time.sleep(arguments.interval)


async def _paper_bitunix(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config.bitunix is None or config.broker != "simulated":
        raise ValueError("Bitunix paper mode requires the simulated futures configuration")
    await run_bitunix_paper(
        config,
        arguments.input,
        arguments.output,
        duration_hours=arguments.duration_hours,
        poll_seconds=arguments.poll_seconds,
    )
    return 0


async def _meme_collect(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    result = await collect_meme_market(
        config,
        duration_hours=arguments.duration_hours,
        api_key=os.getenv("COINGECKO_API_KEY"),
    )
    print(json.dumps(result, indent=2))
    return 0


async def _meme_download_history(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    result = await download_meme_history(
        config, weeks=arguments.weeks, api_key=os.getenv("COINGECKO_API_KEY")
    )
    print(json.dumps(result, indent=2))
    return 0


async def _meme_backtest(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    events = arguments.events or config.storage.raw_directory / "events.jsonl"
    frames = load_recorded_frames(events)
    contracts = load_cached_contracts(config.storage.raw_directory / "universe.json")
    qualities = load_market_qualities(config.storage.raw_directory / "stream.json", frames)
    if not frames or not contracts:
        raise ValueError("meme backtest requires recorded events and a cached universe")
    report = MemePaperEngine(config, contracts).run(frames, qualities)
    output = arguments.output or config.storage.report_path.with_name("backtest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


async def _meme_paper(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    await run_meme_paper(
        config,
        duration_hours=arguments.duration_hours,
        poll_seconds=arguments.poll_seconds,
    )
    return 0


def _meme_build_dataset(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    events = arguments.events or config.storage.raw_directory / "events.jsonl"
    frames = load_recorded_frames(events)
    qualities = load_market_qualities(config.storage.raw_directory / "stream.json", frames)
    dataset = build_shadow_dataset(frames, qualities, config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(arguments.output, index=False)
    print(json.dumps(shadow_status(dataset), indent=2))
    return 0


def _luna_snapshot(config: MemeBotConfig) -> dict[str, object]:
    # The sidecar receives only pre-sanitized market state, never repository files or secrets.
    storage = config.storage
    payload: dict[str, object] = {"generated_at": datetime.now(UTC).isoformat()}
    for name, path in (
        ("market", storage.raw_directory / "stream.json"),
        ("paper", storage.report_path),
    ):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload[name] = {
                key: raw.get(key)
                for key in (
                    "generated_at",
                    "symbols",
                    "universe_scan",
                    "scanner",
                    "position",
                    "net_pnl",
                    "risk",
                )
                if key in raw
            }
        except (OSError, ValueError, TypeError):
            payload[name] = None
    return payload


def _meme_actionable_symbols(report_path: Path) -> set[str]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scanner = report.get("scanner", [])
        return {
            str(item["symbol"])
            for item in scanner
            if isinstance(item, dict)
            and item.get("status") in {"ELIGIBLE", "ELIGIBLE_REDUCED"}
            and item.get("symbol")
        }
    except (OSError, ValueError, TypeError):
        return set()


async def _meme_luna_sidecar(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    if not codex_login_status(config.luna.codex_command):
        raise RuntimeError("Codex CLI is not logged in; run: codex.cmd login")
    if arguments.duration_hours <= 0 or arguments.poll_seconds <= 0:
        raise ValueError("sidecar duration and poll interval must be positive")
    store = PolicyStore(config.luna.storage_directory)
    deadline = asyncio.get_running_loop().time() + arguments.duration_hours * 3600
    refresh_max = arguments.refresh_max
    refresh_requested = False
    last_actionable = _meme_actionable_symbols(config.storage.report_path)
    limit_warning_date = None
    while True:
        now = datetime.now(UTC)
        policy = store.active_policy()
        actionable = _meme_actionable_symbols(config.storage.report_path)
        if actionable - last_actionable:
            refresh_requested = True
        last_actionable = actionable
        cooldown_ready = policy is None or now >= policy.generated_at.astimezone(UTC) + timedelta(
            minutes=config.luna.eligible_refresh_cooldown_minutes
        )
        valid = (
            False
            if policy is None or refresh_max or (refresh_requested and cooldown_ready)
            else validate_policy(policy, config, now)[0]
        )
        if not valid:
            todays_policies = sum(
                1
                for path in store.policies.glob("*.json")
                if path.name != "active.json"
                and datetime.fromtimestamp(path.stat().st_mtime, UTC).date() == now.date()
            )
            if todays_policies >= config.luna.max_runs_per_day:
                if arguments.once or refresh_max:
                    raise RuntimeError(
                        "Luna Max daily run limit reached; paper entries remain paused"
                    )
                if limit_warning_date != now.date():
                    logging.getLogger(__name__).warning(
                        "Luna Max daily run limit reached; paper entries remain paused"
                    )
                    limit_warning_date = now.date()
            else:
                policy = await asyncio.to_thread(run_luna_max, config, _luna_snapshot(config))
                logging.getLogger(__name__).info("Luna Max promoted policy %s", policy.policy_id)
                refresh_max = False
                refresh_requested = False
        for path in sorted(store.requests.glob("*.json")):
            if store.review(path.stem) is not None:
                continue
            try:
                request = LunaLowRequest.model_validate_json(path.read_text(encoding="utf-8"))
                review = await asyncio.to_thread(run_luna_low, config, request)
                logging.getLogger(__name__).info(
                    "Luna Low reviewed %s: %s", request.request_id, review.action.value
                )
            except (OSError, ValueError, RuntimeError) as error:
                logging.getLogger(__name__).error("Luna Low failed closed: %s", error)
        if arguments.once or asyncio.get_running_loop().time() >= deadline:
            return 0
        await asyncio.sleep(arguments.poll_seconds)


def _meme_luna_status(arguments: argparse.Namespace) -> int:
    config = load_meme_config(arguments.config)
    policy = PolicyStore(config.luna.storage_directory).active_policy()
    valid, reason = (
        (False, "missing_policy")
        if policy is None
        else validate_policy(policy, config, datetime.now(UTC))
    )
    print(
        json.dumps(
            {
                "codex_logged_in": codex_login_status(config.luna.codex_command),
                "policy_valid": valid,
                "reason": reason,
                "policy": None if policy is None else policy.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0 if valid else 2


async def _reconcile(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    key, secret = alpaca_credentials()
    broker = AlpacaPaperBroker(
        key,
        secret,
        config.allowed_accounts,
        config.allowed_instruments,
    )
    account = await broker.verify_account()
    state_store = SQLiteStateStore(_sqlite_path(config.database_url))
    await asyncio.to_thread(state_store.initialize)
    payload = await state_store.get("paper_position")
    local_position = None if payload in {None, "null"} else Position.model_validate_json(payload)
    kill_switch = KillSwitch()
    report = await reconcile_before_trading(
        broker,
        config.instrument.symbol,
        local_position,
        kill_switch,
        datetime.now(UTC),
    )
    if not report.reconciled and arguments.accept_broker_state:
        broker_position = report.broker_position
        if broker_position is not None:
            orders = await broker.get_open_orders(config.instrument.symbol)
            closing_side = Side.SELL if broker_position.side is Side.BUY else Side.BUY
            protected = any(
                order.protective
                and order.side is closing_side
                and order.quantity - order.filled_quantity >= broker_position.quantity
                for order in orders
            )
            if not protected:
                raise RuntimeError("cannot accept an unprotected broker position")
        await state_store.set(
            "paper_position",
            "null" if broker_position is None else broker_position.model_dump_json(),
        )
        kill_switch = KillSwitch()
        report = await reconcile_before_trading(
            broker,
            config.instrument.symbol,
            broker_position,
            kill_switch,
            datetime.now(UTC),
        )
    reset = False
    if report.reconciled and arguments.reset_kill_switch:
        if not arguments.actor or not arguments.reason:
            raise ValueError("kill-switch reset requires --actor and --reason")
        await state_store.set("paper_kill_switch", "null")
        reset = True
    print(
        json.dumps(
            {
                "account_id": account.account_id,
                "reconciled": report.reconciled,
                "reason": report.reason,
                "kill_switch": kill_switch.active,
                "kill_switch_reset": reset,
            },
            indent=2,
        )
    )
    return 0 if report.reconciled else 2


async def _shadow(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config.alpaca is None or not config.alpaca.shadow:
        raise ValueError("shadow mode must be enabled in the Alpaca configuration")
    key, secret = alpaca_credentials()
    provider = AlpacaMarketData(
        key,
        secret,
        feed=config.alpaca.feed,
        adjustment=config.alpaca.adjustment,
    )
    await run_shadow(
        config,
        provider,
        arguments.input or config.backtest.input_path,
        arguments.output,
    )
    return 0


async def _paper(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    alpaca = config.alpaca
    if alpaca is None or not alpaca.paper_execution_enabled:
        raise ValueError("Alpaca paper execution must be explicitly enabled in configuration")
    key, secret = alpaca_credentials()
    broker = AlpacaPaperBroker(
        key,
        secret,
        config.allowed_accounts,
        config.allowed_instruments,
    )
    market_data = AlpacaMarketData(
        key,
        secret,
        feed=alpaca.feed,
        adjustment=alpaca.adjustment,
    )
    runtime = PaperRuntime(
        config,
        broker,
        arguments.input or config.backtest.input_path,
        arguments.output,
        state_store=SQLiteStateStore(_sqlite_path(config.database_url)),
    )
    await run_paper(runtime, market_data, AlpacaTradeUpdates(key, secret))
    return 0


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("paper mode currently requires a SQLite DATABASE_URL")
    return Path(database_url.removeprefix(prefix))


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-data":
        frame = ParquetRepository.read(arguments.input)
        report = validate_candles(frame, timeframe_minutes=arguments.timeframe_minutes)
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.passed else 2
    if arguments.command == "download-data":
        return asyncio.run(_download(arguments))
    if arguments.command == "collect-bitunix":
        return asyncio.run(_collect_bitunix(arguments))
    if arguments.command == "collect-bitunix-microstructure":
        return asyncio.run(_collect_bitunix_microstructure(arguments))
    if arguments.command == "paper-bitunix":
        return asyncio.run(_paper_bitunix(arguments))
    if arguments.command == "meme-collect":
        return asyncio.run(_meme_collect(arguments))
    if arguments.command == "meme-download-history":
        return asyncio.run(_meme_download_history(arguments))
    if arguments.command == "meme-backtest":
        return asyncio.run(_meme_backtest(arguments))
    if arguments.command == "meme-paper":
        return asyncio.run(_meme_paper(arguments))
    if arguments.command == "meme-dashboard":
        config = load_meme_config(arguments.config)
        try:
            serve_meme_dashboard(
                config.storage.report_path,
                config.storage.raw_directory / "stream.json",
                arguments.host or config.dashboard_host,
                arguments.port or config.dashboard_port,
                arguments.allow_non_loopback,
            )
        except KeyboardInterrupt:
            return 0
        return 0
    if arguments.command == "meme-build-dataset":
        return _meme_build_dataset(arguments)
    if arguments.command == "meme-luna-sidecar":
        return asyncio.run(_meme_luna_sidecar(arguments))
    if arguments.command == "meme-luna-status":
        return _meme_luna_status(arguments)
    if arguments.command == "backtest":
        return asyncio.run(_backtest(arguments))
    if arguments.command == "research-init":
        try:
            return asyncio.run(_research_init(arguments))
        except Exception as error:
            research_config = load_config(arguments.config)
            write_research_status(research_config, "failed", detail=str(error))
            raise
    if arguments.command == "research-worker":
        return asyncio.run(_research_worker(arguments))
    if arguments.command == "ml-research":
        try:
            return _ml_research(arguments)
        except BrokenPipeError:
            return 0
        except Exception as error:
            failed = load_config(arguments.config).machine_learning
            if failed is not None:
                write_ml_status(failed, "failed", str(error), 0)
            raise
    if arguments.command == "ml-policy-research":
        return _ml_policy_research(arguments)
    if arguments.command == "ml-status":
        return _ml_status(arguments)
    if arguments.command == "ml-download-data":
        return _ml_download(arguments)
    if arguments.command == "ml-finalize":
        return _ml_finalize(arguments)
    if arguments.command == "ml-expert-train":
        try:
            return _ml_expert_train(arguments)
        except BrokenPipeError:
            return 0
        except Exception as error:
            failed = load_config(arguments.config).machine_learning
            if failed is not None:
                write_ml_status(
                    failed.model_copy(
                        update={"status_path": Path("data/reports/ml_expert_research.status.json")}
                    ),
                    "failed",
                    str(error),
                    0,
                )
            raise
    if arguments.command == "ml-expert-status":
        return _ml_expert_status(arguments)
    if arguments.command == "ml-expert-finalize":
        return _ml_expert_finalize(arguments)
    if arguments.command == "ml-hybrid-init":
        return _ml_hybrid_init(arguments)
    if arguments.command == "ml-hybrid-train":
        return _ml_hybrid_train(arguments)
    if arguments.command == "ml-hybrid-status":
        return _ml_hybrid_status(arguments)
    if arguments.command == "ml-hybrid-v11-train":
        return _ml_hybrid_v11_train(arguments)
    if arguments.command == "ml-hybrid-v11-status":
        return _ml_hybrid_v11_status(arguments)
    if arguments.command == "ml-hybrid-v12-train":
        return _ml_hybrid_v12_train(arguments)
    if arguments.command == "ml-hybrid-v12-status":
        return _ml_hybrid_v12_status(arguments)
    if arguments.command == "ml-hybrid-v13-train":
        return _ml_hybrid_v13_train(arguments)
    if arguments.command == "ml-hybrid-v13-status":
        return _ml_hybrid_v13_status(arguments)
    if arguments.command == "ml-hybrid-v14-train":
        return _ml_hybrid_v14_train(arguments)
    if arguments.command == "ml-hybrid-v14-status":
        return _ml_hybrid_v14_status(arguments)
    if arguments.command == "ml-hybrid-v14-freeze-forward":
        return _ml_hybrid_v14_freeze(arguments)
    if arguments.command == "ml-hybrid-v14-forward-status":
        return _ml_hybrid_v14_forward_status(arguments)
    if arguments.command == "musca-btc-policy-train":
        from adaptive_bot import musca_btc_policy

        try:
            report = musca_btc_policy.train(resume=arguments.resume)
        except BrokenPipeError:
            return 0
        except Exception as error:
            musca_btc_policy.failed_status(error)
            raise
        print(json.dumps(report, indent=2, default=str))
        return 0
    if arguments.command == "musca-btc-policy-status":
        return _musca_btc_policy_status(arguments)
    if arguments.command == "research-status":
        return _research_status(arguments)
    if arguments.command == "research-pin":
        return _research_pin(arguments)
    if arguments.command == "dashboard":
        try:
            serve_dashboard(
                arguments.report,
                arguments.host,
                arguments.port,
                arguments.live_data,
                arguments.research_report,
                arguments.ml_report,
            )
        except KeyboardInterrupt:
            return 0
        return 0
    if arguments.command == "reconcile":
        return asyncio.run(_reconcile(arguments))
    if arguments.command == "shadow":
        return asyncio.run(_shadow(arguments))
    if arguments.command == "paper":
        return asyncio.run(_paper(arguments))
    if arguments.command == "live":
        acknowledged = (
            os.getenv("TRADING_MODE") == "live"
            and os.getenv("ALLOW_LIVE_TRADING") == "I_ACKNOWLEDGE_THE_RISK"
        )
        message = (
            "live adapter is unavailable"
            if acknowledged
            else "live trading requires mode=live and the explicit risk acknowledgement"
        )
        logging.getLogger(__name__).critical(message)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
