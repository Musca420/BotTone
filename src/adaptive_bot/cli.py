from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from adaptive_bot.adapters.alpaca.broker import AlpacaPaperBroker
from adaptive_bot.adapters.alpaca.market_data import AlpacaMarketData
from adaptive_bot.adapters.alpaca.trade_updates import AlpacaTradeUpdates
from adaptive_bot.adapters.bitunix.collector import collect_futures_candles
from adaptive_bot.adapters.bitunix.market_data import BitunixMarketData
from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import alpaca_credentials, load_config
from adaptive_bot.dashboard.meme_server import serve_meme_dashboard
from adaptive_bot.dashboard.server import serve_dashboard
from adaptive_bot.data.interfaces import MarketDataProvider
from adaptive_bot.data.repository import ParquetRepository, SQLiteStateStore
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Position
from adaptive_bot.meme.collector import collect_meme_market
from adaptive_bot.meme.config import load_meme_config
from adaptive_bot.meme.research import build_shadow_dataset, shadow_status
from adaptive_bot.meme.runtime import (
    MemePaperEngine,
    load_cached_contracts,
    load_market_qualities,
    load_recorded_frames,
    run_meme_paper,
)
from adaptive_bot.risk.kill_switch import KillSwitch
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
        "--output", type=Path, default=Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
    )
    collect.add_argument("--duration-hours", type=float, default=168)
    collect.add_argument("--poll-seconds", type=float, default=60)

    bitunix_paper = commands.add_parser("paper-bitunix")
    bitunix_paper.add_argument("--config", type=Path, required=True)
    bitunix_paper.add_argument(
        "--input", type=Path, default=Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
    )
    bitunix_paper.add_argument(
        "--output", type=Path, default=Path("data/reports/bitunix_paper.json")
    )
    bitunix_paper.add_argument("--duration-hours", type=float, default=168)
    bitunix_paper.add_argument("--poll-seconds", type=float, default=15)

    meme_collect = commands.add_parser("meme-collect")
    meme_collect.add_argument("--config", type=Path, required=True)
    meme_collect.add_argument("--duration-hours", type=float, default=168)

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

    meme_dataset = commands.add_parser("meme-build-dataset")
    meme_dataset.add_argument("--config", type=Path, required=True)
    meme_dataset.add_argument("--events", type=Path)
    meme_dataset.add_argument(
        "--output", type=Path, default=Path("data/meme/processed/shadow_features.parquet")
    )

    backtest = commands.add_parser("backtest")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.add_argument("--input", type=Path)
    backtest.add_argument("--output", type=Path, default=Path("data/reports/backtest.json"))
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--report", type=Path, default=Path("data/reports/backtest.json"))
    dashboard.add_argument(
        "--live-data", type=Path, default=Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8080)
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


async def _collect_bitunix(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if config.bitunix is None or config.bitunix.market != "futures":
        raise ValueError("collector requires the Bitunix futures configuration")
    count = await collect_futures_candles(
        arguments.output,
        duration_hours=arguments.duration_hours,
        poll_seconds=arguments.poll_seconds,
        timeframe_minutes=config.strategy.timeframe_minutes,
    )
    print(json.dumps({"collected": count, "output": str(arguments.output)}, indent=2))
    return 0


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
    if arguments.command == "paper-bitunix":
        return asyncio.run(_paper_bitunix(arguments))
    if arguments.command == "meme-collect":
        return asyncio.run(_meme_collect(arguments))
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
                config.dashboard_host,
                config.dashboard_port,
            )
        except KeyboardInterrupt:
            return 0
        return 0
    if arguments.command == "meme-build-dataset":
        return _meme_build_dataset(arguments)
    if arguments.command == "backtest":
        return asyncio.run(_backtest(arguments))
    if arguments.command == "dashboard":
        try:
            serve_dashboard(
                arguments.report,
                arguments.host,
                arguments.port,
                arguments.live_data,
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
