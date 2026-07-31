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
from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import alpaca_credentials, load_config
from adaptive_bot.dashboard.server import serve_dashboard
from adaptive_bot.data.repository import ParquetRepository, SQLiteStateStore
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Position
from adaptive_bot.risk.kill_switch import KillSwitch
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

    backtest = commands.add_parser("backtest")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.add_argument("--input", type=Path)
    backtest.add_argument("--output", type=Path, default=Path("data/reports/backtest.json"))
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--report", type=Path, default=Path("data/reports/backtest.json"))
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
    if config.alpaca is None:
        raise ValueError("Alpaca configuration is required")
    key, secret = alpaca_credentials()
    end = arguments.end or datetime.now(UTC)
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
    )
    print(json.dumps(asdict(report), indent=2))
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
    if arguments.command == "backtest":
        return asyncio.run(_backtest(arguments))
    if arguments.command == "dashboard":
        try:
            serve_dashboard(arguments.report, arguments.host, arguments.port)
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
