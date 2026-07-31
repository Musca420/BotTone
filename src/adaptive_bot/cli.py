from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import load_config
from adaptive_bot.dashboard.server import serve_dashboard
from adaptive_bot.data.repository import ParquetRepository
from adaptive_bot.data.validation import validate_candles


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

    backtest = commands.add_parser("backtest")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.add_argument("--input", type=Path)
    backtest.add_argument("--output", type=Path, default=Path("data/reports/backtest.json"))
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--report", type=Path, default=Path("data/reports/backtest.json"))
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8080)
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


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-data":
        frame = ParquetRepository.read(arguments.input)
        report = validate_candles(frame, timeframe_minutes=arguments.timeframe_minutes)
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.passed else 2
    if arguments.command == "backtest":
        return asyncio.run(_backtest(arguments))
    if arguments.command == "dashboard":
        try:
            serve_dashboard(arguments.report, arguments.host, arguments.port)
        except KeyboardInterrupt:
            return 0
        return 0
    if arguments.command == "live":
        acknowledged = (
            os.getenv("TRADING_MODE") == "live"
            and os.getenv("ALLOW_LIVE_TRADING") == "I_ACKNOWLEDGE_THE_RISK"
        )
        message = (
            "live adapter is unavailable in Milestone 1"
            if acknowledged
            else "live trading requires mode=live and the explicit risk acknowledgement"
        )
        logging.getLogger(__name__).critical(message)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
