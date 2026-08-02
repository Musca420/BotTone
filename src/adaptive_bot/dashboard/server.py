from __future__ import annotations

import ipaddress
import json
import math
from datetime import UTC, datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.indicators.adx import adx
from adaptive_bot.indicators.atr import atr
from adaptive_bot.indicators.slope import normalized_ema_slope
from adaptive_bot.indicators.volatility import normalized_distance
from adaptive_bot.indicators.vwap import rolling_vwap

MILESTONES = (
    {"name": "Milestone 1", "label": "Simulation core", "status": "complete"},
    {"name": "Milestone 2", "label": "Alpaca Paper", "status": "complete"},
    {"name": "Milestone 3", "label": "Research & stress", "status": "next"},
    {"name": "Milestone 4", "label": "Bitunix BTCUSDT Futures", "status": "planned"},
    {"name": "Milestone 5", "label": "IBKR Paper", "status": "planned"},
)
LIVE_DATA_PATH = Path("data/raw/bitunix_btcusdt_mark_futures_5m.jsonl")
LIVE_WARMUP_BARS = 288


def build_live_market_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"available": False, "error": "Waiting for the Bitunix collector."}
    rows: dict[int, dict[str, Any]] = {}
    invalid_rows = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            envelope = json.loads(line)
            candle = envelope["candle"]
            timestamp_ms = int(candle["time"])
            values = {name: float(candle[name]) for name in ("open", "high", "low", "close")}
            volume = float(candle.get("baseVol", candle.get("volume", 0)))
            if min(*values.values()) <= 0 or volume < 0:
                raise ValueError("invalid OHLCV")
            envelope_high = max(values["open"], values["high"], values["close"])
            envelope_low = min(values["open"], values["low"], values["close"])
            deviation_bps = (
                ((envelope_high - values["high"]) + (values["low"] - envelope_low))
                / values["close"]
                * 10_000
            )
            if (
                not all(math.isfinite(value) for value in (*values.values(), volume))
                or min(*values.values(), volume) < 0
                or values["high"] < values["low"]
                or deviation_bps > 1
            ):
                raise ValueError("invalid OHLCV")
            if envelope_high != values["high"] or envelope_low != values["low"]:
                invalid_rows += 1
                values["high"] = envelope_high
                values["low"] = envelope_low
            rows[timestamp_ms] = {
                "timestamp": datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat(),
                **values,
                "volume": volume,
                "collected_at": envelope["collected_at"],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid_rows += 1
    candles = [rows[key] for key in sorted(rows)]
    if not candles:
        return {"available": False, "error": "No valid closed Bitunix candles yet."}
    frame = pd.DataFrame(candles)
    atr_values = atr(frame["high"], frame["low"], frame["close"])
    center = rolling_vwap(
        frame["high"], frame["low"], frame["close"], frame["volume"], LIVE_WARMUP_BARS
    )
    adx_values = adx(frame["high"], frame["low"], frame["close"])["adx"]
    slope = normalized_ema_slope(frame["close"], atr_values)
    z_score = normalized_distance(frame["close"], center, atr_values)
    for index, candle in enumerate(candles):
        center_value = _finite_or_none(center.iloc[index])
        atr_value = _finite_or_none(atr_values.iloc[index])
        candle["center"] = center_value
        candle["lower_band"] = (
            None if center_value is None or atr_value is None else center_value - 2 * atr_value
        )
        candle["upper_band"] = (
            None if center_value is None or atr_value is None else center_value + 2 * atr_value
        )
    latest = candles[-1] | {
        "atr": _finite_or_none(atr_values.iloc[-1]),
        "adx": _finite_or_none(adx_values.iloc[-1]),
        "ema_slope": _finite_or_none(slope.iloc[-1]),
        "z_score": _finite_or_none(z_score.iloc[-1]),
        "atr_percentile": None,
        "spread_bps": None,
        "regime": "unknown",
    }
    quote = _live_quote(source.with_suffix(".quote.json"))
    latest["spread_bps"] = quote.get("spread_bps")
    collected_at = datetime.fromisoformat(latest["collected_at"])
    age_seconds = max(0, (datetime.now(UTC) - collected_at).total_seconds())
    ready = len(candles) >= LIVE_WARMUP_BARS and latest["center"] is not None
    return {
        "available": True,
        "status": "live" if age_seconds <= 420 else "stale",
        "age_seconds": round(age_seconds),
        "bars": len(candles),
        "warmup_bars": LIVE_WARMUP_BARS,
        "ready": ready,
        "invalid_rows": invalid_rows,
        "activity": (
            "Indicators ready; paper decisions use closed candles and the observed spread."
            if ready
            else f"Collecting strategy warm-up data: {len(candles)}/{LIVE_WARMUP_BARS} closed bars."
        ),
        "latest": latest,
        "candles": candles[-160:],
        "quote": quote,
    }


def _finite_or_none(value: Any) -> float | None:
    return float(value) if pd.notna(value) else None


def _live_quote(path: Path) -> dict[str, Any]:
    try:
        quote = json.loads(path.read_text(encoding="utf-8"))
        bid = float(quote["best_bid"])
        ask = float(quote["best_ask"])
        spread = float(quote["spread_bps"])
        observed_at = datetime.fromisoformat(quote["observed_at"])
        if not (0 < bid < ask and spread >= 0 and observed_at.tzinfo is not None):
            raise ValueError("invalid quote")
        return {
            "available": True,
            "best_bid": bid,
            "best_ask": ask,
            "spread_bps": spread,
            "observed_at": observed_at.astimezone(UTC).isoformat(),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"available": False}


def build_dashboard_payload(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {
            "available": False,
            "error": f"No report found at {path}",
            "generated_at": datetime.now(UTC).isoformat(),
            "milestones": MILESTONES,
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "error": f"Report cannot be read: {error}",
            "generated_at": datetime.now(UTC).isoformat(),
            "milestones": MILESTONES,
        }
    telemetry = report.get("telemetry", [])
    equity_curve = report.get("equity_curve", [])
    fills = report.get("fills", [])
    latest = telemetry[-1] if telemetry else None
    operations, position = _operations(fills)
    return {
        "available": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "report_updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "summary": {
            "mode": report.get("mode", "backtest"),
            "instrument": report.get("instrument", "—"),
            "timeframe_minutes": report.get("timeframe_minutes"),
            "initial_equity": report.get("initial_equity", "0"),
            "final_equity": report.get("final_equity", "0"),
            "net_pnl": report.get("net_pnl", "0"),
            "gross_pnl": report.get("gross_pnl", "0"),
            "max_drawdown": report.get("max_drawdown", "0"),
            "fees": report.get("fees", "0"),
            "slippage": report.get("slippage", "0"),
            "signals": report.get("signals", 0),
            "rejected_signals": report.get("rejected_signals", 0),
            "kill_switches": report.get("kill_switches", 0),
            "risk_per_trade": report.get("risk_per_trade", "0.01"),
            "max_daily_loss": report.get("max_daily_loss", "0.02"),
            "max_weekly_loss": report.get("max_weekly_loss", "0.10"),
            "fills": len(fills),
            "operations": len(operations),
        },
        "latest": latest,
        "current_position": position,
        "no_trade_reason": (
            None
            if operations
            else f"No order was filled. Latest decision: {(latest or {}).get('activity', 'none')}"
        ),
        "equity_curve": _sample(equity_curve, 500),
        "telemetry": telemetry[-160:],
        "fills": fills[-50:][::-1],
        "operations": operations[-50:][::-1],
        "milestones": MILESTONES,
        "safety": {
            "live_enabled": False,
            "risk_per_trade": 0.01,
            "max_drawdown_limit": 0.08,
            "max_daily_loss": 0.02,
            "max_weekly_loss": 0.10,
            "max_open_positions": 1,
        },
    }


def _operations(fills: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    signed_position = Decimal("0")
    operations: list[dict[str, Any]] = []
    for fill in fills:
        quantity = Decimal(str(fill["quantity"]))
        before = signed_position
        signed_position += quantity if fill["side"] == "buy" else -quantity
        if before == 0:
            event = "OPEN LONG" if signed_position > 0 else "OPEN SHORT"
        elif signed_position == 0:
            event = "CLOSE"
        elif before * signed_position < 0:
            event = "REVERSE"
        elif abs(signed_position) < abs(before):
            event = "REDUCE"
        else:
            event = "INCREASE"
        operations.append(
            {
                "timestamp": fill["exchange_timestamp"],
                "event": event,
                "side": fill["side"],
                "quantity": str(quantity),
                "price": fill["price"],
                "position_after": str(signed_position),
                "details": (
                    f"Fee {fill.get('commission', '0')} · slippage {fill.get('slippage', '0')} · "
                    f"{fill['client_order_id']}"
                ),
            }
        )
    side = "LONG" if signed_position > 0 else "SHORT" if signed_position < 0 else "FLAT"
    return operations, {"status": side, "quantity": str(abs(signed_position))}


def _sample(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return values
    step = max(1, len(values) // maximum)
    sampled = values[::step]
    return sampled if sampled[-1] is values[-1] else [*sampled, values[-1]]


def serve_dashboard(
    report_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    live_data_path: str | Path = LIVE_DATA_PATH,
) -> None:
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("dashboard host must be loopback-only")
    handler = _handler(Path(report_path), Path(live_data_path))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Dashboard available at http://{host}:{port}")
        server.serve_forever()


def _handler(report_path: Path, live_data_path: Path) -> type[BaseHTTPRequestHandler]:
    assets = files("adaptive_bot.dashboard.static")
    routes = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/dashboard":
                self._send_json(build_dashboard_payload(report_path))
                return
            if path == "/api/live":
                self._send_json(build_live_market_payload(live_data_path))
                return
            if path == "/api/health":
                self._send_json({"status": "healthy", "timestamp": datetime.now(UTC).isoformat()})
                return
            asset = routes.get(path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            name, content_type = asset
            self._send(assets.joinpath(name).read_bytes(), content_type)

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send(json.dumps(payload, separators=(",", ":")).encode(), "application/json")

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self' wss://fapi.bitunix.com; "
                "img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return DashboardHandler
