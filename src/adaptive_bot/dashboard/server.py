from __future__ import annotations

import ipaddress
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

MILESTONES = (
    {"name": "Milestone 1", "label": "Simulation core", "status": "complete"},
    {"name": "Milestone 2", "label": "Alpaca Paper", "status": "next"},
    {"name": "Milestone 3", "label": "Research & stress", "status": "planned"},
    {"name": "Milestone 4", "label": "OKX Demo", "status": "planned"},
    {"name": "Milestone 5", "label": "IBKR Paper", "status": "planned"},
)


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
            "risk_per_trade": report.get("risk_per_trade", "0.0025"),
            "fills": len(fills),
        },
        "latest": latest,
        "equity_curve": _sample(equity_curve, 500),
        "telemetry": telemetry[-160:],
        "fills": fills[-50:][::-1],
        "milestones": MILESTONES,
        "safety": {
            "live_enabled": False,
            "max_drawdown_limit": 0.08,
            "max_daily_loss": 0.01,
            "max_open_positions": 1,
        },
    }


def _sample(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return values
    step = max(1, len(values) // maximum)
    sampled = values[::step]
    return sampled if sampled[-1] is values[-1] else [*sampled, values[-1]]


def serve_dashboard(report_path: str | Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("dashboard host must be loopback-only")
    handler = _handler(Path(report_path))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Dashboard available at http://{host}:{port}")
        server.serve_forever()


def _handler(report_path: Path) -> type[BaseHTTPRequestHandler]:
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
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return DashboardHandler
