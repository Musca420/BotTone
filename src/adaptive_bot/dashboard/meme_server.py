from __future__ import annotations

import ipaddress
import json
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any


def build_meme_dashboard_payload(
    report_path: str | Path, stream_path: str | Path
) -> dict[str, Any]:
    report = _read_json(Path(report_path))
    stream = _read_json(Path(stream_path))
    if report is None and stream is None:
        return {
            "available": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "error": "Waiting for the meme collector and paper engine.",
        }
    return {
        "available": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "stream": stream or {"connected": False, "symbols": {}},
        "report": report
        or {
            "mode": "paper",
            "operations": [],
            "audit": [],
            "scanner": [],
            "equity_curve": [],
            "probabilistic": {"status": "collecting_data", "can_trade": False},
        },
        "safety": {
            "live_enabled": False,
            "execution": "simulated",
            "margin_mode": "USDT isolated",
        },
    }


def serve_meme_dashboard(
    report_path: str | Path,
    stream_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8081,
    allow_non_loopback: bool = False,
) -> None:
    if (
        not allow_non_loopback
        and host != "localhost"
        and not ipaddress.ip_address(host).is_loopback
    ):
        raise ValueError("meme dashboard host must be loopback-only")
    handler = _handler(Path(report_path), Path(stream_path))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"Meme dashboard available at http://{host}:{port}")
        server.serve_forever()


def _handler(report_path: Path, stream_path: Path) -> type[BaseHTTPRequestHandler]:
    assets = files("adaptive_bot.dashboard.meme_static")
    routes = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/meme.css": ("meme.css", "text/css; charset=utf-8"),
        "/meme.js": ("meme.js", "text/javascript; charset=utf-8"),
    }

    class MemeDashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/meme":
                self._send_json(build_meme_dashboard_payload(report_path, stream_path))
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
                "img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return MemeDashboardHandler


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
