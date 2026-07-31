from __future__ import annotations

import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from adaptive_bot.domain.enums import HealthLevel
from adaptive_bot.domain.models import HealthStatus


class HealthRegistry:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {"bot_alive": True}

    def set(self, name: str, healthy: bool) -> None:
        self.checks[name] = healthy

    def snapshot(self) -> HealthStatus:
        return HealthStatus(
            timestamp=datetime.now(UTC),
            level=HealthLevel.HEALTHY if all(self.checks.values()) else HealthLevel.UNHEALTHY,
            checks=dict(self.checks),
        )


def serve_health(registry: HealthRegistry, host: str = "127.0.0.1", port: int = 8080) -> Thread:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            payload = registry.snapshot().model_dump(mode="json")
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer((host, port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
