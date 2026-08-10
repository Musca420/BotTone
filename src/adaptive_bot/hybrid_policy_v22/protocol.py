from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL = "hybrid_v22_hierarchical_path_vwap"
ROOT = Path("data/ml/hybrid_v22")
MODEL_ROOT = Path("data/models/expert_policy/v22")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
STATUS_PATH = Path("data/reports/ml_hybrid_v22.status.json")
REPORT_PATH = Path("data/reports/ml_hybrid_v22_audit.json")
V21_REPORT = Path("data/reports/ml_hybrid_v21.json")
V21_PROTOCOL = Path("data/models/expert_policy/v21/protocol.json")
BASE_COST_BPS = 4.0
STRESS_COST_BPS = 8.0
HORIZONS_MINUTES = (5, 15, 30, 60)
ENTRY_MODES = ("immediate", "confirmed_5m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def status(phase: str, detail: str, percent: float, **extra: Any) -> None:
    atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
            **extra,
        },
    )


def exit_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for entry in ENTRY_MODES:
        for stop_buffer in (0.5, 0.75, 1.0):
            for target_name in ("half_distance", "inner_band", "vwap"):
                for timeout in (15, 30, 60):
                    grid.append(
                        {
                            "family": "fade",
                            "entry_mode": entry,
                            "stop_kind": "beyond_extreme",
                            "stop_value": stop_buffer,
                            "target_kind": target_name,
                            "target_value": None,
                            "timeout_minutes": timeout,
                        }
                    )
        stops: tuple[tuple[str, float | None], ...] = (
            ("inside_band", None),
            ("atr", 0.5),
            ("atr", 0.75),
        )
        for stop_kind, stop_value in stops:
            for target_atr in (0.5, 1.0, 1.5):
                for timeout in (15, 30, 60):
                    grid.append(
                        {
                            "family": "follow",
                            "entry_mode": entry,
                            "stop_kind": stop_kind,
                            "stop_value": stop_value,
                            "target_kind": "atr",
                            "target_value": target_atr,
                            "timeout_minutes": timeout,
                        }
                    )
    for number, config in enumerate(grid):
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        config["config_id"] = (
            f"v22-{number:03d}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
        )
    return grid


def payload() -> dict[str, Any]:
    package = Path(__file__).parent
    source_hashes = {path.name: sha256(path) for path in sorted(package.glob("*.py"))}
    immutable: dict[str, Any] = {
        "protocol": PROTOCOL,
        "symbol": "BTCUSDT perpetual",
        "primary_exchange": "binance",
        "path_granularity": "1 minute fallback; no tick/second archive available",
        "events_source": "frozen V21 event states",
        "horizons_minutes": list(HORIZONS_MINUTES),
        "base_cost_bps_round_trip": BASE_COST_BPS,
        "stress_cost_bps_round_trip": STRESS_COST_BPS,
        "exit_grid": exit_grid(),
        "walk_forward_weeks": [52, 4, 4],
        "v22a_gates": {
            "expectancy_4bps": ">0",
            "expectancy_8bps": ">=0",
            "profit_factor": ">=1.10",
            "oos_trades": ">=100",
            "positive_fold_fraction": ">0.50",
            "dominant_month_share": "<=0.50",
        },
        "v21_report_sha256": sha256(V21_REPORT),
        "v21_protocol_sha256": sha256(V21_PROTOCOL),
        "source_hashes": source_hashes,
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    current = payload()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != current["protocol_sha256"]:
            raise RuntimeError("V22 protocol changed after freezing")
        return dict(existing)
    registered = current | {"registered_at": datetime.now(UTC).isoformat()}
    atomic_json(PROTOCOL_PATH, registered)
    return registered
