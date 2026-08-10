from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_bot.hybrid_policy_v22.protocol import sha256

PROTOCOL_NAME = "hybrid_v25_trend_vwap_pullback_meta"
ROOT = Path("data/ml/hybrid_v25")
REPORT_ROOT = Path("data/reports")
MODEL_ROOT = Path("data/models/expert_policy/v25")
BUNDLE_ROOT = Path("data/models/expert_policy/v25_research_bundle")
SPEC_PATH = Path("C:/Users/david/Downloads/V25_VWAP_DEFINITIVE_BOT_SPEC.md")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
STATUS_PATH = REPORT_ROOT / "ml_hybrid_v25.status.json"
ASSETS = ("BTCUSDT", "ETHUSDT")
BASE_COST_BPS = 4.0
STRESS_COST_BPS = 8.0
HOLDOUT_WEEKS = 12
RANDOM_SEED = 20260805

FEATURE_COLUMNS = (
    "asset_code",
    "direction",
    "regime_score",
    "regime_score_opposite",
    "adx_1h",
    "ema_slope_atr",
    "weekly_vwap_slope_atr",
    "volatility_percentile",
    "funding_z",
    "basis_bps",
    "basis_change_bps",
    "return_15m_atr",
    "return_30m_atr",
    "return_60m_atr",
    "donchian_break_atr",
    "impulse_range_atr",
    "relative_volume",
    "trade_intensity",
    "taker_imbalance_15m",
    "spot_taker_imbalance_15m",
    "spot_perp_return_divergence",
    "impulse_efficiency",
    "pullback_depth_atr",
    "pullback_depth_pct_impulse",
    "pullback_duration_minutes",
    "pullback_volume_ratio",
    "pullback_taker_ratio",
    "distance_daily_vwap_atr",
    "distance_weekly_vwap_atr",
    "distance_anchored_vwap_atr",
    "distance_swing_vwap_atr",
    "vwap_zone_touched",
    "vwap_zone_crossed",
    "number_of_touches",
    "pullback_efficiency",
    "flow_persistence",
    "flow_price_divergence",
    "spot_confirmation",
    "stop_distance_bps",
    "stop_distance_atr",
    "cost_to_stop_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def payload() -> dict[str, Any]:
    immutable: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "spec_sha256": sha256(SPEC_PATH),
        "assets": list(ASSETS),
        "strategy": "TREND_VWAP_PULLBACK_META",
        "schema_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "policy_variants": [
            {"id": "base", "regime_threshold": 0.60, "flow_gate": False, "tp1_r": 1.5},
            {"id": "flow", "regime_threshold": 0.60, "flow_gate": True, "tp1_r": 1.0},
        ],
        "stop": {
            "buffer_atr": 0.10,
            "minimum_bps": 12.0,
            "maximum_atr": 2.5,
            "maximum_cost_to_stop": 0.33,
        },
        "outcome": {"timeout_minutes": 240, "partial_tp1": 0.50, "trailing": "swing_15m"},
        "cost_bps": {"base": BASE_COST_BPS, "stress": STRESS_COST_BPS},
        "walk_forward_weeks": {"train": 52, "calibration": 8, "test": 8, "step": 8},
        "inner_splits": 3,
        "embargo_hours": 6,
        "holdout": {"asset": "BTCUSDT", "weeks": HOLDOUT_WEEKS, "opened": False},
        "challenger": {
            "model": "XGBoost",
            "device": "cuda",
            "maximum_configurations": 8,
            "depth": [4, 6],
            "learning_rate": [0.03, 0.07],
            "min_child_weight": [3, 10],
        },
        "ranking": {
            "minimum_expected_net_bps": 2.0,
            "minimum_q25_bps": -2.0,
            "maximum_cost_to_stop": 0.33,
            "maximum_trade_per_asset_hours": 4,
            "maximum_new_positions_per_day": 2,
        },
        "allowed_modes": ["research", "paper", "shadow"],
        "real_capital_allowed": False,
        "auto_promotion": False,
        "random_seed": RANDOM_SEED,
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_hash": hashlib.sha256(canonical.encode()).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def implementation_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def freeze(*, implementation_paths: tuple[Path, ...]) -> dict[str, Any]:
    current = payload()
    implementation_hash = implementation_sha256(implementation_paths)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_hash"] != current["protocol_hash"]:
            raise RuntimeError("V25 protocol changed after preregistration")
        if (
            existing.get("implementation_finalized")
            and existing["implementation_sha256"] != implementation_hash
        ):
            raise RuntimeError("V25 implementation changed after finalization")
        return dict(existing)
    frozen = current | {
        "implementation_sha256": implementation_hash,
        "implementation_finalized": False,
        "registered_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(PROTOCOL_PATH, frozen)
    return frozen


def finalize(protocol: dict[str, Any], *, implementation_paths: tuple[Path, ...]) -> dict[str, Any]:
    finalized = protocol | {
        "implementation_sha256": implementation_sha256(implementation_paths),
        "implementation_finalized": True,
        "finalized_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(PROTOCOL_PATH, finalized)
    return finalized


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
