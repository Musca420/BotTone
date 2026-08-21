from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_bot import hybrid_policy_v11 as engine
from adaptive_bot.config import AppConfig

PROTOCOL = "hybrid_v12_btc_net_admission"
ROOT = Path("data/ml/hybrid_v12")
MODEL_ROOT = Path("data/models/expert_policy/v12")
REPORT_PATH = Path("data/reports/ml_hybrid_v12.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v12.status.json")


def configure_v12() -> None:
    engine.PROTOCOL = PROTOCOL
    engine.ROOT = ROOT
    engine.MATRIX_ROOT = engine.ROOT / "base_matrix"
    engine.OOS_ROOT = engine.ROOT / "oos_folds"
    engine.MODEL_ROOT = MODEL_ROOT
    engine.PROTOCOL_PATH = engine.MODEL_ROOT / "protocol.json"
    engine.BUNDLE_PATH = engine.MODEL_ROOT / "bundle.joblib"
    engine.REPORT_PATH = REPORT_PATH
    engine.STATUS_PATH = STATUS_PATH
    engine.VARIANT_SOURCE_PATHS = (Path(__file__),)


def run_v12(
    app: AppConfig, config_path: Path, *, resume: bool = False, smoke: bool = False
) -> dict[str, Any]:
    configure_v12()
    return engine.run_v11(app, config_path, resume=resume, smoke=smoke)


def write_v12_failure(error: Exception) -> None:
    configure_v12()
    engine._status("failed", f"{type(error).__name__}: {error}", 0, block="FAILED")
