from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_bot import hybrid_policy_v11 as engine
from adaptive_bot.config import AppConfig

PROTOCOL = "hybrid_v13_btc_side_admission"
ROOT = Path("data/ml/hybrid_v13")
MODEL_ROOT = Path("data/models/expert_policy/v13")
REPORT_PATH = Path("data/reports/ml_hybrid_v13.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v13.status.json")


def configure_v13() -> None:
    engine.PROTOCOL = PROTOCOL
    engine.ROOT = ROOT
    engine.MATRIX_ROOT = ROOT / "base_matrix"
    engine.OOS_ROOT = ROOT / "oos_folds"
    engine.MODEL_ROOT = MODEL_ROOT
    engine.PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
    engine.BUNDLE_PATH = MODEL_ROOT / "bundle.joblib"
    engine.REPORT_PATH = REPORT_PATH
    engine.STATUS_PATH = STATUS_PATH
    engine.VARIANT_SOURCE_PATHS = (Path(__file__),)


def run_v13(
    app: AppConfig, config_path: Path, *, resume: bool = False, smoke: bool = False
) -> dict[str, Any]:
    configure_v13()
    return engine.run_v11(app, config_path, resume=resume, smoke=smoke)


def write_v13_failure(error: Exception) -> None:
    configure_v13()
    engine._status("failed", f"{type(error).__name__}: {error}", 0, block="FAILED")
