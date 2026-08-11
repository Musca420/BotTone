from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import subprocess
import time
import zipfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol, cast

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from arch.bootstrap import SPA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from adaptive_bot import musca_btc_auto_moe as discovery
from adaptive_bot import musca_btc_moe as base
from adaptive_bot import musca_v5_microstructure as microstructure
from adaptive_bot.musca_v8_binance import fee_schedule, load_config

SYMBOL = "BTCUSDT"
VENUE = "Binance USD-M futures"
ROOT = Path("data/ml/musca_btc_policy")
LABEL_ROOT_BASE = ROOT / "state_actions"
LABEL_ROOT = LABEL_ROOT_BASE
ORDERED_EVENT_ROOT = ROOT / "ordered_events"
REGISTRY = ROOT / "research_registry.json"
EXPERIMENT_LEDGER = ROOT / "experiment_ledger.json"
REPORT = Path("data/reports/musca_btc_policy.json")
STATUS = Path("data/reports/musca_btc_policy.status.json")
PREFLIGHT_REPORT = Path("data/reports/musca_btc_policy.preflight.json")
BUNDLE = Path("data/models/musca_btc_policy/research_bundle.joblib")
AUDIT_TRADES = ROOT / "audit_trades.parquet"
AUDIT_DECISIONS = ROOT / "audit_decisions.parquet"
PREFLIGHT_TRADES = ROOT / "preflight_trades.parquet"
PREFLIGHT_DECISIONS = ROOT / "preflight_decisions.parquet"
EXECUTION_REPORT = Path("data/reports/musca_btc_execution_contract.json")
CRITIC_CROSSFIT_REPORT = Path("data/reports/musca_btc_policy_critic_crossfit.json")
EQUITY_OBJECTIVE_REPORT = Path("data/reports/musca_btc_policy_equity_objective.json")
WAIT_VALUE_REPORT = Path("data/reports/musca_btc_policy_wait_value.json")
PLAN_EFFICIENCY_REPORT = Path("data/reports/musca_btc_policy_plan_efficiency.json")
VALUE_HEADS_REPORT = Path("data/reports/musca_btc_policy_value_heads.json")
ENTRY_STABILITY_REPORT = Path("data/reports/musca_btc_policy_entry_stability.json")
INTRATRADE_REPORT = Path("data/reports/musca_btc_policy_intratrade.json")
VIEW_AUDIT_REPORT = Path("data/reports/musca_btc_policy_view_audit.json")
BINANCE_L2_ROOT = Path("data/research/binance_l2")
CONFIG = Path("configs/binance_btcusdt_paper.yaml")
AUTO_MOE_REPORT = Path("data/reports/musca_btc_auto_moe.json")
AUTO_MOE_CANDIDATES = Path("data/ml/musca_btc_auto_moe/candidates.parquet")
PARENT_ACTIONS = (
    base.CHECKPOINTS / "oof_actions.parquet",
    base.CHECKPOINTS / "future_actions.parquet",
)

HISTORICAL_START = pd.Timestamp("2025-04-01T00:00:00Z")
HISTORICAL_END = pd.Timestamp("2026-08-01T00:00:00Z")
FUTURE_HOLDOUT_START = pd.Timestamp("2026-08-10T00:00:00Z")
ONE_SECOND_MONTHS = tuple(str(value) for value in pd.period_range("2025-04", "2026-07", freq="M"))
MAXIMUM_HORIZON_SECONDS = max(base.HORIZONS)
PREDICTION_HORIZONS_SECONDS = tuple(int(value) for value in base.HORIZONS)
OUTCOME_TARGET = 0
OUTCOME_STOP = 1
OUTCOME_TIMEOUT = 2
OUTCOME_NAMES = ("TARGET", "STOP", "TIMEOUT")
MODEL_SEEDS = (20260831, 20260901, 20260902)
VALUE_HEADS = ("decomposed", "direct")
THRESHOLDS_BPS = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
MINIMUM_FIT_WEEKS = 16
WINDOW_WEEKS = 4
RISK_PER_TRADE = 0.01
MAXIMUM_LEVERAGE = 10.0
MAXIMUM_DAILY_LOSS = 0.02
MAXIMUM_DRAWDOWN = 0.08
EXECUTION_RESERVE_ROUND_TRIP_BPS = 0.0
GPU_CPU_TOLERANCE_BPS = 1e-6
GLOBAL_PROTOCOL_FLOOR = 53
GLOBAL_EXPERT_FLOOR = 7_691
INNER_CALIBRATION_WEEKS = 2
INNER_MODEL_AUDIT_WEEKS = 2
ROW_CALIBRATION_WEEKS = 2
CROSSFIT_WARMUP_WEEKS = 8
CROSSFIT_BLOCK_WEEKS = 4
CONTINUATION_HALF_LIFE_SECONDS = 24 * 60 * 60
CONTINUATION_CROSSFIT_MINIMUM_WEEKS = 4
CONTINUATION_CROSSFIT_BLOCK_WEEKS = 1
MINIMUM_CONTINUATION_CROSSFIT_BLOCKS = 4
STOP_LOSS_OVERRUN_QUANTILE = 0.999
MINIMUM_EXPERT_OPPORTUNITIES = 100
LOCAL_PLAN_AUDIT_STATES = 2_000
LOCAL_PLAN_TRAINING_STATES = 20_000
LOCAL_PLAN_INNER_CALIBRATION_STATES = 5_000
LOCAL_PLAN_REGRET_MATERIAL_BPS = 2.0
MINIMUM_OOS_TRADES = 300
MINIMUM_SIDE_OOS_TRADES = 100
MINIMUM_CONTROLLER_SELECTION_TRADES = 30
EXPERT_CATALOG_ROOT = ROOT / "fold_experts"
ALPHA_FEATURES = (
    *base.GATING_CONTEXT,
    "side",
    "horizon_fraction",
    "target_1_bps",
    "target_2_bps",
    "stop_bps",
    "trailing_bps",
    "predicted_favorable_q50_bps",
    "predicted_favorable_q75_bps",
    "predicted_adverse_q75_bps",
    "gate_expected_gross_bps",
    "gate_disagreement_bps",
    "gate_entropy",
    "gate_effective_experts",
    "gate_top_weight",
    *(f"view_{view}_prediction_bps" for view in base.VIEWS),
    *(f"proposal_{view}_support" for view in base.VIEWS),
    "proposal_support_fraction",
    "proposal_consensus_selected",
    "proposal_rank_fraction",
    "first_exit_fraction",
)
FOLD_EXPERT_FEATURES = (
    "managed_expert_mean_bps",
    "managed_expert_q90_bps",
    "managed_expert_best_lcb_bps",
    "managed_expert_dispersion_bps",
    "managed_expert_positive_fraction",
    "managed_expert_target_rate",
    "managed_expert_stop_rate",
    "managed_expert_log_opportunities",
    "managed_generator_score_bps",
)
MODEL_FEATURES = (*ALPHA_FEATURES, *FOLD_EXPERT_FEATURES)
GENERATOR_FEATURES = ALPHA_FEATURES
STATE_FEATURES = (
    "daily_pnl_fraction",
    "risk_remaining_fraction",
    "position_side",
    "time_in_position_seconds",
    "unrealized_net_bps",
    "close_cost_bps",
)

LABEL_PROTOCOL = {
    "symbol": SYMBOL,
    "venue": VENUE,
    "parent_protocol_hash": base.PROTOCOL_HASH,
    "path_resolution": (
        "1s state path plus ordered millisecond aggregate-trade entry, target, stop and trailing "
        "fills"
    ),
    "no_trade_seconds": "non-executable; skipped by stop, target, trailing and timeout logic",
    "timeout_execution": "first observed trade bucket at or after the plan horizon",
    "action_space": "variable expert-supported plans preserving distinct horizon proposals",
    "sides": ["LONG", "SHORT"],
    "prediction_horizon_anchors_seconds": list(PREDICTION_HORIZONS_SECONDS),
    "fixed_action_plans": False,
    "proposal_rule": (
        "each specialist proposes its best horizon; identical horizons are deduplicated; "
        "consensus is recorded without averaging incompatible first-passage distributions"
    ),
    "management": "dynamic horizon/TP1/TP2/partial exit/initial stop/non-widening trailing",
    "same_second": "target/stop conflicts are excluded fail-closed",
    "entry": "first observed aggregate trade after decision",
    "terminal_return_prefilter": False,
    "economic_target": "log1p(risk-sized portfolio return after Binance 1x costs and funding)",
    "historical_start": HISTORICAL_START.isoformat(),
    "historical_end": HISTORICAL_END.isoformat(),
}
LABEL_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(LABEL_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
LABEL_ROOT = LABEL_ROOT_BASE / LABEL_PROTOCOL_HASH[:16]

PROTOCOL = {
    "name": "musca_btc_binance_canonical_policy_challenger",
    "symbol": SYMBOL,
    "venue": VENUE,
    "frozen_discovery_control": {
        "protocol_hash": discovery.PROTOCOL_HASH,
        "report": str(AUTO_MOE_REPORT),
        "mutated": False,
    },
    "source": {
        "market": "official Binance USD-M monthly aggTrades",
        "path_resolution_seconds": 1,
        "features": str(base.SOURCE),
        "parent_actions": [str(path) for path in PARENT_ACTIONS],
        "legacy_terminal_expert_outputs_used_by_selector": False,
        "feature_availability": "available_at <= decision; entry at first later trade",
    },
    "execution": {
        "baseline": "TAKER",
        "fee": "signed GET /fapi/v1/commissionRate or labelled official config fallback",
        "non_fee_reserve_round_trip_bps": EXECUTION_RESERVE_ROUND_TRIP_BPS,
        "selection_cost_multiplier": 1.0,
        "diagnostic_cost_multipliers": [1.5, 2.0],
        "maker_enabled": False,
        "invented_spread_or_fill_labels": False,
    },
    "state_action": {
        "label_protocol_hash": LABEL_PROTOCOL_HASH,
        "entry_types": ["WAIT", "ENTER_LONG", "ENTER_SHORT"],
        "position_types": ["HOLD", "REDUCE", "CLOSE", "TIGHTEN_STOP", "UPDATE_TRAIL"],
        "prediction_horizon_anchors_seconds": list(PREDICTION_HORIZONS_SECONDS),
        "fixed_action_plans": False,
        "proposal_rule": (
            "each specialist proposes its best horizon; identical horizons are deduplicated; "
            "consensus is support metadata and cannot average away a specialist"
        ),
        "plan_parameters": [
            "horizon_seconds",
            "target_1_bps",
            "target_2_bps",
            "first_exit_fraction",
            "stop_bps",
            "trailing_bps",
        ],
        "management": "same parameterized management path for labels and replay",
        "same_second": "target/stop conflicts are excluded fail-closed",
        "gpu_cpu_tolerance_bps": GPU_CPU_TOLERANCE_BPS,
        "terminal_return_prefilter": False,
    },
    "probability_heads": {
        "outcomes": list(OUTCOME_NAMES),
        "target_probability": "P(first target before initial stop)",
        "conditional_returns": list(OUTCOME_NAMES),
        "auxiliary": ["MFE quantiles", "MAE quantiles", "time to target"],
        "champion": "logistic plus Ridge",
        "challenger": "XGBoost CUDA",
        "challenger_rule": "strictly better Brier, EV calibration, EV MAE and decision regret",
        "value_benchmarks": ["decomposed event EV", "direct net return", "direct log utility"],
        "post_selection_calibration": (
            "row calibration on the first two weeks followed by winner-only calibration on the "
            "next two weeks; both precede policy selection and outer test"
        ),
    },
    "fold_local_experts": {
        "generator": ("plan-aware XGBRFRegressor CUDA critic trained on exact managed net_bps"),
        "candidate": "every state-plan context leaf by side after expert proposal preservation",
        "terminal_prefilter": False,
        "minimum_support_after_managed_evaluation": MINIMUM_EXPERT_OPPORTUNITIES,
        "compression": "active-leaf managed statistics; leaves do not define the plan",
        "training_scope": "outer-fold fit only",
        "fit_encoding": "strict expanding chronological cross-fit with actual-exit purge",
    },
    "plan_generator": {
        "experts": "OOF heterogeneous horizon/view return and path-quantile predictors",
        "gating": (
            "each view proposes its strongest horizon; identical proposals are deduplicated and "
            "consensus is metadata rather than a destructive average"
        ),
        "plans_per_state": "variable expert-supported LONG/SHORT proposals; not ten templates",
        "objective": "view support, disagreement, horizon-specific path quantiles and Binance cost",
        "local_perturbations": (
            "one parameter family at a time around each expert-composed plan; enabled only by "
            "past-only local-regret audit"
        ),
        "local_training_support": (
            "deterministic past-only sample with every local parameter family labelled by the "
            "same execution engine before any local action can enter calibration or test"
        ),
    },
    "controller": {
        "actions": [
            "WAIT",
            "ENTER_LONG",
            "ENTER_SHORT",
            "HOLD",
            "REDUCE",
            "CLOSE",
            "TIGHTEN_STOP",
            "UPDATE_TRAIL",
        ],
        "maximum_positions": 1,
        "forced_utc_close": False,
        "risk_per_trade": RISK_PER_TRADE,
        "maximum_leverage": MAXIMUM_LEVERAGE,
        "maximum_daily_loss": MAXIMUM_DAILY_LOSS,
        "state_features": list(STATE_FEATURES),
        "entry_value": (
            "myopic coherent utility champion versus chronologically cross-fitted double-Q "
            "semi-Markov challenger; continuation is used only after positive past-only policy "
            "selection evidence and cannot make a negative immediate action enter"
        ),
        "continuation_target": (
            "actual immediate utility plus a temporally split double-estimator value at the next "
            "free state; action selection and evaluation use different past-only regressors, "
            "never the best future realized outcome"
        ),
        "minimum_continuation_crossfit_blocks": MINIMUM_CONTINUATION_CROSSFIT_BLOCKS,
        "minimum_controller_selection_trades": MINIMUM_CONTROLLER_SELECTION_TRADES,
        "utility_consistency": (
            "predicted expected log utility cannot exceed log1p(leverage times predicted net EV)"
        ),
        "continuation_half_life_seconds": CONTINUATION_HALF_LIFE_SECONDS,
        "risk_stop_overrun_reserve": (
            "past-only 99.9% quantile of loss beyond initial stop plus actual 1x costs"
        ),
    },
    "validation": {
        "nested_walk_forward": {
            "minimum_fit_weeks": MINIMUM_FIT_WEEKS,
            "inner_calibration_weeks": INNER_CALIBRATION_WEEKS,
            "inner_model_audit_weeks": INNER_MODEL_AUDIT_WEEKS,
            "calibration_weeks": WINDOW_WEEKS,
            "row_calibration_weeks": ROW_CALIBRATION_WEEKS,
            "winner_calibration_weeks": WINDOW_WEEKS - ROW_CALIBRATION_WEEKS,
            "policy_selection_weeks": WINDOW_WEEKS,
            "outer_test_weeks": WINDOW_WEEKS,
        },
        "purge": "actual exit timestamp",
        "bootstrap_unit": ["day", "week"],
        "global_trials": "research registry; all previously observed periods are contaminated",
        "frequency": (
            "coherent immediate utility positive; paired Q advantage is an optional challenger "
            "selected on the prior policy window; threshold frontier is diagnostic only"
        ),
        "negative_controls": [
            "random prediction",
            "temporal shift",
            "label permutation",
            "FULL only",
            "best active fold expert",
            "equal-weight experts",
            "no gate",
            "train-median constant plan",
            "LONG only",
            "SHORT only",
            "simple momentum",
            "simple mean reversion",
            "always WAIT",
        ],
    },
    "gates": {
        "minimum_oos_trades": MINIMUM_OOS_TRADES,
        "minimum_side_oos_trades": MINIMUM_SIDE_OOS_TRADES,
        "expectancy_net_bps": 0,
        "bootstrap_lcb_bps": 0,
        "profit_factor": 1.15,
        "maximum_drawdown": MAXIMUM_DRAWDOWN,
        "positive_active_days": 0.50,
        "risk_violations": 0,
        "sides_evaluated_separately": True,
    },
    "historical_end": HISTORICAL_END.isoformat(),
    "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
    "future_holdout_opened": False,
    "paper_only": True,
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_RUN_STARTED = time.monotonic()


class Predictor(Protocol):
    def fit(self, values: np.ndarray, target: np.ndarray, **kwargs: Any) -> Predictor: ...

    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SequentialState:
    daily_pnl_fraction: float
    risk_remaining_fraction: float
    position_side: int
    time_in_position_seconds: int
    unrealized_net_bps: float
    close_cost_bps: float
    expert_id: str | None
    regime: str
    vwap_distance_bps: float
    avwap_distance_bps: float
    funding_bps: float
    volatility_bps: float


@dataclass
class ReplayRiskState:
    equity: float = 1.0
    peak_equity: float = 1.0
    current_day: pd.Timestamp | None = None
    day_start_equity: float = 1.0


@dataclass(frozen=True)
class FeeContract:
    maker_bps_per_side: float
    taker_bps_per_side: float
    reserve_round_trip_bps: float
    source: str

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.taker_bps_per_side + self.reserve_round_trip_bps


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    _atomic_replace(temporary, path)


def _atomic_replace(source: Path, destination: Path) -> None:
    """Retry transient Windows reader/antivirus locks without weakening atomicity."""
    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    joblib.dump(payload, temporary)
    _atomic_replace(temporary, path)


def _status(phase: str, detail: str, percent: float, **extra: Any) -> None:
    elapsed = max(0.0, time.monotonic() - _RUN_STARTED)
    eta = (
        elapsed * (100 - percent) / percent
        if elapsed >= 60 and 0 < percent < 100
        else 0.0
        if percent >= 100
        else None
    )
    payload = {
        "phase": phase,
        "detail": detail,
        "percent": round(float(percent), 2),
        "updated_at": datetime.now(UTC).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "pid": os.getpid(),
        "cpu_seconds": round(time.process_time(), 2),
        "ram_gb": _process_ram_gb(),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": None if eta is None else round(eta, 1),
        **extra,
    }
    _atomic_json(STATUS, payload)
    with suppress(BrokenPipeError, OSError):
        print(f"[{payload['percent']:6.2f}%] {phase}: {detail}", flush=True)


def _process_ram_gb() -> float | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return float(round(counters.WorkingSetSize / 1024**3, 3))
    except (AttributeError, OSError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fee_contract() -> FeeContract:
    config = load_config(CONFIG)
    schedule = fee_schedule(config)
    return FeeContract(
        maker_bps_per_side=float(schedule.maker_bps),
        taker_bps_per_side=float(schedule.taker_bps),
        reserve_round_trip_bps=EXECUTION_RESERVE_ROUND_TRIP_BPS,
        source=str(schedule.source),
    )


def _report_protocol_hash(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("protocol_hash")
    if isinstance(value, str) and len(value) == 64:
        return value
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_research_registry() -> dict[str, Any]:
    reports = sorted(Path("data/reports").glob("*.json"))
    protocols: dict[str, list[str]] = {}
    for path in reports:
        protocol_hash = _report_protocol_hash(path)
        if protocol_hash is not None:
            protocols.setdefault(protocol_hash, []).append(str(path))
    ft_log = Path("docs/musca-v5-fine-tuning-log.md")
    ft_trials: list[str] = []
    if ft_log.exists():
        ft_trials = sorted(set(re.findall(r"\bFT-\d{3}\b", ft_log.read_text(encoding="utf-8"))))
    expert_count = 0
    if AUTO_MOE_CANDIDATES.exists():
        expert_count = len(pd.read_parquet(AUTO_MOE_CANDIDATES, columns=["expert_id"]))
    experiments: list[dict[str, Any]] = []
    if EXPERIMENT_LEDGER.exists():
        try:
            existing = json.loads(EXPERIMENT_LEDGER.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                experiments = [item for item in existing if isinstance(item, dict)]
            elif isinstance(existing, dict):
                experiments = [
                    item for item in existing.get("experiments", []) if isinstance(item, dict)
                ]
        except (OSError, ValueError, TypeError):
            raise ValueError("research experiment ledger is not valid JSON") from None
    experiment_id = f"MUSCA-BTC-{PROTOCOL_HASH[:12]}"
    if not any(item.get("experiment_id") == experiment_id for item in experiments):
        experiments.append(
            {
                "experiment_id": experiment_id,
                "protocol_hash": PROTOCOL_HASH,
                "git_commit": _git_commit(),
                "hypothesis": (
                    "preserving distinct specialist horizon proposals and making the managed "
                    "critic plan-aware removes the e556 action-space collapse"
                ),
                "changes": [
                    "variable deduplicated plans proposed by the expert views",
                    "horizon-specific first-passage quantiles without cross-horizon averaging",
                    "plan-aware fold-local managed critic",
                    "versioned state-action labels preserving prior protocol artifacts",
                ],
                "periods_observed": [
                    HISTORICAL_START.isoformat(),
                    HISTORICAL_END.isoformat(),
                ],
                "selection_status": "PREREGISTERED_NOT_RUN",
                "accepted": None,
                "contaminated_after_observation": True,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(EXPERIMENT_LEDGER, {"experiments": experiments})
    ledger_payload = json.loads(EXPERIMENT_LEDGER.read_text(encoding="utf-8"))
    if isinstance(ledger_payload, dict):
        experiments = list(ledger_payload.get("experiments", []))
    elif isinstance(ledger_payload, list):
        experiments = ledger_payload
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "all_observed_periods_contaminated": True,
        "protocol_hashes": protocols,
        "protocol_count_observed": max(GLOBAL_PROTOCOL_FLOOR, len(protocols)),
        "fine_tuning_trials": ft_trials,
        "fine_tuning_trial_count": len(ft_trials),
        "registered_auto_moe_experts": max(GLOBAL_EXPERT_FLOOR, expert_count),
        "frozen_control_protocol_hash": discovery.PROTOCOL_HASH,
        "challenger_protocol_hash": PROTOCOL_HASH,
        "git_commit": _git_commit(),
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
        "future_holdout_opened": False,
        "experiment_ledger": str(EXPERIMENT_LEDGER),
        "experiment_count": len(experiments),
        "current_experiment_id": experiment_id,
    }
    _atomic_json(REGISTRY, payload)
    return payload


def expert_id(side: int, horizon_seconds: int) -> str:
    identity = {
        "label_protocol_hash": LABEL_PROTOCOL_HASH,
        "side": int(side),
        "horizon_seconds": int(horizon_seconds),
        "parent_protocol_hash": base.PROTOCOL_HASH,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"btc-{'long' if side > 0 else 'short'}-{horizon_seconds}s-{digest}"


def parameterized_plan_id(
    side: int,
    horizon_seconds: int,
    target_1_bps: float,
    target_2_bps: float,
    stop_bps: float,
    trailing_bps: float,
    first_exit_fraction: float,
    contributors: str,
) -> str:
    """Identify a generated plan by its executable parameters and expert mixture."""
    identity = {
        "label_protocol_hash": LABEL_PROTOCOL_HASH,
        "side": int(side),
        "horizon_seconds": int(horizon_seconds),
        "target_1_bps": round(float(target_1_bps), 4),
        "target_2_bps": round(float(target_2_bps), 4),
        "stop_bps": round(float(stop_bps), 4),
        "trailing_bps": round(float(trailing_bps), 4),
        "first_exit_fraction": round(float(first_exit_fraction), 4),
        "contributors": contributors,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"plan-{'long' if side > 0 else 'short'}-{digest}"


def fold_expert_id(fold_scope: str, side: int, horizon_seconds: int, tree: int, leaf: int) -> str:
    """Identify a fold-local expert without using any OOS outcome."""
    identity = {
        "protocol_hash": PROTOCOL_HASH,
        "fold_scope": fold_scope,
        "side": int(side),
        "horizon_seconds": int(horizon_seconds),
        "tree": int(tree),
        "leaf": int(leaf),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"btc-{'long' if side > 0 else 'short'}-{horizon_seconds}s-leaf-{digest}"


def _one_second_path(month: str) -> Path:
    return base.MICRO_ROOT / f"{SYMBOL}-aggTrades-1s-{month}.parquet"


def _datetime_ns(values: Any) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    return np.asarray(index.view("int64"), dtype=np.int64)


def _ordered_event_path(month: str) -> Path:
    return ORDERED_EVENT_ROOT / f"month={month}.parquet"


def _ensure_ordered_event_source(month: str) -> Path:
    output = _ordered_event_path(month)
    if output.exists():
        return output
    ORDERED_EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = base.MICRO_ROOT / f"{SYMBOL}-aggTrades-{month}.zip"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"missing official event archive for stop refinement: {archive_path}"
        )
    names = ["id", "price", "quantity", "first", "last", "timestamp", "buyer_maker"]
    temporary = output.with_suffix(f".parquet.{os.getpid()}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise ValueError(f"unexpected aggregate-trade archive layout: {archive_path}")
            chunks = pd.read_csv(
                archive.open(members[0]),
                header=None,
                names=names,
                usecols=["id", "price", "quantity", "timestamp", "buyer_maker"],
                dtype=str,
                chunksize=1_000_000,
            )
            for chunk_number, chunk in enumerate(chunks, start=1):
                event_id = pd.to_numeric(chunk["id"], errors="coerce")
                timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")
                price = pd.to_numeric(chunk["price"], errors="coerce")
                quantity = pd.to_numeric(chunk["quantity"], errors="coerce")
                valid = event_id.notna() & timestamp.notna() & price.notna() & quantity.notna()
                frame = pd.DataFrame(
                    {
                        "event_id": event_id.loc[valid].to_numpy(np.int64),
                        "timestamp_ms": timestamp.loc[valid].to_numpy(np.int64),
                        "second": np.floor_divide(timestamp.loc[valid].to_numpy(np.int64), 1_000),
                        "price": price.loc[valid].to_numpy(float),
                        "quantity": quantity.loc[valid].to_numpy(float),
                        "buyer_maker": chunk.loc[valid, "buyer_maker"]
                        .astype(str)
                        .str.lower()
                        .eq("true")
                        .to_numpy(bool),
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                _status(
                    "ordered_events",
                    f"{month} event chunk {chunk_number}",
                    10,
                    month=month,
                    chunk=chunk_number,
                )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no valid ordered aggregate trades in {archive_path}")
    _atomic_replace(temporary, output)
    return output


def _raw_events_for_seconds(seconds: set[int]) -> dict[int, list[tuple[int, int, float]]]:
    if not seconds:
        return {}
    target = pd.DataFrame({"second": sorted(seconds)})
    paths = [
        str(_ensure_ordered_event_source(month))
        for month in sorted(
            {
                str(pd.Period(pd.Timestamp(second, unit="s", tz="UTC"), freq="M"))
                for second in seconds
            }
        )
    ]
    connection = duckdb.connect()
    try:
        connection.register("target_exit_seconds", target)
        selected = connection.execute(
            """
            SELECT event.second, event.timestamp_ms, event.event_id, event.price
            FROM read_parquet(?) AS event
            INNER JOIN target_exit_seconds AS target USING (second)
            ORDER BY event.timestamp_ms, event.event_id
            """,
            [paths],
        ).df()
    finally:
        connection.close()
    result = {
        int(cast(Any, second)): list(
            zip(
                group["timestamp_ms"].astype("int64").to_list(),
                group["event_id"].astype("int64").to_list(),
                group["price"].astype(float).to_list(),
                strict=True,
            )
        )
        for second, group in selected.groupby("second", sort=False)
    }
    missing = [second for second, events in result.items() if not events]
    missing.extend(second for second in seconds if second not in result)
    if missing:
        examples = [
            pd.Timestamp(second, unit="s", tz="UTC").isoformat()
            for second in sorted(set(missing))[:20]
        ]
        raise ValueError(
            f"missing ordered aggregate trades for {len(set(missing))} exit seconds: {examples}"
        )
    return result


def _raw_event_prices_for_seconds(seconds: set[int]) -> dict[int, list[float]]:
    """Compatibility view used by focused tests and diagnostics."""
    return {
        second: [event[2] for event in events]
        for second, events in _raw_events_for_seconds(seconds).items()
    }


def _refine_entry_events(
    actions: pd.DataFrame,
    source: pd.DataFrame,
    positions: np.ndarray,
) -> pd.DataFrame:
    output = actions.copy()
    bucket_timestamp = pd.to_datetime(source.loc[positions, "timestamp"], utc=True)
    bucket_seconds = _datetime_ns(bucket_timestamp) // 1_000_000_000
    events = _raw_events_for_seconds(set(bucket_seconds.tolist()))
    requested_ns = _datetime_ns(output["entry_timestamp"])
    actual_timestamp_ms = np.empty(len(output), dtype=np.int64)
    actual_price = np.empty(len(output), dtype=float)
    actual_event_id = np.empty(len(output), dtype=np.int64)
    for row, second in enumerate(bucket_seconds):
        candidates = events[int(second)]
        requested_ms = math.ceil(requested_ns[row] / 1_000_000)
        eligible = [event for event in candidates if event[0] >= requested_ms]
        if not eligible:
            raise ValueError("entry bucket does not contain an aggregate trade after decision")
        timestamp_ms, event_id, price = eligible[0]
        actual_timestamp_ms[row] = timestamp_ms
        actual_event_id[row] = event_id
        actual_price[row] = price
    actual_timestamp = pd.to_datetime(actual_timestamp_ms, unit="ms", utc=True)
    output["entry_bucket_timestamp"] = bucket_timestamp.to_numpy()
    output["actual_entry_timestamp"] = actual_timestamp
    output["entry_price"] = actual_price
    output["entry_event_id"] = actual_event_id
    output["entry_delay_seconds"] = (_datetime_ns(actual_timestamp) - requested_ns) / 1_000_000_000
    return output


def refine_stop_fills_with_ordered_events(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    if "exit_bucket_timestamp" in output:
        bucket_timestamp = pd.to_datetime(output["exit_bucket_timestamp"], utc=True)
    elif {"entry_bucket_timestamp", "exit_seconds"}.issubset(output.columns):
        bucket_timestamp = pd.to_datetime(
            output["entry_bucket_timestamp"], utc=True
        ) + pd.to_timedelta(output["exit_seconds"].to_numpy(int) - 1, unit="s")
    else:
        bucket_timestamp = pd.to_datetime(output["exit_timestamp"], utc=True)
    exit_second = _datetime_ns(bucket_timestamp) // 1_000_000_000
    management = output["management_code"].to_numpy(int)
    conflict = (
        output["time_to_target_seconds"].to_numpy(int)
        == output["time_to_stop_seconds"].to_numpy(int)
    ) & output["time_to_target_seconds"].ge(0).to_numpy()
    refinable = np.isin(management, (OUTCOME_STOP, 3, 4, 5)) & ~conflict
    events_by_second = _raw_events_for_seconds(set(exit_second[refinable].tolist()))
    gross = output["gross_bps"].to_numpy(float).copy()
    fill_price = np.full(len(output), np.nan, dtype=float)
    stop_level = np.full(len(output), np.nan, dtype=float)
    slippage = np.full(len(output), np.nan, dtype=float)
    refined = np.zeros(len(output), dtype=bool)
    fill_timestamp_ms = np.full(len(output), -1, dtype=np.int64)
    fill_event_id = np.full(len(output), -1, dtype=np.int64)
    side = output["side"].to_numpy(int)
    entry = output["entry_price"].to_numpy(float)
    target_1 = output["target_1_bps"].to_numpy(float)
    fraction = output["first_exit_fraction"].to_numpy(float)
    initial_stop = output["stop_bps"].to_numpy(float)
    target_time = output["time_to_target_seconds"].to_numpy(int)
    for row in np.flatnonzero(refinable):
        code = management[row]
        events = events_by_second[int(exit_second[row])]
        event_returns = (
            side[row]
            * (np.asarray([event[2] for event in events], dtype=float) / entry[row] - 1)
            * 10_000
        )
        if code == 3:
            crossing = np.flatnonzero(event_returns >= output.iloc[row]["target_2_bps"] - 1e-9)
            if not len(crossing):
                raise ValueError("coarse target exit has no matching ordered aggregate trade")
            event_index = int(crossing[0])
            refined[row] = True
            fill_price[row] = events[event_index][2]
            fill_timestamp_ms[row] = events[event_index][0]
            fill_event_id[row] = events[event_index][1]
            continue
        if code == OUTCOME_STOP:
            threshold = -initial_stop[row]
            filled_fraction = 0.0
            booked = 0.0
        elif code == 4:
            filled_fraction = fraction[row] if target_time[row] >= 0 else 0.0
            booked = filled_fraction * target_1[row]
            event_index = 0
            actual_return = float(event_returns[event_index])
            gross[row] = booked + (1.0 - filled_fraction) * actual_return
            fill_price[row] = events[event_index][2]
            fill_timestamp_ms[row] = events[event_index][0]
            fill_event_id[row] = events[event_index][1]
            refined[row] = True
            continue
        else:
            filled_fraction = fraction[row]
            booked = filled_fraction * target_1[row]
            threshold = (gross[row] - booked) / max(1.0 - filled_fraction, 1e-12)
        crossing = np.flatnonzero(event_returns <= threshold + 1e-9)
        if not len(crossing):
            raise ValueError("coarse stop exit has no matching ordered aggregate trade")
        actual_return = float(event_returns[crossing[0]])
        gross[row] = booked + (1.0 - filled_fraction) * actual_return
        event_index = int(crossing[0])
        fill_price[row] = events[event_index][2]
        fill_timestamp_ms[row] = events[event_index][0]
        fill_event_id[row] = events[event_index][1]
        stop_level[row] = threshold
        slippage[row] = max(0.0, threshold - actual_return)
        refined[row] = True
    exact = fill_timestamp_ms >= 0
    output.loc[exact, "exit_timestamp"] = pd.to_datetime(
        fill_timestamp_ms[exact], unit="ms", utc=True
    )
    output["gross_bps"] = gross
    output["event_fill_price"] = fill_price
    output["exit_event_timestamp_ms"] = fill_timestamp_ms
    output["exit_event_id"] = fill_event_id
    output["event_stop_level_bps"] = stop_level
    output["stop_slippage_bps"] = slippage
    output["event_order_refined"] = refined
    output["same_second_conflict"] = conflict
    output["data_valid"] = ~conflict
    output["event_holding_seconds"] = (
        _datetime_ns(output["exit_timestamp"]) - _datetime_ns(output["actual_entry_timestamp"])
    ) / 1_000_000_000
    return output


def ensure_one_second_sources() -> dict[str, Any]:
    missing = [month for month in ONE_SECOND_MONTHS if not _one_second_path(month).exists()]
    if missing:
        archives = [base.MICRO_ROOT / f"{SYMBOL}-aggTrades-{month}.zip" for month in missing]
        absent = [str(path) for path in archives if not path.exists()]
        if absent:
            raise FileNotFoundError(
                "official Binance archives required for one-second reconstruction: "
                + ", ".join(absent)
            )
        workers = min(2, len(missing))
        _status(
            "dataset_1s",
            f"reconstructing {len(missing)} local official archives with {workers} workers",
            2,
            blocks_completed=0,
            blocks_total=len(missing),
            workers=workers,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            jobs = {
                executor.submit(
                    microstructure._aggregate,
                    archive,
                    month,
                    seconds=1,
                    symbol=SYMBOL,
                    status_path=STATUS,
                    report_progress=False,
                ): month
                for month, archive in zip(missing, archives, strict=True)
            }
            for number, future in enumerate(as_completed(jobs), start=1):
                month = jobs[future]
                future.result()
                _status(
                    "dataset_1s",
                    f"official archive {number}/{len(missing)} complete: {month}",
                    2 + 8 * number / len(missing),
                    month=month,
                    blocks_completed=number,
                    blocks_total=len(missing),
                    workers=workers,
                )
    files: dict[str, Any] = {}
    for month in ONE_SECOND_MONTHS:
        path = _one_second_path(month)
        rows = pd.read_parquet(path, columns=["timestamp", "available_at"])
        timestamp = pd.to_datetime(rows["timestamp"], utc=True)
        available = pd.to_datetime(rows["available_at"], utc=True)
        if rows.empty or not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
            raise ValueError(f"invalid one-second aggTrades ordering in {path}")
        if not available.eq(timestamp + pd.Timedelta(seconds=1)).all():
            raise ValueError(f"invalid one-second availability contract in {path}")
        files[month] = {
            "path": str(path),
            "rows": len(rows),
            "start": timestamp.iloc[0].isoformat(),
            "end": timestamp.iloc[-1].isoformat(),
            "sha256": _sha256(path),
        }
    return {
        "provider": "Binance official public data",
        "symbol": SYMBOL,
        "resolution_seconds": 1,
        "months": files,
    }


def _observed_l2_coverage() -> dict[str, Any]:
    files = sorted(BINANCE_L2_ROOT.glob("btcusdt_????-??-??.jsonl"))
    identity = {
        str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    }
    if EXECUTION_REPORT.exists():
        try:
            cached = json.loads(EXECUTION_REPORT.read_text(encoding="utf-8")).get(
                "shadow_execution", {}
            )
            if cached.get("file_identity") == identity:
                return cast(dict[str, Any], cached)
        except (OSError, ValueError, TypeError):
            pass
    samples: list[dict[str, Any]] = []
    total_rows = 0
    invalid_rows = 0
    for path in files:
        with path.open("r", encoding="utf-8") as source:
            for row_number, line in enumerate(source):
                total_rows += 1
                if row_number % 60:
                    continue
                payload = json.loads(line)
                if "exchange_second" not in payload or "available_at" not in payload:
                    invalid_rows += 1
                    continue
                exchange_time = pd.Timestamp(int(payload["exchange_second"]), unit="s", tz="UTC")
                available_at = pd.Timestamp(payload["available_at"])
                bids = payload.get("bids") or []
                asks = payload.get("asks") or []
                samples.append(
                    {
                        "exchange_time": exchange_time,
                        "available_at": available_at,
                        "latency_seconds": (available_at - exchange_time).total_seconds(),
                        "book_network_latency_ms": payload.get("book_network_latency_ms"),
                        "book_valid": bool(
                            bids
                            and asks
                            and float(bids[0][0]) > 0
                            and float(asks[0][0]) > float(bids[0][0])
                        ),
                    }
                )
    if not samples:
        return {
            "available": False,
            "distinct_days": 0,
            "rows": 0,
            "reason": "NO_OBSERVED_BINANCE_L2",
        }
    latency = np.asarray([item["latency_seconds"] for item in samples], dtype=float)
    network_latency = np.asarray(
        [
            float(item["book_network_latency_ms"])
            for item in samples
            if item["book_network_latency_ms"] is not None
        ],
        dtype=float,
    )
    timestamps = pd.DatetimeIndex([item["exchange_time"] for item in samples])
    return {
        "available": True,
        "source": "Binance official USD-M websocket depth20@100ms routed to 1s snapshots",
        "files": [str(path) for path in files],
        "distinct_days": int(timestamps.floor("D").nunique()),
        "rows": total_rows,
        "sample_stride_seconds": 60,
        "sampled_rows": len(samples),
        "sampled_invalid_rows": invalid_rows,
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "feature_availability_delay_seconds": {
            "p50": float(np.quantile(latency, 0.50)),
            "p90": float(np.quantile(latency, 0.90)),
            "p99": float(np.quantile(latency, 0.99)),
            "maximum": float(latency.max()),
        },
        "book_network_latency_ms": (
            {
                "p50": float(np.quantile(network_latency, 0.50)),
                "p90": float(np.quantile(network_latency, 0.90)),
                "p99": float(np.quantile(network_latency, 0.99)),
                "maximum": float(network_latency.max()),
            }
            if len(network_latency)
            else None
        ),
        "valid_non_crossed_book_fraction": float(
            np.mean([bool(item["book_valid"]) for item in samples])
        ),
        "aggregate_trade_order_retained_within_snapshot": True,
        "aggregate_trade_timestamp_within_second_retained": False,
        "file_identity": identity,
    }


def build_execution_contract(source_manifest: dict[str, Any], fee: FeeContract) -> dict[str, Any]:
    raw_archives = {
        month: base.MICRO_ROOT / f"{SYMBOL}-aggTrades-{month}.zip" for month in ONE_SECOND_MONTHS
    }
    raw_available = all(path.exists() for path in raw_archives.values())
    l2 = _observed_l2_coverage()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "symbol": SYMBOL,
        "venue": VENUE,
        "fee_contract": asdict(fee) | {"round_trip_bps": fee.round_trip_bps},
        "alpha_path": {
            "source": "official monthly Binance aggregate trades",
            "raw_event_archives_available": raw_available,
            "raw_event_archive_paths": {month: str(path) for month, path in raw_archives.items()},
            "current_label_resolution_seconds": int(source_manifest["resolution_seconds"]),
            "ordered_fill_resolution": "aggregate-trade timestamp_ms and event_id",
            "ordered_entry_fill_used_by_current_labels": True,
            "ordered_target_stop_trailing_fill_used_by_current_labels": True,
            "event_order_inside_second_used_by_current_labels": True,
            "same_second_target_stop_conflict": "EXCLUDED_FAIL_CLOSED",
            "fill_classification": "TRADE_PATH_PROXY_NO_HISTORICAL_L2",
            "bid_ask_historical": False,
            "depth_historical": False,
            "partial_fill_historical": False,
            "market_impact_historical": False,
        },
        "shadow_execution": l2,
        "paper_engine": {
            "observed_bid_ask": True,
            "book_walking": True,
            "insufficient_depth_rejection": True,
            "fees": True,
            "network_latency_observed": l2.get("book_network_latency_ms") is not None,
            "feature_availability_delay_observed": True,
        },
        "parity": {
            "label_replay_shadow_differences_zero": False,
            "differences": [
                "historical labels use 1s trade-path proxy",
                "paper uses observed L2 bid/ask and depth",
                "historical partial fills and queue position are unavailable",
            ],
        },
        "gates": {
            "alpha_research_allowed": raw_available,
            "execution_model_allowed": bool(l2.get("distinct_days", 0) >= 30),
            "operational_promotion_allowed": False,
            "real_capital_allowed": False,
        },
    }
    _atomic_json(EXECUTION_REPORT, report)
    return report


def _regularize_seconds(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered["available_at"] = pd.to_datetime(ordered["available_at"], utc=True)
    index = pd.date_range(ordered["timestamp"].iloc[0], ordered["timestamp"].iloc[-1], freq="1s")
    regular = ordered.set_index("timestamp").reindex(index)
    observed = regular["trade_count"].notna() & regular["trade_count"].gt(0)
    previous = regular["close"].ffill()
    if previous.isna().any():
        raise ValueError("one-second path starts before an observable trade price")
    for name in ("open", "high", "low", "close"):
        regular.loc[~observed, name] = previous.loc[~observed]
    for name in ("base_volume", "quote_volume", "signed_quote_volume", "trade_count", "buy_count"):
        regular.loc[~observed, name] = 0
    regular["observed_trade"] = observed
    regular["available_at"] = regular.index + pd.Timedelta(seconds=1)
    regular.index.name = "timestamp"
    return regular.reset_index()


def _load_second_window(month: str) -> pd.DataFrame:
    period = pd.Period(month, freq="M")
    paths = [_one_second_path(month)]
    following = str(period + 1)
    if _one_second_path(following).exists():
        paths.append(_one_second_path(following))
    columns = [
        "timestamp",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "signed_quote_volume",
        "trade_count",
        "buy_count",
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    rows = pd.concat(frames, ignore_index=True)
    end = (period + 1).start_time.tz_localize("UTC") + pd.Timedelta(
        seconds=MAXIMUM_HORIZON_SECONDS + 60
    )
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.loc[rows["timestamp"].lt(end)].copy()
    return _regularize_seconds(rows)


def _parent_columns() -> list[str]:
    return list(
        dict.fromkeys(
            (
                "available_at",
                "entry_timestamp",
                "decision_position",
                "side",
                "horizon_seconds",
                *base.EXPERT_COLUMNS,
                *base.GATING_CONTEXT,
                "predicted_favorable_q50_bps",
                "predicted_favorable_q75_bps",
                "predicted_adverse_q75_bps",
            )
        )
    )


def _load_parent_actions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = _parent_columns()
    selected = ", ".join(f'"{name}"' for name in columns)
    pieces: list[pd.DataFrame] = []
    for path in PARENT_ACTIONS:
        if not path.exists():
            raise FileNotFoundError(f"missing frozen parent action checkpoint: {path}")
        query = (
            f"SELECT {selected} FROM read_parquet(?) "
            "WHERE entry_timestamp >= ? AND entry_timestamp < ?"
        )
        piece = duckdb.execute(query, [str(path), start.to_pydatetime(), end.to_pydatetime()]).df()
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=columns)
    output = pd.concat(pieces, ignore_index=True)
    for name in ("available_at", "entry_timestamp"):
        output[name] = pd.to_datetime(output[name], utc=True)
    if not output["available_at"].le(output["entry_timestamp"]).all():
        raise ValueError("parent actions violate feature availability")
    return output.sort_values(["entry_timestamp", "side", "horizon_seconds"]).reset_index(drop=True)


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -60.0, 0.0))
    return np.asarray(
        exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-12),
        dtype=float,
    )


def compose_parameterized_plans(
    inherited: pd.DataFrame, round_trip_cost_bps: float
) -> pd.DataFrame:
    """Preserve distinct expert proposals instead of averaging incompatible horizons."""
    if inherited.empty:
        return inherited.copy()
    keys = ["available_at", "entry_timestamp", "decision_position", "side"]
    ordered = inherited.sort_values([*keys, "horizon_seconds"], kind="stable").reset_index(
        drop=True
    )
    horizons = np.asarray(PREDICTION_HORIZONS_SECONDS, dtype=int)
    counts = ordered.groupby(keys, sort=False, dropna=False).size().to_numpy()
    if not np.all(counts == len(horizons)):
        raise ValueError("inherited expert predictions do not cover every horizon anchor")
    observed_horizons = ordered["horizon_seconds"].to_numpy(int).reshape(-1, len(horizons))
    if not np.all(observed_horizons == horizons[None, :]):
        raise ValueError("inherited expert horizon anchors are not deterministic")

    representatives = ordered.iloc[:: len(horizons)].reset_index(drop=True).copy()
    views = tuple(base.VIEWS)
    expert_values = np.stack(
        [
            representatives.loc[:, [f"expert_{horizon}s_{view}" for view in views]].to_numpy(float)
            for horizon in horizons
        ],
        axis=1,
    )
    q50 = ordered["predicted_favorable_q50_bps"].to_numpy(float).reshape(-1, len(horizons))
    q75 = ordered["predicted_favorable_q75_bps"].to_numpy(float).reshape(-1, len(horizons))
    adverse = ordered["predicted_adverse_q75_bps"].to_numpy(float).reshape(-1, len(horizons))
    selected_by_view = np.argmax(expert_values, axis=1)
    horizon_center = np.median(expert_values, axis=2)
    horizon_disagreement = np.std(expert_values, axis=2)
    robust_horizon_score = horizon_center - 0.5 * horizon_disagreement
    consensus_horizon = np.argmax(robust_horizon_score, axis=1)
    rank = np.argsort(np.argsort(-robust_horizon_score, axis=1), axis=1)
    proposal_support = np.stack(
        [(selected_by_view == number).sum(axis=1) for number in range(len(horizons))],
        axis=1,
    )
    proposal_mask = proposal_support > 0
    proposal_mask[np.arange(len(representatives)), consensus_horizon] = True
    equal_weight_prediction = expert_values.mean(axis=(1, 2))
    volatility = np.clip(representatives["volatility_percentile"].to_numpy(float), 0.0, 1.0)

    pieces: list[pd.DataFrame] = []
    for horizon_number, horizon in enumerate(horizons):
        active = proposal_mask[:, horizon_number]
        if not active.any():
            continue
        positions = np.flatnonzero(active)
        scores = expert_values[active, horizon_number, :]
        score_scale = np.maximum(np.median(np.abs(scores), axis=1, keepdims=True), 2.0)
        view_weights = _softmax_rows(scores / score_scale)
        entropy = -np.sum(view_weights * np.log(np.maximum(view_weights, 1e-12)), axis=1)
        maximum_entropy = math.log(len(views))
        stop = np.clip(adverse[active, horizon_number], 3.0, base.MAX_STOP_BPS)
        target_1 = np.maximum(q50[active, horizon_number], float(round_trip_cost_bps) + 1.0)
        target_2 = np.maximum(q75[active, horizon_number], target_1 + 1.0)
        support = selected_by_view[active] == horizon_number
        bitmask = np.sum(
            support.astype(np.int16) * (1 << np.arange(len(views), dtype=np.int16)), axis=1
        )
        contributor_lookup = {
            mask: ",".join(view for number, view in enumerate(views) if mask & (1 << number))
            for mask in range(1, 1 << len(views))
        }
        contributors = [
            f"{int(horizon)}s:{contributor_lookup.get(int(mask), 'consensus')}"
            + (":consensus" if consensus_horizon[position] == horizon_number else "")
            for mask, position in zip(bitmask, positions, strict=True)
        ]

        output = representatives.loc[active, [*keys, *base.GATING_CONTEXT]].copy()
        output["side"] = output["side"].astype(int)
        output["horizon_seconds"] = int(horizon)
        output["horizon_fraction"] = float(horizon) / MAXIMUM_HORIZON_SECONDS
        output["target_1_bps"] = np.clip(target_1, 1.0, base.MAX_TARGET_BPS - 1.0)
        output["target_2_bps"] = np.clip(
            np.maximum(target_2, output["target_1_bps"].to_numpy(float) + 1.0),
            2.0,
            base.MAX_TARGET_BPS,
        )
        output["stop_bps"] = stop
        trailing = np.clip(stop * (0.65 + 0.35 * volatility[active]), 3.0, base.MAX_STOP_BPS)
        output["trailing_bps"] = np.minimum(trailing, stop)
        output["first_exit_fraction"] = np.clip(0.35 + 0.30 * entropy / maximum_entropy, 0.35, 0.65)
        output["predicted_favorable_q50_bps"] = q50[active, horizon_number]
        output["predicted_favorable_q75_bps"] = q75[active, horizon_number]
        output["predicted_adverse_q75_bps"] = adverse[active, horizon_number]
        output["gate_expected_gross_bps"] = np.sum(view_weights * scores, axis=1)
        output["gate_disagreement_bps"] = np.sqrt(
            np.sum(
                view_weights
                * (scores - output["gate_expected_gross_bps"].to_numpy(float)[:, None]) ** 2,
                axis=1,
            )
        )
        output["gate_entropy"] = entropy
        output["gate_effective_experts"] = np.exp(entropy)
        output["gate_top_weight"] = view_weights.max(axis=1)
        output["proposal_support_fraction"] = support.mean(axis=1)
        output["proposal_consensus_selected"] = (
            consensus_horizon[active] == horizon_number
        ).astype(float)
        output["proposal_rank_fraction"] = rank[active, horizon_number] / max(len(horizons) - 1, 1)
        output["equal_weight_expert_prediction_bps"] = equal_weight_prediction[active]
        for view_number, view in enumerate(views):
            output[f"view_{view}_prediction_bps"] = scores[:, view_number]
            output[f"proposal_{view}_support"] = support[:, view_number].astype(float)
            output[f"gate_without_{view}_prediction_bps"] = np.mean(
                np.delete(scores, view_number, axis=1), axis=1
            )
        output["plan_contributors"] = contributors
        pieces.append(output)

    if not pieces:
        raise ValueError("expert proposal generator produced no executable plan")
    output = pd.concat(pieces, ignore_index=True)
    output["plan_id"] = [
        parameterized_plan_id(
            int(side),
            int(horizon),
            float(first_target),
            float(second_target),
            float(initial_stop),
            float(trail),
            float(exit_fraction),
            contributors,
        )
        for (
            side,
            horizon,
            first_target,
            second_target,
            initial_stop,
            trail,
            exit_fraction,
            contributors,
        ) in zip(
            output["side"],
            output["horizon_seconds"],
            output["target_1_bps"],
            output["target_2_bps"],
            output["stop_bps"],
            output["trailing_bps"],
            output["first_exit_fraction"],
            output["plan_contributors"],
            strict=True,
        )
    ]
    output["expert_id"] = output["plan_id"]
    if not np.isfinite(output.loc[:, ALPHA_FEATURES].to_numpy(float)).all():
        raise ValueError("parameterized plan features must be finite")
    if output.duplicated([*keys, "horizon_seconds"]).any():
        raise ValueError("expert proposals must be unique per state, side and horizon")
    return output.sort_values(
        ["entry_timestamp", "side", "horizon_seconds"], kind="stable"
    ).reset_index(drop=True)


def _first_observed_positions(
    source: pd.DataFrame, requested_entry: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    observed = source["observed_trade"].to_numpy(bool)
    observed_positions = np.flatnonzero(observed)
    observed_ns = (
        source.loc[observed, "timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    )
    requested_ns = (
        pd.to_datetime(requested_entry, utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )
    locations = np.searchsorted(observed_ns, requested_ns, side="left")
    if np.any(locations >= len(observed_positions)):
        raise ValueError("no observable Binance trade after requested entry")
    positions = observed_positions[locations]
    actual_ns = observed_ns[locations]
    delay = (actual_ns - requested_ns) / 1_000_000_000
    return positions.astype(np.int64), delay.astype(float)


def _observed_mask(source: pd.DataFrame) -> np.ndarray:
    return (
        source["observed_trade"].to_numpy(bool)
        if "observed_trade" in source
        else source["trade_count"].to_numpy(float) > 0
    )


def _timeout_positions(
    observed: np.ndarray, positions: np.ndarray, horizons: np.ndarray
) -> np.ndarray:
    observed_positions = np.flatnonzero(observed)
    starts = np.asarray(positions, dtype=np.int64) + np.asarray(horizons, dtype=np.int64)
    locations = np.searchsorted(observed_positions, starts, side="left")
    valid = locations < len(observed_positions)
    output = np.full(len(starts), -1, dtype=np.int64)
    output[valid] = observed_positions[locations[valid]]
    return output


def _simulate_cpu(
    source: pd.DataFrame,
    positions: np.ndarray,
    side: int,
    horizon: int | np.ndarray,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
    first_exit_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    opens = source["open"].to_numpy(float)
    highs = source["high"].to_numpy(float)
    lows = source["low"].to_numpy(float)
    observed = _observed_mask(source)
    count = len(positions)
    horizons = (
        np.full(count, int(horizon), dtype=np.int32)
        if np.ndim(horizon) == 0
        else np.asarray(horizon, dtype=np.int32)
    )
    timeout_positions = _timeout_positions(observed, positions, horizons)
    gross = np.empty(count, dtype=float)
    exit_seconds = np.empty(count, dtype=np.int32)
    management = np.empty(count, dtype=np.int8)
    first_target = np.full(count, -1, dtype=np.int32)
    first_stop = np.full(count, -1, dtype=np.int32)
    mfe = np.empty(count, dtype=float)
    mae = np.empty(count, dtype=float)
    for row in range(count):
        position = int(positions[row])
        row_horizon = int(horizons[row])
        filled_fraction = float(first_exit_fraction[row])
        remaining_fraction = 1.0 - filled_fraction
        entry = opens[position]
        stop_level = -float(stop[row])
        peak = 0.0
        half = 0.0
        first_filled = False
        done = False
        maximum = -math.inf
        adverse_maximum = -math.inf
        result = 0.0
        result_seconds = row_horizon
        result_code = OUTCOME_TIMEOUT
        for offset in range(row_horizon):
            index = position + offset
            if index >= len(source):
                raise ValueError("incomplete one-second future path")
            if not observed[index]:
                continue
            elapsed = offset + 1
            open_return = side * (opens[index] / entry - 1) * 10_000
            favorable = (
                (highs[index] / entry - 1) * 10_000
                if side > 0
                else (1 - lows[index] / entry) * 10_000
            )
            adverse = (
                (1 - lows[index] / entry) * 10_000
                if side > 0
                else (highs[index] / entry - 1) * 10_000
            )
            maximum = max(maximum, favorable)
            adverse_maximum = max(adverse_maximum, adverse)
            if first_target[row] < 0 and favorable >= target_1[row] - 1e-9:
                first_target[row] = elapsed
            if first_stop[row] < 0 and -adverse <= -stop[row] + 1e-9:
                first_stop[row] = elapsed
            if done:
                continue
            if open_return <= stop_level + 1e-9:
                result = half + remaining_fraction * open_return if first_filled else open_return
                result_seconds = elapsed
                result_code = 4
                done = True
                continue
            if not first_filled and -adverse <= stop_level + 1e-9:
                result = stop_level
                result_seconds = elapsed
                result_code = OUTCOME_STOP
                done = True
                continue
            if not first_filled and favorable >= target_1[row] - 1e-9:
                if favorable >= target_2[row] - 1e-9:
                    result = filled_fraction * target_1[row] + remaining_fraction * target_2[row]
                    result_seconds = elapsed
                    result_code = 3
                    done = True
                    continue
                first_filled = True
                half = filled_fraction * target_1[row]
                stop_level = max(stop_level, 0.0)
            elif first_filled:
                if -adverse <= stop_level + 1e-9:
                    result = half + remaining_fraction * stop_level
                    result_seconds = elapsed
                    result_code = 5
                    done = True
                    continue
                if favorable >= target_2[row] - 1e-9:
                    result = half + remaining_fraction * target_2[row]
                    result_seconds = elapsed
                    result_code = 3
                    done = True
                    continue
                peak = max(peak, favorable)
                stop_level = max(stop_level, peak - trailing[row])
        if not done:
            timeout_position = int(timeout_positions[row])
            if timeout_position < 0:
                raise ValueError("no observable Binance trade at or after plan timeout")
            terminal_price = opens[timeout_position]
            result_seconds = timeout_position - position + 1
            terminal = side * (terminal_price / entry - 1) * 10_000
            result = half + remaining_fraction * terminal if first_filled else terminal
        gross[row] = result
        exit_seconds[row] = result_seconds
        management[row] = result_code
        mfe[row] = maximum
        mae[row] = adverse_maximum
    event = np.where(
        (first_stop >= 0) & ((first_target < 0) | (first_stop <= first_target)),
        OUTCOME_STOP,
        np.where(first_target >= 0, OUTCOME_TARGET, OUTCOME_TIMEOUT),
    ).astype(np.int8)
    return {
        "gross_bps": gross,
        "exit_seconds": exit_seconds,
        "management_code": management,
        "event_class": event,
        "time_to_target_seconds": first_target,
        "time_to_stop_seconds": first_stop,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


_GPU_KERNEL: Any | None = None


def _gpu_kernel() -> Any:
    global _GPU_KERNEL
    if _GPU_KERNEL is not None:
        return _GPU_KERNEL
    import cupy as cp  # type: ignore[import-untyped]

    _GPU_KERNEL = cp.RawKernel(
        r"""
        extern "C" __global__ void managed_path(
            const double* opens, const double* highs, const double* lows, const double* closes,
            const long long* positions, const int* horizons, const double* target1,
            const double* target2, const double* stops, const double* trails,
            const double* first_exit_fraction, const int side,
            const long long source_size, const long long count, double* gross, int* exit_seconds,
            signed char* management, int* first_target, int* first_stop, double* mfe, double* mae) {
          long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
          if (row >= count) return;
          long long position = positions[row];
          int horizon = horizons[row];
          double filled_fraction = first_exit_fraction[row];
          double remaining_fraction = 1.0 - filled_fraction;
          double entry = opens[position];
          double stop_level = -stops[row], peak = 0.0, half = 0.0;
          double maximum = -1.0e300, adverse_maximum = -1.0e300, result = 0.0;
          int first_filled = 0, done = 0, result_seconds = horizon, result_code = 2;
          int target_time = -1, stop_time = -1;
          for (int offset = 0; offset < horizon; ++offset) {
            long long index = position + offset;
            if (index >= source_size) break;
            int elapsed = offset + 1;
            double open_return = side * (opens[index] / entry - 1.0) * 10000.0;
            double favorable = side > 0 ? (highs[index] / entry - 1.0) * 10000.0
                                         : (1.0 - lows[index] / entry) * 10000.0;
            double adverse = side > 0 ? (1.0 - lows[index] / entry) * 10000.0
                                      : (highs[index] / entry - 1.0) * 10000.0;
            maximum = fmax(maximum, favorable);
            adverse_maximum = fmax(adverse_maximum, adverse);
            if (target_time < 0 && favorable >= target1[row] - 1.0e-9) target_time = elapsed;
            if (stop_time < 0 && -adverse <= -stops[row] + 1.0e-9) stop_time = elapsed;
            if (done) continue;
            if (open_return <= stop_level + 1.0e-9) {
              result = first_filled ? half + remaining_fraction * open_return : open_return;
              result_seconds = elapsed; result_code = 4; done = 1; continue;
            }
            if (!first_filled && -adverse <= stop_level + 1.0e-9) {
              result = stop_level; result_seconds = elapsed; result_code = 1; done = 1; continue;
            }
            if (!first_filled && favorable >= target1[row] - 1.0e-9) {
              if (favorable >= target2[row] - 1.0e-9) {
                result = filled_fraction * target1[row] + remaining_fraction * target2[row];
                result_seconds = elapsed; result_code = 3; done = 1; continue;
              }
              first_filled = 1; half = filled_fraction * target1[row];
              stop_level = fmax(stop_level, 0.0);
            } else if (first_filled) {
              if (-adverse <= stop_level + 1.0e-9) {
                result = half + remaining_fraction * stop_level;
                result_seconds = elapsed; result_code = 5; done = 1; continue;
              }
              if (favorable >= target2[row] - 1.0e-9) {
                result = half + remaining_fraction * target2[row];
                result_seconds = elapsed; result_code = 3; done = 1; continue;
              }
              peak = fmax(peak, favorable);
              stop_level = fmax(stop_level, peak - trails[row]);
            }
          }
          if (!done) {
            double terminal = side * (closes[position + horizon - 1] / entry - 1.0) * 10000.0;
            result = first_filled ? half + remaining_fraction * terminal : terminal;
          }
          gross[row] = result; exit_seconds[row] = result_seconds;
          management[row] = (signed char)result_code; first_target[row] = target_time;
          first_stop[row] = stop_time; mfe[row] = maximum; mae[row] = adverse_maximum;
        }
        """,
        "managed_path",
    )
    return _GPU_KERNEL


def _simulate_gpu(
    source: pd.DataFrame,
    positions: np.ndarray,
    side: int,
    horizon: int | np.ndarray,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
    first_exit_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    import cupy as cp

    count = len(positions)
    observed = _observed_mask(source)
    horizons = (
        np.full(count, int(horizon), dtype=np.int32)
        if np.ndim(horizon) == 0
        else np.asarray(horizon, dtype=np.int32)
    )
    host_values = [
        source[name].to_numpy(np.float64).copy() for name in ("open", "high", "low", "close")
    ]
    host_values[0][~observed] = np.nan
    host_values[1][~observed] = -np.inf
    host_values[2][~observed] = np.inf
    device_values = [cp.asarray(values) for values in host_values]
    gpu_positions = cp.asarray(positions, dtype=cp.int64)
    gpu_horizons = cp.asarray(horizons, dtype=cp.int32)
    timeout_positions = _timeout_positions(observed, positions, horizons)
    gpu_parameters = [
        cp.asarray(values, dtype=cp.float64)
        for values in (target_1, target_2, stop, trailing, first_exit_fraction)
    ]
    gross = cp.empty(count, dtype=cp.float64)
    exit_seconds = cp.empty(count, dtype=cp.int32)
    management = cp.empty(count, dtype=cp.int8)
    first_target = cp.empty(count, dtype=cp.int32)
    first_stop = cp.empty(count, dtype=cp.int32)
    mfe = cp.empty(count, dtype=cp.float64)
    mae = cp.empty(count, dtype=cp.float64)
    block = 128
    _gpu_kernel()(
        ((count + block - 1) // block,),
        (block,),
        (
            *device_values,
            gpu_positions,
            gpu_horizons,
            *gpu_parameters,
            np.int32(side),
            np.int64(len(source)),
            np.int64(count),
            gross,
            exit_seconds,
            management,
            first_target,
            first_stop,
            mfe,
            mae,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    gross_values = cp.asnumpy(gross)
    management_values = cp.asnumpy(management)
    exit_values = cp.asnumpy(exit_seconds)
    target_time = cp.asnumpy(first_target)
    stop_time = cp.asnumpy(first_stop)
    timeout_rows = np.flatnonzero(management_values == OUTCOME_TIMEOUT)
    if len(timeout_rows):
        if np.any(timeout_positions[timeout_rows] < 0):
            raise ValueError("no observable Binance trade at or after plan timeout")
        entries = source["open"].to_numpy(float)[positions[timeout_rows]]
        terminal_prices = source["open"].to_numpy(float)[timeout_positions[timeout_rows]]
        terminal = side * (terminal_prices / entries - 1.0) * 10_000.0
        filled = target_time[timeout_rows] >= 0
        fractions = np.asarray(first_exit_fraction, dtype=float)[timeout_rows]
        terminal[filled] = (
            fractions[filled] * np.asarray(target_1, dtype=float)[timeout_rows][filled]
            + (1.0 - fractions[filled]) * terminal[filled]
        )
        gross_values[timeout_rows] = terminal
        exit_values[timeout_rows] = timeout_positions[timeout_rows] - positions[timeout_rows] + 1
    event = np.where(
        (stop_time >= 0) & ((target_time < 0) | (stop_time <= target_time)),
        OUTCOME_STOP,
        np.where(target_time >= 0, OUTCOME_TARGET, OUTCOME_TIMEOUT),
    ).astype(np.int8)
    return {
        "gross_bps": gross_values,
        "exit_seconds": exit_values,
        "management_code": management_values,
        "event_class": event,
        "time_to_target_seconds": target_time,
        "time_to_stop_seconds": stop_time,
        "mfe_bps": cp.asnumpy(mfe),
        "mae_bps": cp.asnumpy(mae),
    }


def simulate_management(
    source: pd.DataFrame,
    positions: np.ndarray,
    side: int,
    horizon: int | np.ndarray,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
    first_exit_fraction: np.ndarray | None = None,
    *,
    backend: str = "auto",
) -> dict[str, np.ndarray]:
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown path backend: {backend}")
    count = len(positions)
    horizons = (
        np.full(count, int(horizon), dtype=np.int32)
        if np.ndim(horizon) == 0
        else np.asarray(horizon, dtype=np.int32)
    )
    fractions = (
        np.full(count, 0.5, dtype=float)
        if first_exit_fraction is None
        else np.asarray(first_exit_fraction, dtype=float)
    )
    if len(horizons) != count or len(fractions) != count:
        raise ValueError("management parameters must match position count")
    if np.any(horizons <= 0):
        raise ValueError("management horizon must be positive")
    if np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError("first exit fraction must be strictly between zero and one")
    if np.any(np.asarray(positions) + horizons > len(source)):
        raise ValueError("incomplete path for requested horizon")
    if backend != "cpu":
        try:
            return _simulate_gpu(
                source,
                positions,
                side,
                horizons,
                target_1,
                target_2,
                stop,
                trailing,
                fractions,
            )
        except (ImportError, RuntimeError):
            if backend == "cuda":
                raise
    return _simulate_cpu(
        source,
        positions,
        side,
        horizons,
        target_1,
        target_2,
        stop,
        trailing,
        fractions,
    )


def _funding_for_actions(actions: pd.DataFrame) -> np.ndarray:
    return base._funding_pnl_bps(
        actions["actual_entry_timestamp"],
        actions["exit_timestamp"],
        actions["side"].to_numpy(int),
        base._funding_curve(),
    )


def _management_name(code: int) -> str:
    return {
        1: "STOP",
        2: "TIMEOUT",
        3: "TARGET_2",
        4: "STOP_GAP",
        5: "TRAIL_STOP",
    }.get(code, "UNKNOWN")


def _label_partition(month: str, fee: FeeContract) -> pd.DataFrame:
    period = pd.Period(month, freq="M")
    start = period.start_time.tz_localize("UTC")
    end = (period + 1).start_time.tz_localize("UTC")
    inherited = _load_parent_actions(start, end)
    if inherited.empty:
        return inherited
    actions = compose_parameterized_plans(inherited, fee.round_trip_bps)
    source = _load_second_window(month)
    positions, _ = _first_observed_positions(source, actions["entry_timestamp"])
    actions = _refine_entry_events(actions, source, positions)
    pieces: list[pd.DataFrame] = []
    groups = list(actions.groupby("side", sort=True))
    for number, (side_value, group) in enumerate(groups, start=1):
        side = int(cast(Any, side_value))
        direction = "LONG" if side > 0 else "SHORT"
        indexes = group.index.to_numpy(int)
        result = simulate_management(
            source,
            positions[indexes],
            side,
            group["horizon_seconds"].to_numpy(int),
            group["target_1_bps"].to_numpy(float),
            group["target_2_bps"].to_numpy(float),
            group["stop_bps"].to_numpy(float),
            group["trailing_bps"].to_numpy(float),
            group["first_exit_fraction"].to_numpy(float),
        )
        labelled = group.copy()
        for name, values in result.items():
            labelled[name] = values
        labelled["outcome"] = [
            _management_name(int(value)) for value in labelled["management_code"]
        ]
        labelled["event"] = [OUTCOME_NAMES[int(value)] for value in labelled["event_class"]]
        labelled["exit_bucket_timestamp"] = pd.to_datetime(
            labelled["entry_bucket_timestamp"], utc=True
        ) + pd.to_timedelta(labelled["exit_seconds"].to_numpy(int) - 1, unit="s")
        labelled["exit_timestamp"] = labelled["exit_bucket_timestamp"] + pd.Timedelta(seconds=1)
        labelled["source_month"] = month
        pieces.append(labelled)
        _status(
            "state_action_labels",
            f"{month} side {number}/{len(groups)}: {direction} parameterized plans",
            12
            + 28 * (ONE_SECOND_MONTHS.index(month) + number / len(groups)) / len(ONE_SECOND_MONTHS),
            month=month,
            action=f"{side}:PARAMETERIZED",
            gpu=_gpu_info(),
        )
    output = pd.concat(pieces, ignore_index=True)
    output = refine_stop_fills_with_ordered_events(output)
    output["funding_bps"] = _funding_for_actions(output)
    output["round_trip_cost_bps"] = fee.round_trip_bps
    output["net_bps"] = output["gross_bps"] + output["funding_bps"] - fee.round_trip_bps
    output["alpha_path_return_bps"] = output["gross_bps"]
    output["proxy_net_bps"] = output["net_bps"]
    output["executable_return_available"] = False
    output["execution_quality"] = "TRADE_PATH_PROXY_NO_HISTORICAL_L2"
    output["stress_1_5x_bps"] = (
        output["gross_bps"] + output["funding_bps"] - 1.5 * fee.round_trip_bps
    )
    output["stress_2x_bps"] = output["gross_bps"] + output["funding_bps"] - 2.0 * fee.round_trip_bps
    leverage = _leverage(output["stop_bps"].to_numpy(float), fee.round_trip_bps)
    output["sized_leverage"] = leverage
    output["sized_portfolio_return"] = leverage * output["net_bps"].to_numpy(float) / 10_000
    if np.any(output["sized_portfolio_return"].to_numpy(float) <= -1):
        raise ValueError("risk-sized state-action can lose all equity")
    output["log_utility"] = np.log1p(output["sized_portfolio_return"].to_numpy(float))
    output["label_protocol_hash"] = LABEL_PROTOCOL_HASH
    output["protocol_hash"] = PROTOCOL_HASH
    if not output["available_at"].le(output["actual_entry_timestamp"]).all():
        raise ValueError("canonical label entered before features were available")
    return output.sort_values(["actual_entry_timestamp", "side"]).reset_index(drop=True)


def local_plan_variants(rows: pd.DataFrame) -> pd.DataFrame:
    """Create a small supported neighborhood around each expert-composed plan."""
    if "local_variant" in rows:
        return rows.copy()
    specifications = (
        ("BASE", None, 1.0),
        ("HORIZON_SHORTER", "horizon_seconds", 0.80),
        ("HORIZON_LONGER", "horizon_seconds", 1.20),
        ("TARGETS_TIGHTER", "targets", 0.85),
        ("TARGETS_WIDER", "targets", 1.15),
        ("STOP_TIGHTER", "stop", 0.85),
        ("STOP_WIDER", "stop", 1.15),
        ("TRAIL_TIGHTER", "trailing_bps", 0.80),
        ("TRAIL_WIDER", "trailing_bps", 1.20),
        ("PARTIAL_SMALLER", "first_exit_fraction", 0.80),
        ("PARTIAL_LARGER", "first_exit_fraction", 1.20),
    )
    pieces: list[pd.DataFrame] = []
    for name, parameter, scale in specifications:
        variant = rows.copy()
        if parameter == "horizon_seconds":
            variant[parameter] = np.clip(
                np.rint(variant[parameter].to_numpy(float) * scale),
                min(PREDICTION_HORIZONS_SECONDS),
                MAXIMUM_HORIZON_SECONDS,
            ).astype(int)
            variant["horizon_fraction"] = variant["horizon_seconds"] / float(
                MAXIMUM_HORIZON_SECONDS
            )
        elif parameter == "targets":
            variant["target_1_bps"] = np.clip(
                variant["target_1_bps"].to_numpy(float) * scale,
                1.0,
                base.MAX_TARGET_BPS - 1.0,
            )
            variant["target_2_bps"] = np.clip(
                np.maximum(
                    variant["target_2_bps"].to_numpy(float) * scale,
                    variant["target_1_bps"].to_numpy(float) + 1.0,
                ),
                2.0,
                base.MAX_TARGET_BPS,
            )
        elif parameter == "stop":
            variant["stop_bps"] = np.clip(
                variant["stop_bps"].to_numpy(float) * scale, 3.0, base.MAX_STOP_BPS
            )
            variant["trailing_bps"] = np.minimum(
                variant["trailing_bps"].to_numpy(float),
                variant["stop_bps"].to_numpy(float),
            )
        elif parameter == "trailing_bps":
            variant[parameter] = np.clip(
                variant[parameter].to_numpy(float) * scale,
                3.0,
                variant["stop_bps"].to_numpy(float),
            )
        elif parameter == "first_exit_fraction":
            variant[parameter] = np.clip(
                variant[parameter].to_numpy(float) * scale,
                0.10,
                0.90,
            )
        variant["local_variant"] = name
        variant["local_parameter_distance"] = abs(math.log(scale)) if parameter else 0.0
        variant["plan_id"] = [
            parameterized_plan_id(
                int(side),
                int(horizon),
                float(target_1),
                float(target_2),
                float(stop),
                float(trailing),
                float(fraction),
                f"{contributors}|{name}",
            )
            for side, horizon, target_1, target_2, stop, trailing, fraction, contributors in zip(
                variant["side"],
                variant["horizon_seconds"],
                variant["target_1_bps"],
                variant["target_2_bps"],
                variant["stop_bps"],
                variant["trailing_bps"],
                variant["first_exit_fraction"],
                variant["plan_contributors"],
                strict=True,
            )
        ]
        variant["expert_id"] = variant["plan_id"]
        pieces.append(variant)
    output = pd.concat(pieces, ignore_index=True)
    return output.drop_duplicates(
        [
            "actual_entry_timestamp",
            "side",
            "horizon_seconds",
            "target_1_bps",
            "target_2_bps",
            "stop_bps",
            "trailing_bps",
            "first_exit_fraction",
        ],
        keep="first",
    ).reset_index(drop=True)


def label_exact_plans(rows: pd.DataFrame, fee: FeeContract) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month, month_rows in rows.groupby("source_month", sort=True):
        source = _load_second_window(str(month))
        positions, _ = _first_observed_positions(source, month_rows["actual_entry_timestamp"])
        month_rows = _refine_entry_events(month_rows, source, positions)
        month_rows["_source_position"] = positions
        side_pieces: list[pd.DataFrame] = []
        for side, side_rows in month_rows.groupby("side", sort=True):
            local_positions = side_rows["_source_position"].to_numpy(np.int64)
            result = simulate_management(
                source,
                local_positions,
                int(cast(Any, side)),
                side_rows["horizon_seconds"].to_numpy(int),
                side_rows["target_1_bps"].to_numpy(float),
                side_rows["target_2_bps"].to_numpy(float),
                side_rows["stop_bps"].to_numpy(float),
                side_rows["trailing_bps"].to_numpy(float),
                side_rows["first_exit_fraction"].to_numpy(float),
            )
            labelled = side_rows.copy()
            for column, values in result.items():
                labelled[column] = values
            labelled = labelled.drop(columns="_source_position")
            side_pieces.append(labelled)
        pieces.append(pd.concat(side_pieces, ignore_index=True))
    output = pd.concat(pieces, ignore_index=True)
    output["outcome"] = [_management_name(int(value)) for value in output["management_code"]]
    output["event"] = [OUTCOME_NAMES[int(value)] for value in output["event_class"]]
    output["exit_bucket_timestamp"] = pd.to_datetime(
        output["entry_bucket_timestamp"], utc=True
    ) + pd.to_timedelta(output["exit_seconds"].to_numpy(int) - 1, unit="s")
    output["exit_timestamp"] = output["exit_bucket_timestamp"] + pd.Timedelta(seconds=1)
    output = refine_stop_fills_with_ordered_events(output)
    output = output.loc[output["data_valid"]].reset_index(drop=True)
    output["funding_bps"] = _funding_for_actions(output)
    output["round_trip_cost_bps"] = fee.round_trip_bps
    output["net_bps"] = output["gross_bps"] + output["funding_bps"] - fee.round_trip_bps
    output["stress_1_5x_bps"] = (
        output["gross_bps"] + output["funding_bps"] - 1.5 * fee.round_trip_bps
    )
    output["stress_2x_bps"] = output["gross_bps"] + output["funding_bps"] - 2.0 * fee.round_trip_bps
    leverage = _leverage(output["stop_bps"].to_numpy(float), fee.round_trip_bps)
    output["sized_leverage"] = leverage
    output["sized_portfolio_return"] = leverage * output["net_bps"].to_numpy(float) / 10_000
    output["log_utility"] = np.log1p(output["sized_portfolio_return"].to_numpy(float))
    output["alpha_path_return_bps"] = output["gross_bps"]
    output["proxy_net_bps"] = output["net_bps"]
    output["executable_return_available"] = False
    output["execution_quality"] = "TRADE_PATH_PROXY_NO_HISTORICAL_L2"
    order = ["actual_entry_timestamp", "side"]
    if "local_variant" in output:
        order.append("local_variant")
    return output.sort_values(order, kind="stable").reset_index(drop=True)


def label_local_plan_variants(rows: pd.DataFrame, fee: FeeContract) -> pd.DataFrame:
    return label_exact_plans(local_plan_variants(rows), fee)


def augment_local_plan_training_support(
    rows: pd.DataFrame,
    fee: FeeContract,
    round_trip_cost_bps: float,
    stop_loss_overrun_reserve_bps: float,
    maximum_states: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Label a bounded, deterministic past-only sample before local plans reach OOS."""
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(rows["actual_entry_timestamp"], utc=True).drop_duplicates()
    )
    count = min(maximum_states, len(timestamps))
    if count == 0:
        return rows.copy(), {"sampled_states": 0, "added_rows": 0}
    positions = np.linspace(0, len(timestamps) - 1, count).astype(int)
    sampled_timestamps = timestamps[np.unique(positions)]
    sampled = rows.loc[rows["actual_entry_timestamp"].isin(sampled_timestamps)].copy()
    variants = label_local_plan_variants(sampled, fee)
    variants = variants.loc[variants["local_variant"].ne("BASE")].reset_index(drop=True)
    base_rows = rows.copy()
    base_rows["local_variant"] = "BASE"
    base_rows["local_parameter_distance"] = 0.0
    output = pd.concat([base_rows, variants], ignore_index=True)
    output = apply_risk_sizing_contract(
        output,
        round_trip_cost_bps,
        stop_loss_overrun_reserve_bps,
    )
    output = output.sort_values(
        ["actual_entry_timestamp", "side", "local_variant"], kind="stable"
    ).reset_index(drop=True)
    return output, {
        "sampled_states": len(sampled_timestamps),
        "added_rows": len(variants),
    }


def train_median_constant_plans(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Build a fixed-management negative control from past-only train medians."""
    parameters = (
        "horizon_seconds",
        "target_1_bps",
        "target_2_bps",
        "stop_bps",
        "trailing_bps",
        "first_exit_fraction",
    )
    output = rows.drop_duplicates(["actual_entry_timestamp", "side"], keep="first").copy()
    medians = reference.groupby("side", sort=False)[list(parameters)].median()
    for side in output["side"].drop_duplicates().to_list():
        if side not in medians.index:
            raise ValueError(f"constant-plan reference is missing side {side}")
        positions = output["side"].eq(side)
        for parameter in parameters:
            output.loc[positions, parameter] = float(cast(Any, medians.at[side, parameter]))
    output["horizon_seconds"] = np.rint(output["horizon_seconds"]).astype(int)
    output["horizon_fraction"] = output["horizon_seconds"] / float(MAXIMUM_HORIZON_SECONDS)
    output["target_2_bps"] = np.maximum(
        output["target_2_bps"].to_numpy(float),
        output["target_1_bps"].to_numpy(float) + 1.0,
    )
    output["trailing_bps"] = np.minimum(
        output["trailing_bps"].to_numpy(float), output["stop_bps"].to_numpy(float)
    )
    output["plan_contributors"] = "TRAIN_MEDIAN_CONSTANT_PLAN"
    output["plan_id"] = [
        parameterized_plan_id(
            int(side),
            int(horizon),
            float(target_1),
            float(target_2),
            float(stop),
            float(trailing),
            float(fraction),
            "TRAIN_MEDIAN_CONSTANT_PLAN",
        )
        for side, horizon, target_1, target_2, stop, trailing, fraction in zip(
            output["side"],
            output["horizon_seconds"],
            output["target_1_bps"],
            output["target_2_bps"],
            output["stop_bps"],
            output["trailing_bps"],
            output["first_exit_fraction"],
            strict=True,
        )
    ]
    output["expert_id"] = output["plan_id"]
    return output


def plan_efficiency_audit(rows: pd.DataFrame, fee: FeeContract) -> dict[str, Any]:
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(rows["actual_entry_timestamp"], utc=True).drop_duplicates()
    )
    if len(timestamps) > LOCAL_PLAN_AUDIT_STATES:
        positions = np.linspace(0, len(timestamps) - 1, LOCAL_PLAN_AUDIT_STATES).astype(int)
        timestamps = timestamps[positions]
    sampled = rows.loc[rows["actual_entry_timestamp"].isin(timestamps)].copy()
    variants = label_local_plan_variants(sampled, fee)
    grouped = variants.groupby(["actual_entry_timestamp", "side"], sort=False)
    best = grouped["net_bps"].max()
    baseline = variants.loc[variants["local_variant"].eq("BASE")].set_index(
        ["actual_entry_timestamp", "side"]
    )["net_bps"]
    regret = best.sub(baseline, fill_value=0.0).clip(lower=0.0)
    winner = variants.loc[variants.groupby(["actual_entry_timestamp", "side"])["net_bps"].idxmax()]
    mean_regret = float(regret.mean())
    return {
        "states": len(grouped),
        "variants": len(variants),
        "mean_local_oracle_regret_bps": mean_regret,
        "median_local_oracle_regret_bps": float(regret.median()),
        "fraction_with_better_local_plan": float(regret.gt(0).mean()),
        "fraction_material_regret": float(regret.gt(LOCAL_PLAN_REGRET_MATERIAL_BPS).mean()),
        "winner_by_variant": winner["local_variant"].value_counts().to_dict(),
        "material": mean_regret > LOCAL_PLAN_REGRET_MATERIAL_BPS,
        "oracle_is_diagnostic_only": True,
    }


def build_state_actions(fee: FeeContract, *, resume: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {}
    for number, month in enumerate(ONE_SECOND_MONTHS, start=1):
        path = LABEL_ROOT / f"month={month}.parquet"
        cached = False
        migrate = False
        if resume and path.exists():
            schema = set(
                duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)])
                .df()["column_name"]
                .astype(str)
            )
            identity_columns = ["protocol_hash", "round_trip_cost_bps"]
            if "label_protocol_hash" in schema:
                identity_columns.append("label_protocol_hash")
            identity = pd.read_parquet(path, columns=identity_columns)
            current_labels = bool(
                "label_protocol_hash" in identity
                and not identity.empty
                and identity["label_protocol_hash"].eq(LABEL_PROTOCOL_HASH).all()
            )
            cached = current_labels
            migrate = cached and (
                not identity["protocol_hash"].eq(PROTOCOL_HASH).all()
                or not np.allclose(
                    identity["round_trip_cost_bps"].to_numpy(float), fee.round_trip_bps
                )
            )
        if cached:
            rows = pd.read_parquet(path)
            if migrate:
                rows["round_trip_cost_bps"] = fee.round_trip_bps
                rows["net_bps"] = rows["gross_bps"] + rows["funding_bps"] - fee.round_trip_bps
                rows["alpha_path_return_bps"] = rows["gross_bps"]
                rows["proxy_net_bps"] = rows["net_bps"]
                rows["executable_return_available"] = False
                rows["execution_quality"] = "TRADE_PATH_PROXY_NO_HISTORICAL_L2"
                rows["stress_1_5x_bps"] = (
                    rows["gross_bps"] + rows["funding_bps"] - 1.5 * fee.round_trip_bps
                )
                rows["stress_2x_bps"] = (
                    rows["gross_bps"] + rows["funding_bps"] - 2.0 * fee.round_trip_bps
                )
                leverage = _leverage(rows["stop_bps"].to_numpy(float), fee.round_trip_bps)
                rows["sized_leverage"] = leverage
                rows["sized_portfolio_return"] = leverage * rows["net_bps"].to_numpy(float) / 10_000
                if np.any(rows["sized_portfolio_return"].to_numpy(float) <= -1):
                    raise ValueError("risk-sized cached state-action can lose all equity")
                rows["log_utility"] = np.log1p(rows["sized_portfolio_return"].to_numpy(float))
                rows["label_protocol_hash"] = LABEL_PROTOCOL_HASH
                rows["protocol_hash"] = PROTOCOL_HASH
                temporary = path.with_suffix(f".parquet.{os.getpid()}.tmp")
                rows.to_parquet(temporary, index=False)
                _atomic_replace(temporary, path)
        else:
            rows = _label_partition(month, fee)
            temporary = path.with_suffix(f".parquet.{os.getpid()}.tmp")
            rows.to_parquet(temporary, index=False)
            _atomic_replace(temporary, path)
        pieces.append(rows)
        manifest[month] = {"rows": len(rows), "path": str(path), "sha256": _sha256(path)}
        _status(
            "state_action_matrix",
            f"month {number}/{len(ONE_SECOND_MONTHS)} complete: {month}",
            12 + 28 * number / len(ONE_SECOND_MONTHS),
            month=month,
            blocks_completed=number,
            blocks_total=len(ONE_SECOND_MONTHS),
            rows=sum(len(piece) for piece in pieces),
        )
    matrix = pd.concat(pieces, ignore_index=True)
    matrix = matrix.loc[matrix["data_valid"].astype(bool)].reset_index(drop=True)
    return matrix, manifest


def _gpu_info() -> dict[str, Any]:
    try:
        import cupy as cp

        device = cp.cuda.Device()
        free, total = device.mem_info
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "available": True,
            "device": int(device.id),
            "name": str(name),
            "free_gb": round(free / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
        }
    except (ImportError, RuntimeError):
        return {"available": False}


def _x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, MODEL_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("model features must be finite")
    return values


def _generator_x(rows: pd.DataFrame) -> np.ndarray:
    values = rows.loc[:, GENERATOR_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("fold-local generator features must be finite")
    return values


def _leaf_matrix(model: Any, values: np.ndarray) -> np.ndarray:
    leaves = np.asarray(model.apply(values), dtype=np.int32)
    if leaves.ndim == 1:
        leaves = leaves[:, None]
    if leaves.ndim != 2 or len(leaves) != len(values):
        raise ValueError("invalid expert leaf matrix")
    return leaves


def _new_expert_generator(seed: int) -> Any:
    return discovery._generator(seed)


def _fold_scope(fold_number: int | str, fit: pd.DataFrame) -> str:
    timestamp = pd.to_datetime(fit["actual_entry_timestamp"], utc=True)
    identity = {
        "protocol_hash": PROTOCOL_HASH,
        "fold": str(fold_number),
        "fit_start": timestamp.min().isoformat(),
        "fit_end": timestamp.max().isoformat(),
        "fit_rows": len(fit),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _leaf_statistics(
    leaves: np.ndarray,
    net_bps: np.ndarray,
    event_class: np.ndarray,
    *,
    fold_scope: str,
    side: int,
    horizon: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    statistics: list[dict[str, np.ndarray]] = []
    catalog: list[dict[str, Any]] = []
    for tree in range(leaves.shape[1]):
        tree_leaves = leaves[:, tree]
        maximum_leaf = int(tree_leaves.max(initial=0))
        arrays = {
            name: np.full(maximum_leaf + 1, np.nan, dtype=float)
            for name in (
                "count",
                "mean",
                "q90",
                "lcb",
                "positive_fraction",
                "target_rate",
                "stop_rate",
                "eligible",
            )
        }
        for leaf in np.unique(tree_leaves):
            active = tree_leaves == leaf
            values = net_bps[active]
            count = len(values)
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
            lcb = mean - 1.645 * standard_deviation / math.sqrt(max(count, 1))
            eligible = count >= MINIMUM_EXPERT_OPPORTUNITIES
            leaf_index = int(leaf)
            arrays["count"][leaf_index] = count
            arrays["mean"][leaf_index] = mean
            arrays["q90"][leaf_index] = float(np.quantile(values, 0.90))
            arrays["lcb"][leaf_index] = lcb
            arrays["positive_fraction"][leaf_index] = float(np.mean(values > 0))
            arrays["target_rate"][leaf_index] = float(
                np.mean(event_class[active] == OUTCOME_TARGET)
            )
            arrays["stop_rate"][leaf_index] = float(np.mean(event_class[active] == OUTCOME_STOP))
            arrays["eligible"][leaf_index] = float(eligible)
            catalog.append(
                {
                    "expert_id": fold_expert_id(fold_scope, side, horizon, tree, leaf_index),
                    "fold_scope": fold_scope,
                    "side": side,
                    "horizon_seconds": horizon,
                    "tree": tree,
                    "leaf": leaf_index,
                    "managed_outcomes_evaluated": True,
                    "opportunities": count,
                    "managed_expectancy_bps": mean,
                    "managed_q90_bps": arrays["q90"][leaf_index],
                    "managed_lcb_bps": lcb,
                    "managed_positive_fraction": arrays["positive_fraction"][leaf_index],
                    "managed_target_rate": arrays["target_rate"][leaf_index],
                    "managed_stop_rate": arrays["stop_rate"][leaf_index],
                    "eligible_after_managed_evaluation": eligible,
                    "elimination_reason": None if eligible else "INSUFFICIENT_MANAGED_SUPPORT",
                    "terminal_return_prefilter": False,
                }
            )
        statistics.append(arrays)
    return statistics, catalog


def fit_fold_expert_library(
    fit: pd.DataFrame,
    fold_number: int | str,
    *,
    resume: bool = False,
    progress: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Generate all leaf experts on exact managed labels from the fold fit only."""
    scope = _fold_scope(fold_number, fit)
    cache = EXPERT_CATALOG_ROOT / f"{scope}.joblib"
    catalog_path = EXPERT_CATALOG_ROOT / f"{scope}.parquet"
    if resume and cache.exists() and catalog_path.exists():
        library = cast(dict[str, Any], joblib.load(cache))
        if library.get("protocol_hash") == PROTOCOL_HASH:
            return library
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    catalog_rows: list[dict[str, Any]] = []
    actions = [(int(side), 0) for side in sorted(fit["side"].unique())]
    for action_number, (side, horizon) in enumerate(actions, start=1):
        active = fit.loc[fit["side"].eq(side)].copy()
        model = _new_expert_generator(20261001 + action_number)
        values = _generator_x(active)
        model.fit(values, active["net_bps"].to_numpy(float))
        leaves = _leaf_matrix(model, values)
        statistics, catalog = _leaf_statistics(
            leaves,
            active["net_bps"].to_numpy(float),
            active["event_class"].to_numpy(int),
            fold_scope=scope,
            side=side,
            horizon=horizon,
        )
        catalog_rows.extend(catalog)
        groups[(side, horizon)] = {
            "model": model,
            "statistics": statistics,
            "fallback": {
                "mean": float(active["net_bps"].mean()),
                "q90": float(active["net_bps"].quantile(0.90)),
                "lcb": float(active["net_bps"].mean()),
                "positive_fraction": float(active["net_bps"].gt(0).mean()),
                "target_rate": float(active["event_class"].eq(OUTCOME_TARGET).mean()),
                "stop_rate": float(active["event_class"].eq(OUTCOME_STOP).mean()),
                "count": len(active),
            },
        }
        if progress is not None:
            completed, total = progress
            _status(
                "fold_expert_generation",
                f"fold {fold_number} side {action_number}/{len(actions)}: "
                f"{'LONG' if side > 0 else 'SHORT'} parameterized critic contexts",
                42 + 20 * (completed + action_number / len(actions)) / max(total, 1),
                fold=str(fold_number),
                action=f"{side}:PARAMETERIZED",
                candidates_generated=len(catalog_rows),
                gpu=_gpu_info(),
            )
    catalog_frame = pd.DataFrame(catalog_rows)
    EXPERT_CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_catalog = catalog_path.with_suffix(f".parquet.{os.getpid()}.tmp")
    catalog_frame.to_parquet(temporary_catalog, index=False)
    _atomic_replace(temporary_catalog, catalog_path)
    library = {
        "protocol_hash": PROTOCOL_HASH,
        "fold_scope": scope,
        "groups": groups,
        "catalog_path": str(catalog_path),
        "candidates_evaluated": len(catalog_frame),
        "candidates_eligible": int(catalog_frame["eligible_after_managed_evaluation"].sum()),
        "terminal_prefilter_rejections": 0,
    }
    _atomic_joblib(cache, library)
    return library


def apply_fold_expert_library(rows: pd.DataFrame, library: dict[str, Any]) -> pd.DataFrame:
    """Map every state-action to the active managed experts without reading future outcomes."""
    output = rows.copy()
    size = len(output)
    columns = {name: np.full(size, np.nan, dtype=float) for name in FOLD_EXPERT_FEATURES}
    best_tree = np.full(size, -1, dtype=np.int16)
    best_leaf = np.full(size, -1, dtype=np.int16)
    for (side, _horizon), group in library["groups"].items():
        positions = np.flatnonzero(output["side"].eq(side).to_numpy())
        if not len(positions):
            continue
        active = output.iloc[positions]
        values = _generator_x(active)
        model = group["model"]
        leaves = _leaf_matrix(model, values)
        count = np.zeros(len(active), dtype=float)
        mean_sum = np.zeros(len(active), dtype=float)
        mean_square_sum = np.zeros(len(active), dtype=float)
        q90_sum = np.zeros(len(active), dtype=float)
        positive_sum = np.zeros(len(active), dtype=float)
        target_sum = np.zeros(len(active), dtype=float)
        stop_sum = np.zeros(len(active), dtype=float)
        log_opportunity_sum = np.zeros(len(active), dtype=float)
        winning_lcb = np.full(len(active), -np.inf, dtype=float)
        winning_tree = np.full(len(active), -1, dtype=np.int16)
        winning_leaf = np.full(len(active), -1, dtype=np.int16)
        for tree, statistics in enumerate(group["statistics"]):
            leaf = leaves[:, tree]
            within = leaf < len(statistics["eligible"])
            eligible = np.zeros(len(active), dtype=bool)
            eligible[within] = statistics["eligible"][leaf[within]] == 1
            if not eligible.any():
                continue
            indexes = leaf[eligible]
            managed_mean = statistics["mean"][indexes]
            count[eligible] += 1
            mean_sum[eligible] += managed_mean
            mean_square_sum[eligible] += managed_mean**2
            q90_sum[eligible] += statistics["q90"][indexes]
            positive_sum[eligible] += statistics["positive_fraction"][indexes]
            target_sum[eligible] += statistics["target_rate"][indexes]
            stop_sum[eligible] += statistics["stop_rate"][indexes]
            log_opportunity_sum[eligible] += np.log1p(statistics["count"][indexes])
            lcb = statistics["lcb"][indexes]
            better = np.zeros(len(active), dtype=bool)
            better[eligible] = lcb > winning_lcb[eligible]
            winning_lcb[better] = statistics["lcb"][leaf[better]]
            winning_tree[better] = tree
            winning_leaf[better] = leaf[better]
        fallback = group["fallback"]
        missing = count == 0
        divisor = np.maximum(count, 1)
        managed_mean = mean_sum / divisor
        managed_mean[missing] = fallback["mean"]
        dispersion = np.sqrt(np.maximum(0.0, mean_square_sum / divisor - managed_mean**2))
        q90 = q90_sum / divisor
        positive = positive_sum / divisor
        target = target_sum / divisor
        stop = stop_sum / divisor
        log_opportunities = log_opportunity_sum / divisor
        q90[missing] = fallback["q90"]
        positive[missing] = fallback["positive_fraction"]
        target[missing] = fallback["target_rate"]
        stop[missing] = fallback["stop_rate"]
        log_opportunities[missing] = math.log1p(fallback["count"])
        winning_lcb[missing] = fallback["lcb"]
        columns["managed_expert_mean_bps"][positions] = managed_mean
        columns["managed_expert_q90_bps"][positions] = q90
        columns["managed_expert_best_lcb_bps"][positions] = winning_lcb
        columns["managed_expert_dispersion_bps"][positions] = dispersion
        columns["managed_expert_positive_fraction"][positions] = positive
        columns["managed_expert_target_rate"][positions] = target
        columns["managed_expert_stop_rate"][positions] = stop
        columns["managed_expert_log_opportunities"][positions] = log_opportunities
        columns["managed_generator_score_bps"][positions] = np.asarray(
            model.predict(values), dtype=float
        )
        best_tree[positions] = winning_tree
        best_leaf[positions] = winning_leaf
    for name, values in columns.items():
        if not np.isfinite(values).all():
            raise ValueError(f"missing fold-local expert feature: {name}")
        output[name] = values
    output["fold_expert_scope"] = str(library["fold_scope"])
    output["expert_tree_index"] = best_tree
    output["expert_leaf_id"] = best_leaf
    return output


def cross_fit_fold_expert_features(
    fit: pd.DataFrame,
    fold_number: int | str,
    *,
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Encode fit rows with critics that have only seen strictly earlier outcomes."""
    if fit.empty:
        raise ValueError("cannot cross-fit an empty outer-fold fit")
    ordered = fit.sort_values("actual_entry_timestamp", kind="stable").reset_index(drop=True)
    timestamp = pd.to_datetime(ordered["actual_entry_timestamp"], utc=True)
    start = timestamp.min().floor("D")
    end = timestamp.max().ceil("D") + pd.Timedelta(days=1)
    first_block = start + pd.Timedelta(weeks=CROSSFIT_WARMUP_WEEKS)
    scope = _fold_scope(f"{fold_number}-crossfit", ordered)
    cache = EXPERT_CATALOG_ROOT / f"{scope}.crossfit.parquet"
    keys = ["actual_entry_timestamp", "side", "plan_id"]
    encoded_columns = [
        *keys,
        *FOLD_EXPERT_FEATURES,
        "fold_expert_scope",
        "expert_tree_index",
        "expert_leaf_id",
    ]
    final_library = fit_fold_expert_library(ordered, fold_number, resume=resume)
    if resume and cache.exists():
        encoded = pd.read_parquet(cache)
        transformed = ordered.merge(encoded, on=keys, how="inner", validate="one_to_one")
        if (
            transformed.empty
            or not np.isfinite(transformed.loc[:, FOLD_EXPERT_FEATURES].to_numpy(float)).all()
        ):
            raise ValueError("invalid cached cross-fitted critic features")
        return (
            transformed,
            final_library,
            {
                "scope": scope,
                "cache": str(cache),
                "rows": len(transformed),
                "blocks": int(encoded["fold_expert_scope"].nunique()),
                "strictly_past_only": True,
            },
        )

    pieces: list[pd.DataFrame] = []
    block_start = first_block
    block_number = 0
    while block_start < end:
        block_end = min(block_start + pd.Timedelta(weeks=CROSSFIT_BLOCK_WEEKS), end)
        history = _period(ordered, None, block_start, purge_exit=True)
        active = timestamp.ge(block_start) & timestamp.lt(block_end)
        held_out = ordered.loc[active].copy()
        if not history.empty and not held_out.empty:
            block_number += 1
            library = fit_fold_expert_library(
                history,
                f"{fold_number}-crossfit-{block_number}",
                resume=resume,
            )
            transformed = apply_fold_expert_library(held_out, library)
            pieces.append(transformed)
            _status(
                "critic_crossfit",
                f"fold {fold_number} block {block_number}: past-only critic encoding",
                42,
                fold=str(fold_number),
                crossfit_block=block_number,
                history_rows=len(history),
                held_out_rows=len(held_out),
                strictly_past_only=True,
            )
        block_start = block_end
    if not pieces:
        raise ValueError("insufficient chronology for critic cross-fitting")
    transformed = pd.concat(pieces, ignore_index=True).sort_values(
        "actual_entry_timestamp", kind="stable"
    )
    EXPERT_CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".parquet.{os.getpid()}.tmp")
    transformed.loc[:, encoded_columns].to_parquet(temporary, index=False)
    _atomic_replace(temporary, cache)
    return (
        transformed,
        final_library,
        {
            "scope": scope,
            "cache": str(cache),
            "rows": len(transformed),
            "blocks": block_number,
            "strictly_past_only": True,
            "warmup_rows_excluded": len(ordered) - len(transformed),
        },
    )


def _timestamp_weights(rows: pd.DataFrame) -> np.ndarray:
    count = rows.groupby("actual_entry_timestamp")["actual_entry_timestamp"].transform("size")
    return np.asarray(1.0 / count.to_numpy(float), dtype=float)


def _permute_outcomes(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = rows.copy()
    columns = [
        "event_class",
        "net_bps",
        "log_utility",
        "mfe_bps",
        "mae_bps",
        "time_to_target_seconds",
        "exit_seconds",
    ]
    permutation = np.random.default_rng(seed).permutation(len(output))
    output.loc[:, columns] = output.loc[:, columns].to_numpy()[permutation]
    return output


def _probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=float)
    result = np.zeros((len(values), 3), dtype=float)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    return result


def _classifier(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, random_state=seed),
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda",
        n_estimators=280,
        learning_rate=0.035,
        max_depth=5,
        min_child_weight=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=30.0,
        n_jobs=4,
        random_state=seed,
    )


def _regressor(kind: str, seed: int) -> Predictor:
    if kind == "ridge":
        return cast(Predictor, make_pipeline(StandardScaler(), Ridge(alpha=20.0)))
    return cast(
        Predictor,
        XGBRegressor(
            objective="reg:pseudohubererror",
            tree_method="hist",
            device="cuda",
            n_estimators=280,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=30.0,
            n_jobs=4,
            random_state=seed,
        ),
    )


def _fit_classifier(
    kind: str,
    model: Any,
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Any:
    if kind == "ridge":
        return model.fit(values, target, logisticregression__sample_weight=weights)
    return model.fit(values, target, sample_weight=weights)


def _fit_regressor(
    kind: str,
    model: Predictor,
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Predictor:
    if kind == "ridge":
        return model.fit(values, target, ridge__sample_weight=weights)
    return model.fit(values, target, sample_weight=weights)


def fit_probability_head(kind: str, fit: pd.DataFrame) -> dict[str, Any]:
    values = _x(fit)
    weights = _timestamp_weights(fit)
    outcome = fit["event_class"].to_numpy(int)
    classifier = _fit_classifier(kind, _classifier(kind, MODEL_SEEDS[0]), values, outcome, weights)
    conditional: dict[int, Predictor] = {}
    conditional_utility: dict[int, Predictor] = {}
    for event_class in range(3):
        active = outcome == event_class
        if active.sum() < 100:
            raise ValueError(f"insufficient {OUTCOME_NAMES[event_class]} labels")
        model = _fit_regressor(
            kind,
            _regressor(kind, MODEL_SEEDS[event_class]),
            values[active],
            fit.loc[active, "net_bps"].to_numpy(float),
            weights[active],
        )
        conditional[event_class] = model
        conditional_utility[event_class] = _fit_regressor(
            kind,
            _regressor(kind, MODEL_SEEDS[event_class] + 30),
            values[active],
            fit.loc[active, "log_utility"].to_numpy(float),
            weights[active],
        )
    auxiliaries: dict[str, Any] = {}
    for number, target in enumerate(("mfe_bps", "mae_bps"), start=10):
        auxiliaries[target] = _fit_regressor(
            kind,
            _regressor(kind, MODEL_SEEDS[0] + number),
            values,
            fit[target].to_numpy(float),
            weights,
        )
    target_rows = outcome == OUTCOME_TARGET
    auxiliaries["time_to_target_seconds"] = _fit_regressor(
        kind,
        _regressor(kind, MODEL_SEEDS[0] + 20),
        values[target_rows],
        fit.loc[target_rows, "time_to_target_seconds"].to_numpy(float),
        weights[target_rows],
    )
    auxiliaries["exit_seconds"] = _fit_regressor(
        kind,
        _regressor(kind, MODEL_SEEDS[0] + 21),
        values,
        fit["exit_seconds"].to_numpy(float),
        weights,
    )
    direct = {
        "net_bps": _fit_regressor(
            kind,
            _regressor(kind, MODEL_SEEDS[0] + 40),
            values,
            fit["net_bps"].to_numpy(float),
            weights,
        ),
        "log_utility": _fit_regressor(
            kind,
            _regressor(kind, MODEL_SEEDS[0] + 41),
            values,
            fit["log_utility"].to_numpy(float),
            weights,
        ),
    }
    return {
        "kind": kind,
        "classifier": classifier,
        "conditional": conditional,
        "conditional_utility": conditional_utility,
        "direct": direct,
        "aux": auxiliaries,
    }


def fit_calibration(head: dict[str, Any], calibration: pd.DataFrame) -> dict[str, Any]:
    values = _x(calibration)
    raw_probability = np.clip(_probabilities(head["classifier"], values), 1e-6, 1.0)
    outcome = calibration["event_class"].to_numpy(int)
    probability = LogisticRegression(C=1.0, max_iter=2_000, random_state=20260831).fit(
        np.log(raw_probability), outcome
    )
    calibrated_probability = _probabilities(probability, np.log(raw_probability))
    conditional_bps = np.column_stack(
        [np.asarray(head["conditional"][event].predict(values), dtype=float) for event in range(3)]
    )
    conditional_utility = np.column_stack(
        [
            np.asarray(head["conditional_utility"][event].predict(values), dtype=float)
            for event in range(3)
        ]
    )
    raw_values = {
        "decomposed": {
            "net_bps": np.sum(calibrated_probability * conditional_bps, axis=1),
            "log_utility": np.sum(calibrated_probability * conditional_utility, axis=1),
        },
        "direct": {
            "net_bps": np.asarray(head["direct"]["net_bps"].predict(values), dtype=float),
            "log_utility": np.asarray(head["direct"]["log_utility"].predict(values), dtype=float),
        },
    }
    value = {
        name: {
            "net_bps": IsotonicRegression(out_of_bounds="clip").fit(
                raw["net_bps"], calibration["net_bps"].to_numpy(float)
            ),
            "log_utility": IsotonicRegression(out_of_bounds="clip").fit(
                raw["log_utility"], calibration["log_utility"].to_numpy(float)
            ),
        }
        for name, raw in raw_values.items()
    }
    residuals: dict[str, list[float]] = {}
    for name in ("mfe_bps", "mae_bps"):
        predicted = np.asarray(head["aux"][name].predict(values), dtype=float)
        residual = calibration[name].to_numpy(float) - predicted
        residuals[name] = [float(np.quantile(residual, value)) for value in (0.10, 0.50, 0.90)]
    return {"probability": probability, "value": value, "residual_quantiles": residuals}


def score_actions(
    rows: pd.DataFrame,
    head: dict[str, Any],
    calibration: dict[str, Any],
    value_head: str = "decomposed",
) -> pd.DataFrame:
    if value_head not in VALUE_HEADS:
        raise ValueError(f"unknown value head: {value_head}")
    values = _x(rows)
    raw_probability = np.clip(_probabilities(head["classifier"], values), 1e-6, 1.0)
    probability = _probabilities(calibration["probability"], np.log(raw_probability))
    conditional_bps = np.column_stack(
        [np.asarray(head["conditional"][event].predict(values), dtype=float) for event in range(3)]
    )
    conditional_utility = np.column_stack(
        [
            np.asarray(head["conditional_utility"][event].predict(values), dtype=float)
            for event in range(3)
        ]
    )
    raw_values = {
        "decomposed": {
            "net_bps": np.sum(probability * conditional_bps, axis=1),
            "log_utility": np.sum(probability * conditional_utility, axis=1),
        },
        "direct": {
            "net_bps": np.asarray(head["direct"]["net_bps"].predict(values), dtype=float),
            "log_utility": np.asarray(head["direct"]["log_utility"].predict(values), dtype=float),
        },
    }
    output = rows.copy()
    output["p_target"] = probability[:, OUTCOME_TARGET]
    output["p_stop"] = probability[:, OUTCOME_STOP]
    output["p_timeout"] = probability[:, OUTCOME_TIMEOUT]
    output["target_probability"] = output["p_target"]
    for event, name in enumerate(OUTCOME_NAMES):
        output[f"expected_{name.lower()}_net_bps"] = conditional_bps[:, event]
        output[f"expected_{name.lower()}_log_utility"] = conditional_utility[:, event]
    for name, raw in raw_values.items():
        output[f"raw_{name}_ev_bps"] = raw["net_bps"]
        output[f"{name}_ev_bps"] = calibration["value"][name]["net_bps"].predict(raw["net_bps"])
        output[f"raw_{name}_log_utility"] = raw["log_utility"]
        unconstrained = calibration["value"][name]["log_utility"].predict(raw["log_utility"])
        output[f"unconstrained_{name}_log_utility"] = unconstrained
        leverage = (
            output["sized_leverage"].to_numpy(float)
            if "sized_leverage" in output
            else np.ones(len(output), dtype=float)
        )
        ev_implied_return = np.maximum(
            leverage * output[f"{name}_ev_bps"].to_numpy(float) / 10_000,
            -0.999999,
        )
        coherent = np.minimum(
            unconstrained,
            np.log1p(ev_implied_return),
        )
        output[f"{name}_utility_consistency_clipped"] = unconstrained > coherent
        output[f"{name}_log_utility"] = coherent
    output["selected_value_head"] = value_head
    output["raw_ev_bps"] = output[f"raw_{value_head}_ev_bps"]
    output["calibrated_ev_bps"] = output[f"{value_head}_ev_bps"]
    output["expected_log_utility"] = output[f"{value_head}_log_utility"]
    horizon = (
        output["horizon_seconds"].to_numpy(float)
        if "horizon_seconds" in output
        else np.full(len(output), MAXIMUM_HORIZON_SECONDS, dtype=float)
    )
    output["expected_time_to_target_seconds"] = np.clip(
        np.asarray(head["aux"]["time_to_target_seconds"].predict(values), dtype=float),
        1.0,
        horizon,
    )
    output["expected_holding_seconds"] = np.clip(
        np.asarray(head["aux"]["exit_seconds"].predict(values), dtype=float),
        1.0,
        horizon,
    )
    output["expected_ev_bps_per_minute"] = (
        output["calibrated_ev_bps"] * 60 / output["expected_holding_seconds"]
    )
    output["expected_log_utility_per_hour"] = (
        output["expected_log_utility"] * 3_600 / output["expected_holding_seconds"]
    )
    for name in ("mfe_bps", "mae_bps"):
        center = np.asarray(head["aux"][name].predict(values), dtype=float)
        for quantile, residual in zip(
            (10, 50, 90), calibration["residual_quantiles"][name], strict=True
        ):
            output[f"predicted_{name.removesuffix('_bps')}_q{quantile}_bps"] = np.maximum(
                center + residual, 0.0
            )
    return output


def _multiclass_brier(truth: np.ndarray, probability: np.ndarray) -> float:
    observed = np.eye(3, dtype=float)[truth]
    return float(np.mean(np.sum((probability - observed) ** 2, axis=1)))


def _calibration_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    if not len(actual):
        return float("inf")
    bins = pd.qcut(pd.Series(predicted), q=min(10, len(np.unique(predicted))), duplicates="drop")
    values = pd.DataFrame({"actual": actual, "predicted": predicted, "bin": bins})
    grouped = values.groupby("bin", observed=True).agg(
        actual=("actual", "mean"), predicted=("predicted", "mean"), size=("actual", "size")
    )
    return float(
        np.average(np.abs(grouped["actual"] - grouped["predicted"]), weights=grouped["size"])
    )


def _decision_regret(rows: pd.DataFrame, prediction_column: str, actual_column: str) -> float:
    values: list[float] = []
    for _, group in rows.groupby("actual_entry_timestamp", sort=False):
        prediction = group[prediction_column].to_numpy(float)
        actual = group[actual_column].to_numpy(float)
        choice = int(np.argmax(prediction))
        selected = actual[choice] if prediction[choice] > 0 else 0.0
        values.append(max(0.0, float(actual.max())) - selected)
    return float(np.mean(values)) if values else float("inf")


def head_metrics(rows: pd.DataFrame, value_head: str) -> dict[str, float]:
    probability = rows.loc[:, ["p_target", "p_stop", "p_timeout"]].to_numpy(float)
    actual = rows["net_bps"].to_numpy(float)
    predicted = rows[f"{value_head}_ev_bps"].to_numpy(float)
    utility = rows["log_utility"].to_numpy(float)
    predicted_utility = rows[f"{value_head}_log_utility"].to_numpy(float)
    return {
        "brier": _multiclass_brier(rows["event_class"].to_numpy(int), probability),
        "ev_calibration_error_bps": _calibration_error(actual, predicted),
        "ev_mae_bps": float(mean_absolute_error(actual, predicted)),
        "decision_regret_bps": _decision_regret(rows, f"{value_head}_ev_bps", "net_bps"),
        "utility_calibration_error": _calibration_error(utility, predicted_utility),
        "utility_mae": float(mean_absolute_error(utility, predicted_utility)),
        "decision_regret_log_utility": _decision_regret(
            rows, f"{value_head}_log_utility", "log_utility"
        ),
    }


def choose_value_head(metrics: dict[str, dict[str, float]]) -> str:
    baseline = metrics["decomposed"]
    challenger = metrics["direct"]
    keys = (
        "ev_calibration_error_bps",
        "ev_mae_bps",
        "decision_regret_bps",
        "utility_calibration_error",
        "utility_mae",
        "decision_regret_log_utility",
    )
    return "direct" if all(challenger[key] < baseline[key] for key in keys) else "decomposed"


def choose_champion(metrics: dict[str, dict[str, float]]) -> str:
    if "xgboost_cuda" not in metrics:
        return "ridge"
    ridge = metrics["ridge"]
    challenger = metrics["xgboost_cuda"]
    keys = (
        "brier",
        "ev_calibration_error_bps",
        "ev_mae_bps",
        "decision_regret_bps",
        "utility_calibration_error",
        "utility_mae",
        "decision_regret_log_utility",
    )
    return "xgboost_cuda" if all(challenger[key] < ridge[key] for key in keys) else "ridge"


def _selected_action_positions(
    rows: pd.DataFrame, value_column: str = "expected_log_utility"
) -> np.ndarray:
    """Return one deterministic predicted winner per state, without reading its outcome."""
    if rows.empty:
        return np.asarray([], dtype=np.int64)
    ranking = pd.DataFrame(
        {
            "position": np.arange(len(rows), dtype=np.int64),
            "timestamp": pd.to_datetime(rows["actual_entry_timestamp"], utc=True).to_numpy(),
            "value": rows[value_column].to_numpy(float),
            "p_target": rows["p_target"].to_numpy(float),
            "expert_id": rows["expert_id"].astype(str).to_numpy(),
        }
    )
    return (
        ranking.sort_values(
            ["timestamp", "value", "p_target", "expert_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .drop_duplicates("timestamp", keep="first")["position"]
        .to_numpy(np.int64)
    )


def fit_post_selection_calibration(rows: pd.DataFrame) -> dict[str, Any]:
    """Calibrate the action that the policy would select, not the unused action rows."""
    positions = _selected_action_positions(rows)
    selected = rows.iloc[positions]
    if len(selected) < 100:
        raise ValueError("insufficient winner-only calibration states")
    raw_ev = selected["calibrated_ev_bps"].to_numpy(float)
    raw_utility = selected["expected_log_utility"].to_numpy(float)
    realized_ev = selected["net_bps"].to_numpy(float)
    realized_utility = selected["log_utility"].to_numpy(float)
    ev_model = IsotonicRegression(out_of_bounds="clip").fit(raw_ev, realized_ev)
    utility_model = IsotonicRegression(out_of_bounds="clip").fit(raw_utility, realized_utility)
    calibrated_ev = np.asarray(ev_model.predict(raw_ev), dtype=float)
    calibrated_utility = np.asarray(utility_model.predict(raw_utility), dtype=float)
    timestamp = pd.to_datetime(selected["actual_entry_timestamp"], utc=True)
    grouped_oracle = rows.groupby("actual_entry_timestamp", sort=False)["net_bps"].max()
    oracle = timestamp.map(grouped_oracle).to_numpy(float)
    audit = {
        "strictly_past_only": True,
        "rows": len(rows),
        "selected_states": len(selected),
        "start": timestamp.min().isoformat(),
        "end": timestamp.max().isoformat(),
        "preselection_predicted_ev_bps": float(np.mean(raw_ev)),
        "realized_selected_ev_bps": float(np.mean(realized_ev)),
        "winner_optimism_bps": float(np.mean(raw_ev - realized_ev)),
        "preselection_calibration_error_bps": _calibration_error(realized_ev, raw_ev),
        "postselection_calibration_error_bps": _calibration_error(realized_ev, calibrated_ev),
        "preselection_utility_calibration_error": _calibration_error(realized_utility, raw_utility),
        "postselection_utility_calibration_error": _calibration_error(
            realized_utility, calibrated_utility
        ),
        "selected_best_realized_action_fraction": float(np.mean(realized_ev >= oracle - 1e-12)),
    }
    return {"net_bps": ev_model, "log_utility": utility_model, "audit": audit}


def apply_post_selection_calibration(
    rows: pd.DataFrame, calibration: dict[str, Any]
) -> pd.DataFrame:
    """Expose only the causally selected winner and correct its post-selection optimism."""
    output = rows.copy()
    positions = _selected_action_positions(output)
    output["preselection_calibrated_ev_bps"] = output["calibrated_ev_bps"]
    output["preselection_expected_log_utility"] = output["expected_log_utility"]
    output["post_selection_candidate"] = False
    output["post_selection_calibrated_ev_bps"] = np.nan
    output["post_selection_expected_log_utility"] = -1.0
    if not len(positions):
        output["expected_log_utility"] = -1.0
        return output
    raw_ev = output.iloc[positions]["calibrated_ev_bps"].to_numpy(float)
    raw_utility = output.iloc[positions]["expected_log_utility"].to_numpy(float)
    calibrated_ev = np.asarray(calibration["net_bps"].predict(raw_ev), dtype=float)
    unconstrained_utility = np.asarray(calibration["log_utility"].predict(raw_utility), dtype=float)
    leverage = output.iloc[positions]["sized_leverage"].to_numpy(float)
    coherent_upper = np.log1p(np.maximum(leverage * calibrated_ev / 10_000, -0.999999))
    calibrated_utility = np.minimum(unconstrained_utility, coherent_upper)
    selected_index = output.index[positions]
    output.loc[selected_index, "post_selection_candidate"] = True
    output.loc[selected_index, "post_selection_calibrated_ev_bps"] = calibrated_ev
    output.loc[selected_index, "post_selection_expected_log_utility"] = calibrated_utility
    output.loc[selected_index, "calibrated_ev_bps"] = calibrated_ev
    output["expected_log_utility"] = -1.0
    output.loc[selected_index, "expected_log_utility"] = calibrated_utility
    output["expected_ev_bps_per_minute"] = (
        output["calibrated_ev_bps"] * 60 / output["expected_holding_seconds"]
    )
    output["expected_log_utility_per_hour"] = (
        output["expected_log_utility"] * 3_600 / output["expected_holding_seconds"]
    )
    return output


def _fit_immediate_backup(rows: pd.DataFrame, seed: int = 20261400) -> Predictor:
    return _fit_regressor(
        "ridge",
        _regressor("ridge", seed),
        _x(rows),
        rows["log_utility"].to_numpy(float),
        _timestamp_weights(rows),
    )


def _fit_double_immediate_backup(rows: pd.DataFrame) -> tuple[Predictor, Predictor]:
    """Fit independent temporal estimators for Double-Q selection/evaluation."""
    timestamp = pd.to_datetime(rows["actual_entry_timestamp"], utc=True)
    week = ((timestamp - timestamp.min()).dt.total_seconds() // (7 * 86_400)).astype(int)
    left = rows.loc[week.mod(2).eq(0)]
    right = rows.loc[week.mod(2).eq(1)]
    if min(len(left), len(right)) < 100:
        raise ValueError("insufficient independent temporal support for double continuation backup")
    return (
        _fit_immediate_backup(left, 20261400),
        _fit_immediate_backup(right, 20261404),
    )


def _double_state_value(
    rows: pd.DataFrame,
    backup: tuple[Predictor, Predictor],
) -> pd.Series:
    """Select with one estimator and evaluate with the other, symmetrically."""
    values = _x(rows)
    prediction_a = np.asarray(backup[0].predict(values), dtype=float)
    prediction_b = np.asarray(backup[1].predict(values), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(rows["actual_entry_timestamp"], utc=True).to_numpy(),
            "a": prediction_a,
            "b": prediction_b,
        }
    )
    selected_a = frame.groupby("timestamp", sort=True)["a"].idxmax()
    selected_b = frame.groupby("timestamp", sort=True)["b"].idxmax()
    evaluated = pd.Series(
        (frame.loc[selected_a, "b"].to_numpy(float) + frame.loc[selected_b, "a"].to_numpy(float))
        / 2,
        index=pd.DatetimeIndex(frame.loc[selected_a, "timestamp"]),
    )
    return evaluated.clip(lower=0.0)


def continuation_targets(
    rows: pd.DataFrame,
    backup_model: Predictor | tuple[Predictor, Predictor] | None = None,
) -> pd.DataFrame:
    """Build one fitted semi-Markov backup without an oracle future maximum."""
    ordered = rows.sort_values(["actual_entry_timestamp", "side"], kind="stable").reset_index(
        drop=True
    )
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(ordered["actual_entry_timestamp"], utc=True).drop_duplicates()
    )
    if len(timestamps) < 2:
        raise ValueError("insufficient states for continuation targets")
    timestamp_ns = _datetime_ns(timestamps)
    exits_ns = _datetime_ns(ordered["exit_timestamp"])
    entries_ns = _datetime_ns(ordered["actual_entry_timestamp"])
    state_index = np.searchsorted(timestamp_ns, entries_ns, side="left")
    next_free = np.searchsorted(timestamp_ns, exits_ns, side="left")
    fitted_value = np.zeros(len(timestamps) + 1, dtype=float)
    if backup_model is not None:
        if isinstance(backup_model, tuple):
            state_value = _double_state_value(ordered, backup_model)
        else:
            predicted_immediate = np.asarray(backup_model.predict(_x(ordered)), dtype=float)
            state_value = (
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(ordered["actual_entry_timestamp"], utc=True),
                        "predicted_immediate": predicted_immediate,
                    }
                )
                .groupby("timestamp", sort=True)["predicted_immediate"]
                .max()
                .clip(lower=0.0)
            )
        fitted_value[:-1] = state_value.reindex(timestamps).fillna(0.0).to_numpy(float)
    q_wait = np.zeros(len(timestamps), dtype=float)
    if len(timestamps) > 1:
        wait_seconds = np.maximum(0.0, np.diff(timestamp_ns) / 1e9)
        q_wait[:-1] = np.exp(-wait_seconds / CONTINUATION_HALF_LIFE_SECONDS) * fitted_value[1:-1]
    utility = ordered["log_utility"].to_numpy(float)
    duration = np.maximum(0.0, (exits_ns - entries_ns) / 1e9)
    q_enter = (
        utility
        + np.exp(-duration / CONTINUATION_HALF_LIFE_SECONDS)
        * fitted_value[np.minimum(next_free, len(timestamps))]
    )
    ordered["target_q_enter_log_utility"] = q_enter
    ordered["target_q_wait_log_utility"] = q_wait[state_index]
    best_enter = ordered.groupby("actual_entry_timestamp", sort=True)[
        "target_q_enter_log_utility"
    ].transform("max")
    ordered["target_state_value_log_utility"] = np.maximum(
        best_enter.to_numpy(float), q_wait[state_index]
    )
    ordered["continuation_target_source"] = (
        "DOUBLE_TEMPORAL_FITTED_IMMEDIATE_VALUE"
        if isinstance(backup_model, tuple)
        else "SINGLE_FITTED_IMMEDIATE_VALUE"
        if backup_model is not None
        else "ZERO_BACKUP"
    )
    complete_before = timestamps[-1] - pd.Timedelta(seconds=MAXIMUM_HORIZON_SECONDS)
    ordered["continuation_target_complete"] = pd.to_datetime(
        ordered["actual_entry_timestamp"], utc=True
    ).le(complete_before)
    return ordered


def _cross_fitted_continuation_targets(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    ordered = rows.sort_values("actual_entry_timestamp", kind="stable")
    timestamp = pd.to_datetime(ordered["actual_entry_timestamp"], utc=True)
    start = timestamp.min().floor("D")
    end = timestamp.max().ceil("D")
    block_start = start + pd.Timedelta(weeks=CONTINUATION_CROSSFIT_MINIMUM_WEEKS)
    pieces: list[pd.DataFrame] = []
    blocks = 0
    while block_start < end:
        block_end = min(block_start + pd.Timedelta(weeks=CONTINUATION_CROSSFIT_BLOCK_WEEKS), end)
        history = _period(ordered, None, block_start, purge_exit=True)
        held_out = _period(ordered, block_start, block_end, purge_exit=True)
        if len(history) >= 100 and len(held_out) >= 100:
            backup = _fit_double_immediate_backup(history)
            targets = continuation_targets(held_out, backup)
            pieces.append(targets.loc[targets["continuation_target_complete"]].copy())
            blocks += 1
        block_start = block_end
    if not pieces:
        raise ValueError("insufficient chronology for continuation cross-fitting")
    output = pd.concat(pieces, ignore_index=True)
    return output, {
        "strictly_past_only": True,
        "blocks": blocks,
        "rows": len(output),
    }


def fit_continuation_models(rows: pd.DataFrame) -> dict[str, Any]:
    complete, crossfit = _cross_fitted_continuation_targets(rows)
    if len(complete) < 100:
        raise ValueError("insufficient complete continuation targets")
    values = _x(complete)
    weights = _timestamp_weights(complete)
    models: dict[str, Any] = {
        "enter": _fit_regressor(
            "ridge",
            _regressor("ridge", 20261401),
            values,
            complete["target_q_enter_log_utility"].to_numpy(float),
            weights,
        ),
        "wait": _fit_regressor(
            "ridge",
            _regressor("ridge", 20261402),
            values,
            complete["target_q_wait_log_utility"].to_numpy(float),
            weights,
        ),
    }
    models["advantage"] = _fit_regressor(
        "ridge",
        _regressor("ridge", 20261403),
        values,
        (
            complete["target_q_enter_log_utility"].to_numpy(float)
            - complete["target_q_wait_log_utility"].to_numpy(float)
        ),
        weights,
    )
    models["backup"] = _fit_double_immediate_backup(rows)
    models["crossfit"] = crossfit
    return models


def fit_continuation_calibration(models: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    targets = continuation_targets(rows, cast(tuple[Predictor, Predictor], models["backup"]))
    complete = targets.loc[targets["continuation_target_complete"]].copy()
    if len(complete) < 100:
        raise ValueError("insufficient complete continuation calibration targets")
    values = _x(complete)
    target = {
        "enter": complete["target_q_enter_log_utility"].to_numpy(float),
        "wait": complete["target_q_wait_log_utility"].to_numpy(float),
        "advantage": (
            complete["target_q_enter_log_utility"].to_numpy(float)
            - complete["target_q_wait_log_utility"].to_numpy(float)
        ),
    }
    calibration = {
        action: IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(cast(Predictor, models[action]).predict(values), dtype=float),
            target[action],
        )
        for action in ("enter", "wait", "advantage")
    }
    calibration["target_source"] = "DOUBLE_TEMPORAL_FITTED_IMMEDIATE_VALUE"
    return calibration


def score_continuation(
    rows: pd.DataFrame,
    models: dict[str, Any],
    calibration: dict[str, Any],
    *,
    copy: bool = True,
) -> pd.DataFrame:
    output = rows.copy() if copy else rows
    values = _x(output)
    for action in ("enter", "wait", "advantage"):
        raw = np.asarray(cast(Predictor, models[action]).predict(values), dtype=float)
        output[f"q_{action}_log_utility"] = cast(IsotonicRegression, calibration[action]).predict(
            raw
        )
    output["q_wait_log_utility"] = output.groupby("actual_entry_timestamp", sort=False)[
        "q_wait_log_utility"
    ].transform("mean")
    output["immediate_expected_log_utility"] = output["expected_log_utility"]
    output["independent_q_difference_log_utility"] = (
        output["q_enter_log_utility"] - output["q_wait_log_utility"]
    )
    output["raw_action_advantage_log_utility"] = output["q_advantage_log_utility"]
    output["continuation_dominance_violation"] = output["raw_action_advantage_log_utility"].gt(
        output["immediate_expected_log_utility"]
    )
    output["action_advantage_log_utility"] = np.minimum(
        output["raw_action_advantage_log_utility"].to_numpy(float),
        output["immediate_expected_log_utility"].to_numpy(float),
    )
    output["expected_log_utility"] = output["action_advantage_log_utility"]
    return output


def continuation_metrics(scored: pd.DataFrame, models: dict[str, Any]) -> dict[str, Any]:
    targets = continuation_targets(scored, cast(tuple[Predictor, Predictor], models["backup"]))
    complete = targets["continuation_target_complete"].to_numpy(bool)
    target_advantage = targets.loc[complete, "target_q_enter_log_utility"].to_numpy(
        float
    ) - targets.loc[complete, "target_q_wait_log_utility"].to_numpy(float)
    predicted_advantage = targets.loc[complete, "raw_action_advantage_log_utility"].to_numpy(float)
    return {
        "enter_mae": float(
            mean_absolute_error(
                targets.loc[complete, "target_q_enter_log_utility"],
                targets.loc[complete, "q_enter_log_utility"],
            )
        ),
        "wait_mae": float(
            mean_absolute_error(
                targets.loc[complete, "target_q_wait_log_utility"],
                targets.loc[complete, "q_wait_log_utility"],
            )
        ),
        "advantage_sign_accuracy": float(
            np.mean(np.sign(target_advantage) == np.sign(predicted_advantage))
        ),
        "advantage_mae": float(mean_absolute_error(target_advantage, predicted_advantage)),
        "positive_target_fraction": float(np.mean(target_advantage > 0)),
        "positive_prediction_fraction": float(np.mean(predicted_advantage > 0)),
        "maximum_absolute_target_q": float(
            np.max(
                np.abs(
                    targets.loc[
                        complete,
                        ["target_q_enter_log_utility", "target_q_wait_log_utility"],
                    ].to_numpy(float)
                ),
                initial=0.0,
            )
        ),
        "dominance_violation_fraction": float(
            targets.loc[complete, "continuation_dominance_violation"].mean()
        ),
        "target_source": "DOUBLE_TEMPORAL_FITTED_IMMEDIATE_VALUE",
        "crossfit": models["crossfit"],
    }


def _leverage(stop_bps: np.ndarray, round_trip_cost_bps: float | np.ndarray) -> np.ndarray:
    risk_fraction = (np.asarray(stop_bps, dtype=float) + round_trip_cost_bps) / 10_000
    return np.minimum(MAXIMUM_LEVERAGE, RISK_PER_TRADE / np.maximum(risk_fraction, 1e-9))


def fit_stop_loss_overrun_reserve(rows: pd.DataFrame, round_trip_cost_bps: float) -> float:
    """Estimate a causal sizing reserve from losses beyond the stated stop and fees."""
    if rows.empty:
        return 0.0
    overrun = np.maximum(
        0.0,
        -rows["net_bps"].to_numpy(float) - rows["stop_bps"].to_numpy(float) - round_trip_cost_bps,
    )
    positive = overrun[overrun > 0]
    if not len(positive):
        return 0.0
    return float(np.quantile(positive, STOP_LOSS_OVERRUN_QUANTILE, method="higher"))


def apply_risk_sizing_contract(
    rows: pd.DataFrame,
    round_trip_cost_bps: float,
    stop_loss_overrun_reserve_bps: float,
) -> pd.DataFrame:
    output = rows.copy()
    reserve = max(0.0, float(stop_loss_overrun_reserve_bps))
    output["risk_stop_overrun_reserve_bps"] = reserve
    leverage = _leverage(output["stop_bps"].to_numpy(float), round_trip_cost_bps + reserve)
    output["sized_leverage"] = leverage
    output["sized_portfolio_return"] = leverage * output["net_bps"].to_numpy(float) / 10_000
    if np.any(output["sized_portfolio_return"].to_numpy(float) <= -1):
        raise ValueError("risk-sized state-action can lose all equity")
    output["log_utility"] = np.log1p(output["sized_portfolio_return"].to_numpy(float))
    return output


def sequential_replay(
    scored: pd.DataFrame,
    threshold_bps: float | dict[int, float],
    round_trip_cost_bps: float,
    *,
    record_decisions: bool = True,
    risk_state: ReplayRiskState | None = None,
    value_column: str = "expected_log_utility",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return scored.copy(), pd.DataFrame(columns=["timestamp", "action", "reason"])
    if value_column not in scored and value_column != "expected_log_utility":
        raise ValueError(f"missing replay value column: {value_column}")
    stop_reserve = (
        scored["risk_stop_overrun_reserve_bps"].to_numpy(float)
        if "risk_stop_overrun_reserve_bps" in scored
        else np.zeros(len(scored), dtype=float)
    )
    predicted_leverage = _leverage(
        scored["stop_bps"].to_numpy(float), round_trip_cost_bps + stop_reserve
    )
    if value_column in scored:
        replay_value = scored[value_column].to_numpy(float)
    else:
        implied_return = predicted_leverage * scored["calibrated_ev_bps"].to_numpy(float) / 10_000
        replay_value = np.log1p(np.maximum(implied_return, -0.999999))
    if isinstance(threshold_bps, dict):
        required_ev_bps = scored["side"].map(threshold_bps).fillna(float("inf")).to_numpy(float)
    else:
        required_ev_bps = np.full(len(scored), float(threshold_bps), dtype=float)
    required_return = predicted_leverage * required_ev_bps / 10_000
    required_log_utility = np.log1p(required_return)
    passes_value_threshold = replay_value > required_log_utility
    ranking = pd.DataFrame(
        {
            "row_number": np.arange(len(scored), dtype=np.int64),
            "actual_entry_timestamp": pd.to_datetime(
                scored["actual_entry_timestamp"], utc=True
            ).to_numpy(),
            "passes_value_threshold": passes_value_threshold,
            "replay_value": replay_value,
            "p_target": scored["p_target"].to_numpy(float),
            "expert_id": scored["expert_id"].astype(str).to_numpy(),
        }
    )
    selected = (
        ranking.sort_values(
            [
                "actual_entry_timestamp",
                "passes_value_threshold",
                "replay_value",
                "p_target",
                "expert_id",
            ],
            ascending=[True, False, False, False, True],
            kind="stable",
        )
        .drop_duplicates("actual_entry_timestamp", keep="first")
        .sort_values("actual_entry_timestamp")
        .reset_index(drop=True)
    )
    candidate_positions = selected["row_number"].to_numpy(np.int64)
    candidates = scored.iloc[candidate_positions].copy().reset_index(drop=True)
    candidates["expected_log_utility"] = replay_value[candidate_positions]
    candidates["required_ev_bps"] = required_ev_bps[candidate_positions]
    candidates["required_log_utility"] = required_log_utility[candidate_positions]
    candidates["passes_value_threshold"] = passes_value_threshold[candidate_positions]
    candidates["passes_ev_threshold"] = candidates["passes_value_threshold"]
    if {
        "fold_expert_scope",
        "expert_tree_index",
        "expert_leaf_id",
    }.issubset(candidates.columns):
        candidates["context_expert_id"] = [
            (
                fold_expert_id(str(scope), int(side), 0, int(tree), int(leaf))
                if int(tree) >= 0 and int(leaf) >= 0
                else f"{scope}:{'LONG' if int(side) > 0 else 'SHORT'}:FALLBACK"
            )
            for scope, side, tree, leaf in zip(
                candidates["fold_expert_scope"],
                candidates["side"],
                candidates["expert_tree_index"],
                candidates["expert_leaf_id"],
                strict=True,
            )
        ]
    if "plan_id" in candidates.columns:
        candidates["expert_id"] = candidates["plan_id"].astype(str)
    selected_rows: list[int] = []
    leverages: list[float] = []
    portfolio_returns: list[float] = []
    equity_before: list[float] = []
    daily_pnl_before: list[float] = []
    risk_remaining_before: list[float] = []
    risk_violation_rows: list[bool] = []
    entry_actions: list[str] = []
    decisions: list[dict[str, Any]] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    active_risk = risk_state if risk_state is not None else ReplayRiskState()
    equity = active_risk.equity
    peak_equity = active_risk.peak_equity
    current_day = active_risk.current_day
    day_start_equity = active_risk.day_start_equity
    risk_violations = 0
    pending_return: float | None = None
    active_side = 0
    active_expert: str | None = None
    active_entry: pd.Timestamp | None = None
    active_entry_price: float | None = None
    for row_number, raw_row in enumerate(candidates.itertuples(index=False)):
        row = cast(Any, raw_row)
        entry = pd.Timestamp(row.actual_entry_timestamp)
        # Realized P&L enters the risk state only when the preceding position exits.
        if pending_return is not None and entry >= free_at:
            exit_day = free_at.floor("D")
            if current_day is None or exit_day != current_day:
                current_day = exit_day
                day_start_equity = equity
            equity *= 1 + pending_return
            peak_equity = max(peak_equity, equity)
            pending_return = None
            active_side = 0
            active_expert = None
            active_entry = None
            active_entry_price = None
        day = entry.floor("D")
        if current_day is None or day != current_day:
            current_day = day
            day_start_equity = equity
        position_open = entry < free_at
        current_price = float(getattr(row, "entry_price", 0.0))
        unrealized_net_bps = (
            active_side * (current_price / active_entry_price - 1) * 10_000 - round_trip_cost_bps
            if position_open
            and active_entry_price is not None
            and active_entry_price > 0
            and current_price > 0
            else 0.0
        )
        daily_floor = day_start_equity * (1 - MAXIMUM_DAILY_LOSS)
        drawdown_floor = peak_equity * (1 - MAXIMUM_DRAWDOWN)
        daily_risk_remaining = max(0.0, 1 - daily_floor / max(equity, 1e-12))
        drawdown_risk_remaining = max(0.0, 1 - drawdown_floor / max(equity, 1e-12))
        state = SequentialState(
            daily_pnl_fraction=equity / day_start_equity - 1,
            risk_remaining_fraction=min(daily_risk_remaining, drawdown_risk_remaining),
            position_side=active_side if position_open else 0,
            time_in_position_seconds=(
                int((entry - active_entry).total_seconds())
                if position_open and active_entry is not None
                else 0
            ),
            unrealized_net_bps=unrealized_net_bps,
            close_cost_bps=round_trip_cost_bps / 2,
            expert_id=active_expert if position_open else None,
            regime="OBSERVED",
            vwap_distance_bps=float(getattr(row, "vwap_distance_bps", 0.0)),
            avwap_distance_bps=float(getattr(row, "vwap_distance_240m_bps", 0.0)),
            funding_bps=float(row.funding_bps),
            volatility_bps=float(getattr(row, "realized_volatility_30m_bps", 0.0)),
        )
        if position_open:
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "HOLD",
                        "reason": "POSITION_OPEN",
                        **asdict(state),
                    }
                )
            continue
        if drawdown_risk_remaining <= 0:
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "WAIT",
                        "reason": "MAXIMUM_DRAWDOWN_VETO",
                        **asdict(state),
                    }
                )
            continue
        if not bool(row.passes_ev_threshold):
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "WAIT",
                        "reason": "EXPECTED_EQUITY_UTILITY_BELOW_THRESHOLD",
                        "candidate_side": int(row.side),
                        "candidate_plan_id": str(getattr(row, "plan_id", row.expert_id)),
                        "candidate_local_variant": str(getattr(row, "local_variant", "BASE")),
                        "candidate_calibrated_ev_bps": float(row.calibrated_ev_bps),
                        "candidate_immediate_log_utility": float(
                            getattr(row, "immediate_expected_log_utility", row.expected_log_utility)
                        ),
                        "candidate_action_advantage_log_utility": float(
                            getattr(row, "action_advantage_log_utility", row.expected_log_utility)
                        ),
                        "candidate_q_enter_log_utility": float(
                            getattr(row, "q_enter_log_utility", 0.0)
                        ),
                        "candidate_q_wait_log_utility": float(
                            getattr(row, "q_wait_log_utility", 0.0)
                        ),
                        "candidate_expected_holding_seconds": float(
                            getattr(
                                row,
                                "expected_holding_seconds",
                                getattr(row, "horizon_seconds", MAXIMUM_HORIZON_SECONDS),
                            )
                        ),
                        "candidate_target_probability": float(row.p_target),
                        **asdict(state),
                    }
                )
            continue
        row_stop_reserve = float(getattr(row, "risk_stop_overrun_reserve_bps", 0.0))
        leverage = float(
            _leverage(
                np.asarray([float(row.stop_bps)]),
                round_trip_cost_bps + row_stop_reserve,
            )[0]
        )
        worst_risk = (
            leverage * (float(row.stop_bps) + round_trip_cost_bps + row_stop_reserve) / 10_000
        )
        if state.risk_remaining_fraction + 1e-12 < worst_risk:
            veto_reason = (
                "MAXIMUM_DRAWDOWN_VETO"
                if drawdown_risk_remaining < worst_risk
                else "DAILY_RISK_VETO"
            )
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "WAIT",
                        "reason": veto_reason,
                        **asdict(state),
                    }
                )
            continue
        portfolio_return = leverage * float(row.net_bps) / 10_000
        risk_violation = portfolio_return < -worst_risk - 1e-9
        if risk_violation:
            risk_violations += 1
        entry_action = "ENTER_LONG" if int(row.side) > 0 else "ENTER_SHORT"
        selected_rows.append(row_number)
        leverages.append(leverage)
        portfolio_returns.append(portfolio_return)
        equity_before.append(equity)
        daily_pnl_before.append(state.daily_pnl_fraction)
        risk_remaining_before.append(state.risk_remaining_fraction)
        risk_violation_rows.append(risk_violation)
        entry_actions.append(entry_action)
        if record_decisions:
            decisions.append(
                {
                    "timestamp": entry,
                    "action": entry_action,
                    "reason": "EXPECTED_EQUITY_UTILITY_AND_RISK_APPROVED",
                    **asdict(state),
                }
            )
        target_seconds = int(row.time_to_target_seconds)
        if record_decisions and 0 < target_seconds < int(row.exit_seconds):
            tightened_state = asdict(state) | {
                "position_side": int(row.side),
                "time_in_position_seconds": target_seconds,
                "unrealized_net_bps": float(getattr(row, "target_1_bps", 0.0))
                - round_trip_cost_bps,
                "expert_id": str(row.expert_id),
            }
            decisions.append(
                {
                    "timestamp": entry + pd.Timedelta(seconds=target_seconds),
                    "action": "REDUCE",
                    "reason": "FIRST_TARGET_FILLED",
                    "reduce_fraction": float(getattr(row, "first_exit_fraction", 0.5)),
                    **tightened_state,
                }
            )
            decisions.append(
                {
                    "timestamp": entry + pd.Timedelta(seconds=target_seconds),
                    "action": "TIGHTEN_STOP",
                    "reason": "FIRST_TARGET_FILLED_NON_WIDENING_STOP",
                    **tightened_state,
                }
            )
            decisions.append(
                {
                    "timestamp": entry + pd.Timedelta(seconds=target_seconds),
                    "action": "UPDATE_TRAIL",
                    "reason": "FIRST_TARGET_ACTIVATED_DYNAMIC_TRAIL",
                    "trailing_bps": float(getattr(row, "trailing_bps", 0.0)),
                    **tightened_state,
                }
            )
        exit_at = pd.Timestamp(row.exit_timestamp)
        if record_decisions:
            close_state = asdict(state) | {
                "position_side": int(row.side),
                "time_in_position_seconds": int(row.exit_seconds),
                "unrealized_net_bps": float(row.net_bps),
                "expert_id": str(row.expert_id),
            }
            decisions.append(
                {
                    "timestamp": exit_at,
                    "action": "CLOSE",
                    "reason": str(row.outcome),
                    **close_state,
                }
            )
        free_at = exit_at
        pending_return = portfolio_return
        active_side = int(row.side)
        active_expert = str(row.expert_id)
        active_entry = entry
        active_entry_price = float(getattr(row, "entry_price", 0.0)) or None
    if pending_return is not None:
        exit_day = free_at.floor("D")
        if current_day is None or exit_day != current_day:
            current_day = exit_day
            day_start_equity = equity
        equity *= 1 + pending_return
        peak_equity = max(peak_equity, equity)
    result = candidates.iloc[selected_rows].copy().reset_index(drop=True)
    if selected_rows:
        result["leverage"] = leverages
        result["portfolio_return"] = portfolio_returns
        result["equity_before"] = equity_before
        result["daily_pnl_before"] = daily_pnl_before
        result["risk_remaining_before"] = risk_remaining_before
        result["risk_violation"] = risk_violation_rows
        result["entry_action"] = entry_actions
        result["close_action"] = "CLOSE"
    result.attrs["risk_violations"] = risk_violations
    active_risk.equity = equity
    active_risk.peak_equity = peak_equity
    active_risk.current_day = current_day
    active_risk.day_start_equity = day_start_equity
    decision_rows = (
        pd.DataFrame(decisions).sort_values("timestamp", kind="stable").reset_index(drop=True)
        if decisions
        else pd.DataFrame(columns=["timestamp", "action", "reason"])
    )
    return result, decision_rows


def _daily_returns(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="1D", inclusive="left")
    if trades.empty:
        return pd.Series(0.0, index=days)
    daily = (
        trades.assign(day=pd.to_datetime(trades["exit_timestamp"], utc=True).dt.floor("D"))
        .groupby("day")["portfolio_return"]
        .apply(lambda values: float(np.prod(1 + values.to_numpy(float)) - 1))
    )
    return daily.reindex(days, fill_value=0.0)


def _block_bootstrap_lcb(values: np.ndarray, block: int, seed: int) -> float | None:
    if len(values) < 20:
        return None
    generator = np.random.default_rng(seed)
    block = min(block, len(values))
    means = np.empty(2_000, dtype=float)
    for draw in range(len(means)):
        sampled: list[np.ndarray] = []
        while sum(len(item) for item in sampled) < len(values):
            start = int(generator.integers(0, len(values) - block + 1))
            sampled.append(values[start : start + block])
        means[draw] = np.concatenate(sampled)[: len(values)].mean()
    return float(np.quantile(means, 0.05))


def policy_metrics(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    daily_bootstrap: bool = True,
    weekly_bootstrap: bool = True,
) -> dict[str, Any]:
    daily = _daily_returns(trades, start, end)
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "expectancy_bps": None,
            "equity_expectancy": None,
            "mean_log_growth": None,
            "profit_factor": None,
            "maximum_drawdown": None,
            "positive_active_days": 0.0,
            "daily_lcb_95": None,
            "weekly_lcb_95": None,
            "risk_violations": int(trades.attrs.get("risk_violations", 0)),
        }
    net = trades["net_bps"].to_numpy(float)
    returns = trades["portfolio_return"].to_numpy(float)
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    notional_gains = float(net[net > 0].sum())
    notional_losses = float(-net[net < 0].sum())
    log_growth = np.log1p(returns)
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    active = daily.loc[daily.ne(0)]
    weekly = (1 + daily).resample("7D").prod() - 1
    risk_violations = (
        int(trades["risk_violation"].astype(bool).sum())
        if "risk_violation" in trades
        else int(trades.attrs.get("risk_violations", 0))
    )
    return {
        "trades": len(trades),
        "trades_per_day": float(len(trades) / max(1, len(daily))),
        "expectancy_bps": float(net.mean()),
        "equity_expectancy": float(returns.mean()),
        "mean_log_growth": float(log_growth.mean()),
        "total_log_growth": float(log_growth.sum()),
        "profit_factor": gains / losses if losses else None,
        "notional_profit_factor": notional_gains / notional_losses if notional_losses else None,
        "win_rate": float((net > 0).mean()),
        "maximum_drawdown": float((1 - equity / peak).max(initial=0.0)),
        "positive_active_days": float(active.gt(0).mean()) if len(active) else 0.0,
        "mean_daily_return": float(daily.mean()),
        "daily_lcb_95": (
            _block_bootstrap_lcb(daily.to_numpy(float), 5, 20260831) if daily_bootstrap else None
        ),
        "weekly_lcb_95": (
            _block_bootstrap_lcb(weekly.to_numpy(float), 3, 20260901) if weekly_bootstrap else None
        ),
        "stress_1_5x_expectancy_bps": float(trades["stress_1_5x_bps"].mean()),
        "stress_2x_expectancy_bps": float(trades["stress_2x_bps"].mean()),
        "risk_violations": risk_violations,
        "long_trades": int(trades["side"].gt(0).sum()),
        "short_trades": int(trades["side"].lt(0).sum()),
    }


def negative_control_metrics(
    scored: pd.DataFrame,
    fee: FeeContract,
    start: pd.Timestamp,
    end: pd.Timestamp,
    constant_plan: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate causal policy controls without using them for model selection."""
    result: dict[str, Any] = {
        "always_wait": policy_metrics(
            scored.iloc[:0],
            start,
            end,
            daily_bootstrap=False,
            weekly_bootstrap=False,
        )
    }

    def evaluate(
        name: str,
        source: pd.DataFrame,
        values: np.ndarray | None = None,
    ) -> None:
        control = source.copy()
        if values is not None:
            control["expected_log_utility"] = values
        if "p_target" not in control:
            control["p_target"] = 0.5
        trades, _ = sequential_replay(control, 0.0, fee.round_trip_bps, record_decisions=False)
        result[name] = policy_metrics(
            trades,
            start,
            end,
            daily_bootstrap=False,
            weekly_bootstrap=False,
        )

    generator = np.random.default_rng(20261101)
    evaluate(
        "random_prediction",
        scored,
        generator.permutation(scored["expected_log_utility"].to_numpy(float)),
    )
    shifted = scored.sort_values(["side", "actual_entry_timestamp"], kind="stable")
    shifted_values = (
        shifted.groupby("side", sort=False)["expected_log_utility"].shift(1).fillna(-1.0)
    )
    evaluate("temporally_shifted_prediction", shifted, shifted_values.to_numpy(float))

    no_gating_return = (
        _leverage(scored["stop_bps"].to_numpy(float), fee.round_trip_bps)
        * scored["managed_expert_mean_bps"].to_numpy(float)
        / 10_000
    )
    evaluate(
        "managed_expert_mean_no_gate",
        scored,
        np.log1p(np.maximum(no_gating_return, -0.999999)),
    )

    def expert_control(name: str, column: str, *, already_net: bool = False) -> None:
        net_prediction = scored[column].to_numpy(float)
        if not already_net:
            net_prediction = net_prediction - fee.round_trip_bps
        predicted_return = (
            _leverage(scored["stop_bps"].to_numpy(float), fee.round_trip_bps)
            * net_prediction
            / 10_000
        )
        evaluate(
            name,
            scored,
            np.log1p(np.maximum(predicted_return, -0.999999)),
        )

    if "view_full_prediction_bps" in scored:
        expert_control("full_only", "view_full_prediction_bps")
    if "equal_weight_expert_prediction_bps" in scored:
        expert_control("equal_weight_experts", "equal_weight_expert_prediction_bps")
    if "gate_expected_gross_bps" in scored:
        expert_control("deterministic_gate_only", "gate_expected_gross_bps")
    if "managed_expert_best_lcb_bps" in scored:
        expert_control("best_active_fold_expert", "managed_expert_best_lcb_bps", already_net=True)

    if constant_plan is not None and not constant_plan.empty:
        direction = np.sign(constant_plan["price_velocity_1m"].to_numpy(float))
        evaluate(
            "train_median_constant_plan",
            constant_plan,
            np.where(constant_plan["side"].to_numpy(int) == direction, 1e-6, -1.0),
        )

    momentum_direction = np.sign(scored["price_velocity_1m"].to_numpy(float))
    evaluate(
        "simple_momentum",
        scored,
        np.where(scored["side"].to_numpy(int) == momentum_direction, 1e-6, -1.0),
    )

    mean_reversion_direction = -np.sign(scored["vwap_distance_bps"].to_numpy(float))
    evaluate(
        "simple_mean_reversion",
        scored,
        np.where(scored["side"].to_numpy(int) == mean_reversion_direction, 1e-6, -1.0),
    )
    evaluate("long_only", scored.loc[scored["side"].gt(0)])
    evaluate("short_only", scored.loc[scored["side"].lt(0)])
    return result


def economic_calibration_buckets(scored: pd.DataFrame) -> dict[str, Any]:
    edges = [-math.inf, 0.0, 2.0, 4.0, 8.0, 12.0, 20.0, math.inf]
    labels = ["<0", "0-2", "2-4", "4-8", "8-12", "12-20", ">20"]
    values = (
        scored.loc[scored["post_selection_candidate"]].copy()
        if "post_selection_candidate" in scored
        else scored.copy()
    )
    values["ev_bucket"] = pd.cut(
        values["calibrated_ev_bps"], bins=edges, labels=labels, right=False
    )
    grouped = (
        values.groupby("ev_bucket", observed=True)
        .agg(
            candidates=("net_bps", "size"),
            predicted_ev_bps=("calibrated_ev_bps", "mean"),
            realized_ev_bps=("net_bps", "mean"),
            predicted_log_utility=("immediate_expected_log_utility", "mean"),
            realized_log_utility=("log_utility", "mean"),
            expected_holding_seconds=("expected_holding_seconds", "mean"),
            expected_ev_bps_per_minute=("expected_ev_bps_per_minute", "mean"),
            expected_log_utility_per_hour=("expected_log_utility_per_hour", "mean"),
            first_timestamp=("actual_entry_timestamp", "min"),
            last_timestamp=("actual_entry_timestamp", "max"),
        )
        .reset_index()
    )
    realized = grouped["realized_log_utility"].to_numpy(float)
    leverage = values["sized_leverage"].to_numpy(float)
    coherent_upper = np.log1p(
        np.maximum(
            leverage * values["calibrated_ev_bps"].to_numpy(float) / 10_000,
            -0.999999,
        )
    )
    return {
        "buckets": grouped.to_dict("records"),
        "utility_ev_consistency_violations": int(
            np.sum(
                values["immediate_expected_log_utility"].to_numpy(float) > coherent_upper + 1e-12
            )
        ),
        "utility_consistency_clip_fraction": float(
            np.mean(
                np.where(
                    values["selected_value_head"].eq("direct"),
                    values["direct_utility_consistency_clipped"],
                    values["decomposed_utility_consistency_clipped"],
                )
            )
        ),
        "realized_utility_monotone_non_decreasing": bool(
            len(realized) < 2 or np.all(np.diff(realized) >= 0)
        ),
    }


def view_gating_audit(scored: pd.DataFrame) -> dict[str, Any]:
    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        return (
            float(np.corrcoef(left, right)[0, 1])
            if len(left) > 1 and float(left.std()) > 0 and float(right.std()) > 0
            else None
        )

    prediction_columns = [f"view_{view}_prediction_bps" for view in base.VIEWS]
    prediction_columns = [name for name in prediction_columns if name in scored]
    if not prediction_columns:
        return {"available": False, "reason": "VIEW_PREDICTIONS_NOT_RETAINED"}
    predictions = scored.loc[:, prediction_columns].astype(float)
    errors = predictions.sub(scored["net_bps"].to_numpy(float), axis=0)
    full_residual = scored["net_bps"].to_numpy(float) - predictions[
        "view_full_prediction_bps"
    ].to_numpy(float)
    leave_one_out: dict[str, Any] = {}
    for view in base.VIEWS:
        column = f"gate_without_{view}_prediction_bps"
        if column not in scored:
            continue
        prediction = scored[column].to_numpy(float)
        leave_one_out[view] = {
            "mae_bps": float(mean_absolute_error(scored["net_bps"], prediction)),
            "calibration_error_bps": _calibration_error(
                scored["net_bps"].to_numpy(float), prediction
            ),
        }
    residual_information = {
        column.removeprefix("view_").removesuffix("_prediction_bps"): correlation(
            predictions[column].to_numpy(float), full_residual
        )
        for column in prediction_columns
    }
    gate_error = np.abs(
        scored["gate_expected_gross_bps"].to_numpy(float) - scored["net_bps"].to_numpy(float)
    )
    dispersion = scored["gate_disagreement_bps"].to_numpy(float)
    dispersion_error_correlation = correlation(dispersion, gate_error)
    prediction_correlation = predictions.corr().astype(object)
    prediction_correlation = prediction_correlation.where(prediction_correlation.notna(), None)
    error_correlation = errors.corr().astype(object)
    error_correlation = error_correlation.where(error_correlation.notna(), None)
    return {
        "available": True,
        "prediction_correlation": prediction_correlation.to_dict(),
        "error_correlation": error_correlation.to_dict(),
        "residual_information_correlation": residual_information,
        "leave_one_view_out": leave_one_out,
        "dispersion_vs_absolute_error_correlation": dispersion_error_correlation,
        "liquidity_view_historical": False,
        "liquidity_view_status": "SHADOW_ONLY_INSUFFICIENT_L2_DAYS",
        "derivatives_view_status": "NOT_PROMOTED_WITHOUT_COVERAGE_AND_PAIRED_ABLATION",
    }


def policy_gates(
    metrics: dict[str, Any], *, minimum_trades: int = MINIMUM_OOS_TRADES
) -> dict[str, bool]:
    economic_expectancy = metrics.get("mean_log_growth")
    if economic_expectancy is None:
        economic_expectancy = metrics.get("expectancy_bps")
    return {
        "minimum_oos_trades": int(metrics.get("trades", 0)) >= minimum_trades,
        "expectancy_positive": float(economic_expectancy or 0) > 0,
        "lower_confidence_bound_positive": min(
            float(metrics.get("daily_lcb_95") or -1),
            float(metrics.get("weekly_lcb_95") or -1),
        )
        > 0,
        "profit_factor_1_15": float(metrics.get("profit_factor") or 0) >= 1.15,
        "drawdown_8pct": float(metrics.get("maximum_drawdown") or 1) <= MAXIMUM_DRAWDOWN,
        "majority_active_days_positive": float(metrics.get("positive_active_days") or 0) > 0.5,
        "risk_respected": int(metrics.get("risk_violations", 1)) == 0,
    }


def selection_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    """Legacy diagnostic only; final statistical gates are never used for fold selection."""
    economic_expectancy = metrics.get("mean_log_growth")
    if economic_expectancy is None:
        economic_expectancy = metrics.get("expectancy_bps")
    return {
        "expectancy_positive": float(economic_expectancy or 0) > 0,
        "daily_lower_confidence_bound_positive": float(metrics.get("daily_lcb_95") or -1) > 0,
        "profit_factor_1_15": float(metrics.get("profit_factor") or 0) >= 1.15,
        "drawdown_8pct": float(metrics.get("maximum_drawdown") or 1) <= MAXIMUM_DRAWDOWN,
        "majority_active_days_positive": float(metrics.get("positive_active_days") or 0) > 0.5,
        "risk_respected": int(metrics.get("risk_violations", 1)) == 0,
    }


def _choose_frequency_threshold(
    scored: pd.DataFrame, fee: FeeContract, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[float, list[dict[str, Any]]]:
    frontier: list[dict[str, Any]] = []
    for threshold in THRESHOLDS_BPS:
        trades, _ = sequential_replay(scored, threshold, fee.round_trip_bps, record_decisions=False)
        metrics = policy_metrics(trades, start, end, weekly_bootstrap=False)
        returns = trades.get("portfolio_return", pd.Series(dtype=float)).to_numpy(float)
        net_log_equity: float | None = (
            float(np.log1p(returns).sum()) if len(returns) and np.all(returns > -1) else None
        )
        risk_approved = (
            int(metrics.get("risk_violations", 1)) == 0
            and float(metrics.get("maximum_drawdown") or 1) <= MAXIMUM_DRAWDOWN
        )
        eligible = risk_approved and net_log_equity is not None and net_log_equity > 0
        frontier.append(
            {
                "threshold_bps": threshold,
                "metrics": metrics,
                "selection_utility": net_log_equity,
                "risk_approved": risk_approved,
                "eligible": eligible,
                "final_statistical_gates_applied": False,
            }
        )
    sustainable = [item for item in frontier if item["eligible"]]
    if not sustainable:
        return float("inf"), frontier
    selected = max(
        sustainable,
        key=lambda item: (
            float(item["selection_utility"]),
            float(item["metrics"]["trades_per_day"]),
        ),
    )
    return float(selected["threshold_bps"]), frontier


def select_entry_controller(
    scored: pd.DataFrame,
    fee: FeeContract,
    start: pd.Timestamp,
    end: pd.Timestamp,
    continuation_blocks: int,
) -> tuple[str, dict[str, Any]]:
    """Promote continuation only when it beats a viable myopic controller past-only."""
    candidates = {
        "MYOPIC": "immediate_expected_log_utility",
        "CONTINUATION": "action_advantage_log_utility",
    }
    audit: dict[str, Any] = {}
    for name, value_column in candidates.items():
        trades, _ = sequential_replay(
            scored,
            0.0,
            fee.round_trip_bps,
            record_decisions=False,
            value_column=value_column,
        )
        metrics = policy_metrics(trades, start, end, weekly_bootstrap=False)
        returns = trades.get("portfolio_return", pd.Series(dtype=float)).to_numpy(float)
        selection_utility = (
            float(np.log1p(returns).sum()) if len(returns) and np.all(returns > -1) else None
        )
        enough_continuation_blocks = (
            name != "CONTINUATION" or continuation_blocks >= MINIMUM_CONTINUATION_CROSSFIT_BLOCKS
        )
        eligible = (
            enough_continuation_blocks
            and len(trades) >= MINIMUM_CONTROLLER_SELECTION_TRADES
            and selection_utility is not None
            and selection_utility > 0
            and int(metrics.get("risk_violations", 1)) == 0
            and metrics.get("maximum_drawdown") is not None
            and float(metrics["maximum_drawdown"]) <= MAXIMUM_DRAWDOWN
        )
        audit[name] = {
            "value_column": value_column,
            "metrics": metrics,
            "selection_utility": selection_utility,
            "minimum_trades": MINIMUM_CONTROLLER_SELECTION_TRADES,
            "continuation_crossfit_blocks": continuation_blocks,
            "minimum_continuation_crossfit_blocks": MINIMUM_CONTINUATION_CROSSFIT_BLOCKS,
            "eligible": eligible,
        }
    viable = [name for name, values in audit.items() if values["eligible"]]
    selected = (
        max(viable, key=lambda name: float(audit[name]["selection_utility"]))
        if viable
        else "DISABLED"
    )
    audit["selected"] = selected
    audit["selection_period_only"] = True
    audit["outer_test_read_for_selection"] = False
    return selected, audit


def apply_entry_controllers(
    rows: pd.DataFrame,
    controllers: dict[int, str],
) -> pd.DataFrame:
    output = rows
    output["selected_entry_controller"] = "DISABLED"
    output["expected_log_utility"] = -1.0
    for side, controller in controllers.items():
        positions = output["side"].eq(side)
        if controller == "MYOPIC":
            output.loc[positions, "expected_log_utility"] = output.loc[
                positions, "immediate_expected_log_utility"
            ]
        elif controller == "CONTINUATION":
            output.loc[positions, "expected_log_utility"] = output.loc[
                positions, "action_advantage_log_utility"
            ]
        elif controller != "DISABLED":
            raise ValueError(f"unknown entry controller: {controller}")
        output.loc[positions, "selected_entry_controller"] = controller
    return output


def _folds(rows: pd.DataFrame) -> list[dict[str, pd.Timestamp]]:
    start = pd.to_datetime(rows["actual_entry_timestamp"], utc=True).min().floor("D")
    end = pd.to_datetime(rows["actual_entry_timestamp"], utc=True).max().ceil("D")
    first_test = start + pd.Timedelta(weeks=MINIMUM_FIT_WEEKS + 3 * WINDOW_WEEKS)
    folds: list[dict[str, pd.Timestamp]] = []
    test_start = first_test
    while test_start + pd.Timedelta(weeks=WINDOW_WEEKS) <= end:
        folds.append(
            {
                "inner_start": test_start - pd.Timedelta(weeks=3 * WINDOW_WEEKS),
                "calibration_start": test_start - pd.Timedelta(weeks=2 * WINDOW_WEEKS),
                "selection_start": test_start - pd.Timedelta(weeks=WINDOW_WEEKS),
                "test_start": test_start,
                "test_end": test_start + pd.Timedelta(weeks=WINDOW_WEEKS),
            }
        )
        test_start += pd.Timedelta(weeks=WINDOW_WEEKS)
    return folds


def _period(
    rows: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp, *, purge_exit: bool
) -> pd.DataFrame:
    timestamp = pd.to_datetime(rows["actual_entry_timestamp"], utc=True)
    mask = timestamp.lt(end)
    if start is not None:
        mask &= timestamp.ge(start)
    if purge_exit:
        mask &= pd.to_datetime(rows["exit_timestamp"], utc=True).lt(end)
    return rows.loc[mask].copy()


def _xgb_available() -> bool:
    return bool(_gpu_info().get("available"))


def walk_forward(
    matrix: pd.DataFrame,
    fee: FeeContract,
    *,
    resume: bool = False,
    fold_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    folds = _folds(matrix)
    if fold_limit is not None:
        if fold_limit < 1:
            raise ValueError("fold_limit must be positive")
        folds = folds[:fold_limit]
    if not folds:
        raise ValueError("insufficient chronology for nested walk-forward")
    trade_pieces: list[pd.DataFrame] = []
    decision_pieces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    walk_forward_risk = ReplayRiskState()
    for number, fold in enumerate(folds, start=1):
        fit = _period(matrix, None, fold["inner_start"], purge_exit=True)
        inner_split = fold["inner_start"] + pd.Timedelta(weeks=INNER_CALIBRATION_WEEKS)
        inner_calibration = _period(matrix, fold["inner_start"], inner_split, purge_exit=True)
        model_audit = _period(matrix, inner_split, fold["calibration_start"], purge_exit=True)
        calibration = _period(
            matrix, fold["calibration_start"], fold["selection_start"], purge_exit=True
        )
        selection = _period(matrix, fold["selection_start"], fold["test_start"], purge_exit=True)
        test = _period(matrix, fold["test_start"], fold["test_end"], purge_exit=True)
        if (
            min(
                len(fit),
                len(inner_calibration),
                len(model_audit),
                len(calibration),
                len(selection),
                len(test),
            )
            == 0
        ):
            continue
        stop_loss_overrun_reserve_bps = fit_stop_loss_overrun_reserve(fit, fee.round_trip_bps)
        fit = apply_risk_sizing_contract(fit, fee.round_trip_bps, stop_loss_overrun_reserve_bps)
        inner_calibration = apply_risk_sizing_contract(
            inner_calibration, fee.round_trip_bps, stop_loss_overrun_reserve_bps
        )
        model_audit = apply_risk_sizing_contract(
            model_audit, fee.round_trip_bps, stop_loss_overrun_reserve_bps
        )
        calibration = apply_risk_sizing_contract(
            calibration, fee.round_trip_bps, stop_loss_overrun_reserve_bps
        )
        selection = apply_risk_sizing_contract(
            selection, fee.round_trip_bps, stop_loss_overrun_reserve_bps
        )
        test = apply_risk_sizing_contract(test, fee.round_trip_bps, stop_loss_overrun_reserve_bps)
        raw_fit = fit
        raw_inner_calibration = inner_calibration
        raw_model_audit = model_audit
        raw_calibration = calibration
        raw_selection = selection
        raw_test = test
        fit, library, crossfit_diagnostics = cross_fit_fold_expert_features(
            raw_fit,
            number,
            resume=resume,
        )
        inner_calibration = apply_fold_expert_library(raw_inner_calibration, library)
        model_audit = apply_fold_expert_library(raw_model_audit, library)
        calibration = apply_fold_expert_library(raw_calibration, library)
        selection = apply_fold_expert_library(raw_selection, library)
        test = apply_fold_expert_library(raw_test, library)
        plan_audit = plan_efficiency_audit(model_audit, fee)
        local_plan_variants_enabled = bool(plan_audit["material"])
        local_plan_training_support = {
            "fit": {"sampled_states": 0, "added_rows": 0},
            "inner_calibration": {"sampled_states": 0, "added_rows": 0},
        }
        if local_plan_variants_enabled:
            augmented_fit, local_plan_training_support["fit"] = augment_local_plan_training_support(
                raw_fit,
                fee,
                fee.round_trip_bps,
                stop_loss_overrun_reserve_bps,
                LOCAL_PLAN_TRAINING_STATES,
            )
            fit, library, crossfit_diagnostics = cross_fit_fold_expert_features(
                augmented_fit,
                f"{number}-local",
                resume=resume,
            )
            augmented_inner, local_plan_training_support["inner_calibration"] = (
                augment_local_plan_training_support(
                    raw_inner_calibration,
                    fee,
                    fee.round_trip_bps,
                    stop_loss_overrun_reserve_bps,
                    LOCAL_PLAN_INNER_CALIBRATION_STATES,
                )
            )
            inner_calibration = apply_fold_expert_library(augmented_inner, library)
            model_audit = apply_fold_expert_library(
                apply_risk_sizing_contract(
                    label_local_plan_variants(raw_model_audit, fee),
                    fee.round_trip_bps,
                    stop_loss_overrun_reserve_bps,
                ),
                library,
            )
            calibration = apply_fold_expert_library(
                apply_risk_sizing_contract(
                    label_local_plan_variants(raw_calibration, fee),
                    fee.round_trip_bps,
                    stop_loss_overrun_reserve_bps,
                ),
                library,
            )
            selection = apply_fold_expert_library(
                apply_risk_sizing_contract(
                    label_local_plan_variants(raw_selection, fee),
                    fee.round_trip_bps,
                    stop_loss_overrun_reserve_bps,
                ),
                library,
            )
            test = apply_fold_expert_library(
                apply_risk_sizing_contract(
                    label_local_plan_variants(raw_test, fee),
                    fee.round_trip_bps,
                    stop_loss_overrun_reserve_bps,
                ),
                library,
            )
        row_calibration_end = fold["calibration_start"] + pd.Timedelta(weeks=ROW_CALIBRATION_WEEKS)
        row_calibration = _period(
            calibration,
            fold["calibration_start"],
            row_calibration_end,
            purge_exit=True,
        )
        winner_calibration = _period(
            calibration,
            row_calibration_end,
            fold["selection_start"],
            purge_exit=True,
        )
        if min(len(row_calibration), len(winner_calibration)) == 0:
            raise ValueError("insufficient disjoint row and winner calibration windows")
        _status(
            "plan_efficiency",
            f"fold {number}/{len(folds)} local variants "
            f"{'enabled' if local_plan_variants_enabled else 'not required'}",
            44 + 12 * number / len(folds),
            fold=f"{number}/{len(folds)}",
            plan_efficiency=plan_audit,
            gpu=_gpu_info(),
        )
        continuation_model = fit_continuation_models(fit)
        continuation_calibration = fit_continuation_calibration(
            continuation_model, inner_calibration
        )
        continuation_audit_rows = model_audit.copy()
        continuation_audit_rows["expected_log_utility"] = 0.0
        continuation_audit = score_continuation(
            continuation_audit_rows,
            continuation_model,
            continuation_calibration,
        )
        continuation_audit_metrics = continuation_metrics(continuation_audit, continuation_model)
        kinds = ["ridge"] + (["xgboost_cuda"] if _xgb_available() else [])
        model_metrics: dict[str, Any] = {}
        champions: dict[int, str] = {}
        value_champions: dict[int, str] = {}
        scored_selection_pieces: list[pd.DataFrame] = []
        scored_test_pieces: list[pd.DataFrame] = []
        scored_winner_calibration_pieces: list[pd.DataFrame] = []
        permuted_test_pieces: list[pd.DataFrame] = []
        model_counter = 0
        total_models = 2 * len(kinds)
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            side_fit = fit.loc[fit["side"].eq(side)]
            side_inner = inner_calibration.loc[inner_calibration["side"].eq(side)]
            side_audit = model_audit.loc[model_audit["side"].eq(side)]
            side_row_calibration = row_calibration.loc[row_calibration["side"].eq(side)]
            side_winner_calibration = winner_calibration.loc[winner_calibration["side"].eq(side)]
            side_selection = selection.loc[selection["side"].eq(side)]
            side_test = test.loc[test["side"].eq(side)]
            side_metrics: dict[str, Any] = {}
            for kind in kinds:
                model_counter += 1
                internal = "xgboost" if kind == "xgboost_cuda" else kind
                _status(
                    "model_fit",
                    f"fold {number}/{len(folds)} {side_name} {model_counter}/{total_models} {kind}",
                    62 + 16 * ((number - 1) + model_counter / total_models) / len(folds),
                    fold=f"{number}/{len(folds)}",
                    side=side_name,
                    model=kind,
                    fit_rows=len(side_fit),
                    gpu=_gpu_info(),
                )
                candidate_head = fit_probability_head(internal, side_fit)
                candidate_calibration = fit_calibration(candidate_head, side_inner)
                scored_inner = score_actions(side_audit, candidate_head, candidate_calibration)
                value_metrics = {
                    value_head: head_metrics(scored_inner, value_head) for value_head in VALUE_HEADS
                }
                selected_value_head = choose_value_head(value_metrics)
                side_metrics[kind] = {
                    "value_heads": value_metrics,
                    "selected_value_head": selected_value_head,
                    "selected_metrics": value_metrics[selected_value_head],
                }
            champion = choose_champion(
                {kind: result["selected_metrics"] for kind, result in side_metrics.items()}
            )
            value_champion = str(side_metrics[champion]["selected_value_head"])
            champions[side] = champion
            value_champions[side] = value_champion
            model_metrics[side_name] = side_metrics
            internal = "xgboost" if champion == "xgboost_cuda" else champion
            side_refit = pd.concat([side_fit, side_inner, side_audit], ignore_index=True)
            head = fit_probability_head(internal, side_refit)
            calibrated = fit_calibration(head, side_row_calibration)
            scored_winner_calibration_pieces.append(
                score_actions(side_winner_calibration, head, calibrated, value_champion)
            )
            scored_selection_pieces.append(
                score_actions(side_selection, head, calibrated, value_champion)
            )
            scored_test_pieces.append(score_actions(side_test, head, calibrated, value_champion))
            permuted_head = fit_probability_head(
                "ridge", _permute_outcomes(side_fit, 20261200 + number * 10 + side)
            )
            permuted_calibration = fit_calibration(
                permuted_head,
                _permute_outcomes(side_inner, 20261300 + number * 10 + side),
            )
            permuted_test_pieces.append(
                score_actions(side_test, permuted_head, permuted_calibration, "decomposed")
            )
        post_selection_calibration = fit_post_selection_calibration(
            pd.concat(scored_winner_calibration_pieces, ignore_index=True)
        )
        scored_selection = apply_post_selection_calibration(
            pd.concat(scored_selection_pieces, ignore_index=True),
            post_selection_calibration,
        )
        scored_test = apply_post_selection_calibration(
            pd.concat(scored_test_pieces, ignore_index=True),
            post_selection_calibration,
        )
        permuted_test = pd.concat(permuted_test_pieces, ignore_index=True)
        continuation_refit = pd.concat([fit, inner_calibration, model_audit], ignore_index=True)
        continuation_model = fit_continuation_models(continuation_refit)
        continuation_calibration = fit_continuation_calibration(continuation_model, calibration)
        scored_selection = score_continuation(
            scored_selection,
            continuation_model,
            continuation_calibration,
            copy=False,
        )
        scored_test = score_continuation(
            scored_test,
            continuation_model,
            continuation_calibration,
            copy=False,
        )
        continuation_blocks = int(continuation_model["crossfit"]["blocks"])
        controllers: dict[int, str] = {}
        controller_audit: dict[str, Any] = {}
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            controller, audit = select_entry_controller(
                scored_selection.loc[scored_selection["side"].eq(side)],
                fee,
                fold["selection_start"],
                fold["test_start"],
                continuation_blocks,
            )
            controllers[side] = controller
            controller_audit[side_name] = audit
        scored_selection = apply_entry_controllers(scored_selection, controllers)
        scored_test = apply_entry_controllers(scored_test, controllers)
        thresholds: dict[int, float] = {1: 0.0, -1: 0.0}
        frontiers: dict[str, list[dict[str, Any]]] = {}
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            _, side_frontier = _choose_frequency_threshold(
                scored_selection.loc[scored_selection["side"].eq(side)],
                fee,
                fold["selection_start"],
                fold["test_start"],
            )
            frontiers[side_name] = side_frontier
        _status(
            "policy_replay",
            f"fold {number}/{len(folds)} LONG={controllers[1]}; SHORT={controllers[-1]}",
            62 + 16 * number / len(folds),
            fold=f"{number}/{len(folds)}",
            selected_entry_controllers={
                "LONG": controllers[1],
                "SHORT": controllers[-1],
            },
            selected_thresholds_bps={
                "LONG": thresholds[1] if math.isfinite(thresholds[1]) else None,
                "SHORT": thresholds[-1] if math.isfinite(thresholds[-1]) else None,
            },
            gpu=_gpu_info(),
        )
        test_frontier: list[dict[str, Any]] = []
        test_daily: dict[str, list[float]] = {}
        for candidate_threshold in THRESHOLDS_BPS:
            candidate_trades, _ = sequential_replay(
                scored_test,
                candidate_threshold,
                fee.round_trip_bps,
                record_decisions=False,
            )
            candidate_metrics = policy_metrics(
                candidate_trades,
                fold["test_start"],
                fold["test_end"],
                daily_bootstrap=False,
                weekly_bootstrap=False,
            )
            test_frontier.append(
                {"threshold_bps": candidate_threshold, "metrics": candidate_metrics}
            )
            test_daily[str(candidate_threshold)] = _daily_returns(
                candidate_trades, fold["test_start"], fold["test_end"]
            ).tolist()
        trades, decisions = sequential_replay(
            scored_test,
            thresholds,
            fee.round_trip_bps,
            risk_state=walk_forward_risk,
        )
        negative_controls = negative_control_metrics(
            scored_test,
            fee,
            fold["test_start"],
            fold["test_end"],
            label_exact_plans(train_median_constant_plans(test, fit), fee),
        )
        permuted_trades, _ = sequential_replay(
            permuted_test,
            0.0,
            fee.round_trip_bps,
            record_decisions=False,
        )
        negative_controls["label_permutation"] = policy_metrics(
            permuted_trades,
            fold["test_start"],
            fold["test_end"],
            daily_bootstrap=False,
            weekly_bootstrap=False,
        )
        if not trades.empty:
            trade_pieces.append(trades)
        if not decisions.empty:
            decision_pieces.append(decisions)
        metrics = policy_metrics(
            trades, fold["test_start"], fold["test_end"], weekly_bootstrap=False
        )
        diagnostics.append(
            {
                "fold": number,
                **{name: value.isoformat() for name, value in fold.items()},
                "fit_rows": len(fit),
                "inner_calibration_rows": len(inner_calibration),
                "model_audit_rows": len(model_audit),
                "calibration_rows": len(calibration),
                "row_calibration_rows": len(row_calibration),
                "winner_calibration_rows": len(winner_calibration),
                "selection_rows": len(selection),
                "test_rows": len(test),
                "candidate_metrics": model_metrics,
                "champions": {
                    "LONG": champions[1],
                    "SHORT": champions[-1],
                },
                "value_champions": {
                    "LONG": value_champions[1],
                    "SHORT": value_champions[-1],
                },
                "selected_thresholds_bps": {
                    "LONG": thresholds[1] if math.isfinite(thresholds[1]) else None,
                    "SHORT": thresholds[-1] if math.isfinite(thresholds[-1]) else None,
                },
                "entry_controller_selection": controller_audit,
                "post_selection_calibration": post_selection_calibration["audit"],
                "frequency_pnl_frontiers": frontiers,
                "test_frequency_pnl_frontier": test_frontier,
                "test_daily_returns_by_threshold": test_daily,
                "test_metrics": metrics,
                "negative_controls": negative_controls,
                "continuation_value_audit": continuation_audit_metrics,
                "economic_calibration": economic_calibration_buckets(scored_test),
                "entry_rule": "PAST_ONLY_MYOPIC_CHAMPION_VS_DOUBLE_Q_CONTINUATION_CHALLENGER",
                "view_gating_audit": view_gating_audit(scored_test),
                "plan_efficiency_audit": plan_audit,
                "local_plan_variants_enabled": local_plan_variants_enabled,
                "local_plan_training_support": local_plan_training_support,
                "stop_loss_overrun_reserve_bps": stop_loss_overrun_reserve_bps,
                "fold_experts": {
                    "fold_scope": library["fold_scope"],
                    "catalog_path": library["catalog_path"],
                    "candidates_evaluated": library["candidates_evaluated"],
                    "candidates_eligible": library["candidates_eligible"],
                    "terminal_prefilter_rejections": 0,
                    "fit_crossfit": crossfit_diagnostics,
                },
            }
        )
        del (
            continuation_refit,
            scored_selection,
            scored_test,
            permuted_test,
            scored_selection_pieces,
            scored_test_pieces,
            scored_winner_calibration_pieces,
            permuted_test_pieces,
            continuation_audit,
            continuation_audit_rows,
        )
        gc.collect()
    risk_violations = sum(int(piece.attrs.get("risk_violations", 0)) for piece in trade_pieces)
    trades = pd.concat(trade_pieces, ignore_index=True) if trade_pieces else matrix.iloc[:0].copy()
    trades.attrs["risk_violations"] = risk_violations
    decisions = (
        pd.concat(decision_pieces, ignore_index=True)
        if decision_pieces
        else pd.DataFrame(columns=["timestamp", "action", "reason"])
    )
    return trades, decisions, diagnostics


def _policy_return_matrix(folds: list[dict[str, Any]]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for threshold in THRESHOLDS_BPS:
        pieces = [
            np.asarray(item["test_daily_returns_by_threshold"][str(threshold)], dtype=float)
            for item in folds
            if str(threshold) in item.get("test_daily_returns_by_threshold", {})
        ]
        if pieces:
            columns.append(np.concatenate(pieces))
    if not columns:
        return np.empty((0, 0))
    length = min(len(column) for column in columns)
    return np.column_stack([column[:length] for column in columns])


def _spa_reality_check(folds: list[dict[str, Any]]) -> dict[str, float] | None:
    returns = _policy_return_matrix(folds)
    if len(returns) < 20 or not returns.shape[1]:
        return None
    test = SPA(
        np.zeros(len(returns)),
        -returns,
        block_size=min(5, len(returns)),
        reps=2_000,
        bootstrap="stationary",
        seed=20260831,
    )
    test.compute()
    return {str(name): float(value) for name, value in test.pvalues.items()}


def _deflated_sharpe_probability(daily: pd.Series, trials: int) -> float | None:
    values = daily.to_numpy(float)
    if len(values) < 20 or float(values.std(ddof=1)) <= 0:
        return None
    sharpe = float(values.mean() / values.std(ddof=1))
    centered = values - values.mean()
    scale = float(values.std(ddof=0))
    skewness = float(np.mean(centered**3) / scale**3)
    excess_kurtosis = float(np.mean(centered**4) / scale**4 - 3)
    benchmark = NormalDist().inv_cdf(1 - 1 / max(trials, 2)) / math.sqrt(len(values))
    standard_error = math.sqrt(
        max(
            1e-12,
            (1 - skewness * sharpe + (excess_kurtosis + 2) * sharpe**2 / 4) / len(values),
        )
    )
    return float(NormalDist().cdf((sharpe - benchmark) / standard_error))


def _pbo(fold_diagnostics: list[dict[str, Any]]) -> float | None:
    if len(fold_diagnostics) < 4:
        return None
    performance = np.asarray(
        [
            [
                float(candidate["metrics"].get("mean_daily_return") or 0)
                for candidate in item["test_frequency_pnl_frontier"]
            ]
            for item in fold_diagnostics
        ],
        dtype=float,
    )
    split = len(performance) // 2
    outcomes: list[bool] = []
    from itertools import combinations

    for selected in combinations(range(len(performance)), split):
        if 0 not in selected:
            continue
        training = np.asarray(selected, dtype=int)
        testing = np.asarray(
            [index for index in range(len(performance)) if index not in selected], dtype=int
        )
        winner = int(np.argmax(performance[training].mean(axis=0)))
        ranks = pd.Series(performance[testing].mean(axis=0)).rank(method="average", pct=True)
        outcomes.append(float(ranks.iloc[winner]) <= 0.5)
    return float(np.mean(outcomes)) if outcomes else None


def fit_forward_bundle(
    matrix: pd.DataFrame,
    fee: FeeContract,
    folds: list[dict[str, Any]],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    end = pd.to_datetime(matrix["actual_entry_timestamp"], utc=True).max().ceil("D")
    selection_start = end - pd.Timedelta(weeks=WINDOW_WEEKS)
    calibration_start = selection_start - pd.Timedelta(weeks=WINDOW_WEEKS)
    fit = _period(matrix, None, calibration_start, purge_exit=True)
    calibration = _period(matrix, calibration_start, selection_start, purge_exit=True)
    selection = _period(matrix, selection_start, end, purge_exit=True)
    stop_loss_overrun_reserve_bps = fit_stop_loss_overrun_reserve(fit, fee.round_trip_bps)
    fit = apply_risk_sizing_contract(fit, fee.round_trip_bps, stop_loss_overrun_reserve_bps)
    calibration = apply_risk_sizing_contract(
        calibration, fee.round_trip_bps, stop_loss_overrun_reserve_bps
    )
    selection = apply_risk_sizing_contract(
        selection, fee.round_trip_bps, stop_loss_overrun_reserve_bps
    )
    raw_fit = fit
    raw_calibration = calibration
    raw_selection = selection
    fit, library, crossfit_diagnostics = cross_fit_fold_expert_features(
        raw_fit,
        "forward",
        resume=resume,
    )
    calibration = apply_fold_expert_library(raw_calibration, library)
    selection = apply_fold_expert_library(raw_selection, library)
    local_plan_variants_enabled = (
        sum(bool(item.get("local_plan_variants_enabled")) for item in folds) > len(folds) / 2
    )
    local_plan_training_support = {"fit": {"sampled_states": 0, "added_rows": 0}}
    if local_plan_variants_enabled:
        augmented_fit, local_plan_training_support["fit"] = augment_local_plan_training_support(
            raw_fit,
            fee,
            fee.round_trip_bps,
            stop_loss_overrun_reserve_bps,
            LOCAL_PLAN_TRAINING_STATES,
        )
        fit, library, crossfit_diagnostics = cross_fit_fold_expert_features(
            augmented_fit,
            "forward-local",
            resume=resume,
        )
        calibration = apply_fold_expert_library(
            apply_risk_sizing_contract(
                label_local_plan_variants(raw_calibration, fee),
                fee.round_trip_bps,
                stop_loss_overrun_reserve_bps,
            ),
            library,
        )
        selection = apply_fold_expert_library(
            apply_risk_sizing_contract(
                label_local_plan_variants(raw_selection, fee),
                fee.round_trip_bps,
                stop_loss_overrun_reserve_bps,
            ),
            library,
        )
    row_calibration_end = calibration_start + pd.Timedelta(weeks=ROW_CALIBRATION_WEEKS)
    row_calibration = _period(calibration, calibration_start, row_calibration_end, purge_exit=True)
    winner_calibration = _period(calibration, row_calibration_end, selection_start, purge_exit=True)
    if min(len(row_calibration), len(winner_calibration)) == 0:
        raise ValueError("insufficient disjoint forward calibration windows")
    champions: dict[str, str] = {}
    value_champions: dict[str, str] = {}
    heads: dict[int, dict[str, Any]] = {}
    calibrations: dict[int, dict[str, Any]] = {}
    thresholds: dict[int, float] = {}
    frontiers: dict[str, list[dict[str, Any]]] = {}
    controllers: dict[int, str] = {}
    controller_audit: dict[str, Any] = {}
    scored_winner_calibration_pieces: list[pd.DataFrame] = []
    scored_selection_pieces: list[pd.DataFrame] = []
    continuation_model = fit_continuation_models(fit)
    continuation_calibration = fit_continuation_calibration(continuation_model, calibration)
    continuation_blocks = int(continuation_model["crossfit"]["blocks"])
    for side, side_name in ((1, "LONG"), (-1, "SHORT")):
        observed = [str(item["champions"][side_name]) for item in folds]
        champion = (
            "xgboost_cuda" if observed.count("xgboost_cuda") > observed.count("ridge") else "ridge"
        )
        champions[side_name] = champion
        observed_value_heads = [
            str(item.get("value_champions", {}).get(side_name, "decomposed")) for item in folds
        ]
        value_champion = (
            "direct"
            if observed_value_heads.count("direct") > observed_value_heads.count("decomposed")
            else "decomposed"
        )
        value_champions[side_name] = value_champion
        internal = "xgboost" if champion == "xgboost_cuda" else champion
        head = fit_probability_head(internal, fit.loc[fit["side"].eq(side)])
        calibrated = fit_calibration(head, row_calibration.loc[row_calibration["side"].eq(side)])
        scored_winner_calibration_pieces.append(
            score_actions(
                winner_calibration.loc[winner_calibration["side"].eq(side)],
                head,
                calibrated,
                value_champion,
            )
        )
        scored_selection_pieces.append(
            score_actions(
                selection.loc[selection["side"].eq(side)], head, calibrated, value_champion
            )
        )
        heads[side] = head
        calibrations[side] = calibrated
        thresholds[side] = 0.0
    post_selection_calibration = fit_post_selection_calibration(
        pd.concat(scored_winner_calibration_pieces, ignore_index=True)
    )
    scored_selection = apply_post_selection_calibration(
        pd.concat(scored_selection_pieces, ignore_index=True),
        post_selection_calibration,
    )
    scored_selection = score_continuation(
        scored_selection,
        continuation_model,
        continuation_calibration,
        copy=False,
    )
    for side, side_name in ((1, "LONG"), (-1, "SHORT")):
        side_selection = scored_selection.loc[scored_selection["side"].eq(side)]
        controller, audit = select_entry_controller(
            side_selection,
            fee,
            selection_start,
            end,
            continuation_blocks,
        )
        controllers[side] = controller
        controller_audit[side_name] = audit
        side_selection = apply_entry_controllers(side_selection, {side: controller})
        _, frontier = _choose_frequency_threshold(side_selection, fee, selection_start, end)
        frontiers[side_name] = frontier
    return {
        "champions": champions,
        "value_champions": value_champions,
        "heads": heads,
        "calibrations": calibrations,
        "post_selection_calibration": post_selection_calibration,
        "post_selection_calibration_audit": post_selection_calibration["audit"],
        "continuation_models": continuation_model,
        "continuation_calibration": continuation_calibration,
        "entry_controllers": {
            "LONG": controllers[1],
            "SHORT": controllers[-1],
        },
        "entry_controller_selection": controller_audit,
        "expert_library": library,
        "thresholds_bps": {
            "LONG": thresholds[1] if math.isfinite(thresholds[1]) else None,
            "SHORT": thresholds[-1] if math.isfinite(thresholds[-1]) else None,
        },
        "fit_end": calibration_start.isoformat(),
        "calibration_period": [calibration_start.isoformat(), selection_start.isoformat()],
        "row_calibration_period": [
            calibration_start.isoformat(),
            row_calibration_end.isoformat(),
        ],
        "winner_calibration_period": [
            row_calibration_end.isoformat(),
            selection_start.isoformat(),
        ],
        "policy_selection_period": [selection_start.isoformat(), end.isoformat()],
        "frequency_pnl_frontiers": frontiers,
        "local_plan_variants_enabled": local_plan_variants_enabled,
        "local_plan_training_support": local_plan_training_support,
        "stop_loss_overrun_reserve_bps": stop_loss_overrun_reserve_bps,
        "fold_experts": {
            "fold_scope": library["fold_scope"],
            "catalog_path": library["catalog_path"],
            "candidates_evaluated": library["candidates_evaluated"],
            "candidates_eligible": library["candidates_eligible"],
            "terminal_prefilter_rejections": 0,
            "fit_crossfit": crossfit_diagnostics,
        },
    }


def _economic_action_set(matrix: pd.DataFrame) -> dict[str, Any]:
    grouped = matrix.groupby("side").agg(
        count=("net_bps", "count"),
        mean_net_bps=("net_bps", "mean"),
        median_net_bps=("net_bps", "median"),
        mean_log_utility=("log_utility", "mean"),
    )
    oracle = matrix.groupby("actual_entry_timestamp")["net_bps"].max()
    utility_oracle = matrix.groupby("actual_entry_timestamp")["log_utility"].max()
    return {
        "parameterized_sides": grouped.reset_index().to_dict("records"),
        "unique_plan_ids": int(matrix["plan_id"].nunique()),
        "unique_horizons": int(matrix["horizon_seconds"].nunique()),
        "horizon_seconds": {
            "minimum": int(matrix["horizon_seconds"].min()),
            "median": float(matrix["horizon_seconds"].median()),
            "maximum": int(matrix["horizon_seconds"].max()),
        },
        "fixed_action_plans": False,
        "oracle_positive_fraction": float(oracle.gt(0).mean()),
        "oracle_mean_net_bps": float(oracle.mean()),
        "oracle_mean_log_utility": float(utility_oracle.mean()),
        "oracle_is_not_tradable": True,
        "has_positive_unconditional_action": bool(grouped["mean_log_utility"].gt(0).any()),
    }


def _side_metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, name in ((1, "LONG"), (-1, "SHORT")):
        active = trades.loc[trades["side"].eq(side)].copy()
        active.attrs.update(trades.attrs)
        metrics = policy_metrics(active, start, end)
        output[name] = {
            "metrics": metrics,
            "gates": policy_gates(metrics, minimum_trades=MINIMUM_SIDE_OOS_TRADES),
        }
    return output


def _verdict(
    economics: dict[str, Any],
    folds: list[dict[str, Any]],
    metrics: dict[str, Any],
    sides: dict[str, Any],
) -> str:
    if (
        not economics["has_positive_unconditional_action"]
        and float(economics.get("oracle_mean_log_utility", economics.get("oracle_mean_net_bps", 0)))
        <= 0
    ):
        return "NO_ECONOMIC_ACTION_SET"
    if not folds or sum(int(item["test_metrics"]["trades"]) for item in folds) == 0:
        return "NO_PREDICTABLE_EDGE"
    audited: list[float] = []

    def collect_numbers(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect_numbers(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            audited.append(float(value))

    for item in folds:
        collect_numbers(item.get("candidate_metrics", {}))
    if not audited or not all(math.isfinite(float(value)) for value in audited):
        return "NO_CALIBRATED_POLICY"
    side_enabled = any(all(result["gates"].values()) for result in sides.values())
    if not all(policy_gates(metrics).values()) or not side_enabled:
        return "NO_STABLE_OOS_POLICY"
    return "RESEARCH_PAPER_READY"


def write_stage_reports(
    folds: list[dict[str, Any]], metrics: dict[str, Any], gates: dict[str, bool]
) -> dict[str, str]:
    common = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "outer_folds": len(folds),
    }
    payloads = {
        CRITIC_CROSSFIT_REPORT: common
        | {
            "folds": [item.get("fold_experts", {}).get("fit_crossfit", {}) for item in folds],
            "strictly_past_only": bool(folds)
            and all(
                item.get("fold_experts", {})
                .get("fit_crossfit", {})
                .get("strictly_past_only", False)
                for item in folds
            ),
        },
        EQUITY_OBJECTIVE_REPORT: common
        | {
            "target": "log1p(risk-sized portfolio return)",
            "oos_metrics": metrics,
            "accounting_units_separated": ["notional_bps", "portfolio_return", "log_growth"],
        },
        WAIT_VALUE_REPORT: common
        | {
            "entry_rule": (
                "MIN(PAIRED_Q_ADVANTAGE, IMMEDIATE_EXPECTED_LOG_UTILITY)_GREATER_THAN_ZERO"
            ),
            "folds": [item.get("continuation_value_audit", {}) for item in folds],
            "wait_is_constant_zero": False,
        },
        PLAN_EFFICIENCY_REPORT: common
        | {
            "folds": [item.get("plan_efficiency_audit", {}) for item in folds],
            "enabled_by_fold": [bool(item.get("local_plan_variants_enabled")) for item in folds],
        },
        VALUE_HEADS_REPORT: common
        | {
            "candidate_metrics": [item.get("candidate_metrics", {}) for item in folds],
            "economic_calibration": [item.get("economic_calibration", {}) for item in folds],
        },
        ENTRY_STABILITY_REPORT: common
        | {
            "entry_rule": "DOMINANCE_SAFE_PAIRED_Q_ADVANTAGE_GT_ZERO",
            "threshold_tuning_enabled": False,
            "diagnostic_frontiers": [item.get("frequency_pnl_frontiers", {}) for item in folds],
        },
        INTRATRADE_REPORT: common
        | {
            "entry_policy_gate_passed": all(gates.values()),
            "activated": False,
            "reason": (
                "ENTRY_POLICY_PREREQUISITE_NOT_PASSED"
                if not all(gates.values())
                else "COUNTERFACTUAL_INTRATRADE_STAGE_REQUIRES_SEPARATE_FUTURE_CONFIRMATION"
            ),
            "deterministic_management_retained": True,
            "fake_learned_actions_forbidden": True,
        },
        VIEW_AUDIT_REPORT: common
        | {"folds": [item.get("view_gating_audit", {}) for item in folds]},
    }
    for path, payload in payloads.items():
        _atomic_json(path, payload)
    return {path.stem: str(path) for path in payloads}


def append_experiment_result(verdict: str, metrics: dict[str, Any]) -> None:
    payload = json.loads(EXPERIMENT_LEDGER.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", []) if isinstance(payload, dict) else payload
    result_id = f"MUSCA-BTC-{PROTOCOL_HASH[:12]}:RESULT"
    if any(item.get("event_id") == result_id for item in experiments):
        return
    experiments.append(
        {
            "event_id": result_id,
            "experiment_id": f"MUSCA-BTC-{PROTOCOL_HASH[:12]}",
            "event": "RESULT",
            "protocol_hash": PROTOCOL_HASH,
            "verdict": verdict,
            "accepted": verdict == "RESEARCH_PAPER_READY",
            "performance": metrics,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(EXPERIMENT_LEDGER, {"experiments": experiments})


def preflight_gates(
    metrics: dict[str, Any], folds: list[dict[str, Any]], total_fold_count: int
) -> dict[str, bool]:
    required_trades = math.ceil(MINIMUM_OOS_TRADES * len(folds) / max(total_fold_count, 1))
    fold_expectancy = [item["test_metrics"].get("expectancy_bps") for item in folds]
    return {
        "folds_complete": len(folds) == 2,
        "minimum_proportional_trades": int(metrics.get("trades", 0)) >= required_trades,
        "expectancy_positive": float(metrics.get("expectancy_bps") or -math.inf) > 0,
        "lower_confidence_bound_positive": float(metrics.get("daily_lcb_95") or -math.inf) > 0,
        "profit_factor_1_15": float(metrics.get("profit_factor") or -math.inf) >= 1.15,
        "drawdown_8pct": metrics.get("maximum_drawdown") is not None
        and float(metrics["maximum_drawdown"]) <= MAXIMUM_DRAWDOWN,
        "majority_active_days_positive": float(metrics.get("positive_active_days") or 0.0) > 0.5,
        "majority_folds_positive": sum(
            value is not None and float(value) > 0 for value in fold_expectancy
        )
        > len(folds) / 2,
        "risk_respected": int(metrics.get("risk_violations", 0)) == 0,
        "utility_ev_consistent": all(
            int(item["economic_calibration"].get("utility_ev_consistency_violations", 1)) == 0
            for item in folds
        ),
        "fitted_continuation_only": all(
            item["continuation_value_audit"].get("target_source")
            == "DOUBLE_TEMPORAL_FITTED_IMMEDIATE_VALUE"
            and bool(item["continuation_value_audit"].get("crossfit", {}).get("strictly_past_only"))
            and int(item["continuation_value_audit"].get("crossfit", {}).get("blocks", 0))
            >= MINIMUM_CONTINUATION_CROSSFIT_BLOCKS
            for item in folds
        ),
        "controller_selection_past_only": all(
            all(
                bool(side_audit.get("selection_period_only"))
                and not bool(side_audit.get("outer_test_read_for_selection", True))
                for side_audit in item.get("entry_controller_selection", {}).values()
            )
            for item in folds
        ),
        "post_selection_calibration_past_only": all(
            bool(item.get("post_selection_calibration", {}).get("strictly_past_only"))
            and pd.Timestamp(item["post_selection_calibration"]["end"])
            < pd.Timestamp(item["selection_start"])
            for item in folds
        ),
        "local_actions_supported_in_fit": all(
            not item.get("local_plan_variants_enabled")
            or int(item.get("local_plan_training_support", {}).get("fit", {}).get("added_rows", 0))
            > 0
            for item in folds
        ),
    }


def preflight(*, resume: bool = False) -> dict[str, Any]:
    """Run two frozen discovery folds; the ten-fold audit is forbidden until this passes."""
    global _RUN_STARTED
    _RUN_STARTED = time.monotonic()
    started = time.monotonic()
    _status("preflight", "loading canonical Binance state-actions", 0.5, gpu=_gpu_info())
    registry = build_research_registry()
    fee = resolve_fee_contract()
    source_manifest = ensure_one_second_sources()
    execution_contract = build_execution_contract(source_manifest, fee)
    matrix, partitions = build_state_actions(fee, resume=resume)
    if pd.to_datetime(matrix["actual_entry_timestamp"], utc=True).ge(FUTURE_HOLDOUT_START).any():
        raise ValueError("sealed future holdout was read")
    all_folds = _folds(matrix)
    trades, decisions, folds = walk_forward(matrix, fee, resume=resume, fold_limit=2)
    audit_start = min(pd.Timestamp(item["test_start"]) for item in folds)
    audit_end = max(pd.Timestamp(item["test_end"]) for item in folds)
    metrics = policy_metrics(trades, audit_start, audit_end)
    gates = preflight_gates(metrics, folds, len(all_folds))
    verdict = "PREFLIGHT_PASSED" if all(gates.values()) else "PREFLIGHT_FAILED"
    PREFLIGHT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREFLIGHT_TRADES.with_suffix(".parquet.tmp")
    trades.to_parquet(temporary, index=False)
    os.replace(temporary, PREFLIGHT_TRADES)
    temporary_decisions = PREFLIGHT_DECISIONS.with_suffix(".parquet.tmp")
    decisions.to_parquet(temporary_decisions, index=False)
    os.replace(temporary_decisions, PREFLIGHT_DECISIONS)
    report = {
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "discovery_only": True,
        "future_holdout_opened": False,
        "full_training_authorized": all(gates.values()),
        "fee_contract": asdict(fee) | {"round_trip_bps": fee.round_trip_bps},
        "data": {
            "source_manifest": source_manifest,
            "label_partitions": partitions,
            "state_action_rows": len(matrix),
            "future_holdout_rows_read": 0,
        },
        "execution": execution_contract,
        "registry": registry,
        "folds": folds,
        "metrics": metrics,
        "gates": gates,
        "verdict": verdict,
    }
    _atomic_json(PREFLIGHT_REPORT, report)
    _status(
        "preflight_complete",
        verdict,
        100,
        verdict=verdict,
        full_training_authorized=all(gates.values()),
        gpu=_gpu_info(),
    )
    return report


def train(*, resume: bool = False) -> dict[str, Any]:
    global _RUN_STARTED
    _RUN_STARTED = time.monotonic()
    started = time.monotonic()
    try:
        preflight_report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        preflight_report = {}
    if preflight_report.get("protocol_hash") != PROTOCOL_HASH or not bool(
        preflight_report.get("full_training_authorized")
    ):
        raise RuntimeError(
            "full training is forbidden until musca-btc-policy-train --preflight-only passes"
        )
    frozen_hash_before = _sha256(AUTO_MOE_REPORT)
    _status("registry", "registering all prior protocols and experts", 0.5, gpu=_gpu_info())
    registry = build_research_registry()
    fee = resolve_fee_contract()
    _status(
        "dataset_preflight",
        f"Binance taker {fee.taker_bps_per_side:.4f} bps/side; 1x RT {fee.round_trip_bps:.4f} bps",
        1,
        fee_source=fee.source,
        gpu=_gpu_info(),
    )
    source_manifest = ensure_one_second_sources()
    execution_contract = build_execution_contract(source_manifest, fee)
    matrix, partitions = build_state_actions(fee, resume=resume)
    if pd.to_datetime(matrix["actual_entry_timestamp"], utc=True).ge(FUTURE_HOLDOUT_START).any():
        raise ValueError("sealed future holdout was read")
    economics = _economic_action_set(matrix)
    _status(
        "economic_action_set",
        f"{len(matrix):,} exact managed state-actions; oracle is diagnostic only",
        41,
        economic_action_set=economics,
    )
    if (
        not economics["has_positive_unconditional_action"]
        and economics["oracle_mean_log_utility"] <= 0
    ):
        trades = matrix.iloc[:0].copy()
        decisions = pd.DataFrame()
        folds: list[dict[str, Any]] = []
    else:
        trades, decisions, folds = walk_forward(matrix, fee, resume=resume)
    audit_start = min(
        (pd.Timestamp(item["test_start"]) for item in folds), default=HISTORICAL_START
    )
    audit_end = max((pd.Timestamp(item["test_end"]) for item in folds), default=HISTORICAL_END)
    metrics = policy_metrics(trades, audit_start, audit_end)
    sides = _side_metrics(trades, audit_start, audit_end)
    verdict = _verdict(economics, folds, metrics, sides)
    gates = policy_gates(metrics)
    stage_reports = write_stage_reports(folds, metrics, gates)
    append_experiment_result(verdict, metrics)
    forward_bundle: dict[str, Any] | None = None
    if folds:
        _status("forward_bundle", "fitting frozen research-paper controller", 82, gpu=_gpu_info())
        forward_bundle = fit_forward_bundle(matrix, fee, folds, resume=resume)
        if verdict == "RESEARCH_PAPER_READY" and not any(
            value != "DISABLED" for value in forward_bundle["entry_controllers"].values()
        ):
            verdict = "NO_CALIBRATED_POLICY"
    daily = _daily_returns(trades, audit_start, audit_end)
    generated_expert_candidates = sum(
        int(item.get("fold_experts", {}).get("candidates_evaluated", 0)) for item in folds
    )
    if forward_bundle is not None:
        generated_expert_candidates += int(forward_bundle["fold_experts"]["candidates_evaluated"])
    generated_experts_eligible = sum(
        int(item.get("fold_experts", {}).get("candidates_eligible", 0)) for item in folds
    )
    if forward_bundle is not None:
        generated_experts_eligible += int(forward_bundle["fold_experts"]["candidates_eligible"])
    registry["registered_fold_local_expert_candidates"] = generated_expert_candidates
    registry["registered_fold_local_experts_eligible_after_management"] = generated_experts_eligible
    registry["total_registered_expert_attempts"] = (
        int(registry["registered_auto_moe_experts"]) + generated_expert_candidates
    )
    ledger = json.loads(EXPERIMENT_LEDGER.read_text(encoding="utf-8"))
    registry["experiment_count"] = len(ledger.get("experiments", []))
    _atomic_json(REGISTRY, registry)
    multiple_comparison = {
        "global_protocols": int(registry["protocol_count_observed"]),
        "global_experts": int(registry["registered_auto_moe_experts"])
        + generated_expert_candidates,
        "fold_local_expert_candidates": generated_expert_candidates,
        "fold_local_experts_eligible_after_management": generated_experts_eligible,
        "terminal_prefilter_rejections": 0,
        "spa_reality_check": _spa_reality_check(folds),
        "pbo": _pbo(folds),
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            daily,
            int(registry["protocol_count_observed"])
            + int(registry["registered_auto_moe_experts"])
            + generated_expert_candidates,
        ),
    }
    AUDIT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_TRADES.with_suffix(".parquet.tmp")
    trades.to_parquet(temporary, index=False)
    os.replace(temporary, AUDIT_TRADES)
    temporary_decisions = AUDIT_DECISIONS.with_suffix(".parquet.tmp")
    decisions.to_parquet(temporary_decisions, index=False)
    os.replace(temporary_decisions, AUDIT_DECISIONS)
    frozen_hash_after = _sha256(AUTO_MOE_REPORT)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("frozen Auto-MoE report changed during challenger training")
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "fee_contract": asdict(fee) | {"round_trip_bps": fee.round_trip_bps},
        "data": {
            "source_manifest": source_manifest,
            "label_partitions": partitions,
            "state_action_rows": len(matrix),
            "future_holdout_rows_read": 0,
            "fold_expert_catalog_root": str(EXPERT_CATALOG_ROOT),
            "execution_contract": str(EXECUTION_REPORT),
        },
        "execution": execution_contract,
        "registry": registry,
        "frozen_auto_moe_report_sha256": frozen_hash_after,
        "economic_action_set": economics,
        "walk_forward": folds,
        "oos_metrics": metrics,
        "side_controls": sides,
        "gates": gates,
        "stage_reports": stage_reports,
        "multiple_comparison": multiple_comparison,
        "forward_bundle": (
            None
            if forward_bundle is None
            else {
                name: value
                for name, value in forward_bundle.items()
                if name
                not in {
                    "heads",
                    "calibrations",
                    "post_selection_calibration",
                    "continuation_models",
                    "continuation_calibration",
                    "expert_library",
                }
            }
        ),
        "verdict": verdict,
        "paper_orders_enabled": verdict == "RESEARCH_PAPER_READY",
        "live_orders_enabled": False,
        "future_holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_joblib(
        BUNDLE,
        {
            "protocol": PROTOCOL,
            "protocol_hash": PROTOCOL_HASH,
            "verdict": verdict,
            "research_only": True,
            "paper_orders_enabled": verdict == "RESEARCH_PAPER_READY",
            "live_orders_enabled": False,
            "future_holdout_opened": False,
            "real_capital_allowed": False,
            "alpha_features": ALPHA_FEATURES,
            "fold_expert_features": FOLD_EXPERT_FEATURES,
            "model_features": MODEL_FEATURES,
            "models_by_side": None if forward_bundle is None else forward_bundle["heads"],
            "calibrations_by_side": (
                None if forward_bundle is None else forward_bundle["calibrations"]
            ),
            "post_selection_calibration": (
                None if forward_bundle is None else forward_bundle["post_selection_calibration"]
            ),
            "continuation_models": (
                None if forward_bundle is None else forward_bundle["continuation_models"]
            ),
            "continuation_calibration": (
                None if forward_bundle is None else forward_bundle["continuation_calibration"]
            ),
            "expert_library": (
                None if forward_bundle is None else forward_bundle["expert_library"]
            ),
            "thresholds_bps": (
                None if forward_bundle is None else forward_bundle["thresholds_bps"]
            ),
            "entry_controllers": (
                None if forward_bundle is None else forward_bundle["entry_controllers"]
            ),
            "enabled_sides": [
                name for name, result in sides.items() if all(result["gates"].values())
            ],
            "note": "research-paper model; confirmation requires sealed future holdout",
        },
    )
    _atomic_json(REPORT, report)
    _status("complete", verdict, 100, verdict=verdict, gpu=_gpu_info())
    return report


def status(*, watch: bool = False, interval: float = 5.0) -> int:
    last = ""
    while True:
        if STATUS.exists():
            content = STATUS.read_text(encoding="utf-8")
            if content != last:
                print(content, flush=True)
                last = content
            payload = json.loads(content)
            if payload.get("phase") in {"complete", "failed"}:
                return 0 if payload.get("phase") == "complete" else 2
        else:
            print(json.dumps({"phase": "not_started", "percent": 0}), flush=True)
        if not watch:
            return 0
        time.sleep(max(0.5, interval))


def failed_status(error: Exception) -> None:
    _status("failed", f"{type(error).__name__}: {error}", 0, error_type=type(error).__name__)


def preregistered_thresholds() -> Iterable[float]:
    return THRESHOLDS_BPS
