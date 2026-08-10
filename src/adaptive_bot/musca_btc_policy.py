from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
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
LABEL_ROOT = ROOT / "state_actions"
REGISTRY = ROOT / "research_registry.json"
REPORT = Path("data/reports/musca_btc_policy.json")
STATUS = Path("data/reports/musca_btc_policy.status.json")
BUNDLE = Path("data/models/musca_btc_policy/research_bundle.joblib")
AUDIT_TRADES = ROOT / "audit_trades.parquet"
AUDIT_DECISIONS = ROOT / "audit_decisions.parquet"
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
OUTCOME_TARGET = 0
OUTCOME_STOP = 1
OUTCOME_TIMEOUT = 2
OUTCOME_NAMES = ("TARGET", "STOP", "TIMEOUT")
MODEL_SEEDS = (20260831, 20260901, 20260902)
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
MINIMUM_EXPERT_OPPORTUNITIES = 100
MINIMUM_OOS_TRADES = 300
MINIMUM_SIDE_OOS_TRADES = 100
EXPERT_CATALOG_ROOT = ROOT / "fold_experts"
LEGACY_LABEL_PROTOCOL_HASHES = {"a20eb033eb076bfd9d0019e0bbb3e3956037b9ffdd28f618400f991494fac258"}

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
    "path_resolution_seconds": 1,
    "sides": ["LONG", "SHORT"],
    "horizons_seconds": list(base.HORIZONS),
    "management": "TP1/TP2/initial stop/non-widening trailing/timeout",
    "same_second": "stop wins",
    "entry": "first observed aggregate trade after decision",
    "terminal_return_prefilter": False,
    "historical_start": HISTORICAL_START.isoformat(),
    "historical_end": HISTORICAL_END.isoformat(),
}
LABEL_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(LABEL_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

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
        "sides": ["LONG", "SHORT"],
        "horizons_seconds": list(base.HORIZONS),
        "management": "same TP1/TP2/stop/non-widening trailing path for labels and replay",
        "same_second": "stop wins",
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
    },
    "fold_local_experts": {
        "generator": "XGBRFRegressor CUDA trained on exact managed net_bps",
        "candidate": "every tree leaf for LONG/SHORT and every horizon",
        "terminal_prefilter": False,
        "minimum_support_after_managed_evaluation": MINIMUM_EXPERT_OPPORTUNITIES,
        "compression": "active-leaf managed statistics plus deterministic winning expert",
        "training_scope": "outer-fold fit only",
    },
    "controller": {
        "actions": ["WAIT", "ENTER_LONG", "ENTER_SHORT", "HOLD", "CLOSE", "TIGHTEN_STOP"],
        "maximum_positions": 1,
        "forced_utc_close": False,
        "risk_per_trade": RISK_PER_TRADE,
        "maximum_leverage": MAXIMUM_LEVERAGE,
        "maximum_daily_loss": MAXIMUM_DAILY_LOSS,
        "state_features": list(STATE_FEATURES),
    },
    "validation": {
        "nested_walk_forward": {
            "minimum_fit_weeks": MINIMUM_FIT_WEEKS,
            "inner_calibration_weeks": INNER_CALIBRATION_WEEKS,
            "inner_model_audit_weeks": INNER_MODEL_AUDIT_WEEKS,
            "calibration_weeks": WINDOW_WEEKS,
            "policy_selection_weeks": WINDOW_WEEKS,
            "outer_test_weeks": WINDOW_WEEKS,
        },
        "purge": "actual exit timestamp",
        "bootstrap_unit": ["day", "week"],
        "global_trials": "research registry; all previously observed periods are contaminated",
        "frequency": "maximum net log-equity utility on policy-selection data; no quota",
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
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
        "future_holdout_opened": False,
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


def fold_expert_id(
    fold_scope: str, side: int, horizon_seconds: int, tree: int, leaf: int
) -> str:
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
                "target_1_bps",
                "target_2_bps",
                "stop_bps",
                "trailing_bps",
                *ALPHA_FEATURES,
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


def _simulate_cpu(
    source: pd.DataFrame,
    positions: np.ndarray,
    side: int,
    horizon: int,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
) -> dict[str, np.ndarray]:
    opens = source["open"].to_numpy(float)
    highs = source["high"].to_numpy(float)
    lows = source["low"].to_numpy(float)
    closes = source["close"].to_numpy(float)
    count = len(positions)
    gross = np.empty(count, dtype=float)
    exit_seconds = np.empty(count, dtype=np.int32)
    management = np.empty(count, dtype=np.int8)
    first_target = np.full(count, -1, dtype=np.int32)
    first_stop = np.full(count, -1, dtype=np.int32)
    mfe = np.empty(count, dtype=float)
    mae = np.empty(count, dtype=float)
    for row in range(count):
        position = int(positions[row])
        entry = opens[position]
        stop_level = -float(stop[row])
        peak = 0.0
        half = 0.0
        first_filled = False
        done = False
        maximum = -math.inf
        adverse_maximum = -math.inf
        result = 0.0
        result_seconds = horizon
        result_code = OUTCOME_TIMEOUT
        for offset in range(horizon):
            index = position + offset
            if index >= len(source):
                raise ValueError("incomplete one-second future path")
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
                result = half + 0.5 * open_return if first_filled else open_return
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
                    result = 0.5 * target_1[row] + 0.5 * target_2[row]
                    result_seconds = elapsed
                    result_code = 3
                    done = True
                    continue
                first_filled = True
                half = 0.5 * target_1[row]
                stop_level = max(stop_level, 0.0)
            elif first_filled:
                if -adverse <= stop_level + 1e-9:
                    result = half + 0.5 * stop_level
                    result_seconds = elapsed
                    result_code = 5
                    done = True
                    continue
                if favorable >= target_2[row] - 1e-9:
                    result = half + 0.5 * target_2[row]
                    result_seconds = elapsed
                    result_code = 3
                    done = True
                    continue
                peak = max(peak, favorable)
                stop_level = max(stop_level, peak - trailing[row])
        if not done:
            terminal = side * (closes[position + horizon - 1] / entry - 1) * 10_000
            result = half + 0.5 * terminal if first_filled else terminal
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
            const long long* positions, const double* target1, const double* target2,
            const double* stops, const double* trails, const int side, const int horizon,
            const long long source_size, const long long count, double* gross, int* exit_seconds,
            signed char* management, int* first_target, int* first_stop, double* mfe, double* mae) {
          long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
          if (row >= count) return;
          long long position = positions[row];
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
              result = first_filled ? half + 0.5 * open_return : open_return;
              result_seconds = elapsed; result_code = 4; done = 1; continue;
            }
            if (!first_filled && -adverse <= stop_level + 1.0e-9) {
              result = stop_level; result_seconds = elapsed; result_code = 1; done = 1; continue;
            }
            if (!first_filled && favorable >= target1[row] - 1.0e-9) {
              if (favorable >= target2[row] - 1.0e-9) {
                result = 0.5 * target1[row] + 0.5 * target2[row];
                result_seconds = elapsed; result_code = 3; done = 1; continue;
              }
              first_filled = 1; half = 0.5 * target1[row]; stop_level = fmax(stop_level, 0.0);
            } else if (first_filled) {
              if (-adverse <= stop_level + 1.0e-9) {
                result = half + 0.5 * stop_level;
                result_seconds = elapsed; result_code = 5; done = 1; continue;
              }
              if (favorable >= target2[row] - 1.0e-9) {
                result = half + 0.5 * target2[row];
                result_seconds = elapsed; result_code = 3; done = 1; continue;
              }
              peak = fmax(peak, favorable);
              stop_level = fmax(stop_level, peak - trails[row]);
            }
          }
          if (!done) {
            double terminal = side * (closes[position + horizon - 1] / entry - 1.0) * 10000.0;
            result = first_filled ? half + 0.5 * terminal : terminal;
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
    horizon: int,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
) -> dict[str, np.ndarray]:
    import cupy as cp

    count = len(positions)
    device_values = [
        cp.asarray(source[name].to_numpy(np.float64)) for name in ("open", "high", "low", "close")
    ]
    gpu_positions = cp.asarray(positions, dtype=cp.int64)
    gpu_parameters = [
        cp.asarray(values, dtype=cp.float64) for values in (target_1, target_2, stop, trailing)
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
            *gpu_parameters,
            np.int32(side),
            np.int32(horizon),
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
    target_time = cp.asnumpy(first_target)
    stop_time = cp.asnumpy(first_stop)
    event = np.where(
        (stop_time >= 0) & ((target_time < 0) | (stop_time <= target_time)),
        OUTCOME_STOP,
        np.where(target_time >= 0, OUTCOME_TARGET, OUTCOME_TIMEOUT),
    ).astype(np.int8)
    return {
        "gross_bps": cp.asnumpy(gross),
        "exit_seconds": cp.asnumpy(exit_seconds),
        "management_code": cp.asnumpy(management),
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
    horizon: int,
    target_1: np.ndarray,
    target_2: np.ndarray,
    stop: np.ndarray,
    trailing: np.ndarray,
    *,
    backend: str = "auto",
) -> dict[str, np.ndarray]:
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown path backend: {backend}")
    if np.any(np.asarray(positions) + horizon > len(source)):
        raise ValueError("incomplete path for requested horizon")
    if backend != "cpu":
        try:
            return _simulate_gpu(
                source, positions, side, horizon, target_1, target_2, stop, trailing
            )
        except (ImportError, RuntimeError):
            if backend == "cuda":
                raise
    return _simulate_cpu(source, positions, side, horizon, target_1, target_2, stop, trailing)


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
    actions = _load_parent_actions(start, end)
    if actions.empty:
        return actions
    source = _load_second_window(month)
    positions, entry_delay = _first_observed_positions(source, actions["entry_timestamp"])
    actions["actual_entry_timestamp"] = source.loc[positions, "timestamp"].to_numpy()
    actions["entry_price"] = source.loc[positions, "open"].to_numpy(float)
    actions["entry_delay_seconds"] = entry_delay
    pieces: list[pd.DataFrame] = []
    groups = list(actions.groupby(["side", "horizon_seconds"], sort=True))
    for number, ((side_value, horizon_value), group) in enumerate(groups, start=1):
        side = int(cast(Any, side_value))
        horizon = int(cast(Any, horizon_value))
        direction = "LONG" if side > 0 else "SHORT"
        indexes = group.index.to_numpy(int)
        result = simulate_management(
            source,
            positions[indexes],
            side,
            horizon,
            group["target_1_bps"].to_numpy(float),
            group["target_2_bps"].to_numpy(float),
            group["stop_bps"].to_numpy(float),
            group["trailing_bps"].to_numpy(float),
        )
        labelled = group.copy()
        for name, values in result.items():
            labelled[name] = values
        labelled["expert_id"] = expert_id(side, horizon)
        labelled["outcome"] = [
            _management_name(int(value)) for value in labelled["management_code"]
        ]
        labelled["event"] = [OUTCOME_NAMES[int(value)] for value in labelled["event_class"]]
        labelled["exit_timestamp"] = labelled["actual_entry_timestamp"] + pd.to_timedelta(
            labelled["exit_seconds"], unit="s"
        )
        labelled["source_month"] = month
        pieces.append(labelled)
        _status(
            "state_action_labels",
            f"{month} action {number}/{len(groups)}: {direction} {horizon}s",
            12
            + 28 * (ONE_SECOND_MONTHS.index(month) + number / len(groups)) / len(ONE_SECOND_MONTHS),
            month=month,
            action=f"{side}:{horizon}",
            gpu=_gpu_info(),
        )
    output = pd.concat(pieces, ignore_index=True)
    output["funding_bps"] = _funding_for_actions(output)
    output["round_trip_cost_bps"] = fee.round_trip_bps
    output["net_bps"] = output["gross_bps"] + output["funding_bps"] - fee.round_trip_bps
    output["stress_1_5x_bps"] = (
        output["gross_bps"] + output["funding_bps"] - 1.5 * fee.round_trip_bps
    )
    output["stress_2x_bps"] = output["gross_bps"] + output["funding_bps"] - 2.0 * fee.round_trip_bps
    output["label_protocol_hash"] = LABEL_PROTOCOL_HASH
    output["protocol_hash"] = PROTOCOL_HASH
    if not output["available_at"].le(output["actual_entry_timestamp"]).all():
        raise ValueError("canonical label entered before features were available")
    return output.sort_values(["actual_entry_timestamp", "side", "horizon_seconds"]).reset_index(
        drop=True
    )


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
            legacy_labels = bool(
                "label_protocol_hash" not in identity
                and not identity.empty
                and identity["protocol_hash"].isin(LEGACY_LABEL_PROTOCOL_HASHES).all()
            )
            cached = current_labels or legacy_labels
            migrate = cached and (
                legacy_labels
                or not identity["protocol_hash"].eq(PROTOCOL_HASH).all()
                or not np.allclose(
                    identity["round_trip_cost_bps"].to_numpy(float), fee.round_trip_bps
                )
            )
        if cached:
            rows = pd.read_parquet(path)
            if migrate:
                rows["round_trip_cost_bps"] = fee.round_trip_bps
                rows["net_bps"] = rows["gross_bps"] + rows["funding_bps"] - fee.round_trip_bps
                rows["stress_1_5x_bps"] = (
                    rows["gross_bps"] + rows["funding_bps"] - 1.5 * fee.round_trip_bps
                )
                rows["stress_2x_bps"] = (
                    rows["gross_bps"] + rows["funding_bps"] - 2.0 * fee.round_trip_bps
                )
                rows["expert_id"] = [
                    expert_id(int(side), int(horizon))
                    for side, horizon in zip(rows["side"], rows["horizon_seconds"], strict=True)
                ]
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
    values = rows.loc[:, base.GATING_CONTEXT].to_numpy(np.float32)
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
            arrays["stop_rate"][leaf_index] = float(
                np.mean(event_class[active] == OUTCOME_STOP)
            )
            arrays["eligible"][leaf_index] = float(eligible)
            catalog.append(
                {
                    "expert_id": fold_expert_id(
                        fold_scope, side, horizon, tree, leaf_index
                    ),
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
    actions = sorted(
        (int(side), int(horizon))
        for side, horizon in fit.loc[:, ["side", "horizon_seconds"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for action_number, (side, horizon) in enumerate(actions, start=1):
        active = fit.loc[
            fit["side"].eq(side) & fit["horizon_seconds"].eq(horizon)
        ].copy()
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
                f"fold {fold_number} action {action_number}/{len(actions)}: "
                f"{'LONG' if side > 0 else 'SHORT'} {horizon}s",
                42 + 20 * (completed + action_number / len(actions)) / max(total, 1),
                fold=str(fold_number),
                action=f"{side}:{horizon}",
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
        "candidates_eligible": int(
            catalog_frame["eligible_after_managed_evaluation"].sum()
        ),
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
    for (side, horizon), group in library["groups"].items():
        positions = np.flatnonzero(
            output["side"].eq(side).to_numpy()
            & output["horizon_seconds"].eq(horizon).to_numpy()
        )
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


def _timestamp_weights(rows: pd.DataFrame) -> np.ndarray:
    count = rows.groupby("actual_entry_timestamp")["actual_entry_timestamp"].transform("size")
    return np.asarray(1.0 / count.to_numpy(float), dtype=float)


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
    return {"kind": kind, "classifier": classifier, "conditional": conditional, "aux": auxiliaries}


def fit_calibration(head: dict[str, Any], calibration: pd.DataFrame) -> dict[str, Any]:
    values = _x(calibration)
    raw_probability = np.clip(_probabilities(head["classifier"], values), 1e-6, 1.0)
    outcome = calibration["event_class"].to_numpy(int)
    probability = LogisticRegression(C=1.0, max_iter=2_000, random_state=20260831).fit(
        np.log(raw_probability), outcome
    )
    calibrated_probability = _probabilities(probability, np.log(raw_probability))
    conditional = np.column_stack(
        [np.asarray(head["conditional"][event].predict(values), dtype=float) for event in range(3)]
    )
    raw_ev = np.sum(calibrated_probability * conditional, axis=1)
    ev = IsotonicRegression(out_of_bounds="clip").fit(
        raw_ev, calibration["net_bps"].to_numpy(float)
    )
    residuals: dict[str, list[float]] = {}
    for name in ("mfe_bps", "mae_bps"):
        predicted = np.asarray(head["aux"][name].predict(values), dtype=float)
        residual = calibration[name].to_numpy(float) - predicted
        residuals[name] = [float(np.quantile(residual, value)) for value in (0.10, 0.50, 0.90)]
    return {"probability": probability, "ev": ev, "residual_quantiles": residuals}


def score_actions(
    rows: pd.DataFrame, head: dict[str, Any], calibration: dict[str, Any]
) -> pd.DataFrame:
    values = _x(rows)
    raw_probability = np.clip(_probabilities(head["classifier"], values), 1e-6, 1.0)
    probability = _probabilities(calibration["probability"], np.log(raw_probability))
    conditional = np.column_stack(
        [np.asarray(head["conditional"][event].predict(values), dtype=float) for event in range(3)]
    )
    raw_ev = np.sum(probability * conditional, axis=1)
    output = rows.copy()
    output["p_target"] = probability[:, OUTCOME_TARGET]
    output["p_stop"] = probability[:, OUTCOME_STOP]
    output["p_timeout"] = probability[:, OUTCOME_TIMEOUT]
    output["target_probability"] = output["p_target"]
    for event, name in enumerate(OUTCOME_NAMES):
        output[f"expected_{name.lower()}_net_bps"] = conditional[:, event]
    output["raw_ev_bps"] = raw_ev
    output["calibrated_ev_bps"] = calibration["ev"].predict(raw_ev)
    output["expected_time_to_target_seconds"] = np.maximum(
        np.asarray(head["aux"]["time_to_target_seconds"].predict(values), dtype=float), 1.0
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


def _decision_regret(rows: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in rows.groupby("actual_entry_timestamp", sort=False):
        prediction = group["calibrated_ev_bps"].to_numpy(float)
        actual = group["net_bps"].to_numpy(float)
        choice = int(np.argmax(prediction))
        selected = actual[choice] if prediction[choice] > 0 else 0.0
        values.append(max(0.0, float(actual.max())) - selected)
    return float(np.mean(values)) if values else float("inf")


def head_metrics(rows: pd.DataFrame) -> dict[str, float]:
    probability = rows.loc[:, ["p_target", "p_stop", "p_timeout"]].to_numpy(float)
    actual = rows["net_bps"].to_numpy(float)
    predicted = rows["calibrated_ev_bps"].to_numpy(float)
    return {
        "brier": _multiclass_brier(rows["event_class"].to_numpy(int), probability),
        "ev_calibration_error_bps": _calibration_error(actual, predicted),
        "ev_mae_bps": float(mean_absolute_error(actual, predicted)),
        "decision_regret_bps": _decision_regret(rows),
    }


def choose_champion(metrics: dict[str, dict[str, float]]) -> str:
    if "xgboost_cuda" not in metrics:
        return "ridge"
    ridge = metrics["ridge"]
    challenger = metrics["xgboost_cuda"]
    keys = ("brier", "ev_calibration_error_bps", "ev_mae_bps", "decision_regret_bps")
    return "xgboost_cuda" if all(challenger[key] < ridge[key] for key in keys) else "ridge"


def _leverage(stop_bps: np.ndarray, round_trip_cost_bps: float) -> np.ndarray:
    risk_fraction = (np.asarray(stop_bps, dtype=float) + round_trip_cost_bps) / 10_000
    return np.minimum(MAXIMUM_LEVERAGE, RISK_PER_TRADE / np.maximum(risk_fraction, 1e-9))


def sequential_replay(
    scored: pd.DataFrame,
    threshold_bps: float | dict[int, float],
    round_trip_cost_bps: float,
    *,
    record_decisions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return scored.copy(), pd.DataFrame(columns=["timestamp", "action", "reason"])
    ranked = scored.copy()
    if isinstance(threshold_bps, dict):
        ranked["required_ev_bps"] = ranked["side"].map(threshold_bps).fillna(float("inf"))
    else:
        ranked["required_ev_bps"] = float(threshold_bps)
    ranked["passes_ev_threshold"] = ranked["calibrated_ev_bps"].gt(
        ranked["required_ev_bps"]
    )
    candidates = (
        ranked.sort_values(
            [
                "actual_entry_timestamp",
                "passes_ev_threshold",
                "calibrated_ev_bps",
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
    if {
        "fold_expert_scope",
        "expert_tree_index",
        "expert_leaf_id",
    }.issubset(candidates.columns):
        candidates["expert_id"] = [
            (
                fold_expert_id(
                    str(scope), int(side), int(horizon), int(tree), int(leaf)
                )
                if int(tree) >= 0 and int(leaf) >= 0
                else expert_id(int(side), int(horizon))
            )
            for scope, side, horizon, tree, leaf in zip(
                candidates["fold_expert_scope"],
                candidates["side"],
                candidates["horizon_seconds"],
                candidates["expert_tree_index"],
                candidates["expert_leaf_id"],
                strict=True,
            )
        ]
    selected_rows: list[int] = []
    leverages: list[float] = []
    portfolio_returns: list[float] = []
    equity_before: list[float] = []
    daily_pnl_before: list[float] = []
    risk_remaining_before: list[float] = []
    entry_actions: list[str] = []
    decisions: list[dict[str, Any]] = []
    free_at = pd.Timestamp.min.tz_localize("UTC")
    equity = 1.0
    current_day: pd.Timestamp | None = None
    day_start_equity = 1.0
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
        state = SequentialState(
            daily_pnl_fraction=equity / day_start_equity - 1,
            risk_remaining_fraction=max(0.0, MAXIMUM_DAILY_LOSS + equity / day_start_equity - 1),
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
        if not bool(row.passes_ev_threshold):
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "WAIT",
                        "reason": "EV_BELOW_THRESHOLD",
                        **asdict(state),
                    }
                )
            continue
        leverage = float(_leverage(np.asarray([float(row.stop_bps)]), round_trip_cost_bps)[0])
        worst_risk = leverage * (float(row.stop_bps) + round_trip_cost_bps) / 10_000
        if state.risk_remaining_fraction + 1e-12 < worst_risk:
            if record_decisions:
                decisions.append(
                    {
                        "timestamp": entry,
                        "action": "WAIT",
                        "reason": "DAILY_RISK_VETO",
                        **asdict(state),
                    }
                )
            continue
        portfolio_return = leverage * float(row.net_bps) / 10_000
        if portfolio_return < -worst_risk - 1e-9:
            risk_violations += 1
        entry_action = "ENTER_LONG" if int(row.side) > 0 else "ENTER_SHORT"
        selected_rows.append(row_number)
        leverages.append(leverage)
        portfolio_returns.append(portfolio_return)
        equity_before.append(equity)
        daily_pnl_before.append(state.daily_pnl_fraction)
        risk_remaining_before.append(state.risk_remaining_fraction)
        entry_actions.append(entry_action)
        if record_decisions:
            decisions.append(
                {
                    "timestamp": entry,
                    "action": entry_action,
                    "reason": "CALIBRATED_EV_AND_RISK_APPROVED",
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
                    "action": "TIGHTEN_STOP",
                    "reason": "FIRST_TARGET_FILLED_NON_WIDENING_STOP",
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
    result = candidates.iloc[selected_rows].copy().reset_index(drop=True)
    if selected_rows:
        result["leverage"] = leverages
        result["portfolio_return"] = portfolio_returns
        result["equity_before"] = equity_before
        result["daily_pnl_before"] = daily_pnl_before
        result["risk_remaining_before"] = risk_remaining_before
        result["entry_action"] = entry_actions
        result["close_action"] = "CLOSE"
    result.attrs["risk_violations"] = risk_violations
    decision_rows = (
        pd.DataFrame(decisions).sort_values("timestamp").reset_index(drop=True)
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
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    active = daily.loc[daily.ne(0)]
    weekly = (1 + daily).resample("7D").prod() - 1
    return {
        "trades": len(trades),
        "trades_per_day": float(len(trades) / max(1, len(daily))),
        "expectancy_bps": float(net.mean()),
        "profit_factor": gains / losses if losses else None,
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
        "risk_violations": int(trades.attrs.get("risk_violations", 0)),
        "long_trades": int(trades["side"].gt(0).sum()),
        "short_trades": int(trades["side"].lt(0).sum()),
    }


def policy_gates(
    metrics: dict[str, Any], *, minimum_trades: int = MINIMUM_OOS_TRADES
) -> dict[str, bool]:
    return {
        "minimum_oos_trades": int(metrics.get("trades", 0)) >= minimum_trades,
        "expectancy_positive": float(metrics.get("expectancy_bps") or 0) > 0,
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
    return {
        "expectancy_positive": float(metrics.get("expectancy_bps") or 0) > 0,
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
        net_log_equity = (
            float(np.log1p(returns).sum())
            if len(returns) and np.all(returns > -1)
            else float("-inf")
        )
        risk_approved = (
            int(metrics.get("risk_violations", 1)) == 0
            and float(metrics.get("maximum_drawdown") or 1) <= MAXIMUM_DRAWDOWN
        )
        eligible = risk_approved and net_log_equity > 0
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
    matrix: pd.DataFrame, fee: FeeContract, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    folds = _folds(matrix)
    if not folds:
        raise ValueError("insufficient chronology for nested walk-forward")
    trade_pieces: list[pd.DataFrame] = []
    decision_pieces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
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
        library = fit_fold_expert_library(
            fit,
            number,
            resume=resume,
            progress=(number - 1, len(folds)),
        )
        fit = apply_fold_expert_library(fit, library)
        inner_calibration = apply_fold_expert_library(inner_calibration, library)
        model_audit = apply_fold_expert_library(model_audit, library)
        calibration = apply_fold_expert_library(calibration, library)
        selection = apply_fold_expert_library(selection, library)
        test = apply_fold_expert_library(test, library)
        kinds = ["ridge"] + (["xgboost_cuda"] if _xgb_available() else [])
        model_metrics: dict[str, dict[str, dict[str, float]]] = {}
        champions: dict[int, str] = {}
        scored_selection_pieces: list[pd.DataFrame] = []
        scored_test_pieces: list[pd.DataFrame] = []
        model_counter = 0
        total_models = 2 * len(kinds)
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            side_fit = fit.loc[fit["side"].eq(side)]
            side_inner = inner_calibration.loc[inner_calibration["side"].eq(side)]
            side_audit = model_audit.loc[model_audit["side"].eq(side)]
            side_calibration = calibration.loc[calibration["side"].eq(side)]
            side_selection = selection.loc[selection["side"].eq(side)]
            side_test = test.loc[test["side"].eq(side)]
            side_metrics: dict[str, dict[str, float]] = {}
            for kind in kinds:
                model_counter += 1
                internal = "xgboost" if kind == "xgboost_cuda" else kind
                _status(
                    "model_fit",
                    f"fold {number}/{len(folds)} {side_name} "
                    f"{model_counter}/{total_models} {kind}",
                    62
                    + 16
                    * ((number - 1) + model_counter / total_models)
                    / len(folds),
                    fold=f"{number}/{len(folds)}",
                    side=side_name,
                    model=kind,
                    fit_rows=len(side_fit),
                    gpu=_gpu_info(),
                )
                candidate_head = fit_probability_head(internal, side_fit)
                candidate_calibration = fit_calibration(candidate_head, side_inner)
                scored_inner = score_actions(side_audit, candidate_head, candidate_calibration)
                side_metrics[kind] = head_metrics(scored_inner)
            champion = choose_champion(side_metrics)
            champions[side] = champion
            model_metrics[side_name] = side_metrics
            internal = "xgboost" if champion == "xgboost_cuda" else champion
            side_refit = pd.concat([side_fit, side_inner, side_audit], ignore_index=True)
            head = fit_probability_head(internal, side_refit)
            calibrated = fit_calibration(head, side_calibration)
            scored_selection_pieces.append(score_actions(side_selection, head, calibrated))
            scored_test_pieces.append(score_actions(side_test, head, calibrated))
        scored_selection = pd.concat(scored_selection_pieces, ignore_index=True)
        scored_test = pd.concat(scored_test_pieces, ignore_index=True)
        thresholds: dict[int, float] = {}
        frontiers: dict[str, list[dict[str, Any]]] = {}
        for side, side_name in ((1, "LONG"), (-1, "SHORT")):
            threshold, side_frontier = _choose_frequency_threshold(
                scored_selection.loc[scored_selection["side"].eq(side)],
                fee,
                fold["selection_start"],
                fold["test_start"],
            )
            thresholds[side] = threshold
            frontiers[side_name] = side_frontier
        _status(
            "policy_replay",
            f"fold {number}/{len(folds)} LONG="
            f"{thresholds[1] if math.isfinite(thresholds[1]) else 'OFF'} bps; SHORT="
            f"{thresholds[-1] if math.isfinite(thresholds[-1]) else 'OFF'} bps",
            62 + 16 * number / len(folds),
            fold=f"{number}/{len(folds)}",
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
        trades, decisions = sequential_replay(scored_test, thresholds, fee.round_trip_bps)
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
                "selection_rows": len(selection),
                "test_rows": len(test),
                "candidate_metrics": model_metrics,
                "champions": {
                    "LONG": champions[1],
                    "SHORT": champions[-1],
                },
                "selected_thresholds_bps": {
                    "LONG": thresholds[1] if math.isfinite(thresholds[1]) else None,
                    "SHORT": thresholds[-1] if math.isfinite(thresholds[-1]) else None,
                },
                "frequency_pnl_frontiers": frontiers,
                "test_frequency_pnl_frontier": test_frontier,
                "test_daily_returns_by_threshold": test_daily,
                "test_metrics": metrics,
                "fold_experts": {
                    "fold_scope": library["fold_scope"],
                    "catalog_path": library["catalog_path"],
                    "candidates_evaluated": library["candidates_evaluated"],
                    "candidates_eligible": library["candidates_eligible"],
                    "terminal_prefilter_rejections": 0,
                },
            }
        )
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
    library = fit_fold_expert_library(fit, "forward", resume=resume)
    fit = apply_fold_expert_library(fit, library)
    calibration = apply_fold_expert_library(calibration, library)
    selection = apply_fold_expert_library(selection, library)
    champions: dict[str, str] = {}
    heads: dict[int, dict[str, Any]] = {}
    calibrations: dict[int, dict[str, Any]] = {}
    thresholds: dict[int, float] = {}
    frontiers: dict[str, list[dict[str, Any]]] = {}
    for side, side_name in ((1, "LONG"), (-1, "SHORT")):
        observed = [str(item["champions"][side_name]) for item in folds]
        champion = (
            "xgboost_cuda"
            if observed.count("xgboost_cuda") > observed.count("ridge")
            else "ridge"
        )
        champions[side_name] = champion
        internal = "xgboost" if champion == "xgboost_cuda" else champion
        head = fit_probability_head(internal, fit.loc[fit["side"].eq(side)])
        calibrated = fit_calibration(
            head, calibration.loc[calibration["side"].eq(side)]
        )
        scored_selection = score_actions(
            selection.loc[selection["side"].eq(side)], head, calibrated
        )
        threshold, frontier = _choose_frequency_threshold(
            scored_selection, fee, selection_start, end
        )
        heads[side] = head
        calibrations[side] = calibrated
        thresholds[side] = threshold
        frontiers[side_name] = frontier
    return {
        "champions": champions,
        "heads": heads,
        "calibrations": calibrations,
        "expert_library": library,
        "thresholds_bps": {
            "LONG": thresholds[1] if math.isfinite(thresholds[1]) else None,
            "SHORT": thresholds[-1] if math.isfinite(thresholds[-1]) else None,
        },
        "fit_end": calibration_start.isoformat(),
        "calibration_period": [calibration_start.isoformat(), selection_start.isoformat()],
        "policy_selection_period": [selection_start.isoformat(), end.isoformat()],
        "frequency_pnl_frontiers": frontiers,
        "fold_experts": {
            "fold_scope": library["fold_scope"],
            "catalog_path": library["catalog_path"],
            "candidates_evaluated": library["candidates_evaluated"],
            "candidates_eligible": library["candidates_eligible"],
            "terminal_prefilter_rejections": 0,
        },
    }


def _economic_action_set(matrix: pd.DataFrame) -> dict[str, Any]:
    grouped = matrix.groupby(["side", "horizon_seconds"])["net_bps"].agg(
        ["count", "mean", "median"]
    )
    oracle = matrix.groupby("actual_entry_timestamp")["net_bps"].max()
    return {
        "actions": grouped.reset_index().to_dict("records"),
        "oracle_positive_fraction": float(oracle.gt(0).mean()),
        "oracle_mean_net_bps": float(oracle.mean()),
        "oracle_is_not_tradable": True,
        "has_positive_unconditional_action": bool(grouped["mean"].gt(0).any()),
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
    if not economics["has_positive_unconditional_action"] and economics["oracle_mean_net_bps"] <= 0:
        return "NO_ECONOMIC_ACTION_SET"
    if not folds or sum(int(item["test_metrics"]["trades"]) for item in folds) == 0:
        return "NO_PREDICTABLE_EDGE"
    audited: list[float] = []
    for item in folds:
        candidate_metrics = item.get("candidate_metrics", {})
        for candidate in candidate_metrics.values():
            if candidate and all(isinstance(value, (int, float)) for value in candidate.values()):
                audited.extend(float(value) for value in candidate.values())
            else:
                for model in candidate.values():
                    audited.extend(float(value) for value in model.values())
    if not audited or not all(math.isfinite(float(value)) for value in audited):
        return "NO_CALIBRATED_POLICY"
    side_enabled = any(all(result["gates"].values()) for result in sides.values())
    if not all(policy_gates(metrics).values()) or not side_enabled:
        return "NO_STABLE_OOS_POLICY"
    return "RESEARCH_PAPER_READY"


def train(*, resume: bool = False) -> dict[str, Any]:
    global _RUN_STARTED
    _RUN_STARTED = time.monotonic()
    started = time.monotonic()
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
    if not economics["has_positive_unconditional_action"] and economics["oracle_mean_net_bps"] <= 0:
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
    forward_bundle: dict[str, Any] | None = None
    if folds:
        _status("forward_bundle", "fitting frozen research-paper controller", 82, gpu=_gpu_info())
        forward_bundle = fit_forward_bundle(matrix, fee, folds, resume=resume)
        if verdict == "RESEARCH_PAPER_READY" and not any(
            value is not None for value in forward_bundle["thresholds_bps"].values()
        ):
            verdict = "NO_CALIBRATED_POLICY"
    daily = _daily_returns(trades, audit_start, audit_end)
    generated_expert_candidates = sum(
        int(item.get("fold_experts", {}).get("candidates_evaluated", 0)) for item in folds
    )
    if forward_bundle is not None:
        generated_expert_candidates += int(
            forward_bundle["fold_experts"]["candidates_evaluated"]
        )
    generated_experts_eligible = sum(
        int(item.get("fold_experts", {}).get("candidates_eligible", 0)) for item in folds
    )
    if forward_bundle is not None:
        generated_experts_eligible += int(
            forward_bundle["fold_experts"]["candidates_eligible"]
        )
    registry["registered_fold_local_expert_candidates"] = generated_expert_candidates
    registry["registered_fold_local_experts_eligible_after_management"] = (
        generated_experts_eligible
    )
    registry["total_registered_expert_attempts"] = (
        int(registry["registered_auto_moe_experts"]) + generated_expert_candidates
    )
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
        },
        "registry": registry,
        "frozen_auto_moe_report_sha256": frozen_hash_after,
        "economic_action_set": economics,
        "walk_forward": folds,
        "oos_metrics": metrics,
        "side_controls": sides,
        "gates": policy_gates(metrics),
        "multiple_comparison": multiple_comparison,
        "forward_bundle": (
            None
            if forward_bundle is None
            else {
                name: value
                for name, value in forward_bundle.items()
                if name not in {"heads", "calibrations", "expert_library"}
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
            "expert_library": (
                None if forward_bundle is None else forward_bundle["expert_library"]
            ),
            "thresholds_bps": (
                None if forward_bundle is None else forward_bundle["thresholds_bps"]
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
