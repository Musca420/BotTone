from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise, product
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_bot.adapters.bitunix.market_data import JsonGetter, _get_json
from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig, MachineLearningConfig
from adaptive_bot.domain.enums import MarketRegime, Side, SignalAction
from adaptive_bot.domain.models import Position, Signal, StrategyState
from adaptive_bot.ml_research import (
    FEATURE_COLUMNS,
    _download_funding,
    _download_klines,
    add_triple_barrier_labels,
    build_ml_features,
    write_ml_status,
)
from adaptive_bot.research import combinatorial_pbo, deflated_sharpe_probability
from adaptive_bot.strategy.signals import MarketSnapshot, initial_stop

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional GPU dependency
    XGBClassifier = None  # type: ignore[misc,assignment]


CORE_FEATURES = tuple(name for name in FEATURE_COLUMNS if name != "funding_rate")
PROTOCOL_VERSION = "scientific_v2"
BASE_EXTRA_COST_BPS = 7.0  # 3 bps full spread + 2 bps slippage per side.
MAX_ENVELOPE_CORRECTION_BPS = 5.0
INVALID_SCORE = -1e9
CALIBRATION_CV_SPLITS = 5
_SCREEN_APP: AppConfig | None = None
_SCREEN_FRAMES: dict[int, pd.DataFrame] = {}
_SCREEN_CONFIG: MachineLearningConfig | None = None
_SCREEN_FEATURE_CACHE: dict[tuple[int, int, int, int], pd.DataFrame] = {}


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    timeframe_minutes: int
    vwap_hours: int
    atr_period: int
    adx_period: int
    entry_z_long: float
    entry_z_short: float
    range_adx_threshold: float
    regime_policy: Literal["range_only", "block_with_trend", "any_nonshock"]
    entry_rule: Literal["touch", "exhaustion", "confirmed_reentry"]
    confirmation_bars: int
    stop_atr: float
    exit_z: float
    time_stop_hours: int
    cooldown_bars: int

    @property
    def vwap_bars(self) -> int:
        return self.vwap_hours * 60 // self.timeframe_minutes

    @property
    def holding_bars(self) -> int:
        return self.time_stop_hours * 60 // self.timeframe_minutes


@dataclass(frozen=True)
class PurgedFold:
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class NestedFold:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class OofEvaluation:
    events: list[tuple[int, int, float, bool]]
    stress_events: list[tuple[int, int, float, bool]]
    probabilities: list[float]
    targets: list[int]

    def returns(self, cooldown_bars: int) -> list[float]:
        return _non_overlapping_returns(self.events, cooldown_bars)


class CandidateReplayStrategy:
    """Exact candidate entry/exit policy for the shared event-driven execution engine."""

    def __init__(
        self,
        candidate: StrategyCandidate,
        allowed_entries: set[tuple[datetime, str]] | None = None,
    ) -> None:
        self.candidate = candidate
        self.allowed_entries = allowed_entries
        self._z: list[float] = []

    def evaluate(
        self, snapshot: MarketSnapshot, state: StrategyState, position: Position | None
    ) -> Signal | None:
        try:
            if position is not None:
                reason = self._exit_reason(snapshot, position)
                return self._signal(snapshot, SignalAction.EXIT, reason) if reason else None
            if (
                not snapshot.data_reliable
                or snapshot.regime in {MarketRegime.UNKNOWN, MarketRegime.SHOCK}
                or snapshot.spread_bps > 3
            ):
                return None
            side = "long" if snapshot.z_score < 0 else "short"
            if (
                self.allowed_entries is not None
                and (
                    snapshot.candle.exchange_timestamp,
                    side,
                )
                not in self.allowed_entries
            ):
                return None
            if not self._entry_allowed(snapshot, side):
                return None
            order_side = Side.BUY if side == "long" else Side.SELL
            action = SignalAction.ENTER_LONG if side == "long" else SignalAction.ENTER_SHORT
            stop = initial_stop(
                snapshot.candle.close,
                snapshot.atr,
                order_side,
                Decimal(str(self.candidate.stop_atr)),
            )
            return self._signal(snapshot, action, "scientific candidate", stop, snapshot.center)
        finally:
            self._z.append(snapshot.z_score)
            self._z = self._z[-4:]

    def entry_score(self, snapshot: MarketSnapshot, state: StrategyState) -> None:
        del snapshot, state
        return None

    def _entry_allowed(self, snapshot: MarketSnapshot, side: str) -> bool:
        threshold = self.candidate.entry_z_long if side == "long" else self.candidate.entry_z_short
        if (side == "long" and snapshot.z_score > -threshold) or (
            side == "short" and snapshot.z_score < threshold
        ):
            return False
        if (
            self.candidate.regime_policy == "range_only"
            and snapshot.regime is not MarketRegime.RANGE
        ):
            return False
        if self.candidate.regime_policy == "block_with_trend" and (
            (side == "long" and snapshot.regime is MarketRegime.TREND_DOWN)
            or (side == "short" and snapshot.regime is MarketRegime.TREND_UP)
        ):
            return False
        bars = max(1, self.candidate.confirmation_bars)
        if self.candidate.entry_rule == "touch":
            return True
        sequence = [abs(value) for value in (*self._z[-bars:], snapshot.z_score)]
        return len(sequence) == bars + 1 and all(left > right for left, right in pairwise(sequence))

    def _exit_reason(self, snapshot: MarketSnapshot, position: Position) -> str | None:
        if not snapshot.data_reliable:
            return "data integrity compromised"
        if snapshot.regime is MarketRegime.SHOCK:
            return "shock regime"
        if position.bars_held >= self.candidate.holding_bars:
            return "time stop"
        if position.side is Side.BUY and snapshot.z_score >= -self.candidate.exit_z:
            return "VWAP approach"
        if position.side is Side.SELL and snapshot.z_score <= self.candidate.exit_z:
            return "VWAP approach"
        return None

    @staticmethod
    def _signal(
        snapshot: MarketSnapshot,
        action: SignalAction,
        reason: str,
        stop: Decimal | None = None,
        target: Decimal | None = None,
    ) -> Signal:
        candle = snapshot.candle
        return Signal(
            exchange_timestamp=candle.exchange_timestamp,
            received_timestamp=candle.received_timestamp,
            source="scientific_replay",
            instrument=candle.instrument,
            correlation_id=candle.correlation_id,
            action=action,
            reference_price=candle.close,
            stop_price=stop,
            target_price=target,
            z_score=snapshot.z_score,
            regime=snapshot.regime,
            reason=reason,
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_preflight(*, required: bool = True) -> dict[str, Any]:
    if XGBClassifier is None:
        if required:
            raise RuntimeError("scientific ML requires the gpu extra with xgboost installed")
        return {"backend": "unavailable", "device": None}
    try:
        model = XGBClassifier(
            n_estimators=2,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=7,
            verbosity=0,
        )
        model.fit(np.asarray([[0.0], [1.0], [2.0], [3.0]]), np.asarray([0, 0, 1, 1]))
        model.predict_proba(np.asarray([[1.5]]))
    except Exception as error:
        if required:
            raise RuntimeError(f"CUDA XGBoost preflight failed: {error}") from error
        return {"backend": "cpu", "device": None, "warning": str(error)}
    return {"backend": "cuda", "device": 0}


def scientific_archive_path(config: MachineLearningConfig, symbol: str) -> Path:
    return config.archive_directory / f"bitunix_{symbol.lower()}_observed_1m.parquet"


def _monthly_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = pd.Timestamp(start)
    boundary = pd.Timestamp(end)
    while cursor < boundary:
        next_cursor = min(cursor + pd.DateOffset(months=1), boundary)
        chunks.append((cursor.to_pydatetime(), next_cursor.to_pydatetime()))
        cursor = next_cursor
    return chunks


def _parallel_price_history(
    config: MachineLearningConfig,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    get_json: JsonGetter,
    phase_start: float,
    phase_width: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = _monthly_chunks(start, end)
    jobs = [
        (price_type, chunk_start, chunk_end)
        for price_type in ("LAST_PRICE", "MARK_PRICE")
        for chunk_start, chunk_end in chunks
    ]
    chunk_root = config.archive_directory / "chunks" / symbol.lower()
    chunk_root.mkdir(parents=True, exist_ok=True)
    state_lock = Lock()
    rate_lock = Lock()
    next_request = [time.monotonic()]
    progress: dict[str, float] = {}
    row_counts: dict[str, int] = {}

    def limited_get(url: str) -> dict[str, Any]:
        for attempt in range(6):
            with rate_lock:
                now = time.monotonic()
                wait = max(0.0, next_request[0] - now)
                next_request[0] = max(now, next_request[0]) + 0.125
            if wait:
                time.sleep(wait)
            try:
                return get_json(url)
            except HTTPError as error:
                if error.code != 429 and error.code < 500:
                    raise
                retry_after = float(error.headers.get("Retry-After", 0) or 0)
                time.sleep(max(retry_after, min(30.0, 0.5 * 2**attempt)))
            except (URLError, TimeoutError):
                time.sleep(min(30.0, 0.5 * 2**attempt))
        raise RuntimeError("Bitunix REST retries exhausted")

    def download_job(
        price_type: Literal["LAST_PRICE", "MARK_PRICE"],
        chunk_start: datetime,
        chunk_end: datetime,
    ) -> tuple[str, datetime, Path, int]:
        key = f"{price_type}:{chunk_start.isoformat()}"
        target = chunk_root / (
            f"{price_type.lower()}_{chunk_start:%Y%m%dT%H%M}_{chunk_end:%Y%m%dT%H%M}.parquet"
        )
        if target.exists():
            saved = pd.read_parquet(target)
            with state_lock:
                progress[key] = 1.0
                row_counts[key] = len(saved)
            return price_type, chunk_start, target, len(saved)

        start_ms = int(chunk_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        def update(cursor_ms: int, rows: int) -> None:
            fraction = min(1.0, max(0.0, (end_ms - cursor_ms) / (end_ms - start_ms)))
            with state_lock:
                progress[key] = fraction
                row_counts[key] = rows
                overall = sum(progress.values()) / len(jobs)
                write_ml_status(
                    config,
                    "download",
                    f"{symbol}: LAST + MARK 1m paralleli",
                    phase_start + phase_width * overall,
                    symbol=symbol,
                    completed_chunks=sum(value >= 1 for value in progress.values()),
                    total_chunks=len(jobs),
                    downloaded_rows=sum(row_counts.values()),
                    active_workers=config.download_workers,
                    backend=f"REST {config.download_workers} workers, <=8 req/s",
                )

        data = _download_klines(
            limited_get,
            chunk_start,
            chunk_end,
            price_type,
            symbol=symbol,
            interval="1m",
            progress=update,
        )
        temporary = target.with_suffix(".tmp.parquet")
        data.to_parquet(temporary, index=False)
        temporary.replace(target)
        return price_type, chunk_start, target, len(data)

    results: dict[str, list[tuple[datetime, Path]]] = {
        "LAST_PRICE": [],
        "MARK_PRICE": [],
    }
    with ThreadPoolExecutor(max_workers=config.download_workers) as pool:
        futures = {
            pool.submit(download_job, cast(Any, price_type), chunk_start, chunk_end): price_type
            for price_type, chunk_start, chunk_end in jobs
        }
        for future in as_completed(futures):
            price_type, chunk_start, path, _ = future.result()
            results[price_type].append((chunk_start, path))
    frames: dict[str, pd.DataFrame] = {}
    for price_type, parts in results.items():
        frames[price_type] = (
            pd.concat(
                [pd.read_parquet(path) for _, path in sorted(parts)],
                ignore_index=True,
            )
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    return frames["LAST_PRICE"], frames["MARK_PRICE"]


def download_scientific_archive(
    app: AppConfig,
    start: datetime,
    end: datetime,
    *,
    get_json: JsonGetter = _get_json,
) -> dict[str, Any]:
    config = _scientific_config(app)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("archive boundaries must be timezone-aware and ordered")
    files: list[dict[str, Any]] = []
    total = len(config.symbols)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    def progress_callback(
        symbol: str, detail: str, phase_start: float, phase_width: float
    ) -> Callable[[int, int], None]:
        def update(cursor_ms: int, rows: int) -> None:
            fraction = min(1.0, max(0.0, (end_ms - cursor_ms) / (end_ms - start_ms)))
            write_ml_status(
                config,
                "download",
                f"{symbol}: {detail}",
                phase_start + phase_width * fraction,
                symbol=symbol,
                downloaded_rows=rows,
                cursor=pd.to_datetime(cursor_ms, unit="ms", utc=True).isoformat(),
                backend="REST rate-limited",
            )

        return update

    for position, symbol in enumerate(config.symbols):
        base = position / total * 100
        span = 100 / total
        write_ml_status(
            config,
            "download",
            f"{symbol}: preparing parallel LAST + MARK 1m",
            base,
            symbol=symbol,
            completed=position,
            total=total,
            backend=f"REST {config.download_workers} workers, <=8 req/s",
        )
        last, mark = _parallel_price_history(
            config,
            symbol,
            start,
            end,
            get_json=get_json,
            phase_start=base,
            phase_width=span * 0.8,
        )
        mark = mark.rename(
            columns={name: f"mark_{name}" for name in ("open", "high", "low", "close")}
        )
        funding = _download_funding(
            get_json,
            start - timedelta(days=1),
            end,
            symbol=symbol,
            progress=progress_callback(symbol, "official funding", base + span * 0.8, span * 0.2),
        )
        data = last.merge(
            mark.loc[:, ["timestamp", "mark_open", "mark_high", "mark_low", "mark_close"]],
            on="timestamp",
            how="inner",
            validate="one_to_one",
        ).sort_values("timestamp")
        if funding.empty:
            data["funding_rate"] = np.nan
            data["funding_event_rate"] = np.nan
        else:
            events = funding.rename(columns={"funding_rate": "funding_event_rate"})
            data = data.merge(events, on="timestamp", how="left", validate="one_to_one")
            data = pd.merge_asof(
                data.sort_values("timestamp"),
                funding.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        data["raw_high"] = data["high"]
        data["raw_low"] = data["low"]
        envelope_error = (
            pd.concat(
                [
                    data.loc[:, ["open", "close"]].max(axis=1) - data["high"],
                    data["low"] - data.loc[:, ["open", "close"]].min(axis=1),
                ],
                axis=1,
            )
            .max(axis=1)
            .clip(lower=0)
        )
        data["envelope_deviation_bps"] = envelope_error / data["close"] * 10_000
        data["data_valid"] = data["envelope_deviation_bps"].le(MAX_ENVELOPE_CORRECTION_BPS)
        data["ohlc_adjusted"] = envelope_error.gt(0) & data["data_valid"]
        data["high"] = data.loc[:, ["open", "high", "close"]].max(axis=1)
        data["low"] = data.loc[:, ["open", "low", "close"]].min(axis=1)
        data["round_trip_cost_bps"] = config.taker_fee_bps * 2
        data["market_data_source"] = "observed:bitunix-official-rest"
        data["price_source"] = "observed"
        data["volume_source"] = "observed"
        data["funding_source"] = np.where(data["funding_rate"].notna(), "observed", "unavailable")
        data["spread_source"] = "unavailable"
        data["cost_source"] = "official_vip0_taker_fee_no_spread"
        target = scientific_archive_path(config, symbol)
        target.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target, index=False)
        funding_index = data["funding_rate"].first_valid_index()
        files.append(
            {
                "symbol": symbol,
                "path": str(target),
                "sha256": _sha256(target),
                "rows": len(data),
                "start": data["timestamp"].min().isoformat(),
                "end": data["timestamp"].max().isoformat(),
                "invalid_envelope_rows": int((~data["data_valid"]).sum()),
                "funding_start": (
                    None
                    if funding_index is None
                    else cast(pd.Timestamp, data.at[funding_index, "timestamp"]).isoformat()
                ),
            }
        )
    manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "Bitunix official REST",
        "interval": "1m",
        "requested_start": start.astimezone(UTC).isoformat(),
        "requested_end": end.astimezone(UTC).isoformat(),
        "files": files,
    }
    _atomic_json(config.manifest_path, manifest)
    write_ml_status(config, "download_complete", "Immutable scientific archive ready", 100)
    return manifest


def resample_observed(frame: pd.DataFrame, minutes: int, tick_size: float) -> pd.DataFrame:
    if minutes not in {5, 15, 30, 60}:
        raise ValueError("scientific protocol supports 5m, 15m, 30m and 60m")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.drop_duplicates("timestamp", keep=False).sort_values("timestamp")
    expected = pd.Timedelta(minutes=1)
    contiguous = data["timestamp"].diff().eq(expected) | data["timestamp"].diff().isna()
    raw_valid = data.get("data_valid", pd.Series(True, index=data.index)).astype(bool)
    if {"raw_high", "raw_low"}.issubset(data.columns):
        envelope_error = (
            pd.concat(
                [
                    data.loc[:, ["open", "close"]].max(axis=1) - data["raw_high"],
                    data["raw_low"] - data.loc[:, ["open", "close"]].min(axis=1),
                ],
                axis=1,
            )
            .max(axis=1)
            .clip(lower=0)
        )
        correction_bps = envelope_error / data["close"] * 10_000
        # Recompute from immutable raw OHLC: old archives may contain a validity flag
        # produced with a superseded envelope threshold.
        raw_valid = correction_bps.le(MAX_ENVELOPE_CORRECTION_BPS)
    data["minute_valid"] = contiguous & raw_valid
    data = data.set_index("timestamp")
    aggregation: dict[Any, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "mark_open": "first",
        "mark_high": "max",
        "mark_low": "min",
        "mark_close": "last",
        "funding_rate": "last",
        "funding_event_rate": "sum",
        "minute_valid": "all",
    }
    grouped = data.resample(f"{minutes}min", origin="epoch", closed="left", label="left")
    result = grouped.agg(aggregation)
    result["minute_count"] = grouped["close"].count()
    result["data_valid"] = result["minute_valid"] & result["minute_count"].eq(minutes)
    result = result[result["minute_count"].gt(0)].reset_index()
    result["round_trip_cost_bps"] = 12.0
    result["market_data_source"] = "observed:bitunix-official-rest"
    result["price_source"] = "observed"
    result["volume_source"] = "observed"
    result["funding_source"] = np.where(result["funding_rate"].notna(), "observed", "unavailable")
    result["spread_source"] = "unavailable"
    result["cost_source"] = "official_vip0_taker_fee_no_spread"
    result["high"] = result.loc[:, ["open", "high", "close"]].max(axis=1)
    result["low"] = result.loc[:, ["open", "low", "close"]].min(axis=1)
    result["tick_size"] = tick_size
    return result


def generate_candidates(config: MachineLearningConfig) -> list[StrategyCandidate]:
    randomizer = random.Random(config.random_seed)
    choices: dict[str, tuple[Any, ...]] = {
        "timeframe_minutes": config.timeframes,
        "vwap_hours": (8, 12, 24, 48),
        "atr_period": (7, 14, 21, 28),
        "adx_period": (7, 14, 21, 28),
        "entry_z_long": tuple(np.arange(1.0, 3.01, 0.25)),
        "entry_z_short": tuple(np.arange(1.0, 3.01, 0.25)),
        "range_adx_threshold": (15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0),
        "regime_policy": ("range_only", "block_with_trend", "any_nonshock"),
        "entry_rule": ("touch", "exhaustion", "confirmed_reentry"),
        "confirmation_bars": (0, 1, 2, 3),
        "stop_atr": (1.5, 2.0, 2.5, 3.0),
        "exit_z": (0.0, 0.25, 0.5),
        "time_stop_hours": (1, 2, 4, 8),
        "cooldown_bars": (0, 3, 6, 9, 12),
    }
    rows: dict[str, StrategyCandidate] = {}
    for threshold in (20.0, 22.0, 23.0, 24.0, 25.0):
        candidate = StrategyCandidate(
            candidate_id=f"baseline-adx{int(threshold)}",
            timeframe_minutes=5,
            vwap_hours=24,
            atr_period=14,
            adx_period=14,
            entry_z_long=1.5,
            entry_z_short=1.5,
            range_adx_threshold=threshold,
            regime_policy="range_only",
            entry_rule="touch",
            confirmation_bars=0,
            stop_atr=2.5,
            exit_z=0.0,
            time_stop_hours=1,
            cooldown_bars=0,
        )
        if len(rows) < config.strategy_candidates:
            rows[candidate.candidate_id] = candidate
    while len(rows) < config.strategy_candidates:
        values = {name: randomizer.choice(options) for name, options in choices.items()}
        fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]
        rows[fingerprint] = StrategyCandidate(candidate_id=fingerprint, **values)
    return list(rows.values())


def purged_time_splits(
    data: pd.DataFrame, *, n_splits: int, embargo_bars: int, exit_column: str
) -> tuple[PurgedFold, ...]:
    if n_splits < 2:
        raise ValueError("at least two purged folds are required")
    indexes = np.arange(len(data))
    block = len(data) // (n_splits + 1)
    if block <= embargo_bars:
        raise ValueError("not enough observations for purged validation")
    exits = data[exit_column].to_numpy(dtype=int)
    event_indexes = (
        data["event_index"].to_numpy(dtype=int) if "event_index" in data else np.arange(len(data))
    )
    folds: list[PurgedFold] = []
    for fold in range(n_splits):
        validation_start = block * (fold + 1)
        validation_end = len(data) if fold == n_splits - 1 else validation_start + block
        train_end = max(0, validation_start - embargo_bars)
        train = indexes[:train_end]
        validation_event_start = event_indexes[validation_start]
        train = train[exits[train] < validation_event_start]
        validation = indexes[validation_start:validation_end]
        if len(train) and len(validation):
            folds.append(PurgedFold(train, validation))
    if len(folds) < 2:
        raise ValueError("purging left too few validation folds")
    return tuple(folds)


def rolling_purged_splits(
    data: pd.DataFrame,
    *,
    train_weeks: int,
    calibration_weeks: int,
    test_weeks: int,
    step_weeks: int,
    exit_column: str,
) -> tuple[PurgedFold, ...]:
    return tuple(
        PurgedFold(fold.train, fold.test)
        for fold in rolling_nested_splits(
            data,
            train_weeks=train_weeks,
            calibration_weeks=calibration_weeks,
            test_weeks=test_weeks,
            step_weeks=step_weeks,
            exit_column=exit_column,
        )
    )


def rolling_nested_splits(
    data: pd.DataFrame,
    *,
    train_weeks: int,
    calibration_weeks: int,
    test_weeks: int,
    step_weeks: int,
    exit_column: str,
) -> tuple[NestedFold, ...]:
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    starts = (
        data["event_index"].to_numpy(dtype=int) if "event_index" in data else np.arange(len(data))
    )
    exits = data[exit_column].to_numpy(dtype=int)
    cursor = timestamps.min() + timedelta(weeks=train_weeks)
    end = timestamps.max()
    folds: list[NestedFold] = []
    while True:
        calibration_end = cursor + timedelta(weeks=calibration_weeks)
        test_end = calibration_end + timedelta(weeks=test_weeks)
        if test_end > end:
            break
        train_start = cursor - timedelta(weeks=train_weeks)
        train = np.flatnonzero(timestamps.ge(train_start) & timestamps.lt(cursor))
        calibration = np.flatnonzero(timestamps.ge(cursor) & timestamps.lt(calibration_end))
        test = np.flatnonzero(timestamps.ge(calibration_end) & timestamps.lt(test_end))
        if len(train) and len(calibration) and len(test):
            first_calibration_event = starts[calibration[0]]
            first_test_event = starts[test[0]]
            train = train[exits[train] < first_calibration_event]
            calibration = calibration[exits[calibration] < first_test_event]
            if len(train) and len(calibration):
                folds.append(NestedFold(train, calibration, test))
        cursor += timedelta(weeks=step_weeks)
    if len(folds) < 2:
        raise ValueError("fewer than two chronological outer walk-forward folds are available")
    return tuple(folds)


def _candidate_frame(
    frame: pd.DataFrame, app: AppConfig, candidate: StrategyCandidate
) -> pd.DataFrame:
    strategy = app.strategy.model_copy(
        update={
            "timeframe_minutes": candidate.timeframe_minutes,
            "crypto_vwap_window": candidate.vwap_bars,
            "atr_period": candidate.atr_period,
            "adx_period": candidate.adx_period,
        }
    )
    configured = app.model_copy(update={"strategy": strategy})
    complete = frame[frame["funding_rate"].notna()].copy()
    complete["funding_source"] = "observed"
    features, _ = build_ml_features(complete, configured)
    return _label_candidate_frame(features, candidate)


def _label_candidate_frame(features: pd.DataFrame, candidate: StrategyCandidate) -> pd.DataFrame:
    labelled = add_triple_barrier_labels(
        features,
        max_holding_bars=candidate.holding_bars,
        stop_atr=candidate.stop_atr,
        target_z=candidate.exit_z,
    )
    labelled["entry_price"] = labelled["open"].shift(-1)
    funding = labelled["funding_event_rate"].fillna(0).to_numpy(dtype=float)
    cumulative_funding = np.concatenate(([0.0], np.cumsum(funding)))
    bar_valid = labelled["data_valid"].to_numpy(dtype=bool)
    cumulative_invalid = np.concatenate(([0], np.cumsum(~bar_valid)))
    for side, direction in (("long", -1.0), ("short", 1.0)):
        exits = labelled[f"exit_index_{side}"].to_numpy(dtype=int)
        valid = exits >= 0
        starts = np.arange(len(labelled)) + 1
        crosses_invalid = np.zeros(len(labelled), dtype=bool)
        crosses_invalid[valid] = (
            cumulative_invalid[exits[valid] + 1] - cumulative_invalid[starts[valid]]
        ) > 0
        payments = np.zeros(len(labelled), dtype=float)
        clean = valid & ~crosses_invalid
        payments[clean] = cumulative_funding[exits[clean] + 1] - cumulative_funding[starts[clean]]
        labelled.loc[clean, f"net_return_{side}"] += direction * payments[clean]
        labelled.loc[crosses_invalid, f"target_{side}"] = -1
        labelled.loc[crosses_invalid, f"net_return_{side}"] = np.nan
        labelled.loc[crosses_invalid, f"exit_index_{side}"] = -1
    labelled["data_valid"] = bar_valid
    usable = labelled[list(FEATURE_COLUMNS)].notna().all(axis="columns").to_numpy(dtype=bool)
    retained = np.flatnonzero(usable)
    remap = np.full(len(labelled), -1, dtype=np.int64)
    remap[retained] = np.arange(len(retained))
    for side in ("long", "short"):
        exits = labelled[f"exit_index_{side}"].to_numpy(dtype=int)
        mapped = np.full(len(labelled), -1, dtype=np.int64)
        valid = (exits >= 0) & (exits < len(labelled))
        mapped[valid] = remap[exits[valid]]
        invalid_exit = usable & (mapped < 0)
        labelled.loc[invalid_exit, f"target_{side}"] = -1
        labelled.loc[invalid_exit, f"net_return_{side}"] = np.nan
        labelled[f"exit_index_{side}"] = mapped
    return labelled.loc[usable].reset_index(drop=True)


def _entry_mask(data: pd.DataFrame, candidate: StrategyCandidate, side: str) -> pd.Series:
    z = data["distance_vwap_atr"]
    atr_change = data["atr"].pct_change(fill_method=None)
    cumulative_move = (data["close"] - data["close"].shift(3)) / data["atr"]
    threshold = candidate.entry_z_long if side == "long" else candidate.entry_z_short
    direction = z <= -threshold if side == "long" else z >= threshold
    shock = data["atr_percentile"].gt(90) | atr_change.gt(0.5) | cumulative_move.abs().gt(3)
    trend_up = data["adx"].gt(25) & data["ema50_slope"].ge(0.05) & cumulative_move.ge(0)
    trend_down = data["adx"].gt(25) & data["ema50_slope"].le(-0.05) & cumulative_move.le(0)
    range_regime = data["adx"].lt(candidate.range_adx_threshold)
    known = range_regime | trend_up | trend_down
    if candidate.regime_policy == "range_only":
        regime = range_regime
    elif candidate.regime_policy == "block_with_trend":
        regime = known & (~trend_down if side == "long" else ~trend_up)
    else:
        regime = known
    if candidate.entry_rule == "exhaustion":
        confirmation = z.abs().lt(z.abs().shift(1))
    elif candidate.entry_rule == "confirmed_reentry":
        confirmation = z.abs().lt(z.abs().shift(1))
        for lag in range(2, max(2, candidate.confirmation_bars + 1)):
            confirmation &= z.abs().shift(lag - 1).lt(z.abs().shift(lag))
    else:
        confirmation = pd.Series(True, index=data.index)
    valid = (
        data["data_valid"].rolling(candidate.vwap_bars, min_periods=candidate.vwap_bars).min().eq(1)
    )
    executable = data[f"net_return_{side}"].notna() & data[f"net_return_{side}"].ne(0)
    return (
        direction
        & regime
        & confirmation
        & ~shock
        & valid
        & data[f"target_{side}"].ge(0)
        & executable
    )


def _candidate_events(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    side: str,
    probabilities: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    expected_win: float | None = None,
    expected_loss: float | None = None,
    cost_multiplier: float = 1.0,
) -> list[tuple[int, int, float, bool]]:
    eligibility = (
        data[f"eligible_{side}"]
        if f"eligible_{side}" in data
        else _entry_mask(data, candidate, side)
    )
    eligible = data.loc[eligibility].copy()
    if probabilities is not None and len(probabilities) != len(eligible):
        raise ValueError("probability count does not match eligible events")
    returns = eligible[f"net_return_{side}"].to_numpy(dtype=float)
    extra_cost = (BASE_EXTRA_COST_BPS * cost_multiplier + 12.0 * (cost_multiplier - 1)) / 10_000
    returns = returns - extra_cost
    stop_fraction = (
        candidate.stop_atr
        * eligible["atr"].to_numpy(dtype=float)
        / eligible["entry_price"].to_numpy(dtype=float)
    )
    risk_returns = np.divide(
        returns,
        stop_fraction,
        out=np.zeros_like(returns),
        where=stop_fraction > 0,
    )
    accepted = np.ones(len(eligible), dtype=bool)
    if probabilities is not None:
        if expected_win is None or expected_loss is None:
            raise ValueError("training payoff priors are required for probability filtering")
        ev = probabilities * expected_win - (1 - probabilities) * expected_loss
        accepted = (probabilities >= threshold) & (ev > 0)
    return [
        (
            int(cast(Any, index)),
            int(row[f"exit_index_{side}"]),
            float(value),
            bool(row[f"target_{side}"]),
        )
        for (index, row), value, keep in zip(
            eligible.iterrows(), risk_returns, accepted, strict=True
        )
        if keep
    ]


def _non_overlapping_returns(
    events: list[tuple[int, int, float, bool]], cooldown_bars: int
) -> list[float]:
    return [event[2] for event in _non_overlapping_events(events, cooldown_bars)]


def _non_overlapping_events(
    events: list[tuple[int, int, float, bool]], cooldown_bars: int
) -> list[tuple[int, int, float, bool]]:
    blocked_until = -1
    cooldown_until = -1
    selected: list[tuple[int, int, float, bool]] = []
    for index, exit_index, value, won in sorted(events, key=lambda event: (event[0], event[1])):
        if index <= max(blocked_until, cooldown_until):
            continue
        selected.append((index, exit_index, value, won))
        blocked_until = exit_index
        if not won:
            cooldown_until = exit_index + cooldown_bars
    return selected


def _third_cost_events(
    base: list[tuple[int, int, float, bool]],
    double: list[tuple[int, int, float, bool]],
) -> list[tuple[int, int, float, bool]]:
    if [(item[0], item[1]) for item in base] != [(item[0], item[1]) for item in double]:
        raise ValueError("stress scenarios must use identical trade decisions")
    return [
        (event[0], event[1], stressed[2] + (stressed[2] - event[2]), event[3])
        for event, stressed in zip(base, double, strict=True)
    ]


def _selected_returns(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    side: str,
    probabilities: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    expected_win: float | None = None,
    expected_loss: float | None = None,
    cost_multiplier: float = 1.0,
) -> list[float]:
    events = _candidate_events(
        data,
        candidate,
        side,
        probabilities,
        threshold=threshold,
        expected_win=expected_win,
        expected_loss=expected_loss,
        cost_multiplier=cost_multiplier,
    )
    return _non_overlapping_returns(events, candidate.cooldown_bars)


def _combined_deterministic_returns(
    data: pd.DataFrame, candidate: StrategyCandidate, *, cost_multiplier: float = 1.0
) -> list[float]:
    events = [
        event
        for side in ("long", "short")
        for event in _candidate_events(data, candidate, side, cost_multiplier=cost_multiplier)
    ]
    return _non_overlapping_returns(events, candidate.cooldown_bars)


def _outer_fold_returns_for_events(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    events: list[tuple[int, int, float, bool]],
    config: MachineLearningConfig,
) -> list[list[float]]:
    frame = data.copy()
    frame["exit_index_max"] = frame.loc[:, ["exit_index_long", "exit_index_short"]].max(axis=1)
    folds = rolling_purged_splits(
        frame,
        train_weeks=config.outer_train_weeks,
        calibration_weeks=config.outer_calibration_weeks,
        test_weeks=config.outer_test_weeks,
        step_weeks=config.outer_step_weeks,
        exit_column="exit_index_max",
    )
    returns: list[list[float]] = []
    for fold in folds:
        first = int(fold.validation[0])
        last = int(fold.validation[-1])
        selected = [event for event in events if first <= event[0] <= last]
        returns.append(_non_overlapping_returns(selected, candidate.cooldown_bars))
    return returns


def _metrics(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "trades": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "net_equity_return": 0.0,
            "win_rate": 0.0,
        }
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= max(0.0, 1 + 0.01 * value)
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    wins = sum(value for value in returns if value > 0)
    losses = abs(sum(value for value in returns if value < 0))
    return {
        "trades": float(len(returns)),
        "expectancy_r": float(statistics.mean(returns)),
        "profit_factor": wins / losses if losses else (999.0 if wins else 0.0),
        "max_drawdown": drawdown,
        "net_equity_return": equity - 1,
        "win_rate": sum(value > 0 for value in returns) / len(returns),
    }


def _robust_score(fold_returns: list[list[float]]) -> float:
    expectancy = [statistics.mean(values) for values in fold_returns if values]
    trades = sum(map(len, fold_returns))
    if len(expectancy) < 2 or trades < 20:
        return INVALID_SCORE
    median = statistics.median(expectancy)
    mad = statistics.median(abs(value - median) for value in expectancy)
    return median - 1.4826 * mad


def _evaluate_candidate(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    config: MachineLearningConfig,
) -> dict[str, Any]:
    data = data.copy()
    for side in ("long", "short"):
        data[f"eligible_{side}"] = _entry_mask(data, candidate, side)
    data["exit_index_max"] = data.loc[:, ["exit_index_long", "exit_index_short"]].max(axis=1)
    folds = rolling_purged_splits(
        data,
        train_weeks=config.outer_train_weeks,
        calibration_weeks=config.outer_calibration_weeks,
        test_weeks=config.outer_test_weeks,
        step_weeks=config.outer_step_weeks,
        exit_column="exit_index_max",
    )
    side_returns: dict[str, list[float]] = {"long": [], "short": []}
    fold_returns: list[list[float]] = []
    combined: list[float] = []
    for fold in folds:
        validation = data.iloc[fold.validation]
        side_events = {
            side: _candidate_events(validation, candidate, side) for side in ("long", "short")
        }
        executed = _non_overlapping_events(
            [event for events in side_events.values() for event in events],
            candidate.cooldown_bars,
        )
        selected = [event[2] for event in executed]
        combined.extend(selected)
        fold_returns.append(selected)
        for side, events in side_events.items():
            event_set = set(events)
            side_returns[side].extend(event[2] for event in executed if event in event_set)
    return {
        "candidate_id": candidate.candidate_id,
        "parameters": asdict(candidate),
        "score": _robust_score(fold_returns),
        "metrics": _metrics(combined),
        "sides": {side: _metrics(values) for side, values in side_returns.items()},
        "fold_returns": fold_returns,
        "returns": combined,
    }


def _initialize_screen_worker(
    app: AppConfig,
    frames: dict[int, pd.DataFrame],
    config: MachineLearningConfig,
) -> None:
    global _SCREEN_APP, _SCREEN_FRAMES, _SCREEN_CONFIG, _SCREEN_FEATURE_CACHE
    _SCREEN_APP = app
    _SCREEN_FRAMES = frames
    _SCREEN_CONFIG = config
    _SCREEN_FEATURE_CACHE = {}


def _feature_key(candidate: StrategyCandidate) -> tuple[int, int, int, int]:
    return (
        candidate.timeframe_minutes,
        candidate.vwap_hours,
        candidate.atr_period,
        candidate.adx_period,
    )


def _screen_candidate_worker(candidate: StrategyCandidate) -> dict[str, Any]:
    if _SCREEN_APP is None or _SCREEN_CONFIG is None:
        raise RuntimeError("scientific screening worker is not initialized")
    key = _feature_key(candidate)
    features = _SCREEN_FEATURE_CACHE.get(key)
    if features is None:
        frame = _SCREEN_FRAMES[candidate.timeframe_minutes]
        strategy = _SCREEN_APP.strategy.model_copy(
            update={
                "timeframe_minutes": candidate.timeframe_minutes,
                "crypto_vwap_window": candidate.vwap_bars,
                "atr_period": candidate.atr_period,
                "adx_period": candidate.adx_period,
            }
        )
        configured = _SCREEN_APP.model_copy(update={"strategy": strategy})
        complete = frame[frame["funding_rate"].notna()].copy()
        complete["funding_source"] = "observed"
        features, _ = build_ml_features(complete, configured)
        _SCREEN_FEATURE_CACHE.clear()
        _SCREEN_FEATURE_CACHE[key] = features
    data = _label_candidate_frame(features, candidate)
    try:
        return _evaluate_candidate(data, candidate, _SCREEN_CONFIG)
    except (ValueError, IndexError):
        return {
            "candidate_id": candidate.candidate_id,
            "parameters": asdict(candidate),
            "score": INVALID_SCORE,
            "metrics": _metrics([]),
            "sides": {"long": _metrics([]), "short": _metrics([])},
            "fold_returns": [],
            "returns": [],
        }


def _logistic(seed: int) -> ClassifierMixin:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5, max_iter=1000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )


def _xgb(params: dict[str, Any], seed: int) -> ClassifierMixin:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )


def _fit_calibrated(
    estimator: ClassifierMixin,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    side: str,
) -> ClassifierMixin:
    target = f"target_{side}"
    counts = calibration[target].value_counts()
    if len(counts) < 2 or int(counts.min()) < CALIBRATION_CV_SPLITS:
        raise ValueError("calibration fold has insufficient minority-class support")
    estimator.fit(train.loc[:, FEATURE_COLUMNS], train[target])
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(estimator), method="sigmoid", cv=CALIBRATION_CV_SPLITS
    )
    calibrated.fit(calibration.loc[:, FEATURE_COLUMNS], calibration[target])
    return calibrated


def _payoff_priors(
    data: pd.DataFrame,
    side: str,
    candidate: StrategyCandidate | None = None,
) -> tuple[float, float]:
    values = data[f"net_return_{side}"].to_numpy(dtype=float) - BASE_EXTRA_COST_BPS / 10_000
    if candidate is not None:
        stop_fraction = (
            candidate.stop_atr
            * data["atr"].to_numpy(dtype=float)
            / data["entry_price"].to_numpy(dtype=float)
        )
        values = np.divide(
            values,
            stop_fraction,
            out=np.full_like(values, np.nan),
            where=stop_fraction > 0,
        )
    values = values[np.isfinite(values)]
    wins = values[values > 0]
    losses = values[values <= 0]
    return (
        float(wins.mean()) if len(wins) else 0.0,
        abs(float(losses.mean())) if len(losses) else 1.0,
    )


def _economic_threshold(expected_win: float, expected_loss: float) -> float:
    total = expected_win + expected_loss
    return expected_loss / total if total > 0 else 1.0


def _tune_xgb_on_eligible(
    eligible: pd.DataFrame,
    candidate: StrategyCandidate,
    side: str,
    config: MachineLearningConfig,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], float]:
    folds = purged_time_splits(
        eligible,
        n_splits=config.n_splits,
        embargo_bars=candidate.holding_bars,
        exit_column=f"exit_index_{side}",
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100, log=True),
        }
        scores: list[list[float]] = []
        for fold in folds:
            train = eligible.iloc[fold.train]
            validation = eligible.iloc[fold.validation]
            split = int(len(train) * 0.8)
            calibration = train.iloc[split:]
            first_calibration_event = int(calibration.iloc[0]["event_index"])
            base = train.iloc[:split]
            base = base.loc[base[f"exit_index_{side}"].lt(first_calibration_event)]
            if (
                min(len(base), len(calibration), len(validation)) < 50
                or base[f"target_{side}"].nunique() < 2
                or calibration[f"target_{side}"].nunique() < 2
                or int(calibration[f"target_{side}"].value_counts().min()) < CALIBRATION_CV_SPLITS
            ):
                raise optuna.TrialPruned()
            model = _fit_calibrated(_xgb(params, config.random_seed), base, calibration, side)
            probability = model.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1]
            win, loss = _payoff_priors(train, side, candidate)
            threshold = _economic_threshold(win, loss)
            validation = validation.set_index("event_index", drop=False)
            validation[f"eligible_{side}"] = True
            selected = _selected_returns(
                validation,
                candidate,
                side,
                probability,
                threshold=threshold,
                expected_win=win,
                expected_loss=loss,
            )
            scores.append(selected)
        return _robust_score(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
    if progress is not None:
        callbacks.append(
            lambda _study, trial: progress(trial.number + 1, config.model_trials_per_side)
        )
    study.optimize(
        objective,
        n_trials=config.model_trials_per_side,
        gc_after_trial=True,
        callbacks=callbacks,
    )
    return dict(study.best_params), float(study.best_value)


def _append_outer_predictions(
    destination: OofEvaluation,
    model: ClassifierMixin,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: StrategyCandidate,
    side: str,
) -> None:
    probability = model.predict_proba(test.loc[:, FEATURE_COLUMNS])[:, 1]
    win, loss = _payoff_priors(train, side, candidate)
    threshold = _economic_threshold(win, loss)
    indexed = test.set_index("event_index", drop=False)
    indexed[f"eligible_{side}"] = True
    destination.events.extend(
        _candidate_events(
            indexed,
            candidate,
            side,
            probability,
            threshold=threshold,
            expected_win=win,
            expected_loss=loss,
        )
    )
    destination.stress_events.extend(
        _candidate_events(
            indexed,
            candidate,
            side,
            probability,
            threshold=threshold,
            expected_win=win,
            expected_loss=loss,
            cost_multiplier=2,
        )
    )
    destination.probabilities.extend(probability.tolist())
    destination.targets.extend(test[f"target_{side}"].to_numpy(dtype=int).tolist())


def _nested_meta_evaluations(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    side: str,
    config: MachineLearningConfig,
    progress: Callable[[int, int, float], None] | None = None,
    trial_progress: Callable[[int, int, int, int], None] | None = None,
) -> tuple[OofEvaluation, OofEvaluation, dict[str, Any], float]:
    eligible = data.loc[_entry_mask(data, candidate, side)].copy()
    eligible[f"target_{side}"] = eligible[f"net_return_{side}"].gt(0).astype(np.int8)
    eligible["event_index"] = eligible.index
    eligible = eligible.reset_index(drop=True)
    data_with_exits = data.copy()
    data_with_exits["exit_index_max"] = data_with_exits[
        ["exit_index_long", "exit_index_short"]
    ].max(axis=1)
    folds = rolling_nested_splits(
        data_with_exits,
        train_weeks=config.outer_train_weeks,
        calibration_weeks=config.outer_calibration_weeks,
        test_weeks=config.outer_test_weeks,
        step_weeks=config.outer_step_weeks,
        exit_column="exit_index_max",
    )
    logistic = OofEvaluation([], [], [], [])
    xgboost = OofEvaluation([], [], [], [])
    latest_params: dict[str, Any] = {}
    latest_objective = INVALID_SCORE
    for fold_number, fold in enumerate(folds, 1):
        train_indexes = set(fold.train.tolist())
        calibration_indexes = set(fold.calibration.tolist())
        test_indexes = set(fold.test.tolist())
        train = eligible.loc[eligible["event_index"].isin(train_indexes)].reset_index(drop=True)
        calibration = eligible.loc[eligible["event_index"].isin(calibration_indexes)].reset_index(
            drop=True
        )
        test = eligible.loc[eligible["event_index"].isin(test_indexes)].reset_index(drop=True)
        if (
            min(len(train), len(calibration), len(test)) < 50
            or train[f"target_{side}"].nunique() < 2
            or calibration[f"target_{side}"].nunique() < 2
            or int(calibration[f"target_{side}"].value_counts().min()) < CALIBRATION_CV_SPLITS
        ):
            continue

        def notify_trial(trial: int, trial_total: int, fold: int = fold_number) -> None:
            if trial_progress is not None:
                trial_progress(fold, len(folds), trial, trial_total)

        latest_params, latest_objective = _tune_xgb_on_eligible(
            train,
            candidate,
            side,
            config,
            progress=notify_trial if trial_progress is not None else None,
        )
        logistic_model = _fit_calibrated(_logistic(config.random_seed), train, calibration, side)
        xgb_model = _fit_calibrated(
            _xgb(latest_params, config.random_seed), train, calibration, side
        )
        _append_outer_predictions(logistic, logistic_model, train, test, candidate, side)
        _append_outer_predictions(xgboost, xgb_model, train, test, candidate, side)
        if progress is not None:
            progress(fold_number, len(folds), latest_objective)
    if not latest_params:
        raise ValueError(f"no valid nested folds for {side}")
    return logistic, xgboost, latest_params, latest_objective


def _bootstrap_lower(returns: list[float], seed: int, repetitions: int = 2000) -> float:
    if len(returns) < 2:
        return -math.inf
    generator = np.random.default_rng(seed)
    values = np.asarray(returns, dtype=float)
    block = max(1, min(20, int(math.sqrt(len(values)))))
    means = np.empty(repetitions)
    for position in range(repetitions):
        sample: list[float] = []
        while len(sample) < len(values):
            start = int(generator.integers(0, len(values)))
            sample.extend(values.take(np.arange(start, start + block) % len(values)).tolist())
        means[position] = np.mean(sample[: len(values)])
    return float(np.quantile(means, 0.025))


def paired_superiority_pvalue(
    champion_folds: list[list[float]], baseline_folds: list[list[float]], seed: int
) -> float:
    """One-sided paired randomization test on aligned walk-forward folds."""
    count = min(len(champion_folds), len(baseline_folds))
    if count < 2:
        return 1.0
    differences = np.asarray(
        [
            (statistics.mean(champion_folds[index]) if champion_folds[index] else 0.0)
            - (statistics.mean(baseline_folds[index]) if baseline_folds[index] else 0.0)
            for index in range(count)
        ],
        dtype=float,
    )
    observed = float(differences.mean())
    if observed <= 0:
        return 1.0
    if count <= 16:
        null = [float(np.mean(differences * signs)) for signs in product((-1, 1), repeat=count)]
    else:
        generator = np.random.default_rng(seed)
        null = [
            float(np.mean(differences * generator.choice((-1, 1), size=count)))
            for _ in range(20_000)
        ]
    return (1 + sum(value >= observed for value in null)) / (len(null) + 1)


def reality_check_pvalue(
    candidate_folds: list[list[list[float]]],
    baseline_folds: list[list[float]],
    seed: int,
    repetitions: int = 5000,
) -> float:
    """White-style block bootstrap over every tried strategy and aligned OOS folds."""
    count = min([len(baseline_folds), *(len(item) for item in candidate_folds)])
    if count < 2 or not candidate_folds:
        return 1.0
    baseline = np.asarray(
        [statistics.mean(values) if values else 0.0 for values in baseline_folds[:count]]
    )
    differences = (
        np.asarray(
            [
                [statistics.mean(values) if values else 0.0 for values in folds[:count]]
                for folds in candidate_folds
            ]
        )
        - baseline
    )
    observed = float(differences.mean(axis=1).max())
    if observed <= 0:
        return 1.0
    centered = differences - differences.mean(axis=1, keepdims=True)
    generator = np.random.default_rng(seed)
    block_size = max(2, int(np.sqrt(count)))
    block_count = int(np.ceil(count / block_size))
    starts = generator.integers(0, count, size=(repetitions, block_count))
    samples = ((starts[..., None] + np.arange(block_size)) % count).reshape(repetitions, -1)[
        :, :count
    ]
    null = np.full(repetitions, -np.inf)
    for candidate in centered:
        np.maximum(null, candidate[samples].mean(axis=1), out=null)
    return float((1 + np.count_nonzero(null >= observed)) / (repetitions + 1))


def _gate(
    metrics: dict[str, float],
    stress: dict[str, float],
    *,
    returns: list[float],
    config: MachineLearningConfig,
    pbo: float | None,
    dsr: float,
    reality_pvalue: float | None = None,
    side_metrics: dict[str, dict[str, float]] | None = None,
    holdout: bool = False,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    minimum = config.minimum_holdout_trades if holdout else config.minimum_oos_trades
    checks = {
        f"trades<{minimum}": metrics["trades"] >= minimum,
        "expectancy<=0": metrics["expectancy_r"] > 0,
        "profit_factor<gate": metrics["profit_factor"] >= config.minimum_profit_factor,
        "drawdown>8%": metrics["max_drawdown"] <= 0.08,
        "stress_2x_negative": stress["expectancy_r"] >= 0,
        "bootstrap_lower<=0": _bootstrap_lower(returns, config.random_seed) > 0,
    }
    if not holdout:
        checks["pbo>gate"] = pbo is not None and pbo <= config.maximum_pbo
        checks["dsr<gate"] = dsr >= config.minimum_dsr_probability
        checks["reality_check_pvalue>gate"] = (
            reality_pvalue is not None and reality_pvalue <= config.maximum_reality_check_pvalue
        )
    if side_metrics is not None:
        for side, values in side_metrics.items():
            checks[f"{side}_trades<{config.minimum_side_trades}"] = (
                values["trades"] >= config.minimum_side_trades
            )
    for reason, passed in checks.items():
        if not passed:
            failures.append(reason)
    return not failures, failures


def _event_driven_replay(
    app: AppConfig,
    frame: pd.DataFrame,
    candidate: StrategyCandidate,
    allowed_entries: set[tuple[datetime, str]],
) -> dict[str, Any]:
    strategy_config = app.strategy.model_copy(
        update={
            "timeframe_minutes": candidate.timeframe_minutes,
            "crypto_vwap_window": candidate.vwap_bars,
            "atr_period": candidate.atr_period,
            "adx_period": candidate.adx_period,
            "range_adx_threshold": candidate.range_adx_threshold,
            "entry_z": min(candidate.entry_z_long, candidate.entry_z_short),
            "stop_atr": candidate.stop_atr,
            "time_stop_bars": candidate.holding_bars,
            "confirmation_bars": 1,
            "short_enabled": True,
            "session_flatten_enabled": False,
            "fixed_stop_fraction": None,
            "fixed_target_fraction": None,
            "max_spread_bps": 3.0,
        }
    )
    configured = app.model_copy(update={"strategy": strategy_config})
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    segment_ids = timestamps.diff().ne(pd.Timedelta(minutes=candidate.timeframe_minutes)).cumsum()
    segments = [segment.reset_index(drop=True) for _, segment in frame.groupby(segment_ids)]
    results = [
        (
            segment,
            asyncio.run(
                BacktestEngine(
                    configured,
                    strategy=CandidateReplayStrategy(candidate, allowed_entries),
                ).run(segment, mode="scientific_event_replay")
            ),
        )
        for segment in segments
    ]
    breaches = 0
    missing_mark_intervals = 0
    ending_positions = 0
    distance = Decimal("1") / app.instrument.max_leverage - app.risk.liquidation_buffer_fraction
    for segment, result in results:
        position = Decimal("0")
        entry: tuple[datetime, Side, Decimal] | None = None
        segment_timestamps = pd.to_datetime(segment["timestamp"], utc=True)
        for fill in result.fills:
            signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
            previous = position
            position += signed
            if previous == 0 and position != 0:
                entry = (fill.exchange_timestamp, fill.side, fill.price)
            if position == 0 and entry is not None:
                opened, side, price = entry
                interval = segment.loc[
                    segment_timestamps.ge(opened) & segment_timestamps.le(fill.exchange_timestamp)
                ]
                liquidation = price * (
                    Decimal("1") - distance if side is Side.BUY else Decimal("1") + distance
                )
                mark = pd.to_numeric(
                    interval["mark_low" if side is Side.BUY else "mark_high"], errors="coerce"
                )
                observed = mark.min() if side is Side.BUY else mark.max()
                if interval.empty or not math.isfinite(float(observed)):
                    missing_mark_intervals += 1
                elif side is Side.BUY:
                    breaches += int(Decimal(str(observed)) <= liquidation)
                else:
                    breaches += int(Decimal(str(observed)) >= liquidation)
                entry = None
        ending_positions += int(position != 0)
    kill_switches = sum(result.kill_switches for _, result in results)
    passed = (
        not kill_switches
        and breaches == 0
        and missing_mark_intervals == 0
        and ending_positions == 0
    )
    return {
        "passed": passed,
        "engine": "shared BacktestEngine + DefaultRiskEngine + SimulatedBroker",
        "signals": sum(result.signals for _, result in results),
        "rejected_signals": sum(result.rejected_signals for _, result in results),
        "fills": sum(len(result.fills) for _, result in results),
        "kill_switches": kill_switches,
        "ending_position": "0" if ending_positions == 0 else "unclosed_segment",
        "continuous_segments": len(segments),
        "observed_data_gaps": len(segments) - 1,
        "liquidation_breaches": breaches,
        "missing_mark_intervals": missing_mark_intervals,
        "mark_price_source": "observed Bitunix mark OHLC",
        "liquidation_price_source": "independent conservative 10x estimator",
        "risk_per_trade": str(app.risk.risk_per_trade),
    }


def run_scientific_research(
    app: AppConfig, one_minute: pd.DataFrame, *, resume: bool = False
) -> dict[str, Any]:
    config = _scientific_config(app)
    if app.risk.risk_per_trade != Decimal("0.01"):
        raise ValueError("scientific research requires fixed risk_per_trade=1%")
    if app.bitunix is None or str(app.bitunix.leverage) != "10":
        raise ValueError("scientific research requires fixed leverage=10x")
    gpu = gpu_preflight(required=config.gpu_required)
    started = time.monotonic()
    previous_status: dict[str, Any] = {}
    if resume and config.status_path.exists():
        previous_status = json.loads(config.status_path.read_text(encoding="utf-8"))
    run_id = str(previous_status.get("run_id") or "")
    if resume:
        checkpoints = tuple(Path("data/research/mlv2").glob("mlv2-*/strategy_screen.jsonl"))
        if checkpoints:
            run_id = max(checkpoints, key=lambda path: path.stat().st_size).parent.name
    if not run_id:
        run_id = datetime.now(UTC).strftime("mlv2-%Y%m%dT%H%M%SZ")
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True)
    holdout_end = timestamps.max() + pd.Timedelta(minutes=1)
    holdout_start = holdout_end - pd.Timedelta(weeks=config.holdout_weeks)
    development_raw = one_minute.loc[timestamps.lt(holdout_start)].copy()
    if development_raw.empty:
        raise ValueError("no development data precedes the sealed holdout")
    funding_rows = development_raw.loc[development_raw["funding_rate"].notna(), "timestamp"]
    if funding_rows.empty:
        raise ValueError("no observed Bitunix funding precedes the sealed holdout")
    frames = {
        minutes: resample_observed(development_raw, minutes, float(app.instrument.tick_size))
        for minutes in config.timeframes
    }
    candidates = generate_candidates(config)
    checkpoint = Path("data/research/mlv2") / run_id / "strategy_screen.jsonl"
    evaluations: list[dict[str, Any]] = []
    if resume and checkpoint.exists():
        evaluations = [
            json.loads(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_ids = {str(item["candidate_id"]) for item in evaluations}
    write_ml_status(
        config,
        "strategy_screen",
        "Scientific deterministic strategy screening",
        2,
        run_id=run_id,
        completed=len(evaluations),
        total=len(candidates),
        backend=gpu["backend"],
        elapsed_seconds=0,
        compute=f"{config.parallel_workers} CPU workers + CUDA meta-model",
    )
    pending = sorted(
        (candidate for candidate in candidates if candidate.candidate_id not in completed_ids),
        key=_feature_key,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output = checkpoint.open("a", encoding="utf-8")

    def record(candidate: StrategyCandidate, evaluation: dict[str, Any]) -> None:
        evaluations.append(evaluation)
        checkpoint_output.write(json.dumps(evaluation, default=str, allow_nan=False) + "\n")
        completed = len(evaluations)
        processed = completed - len(completed_ids)
        if processed == 1 or completed % 10 == 0 or completed == len(candidates):
            checkpoint_output.flush()
            elapsed = time.monotonic() - started
            rate = elapsed / max(processed, 1)
            best = max(evaluations, key=lambda item: float(item["score"]))
            write_ml_status(
                config,
                "strategy_screen",
                f"Candidate {completed}/{len(candidates)}",
                2 + completed / len(candidates) * 58,
                run_id=run_id,
                completed=completed,
                total=len(candidates),
                backend=gpu["backend"],
                elapsed_seconds=round(elapsed),
                eta_seconds=round(rate * (len(candidates) - completed)),
                current_candidate=candidate.candidate_id,
                timeframe_minutes=candidate.timeframe_minutes,
                compute=f"{config.parallel_workers} CPU workers + CUDA meta-model",
                best_candidate=best["candidate_id"],
                best_score_r=round(float(best["score"]), 6),
                best_expectancy_r=round(float(best["metrics"]["expectancy_r"]), 6),
                best_profit_factor=round(float(best["metrics"]["profit_factor"]), 4),
                best_trades=int(best["metrics"]["trades"]),
            )

    try:
        if config.parallel_workers == 1:
            _initialize_screen_worker(app, frames, config)
            for candidate in pending:
                record(candidate, _screen_candidate_worker(candidate))
        else:
            with ProcessPoolExecutor(
                max_workers=config.parallel_workers,
                initializer=_initialize_screen_worker,
                initargs=(app, frames, config),
            ) as pool:
                futures = {
                    pool.submit(_screen_candidate_worker, candidate): candidate
                    for candidate in pending
                }
                for future in as_completed(futures):
                    record(futures[future], future.result())
    finally:
        checkpoint_output.close()
    evaluations.sort(key=lambda item: (float(item["score"]), item["candidate_id"]), reverse=True)
    valid = [item for item in evaluations if item["fold_returns"] and item["score"] > INVALID_SCORE]
    if not valid:
        raise ValueError("no valid strategy survived deterministic walk-forward screening")
    full = valid[: min(config.full_candidates, len(valid))]
    write_ml_status(
        config,
        "selection_bias_audit",
        f"Vectorized combinatorial PBO across {len(valid)} valid candidates",
        60,
        run_id=run_id,
        completed=0,
        total=252,
        backend="cpu-vectorized",
        elapsed_seconds=round(time.monotonic() - started),
    )
    pbo_audit = combinatorial_pbo([item["fold_returns"] for item in valid])
    trial_sharpes = [
        statistics.mean(item["returns"]) / statistics.stdev(item["returns"])
        if len(item["returns"]) > 1 and statistics.stdev(item["returns"])
        else 0.0
        for item in valid
    ]
    finalists = full[: min(config.meta_candidates, len(full))]
    artifacts: list[dict[str, Any]] = []
    candidate_root = config.candidate_directory / run_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    for position, evaluation in enumerate(finalists, 1):
        candidate = StrategyCandidate(**evaluation["parameters"])
        data = _candidate_frame(frames[candidate.timeframe_minutes], app, candidate)
        models: dict[str, ClassifierMixin] = {}
        model_data: dict[str, Any] = {}
        selected_oof: dict[str, OofEvaluation] = {}
        for side_position, side in enumerate(("long", "short")):
            eligible = data.loc[_entry_mask(data, candidate, side)].copy()
            eligible[f"target_{side}"] = eligible[f"net_return_{side}"].gt(0).astype(np.int8)
            eligible["event_index"] = eligible.index
            eligible = eligible.reset_index(drop=True)
            if len(eligible) < 300 or eligible[f"target_{side}"].nunique() < 2:
                model_data[side] = {"status": "insufficient", "samples": len(eligible)}
                continue

            def report_nested_progress(
                fold: int,
                total: int,
                score: float,
                candidate_id: str = candidate.candidate_id,
                current_side: str = side,
                finalist_position: int = position,
                current_side_position: int = side_position,
            ) -> None:
                write_ml_status(
                    config,
                    "nested_meta_model",
                    f"{candidate_id} {current_side}: outer fold {fold}/{total}",
                    60
                    + ((finalist_position - 1) + (current_side_position + fold / total) / 2)
                    / len(finalists)
                    * 35,
                    run_id=run_id,
                    completed=fold,
                    total=total,
                    backend=gpu["backend"],
                    model="XGBoost CUDA + Logistic control",
                    side=current_side,
                    inner_best_robust_score_r=round(score, 6),
                )

            def report_trial_progress(
                fold: int,
                fold_total: int,
                trial: int,
                trial_total: int,
                candidate_id: str = candidate.candidate_id,
                current_side: str = side,
                finalist_position: int = position,
                current_side_position: int = side_position,
            ) -> None:
                write_ml_status(
                    config,
                    "nested_meta_model",
                    (
                        f"{candidate_id} {current_side}: fold {fold}/{fold_total}, "
                        f"trial {trial}/{trial_total}"
                    ),
                    60
                    + (
                        (finalist_position - 1)
                        + (current_side_position + (fold - 1 + trial / trial_total) / fold_total)
                        / 2
                    )
                    / len(finalists)
                    * 35,
                    run_id=run_id,
                    completed=trial,
                    total=trial_total,
                    backend=gpu["backend"],
                    model="XGBoost CUDA + Logistic control",
                    side=current_side,
                    outer_fold=fold,
                    outer_folds=fold_total,
                )

            try:
                logistic_oof, xgb_oof, params, objective = _nested_meta_evaluations(
                    data,
                    candidate,
                    side,
                    config,
                    progress=report_nested_progress,
                    trial_progress=report_trial_progress,
                )
            except ValueError:
                model_data[side] = {"status": "insufficient_nested_folds", "samples": len(eligible)}
                continue
            logistic_folds = _outer_fold_returns_for_events(
                data, candidate, logistic_oof.events, config
            )
            xgb_folds = _outer_fold_returns_for_events(data, candidate, xgb_oof.events, config)
            logistic_score = _robust_score(logistic_folds)
            xgb_score = _robust_score(xgb_folds)
            model_kind = "logistic" if logistic_score > xgb_score else "xgboost"
            side_oof = logistic_oof if model_kind == "logistic" else xgb_oof
            selected_oof[side] = side_oof
            split = int(len(eligible) * 0.8)
            base = eligible.iloc[:split]
            calibration = eligible.iloc[split:]
            first_calibration_event = int(calibration.iloc[0]["event_index"])
            base = base.loc[base[f"exit_index_{side}"].lt(first_calibration_event)]
            calibration_counts = calibration[f"target_{side}"].value_counts()
            if len(calibration_counts) < 2 or int(calibration_counts.min()) < CALIBRATION_CV_SPLITS:
                model_data[side] = {
                    "status": "insufficient_calibration_class_support",
                    "samples": len(eligible),
                }
                continue
            estimator = (
                _logistic(config.random_seed)
                if model_kind == "logistic"
                else _xgb(params, config.random_seed)
            )
            model = _fit_calibrated(estimator, base, calibration, side)
            win, loss = _payoff_priors(base, side, candidate)
            threshold = _economic_threshold(win, loss)
            target = np.asarray(side_oof.targets, dtype=int)
            probability = np.asarray(side_oof.probabilities, dtype=float)
            side_returns = side_oof.returns(candidate.cooldown_bars)
            models[side] = model
            model_data[side] = {
                "status": "trained",
                "samples": len(eligible),
                "model_kind": model_kind,
                "objective": objective,
                "oof_metrics": _metrics(side_returns),
                "comparison": {
                    "logistic_robust_score_r": logistic_score,
                    "xgboost_robust_score_r": xgb_score,
                },
                "parameters": {
                    **(params if model_kind == "xgboost" else {}),
                    "threshold": threshold,
                },
                "average_precision": float(average_precision_score(target, probability)),
                "brier": float(brier_score_loss(target, probability)),
                "log_loss": float(log_loss(target, probability, labels=[0, 1])),
                "payoff_priors": (win, loss),
                "metrics_source": "nested_outer_test_only",
            }
        meta_events = [event for result in selected_oof.values() for event in result.events]
        meta_stress_events = [
            event for result in selected_oof.values() for event in result.stress_events
        ]
        meta_stress_3x_events = _third_cost_events(meta_events, meta_stress_events)
        executed_events = _non_overlapping_events(meta_events, candidate.cooldown_bars)
        meta_returns = [event[2] for event in executed_events]
        meta_stress_returns = _non_overlapping_returns(meta_stress_events, candidate.cooldown_bars)
        meta_stress_3x_returns = _non_overlapping_returns(
            meta_stress_3x_events, candidate.cooldown_bars
        )
        meta_folds = _outer_fold_returns_for_events(data, candidate, meta_events, config)
        meta_score = _robust_score(meta_folds)
        meta_sides = {
            side: _metrics([event[2] for event in executed_events if event in set(result.events)])
            for side, result in selected_oof.items()
        }
        artifact_path = candidate_root / f"{candidate.candidate_id}.joblib"
        evaluation["artifact"] = str(artifact_path)
        evaluation["models"] = model_data
        evaluation["meta_metrics"] = _metrics(meta_returns)
        evaluation["meta_stress_metrics"] = _metrics(meta_stress_returns)
        evaluation["meta_stress_3x_metrics"] = _metrics(meta_stress_3x_returns)
        evaluation["meta_sides"] = meta_sides
        evaluation["meta_fold_returns"] = meta_folds
        evaluation["meta_score"] = meta_score
        evaluation["selected_mode"] = (
            "meta_label"
            if float(evaluation["meta_score"]) > float(evaluation["score"])
            else "deterministic"
        )
        evaluation["meta_returns"] = meta_returns
        evaluation["meta_stress_returns"] = meta_stress_returns
        evaluation["meta_event_indexes"] = [event[0] for event in executed_events]
        joblib.dump(
            {
                "protocol": PROTOCOL_VERSION,
                "run_id": run_id,
                "candidate": asdict(candidate),
                "models": models,
                "model_data": model_data,
                "selected_mode": evaluation["selected_mode"],
                "features": FEATURE_COLUMNS,
                "trained_until": holdout_start.isoformat(),
            },
            artifact_path,
        )
        artifacts.append(evaluation)
        write_ml_status(
            config,
            "meta_model",
            f"GPU meta-model {position}/{len(finalists)}",
            60 + position / len(finalists) * 35,
            run_id=run_id,
            completed=position,
            total=len(finalists),
            backend=gpu["backend"],
            elapsed_seconds=round(time.monotonic() - started),
            current_candidate=candidate.candidate_id,
            deterministic_score_r=round(float(evaluation["score"]), 6),
            meta_score_r=round(float(evaluation["meta_score"]), 6),
            selected_mode=evaluation["selected_mode"],
        )
    artifacts.sort(
        key=lambda item: max(float(item["score"]), float(item.get("meta_score", INVALID_SCORE))),
        reverse=True,
    )
    champion = artifacts[0] if artifacts else full[0]
    returns = cast(
        list[float],
        champion.get("meta_returns", champion["returns"])
        if champion.get("selected_mode") == "meta_label"
        else champion["returns"],
    )
    dsr = deflated_sharpe_probability(returns, trial_sharpes)
    base_metrics = cast(
        dict[str, float],
        champion["meta_metrics"]
        if champion.get("selected_mode") == "meta_label"
        else champion["metrics"],
    )
    candidate = StrategyCandidate(**champion["parameters"])
    champion_data = _candidate_frame(frames[candidate.timeframe_minutes], app, candidate)
    stress_returns = cast(
        list[float],
        champion.get("meta_stress_returns", [])
        if champion.get("selected_mode") == "meta_label"
        else _combined_deterministic_returns(champion_data, candidate, cost_multiplier=2),
    )
    stress_metrics = _metrics(stress_returns)
    stress_3x_metrics = cast(
        dict[str, float],
        champion.get("meta_stress_3x_metrics")
        if champion.get("selected_mode") == "meta_label"
        else _metrics(_combined_deterministic_returns(champion_data, candidate, cost_multiplier=3)),
    )
    replay_allowed = set(cast(list[int], champion.get("meta_event_indexes", [])))
    if champion.get("selected_mode") != "meta_label":
        replay_allowed = {
            event[0]
            for event in _non_overlapping_events(
                [
                    event
                    for side in ("long", "short")
                    for event in _candidate_events(champion_data, candidate, side)
                ],
                candidate.cooldown_bars,
            )
        }
    write_ml_status(
        config,
        "event_replay",
        "Shared execution/risk replay with observed mark-price liquidation audit",
        96,
        run_id=run_id,
        backend=gpu["backend"],
        candidate=candidate.candidate_id,
        selected_mode=champion.get("selected_mode", "deterministic"),
        allowed_entries=len(replay_allowed),
    )
    replay = _event_driven_replay(
        app,
        frames[candidate.timeframe_minutes],
        candidate,
        {
            (pd.Timestamp(champion_data.iloc[index]["timestamp"]).to_pydatetime(), side)
            for index in replay_allowed
            for side in ("long", "short")
            if bool(_entry_mask(champion_data, candidate, side).iloc[index])
        },
    )
    baseline = next(item for item in evaluations if item["candidate_id"] == "baseline-adx20")
    write_ml_status(
        config,
        "multiple_testing",
        f"PBO, DSR and Reality Check across {len(valid)} valid candidates",
        98,
        run_id=run_id,
        backend=gpu["backend"],
        candidate_count=len(valid),
    )
    reality_pvalue = reality_check_pvalue(
        [cast(list[list[float]], item["fold_returns"]) for item in valid],
        cast(list[list[float]], baseline["fold_returns"]),
        config.random_seed,
    )
    side_metrics = cast(
        dict[str, dict[str, float]],
        champion.get("meta_sides", champion["sides"])
        if champion.get("selected_mode") == "meta_label"
        else champion["sides"],
    )
    accepted, failures = _gate(
        base_metrics,
        stress_metrics,
        returns=returns,
        config=config,
        pbo=cast(float | None, pbo_audit.get("pbo")),
        dsr=dsr,
        reality_pvalue=reality_pvalue,
        side_metrics=side_metrics,
    )
    if not replay["passed"]:
        accepted = False
        failures.append("event_driven_replay_failed")
    for item in evaluations:
        item.pop("returns", None)
        item.pop("fold_returns", None)
        item.pop("meta_returns", None)
        item.pop("meta_stress_returns", None)
        item.pop("meta_fold_returns", None)
        item.pop("meta_event_indexes", None)
    payload = {
        "schema_version": 2,
        "protocol": PROTOCOL_VERSION,
        "run_id": run_id,
        "mode": "research_development_only",
        "started_at": (
            datetime.now(UTC) - timedelta(seconds=time.monotonic() - started)
        ).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "data": {
            "development_start": timestamps.min().isoformat(),
            "economic_development_start": pd.to_datetime(funding_rows.min(), utc=True).isoformat(),
            "development_end": holdout_start.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "holdout_end": holdout_end.isoformat(),
            "historical_spread": "unavailable_not_estimated",
            "spread_assumption_bps": float(app.backtest.spread_bps),
            "slippage_assumption_bps_per_side": float(app.backtest.slippage_bps),
            "funding": "observed_rows_only",
        },
        "compute": gpu,
        "search": {
            "strategy_candidates": len(candidates),
            "full_candidates": len(full),
            "meta_candidates": len(artifacts),
            "timeframes": list(config.timeframes),
            "outer_walk_forward_weeks": {
                "train": config.outer_train_weeks,
                "calibration": config.outer_calibration_weeks,
                "test": config.outer_test_weeks,
                "step": config.outer_step_weeks,
            },
        },
        "selection_bias": {
            **pbo_audit,
            "candidate_count": len(valid),
            "deflated_sharpe_probability": dsr,
            "baseline": "baseline-adx20",
            "white_style_reality_check_pvalue": reality_pvalue,
        },
        "champion": champion,
        "development_metrics": base_metrics,
        "stress_2x_metrics": stress_metrics,
        "stress_3x_metrics": stress_3x_metrics,
        "development_gate_passed": accepted,
        "development_verdict": (
            "ELIGIBLE_FOR_FINAL_HOLDOUT" if accepted else "NO_DEMONSTRABLE_EDGE"
        ),
        "gate_failures": failures,
        "event_driven_replay": replay,
        "holdout": {"status": "sealed", "opened": False},
        "accepted": False,
        "risk_policy": {
            "risk_per_trade": "0.01",
            "leverage": "10",
            "model_controls_size": False,
            "automatic_promotion": False,
        },
        "evaluations": evaluations[: config.full_candidates],
    }
    _atomic_json(config.report_path, payload)
    write_ml_status(
        config,
        "development_complete",
        "Development complete; final holdout remains sealed",
        100,
        run_id=run_id,
        backend=gpu["backend"],
        elapsed_seconds=round(time.monotonic() - started),
        gate_passed=accepted,
    )
    return payload


def _evaluate_frozen_candidate(
    app: AppConfig,
    raw: pd.DataFrame,
    candidate: StrategyCandidate,
    artifact: dict[str, Any],
) -> tuple[list[float], list[float], list[float], dict[str, Any], set[int]]:
    resampled = resample_observed(raw, candidate.timeframe_minutes, float(app.instrument.tick_size))
    data = _candidate_frame(resampled, app, candidate)
    combined_events: list[tuple[int, int, float, bool]] = []
    stress_events: list[tuple[int, int, float, bool]] = []
    stress_3x_events: list[tuple[int, int, float, bool]] = []
    events_by_side: dict[str, list[tuple[int, int, float, bool]]] = {}
    for side in ("long", "short"):
        data[f"eligible_{side}"] = _entry_mask(data, candidate, side)
        eligible = data.loc[data[f"eligible_{side}"]].copy()
        model = (
            artifact["models"].get(side) if artifact.get("selected_mode") == "meta_label" else None
        )
        metadata = artifact["model_data"].get(side, {})
        if model is None:
            selected_events = _candidate_events(data, candidate, side)
            stressed_events = _candidate_events(data, candidate, side, cost_multiplier=2)
            stressed_3x_events = _candidate_events(data, candidate, side, cost_multiplier=3)
        else:
            probability = model.predict_proba(eligible.loc[:, FEATURE_COLUMNS])[:, 1]
            win, loss = metadata["payoff_priors"]
            threshold = float(metadata["parameters"]["threshold"])
            selected_events = _candidate_events(
                data,
                candidate,
                side,
                probability,
                threshold=threshold,
                expected_win=float(win),
                expected_loss=float(loss),
            )
            stressed_events = _candidate_events(
                data,
                candidate,
                side,
                probability,
                threshold=threshold,
                expected_win=float(win),
                expected_loss=float(loss),
                cost_multiplier=2,
            )
            stressed_3x_events = _candidate_events(
                data,
                candidate,
                side,
                probability,
                threshold=threshold,
                expected_win=float(win),
                expected_loss=float(loss),
                cost_multiplier=3,
            )
        combined_events.extend(selected_events)
        stress_events.extend(stressed_events)
        stress_3x_events.extend(stressed_3x_events)
        events_by_side[side] = selected_events
    executed = _non_overlapping_events(combined_events, candidate.cooldown_bars)
    combined = [event[2] for event in executed]
    stress = _non_overlapping_returns(stress_events, candidate.cooldown_bars)
    stress_3x = _non_overlapping_returns(stress_3x_events, candidate.cooldown_bars)
    sides = {
        side: _metrics([event[2] for event in executed if event in set(side_events)])
        for side, side_events in events_by_side.items()
    }
    return combined, stress, stress_3x, sides, {event[0] for event in executed}


def finalize_scientific_research(
    app: AppConfig,
    one_minute: pd.DataFrame,
    run_id: str,
    external_control: pd.DataFrame | None = None,
) -> dict[str, Any]:
    config = _scientific_config(app)
    report = cast(dict[str, Any], json.loads(config.report_path.read_text(encoding="utf-8")))
    if report.get("run_id") != run_id or report.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("run-id does not identify the current scientific development report")
    if not report.get("development_gate_passed"):
        raise RuntimeError(
            "development found no demonstrable edge; sealed holdout will not be opened"
        )
    lock = config.candidate_directory / run_id / "holdout.opened.json"
    if lock.exists():
        raise RuntimeError("this holdout has already been opened for the run")
    _atomic_json(lock, {"run_id": run_id, "opened_at": datetime.now(UTC).isoformat()})
    candidate = StrategyCandidate(**report["champion"]["parameters"])
    artifact = cast(dict[str, Any], joblib.load(report["champion"]["artifact"]))
    start = pd.Timestamp(report["data"]["holdout_start"])
    end = pd.Timestamp(report["data"]["holdout_end"])
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True)
    raw = one_minute.loc[timestamps.ge(start) & timestamps.lt(end)].copy()
    combined, stress, stress_3x, sides, event_indexes = _evaluate_frozen_candidate(
        app, raw, candidate, artifact
    )
    metrics = _metrics(combined)
    stress_metrics = _metrics(stress)
    accepted, failures = _gate(
        metrics,
        stress_metrics,
        returns=combined,
        config=config,
        pbo=None,
        dsr=0,
        side_metrics=sides,
        holdout=True,
    )
    replay_frame = resample_observed(
        raw, candidate.timeframe_minutes, float(app.instrument.tick_size)
    )
    replay_data = _candidate_frame(replay_frame, app, candidate)
    replay = _event_driven_replay(
        app,
        replay_frame,
        candidate,
        {
            (pd.Timestamp(replay_data.iloc[index]["timestamp"]).to_pydatetime(), side)
            for index in event_indexes
            for side in ("long", "short")
            if bool(_entry_mask(replay_data, candidate, side).iloc[index])
        },
    )
    if not replay["passed"]:
        accepted = False
        failures.append("event_driven_replay_failed")
    report["holdout"] = {
        "status": "opened",
        "opened": True,
        "opened_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "stress_2x_metrics": stress_metrics,
        "stress_3x_metrics": _metrics(stress_3x),
        "sides": sides,
        "gate_passed": accepted,
        "gate_failures": failures,
        "event_driven_replay": replay,
    }
    if external_control is not None:
        external_timestamps = pd.to_datetime(external_control["timestamp"], utc=True)
        external_raw = external_control.loc[
            external_timestamps.ge(start) & external_timestamps.lt(end)
        ].copy()
        eth_returns, eth_stress, _, eth_sides, _ = _evaluate_frozen_candidate(
            app, external_raw, candidate, artifact
        )
        eth_metrics = _metrics(eth_returns)
        external_veto = bool(
            eth_metrics["trades"] >= 100
            and _bootstrap_lower([-value for value in eth_returns], config.random_seed) > 0
        )
        report["external_control"] = {
            "symbol": "ETHUSDT",
            "trained_on": False,
            "metrics": eth_metrics,
            "stress_2x_metrics": _metrics(eth_stress),
            "sides": eth_sides,
            "veto": external_veto,
            "status": "failed" if external_veto else "diagnostic_pass_or_inconclusive",
        }
        if external_veto:
            report["holdout"]["gate_failures"].append("eth_external_control_negative")
            report["holdout"]["gate_passed"] = False
            accepted = False
    report["accepted"] = bool(report["development_gate_passed"] and accepted)
    report["verdict"] = "ACCEPTED" if report["accepted"] else "REJECTED"
    report["mode"] = "research_finalized"
    _atomic_json(config.report_path, report)
    write_ml_status(
        config,
        "complete",
        "Final holdout evaluated; no automatic promotion",
        100,
        run_id=run_id,
        backend=report["compute"]["backend"],
        gate_passed=report["accepted"],
    )
    return report


def _scientific_config(app: AppConfig) -> MachineLearningConfig:
    config = app.machine_learning
    if config is None or not config.enabled:
        raise ValueError("machine-learning research is disabled")
    if config.protocol_version != PROTOCOL_VERSION:
        raise ValueError(f"expected machine-learning protocol {PROTOCOL_VERSION}")
    if config.full_candidates > config.strategy_candidates:
        raise ValueError("full_candidates cannot exceed strategy_candidates")
    if config.meta_candidates > config.full_candidates:
        raise ValueError("meta_candidates cannot exceed full_candidates")
    return config
