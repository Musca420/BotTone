from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import random
import statistics
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import combinations
from math import exp, log, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd

from adaptive_bot.backtest.engine import BacktestEngine, BacktestResult
from adaptive_bot.backtest.walk_forward import WalkForwardWindow, walk_forward_windows
from adaptive_bot.config import AppConfig
from adaptive_bot.domain.enums import Side

if TYPE_CHECKING:
    from adaptive_bot.backtest.gpu_features import GpuFeatureFactory

FIXED_PARAMETERS = ("risk_per_trade", "leverage", "timeframe_minutes", "instrument")
_WORKER_CONFIG: AppConfig | None = None
_WORKER_FRAME: pd.DataFrame | None = None
_WORKER_WINDOWS: tuple[WalkForwardWindow, ...] = ()


def research_status_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.status.json")


def write_research_status(
    config: AppConfig,
    phase: str,
    completed: int = 0,
    total: int = 0,
    detail: str = "",
) -> None:
    research = config.research
    if research is None:
        return
    target = research_status_path(research.report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "completed": completed,
        "total": total,
        "percent": round(completed / total * 100, 1) if total else 0.0,
        "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for _ in range(5):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            time.sleep(0.02)
    # Progress telemetry must never stop downloads or simulations.
    temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]


def _candidate(family: str, parameters: dict[str, Any]) -> Candidate:
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
    identifier = hashlib.sha256(f"{family}:{canonical}".encode()).hexdigest()[:16]
    return Candidate(identifier, family, parameters)


def core_candidates() -> tuple[Candidate, ...]:
    return tuple(
        _candidate(
            "v11_core",
            {
                "entry_mode": "weighted_reversion_v11",
                "entry_z": entry_z,
                "weighted_entry_threshold": threshold,
                "time_stop_bars": time_stop,
            },
        )
        for entry_z in (1.25, 1.5, 1.75, 2.0)
        for threshold in (0.55, 0.60, 0.65, 0.70, 0.75)
        for time_stop in (6, 8, 12)
    )


def broad_candidates(count: int, seed: int) -> tuple[Candidate, ...]:
    spaces: dict[str, tuple[Any, ...]] = {
        "crypto_vwap_window": (144, 288, 576),
        "entry_z": (1.25, 1.5, 1.75, 2.0),
        "weighted_entry_threshold": (0.55, 0.60, 0.65, 0.70, 0.75),
        "v11_exit_z": (0.25, 0.5, 0.75),
        "v11_cooldown_bars": (3, 6, 9),
        "time_stop_bars": (6, 8, 12),
        "fixed_stop_fraction": ("0.0075", "0.01", "0.0125"),
        "fixed_target_fraction": ("0.0075", "0.01", "0.0125"),
        "confirmation_bars": (2, 3, 4),
    }
    rng = random.Random(seed)
    selected: dict[str, Candidate] = {}
    while len(selected) < count:
        values = {name: rng.choice(options) for name, options in spaces.items()}
        values["entry_mode"] = "weighted_reversion_v11"
        values["fixed_target_fraction"] = values["fixed_stop_fraction"]
        item = _candidate("v11_broad", values)
        selected[item.candidate_id] = item
    return tuple(selected[key] for key in sorted(selected))


def _trade_pnls(result: BacktestResult) -> tuple[Decimal, ...]:
    position = Decimal("0")
    cash = Decimal("0")
    trades: list[Decimal] = []
    for fill in result.fills:
        quantity = fill.quantity if fill.side is Side.BUY else -fill.quantity
        cash -= quantity * fill.price
        cash -= fill.commission
        position += quantity
        if position == 0:
            trades.append(cash)
            cash = Decimal("0")
    return tuple(trades)


def _sharpe(returns: Iterable[float]) -> float:
    values = tuple(returns)
    if len(values) < 2:
        return 0.0
    deviation = statistics.stdev(values)
    return statistics.mean(values) / deviation if deviation else 0.0


def deflated_sharpe_probability(returns: Iterable[float], trial_sharpes: Iterable[float]) -> float:
    """Probability that Sharpe exceeds the expected best result from all trials."""
    values = tuple(returns)
    sharpes = tuple(trial_sharpes)
    if len(values) < 3 or len(sharpes) < 2:
        return 0.0
    observed = _sharpe(values)
    trial_sigma = statistics.stdev(sharpes)
    trials = len(sharpes)
    euler_gamma = 0.5772156649015329
    normal = NormalDist()
    expected_max = statistics.mean(sharpes) + trial_sigma * (
        (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
        + euler_gamma * normal.inv_cdf(1 - 1 / (trials * exp(1)))
    )
    mean = statistics.mean(values)
    sigma = statistics.stdev(values)
    centered = tuple((value - mean) / sigma for value in values)
    skew = statistics.mean(value**3 for value in centered)
    kurtosis = statistics.mean(value**4 for value in centered)
    denominator = 1 - skew * observed + ((kurtosis - 1) / 4) * observed**2
    if denominator <= 0:
        return 0.0
    statistic = (observed - expected_max) * sqrt(len(values) - 1) / sqrt(denominator)
    return normal.cdf(statistic)


def combinatorial_pbo(blocks_by_candidate: list[list[list[float]]]) -> dict[str, Any]:
    """Estimate PBO with combinatorially symmetric cross-validation."""
    if len(blocks_by_candidate) < 2:
        return {"pbo": None, "splits": 0, "blocks": 0}
    available = min(len(blocks) for blocks in blocks_by_candidate)
    block_count = min(10, available - available % 2)
    if block_count < 4:
        return {"pbo": None, "splits": 0, "blocks": block_count}
    indexes = tuple(range(block_count))
    splits = tuple(combinations(indexes, block_count // 2))
    counts = np.zeros((len(blocks_by_candidate), block_count), dtype=np.int64)
    sums = np.zeros_like(counts, dtype=np.float64)
    sums_of_squares = np.zeros_like(counts, dtype=np.float64)
    for candidate_index, blocks in enumerate(blocks_by_candidate):
        for block_index, block in enumerate(blocks[:block_count]):
            values = np.asarray(block, dtype=np.float64)
            counts[candidate_index, block_index] = values.size
            sums[candidate_index, block_index] = values.sum()
            sums_of_squares[candidate_index, block_index] = np.square(values).sum()

    def sharpes(selected: tuple[int, ...]) -> np.ndarray:
        selected_counts = counts[:, selected].sum(axis=1)
        selected_sums = sums[:, selected].sum(axis=1)
        selected_squares = sums_of_squares[:, selected].sum(axis=1)
        means = np.divide(
            selected_sums,
            selected_counts,
            out=np.zeros_like(selected_sums),
            where=selected_counts > 0,
        )
        variance = np.divide(
            selected_squares
            - np.divide(
                np.square(selected_sums),
                selected_counts,
                out=np.zeros_like(selected_sums),
                where=selected_counts > 0,
            ),
            selected_counts - 1,
            out=np.zeros_like(selected_sums),
            where=selected_counts > 1,
        )
        deviation = np.sqrt(np.maximum(variance, 0.0))
        return np.divide(means, deviation, out=np.zeros_like(means), where=deviation > 0)

    overfit = 0
    logits: list[float] = []
    for in_sample in splits:
        in_set = set(in_sample)
        out_sample = tuple(index for index in indexes if index not in in_set)
        in_scores = sharpes(in_sample)
        winner = int(np.argmax(in_scores))
        out_scores = sharpes(out_sample)
        selected = out_scores[winner]
        percentile = (
            np.count_nonzero(out_scores < selected) + 0.5 * np.count_nonzero(out_scores == selected)
        ) / out_scores.size
        clipped = min(max(percentile, 1e-9), 1 - 1e-9)
        logits.append(log(clipped / (1 - clipped)))
        overfit += int(percentile <= 0.5)
    return {
        "pbo": overfit / len(splits),
        "splits": len(splits),
        "blocks": block_count,
        "median_logit": statistics.median(logits),
    }


def _metrics(results: Iterable[BacktestResult]) -> dict[str, Any]:
    reports = tuple(results)
    trades_by_window = tuple(_trade_pnls(result) for result in reports)
    trades = tuple(pnl for window in trades_by_window for pnl in window)
    wins = sum((value for value in trades if value > 0), Decimal("0"))
    losses = abs(sum((value for value in trades if value < 0), Decimal("0")))
    net = sum((result.net_pnl for result in reports), Decimal("0"))
    costs = sum((result.fees + result.slippage for result in reports), Decimal("0"))
    extra_cost = costs / len(trades) if trades else Decimal("0")
    cost_2x_trades = tuple(value - extra_cost for value in trades)
    wins_2x = sum((value for value in cost_2x_trades if value > 0), Decimal("0"))
    losses_2x = abs(sum((value for value in cost_2x_trades if value < 0), Decimal("0")))
    window_expectancies = [
        sum(window, Decimal("0")) / len(window) for window in trades_by_window if window
    ]
    initial_equity = reports[0].initial_equity if reports else Decimal("1")
    block_returns = [
        [float(value / initial_equity) for value in window] for window in trades_by_window
    ]
    returns = [value for window in block_returns for value in window]
    return {
        "windows": len(reports),
        "positive_windows": sum(result.net_pnl > 0 for result in reports),
        "positive_window_rate": (
            sum(result.net_pnl > 0 for result in reports) / len(reports) if reports else 0.0
        ),
        "net_pnl": str(net),
        "expectancy": str(sum(trades, Decimal("0")) / len(trades) if trades else Decimal("0")),
        "median_oos_expectancy": str(
            statistics.median(window_expectancies) if window_expectancies else Decimal("0")
        ),
        "profit_factor": str(
            wins / losses if losses else (Decimal("999") if wins else Decimal("0"))
        ),
        "profit_factor_cost_2x": str(
            wins_2x / losses_2x if losses_2x else (Decimal("999") if wins_2x else Decimal("0"))
        ),
        "net_pnl_cost_2x": str(net - costs),
        "max_drawdown": str(max((result.max_drawdown for result in reports), default=Decimal("0"))),
        "trades": len(trades),
        "safety_violations": sum(result.kill_switches for result in reports),
        "liquidations": 0,
        "risk_budget_violations": 0,
        "duplicate_orders": 0,
        "orphan_positions": 0,
        "sharpe_per_trade": _sharpe(returns),
        "_block_returns": block_returns,
        "_returns": returns,
    }


def _status(metrics: dict[str, Any], stable: bool, pbo: float | None) -> str:
    trades = int(metrics["trades"])
    if trades < 30:
        return "insufficient"
    gates = (
        Decimal(metrics["expectancy"]) > 0,
        Decimal(metrics["profit_factor"]) >= Decimal("1.15"),
        Decimal(metrics["max_drawdown"]) <= Decimal("0.10"),
        float(metrics["positive_window_rate"]) > 0.5,
        Decimal(metrics["net_pnl_cost_2x"]) >= 0,
        int(metrics["safety_violations"]) == 0,
        int(metrics["liquidations"]) == 0,
        int(metrics["risk_budget_violations"]) == 0,
        int(metrics["duplicate_orders"]) == 0,
        int(metrics["orphan_positions"]) == 0,
        float(metrics["deflated_sharpe_probability"]) >= 0.95,
        pbo is not None and pbo <= 0.20,
        stable,
    )
    return "validated" if trades >= 300 and all(gates) else "provisional"


def _rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    return (
        -float(metrics["deflated_sharpe_probability"]),
        -float(metrics["positive_window_rate"]),
        -float(metrics["median_oos_expectancy"]),
        -float(metrics["profit_factor_cost_2x"]),
        float(metrics["max_drawdown"]),
        -int(metrics["trades"]),
        item["candidate_id"],
    )


def _stable(candidate: dict[str, Any], evaluations: list[dict[str, Any]]) -> bool:
    others = [item for item in evaluations if item is not candidate]
    nearest = sorted(
        others,
        key=lambda item: (
            sum(
                candidate["parameters"].get(key) != item["parameters"].get(key)
                for key in set(candidate["parameters"]) | set(item["parameters"])
            ),
            item["candidate_id"],
        ),
    )[:5]
    return (
        len(nearest) >= 2
        and sum(Decimal(item["metrics"]["expectancy"]) > 0 for item in nearest) >= 3
    )


class ResearchRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.path)) as database:
            database.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status VARCHAR
                )"""
            )
            database.execute(
                """CREATE TABLE IF NOT EXISTS evaluations (
                    run_id VARCHAR,
                    candidate_id VARCHAR,
                    rank INTEGER,
                    status VARCHAR,
                    parameters JSON,
                    metrics JSON,
                    PRIMARY KEY(run_id, candidate_id)
                )"""
            )
            database.execute(
                """CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id VARCHAR PRIMARY KEY,
                    family VARCHAR,
                    parameters JSON
                )"""
            )
            database.execute(
                """CREATE TABLE IF NOT EXISTS champions (
                    candidate_id VARCHAR PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    pinned BOOLEAN,
                    parameters JSON,
                    metrics JSON
                )"""
            )

    def save(self, payload: dict[str, Any]) -> None:
        self.initialize()
        with duckdb.connect(str(self.path)) as database:
            database.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET
                    completed_at=excluded.completed_at,
                    status=excluded.status""",
                [payload["run_id"], payload["started_at"], payload["completed_at"], "complete"],
            )
            for item in payload["evaluations"]:
                database.execute(
                    """INSERT INTO candidates VALUES (?, ?, ?)
                    ON CONFLICT(candidate_id) DO UPDATE SET
                        family=excluded.family,
                        parameters=excluded.parameters""",
                    [
                        item["candidate_id"],
                        item["family"],
                        json.dumps(item["parameters"]),
                    ],
                )
                database.execute(
                    "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    [
                        payload["run_id"],
                        item["candidate_id"],
                        item["rank"],
                        item["status"],
                        json.dumps(item["parameters"]),
                        json.dumps(item["metrics"]),
                    ],
                )
                if item["status"] == "validated":
                    database.execute(
                        """INSERT INTO champions VALUES (?, ?, ?, false, ?, ?)
                        ON CONFLICT(candidate_id) DO UPDATE SET
                            last_seen=excluded.last_seen,
                            parameters=excluded.parameters,
                            metrics=excluded.metrics""",
                        [
                            item["candidate_id"],
                            payload["completed_at"],
                            payload["completed_at"],
                            json.dumps(item["parameters"]),
                            json.dumps(item["metrics"]),
                        ],
                    )

    def pin(self, candidate_id: str, pinned: bool = True) -> bool:
        self.initialize()
        with duckdb.connect(str(self.path)) as database:
            result = database.execute(
                "UPDATE champions SET pinned=? WHERE candidate_id=? RETURNING candidate_id",
                [pinned, candidate_id],
            ).fetchone()
        return result is not None

    def champions(self) -> list[dict[str, Any]]:
        self.initialize()
        with duckdb.connect(str(self.path)) as database:
            rows = database.execute(
                """SELECT candidate_id, first_seen, last_seen, pinned, parameters, metrics
                FROM champions ORDER BY pinned DESC, last_seen DESC"""
            ).fetchall()
        return [
            {
                "candidate_id": row[0],
                "first_seen": str(row[1]),
                "last_seen": str(row[2]),
                "pinned": row[3],
                "parameters": json.loads(row[4]),
                "metrics": json.loads(row[5]),
            }
            for row in rows
        ]


async def _add_horizons(
    config: AppConfig,
    frame: pd.DataFrame,
    evaluations: list[dict[str, Any]],
    feature_factory: GpuFeatureFactory | None = None,
) -> None:
    research = config.research
    assert research is not None
    shadow_evaluations = evaluations[: research.top_shadow_candidates]
    shadow_ids = {item["candidate_id"] for item in shadow_evaluations}
    shadow_evaluations.extend(
        item
        for item in evaluations
        if (item["status"] == "validated" or item["family"] == "champion")
        and item["candidate_id"] not in shadow_ids
    )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    data_end = timestamps.max()
    warm_frame = frame[timestamps >= data_end - pd.Timedelta(days=32)]
    for index, item in enumerate(shadow_evaluations, 1):
        write_research_status(
            config,
            "shadow",
            index - 1,
            len(shadow_evaluations),
            f"Candidate {item['candidate_id']}",
        )
        strategy = type(config.strategy).model_validate(
            {**config.strategy.model_dump(), **item["parameters"]}
        )
        candidate_config = config.model_copy(update={"strategy": strategy})
        candidate_features = (
            feature_factory.build(strategy) if feature_factory is not None else None
        )
        item["metrics"].setdefault("horizons", {})["all"] = {
            key: value
            for key, value in item["metrics"].items()
            if key != "horizons" and not key.startswith("_")
        }
        for label, days in (("1d", 1), ("7d", 7), ("30d", 30)):
            result = await BacktestEngine(candidate_config).run(
                warm_frame,
                trade_after=(data_end - pd.Timedelta(days=days)).to_pydatetime(),
                mode="research-shadow",
                precomputed_features=(
                    candidate_features.loc[warm_frame.index]
                    if candidate_features is not None
                    else None
                ),
            )
            horizon_metrics = _metrics((result,))
            item["metrics"]["horizons"][label] = {
                key: value for key, value in horizon_metrics.items() if not key.startswith("_")
            }
    write_research_status(
        config, "shadow", len(shadow_evaluations), len(shadow_evaluations), "Updated"
    )


async def refresh_research_shadow(
    config: AppConfig, frame: pd.DataFrame, payload: dict[str, Any]
) -> dict[str, Any]:
    research = config.research
    if research is None or not research.enabled:
        raise ValueError("research mode is disabled")
    if research.gpu_enabled:
        from adaptive_bot.backtest.gpu_features import GpuFeatureFactory

        feature_factory = GpuFeatureFactory(frame)
    else:
        feature_factory = None
    await _add_horizons(config, frame, payload["evaluations"], feature_factory)
    payload["shadow_updated_at"] = datetime.now(UTC).isoformat()
    temporary = research.report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(research.report_path)
    return payload


def _initialize_worker(
    config: AppConfig, frame: pd.DataFrame, windows: tuple[WalkForwardWindow, ...]
) -> None:
    global _WORKER_CONFIG, _WORKER_FRAME, _WORKER_WINDOWS
    _WORKER_CONFIG = config
    _WORKER_FRAME = frame
    _WORKER_WINDOWS = windows


def _evaluate_in_worker(
    candidate: Candidate, feature_values: pd.DataFrame | None
) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_FRAME is None:
        raise RuntimeError("research worker was not initialized")
    return asyncio.run(
        _evaluate_candidate(
            _WORKER_CONFIG,
            _WORKER_FRAME,
            _WORKER_WINDOWS,
            candidate,
            feature_values,
        )
    )


async def _evaluate_candidate(
    config: AppConfig,
    frame: pd.DataFrame,
    windows: tuple[WalkForwardWindow, ...],
    candidate: Candidate,
    candidate_features: pd.DataFrame | None,
) -> dict[str, Any]:
    strategy = type(config.strategy).model_validate(
        {**config.strategy.model_dump(), **candidate.parameters}
    )
    candidate_config = config.model_copy(update={"strategy": strategy})
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    reports: list[BacktestResult] = []
    if windows:
        warmup_bars = (
            max(
                strategy.crypto_vwap_window,
                strategy.ema_period + strategy.slope_lookback,
                strategy.atr_percentile_window + strategy.atr_period,
                strategy.adx_period * 2,
            )
            * 3
        )
        warmup = pd.Timedelta(minutes=warmup_bars * strategy.timeframe_minutes)
        for window in windows:
            sample = frame[
                (timestamps >= window.validation_end - warmup) & (timestamps < window.test_end)
            ]
            reports.append(
                await BacktestEngine(candidate_config).run(
                    sample,
                    trade_after=window.validation_end.to_pydatetime(),
                    mode="research",
                    precomputed_features=(
                        candidate_features.loc[sample.index]
                        if candidate_features is not None
                        else None
                    ),
                )
            )
    else:
        reports.append(
            await BacktestEngine(candidate_config).run(
                frame,
                mode="research",
                precomputed_features=candidate_features,
            )
        )
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "parameters": candidate.parameters,
        "metrics": _metrics(reports),
    }


def _feature_values(
    feature_factory: GpuFeatureFactory | None,
    config: AppConfig,
    candidate: Candidate,
) -> pd.DataFrame | None:
    if feature_factory is None:
        return None
    strategy = type(config.strategy).model_validate(
        {**config.strategy.model_dump(), **candidate.parameters}
    )
    return feature_factory.build(strategy)


def _evaluate_parallel(
    config: AppConfig,
    frame: pd.DataFrame,
    windows: tuple[WalkForwardWindow, ...],
    candidates: list[Candidate],
    feature_factory: GpuFeatureFactory | None,
    workers: int,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(config, frame, windows),
    ) as executor:
        candidate_iterator = iter(candidates)
        pending: dict[Any, Candidate] = {}

        def submit_next() -> bool:
            try:
                candidate = next(candidate_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _evaluate_in_worker,
                candidate,
                _feature_values(feature_factory, config, candidate),
            )
            pending[future] = candidate
            return True

        for _ in range(min(len(candidates), workers * 2)):
            submit_next()
        while pending:
            completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                candidate = pending.pop(future)
                evaluations.append(future.result())
                write_research_status(
                    config,
                    "evaluating",
                    len(evaluations),
                    len(candidates),
                    f"Candidate {candidate.candidate_id} | {workers} CPU workers + CUDA",
                )
                submit_next()
    return evaluations


async def run_research(
    config: AppConfig,
    frame: pd.DataFrame,
    *,
    include_broad: bool = True,
    candidate_limit: int | None = None,
) -> dict[str, Any]:
    research = config.research
    if research is None or not research.enabled:
        raise ValueError("research mode is disabled")
    if config.risk.risk_per_trade != Decimal("0.01"):
        raise ValueError("research requires fixed risk_per_trade=1%")
    if config.bitunix is None or config.bitunix.leverage != Decimal("10"):
        raise ValueError("research requires fixed Bitunix leverage=10x")
    registry = ResearchRegistry(research.database_path)
    candidates = list(core_candidates())
    if include_broad:
        candidates.extend(broad_candidates(research.broad_candidates, research.random_seed))
    candidates.extend(
        Candidate(champion["candidate_id"], "champion", champion["parameters"])
        for champion in registry.champions()
        if champion["parameters"].get("entry_mode") == "weighted_reversion_v11"
    )
    candidates = list({item.candidate_id: item for item in candidates}.values())
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    windows = walk_forward_windows(frame)
    started = datetime.now(UTC)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if research.gpu_enabled:
        from adaptive_bot.backtest.gpu_features import GpuFeatureFactory

        feature_factory = GpuFeatureFactory(frame)
    else:
        feature_factory = None
    workers = min(research.parallel_workers, len(candidates))
    backend = "CUDA" if feature_factory is not None else "CPU"
    write_research_status(
        config,
        "evaluating",
        0,
        len(candidates),
        f"Starting | {workers} CPU workers + {backend}",
    )
    if workers > 1:
        evaluations = await asyncio.to_thread(
            _evaluate_parallel,
            config,
            frame,
            windows,
            candidates,
            feature_factory,
            workers,
        )
    else:
        evaluations = []
        for index, candidate in enumerate(candidates, 1):
            evaluations.append(
                await _evaluate_candidate(
                    config,
                    frame,
                    windows,
                    candidate,
                    _feature_values(feature_factory, config, candidate),
                )
            )
            write_research_status(
                config,
                "evaluating",
                index,
                len(candidates),
                f"Candidate {candidate.candidate_id} | {backend}",
            )
    trial_sharpes = [float(item["metrics"]["sharpe_per_trade"]) for item in evaluations]
    selection_bias = combinatorial_pbo([item["metrics"]["_block_returns"] for item in evaluations])
    pbo = selection_bias["pbo"]
    for item in evaluations:
        stable = _stable(item, evaluations)
        item["metrics"]["neighborhood_stable"] = stable
        item["metrics"]["deflated_sharpe_probability"] = deflated_sharpe_probability(
            item["metrics"]["_returns"], trial_sharpes
        )
        item["status"] = _status(item["metrics"], stable, pbo)
    evaluations.sort(key=_rank_key)
    for rank, item in enumerate(evaluations, 1):
        item["rank"] = rank
    data_end = timestamps.max()
    await _add_horizons(config, frame, evaluations, feature_factory)
    for item in evaluations:
        item["metrics"].pop("_block_returns", None)
        item["metrics"].pop("_returns", None)
    completed = datetime.now(UTC)
    payload = {
        "schema_version": 2,
        "run_id": uuid.uuid4().hex,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "data_start": timestamps.min().isoformat(),
        "data_end": data_end.isoformat(),
        "walk_forward": [asdict(window) for window in windows],
        "selection_bias": {
            **selection_bias,
            "trials": len(evaluations),
            "dsr_gate": 0.95,
            "pbo_gate": 0.20,
            "recent_horizons": "diagnostic only; excluded from ranking and validation",
        },
        "fixed": {
            "risk_per_trade": "0.01",
            "leverage": "10",
            "timeframe_minutes": config.strategy.timeframe_minutes,
            "instrument": config.instrument.symbol,
            "feature_backend": "cuda" if research.gpu_enabled else "cpu",
        },
        "excluded": {
            "safety_limits": "not optimizable",
            "risk_and_leverage": "fixed by policy",
            "llm": "not used for trading decisions",
        },
        "evaluations": evaluations,
    }
    registry.save(payload)
    payload["champions"] = registry.champions()
    research.report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = research.report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(research.report_path)
    write_research_status(config, "complete", len(evaluations), len(evaluations), "Ready")
    return payload
