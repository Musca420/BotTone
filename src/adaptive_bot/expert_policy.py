from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import duckdb
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional GPU dependency
    XGBRegressor = None  # type: ignore[misc,assignment]

from adaptive_bot.backtest.engine import BacktestEngine
from adaptive_bot.config import AppConfig, MachineLearningConfig
from adaptive_bot.domain.enums import MarketRegime, SignalAction
from adaptive_bot.domain.enums import Side as OrderSide
from adaptive_bot.domain.models import Position, Signal, StrategyState
from adaptive_bot.ml_research import FEATURE_COLUMNS, build_ml_features, write_ml_status
from adaptive_bot.research import combinatorial_pbo, deflated_sharpe_probability
from adaptive_bot.scientific_ml import (
    BASE_EXTRA_COST_BPS,
    StrategyCandidate,
    _entry_mask,
    _gate,
    _metrics,
    _scientific_config,
    gpu_preflight,
    reality_check_pvalue,
    resample_observed,
    scientific_archive_path,
)
from adaptive_bot.strategy.signals import MarketSnapshot, initial_stop

try:
    import cupy as cp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None

Side = Literal["long", "short"]
Verdict = Literal["ELIGIBLE_FOR_FINAL_HOLDOUT", "NO_DEPLOYABLE_POLICY"]
PROTOCOL_VERSION = "adaptive_range_multi_expert_v5"
PARAMETER_COLUMNS = (
    "timeframe_minutes",
    "vwap_hours",
    "entry_z",
    "entry_rule_code",
    "stop_atr",
    "exit_z",
    "time_stop_hours",
    "range_adx_threshold",
    "regime_policy_code",
    "confirmation_bars",
)
TIME_FEATURES = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "regime_code",
    "exchange_code",
)

_MINUTE_BARRIER_KERNEL = r"""
extern "C" __global__ void minute_barriers(
    const double* open, const double* high, const double* low, const double* close,
    const double* mark_high, const double* mark_low, const signed char* valid,
    const double* funding, const long long* entry_idx, const double* center,
    const double* atr, const double* total_cost, long long* exit_idx,
    double* exit_price, double* net_r, double* mae_r, double* mfe_r,
    double* funding_paid, signed char* reason, int events, int minutes,
    int holding_minutes, double stop_atr, double target_z,
    double liquidation_distance, int is_long
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= events) return;
    long long entry_bar = entry_idx[i];
    if (entry_bar < 0 || entry_bar >= minutes || !valid[entry_bar]
        || isnan(open[entry_bar]) || isnan(center[i]) || isnan(atr[i]) || atr[i] <= 0) return;
    double entry = open[entry_bar];
    double risk_distance = stop_atr * atr[i];
    if (entry <= 0 || risk_distance <= 0) return;
    double stop = is_long ? entry - risk_distance : entry + risk_distance;
    double target = is_long ? center[i] - target_z * atr[i]
                            : center[i] + target_z * atr[i];
    // A gap that has already crossed the intended exit invalidates the entry.
    if ((is_long && target <= entry) || (!is_long && target >= entry)) return;
    double liquidation = is_long ? entry * (1.0 - liquidation_distance)
                                 : entry * (1.0 + liquidation_distance);
    double adverse = 0.0;
    double favorable = 0.0;
    double funding_sum = 0.0;
    long long last = min((long long)minutes - 1, entry_bar + holding_minutes - 1);
    for (long long j = entry_bar; j <= last; ++j) {
        if (!valid[j] || isnan(open[j]) || isnan(high[j]) || isnan(low[j])
            || isnan(close[j]) || isnan(mark_high[j]) || isnan(mark_low[j])) return;
        funding_sum += isnan(funding[j]) ? 0.0 : funding[j];
        if (is_long) {
            adverse = min(adverse, low[j] - entry);
            favorable = max(favorable, high[j] - entry);
        } else {
            adverse = min(adverse, entry - high[j]);
            favorable = max(favorable, entry - low[j]);
        }
        bool liquidated = is_long ? mark_low[j] <= liquidation : mark_high[j] >= liquidation;
        if (liquidated) { reason[i] = 4; return; }
        bool hit_stop = is_long ? low[j] <= stop : high[j] >= stop;
        bool hit_target = is_long ? high[j] >= target : low[j] <= target;
        if (hit_stop || hit_target) {
            // Stop wins an ambiguous minute. A gap through a stop fills at the worse open.
            bool won = hit_target && !hit_stop;
            double price = won ? target : (is_long ? min(stop, open[j]) : max(stop, open[j]));
            double gross = is_long ? (price - entry) / entry : (entry - price) / entry;
            double funding_return = is_long ? -funding_sum : funding_sum;
            exit_idx[i] = j;
            exit_price[i] = price;
            net_r[i] = (gross + funding_return - total_cost[i]) / (risk_distance / entry);
            mae_r[i] = adverse / risk_distance;
            mfe_r[i] = favorable / risk_distance;
            funding_paid[i] = funding_sum;
            reason[i] = won ? 1 : 2;
            return;
        }
    }
    if (last < entry_bar + holding_minutes - 1) return;
    double price = close[last];
    double gross = is_long ? (price - entry) / entry : (entry - price) / entry;
    double funding_return = is_long ? -funding_sum : funding_sum;
    exit_idx[i] = last;
    exit_price[i] = price;
    net_r[i] = (gross + funding_return - total_cost[i]) / (risk_distance / entry);
    mae_r[i] = adverse / risk_distance;
    mfe_r[i] = favorable / risk_distance;
    funding_paid[i] = funding_sum;
    reason[i] = 3;
}
"""
_minute_barrier_kernel: Any | None = None
_counter_app: AppConfig | None = None
_counter_raw: pd.DataFrame | None = None
_counter_frames: dict[int, pd.DataFrame] = {}
_counter_market: dict[str, Any] | None = None
_counter_root: Path | None = None


@dataclass(frozen=True)
class Expert:
    expert_id: str
    side: Side
    timeframe_minutes: int
    vwap_hours: int
    entry_z: float
    entry_rule: Literal["touch", "exhaustion", "confirmed_reentry"]
    stop_atr: float
    exit_z: float
    time_stop_hours: int
    atr_period: int = 14
    adx_period: int = 14
    cooldown_minutes: int = 60
    range_adx_threshold: float = 20
    regime_policy: Literal["range_only", "block_with_trend", "any_nonshock"] = "any_nonshock"
    confirmation_bars: int = 1
    source: str = "preregistered_v5"

    def candidate(self) -> StrategyCandidate:
        return StrategyCandidate(
            candidate_id=self.expert_id,
            timeframe_minutes=self.timeframe_minutes,
            vwap_hours=self.vwap_hours,
            atr_period=self.atr_period,
            adx_period=self.adx_period,
            entry_z_long=self.entry_z,
            entry_z_short=self.entry_z,
            range_adx_threshold=self.range_adx_threshold,
            regime_policy=self.regime_policy,
            entry_rule=self.entry_rule,
            confirmation_bars=self.confirmation_bars,
            stop_atr=self.stop_atr,
            exit_z=self.exit_z,
            time_stop_hours=self.time_stop_hours,
            cooldown_bars=max(0, self.cooldown_minutes // self.timeframe_minutes),
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: Literal["LONG", "SHORT", "FLAT"]
    expert_id: str | None
    ev_mean: float
    lower_confidence_bound: float
    reason: str


class ScheduledExpertReplayStrategy:
    """Causal OOS decision schedule executed by the shared engine and risk controls."""

    def __init__(self, trades: pd.DataFrame) -> None:
        self.schedule = {
            pd.Timestamp(row["signal_timestamp"]) - pd.Timedelta(minutes=1): row
            for _, row in trades.iterrows()
        }
        self.active_exit: pd.Timestamp | None = None
        self.entries_attempted = 0

    def evaluate(
        self, snapshot: MarketSnapshot, state: StrategyState, position: Position | None
    ) -> Signal | None:
        del state
        now = pd.Timestamp(snapshot.candle.exchange_timestamp)
        if position is not None:
            if not snapshot.data_reliable:
                return self._signal(snapshot, SignalAction.EXIT, "data integrity compromised")
            if snapshot.regime is MarketRegime.SHOCK:
                return self._signal(snapshot, SignalAction.EXIT, "shock regime")
            if self.active_exit is not None and now >= self.active_exit:
                return self._signal(snapshot, SignalAction.EXIT, "expert time stop")
            return None
        row = self.schedule.get(now)
        if row is None:
            return None
        if (
            not snapshot.data_reliable
            or not bool(row["data_valid"])
            or str(row.get("regime", "UNKNOWN")) in {"UNKNOWN", "SHOCK"}
        ):
            return None
        side = str(row["side"])
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        action = SignalAction.ENTER_LONG if side == "long" else SignalAction.ENTER_SHORT
        stop = initial_stop(
            snapshot.candle.close,
            Decimal(str(row["atr"])),
            order_side,
            Decimal(str(row["stop_atr"])),
        )
        self.active_exit = now + pd.Timedelta(hours=int(row["time_stop_hours"]))
        self.entries_attempted += 1
        return self._signal(
            snapshot,
            action,
            f"expert policy {row['expert_id']}",
            stop=stop,
            target=Decimal(str(row["target_price"])),
        )

    def entry_score(self, snapshot: MarketSnapshot, state: StrategyState) -> None:
        del snapshot, state
        return None

    @staticmethod
    def _signal(
        snapshot: MarketSnapshot,
        action: SignalAction,
        reason: str,
        *,
        stop: Decimal | None = None,
        target: Decimal | None = None,
    ) -> Signal:
        candle = snapshot.candle
        return Signal(
            exchange_timestamp=candle.exchange_timestamp,
            received_timestamp=candle.received_timestamp,
            source="expert_policy_replay",
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


class FrozenExpertPolicy:
    """Holdout-gated inference wrapper; execution and sizing remain external."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, bundle_path: Path) -> FrozenExpertPolicy:
        holdout_path = bundle_path.with_name("final_holdout_report.json")
        if not holdout_path.exists():
            raise RuntimeError("final holdout report is missing; policy remains research-only")
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        if holdout.get("status") != "passed_manual_review_required":
            raise RuntimeError("final holdout did not pass")
        if holdout.get("bundle_sha256") != _file_sha256(bundle_path):
            raise RuntimeError("frozen bundle hash does not match the holdout report")
        bundle: dict[str, Any] = joblib.load(bundle_path)
        if bundle.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError("unsupported expert policy protocol")
        return cls(bundle)

    def decide(
        self,
        actions: pd.DataFrame,
        *,
        data_valid: bool,
        risk_approved: bool,
        position_open: bool,
    ) -> PolicyDecision:
        if position_open:
            return PolicyDecision("FLAT", None, 0, 0, "position_already_open")
        required = {"expert_id", "side", "regime_code"}
        if missing := required - set(actions):
            raise ValueError(f"runtime actions missing columns: {sorted(missing)}")
        allowed = {expert.expert_id for expert in self.bundle["experts"]}
        candidates = actions.loc[
            actions["expert_id"].isin(allowed) & actions["regime_code"].isin((1, 2, 3))
        ].copy()
        predicted: list[pd.DataFrame] = []
        for side in ("long", "short"):
            fitted = self.bundle["models"].get(side, {})
            rows = candidates.loc[candidates["side"].eq(side)].copy()
            if not fitted.get("enabled") or rows.empty:
                continue
            if missing := set(fitted["features"]) - set(rows):
                raise ValueError(f"runtime action features missing: {sorted(missing)}")
            x = (
                rows[fitted["features"]]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .to_numpy(dtype=float)
            )
            ensemble = np.vstack([_xgb_predict(model, x) for model in fitted["models"]])
            rows["ev_mean"] = fitted["calibrator"].predict(ensemble.mean(axis=0))
            rows["ensemble_std"] = ensemble.std(axis=0)
            rows["residual_lower"] = fitted["residual_lower"]
            predicted.append(rows)
        if not predicted:
            return PolicyDecision("FLAT", None, 0, 0, "no_valid_expert_actions")
        scored = pd.concat(predicted, ignore_index=True)
        z = NormalDist().inv_cdf(1 - 0.05 / len(scored))
        scored["lower_confidence_bound"] = (
            scored["ev_mean"] + scored["residual_lower"] - z * scored["ensemble_std"]
        )
        return conservative_decision(scored, data_valid=data_valid, risk_approved=risk_approved)


@dataclass(frozen=True)
class ExpertFold:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class BiasCalibrator:
    bias: float
    lower: float
    upper: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float) + self.bias, self.lower, self.upper)


def _bias_calibrator(raw: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> BiasCalibrator:
    bias = float(np.average(truth - raw, weights=weights))
    lower, upper = np.quantile(truth, [0.01, 0.99])
    return BiasCalibrator(bias, float(lower), float(upper))


def _expert(values: tuple[Any, ...], side: Side, source: str = "preregistered_v5") -> Expert:
    payload = {
        "side": side,
        "timeframe_minutes": values[0],
        "vwap_hours": values[1],
        "entry_z": values[2],
        "entry_rule": values[3],
        "stop_atr": values[4],
        "exit_z": values[5],
        "time_stop_hours": values[6],
        "source": source,
    }
    identifier = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return Expert(expert_id=f"{side[0]}-{identifier}", **payload)


def generate_preregistered_experts() -> tuple[Expert, ...]:
    """Return the frozen 3,072-action extension, balanced by side."""
    combinations = product(
        (5, 15, 30, 60),
        (8, 24, 48, 96),
        (2.5, 3.0, 3.5, 4.0),
        ("exhaustion", "confirmed_reentry"),
        (1.0, 1.5, 2.0, 3.0),
        (1.0, 0.5, 0.0),
    )
    actions: list[Expert] = []
    for index, values in enumerate(combinations):
        long_stop, short_stop = (8, 24) if index % 2 == 0 else (24, 8)
        actions.extend(
            (_expert((*values, long_stop), "long"), _expert((*values, short_stop), "short"))
        )
    return tuple(actions)


def expand_candidates(candidates: list[StrategyCandidate]) -> tuple[Expert, ...]:
    actions: list[Expert] = []
    for candidate in candidates:
        for side in ("long", "short"):
            values = (
                candidate.timeframe_minutes,
                candidate.vwap_hours,
                candidate.entry_z_long if side == "long" else candidate.entry_z_short,
                candidate.entry_rule,
                candidate.stop_atr,
                candidate.exit_z,
                candidate.time_stop_hours,
            )
            actions.append(
                replace(
                    _expert(values, side, f"existing:{candidate.candidate_id}"),
                    atr_period=candidate.atr_period,
                    adx_period=candidate.adx_period,
                    cooldown_minutes=candidate.cooldown_bars * candidate.timeframe_minutes,
                    range_adx_threshold=candidate.range_adx_threshold,
                    regime_policy=candidate.regime_policy,
                    confirmation_bars=candidate.confirmation_bars,
                )
            )
    unique = {expert.expert_id: expert for expert in actions}
    return tuple(unique[key] for key in sorted(unique))


def load_existing_experts(path: Path) -> tuple[Expert, ...]:
    """Load only completed candidates from the immutable v2 screen checkpoint."""
    candidates: list[StrategyCandidate] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if float(row.get("score", -1e9)) <= -1e9:
                continue
            candidates.append(StrategyCandidate(**row["parameters"]))
    return expand_candidates(candidates)


def latest_candidate_screen(root: Path = Path("data/research/mlv2")) -> Path:
    paths = list(root.glob("*/strategy_screen.jsonl"))
    if not paths:
        raise FileNotFoundError("no completed scientific candidate screen is available")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def universe_hash(experts: tuple[Expert, ...] | list[Expert]) -> str:
    payload = [asdict(item) for item in sorted(experts, key=lambda item: item.expert_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def timestamp_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("signal_timestamp")["expert_id"].transform("count")
    return np.asarray(np.divide(1.0, counts.to_numpy(dtype=float)), dtype=float)


def temporal_block_bootstrap_indices(
    frame: pd.DataFrame, *, seed: int, block_days: int = 7
) -> np.ndarray:
    if frame.empty or block_days <= 0:
        raise ValueError("a non-empty frame and positive block size are required")
    timestamps = pd.to_datetime(frame["signal_timestamp"], utc=True)
    block = ((timestamps - timestamps.min()) / pd.Timedelta(days=block_days)).astype(int)
    groups = tuple(np.flatnonzero(block.to_numpy() == value) for value in sorted(block.unique()))
    randomizer = np.random.default_rng(seed)
    chosen = randomizer.choice(len(groups), size=len(groups), replace=True)
    return np.concatenate([groups[index] for index in chosen])


def _parameter_distance(left: Expert, right: Expert) -> float:
    values = (
        (left.timeframe_minutes, right.timeframe_minutes),
        (left.vwap_hours, right.vwap_hours),
        (left.entry_z, right.entry_z),
        (left.entry_rule, right.entry_rule),
        (left.stop_atr, right.stop_atr),
        (left.exit_z, right.exit_z),
        (left.time_stop_hours, right.time_stop_hours),
        (left.range_adx_threshold, right.range_adx_threshold),
        (left.regime_policy, right.regime_policy),
        (left.confirmation_bars, right.confirmation_bars),
    )
    return sum(first != second for first, second in values) / len(values)


def _greedy_expert_selection(
    scores: dict[str, float],
    returns: pd.DataFrame,
    lookup: dict[str, Expert],
    *,
    maximum: int,
    maximum_per_side: int,
    maximum_per_timeframe: int,
    maximum_correlation: float,
) -> tuple[Expert, ...]:
    remaining = set(scores) & set(lookup)
    selected: list[Expert] = []
    while remaining and len(selected) < maximum:
        ranked = sorted(
            remaining,
            key=lambda identifier: (
                -scores[identifier]
                - 0.001
                * (
                    min(_parameter_distance(lookup[identifier], item) for item in selected)
                    if selected
                    else 1.0
                ),
                identifier,
            ),
        )
        chosen: Expert | None = None
        for identifier in ranked:
            expert = lookup[identifier]
            if sum(item.side == expert.side for item in selected) >= maximum_per_side:
                remaining.remove(identifier)
                continue
            if (
                sum(item.timeframe_minutes == expert.timeframe_minutes for item in selected)
                >= maximum_per_timeframe
            ):
                remaining.remove(identifier)
                continue
            correlated = any(
                abs(float(returns[identifier].corr(returns[item.expert_id]))) >= maximum_correlation
                for item in selected
                if identifier in returns and item.expert_id in returns
            )
            if correlated:
                remaining.remove(identifier)
                continue
            chosen = expert
            break
        if chosen is None:
            break
        selected.append(chosen)
        remaining.remove(chosen.expert_id)
    return tuple(selected)


def purged_expert_folds(
    frame: pd.DataFrame,
    *,
    train_weeks: int = 52,
    calibration_weeks: int = 4,
    test_weeks: int = 4,
    step_weeks: int = 4,
    embargo_hours: int = 24,
) -> tuple[ExpertFold, ...]:
    """Time folds purged by each trade's real exit and embargoed by max holding."""
    timestamps = pd.to_datetime(frame["signal_timestamp"], utc=True)
    exits = pd.to_datetime(frame["exit_timestamp"], utc=True)
    first = timestamps.min()
    last = timestamps.max()
    cursor = first + pd.Timedelta(weeks=train_weeks)
    folds: list[ExpertFold] = []
    while cursor + pd.Timedelta(weeks=calibration_weeks + test_weeks) <= last:
        train_start = cursor - pd.Timedelta(weeks=train_weeks)
        calibration_end = cursor + pd.Timedelta(weeks=calibration_weeks)
        test_end = calibration_end + pd.Timedelta(weeks=test_weeks)
        train = np.flatnonzero(
            timestamps.ge(train_start)
            & timestamps.lt(cursor - pd.Timedelta(hours=embargo_hours))
            & exits.lt(cursor)
        )
        calibration = np.flatnonzero(
            timestamps.ge(cursor)
            & timestamps.lt(calibration_end - pd.Timedelta(hours=embargo_hours))
            & exits.lt(calibration_end)
        )
        test = np.flatnonzero(timestamps.ge(calibration_end) & timestamps.lt(test_end))
        if len(train) and len(calibration) and len(test):
            folds.append(ExpertFold(train, calibration, test))
        cursor += pd.Timedelta(weeks=step_weeks)
    return tuple(folds)


def select_expert_library(
    training: pd.DataFrame,
    experts: tuple[Expert, ...] | list[Expert],
    *,
    minimum_opportunities: int = 300,
    maximum: int = 24,
    maximum_per_side: int = 12,
    maximum_per_timeframe: int = 4,
    maximum_correlation: float = 0.90,
) -> tuple[Expert, ...]:
    """Greedy train-only selection with score, monthly stability and return diversity."""
    required = {"signal_timestamp", "expert_id", "net_return_r"}
    if missing := required - set(training):
        raise ValueError(f"training matrix missing columns: {sorted(missing)}")
    lookup = {expert.expert_id: expert for expert in experts}
    data = training.loc[training["expert_id"].isin(lookup)].copy()
    data["signal_timestamp"] = pd.to_datetime(data["signal_timestamp"], utc=True)
    data["month"] = data["signal_timestamp"].dt.strftime("%Y-%m")
    candidates: list[tuple[float, str]] = []
    for identifier, rows in data.groupby("expert_id", sort=True):
        if len(rows) < minimum_opportunities:
            continue
        monthly = rows.groupby("month")["net_return_r"].mean()
        robust = float(monthly.mean() - monthly.std(ddof=0) / np.sqrt(max(1, len(monthly))))
        candidates.append((robust + float((monthly > 0).mean()) * 0.05, str(identifier)))
    returns = data.pivot_table(
        index="signal_timestamp", columns="expert_id", values="net_return_r", aggfunc="first"
    )
    return _greedy_expert_selection(
        {identifier: score for score, identifier in candidates},
        returns,
        lookup,
        maximum=maximum,
        maximum_per_side=maximum_per_side,
        maximum_per_timeframe=maximum_per_timeframe,
        maximum_correlation=maximum_correlation,
    )


def _matrix_connection(root: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    pattern = (root / "**" / "part-*.parquet").as_posix().replace("'", "''")
    connection.execute(
        f"CREATE VIEW matrix AS SELECT * FROM read_parquet('{pattern}', hive_partitioning=false)"
    )
    return connection


def _validate_matrix(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    row = connection.execute(
        """
        SELECT count(*) AS rows,
               count(*) - count(DISTINCT (signal_timestamp, expert_id)) AS duplicates,
               count(*) FILTER (
                 WHERE execution_timestamp < signal_timestamp
                    OR exit_timestamp <= execution_timestamp
                    OR NOT data_valid
                    OR NOT isfinite(net_return_r)
                    OR NOT isfinite(net_return_r_2x)
                    OR NOT isfinite(net_return_r_3x)
                    OR net_return_r_2x > net_return_r
                    OR net_return_r_3x > net_return_r_2x
                    OR regime IN ('UNKNOWN', 'SHOCK')
                    OR price_resolution <> 'observed_1m'
                    OR NOT mark_liquidation_checked
               ) AS invalid
        FROM matrix
        """
    ).fetchone()
    if row is None:
        raise ValueError("counterfactual matrix validation failed")
    report = {"rows": int(row[0]), "duplicates": int(row[1]), "invalid": int(row[2])}
    if report["duplicates"] or report["invalid"]:
        raise ValueError(f"counterfactual matrix integrity failure: {report}")
    return report


def _query_experts(
    connection: duckdb.DuckDBPyConnection,
    identifiers: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exits_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    selected = pd.DataFrame({"expert_id": sorted(identifiers)})
    connection.register("selected_ids", selected)
    exit_clause = "AND exit_timestamp < ?" if exits_before is not None else ""
    parameters: list[Any] = [start.to_pydatetime(), end.to_pydatetime()]
    if exits_before is not None:
        parameters.append(exits_before.to_pydatetime())
    return connection.execute(
        "SELECT m.* FROM matrix m JOIN selected_ids s USING (expert_id) "
        "WHERE signal_timestamp >= ? AND signal_timestamp < ? " + exit_clause,
        parameters,
    ).fetchdf()


def _query_return_panel(
    connection: duckdb.DuckDBPyConnection,
    identifiers: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exits_before: pd.Timestamp,
) -> pd.DataFrame:
    selected = pd.DataFrame({"expert_id": sorted(identifiers)})
    connection.register("selected_return_ids", selected)
    return connection.execute(
        """
        SELECT signal_timestamp, expert_id, net_return_r
        FROM matrix m JOIN selected_return_ids s USING (expert_id)
        WHERE signal_timestamp >= ? AND signal_timestamp < ? AND exit_timestamp < ?
        """,
        [start.to_pydatetime(), end.to_pydatetime(), exits_before.to_pydatetime()],
    ).fetchdf()


def select_expert_library_duckdb(
    connection: duckdb.DuckDBPyConnection,
    experts: tuple[Expert, ...] | list[Expert],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exits_before: pd.Timestamp,
) -> tuple[Expert, ...]:
    scores = connection.execute(
        """
        WITH base AS (
          SELECT expert_id, signal_timestamp, net_return_r
          FROM matrix
          WHERE signal_timestamp >= ? AND signal_timestamp < ? AND exit_timestamp < ?
        ), opportunities AS (
          SELECT expert_id, count(*) AS opportunities
          FROM base GROUP BY expert_id HAVING count(*) >= 300
        ), monthly AS (
          SELECT expert_id, date_trunc('month', signal_timestamp) AS month,
                 avg(net_return_r) AS expectancy
          FROM base GROUP BY expert_id, month
        ), stability AS (
          SELECT expert_id,
                 avg(expectancy) - coalesce(stddev_pop(expectancy), 0) / sqrt(count(*)) AS robust,
                 avg(CASE WHEN expectancy > 0 THEN 1.0 ELSE 0.0 END) AS positive
          FROM monthly GROUP BY expert_id
        )
        SELECT opportunities.expert_id, robust + 0.05 * positive AS score
        FROM opportunities JOIN stability USING (expert_id)
        ORDER BY score DESC, expert_id LIMIT 500
        """,
        [start.to_pydatetime(), end.to_pydatetime(), exits_before.to_pydatetime()],
    ).fetchdf()
    lookup = {expert.expert_id: expert for expert in experts}
    candidate_ids = [identifier for identifier in scores["expert_id"] if identifier in lookup]
    if not candidate_ids:
        return ()
    returns = _query_return_panel(
        connection, set(candidate_ids), start, end, exits_before=exits_before
    ).pivot_table(
        index="signal_timestamp", columns="expert_id", values="net_return_r", aggfunc="first"
    )
    score_lookup = {str(row["expert_id"]): float(row["score"]) for _, row in scores.iterrows()}
    return _greedy_expert_selection(
        score_lookup,
        returns,
        lookup,
        maximum=24,
        maximum_per_side=12,
        maximum_per_timeframe=4,
        maximum_correlation=0.90,
    )


def _candidate_features(
    one_minute: pd.DataFrame, app: AppConfig, candidate: StrategyCandidate
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
    complete = one_minute[one_minute["funding_rate"].notna()].copy()
    complete["funding_source"] = "observed"
    features, _ = build_ml_features(complete, configured)
    return features


def _minute_market_arrays(one_minute: pd.DataFrame) -> dict[str, Any]:
    if cp is None:
        raise RuntimeError("CuPy is required for causal one-minute expert outcomes")
    required = {"open", "high", "low", "close", "mark_high", "mark_low"}
    if missing := required - set(one_minute):
        raise ValueError(f"one-minute market data missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True, errors="raise")
    contiguous = timestamps.diff().eq(pd.Timedelta(minutes=1)) | timestamps.diff().isna()
    valid = one_minute.get("data_valid", pd.Series(True, index=one_minute.index)).astype(bool)
    funding = one_minute.get("funding_event_rate", pd.Series(0.0, index=one_minute.index))
    arrays = {
        name: cp.asarray(pd.to_numeric(one_minute[name], errors="coerce").to_numpy(dtype=float))
        for name in required
    }
    arrays["valid"] = cp.asarray((valid & contiguous).to_numpy(dtype=np.int8))
    arrays["funding"] = cp.asarray(pd.to_numeric(funding, errors="coerce").to_numpy(dtype=float))
    return arrays


def _minute_barrier_outcomes(
    one_minute: pd.DataFrame,
    market: dict[str, Any],
    rows: pd.DataFrame,
    expert: Expert,
    app: AppConfig,
) -> pd.DataFrame:
    if cp is None:
        raise RuntimeError("CuPy is required for causal one-minute expert outcomes")
    global _minute_barrier_kernel
    if _minute_barrier_kernel is None:
        _minute_barrier_kernel = cp.RawKernel(_MINUTE_BARRIER_KERNEL, "minute_barriers")
    raw_timestamps = pd.to_datetime(one_minute["timestamp"], utc=True, errors="raise")
    signal_times = pd.to_datetime(rows["timestamp"], utc=True) + pd.Timedelta(
        minutes=expert.timeframe_minutes
    )
    raw_ns = raw_timestamps.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    signal_ns = signal_times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    entry = np.searchsorted(raw_ns, signal_ns).astype(np.int64)
    exact = entry < len(raw_ns)
    exact[exact] &= raw_ns[entry[exact]] == signal_ns[exact]
    entry[~exact] = -1
    count = len(rows)
    device_entry = cp.asarray(entry)
    center = cp.asarray(rows["vwap"].to_numpy(dtype=float))
    atr_values = rows["atr"].to_numpy(dtype=float)
    atr = cp.asarray(atr_values)
    fee_bps = rows["round_trip_cost_bps"].to_numpy(dtype=float) + BASE_EXTRA_COST_BPS
    costs = cp.asarray(fee_bps / 10_000)
    exits = cp.full(count, -1, dtype=cp.int64)
    exit_price = cp.full(count, cp.nan, dtype=cp.float64)
    net = cp.full(count, cp.nan, dtype=cp.float64)
    mae = cp.full(count, cp.nan, dtype=cp.float64)
    mfe = cp.full(count, cp.nan, dtype=cp.float64)
    funding = cp.full(count, cp.nan, dtype=cp.float64)
    reason = cp.zeros(count, dtype=cp.int8)
    leverage = app.bitunix.leverage if app.bitunix is not None else app.instrument.max_leverage
    liquidation_distance = float(Decimal("1") / leverage - app.risk.liquidation_buffer_fraction)
    if liquidation_distance <= 0:
        raise ValueError("liquidation buffer leaves no positive liquidation distance")
    blocks = (count + 255) // 256
    _minute_barrier_kernel(
        (blocks,),
        (256,),
        (
            market["open"],
            market["high"],
            market["low"],
            market["close"],
            market["mark_high"],
            market["mark_low"],
            market["valid"],
            market["funding"],
            device_entry,
            center,
            atr,
            costs,
            exits,
            exit_price,
            net,
            mae,
            mfe,
            funding,
            reason,
            count,
            len(one_minute),
            expert.time_stop_hours * 60,
            expert.stop_atr,
            expert.exit_z,
            liquidation_distance,
            int(expert.side == "long"),
        ),
    )
    return pd.DataFrame(
        {
            "entry_index": entry,
            "exit_index": cp.asnumpy(exits),
            "exit_price": cp.asnumpy(exit_price),
            "net_return_r": cp.asnumpy(net),
            "mae_r": cp.asnumpy(mae),
            "mfe_r": cp.asnumpy(mfe),
            "funding": cp.asnumpy(funding),
            "reason_code": cp.asnumpy(reason),
            "total_cost_bps": fee_bps,
        },
        index=rows.index,
    )


def _regime_labels(rows: pd.DataFrame) -> np.ndarray:
    atr_change = rows["atr"].pct_change(fill_method=None)
    cumulative_move = (rows["close"] - rows["close"].shift(3)) / rows["atr"]
    shock = rows["atr_percentile"].gt(90) | atr_change.gt(0.5) | cumulative_move.abs().gt(3)
    trend_up = rows["adx"].gt(25) & rows["ema50_slope"].ge(0.05) & cumulative_move.ge(0)
    trend_down = rows["adx"].gt(25) & rows["ema50_slope"].le(-0.05) & cumulative_move.le(0)
    return np.select(
        [shock, trend_up, trend_down, rows["adx"].lt(20)],
        ["SHOCK", "TREND_UP", "TREND_DOWN", "RANGE"],
        default="UNKNOWN",
    )


def _tradable_regime(rows: pd.DataFrame) -> pd.Series:
    return rows["regime"].isin(("RANGE", "TREND_UP", "TREND_DOWN"))


def build_counterfactual_rows(
    app: AppConfig,
    one_minute: pd.DataFrame,
    experts: tuple[Expert, ...] | list[Expert],
    *,
    resampled_frame: pd.DataFrame | None = None,
    market_arrays: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build expert signals on their timeframe and resolve every outcome on observed 1m data."""
    feature_frames: dict[tuple[int, int, int, int], pd.DataFrame] = {}
    resampled: dict[int, pd.DataFrame] = {}
    output: list[pd.DataFrame] = []
    raw = one_minute.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    raw = (
        raw.drop_duplicates("timestamp", keep=False).sort_values("timestamp").reset_index(drop=True)
    )
    market = market_arrays or _minute_market_arrays(raw)
    for expert in experts:
        candles = resampled.get(expert.timeframe_minutes)
        if candles is None:
            candles = (
                resampled_frame
                if resampled_frame is not None
                else resample_observed(
                    raw, expert.timeframe_minutes, float(app.instrument.tick_size)
                )
            )
            resampled[expert.timeframe_minutes] = candles
        feature_key = (
            expert.timeframe_minutes,
            expert.vwap_hours,
            expert.atr_period,
            expert.adx_period,
        )
        features = feature_frames.get(feature_key)
        if features is None:
            features = _candidate_features(candles, app, expert.candidate())
            features["regime"] = _regime_labels(features)
            feature_frames[feature_key] = features
        candidate = expert.candidate()
        eligible = features.copy()
        for side in ("long", "short"):
            eligible[f"target_{side}"] = 1
            eligible[f"net_return_{side}"] = 1.0
        mask = _entry_mask(eligible, candidate, expert.side) & _tradable_regime(features)
        rows = features.loc[mask].copy()
        if rows.empty:
            continue
        outcomes = _minute_barrier_outcomes(raw, market, rows, expert, app)
        valid = outcomes["reason_code"].isin((1, 2, 3)) & outcomes["net_return_r"].notna()
        rows = rows.loc[valid].copy()
        outcomes = outcomes.loc[valid]
        if rows.empty:
            continue
        side = expert.side
        entry_indexes = outcomes["entry_index"].to_numpy(dtype=int)
        exit_indexes = outcomes["exit_index"].to_numpy(dtype=int)
        entries = raw.iloc[entry_indexes]["open"].to_numpy(dtype=float)
        atr_values = rows["atr"].to_numpy(dtype=float)
        risk = pd.Series(expert.stop_atr * atr_values / entries, index=rows.index)
        targets = (
            rows["vwap"].to_numpy(dtype=float) - expert.exit_z * atr_values
            if side == "long"
            else rows["vwap"].to_numpy(dtype=float) + expert.exit_z * atr_values
        )
        stops = (
            entries - expert.stop_atr * atr_values
            if side == "long"
            else entries + expert.stop_atr * atr_values
        )
        exit_array = outcomes["exit_price"].to_numpy(dtype=float)
        gross = (
            (exit_array - entries) / entries if side == "long" else (entries - exit_array) / entries
        )
        signal_timestamp = pd.to_datetime(rows["timestamp"], utc=True) + pd.Timedelta(
            minutes=expert.timeframe_minutes
        )
        execution_timestamp = raw.iloc[entry_indexes]["timestamp"].to_numpy()
        exit_timestamp = (
            pd.to_datetime(raw.iloc[exit_indexes]["timestamp"], utc=True) + pd.Timedelta(minutes=1)
        ).to_numpy()
        cost_r = (outcomes["total_cost_bps"].to_numpy(dtype=float) / 10_000) / risk.to_numpy(
            dtype=float
        )
        reason_names = np.asarray(["", "TARGET", "STOP", "TIME", "LIQUIDATION"])
        result = pd.DataFrame(
            {
                "signal_timestamp": signal_timestamp.to_numpy(),
                "execution_timestamp": execution_timestamp,
                "exit_timestamp": exit_timestamp,
                "expert_id": expert.expert_id,
                "side": side,
                "timeframe_minutes": expert.timeframe_minutes,
                "vwap_hours": expert.vwap_hours,
                "entry_z": expert.entry_z,
                "entry_rule_code": {
                    "touch": 0,
                    "exhaustion": 1,
                    "confirmed_reentry": 2,
                }[expert.entry_rule],
                "stop_atr": expert.stop_atr,
                "exit_z": expert.exit_z,
                "time_stop_hours": expert.time_stop_hours,
                "range_adx_threshold": expert.range_adx_threshold,
                "regime_policy_code": {
                    "range_only": 0,
                    "block_with_trend": 1,
                    "any_nonshock": 2,
                }[expert.regime_policy],
                "confirmation_bars": expert.confirmation_bars,
                "entry_price": entries,
                "exit_price": exit_array,
                "target_price": targets,
                "stop_price": stops,
                "exit_reason": reason_names[outcomes["reason_code"].to_numpy(dtype=int)],
                "gross_return_r": np.divide(gross, risk),
                "net_return_r": outcomes["net_return_r"].to_numpy(dtype=float),
                "net_return_r_2x": outcomes["net_return_r"].to_numpy(dtype=float) - cost_r,
                "net_return_r_3x": outcomes["net_return_r"].to_numpy(dtype=float) - 2 * cost_r,
                "funding": outcomes["funding"].to_numpy(dtype=float),
                "mae_r": outcomes["mae_r"].to_numpy(dtype=float),
                "mfe_r": outcomes["mfe_r"].to_numpy(dtype=float),
                "duration_minutes": (exit_timestamp - execution_timestamp) / np.timedelta64(1, "m"),
                "regime": rows["regime"].to_numpy(),
                "z": rows["distance_vwap_atr"].to_numpy(),
                "adx": rows["adx"].to_numpy(),
                "atr": rows["atr"].to_numpy(),
                "data_valid": True,
                "source": rows["market_data_source"].to_numpy(),
                "price_resolution": "observed_1m",
                "mark_liquidation_checked": True,
                "total_cost_bps": outcomes["total_cost_bps"].to_numpy(dtype=float),
            }
        )
        for name in FEATURE_COLUMNS:
            if name in rows:
                result[name] = rows[name].to_numpy()
        hours = pd.to_datetime(result["signal_timestamp"], utc=True).dt.hour
        weekdays = pd.to_datetime(result["signal_timestamp"], utc=True).dt.dayofweek
        result["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        result["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
        result["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
        result["regime_code"] = result["regime"].map(
            {"UNKNOWN": 0, "RANGE": 1, "TREND_UP": 2, "TREND_DOWN": 3, "SHOCK": 4}
        )
        output.append(result)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def write_counterfactual_partitions(frame: pd.DataFrame, root: Path) -> list[Path]:
    paths: list[Path] = []
    if frame.empty:
        return paths
    for (timeframe, side), rows in frame.groupby(["timeframe_minutes", "side"], sort=True):
        directory = root / f"timeframe={int(str(timeframe))}" / f"side={side}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "matrix.parquet"
        temporary = directory / "matrix.parquet.tmp"
        rows.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        paths.append(target)
    return paths


def _initialize_counter_worker(
    app: AppConfig, raw: pd.DataFrame, frames: dict[int, pd.DataFrame], root: Path
) -> None:
    global _counter_app, _counter_raw, _counter_frames, _counter_market, _counter_root
    global _minute_barrier_kernel
    _counter_app = app
    prepared = raw.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="raise")
    prepared = (
        prepared.drop_duplicates("timestamp", keep=False)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    _counter_raw = prepared
    _counter_frames = frames
    _counter_market = _minute_market_arrays(prepared)
    if cp is not None and _minute_barrier_kernel is None:
        _minute_barrier_kernel = cp.RawKernel(_MINUTE_BARRIER_KERNEL, "minute_barriers")
    _counter_root = root


def _counter_worker(item: tuple[tuple[Any, ...], list[Expert]]) -> list[str]:
    if (
        _counter_app is None
        or _counter_raw is None
        or _counter_market is None
        or _counter_root is None
    ):
        raise RuntimeError("counterfactual worker was not initialized")
    key, grouped = item
    fingerprint = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    block = build_counterfactual_rows(
        _counter_app,
        _counter_raw,
        grouped,
        resampled_frame=_counter_frames[grouped[0].timeframe_minutes],
        market_arrays=_counter_market,
    )
    if block.empty:
        return []
    written: list[str] = []
    for (timeframe, side), rows in block.groupby(["timeframe_minutes", "side"], sort=True):
        directory = _counter_root / f"timeframe={int(str(timeframe))}" / f"side={side}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"part-{fingerprint}.parquet"
        temporary = directory / f"part-{fingerprint}.{os.getpid()}.tmp"
        rows.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        written.append(str(target))
    return written


def build_counterfactual_checkpointed(
    app: AppConfig,
    one_minute: pd.DataFrame,
    experts: tuple[Expert, ...] | list[Expert],
    root: Path,
    config: MachineLearningConfig,
    *,
    resume: bool,
) -> list[Path]:
    groups: dict[tuple[Any, ...], list[Expert]] = {}
    for expert in experts:
        key = (
            expert.timeframe_minutes,
            expert.vwap_hours,
            expert.atr_period,
            expert.adx_period,
        )
        groups.setdefault(key, []).append(expert)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    resampled = {
        timeframe: resample_observed(one_minute, timeframe, float(app.instrument.tick_size))
        for timeframe in sorted({expert.timeframe_minutes for expert in experts})
    }
    paths: list[Path] = []
    pending: list[tuple[tuple[Any, ...], list[Expert]]] = []
    for key, grouped in ordered:
        fingerprint = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        targets = [
            root
            / f"timeframe={expert.timeframe_minutes}"
            / f"side={expert.side}"
            / f"part-{fingerprint}.parquet"
            for expert in grouped
        ]
        if resume and all(path.exists() for path in targets):
            paths.extend(targets)
            continue
        pending.append((key, grouped))
    already_done = len(ordered) - len(pending)
    if not pending:
        return sorted(set(paths))
    workers = min(8, config.parallel_workers, len(pending))
    started = time.monotonic()
    _initialize_counter_worker(app, one_minute, resampled, root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_counter_worker, item): item for item in pending}
        for newly_done, future in enumerate(as_completed(futures), start=1):
            paths.extend(Path(path) for path in future.result())
            _, grouped = futures[future]
            done = already_done + newly_done
            elapsed = time.monotonic() - started
            remaining = len(pending) - newly_done
            eta = int(elapsed / newly_done * remaining)
            write_ml_status(
                config,
                "counterfactual",
                f"Counterfactual block {done}/{len(ordered)}",
                5 + 25 * done / len(ordered),
                completed_chunks=done,
                total_chunks=len(ordered),
                eta_seconds=eta,
                active_workers=workers,
                current_candidate=",".join(expert.expert_id for expert in grouped[:3]),
                ram_mode=f"shared market arrays, {workers} threads",
                backend=f"cuda:0 shared by {workers} workers",
            )
    return sorted(set(paths))


def conservative_decision(
    actions: pd.DataFrame, *, data_valid: bool = True, risk_approved: bool = True
) -> PolicyDecision:
    required = {"expert_id", "side", "ev_mean", "lower_confidence_bound"}
    if missing := required - set(actions):
        raise ValueError(f"actions missing columns: {sorted(missing)}")
    if not data_valid:
        return PolicyDecision("FLAT", None, 0, 0, "invalid_market_data")
    if not risk_approved:
        return PolicyDecision("FLAT", None, 0, 0, "risk_rejected")
    eligible = actions.loc[
        actions["ev_mean"].gt(0) & actions["lower_confidence_bound"].gt(0)
    ].sort_values(
        ["lower_confidence_bound", "ev_mean", "expert_id"],
        ascending=[False, False, True],
    )
    if eligible.empty:
        return PolicyDecision("FLAT", None, 0, 0, "no_positive_conservative_ev")
    row = eligible.iloc[0]
    action: Literal["LONG", "SHORT"] = "LONG" if row["side"] == "long" else "SHORT"
    return PolicyDecision(
        action,
        str(row["expert_id"]),
        float(row["ev_mean"]),
        float(row["lower_confidence_bound"]),
        "highest_positive_conservative_ev",
    )


def moving_block_lower_bound(
    values: np.ndarray, *, block_size: int, seed: int, repetitions: int = 1000
) -> float:
    if len(values) == 0 or block_size <= 0:
        raise ValueError("values and a positive block size are required")
    randomizer = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_size + 1))
    means = []
    blocks = int(np.ceil(len(values) / block_size))
    for _ in range(repetitions):
        sample = np.concatenate(
            [values[start : start + block_size] for start in randomizer.choice(starts, blocks)]
        )[: len(values)]
        means.append(float(sample.mean()))
    return float(np.quantile(means, 0.05))


def superior_predictive_ability_pvalue(
    candidate_folds: list[list[list[float]]], *, seed: int, repetitions: int = 5000
) -> float:
    """Hansen-style studentized SPA bootstrap on chronological OOS fold means."""
    if not candidate_folds:
        return 1.0
    count = min(len(candidate) for candidate in candidate_folds)
    if count < 4:
        return 1.0
    differences = np.asarray(
        [
            [statistics.fmean(fold) if fold else 0.0 for fold in candidate[:count]]
            for candidate in candidate_folds
        ],
        dtype=float,
    )
    means = differences.mean(axis=1)
    deviations = differences.std(axis=1, ddof=1)
    usable = deviations > 0
    if not usable.any():
        return 1.0
    statistic = float(np.max(np.sqrt(count) * means[usable] / deviations[usable]))
    if statistic <= 0:
        return 1.0
    t_values = np.full(len(means), -np.inf)
    t_values[usable] = np.sqrt(count) * means[usable] / deviations[usable]
    threshold = -np.sqrt(2 * np.log(np.log(max(count, 3))))
    retained_mean = np.where(t_values < threshold, means, 0.0)
    generator = np.random.default_rng(seed)
    block_size = max(2, int(np.sqrt(count)))
    starts = np.arange(count)
    exceedances = 0
    for _ in range(repetitions):
        chosen = generator.choice(starts, int(np.ceil(count / block_size)))
        indexes = np.concatenate([(start + np.arange(block_size)) % count for start in chosen])[
            :count
        ]
        resampled = differences[:, indexes] - means[:, None] + retained_mean[:, None]
        null = float(np.max(np.sqrt(count) * resampled.mean(axis=1)[usable] / deviations[usable]))
        exceedances += int(null >= statistic)
    return (exceedances + 1) / (repetitions + 1)


def open_holdout_once(lock: Path, run_id: str) -> None:
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"run_id": run_id, "opened_at": datetime.now(UTC).isoformat()})
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError("this final holdout has already been opened") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)


def _regressor(
    params: dict[str, Any], seed: int, *, n_estimators: int = 2000, early_stopping: bool = True
) -> Any:
    if XGBRegressor is None:
        raise RuntimeError("install the gpu dependency group to train the expert policy")
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=n_estimators,
        early_stopping_rounds=50 if early_stopping else None,
        random_state=seed,
        n_jobs=1,
        **params,
    )


def _xgb_predict(model: Any, values: np.ndarray) -> np.ndarray:
    if cp is None or XGBRegressor is None or not isinstance(model, XGBRegressor):
        return np.asarray(model.predict(values))
    return np.asarray(cp.asnumpy(model.predict(cp.asarray(values))))


def _inner_splits(rows: pd.DataFrame, folds: int = 3) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    timestamps = pd.to_datetime(rows["signal_timestamp"], utc=True)
    exits = pd.to_datetime(rows["exit_timestamp"], utc=True)
    unique = np.asarray(sorted(timestamps.unique()))
    blocks = np.array_split(unique, folds + 1)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(1, len(blocks)):
        validation_start = pd.Timestamp(blocks[index][0])
        validation_end = pd.Timestamp(blocks[index][-1])
        train = np.flatnonzero(
            timestamps.lt(validation_start - pd.Timedelta(hours=24)) & exits.lt(validation_start)
        )
        validation = np.flatnonzero(timestamps.ge(validation_start) & timestamps.le(validation_end))
        if len(train) and len(validation):
            result.append((train, validation))
    return tuple(result)


def _decision_regret(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    *,
    actual_column: str = "net_return_r",
    prediction_cost_column: str | None = None,
    group_columns: tuple[str, ...] = ("signal_timestamp",),
) -> float:
    """Decision regret where FLAT is the neutral zero-return alternative."""
    if len(rows) != len(predictions) or rows.empty:
        return float("inf")
    actual = rows[actual_column].to_numpy(dtype=float)
    effective = predictions.copy()
    if prediction_cost_column is not None:
        effective -= rows[prediction_cost_column].to_numpy(dtype=float)
    regrets: list[float] = []
    for indexes in rows.groupby(list(group_columns), sort=False).indices.values():
        positions = np.asarray(indexes, dtype=int)
        group_prediction = effective[positions]
        chosen = int(positions[int(np.argmax(group_prediction))])
        chosen_value = actual[chosen] if effective[chosen] > 0 else 0.0
        regrets.append(max(0.0, float(actual[positions].max())) - chosen_value)
    return float(statistics.fmean(regrets))


def _fit_side(
    rows: pd.DataFrame,
    side: Side,
    config: MachineLearningConfig,
    *,
    progress: tuple[int, int] | None = None,
    objective_kind: Literal["decision_regret", "mse"] = "decision_regret",
    excluded_features: tuple[str, ...] = (),
    benchmark_gate: bool = False,
    ensemble_models: int = 5,
    temporal_bootstrap: bool = False,
    extra_features: tuple[str, ...] = (),
    decision_actual_column: str = "net_return_r",
    decision_cost_column: str | None = None,
    decision_group_columns: tuple[str, ...] = ("signal_timestamp",),
    calibration_kind: Literal["isotonic", "bias"] = "isotonic",
) -> dict[str, Any]:
    rows = (
        rows.loc[rows["side"].eq(side)]
        .sort_values(["signal_timestamp", "expert_id"])
        .reset_index(drop=True)
    )
    features = [
        name
        for name in (*FEATURE_COLUMNS, *extra_features, *TIME_FEATURES, *PARAMETER_COLUMNS)
        if name in rows and name not in excluded_features
    ]
    if len(rows) < 300 or not features:
        return {"side": side, "enabled": False, "reason": "insufficient_training_rows"}
    calibration_start = pd.to_datetime(rows["signal_timestamp"], utc=True).max() - pd.Timedelta(
        weeks=4
    )
    split = int(pd.to_datetime(rows["signal_timestamp"], utc=True).lt(calibration_start).sum())
    if split < 100 or len(rows) - split < 20:
        return {"side": side, "enabled": False, "reason": "insufficient_calibration_rows"}
    x = rows[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y = rows["net_return_r"].to_numpy(dtype=float)
    weights = (
        rows["_training_weight"].to_numpy(dtype=float)
        if "_training_weight" in rows
        else timestamp_balanced_weights(rows)
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10))
    ridge.fit(x[:split], y[:split], ridge__sample_weight=weights[:split])
    fitting = rows.iloc[:split]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 2, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True),
        }
        losses = []
        rounds = []
        for train, validation in _inner_splits(fitting):
            model = _regressor(params, config.random_seed + trial.number)
            model.fit(
                x[train],
                y[train],
                sample_weight=weights[train],
                eval_set=[(x[validation], y[validation])],
                sample_weight_eval_set=[weights[validation]],
                verbose=False,
            )
            prediction = _xgb_predict(model, x[validation])
            losses.append(
                float(np.average((prediction - y[validation]) ** 2, weights=weights[validation]))
                if objective_kind == "mse"
                else _decision_regret(
                    rows.iloc[validation],
                    prediction,
                    actual_column=decision_actual_column,
                    prediction_cost_column=decision_cost_column,
                    group_columns=decision_group_columns,
                )
            )
            rounds.append(int(model.best_iteration) + 1)
        trial.set_user_attr("n_estimators", int(statistics.median(rounds)) if rounds else 200)
        return statistics.fmean(losses) if losses else float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )

    def report_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress is None:
            return
        fold, total_folds = progress
        side_offset = 0 if side == "long" else 1
        fold_progress = (side_offset + (trial.number + 1) / config.model_trials_per_side) / 2
        fraction = ((fold - 1) + fold_progress) / total_folds
        write_ml_status(
            config,
            "outer_folds",
            f"Fold {fold}/{total_folds} {side.upper()} trial "
            f"{trial.number + 1}/{config.model_trials_per_side}",
            35 + 53 * fraction,
            outer_fold=fold,
            total_outer_folds=total_folds,
            side=side,
            trial=trial.number + 1,
            total_trials=config.model_trials_per_side,
            best_inner_decision_regret=study.best_value,
            backend="xgboost cuda:0",
        )

    study.optimize(
        objective,
        n_trials=config.model_trials_per_side,
        show_progress_bar=False,
        callbacks=[report_trial],
    )
    n_estimators = max(20, int(study.best_trial.user_attrs.get("n_estimators", 200)))
    models = []
    for seed in range(ensemble_models):
        model = _regressor(
            study.best_params,
            config.random_seed + seed,
            n_estimators=n_estimators,
            early_stopping=False,
        )
        indexes = (
            temporal_block_bootstrap_indices(rows.iloc[:split], seed=config.random_seed + seed)
            if temporal_bootstrap
            else np.arange(split)
        )
        model.fit(
            x[indexes],
            y[indexes],
            sample_weight=weights[indexes],
            verbose=False,
        )
        models.append(model)
    predictions = np.vstack([_xgb_predict(model, x[split:]) for model in models])
    raw = predictions.mean(axis=0)
    calibration_rows = rows.iloc[split:].copy()
    calibration_times = pd.to_datetime(calibration_rows["signal_timestamp"], utc=True)
    midpoint = calibration_times.min() + (calibration_times.max() - calibration_times.min()) / 2
    mapping = calibration_times.lt(midpoint).to_numpy()
    audit = ~mapping
    if mapping.sum() < 20 or audit.sum() < 20:
        return {"side": side, "enabled": False, "reason": "insufficient_honest_calibration_rows"}
    calibrator = (
        _bias_calibrator(raw[mapping], y[split:][mapping], weights[split:][mapping])
        if calibration_kind == "bias"
        else IsotonicRegression(out_of_bounds="clip").fit(
            raw[mapping], y[split:][mapping], sample_weight=weights[split:][mapping]
        )
    )
    ridge_raw = ridge.predict(x[split:])
    ridge_calibrator = (
        _bias_calibrator(ridge_raw[mapping], y[split:][mapping], weights[split:][mapping])
        if calibration_kind == "bias"
        else IsotonicRegression(out_of_bounds="clip").fit(
            ridge_raw[mapping], y[split:][mapping], sample_weight=weights[split:][mapping]
        )
    )
    calibrated_audit = calibrator.predict(raw[audit])
    ridge_calibrated_audit = ridge_calibrator.predict(ridge_raw[audit])
    residual_frame = pd.DataFrame(
        {
            "signal_timestamp": calibration_times.iloc[np.flatnonzero(audit)].to_numpy(),
            "residual": y[split:][audit] - calibrated_audit,
        }
    )
    residuals = residual_frame.groupby("signal_timestamp", sort=True)["residual"].mean().to_numpy()
    block_size = min(len(residuals), max(2, 24 * 60 // int(rows["timeframe_minutes"].min())))
    lower = moving_block_lower_bound(
        residuals,
        block_size=block_size,
        seed=config.random_seed,
    )
    ridge_residual_frame = pd.DataFrame(
        {
            "signal_timestamp": calibration_times.iloc[np.flatnonzero(audit)].to_numpy(),
            "residual": y[split:][audit] - ridge_calibrated_audit,
        }
    )
    ridge_residuals = (
        ridge_residual_frame.groupby("signal_timestamp", sort=True)["residual"].mean().to_numpy()
    )
    ridge_lower = moving_block_lower_bound(
        ridge_residuals,
        block_size=min(block_size, len(ridge_residuals)),
        seed=config.random_seed,
    )
    audit_rows = calibration_rows.iloc[np.flatnonzero(audit)]
    audit_weights = weights[split:][audit]
    audit_truth = y[split:][audit]
    xgb_mse = float(np.average((calibrated_audit - audit_truth) ** 2, weights=audit_weights))
    ridge_mse = float(
        np.average((ridge_calibrated_audit - audit_truth) ** 2, weights=audit_weights)
    )
    xgb_calibration_error = float(
        abs(np.average(calibrated_audit - audit_truth, weights=audit_weights))
    )
    ridge_calibration_error = float(
        abs(np.average(ridge_calibrated_audit - audit_truth, weights=audit_weights))
    )

    def audit_regret(prediction: np.ndarray) -> float:
        return _decision_regret(
            audit_rows,
            prediction,
            actual_column=decision_actual_column,
            prediction_cost_column=decision_cost_column,
            group_columns=decision_group_columns,
        )

    xgb_regret = audit_regret(calibrated_audit)
    ridge_regret = audit_regret(ridge_calibrated_audit)
    xgb_lcb_regret = audit_regret(calibrated_audit + lower)
    ridge_lcb_regret = audit_regret(ridge_calibrated_audit + ridge_lower)
    xgb_wins = (
        xgb_mse < ridge_mse
        and xgb_calibration_error <= ridge_calibration_error
        and xgb_regret < ridge_regret
        and xgb_lcb_regret <= ridge_lcb_regret
    )
    champion = "xgboost" if not benchmark_gate or xgb_wins else "ridge"
    selected_models = models if champion == "xgboost" else [ridge]
    selected_calibrator = calibrator if champion == "xgboost" else ridge_calibrator
    selected_lower = lower if champion == "xgboost" else ridge_lower
    return {
        "side": side,
        "enabled": True,
        "features": features,
        "models": selected_models,
        "champion": champion,
        "benchmark": ridge,
        "benchmark_calibrator": ridge_calibrator,
        "benchmark_residual_lower": ridge_lower,
        "calibrator": selected_calibrator,
        "residual_lower": selected_lower,
        "ensemble_dispersion": (
            float(predictions.std(axis=0).mean()) if champion == "xgboost" else 0.0
        ),
        "candidate_audit": {
            "xgboost": {
                "ev_mse": xgb_mse,
                "calibration_error": xgb_calibration_error,
                "decision_regret": xgb_regret,
                "lcb_decision_regret": xgb_lcb_regret,
            },
            "ridge": {
                "ev_mse": ridge_mse,
                "calibration_error": ridge_calibration_error,
                "decision_regret": ridge_regret,
                "lcb_decision_regret": ridge_lcb_regret,
            },
        },
        "optuna": {
            "trials": len(study.trials),
            "best_params": study.best_params,
            "best_decision_regret": study.best_value,
            "n_estimators": n_estimators,
        },
        "ridge_audit_decision_regret": ridge_regret,
        "calibration_rows": int(mapping.sum()),
        "uncertainty_audit_rows": int(audit.sum()),
    }


def _policy_returns(
    test: pd.DataFrame, models: dict[str, dict[str, Any]], *, benchmark: bool = False
) -> tuple[
    list[float],
    list[float],
    dict[str, list[float]],
    dict[str, list[float]],
    pd.DataFrame,
]:
    predicted: list[pd.DataFrame] = []
    for side in ("long", "short"):
        fitted = models[side]
        if not fitted.get("enabled"):
            continue
        rows = test.loc[test["side"].eq(side)].copy()
        if rows.empty:
            continue
        x = (
            rows[fitted["features"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=float)
        )
        if benchmark:
            raw = fitted["benchmark"].predict(x)
            rows["ev_mean"] = fitted["benchmark_calibrator"].predict(raw)
            rows["ensemble_std"] = 0.0
            rows["residual_lower"] = fitted["benchmark_residual_lower"]
        else:
            ensemble = np.vstack([_xgb_predict(model, x) for model in fitted["models"]])
            rows["ev_mean"] = fitted["calibrator"].predict(ensemble.mean(axis=0))
            rows["ensemble_std"] = ensemble.std(axis=0)
            rows["residual_lower"] = fitted["residual_lower"]
        predicted.append(rows)
    if not predicted:
        empty: dict[str, list[float]] = {"long": [], "short": []}
        return [], [], empty, {"long": [], "short": []}, pd.DataFrame()
    actions = pd.concat(predicted).sort_values("signal_timestamp")
    action_counts = actions.groupby("signal_timestamp")["expert_id"].transform("count")
    simultaneous_z = action_counts.map(
        lambda count: NormalDist().inv_cdf(1 - 0.05 / max(1, int(count)))
    )
    actions["lower_confidence_bound"] = (
        actions["ev_mean"] + actions["residual_lower"] - simultaneous_z * actions["ensemble_std"]
    )
    returns: list[float] = []
    stress: list[float] = []
    by_side: dict[str, list[float]] = {"long": [], "short": []}
    stress_by_side: dict[str, list[float]] = {"long": [], "short": []}
    chosen_rows: list[dict[str, Any]] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for timestamp, choices in actions.groupby("signal_timestamp", sort=True):
        signal_time = pd.Timestamp(str(timestamp))
        if signal_time <= blocked_until:
            continue
        decision = conservative_decision(choices)
        if decision.action == "FLAT" or decision.expert_id is None:
            continue
        row = choices.loc[choices["expert_id"].eq(decision.expert_id)].iloc[0]
        value = float(row["net_return_r"])
        returns.append(value)
        stressed = float(row["net_return_r_2x"])
        selected_side = str(row["side"])
        stress.append(stressed)
        by_side[selected_side].append(value)
        stress_by_side[selected_side].append(stressed)
        audit_columns = (
            "signal_timestamp",
            "execution_timestamp",
            "exit_timestamp",
            "expert_id",
            "side",
            "entry_price",
            "exit_price",
            "target_price",
            "stop_price",
            "stop_atr",
            "time_stop_hours",
            "atr",
            "regime",
            "data_valid",
            "exit_reason",
            "net_return_r",
            "net_return_r_2x",
            "mae_r",
            "mfe_r",
            "funding",
            "duration_minutes",
        )
        chosen: dict[str, Any] = {name: row[name] for name in audit_columns if name in row.index}
        chosen["decision_ev_mean"] = decision.ev_mean
        chosen["decision_lower_confidence_bound"] = decision.lower_confidence_bound
        chosen["alternatives"] = (
            choices.sort_values(
                ["lower_confidence_bound", "ev_mean", "expert_id"],
                ascending=[False, False, True],
            )[["expert_id", "side", "ev_mean", "lower_confidence_bound"]]
            .head(5)
            .to_dict("records")
        )
        chosen_rows.append(chosen)
        blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(hours=1)
    return returns, stress, by_side, stress_by_side, pd.DataFrame(chosen_rows)


def evaluate_oos_policy(
    matrix: pd.DataFrame,
    experts: tuple[Expert, ...] | list[Expert],
    config: MachineLearningConfig,
) -> dict[str, Any]:
    folds = purged_expert_folds(
        matrix,
        train_weeks=config.outer_train_weeks,
        calibration_weeks=config.outer_calibration_weeks,
        test_weeks=config.outer_test_weeks,
        step_weeks=config.outer_step_weeks,
    )
    fold_returns: list[list[float]] = []
    fold_stress: list[list[float]] = []
    benchmark_fold_returns: list[list[float]] = []
    side_returns: dict[str, list[float]] = {"long": [], "short": []}
    side_stress: dict[str, list[float]] = {"long": [], "short": []}
    side_fold_returns: dict[str, list[list[float]]] = {"long": [], "short": []}
    side_fold_stress: dict[str, list[list[float]]] = {"long": [], "short": []}
    side_decision_frames: dict[str, list[pd.DataFrame]] = {"long": [], "short": []}
    candidate_blocks: dict[str, list[list[float]]] = {}
    decision_frames: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds):
        train = matrix.iloc[fold.train]
        selected = select_expert_library(train, experts)
        identifiers = {item.expert_id for item in selected}
        fitting = matrix.iloc[np.concatenate((fold.train, fold.calibration))]
        fitting = fitting.loc[fitting["expert_id"].isin(identifiers)]
        models = {
            side: _fit_side(fitting, side, config, progress=(fold_number + 1, len(folds)))
            for side in ("long", "short")
        }
        test = matrix.iloc[fold.test]
        test = test.loc[test["expert_id"].isin(identifiers)]
        values, stress, _, _, decisions = _policy_returns(test, models)
        if not decisions.empty:
            decisions["outer_fold"] = fold_number
            decision_frames.append(decisions)
        benchmark_values, _, _, _, _ = _policy_returns(test, models, benchmark=True)
        for isolated_side in ("long", "short"):
            isolated_models = {
                name: model if name == isolated_side else {"enabled": False}
                for name, model in models.items()
            }
            isolated_values, isolated_stress, _, _, isolated_decisions = _policy_returns(
                test, isolated_models
            )
            side_fold_returns[isolated_side].append(isolated_values)
            side_fold_stress[isolated_side].append(isolated_stress)
            side_returns[isolated_side].extend(isolated_values)
            side_stress[isolated_side].extend(isolated_stress)
            if not isolated_decisions.empty:
                isolated_decisions["outer_fold"] = fold_number
                side_decision_frames[isolated_side].append(isolated_decisions)
        fold_returns.append(values)
        fold_stress.append(stress)
        benchmark_fold_returns.append(benchmark_values)
        for blocks in candidate_blocks.values():
            blocks.append([])
        for identifier in identifiers:
            if identifier not in candidate_blocks:
                candidate_blocks[identifier] = [[] for _ in range(fold_number + 1)]
            candidate_blocks[identifier][-1] = test.loc[
                test["expert_id"].eq(identifier), "net_return_r"
            ].tolist()
    return _summarize_oos(
        fold_returns,
        fold_stress,
        side_returns,
        side_stress,
        candidate_blocks,
        benchmark_fold_returns,
        decision_frames,
        side_fold_returns,
        side_fold_stress,
        side_decision_frames,
        config,
    )


def evaluate_oos_policy_duckdb(
    connection: duckdb.DuckDBPyConnection,
    experts: tuple[Expert, ...] | list[Expert],
    config: MachineLearningConfig,
) -> dict[str, Any]:
    bounds = connection.execute(
        "SELECT min(signal_timestamp), max(signal_timestamp) FROM matrix"
    ).fetchone()
    if bounds is None:
        raise ValueError("counterfactual matrix is empty")
    first, last = bounds
    start = pd.Timestamp(first)
    end = pd.Timestamp(last)
    cursor = start + pd.Timedelta(weeks=config.outer_train_weeks)
    fold_returns: list[list[float]] = []
    fold_stress: list[list[float]] = []
    benchmark_fold_returns: list[list[float]] = []
    side_returns: dict[str, list[float]] = {"long": [], "short": []}
    side_stress: dict[str, list[float]] = {"long": [], "short": []}
    side_fold_returns: dict[str, list[list[float]]] = {"long": [], "short": []}
    side_fold_stress: dict[str, list[list[float]]] = {"long": [], "short": []}
    side_decision_frames: dict[str, list[pd.DataFrame]] = {"long": [], "short": []}
    candidate_blocks: dict[str, list[list[float]]] = {}
    decision_frames: list[pd.DataFrame] = []
    fold_number = 0
    probe = cursor
    total_folds = 0
    while (
        probe + pd.Timedelta(weeks=config.outer_calibration_weeks + config.outer_test_weeks) <= end
    ):
        total_folds += 1
        probe += pd.Timedelta(weeks=config.outer_step_weeks)
    while (
        cursor + pd.Timedelta(weeks=config.outer_calibration_weeks + config.outer_test_weeks) <= end
    ):
        train_start = cursor - pd.Timedelta(weeks=config.outer_train_weeks)
        calibration_end = cursor + pd.Timedelta(weeks=config.outer_calibration_weeks)
        test_end = calibration_end + pd.Timedelta(weeks=config.outer_test_weeks)
        selected = select_expert_library_duckdb(
            connection,
            experts,
            train_start,
            cursor - pd.Timedelta(hours=24),
            exits_before=cursor,
        )
        identifiers = {expert.expert_id for expert in selected}
        if identifiers:
            fitting = _query_experts(
                connection,
                identifiers,
                train_start,
                calibration_end,
                exits_before=calibration_end,
            )
            models = {
                side: _fit_side(fitting, side, config, progress=(fold_number + 1, total_folds))
                for side in ("long", "short")
            }
            test = _query_experts(connection, identifiers, calibration_end, test_end)
            values, stress, _, _, decisions = _policy_returns(test, models)
            if not decisions.empty:
                decisions["outer_fold"] = fold_number
                decision_frames.append(decisions)
            benchmark_values, _, _, _, _ = _policy_returns(test, models, benchmark=True)
            for isolated_side in ("long", "short"):
                isolated_models = {
                    name: model if name == isolated_side else {"enabled": False}
                    for name, model in models.items()
                }
                isolated_values, isolated_stress, _, _, isolated_decisions = _policy_returns(
                    test, isolated_models
                )
                side_fold_returns[isolated_side].append(isolated_values)
                side_fold_stress[isolated_side].append(isolated_stress)
                side_returns[isolated_side].extend(isolated_values)
                side_stress[isolated_side].extend(isolated_stress)
                if not isolated_decisions.empty:
                    isolated_decisions["outer_fold"] = fold_number
                    side_decision_frames[isolated_side].append(isolated_decisions)
        else:
            test = pd.DataFrame()
            values, stress = [], []
            benchmark_values = []
            for isolated_side in ("long", "short"):
                side_fold_returns[isolated_side].append([])
                side_fold_stress[isolated_side].append([])
        fold_returns.append(values)
        fold_stress.append(stress)
        benchmark_fold_returns.append(benchmark_values)
        for blocks in candidate_blocks.values():
            blocks.append([])
        for identifier in identifiers:
            if identifier not in candidate_blocks:
                candidate_blocks[identifier] = [[] for _ in range(fold_number + 1)]
            candidate_blocks[identifier][-1] = test.loc[
                test["expert_id"].eq(identifier), "net_return_r"
            ].tolist()
        fold_number += 1
        cursor += pd.Timedelta(weeks=config.outer_step_weeks)
    return _summarize_oos(
        fold_returns,
        fold_stress,
        side_returns,
        side_stress,
        candidate_blocks,
        benchmark_fold_returns,
        decision_frames,
        side_fold_returns,
        side_fold_stress,
        side_decision_frames,
        config,
    )


def _summarize_oos(
    fold_returns: list[list[float]],
    fold_stress: list[list[float]],
    side_returns: dict[str, list[float]],
    side_stress: dict[str, list[float]],
    candidate_blocks: dict[str, list[list[float]]],
    benchmark_fold_returns: list[list[float]],
    decision_frames: list[pd.DataFrame],
    side_fold_returns: dict[str, list[list[float]]],
    side_fold_stress: dict[str, list[list[float]]],
    side_decision_frames: dict[str, list[pd.DataFrame]],
    config: MachineLearningConfig,
) -> dict[str, Any]:
    side_metrics = {side: _metrics(values) for side, values in side_returns.items()}
    side_enabled = {
        side: values["trades"] >= config.minimum_side_trades
        and values["expectancy_r"] > 0
        and _metrics(side_stress[side])["expectancy_r"] >= 0
        for side, values in side_metrics.items()
    }
    enabled_sides = [side for side, enabled in side_enabled.items() if enabled]
    if len(enabled_sides) == 1:
        enabled_side = enabled_sides[0]
        fold_returns = side_fold_returns[enabled_side]
        fold_stress = side_fold_stress[enabled_side]
        decision_frames = side_decision_frames[enabled_side]
    returns = [value for fold in fold_returns for value in fold]
    stress = [value for fold in fold_stress for value in fold]
    metrics = _metrics(returns)
    stress_metrics = _metrics(stress)
    blocks = list(candidate_blocks.values())
    pbo_result = combinatorial_pbo(blocks)
    trial_sharpes = [
        statistics.fmean(values) / statistics.stdev(values)
        for candidate in blocks
        if len(values := [value for block in candidate for value in block]) > 1
        and statistics.stdev(values) > 0
    ]
    dsr = deflated_sharpe_probability(returns, trial_sharpes)
    tested_policies = [fold_returns]
    baseline = [[0.0] for _ in fold_returns]
    reality = reality_check_pvalue(tested_policies, baseline, config.random_seed)
    spa = superior_predictive_ability_pvalue(tested_policies, seed=config.random_seed)
    passed, failures = _gate(
        metrics,
        stress_metrics,
        returns=returns,
        config=config,
        pbo=pbo_result["pbo"],
        dsr=dsr,
        reality_pvalue=reality,
    )
    nonempty = [fold for fold in fold_returns if fold]
    if not nonempty or sum(statistics.fmean(fold) > 0 for fold in nonempty) / len(nonempty) < 0.60:
        failures.append("positive_windows<60%")
        passed = False
    if spa > config.maximum_reality_check_pvalue:
        failures.append("spa_pvalue>gate")
        passed = False
    if not any(side_enabled.values()):
        failures.append("no_side_passed_independent_gate")
        passed = False
    return {
        "passed": passed,
        "failures": sorted(set(failures)),
        "metrics": metrics,
        "stress_2x": stress_metrics,
        "ridge_benchmark": _metrics([value for fold in benchmark_fold_returns for value in fold]),
        "fold_returns": fold_returns,
        "side_metrics": side_metrics,
        "side_enabled": side_enabled,
        "pbo": pbo_result,
        "dsr_probability": dsr,
        "reality_check_pvalue": reality,
        "spa_pvalue": spa,
        "decision_records": (
            pd.concat(decision_frames, ignore_index=True).to_dict("records")
            if decision_frames
            else []
        ),
    }


def _event_driven_policy_replay(
    app: AppConfig, one_minute: pd.DataFrame, trades: pd.DataFrame
) -> dict[str, Any]:
    if trades.empty:
        return {
            "passed": True,
            "engine": "shared BacktestEngine + DefaultRiskEngine + SimulatedBroker",
            "signals": 0,
            "rejected_signals": 0,
            "fills": 0,
            "kill_switches": 0,
            "ending_positions": 0,
            "liquidation_breaches": 0,
            "missing_mark_intervals": 0,
            "risk_budget_violations": 0,
            "returns_r": [],
            "metrics": _metrics([]),
        }
    strategy = app.strategy.model_copy(
        update={
            "timeframe_minutes": 1,
            "crypto_vwap_window": 96,
            "atr_period": 14,
            "adx_period": 14,
            "short_enabled": True,
            "session_flatten_enabled": False,
            "fixed_stop_fraction": None,
            "fixed_target_fraction": None,
        }
    )
    configured = app.model_copy(update={"strategy": strategy})
    frame = one_minute.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    segment_ids = frame["timestamp"].diff().ne(pd.Timedelta(minutes=1)).cumsum()
    results = []
    replay_strategies: list[ScheduledExpertReplayStrategy] = []
    for _, segment in frame.groupby(segment_ids, sort=True):
        segment = segment.reset_index(drop=True)
        first = pd.Timestamp(segment["timestamp"].iloc[0])
        last = pd.Timestamp(segment["timestamp"].iloc[-1]) + pd.Timedelta(minutes=1)
        scheduled = trades.loc[
            pd.to_datetime(trades["signal_timestamp"], utc=True).between(
                first, last, inclusive="left"
            )
        ]
        replay_strategy = ScheduledExpertReplayStrategy(scheduled)
        replay_strategies.append(replay_strategy)
        results.append(
            (
                segment,
                asyncio.run(
                    BacktestEngine(configured, strategy=replay_strategy).run(
                        segment, mode="expert_policy_oos_replay"
                    )
                ),
            )
        )
    liquidation_distance = (
        Decimal("1") / (app.bitunix.leverage if app.bitunix else app.instrument.max_leverage)
        - app.risk.liquidation_buffer_fraction
    )
    liquidation_breaches = 0
    missing_mark_intervals = 0
    ending_positions = 0
    replay_returns: list[float] = []
    for segment, result in results:
        timestamps = pd.to_datetime(segment["timestamp"], utc=True)
        position = Decimal("0")
        opened: tuple[pd.Timestamp, OrderSide, Decimal] | None = None
        trade_cash = Decimal("0")
        risk_budget = Decimal("0")
        curve_times = np.asarray(
            [pd.Timestamp(point.timestamp).value for point in result.equity_curve], dtype=np.int64
        )
        for fill in result.fills:
            signed = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
            previous = position
            position += signed
            trade_cash += (
                -fill.price * fill.quantity
                if fill.side is OrderSide.BUY
                else fill.price * fill.quantity
            ) - fill.commission
            if previous == 0 and position != 0:
                opened = (pd.Timestamp(fill.exchange_timestamp), fill.side, fill.price)
                curve_index = max(
                    0,
                    int(
                        np.searchsorted(
                            curve_times, pd.Timestamp(fill.exchange_timestamp).value, side="left"
                        )
                    )
                    - 1,
                )
                risk_budget = (
                    result.equity_curve[curve_index].equity * configured.risk.risk_per_trade
                )
            if position == 0 and opened is not None:
                if risk_budget > 0:
                    replay_returns.append(float(trade_cash / risk_budget))
                opened_at, side, price = opened
                interval = segment.loc[
                    timestamps.ge(opened_at) & timestamps.le(fill.exchange_timestamp)
                ]
                mark_column = "mark_low" if side is OrderSide.BUY else "mark_high"
                mark = pd.to_numeric(interval[mark_column], errors="coerce")
                observed = mark.min() if side is OrderSide.BUY else mark.max()
                if interval.empty or not np.isfinite(float(observed)):
                    missing_mark_intervals += 1
                else:
                    liquidation = price * (
                        Decimal("1") - liquidation_distance
                        if side is OrderSide.BUY
                        else Decimal("1") + liquidation_distance
                    )
                    liquidation_breaches += int(
                        Decimal(str(observed)) <= liquidation
                        if side is OrderSide.BUY
                        else Decimal(str(observed)) >= liquidation
                    )
                opened = None
                trade_cash = Decimal("0")
                risk_budget = Decimal("0")
        ending_positions += int(position != 0)
    kill_switches = sum(result.kill_switches for _, result in results)
    rejected_signals = sum(result.rejected_signals for _, result in results)
    entries_attempted = sum(strategy.entries_attempted for strategy in replay_strategies)
    return {
        "passed": not kill_switches
        and not ending_positions
        and not liquidation_breaches
        and not missing_mark_intervals
        and not rejected_signals
        and entries_attempted == len(trades)
        and len(replay_returns) == entries_attempted,
        "engine": "shared BacktestEngine + DefaultRiskEngine + SimulatedBroker",
        "signals": sum(result.signals for _, result in results),
        "rejected_signals": rejected_signals,
        "scheduled_entries": len(trades),
        "entries_attempted": entries_attempted,
        "fills": sum(len(result.fills) for _, result in results),
        "kill_switches": kill_switches,
        "ending_positions": ending_positions,
        "liquidation_breaches": liquidation_breaches,
        "missing_mark_intervals": missing_mark_intervals,
        "risk_budget_violations": 0,
        "returns_r": replay_returns,
        "metrics": _metrics(replay_returns),
        "duplicate_order_protection": "OrderManager deterministic client IDs",
        "continuous_segments": len(results),
        "net_pnl": str(sum((result.net_pnl for _, result in results), Decimal("0"))),
        "fees": str(sum((result.fees for _, result in results), Decimal("0"))),
        "slippage": str(sum((result.slippage for _, result in results), Decimal("0"))),
    }


def run_expert_training(
    app: AppConfig, one_minute: pd.DataFrame, *, resume: bool = False
) -> dict[str, Any]:
    config = _scientific_config(app).model_copy(
        update={"status_path": Path("data/reports/ml_expert_research.status.json")}
    )
    report_path = Path("data/reports/ml_expert_research_v5.json")
    if resume and report_path.exists():
        checkpoint: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(str(checkpoint.get("bundle", ""))).exists():
            return checkpoint
    gpu = gpu_preflight(required=config.gpu_required)
    run_id = datetime.now(UTC).strftime("expert-v5-%Y%m%dT%H%M%SZ")
    existing = load_existing_experts(latest_candidate_screen())
    unique = {expert.expert_id: expert for expert in (*generate_preregistered_experts(), *existing)}
    experts = tuple(unique[key] for key in sorted(unique))
    if len(experts) > 8766:
        raise ValueError("expert universe exceeds the preregistered 8,766-action cap")
    root = Path("data/ml/counterfactual_v5")
    timestamps = pd.to_datetime(one_minute["timestamp"], utc=True)
    holdout_end = timestamps.max() + pd.Timedelta(minutes=1)
    holdout_start = holdout_end - pd.Timedelta(weeks=config.holdout_weeks)
    development = one_minute.loc[timestamps.lt(holdout_start)].copy()
    if development.empty:
        raise ValueError("no development data precedes the sealed final holdout")
    identity_path = root / "universe.json"
    identity = {"protocol": PROTOCOL_VERSION, "universe_sha256": universe_hash(experts)}
    if identity_path.exists():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != identity:
            raise RuntimeError("counterfactual directory belongs to a different frozen universe")
    else:
        _atomic_json(identity_path, identity)
    write_ml_status(config, "counterfactual", "Building frozen expert matrix", 5, backend=str(gpu))
    paths = build_counterfactual_checkpointed(
        app, development, experts, root, config, resume=resume
    )
    if not paths:
        row_count = 0
        verdict: Verdict = "NO_DEPLOYABLE_POLICY"
        selected: tuple[Expert, ...] = ()
        models: dict[str, Any] = {}
        evaluation: dict[str, Any] = {"passed": False, "failures": ["empty_matrix"]}
        integrity: dict[str, int] = {"rows": 0, "duplicates": 0, "invalid": 0}
    else:
        connection = _matrix_connection(root)
        write_ml_status(config, "matrix_validation", "Causal schema and integrity audit", 31)
        integrity = _validate_matrix(connection)
        row_count = integrity["rows"]
        manifest = {
            "protocol": PROTOCOL_VERSION,
            "universe_sha256": universe_hash(experts),
            "rows": row_count,
            "integrity": integrity,
            "partitions": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for path in paths
            ],
        }
        _atomic_json(root / "manifest.json", manifest)
        write_ml_status(config, "outer_folds", "Nested purged OOS evaluation", 35)
        evaluation = evaluate_oos_policy_duckdb(connection, experts, config)
        oos_decisions = pd.DataFrame(evaluation.get("decision_records", []))
        if oos_decisions.empty:
            replay_frame = development.iloc[:0]
        else:
            replay_start = pd.to_datetime(
                oos_decisions["signal_timestamp"], utc=True
            ).min() - pd.Timedelta(days=7)
            replay_end = pd.to_datetime(oos_decisions["exit_timestamp"], utc=True).max()
            replay_frame = development.loc[
                pd.to_datetime(development["timestamp"], utc=True).between(
                    replay_start, replay_end, inclusive="both"
                )
            ]
        replay = _event_driven_policy_replay(app, replay_frame, oos_decisions)
        evaluation["event_driven_replay"] = replay
        if not replay["passed"]:
            evaluation["passed"] = False
            evaluation.setdefault("failures", []).append("event_driven_replay_failed")
            evaluation["failures"] = sorted(set(evaluation["failures"]))
        bounds = connection.execute(
            "SELECT min(signal_timestamp), max(signal_timestamp) FROM matrix"
        ).fetchone()
        if bounds is None:
            raise ValueError("counterfactual matrix bounds failed")
        matrix_start, matrix_end = bounds
        start = pd.Timestamp(matrix_start)
        end = pd.Timestamp(matrix_end) + pd.Timedelta(minutes=1)
        selected = select_expert_library_duckdb(connection, experts, start, end, exits_before=end)
        selected_rows = _query_experts(
            connection,
            {expert.expert_id for expert in selected},
            start,
            end,
            exits_before=end,
        )
        write_ml_status(config, "final_fit", "Freezing five-seed pooled models", 90)
        models = {side: _fit_side(selected_rows, side, config) for side in ("long", "short")}
        for side, enabled in evaluation.get("side_enabled", {}).items():
            if not enabled:
                models[side] = {"side": side, "enabled": False, "reason": "side_gate_disabled"}
        verdict = "ELIGIBLE_FOR_FINAL_HOLDOUT" if evaluation["passed"] else "NO_DEPLOYABLE_POLICY"
        connection.close()
    bundle_dir = Path("data/models/expert_policy") / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "protocol": PROTOCOL_VERSION,
        "run_id": run_id,
        "universe_hash": universe_hash(experts),
        "experts": selected,
        "models": models,
        "side_enabled": evaluation.get("side_enabled", {}),
        "config": app.model_dump(mode="json"),
        "source_tree_sha256": _tree_sha256(Path("src/adaptive_bot")),
        "git_commit": _git_commit(Path(".git")),
        "verdict": verdict,
        "created_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(bundle, bundle_dir / "bundle.joblib")
    report = {
        "protocol": PROTOCOL_VERSION,
        "run_id": run_id,
        "verdict": verdict,
        "holdout": {"status": "sealed", "opened": False},
        "universe": {"actions": len(experts), "sha256": universe_hash(experts)},
        "frozen": {
            "source_tree_sha256": bundle["source_tree_sha256"],
            "git_commit": bundle["git_commit"],
            "config_sha256": hashlib.sha256(
                json.dumps(bundle["config"], sort_keys=True).encode()
            ).hexdigest(),
            "dataset_manifest_sha256": (
                _file_sha256(config.manifest_path) if config.manifest_path.exists() else None
            ),
        },
        "selected_experts": [asdict(item) for item in selected],
        "oos": evaluation,
        "counterfactual": {
            "rows": row_count,
            "partitions": [str(path) for path in paths],
            "integrity": integrity,
            "manifest": str(root / "manifest.json"),
        },
        "data": {
            "development_end": holdout_start.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "holdout_end": holdout_end.isoformat(),
        },
        "bundle": str(bundle_dir / "bundle.joblib"),
        "resume_requested": resume,
    }
    _atomic_json(report_path, report)
    write_ml_status(config, "development_complete", verdict, 100, run_id=run_id)
    return report


def finalize_expert_training(app: AppConfig, run_id: str) -> dict[str, Any]:
    config = _scientific_config(app).model_copy(
        update={"status_path": Path("data/reports/ml_expert_research.status.json")}
    )
    report_path = Path("data/reports/ml_expert_research_v5.json")
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id:
        raise ValueError("run id does not match the frozen research report")
    if report.get("verdict") != "ELIGIBLE_FOR_FINAL_HOLDOUT":
        raise RuntimeError("development gates did not authorize opening the holdout")
    lock = Path("data/models/expert_policy") / run_id / "holdout.opened.json"
    open_holdout_once(lock, run_id)
    report["holdout"] = {"status": "opened_evaluating", "opened": True, "lock": str(lock)}
    _atomic_json(report_path, report)
    write_ml_status(config, "final_holdout", "One-shot holdout evaluation", 5, run_id=run_id)
    try:
        archive = scientific_archive_path(config, "BTCUSDT")
        raw = pd.read_parquet(archive)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
        holdout_start = pd.Timestamp(report["data"]["holdout_start"])
        holdout_end = pd.Timestamp(report["data"]["holdout_end"])
        window = raw.loc[
            raw["timestamp"].ge(holdout_start - pd.Timedelta(days=7))
            & raw["timestamp"].lt(holdout_end)
        ].copy()
        bundle = joblib.load(report["bundle"])
        selected = tuple(bundle["experts"])
        matrix = build_counterfactual_rows(app, window, selected)
        if matrix.empty:
            returns: list[float] = []
            stress: list[float] = []
            trades = pd.DataFrame()
        else:
            matrix = matrix.loc[
                pd.to_datetime(matrix["signal_timestamp"], utc=True).ge(holdout_start)
                & pd.to_datetime(matrix["signal_timestamp"], utc=True).lt(holdout_end)
                & pd.to_datetime(matrix["exit_timestamp"], utc=True).le(holdout_end)
            ].copy()
            returns, stress, _, _, trades = _policy_returns(matrix, bundle["models"])
        metrics = _metrics(returns)
        stress_metrics = _metrics(stress)
        passed, failures = _gate(
            metrics,
            stress_metrics,
            returns=returns,
            config=config,
            pbo=None,
            dsr=0,
            holdout=True,
        )
        replay = _event_driven_policy_replay(app, window, trades)
        if not replay["passed"]:
            failures.append("event_driven_replay_failed")
            passed = False
        holdout_report = {
            "status": "passed_manual_review_required" if passed else "failed_retired",
            "opened": True,
            "lock": str(lock),
            "metrics": metrics,
            "stress_2x": stress_metrics,
            "failures": sorted(set(failures)),
            "replay": replay,
            "decisions": trades.to_dict("records"),
            "matrix_rows": len(matrix),
            "archive": str(archive),
            "archive_sha256": _file_sha256(archive),
            "bundle_sha256": _file_sha256(Path(report["bundle"])),
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        report["holdout"] = holdout_report
        report["final_verdict"] = (
            "FINAL_HOLDOUT_PASSED_MANUAL_REVIEW_REQUIRED"
            if passed
            else "FINAL_HOLDOUT_FAILED_RETIRED"
        )
        _atomic_json(lock.parent / "final_holdout_report.json", holdout_report)
        _atomic_json(report_path, report)
        write_ml_status(
            config,
            "final_holdout_complete",
            report["final_verdict"],
            100,
            run_id=run_id,
        )
        return report
    except Exception as error:
        report["holdout"] = {
            "status": "failed_retired",
            "opened": True,
            "lock": str(lock),
            "error": str(error),
        }
        report["final_verdict"] = "FINAL_HOLDOUT_FAILED_RETIRED"
        _atomic_json(report_path, report)
        write_ml_status(config, "failed", str(error), 100, run_id=run_id)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(git: Path) -> str | None:
    head = git / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value
    reference = git / value.removeprefix("ref: ")
    return reference.read_text(encoding="utf-8").strip() if reference.exists() else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)
