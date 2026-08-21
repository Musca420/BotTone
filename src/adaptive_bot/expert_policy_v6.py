from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from adaptive_bot.config import AppConfig, MachineLearningConfig

PROTOCOL_VERSION = "adaptive_range_dual_family_v6"
Family = Literal["mean_reversion", "momentum"]
PolicySide = Literal["long", "short"]


@dataclass(frozen=True)
class V6Expert:
    expert_id: str
    family: Family
    side: PolicySide
    timeframe_minutes: int
    entry_rule: str
    stop_atr: float
    exit_rule: str
    exit_value: float
    time_stop_hours: int
    vwap_hours: int | None = None
    entry_threshold: float | None = None
    lookback_bars: int | None = None
    adx_min: float | None = None


@dataclass(frozen=True)
class PassiveFill:
    status: Literal["full", "partial", "none", "invalid"]
    side: PolicySide
    order_timestamp: datetime | None
    fill_timestamp: datetime | None
    limit_price: Decimal | None
    requested_quantity: Decimal
    filled_quantity: Decimal
    queue_ahead: Decimal
    adverse_selection_bps_1s: float | None
    adverse_selection_bps_5s: float | None
    adverse_selection_bps_30s: float | None
    reason: str


@dataclass(frozen=True)
class EVCalibration:
    ev_mean: np.ndarray
    lower_confidence_bound: np.ndarray
    calibration_rows: int
    audit_rows: int
    audit_bias: float
    residual_lower: float


def generate_v6_experts() -> tuple[V6Expert, ...]:
    """Return 1,024 preregistered actions with balanced deterministic coverage."""
    candidates: list[V6Expert] = []
    for side, timeframe, entry_rule in product(
        ("long", "short"), (15, 30, 60, 240), ("touch", "reentry_2bar")
    ):
        for vwap, threshold, stop, (exit_rule, exit_value), timeout in product(
            (8, 24, 48, 96),
            (1.0, 1.5, 2.0),
            (2.0, 3.0, 4.0),
            (("z", 0.5), ("vwap", 0.0)),
            (4, 8, 24),
        ):
            candidates.append(
                _v6_expert(
                    "mean_reversion",
                    side,
                    timeframe,
                    entry_rule,
                    stop,
                    exit_rule,
                    exit_value,
                    timeout,
                    vwap_hours=vwap,
                    entry_threshold=threshold,
                )
            )
    for side, timeframe, entry_rule in product(
        ("long", "short"), (15, 30, 60, 240), ("donchian_breakout", "ema20_pullback")
    ):
        for lookback, adx, stop, (exit_rule, exit_value), timeout in product(
            (12, 24, 48),
            (20.0, 25.0),
            (1.5, 2.0, 3.0),
            (("atr_trailing", 2.0), ("atr_trailing", 3.0), ("ema_cross", 20.0)),
            (8, 24),
        ):
            candidates.append(
                _v6_expert(
                    "momentum",
                    side,
                    timeframe,
                    entry_rule,
                    stop,
                    exit_rule,
                    exit_value,
                    timeout,
                    lookback_bars=lookback,
                    adx_min=adx,
                )
            )
    buckets: dict[tuple[str, str, int, str], list[V6Expert]] = {}
    for expert in candidates:
        buckets.setdefault(
            (expert.family, expert.side, expert.timeframe_minutes, expert.entry_rule), []
        ).append(expert)
    selected = [
        expert
        for key in sorted(buckets)
        for expert in sorted(buckets[key], key=lambda item: item.expert_id)[:32]
    ]
    if len(selected) != 1024 or len({item.expert_id for item in selected}) != 1024:
        raise AssertionError("V6 universe must contain exactly 1,024 unique actions")
    return tuple(selected)


def _v6_expert(
    family: Family,
    side: str,
    timeframe: int,
    entry_rule: str,
    stop: float,
    exit_rule: str,
    exit_value: float,
    timeout: int,
    **optional: Any,
) -> V6Expert:
    payload = {
        "family": family,
        "side": side,
        "timeframe_minutes": timeframe,
        "entry_rule": entry_rule,
        "stop_atr": stop,
        "exit_rule": exit_rule,
        "exit_value": exit_value,
        "time_stop_hours": timeout,
        **optional,
    }
    identifier = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return V6Expert(expert_id=f"v6-{family[0]}-{side[0]}-{identifier}", **payload)


def v6_universe_hash(experts: tuple[V6Expert, ...] | None = None) -> str:
    rows = [asdict(item) for item in (experts or generate_v6_experts())]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def reconstruct_passive_fill(
    events: pd.DataFrame,
    *,
    signal_timestamp: datetime | pd.Timestamp,
    side: PolicySide,
    quantity: Decimal,
    latency_ms: int = 250,
    timeout_seconds: int = 60,
) -> PassiveFill:
    """Conservative price-time fill: cancellations never reduce queue ahead."""
    if quantity <= 0 or latency_ms < 0 or timeout_seconds <= 0:
        raise ValueError("quantity, latency and timeout must be valid")
    required = {"event_type", "exchange_timestamp"}
    if missing := required - set(events):
        raise ValueError(f"microstructure events missing columns: {sorted(missing)}")
    data = events.copy()
    data["exchange_timestamp"] = pd.to_datetime(
        data["exchange_timestamp"], format="mixed", utc=True
    )
    data = data.sort_values(["exchange_timestamp", "event_type"]).reset_index(drop=True)
    signal = pd.Timestamp(signal_timestamp)
    signal = signal.tz_localize("UTC") if signal.tzinfo is None else signal.tz_convert("UTC")
    submitted = signal + pd.Timedelta(milliseconds=latency_ms)
    deadline = submitted + pd.Timedelta(seconds=timeout_seconds)
    books = data.loc[
        data["event_type"].eq("book") & data["exchange_timestamp"].ge(submitted)
    ]
    if books.empty:
        return _empty_fill(side, quantity, "invalid", "no_book_after_latency")
    book = books.iloc[0]
    levels = _row_levels(book, "bids" if side == "long" else "asks")
    if not levels:
        return _empty_fill(side, quantity, "invalid", "book_side_is_empty")
    limit_price, queue_initial = levels[0]
    queue = queue_initial
    filled = Decimal("0")
    filled_at: pd.Timestamp | None = None
    trades = data.loc[
        data["event_type"].eq("trade")
        & data["exchange_timestamp"].ge(pd.Timestamp(book["exchange_timestamp"]))
        & data["exchange_timestamp"].le(deadline)
    ]
    for _, trade in trades.iterrows():
        try:
            price = Decimal(str(trade["price"]))
            volume = Decimal(str(trade["quantity"]))
        except (KeyError, ArithmeticError):
            continue
        aggressive = str(trade.get("aggressor_side", "")).lower()
        matches = (
            aggressive == "sell" and price <= limit_price
            if side == "long"
            else aggressive == "buy" and price >= limit_price
        )
        if not matches:
            continue
        if (side == "long" and price < limit_price) or (
            side == "short" and price > limit_price
        ):
            queue = Decimal("0")
        consumed = min(queue, volume)
        queue -= consumed
        available = volume - consumed
        if available > 0:
            filled += min(quantity - filled, available)
            filled_at = pd.Timestamp(trade["exchange_timestamp"])
        if filled >= quantity:
            break
    if filled <= 0:
        return PassiveFill(
            "none",
            side,
            pd.Timestamp(book["exchange_timestamp"]).to_pydatetime(),
            None,
            limit_price,
            quantity,
            Decimal("0"),
            queue_initial,
            None,
            None,
            None,
            "queue_not_cleared_before_timeout",
        )
    adverse = tuple(
        _adverse_selection_bps(data, filled_at, limit_price, side, seconds)
        for seconds in (1, 5, 30)
    )
    return PassiveFill(
        "full" if filled == quantity else "partial",
        side,
        pd.Timestamp(book["exchange_timestamp"]).to_pydatetime(),
        None if filled_at is None else filled_at.to_pydatetime(),
        limit_price,
        quantity,
        filled,
        queue_initial,
        adverse[0],
        adverse[1],
        adverse[2],
        "queue_cleared",
    )


def build_passive_fill_labels(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    *,
    latency_ms: int = 250,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Attach conservative maker-fill labels to preregistered signal opportunities."""
    required = {"signal_timestamp", "side", "quantity"}
    if missing := required - set(signals):
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    # ponytail: O(signals * events); replace with an indexed event cursor if profiling requires it.
    rows: list[dict[str, Any]] = []
    for raw_signal in signals.to_dict("records"):
        signal = {str(key): value for key, value in raw_signal.items()}
        side = str(signal["side"])
        if side not in {"long", "short"}:
            raise ValueError(f"invalid policy side: {side}")
        outcome = reconstruct_passive_fill(
            events,
            signal_timestamp=pd.Timestamp(signal["signal_timestamp"]),
            side=side,  # type: ignore[arg-type]
            quantity=Decimal(str(signal["quantity"])),
            latency_ms=latency_ms,
            timeout_seconds=timeout_seconds,
        )
        rows.append(
            {
                **signal,
                "maker_fill_status": outcome.status,
                "maker_fill_probability_target": float(
                    outcome.filled_quantity / outcome.requested_quantity
                ),
                "maker_order_timestamp": outcome.order_timestamp,
                "maker_fill_timestamp": outcome.fill_timestamp,
                "maker_limit_price": outcome.limit_price,
                "maker_filled_quantity": outcome.filled_quantity,
                "maker_queue_ahead": outcome.queue_ahead,
                "adverse_selection_bps_1s": outcome.adverse_selection_bps_1s,
                "adverse_selection_bps_5s": outcome.adverse_selection_bps_5s,
                "adverse_selection_bps_30s": outcome.adverse_selection_bps_30s,
                "execution_valid": outcome.status not in {"invalid"},
                "maker_fill_reason": outcome.reason,
            }
        )
    return pd.DataFrame(rows)


def load_microstructure_events(directory: Path, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Load deduplicated daily event partitions for execution-model fitting."""
    paths = sorted(directory.glob(f"{symbol.lower()}_*.parquet"))
    if not paths:
        return pd.DataFrame()
    data = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    if "event_id" in data:
        data = data.drop_duplicates("event_id")
    data["exchange_timestamp"] = pd.to_datetime(
        data["exchange_timestamp"], format="mixed", utc=True
    )
    return data.sort_values(["exchange_timestamp", "event_type"]).reset_index(drop=True)


def _empty_fill(
    side: PolicySide,
    quantity: Decimal,
    status: Literal["invalid"],
    reason: str,
) -> PassiveFill:
    return PassiveFill(
        status,
        side,
        None,
        None,
        None,
        quantity,
        Decimal("0"),
        Decimal("0"),
        None,
        None,
        None,
        reason,
    )


def _row_levels(row: pd.Series, name: str) -> list[tuple[Decimal, Decimal]]:
    value = row.get(name, row.get(f"{name}_json", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    levels: list[tuple[Decimal, Decimal]] = []
    for item in value:
        try:
            price, quantity = Decimal(str(item[0])), Decimal(str(item[1]))
        except (ArithmeticError, IndexError, TypeError):
            continue
        if price > 0 and quantity > 0:
            levels.append((price, quantity))
    return levels


def _adverse_selection_bps(
    events: pd.DataFrame,
    filled_at: pd.Timestamp | None,
    fill_price: Decimal,
    side: PolicySide,
    seconds: int,
) -> float | None:
    if filled_at is None:
        return None
    target = filled_at + pd.Timedelta(seconds=seconds)
    books = events.loc[
        events["event_type"].eq("book") & events["exchange_timestamp"].ge(target)
    ]
    if books.empty:
        return None
    book = books.iloc[0]
    try:
        midpoint = (Decimal(str(book["best_bid"])) + Decimal(str(book["best_ask"]))) / 2
    except (ArithmeticError, KeyError):
        return None
    adverse = fill_price - midpoint if side == "long" else midpoint - fill_price
    return float(adverse / fill_price * Decimal("10000"))


def calibrate_expected_value(
    raw_prediction: np.ndarray,
    actual: np.ndarray,
    timestamps: pd.Series | np.ndarray,
    *,
    ensemble_std: np.ndarray | None = None,
    seed: int = 20260803,
) -> EVCalibration:
    """Chronological calibration with a disjoint residual audit and honest lower bound."""
    raw = np.asarray(raw_prediction, dtype=float)
    truth = np.asarray(actual, dtype=float)
    if len(raw) != len(truth) or len(raw) < 40 or not np.isfinite(raw).all():
        raise ValueError("at least 40 finite paired calibration observations are required")
    times = pd.to_datetime(timestamps, utc=True)
    order = np.argsort(times.astype("int64"))
    raw, truth = raw[order], truth[order]
    split = len(raw) // 2
    if split < 20 or len(raw) - split < 20:
        raise ValueError("calibration and audit each require at least 20 observations")
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw[:split], truth[:split])
    calibrated = np.asarray(calibrator.predict(raw), dtype=float)
    residuals = truth[split:] - calibrated[split:]
    bias = float(residuals.mean())
    block_size = min(len(residuals), max(2, int(np.sqrt(len(residuals)))))
    residual_lower = _block_mean_quantile(residuals, block_size, seed)
    corrected = calibrated + bias
    dispersion = (
        np.zeros(len(raw), dtype=float)
        if ensemble_std is None
        else np.asarray(ensemble_std, dtype=float)[order]
    )
    if len(dispersion) != len(raw) or (dispersion < 0).any():
        raise ValueError("ensemble dispersion must align and be non-negative")
    z = NormalDist().inv_cdf(0.95)
    lower = calibrated + residual_lower - z * dispersion
    lower = np.minimum(lower, corrected)
    inverse = np.argsort(order)
    return EVCalibration(
        corrected[inverse],
        lower[inverse],
        split,
        len(raw) - split,
        bias,
        residual_lower,
    )


def _block_mean_quantile(values: np.ndarray, block_size: int, seed: int) -> float:
    generator = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_size + 1))
    blocks = int(np.ceil(len(values) / block_size))
    means = np.empty(1000)
    for index in range(len(means)):
        sampled = np.concatenate(
            [values[start : start + block_size] for start in generator.choice(starts, blocks)]
        )[: len(values)]
        means[index] = sampled.mean()
    return float(np.quantile(means, 0.05))


def robust_expert_candidates(
    frame: pd.DataFrame,
    *,
    minimum_opportunities: int = 100,
    minimum_active_months: int = 12,
) -> pd.DataFrame:
    """Reject losing experts before GPU fitting using train-only 2x-cost outcomes."""
    required = {"expert_id", "signal_timestamp", "net_return_r_2x"}
    if missing := required - set(frame):
        raise ValueError(f"expert screen missing columns: {sorted(missing)}")
    data = frame.copy()
    data["month"] = pd.to_datetime(data["signal_timestamp"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for identifier, expert_rows in data.groupby("expert_id", sort=True):
        monthly = expert_rows.groupby("month")["net_return_r_2x"].mean()
        if len(expert_rows) < minimum_opportunities or len(monthly) < minimum_active_months:
            continue
        median = float(monthly.median())
        mad = float((monthly - median).abs().median())
        robust = median - 1.4826 * mad / np.sqrt(len(monthly))
        if robust > 0:
            rows.append(
                {
                    "expert_id": str(identifier),
                    "opportunities": len(expert_rows),
                    "active_months": len(monthly),
                    "robust_net_return_r_2x": robust,
                }
            )
    columns = ["expert_id", "opportunities", "active_months", "robust_net_return_r_2x"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["robust_net_return_r_2x", "expert_id"], ascending=[False, True]
    )


def maker_ev_scenarios(
    gross_ev_r: np.ndarray,
    fill_probability: np.ndarray,
    stop_probability: np.ndarray,
    risk_distance_bps: np.ndarray,
    adverse_selection_bps: np.ndarray,
    *,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert conditional gross alpha into opportunity EV with real maker economics."""
    gross, fill, stopped, risk, adverse = np.broadcast_arrays(
        np.asarray(gross_ev_r, dtype=float),
        np.asarray(fill_probability, dtype=float),
        np.asarray(stop_probability, dtype=float),
        np.asarray(risk_distance_bps, dtype=float),
        np.asarray(adverse_selection_bps, dtype=float),
    )
    if (
        not all(np.isfinite(value).all() for value in (gross, fill, stopped, risk, adverse))
        or ((fill < 0) | (fill > 1)).any()
        or ((stopped < 0) | (stopped > 1)).any()
        or (risk <= 0).any()
        or maker_fee_bps < 0
        or taker_fee_bps < 0
    ):
        raise ValueError("maker EV inputs must be finite probabilities, costs and positive risk")
    exit_fee = stopped * taker_fee_bps + (1 - stopped) * maker_fee_bps
    base_cost = maker_fee_bps + exit_fee + np.maximum(0, adverse)
    return tuple(fill * (gross - multiplier * base_cost / risk) for multiplier in (1, 2, 3))


def select_v6_decisions(actions: pd.DataFrame, *, cooldown_minutes: int = 60) -> pd.DataFrame:
    """Choose one family/side action or FLAT without overlapping positions."""
    required = {
        "signal_timestamp",
        "exit_timestamp",
        "expert_id",
        "family",
        "side",
        "ev_mean",
        "lower_confidence_bound",
        "execution_valid",
    }
    if missing := required - set(actions):
        raise ValueError(f"V6 actions missing columns: {sorted(missing)}")
    if cooldown_minutes < 0:
        raise ValueError("cooldown cannot be negative")
    data = actions.copy()
    data["signal_timestamp"] = pd.to_datetime(data["signal_timestamp"], utc=True)
    data["exit_timestamp"] = pd.to_datetime(data["exit_timestamp"], utc=True)
    data = data.loc[
        data["execution_valid"].astype(bool)
        & data["ev_mean"].gt(0)
        & data["lower_confidence_bound"].gt(0)
        & data["lower_confidence_bound"].le(data["ev_mean"])
    ]
    selected: list[pd.Series] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for timestamp, choices in data.groupby("signal_timestamp", sort=True):
        signal = pd.Timestamp(str(timestamp))
        if signal <= blocked_until:
            continue
        row = choices.sort_values(
            ["lower_confidence_bound", "ev_mean", "family", "expert_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selected.append(row)
        blocked_until = pd.Timestamp(row["exit_timestamp"]) + pd.Timedelta(
            minutes=cooldown_minutes
        )
    return pd.DataFrame(selected).reset_index(drop=True) if selected else data.iloc[:0].copy()


def summarize_v6_groups(
    decisions: pd.DataFrame,
    *,
    minimum_trades: int = 100,
    minimum_profit_factor: float = 1.15,
    maximum_drawdown: float = 0.08,
) -> dict[str, Any]:
    required = {"family", "side", "net_return_r", "net_return_r_2x"}
    if missing := required - set(decisions):
        raise ValueError(f"V6 decisions missing columns: {sorted(missing)}")
    groups: dict[str, Any] = {}
    for (family, side), rows in decisions.groupby(["family", "side"], sort=True):
        normal = _return_metrics(rows["net_return_r"].to_numpy(dtype=float))
        stress = _return_metrics(rows["net_return_r_2x"].to_numpy(dtype=float))
        enabled = (
            normal["trades"] >= minimum_trades
            and normal["expectancy_r"] > 0
            and normal["profit_factor"] >= minimum_profit_factor
            and normal["max_drawdown"] <= maximum_drawdown
            and stress["expectancy_r"] >= 0
        )
        groups[f"{family}:{side}"] = {
            "enabled": enabled,
            "metrics": normal,
            "stress_2x": stress,
        }
    return groups


def _return_metrics(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {
            "trades": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        equity *= max(0.0, 1 + 0.01 * float(value))
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    wins = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    return {
        "trades": float(len(values)),
        "expectancy_r": float(values.mean()),
        "profit_factor": wins / losses if losses else (999.0 if wins else 0.0),
        "max_drawdown": drawdown,
        "win_rate": float((values > 0).mean()),
    }


def microstructure_coverage(directory: Path, symbol: str = "BTCUSDT") -> dict[str, Any]:
    live_path = directory / f"status_{symbol.lower()}.json"
    live = (
        json.loads(live_path.read_text(encoding="utf-8"))
        if live_path.exists()
        else {"connected": False}
    )
    paths = sorted(directory.glob(f"{symbol.lower()}_*.parquet"))
    if not paths:
        return {
            "ready": False,
            "days": 0,
            "book_events": 0,
            "trade_events": 0,
            "live": live,
        }
    frames = [pd.read_parquet(path, columns=["event_type", "exchange_timestamp"]) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        return {
            "ready": False,
            "days": 0,
            "book_events": 0,
            "trade_events": 0,
            "live": live,
        }
    data["exchange_timestamp"] = pd.to_datetime(data["exchange_timestamp"], utc=True)
    counts = data["event_type"].value_counts()
    days_by_type = {
        event: set(data.loc[data["event_type"].eq(event), "exchange_timestamp"].dt.date)
        for event in ("book", "trade")
    }
    common_days = days_by_type["book"] & days_by_type["trade"]
    span_days = (
        0
        if data.empty
        else int((data["exchange_timestamp"].max() - data["exchange_timestamp"].min()).days + 1)
    )
    coverage = len(common_days) / span_days if span_days else 0.0
    return {
        "ready": span_days >= 56 and coverage >= 0.95,
        "days": span_days,
        "common_days": len(common_days),
        "coverage": coverage,
        "book_events": int(counts.get("book", 0)),
        "trade_events": int(counts.get("trade", 0)),
        "start": None if data.empty else data["exchange_timestamp"].min().isoformat(),
        "end": None if data.empty else data["exchange_timestamp"].max().isoformat(),
        "live": live,
    }


def preregister_v6(app: AppConfig, *, now: datetime | None = None) -> dict[str, Any]:
    config = _ml_config(app)
    root = Path("data/models/expert_policy/v6")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "protocol.json"
    experts = generate_v6_experts()
    created = (now or datetime.now(UTC)).astimezone(UTC).replace(second=0, microsecond=0)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "run_id": created.strftime("expert-v6-%Y%m%dT%H%M%SZ"),
        "created_at": created.isoformat(),
        "universe_sha256": v6_universe_hash(experts),
        "actions": len(experts),
        "experts": [asdict(item) for item in experts],
        "config_sha256": hashlib.sha256(
            json.dumps(app.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "source_tree_sha256": _tree_sha256(Path("src/adaptive_bot")),
        "schedule": {
            "fit_end": (created + pd.Timedelta(weeks=config.maker_fit_weeks)).isoformat(),
            "calibration_end": (
                created
                + pd.Timedelta(weeks=config.maker_fit_weeks + config.maker_calibration_weeks)
            ).isoformat(),
            "holdout_end": (
                created
                + pd.Timedelta(
                    weeks=config.maker_fit_weeks
                    + config.maker_calibration_weeks
                    + config.maker_holdout_weeks
                )
            ).isoformat(),
        },
        "execution": {
            "entry": "post_only_best_quote_no_reprice_60s",
            "target": "post_only_reduce_only",
            "stop": "market_reduce_only",
            "latency_ms": 250,
            "maker_fee_bps": config.maker_fee_bps,
            "taker_fee_bps": config.maker_taker_fee_bps,
        },
    }
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        for key in (
            "protocol",
            "universe_sha256",
            "actions",
            "config_sha256",
            "source_tree_sha256",
        ):
            if existing.get(key) != payload.get(key):
                raise RuntimeError("V6 protocol is already frozen with a different identity")
        return cast(dict[str, Any], existing)
    _exclusive_json(target, payload)
    _retire_v5_holdout_unopened(root, created)
    return payload


def run_v6_training(app: AppConfig, *, preregister_only: bool = False) -> dict[str, Any]:
    config = _ml_config(app)
    protocol = preregister_v6(app)
    coverage = microstructure_coverage(config.v6_microstructure_directory, app.instrument.symbol)
    control = microstructure_coverage(config.v6_microstructure_directory, "ETHUSDT")
    verdict = "PREREGISTERED" if preregister_only else (
        "READY_FOR_DEVELOPMENT_FIT" if coverage["ready"] else "COLLECTING_MICROSTRUCTURE"
    )
    report = {
        "protocol": PROTOCOL_VERSION,
        "run_id": protocol["run_id"],
        "verdict": verdict,
        "deployable": False,
        "reason": (
            "preregistered_only"
            if preregister_only
            else "eight_week_microstructure_gate_passed"
            if coverage["ready"]
            else "eight_weeks_of_real_book_and_trade_data_are_required"
        ),
        "universe": {
            "actions": protocol["actions"],
            "sha256": protocol["universe_sha256"],
        },
        "schedule": protocol["schedule"],
        "microstructure": coverage,
        "external_control": {"symbol": "ETHUSDT", "gate": False, **control},
        "holdout": {"status": "sealed", "opened": False},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.v6_report_path, report)
    _atomic_json(
        config.v6_status_path,
        {
            "phase": verdict.lower(),
            "detail": report["reason"],
            "percent": min(66, round(coverage.get("days", 0) / 56 * 66, 2)),
            "updated_at": report["updated_at"],
            "microstructure": coverage,
            "external_control": {"symbol": "ETHUSDT", "gate": False, **control},
        },
    )
    return report


def finalize_v6_training(app: AppConfig, run_id: str) -> dict[str, Any]:
    config = _ml_config(app)
    report = json.loads(config.v6_report_path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id:
        raise ValueError("run id does not match the V6 report")
    if report.get("verdict") != "ELIGIBLE_FOR_FINAL_HOLDOUT":
        raise RuntimeError("V6 development and execution gates did not authorize the holdout")
    if datetime.now(UTC) < datetime.fromisoformat(report["schedule"]["holdout_end"]):
        raise RuntimeError("the twelve-week forward holdout window is not complete")
    raise RuntimeError(
        "V6 final evaluation is unavailable until a frozen development bundle exists"
    )


def _retire_v5_holdout_unopened(root: Path, retired_at: datetime) -> None:
    report_path = Path("data/reports/ml_expert_research_v5.json")
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("holdout") != {"status": "sealed", "opened": False}:
        raise RuntimeError("V5 holdout was not sealed and cannot be retired as unopened")
    marker = root / "v5_holdout_retired_unopened.json"
    payload = {
        "v5_run_id": report.get("run_id"),
        "v5_verdict": report.get("verdict"),
        "status": "RETIRED_UNOPENED",
        "retired_for": PROTOCOL_VERSION,
        "retired_at": retired_at.isoformat(),
        "v5_report_sha256": _file_sha256(report_path),
    }
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("v5_report_sha256") != payload["v5_report_sha256"]:
            raise RuntimeError("V5 retirement marker does not match the immutable report")
        return
    _exclusive_json(marker, payload)


def _ml_config(app: AppConfig) -> MachineLearningConfig:
    config = app.machine_learning
    if config is None or not config.enabled:
        raise ValueError("machine-learning research is disabled")
    return config


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


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)
