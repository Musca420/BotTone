from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.hybrid_policy_v11 import btc_inventory
from adaptive_bot.hybrid_policy_v21 import STATE_PATH as V21_STATE_PATH
from adaptive_bot.hybrid_policy_v22.protocol import (
    BASE_COST_BPS,
    HORIZONS_MINUTES,
    ROOT,
    STRESS_COST_BPS,
    exit_grid,
    status,
)

EVENT_STATES_PATH = ROOT / "event_states.parquet"
EVENT_PATHS_PATH = ROOT / "event_paths.parquet"
PATH_LABELS_PATH = ROOT / "path_labels.parquet"
GRID_RESULTS_PATH = ROOT / "exit_grid_results.parquet"
OOS_PATH = ROOT / "oos_exit_audit.parquet"


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def event_id(timestamp: pd.Timestamp, event_name: str, protocol_hash: str) -> str:
    value = f"BTCUSDT|binance|{timestamp.isoformat()}|{event_name}|{protocol_hash}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def build_paths(protocol_hash: str, *, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resume and EVENT_STATES_PATH.exists() and EVENT_PATHS_PATH.exists():
        return pd.read_parquet(EVENT_STATES_PATH), pd.read_parquet(EVENT_PATHS_PATH)
    states = pd.read_parquet(V21_STATE_PATH).copy()
    states["decision_timestamp"] = pd.to_datetime(states["signal_timestamp"], utc=True)
    states["available_at"] = pd.to_datetime(states["context_available_at"], utc=True)
    if states["available_at"].gt(states["decision_timestamp"]).any():
        raise RuntimeError("V22 causal feature availability violation")
    states["event_id"] = [
        event_id(timestamp, name, protocol_hash)
        for timestamp, name in zip(states["decision_timestamp"], states["event_name"], strict=True)
    ]
    if states["event_id"].duplicated().any():
        raise RuntimeError("V22 duplicate event_id")
    raw_path = Path(btc_inventory()["binance"]["path"])
    raw = pd.read_parquet(raw_path).copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    output: list[pd.DataFrame] = []
    for number, row in enumerate(states.itertuples(index=False), start=1):
        decision = pd.Timestamp(cast(Any, row.decision_timestamp))
        path = raw.loc[decision : decision + pd.Timedelta(minutes=59)].copy()
        expected = pd.date_range(decision, periods=60, freq="1min", tz="UTC")
        if len(path) != 60 or not path.index.equals(expected) or not path["data_valid"].all():
            continue
        path = path.reset_index()
        path["event_id"] = row.event_id
        path["path_timestamp"] = path["timestamp"]
        path["available_at"] = path["path_timestamp"] + pd.Timedelta(minutes=1)
        path["elapsed_seconds"] = np.arange(60) * 60
        path["last_price"] = path["close"]
        path["mark_price"] = path["mark_close"]
        path["cumulative_base_volume"] = path["volume"].cumsum()
        path["cumulative_quote_volume"] = path["quote_volume"].cumsum()
        output.append(
            path[
                [
                    "event_id",
                    "path_timestamp",
                    "available_at",
                    "elapsed_seconds",
                    "open",
                    "high",
                    "low",
                    "last_price",
                    "mark_price",
                    "cumulative_base_volume",
                    "cumulative_quote_volume",
                ]
            ]
        )
        if number % 500 == 0:
            status(
                "path_builder", f"Event path {number}/{len(states)}", 5 + 20 * number / len(states)
            )
    paths = pd.concat(output, ignore_index=True)
    valid_ids = set(paths["event_id"])
    states = states.loc[states["event_id"].isin(valid_ids)].reset_index(drop=True)
    states["protocol_hash"] = protocol_hash
    atomic_parquet(EVENT_STATES_PATH, states)
    atomic_parquet(EVENT_PATHS_PATH, paths)
    return states, paths


def _first_hit(condition: np.ndarray) -> tuple[float, bool]:
    hits = np.flatnonzero(condition)
    return (float((hits[0] + 1) * 60), False) if len(hits) else (np.nan, True)


def build_path_labels(states: pd.DataFrame, paths: pd.DataFrame, *, resume: bool) -> pd.DataFrame:
    if resume and PATH_LABELS_PATH.exists():
        return pd.read_parquet(PATH_LABELS_PATH)
    indexed = {key: rows for key, rows in paths.groupby("event_id", sort=False)}
    labels: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        path = indexed[row.event_id]
        entry = float(cast(Any, path.iloc[0]["open"]))
        atr = float(cast(Any, row.atr))
        side = int(np.sign(float(cast(Any, row.deviation_side))))
        high = path["high"].to_numpy(float)
        low = path["low"].to_numpy(float)
        record: dict[str, Any] = {
            "event_id": row.event_id,
            "decision_timestamp": row.decision_timestamp,
            "label_end_timestamp": path.iloc[-1]["path_timestamp"],
            "event_name": row.event_name,
        }
        for family, direction in (("follow", side), ("fade", -side)):
            favorable = (high - entry) * direction if direction > 0 else (entry - low)
            adverse = (entry - low) if direction > 0 else (high - entry)
            for horizon in HORIZONS_MINUTES:
                record[f"{family}_mfe_r_{horizon}m"] = float(np.nanmax(favorable[:horizon]) / atr)
                record[f"{family}_mae_r_{horizon}m"] = float(np.nanmax(adverse[:horizon]) / atr)
        vwap = float(cast(Any, row.vwap))
        half = (entry + vwap) / 2
        if side > 0:
            time_vwap = _first_hit(low <= vwap)
            time_half = _first_hit(low <= half)
            time_inner = _first_hit(low <= vwap + 0.5 * atr)
        else:
            time_vwap = _first_hit(high >= vwap)
            time_half = _first_hit(high >= half)
            time_inner = _first_hit(high >= vwap - 0.5 * atr)
        for name, value in (
            ("vwap", time_vwap),
            ("half_vwap", time_half),
            ("inner_band", time_inner),
        ):
            record[f"time_to_{name}_seconds"], record[f"{name}_censored"] = value
        labels.append(record)
    result = pd.DataFrame(labels)
    atomic_parquet(PATH_LABELS_PATH, result)
    return result


@dataclass(frozen=True)
class Outcome:
    gross_r: float
    net_4bps_r: float
    net_8bps_r: float
    exit_timestamp: pd.Timestamp
    exit_reason: str
    risk_price: float


def simulate(path: pd.DataFrame, state: Any, config: dict[str, Any]) -> Outcome | None:
    family = str(config["family"])
    deviation = int(np.sign(state.deviation_side))
    direction = deviation if family == "follow" else -deviation
    delay = 0 if config["entry_mode"] == "immediate" else 5
    initial = float(path.iloc[0]["open"])
    if delay and direction * (float(path.iloc[delay - 1]["last_price"]) - initial) <= 0:
        return None
    entry = float(path.iloc[delay]["open"])
    atr = float(state.atr)
    vwap = float(state.vwap)
    if family == "fade":
        extreme_z = float(state.excursion_max_z)
        stop = vwap + deviation * (extreme_z + float(config["stop_value"])) * atr
        target_kind = config["target_kind"]
        target = (
            (entry + vwap) / 2
            if target_kind == "half_distance"
            else vwap + deviation * 0.5 * atr
            if target_kind == "inner_band"
            else vwap
        )
    else:
        stop = (
            vwap + deviation * 0.5 * atr
            if config["stop_kind"] == "inside_band"
            else entry - direction * float(config["stop_value"]) * atr
        )
        target = entry + direction * float(config["target_value"]) * atr
    risk = direction * (entry - stop)
    reward = direction * (target - entry)
    if not np.isfinite(risk) or risk <= 0 or reward <= 0:
        return None
    window = path.iloc[delay : delay + int(config["timeout_minutes"])]
    for bar in window.itertuples(index=False):
        low, high = float(cast(Any, bar.low)), float(cast(Any, bar.high))
        stop_hit = low <= stop if direction > 0 else high >= stop
        target_hit = high >= target if direction > 0 else low <= target
        if stop_hit:  # worst case when both are touched in one minute
            exit_price, reason = stop, "stop"
        elif target_hit:
            exit_price, reason = target, "target"
        else:
            continue
        gross = direction * (exit_price - entry) / risk
        cost_4 = BASE_COST_BPS / (risk / entry * 10_000)
        cost_8 = STRESS_COST_BPS / (risk / entry * 10_000)
        return Outcome(
            gross,
            gross - cost_4,
            gross - cost_8,
            pd.Timestamp(cast(Any, bar.path_timestamp)),
            reason,
            risk,
        )
    final = window.iloc[-1]
    gross = direction * (float(final["last_price"]) - entry) / risk
    cost_4 = BASE_COST_BPS / (risk / entry * 10_000)
    cost_8 = STRESS_COST_BPS / (risk / entry * 10_000)
    return Outcome(
        gross,
        gross - cost_4,
        gross - cost_8,
        pd.Timestamp(cast(Any, final["path_timestamp"])),
        "timeout",
        risk,
    )


def build_grid(states: pd.DataFrame, paths: pd.DataFrame, *, resume: bool) -> pd.DataFrame:
    if resume and GRID_RESULTS_PATH.exists():
        return pd.read_parquet(GRID_RESULTS_PATH)
    indexed = {key: rows for key, rows in paths.groupby("event_id", sort=False)}
    rows: list[dict[str, Any]] = []
    configs = exit_grid()
    for number, state in enumerate(states.itertuples(index=False), start=1):
        path = indexed[state.event_id]
        for config in configs:
            outcome = simulate(path, state, config)
            if outcome is None:
                continue
            rows.append(
                {
                    "event_id": state.event_id,
                    "decision_timestamp": state.decision_timestamp,
                    "event_name": state.event_name,
                    "regime_code": state.regime_code,
                    "family": config["family"],
                    "config_id": config["config_id"],
                    "entry_mode": config["entry_mode"],
                    "exit_timestamp": outcome.exit_timestamp,
                    "exit_reason": outcome.exit_reason,
                    "gross_r": outcome.gross_r,
                    "net_4bps_r": outcome.net_4bps_r,
                    "net_8bps_r": outcome.net_8bps_r,
                    "risk_price": outcome.risk_price,
                }
            )
        if number % 250 == 0:
            status(
                "exit_grid",
                f"Event {number}/{len(states)} x {len(configs)} configs",
                35 + 30 * number / len(states),
            )
    result = pd.DataFrame(rows)
    atomic_parquet(GRID_RESULTS_PATH, result)
    return result


def metrics(rows: pd.DataFrame) -> dict[str, float]:
    values = rows["net_4bps_r"].to_numpy(float)
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return {
        "trades": float(len(rows)),
        "expectancy_4bps": float(np.mean(values)) if len(values) else np.nan,
        "expectancy_8bps": float(rows["net_8bps_r"].mean()) if len(rows) else np.nan,
        "profit_factor": float(gains / losses) if losses else float("inf"),
    }


def non_overlapping(rows: pd.DataFrame) -> pd.DataFrame:
    selected: list[Any] = []
    blocked = pd.Timestamp("1900", tz="UTC")
    for index, row in rows.sort_values("decision_timestamp").iterrows():
        signal = pd.Timestamp(row["decision_timestamp"])
        if signal > blocked:
            selected.append(index)
            blocked = pd.Timestamp(row["exit_timestamp"])
    return rows.loc[selected].copy()


def walk_forward_exit_audit(
    grid: pd.DataFrame, *, smoke: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    times = pd.to_datetime(grid["decision_timestamp"], utc=True)
    first = times.min().floor("D") + pd.Timedelta(weeks=52)
    last = times.max().floor("D")
    fold_starts = list(pd.date_range(first, last - pd.Timedelta(weeks=8), freq="4W", tz="UTC"))
    if smoke:
        fold_starts = fold_starts[-2:]
    output: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for number, start in enumerate(fold_starts, start=1):
        train_start, train_end = start - pd.Timedelta(weeks=52), start
        test_start, test_end = start + pd.Timedelta(weeks=4), start + pd.Timedelta(weeks=8)
        train = grid.loc[times.ge(train_start) & times.lt(train_end)].copy()
        train = train.loc[pd.to_datetime(train["exit_timestamp"], utc=True).lt(train_end)]
        test = grid.loc[times.ge(test_start) & times.lt(test_end)]
        for family in ("fade", "follow"):
            summaries = []
            for config_id, rows in train.loc[train["family"].eq(family)].groupby("config_id"):
                chosen = non_overlapping(rows)
                score = metrics(chosen)
                summaries.append({"config_id": config_id, **score})
            if not summaries:
                continue
            ranked = pd.DataFrame(summaries).sort_values(
                ["expectancy_8bps", "profit_factor", "config_id"],
                ascending=[False, False, True],
            )
            winner = str(ranked.iloc[0]["config_id"])
            audited = non_overlapping(test.loc[test["config_id"].eq(winner)]).copy()
            audited["outer_fold"] = number
            output.append(audited)
            selections.append(
                {
                    "outer_fold": number,
                    "family": family,
                    "config_id": winner,
                    "train": metrics(non_overlapping(train.loc[train["config_id"].eq(winner)])),
                    "test": metrics(audited),
                }
            )
        status(
            "v22a_oos",
            f"Fold {number}/{len(fold_starts)}",
            68 + 25 * number / max(len(fold_starts), 1),
        )
    oos = pd.concat(output, ignore_index=True) if output else grid.iloc[:0].copy()
    atomic_parquet(OOS_PATH, oos)
    families: dict[str, Any] = {}
    for family_value, rows in oos.groupby("family"):
        family = str(family_value)
        result = metrics(rows)
        fold_ev = rows.groupby("outer_fold")["net_4bps_r"].mean()
        months = pd.to_datetime(rows["decision_timestamp"], utc=True).dt.to_period("M")
        monthly = rows["net_4bps_r"].groupby(months).sum()
        positive_profit = monthly.clip(lower=0)
        dominant = (
            float(positive_profit.max() / positive_profit.sum())
            if positive_profit.sum() > 0
            else 1.0
        )
        result |= {
            "positive_fold_fraction": float(fold_ev.gt(0).mean()),
            "dominant_positive_month_share": dominant,
        }
        result["gate_passed"] = bool(
            result["trades"] >= 100
            and result["expectancy_4bps"] > 0
            and result["expectancy_8bps"] >= 0
            and result["profit_factor"] >= 1.10
            and result["positive_fold_fraction"] > 0.5
            and dominant <= 0.5
        )
        families[family] = result
    return oos, {"fold_selections": selections, "families": families}
