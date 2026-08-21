from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from adaptive_bot.expert_policy import moving_block_lower_bound
from adaptive_bot.hybrid_policy_v22.path_audit import atomic_parquet, metrics, non_overlapping
from adaptive_bot.hybrid_policy_v22.protocol import sha256

PROTOCOL = "hybrid_v23a_causal_confirmed_entry_audit"
SOURCE_ROOT = Path("data/ml/hybrid_v22")
ORDERFLOW_PATH = Path("data/ml/hybrid_v14/binance_reference_orderflow_1m.parquet")
ROOT = Path("data/ml/hybrid_v23")
MODEL_ROOT = Path("data/models/expert_policy/v23")
PROTOCOL_PATH = MODEL_ROOT / "protocol.json"
REPORT_PATH = Path("data/reports/ml_hybrid_v23_audit.json")
STATUS_PATH = Path("data/reports/ml_hybrid_v23.status.json")
BASE_COST_BPS = 4.0
STRESS_COST_BPS = 8.0
RULES = (
    "fade_flow_reversal",
    "fade_failure_break",
    "follow_acceptance_3m",
    "follow_acceptance_breakout",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def exit_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stop in (0.5, 0.75):
        for target in ("half_distance", "vwap"):
            for timeout in (30, 60):
                rows.append(
                    {
                        "family": "fade",
                        "stop_kind": "event_extreme",
                        "stop_value": stop,
                        "target_kind": target,
                        "target_value": None,
                        "timeout_minutes": timeout,
                    }
                )
    follow_stops: tuple[tuple[str, float | None], ...] = (("inner_band", None), ("atr", 0.75))
    for stop_kind, stop_value in follow_stops:
        for target_atr in (1.0, 1.5):
            for timeout in (30, 60):
                rows.append(
                    {
                        "family": "follow",
                        "stop_kind": stop_kind,
                        "stop_value": stop_value,
                        "target_kind": "atr",
                        "target_value": target_atr,
                        "timeout_minutes": timeout,
                    }
                )
    for number, row in enumerate(rows):
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["config_id"] = f"v23-{number:02d}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
    return rows


def _payload() -> dict[str, Any]:
    immutable = {
        "protocol": PROTOCOL,
        "source_sha256": sha256(Path(__file__)),
        "v22_protocol_sha256": sha256(Path("data/models/expert_policy/v22/protocol.json")),
        "v22_1_protocol_sha256": sha256(Path("data/models/expert_policy/v22_1/protocol.json")),
        "event_role": "candidate only; never an automatic order",
        "confirmation_window_minutes": 5,
        "rules": list(RULES),
        "entry": "next one-minute open after confirmation becomes available",
        "base_cost_bps_round_trip": BASE_COST_BPS,
        "stress_cost_bps_round_trip": STRESS_COST_BPS,
        "exit_configs": exit_configs(),
        "walk_forward_weeks": [52, 4, 4],
        "gates": {
            "trades": ">=100",
            "expectancy_4bps": ">0",
            "expectancy_8bps": ">=0",
            "profit_factor": ">=1.10",
            "positive_fold_fraction": ">0.50",
            "bootstrap_lcb": ">0",
        },
    }
    canonical = json.dumps(immutable, sort_keys=True, separators=(",", ":"))
    return immutable | {"protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def preregister() -> dict[str, Any]:
    current = _payload()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != current["protocol_sha256"]:
            raise RuntimeError("V23 protocol changed after freezing")
        return dict(existing)
    payload = current | {"registered_at": datetime.now(UTC).isoformat()}
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def _flow(paths: pd.DataFrame) -> pd.DataFrame:
    orderflow = pd.read_parquet(ORDERFLOW_PATH).copy()
    orderflow["path_timestamp"] = pd.to_datetime(orderflow["timestamp"], utc=True)
    orderflow["aggressive_net"] = 2 * orderflow["taker_buy_quote"] - orderflow["quote_volume"]
    orderflow = orderflow.rename(columns={"quote_volume": "flow_quote_volume"})
    joined = paths.merge(
        orderflow[["path_timestamp", "flow_quote_volume", "aggressive_net"]],
        on="path_timestamp",
        how="left",
        validate="many_to_one",
        suffixes=("", "_flow"),
    )
    joined["flow_available_at"] = pd.to_datetime(joined["path_timestamp"], utc=True) + pd.Timedelta(
        minutes=1
    )
    return joined


def confirmations(states: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        key: rows.reset_index(drop=True)
        for key, rows in _flow(paths).groupby("event_id", sort=False)
    }
    output: list[dict[str, Any]] = []
    for state in states.itertuples(index=False):
        path = indexed[state.event_id]
        if path[["flow_quote_volume", "aggressive_net"]].isna().any().any():
            continue
        side = int(np.sign(float(cast(Any, state.deviation_side))))
        vwap, atr = float(cast(Any, state.vwap)), float(cast(Any, state.atr))
        event_extreme = float(cast(Any, state.high if side > 0 else state.low))
        prior_flow = side * float(cast(Any, state.binance_taker_imbalance_15m))
        closes = path["last_price"].to_numpy(float)
        highs, lows = path["high"].to_numpy(float), path["low"].to_numpy(float)
        aggressive = path["aggressive_net"].to_numpy(float)
        volume = path["flow_quote_volume"].to_numpy(float)
        signed_flow = side * np.cumsum(aggressive[:5]) / np.cumsum(volume[:5])
        outer = vwap + side * atr

        failure_index: int | None = None
        for index in range(5):
            prior_close = float(cast(Any, state.close)) if index == 0 else closes[index - 1]
            toward_vwap = side * (closes[index] - prior_close) < 0
            inside_outer = side * (closes[index] - outer) <= 0
            no_new_extreme = (
                highs[index] <= event_extreme + 0.1 * atr
                if side > 0
                else lows[index] >= event_extreme - 0.1 * atr
            )
            flow_reversed = signed_flow[index] <= min(0.0, prior_flow * 0.5)
            if toward_vwap and inside_outer and no_new_extreme and flow_reversed:
                failure_index = index
                output.append(
                    _confirmation_row(state, "fade_flow_reversal", index + 1, signed_flow[index])
                )
                break
        if failure_index is not None:
            failure_level = lows[failure_index] if side > 0 else highs[failure_index]
            for index in range(failure_index + 1, min(failure_index + 4, 59)):
                broken = lows[index] < failure_level if side > 0 else highs[index] > failure_level
                if broken:
                    output.append(
                        _confirmation_row(
                            state, "fade_failure_break", index + 1, signed_flow[min(index, 4)]
                        )
                    )
                    break

        acceptance_index: int | None = None
        for index in range(2, 5):
            accepted = np.all(side * (closes[index - 2 : index + 1] - outer) > 0)
            if accepted and signed_flow[index] > 0:
                acceptance_index = index
                output.append(
                    _confirmation_row(state, "follow_acceptance_3m", index + 1, signed_flow[index])
                )
                break
        if acceptance_index is not None:
            prior_extreme = (
                np.max(highs[: acceptance_index + 1])
                if side > 0
                else np.min(lows[: acceptance_index + 1])
            )
            for index in range(acceptance_index + 1, min(acceptance_index + 6, 59)):
                broken = highs[index] > prior_extreme if side > 0 else lows[index] < prior_extreme
                if broken:
                    output.append(
                        _confirmation_row(
                            state,
                            "follow_acceptance_breakout",
                            index + 1,
                            signed_flow[min(index, 4)],
                        )
                    )
                    break
    return pd.DataFrame(output)


def _confirmation_row(
    state: Any, rule: str, entry_index: int, signed_flow: float
) -> dict[str, Any]:
    decision = pd.Timestamp(cast(Any, state.decision_timestamp))
    confirmation = decision + pd.Timedelta(minutes=entry_index)
    return {
        "event_id": state.event_id,
        "decision_timestamp": decision,
        "confirmation_timestamp": confirmation,
        "entry_timestamp": confirmation,
        "entry_index": entry_index,
        "rule": rule,
        "family": "fade" if rule.startswith("fade") else "follow",
        "event_name": state.event_name,
        "regime_code": state.regime_code,
        "signed_confirmation_flow": signed_flow,
    }


def _simulate(
    path: pd.DataFrame, state: Any, candidate: Any, config: dict[str, Any]
) -> dict[str, Any] | None:
    entry_index = int(candidate.entry_index)
    if entry_index >= len(path):
        return None
    family = str(candidate.family)
    deviation = int(np.sign(float(cast(Any, state.deviation_side))))
    direction = deviation if family == "follow" else -deviation
    entry = float(cast(Any, path.iloc[entry_index]["open"]))
    vwap, atr = float(cast(Any, state.vwap)), float(cast(Any, state.atr))
    if family == "fade":
        extreme = float(cast(Any, state.high if deviation > 0 else state.low))
        stop = extreme + deviation * float(config["stop_value"]) * atr
        target = (entry + vwap) / 2 if config["target_kind"] == "half_distance" else vwap
    else:
        stop = (
            vwap + deviation * 0.5 * atr
            if config["stop_kind"] == "inner_band"
            else entry - direction * float(config["stop_value"]) * atr
        )
        target = entry + direction * float(config["target_value"]) * atr
    risk = direction * (entry - stop)
    reward = direction * (target - entry)
    if risk <= 0 or reward <= 0:
        return None
    window = path.iloc[entry_index : entry_index + int(config["timeout_minutes"])]
    for bar in window.itertuples(index=False):
        low, high = float(cast(Any, bar.low)), float(cast(Any, bar.high))
        stop_hit = low <= stop if direction > 0 else high >= stop
        target_hit = high >= target if direction > 0 else low <= target
        if stop_hit:
            exit_price, reason = stop, "stop"
        elif target_hit:
            exit_price, reason = target, "target"
        else:
            continue
        return _outcome(
            entry, exit_price, risk, direction, pd.Timestamp(cast(Any, bar.path_timestamp)), reason
        )
    final = window.iloc[-1]
    return _outcome(
        entry,
        float(cast(Any, final["last_price"])),
        risk,
        direction,
        pd.Timestamp(cast(Any, final["path_timestamp"])),
        "timeout",
    )


def _outcome(
    entry: float,
    exit_price: float,
    risk: float,
    direction: int,
    timestamp: pd.Timestamp,
    reason: str,
) -> dict[str, Any]:
    gross = direction * (exit_price - entry) / risk
    return {
        "exit_timestamp": timestamp,
        "exit_reason": reason,
        "gross_r": gross,
        "net_4bps_r": gross - BASE_COST_BPS / (risk / entry * 10_000),
        "net_8bps_r": gross - STRESS_COST_BPS / (risk / entry * 10_000),
    }


def build_matrix(
    states: pd.DataFrame, paths: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    state_index = {row.event_id: row for row in states.itertuples(index=False)}
    path_index = {
        key: rows.reset_index(drop=True) for key, rows in paths.groupby("event_id", sort=False)
    }
    output: list[dict[str, Any]] = []
    configs = exit_configs()
    for candidate in candidates.itertuples(index=False):
        state, path = state_index[candidate.event_id], path_index[candidate.event_id]
        for config in configs:
            if config["family"] != candidate.family:
                continue
            outcome = _simulate(path, state, candidate, config)
            if outcome is not None:
                output.append(
                    cast(Any, candidate)._asdict() | {"config_id": config["config_id"]} | outcome
                )
    return pd.DataFrame(output)


def walk_forward(matrix: pd.DataFrame, *, smoke: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    times = pd.to_datetime(matrix["decision_timestamp"], utc=True)
    first, last = times.min().floor("D") + pd.Timedelta(weeks=52), times.max().floor("D")
    starts = list(pd.date_range(first, last - pd.Timedelta(weeks=8), freq="4W", tz="UTC"))
    if smoke:
        starts = starts[-2:]
    output: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for number, start in enumerate(starts, start=1):
        train_end = start
        test_start, test_end = start + pd.Timedelta(weeks=4), start + pd.Timedelta(weeks=8)
        train = matrix.loc[times.ge(start - pd.Timedelta(weeks=52)) & times.lt(train_end)]
        train = train.loc[pd.to_datetime(train["exit_timestamp"], utc=True).lt(train_end)]
        test = matrix.loc[times.ge(test_start) & times.lt(test_end)]
        for rule in RULES:
            ranked = []
            for config_id, rows in train.loc[train["rule"].eq(rule)].groupby("config_id"):
                score = metrics(non_overlapping(rows))
                ranked.append({"config_id": config_id, **score})
            if not ranked:
                continue
            winner = str(
                pd.DataFrame(ranked)
                .sort_values(
                    ["expectancy_8bps", "profit_factor", "config_id"],
                    ascending=[False, False, True],
                )
                .iloc[0]["config_id"]
            )
            audited = non_overlapping(
                test.loc[test["rule"].eq(rule) & test["config_id"].eq(winner)]
            ).copy()
            audited["outer_fold"] = number
            output.append(audited)
            selections.append(
                {"fold": number, "rule": rule, "config_id": winner, "test": metrics(audited)}
            )
        _status("v23a_oos", f"Fold {number}/{len(starts)}", 70 + 25 * number / max(len(starts), 1))
    oos = pd.concat(output, ignore_index=True) if output else matrix.iloc[:0].copy()
    rules: dict[str, Any] = {}
    for rule_value, rows in oos.groupby("rule"):
        rule = str(rule_value)
        result = metrics(rows)
        fold_ev = rows.groupby("outer_fold")["net_4bps_r"].mean()
        lower = moving_block_lower_bound(
            rows["net_4bps_r"].to_numpy(float), block_size=min(20, len(rows)), seed=20260805
        )
        result |= {"positive_fold_fraction": float(fold_ev.gt(0).mean()), "bootstrap_lcb": lower}
        result["gate_passed"] = bool(
            result["trades"] >= 100
            and result["expectancy_4bps"] > 0
            and result["expectancy_8bps"] >= 0
            and result["profit_factor"] >= 1.10
            and result["positive_fold_fraction"] > 0.5
            and lower > 0
        )
        rules[rule] = result
    return oos, {"selections": selections, "rules": rules}


def run(*, smoke: bool) -> dict[str, Any]:
    protocol = preregister()
    states = pd.read_parquet(SOURCE_ROOT / "event_states.parquet")
    paths = pd.read_parquet(SOURCE_ROOT / "event_paths.parquet")
    _status("confirmations", "Causal 1-5 minute entry confirmations", 10)
    candidates = confirmations(states, paths)
    _status("matrix", f"{len(candidates)} confirmed candidates", 35)
    matrix = build_matrix(states, paths, candidates)
    oos, audit = walk_forward(matrix, smoke=smoke)
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_parquet(ROOT / "confirmations.parquet", candidates)
    atomic_parquet(ROOT / "confirmed_entry_matrix.parquet", matrix)
    atomic_parquet(ROOT / "oos_confirmed_entries.parquet", oos)
    passed = any(value["gate_passed"] for value in audit["rules"].values())
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "stage": "V23-A_CONFIRMED_ENTRY_AUDIT",
        "verdict": "CONFIRMED_ENTRY_FAMILY_FOUND" if passed else "NO_CONFIRMED_ENTRY_EDGE_FOUND",
        "artifact_class": "RESEARCH_ONLY",
        "ml_authorized": passed and not smoke,
        "paper_only": False,
        "deployable": False,
        "events": len(states),
        "confirmed_candidates": len(candidates),
        "matrix_rows": len(matrix),
        "oos_rows": len(oos),
        "audit": audit,
        "smoke": smoke,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT_PATH, report)
    _status("complete", report["verdict"], 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V23-A confirmed VWAP entry audit")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
