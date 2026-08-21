from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.hybrid_policy_v25.protocol import atomic_json


def data_report(audits: dict[str, Any], *, protocol_hash: str) -> dict[str, Any]:
    ready = all(
        value["mandatory_coverage"] >= 0.99
        and value["mark_coverage"] >= 0.99
        and value["funding_coverage"] >= 0.99
        and value["future_feature_violations"] == 0
        for value in audits.values()
    )
    report = {
        "verdict": "DATA_READY" if ready else "DATA_FAILURE",
        "protocol_hash": protocol_hash,
        "assets": audits,
        "optional_missing": [
            "historical bid/ask",
            "historical L2",
            "liquidations",
            "open interest where context_coverage=false",
        ],
    }
    atomic_json(Path("data/reports/ml_hybrid_v25_data_audit.json"), report)
    lines = [
        "# V25 data audit",
        "",
        f"Verdetto: `{report['verdict']}`. Protocollo: `{protocol_hash}`.",
        "",
        "| Asset | Minuti | Copertura obbligatoria | Mark | Funding | OI opzionale |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for asset, value in audits.items():
        lines.append(
            f"| {asset} | {value['rows']:,} | {value['mandatory_coverage']:.2%} | "
            f"{value['mark_coverage']:.2%} | {value['funding_coverage']:.2%} | "
            f"{value['open_interest_optional_coverage']:.2%} |"
        )
    lines += [
        "",
        "Bid/ask, L2 e liquidazioni mancanti sono opzionali e non vengono inventati. "
        "Il costo storico resta il proxy composito preregistrato.",
    ]
    Path("docs/hybrid-policy-v25-data-audit.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def policy_report(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    model_audit: dict[str, Any],
    *,
    protocol_hash: str,
) -> dict[str, Any]:
    metrics = model_audit["metrics"]
    values = decisions.get("observed_net_bps", pd.Series(dtype=float))
    months = (
        decisions.groupby(
            pd.to_datetime(decisions["entry_timestamp"], utc=True).dt.strftime("%Y-%m")
        )["observed_net_bps"].sum()
        if not decisions.empty
        else pd.Series(dtype=float)
    )
    weeks = (
        decisions.groupby(
            pd.to_datetime(decisions["entry_timestamp"], utc=True).dt.strftime("%G-%V")
        )["observed_net_bps"].sum()
        if not decisions.empty
        else pd.Series(dtype=float)
    )
    positive_total = float(values.clip(lower=0).sum())
    trade_concentration = (
        float(values.clip(lower=0).max() / positive_total) if positive_total > 0 else 1.0
    )
    weekly_positive = weeks.clip(lower=0)
    weekly_concentration = (
        float(weekly_positive.max() / weekly_positive.sum()) if weekly_positive.sum() > 0 else 1.0
    )
    has_trades = len(decisions) > 0
    paper_gates = {
        "oos_trades_300": len(decisions) >= 300,
        "expectancy_4bps_positive": has_trades and metrics["expectancy_net_bps_4"] > 0,
        "expectancy_8bps_nonnegative": has_trades and metrics["expectancy_net_bps_8"] >= 0,
        "profit_factor_1_20": has_trades and metrics["profit_factor"] >= 1.20,
        "bootstrap_lcb_positive": has_trades and metrics["bootstrap_lcb_bps"] > 0,
        "max_drawdown_10pct": has_trades and metrics["max_drawdown_bps"] <= 1_000,
        "majority_months_positive": float(months.gt(0).mean()) > 0.5 if len(months) else False,
        "single_trade_15pct": trade_concentration <= 0.15,
        "single_week_25pct": weekly_concentration <= 0.25,
    }
    if len(events) < 2_000:
        verdict = "INSUFFICIENT_DATA"
    elif not model_audit["gates_passed"]:
        verdict = "NO_META_EDGE"
    elif not all(paper_gates.values()):
        verdict = (
            "RESEARCH_POLICY_READY" if metrics["expectancy_net_bps_4"] > 0 else "NO_POLICY_EDGE"
        )
    else:
        verdict = "PAPER_VALIDATED"
    report = {
        "verdict": verdict,
        "protocol_hash": protocol_hash,
        "independent_events": len(events),
        "oos_events": len(predictions),
        "oos_trades": len(decisions),
        "metrics": metrics,
        "research_gates": model_audit["gates"],
        "paper_gates": paper_gates,
        "trade_pnl_concentration": trade_concentration,
        "weekly_pnl_concentration": weekly_concentration,
        "holdout": model_audit["holdout"],
        "live_allowed": False,
        "auto_promotion": False,
    }
    atomic_json(Path("data/reports/ml_hybrid_v25_policy_audit.json"), report)
    lines = [
        "# V25 policy audit",
        "",
        f"Verdetto: `{verdict}`. Il live è vietato.",
        "",
        f"- Eventi indipendenti: {len(events):,}",
        f"- Eventi OOS: {len(predictions):,}",
        f"- Trade OOS: {len(decisions):,}",
        f"- EV netta 4 bps: {metrics['expectancy_net_bps_4']:.4f} bps",
        f"- EV netta 8 bps: {metrics['expectancy_net_bps_8']:.4f} bps",
        f"- Profit factor: {metrics['profit_factor']:.4f}",
        f"- Bootstrap LCB: {metrics['bootstrap_lcb_bps']:.4f} bps",
        "",
        "L'holdout BTC futuro resta sigillato e `opened=false`. "
        "Nessuna metrica deriva da quel periodo.",
    ]
    Path("docs/hybrid-policy-v25-policy-audit.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def model_markdown(audit: dict[str, Any]) -> None:
    metrics = audit["metrics"]
    lines = [
        "# V25 model audit",
        "",
        f"Verdetto: `{audit['verdict']}`.",
        "",
        f"- Eventi OOS: {audit['oos_events']:,}",
        f"- Trade selezionati: {audit['oos_trades']:,}",
        f"- EV netta: {metrics['expectancy_net_bps_4']:.4f} bps",
        f"- PF: {metrics['profit_factor']:.4f}",
        f"- Modelli scelti: `{json.dumps(audit['chosen_models'], sort_keys=True)}`",
        "",
        "Il test di ogni fold non partecipa a modello, iperparametri, calibrazione o soglia.",
    ]
    Path("docs/hybrid-policy-v25-model-audit.md").write_text("\n".join(lines), encoding="utf-8")
