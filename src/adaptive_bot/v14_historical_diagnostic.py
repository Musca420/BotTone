from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from adaptive_bot.config import AppConfig, load_config
from adaptive_bot.hybrid_policy_v11 import (
    EXCHANGES,
    SHADOW_COST_BPS,
    btc_inventory,
    build_feature_frame,
    cross_exchange_features,
    evaluate_expert,
    return_metrics,
)
from adaptive_bot.hybrid_policy_v14 import (
    FEATURES,
    ORDERFLOW_PATH,
    RESEARCH_BUNDLE_PATH,
    build_orderflow_context,
    first_reentry_events,
    predict_meta_model,
    primary_expert,
    protocol_hash,
    select_non_overlapping,
)

BITUNIX_PATH = Path(
    "data/ml/hybrid_v7/alpha_raw/exchange=bitunix/symbol=BTCUSDT/data.parquet"
)
REPORT_PATH = Path("data/reports/v14_bitunix_historical_diagnostic.json")
MATRIX_PATH = Path("data/ml/hybrid_v14/bitunix_historical_diagnostic.parquet")


def run_diagnostic(app: AppConfig) -> dict[str, Any]:
    bundle = joblib.load(RESEARCH_BUNDLE_PATH)
    if bundle["protocol"]["protocol_sha256"] != protocol_hash():
        raise RuntimeError("V14 frozen model no longer matches its protocol")
    inventory = btc_inventory()
    orderflow = build_orderflow_context(pd.read_parquet(ORDERFLOW_PATH))
    outcomes: list[pd.DataFrame] = []
    for timeframe in (15, 30, 60):
        print(f"V14 historical diagnostic: {timeframe}m features", flush=True)
        external = {
            exchange: build_feature_frame(
                Path(inventory[exchange]["path"]),
                app,
                exchange,
                timeframe_minutes=timeframe,
            )[1]
            for exchange in EXCHANGES
        }
        context = cross_exchange_features(external, EXCHANGES)[EXCHANGES[0]][
            [
                "timestamp",
                "cross_exchange_return_median",
                "cross_exchange_return_dispersion",
                "cross_feature_available_at",
            ]
        ]
        raw, local = build_feature_frame(
            BITUNIX_PATH, app, "bitunix", timeframe_minutes=timeframe
        )
        local = local.merge(context, on="timestamp", how="left", validate="one_to_one")
        local["lookahead_valid"] = pd.to_datetime(
            local["cross_feature_available_at"], utc=True
        ).le(pd.to_datetime(local["signal_timestamp"], utc=True))
        local = pd.merge_asof(
            local.sort_values("signal_timestamp"),
            orderflow.sort_values("orderflow_available_at"),
            left_on="signal_timestamp",
            right_on="orderflow_available_at",
            direction="backward",
            tolerance=pd.Timedelta(minutes=15),
        )
        local["orderflow_lookahead_valid"] = pd.to_datetime(
            local["orderflow_available_at"], utc=True
        ).le(pd.to_datetime(local["signal_timestamp"], utc=True))
        local["feature_coverage"] = (
            local["local_feature_coverage"].fillna(False).astype(bool)
            & local["lookahead_valid"].fillna(False).astype(bool)
            & local["orderflow_coverage"].fillna(False).astype(bool)
            & local["orderflow_lookahead_valid"].fillna(False).astype(bool)
        )
        for side in ("long", "short"):
            events = first_reentry_events(local, side)
            mask = events["event_signal"] & events["feature_coverage"]
            evaluated = evaluate_expert(
                events,
                raw,
                primary_expert(side, timeframe),
                cost_bps=SHADOW_COST_BPS,
                entry_mask=mask,
                stop_prices=events["event_stop_price"],
                timeframe_minutes=timeframe,
                vwap_hours=24,
            )
            if evaluated.empty:
                continue
            evaluated["context_code"] = (
                evaluated["side_code"] * 1_000
                + evaluated["regime_code"] * 100
                + timeframe
            )
            outcomes.append(evaluated)
    opportunities = pd.concat(outcomes, ignore_index=True).sort_values("signal_timestamp")
    complete = opportunities.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=list(FEATURES)
    )
    scored = predict_meta_model(complete, bundle["models"])
    strict = select_non_overlapping(scored, require_lcb=True)
    ev_only = select_non_overlapping(scored, require_lcb=False)
    report = {
        "status": "DISCOVERY_DIAGNOSTIC_NOT_FORWARD_EVIDENCE",
        "purpose": "measure frozen V14 behavior without tuning or promotion",
        "protocol_sha256": protocol_hash(),
        "symbol": "BTCUSDT",
        "exchange": "bitunix",
        "training_changed": False,
        "thresholds_changed": False,
        "history_previously_observed": True,
        "opportunities": len(opportunities),
        "scored_opportunities": len(scored),
        "strict_lcb_trades": len(strict),
        "positive_ev_trades": len(ev_only),
        "by_side_timeframe": _counts(scored),
        "all_counterfactual_metrics": _metrics(scored),
        "positive_ev_policy_metrics": _metrics(ev_only),
        "strict_lcb_policy_metrics": _metrics(strict),
        "transfer_diagnostic": {
            "ev_realized_correlation": float(scored["ev_net"].corr(scored["gross_return_r"])),
            "positive_ev_gross_expectancy_r": float(
                scored.loc[scored["ev_net"].gt(0), "gross_return_r"].mean()
            ),
            "nonpositive_ev_gross_expectancy_r": float(
                scored.loc[scored["ev_net"].le(0), "gross_return_r"].mean()
            ),
            "positive_ev_improves_selection": bool(
                scored.loc[scored["ev_net"].gt(0), "gross_return_r"].mean()
                > scored["gross_return_r"].mean()
            ),
        },
        "deployable": False,
        "forward_gate_unchanged": True,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_parquet(MATRIX_PATH, scored)
    _atomic_json(REPORT_PATH, report)
    return report


def _counts(rows: pd.DataFrame) -> dict[str, int]:
    return {
        f"{side}_{int(str(timeframe))}m": len(group)
        for (side, timeframe), group in rows.groupby(["side", "timeframe_minutes"])
    }


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"trades": 0}
    scenarios = rows.copy()
    scenarios["net_4bps"] = scenarios["gross_return_r"] - scenarios["cost_r_1x"] * (4 / 19)
    scenarios["net_12bps"] = scenarios["gross_return_r"] - scenarios["cost_r_1x"] * (12 / 19)
    return {
        "gross": return_metrics(rows, "gross_return_r"),
        "maker_fee_4bps": return_metrics(scenarios, "net_4bps"),
        "taker_fee_12bps": return_metrics(scenarios, "net_12bps"),
        "costs_19bps": return_metrics(rows, "net_return_r_shadow_19bps"),
        "costs_38bps": return_metrics(rows, "net_return_r_shadow_38bps"),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen V14 Bitunix historical diagnostic")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bitunix_btc_futures_simulated.yaml")
    )
    print(json.dumps(run_diagnostic(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
