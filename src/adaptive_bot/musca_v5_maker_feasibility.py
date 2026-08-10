from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_bot.musca_v5_fine_tuning import _atomic_json

BITUNIX_RAW = Path("data/raw/bitunix_microstructure")
BITUNIX_L2 = Path("data/research/bitunix_l2/btcusdt_l2_features.parquet")
OBSERVED_LABELS = Path(
    "data/research/bitunix_execution/btcusdt_observed_post_only_labels.parquet"
)
REPORT = Path("data/reports/musca_v5_maker_feasibility.json")
MINIMUM_INDEPENDENT_DAYS = 30
MINIMUM_COMPLETE_ORDERS = 100
MINIMUM_CLASS_OBSERVATIONS = 20
ADVERSE_COLUMNS = tuple(f"adverse_selection_bps_{seconds}s" for seconds in (1, 5, 30))

PROTOCOL = {
    "name": "musca_v5_bitunix_observed_maker_feasibility_v1",
    "symbol": "BTCUSDT",
    "minimum_independent_execution_days": MINIMUM_INDEPENDENT_DAYS,
    "minimum_complete_post_only_orders": MINIMUM_COMPLETE_ORDERS,
    "minimum_filled_and_unfilled_orders": MINIMUM_CLASS_OBSERVATIONS,
    "required_adverse_horizons_seconds": [1, 5, 30],
    "fill_truth": "private_Bitunix_orders_and_trades_only",
    "public_l2_role": "features_only_not_fill_truth",
    "historical_fill_simulation": False,
    "official_sources": {
        "bitunix_post_only": (
            "https://www.bitunix.com/api-docs/futures/trade/place_order.html"
        ),
        "bitunix_history_orders": (
            "https://www.bitunix.com/api-docs/futures/trade/get_history_orders.html"
        ),
        "bitunix_history_trades": (
            "https://www.bitunix.com/api-docs/futures/trade/get_history_trades.html"
        ),
        "bitunix_private_order_channel": (
            "https://www.bitunix.com/api-docs/futures/websocket/private/Order%20Channel.html"
        ),
        "binance_public_archive": (
            "https://github.com/binance/binance-public-data/blob/master/README.md"
        ),
    },
    "live_orders_enabled": False,
    "holdout_opened": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def l2_coverage(
    materialized: Path = BITUNIX_L2,
    raw_root: Path = BITUNIX_RAW,
) -> dict[str, Any]:
    raw_days = {
        path.stem.removeprefix("btcusdt_")
        for suffix in ("*.parquet", "*.jsonl")
        for path in raw_root.glob(f"btcusdt_{suffix}")
    }
    if not materialized.exists():
        return {
            "observed_raw_days": len(raw_days),
            "materialized_rows": 0,
            "materialized_days": 0,
            "valid_rows": 0,
            "valid_fraction": 0.0,
            "start": None,
            "end": None,
        }
    frame = pd.read_parquet(materialized, columns=["available_at", "feature_valid"])
    timestamps = pd.to_datetime(frame["available_at"], format="mixed", utc=True)
    valid = frame["feature_valid"].fillna(False).astype(bool)
    return {
        "observed_raw_days": len(raw_days),
        "materialized_rows": len(frame),
        "materialized_days": int(timestamps.dt.floor("D").nunique()),
        "valid_rows": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "start": timestamps.min().isoformat() if len(timestamps) else None,
        "end": timestamps.max().isoformat() if len(timestamps) else None,
    }


def observed_label_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "symbol",
        "created_at",
        "available_at",
        "fill_probability_target",
        "fill_fraction_target",
        "fill_latency_ms",
        "maker_only",
        "observation_complete",
        *ADVERSE_COLUMNS,
    }
    if missing := required - set(frame):
        raise ValueError(f"Observed maker labels missing columns: {sorted(missing)}")
    labels = frame.loc[frame["symbol"].astype(str).str.upper().eq("BTCUSDT")].copy()
    labels["created_at"] = pd.to_datetime(labels["created_at"], format="mixed", utc=True)
    labels["available_at"] = pd.to_datetime(labels["available_at"], format="mixed", utc=True)
    if labels["created_at"].isna().any() or labels["available_at"].isna().any():
        raise ValueError("Observed maker labels contain invalid timestamps")
    if labels["available_at"].lt(labels["created_at"]).any():
        raise ValueError("Observed maker label is available before order creation")
    probabilities = pd.to_numeric(labels["fill_probability_target"], errors="coerce")
    fractions = pd.to_numeric(labels["fill_fraction_target"], errors="coerce")
    if not probabilities.isin((0.0, 1.0)).all() or not fractions.between(0.0, 1.0).all():
        raise ValueError("Observed maker fill targets are outside their valid range")
    complete = labels.loc[labels["observation_complete"].fillna(False).astype(bool)]
    filled = complete["fill_probability_target"].eq(1.0)
    unfilled = complete["fill_probability_target"].eq(0.0)
    maker_fills = complete.loc[filled & complete["maker_only"].fillna(False).astype(bool)]
    fully_labeled = maker_fills.dropna(subset=["fill_latency_ms", *ADVERSE_COLUMNS])
    return {
        "rows": len(labels),
        "complete_orders": len(complete),
        "independent_days": int(complete["created_at"].dt.floor("D").nunique()),
        "filled_orders": int(filled.sum()),
        "unfilled_orders": int(unfilled.sum()),
        "maker_fills": len(maker_fills),
        "fully_labeled_maker_fills": len(fully_labeled),
        "causal_timestamp_violations": 0,
    }


def run(
    *,
    labels_path: Path = OBSERVED_LABELS,
    report_path: Path = REPORT,
    l2_path: Path = BITUNIX_L2,
    raw_root: Path = BITUNIX_RAW,
) -> dict[str, Any]:
    labels = (
        observed_label_coverage(pd.read_parquet(labels_path))
        if labels_path.exists()
        else {
            "rows": 0,
            "complete_orders": 0,
            "independent_days": 0,
            "filled_orders": 0,
            "unfilled_orders": 0,
            "maker_fills": 0,
            "fully_labeled_maker_fills": 0,
            "causal_timestamp_violations": 0,
        }
    )
    gates = {
        "execution_days_30": labels["independent_days"] >= MINIMUM_INDEPENDENT_DAYS,
        "complete_orders_100": labels["complete_orders"] >= MINIMUM_COMPLETE_ORDERS,
        "filled_orders_20": labels["filled_orders"] >= MINIMUM_CLASS_OBSERVATIONS,
        "unfilled_orders_20": labels["unfilled_orders"] >= MINIMUM_CLASS_OBSERVATIONS,
        "all_fills_observed_as_maker": (
            labels["filled_orders"] > 0 and labels["maker_fills"] == labels["filled_orders"]
        ),
        "all_maker_fills_have_latency_and_adverse": (
            labels["maker_fills"] > 0
            and labels["fully_labeled_maker_fills"] == labels["maker_fills"]
        ),
        "causal_timestamps": labels["causal_timestamp_violations"] == 0,
    }
    authorized = all(gates.values())
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "bitunix_public_l2": l2_coverage(l2_path, raw_root),
        "bitunix_private_post_only": labels,
        "gates": gates,
        "maker_training_authorized": authorized,
        "maker_cost_allowed_in_alpha_or_paper": False,
        "verdict": (
            "OBSERVED_MAKER_DATA_READY" if authorized else "COLLECTING_NO_OBSERVED_MAKER_MODEL"
        ),
        "changes_to_active_paper": False,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }
    _atomic_json(report_path, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
