from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from adaptive_bot.musca_v5_fine_tuning import _atomic_json
from adaptive_bot.musca_v5_post_only_upper_bound import (
    DATASET as BITUNIX_L2,
)
from adaptive_bot.musca_v5_post_only_upper_bound import (
    MAKER_EXIT_REPORT,
)

REPORT = Path("data/reports/musca_v5_l2_toxicity_model.json")
PREFILL_REPORT = Path("data/reports/musca_v5_prefill_lead_lag.json")
BINANCE_SIGNAL_REPORT = Path("data/reports/musca_v5_binance_l2_signal_validity.json")
BINANCE_L2 = Path("data/research/binance_l2/btcusdt_l2_features.parquet")
PROTOCOL = {
    "name": "musca_v5_vip5_preregistered_l2_toxicity_model_v1",
    "source": str(MAKER_EXIT_REPORT),
    "profile": "VIP5",
    "train": "2026-08-03_to_04",
    "selection": "2026-08-05",
    "audit": "2026-08-06_to_08",
    "ridge": {"alpha": 10.0, "champion_default": True},
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 10.0,
        "device": "cpu_diagnostic_dataset_is_small",
    },
    "probability": "logistic_target_before_timeout_l2",
    "decision": "predicted_net_ev_above_zero_and_probability_above_train_break_even",
    "xgb_champion_rule": "lower_selection_mae_and_positive_selection_policy_pf_1_15_n_20",
    "threshold_search": False,
    "holdout_opened": False,
    "paper_authority": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
PREFILL_PROTOCOL = {
    "name": "musca_v5_vip5_binance_bitunix_prefill_lead_lag_v1",
    "source": str(MAKER_EXIT_REPORT),
    "binance_l2": str(BINANCE_L2),
    "bitunix_l2": str(BITUNIX_L2),
    "profile": "VIP5",
    "observation_time": "touch_time_minus_250ms_cancel_latency",
    "join": "available_at_backward_only",
    "maximum_age_seconds": {"binance": 5, "bitunix": 2},
    "train": "2026-08-06",
    "selection": "2026-08-07",
    "audit": "2026-08-08",
    "ridge": {"alpha": 10.0, "champion_default": True},
    "xgboost": PROTOCOL["xgboost"],
    "probability": "logistic_target_before_timeout_l2",
    "decision": "predicted_net_ev_above_zero_and_probability_above_train_break_even",
    "threshold_search": False,
    "timing_upper_bound": "live_does_not_know_future_touch_time",
    "paper_authority": False,
    "holdout_opened": False,
}
PREFILL_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PREFILL_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
BINANCE_SIGNAL_HORIZONS = (5, 30, 60)
BINANCE_SIGNAL_COST_BPS = 3.0
BINANCE_SIGNAL_PROTOCOL = {
    "name": "musca_v5_binance_l2_instrument_validity_v1",
    "source": str(BINANCE_L2),
    "horizons_seconds": list(BINANCE_SIGNAL_HORIZONS),
    "cadence": "max_30s_horizon_non_overlapping",
    "train": "2026-08-06",
    "selection": "2026-08-07",
    "audit": "2026-08-08",
    "purge": "label_available_at_inside_period",
    "ridge": {"alpha": 10.0, "champion_default": True},
    "xgboost": PROTOCOL["xgboost"],
    "entry_rule": "absolute_predicted_return_above_3bps",
    "cost_bps": BINANCE_SIGNAL_COST_BPS,
    "threshold_search": False,
    "paper_authority": False,
    "holdout_opened": False,
}
BINANCE_SIGNAL_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(BINANCE_SIGNAL_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

DIRECTIONAL = (
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_15s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "rolling_vwap_5m_distance_bps",
    "rolling_vwap_slope_bps_60s",
    "rolling_vwap_slope_change_bps",
)
NONDIRECTIONAL = (
    "trade_arrival_rate_30s",
    "range_60s_bps",
    "realized_volatility",
    "volatility_percentile",
    "spread_bps",
    "time_since_last_vwap_cross_seconds",
)
PREFILL_DIRECTIONAL = (
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_15s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "mid_return_5s_bps",
    "mid_return_30s_bps",
    "rolling_vwap_5m_distance_bps",
)
PREFILL_NONDIRECTIONAL = (
    "spread_bps",
    "trade_arrival_rate_30s",
    "range_60s_bps",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
)
BINANCE_SIGNAL_FEATURES = tuple(
    dict.fromkeys(
        (
            *PREFILL_DIRECTIONAL,
            *PREFILL_NONDIRECTIONAL,
            "realized_volatility",
            "volatility_percentile",
        )
    )
)


def feature_frame(trades: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows: list[dict[str, Any]] = []
    returns: list[float] = []
    targets: list[int] = []
    for trade in trades:
        side = 1.0 if trade.get("side") == "long" else -1.0
        decision_at = pd.Timestamp(trade["decision_at"])
        row: dict[str, object] = {
            f"signed_{name}": _number(trade.get(f"decision_{name}")) * side
            for name in DIRECTIONAL
        }
        row.update(
            {name: _number(trade.get(f"decision_{name}")) for name in NONDIRECTIONAL}
        )
        bid_cancel = _number(trade.get("decision_bid_cancel_rate_5s"))
        ask_cancel = _number(trade.get("decision_ask_cancel_rate_5s"))
        row.update(
            {
                "signed_cancel_balance": (ask_cancel - bid_cancel) * side,
                "side": side,
                "hour_sin": float(np.sin(2 * np.pi * decision_at.hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * decision_at.hour / 24)),
                "decision_at": decision_at,
            }
        )
        rows.append(row)
        returns.append(float(trade["net_bps"]))
        targets.append(int(trade["reason"] == "VWAP_TARGET"))
    return pd.DataFrame(rows), pd.Series(returns, dtype=float), pd.Series(targets, dtype=int)


def run(source: Path = MAKER_EXIT_REPORT, report_path: Path = REPORT) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    trades = payload["profiles"]["VIP5"]["trades"]
    features, returns, targets = feature_frame(trades)
    timestamps = pd.to_datetime(features.pop("decision_at"), utc=True)
    train = timestamps.lt(pd.Timestamp("2026-08-05T00:00:00Z"))
    selection = timestamps.ge(pd.Timestamp("2026-08-05T00:00:00Z")) & timestamps.lt(
        pd.Timestamp("2026-08-06T00:00:00Z")
    )
    audit = timestamps.ge(pd.Timestamp("2026-08-06T00:00:00Z"))
    if min(int(train.sum()), int(selection.sum()), int(audit.sum())) < 20:
        raise ValueError("L2 toxicity split has fewer than 20 trades")
    ridge = TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(10.0)),
        transformer=StandardScaler(),
    )
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        n_jobs=6,
        random_state=42,
    )
    probability = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=42),
    )
    ridge.fit(features.loc[train], returns.loc[train])
    xgb.fit(features.loc[train], returns.loc[train])
    probability.fit(features.loc[train], targets.loc[train])
    wins = returns.loc[train & targets.eq(1)]
    misses = returns.loc[train & targets.eq(0)]
    break_even = float(-misses.mean() / (wins.mean() - misses.mean()))
    predictions = {
        "ridge": np.asarray(ridge.predict(features), dtype=float),
        "xgboost": np.asarray(xgb.predict(features), dtype=float),
    }
    target_probability = probability.predict_proba(features)[:, 1]
    model_reports = {
        name: _model_report(
            predicted,
            target_probability,
            returns.to_numpy(float),
            selection.to_numpy(bool),
            audit.to_numpy(bool),
            break_even,
        )
        for name, predicted in predictions.items()
    }
    ridge_selection = model_reports["ridge"]["selection"]
    xgb_selection = model_reports["xgboost"]["selection"]
    xgb_allowed = bool(
        model_reports["xgboost"]["selection_mae"]
        < model_reports["ridge"]["selection_mae"]
        and xgb_selection["trades"] >= 20
        and (xgb_selection["expectancy_bps"] or -1) > 0
        and (xgb_selection["profit_factor"] or 0) >= 1.15
    )
    champion = "xgboost" if xgb_allowed else "ridge"
    selected = model_reports[champion]
    gate = bool(
        ridge_selection["trades"] >= 20
        and selected["selection"]["trades"] >= 20
        and selected["audit"]["trades"] >= 20
        and (selected["selection"]["expectancy_bps"] or -1) > 0
        and (selected["audit"]["expectancy_bps"] or -1) > 0
        and (selected["selection"]["profit_factor"] or 0) >= 1.15
        and (selected["audit"]["profit_factor"] or 0) >= 1.15
    )
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "source_protocol_hash": payload["protocol_hash"],
        "rows": len(features),
        "split_rows": {
            "train": int(train.sum()),
            "selection": int(selection.sum()),
            "audit": int(audit.sum()),
        },
        "features": list(features.columns),
        "train_break_even_target_probability": break_even,
        "probability_brier": {
            "selection": float(
                brier_score_loss(targets.loc[selection], target_probability[selection])
            ),
            "audit": float(brier_score_loss(targets.loc[audit], target_probability[audit])),
        },
        "models": model_reports,
        "champion": champion,
        "xgboost_allowed": xgb_allowed,
        "economic_gate": gate,
        "verdict": "L2_TOXICITY_SIGNAL_FOUND" if gate else "NO_CAUSAL_L2_TOXICITY_MODEL",
        "paper_change_authorized": False,
        "holdout_opened": False,
    }
    _atomic_json(report_path, report)
    return report


def prefill_feature_frame(
    source: Path = MAKER_EXIT_REPORT,
    binance_path: Path = BINANCE_L2,
    bitunix_path: Path = BITUNIX_L2,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, int]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    trades = pd.DataFrame(payload["profiles"]["VIP5"]["trades"])
    trades["entered_at"] = pd.to_datetime(trades["entered_at"], utc=True, format="mixed")
    trades["query_at"] = trades["entered_at"] - pd.Timedelta(milliseconds=250)
    trades["side_sign"] = trades["side"].map({"long": 1.0, "short": -1.0})
    columns = [
        "available_at",
        "feature_valid",
        "mid",
        *PREFILL_DIRECTIONAL,
        *PREFILL_NONDIRECTIONAL,
    ]
    columns = list(dict.fromkeys(columns))
    binance = _venue_frame(binance_path, columns, "binance")
    bitunix = _venue_frame(bitunix_path, columns, "bitunix")
    joined = pd.merge_asof(
        trades.sort_values("query_at"),
        binance,
        left_on="query_at",
        right_on="binance_available_at",
        direction="backward",
    )
    joined = pd.merge_asof(
        joined.sort_values("query_at"),
        bitunix,
        left_on="query_at",
        right_on="bitunix_available_at",
        direction="backward",
    )
    binance_age = (joined["query_at"] - joined["binance_available_at"]).dt.total_seconds()
    bitunix_age = (joined["query_at"] - joined["bitunix_available_at"]).dt.total_seconds()
    missing_source = joined[["binance_available_at", "bitunix_available_at"]].isna().any(axis=1)
    causal = binance_age.ge(0) & bitunix_age.ge(0)
    covered = (
        causal
        & binance_age.le(5)
        & bitunix_age.le(2)
        & joined["binance_feature_valid"].fillna(False).astype(bool)
        & joined["bitunix_feature_valid"].fillna(False).astype(bool)
    )
    joined = joined.loc[covered].reset_index(drop=True)
    side = joined["side_sign"].to_numpy(float)
    feature_data: dict[str, Any] = {
        "side": side,
        "quote_age_seconds": pd.to_numeric(
            joined["time_to_touch_seconds"], errors="coerce"
        ).to_numpy(float),
        "hour_sin": np.sin(2 * np.pi * joined["query_at"].dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * joined["query_at"].dt.hour / 24),
    }
    for venue in ("binance", "bitunix"):
        for name in PREFILL_DIRECTIONAL:
            feature_data[f"signed_{venue}_{name}"] = (
                pd.to_numeric(joined[f"{venue}_{name}"], errors="coerce").to_numpy(float)
                * side
            )
        for name in PREFILL_NONDIRECTIONAL:
            feature_data[f"{venue}_{name}"] = pd.to_numeric(
                joined[f"{venue}_{name}"], errors="coerce"
            ).to_numpy(float)
    feature_data["signed_cross_venue_basis_bps"] = (
        (joined["binance_mid"] / joined["bitunix_mid"] - 1).to_numpy(float)
        * 10_000
        * side
    )
    for name in ("mid_return_5s_bps", "mid_return_30s_bps", "aggressive_imbalance_5s"):
        feature_data[f"signed_cross_venue_{name}_difference"] = (
            (
                pd.to_numeric(joined[f"binance_{name}"], errors="coerce")
                - pd.to_numeric(joined[f"bitunix_{name}"], errors="coerce")
            ).to_numpy(float)
            * side
        )
    features = pd.DataFrame(feature_data)
    complete = features.notna().all(axis=1)
    missing_features = int((~complete).sum())
    features = features.loc[complete].reset_index(drop=True)
    joined = joined.loc[complete].reset_index(drop=True)
    returns = pd.to_numeric(joined["net_bps"], errors="coerce").reset_index(drop=True)
    targets = joined["reason"].eq("VWAP_TARGET").astype(int).reset_index(drop=True)
    timestamps = joined["entered_at"].reset_index(drop=True)
    coverage = {
        "source_trades": len(trades),
        "covered_trades": len(joined),
        "missing_source_matches": int(missing_source.sum()),
        "stale_or_missing_source_matches": int((~covered).sum()),
        "missing_feature_trades": missing_features,
        "causality_violations": int((binance_age.lt(0) | bitunix_age.lt(0)).sum()),
    }
    return features, returns, targets, timestamps, coverage


def run_prefill_lead_lag(report_path: Path = PREFILL_REPORT) -> dict[str, Any]:
    features, returns, targets, timestamps, coverage = prefill_feature_frame()
    train = timestamps.lt(pd.Timestamp("2026-08-07T00:00:00Z"))
    selection = timestamps.ge(pd.Timestamp("2026-08-07T00:00:00Z")) & timestamps.lt(
        pd.Timestamp("2026-08-08T00:00:00Z")
    )
    audit = timestamps.ge(pd.Timestamp("2026-08-08T00:00:00Z")) & timestamps.lt(
        pd.Timestamp("2026-08-09T00:00:00Z")
    )
    if min(int(train.sum()), int(selection.sum()), int(audit.sum())) < 20:
        raise ValueError("pre-fill lead-lag split has fewer than 20 covered trades")
    ridge = TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(10.0)),
        transformer=StandardScaler(),
    )
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        n_jobs=6,
        random_state=42,
    )
    probability = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=42),
    )
    ridge.fit(features.loc[train], returns.loc[train])
    xgb.fit(features.loc[train], returns.loc[train])
    probability.fit(features.loc[train], targets.loc[train])
    wins = returns.loc[train & targets.eq(1)]
    misses = returns.loc[train & targets.eq(0)]
    break_even = float(-misses.mean() / (wins.mean() - misses.mean()))
    predictions = {
        "ridge": np.asarray(ridge.predict(features), dtype=float),
        "xgboost": np.asarray(xgb.predict(features), dtype=float),
    }
    target_probability = probability.predict_proba(features)[:, 1]
    model_reports = {
        name: _model_report(
            predicted,
            target_probability,
            returns.to_numpy(float),
            selection.to_numpy(bool),
            audit.to_numpy(bool),
            break_even,
        )
        for name, predicted in predictions.items()
    }
    xgb_selection = model_reports["xgboost"]["selection"]
    xgb_allowed = bool(
        model_reports["xgboost"]["selection_mae"]
        < model_reports["ridge"]["selection_mae"]
        and xgb_selection["trades"] >= 20
        and (xgb_selection["expectancy_bps"] or -1) > 0
        and (xgb_selection["profit_factor"] or 0) >= 1.15
    )
    champion = "xgboost" if xgb_allowed else "ridge"
    selected = model_reports[champion]
    gate = all(
        period["trades"] >= 20
        and (period["expectancy_bps"] or -1) > 0
        and (period["profit_factor"] or 0) >= 1.15
        for period in (selected["selection"], selected["audit"])
    )
    report = {
        "protocol": PREFILL_PROTOCOL,
        "protocol_hash": PREFILL_PROTOCOL_HASH,
        "source_protocol_hash": json.loads(
            MAKER_EXIT_REPORT.read_text(encoding="utf-8")
        )["protocol_hash"],
        "coverage": coverage,
        "split_rows": {
            "train": int(train.sum()),
            "selection": int(selection.sum()),
            "audit": int(audit.sum()),
        },
        "features": list(features.columns),
        "train_break_even_target_probability": break_even,
        "probability_brier": {
            "selection": float(
                brier_score_loss(targets.loc[selection], target_probability[selection])
            ),
            "audit": float(brier_score_loss(targets.loc[audit], target_probability[audit])),
        },
        "models": model_reports,
        "champion": champion,
        "xgboost_allowed": xgb_allowed,
        "economic_gate": gate,
        "verdict": (
            "PREFILL_LEAD_LAG_SIGNAL_FOR_CONTINUOUS_REPLAY"
            if gate
            else "NO_PREFILL_LEAD_LAG_SIGNAL"
        ),
        "timing_upper_bound": True,
        "paper_change_authorized": False,
        "holdout_opened": False,
    }
    _atomic_json(report_path, report)
    return report


def run_binance_signal_validity(
    source: Path = BINANCE_L2,
    report_path: Path = BINANCE_SIGNAL_REPORT,
) -> dict[str, Any]:
    label_columns = [
        item
        for horizon in BINANCE_SIGNAL_HORIZONS
        for item in (
            f"future_return_{horizon}s_bps",
            f"label_available_at_{horizon}s",
        )
    ]
    frame = pd.read_parquet(
        source,
        columns=["available_at", "feature_valid", *BINANCE_SIGNAL_FEATURES, *label_columns],
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, format="mixed"
    )
    frame = frame.loc[frame["feature_valid"].fillna(False).astype(bool)].copy()
    features = frame[list(BINANCE_SIGNAL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    features["hour_sin"] = np.sin(2 * np.pi * frame["available_at"].dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * frame["available_at"].dt.hour / 24)
    complete = features.notna().all(axis=1)
    frame = frame.loc[complete].copy()
    features = features.loc[complete].copy()
    horizons: dict[str, Any] = {}
    for horizon in BINANCE_SIGNAL_HORIZONS:
        cadence = max(30, horizon)
        sampled = frame.assign(
            bucket=frame["available_at"].dt.floor(f"{cadence}s")
        ).drop_duplicates("bucket", keep="first")
        horizon_features = features.loc[sampled.index].reset_index(drop=True)
        timestamps = sampled["available_at"].reset_index(drop=True)
        label_available = pd.to_datetime(
            sampled[f"label_available_at_{horizon}s"], utc=True, format="mixed"
        ).reset_index(drop=True)
        returns = pd.to_numeric(
            sampled[f"future_return_{horizon}s_bps"], errors="coerce"
        ).reset_index(drop=True)
        valid = returns.notna() & label_available.notna() & label_available.ge(timestamps)
        horizon_features = horizon_features.loc[valid].reset_index(drop=True)
        timestamps = timestamps.loc[valid].reset_index(drop=True)
        label_available = label_available.loc[valid].reset_index(drop=True)
        returns = returns.loc[valid].reset_index(drop=True)
        train = timestamps.lt(pd.Timestamp("2026-08-07T00:00:00Z")) & label_available.lt(
            pd.Timestamp("2026-08-07T00:00:00Z")
        )
        selection = (
            timestamps.ge(pd.Timestamp("2026-08-07T00:00:00Z"))
            & timestamps.lt(pd.Timestamp("2026-08-08T00:00:00Z"))
            & label_available.lt(pd.Timestamp("2026-08-08T00:00:00Z"))
        )
        audit = (
            timestamps.ge(pd.Timestamp("2026-08-08T00:00:00Z"))
            & timestamps.lt(pd.Timestamp("2026-08-09T00:00:00Z"))
            & label_available.lt(pd.Timestamp("2026-08-09T00:00:00Z"))
        )
        if min(int(train.sum()), int(selection.sum()), int(audit.sum())) < 50:
            raise ValueError(f"Binance L2 {horizon}s split has fewer than 50 rows")
        ridge = TransformedTargetRegressor(
            regressor=make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(10.0)
            ),
            transformer=StandardScaler(),
        )
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            n_jobs=6,
            random_state=42,
        )
        ridge.fit(horizon_features.loc[train], returns.loc[train])
        xgb.fit(horizon_features.loc[train], returns.loc[train])
        predictions = {
            "ridge": np.asarray(ridge.predict(horizon_features), dtype=float),
            "xgboost": np.asarray(xgb.predict(horizon_features), dtype=float),
        }
        model_reports = {
            name: _directional_model_report(
                predicted,
                returns.to_numpy(float),
                selection.to_numpy(bool),
                audit.to_numpy(bool),
            )
            for name, predicted in predictions.items()
        }
        ridge_selection = model_reports["ridge"]["selection"]
        xgb_selection = model_reports["xgboost"]["selection"]
        xgb_allowed = bool(
            model_reports["xgboost"]["selection_mae"]
            < model_reports["ridge"]["selection_mae"]
            and xgb_selection["trades"] >= 50
            and (xgb_selection["expectancy_bps"] or -1) > 0
            and (xgb_selection["profit_factor"] or 0) >= 1.15
        )
        champion = "xgboost" if xgb_allowed else "ridge"
        selected = model_reports[champion]
        gate = bool(
            selected["selection_mae"] < selected["zero_selection_mae"]
            and selected["audit_mae"] < selected["zero_audit_mae"]
            and all(
                period["trades"] >= 50
                and (period["expectancy_bps"] or -1) > 0
                and (period["profit_factor"] or 0) >= 1.15
                for period in (selected["selection"], selected["audit"])
            )
        )
        horizons[str(horizon)] = {
            "cadence_seconds": cadence,
            "split_rows": {
                "train": int(train.sum()),
                "selection": int(selection.sum()),
                "audit": int(audit.sum()),
            },
            "models": model_reports,
            "champion": champion,
            "xgboost_allowed": xgb_allowed,
            "economic_gate": gate,
            "ridge_selection_trades": ridge_selection["trades"],
        }
    report = {
        "protocol": BINANCE_SIGNAL_PROTOCOL,
        "protocol_hash": BINANCE_SIGNAL_PROTOCOL_HASH,
        "source": str(source),
        "source_rows": len(frame),
        "excluded_missing_features": int((~complete).sum()),
        "features": list(features.columns),
        "horizons": horizons,
        "verdict": (
            "BINANCE_L2_INSTRUMENT_SIGNAL_FOUND"
            if any(item["economic_gate"] for item in horizons.values())
            else "NO_BINANCE_L2_INSTRUMENT_SIGNAL"
        ),
        "paper_change_authorized": False,
        "holdout_opened": False,
    }
    _atomic_json(report_path, report)
    return report


def _venue_frame(path: Path, columns: list[str], prefix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, format="mixed"
    )
    return (
        frame.sort_values("available_at")
        .rename(columns={column: f"{prefix}_{column}" for column in columns})
        .drop_duplicates(f"{prefix}_available_at", keep="last")
    )


def _model_report(
    prediction: np.ndarray,
    target_probability: np.ndarray,
    returns: np.ndarray,
    selection: np.ndarray,
    audit: np.ndarray,
    break_even: float,
) -> dict[str, Any]:
    accepted = (prediction > 0) & (target_probability >= break_even)
    return {
        "selection_mae": float(mean_absolute_error(returns[selection], prediction[selection])),
        "audit_mae": float(mean_absolute_error(returns[audit], prediction[audit])),
        "selection": _metrics(returns[selection & accepted]),
        "audit": _metrics(returns[audit & accepted]),
        "predicted_positive_selection": int((selection & accepted).sum()),
        "predicted_positive_audit": int((audit & accepted).sum()),
    }


def _directional_model_report(
    prediction: np.ndarray,
    returns: np.ndarray,
    selection: np.ndarray,
    audit: np.ndarray,
) -> dict[str, Any]:
    accepted = np.abs(prediction) > BINANCE_SIGNAL_COST_BPS
    realized = np.sign(prediction) * returns - BINANCE_SIGNAL_COST_BPS
    return {
        "selection_mae": float(mean_absolute_error(returns[selection], prediction[selection])),
        "audit_mae": float(mean_absolute_error(returns[audit], prediction[audit])),
        "zero_selection_mae": float(
            mean_absolute_error(returns[selection], np.zeros(selection.sum()))
        ),
        "zero_audit_mae": float(mean_absolute_error(returns[audit], np.zeros(audit.sum()))),
        "selection": _metrics(realized[selection & accepted]),
        "audit": _metrics(realized[audit & accepted]),
    }


def _metrics(values: np.ndarray) -> dict[str, float | int | None]:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "trades": len(values),
        "expectancy_bps": float(values.mean()) if len(values) else None,
        "profit_factor": gains / losses if losses else None,
        "positive_fraction": float((values > 0).mean()) if len(values) else None,
    }


def _number(value: object) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
