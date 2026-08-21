from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_bot.bitunix_fees import FUTURES_VIP_FEE_BPS
from adaptive_bot.musca_v5_event_policy import (
    MAXIMUM_MICRO_STOP_BPS,
    MINIMUM_STOP_BPS,
    POLICY_CROSS_FEATURES,
    POLICY_MARKET_FEATURES,
    _attach_cross_exchange,
    _attach_market_state,
)
from adaptive_bot.musca_v5_micro_model import (
    DIRECTIONAL_ONE_SECOND_FEATURES,
    ONE_SECOND_MICRO_FEATURES,
    build_one_second_features,
)
from adaptive_bot.musca_v5_micro_model import (
    build_micro_features as build_five_second_features,
)
from adaptive_bot.musca_v5_microstructure import ROOT as AGGTRADES_ROOT
from adaptive_bot.musca_v5_research import EVENTS, ML_FEATURES

ROOT = Path("data/ml/musca_v5")
MATRIX = ROOT / "economic_alpha_matrix.parquet"
BUNDLE = Path("data/models/musca_v5/economic_alpha.joblib")
REPORT = Path("data/reports/musca_v5_economic_alpha.json")
STATUS = Path("data/reports/musca_v5_economic_alpha.status.json")
WALK_FORWARD_DIR = ROOT / "walk_forward"

TARGETS_BPS = (30, 40, 50, 70, 100)
HORIZONS_MINUTES = (5, 15, 30, 60)
PLANS = tuple((target, horizon) for target in TARGETS_BPS for horizon in HORIZONS_MINUTES)
DISCOVERY_COVERAGE = 0.01
SLIPPAGE_RESERVE_BPS = 1.0
MINIMUM_TARGET_TO_COST = 3.0
PATH_BUCKETS_PER_MINUTE = 60
WALK_FORWARD_CALIBRATION_DAYS = 14
WALK_FORWARD_SELECTION_DAYS = 14
WALK_FORWARD_TEST_DAYS = 7
TRAINING_WINDOW_DAYS = 84
HOLDOUT_START = pd.Timestamp("2026-07-01T00:00:00Z")
FIT_END = pd.Timestamp("2026-05-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2026-05-16T00:00:00Z")
AUDIT_START = pd.Timestamp("2026-06-01T00:00:00Z")
AUDIT_END = HOLDOUT_START
TRAINING_MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 7))
ENTRY_CONFIRMATION_BUCKETS_5S = 60
MICRO_STOP_LOOKBACK_SECONDS = 180

EVENT_FAMILIES = (
    "DAILY_FADE",
    "DAILY_RECLAIM",
    "ROLLING_4H_FADE",
    "ROLLING_4H_RETEST",
    "SWING_AVWAP_RETEST",
)
EVENT_FAMILY_FEATURES = tuple(
    f"event_family_{family.lower()}" for family in EVENT_FAMILIES
)
DIRECTIONAL_MARKET_FEATURES = (
    "trend_vote",
    "return_1h",
    "return_4h",
    "ema_spread_atr",
    "taker_imbalance_15m",
    "spot_taker_imbalance_15m",
    "distance_vwap_atr",
    "funding_z",
    "basis_bps",
    "return_oi_interaction",
)
BASE_ALPHA_FEATURES = tuple(
    feature for feature in ML_FEATURES if feature != "event_family_code"
)

FEATURES = tuple(
    dict.fromkeys(
        (
            *BASE_ALPHA_FEATURES,
            "event_direction_alignment",
            *EVENT_FAMILY_FEATURES,
            *ONE_SECOND_MICRO_FEATURES,
            *POLICY_CROSS_FEATURES,
            *POLICY_MARKET_FEATURES,
        )
    )
)

PROTOCOL = {
    "name": "musca_v5_neutral_vwap_economic_alpha",
    "asset": "BTCUSDT",
    "feature_venue": "Binance USD-M official aggTrades plus causal market context",
    "execution_venue": "Bitunix futures VIP configurable shadow",
    "actions": "both LONG and SHORT for every neutral VWAP event, plus implicit WAIT",
    "action_relative_features": (
        "signed market and flow inputs are aligned to the candidate side; original event "
        "direction agreement is explicit; event family uses deterministic one-hot encoding"
    ),
    "entry": "first causal 5-second restart within five minutes; execute next 1s open",
    "path": "entry second included; first exit timestamp releases the single-position lock",
    "targets_bps": list(TARGETS_BPS),
    "horizons_minutes": list(HORIZONS_MINUTES),
    "outcomes": ["TARGET", "STOP", "TIMEOUT"],
    "same_bucket": "STOP wins",
    "stop": "previous 3-minute micro extreme plus range buffer; 12..60 bps; never widens",
    "selection_coverage": DISCOVERY_COVERAGE,
    "selection_note": "1% preregistered from January-March discovery; not retuned on audit",
    "walk_forward": (
        "rolling 12-week fit; 14d probability calibration; 14d threshold selection; "
        "7d test; 60m purge"
    ),
    "costs": (
        "conservative taker entry and taker exit at the selected Bitunix VIP level; "
        "1 bp round-trip slippage reserve; observed Bitunix spread/depth/funding are "
        "additional fail-closed shadow gates"
    ),
    "economic_plan_gate": (
        "a target is eligible for a VIP profile only when its gross distance is at "
        "least three times that profile's expected round-trip cost"
    ),
    "maker_execution": (
        "disabled for Alpha economics until private Bitunix POST_ONLY fill labels are sufficient"
    ),
    "funding": (
        "not inferred from Binance for Bitunix; the Bitunix shadow economic gate must add "
        "the observed expected funding before entry"
    ),
    "vip_profiles": [f"VIP{level}" for level in range(6)],
    "fit_end": FIT_END.isoformat(),
    "probability_calibration_end": CALIBRATION_END.isoformat(),
    "audit": [AUDIT_START.isoformat(), AUDIT_END.isoformat()],
    "sealed_holdout_start": HOLDOUT_START.isoformat(),
    "features": list(FEATURES),
    "historical_depth_in_alpha": False,
    "depth_transfer": (
        "official +/-1/2/5% history is diagnostic-only because Binance live depth1000 "
        "covered about +/-0.18% at audit; Binance/Bitunix live depth is an execution veto"
    ),
    "real_capital_allowed": False,
}
PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _status(phase: str, percent: float, detail: str) -> None:
    _atomic_json(
        STATUS,
        {
            "phase": phase,
            "percent": round(percent, 2),
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _features(rows: pd.DataFrame) -> np.ndarray:
    values = np.asarray(rows.loc[:, FEATURES].to_numpy(float), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Musca V5 Alpha features must be finite and causally covered")
    return cast(np.ndarray, values)


def _load_training_1s() -> pd.DataFrame:
    """Load January-June only; the July archive remains a sealed future holdout."""
    frames = [
        pd.read_parquet(AGGTRADES_ROOT / f"BTCUSDT-aggTrades-1s-{month}.parquet")
        for month in TRAINING_MONTHS
    ]
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    return result.reset_index(drop=True)


def _load_training_5s() -> pd.DataFrame:
    frames = [
        pd.read_parquet(AGGTRADES_ROOT / f"BTCUSDT-aggTrades-5s-{month}.parquet")
        for month in TRAINING_MONTHS
    ]
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    return result.reset_index(drop=True)


def _costs(vip_level: int, multiplier: float = 1.0) -> tuple[float, float]:
    if multiplier <= 0:
        raise ValueError("Cost multiplier must be positive")
    _, taker = FUTURES_VIP_FEE_BPS[vip_level]
    round_trip = (2 * taker + SLIPPAGE_RESERVE_BPS) * multiplier
    return round_trip, round_trip


def _encode_action_context(
    record: dict[str, Any],
    *,
    side: int,
    event_direction: int,
    event_family: str,
) -> None:
    """Encode signed state relative to the action without losing event intent."""
    record["event_direction_alignment"] = float(side * event_direction)
    for family, feature in zip(
        EVENT_FAMILIES, EVENT_FAMILY_FEATURES, strict=True
    ):
        record[feature] = float(event_family == family)
    for feature in DIRECTIONAL_MARKET_FEATURES:
        record[feature] = side * float(record[feature])


def _candidate_actions(
    data: pd.DataFrame,
    events: pd.DataFrame,
    confirmation_data: pd.DataFrame,
) -> pd.DataFrame:
    timestamp = pd.to_datetime(data["timestamp"], utc=True).to_numpy(
        dtype="datetime64[ns]"
    )
    timestamp_ns: np.ndarray = np.asarray(timestamp.astype("int64"), dtype=np.int64)
    available = pd.to_datetime(data["available_at"], utc=True).to_numpy(
        dtype="datetime64[ns]"
    )
    available_ns: np.ndarray = np.asarray(available.astype("int64"), dtype=np.int64)
    confirmation_available_ns: np.ndarray = np.asarray(
        pd.to_datetime(confirmation_data["available_at"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64"),
        dtype=np.int64,
    )
    event_ns: np.ndarray = np.asarray(
        pd.to_datetime(events["available_at"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64"),
        dtype=np.int64,
    )
    first_indexes = cast(
        np.ndarray, np.searchsorted(confirmation_available_ns, event_ns)
    )
    high = data["high"].to_numpy(float)
    low = data["low"].to_numpy(float)
    open_price = data["open"].to_numpy(float)
    close = data["close"].to_numpy(float)
    bucket_range = (high - low) / close * 10_000
    range_median = (
        pd.Series(bucket_range)
        .shift(1)
        .rolling(MICRO_STOP_LOOKBACK_SECONDS, min_periods=60)
        .median()
        .to_numpy()
    )
    previous_low = (
        pd.Series(low)
        .shift(1)
        .rolling(MICRO_STOP_LOOKBACK_SECONDS, min_periods=60)
        .min()
        .to_numpy()
    )
    previous_high = (
        pd.Series(high)
        .shift(1)
        .rolling(MICRO_STOP_LOOKBACK_SECONDS, min_periods=60)
        .max()
        .to_numpy()
    )
    price_velocity = confirmation_data["price_velocity_15s"].to_numpy(float)
    ofi_15s = confirmation_data["ofi_15s"].to_numpy(float)
    ofi_1m = confirmation_data["ofi_1m"].to_numpy(float)
    intensity = confirmation_data["trade_intensity_15s"].to_numpy(float)
    maximum_path = max(HORIZONS_MINUTES) * 60
    records: list[dict[str, Any]] = []
    for event, first in zip(
        events.to_dict("records"), first_indexes.tolist(), strict=True
    ):
        event_direction = int(event["direction"])
        event_family = str(event["event_family"])
        for side in (-1, 1):
            last = min(
                first + ENTRY_CONFIRMATION_BUCKETS_5S,
                len(confirmation_data) - 1,
            )
            indexes = np.arange(first, last + 1)
            confirmed = (
                (side * price_velocity[indexes] >= 0.5)
                & (side * ofi_15s[indexes] >= 0.05)
                & (side * ofi_1m[indexes] >= -0.02)
                & (intensity[indexes] >= 0.8)
            )
            if not confirmed.any():
                continue
            confirmation = int(indexes[np.flatnonzero(confirmed)[0]])
            confirmation_ns = int(confirmation_available_ns[confirmation])
            entry_index = int(np.searchsorted(timestamp_ns, confirmation_ns, side="left"))
            if entry_index + maximum_path > len(data):
                continue
            feature_index = int(
                np.searchsorted(available_ns, confirmation_ns, side="right") - 1
            )
            if feature_index < 0:
                continue
            entry = open_price[entry_index]
            extreme = previous_low[entry_index] if side > 0 else previous_high[entry_index]
            raw_stop = (
                side * (extreme / entry - 1) * 10_000
                - max(2.0, 0.25 * range_median[entry_index])
            )
            stop_bps = -max(MINIMUM_STOP_BPS, -raw_stop)
            if not np.isfinite(stop_bps) or stop_bps < -MAXIMUM_MICRO_STOP_BPS:
                continue
            record: dict[str, Any] = {str(key): value for key, value in event.items()}
            record.update({
                "signal_at": event["available_at"],
                "available_at": pd.Timestamp(confirmation_ns, unit="ns", tz="UTC"),
                "entry_timestamp": pd.Timestamp(
                    int(timestamp_ns[entry_index]), unit="ns", tz="UTC"
                ),
                "entry_index": entry_index,
                "entry_price": entry,
                "direction": side,
                "risk_bps": -stop_bps,
                "initial_stop_bps": stop_bps,
                "protocol_hash": PROTOCOL_HASH,
            })
            _encode_action_context(
                record,
                side=side,
                event_direction=event_direction,
                event_family=event_family,
            )
            confirmation_row = data.iloc[feature_index]
            for feature in ONE_SECOND_MICRO_FEATURES:
                value = float(confirmation_row[feature])
                if feature in DIRECTIONAL_ONE_SECOND_FEATURES:
                    value *= side
                record[feature] = value
            records.append(record)
    return pd.DataFrame(records)


def build_matrix(*, force: bool = False) -> pd.DataFrame:
    if MATRIX.exists() and not force:
        cached = pd.read_parquet(MATRIX)
        if len(cached) and cached["protocol_hash"].eq(PROTOCOL_HASH).all():
            return cached
    _status("matrix", 2, "Loading checksum-verified Binance 1-second aggTrades")
    raw = _load_training_1s()
    micro = build_one_second_features(raw)
    data = (
        raw.merge(micro, on="available_at", how="inner", validate="one_to_one")
        .sort_values("available_at")
        .reset_index(drop=True)
    )
    raw_5s = _load_training_5s()
    micro_5s = build_five_second_features(raw_5s)
    confirmation_data = (
        raw_5s.merge(micro_5s, on="available_at", how="inner", validate="one_to_one")
        .sort_values("available_at")
        .reset_index(drop=True)
    )
    source_events = pd.read_parquet(EVENTS)
    source_events["available_at"] = pd.to_datetime(source_events["available_at"], utc=True)
    start = pd.Timestamp(data["available_at"].min())
    end = pd.Timestamp(data["available_at"].max())
    source_events = (
        source_events.loc[
            source_events["available_at"].ge(start)
            & source_events["available_at"].le(end)
            & source_events["available_at"].lt(
                AUDIT_END - pd.Timedelta(minutes=max(HORIZONS_MINUTES))
            )
        ]
        .sort_values("available_at")
        .drop_duplicates(["available_at", "event_family"])
    )
    actions = _candidate_actions(data, source_events, confirmation_data)
    if actions.empty:
        raise ValueError("No causally confirmed neutral VWAP actions")
    _status("matrix", 20, f"{len(actions):,} LONG/SHORT actions; vectorized paths")
    indexes = actions["entry_index"].to_numpy(int)
    entry = actions["entry_price"].to_numpy(float)
    side = actions["direction"].to_numpy(float)
    stop = actions["initial_stop_bps"].to_numpy(float)
    maximum_path = max(HORIZONS_MINUTES) * PATH_BUCKETS_PER_MINUTE
    high = np.lib.stride_tricks.sliding_window_view(
        data["high"].to_numpy(float), maximum_path
    )[indexes]
    low = np.lib.stride_tricks.sliding_window_view(
        data["low"].to_numpy(float), maximum_path
    )[indexes]
    favorable = np.where(
        side[:, None] > 0,
        (high / entry[:, None] - 1) * 10_000,
        (entry[:, None] / low - 1) * 10_000,
    )
    adverse = np.where(
        side[:, None] > 0,
        (entry[:, None] / low - 1) * 10_000,
        (high / entry[:, None] - 1) * 10_000,
    )
    close = data["close"].to_numpy(float)
    for horizon in HORIZONS_MINUTES:
        count = horizon * 60
        terminal = side * (close[indexes + count - 1] / entry - 1) * 10_000
        actions[f"mfe_{horizon}m_bps"] = favorable[:, :count].max(axis=1)
        actions[f"mae_{horizon}m_bps"] = adverse[:, :count].max(axis=1)
        for target in TARGETS_BPS:
            target_hit = favorable[:, :count] >= target
            stop_hit = adverse[:, :count] >= -stop[:, None]
            has_target = target_hit.any(axis=1)
            has_stop = stop_hit.any(axis=1)
            first_target = np.where(has_target, target_hit.argmax(axis=1), count)
            first_stop = np.where(has_stop, stop_hit.argmax(axis=1), count)
            outcome = np.select(
                (has_target & (first_target < first_stop), has_stop & (first_stop <= first_target)),
                (0, 1),
                default=2,
            ).astype(np.int8)
            realized = np.select((outcome == 0, outcome == 1), (float(target), stop), terminal)
            stem = f"{target}bps_{horizon}m"
            actions[f"outcome_{stem}"] = outcome
            actions[f"plan_return_{stem}"] = realized
            actions[f"exit_buckets_{stem}"] = np.select(
                (outcome == 0, outcome == 1),
                (first_target + 1, first_stop + 1),
                default=count,
            ).astype(np.int16)
            actions[f"time_to_target_{stem}"] = np.where(
                outcome == 0, (first_target + 1) / 60, np.nan
            )
    timestamp = pd.to_datetime(actions["available_at"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    weekday = timestamp.dt.dayofweek
    actions["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    actions["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    actions["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    actions["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    actions["available_at"] = timestamp.dt.as_unit("us")
    actions = _attach_market_state(_attach_cross_exchange(actions))
    actions = actions.dropna(subset=list(FEATURES)).reset_index(drop=True)
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".parquet.tmp")
    actions.to_parquet(temporary, index=False)
    temporary.replace(MATRIX)
    _status("matrix", 40, f"{len(actions):,} fully covered causal actions")
    return pd.DataFrame(actions)


def _classifier(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2_000, random_state=seed),
        )
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12,
        tree_method="hist",
        device="cuda",
        n_jobs=8,
        random_state=seed,
    )


def _regressor(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=12,
        tree_method="hist",
        device="cuda",
        n_jobs=8,
        random_state=seed,
    )


def _three_class_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    """Map estimator probabilities to TARGET/STOP/TIMEOUT, including absent classes."""
    predicted = np.asarray(model.predict_proba(values), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    result = np.zeros((len(values), 3), dtype=float)
    result[:, classes] = predicted
    return result


def _fit_plan(
    kind: str, fit: pd.DataFrame, calibration: pd.DataFrame, target: int, horizon: int
) -> dict[str, Any]:
    stem = f"{target}bps_{horizon}m"
    truth = fit[f"outcome_{stem}"].to_numpy(int)
    model = _classifier(kind, target + horizon).fit(_features(fit), truth)
    raw = np.clip(
        _three_class_probabilities(model, _features(calibration)), 1e-6, 1
    )
    calibrator = LogisticRegression(C=1.0, max_iter=2_000, random_state=target * 10 + horizon)
    calibrator.fit(np.log(raw), calibration[f"outcome_{stem}"].to_numpy(int))
    timeout = truth == 2
    timeout_model = _regressor(kind, 100 + target + horizon).fit(
        _features(fit)[timeout], fit.loc[timeout, f"plan_return_{stem}"]
    )
    return {
        "target_bps": target,
        "horizon_minutes": horizon,
        "outcome_model": model,
        "calibrator": calibrator,
        "timeout_model": timeout_model,
    }


def _score_plan(
    plan: dict[str, Any],
    rows: pd.DataFrame,
    vip_level: int = 0,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    target = int(plan["target_bps"])
    horizon = int(plan["horizon_minutes"])
    raw = np.clip(
        _three_class_probabilities(plan["outcome_model"], _features(rows)),
        1e-6,
        1,
    )
    probability = _three_class_probabilities(plan["calibrator"], np.log(raw))
    timeout_return = np.asarray(plan["timeout_model"].predict(_features(rows)), float)
    target_cost, forced_cost = _costs(vip_level, cost_multiplier)
    stop = rows["initial_stop_bps"].to_numpy(float)
    expected_net = (
        probability[:, 0] * (target - target_cost)
        + probability[:, 1] * (stop - forced_cost)
        + probability[:, 2] * (timeout_return - forced_cost)
    )
    return pd.DataFrame(
        {
            "target_bps": target,
            "horizon_minutes": horizon,
            "p_target": probability[:, 0],
            "p_stop": probability[:, 1],
            "p_timeout": probability[:, 2],
            "expected_timeout_bps": timeout_return,
            "predicted_net_ev_bps": expected_net,
        },
        index=rows.index,
    )


def _best_actions(
    plans: list[dict[str, Any]],
    rows: pd.DataFrame,
    vip_level: int = 0,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    target_cost, _ = _costs(vip_level, cost_multiplier)
    for plan in plans:
        if float(plan["target_bps"]) < MINIMUM_TARGET_TO_COST * target_cost:
            continue
        scored = _score_plan(
            plan,
            rows,
            vip_level=vip_level,
            cost_multiplier=cost_multiplier,
        )
        scored = pd.concat(
            [
                rows[
                    [
                        "available_at",
                        "entry_timestamp",
                        "direction",
                        "event_family",
                        "initial_stop_bps",
                    ]
                ],
                scored,
            ],
            axis=1,
        )
        stem = f"{plan['target_bps']}bps_{plan['horizon_minutes']}m"
        outcome = rows[f"outcome_{stem}"].to_numpy(int)
        target_cost, forced_cost = _costs(vip_level, cost_multiplier)
        realized_cost = np.where(outcome == 0, target_cost, forced_cost)
        exit_buckets = rows[f"exit_buckets_{stem}"].to_numpy(int)
        scored["realized_gross_bps"] = rows[f"plan_return_{stem}"].to_numpy(float)
        scored["realized_cost_bps"] = realized_cost
        scored["realized_net_bps"] = scored["realized_gross_bps"] - realized_cost
        scored["risk_budget_distance_bps"] = (
            rows["initial_stop_bps"].abs().to_numpy(float) + forced_cost
        )
        scored["exit_buckets"] = exit_buckets
        scored["exit_timestamp"] = pd.to_datetime(
            scored["entry_timestamp"], utc=True
        ) + pd.to_timedelta(exit_buckets, unit="s")
        scored["outcome"] = outcome
        candidates.append(scored)
    expanded = pd.concat(candidates, ignore_index=True)
    return expanded.loc[
        expanded.groupby("available_at")["predicted_net_ev_bps"].idxmax()
    ].sort_values("available_at")


def _one_position(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = rows.loc[
        rows["predicted_net_ev_bps"].ge(threshold)
        & rows["predicted_net_ev_bps"].gt(0)
    ].sort_values("available_at")
    chosen: list[Any] = []
    free_ns = np.iinfo(np.int64).min
    for index, row in candidates.iterrows():
        current_ns = pd.Timestamp(row["available_at"]).value
        if current_ns >= free_ns:
            chosen.append(index)
            if "exit_timestamp" in row.index:
                free_ns = pd.Timestamp(row["exit_timestamp"]).value
            else:
                free_ns = (
                    pd.Timestamp(row["entry_timestamp"]).value
                    + int(row["horizon_minutes"]) * 60 * 1_000_000_000
                )
    return candidates.loc[chosen]


def _causal_coverage_filter(
    history: pd.DataFrame,
    test: pd.DataFrame,
    *,
    coverage: float = DISCOVERY_COVERAGE,
) -> pd.DataFrame:
    """Keep the top causal score percentile while adapting to score-scale drift."""
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    observed = list(history["predicted_net_ev_bps"].to_numpy(float))
    accepted: list[Any] = []
    thresholds: list[float] = []
    for index, row in test.sort_values("available_at").iterrows():
        threshold = float(np.quantile(observed, 1 - coverage))
        score = float(row["predicted_net_ev_bps"])
        if score > max(0.0, threshold):
            accepted.append(index)
            thresholds.append(threshold)
        observed.append(score)
    result = test.loc[accepted].copy()
    result["causal_threshold_bps"] = thresholds
    return result


def _bootstrap_lower_mean(
    values: np.ndarray,
    *,
    block_size: int = 10,
    samples: int = 2_000,
) -> float:
    if not len(values):
        return 0.0
    if len(values) < block_size:
        return float(np.quantile(values, 0.05))
    random = np.random.default_rng(5_2026)
    starts = np.arange(len(values) - block_size + 1)
    blocks_needed = int(np.ceil(len(values) / block_size))
    means = np.empty(samples)
    for sample in range(samples):
        chosen = random.choice(starts, size=blocks_needed, replace=True)
        bootstrapped = np.concatenate(
            [values[start : start + block_size] for start in chosen]
        )[: len(values)]
        means[sample] = bootstrapped.mean()
    return float(np.quantile(means, 0.05))


def _max_account_drawdown(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    risk_bps = (
        rows["risk_budget_distance_bps"].clip(lower=1).to_numpy(float)
        if "risk_budget_distance_bps" in rows
        else rows["initial_stop_bps"].abs().clip(lower=1).to_numpy(float)
    )
    trade_returns = np.maximum(rows["realized_net_bps"].to_numpy(float) / risk_bps, -2.0) * 0.01
    equity: np.ndarray = np.cumprod(1 + trade_returns)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdowns = 1 - np.r_[1.0, equity] / peaks
    return float(drawdowns.max())


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    values = rows["realized_net_bps"].to_numpy(float)
    if not len(values):
        return {
            "trades": 0,
            "expectancy_bps": 0.0,
            "expectancy_bootstrap_lcb_95_bps": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_account_drawdown": 0.0,
            "positive_week_fraction": 0.0,
        }
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    gross = rows["realized_gross_bps"].to_numpy(float)
    costs = rows["realized_cost_bps"].to_numpy(float)
    timestamps = pd.to_datetime(rows["available_at"], utc=True)
    weekly = pd.Series(values, index=timestamps).resample("7D").sum()
    gross_winners = gross[gross > 0]
    return {
        "trades": len(values),
        "expectancy_bps": float(values.mean()),
        "expectancy_bootstrap_lcb_95_bps": _bootstrap_lower_mean(values),
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()),
        "gross_expectancy_bps": float(gross.mean()),
        "average_cost_bps": float(costs.mean()),
        "cost_to_gross_profit": (
            float(costs.sum() / gross_winners.sum()) if gross_winners.sum() else None
        ),
        "average_winner_gross_to_cost": (
            float(gross_winners.mean() / costs.mean())
            if len(gross_winners) and costs.mean() > 0
            else None
        ),
        "max_account_drawdown": _max_account_drawdown(rows),
        "positive_week_fraction": float((weekly > 0).mean()) if len(weekly) else 0.0,
        "long": int(rows["direction"].gt(0).sum()),
        "short": int(rows["direction"].lt(0).sum()),
        "targets": int(rows["outcome"].eq(0).sum()),
        "stops": int(rows["outcome"].eq(1).sum()),
        "timeouts": int(rows["outcome"].eq(2).sum()),
    }


def _stress_same_trades(rows: pd.DataFrame) -> pd.DataFrame:
    stressed = rows.copy()
    stressed["realized_cost_bps"] *= 2
    stressed["realized_net_bps"] = (
        stressed["realized_gross_bps"] - stressed["realized_cost_bps"]
    )
    return stressed


def _score_distribution(rows: pd.DataFrame) -> dict[str, float]:
    scores = rows["predicted_net_ev_bps"].to_numpy(float)
    return {
        "positive_fraction": float((scores > 0).mean()),
        "p50_bps": float(np.quantile(scores, 0.50)),
        "p90_bps": float(np.quantile(scores, 0.90)),
        "p95_bps": float(np.quantile(scores, 0.95)),
        "p99_bps": float(np.quantile(scores, 0.99)),
        "maximum_bps": float(scores.max()),
    }


def _oracle_diagnostics(rows: pd.DataFrame, vip_level: int = 0) -> dict[str, Any]:
    candidates: list[pd.DataFrame] = []
    target_cost, forced_cost = _costs(vip_level)
    identity = rows[
        [
            "available_at",
            "entry_timestamp",
            "direction",
            "event_family",
            "initial_stop_bps",
        ]
    ]
    for target, horizon in PLANS:
        if target < MINIMUM_TARGET_TO_COST * target_cost:
            continue
        stem = f"{target}bps_{horizon}m"
        outcome = rows[f"outcome_{stem}"].to_numpy(int)
        gross = rows[f"plan_return_{stem}"].to_numpy(float)
        cost = np.where(outcome == 0, target_cost, forced_cost)
        plan = identity.copy()
        plan["target_bps"] = target
        plan["horizon_minutes"] = horizon
        plan["outcome"] = outcome
        exit_buckets = rows[f"exit_buckets_{stem}"].to_numpy(int)
        plan["exit_buckets"] = exit_buckets
        plan["exit_timestamp"] = pd.to_datetime(
            plan["entry_timestamp"], utc=True
        ) + pd.to_timedelta(exit_buckets, unit="s")
        plan["realized_gross_bps"] = gross
        plan["realized_cost_bps"] = cost
        plan["realized_net_bps"] = gross - cost
        plan["risk_budget_distance_bps"] = (
            rows["initial_stop_bps"].abs().to_numpy(float) + forced_cost
        )
        plan["predicted_net_ev_bps"] = plan["realized_net_bps"]
        candidates.append(plan)
    expanded = pd.concat(candidates, ignore_index=True)
    best = expanded.loc[
        expanded.groupby("available_at")["realized_net_bps"].idxmax()
    ].sort_values("available_at")
    positive = best.loc[best["realized_net_bps"].gt(0)]
    return {
        "note": "diagnostic upper bound uses future outcomes and is never deployable",
        "available_decisions": len(best),
        "positive_decision_fraction": float(len(positive) / len(best)) if len(best) else 0.0,
        "one_position_positive_oracle": _metrics(_one_position(best, threshold=0.0)),
    }


def _rolling_fit(
    matrix: pd.DataFrame,
    timestamps: pd.Series,
    fit_end: pd.Timestamp,
) -> pd.DataFrame:
    fit_start = fit_end - pd.Timedelta(days=TRAINING_WINDOW_DAYS)
    return matrix.loc[timestamps.ge(fit_start) & timestamps.lt(fit_end)]


def _walk_forward(
    matrix: pd.DataFrame,
    kind: str,
    *,
    status_offset: float,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(matrix["available_at"], utc=True)
    fold_starts = list(
        pd.date_range(
            AUDIT_START,
            AUDIT_END,
            freq=f"{WALK_FORWARD_TEST_DAYS}D",
            inclusive="left",
        )
    )
    qualified: dict[int, list[pd.DataFrame]] = {level: [] for level in range(6)}
    folds: list[dict[str, Any]] = []
    purge = pd.Timedelta(minutes=max(HORIZONS_MINUTES))
    for fold_number, fold_start in enumerate(fold_starts, start=1):
        fold_end = min(
            fold_start + pd.Timedelta(days=WALK_FORWARD_TEST_DAYS),
            AUDIT_END,
        )
        selection_end = fold_start - purge
        selection_start = selection_end - pd.Timedelta(
            days=WALK_FORWARD_SELECTION_DAYS
        )
        calibration_end = selection_start - purge
        calibration_start = calibration_end - pd.Timedelta(
            days=WALK_FORWARD_CALIBRATION_DAYS
        )
        fit_end = calibration_start - purge
        fit = _rolling_fit(matrix, timestamps, fit_end)
        calibration = matrix.loc[
            timestamps.ge(calibration_start) & timestamps.lt(calibration_end)
        ]
        selection = matrix.loc[
            timestamps.ge(selection_start) & timestamps.lt(selection_end)
        ]
        test = matrix.loc[timestamps.ge(fold_start) & timestamps.lt(fold_end)]
        if min(len(fit), len(calibration), len(selection), len(test)) < 100:
            raise ValueError(f"Insufficient walk-forward coverage in fold {fold_number}")
        percent = status_offset + 8 * fold_number / len(fold_starts)
        _status(
            "walk_forward",
            percent,
            f"{kind} fold {fold_number}/{len(fold_starts)}",
        )
        plans = [
            _fit_plan(kind, fit, calibration, target, horizon)
            for target, horizon in PLANS
        ]
        fold_profiles: dict[str, Any] = {}
        for level in range(6):
            selection_scores = _best_actions(plans, selection, vip_level=level)
            threshold = float(
                selection_scores["predicted_net_ev_bps"].quantile(
                    1 - DISCOVERY_COVERAGE
                )
            )
            test_scores = _best_actions(plans, test, vip_level=level)
            accepted = _causal_coverage_filter(
                selection_scores,
                test_scores,
                coverage=DISCOVERY_COVERAGE,
            )
            qualified[level].append(accepted)
            fold_trades = _one_position(accepted, threshold=0.0)
            fold_profiles[f"VIP{level}"] = {
                "threshold_bps": threshold,
                "available_actions": len(test_scores),
                "accepted_before_position_lock": len(accepted),
                "score_distribution": _score_distribution(test_scores),
                "metrics": _metrics(fold_trades),
            }
        folds.append(
            {
                "fold": fold_number,
                "fit_end": fit_end.isoformat(),
                "calibration": [
                    calibration_start.isoformat(),
                    calibration_end.isoformat(),
                ],
                "selection": [selection_start.isoformat(), selection_end.isoformat()],
                "test": [fold_start.isoformat(), fold_end.isoformat()],
                "rows": {
                    "fit": len(fit),
                    "calibration": len(calibration),
                    "selection": len(selection),
                    "test": len(test),
                },
                "profiles": fold_profiles,
            }
        )
    profiles: dict[str, Any] = {}
    WALK_FORWARD_DIR.mkdir(parents=True, exist_ok=True)
    for level in range(6):
        joined = pd.concat(qualified[level], ignore_index=True).sort_values(
            "available_at"
        )
        trades = _one_position(joined, threshold=0.0)
        trades = trades.copy()
        trades["protocol_hash"] = PROTOCOL_HASH
        trades["model_kind"] = kind
        trades["vip_level"] = level
        output = WALK_FORWARD_DIR / f"{kind}_vip{level}.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        trades.to_parquet(temporary, index=False)
        temporary.replace(output)
        profiles[f"VIP{level}"] = {
            "audit_metrics": _metrics(trades),
            "audit_cost_stress_2x_metrics": _metrics(_stress_same_trades(trades)),
        }
    return {"kind": kind, "folds": folds, "profiles": profiles}


def _fit_shadow_challenger(
    matrix: pd.DataFrame,
    kind: str,
) -> dict[str, Any]:
    """Refit after the June audit for future paper only; July is never read."""
    timestamps = pd.to_datetime(matrix["available_at"], utc=True)
    purge = pd.Timedelta(minutes=max(HORIZONS_MINUTES))
    calibration_start = AUDIT_START
    calibration_end = AUDIT_START + pd.Timedelta(days=14) - purge
    selection_start = AUDIT_START + pd.Timedelta(days=14)
    selection_end = AUDIT_END - purge
    fit = _rolling_fit(matrix, timestamps, calibration_start - purge)
    calibration = matrix.loc[
        timestamps.ge(calibration_start) & timestamps.lt(calibration_end)
    ]
    selection = matrix.loc[
        timestamps.ge(selection_start) & timestamps.lt(selection_end)
    ]
    _status("shadow_refit", 98, f"{kind}: future paper bundle, July remains sealed")
    plans = [
        _fit_plan(kind, fit, calibration, target, horizon)
        for target, horizon in PLANS
    ]
    profiles: dict[str, Any] = {}
    for level in range(6):
        scores = _best_actions(plans, selection, vip_level=level)
        profiles[f"VIP{level}"] = {
            "threshold_bps": float(
                scores["predicted_net_ev_bps"].quantile(1 - DISCOVERY_COVERAGE)
            ),
            "score_history_bps": scores["predicted_net_ev_bps"].astype(float).tolist(),
            "selection_score_distribution": _score_distribution(scores),
        }
    return {
        "kind": kind,
        "fit_end": (calibration_start - purge).isoformat(),
        "calibration": [calibration_start.isoformat(), calibration_end.isoformat()],
        "threshold_selection": [selection_start.isoformat(), selection_end.isoformat()],
        "plans": plans,
        "profiles": profiles,
        "status": "DISCOVERY_ONLY_SHADOW_CHALLENGER",
        "live_orders_enabled": False,
    }


def train(*, force_matrix: bool = False) -> dict[str, Any]:
    matrix = build_matrix(force=force_matrix).sort_values("available_at").reset_index(drop=True)
    timestamp = pd.to_datetime(matrix["available_at"], utc=True)
    if timestamp.max() < AUDIT_END - pd.Timedelta(days=1):
        raise ValueError("April-June Binance aggTrades are not yet fully available")
    fit = _rolling_fit(matrix, timestamp, FIT_END).copy()
    calibration = matrix.loc[
        timestamp.ge(FIT_END) & timestamp.lt(CALIBRATION_END)
    ].copy()
    selection = matrix.loc[
        timestamp.ge(CALIBRATION_END) & timestamp.lt(AUDIT_START)
    ].copy()
    audit = matrix.loc[timestamp.ge(AUDIT_START) & timestamp.lt(AUDIT_END)].copy()
    holdout_rows = int(timestamp.ge(HOLDOUT_START).sum())
    if min(len(fit), len(calibration), len(selection), len(audit)) < 100:
        raise ValueError(
            "Chronological V5 fit/calibration/selection/audit coverage is insufficient"
        )
    candidates: dict[str, dict[str, Any]] = {}
    for number, kind in enumerate(("ridge", "xgboost"), start=1):
        _status(
            "training",
            40 + number * 20,
            f"{kind}: {len(PLANS)} TARGET/STOP/TIMEOUT plans",
        )
        plans = [
            _fit_plan(kind, fit, calibration, target, horizon)
            for target, horizon in PLANS
        ]
        selected_scores = _best_actions(plans, selection, vip_level=0)
        threshold = float(
            selected_scores["predicted_net_ev_bps"].quantile(1 - DISCOVERY_COVERAGE)
        )
        selected = _one_position(selected_scores, threshold)
        audit_scores_for_candidate = _best_actions(plans, audit, vip_level=0)
        audit_trades_for_candidate = _one_position(
            audit_scores_for_candidate, threshold
        )
        candidates[kind] = {
            "plans": plans,
            "threshold_bps": threshold,
            "selection_metrics": _metrics(selected),
            "selection_score_distribution": _score_distribution(selected_scores),
            "audit_diagnostic_metrics": _metrics(audit_trades_for_candidate),
            "audit_score_distribution": _score_distribution(audit_scores_for_candidate),
        }
    ridge_metrics = candidates["ridge"]["selection_metrics"]
    xgb_metrics = candidates["xgboost"]["selection_metrics"]
    xgb_wins = (
        xgb_metrics["trades"] >= 10
        and xgb_metrics["expectancy_bps"] > ridge_metrics["expectancy_bps"]
        and (xgb_metrics["profit_factor"] or 0) >= 1.15
    )
    champion = "xgboost" if xgb_wins else "ridge"
    walk_forward = {
        "ridge": _walk_forward(matrix, "ridge", status_offset=80),
        "xgboost": _walk_forward(matrix, "xgboost", status_offset=88),
    }
    xgb_shadow_metrics = walk_forward["xgboost"]["profiles"]["VIP0"][
        "audit_metrics"
    ]
    shadow_profile_gates: dict[str, dict[str, bool]] = {}
    for level in range(6):
        name = f"VIP{level}"
        metrics_for_level = walk_forward["xgboost"]["profiles"][name][
            "audit_metrics"
        ]
        stress_for_level = walk_forward["xgboost"]["profiles"][name][
            "audit_cost_stress_2x_metrics"
        ]
        shadow_profile_gates[name] = {
            "expectancy_positive": metrics_for_level["expectancy_bps"] > 0,
            "expectancy_lcb_positive": (
                metrics_for_level["expectancy_bootstrap_lcb_95_bps"] > 0
            ),
            "profit_factor_1_15": (metrics_for_level["profit_factor"] or 0) >= 1.15,
            "max_drawdown_8pct": metrics_for_level["max_account_drawdown"] <= 0.08,
            "majority_positive_weeks": (
                metrics_for_level["positive_week_fraction"] > 0.5
            ),
            "cost_stress_2x_nonnegative": stress_for_level["expectancy_bps"] >= 0,
            "gross_winner_at_least_3x_cost": (
                metrics_for_level.get("average_winner_gross_to_cost") or 0
            )
            >= 3,
        }
    shadow_challenger_financial_gates = shadow_profile_gates["VIP0"]
    base_viable_profiles = [
        name
        for name, profile_gates in shadow_profile_gates.items()
        if all(
            passed
            for gate, passed in profile_gates.items()
            if gate != "cost_stress_2x_nonnegative"
        )
    ]
    shadow_candidate_available = (
        bool(base_viable_profiles)
        and xgb_shadow_metrics["expectancy_bps"]
        > walk_forward["ridge"]["profiles"]["VIP0"]["audit_metrics"][
            "expectancy_bps"
        ]
    )
    shadow_challenger = (
        _fit_shadow_challenger(matrix, "xgboost")
        if shadow_candidate_available
        else None
    )
    profiles: dict[str, dict[str, Any]] = {}
    for level in range(6):
        profile_selection = _best_actions(
            candidates[champion]["plans"], selection, vip_level=level
        )
        profile_threshold = float(
            profile_selection["predicted_net_ev_bps"].quantile(
                1 - DISCOVERY_COVERAGE
            )
        )
        profile_audit = _best_actions(
            candidates[champion]["plans"], audit, vip_level=level
        )
        audit_trades = _one_position(profile_audit, profile_threshold)
        audit_metrics_for_profile = _metrics(audit_trades)
        profiles[f"VIP{level}"] = {
            "vip_level": level,
            "target_cost_bps": _costs(level)[0],
            "forced_exit_cost_bps": _costs(level)[1],
            "threshold_bps": profile_threshold,
            "selection_metrics": _metrics(
                _one_position(profile_selection, profile_threshold)
            ),
            "static_audit_diagnostic_metrics": audit_metrics_for_profile,
            "static_audit_cost_stress_2x_diagnostic_metrics": _metrics(
                _stress_same_trades(audit_trades)
            ),
            "audit_metrics": walk_forward[champion]["profiles"][f"VIP{level}"][
                "audit_metrics"
            ],
            "audit_cost_stress_2x_metrics": walk_forward[champion]["profiles"][
                f"VIP{level}"
            ]["audit_cost_stress_2x_metrics"],
        }
    threshold = float(profiles["VIP0"]["threshold_bps"])
    audit_metrics = profiles["VIP0"]["audit_metrics"]
    stress_metrics = profiles["VIP0"]["audit_cost_stress_2x_metrics"]
    gates = {
        "audit_trades_300": audit_metrics["trades"] >= 300,
        "expectancy_positive": audit_metrics["expectancy_bps"] > 0,
        "expectancy_lcb_positive": audit_metrics["expectancy_bootstrap_lcb_95_bps"] > 0,
        "profit_factor_1_15": (audit_metrics["profit_factor"] or 0) >= 1.15,
        "max_drawdown_8pct": audit_metrics["max_account_drawdown"] <= 0.08,
        "majority_positive_weeks": audit_metrics["positive_week_fraction"] > 0.5,
        "cost_stress_2x_nonnegative": stress_metrics["expectancy_bps"] >= 0,
        "gross_winner_at_least_3x_cost": (
            audit_metrics.get("average_winner_gross_to_cost") or 0
        )
        >= 3,
        "future_holdout_opened": False,
    }
    deployable_verdict = (
        "RESEARCH_ONLY_CANDIDATE"
        if all(value for key, value in gates.items() if key != "future_holdout_opened")
        else "NO_ECONOMIC_ALPHA"
    )
    verdict = (
        "NO_DEPLOYABLE_POLICY_SHADOW_CHALLENGER_ONLY"
        if shadow_challenger is not None and deployable_verdict == "NO_ECONOMIC_ALPHA"
        else deployable_verdict
    )
    bundle = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "features": FEATURES,
        "champion": champion,
        "plans": candidates[champion]["plans"],
        "threshold_bps": threshold,
        "vip_profiles": profiles,
        "shadow_challenger": shadow_challenger,
        "shadow_challenger_vip_audit": walk_forward["xgboost"]["profiles"],
        "verdict": verdict,
        "live_orders_enabled": False,
        "holdout_opened": False,
    }
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(BUNDLE)
    report = {
        "status": verdict,
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "rows": {
            "matrix": len(matrix),
            "fit": len(fit),
            "calibration": len(calibration),
            "model_selection": len(selection),
            "audit": len(audit),
            "sealed_holdout": holdout_rows,
        },
        "candidate_models": {
            kind: {
                "threshold_bps": value["threshold_bps"],
                "selection_metrics": value["selection_metrics"],
                "selection_score_distribution": value["selection_score_distribution"],
                "audit_diagnostic_metrics": value["audit_diagnostic_metrics"],
                "audit_score_distribution": value["audit_score_distribution"],
                "walk_forward_audit_metrics": walk_forward[kind]["profiles"][
                    "VIP0"
                ]["audit_metrics"],
            }
            for kind, value in candidates.items()
        },
        "champion": champion,
        "audit_metrics": audit_metrics,
        "vip_profiles": profiles,
        "shadow_challenger": {
            "status": (
                shadow_challenger["status"]
                if shadow_challenger is not None
                else "NOT_QUALIFIED"
            ),
            "model": "xgboost",
            "financial_gates": shadow_challenger_financial_gates,
            "profile_financial_gates": shadow_profile_gates,
            "base_viable_profiles": base_viable_profiles,
            "vip_audit": walk_forward["xgboost"]["profiles"],
            "trade_count_gate_300": xgb_shadow_metrics["trades"] >= 300,
            "independent_future_holdout_passed": False,
        },
        "walk_forward": walk_forward,
        "audit_oracle_diagnostic": _oracle_diagnostics(audit, vip_level=0),
        "gates": gates,
        "bundle": str(BUNDLE),
        "real_capital_allowed": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT, report)
    _status("complete", 100, verdict)
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
