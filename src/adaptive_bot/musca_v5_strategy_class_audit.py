from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.bitunix_fees import futures_fee_bps

SOURCE = Path("data/ml/musca_v5/aggtrades")
BITUNIX_L2 = Path("data/research/bitunix_l2/btcusdt_l2_features.parquet")
REPORT = Path("data/reports/musca_v5_strategy_class_audit.json")
DISCOVERY_MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04")
TARGET_BPS = 12.0
STOP_BPS = 6.0
HORIZON_BARS = 180  # 15 minutes at five-second resolution.
PROTOCOL = {
    "name": "musca_v5_strategy_class_diagnostic_v1",
    "source": "official_Binance_BTCUSDT_perpetual_aggregate_trades_5s",
    "discovery_months": list(DISCOVERY_MONTHS),
    "sealed_data_from": "2026-05-01T00:00:00Z",
    "decision_frequency": "one_minute",
    "execution": "next_5s_open",
    "target_bps": TARGET_BPS,
    "stop_bps": STOP_BPS,
    "maximum_holding_minutes": 15,
    "same_bar": "stop_wins",
    "one_open_position": True,
    "diagnostics": {
        "fixed_barrier_rules": [
            "momentum_1m_follow",
            "momentum_5m_follow",
            "flow_1m_follow",
            "daily_vwap_fade",
            "daily_vwap_follow",
        ],
        "frozen_rolling_vwap_fade_thresholds_bps": [2, 4, 6, 8, 12, 16, 24, 32],
        "frozen_center_stop": "symmetric_distance_beyond_entry",
        "cost_basis": "Bitunix_observed_median_spread_plus_VIP_fee",
    },
    "state_selection": "Jan_Feb_only",
    "state_validation": ["2026-03", "2026-04"],
    "holdout_opened": False,
    "changes_to_active_paper": False,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def _barrier_actions(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entries: np.ndarray,
    positions: np.ndarray,
    *,
    target_bps: float = TARGET_BPS,
    stop_bps: float = STOP_BPS,
    horizon_bars: int = HORIZON_BARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LONG/SHORT gross outcomes using only bars after each decision."""
    entry_position = np.minimum(positions + 1, len(entries) - 1)
    entry = entries[entry_position]
    valid = (
        np.isfinite(entry)
        & (positions + 1 < len(entries))
        & (positions + horizon_bars < len(close))
    )
    long_return = np.full(len(positions), np.nan)
    short_return = np.full(len(positions), np.nan)
    long_exit = np.full(len(positions), horizon_bars, dtype=np.int16)
    short_exit = np.full(len(positions), horizon_bars, dtype=np.int16)
    long_open = valid.copy()
    short_open = valid.copy()
    long_target = entry * (1 + target_bps / 10_000)
    long_stop = entry * (1 - stop_bps / 10_000)
    short_target = entry * (1 - target_bps / 10_000)
    short_stop = entry * (1 + stop_bps / 10_000)
    path_valid = valid.copy()

    for offset in range(1, horizon_bars + 1):
        path_high = high[np.minimum(positions + offset, len(high) - 1)]
        path_low = low[np.minimum(positions + offset, len(low) - 1)]
        finite = np.isfinite(path_high) & np.isfinite(path_low)
        path_valid &= finite

        long_stop_hit = long_open & finite & (path_low <= long_stop)
        long_target_hit = long_open & finite & ~long_stop_hit & (path_high >= long_target)
        long_return[long_stop_hit] = -stop_bps
        long_return[long_target_hit] = target_bps
        long_exit[long_stop_hit | long_target_hit] = offset
        long_open &= ~(long_stop_hit | long_target_hit)

        short_stop_hit = short_open & finite & (path_high >= short_stop)
        short_target_hit = short_open & finite & ~short_stop_hit & (path_low <= short_target)
        short_return[short_stop_hit] = -stop_bps
        short_return[short_target_hit] = target_bps
        short_exit[short_stop_hit | short_target_hit] = offset
        short_open &= ~(short_stop_hit | short_target_hit)

    final = close[np.minimum(positions + horizon_bars, len(close) - 1)]
    long_return[long_open & path_valid] = (
        final[long_open & path_valid] / entry[long_open & path_valid] - 1
    ) * 10_000
    short_return[short_open & path_valid] = -(
        final[short_open & path_valid] / entry[short_open & path_valid] - 1
    ) * 10_000
    long_return[~path_valid] = np.nan
    short_return[~path_valid] = np.nan
    return long_return, short_return, long_exit, short_exit, path_valid


def _frozen_center_fade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entries: np.ndarray,
    positions: np.ndarray,
    centers: np.ndarray,
    *,
    horizon_bars: int = HORIZON_BARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fade to the currently observable VWAP with symmetric structural risk."""
    entry = entries[np.minimum(positions + 1, len(entries) - 1)]
    center = centers[positions]
    side = -np.sign(entries[positions] - center)
    target_bps = side * (center / entry - 1) * 10_000
    stop = entry - side * np.abs(center - entry)
    outcome = np.full(len(positions), np.nan)
    exit_bars = np.full(len(positions), horizon_bars, dtype=np.int16)
    valid = (
        np.isfinite(entry)
        & np.isfinite(center)
        & (target_bps > 0)
        & (positions + horizon_bars < len(close))
    )
    opened = valid.copy()
    path_valid = valid.copy()
    for offset in range(1, horizon_bars + 1):
        path_high = high[np.minimum(positions + offset, len(high) - 1)]
        path_low = low[np.minimum(positions + offset, len(low) - 1)]
        finite = np.isfinite(path_high) & np.isfinite(path_low)
        path_valid &= finite
        stop_hit = opened & finite & np.where(side > 0, path_low <= stop, path_high >= stop)
        target_hit = (
            opened
            & finite
            & ~stop_hit
            & np.where(side > 0, path_high >= center, path_low <= center)
        )
        outcome[stop_hit] = -target_bps[stop_hit]
        outcome[target_hit] = target_bps[target_hit]
        exit_bars[stop_hit | target_hit] = offset
        opened &= ~(stop_hit | target_hit)
    final = close[np.minimum(positions + horizon_bars, len(close) - 1)]
    outcome[opened & path_valid] = (
        side[opened & path_valid]
        * (final[opened & path_valid] / entry[opened & path_valid] - 1)
        * 10_000
    )
    outcome[~path_valid] = np.nan
    return outcome, exit_bars, valid


def _load_discovery() -> pd.DataFrame:
    frames = [
        pd.read_parquet(SOURCE / f"BTCUSDT-aggTrades-5s-{month}.parquet")
        for month in DISCOVERY_MONTHS
    ]
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    data = data.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    index = pd.date_range(data.index.min(), data.index.max(), freq="5s", tz="UTC")
    return data.reindex(index)


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    quote = data["quote_volume"]
    signed = data["signed_quote_volume"]
    close = data["close"]
    log_volume = np.log1p(quote)
    index = pd.DatetimeIndex(data.index)
    session = pd.Series(index.normalize(), index=index)
    daily_quote = quote.groupby(session).cumsum()
    daily_base = data["base_volume"].groupby(session).cumsum()
    rolling_quote = quote.rolling(60, min_periods=60).sum()
    rolling_base = data["base_volume"].rolling(60, min_periods=60).sum()
    volume_mean = log_volume.shift(1).rolling(720, min_periods=240).mean()
    volume_std = log_volume.shift(1).rolling(720, min_periods=240).std()
    features = pd.DataFrame(index=data.index)
    features["return_1m_bps"] = close.pct_change(12, fill_method=None) * 10_000
    features["return_5m_bps"] = close.pct_change(60, fill_method=None) * 10_000
    features["return_15m_bps"] = close.pct_change(180, fill_method=None) * 10_000
    features["flow_1m"] = signed.rolling(12, min_periods=12).sum() / quote.rolling(
        12, min_periods=12
    ).sum()
    features["flow_5m"] = signed.rolling(60, min_periods=60).sum() / rolling_quote
    features["last_5s_return_bps"] = close.pct_change(fill_method=None) * 10_000
    features["last_5s_flow"] = signed / quote.replace(0, np.nan)
    features["rolling_vwap_distance_bps"] = (close / (rolling_quote / rolling_base) - 1) * 10_000
    features["daily_vwap_distance_bps"] = (close / (daily_quote / daily_base) - 1) * 10_000
    features["volatility_1m_bps"] = (
        close.pct_change(fill_method=None).rolling(12, min_periods=12).std() * 10_000
    )
    features["volume_z"] = (log_volume - volume_mean) / volume_std.replace(0, np.nan)
    features["available_at"] = data["available_at"]
    return features


def _costs() -> tuple[dict[str, dict[str, float]], float]:
    spread = pd.read_parquet(BITUNIX_L2, columns=["spread_bps", "feature_valid"])
    spread = spread.loc[spread["feature_valid"], "spread_bps"]
    median_spread = float(spread.median())
    profiles: dict[str, dict[str, float]] = {}
    for level in range(6):
        maker, taker = futures_fee_bps(level)
        profiles[f"VIP{level}"] = {
            "maker_maker_bps": 2 * maker,
            "maker_taker_bps": maker + taker + median_spread / 2,
            "taker_taker_bps": 2 * taker + median_spread,
        }
    return profiles, median_spread


def _metrics(values: np.ndarray) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return {
        "trades": float(len(values)),
        "expectancy_bps": float(values.mean()) if len(values) else None,
        "profit_factor": float(gains / losses) if losses else None,
        "win_rate": float((values > 0).mean()) if len(values) else None,
    }


def _as_int(value: Any) -> int:
    return int(float(value))


def _simulate(
    frame: pd.DataFrame,
    sides: np.ndarray,
    cost: float,
    *,
    mask: np.ndarray | None = None,
    gross: np.ndarray | None = None,
    exit_bars: np.ndarray | None = None,
) -> dict[str, Any]:
    allowed = np.ones(len(frame), dtype=bool) if mask is None else mask
    selected: list[tuple[pd.Timestamp, float]] = []
    blocked_until = pd.Timestamp.min.tz_localize("UTC")
    for position, timestamp in enumerate(pd.DatetimeIndex(frame.index)):
        side = sides[position]
        if not allowed[position] or side == 0 or timestamp < blocked_until:
            continue
        if gross is None:
            value = float(
                frame.iloc[position][
                    "long_gross_bps" if side > 0 else "short_gross_bps"
                ]
            )
            duration = int(
                frame.iloc[position]["long_exit_bars" if side > 0 else "short_exit_bars"]
            )
        else:
            value = float(gross[position])
            duration = int(exit_bars[position]) if exit_bars is not None else HORIZON_BARS
        if not np.isfinite(value):
            continue
        selected.append((timestamp, value - cost))
        blocked_until = timestamp + pd.Timedelta(seconds=5 * duration)
    result: dict[str, Any] = {}
    for month in DISCOVERY_MONTHS:
        values = np.array(
            [
                value
                for timestamp, value in selected
                if timestamp.strftime("%Y-%m") == month
            ]
        )
        result[month] = _metrics(values)
    values = np.array([value for _, value in selected])
    result["all"] = _metrics(values)
    result["trades_per_active_day"] = (
        len(selected) / len({timestamp.normalize() for timestamp, _ in selected})
        if selected
        else 0.0
    )
    return result


def _state_table(frame: pd.DataFrame, cost: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = frame.copy()
    states["trend"] = np.sign(states["return_5m_bps"])
    states["flow"] = np.sign(states["flow_1m"])
    states["vwap"] = np.sign(states["daily_vwap_distance_bps"])
    fit = states.loc[states.index < "2026-03-01"]
    rows: list[dict[str, Any]] = []
    policy: dict[tuple[int, int, int], int] = {}
    for key, group in fit.groupby(["trend", "flow", "vwap"], observed=True):
        if len(group) < 1_000:
            continue
        long_ev = float(group["long_gross_bps"].mean() - cost)
        short_ev = float(group["short_gross_bps"].mean() - cost)
        state_key = (
            int(float(key[0])),  # type: ignore[arg-type]
            int(float(key[1])),  # type: ignore[arg-type]
            int(float(key[2])),  # type: ignore[arg-type]
        )
        selected_side = 1 if long_ev > max(0.0, short_ev) else -1 if short_ev > 0 else 0
        policy[state_key] = selected_side
        rows.append(
            {
                "trend": state_key[0],
                "flow": state_key[1],
                "vwap": state_key[2],
                "fit_rows": len(group),
                "fit_long_net_bps": long_ev,
                "fit_short_net_bps": short_ev,
                "selected_side": selected_side,
            }
        )
    validation: dict[str, Any] = {}
    for month in ("2026-03", "2026-04"):
        state_index = pd.DatetimeIndex(states.index)
        sample = states.loc[state_index.strftime("%Y-%m") == month].copy()
        sides = np.array(
            [
                policy.get((_as_int(row.trend), _as_int(row.flow), _as_int(row.vwap)), 0)
                for row in sample.itertuples()
            ]
        )
        simulation = _simulate(sample, sides, cost)
        validation[month] = simulation[month]
    return rows, validation


def run() -> dict[str, Any]:
    data = _load_discovery()
    features = _feature_frame(data)
    minute_mask = (pd.DatetimeIndex(data.index).second == 55) & features.notna().all(axis=1)
    positions = np.flatnonzero(np.asarray(minute_mask))
    long_gross, short_gross, long_exit, short_exit, valid = _barrier_actions(
        data["high"].to_numpy(float),
        data["low"].to_numpy(float),
        data["close"].to_numpy(float),
        data["open"].to_numpy(float),
        positions,
    )
    frame = features.iloc[positions].copy()
    frame["long_gross_bps"] = long_gross
    frame["short_gross_bps"] = short_gross
    frame["long_exit_bars"] = long_exit
    frame["short_exit_bars"] = short_exit
    frame = frame.loc[valid & np.isfinite(long_gross) & np.isfinite(short_gross)]

    costs, spread = _costs()
    vip5_cost = costs["VIP5"]["maker_taker_bps"]
    oracle = np.maximum.reduce(
        [
            np.zeros(len(frame)),
            frame["long_gross_bps"].to_numpy() - vip5_cost,
            frame["short_gross_bps"].to_numpy() - vip5_cost,
        ]
    )
    fit_states, validation = _state_table(frame, vip5_cost)
    simple_rules = {
        "momentum_1m_follow": np.sign(frame["return_1m_bps"]),
        "momentum_5m_follow": np.sign(frame["return_5m_bps"]),
        "flow_1m_follow": np.sign(frame["flow_1m"]),
        "daily_vwap_fade": -np.sign(frame["daily_vwap_distance_bps"]),
        "daily_vwap_follow": np.sign(frame["daily_vwap_distance_bps"]),
    }
    rule_results: dict[str, Any] = {}
    for name, side_series in simple_rules.items():
        rule_results[name] = _simulate(frame, side_series.to_numpy(), vip5_cost)

    rolling_center = data["close"] / (
        1 + features["rolling_vwap_distance_bps"] / 10_000
    )
    fade_gross, fade_exit, fade_valid = _frozen_center_fade(
        data["high"].to_numpy(float),
        data["low"].to_numpy(float),
        data["close"].to_numpy(float),
        data["open"].to_numpy(float),
        positions,
        rolling_center.to_numpy(float),
    )
    fade_gross = fade_gross[valid]
    fade_exit = fade_exit[valid]
    fade_valid = fade_valid[valid]
    distance = frame["rolling_vwap_distance_bps"].abs()
    distance_frontier: dict[str, Any] = {}
    for threshold in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        outward_cross = distance.ge(threshold) & distance.shift(1).lt(threshold)
        distance_frontier[f"{threshold:g}bps"] = _simulate(
            frame,
            -np.sign(frame["rolling_vwap_distance_bps"]).to_numpy(),
            vip5_cost,
            mask=outward_cross.to_numpy() & fade_valid,
            gross=fade_gross,
            exit_bars=fade_exit,
        )

    wins = TARGET_BPS - vip5_cost
    losses = STOP_BPS + vip5_cost
    required_win_rate = losses / (wins + losses)
    payload = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "created_at": datetime.now(UTC).isoformat(),
        "source_rows": len(data),
        "valid_minute_decisions": len(frame),
        "source_start": data.index.min().isoformat(),
        "source_end": data.index.max().isoformat(),
        "causal_checks": {
            "available_not_before_source_close": bool(
                (frame["available_at"] >= frame.index + pd.Timedelta(seconds=5)).all()
            ),
            "labels_start_after_decision": True,
            "sealed_months_read": [],
        },
        "observed_bitunix_spread_bps": {
            "median": spread,
            "best_quote_capture_minus_maker_round_trip": {
                profile: spread - values["maker_maker_bps"]
                for profile, values in costs.items()
            },
        },
        "costs": costs,
        "vip5_barrier_economics": {
            "maker_taker_cost_bps": vip5_cost,
            "target_net_bps": wins,
            "stop_net_bps": -losses,
            "break_even_win_rate": required_win_rate,
        },
        "vip5_oracle": {
            "positive_decision_fraction": float((oracle > 0).mean()),
            "flat_inclusive_expectancy_bps": float(oracle.mean()),
            "not_actionable": True,
        },
        "simple_rule_results": rule_results,
        "frozen_rolling_vwap_fade_frontier": distance_frontier,
        "state_policy_fit": fit_states,
        "state_policy_validation": validation,
        "decision": {
            "directional_state_policy_valid": all(
                result["trades"] >= 100
                and (result["expectancy_bps"] or float("-inf")) > 0
                and (result["profit_factor"] or 0.0) >= 1.10
                for result in validation.values()
            ),
            "best_quote_market_making_economically_valid": bool(
                spread - costs["VIP5"]["maker_maker_bps"] > 0
            ),
            "paper_policy_changed": False,
            "holdout_opened": False,
        },
    }
    directional = payload["decision"]["directional_state_policy_valid"]
    maker = payload["decision"]["best_quote_market_making_economically_valid"]
    payload["verdict"] = (
        "DIRECTIONAL_MICRO_POLICY_FOUND"
        if directional
        else "BEST_QUOTE_MARKET_MAKER_FOUND"
        if maker
        else "NO_FREQUENT_POLICY_IN_AVAILABLE_FEATURES"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(REPORT)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
