from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot import binance_l2_dataset, btc_vwap_alpha

ROOT = Path("data/raw/btc_context")
OUTPUT = Path("data/research/btc_cross_exchange_features.parquet")
EXCHANGES = ("binance", "bybit", "okx", "bitunix")
HORIZONS = (5, 15, 30)
L2_JOIN_FEATURES = (
    "bids",
    "asks",
    "best_bid",
    "best_ask",
    "spread_bps",
    "range_60s_bps",
    "mid_return_15m_bps",
    "mid_return_30m_bps",
    "rolling_vwap_5m_distance_bps",
    "aggressive_imbalance_5s",
    "aggressive_imbalance_30s",
    "aggressive_imbalance_60s",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_20",
    "microprice_distance_bps",
    "anchored_vwap_distance_bps",
    "anchor_direction",
    "anchor_age_seconds",
    "anchor_available_at",
    "aggressive_imbalance_1s",
    "aggressive_imbalance_3s",
    "aggressive_imbalance_15s",
    "trade_arrival_rate_30s",
    "bid_depth_5bps",
    "ask_depth_5bps",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_20bps",
    "ask_depth_20bps",
    "depth_imbalance_5bps",
    "depth_imbalance_10bps",
    "depth_imbalance_20bps",
    "bid_cancel_rate_5s",
    "ask_cancel_rate_5s",
    "spread_change_bps",
    "last_trade_update_age_ms",
    "last_book_update_age_ms",
    "clock_drift_ms",
    "book_is_synced",
    "trade_feed_alive",
    "orderbook_feed_alive",
    "sequence_gap_detected",
    "rolling_vwap",
    "rolling_vwap_slope_bps_60s",
    "rolling_vwap_slope_change_bps",
    "rolling_vwap_tests_30m",
    "rolling_vwap_rejections_30m",
    "time_since_last_vwap_cross_seconds",
    "volume_on_vwap_test",
    "vwap_rejection_strength_bps",
    "vwap_band_position",
    "anchored_vwap",
    "anchor_type",
    "anchor_timestamp",
    "anchor_age_bars",
    "anchor_price",
    "volume_since_anchor",
    "return_since_anchor_bps",
    "avwap_slope_bps_60s",
    "avwap_slope_change_bps",
    "avwap_tests",
    "avwap_rejections",
    "avwap_rejection_strength_bps",
    "rolling_vwap_vs_avwap_distance_bps",
    "rolling_vwap_avwap_convergence_bps",
    "atr_1m",
    "atr_5m",
    "atr_15m",
    "atr_30m",
    "realized_volatility",
    "range_n_bps",
    "volatility_percentile",
    "volume_percentile",
    "spread_percentile",
    "depth_percentile",
    "trend_1m",
    "trend_5m",
    "trend_15m",
    "trend_30m",
    "vwap_state_1m",
    "vwap_state_5m",
    "vwap_state_15m",
    "vwap_state_30m",
    "market_regime",
    "aggressive_buy_volume",
    "aggressive_sell_volume",
    "aggressor_ratio",
    "market_buy_sell_delta",
    "alpha_rolling_vwap",
    "alpha_recent_low_5m",
    "alpha_recent_high_5m",
    *sorted(
        column
        for column in btc_vwap_alpha.LIVE_REQUIRED
        if column.startswith("alpha_")
    ),
)


def load_snapshots(root: Path = ROOT) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("btc_context_*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                snapshot = json.loads(line)
                available = pd.Timestamp(snapshot["received_at"])
                minute = available.floor("min")
                for record in snapshot.get("records", []):
                    exchange = record.get("exchange")
                    if exchange not in EXCHANGES or not record.get("coverage"):
                        continue
                    price = record.get("mark_price") or record.get("last_price")
                    funding = pd.to_numeric(record.get("funding_rate"), errors="coerce")
                    if exchange == "bitunix" and pd.notna(funding):
                        # The current-rate endpoint emits the UI percentage value
                        # (for example -0.006569), while settled history emits the
                        # fractional rate (for example -0.00006569).
                        funding /= 100
                    rows.append(
                        {
                            "minute": minute,
                            "available_at": available,
                            "exchange": exchange,
                            "price": pd.to_numeric(price, errors="coerce"),
                            "basis_bps": pd.to_numeric(
                                record.get("mark_spot_basis_bps")
                                if exchange == "binance"
                                else record.get("mark_index_basis_bps"),
                                errors="coerce",
                            ),
                            "funding": funding,
                            "next_funding_timestamp": record.get("next_funding_timestamp"),
                            "open_interest": pd.to_numeric(
                                record.get("open_interest"), errors="coerce"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _wide(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    return rows.pivot(index="minute", columns="exchange", values=value).add_prefix(f"{value}_")


def build_features(
    rows: pd.DataFrame,
    l2: pd.DataFrame | None = None,
    *,
    alpha_clock: bool = False,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    if "next_funding_timestamp" not in rows:
        rows = rows.assign(next_funding_timestamp=pd.Series(pd.NaT, index=rows.index))
    rows = rows.sort_values("available_at").drop_duplicates(["minute", "exchange"], keep="last")
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True, format="mixed")
    # Alpha decisions are timed from Binance only. Other venues remain in the
    # research frame for diagnostics, but their arrival cannot delay or validate
    # a Binance signal.
    available = (
        rows.loc[rows["exchange"].eq("binance"), ["minute", "available_at"]]
        .set_index("minute")["available_at"]
        .rename("available_at")
    )
    frame = pd.concat(
        [
            available,
            *(
                _wide(rows, value)
                for value in (
                    "price",
                    "basis_bps",
                    "funding",
                    "next_funding_timestamp",
                    "open_interest",
                )
            ),
        ],
        axis=1,
    ).sort_index()
    if len(frame):
        frame = frame.reindex(
            pd.date_range(frame.index.min(), frame.index.max(), freq="1min", tz="UTC")
        )
        frame.index.name = "minute"
    for value in (
        "price",
        "basis_bps",
        "funding",
        "next_funding_timestamp",
        "open_interest",
    ):
        for exchange in EXCHANGES:
            column = f"{value}_{exchange}"
            if column not in frame:
                frame[column] = np.nan
    for exchange in EXCHANGES:
        price = frame[f"price_{exchange}"]
        frame[f"return_1m_{exchange}_bps"] = price.pct_change(fill_method=None) * 10_000
        frame[f"return_5m_{exchange}_bps"] = price.pct_change(5, fill_method=None) * 10_000
    return_1m = frame[[f"return_1m_{exchange}_bps" for exchange in EXCHANGES]]
    return_5m = frame[[f"return_5m_{exchange}_bps" for exchange in EXCHANGES]]
    frame["median_return_1m_bps"] = return_1m.median(axis=1)
    frame["dispersion_return_1m_bps"] = return_1m.std(axis=1)
    frame["median_return_5m_bps"] = return_5m.median(axis=1)
    frame["dispersion_return_5m_bps"] = return_5m.std(axis=1)
    external_1m = frame[["return_1m_bybit_bps", "return_1m_okx_bps"]].median(axis=1)
    frame["binance_lag_1m_bps"] = external_1m - frame["return_1m_binance_bps"]
    frame["funding_median"] = frame[[f"funding_{exchange}" for exchange in EXCHANGES]].median(
        axis=1
    )
    frame["funding_dispersion"] = frame[[f"funding_{exchange}" for exchange in EXCHANGES]].std(
        axis=1
    )
    funding_history = frame["funding_median"].shift(1).rolling(1_440, min_periods=240)
    frame["funding_z"] = (
        frame["funding_median"] - funding_history.mean()
    ) / funding_history.std().replace(0, np.nan)
    binance_funding_history = (
        frame["funding_binance"].shift(1).rolling(10_080, min_periods=1_440)
    )
    frame["funding_z_binance"] = (
        frame["funding_binance"] - binance_funding_history.mean()
    ) / binance_funding_history.std().replace(0, np.nan)
    frame["basis_median_bps"] = frame[[f"basis_bps_{exchange}" for exchange in EXCHANGES]].median(
        axis=1
    )
    for exchange in ("binance", "bybit", "okx"):
        frame[f"oi_change_1h_{exchange}"] = frame[f"open_interest_{exchange}"].pct_change(
            60, fill_method=None
        )
    frame["oi_change_1h_median"] = frame[
        [f"oi_change_1h_{exchange}" for exchange in ("binance", "bybit", "okx")]
    ].median(axis=1)
    frame = frame.reset_index()
    if l2 is not None and not l2.empty:
        causal_l2 = l2.loc[l2["feature_valid"]].sort_values("available_at").copy()
        causal_l2["alpha_l2_available_at"] = causal_l2["available_at"]
        missing_l2_features = [
            feature for feature in L2_JOIN_FEATURES if feature not in causal_l2
        ]
        if missing_l2_features:
            causal_l2 = pd.concat(
                [
                    causal_l2,
                    pd.DataFrame(
                        np.nan,
                        index=causal_l2.index,
                        columns=missing_l2_features,
                    ),
                ],
                axis=1,
            )
        causal_l2["available_at"] = pd.to_datetime(causal_l2["available_at"], utc=True)
        available_frame = frame.loc[frame["available_at"].notna()].sort_values(
            "available_at"
        )
        unavailable_frame = frame.loc[frame["available_at"].isna()].copy()
        if alpha_clock:
            alpha_frame = causal_l2[
                ["available_at", "alpha_l2_available_at", *L2_JOIN_FEATURES]
            ].copy()
            alpha_frame["minute"] = alpha_frame["available_at"].dt.floor("min")
            context_frame = available_frame.rename(
                columns={
                    "available_at": "context_available_at",
                    "minute": "context_minute",
                }
            )
            frame = pd.merge_asof(
                alpha_frame.sort_values("available_at"),
                context_frame.sort_values("context_available_at"),
                left_on="available_at",
                right_on="context_available_at",
                direction="backward",
                tolerance=pd.Timedelta(seconds=90),
            )
        else:
            frame = pd.merge_asof(
                available_frame,
                causal_l2[
                    ["available_at", "alpha_l2_available_at", *L2_JOIN_FEATURES]
                ],
                on="available_at",
                direction="backward",
                tolerance=pd.Timedelta(seconds=90),
            )
            if not unavailable_frame.empty:
                unavailable_frame = pd.concat(
                    [
                        unavailable_frame,
                        pd.DataFrame(
                            {
                                "alpha_l2_available_at": pd.Series(
                                    pd.NaT,
                                    index=unavailable_frame.index,
                                    dtype="datetime64[ns, UTC]",
                                ),
                                **{
                                    feature: pd.Series(
                                        np.nan, index=unavailable_frame.index
                                    )
                                    for feature in L2_JOIN_FEATURES
                                },
                            }
                        ),
                    ],
                    axis=1,
                )
                frame = pd.concat(
                    [frame, unavailable_frame], ignore_index=True
                ).sort_values("minute")
        frame = frame.reset_index(drop=True).copy()
    else:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "alpha_l2_available_at": pd.Series(
                            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
                        ),
                        **{
                            feature: pd.Series(np.nan, index=frame.index)
                            for feature in L2_JOIN_FEATURES
                        },
                    }
                ),
            ],
            axis=1,
        )
    required = [
        "price_binance",
        "return_1m_binance_bps",
        "return_5m_binance_bps",
        "rolling_vwap_5m_distance_bps",
    ]
    frame["feature_valid"] = frame[required].notna().all(axis=1)
    for horizon in HORIZONS:
        future = frame["price_binance"].shift(-horizon)
        future_minute = frame["minute"].shift(-horizon)
        exact = future_minute.sub(frame["minute"]).eq(pd.Timedelta(minutes=horizon))
        frame[f"future_return_{horizon}m_bps"] = (
            (future / frame["price_binance"] - 1) * 10_000
        ).where(exact)
        frame[f"label_available_at_{horizon}m"] = frame["available_at"].shift(-horizon).where(exact)
    return frame


def materialize() -> Path:
    l2 = binance_l2_dataset.build_features(binance_l2_dataset.load_records(binance_l2_dataset.ROOT))
    frame = build_features(load_snapshots(), l2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(materialize())
