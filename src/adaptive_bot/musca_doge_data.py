from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_bot.hybrid_policy_v25.data import _checked_archive, _parse_archive
from adaptive_bot.musca_v5_microstructure import build as build_microstructure

SYMBOL = "DOGEUSDT"
INFORMATIVE_SYMBOL = "BTCUSDT"
MONTHS = tuple(
    str(value) for value in pd.period_range("2025-01", "2026-07", freq="M")
)
OUTPUT = Path(f"data/ml/hybrid_v25/asset={SYMBOL}/minutes.parquet")
BTC_SOURCE = Path("data/ml/hybrid_v25/asset=BTCUSDT/minutes.parquet")
STATUS = Path("data/reports/musca_doge_auto_moe.status.json")
AUDIT = Path("data/reports/musca_doge_data_audit.json")


def _utc_ns(values: pd.Series) -> pd.Series:
    """Normalize mixed official archive precisions without changing instants."""
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _status(phase: str, detail: str, percent: float) -> None:
    _atomic_json(
        STATUS,
        {
            "symbol": SYMBOL,
            "phase": phase,
            "detail": detail,
            "percent": round(percent, 2),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _monthly_klines(market: str) -> tuple[pd.DataFrame, list[str]]:
    base = "spot" if market == "spot" else "futures/um"
    dataset = "markPriceKlines" if market == "mark" else "klines"
    prefix = f"{base}/monthly/{dataset}/{SYMBOL}/1m"
    frames: list[pd.DataFrame] = []
    paths: list[str] = []
    for number, month in enumerate(MONTHS, start=1):
        relative = f"{prefix}/{SYMBOL}-1m-{month}.zip"
        path = _checked_archive(relative)
        frames.append(_parse_archive(path))
        paths.append(str(path))
        _status(
            "official_minutes",
            f"{market} {number}/{len(MONTHS)}: {month}",
            2 + 18 * number / (3 * len(MONTHS)),
        )
    rows = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return rows, paths


def _funding() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    paths: list[str] = []
    for number, month in enumerate(MONTHS, start=1):
        relative = (
            f"futures/um/monthly/fundingRate/{SYMBOL}/"
            f"{SYMBOL}-fundingRate-{month}.zip"
        )
        path = _checked_archive(relative)
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise RuntimeError(f"unexpected Binance funding archive: {path}")
            frame = pd.read_csv(archive.open(members[0]))
        required = {"calc_time", "last_funding_rate"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"invalid Binance funding schema: {path}")
        frames.append(frame.loc[:, sorted(required)])
        paths.append(str(path))
        _status(
            "official_funding",
            f"funding {number}/{len(MONTHS)}: {month}",
            20 + 5 * number / len(MONTHS),
        )
    rows = pd.concat(frames, ignore_index=True)
    rows["funding_available_at"] = _utc_ns(
        pd.to_datetime(
            pd.to_numeric(rows["calc_time"], errors="raise"), unit="ms", utc=True
        )
    )
    rows["funding_rate"] = pd.to_numeric(
        rows["last_funding_rate"], errors="raise"
    )
    rows = (
        rows.loc[:, ["funding_available_at", "funding_rate"]]
        .drop_duplicates("funding_available_at", keep="last")
        .sort_values("funding_available_at")
        .reset_index(drop=True)
    )
    return rows, paths


def _btc_context(rows: pd.DataFrame) -> pd.DataFrame:
    if not BTC_SOURCE.exists():
        raise RuntimeError(f"BTC informative source is missing: {BTC_SOURCE}")
    btc = pd.read_parquet(
        BTC_SOURCE, columns=["timestamp", "available_at", "perp_close"]
    ).rename(
        columns={
            "available_at": "btc_context_available_at",
            "perp_close": "btc_close",
        }
    )
    btc["timestamp"] = _utc_ns(btc["timestamp"])
    btc["btc_context_available_at"] = _utc_ns(btc["btc_context_available_at"])
    output = rows.merge(btc, on="timestamp", how="left", validate="one_to_one")
    doge_close = output["perp_close"]
    btc_close = output["btc_close"]
    doge_return_1m = doge_close.pct_change(fill_method=None) * 10_000
    btc_return_1m = btc_close.pct_change(fill_method=None) * 10_000
    for minutes in (1, 5, 15, 60):
        output[f"btc_return_{minutes}m_bps"] = (
            btc_close.pct_change(minutes, fill_method=None) * 10_000
        )
    output["btc_realized_volatility_60m_bps"] = btc_return_1m.rolling(
        60, min_periods=60
    ).std(ddof=0)
    prior_doge = doge_return_1m.shift(1)
    prior_btc = btc_return_1m.shift(1)
    covariance = prior_doge.rolling(240, min_periods=60).cov(prior_btc)
    variance = prior_btc.rolling(240, min_periods=60).var(ddof=0)
    output["btc_beta_4h"] = covariance / variance.replace(0, np.nan)
    output["btc_correlation_1h"] = prior_doge.rolling(60, min_periods=30).corr(
        prior_btc
    )
    output["btc_correlation_4h"] = prior_doge.rolling(240, min_periods=60).corr(
        prior_btc
    )
    for minutes in (1, 5, 15):
        doge_return = doge_close.pct_change(minutes, fill_method=None) * 10_000
        output[f"residual_return_{minutes}m_bps"] = (
            doge_return - output["btc_beta_4h"] * output[f"btc_return_{minutes}m_bps"]
        )
    output["btc_shock_5m"] = output["btc_return_5m_bps"].abs() / output[
        "btc_realized_volatility_60m_bps"
    ].replace(0, np.nan)
    doge_return_5m = doge_close.pct_change(5, fill_method=None) * 10_000
    output["direction_agreement_5m"] = np.sign(doge_return_5m) * np.sign(
        output["btc_return_5m_bps"]
    )
    output["btc_context_coverage"] = (
        output["btc_close"].notna()
        & output["btc_context_available_at"].notna()
        & output["btc_context_available_at"].le(output["available_at"])
    )
    return output


def build_minutes(*, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    if OUTPUT.exists() and AUDIT.exists() and not force:
        return pd.read_parquet(OUTPUT), json.loads(AUDIT.read_text(encoding="utf-8"))
    perpetual, perp_paths = _monthly_klines("perpetual")
    spot, spot_paths = _monthly_klines("spot")
    mark, mark_paths = _monthly_klines("mark")
    funding, funding_paths = _funding()
    for frame in (perpetual, spot, mark):
        frame["timestamp"] = _utc_ns(frame["timestamp"])
        frame["available_at"] = _utc_ns(frame["available_at"])

    perpetual = perpetual.rename(
        columns={column: f"perp_{column}" for column in perpetual if column != "timestamp"}
    )
    spot = spot.rename(
        columns={column: f"spot_{column}" for column in spot if column != "timestamp"}
    )
    mark = mark.loc[:, ["timestamp", "close", "available_at"]].rename(
        columns={"close": "perp_mark_close", "available_at": "mark_available_at"}
    )
    rows = perpetual.merge(spot, on="timestamp", how="inner", validate="one_to_one")
    rows = rows.merge(mark, on="timestamp", how="inner", validate="one_to_one")
    rows["available_at"] = rows[
        ["perp_available_at", "spot_available_at", "mark_available_at"]
    ].max(axis=1)
    rows = pd.merge_asof(
        rows.sort_values("available_at"),
        funding,
        left_on="available_at",
        right_on="funding_available_at",
        direction="backward",
    ).rename(columns={"funding_rate": "perp_funding_rate"})
    event_rate = funding.assign(
        timestamp=funding["funding_available_at"].dt.floor("min")
    ).groupby("timestamp", as_index=False)["funding_rate"].sum()
    rows = rows.merge(event_rate, on="timestamp", how="left", validate="one_to_one").rename(
        columns={"funding_rate": "perp_funding_event_rate"}
    )
    rows["perp_funding_event_rate"] = rows["perp_funding_event_rate"].fillna(0.0)
    rows["context_available_at"] = pd.NaT
    rows["oi_change_1h"] = np.nan
    rows["context_coverage"] = False
    rows["perp_data_valid"] = True
    rows["is_available"] = rows[
        [
            "perp_open",
            "perp_high",
            "perp_low",
            "perp_close",
            "perp_volume",
            "perp_quote_volume",
            "perp_taker_buy_quote",
            "perp_trade_count",
            "spot_close",
            "perp_mark_close",
            "perp_funding_rate",
        ]
    ].notna().all(axis=1)
    rows = _btc_context(rows)
    rows["max_input_available_at"] = rows[
        ["available_at", "btc_context_available_at"]
    ].max(axis=1)
    rows["available_at"] = rows["max_input_available_at"]
    rows["is_available"] &= rows["btc_context_coverage"]
    rows = rows.sort_values("timestamp").reset_index(drop=True)
    if rows["timestamp"].duplicated().any() or not rows["timestamp"].is_monotonic_increasing:
        raise RuntimeError("DOGE official minute timestamps are duplicate or unordered")
    if not rows["available_at"].ge(rows["timestamp"] + pd.Timedelta(minutes=1)).all():
        raise RuntimeError("DOGE minute feature availability is non-causal")
    _atomic_parquet(OUTPUT, rows)
    audit = {
        "symbol": SYMBOL,
        "informative_symbol": INFORMATIVE_SYMBOL,
        "rows": len(rows),
        "start": rows["timestamp"].min().isoformat(),
        "end": rows["timestamp"].max().isoformat(),
        "available_coverage": float(rows["is_available"].mean()),
        "btc_context_coverage": float(rows["btc_context_coverage"].mean()),
        "open_interest": "excluded_no_common_multi_year_official_archive",
        "archives": {
            "perpetual": len(perp_paths),
            "spot": len(spot_paths),
            "mark": len(mark_paths),
            "funding": len(funding_paths),
        },
        "output_sha256": _sha256(OUTPUT),
        "holdout_rows_read": 0,
    }
    _atomic_json(AUDIT, audit)
    _status("official_minutes_complete", f"{len(rows):,} causal DOGE minutes", 25)
    return rows, audit


def build(*, force: bool = False) -> dict[str, Any]:
    _, minute_audit = build_minutes(force=force)
    micro_audit = build_microstructure(
        SYMBOL,
        MONTHS,
        (5,),
        STATUS,
    )
    return {"minutes": minute_audit, "microstructure": micro_audit}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DOGEUSDT official Binance research data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--minutes-only", action="store_true")
    args = parser.parse_args()
    try:
        payload = (
            {"minutes": build_minutes(force=args.force)[1]}
            if args.minutes_only
            else build(force=args.force)
        )
    except Exception as error:
        _status("failed", str(error), 0)
        raise
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
