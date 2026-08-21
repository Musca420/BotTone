from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from adaptive_bot import (
    binance_l2_dataset,
    btc_vwap_alpha,
    musca_btc_auto_moe,
)
from adaptive_bot.musca_v5_execution import (
    quote_taker_round_trip,
    size_for_technical_stop,
)
from adaptive_bot.musca_v5_paper import advance_account, new_paper_account

CONFIG = Path("configs/binance_btcusdt_paper.yaml")
STATE = Path("data/research/musca_btc_auto_moe_paper_state.json")
REPORT = Path("data/reports/musca_v8_binance_paper.json")
SHADOW_REPORT = Path("data/reports/musca_v8_binance_shadow.json")
PROFILE = "BINANCE"
STRATEGY_PROFILE = "musca_btc_auto_moe_vwap"
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
COMMISSION_URL = "https://fapi.binance.com/fapi/v1/commissionRate"
OPEN_INTEREST_HISTORY_URL = (
    "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=30"
)
FUNDING_HISTORY_URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=200"
MARK_KLINES_URL = (
    "https://fapi.binance.com/fapi/v1/markPriceKlines?symbol=BTCUSDT&interval=1m&limit=5"
)
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5"
_FUNDING_CACHE: tuple[float, dict[str, Any]] | None = None
_CONTEXT_CACHE: tuple[float, pd.DataFrame] | None = None


@dataclass(frozen=True)
class BinancePaperConfig:
    symbol: str
    maker_fee_bps: float
    taker_fee_bps: float
    non_fee_reserve_bps: float
    fee_source: str
    max_book_age_seconds: float

    @property
    def modeled_round_trip_cost_bps(self) -> float:
        return 2 * self.taker_fee_bps + self.non_fee_reserve_bps


@dataclass(frozen=True)
class FeeSchedule:
    maker_bps: float
    taker_bps: float
    source: str


_FEE_CACHE: tuple[float, FeeSchedule] | None = None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def load_config(path: Path = CONFIG) -> BinancePaperConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("mode") != "paper":
        raise ValueError("Binance Musca Auto-MoE requires an explicit paper configuration")
    if raw.get("live_trading_enabled") is not False:
        raise ValueError("live trading must remain disabled")
    execution = raw.get("execution")
    risk = raw.get("risk")
    if not isinstance(execution, dict) or not isinstance(risk, dict):
        raise ValueError("Binance paper execution and risk configuration are required")
    if float(risk.get("initial_equity", 0)) != 10_000:
        raise ValueError("the frozen paper protocol starts from 10,000 USDT")
    if (
        float(risk.get("risk_per_trade", 0)) != 0.01
        or float(risk.get("margin_fraction", 0)) != 0.10
        or float(risk.get("max_leverage", 0)) != 10
    ):
        raise ValueError("the frozen risk protocol is 1% risk, 10% margin and 10x cap")
    values = BinancePaperConfig(
        symbol=str(raw.get("symbol")),
        maker_fee_bps=float(execution["maker_fee_bps"]),
        taker_fee_bps=float(execution["taker_fee_bps"]),
        non_fee_reserve_bps=float(execution["non_fee_reserve_bps"]),
        fee_source=str(execution["fee_source"]),
        max_book_age_seconds=float(execution["max_book_age_seconds"]),
    )
    if (
        values.symbol != "BTCUSDT"
        or min(
            values.maker_fee_bps,
            values.taker_fee_bps,
            values.non_fee_reserve_bps,
            values.max_book_age_seconds,
        )
        < 0
    ):
        raise ValueError("invalid Binance paper configuration")
    return values


def _fetch_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BotTone-MuscaAutoMoE/1",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def fee_schedule(
    config: BinancePaperConfig,
    *,
    fetch: Callable[..., Any] = _fetch_json,
) -> FeeSchedule:
    """Read the signed account rate, or return the labelled config fallback."""
    global _FEE_CACHE
    now = time.monotonic()
    if _FEE_CACHE is not None and now - _FEE_CACHE[0] < 3_600:
        return _FEE_CACHE[1]
    key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    result = FeeSchedule(
        config.maker_fee_bps,
        config.taker_fee_bps,
        f"CONFIG_FALLBACK:{config.fee_source}",
    )
    if key and secret:
        params = {
            "symbol": config.symbol,
            "recvWindow": "5000",
            "timestamp": str(int(datetime.now(UTC).timestamp() * 1_000)),
        }
        query = urllib.parse.urlencode(params)
        signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        payload = fetch(
            f"{COMMISSION_URL}?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": key},
        )
        if not isinstance(payload, dict) or payload.get("symbol") != config.symbol:
            raise ValueError("Binance commissionRate returned an invalid response")
        result = FeeSchedule(
            float(payload["makerCommissionRate"]) * 10_000,
            float(payload["takerCommissionRate"]) * 10_000,
            "BINANCE_SIGNED_COMMISSION_RATE",
        )
    _FEE_CACHE = (now, result)
    return result


def funding_snapshot(*, fetch: Callable[..., Any] = _fetch_json) -> dict[str, Any]:
    global _FUNDING_CACHE
    now = time.monotonic()
    if _FUNDING_CACHE is not None and now - _FUNDING_CACHE[0] < 60:
        return _FUNDING_CACHE[1]
    payload = fetch(PREMIUM_INDEX_URL)
    if not isinstance(payload, dict) or payload.get("symbol") != "BTCUSDT":
        raise ValueError("Binance premiumIndex returned an invalid response")
    observed_at = pd.to_datetime(int(payload["time"]), unit="ms", utc=True)
    mark_price = float(payload["markPrice"])
    index_price = float(payload["indexPrice"])
    result = {
        "observed_at": observed_at.isoformat(),
        "mark_price": mark_price,
        "index_price": index_price,
        "funding_rate": float(payload["lastFundingRate"]),
        "next_funding_timestamp": pd.to_datetime(
            int(payload["nextFundingTime"]), unit="ms", utc=True
        ).isoformat(),
    }
    if min(mark_price, index_price) <= 0:
        raise ValueError("Binance mark/index prices must be positive")
    _FUNDING_CACHE = (now, result)
    return result


def _closed_kline_close(
    payload: Any, candle_timestamp: pd.Timestamp, observed_at: pd.Timestamp
) -> tuple[float, pd.Timestamp]:
    if not isinstance(payload, list):
        raise ValueError("Binance kline history returned an invalid response")
    for row in payload:
        if not isinstance(row, list) or len(row) < 7:
            continue
        opened = pd.to_datetime(int(row[0]), unit="ms", utc=True)
        closed = pd.to_datetime(int(row[6]), unit="ms", utc=True)
        price = float(row[4])
        if opened == candle_timestamp and closed <= observed_at and price > 0:
            return price, closed
    raise ValueError("Binance closed basis candle is unavailable")


def derivatives_alpha_features(
    *,
    candle_timestamp: pd.Timestamp,
    observed_at: pd.Timestamp,
    return_5m_bps: float,
    fetch: Callable[..., Any] = _fetch_json,
) -> dict[str, Any]:
    """Recreate the four historical derivatives features from public Binance data."""
    candle_timestamp = (
        candle_timestamp.tz_localize("UTC")
        if candle_timestamp.tzinfo is None
        else candle_timestamp.tz_convert("UTC")
    )
    observed_at = (
        observed_at.tz_localize("UTC")
        if observed_at.tzinfo is None
        else observed_at.tz_convert("UTC")
    )
    oi_payload = fetch(OPEN_INTEREST_HISTORY_URL)
    funding_payload = fetch(FUNDING_HISTORY_URL)
    mark_payload = fetch(MARK_KLINES_URL)
    spot_payload = fetch(SPOT_KLINES_URL)
    if not isinstance(oi_payload, list):
        raise ValueError("Binance open-interest history returned an invalid response")

    oi = pd.DataFrame(oi_payload)
    if not {"timestamp", "sumOpenInterest"}.issubset(oi.columns):
        raise ValueError("Binance open-interest history is incomplete")
    oi["timestamp"] = pd.to_datetime(
        pd.to_numeric(oi["timestamp"], errors="raise"), unit="ms", utc=True
    )
    oi["open_interest"] = pd.to_numeric(oi["sumOpenInterest"], errors="raise")
    oi = oi.loc[oi["timestamp"].le(observed_at)].sort_values("timestamp").tail(13)
    if (
        len(oi) != 13
        or oi["timestamp"].duplicated().any()
        or observed_at - oi["timestamp"].iloc[-1] > pd.Timedelta(minutes=10)
        or float(oi["open_interest"].iloc[0]) <= 0
    ):
        raise ValueError("Binance open-interest 1h coverage is stale or incomplete")
    oi_change_1h = float(oi["open_interest"].iloc[-1] / oi["open_interest"].iloc[0] - 1)

    if not isinstance(funding_payload, list):
        raise ValueError("Binance funding history returned an invalid response")
    funding = pd.DataFrame(funding_payload)
    if not {"fundingTime", "fundingRate"}.issubset(funding.columns):
        raise ValueError("Binance funding history is incomplete")
    funding["timestamp"] = pd.to_datetime(
        pd.to_numeric(funding["fundingTime"], errors="raise"), unit="ms", utc=True
    ).dt.as_unit("ns")
    funding["rate"] = pd.to_numeric(funding["fundingRate"], errors="raise")
    funding = funding.loc[funding["timestamp"].le(observed_at)].sort_values("timestamp")
    if funding.empty or funding["timestamp"].duplicated().any():
        raise ValueError("Binance funding history is unavailable")
    minute_grid = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                observed_at.floor("min") - pd.Timedelta(minutes=10_080),
                observed_at.floor("min"),
                freq="1min",
            ).as_unit("ns")
        }
    )
    minute_rates = pd.merge_asof(
        minute_grid,
        funding[["timestamp", "rate"]],
        on="timestamp",
        direction="backward",
    )["rate"]
    history = minute_rates.iloc[:-1].dropna()
    if len(history) < 1_440 or not pd.notna(minute_rates.iloc[-1]):
        raise ValueError("Binance funding z-score warm-up is incomplete")
    funding_std = float(history.std())
    if not funding_std > 0:
        raise ValueError("Binance funding history has zero dispersion")
    funding_z = float((float(minute_rates.iloc[-1]) - float(history.mean())) / funding_std)

    mark_close, mark_closed_at = _closed_kline_close(mark_payload, candle_timestamp, observed_at)
    spot_close, spot_closed_at = _closed_kline_close(spot_payload, candle_timestamp, observed_at)
    basis_bps = (mark_close / spot_close - 1) * 10_000
    values = np.array(
        [oi_change_1h, funding_z, basis_bps, return_5m_bps * oi_change_1h],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Binance derivatives features are non-finite")
    return {
        "oi_change_1h": oi_change_1h,
        "return_oi_interaction_raw": float(return_5m_bps * oi_change_1h),
        "basis_bps": float(basis_bps),
        "funding_z": funding_z,
        "derivatives_feature_valid": True,
        "derivatives_feature_available_at": observed_at.isoformat(),
        "derivatives_source_timestamps": {
            "open_interest": oi["timestamp"].iloc[-1].isoformat(),
            "funding": funding["timestamp"].iloc[-1].isoformat(),
            "mark": mark_closed_at.isoformat(),
            "spot": spot_closed_at.isoformat(),
        },
    }


def market_context() -> pd.DataFrame:
    global _CONTEXT_CACHE
    now = time.monotonic()
    if _CONTEXT_CACHE is not None and now - _CONTEXT_CACHE[0] < 20:
        return _CONTEXT_CACHE[1]
    minutes = binance_l2_dataset.load_recent_official_minutes()
    official_minute_rows = len(minutes)
    context = btc_vwap_alpha.canonical_minute_market_features(minutes)
    context = context.loc[context["feature_contract_valid"].fillna(False)].copy()
    if context.empty:
        raise ValueError("Binance minute Alpha has not completed its causal warm-up")
    latest = context.iloc[-1]
    derivatives = derivatives_alpha_features(
        candle_timestamp=pd.Timestamp(latest["timestamp"]),
        observed_at=pd.Timestamp.now(tz="UTC"),
        return_5m_bps=float(latest["return_5m_bps"]),
    )
    for name in (
        "oi_change_1h",
        "return_oi_interaction_raw",
        "basis_bps",
        "funding_z",
        "derivatives_feature_valid",
        "derivatives_feature_available_at",
    ):
        context.at[context.index[-1], name] = derivatives[name]
    context.attrs["derivatives_source_timestamps"] = derivatives["derivatives_source_timestamps"]
    context.attrs["official_minute_rows"] = official_minute_rows
    _CONTEXT_CACHE = (now, context)
    return context


def latest_book(evaluated_at: pd.Timestamp, config: BinancePaperConfig) -> pd.DataFrame:
    rows = binance_l2_dataset.load_recent_records(max_lines=30)
    if rows.empty:
        return rows
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True, format="mixed")
    rows = rows.loc[rows["available_at"].le(evaluated_at)].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values("exchange_second").tail(10).reset_index(drop=True)
    latest = rows.iloc[-1]
    bids, asks = latest.get("bids"), latest.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        return pd.DataFrame()
    bid, ask = Decimal(str(bids[0][0])), Decimal(str(asks[0][0]))
    available = pd.Timestamp(latest["available_at"])
    exchange_at = pd.to_datetime(int(latest["exchange_second"]), unit="s", utc=True)
    gaps = pd.to_numeric(rows["exchange_second"], errors="coerce").diff().dropna()
    valid = (
        latest.get("source") == "binance-official-usdm-websocket-routed"
        and 0 < bid < ask
        and exchange_at <= available <= evaluated_at
        and (evaluated_at - available).total_seconds() <= config.max_book_age_seconds
        and (gaps.le(2).all() if len(gaps) else True)
    )
    mid = (bid + ask) / 2
    return pd.DataFrame(
        [
            {
                "available_at": available,
                "mid": float(mid),
                "best_bid": float(bid),
                "best_ask": float(ask),
                "bids": bids,
                "asks": asks,
                "spread_bps": float((ask - bid) / mid * Decimal("10000")),
                "feature_valid": bool(valid),
            }
        ]
    )


def _source(
    name: str,
    purpose: str,
    observed_at: object,
    evaluated_at: pd.Timestamp,
    *,
    maximum_age_seconds: float,
    structurally_valid: bool,
) -> dict[str, Any]:
    try:
        observed = pd.Timestamp(str(observed_at))
        observed = (
            observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
        )
        age = (evaluated_at - observed).total_seconds()
    except (TypeError, ValueError):
        observed, age = None, None
    valid = bool(
        structurally_valid
        and observed is not None
        and age is not None
        and 0 <= age <= maximum_age_seconds
    )
    return {
        "name": name,
        "purpose": purpose,
        "state": "FRESH" if valid else "STALE_OR_INVALID",
        "valid": valid,
        "observed_at": observed.isoformat() if observed is not None else None,
        "age_seconds": age,
        "max_age_seconds": maximum_age_seconds,
    }


def build_assessment(
    *,
    candidate: dict[str, Any] | None,
    context: pd.DataFrame,
    book: pd.DataFrame,
    funding: dict[str, Any],
    fees: FeeSchedule,
    config: BinancePaperConfig,
    equity: float,
    evaluated_at: pd.Timestamp,
    model_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = context.iloc[-1]
    book_row = book.iloc[-1] if not book.empty else pd.Series(dtype=object)
    alpha_source = _source(
        "Binance",
        "Alpha: closed BTCUSDT perpetual/spot candles and causal VWAP event",
        latest.get("available_at"),
        evaluated_at,
        maximum_age_seconds=musca_btc_auto_moe.LIVE_ALPHA_MAX_AGE_SECONDS,
        structurally_valid=bool(latest.get("feature_contract_valid", False))
        and bool(latest.get("derivatives_feature_valid", False)),
    )
    execution_source = _source(
        "Binance",
        "Paper execution: observed USD-M depth, mark, funding and account fee",
        book_row.get("available_at"),
        evaluated_at,
        maximum_age_seconds=config.max_book_age_seconds,
        structurally_valid=bool(book_row.get("feature_valid", False)),
    )
    funding_source = _source(
        "Binance",
        "Public mark price and current funding schedule",
        funding.get("observed_at"),
        evaluated_at,
        maximum_age_seconds=120,
        structurally_valid=True,
    )
    sources_valid = all(
        source["valid"] for source in (alpha_source, execution_source, funding_source)
    )
    model_evaluation = model_evaluation or {
        "status": "READY_CANDIDATE" if candidate else "READY_FLAT",
        "reason": "LEGACY_CALL",
        "feature_coverage": {"complete": True, "missing": []},
        "frozen_expert_count": None,
        "active_expert_count": int(candidate is not None),
        "horizons_minutes": [360],
        "best_action": None,
        "alternatives": [],
    }
    model_ready = str(model_evaluation.get("status", "")).startswith("READY_")
    best_action = model_evaluation.get("best_action")
    if not isinstance(best_action, dict):
        best_action = {}
    direction = candidate.get("direction") if candidate else best_action.get("direction")
    sign = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
    stop_bps: float | None = None
    quote = risk = None
    expected_funding_bps = 0.0
    expected_cost_bps = config.modeled_round_trip_cost_bps
    expected_net_ev_bps: float | None = (
        float(best_action["calibrated_ev_bps"])
        + musca_btc_auto_moe.TRAINED_ROUND_TRIP_COST_BPS
        - expected_cost_bps
        if best_action.get("calibrated_ev_bps") is not None
        else None
    )
    target_bps: float | None = None
    if candidate and not book.empty and sign:
        mid = float(book_row["mid"])
        stop_bps = (
            float(candidate["stop_bps"])
            if candidate.get("stop_bps") is not None
            else sign * (mid - float(candidate["stop_price"])) / mid * 10_000
        )
        next_funding = pd.Timestamp(funding["next_funding_timestamp"])
        crosses_funding = next_funding <= evaluated_at + pd.Timedelta(
            minutes=int(candidate["maximum_hold_minutes"])
        )
        expected_funding_bps = (
            max(0.0, sign * float(funding["funding_rate"]) * 10_000) if crosses_funding else 0.0
        )
        if sources_valid and 0 < stop_bps <= 200:
            for quantity in (Decimal("0.001"),):
                quote = quote_taker_round_trip(
                    side=str(direction),
                    quantity=quantity,
                    bids=book_row["bids"],
                    asks=book_row["asks"],
                    best_bid=Decimal(str(book_row["best_bid"])),
                    best_ask=Decimal(str(book_row["best_ask"])),
                    taker_fee_bps=Decimal(str(fees.taker_bps)),
                    expected_funding_bps=Decimal(str(expected_funding_bps)),
                )
            if quote is not None:
                for _ in range(2):
                    expected_cost_bps = (
                        float(quote.expected_total_cost_bps) + config.non_fee_reserve_bps
                    )
                    risk = size_for_technical_stop(
                        equity=Decimal(str(equity)),
                        entry_price=quote.entry.execution_vwap,
                        side=str(direction),
                        technical_stop_bps=Decimal(str(stop_bps)),
                        expected_cost_bps=Decimal(str(expected_cost_bps)),
                        lot_size=Decimal("0.001"),
                        minimum_quantity=Decimal("0.001"),
                        minimum_notional=Decimal("5"),
                    )
                    if not risk.approved:
                        break
                    sized = quote_taker_round_trip(
                        side=str(direction),
                        quantity=risk.quantity,
                        bids=book_row["bids"],
                        asks=book_row["asks"],
                        best_bid=Decimal(str(book_row["best_bid"])),
                        best_ask=Decimal(str(book_row["best_ask"])),
                        taker_fee_bps=Decimal(str(fees.taker_bps)),
                        expected_funding_bps=Decimal(str(expected_funding_bps)),
                    )
                    if sized is None:
                        quote = None
                        break
                    quote = sized
        expected_net_ev_bps = float(candidate["robust_expected_gross_bps"]) - expected_cost_bps
        target_bps = float(candidate.get("target_1_bps", 1.5 * stop_bps)) if stop_bps else None
    approved = bool(
        candidate
        and sources_valid
        and quote is not None
        and risk is not None
        and risk.approved
        and expected_net_ev_bps is not None
        and expected_net_ev_bps > 0
        and target_bps is not None
    )
    decision = "TRADE" if approved else "FLAT" if sources_valid and model_ready else "WAIT"
    reason = (
        "BASE_ALPHA_NET_POSITIVE"
        if approved
        else "BINANCE_DATA_FAIL_CLOSED"
        if not sources_valid
        else "MODEL_INPUT_FAIL_CLOSED"
        if not model_ready
        else "FLAT_NO_POSITIVE_AUTO_MOE_ACTION"
        if candidate is None
        else "INVALID_STRUCTURAL_STOP"
        if stop_bps is None or not 0 < stop_bps <= 200
        else "INSUFFICIENT_BINANCE_DEPTH"
        if quote is None
        else risk.reason
        if risk is not None and not risk.approved
        else "BASE_ALPHA_COST_BLOCKED"
    )
    entry = float(quote.entry.execution_vwap) if quote is not None else None
    setup = candidate or {
        "setup": "AUTO_MOE_VWAP_CONTROLLER",
        "direction": best_action.get("direction"),
        "candidate": False,
        "setup_active": bool(model_evaluation.get("active_expert_count", 0)),
        "passed_checks": 0,
        "total_checks": 1,
        "first_failed_check": (
            "auto_moe_input_contract" if not model_ready else "auto_moe_calibrated_ev_positive"
        ),
        "checks": [
            {
                "name": "auto_moe_calibrated_ev_positive",
                "passed": False,
                "actual": best_action.get("calibrated_ev_bps"),
                "requirement": "> 0 bps netti",
            }
        ],
        "policy_source": "BTC_AUTO_MOE_RESEARCH_PAPER_MONITOR",
        "expert_id": best_action.get("expert_id"),
        "maximum_hold_minutes": best_action.get("horizon_minutes", 360),
    }
    context_observed = pd.Timestamp(latest["available_at"]).isoformat()
    return {
        "evaluated_at": evaluated_at.isoformat(),
        "observed_at": (
            candidate.get("available_at", context_observed) if candidate else context_observed
        ),
        "decision": decision,
        "reason": reason,
        "fee_profile": PROFILE,
        "setup": setup["setup"],
        "direction": direction,
        "policy_source": setup.get("policy_source", "MUSCA_AUTO_MOE_FROZEN_BASE"),
        "expert_id": setup.get("expert_id"),
        "alpha_signal_at": setup.get("available_at"),
        "management_style": setup.get("management_style", "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M"),
        "partial_target_fraction": setup.get("partial_target_fraction", 0.5),
        "maximum_hold_minutes": int(setup.get("maximum_hold_minutes", 360)),
        "candidate_complete": candidate is not None,
        "model_context": (
            "COMPLETE_CANDIDATE"
            if candidate
            else "READY_FLAT_NO_POSITIVE_EV"
            if model_ready
            else "MODEL_INPUT_INCOMPLETE"
        ),
        "model_feature_coverage": model_evaluation.get(
            "feature_coverage", {"complete": False, "missing": []}
        ),
        "probability_status": (
            "AUTO_MOE_CALIBRATED_ENTRY"
            if candidate
            else "AUTO_MOE_READY_FLAT"
            if model_ready
            else "AUTO_MOE_INPUT_FAIL_CLOSED"
        ),
        "target_probability": (
            candidate.get("target_probability")
            if candidate
            else best_action.get("probability_net_positive")
        ),
        "stop_probability": None,
        "timeout_probability": None,
        "expected_net_ev_bps": expected_net_ev_bps,
        "expected_mfe_60m_bps": None,
        "expected_mae_60m_bps": None,
        "expected_time_to_target_minutes": None,
        "vip_ev_bps": {PROFILE: expected_net_ev_bps},
        "price": float(book_row["mid"]) if not book.empty else float(latest["close"]),
        "execution_price": float(book_row["mid"]) if not book.empty else None,
        "rolling_vwap": float(latest["rolling_vwap"]),
        "anchored_vwap": (
            candidate.get("operating_vwap") if candidate else float(latest["rolling_vwap"])
        ),
        "spread_bps": float(book_row["spread_bps"]) if not book.empty else None,
        "flow_vote": float(latest["taker_imbalance_60s"]),
        "binance_return_1m_bps": float(latest["return_1m_bps"]),
        "binance_return_5m_bps": float(latest["return_5m_bps"]),
        "vwap_distance_bps": float(latest["vwap_distance_bps"]),
        "stop_bps": stop_bps,
        "target_bps": target_bps,
        "target_2_bps": candidate.get("target_2_bps") if candidate else None,
        "trailing_bps": candidate.get("trailing_bps") if candidate else None,
        "expected_cost_bps": expected_cost_bps,
        "expected_funding_bps": expected_funding_bps,
        "execution_status": "QUOTED" if quote is not None else "NOT_EVALUATED",
        "execution_type": quote.execution_type if quote is not None else None,
        "entry_execution_vwap": entry,
        "estimated_exit_execution_vwap": (
            float(quote.estimated_exit.execution_vwap) if quote is not None else None
        ),
        "execution_levels_entry": quote.entry.levels_consumed if quote is not None else None,
        "risk_status": (
            "APPROVED"
            if risk is not None and risk.approved
            else "NOT_EVALUATED"
            if risk is None
            else "REJECTED"
        ),
        "risk_approved": risk.approved if risk is not None else None,
        "risk_reason": risk.reason if risk is not None else "NOT_EVALUATED_NO_CANDIDATE",
        "risk_budget": float(risk.risk_budget) if risk is not None else None,
        "notional": float(risk.notional) if risk is not None else None,
        "quantity_btc": float(risk.quantity) if risk is not None else None,
        "break_even_price": entry * (1 + sign * expected_cost_bps / 10_000) if entry else None,
        "stop_price": entry * (1 - sign * stop_bps / 10_000) if entry and stop_bps else None,
        "target_price": entry * (1 + sign * target_bps / 10_000) if entry and target_bps else None,
        "target_2_price": (
            entry * (1 + sign * float(candidate["target_2_bps"]) / 10_000)
            if entry and candidate and candidate.get("target_2_bps")
            else None
        ),
        "anchor": {
            "state": "OPERATING_VWAP",
            "valid": True,
            "detected_at": candidate.get("impulse_anchor_at") if candidate else None,
            "age_seconds": None,
            "direction": direction,
            "price": (
                candidate.get("operating_vwap") if candidate else float(latest["rolling_vwap"])
            ),
        },
        "sources": {
            "alpha": alpha_source,
            "execution": execution_source,
            "funding": funding_source,
        },
        "market_inputs": {
            "binance": {
                "price": float(book_row["mid"]) if not book.empty else float(latest["close"]),
                "return_1m_bps": float(latest["return_1m_bps"]),
                "return_5m_bps": float(latest["return_5m_bps"]),
                "return_15m_bps": float(latest["return_15m_bps"]),
                "return_30m_bps": float(latest["return_30m_bps"]),
                "rolling_vwap": float(latest["rolling_vwap"]),
                "vwap_distance_bps": float(latest["vwap_distance_bps"]),
                "vwap_slope_bps": float(latest["vwap_slope_bps"]),
                "range_60s_bps": float(latest["range_60s_bps"]),
                "taker_imbalance_60s": float(latest["taker_imbalance_60s"]),
                "oi_change_1h": float(latest["oi_change_1h"]),
                "basis_bps": float(latest["basis_bps"]),
                "funding_z": float(latest["funding_z"]),
                "ofi_1m": model_evaluation.get("micro_features", {}).get("ofi_1m"),
                "ofi_5m": model_evaluation.get("micro_features", {}).get("ofi_5m"),
                "trade_intensity_1m": model_evaluation.get("micro_features", {}).get(
                    "trade_intensity_1m"
                ),
                "price_velocity_1m": model_evaluation.get("micro_features", {}).get(
                    "price_velocity_1m"
                ),
                "mid": float(book_row["mid"]) if not book.empty else None,
                "mark_price": funding["mark_price"],
                "index_price": funding["index_price"],
                "best_bid": float(book_row["best_bid"]) if not book.empty else None,
                "best_ask": float(book_row["best_ask"]) if not book.empty else None,
                "spread_bps": float(book_row["spread_bps"]) if not book.empty else None,
                "funding_rate": funding["funding_rate"],
                "next_funding_timestamp": funding["next_funding_timestamp"],
                "book_synced": bool(book_row.get("feature_valid", False)),
            }
        },
        "setups": [setup],
        "model_evaluation": {
            key: model_evaluation.get(key)
            for key in (
                "status",
                "reason",
                "frozen_expert_count",
                "active_expert_count",
                "best_action",
                "alternatives",
            )
        },
        "evaluation_frequency": (
            "Auto-MoE decision on completed Binance 1m context; execution on next Binance book"
        ),
        "outcome_horizons_minutes": model_evaluation.get("horizons_minutes", [360]),
    }


def _load_account(fees: FeeSchedule) -> dict[str, Any]:
    if STATE.exists():
        payload = json.loads(STATE.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("account"), dict):
            raise ValueError("Binance paper state has an unsupported schema")
        account = cast(dict[str, Any], payload["account"])
        if (
            float(account.get("maker_fees_per_side_bps", -1)) != fees.maker_bps
            or float(account.get("fees_per_side_bps", -1)) != fees.taker_bps
        ):
            account["risk_block_reason"] = "FEE_SCHEDULE_CHANGED_NEW_PAPER_RUN_REQUIRED"
        return account
    started = datetime.now(UTC).isoformat()
    return new_paper_account(
        PROFILE,
        started,
        execution_venue="BINANCE",
        maker_fee_bps=fees.maker_bps,
        taker_fee_bps=fees.taker_bps,
    )


def _historical_alpha(config: BinancePaperConfig, fees: FeeSchedule) -> dict[str, Any]:
    report = json.loads(musca_btc_auto_moe.REPORT.read_text(encoding="utf-8"))
    cost = 2 * fees.taker_bps + config.non_fee_reserve_bps
    trained_cost = float(report.get("protocol", {}).get("round_trip_cost_bps", -1))
    return {
        "status": report.get("verdict"),
        "protocol_hash": report.get("protocol_hash"),
        "historical_audit": report.get("historical_audit"),
        "paper_profiles": {PROFILE: report.get("historical_audit")},
        "cost_matched_without_retraining": abs(trained_cost - cost) < 1e-9,
        "modeled_round_trip_cost_bps": cost,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }


def _chart(context: pd.DataFrame, assessment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": pd.Timestamp(row["available_at"]).isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "execution_price": None,
            "center": float(row["rolling_vwap"]),
            "anchored_vwap": None,
            "break_even": assessment.get("break_even_price"),
            "stop": assessment.get("stop_price"),
            "target": assessment.get("target_price"),
        }
        for _, row in context.tail(360).iterrows()
    ]


def _shadow(audit: dict[str, Any]) -> dict[str, Any]:
    account = audit["one_position_diagnostics"]["paper_account"]
    assessment = audit["current_market_assessment"]
    trades = account.get("trades", [])
    fills: list[dict[str, Any]] = []
    for number, trade in enumerate(trades):
        for entry in (True, False):
            fills.append(
                {
                    "exchange_timestamp": trade["entry_at"] if entry else trade["exit_at"],
                    "received_timestamp": trade["entry_at"] if entry else trade["exit_at"],
                    "source": "musca_btc_auto_moe_observed_depth_paper",
                    "instrument": "BTCUSDT",
                    "client_order_id": f"musca-auto-moe-{2 * number + int(not entry)}",
                    "side": "buy" if (trade["side"] == "LONG") == entry else "sell",
                    "price": str(
                        trade["entry_execution_price"] if entry else trade["exit_execution_price"]
                    ),
                    "quantity": str(trade["quantity_btc"]),
                    "commission": str(float(trade["fees"]) / 2),
                    "slippage": str(float(trade.get("realized_slippage", 0)) / 2),
                    "liquidity_role": "TAKER_MARKET_DEPTH_VWAP",
                }
            )
    return {
        "mode": "paper",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 1,
        "strategy_profile": STRATEGY_PROFILE,
        "benchmark_venue": "Binance BTCUSDT perpetual/spot Alpha",
        "execution_venue": "Binance USD-M BTCUSDT observed L2",
        "vwap_session": "causal daily/impulse/swing VWAP zones",
        "validation_status": audit["validation_status"],
        "policy_action": assessment["decision"],
        "live_orders_enabled": False,
        "initial_equity": str(account["initial_equity"]),
        "final_equity": str(account["final_equity"]),
        "gross_pnl": str(account.get("gross_pnl", 0)),
        "net_pnl": str(account.get("net_pnl", 0)),
        "fees": str(account.get("fees", 0)),
        "slippage": str(account.get("realized_slippage", 0)),
        "funding": str(account.get("funding", 0)),
        "max_drawdown": str(account.get("max_drawdown", 0)),
        "risk_per_trade": str(account["risk_per_trade"]),
        "max_leverage": str(account["max_leverage"]),
        "signals": account.get("complete_candidate_count", 0),
        "rejected_signals": max(
            0,
            int(account.get("complete_candidate_count", 0))
            - int(account.get("trade_signal_count", 0)),
        ),
        "kill_switches": int(account.get("risk_block_reason") is not None),
        "fills": fills,
        "equity_curve": account.get("equity_curve", []),
        "telemetry": [
            {
                **point,
                "activity": assessment["decision"],
                "decision_reason": assessment["reason"],
                "entry_score": assessment["expected_net_ev_bps"],
                "win_probability": assessment["target_probability"],
                "spread_bps": assessment["spread_bps"],
                "regime": assessment["probability_status"],
            }
            for point in audit["market_chart"]
        ],
        "shadow_closed_trades": len(trades),
        "shadow_position": account.get("open_position"),
        "pending_order": account.get("pending_order"),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def refresh(config_path: Path = CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    fees = fee_schedule(config)
    funding = funding_snapshot()
    context = market_context()
    evaluated_at = pd.Timestamp.now(tz="UTC")
    book = latest_book(evaluated_at, config)
    cost = 2 * fees.taker_bps + config.non_fee_reserve_bps
    recent_l2 = binance_l2_dataset.load_recent_records(max_lines=7_200)
    model_evaluation = musca_btc_auto_moe.evaluate_live_actions(context, recent_l2, evaluated_at)
    candidate = cast(dict[str, Any] | None, model_evaluation["candidate"])
    account = _load_account(fees)
    assessment = build_assessment(
        candidate=candidate,
        context=context,
        book=book,
        funding=funding,
        fees=fees,
        config=config,
        equity=float(account.get("final_equity", 10_000)),
        evaluated_at=evaluated_at,
        model_evaluation=model_evaluation,
    )
    advance_account(account, assessment, book)
    _atomic_json(
        STATE,
        {
            "schema_version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "account": account,
        },
    )
    alpha = _historical_alpha(config, fees)
    status = json.loads(
        Path("data/reports/binance_l2_collector.status.json").read_text(encoding="utf-8")
    )
    audit = {
        "protocol": musca_btc_auto_moe.PROTOCOL,
        "protocol_hash": musca_btc_auto_moe.PROTOCOL_HASH,
        "validation_status": "RESEARCH_PAPER_NO_REAL_CAPITAL",
        "real_capital_allowed": False,
        "selected_fee_profile": PROFILE,
        "alpha": alpha,
        "cost_interpretation": {
            "fee_venue": "Binance USD-M Futures",
            "maker_bps_per_side": fees.maker_bps,
            "taker_bps_per_side": fees.taker_bps,
            "non_fee_reserve_bps_round_trip": config.non_fee_reserve_bps,
            "modeled_round_trip_cost_bps": cost,
            "fee_source": fees.source,
            "stress_costs_are_not_leverage": True,
        },
        "selector": {"minimum_holdout_days": 10, "minimum_holdout_trades": 100},
        "data_coverage": {
            "binance_l2_rows": int(status.get("records", 0)),
            "binance_l2_utc_days": len(
                list(binance_l2_dataset.ROOT.glob("btcusdt_????-??-??.jsonl"))
            ),
            "execution_venue": "BINANCE",
        },
        "live_readiness": {
            "ready": bool(model_evaluation.get("feature_coverage", {}).get("complete")),
            "official_closed_minutes_loaded": int(
                context.attrs.get("official_minute_rows", len(context))
            ),
            "model_ready_rows": len(context),
            "decision_cadence_minutes": 1,
            "chart_rows": min(len(context), 360),
        },
        "current_market_assessment": assessment,
        "current_market_assessments": {PROFILE: assessment},
        "one_position_diagnostics": {
            "paper_accounts": {PROFILE: account},
            "paper_account": account,
        },
        "market_chart": _chart(context, assessment),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(REPORT, audit)
    _atomic_json(SHADOW_REPORT, _shadow(audit))
    return audit


async def worker(*, once: bool, config_path: Path = CONFIG) -> None:
    while True:
        try:
            refresh(config_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            _atomic_json(
                SHADOW_REPORT,
                {
                    "mode": "paper",
                    "strategy_profile": STRATEGY_PROFILE,
                    "validation_status": "RESEARCH_ONLY_FAIL_CLOSED",
                    "policy_action": "DATA_UNAVAILABLE",
                    "live_orders_enabled": False,
                    "detail": str(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        if once:
            return
        await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca BTC Auto-MoE Binance paper worker")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    asyncio.run(worker(once=args.once, config_path=args.config))


if __name__ == "__main__":
    main()
