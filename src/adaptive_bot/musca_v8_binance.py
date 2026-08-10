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

import pandas as pd
import yaml

from adaptive_bot import binance_l2_dataset, btc_vwap_alpha, musca_v8_multi_horizon
from adaptive_bot.musca_v5_execution import (
    quote_taker_round_trip,
    size_for_technical_stop,
)
from adaptive_bot.musca_v5_paper import advance_account, new_paper_account

CONFIG = Path("configs/binance_btcusdt_paper.yaml")
STATE = Path("data/research/musca_v8_binance_paper_state.json")
REPORT = Path("data/reports/musca_v8_binance_paper.json")
SHADOW_REPORT = Path("data/reports/musca_v8_binance_shadow.json")
PROFILE = "BINANCE"
STRATEGY_PROFILE = "musca_v5_stable_multi_horizon_vwap"
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
COMMISSION_URL = "https://fapi.binance.com/fapi/v1/commissionRate"
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
    temporary.replace(path)


def load_config(path: Path = CONFIG) -> BinancePaperConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("mode") != "paper":
        raise ValueError("Binance Musca V8 requires an explicit paper configuration")
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
            "User-Agent": "BotTone-MuscaV8/1",
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


def market_context() -> pd.DataFrame:
    global _CONTEXT_CACHE
    now = time.monotonic()
    if _CONTEXT_CACHE is not None and now - _CONTEXT_CACHE[0] < 50:
        return _CONTEXT_CACHE[1]
    minutes = binance_l2_dataset.load_recent_official_minutes()
    context = btc_vwap_alpha.canonical_minute_market_features(minutes)
    context = context.loc[context["feature_contract_valid"].fillna(False)].copy()
    if context.empty:
        raise ValueError("Binance minute Alpha has not completed its causal warm-up")
    _CONTEXT_CACHE = (now, context)
    return context


def latest_book(evaluated_at: pd.Timestamp, config: BinancePaperConfig) -> pd.DataFrame:
    rows = binance_l2_dataset.load_recent_records(max_lines=30)
    if rows.empty:
        return rows
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True, format="mixed")
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
) -> dict[str, Any]:
    latest = context.iloc[-1]
    book_row = book.iloc[-1] if not book.empty else pd.Series(dtype=object)
    alpha_source = _source(
        "Binance",
        "Alpha: closed BTCUSDT perpetual/spot candles and causal VWAP event",
        latest.get("available_at"),
        evaluated_at,
        maximum_age_seconds=90,
        structurally_valid=bool(latest.get("feature_contract_valid", False)),
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
    direction = candidate.get("direction") if candidate else None
    sign = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
    stop_bps: float | None = None
    quote = risk = None
    expected_funding_bps = 0.0
    expected_cost_bps = config.modeled_round_trip_cost_bps
    expected_net_ev_bps: float | None = None
    target_bps: float | None = None
    if candidate and not book.empty and sign:
        mid = float(book_row["mid"])
        stop_bps = sign * (mid - float(candidate["stop_price"])) / mid * 10_000
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
        target_bps = 1.5 * stop_bps if stop_bps and stop_bps > 0 else None
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
    decision = "TRADE" if approved else "FLAT" if candidate and sources_valid else "WAIT"
    reason = (
        "BASE_ALPHA_NET_POSITIVE"
        if approved
        else "WAIT_NEW_IMPULSE_PULLBACK_RESTART"
        if candidate is None
        else "BINANCE_DATA_FAIL_CLOSED"
        if not sources_valid
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
        "setup": "IMPULSE_PULLBACK_MULTI_HORIZON",
        "direction": None,
        "candidate": False,
        "setup_active": False,
        "passed_checks": 0,
        "total_checks": 7,
        "first_failed_check": "waiting_new_impulse_pullback_restart",
        "checks": [],
        "policy_source": "MUSCA_V8_FROZEN_BASE_MONITOR",
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
        "policy_source": setup.get("policy_source", "MUSCA_V8_FROZEN_BASE"),
        "expert_id": setup.get("expert_id"),
        "alpha_signal_at": setup.get("available_at"),
        "management_style": setup.get("management_style", "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M"),
        "partial_target_fraction": setup.get("partial_target_fraction", 0.5),
        "maximum_hold_minutes": int(setup.get("maximum_hold_minutes", 360)),
        "candidate_complete": candidate is not None,
        "model_context": "COMPLETE_CANDIDATE" if candidate else "WAITING_FROZEN_EVENT",
        "model_feature_coverage": {"complete": sources_valid, "missing": []},
        "probability_status": (
            "FROZEN_BASE_HISTORICAL_CALIBRATION" if candidate else "FROZEN_BASE_WAITING_EVENT"
        ),
        "target_probability": candidate.get("target_probability") if candidate else None,
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
        "anchored_vwap": candidate.get("operating_vwap") if candidate else None,
        "spread_bps": float(book_row["spread_bps"]) if not book.empty else None,
        "flow_vote": float(latest["taker_imbalance_60s"]),
        "binance_return_1m_bps": float(latest["return_1m_bps"]),
        "binance_return_5m_bps": float(latest["return_5m_bps"]),
        "vwap_distance_bps": float(latest["vwap_distance_bps"]),
        "stop_bps": stop_bps,
        "target_bps": target_bps,
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
        "anchor": {
            "state": "VALID" if candidate else "NONE",
            "valid": candidate is not None,
            "detected_at": candidate.get("impulse_anchor_at") if candidate else None,
            "age_seconds": None,
            "direction": direction,
            "price": candidate.get("operating_vwap") if candidate else None,
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
        "evaluation_frequency": (
            "frozen V8 event on completed Binance 5m bars; execution on next Binance book"
        ),
        "outcome_horizons_minutes": [360],
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
    report = json.loads(musca_v8_multi_horizon.REPORT.read_text(encoding="utf-8"))
    cost = 2 * fees.taker_bps + config.non_fee_reserve_bps
    matching = next(
        (
            audit
            for audit in report.get("paper_profiles", {}).values()
            if abs(float(audit.get("assumed_round_trip_cost_bps", -1)) - cost) < 1e-9
        ),
        None,
    )
    return {
        "status": report.get("verdict"),
        "protocol_hash": report.get("protocol_hash"),
        "paper_profiles": {PROFILE: matching} if matching else {},
        "cost_matched_without_retraining": matching is not None,
        "modeled_round_trip_cost_bps": cost,
        "holdout_opened": False,
        "real_capital_allowed": False,
    }


def _chart(context: pd.DataFrame, assessment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": pd.Timestamp(row["available_at"]).isoformat(),
            "close": float(row["close"]),
            "execution_price": float(row["close"]),
            "center": float(row["rolling_vwap"]),
            "anchored_vwap": assessment.get("anchored_vwap"),
            "break_even": assessment.get("break_even_price"),
            "stop": assessment.get("stop_price"),
            "target": assessment.get("target_price"),
        }
        for _, row in context.tail(360).iloc[::3].iterrows()
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
                    "source": "musca_v8_binance_observed_depth_paper",
                    "instrument": "BTCUSDT",
                    "client_order_id": f"musca-v8-binance-{2 * number + int(not entry)}",
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
        "timeframe_minutes": 5,
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
                "timestamp": assessment["observed_at"],
                "activity": assessment["decision"],
                "decision_reason": assessment["reason"],
                "close": assessment["price"],
                "center": assessment["rolling_vwap"],
                "anchored_vwap": assessment["anchored_vwap"],
                "entry_score": assessment["expected_net_ev_bps"],
                "win_probability": assessment["target_probability"],
                "spread_bps": assessment["spread_bps"],
                "target": assessment["target_price"],
                "stop": assessment["stop_price"],
                "break_even": assessment["break_even_price"],
                "regime": assessment["probability_status"],
            }
        ],
        "shadow_closed_trades": len(trades),
        "shadow_position": account.get("open_position"),
        "pending_order": account.get("pending_order"),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def refresh(config_path: Path = CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    evaluated_at = pd.Timestamp.now(tz="UTC")
    fees = fee_schedule(config)
    funding = funding_snapshot()
    context = market_context()
    book = latest_book(evaluated_at, config)
    cost = 2 * fees.taker_bps + config.non_fee_reserve_bps
    candidate = musca_v8_multi_horizon.live_candidate_for_cost(cost, evaluated_at)
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
        "protocol": musca_v8_multi_horizon.PROTOCOL,
        "protocol_hash": musca_v8_multi_horizon.PROTOCOL_HASH,
        "validation_status": "RESEARCH_ONLY_BINANCE_PAPER_REQUIRED",
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
    parser = argparse.ArgumentParser(description="Musca V8 Binance USD-M paper worker")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    asyncio.run(worker(once=args.once, config_path=args.config))


if __name__ == "__main__":
    main()
