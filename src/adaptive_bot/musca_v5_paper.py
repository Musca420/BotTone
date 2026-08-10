from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import pandas as pd

from adaptive_bot.bitunix_fees import futures_fee_bps
from adaptive_bot.musca_v5_execution import (
    TEN_THOUSAND,
    quote_taker_round_trip,
    size_for_technical_stop,
    walk_book,
)

STATE = Path("data/research/musca_v5_live_paper_state.json")
INITIAL_EQUITY = Decimal("10000")
MAX_HOLD_MINUTES = 60
SLIPPAGE_RESERVE_BPS_PER_SIDE = Decimal("0.5")
LOT_SIZE = Decimal("0.001")
MINIMUM_NOTIONAL = Decimal("5")
MAX_HISTORY = 2_000


def _number(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default
    return number if number.is_finite() else default


def _timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        result = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _new_account(
    profile: str,
    started_at: str,
    *,
    execution_venue: str = "BITUNIX",
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
) -> dict[str, Any]:
    if maker_fee_bps is None or taker_fee_bps is None:
        level = int(profile.removeprefix("VIP"))
        maker_fee_bps, taker_fee_bps = futures_fee_bps(level)
    venue = execution_venue.upper()
    initial = float(INITIAL_EQUITY)
    return {
        "status": f"{venue}_PAPER_SIMULATION",
        "execution_venue": venue,
        "initial_equity": initial,
        "paper_start": started_at,
        "risk_per_trade": 0.01,
        "max_daily_loss": 0.02,
        "max_strategy_drawdown": 0.08,
        "margin_fraction": 0.10,
        "max_leverage": 10,
        "fee_profile": profile,
        "maker_fees_per_side_bps": maker_fee_bps,
        "fees_per_side_bps": taker_fee_bps,
        "realized_balance": initial,
        "final_equity": initial,
        "available_balance": initial,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "estimated_net_if_closed": 0.0,
        "fees": 0.0,
        "funding": 0.0,
        "realized_slippage": 0.0,
        "modeled_costs": 0.0,
        "peak_equity": initial,
        "max_drawdown": 0.0,
        "pending_order": None,
        "open_position": None,
        "orders": [],
        "trades": [],
        "equity_curve": [{"timestamp": started_at, "equity": initial}],
        "last_book_at": None,
        "trade_signal_armed": True,
        "decision_tracking_start": started_at,
        "assessment_count": 0,
        "complete_candidate_count": 0,
        "trade_signal_count": 0,
        "wait_count": 0,
        "flat_count": 0,
        "decision_reason_counts": {},
        "assessment_log": [],
        "last_assessment_id": None,
        "last_decision": None,
        "last_candidate_complete": False,
        "last_assessment": None,
        "risk_block_reason": None,
        "current_day": None,
        "day_start_equity": initial,
        "last_event": "PAPER_ACCOUNT_CREATED",
    }


def _account_venue(account: dict[str, Any]) -> str:
    return str(account.get("execution_venue") or "BITUNIX").upper()


def _account_fee_bps(account: dict[str, Any]) -> tuple[Decimal, Decimal]:
    maker = _number(account.get("maker_fees_per_side_bps"), Decimal("-1"))
    taker = _number(account.get("fees_per_side_bps"), Decimal("-1"))
    if maker >= 0 and taker >= 0:
        return maker, taker
    level = int(str(account["fee_profile"]).removeprefix("VIP"))
    legacy_maker, legacy_taker = futures_fee_bps(level)
    return Decimal(str(legacy_maker)), Decimal(str(legacy_taker))


def new_paper_account(
    profile: str,
    started_at: str,
    *,
    execution_venue: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> dict[str, Any]:
    return _new_account(
        profile,
        started_at,
        execution_venue=execution_venue,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
    )


def _new_state() -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "created_at": started_at,
        "updated_at": started_at,
        "accounts": {f"VIP{level}": _new_account(f"VIP{level}", started_at) for level in range(6)},
    }


def load_accounts(path: Path = STATE) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return cast(dict[str, dict[str, Any]], _new_state()["accounts"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Musca V5 paper state is unreadable: {error}") from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("accounts"), dict):
        raise ValueError("Musca V5 paper state has an unsupported schema")
    return cast(dict[str, dict[str, Any]], payload["accounts"])


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{id(payload)}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _latest_book(execution_l2: pd.DataFrame) -> dict[str, Any] | None:
    if execution_l2.empty or "available_at" not in execution_l2:
        return None
    row = execution_l2.sort_values("available_at").iloc[-1]
    observed_at = _timestamp(row.get("available_at"))
    mid = _number(row.get("mid"))
    if observed_at is None or mid <= 0:
        return None
    return {
        "available_at": observed_at,
        "mid": mid,
        "bids": row.get("bids"),
        "asks": row.get("asks"),
        "best_bid": _number(row.get("best_bid")),
        "best_ask": _number(row.get("best_ask")),
        "valid": bool(row.get("feature_valid", False)),
    }


def _fill(book: dict[str, Any], side: str, quantity: Decimal) -> Any:
    levels = book["asks"] if side == "BUY" else book["bids"]
    return walk_book(levels, quantity, reference_mid=book["mid"])


def _signal_id(profile: str, assessment: dict[str, Any]) -> str:
    signal_at = assessment.get("alpha_signal_at") or assessment.get("observed_at")
    source = "|".join(
        [str(signal_at)]
        + [str(assessment.get(key)) for key in ("setup", "direction", "target_bps", "stop_bps")]
    )
    return hashlib.sha256(f"{profile}|{source}".encode()).hexdigest()[:24]


def _append_order(account: dict[str, Any], order: dict[str, Any]) -> None:
    account["orders"] = [*account.get("orders", []), order][-MAX_HISTORY:]


def _record_assessment(account: dict[str, Any], assessment: dict[str, Any]) -> None:
    observed_at = assessment.get("observed_at")
    if observed_at is None:
        return
    decision = str(assessment.get("decision") or "WAIT")
    reason = str(assessment.get("reason") or "NO_REASON")
    candidate_complete = bool(assessment.get("candidate_complete", False))
    assessment_id = hashlib.sha256(
        "|".join(
            str(value)
            for value in (
                observed_at,
                decision,
                reason,
                assessment.get("setup"),
                assessment.get("direction"),
            )
        ).encode()
    ).hexdigest()[:24]
    if account.get("last_assessment_id") == assessment_id:
        return
    tracking_start = account.setdefault(
        "decision_tracking_start",
        assessment.get("evaluated_at") or datetime.now(UTC).isoformat(),
    )
    account.setdefault("assessment_count", 0)
    account.setdefault("complete_candidate_count", 0)
    account.setdefault("trade_signal_count", 0)
    account.setdefault("wait_count", 0)
    account.setdefault("flat_count", 0)
    account.setdefault("decision_reason_counts", {})
    account.setdefault("assessment_log", [])
    previous_decision = account.get("last_decision")
    previous_complete = bool(account.get("last_candidate_complete", False))
    account["assessment_count"] += 1
    if candidate_complete and not previous_complete:
        account["complete_candidate_count"] += 1
    if decision == "TRADE" and previous_decision != "TRADE":
        account["trade_signal_count"] += 1
    if decision == "WAIT":
        account["wait_count"] += 1
    elif decision == "FLAT":
        account["flat_count"] += 1
    reasons = account["decision_reason_counts"]
    reasons[reason] = int(reasons.get(reason, 0)) + 1
    record = {
        "assessment_id": assessment_id,
        "evaluated_at": assessment.get("evaluated_at"),
        "observed_at": observed_at,
        "decision": decision,
        "reason": reason,
        "setup": assessment.get("setup"),
        "direction": assessment.get("direction"),
        "candidate_complete": candidate_complete,
        "expected_net_ev_bps": assessment.get("expected_net_ev_bps"),
        "risk_status": assessment.get("risk_status"),
        "execution_status": assessment.get("execution_status"),
        "tracking_start": tracking_start,
    }
    account["assessment_log"] = [*account["assessment_log"], record][-MAX_HISTORY:]
    account["last_assessment"] = record
    account["last_assessment_id"] = assessment_id
    account["last_decision"] = decision
    account["last_candidate_complete"] = candidate_complete


def _enqueue(
    account: dict[str, Any], profile: str, assessment: dict[str, Any], book_at: str
) -> None:
    order_id = f"musca-v5-{profile.lower()}-{_signal_id(profile, assessment)}"
    if any(order.get("client_order_id") == order_id for order in account.get("orders", [])):
        return
    venue_inputs = assessment.get("market_inputs", {}).get(_account_venue(account).lower(), {})
    pending = {
        "client_order_id": order_id,
        "status": "ACKNOWLEDGED",
        "order_type": "TAKER_MARKET",
        "reduce_only": False,
        "signal_at": assessment.get("observed_at"),
        "created_at": assessment.get("evaluated_at"),
        "eligible_after": book_at,
        "setup": assessment.get("setup"),
        "side": assessment.get("direction"),
        "quantity_btc": assessment.get("quantity_btc"),
        "target_bps": assessment.get("target_bps"),
        "stop_bps": assessment.get("stop_bps"),
        "expected_cost_bps": assessment.get("expected_cost_bps"),
        "expected_funding_bps": assessment.get("expected_funding_bps"),
        "risk_budget": assessment.get("risk_budget"),
        "target_probability": assessment.get("target_probability"),
        "stop_probability": assessment.get("stop_probability"),
        "timeout_probability": assessment.get("timeout_probability"),
        "expected_net_ev_bps": assessment.get("expected_net_ev_bps"),
        "expected_time_to_target_minutes": assessment.get("expected_time_to_target_minutes"),
        "policy_source": assessment.get("policy_source"),
        "expert_id": assessment.get("expert_id"),
        "alpha_signal_at": assessment.get("alpha_signal_at"),
        "management_style": assessment.get("management_style", "FIXED_TARGET_STOP"),
        "partial_target_fraction": assessment.get("partial_target_fraction", 0.0),
        "maximum_hold_minutes": assessment.get("maximum_hold_minutes", MAX_HOLD_MINUTES),
        "funding_rate": venue_inputs.get("funding_rate"),
        "next_funding_at": venue_inputs.get("next_funding_timestamp"),
    }
    account["pending_order"] = pending
    account["trade_signal_armed"] = False
    account["last_event"] = (
        f"ORDER_ACKNOWLEDGED {order_id}; waits for the next {_account_venue(account)} book"
    )
    _append_order(account, dict(pending))


def _update_order(account: dict[str, Any], order_id: str, **updates: object) -> None:
    for order in reversed(account.get("orders", [])):
        if order.get("client_order_id") == order_id:
            order.update(updates)
            return


def _open_pending(account: dict[str, Any], book: dict[str, Any]) -> bool:
    pending = account.get("pending_order")
    if not isinstance(pending, dict):
        return False
    eligible_after = _timestamp(pending.get("eligible_after"))
    if eligible_after is None or book["available_at"] <= eligible_after or not book["valid"]:
        return False
    if book["available_at"] > eligible_after + pd.Timedelta(seconds=30):
        _update_order(
            account,
            str(pending["client_order_id"]),
            status="EXPIRED",
            reason="MARKET_ORDER_NEXT_EVENT_TIMEOUT",
            updated_at=book["available_at"].isoformat(),
        )
        account["pending_order"] = None
        account["last_event"] = "ENTRY_EXPIRED_NEXT_EVENT_TIMEOUT"
        return False
    side = str(pending.get("side"))
    stop_bps = _number(pending.get("stop_bps"))
    level = 0
    _, taker_fee_bps = _account_fee_bps(account)
    tentative_quantity = _number(pending.get("quantity_btc"))
    expected_funding_bps = max(Decimal("0"), _number(pending.get("expected_funding_bps")))
    quote = quote_taker_round_trip(
        side=side,
        quantity=tentative_quantity,
        bids=book["bids"],
        asks=book["asks"],
        best_bid=book["best_bid"],
        best_ask=book["best_ask"],
        vip_level=level,
        taker_fee_bps=taker_fee_bps,
        expected_funding_bps=expected_funding_bps,
    )
    if quote is None:
        _update_order(
            account,
            str(pending["client_order_id"]),
            status="REJECTED",
            reason=f"INSUFFICIENT_OBSERVED_{_account_venue(account)}_DEPTH",
            updated_at=book["available_at"].isoformat(),
        )
        account["pending_order"] = None
        account["last_event"] = "ENTRY_REJECTED_INSUFFICIENT_OBSERVED_DEPTH"
        return False
    risk = None
    expected_cost_bps = Decimal("0")
    for _ in range(2):
        expected_cost_bps = quote.expected_total_cost_bps + 2 * SLIPPAGE_RESERVE_BPS_PER_SIDE
        risk = size_for_technical_stop(
            equity=_number(account.get("final_equity")),
            entry_price=quote.entry.execution_vwap,
            side=side,
            technical_stop_bps=stop_bps,
            expected_cost_bps=expected_cost_bps,
            lot_size=LOT_SIZE,
            minimum_quantity=LOT_SIZE,
            minimum_notional=MINIMUM_NOTIONAL,
        )
        if not risk.approved:
            break
        sized = quote_taker_round_trip(
            side=side,
            quantity=risk.quantity,
            bids=book["bids"],
            asks=book["asks"],
            best_bid=book["best_bid"],
            best_ask=book["best_ask"],
            vip_level=level,
            taker_fee_bps=taker_fee_bps,
            expected_funding_bps=expected_funding_bps,
        )
        if sized is None:
            quote = None
            break
        quote = sized
    if risk is None or not risk.approved or quote is None:
        reason = (
            risk.reason
            if risk is not None and not risk.approved
            else f"INSUFFICIENT_OBSERVED_{_account_venue(account)}_DEPTH"
        )
        _update_order(
            account,
            str(pending["client_order_id"]),
            status="REJECTED",
            reason=reason,
            updated_at=book["available_at"].isoformat(),
        )
        account["pending_order"] = None
        account["last_event"] = f"ENTRY_REJECTED_{reason}"
        return False
    actual_fill = quote.entry
    entry_fee = actual_fill.notional * taker_fee_bps / TEN_THOUSAND
    balance = _number(account["realized_balance"]) - entry_fee
    direction = Decimal("1") if side == "LONG" else Decimal("-1")
    target_bps = _number(pending.get("target_bps"))
    entry = actual_fill.execution_vwap
    current_stop = risk.technical_stop_price
    target = entry * (Decimal("1") + direction * target_bps / TEN_THOUSAND)
    break_even = entry * (Decimal("1") + direction * expected_cost_bps / TEN_THOUSAND)
    now = book["available_at"].isoformat()
    slippage = actual_fill.notional * actual_fill.slippage_bps / TEN_THOUSAND
    account["realized_balance"] = float(balance)
    account["fees"] = float(_number(account.get("fees")) + entry_fee)
    account["realized_slippage"] = float(_number(account.get("realized_slippage")) + slippage)
    account["open_position"] = {
        "status": "OPEN",
        "side": side,
        "setup": pending.get("setup"),
        "signal_at": pending.get("signal_at"),
        "entry_at": now,
        "entry_execution_price": float(entry),
        "entry_reference_mid": float(book["mid"]),
        "quantity_btc": float(risk.quantity),
        "notional": float(actual_fill.notional),
        "initial_notional": float(actual_fill.notional),
        "margin_used": float(risk.margin_required),
        "leverage": float(risk.leverage),
        "entry_fee": float(entry_fee),
        "initial_quantity_btc": float(risk.quantity),
        "realized_trading_pnl": 0.0,
        "realized_exit_fees": 0.0,
        "realized_gross_pnl": 0.0,
        "realized_slippage_cost": 0.0,
        "exit_quantity_btc": 0.0,
        "exit_notional_sum": 0.0,
        "exit_fills": [],
        "entry_slippage": float(slippage),
        "target_bps": float(target_bps),
        "stop_bps": float(stop_bps),
        "expected_cost_bps": float(expected_cost_bps),
        "target_price": float(target),
        "initial_stop_price": float(current_stop),
        "current_stop_price": float(current_stop),
        "catastrophic_stop_price": float(risk.catastrophic_stop_price),
        "break_even_price": float(break_even),
        "target_probability": pending.get("target_probability"),
        "stop_probability": pending.get("stop_probability"),
        "timeout_probability": pending.get("timeout_probability"),
        "expected_net_ev_bps": pending.get("expected_net_ev_bps"),
        "expected_time_to_target_minutes": pending.get("expected_time_to_target_minutes"),
        "policy_source": pending.get("policy_source"),
        "expert_id": pending.get("expert_id"),
        "alpha_signal_at": pending.get("alpha_signal_at"),
        "management_style": pending.get("management_style", "FIXED_TARGET_STOP"),
        "partial_target_fraction": pending.get("partial_target_fraction", 0.0),
        "maximum_hold_minutes": pending.get("maximum_hold_minutes", MAX_HOLD_MINUTES),
        "tp1_hit": False,
        "trail_marks": [],
        "mfe_bps": 0.0,
        "mae_bps": 0.0,
        "trailing_status": "FIXED_ALPHA_PLAN",
        "last_funding_rate": pending.get("funding_rate"),
        "next_funding_at": pending.get("next_funding_at"),
        "funding_pnl": 0.0,
    }
    account["pending_order"] = None
    account["last_event"] = f"POSITION_OPENED_{side}"
    _update_order(
        account,
        str(pending["client_order_id"]),
        status="FILLED",
        filled_at=now,
        fill_price=float(entry),
        quantity_btc=float(risk.quantity),
        fee=float(entry_fee),
        levels_consumed=actual_fill.levels_consumed,
    )
    return True


def _apply_funding(
    account: dict[str, Any], position: dict[str, Any], assessment: dict[str, Any], now: pd.Timestamp
) -> None:
    venue_inputs = assessment.get("market_inputs", {}).get(_account_venue(account).lower(), {})
    current_next = _timestamp(venue_inputs.get("next_funding_timestamp"))
    current_rate_value = venue_inputs.get("funding_rate")
    current_rate = _number(current_rate_value) if current_rate_value is not None else None
    previous_next = _timestamp(position.get("next_funding_at"))
    previous_rate = _number(position.get("last_funding_rate"))
    if (
        previous_next is not None
        and now >= previous_next
        and (current_next is None or current_next > previous_next)
    ):
        direction = Decimal("1") if position["side"] == "LONG" else Decimal("-1")
        cashflow = -direction * _number(position["notional"]) * previous_rate
        position["funding_pnl"] = float(_number(position.get("funding_pnl")) + cashflow)
        account["funding"] = float(_number(account.get("funding")) - cashflow)
        account["realized_balance"] = float(_number(account["realized_balance"]) + cashflow)
    if current_next is not None:
        position["next_funding_at"] = current_next.isoformat()
    if current_rate is not None:
        position["last_funding_rate"] = float(current_rate)


def _execute_exit(
    account: dict[str, Any],
    book: dict[str, Any],
    position: dict[str, Any],
    reason: str,
    quantity: Decimal,
) -> bool:
    side = str(position["side"])
    exit_side = "SELL" if side == "LONG" else "BUY"
    fill = _fill(book, exit_side, quantity)
    if fill is None:
        account["last_event"] = f"EXIT_PENDING_{reason}_INSUFFICIENT_OBSERVED_DEPTH"
        return False
    _, taker_bps = _account_fee_bps(account)
    exit_fee = fill.notional * taker_bps / TEN_THOUSAND
    entry = _number(position["entry_execution_price"])
    entry_reference = _number(position["entry_reference_mid"])
    direction = Decimal("1") if side == "LONG" else Decimal("-1")
    trading_pnl = direction * quantity * (fill.execution_vwap - entry)
    gross_pnl = direction * quantity * (book["mid"] - entry_reference)
    realized_slippage_cost = gross_pnl - trading_pnl
    balance = _number(account["realized_balance"]) + trading_pnl - exit_fee
    exit_slippage = fill.notional * fill.slippage_bps / TEN_THOUSAND
    account["realized_balance"] = float(balance)
    account["fees"] = float(_number(account.get("fees")) + exit_fee)
    account["realized_slippage"] = float(_number(account.get("realized_slippage")) + exit_slippage)
    account["gross_pnl"] = float(_number(account.get("gross_pnl")) + gross_pnl)
    now = book["available_at"].isoformat()
    order_id = f"{position['signal_at']}-{account['fee_profile']}-exit-{len(account['orders'])}"
    _append_order(
        account,
        {
            "client_order_id": hashlib.sha256(order_id.encode()).hexdigest()[:24],
            "status": "FILLED",
            "order_type": "TAKER_MARKET",
            "reduce_only": True,
            "side": exit_side,
            "created_at": now,
            "filled_at": now,
            "fill_price": float(fill.execution_vwap),
            "quantity_btc": float(quantity),
            "fee": float(exit_fee),
            "levels_consumed": fill.levels_consumed,
            "reason": reason,
        },
    )
    position["realized_trading_pnl"] = float(
        _number(position.get("realized_trading_pnl")) + trading_pnl
    )
    position["realized_exit_fees"] = float(_number(position.get("realized_exit_fees")) + exit_fee)
    position["realized_gross_pnl"] = float(_number(position.get("realized_gross_pnl")) + gross_pnl)
    position["realized_slippage_cost"] = float(
        _number(position.get("realized_slippage_cost")) + realized_slippage_cost
    )
    position["exit_quantity_btc"] = float(_number(position.get("exit_quantity_btc")) + quantity)
    position["exit_notional_sum"] = float(
        _number(position.get("exit_notional_sum")) + quantity * fill.execution_vwap
    )
    position["exit_fills"] = [
        *position.get("exit_fills", []),
        {
            "timestamp": now,
            "reason": reason,
            "quantity_btc": float(quantity),
            "price": float(fill.execution_vwap),
            "fee": float(exit_fee),
        },
    ]
    previous_quantity = _number(position["quantity_btc"])
    remaining = max(Decimal("0"), previous_quantity - quantity)
    ratio = remaining / previous_quantity if previous_quantity > 0 else Decimal("0")
    position["quantity_btc"] = float(remaining)
    position["notional"] = float(_number(position["notional"]) * ratio)
    position["margin_used"] = float(_number(position["margin_used"]) * ratio)
    account["last_event"] = f"POSITION_REDUCED_{reason}"
    return True


def _finalize_position(
    account: dict[str, Any], position: dict[str, Any], reason: str, exited_at: str
) -> None:
    entry_fee = _number(position["entry_fee"])
    exit_fees = _number(position.get("realized_exit_fees"))
    trading_pnl = _number(position.get("realized_trading_pnl"))
    gross_pnl = _number(position.get("realized_gross_pnl"))
    realized_slippage_cost = _number(position.get("realized_slippage_cost"))
    funding_pnl = _number(position.get("funding_pnl"))
    net_pnl = trading_pnl - entry_fee - exit_fees + funding_pnl
    stress_pnl = (
        net_pnl
        - entry_fee
        - exit_fees
        - max(Decimal("0"), realized_slippage_cost)
        - max(Decimal("0"), -funding_pnl)
    )
    total_cost = entry_fee + exit_fees + realized_slippage_cost - funding_pnl
    exit_quantity = _number(position.get("exit_quantity_btc"))
    average_exit = (
        _number(position.get("exit_notional_sum")) / exit_quantity
        if exit_quantity > 0
        else Decimal("0")
    )
    balance = _number(account["realized_balance"])
    side = str(position["side"])
    entry = _number(position["entry_execution_price"])
    trade = {
        "signal_at": position["signal_at"],
        "entry_at": position["entry_at"],
        "exit_at": exited_at,
        "expert": str(position["setup"]).lower() + "_dynamic",
        "side": side,
        "entry_execution_price": float(entry),
        "exit_execution_price": float(average_exit),
        "stop_bps": position["stop_bps"],
        "target_bps": position["target_bps"],
        "alpha_target_bps": position["target_bps"],
        "alpha_target_probability": position.get("target_probability"),
        "alpha_expected_time_to_target_minutes": position.get("expected_time_to_target_minutes"),
        "notional": position["initial_notional"],
        "quantity_btc": float(_number(position["initial_quantity_btc"])),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "stress_pnl": float(stress_pnl),
        "fees": float(entry_fee + exit_fees),
        "funding": float(-funding_pnl),
        "slippage_reserve": 0.0,
        "realized_slippage": float(realized_slippage_cost),
        "modeled_costs": float(total_cost),
        "mfe_bps": position["mfe_bps"],
        "mae_bps": position["mae_bps"],
        "balance": float(balance),
        "exit_reason": reason,
        "fee_profile": account["fee_profile"],
        "policy_source": position.get("policy_source"),
        "expert_id": position.get("expert_id"),
        "management_style": position.get("management_style"),
        "tp1_hit": bool(position.get("tp1_hit", False)),
        "exit_fills": position.get("exit_fills", []),
    }
    account["trades"] = [*account.get("trades", []), trade][-MAX_HISTORY:]
    account["equity_curve"] = [
        *account.get("equity_curve", []),
        {"timestamp": exited_at, "equity": float(balance)},
    ][-MAX_HISTORY:]
    account["open_position"] = None
    account["last_event"] = f"POSITION_CLOSED_{reason}"


def _close_position(
    account: dict[str, Any], book: dict[str, Any], position: dict[str, Any], reason: str
) -> bool:
    quantity = _number(position["quantity_btc"])
    if not _execute_exit(account, book, position, reason, quantity):
        return False
    _finalize_position(account, position, reason, book["available_at"].isoformat())
    return True


def _take_partial_target(
    account: dict[str, Any], book: dict[str, Any], position: dict[str, Any]
) -> bool:
    initial = _number(position["initial_quantity_btc"])
    fraction = _number(position.get("partial_target_fraction"), Decimal("0.5"))
    quantity = (initial * fraction // LOT_SIZE) * LOT_SIZE
    remaining = _number(position["quantity_btc"]) - quantity
    if quantity < LOT_SIZE or remaining < LOT_SIZE:
        return _close_position(account, book, position, "DYNAMIC_TARGET")
    if not _execute_exit(account, book, position, "PARTIAL_TARGET_1", quantity):
        return False
    side = str(position["side"])
    protected = _number(position["break_even_price"])
    current = _number(position["current_stop_price"])
    position["current_stop_price"] = float(
        max(current, protected) if side == "LONG" else min(current, protected)
    )
    position["tp1_hit"] = True
    position["trailing_status"] = "COST_PROTECTED_TRAILING_15M"
    account["last_event"] = "PARTIAL_TARGET_FILLED_STOP_COST_PROTECTED"
    return True


def _manage_open(account: dict[str, Any], assessment: dict[str, Any], book: dict[str, Any]) -> None:
    position = account.get("open_position")
    if not isinstance(position, dict) or not book["valid"]:
        return
    now = book["available_at"]
    _apply_funding(account, position, assessment, now)
    entry = _number(position["entry_execution_price"])
    # The public mark feed is sampled once per minute, while the observed book is
    # event-time current. Use the current executable market for TP/SL triggering;
    # the mark remains a separate liquidation diagnostic and is never forward-filled.
    mark = book["mid"]
    direction = Decimal("1") if position["side"] == "LONG" else Decimal("-1")
    excursion = direction * (mark / entry - Decimal("1")) * TEN_THOUSAND
    position["mfe_bps"] = max(float(position.get("mfe_bps", 0.0)), float(excursion))
    position["mae_bps"] = min(float(position.get("mae_bps", 0.0)), float(excursion))
    marks = [
        mark_row
        for mark_row in position.get("trail_marks", [])
        if (_timestamp(mark_row.get("timestamp")) or now) >= now - pd.Timedelta(minutes=15)
    ]
    marks.append({"timestamp": now.isoformat(), "mid": float(mark)})
    position["trail_marks"] = marks[-1_000:]
    catastrophic = _number(position["catastrophic_stop_price"])
    stop = _number(position["current_stop_price"])
    target = _number(position["target_price"])
    catastrophic_hit = (position["side"] == "LONG" and mark <= catastrophic) or (
        position["side"] == "SHORT" and mark >= catastrophic
    )
    stop_hit = (position["side"] == "LONG" and mark <= stop) or (
        position["side"] == "SHORT" and mark >= stop
    )
    target_hit = not bool(position.get("tp1_hit", False)) and (
        (position["side"] == "LONG" and mark >= target)
        or (position["side"] == "SHORT" and mark <= target)
    )
    opened = _timestamp(position.get("entry_at"))
    maximum_hold = int(position.get("maximum_hold_minutes", MAX_HOLD_MINUTES))
    timed_out = opened is not None and now >= opened + pd.Timedelta(minutes=maximum_hold)
    reason = (
        "CATASTROPHIC_STOP"
        if catastrophic_hit
        else "DYNAMIC_STOP"
        if stop_hit
        else str(account["risk_block_reason"])
        if account.get("risk_block_reason")
        else "TIME"
        if timed_out
        else None
    )
    if reason is not None:
        _close_position(account, book, position, reason)
        return
    if target_hit:
        if position.get("management_style") == "HALF_AT_1_5R_COST_PROTECTED_TRAIL_15M":
            _take_partial_target(account, book, position)
        else:
            _close_position(account, book, position, "DYNAMIC_TARGET")
        return
    if (
        bool(position.get("tp1_hit", False))
        and opened is not None
        and now >= opened + pd.Timedelta(minutes=15)
    ):
        observed = [_number(row.get("mid")) for row in marks]
        proposal = min(observed) if position["side"] == "LONG" else max(observed)
        current_stop = _number(position["current_stop_price"])
        tightened = (
            max(current_stop, proposal)
            if position["side"] == "LONG"
            else min(current_stop, proposal)
        )
        position["current_stop_price"] = float(tightened)


def _mark_account(account: dict[str, Any], book: dict[str, Any] | None) -> None:
    balance = _number(account["realized_balance"])
    unrealized = Decimal("0")
    estimated_exit_fee = Decimal("0")
    margin = Decimal("0")
    position = account.get("open_position")
    if isinstance(position, dict) and book is not None:
        direction = Decimal("1") if position["side"] == "LONG" else Decimal("-1")
        quantity = _number(position["quantity_btc"])
        exit_side = "SELL" if position["side"] == "LONG" else "BUY"
        fill = _fill(book, exit_side, quantity)
        mark = fill.execution_vwap if fill is not None else book["mid"]
        unrealized = direction * quantity * (mark - _number(position["entry_execution_price"]))
        _, taker_bps = _account_fee_bps(account)
        estimated_exit_fee = quantity * mark * taker_bps / TEN_THOUSAND
        margin = _number(position["margin_used"])
        position["mark_price"] = float(book["mid"])
        position["estimated_exit_price"] = float(mark)
        position["unrealized_pnl"] = float(unrealized)
        position["estimated_net_if_closed"] = float(unrealized - estimated_exit_fee)
    equity = balance + unrealized
    peak = max(_number(account.get("peak_equity")), equity)
    drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
    account["final_equity"] = float(equity)
    account["available_balance"] = float(max(Decimal("0"), equity - margin))
    account["unrealized_pnl"] = float(unrealized)
    account["estimated_net_if_closed"] = float(unrealized - estimated_exit_fee)
    account["net_pnl"] = float(equity - INITIAL_EQUITY)
    account["peak_equity"] = float(peak)
    account["max_drawdown"] = max(float(account.get("max_drawdown", 0.0)), float(drawdown))
    account["modeled_costs"] = float(
        _number(account.get("fees"))
        + _number(account.get("funding"))
        + _number(account.get("realized_slippage"))
    )


def _update_loss_limits(account: dict[str, Any], book: dict[str, Any]) -> None:
    day = book["available_at"].date().isoformat()
    if account.get("current_day") != day:
        account["current_day"] = day
        account["day_start_equity"] = account["final_equity"]
    equity = _number(account["final_equity"])
    day_start = _number(account["day_start_equity"])
    reason = None
    if day_start > 0 and equity <= day_start * Decimal("0.98"):
        reason = "DAILY_DRAWDOWN_LIMIT"
    if _number(account.get("max_drawdown")) >= Decimal("0.08"):
        reason = "MAX_STRATEGY_DRAWDOWN"
    account["risk_block_reason"] = reason


def advance_account(
    account: dict[str, Any], assessment: dict[str, Any], execution_l2: pd.DataFrame
) -> dict[str, Any]:
    """Advance one persistent paper account on one newly observed venue book."""
    _record_assessment(account, assessment)
    if assessment.get("decision") != "TRADE":
        account["trade_signal_armed"] = True
    book = _latest_book(execution_l2)
    if book is None:
        account["last_event"] = f"WAITING_FOR_{_account_venue(account)}_BOOK"
        _mark_account(account, None)
        return account
    last_book = _timestamp(account.get("last_book_at"))
    if last_book is not None and book["available_at"] <= last_book:
        _mark_account(account, book)
        return account
    account["last_book_at"] = book["available_at"].isoformat()
    opened = _open_pending(account, book)
    if not opened:
        _manage_open(account, assessment, book)
    _mark_account(account, book)
    _update_loss_limits(account, book)
    if (
        account.get("open_position") is None
        and account.get("pending_order") is None
        and assessment.get("decision") == "TRADE"
        and bool(account.get("trade_signal_armed", True))
        and book["valid"]
        and account.get("risk_block_reason") is None
    ):
        _enqueue(
            account,
            str(account["fee_profile"]),
            assessment,
            book["available_at"].isoformat(),
        )
    _mark_account(account, book)
    return account


def advance_accounts(
    assessments: dict[str, dict[str, Any]],
    execution_l2: pd.DataFrame,
    *,
    path: Path = STATE,
) -> dict[str, dict[str, Any]]:
    state = _new_state()
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Musca V5 paper state is unreadable: {error}") from error
    if state.get("schema_version") != 1 or not isinstance(state.get("accounts"), dict):
        raise ValueError("Musca V5 paper state has an unsupported schema")
    for level in range(6):
        profile = f"VIP{level}"
        account = state["accounts"].setdefault(profile, _new_account(profile, state["created_at"]))
        assessment = assessments.get(profile, {})
        advance_account(account, assessment, execution_l2)
    state["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_write(path, state)
    return cast(dict[str, dict[str, Any]], state["accounts"])
