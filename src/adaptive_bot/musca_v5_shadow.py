from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_bot import btc_cross_exchange_forward_audit

REPORT = Path("data/reports/musca_v5_shadow.json")
AUDIT = btc_cross_exchange_forward_audit.REPORT
STRATEGY_PROFILE = "musca_v5_stable_multi_horizon_vwap"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _fill(trade: dict[str, Any], *, entry: bool, sequence: int) -> dict[str, Any]:
    long = trade["side"] == "LONG"
    timestamp = trade["entry_at"] if entry else trade["exit_at"]
    price = trade["entry_execution_price"] if entry else trade["exit_execution_price"]
    return {
        "exchange_timestamp": timestamp,
        "received_timestamp": timestamp,
        "source": "musca_v5_bitunix_depth_replay",
        "instrument": "BTCUSDT",
        "client_order_id": f"musca-v5-{sequence}",
        "side": "buy" if long == entry else "sell",
        "price": str(price),
        "quantity": str(trade["quantity_btc"]),
        "commission": str(trade["fees"] / 2),
        "slippage": str(trade["slippage_reserve"] / 2),
        "liquidity_role": "TAKER_MARKET_DEPTH_VWAP",
    }


def build_shadow_report(audit: dict[str, Any]) -> dict[str, Any]:
    diagnostics = audit.get("one_position_diagnostics", {})
    paper = diagnostics.get("paper_accounts", {}).get(
        "VIP0", diagnostics.get("paper_account", {})
    )
    assessment = audit.get("current_market_assessment", {})
    trades = paper.get("trades", [])
    fills = [
        _fill(trade, entry=entry, sequence=2 * number + int(not entry))
        for number, trade in enumerate(trades)
        for entry in (True, False)
    ]
    fees = sum(float(trade.get("fees", 0.0)) for trade in trades)
    slippage = sum(float(trade.get("slippage_reserve", 0.0)) for trade in trades)
    funding = sum(float(trade.get("funding", 0.0)) for trade in trades)
    observed_at = assessment.get("observed_at", datetime.now(UTC).isoformat())
    complete_candidates = int(paper.get("complete_candidate_count", 0))
    trade_signals = int(paper.get("trade_signal_count", 0))
    telemetry = [
        {
            "timestamp": observed_at,
            "activity": assessment.get("decision", "WAIT"),
            "decision_reason": assessment.get("reason", "No synchronized setup"),
            "close": assessment.get("price"),
            "center": assessment.get("rolling_vwap"),
            "anchored_vwap": assessment.get("anchored_vwap"),
            "entry_score": assessment.get("expected_net_ev_bps"),
            "win_probability": assessment.get("target_probability"),
            "spread_bps": assessment.get("spread_bps"),
            "target": assessment.get("target_price"),
            "stop": assessment.get("stop_price"),
            "break_even": assessment.get("break_even_price"),
            "regime": assessment.get("probability_status", "COLLECTING"),
        }
    ]
    return {
        "mode": "shadow",
        "instrument": "BTCUSDT",
        "timeframe_minutes": 5,
        "strategy_profile": STRATEGY_PROFILE,
        "benchmark_venue": "Binance BTCUSDT perpetual/spot Alpha",
        "execution_venue": "Bitunix BTCUSDT observed L2",
        "vwap_session": (
            "active Alpha uses causal daily/impulse/swing VWAP zones; "
            "rolling and live anchor lines are chart diagnostics"
        ),
        "validation_status": audit.get("validation_status", "RESEARCH_ONLY"),
        "policy_action": assessment.get("decision", "WAIT"),
        "live_orders_enabled": False,
        "initial_equity": str(paper.get("initial_equity", 10_000)),
        "final_equity": str(paper.get("final_equity", 10_000)),
        "gross_pnl": str(float(paper.get("net_pnl", 0.0)) + fees + slippage + funding),
        "net_pnl": str(paper.get("net_pnl", 0.0)),
        "fees": str(fees),
        "slippage": str(slippage),
        "funding": str(funding),
        "max_drawdown": str(paper.get("max_drawdown", 0.0)),
        "risk_per_trade": str(paper.get("risk_per_trade", 0.01)),
        "max_leverage": str(paper.get("max_leverage", 10)),
        "signals": complete_candidates,
        "rejected_signals": max(0, complete_candidates - trade_signals),
        "kill_switches": 0,
        "fills": fills,
        "equity_curve": [
            {"timestamp": paper.get("paper_start"), "equity": paper.get("initial_equity", 10_000)},
            *(
                {"timestamp": trade["exit_at"], "equity": trade["balance"]}
                for trade in trades
            ),
        ],
        "telemetry": telemetry,
        "shadow_closed_trades": len(trades),
        "shadow_position": paper.get("open_position"),
        "pending_order": paper.get("pending_order"),
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def worker(*, once: bool) -> None:
    while True:
        try:
            audit = btc_cross_exchange_forward_audit.refresh_current_report()
            _atomic_json(REPORT, build_shadow_report(audit))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _atomic_json(
                REPORT,
                {
                    "mode": "shadow",
                    "strategy_profile": STRATEGY_PROFILE,
                    "validation_status": "RESEARCH_ONLY_FAIL_CLOSED",
                    "policy_action": "DATA_UNAVAILABLE",
                    "detail": str(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        if once:
            return
        await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musca V5 realistic Bitunix shadow view")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset-from-v2", action="store_true")
    args = parser.parse_args()
    asyncio.run(worker(once=args.once))


if __name__ == "__main__":
    main()
