import asyncio
import json
from pathlib import Path

import pytest

from adaptive_bot import musca_v5_shadow as shadow
from adaptive_bot.musca_v5_shadow import build_shadow_report


def test_shadow_report_uses_only_realistic_v5_paper_account() -> None:
    report = build_shadow_report(
        {
            "validation_status": "RESEARCH_ONLY",
            "alpha": {"scored_candidates": 2, "accepted_candidates": 1},
            "current_market_assessment": {
                "observed_at": "2026-08-08T12:00:00Z",
                "decision": "WAIT",
                "reason": "NO_SETUP",
                "price": 100,
            },
            "one_position_diagnostics": {
                "paper_account": {
                    "initial_equity": 10_000,
                    "final_equity": 10_000,
                    "net_pnl": 0,
                    "max_drawdown": 0,
                    "risk_per_trade": 0.01,
                    "max_leverage": 10,
                    "paper_start": "2026-08-08T10:00:00Z",
                    "trades": [],
                    "complete_candidate_count": 3,
                    "trade_signal_count": 1,
                }
            },
        }
    )
    assert report["initial_equity"] == "10000"
    assert report["final_equity"] == "10000"
    assert report["strategy_profile"] == "musca_v5_stable_multi_horizon_vwap"
    assert report["timeframe_minutes"] == 5
    assert "daily/impulse/swing VWAP" in report["vwap_session"]
    assert report["fills"] == []
    assert report["signals"] == 3
    assert report["rejected_signals"] == 2


def test_worker_refreshes_existing_forward_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "shadow.json"
    audit_path.write_text("{}", encoding="utf-8")
    payload = {
        "validation_status": "RESEARCH_ONLY_NO_ECONOMIC_ALPHA",
        "current_market_assessment": {
            "observed_at": "2026-08-09T08:45:00Z",
            "decision": "WAIT",
            "reason": "WAIT_NEXT_5M_DECISION",
        },
        "one_position_diagnostics": {"paper_account": {"trades": []}},
    }
    monkeypatch.setattr(shadow, "AUDIT", audit_path)
    monkeypatch.setattr(shadow, "REPORT", report_path)
    monkeypatch.setattr(
        shadow.btc_cross_exchange_forward_audit,
        "refresh_current_report",
        lambda: payload,
    )

    asyncio.run(shadow.worker(once=True))

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["telemetry"][0]["decision_reason"] == "WAIT_NEXT_5M_DECISION"
