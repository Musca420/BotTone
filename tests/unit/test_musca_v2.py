from pathlib import Path

import pandas as pd

import adaptive_bot.musca_v2 as musca_v2
from adaptive_bot.services.bitunix_paper_service import read_collected_candles


def test_ranked_selector_uses_fold_local_ev_and_one_position() -> None:
    rows = pd.DataFrame(
        {
            "outer_fold": [1] * 10 + [2] * 10,
            "expected_net_bps": list(range(10)) * 2,
            "entry_timestamp": pd.date_range("2026-01-01", periods=20, freq="6h", tz="UTC"),
            "exit_timestamp": pd.date_range("2026-01-01 01:00", periods=20, freq="6h", tz="UTC"),
            "asset": ["BTCUSDT"] * 20,
        }
    )
    chosen = musca_v2.select_ranked_oos(rows)
    assert chosen["expected_net_bps"].tolist() == [9, 9]
    assert len(chosen) == 2


def test_v25_report_remains_present() -> None:
    assert Path("data/reports/ml_hybrid_v25_model_audit.json").exists()


def test_dashboard_hides_generic_panels_for_musca_v2() -> None:
    html = Path("src/adaptive_bot/dashboard/static/index.html").read_text(encoding="utf-8")
    script = Path("src/adaptive_bot/dashboard/static/app.js").read_text(encoding="utf-8")
    assert 'id="ml-panel" class="panel decision-panel wide generic-profile-only"' in html
    assert 'id="research-panel" class="panel fills-panel wide generic-profile-only"' in html
    assert "setMuscaView(muscaV2, muscaV4 || muscaV5 || muscaLiquidity)" in script


def test_shadow_starts_now_without_retroactive_fills() -> None:
    state_path = Path("data/research/test_musca_v2_state.json")
    original = musca_v2.STATE
    try:
        state_path.unlink(missing_ok=True)
        musca_v2.STATE = state_path
        candles = read_collected_candles(musca_v2.INPUT)
        state = musca_v2._advance_shadow(candles)
        assert state["fills"] == []
        assert pd.Timestamp(state["baseline"]) == pd.Timestamp(candles["timestamp"].max())
    finally:
        musca_v2.STATE = original
        state_path.unlink(missing_ok=True)
