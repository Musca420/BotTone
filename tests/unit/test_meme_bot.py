import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from importlib.resources import files

import pandas as pd
import pytest

from adaptive_bot.dashboard.meme_server import build_meme_dashboard_payload
from adaptive_bot.domain.enums import MarketRegime, Side, TradingMode
from adaptive_bot.domain.exceptions import LiveTradingDisabled
from adaptive_bot.meme.collector import (
    MemeCollectorState,
    SymbolStreamState,
    adaptive_range_metrics,
    apply_ws_message,
)
from adaptive_bot.meme.config import MemeBotConfig, MemeStrategyConfig, load_meme_config
from adaptive_bot.meme.research import shadow_status
from adaptive_bot.meme.runtime import (
    MemePaperEngine,
    PaperPosition,
    load_market_qualities,
    load_recorded_frames,
)
from adaptive_bot.meme.strategy import (
    MemeDecision,
    MemeMomentumStrategy,
    MemeStrategyState,
    build_meme_features,
    triple_barrier_label,
)
from adaptive_bot.meme.universe import (
    MarketQuality,
    choose_leverage,
    intersect_meme_contracts,
    rank_candidates,
)


def test_meme_config_is_isolated_and_live_locked() -> None:
    config = load_meme_config("configs/bitunix_meme_paper.yaml")
    assert config.dashboard_port == 8081
    assert config.initial_equity == Decimal("100")
    assert config.risk.risk_per_trade == Decimal("0.0025")
    assert config.risk.hard_risk_cap == Decimal("0.003")
    assert config.risk.hard_notional_cap == Decimal("40")
    assert config.risk.leverage_ceiling == 3
    assert config.risk.max_open_positions == 2
    with pytest.raises(LiveTradingDisabled):
        MemeBotConfig(trading_mode=TradingMode.LIVE)


def test_universe_intersection_rejects_ambiguous_symbols_and_ranks_quality() -> None:
    pairs = {
        "code": 0,
        "data": [
            {
                "symbol": "DOGEUSDT",
                "base": "DOGE",
                "quote": "USDT",
                "status": "OPEN",
                "tickSize": "0.00001",
                "lotSize": "1",
                "minTradeVolume": "10",
                "minNotional": "5",
                "maxLeverage": "20",
            },
            {"symbol": "MEMEUSDT", "base": "MEME", "quote": "USDT"},
        ],
    }
    coins = [
        {"id": "dogecoin", "symbol": "doge"},
        {"id": "meme-one", "symbol": "meme"},
        {"id": "meme-two", "symbol": "meme"},
    ]
    contracts = intersect_meme_contracts(pairs, coins)
    assert [contract.symbol for contract in contracts] == ["DOGEUSDT"]
    assert contracts[0].instrument(5).max_leverage == Decimal("5")

    quality = MarketQuality(
        quote_volume_24h=Decimal("10000000"),
        spread_bps=Decimal("5"),
        depth_half_percent=Decimal("50000"),
        mark_divergence=Decimal("0.001"),
        funding_8h=Decimal("0.0002"),
        history_hours=200,
        momentum_atr=Decimal("3"),
        volume_zscore=Decimal("2.5"),
    )
    ranked = rank_candidates(
        contracts,
        {"DOGEUSDT": quality},
        load_meme_config("configs/bitunix_meme_paper.yaml").universe,
        Decimal("1000"),
    )
    assert ranked[0].eligible
    assert ranked[0].rank == 1


def test_leverage_uses_lowest_value_under_margin_cap() -> None:
    assert choose_leverage(Decimal("1900"), Decimal("10000"), Decimal("0.10"), 5) == 2
    assert choose_leverage(Decimal("2500"), Decimal("10000"), Decimal("0.10"), 5) == 3
    assert choose_leverage(Decimal("4500"), Decimal("10000"), Decimal("0.10"), 5) == 5
    assert choose_leverage(Decimal("6000"), Decimal("10000"), Decimal("0.10"), 5) is None


def test_breakout_requires_later_retest_and_builds_one_r_levels() -> None:
    strategy = MemeMomentumStrategy(MemeStrategyConfig())
    timestamp = datetime(2026, 8, 2, 8, tzinfo=UTC)
    breakout = _strategy_row(timestamp, close=101, low=100.5, high=102, breakout_high=100)
    decision, state = strategy.evaluate(breakout, "DOGEUSDT", MemeStrategyState())
    assert decision.action == "watch"
    assert state.pending_side is Side.BUY

    retest = _strategy_row(
        timestamp.replace(minute=5), close=100.5, low=99.8, high=101, breakout_high=100
    )
    decision, state = strategy.evaluate(retest, "DOGEUSDT", state)
    assert decision.action == "enter_long"
    assert decision.stop_price == Decimal("99.30")
    assert decision.target_price == Decimal("102.6000")
    assert decision.regime is MarketRegime.TREND_UP
    assert state.initial_risk == Decimal("1.20")


def test_feature_breakout_level_excludes_current_candle() -> None:
    timestamps = pd.date_range("2026-01-01", periods=800, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 800,
            "high": [101.0] * 799 + [999.0],
            "low": [99.0] * 800,
            "close": [100.0] * 800,
            "volume": [1000.0] * 800,
        }
    )
    features = build_meme_features(frame, MemeStrategyConfig())
    assert features.iloc[-1]["breakout_high"] == 101


def test_meme_adaptive_range_enters_only_in_sideways_extreme() -> None:
    strategy = MemeMomentumStrategy(MemeStrategyConfig())
    row = _strategy_row(
        datetime(2026, 8, 2, 8, tzinfo=UTC),
        close=100,
        low=99,
        high=101,
        breakout_high=110,
    )
    row["adx_1h"] = 10
    row["adaptive_center"] = 104
    row["adaptive_z"] = -2
    decision, _ = strategy.evaluate(row, "DOGEUSDT", MemeStrategyState())
    assert decision.action == "enter_long"
    assert decision.strategy_name == "adaptive_range"
    assert decision.stop_price == Decimal("97.50")
    assert decision.target_price == Decimal("104")


def test_adaptive_range_scanner_scores_sideways_market() -> None:
    candles = []
    for index in range(200):
        close = Decimal("100") + Decimal(index % 6 - 3) / Decimal("10")
        candles.append(
            {
                "time": index,
                "open": str(close),
                "high": str(close + Decimal("0.5")),
                "low": str(close - Decimal("0.5")),
                "close": str(close),
                "baseVol": "1000",
            }
        )
    metrics = adaptive_range_metrics(candles, MemeBotConfig())
    assert metrics["range_favorable"] is True


def test_triple_barrier_is_next_event_and_worst_case() -> None:
    frame = pd.DataFrame(
        [
            {"high": 100, "low": 100},
            {"high": 102, "low": 98},
            {"high": 103, "low": 101},
        ]
    )
    assert (
        triple_barrier_label(frame, 0, Side.BUY, Decimal("100"), Decimal("99"), Decimal("101"), 2)
        == -1
    )


def test_websocket_depth_and_trade_flow_are_normalized() -> None:
    state = MemeCollectorState(symbols={"DOGEUSDT": SymbolStreamState("DOGEUSDT")})
    apply_ws_message(
        state,
        {
            "ch": "depth_book15",
            "symbol": "DOGEUSDT",
            "data": {"b": [["99", "10"]], "a": [["101", "20"]]},
        },
    )
    apply_ws_message(
        state,
        {
            "ch": "price",
            "symbol": "DOGEUSDT",
            "data": {
                "mp": "100",
                "ip": "100.1",
                "fr": "0.0001",
                "ft": "2026-08-02T00:00:00Z",
                "nft": "2026-08-02T04:00:00Z",
            },
        },
    )
    apply_ws_message(
        state,
        {
            "ch": "trade",
            "symbol": "DOGEUSDT",
            "data": [{"p": "100", "v": "3", "s": "buy"}],
        },
    )
    stream = state.symbols["DOGEUSDT"]
    assert Decimal(stream.spread_bps or "0") == Decimal("200")
    assert stream.depth_half_percent == "0"
    assert stream.buy_volume == "3"
    assert stream.trade_count == 1
    assert Decimal(stream.funding_interval_hours) == Decimal("4")


def test_invalid_funding_is_unknown_and_fails_closed() -> None:
    state = MemeCollectorState(symbols={"DOGEUSDT": SymbolStreamState("DOGEUSDT")})
    apply_ws_message(
        state,
        {
            "ch": "price",
            "symbol": "DOGEUSDT",
            "data": {
                "mp": "1",
                "ip": "1",
                "fr": "NaN",
                "ft": "2026-08-02T00:00:00Z",
                "nft": "2026-08-02T08:00:00Z",
            },
        },
    )
    assert state.symbols["DOGEUSDT"].funding_rate is None

    contract = intersect_meme_contracts(
        {"data": [{"symbol": "DOGEUSDT", "base": "DOGE", "quote": "USDT"}]},
        [{"id": "dogecoin", "symbol": "doge"}],
    )[0]
    quality = MarketQuality(
        quote_volume_24h=Decimal("10000000"),
        spread_bps=Decimal("1"),
        depth_half_percent=Decimal("10000"),
        mark_divergence=Decimal("0"),
        funding_8h=None,
        history_hours=200,
        momentum_atr=Decimal("1"),
        volume_zscore=Decimal("2"),
    )
    ranked = rank_candidates(
        (contract,),
        {"DOGEUSDT": quality},
        load_meme_config("configs/bitunix_meme_paper.yaml").universe,
        Decimal("40"),
    )
    assert not ranked[0].eligible
    assert "funding_unavailable" in ranked[0].reasons


def test_paper_sizing_respects_margin_and_stop_wins_ambiguous_bar() -> None:
    config = load_meme_config("configs/bitunix_meme_paper.yaml")
    contract = intersect_meme_contracts(
        {
            "data": [
                {
                    "symbol": "DOGEUSDT",
                    "base": "DOGE",
                    "quote": "USDT",
                    "tickSize": "0.01",
                    "lotSize": "1",
                    "minTradeVolume": "1",
                    "minNotional": "1",
                    "maxLeverage": "5",
                }
            ]
        },
        [{"id": "dogecoin", "symbol": "doge"}],
    )[0]
    engine = MemePaperEngine(config, (contract,))
    decision = MemeDecision(
        datetime(2026, 1, 1, tzinfo=UTC),
        "DOGEUSDT",
        "enter_long",
        "test",
        Decimal("1"),
        Decimal("0.98"),
        Decimal("1.02"),
        MarketRegime.TREND_UP,
    )
    fill = engine._entry_fill(decision, pd.Series({"open": 1}), contract, Decimal("100"))
    assert fill is not None
    price, sizing = fill
    assert sizing.leverage in {1, 2, 3}
    assert price * sizing.quantity <= Decimal("40")
    assert price * sizing.quantity / sizing.leverage <= Decimal("20")
    assert sizing.effective_risk <= sizing.risk_budget
    assert sizing.effective_risk <= Decimal("0.30")

    position = PaperPosition(
        "DOGEUSDT",
        Side.BUY,
        Decimal("10"),
        Decimal("100"),
        Decimal("98"),
        Decimal("102"),
        2,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    remaining, _, operations = engine._process_position(
        position,
        pd.Series({"open": 100, "high": 103, "low": 97, "close": 101, "atr": 2}),
        datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        Decimal("10000"),
    )
    assert remaining is None
    assert operations[0]["event"] == "STOP"


def test_recorded_frames_deduplicate_live_kline_updates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "events.jsonl"
    messages = [
        {
            "received_at": "2026-01-01T00:00:01+00:00",
            "message": {
                "ch": "market_kline_5min",
                "symbol": "DOGEUSDT",
                "ts": 1767225601000,
                "data": {"o": "1", "h": "2", "l": "0.5", "c": close, "v": "10"},
            },
        }
        for close in ("1.1", "1.2")
    ]
    path.write_text("\n".join(json.dumps(message) for message in messages), encoding="utf-8")
    frames = load_recorded_frames(path)
    assert len(frames["DOGEUSDT"]) == 1
    assert frames["DOGEUSDT"].iloc[0]["close"] == "1.2"


def test_market_quality_reader_fails_closed_during_atomic_replace(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "stream.json"
    path.write_text("{}", encoding="utf-8")

    def missing(*_args, **_kwargs) -> str:  # type: ignore[no-untyped-def]
        raise FileNotFoundError

    monkeypatch.setattr(path.__class__, "read_text", missing)
    assert load_market_qualities(path, {}) == {}


def test_shadow_gate_and_dashboard_never_claim_live_execution(tmp_path) -> None:  # type: ignore[no-untyped-def]
    report = tmp_path / "paper.json"
    stream = tmp_path / "stream.json"
    report.write_text(
        json.dumps(
            {
                "final_equity": "10000",
                "operations": [],
                "audit": [],
                "scanner": [],
                "equity_curve": [],
                "probabilistic": shadow_status(pd.DataFrame()),
            }
        ),
        encoding="utf-8",
    )
    stream.write_text(json.dumps({"connected": True, "symbols": {}}), encoding="utf-8")
    payload = build_meme_dashboard_payload(report, stream)
    assert payload["available"]
    assert payload["safety"]["live_enabled"] is False
    assert payload["report"]["probabilistic"]["can_trade"] is False


def test_meme_dashboard_script_only_references_existing_elements() -> None:
    assets = files("adaptive_bot.dashboard.meme_static")
    html = assets.joinpath("index.html").read_text(encoding="utf-8")
    script = assets.joinpath("meme.js").read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("([^"]+)"\)', script))
    available = set(re.findall(r'id="([^"]+)"', html))
    assert referenced <= available


def _strategy_row(
    timestamp: datetime,
    *,
    close: float,
    low: float,
    high: float,
    breakout_high: float,
) -> pd.Series:
    return pd.Series(
        {
            "timestamp": timestamp,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 10000,
            "atr": 2,
            "breakout_high": breakout_high,
            "breakout_low": 90,
            "volume_zscore": 3,
            "range_atr": 1,
            "three_bar_atr": 1,
            "momentum_atr": 2,
            "adaptive_center": close,
            "adaptive_z": 0,
            "ema_fast_1h": 105,
            "ema_slow_1h": 100,
            "ema_slope_1h": 0.01,
            "adx_1h": 30,
            "ema_fast_5m": 99,
            "previous_close": close - 1,
        }
    )
