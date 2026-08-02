from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_bot.domain.enums import TradingMode
from adaptive_bot.domain.exceptions import LiveTradingDisabled


class MemeConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemeUniverseConfig(MemeConfigModel):
    category: str = "meme-token"
    quote_currency: Literal["USDT"] = "USDT"
    minimum_listing_days: int = Field(default=7, ge=7)
    minimum_quote_volume: Decimal = Field(default=Decimal("5000000"), ge=0)
    maximum_spread_bps: Decimal = Field(default=Decimal("20"), gt=0)
    preferred_spread_bps: Decimal = Field(default=Decimal("10"), gt=0)
    minimum_depth_multiple: Decimal = Field(default=Decimal("20"), gt=0)
    maximum_mark_divergence: Decimal = Field(default=Decimal("0.005"), gt=0)
    maximum_funding_8h: Decimal = Field(default=Decimal("0.001"), gt=0)
    preferred_funding_8h: Decimal = Field(default=Decimal("0.0005"), ge=0)
    minimum_liquidity_score: Decimal = Field(default=Decimal("0.75"), ge=0, le=1)
    maximum_manipulation_probability: Decimal = Field(default=Decimal("0.55"), ge=0, le=1)
    detailed_symbols: int = Field(default=10, gt=0, le=50)
    catalog_cache_hours: int = Field(default=24, gt=0)
    fail_closed_after_hours: int = Field(default=72, gt=0)


class MemeStrategyConfig(MemeConfigModel):
    timeframe_minutes: Literal[5] = 5
    context_minutes: Literal[60] = 60
    atr_period: int = Field(default=14, gt=1)
    adx_period: int = Field(default=14, gt=1)
    fast_ema: int = Field(default=20, gt=1)
    slow_ema: int = Field(default=50, gt=2)
    minimum_adx: Decimal = Field(default=Decimal("25"), ge=0)
    breakout_bars: int = Field(default=20, gt=2)
    volume_window: int = Field(default=20, gt=2)
    minimum_volume_zscore: Decimal = Decimal("0")
    minimum_momentum_atr: Decimal = Field(default=Decimal("1.5"), gt=0)
    aggressive_momentum_atr: Decimal = Field(default=Decimal("2.5"), gt=0)
    confirmed_reward_risk: Decimal = Field(default=Decimal("1.75"), ge=Decimal("1.6"))
    short_reward_risk: Decimal = Field(default=Decimal("1.8"), ge=Decimal("1.8"))
    aggressive_reward_risk: Decimal = Field(default=Decimal("1.6"), ge=Decimal("1.6"))
    retest_bars: int = Field(default=3, gt=0)
    retest_tolerance_atr: Decimal = Field(default=Decimal("0.25"), gt=0)
    stop_buffer_atr: Decimal = Field(default=Decimal("0.25"), ge=0)
    maximum_stop_atr: Decimal = Field(default=Decimal("1.5"), gt=0)
    shock_candle_atr: Decimal = Field(default=Decimal("2.5"), gt=0)
    shock_three_bar_atr: Decimal = Field(default=Decimal("4"), gt=0)
    trailing_atr: Decimal = Field(default=Decimal("2"), gt=0)
    time_stop_bars: int = Field(default=24, gt=0)
    short_enabled: bool = True
    short_time_stop_bars: int = Field(default=12, gt=0)

    @model_validator(mode="after")
    def valid_ema_order(self) -> MemeStrategyConfig:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast EMA must be shorter than slow EMA")
        return self


class MemeRiskConfig(MemeConfigModel):
    risk_per_trade: Decimal = Field(default=Decimal("0.0025"), gt=0, le=Decimal("0.003"))
    early_entry_risk: Decimal = Field(default=Decimal("0.00125"), gt=0, le=Decimal("0.0015"))
    hard_risk_cap: Decimal = Field(default=Decimal("0.003"), gt=0, le=Decimal("0.003"))
    hard_notional_cap: Decimal = Field(default=Decimal("40"), gt=0)
    max_daily_loss: Decimal = Field(default=Decimal("0.015"), gt=0, le=1)
    max_weekly_loss: Decimal = Field(default=Decimal("0.04"), gt=0, le=1)
    max_strategy_drawdown: Decimal = Field(default=Decimal("0.08"), gt=0, le=1)
    max_open_positions: Literal[1, 2] = 2
    max_portfolio_heat: Decimal = Field(default=Decimal("0.01"), gt=0, le=Decimal("0.01"))
    max_consecutive_losses: int = Field(default=3, gt=0)
    cooldown_bars: int = Field(default=8, gt=0)
    max_margin_fraction: Decimal = Field(default=Decimal("0.20"), gt=0, le=Decimal("0.20"))
    leverage_ceiling: Literal[1, 2, 3, 5] = 3
    experimental_5x_enabled: bool = False
    short_risk_multiplier: Decimal = Field(default=Decimal("0.75"), gt=0, le=1)
    liquidation_buffer_fraction: Decimal = Field(default=Decimal("0.01"), ge=0, lt=1)
    estimated_round_trip_cost_bps: Decimal = Field(default=Decimal("20"), ge=0)

    @model_validator(mode="after")
    def valid_caps(self) -> MemeRiskConfig:
        if self.risk_per_trade > self.hard_risk_cap:
            raise ValueError("risk per trade cannot exceed the hard risk cap")
        return self


class MemeModelConfig(MemeConfigModel):
    mode: Literal["paper_bootstrap", "paper_validated"] = "paper_bootstrap"
    minimum_p_win: Decimal = Field(default=Decimal("0.56"), ge=0, le=1)
    minimum_short_p_win: Decimal = Field(default=Decimal("0.58"), ge=0, le=1)
    minimum_expected_value_r: Decimal = Decimal("0")
    minimum_mfe_mae_ratio: Decimal = Field(default=Decimal("1.25"), gt=0)
    minimum_samples: int = Field(default=1000, gt=0)
    minimum_weeks: int = Field(default=20, gt=0)
    minimum_symbols: int = Field(default=10, gt=0)


class MemeLunaConfig(MemeConfigModel):
    enabled: bool = True
    policy_required: bool = True
    minimum_regime_confidence: Decimal = Field(default=Decimal("0.60"), ge=0, le=1)
    maximum_systemic_risk: Decimal = Field(default=Decimal("0.75"), ge=0, le=1)
    policy_hours: int = Field(default=6, gt=0, le=24)
    max_runs_per_day: int = Field(default=12, gt=0, le=24)
    eligible_refresh_cooldown_minutes: int = Field(default=15, gt=0, le=60)
    low_timeout_seconds: int = Field(default=90, gt=0, le=300)
    codex_command: str = "codex.cmd"
    storage_directory: Path = Path("data/meme/luna")
    allowed_source_domains: tuple[str, ...] = (
        "bitunix.com",
        "coingecko.com",
        "federalreserve.gov",
        "ecb.europa.eu",
        "sec.gov",
        "cftc.gov",
    )


class MemeStorageConfig(MemeConfigModel):
    database_url: str = "sqlite:///data/meme/runtime.sqlite"
    raw_directory: Path = Path("data/meme/raw")
    processed_directory: Path = Path("data/meme/processed")
    report_path: Path = Path("data/meme/reports/paper.json")


class MemeBotConfig(MemeConfigModel):
    trading_mode: TradingMode = TradingMode.PAPER
    initial_equity: Decimal = Field(default=Decimal("100"), gt=0)
    universe: MemeUniverseConfig = MemeUniverseConfig()
    strategy: MemeStrategyConfig = MemeStrategyConfig()
    risk: MemeRiskConfig = MemeRiskConfig()
    models: MemeModelConfig = MemeModelConfig()
    luna: MemeLunaConfig = MemeLunaConfig()
    storage: MemeStorageConfig = MemeStorageConfig()
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(default=8081, ge=1024, le=65535)
    simulated_execution_only: bool = True

    @model_validator(mode="after")
    def live_is_disabled(self) -> MemeBotConfig:
        if self.trading_mode is TradingMode.LIVE or not self.simulated_execution_only:
            raise LiveTradingDisabled("meme live execution is implemented as a locked boundary")
        return self


def load_meme_config(path: str | Path) -> MemeBotConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("meme configuration root must be a mapping")
    return MemeBotConfig.model_validate(raw)
