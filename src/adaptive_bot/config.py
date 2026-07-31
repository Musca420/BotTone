from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptive_bot.domain.enums import TradingMode
from adaptive_bot.domain.exceptions import LiveTradingDisabled
from adaptive_bot.domain.models import Instrument


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StrategyConfig(ConfigModel):
    timeframe_minutes: int = Field(default=15, gt=0)
    atr_period: int = Field(default=14, gt=1)
    adx_period: int = Field(default=14, gt=1)
    ema_period: int = Field(default=50, gt=1)
    atr_percentile_window: int = Field(default=100, gt=10)
    crypto_vwap_window: int = Field(default=96, gt=1)
    slope_lookback: int = Field(default=5, gt=0)
    slope_threshold: float = Field(default=0.05, ge=0)
    atr_change_threshold: float = Field(default=0.5, gt=0)
    cumulative_move_atr: float = Field(default=3.0, gt=0)
    range_multiplier: float = Field(default=2.0, gt=0)
    entry_z: float = Field(default=1.5, gt=0)
    stop_atr: float = Field(default=2.5, gt=0)
    time_stop_bars: int = Field(default=8, gt=0)
    confirmation_bars: int = Field(default=3, gt=0)
    max_spread_bps: float = Field(default=5.0, gt=0)
    partial_exit_enabled: bool = False
    short_enabled: bool = False
    no_entry_minutes_after_open: int = Field(default=15, ge=0)
    flatten_minutes_before_close: int = Field(default=15, ge=0)


class RiskConfig(ConfigModel):
    risk_per_trade: Decimal = Field(default=Decimal("0.0025"), gt=0, le=1)
    max_daily_loss: Decimal = Field(default=Decimal("0.01"), gt=0, le=1)
    max_weekly_loss: Decimal = Field(default=Decimal("0.025"), gt=0, le=1)
    max_strategy_drawdown: Decimal = Field(default=Decimal("0.08"), gt=0, le=1)
    max_open_positions: int = Field(default=1, gt=0)
    max_correlated_positions: int = Field(default=1, gt=0)
    max_consecutive_losses: int = Field(default=3, gt=0)
    cooldown_after_losses: int = Field(default=8, gt=0)
    hard_notional_cap: Decimal = Field(default=Decimal("25000"), gt=0)


class BacktestConfig(ConfigModel):
    input_path: Path
    initial_equity: Decimal = Field(gt=0)
    spread_bps: Decimal = Field(default=Decimal("2"), ge=0)
    slippage_bps: Decimal = Field(default=Decimal("1"), ge=0)
    commission_per_unit: Decimal = Field(default=Decimal("0.005"), ge=0)
    max_volume_participation: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    minimum_quality_score: float = Field(default=1.0, ge=0, le=1)


class AlpacaConfig(ConfigModel):
    enabled: bool = False
    paper: bool = True
    feed: str = "iex"
    adjustment: str = "all"
    historical_days: int = Field(default=30, gt=0)
    stale_after_seconds: int = Field(default=90, gt=0)
    shadow: bool = True


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    trading_mode: TradingMode | None = None
    allow_live_trading: str | None = None
    database_url: str | None = None
    alpaca_api_key: SecretStr | None = None
    alpaca_api_secret: SecretStr | None = None
    alpaca_account_id: str | None = None
    okx_api_key: SecretStr | None = None
    okx_api_secret: SecretStr | None = None
    okx_api_passphrase: SecretStr | None = None
    ibkr_account_id: str | None = None


class AppConfig(ConfigModel):
    trading_mode: TradingMode
    broker: str
    database_url: str
    instrument: Instrument
    strategy: StrategyConfig
    risk: RiskConfig
    backtest: BacktestConfig
    alpaca: AlpacaConfig | None = None
    allow_live_trading: str | None = None
    allowed_instruments: tuple[str, ...] = ("QQQ",)
    allowed_accounts: tuple[str, ...] = ("SIM-QQQ",)

    @model_validator(mode="after")
    def fail_closed(self) -> AppConfig:
        if self.instrument.symbol not in self.allowed_instruments:
            raise ValueError("instrument is not allowlisted")
        if self.broker == "alpaca":
            if self.alpaca is None or not self.alpaca.enabled:
                raise ValueError("Alpaca adapter is disabled")
            if not self.alpaca.paper:
                raise LiveTradingDisabled("Alpaca live endpoint is disabled")
            if self.trading_mode is not TradingMode.PAPER:
                raise ValueError("Alpaca MVP requires paper trading mode")
        if self.trading_mode is TradingMode.LIVE:
            if self.allow_live_trading != "I_ACKNOWLEDGE_THE_RISK":
                raise LiveTradingDisabled("live acknowledgement missing")
            raise LiveTradingDisabled("live adapter is unavailable")
        return self


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    env = EnvironmentSettings()
    if env.trading_mode is not None:
        raw["trading_mode"] = env.trading_mode.value
    if env.database_url is not None:
        raw["database_url"] = env.database_url
    if env.allow_live_trading is not None:
        raw["allow_live_trading"] = env.allow_live_trading
    if env.alpaca_account_id is not None:
        raw["allowed_accounts"] = [env.alpaca_account_id]
    return AppConfig.model_validate(raw)


def alpaca_credentials() -> tuple[str, str]:
    env = EnvironmentSettings()
    if env.alpaca_api_key is None or env.alpaca_api_secret is None:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
    return env.alpaca_api_key.get_secret_value(), env.alpaca_api_secret.get_secret_value()


def redact_environment() -> dict[str, str]:
    return {
        key: "***"
        if any(token in key for token in ("KEY", "SECRET", "PASSPHRASE", "ACCOUNT"))
        else value
        for key, value in os.environ.items()
        if key.startswith(("TRADING_", "ALPACA_", "OKX_", "IBKR_", "DATABASE_"))
    }
