from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptive_bot.domain.enums import TradingMode
from adaptive_bot.domain.exceptions import LiveTradingDisabled
from adaptive_bot.domain.models import Instrument


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StrategyConfig(ConfigModel):
    entry_mode: Literal[
        "range", "weighted_reversion", "weighted_reversion_v11", "weighted_reversion_v2"
    ] = "range"
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
    range_adx_threshold: float = Field(default=20.0, gt=0)
    trend_adx_threshold: float = Field(default=25.0, gt=0)
    range_multiplier: float = Field(default=2.0, gt=0)
    entry_z: float = Field(default=1.5, gt=0)
    weighted_entry_threshold: float = Field(default=0.65, gt=0, le=1)
    v11_exit_z: float = Field(default=0.5, ge=0)
    v11_cooldown_bars: int = Field(default=6, ge=0)
    stop_atr: float = Field(default=2.5, gt=0)
    time_stop_bars: int = Field(default=8, gt=0)
    confirmation_bars: int = Field(default=3, gt=0)
    max_spread_bps: float = Field(default=5.0, gt=0)
    partial_exit_enabled: bool = False
    short_enabled: bool = False
    no_entry_minutes_after_open: int = Field(default=15, ge=0)
    flatten_minutes_before_close: int = Field(default=15, ge=0)
    session_flatten_enabled: bool = True
    fixed_stop_fraction: Decimal | None = Field(default=None, gt=0, lt=1)
    fixed_target_fraction: Decimal | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def fixed_risk_reward_is_complete(self) -> StrategyConfig:
        if self.range_adx_threshold > self.trend_adx_threshold:
            raise ValueError("range ADX threshold must not exceed trend ADX threshold")
        if (self.fixed_stop_fraction is None) != (self.fixed_target_fraction is None):
            raise ValueError("fixed stop and target fractions must be configured together")
        if (
            self.fixed_stop_fraction is not None
            and self.fixed_stop_fraction != self.fixed_target_fraction
        ):
            raise ValueError("fixed stop and target must preserve 1:1 risk/reward")
        return self


class RiskConfig(ConfigModel):
    risk_per_trade: Decimal = Field(default=Decimal("0.01"), gt=0, le=1)
    max_daily_loss: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    max_weekly_loss: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_strategy_drawdown: Decimal = Field(default=Decimal("0.08"), gt=0, le=1)
    max_open_positions: int = Field(default=1, gt=0)
    max_correlated_positions: int = Field(default=1, gt=0)
    max_consecutive_losses: int = Field(default=3, gt=0)
    cooldown_after_losses: int = Field(default=8, gt=0)
    hard_notional_cap: Decimal = Field(default=Decimal("25000"), gt=0)
    target_exposure_fraction: Decimal = Field(default=Decimal("1"), gt=0, le=1)
    max_margin_fraction: Decimal = Field(default=Decimal("1"), gt=0, le=1)
    allow_simulated_leverage: bool = False
    liquidation_buffer_fraction: Decimal = Field(default=Decimal("0.01"), ge=0, lt=1)


class BacktestConfig(ConfigModel):
    input_path: Path
    initial_equity: Decimal = Field(gt=0)
    spread_bps: Decimal = Field(default=Decimal("2"), ge=0)
    slippage_bps: Decimal = Field(default=Decimal("1"), ge=0)
    commission_per_unit: Decimal = Field(default=Decimal("0.005"), ge=0)
    maker_fee_bps: Decimal = Field(default=Decimal("0"), ge=0)
    taker_fee_bps: Decimal = Field(default=Decimal("0"), ge=0)
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
    paper_execution_enabled: bool = False


class BitunixConfig(ConfigModel):
    enabled: bool = False
    market: Literal["spot", "futures"]
    simulated_execution_only: bool = True
    margin_coin: Literal["USDT"] = "USDT"
    margin_mode: Literal["isolated"] = "isolated"
    leverage: Decimal = Field(default=Decimal("10"), ge=1)


class ResearchConfig(ConfigModel):
    enabled: bool = False
    history_months: int = Field(default=12, gt=0)
    broad_candidates: int = Field(default=200, ge=0)
    random_seed: int = 20260802
    database_path: Path = Path("data/research/research.duckdb")
    report_path: Path = Path("data/reports/research.json")
    history_path: Path = Path("data/processed/bitunix_btcusdt_mark_5m.parquet")
    top_shadow_candidates: int = Field(default=10, gt=0)
    gpu_enabled: bool = False
    parallel_workers: int = Field(default=1, gt=0, le=16)


class MachineLearningConfig(ConfigModel):
    enabled: bool = False
    protocol_version: Literal["legacy_v1", "scientific_v2"] = "scientific_v2"
    history_path: Path = Path("data/processed/bitunix_btcusdt_observed_5m.parquet")
    archive_directory: Path = Path("data/ml/raw")
    manifest_path: Path = Path("data/ml/dataset_manifest.json")
    report_path: Path = Path("data/reports/ml_research.json")
    status_path: Path = Path("data/reports/ml_research.status.json")
    model_path: Path = Path("data/models/adaptive_range_ml.joblib")
    candidate_directory: Path = Path("data/models/candidates")
    model_card_path: Path = Path("data/reports/ml_model_card.md")
    optuna_path: Path = Path("data/research/ml_optuna.db")
    trials: int = Field(default=20, gt=0)
    n_splits: int = Field(default=5, ge=3, le=10)
    max_holding_bars: int = Field(default=12, gt=1)
    stop_atr: float = Field(default=2.5, gt=0)
    probability_threshold: float = Field(default=0.67, gt=0.5, lt=1)
    taker_fee_bps: float = Field(default=6.0, ge=0)
    random_seed: int = 20260803
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    timeframes: tuple[int, ...] = (5, 15, 30)
    strategy_candidates: int = Field(default=3000, gt=0)
    full_candidates: int = Field(default=120, gt=0)
    meta_candidates: int = Field(default=12, gt=0)
    model_trials_per_side: int = Field(default=40, gt=0)
    gpu_required: bool = True
    download_workers: int = Field(default=4, gt=0, le=8)
    parallel_workers: int = Field(default=4, gt=0, le=16)
    holdout_weeks: int = Field(default=12, gt=0)
    minimum_oos_trades: int = Field(default=300, gt=0)
    minimum_holdout_trades: int = Field(default=30, gt=0)
    minimum_side_trades: int = Field(default=100, gt=0)
    minimum_profit_factor: float = Field(default=1.15, gt=0)
    maximum_pbo: float = Field(default=0.20, ge=0, le=1)
    minimum_dsr_probability: float = Field(default=0.95, ge=0, le=1)
    maximum_reality_check_pvalue: float = Field(default=0.05, ge=0, le=1)
    outer_train_weeks: int = Field(default=52, gt=0)
    outer_calibration_weeks: int = Field(default=4, gt=0)
    outer_test_weeks: int = Field(default=4, gt=0)
    outer_step_weeks: int = Field(default=4, gt=0)
    v6_microstructure_directory: Path = Path("data/raw/bitunix_microstructure")
    v6_report_path: Path = Path("data/reports/ml_expert_research_v6.json")
    v6_status_path: Path = Path("data/reports/ml_expert_research_v6.status.json")
    maker_fee_bps: float = Field(default=2.0, ge=0)
    maker_taker_fee_bps: float = Field(default=6.0, ge=0)
    maker_fit_weeks: int = Field(default=6, gt=0)
    maker_calibration_weeks: int = Field(default=2, gt=0)
    maker_holdout_weeks: int = Field(default=4, gt=0)


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
    bitunix: BitunixConfig | None = None
    research: ResearchConfig | None = None
    machine_learning: MachineLearningConfig | None = None
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
        if self.bitunix is not None:
            if not self.bitunix.enabled:
                raise ValueError("Bitunix market data is disabled")
            if not self.bitunix.simulated_execution_only or self.broker != "simulated":
                raise LiveTradingDisabled("Bitunix execution is limited to the simulated broker")
            if (
                self.bitunix.market != "futures"
                or self.instrument.symbol != "BTCUSDT"
                or self.instrument.currency != "USDT"
            ):
                raise ValueError("Bitunix is restricted to BTCUSDT USDT-margined futures")
            if self.bitunix.leverage != self.instrument.max_leverage:
                raise ValueError("Bitunix leverage must match the instrument leverage")
            margin_fraction = self.risk.target_exposure_fraction / self.bitunix.leverage
            if margin_fraction > self.risk.max_margin_fraction:
                raise ValueError("target exposure exceeds the margin allocation cap")
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
