from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from adaptive_bot.config import RiskConfig
from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import AccountSnapshot, Instrument, RiskDecision, Signal
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.risk.limits import drawdown, loss_limit_breached
from adaptive_bot.risk.position_sizing import SizingInput, size_position


@dataclass(frozen=True)
class RiskState:
    day_start_equity: Decimal
    week_start_equity: Decimal
    peak_equity: Decimal
    open_positions: int = 0
    correlated_positions: int = 0
    consecutive_losses: int = 0
    cooldown_bars: int = 0


class DefaultRiskEngine:
    def __init__(self, config: RiskConfig, kill_switch: KillSwitch) -> None:
        self.config = config
        self.kill_switch = kill_switch

    def assess(
        self,
        signal: Signal,
        instrument: Instrument,
        account: AccountSnapshot,
        state: RiskState,
        estimated_cost_per_unit: Decimal,
    ) -> RiskDecision:
        blocked = self._blocked_reason(account, state)
        if blocked is not None:
            return RiskDecision(approved=False, reason=blocked)
        if signal.stop_price is None:
            return RiskDecision(approved=False, reason="entry requires a protective stop")
        if instrument.max_leverage > 1:
            if not self.config.allow_simulated_leverage:
                return RiskDecision(
                    approved=False,
                    reason="reliable broker liquidation data is required for leveraged instruments",
                )
            stop_fraction = abs(signal.reference_price - signal.stop_price) / signal.reference_price
            liquidation_distance = (
                Decimal("1") / instrument.max_leverage - self.config.liquidation_buffer_fraction
            )
            if liquidation_distance < Decimal("3") * stop_fraction:
                return RiskDecision(approved=False, reason="simulated liquidation buffer is unsafe")
        side = Side.BUY if signal.action.value.endswith("long") else Side.SELL
        if side is Side.SELL and not instrument.shortable:
            return RiskDecision(approved=False, reason="short selling is disabled")
        return size_position(
            SizingInput(
                equity=account.equity,
                buying_power=account.buying_power,
                entry_price=signal.reference_price,
                stop_price=signal.stop_price,
                estimated_cost_per_unit=estimated_cost_per_unit,
                risk_fraction=self.config.risk_per_trade,
                hard_notional_cap=min(
                    self.config.hard_notional_cap,
                    account.equity * self.config.target_exposure_fraction,
                ),
                side=side,
            ),
            instrument,
        )

    def _blocked_reason(self, account: AccountSnapshot, state: RiskState) -> str | None:
        if self.kill_switch.active:
            return "kill switch active"
        if account.equity <= 0:
            return "equity is zero"
        if loss_limit_breached(account.equity, state.day_start_equity, self.config.max_daily_loss):
            return "daily loss limit"
        if loss_limit_breached(
            account.equity, state.week_start_equity, self.config.max_weekly_loss
        ):
            return "weekly loss limit"
        if drawdown(account.equity, state.peak_equity) >= self.config.max_strategy_drawdown:
            return "strategy drawdown limit"
        if state.open_positions >= self.config.max_open_positions:
            return "maximum open positions"
        if state.correlated_positions >= self.config.max_correlated_positions:
            return "maximum correlated positions"
        if state.consecutive_losses >= self.config.max_consecutive_losses:
            return "consecutive loss limit"
        if state.cooldown_bars > 0:
            return "cooldown active"
        return None
