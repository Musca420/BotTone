from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
from pydantic import BaseModel, ConfigDict

from adaptive_bot.adapters.simulated.broker import SimulatedBroker
from adaptive_bot.backtest.costs import estimated_round_trip_cost_per_unit
from adaptive_bot.backtest.metrics import maximum_drawdown
from adaptive_bot.config import AppConfig
from adaptive_bot.data.validation import ValidationReport, validate_candles
from adaptive_bot.domain.enums import (
    AssetClass,
    KillSwitchCause,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
)
from adaptive_bot.domain.models import Candle, Fill, OrderRequest, Signal, StrategyState
from adaptive_bot.execution.idempotency import client_order_id
from adaptive_bot.execution.order_manager import OrderManager
from adaptive_bot.risk.engine import DefaultRiskEngine, RiskState
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.risk.position_sizing import floor_to_lot
from adaptive_bot.strategy.adaptive_range import AdaptiveRangeStrategy, build_features
from adaptive_bot.strategy.regime import RegimeClassifier, RegimeFeatures
from adaptive_bot.strategy.signals import MarketSnapshot


class ResultModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class EquityPoint(ResultModel):
    timestamp: datetime
    equity: Decimal


class TelemetryPoint(ResultModel):
    timestamp: datetime
    close: Decimal
    center: Decimal | None
    lower_band: Decimal | None
    upper_band: Decimal | None
    atr: float | None
    adx: float | None
    z_score: float | None
    atr_percentile: float | None
    ema_slope: float | None
    spread_bps: float
    regime: str
    equity: Decimal
    position_quantity: Decimal
    activity: str


class BacktestResult(ResultModel):
    mode: str = "backtest"
    instrument: str
    timeframe_minutes: int
    risk_per_trade: Decimal
    max_daily_loss: Decimal
    max_weekly_loss: Decimal
    initial_equity: Decimal
    final_equity: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    fees: Decimal
    slippage: Decimal
    max_drawdown: Decimal
    fills: tuple[Fill, ...]
    equity_curve: tuple[EquityPoint, ...]
    telemetry: tuple[TelemetryPoint, ...]
    signals: int
    rejected_signals: int
    kill_switches: int

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)


class BacktestEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.kill_switch = KillSwitch()
        self.broker = SimulatedBroker(
            config.instrument,
            config.backtest.initial_equity,
            spread_bps=config.backtest.spread_bps,
            slippage_bps=config.backtest.slippage_bps,
            commission_per_unit=config.backtest.commission_per_unit,
            max_volume_participation=config.backtest.max_volume_participation,
        )
        self.order_manager = OrderManager(self.broker, self.kill_switch)
        self.risk_engine = DefaultRiskEngine(config.risk, self.kill_switch)
        self.strategy = AdaptiveRangeStrategy(config.strategy)
        self.classifier = RegimeClassifier(config.strategy)

    async def run(
        self,
        frame: pd.DataFrame,
        *,
        trade_after: datetime | None = None,
        mode: str = "backtest",
    ) -> BacktestResult:
        report = validate_candles(
            frame,
            timeframe_minutes=self.config.strategy.timeframe_minutes,
            calendar_name=(
                None if self.config.instrument.asset_class is AssetClass.CRYPTO else "NYSE"
            ),
        )
        report.require(self.config.backtest.minimum_quality_score)
        data, sessions, opens, closes = self._prepare(frame, report)
        features = build_features(
            data,
            sessions,
            self.config.strategy,
            self.config.instrument.asset_class,
        )
        strategy_state = StrategyState()
        initial = self.config.backtest.initial_equity
        risk_state = RiskState(initial, initial, initial)
        equity_curve: list[EquityPoint] = []
        telemetry: list[TelemetryPoint] = []
        signal_count = 0
        rejected = 0
        last_session: str | None = None
        last_week: tuple[int, int] | None = None
        last_realized = Decimal("0")

        for position_index, (_, row) in enumerate(features.iterrows()):
            candle = self._candle(row)
            spread_bps = self._spread_bps(row)
            self.broker.spread_bps = spread_bps
            session = str(sessions.iloc[position_index])
            week = candle.exchange_timestamp.isocalendar()[:2]
            await self.broker.process_candle(candle)
            await self._cancel_orphaned_brackets()
            account = await self.broker.get_account()
            consecutive = risk_state.consecutive_losses
            cooldown = max(0, risk_state.cooldown_bars - 1)
            if account.realized_pnl != last_realized:
                consecutive = consecutive + 1 if account.realized_pnl < last_realized else 0
                last_realized = account.realized_pnl
                if consecutive >= self.config.risk.max_consecutive_losses:
                    cooldown = self.config.risk.cooldown_after_losses
            if cooldown == 0 and consecutive >= self.config.risk.max_consecutive_losses:
                consecutive = 0
            day_start = account.equity if session != last_session else risk_state.day_start_equity
            week_start = account.equity if week != last_week else risk_state.week_start_equity
            risk_state = RiskState(
                day_start_equity=day_start,
                week_start_equity=week_start,
                peak_equity=max(risk_state.peak_equity, account.equity),
                open_positions=1 if self.broker.position else 0,
                correlated_positions=1 if self.broker.position else 0,
                consecutive_losses=consecutive,
                cooldown_bars=cooldown,
            )
            last_session = session
            last_week = week
            self._activate_risk_kill_switch(account.equity, risk_state, candle)

            values = RegimeFeatures(
                adx=float(row["adx"]),
                atr_percentile=float(row["atr_percentile"]),
                ema_slope=float(row["ema_slope"]),
                atr_change=float(row["atr_change"]),
                spread_bps=float(spread_bps),
                missing_ratio=0.0,
                cumulative_move=float(row["cumulative_move"]),
            )
            classified = self.classifier.update(
                values,
                strategy_state.regime,
                strategy_state.pending_regime,
                strategy_state.pending_regime_count,
            )
            previous_z = strategy_state.last_z
            strategy_state = StrategyState(
                regime=classified.regime,
                pending_regime=classified.pending,
                pending_regime_count=classified.pending_count,
                cooldown_bars=cooldown,
                last_z=previous_z,
            )
            activity = "Indicators warming up — no trading decision"
            usable = self._usable(row)
            trading_enabled = trade_after is None or candle.exchange_timestamp > trade_after
            if usable and not trading_enabled:
                activity = "Historical warm-up — paper orders disabled for this candle"
            if usable and trading_enabled:
                snapshot = MarketSnapshot(
                    candle=candle,
                    atr=Decimal(str(row["atr"])),
                    center=Decimal(str(row["center"])),
                    z_score=float(row["z"]),
                    spread_bps=float(spread_bps),
                    regime=classified.regime,
                    session_open=opens.iloc[position_index],
                    session_close=closes.iloc[position_index],
                    data_reliable=report.passed,
                )
                signal = self.strategy.evaluate(snapshot, strategy_state, self.broker.position)
                if signal is not None:
                    signal_count += 1
                    accepted, detail = await self._handle_signal(signal, risk_state, spread_bps)
                    rejected += int(not accepted)
                    action = signal.action.value.replace("_", " ").title()
                    activity = f"{action}: {signal.reason}. {detail}"
                else:
                    activity = self._waiting_reason(snapshot, risk_state)
            equity_curve.append(
                EquityPoint(timestamp=candle.exchange_timestamp, equity=account.equity)
            )
            atr_value = self._finite(row["atr"])
            center_value = self._decimal_or_none(row["center"])
            band_distance = (
                Decimal(str(self.config.strategy.range_multiplier)) * Decimal(str(atr_value))
                if atr_value is not None and center_value is not None
                else None
            )
            lower_band = (
                center_value - band_distance
                if center_value is not None and band_distance is not None
                else None
            )
            upper_band = (
                center_value + band_distance
                if center_value is not None and band_distance is not None
                else None
            )
            telemetry.append(
                TelemetryPoint(
                    timestamp=candle.exchange_timestamp,
                    close=candle.close,
                    center=center_value,
                    lower_band=lower_band,
                    upper_band=upper_band,
                    atr=atr_value,
                    adx=self._finite(row["adx"]),
                    z_score=self._finite(row["z"]),
                    atr_percentile=self._finite(row["atr_percentile"]),
                    ema_slope=self._finite(row["ema_slope"]),
                    spread_bps=float(spread_bps),
                    regime=classified.regime.value,
                    equity=account.equity,
                    position_quantity=(
                        self.broker.position.quantity
                        if self.broker.position is not None
                        else Decimal("0")
                    ),
                    activity=activity,
                )
            )
            strategy_state = strategy_state.model_copy(
                update={"last_z": None if pd.isna(row["z"]) else float(row["z"])}
            )

        final_account = await self.broker.get_account()
        fees = sum((fill.commission for fill in self.broker.fills), Decimal("0"))
        slippage = sum((fill.slippage for fill in self.broker.fills), Decimal("0"))
        net = final_account.equity - initial
        return BacktestResult(
            mode=mode,
            instrument=self.config.instrument.symbol,
            timeframe_minutes=self.config.strategy.timeframe_minutes,
            risk_per_trade=self.config.risk.risk_per_trade,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_weekly_loss=self.config.risk.max_weekly_loss,
            initial_equity=initial,
            final_equity=final_account.equity,
            gross_pnl=net + fees + slippage,
            net_pnl=net,
            fees=fees,
            slippage=slippage,
            max_drawdown=maximum_drawdown([point.equity for point in equity_curve]),
            fills=tuple(self.broker.fills),
            equity_curve=tuple(equity_curve),
            telemetry=tuple(telemetry),
            signals=signal_count,
            rejected_signals=rejected,
            kill_switches=int(self.kill_switch.active),
        )

    def _prepare(
        self, frame: pd.DataFrame, report: ValidationReport
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        del report
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        for column in ("open", "high", "low", "close", "volume"):
            data[column] = pd.to_numeric(data[column])
        if self.config.instrument.asset_class is AssetClass.CRYPTO:
            timestamps = data["timestamp"]
            opens = timestamps.dt.floor("1D")
            return (
                data,
                timestamps.dt.date.astype(str),
                opens,
                opens + timedelta(days=1),
            )
        calendar = mcal.get_calendar("NYSE")
        schedule = calendar.schedule(
            start_date=data["timestamp"].iloc[0].date(),
            end_date=data["timestamp"].iloc[-1].date(),
        )
        session_labels: list[str] = []
        session_opens: list[datetime] = []
        session_closes: list[datetime] = []
        for timestamp in data["timestamp"]:
            matches = schedule[
                (schedule["market_open"] < timestamp) & (schedule["market_close"] >= timestamp)
            ]
            if matches.empty:
                raise ValueError(f"candle outside regular session: {timestamp}")
            label = matches.index[0]
            session_labels.append(str(label.date()))
            session_opens.append(matches.iloc[0]["market_open"].to_pydatetime())
            session_closes.append(matches.iloc[0]["market_close"].to_pydatetime())
        index = data.index
        return (
            data,
            pd.Series(session_labels, index=index),
            pd.Series(session_opens, index=index),
            pd.Series(session_closes, index=index),
        )

    def _candle(self, row: pd.Series) -> Candle:
        timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()
        return Candle(
            exchange_timestamp=timestamp,
            received_timestamp=timestamp,
            source="backtest",
            instrument=self.config.instrument.symbol,
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=Decimal(str(row["volume"])),
            timeframe_minutes=self.config.strategy.timeframe_minutes,
        )

    @staticmethod
    def _usable(row: pd.Series) -> bool:
        return not row[["atr", "adx", "center", "z", "atr_percentile", "ema_slope"]].isna().any()

    def _spread_bps(self, row: pd.Series) -> Decimal:
        value = row.get("spread_bps", self.config.backtest.spread_bps)
        spread = self.config.backtest.spread_bps if pd.isna(value) else Decimal(str(value))
        if not spread.is_finite() or spread < 0:
            raise ValueError("spread_bps must be finite and non-negative")
        return spread

    async def _handle_signal(
        self, signal: Signal, risk_state: RiskState, spread_bps: Decimal
    ) -> tuple[bool, str]:
        if signal.action in {SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT}:
            account = await self.broker.get_account()
            costs = estimated_round_trip_cost_per_unit(
                signal.reference_price,
                spread_bps,
                self.config.backtest.slippage_bps,
                self.config.backtest.commission_per_unit,
            )
            decision = self.risk_engine.assess(
                signal, self.config.instrument, account, risk_state, costs
            )
            if not decision.approved:
                return False, f"Risk rejected: {decision.reason}"
            side = Side.BUY if signal.action is SignalAction.ENTER_LONG else Side.SELL
            await self._submit(signal, "entry", side, OrderType.MARKET, decision.quantity)
            opposite = Side.SELL if side is Side.BUY else Side.BUY
            assert signal.stop_price is not None and signal.target_price is not None
            await self._submit(
                signal,
                "stop",
                opposite,
                OrderType.STOP,
                decision.quantity,
                stop_price=signal.stop_price,
                reduce_only=True,
                protective=True,
            )
            await self._submit(
                signal,
                "target",
                opposite,
                OrderType.LIMIT,
                decision.quantity,
                limit_price=signal.target_price,
                reduce_only=True,
            )
            return True, f"Risk approved · quantity {decision.quantity}"

        position = self.broker.position
        if position is None:
            return False, "No position is available to close"
        quantity = position.quantity
        if signal.action is SignalAction.REDUCE:
            quantity = floor_to_lot(quantity / 2, self.config.instrument.lot_size)
        if quantity <= 0:
            return False, "Rounded exit quantity is zero"
        side = Side.SELL if position.side is Side.BUY else Side.BUY
        await self._submit(signal, "exit", side, OrderType.MARKET, quantity, reduce_only=True)
        return True, f"Reduce-only order submitted · quantity {quantity}"

    def _waiting_reason(self, snapshot: MarketSnapshot, state: RiskState) -> str:
        if self.kill_switch.active:
            return "Kill switch active — new entries blocked"
        if state.cooldown_bars > 0:
            return f"Cooldown active — {state.cooldown_bars} bars remaining"
        if snapshot.regime.value != "range":
            return f"No entry — market regime is {snapshot.regime.value.replace('_', ' ')}"
        if abs(snapshot.z_score) < self.config.strategy.entry_z:
            return (
                f"No entry — |z| {abs(snapshot.z_score):.2f} is below "
                f"{self.config.strategy.entry_z:.2f}"
            )
        return "No order — position, session or direction constraints are not satisfied"

    @staticmethod
    def _finite(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    @classmethod
    def _decimal_or_none(cls, value: object) -> Decimal | None:
        number = cls._finite(value)
        return Decimal(str(number)) if number is not None else None

    async def _submit(
        self,
        signal: Signal,
        purpose: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        reduce_only: bool = False,
        protective: bool = False,
    ) -> None:
        request = OrderRequest(
            exchange_timestamp=signal.exchange_timestamp,
            received_timestamp=signal.received_timestamp,
            source="backtest",
            instrument=signal.instrument,
            sequence_number=signal.sequence_number,
            correlation_id=signal.correlation_id,
            client_order_id=client_order_id(signal, purpose),
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            protective=protective,
        )
        await self.order_manager.submit(request)

    async def _cancel_orphaned_brackets(self) -> None:
        active_entry = any(
            not order.reduce_only
            and order.status in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}
            for order in self.broker.orders.values()
        )
        if self.broker.position is not None or active_entry:
            return
        for order in tuple(self.broker.orders.values()):
            if order.reduce_only and order.status in {
                OrderStatus.ACKNOWLEDGED,
                OrderStatus.PARTIALLY_FILLED,
            }:
                await self.broker.cancel_order(order.client_order_id)

    def _activate_risk_kill_switch(self, equity: Decimal, state: RiskState, candle: Candle) -> None:
        if equity <= state.day_start_equity * (Decimal("1") - self.config.risk.max_daily_loss):
            self.kill_switch.trigger(
                KillSwitchCause.DAILY_LOSS,
                candle.exchange_timestamp,
                "daily loss limit reached",
            )
        if equity <= state.week_start_equity * (Decimal("1") - self.config.risk.max_weekly_loss):
            self.kill_switch.trigger(
                KillSwitchCause.WEEKLY_LOSS,
                candle.exchange_timestamp,
                "weekly loss limit reached",
            )
        if (
            state.peak_equity > 0
            and (state.peak_equity - equity) / state.peak_equity
            >= self.config.risk.max_strategy_drawdown
        ):
            self.kill_switch.trigger(
                KillSwitchCause.DRAWDOWN,
                candle.exchange_timestamp,
                "strategy drawdown limit reached",
                close_position=True,
            )
