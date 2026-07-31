from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
from pydantic import BaseModel, ConfigDict

from adaptive_bot.adapters.simulated.broker import SimulatedBroker
from adaptive_bot.backtest.costs import estimated_round_trip_cost_per_unit
from adaptive_bot.backtest.metrics import maximum_drawdown
from adaptive_bot.config import AppConfig
from adaptive_bot.data.validation import ValidationReport, validate_candles
from adaptive_bot.domain.enums import (
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


class BacktestResult(ResultModel):
    initial_equity: Decimal
    final_equity: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    fees: Decimal
    slippage: Decimal
    max_drawdown: Decimal
    fills: tuple[Fill, ...]
    equity_curve: tuple[EquityPoint, ...]
    signals: int
    rejected_signals: int
    kill_switches: int

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")


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

    async def run(self, frame: pd.DataFrame) -> BacktestResult:
        report = validate_candles(frame, timeframe_minutes=self.config.strategy.timeframe_minutes)
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
        signal_count = 0
        rejected = 0
        last_session: str | None = None
        last_week: tuple[int, int] | None = None
        last_realized = Decimal("0")

        for position_index, (_, row) in enumerate(features.iterrows()):
            candle = self._candle(row)
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
                spread_bps=float(self.config.backtest.spread_bps),
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
            if self._usable(row):
                snapshot = MarketSnapshot(
                    candle=candle,
                    atr=Decimal(str(row["atr"])),
                    center=Decimal(str(row["center"])),
                    z_score=float(row["z"]),
                    spread_bps=float(self.config.backtest.spread_bps),
                    regime=classified.regime,
                    session_open=opens.iloc[position_index],
                    session_close=closes.iloc[position_index],
                    data_reliable=report.passed,
                )
                signal = self.strategy.evaluate(snapshot, strategy_state, self.broker.position)
                if signal is not None:
                    signal_count += 1
                    accepted = await self._handle_signal(signal, risk_state)
                    rejected += int(not accepted)
            equity_curve.append(
                EquityPoint(timestamp=candle.exchange_timestamp, equity=account.equity)
            )
            strategy_state = strategy_state.model_copy(
                update={"last_z": None if pd.isna(row["z"]) else float(row["z"])}
            )

        final_account = await self.broker.get_account()
        fees = sum((fill.commission for fill in self.broker.fills), Decimal("0"))
        slippage = sum((fill.slippage for fill in self.broker.fills), Decimal("0"))
        net = final_account.equity - initial
        return BacktestResult(
            initial_equity=initial,
            final_equity=final_account.equity,
            gross_pnl=net + fees + slippage,
            net_pnl=net,
            fees=fees,
            slippage=slippage,
            max_drawdown=maximum_drawdown([point.equity for point in equity_curve]),
            fills=tuple(self.broker.fills),
            equity_curve=tuple(equity_curve),
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

    async def _handle_signal(self, signal: Signal, risk_state: RiskState) -> bool:
        if signal.action in {SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT}:
            account = await self.broker.get_account()
            costs = estimated_round_trip_cost_per_unit(
                signal.reference_price,
                self.config.backtest.spread_bps,
                self.config.backtest.slippage_bps,
                self.config.backtest.commission_per_unit,
            )
            decision = self.risk_engine.assess(
                signal, self.config.instrument, account, risk_state, costs
            )
            if not decision.approved:
                return False
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
            return True

        position = self.broker.position
        if position is None:
            return False
        quantity = position.quantity
        if signal.action is SignalAction.REDUCE:
            quantity = floor_to_lot(quantity / 2, self.config.instrument.lot_size)
        if quantity <= 0:
            return False
        side = Side.SELL if position.side is Side.BUY else Side.BUY
        await self._submit(signal, "exit", side, OrderType.MARKET, quantity, reduce_only=True)
        return True

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
                close_position=True,
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
