from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from adaptive_bot.adapters.alpaca.trade_updates import AlpacaTradeUpdates
from adaptive_bot.backtest.costs import estimated_round_trip_cost_per_unit
from adaptive_bot.backtest.engine import EquityPoint, ResultModel, TelemetryPoint
from adaptive_bot.clock import Clock, SystemClock
from adaptive_bot.config import AppConfig
from adaptive_bot.data.interfaces import MarketDataProvider
from adaptive_bot.data.repository import ParquetRepository, SQLiteStateStore
from adaptive_bot.data.validation import validate_candles
from adaptive_bot.domain.enums import KillSwitchCause, MarketRegime, OrderType, Side, SignalAction
from adaptive_bot.domain.models import (
    AccountSnapshot,
    Candle,
    Fill,
    OrderRequest,
    Position,
    Quote,
    Signal,
    StrategyState,
)
from adaptive_bot.execution.idempotency import client_order_id
from adaptive_bot.execution.interfaces import PaperBroker
from adaptive_bot.risk.engine import DefaultRiskEngine, RiskState
from adaptive_bot.risk.kill_switch import KillSwitch
from adaptive_bot.risk.limits import drawdown, loss_limit_breached
from adaptive_bot.services.recovery_service import reconcile_before_trading
from adaptive_bot.services.trading_service import CandleAggregator, append_candle
from adaptive_bot.strategy.adaptive_range import AdaptiveRangeStrategy, build_features
from adaptive_bot.strategy.regime import RegimeClassifier, RegimeFeatures
from adaptive_bot.strategy.signals import MarketSnapshot


class PaperReport(ResultModel):
    mode: str = "paper"
    instrument: str
    timeframe_minutes: int
    risk_per_trade: Decimal
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
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class PaperRuntime:
    def __init__(
        self,
        config: AppConfig,
        broker: PaperBroker,
        input_path: str | Path,
        output_path: str | Path,
        *,
        clock: Clock | None = None,
        state_store: SQLiteStateStore | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.output_path = Path(output_path)
        self.clock = clock or SystemClock()
        self.state_store = state_store
        self.frame = ParquetRepository.read(input_path)
        self.frame.attrs["split_adjusted"] = True
        self.kill_switch = KillSwitch()
        self.risk_engine = DefaultRiskEngine(config.risk, self.kill_switch)
        self.strategy = AdaptiveRangeStrategy(config.strategy)
        self.classifier = RegimeClassifier(config.strategy)
        self.strategy_state = StrategyState()
        self.risk_state: RiskState | None = None
        self.initial_equity = Decimal("0")
        self.last_quote: Quote | None = None
        self.telemetry: list[TelemetryPoint] = []
        self.equity_curve: list[EquityPoint] = []
        self.fills: list[Fill] = []
        self.signal_count = 0
        self.rejected_count = 0
        self._bars_held = 0
        self._last_session: str | None = None
        self._last_week: tuple[int, int] | None = None

    async def initialize(self) -> None:
        account = await self.broker.verify_account()
        if self.state_store is not None:
            await asyncio.to_thread(self.state_store.initialize)
            latched = await self.state_store.get("paper_kill_switch")
            if latched not in {None, "null"}:
                raise RuntimeError("paper kill switch is latched; manual reset is required")
        local_position = await self._load_position()
        recovery = await reconcile_before_trading(
            self.broker,
            self.config.instrument.symbol,
            local_position,
            self.kill_switch,
            self.clock.now(),
        )
        if not recovery.reconciled:
            await self.persist_kill_switch()
            raise RuntimeError(f"paper startup reconciliation failed: {recovery.reason}")
        open_orders = await self.broker.get_open_orders(self.config.instrument.symbol)
        if local_position is None and open_orders:
            self.kill_switch.trigger(
                KillSwitchCause.STATE_DIVERGENCE,
                self.clock.now(),
                "open orders exist without restored local state",
            )
            await self.persist_kill_switch()
            raise RuntimeError("paper startup found unknown open orders")
        if local_position is not None and not await self._protective_stop_present(local_position):
            self.kill_switch.trigger(
                KillSwitchCause.MISSING_PROTECTIVE_STOP,
                self.clock.now(),
                "restored paper position has no protective stop",
                close_position=True,
            )
            await self.broker.flatten_all()
            await self.persist_kill_switch()
            raise RuntimeError("paper startup flattened an unprotected position")
        self.initial_equity = account.equity
        self.risk_state = await self._load_risk_state(account.equity)

    def on_quote(self, quote: Quote) -> None:
        if quote.instrument == self.config.instrument.symbol:
            self.last_quote = quote

    async def on_fill(self, fill: Fill) -> None:
        if fill.instrument == self.config.instrument.symbol:
            self.fills.append(fill)
            account = await self.broker.get_account()
            positions = await self.broker.get_positions()
            await self._save_position(self._position(positions))
            self._write_report(account.equity)

    async def process(self, candle: Candle) -> None:
        if self.risk_state is None:
            raise RuntimeError("paper runtime is not initialized")
        self.frame = append_candle(self.frame, candle)
        report = validate_candles(
            self.frame,
            timeframe_minutes=self.config.strategy.timeframe_minutes,
        )
        report.require(self.config.backtest.minimum_quality_score)
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        try:
            position = self._position(positions)
        except RuntimeError:
            await self.persist_kill_switch()
            raise
        await self._save_position(position)
        session, session_open, session_close = self._session(candle.exchange_timestamp)
        self._update_risk_state(account.equity, position, session, candle.exchange_timestamp)
        self._activate_loss_switch(account.equity, candle.exchange_timestamp)
        if position is not None and not await self._protective_stop_present(position):
            self.kill_switch.trigger(
                KillSwitchCause.MISSING_PROTECTIVE_STOP,
                candle.exchange_timestamp,
                "Alpaca paper position has no protective stop",
                close_position=True,
            )
        event = self.kill_switch.event
        if position is not None and event is not None and event.close_position:
            await self.broker.flatten_all()
            await self.persist_kill_switch()
            await self._save_risk_state()
            self._record(
                candle,
                account.equity,
                position,
                MarketRegime.UNKNOWN,
                pd.Series(dtype=float),
                f"Emergency paper flatten requested: {event.details}",
            )
            self._write_report(account.equity)
            return

        row = self._latest_features()
        if not self._usable(row):
            self._record(
                candle, account.equity, position, MarketRegime.UNKNOWN, row, "Indicators warming up"
            )
            await self._save_risk_state()
            await self.persist_kill_switch()
            self._write_report(account.equity)
            return

        spread, reliable = self._market_quality()
        values = RegimeFeatures(
            adx=float(row["adx"]),
            atr_percentile=float(row["atr_percentile"]),
            ema_slope=float(row["ema_slope"]),
            atr_change=float(row["atr_change"]),
            spread_bps=spread,
            missing_ratio=0.0 if reliable else 1.0,
            cumulative_move=float(row["cumulative_move"]),
        )
        classified = self.classifier.update(
            values,
            self.strategy_state.regime,
            self.strategy_state.pending_regime,
            self.strategy_state.pending_regime_count,
        )
        self.strategy_state = self.strategy_state.model_copy(
            update={
                "regime": classified.regime,
                "pending_regime": classified.pending,
                "pending_regime_count": classified.pending_count,
            }
        )
        snapshot = MarketSnapshot(
            candle=candle,
            atr=Decimal(str(row["atr"])),
            center=Decimal(str(row["center"])),
            z_score=float(row["z"]),
            spread_bps=spread,
            regime=classified.regime,
            session_open=session_open,
            session_close=session_close,
            data_reliable=reliable,
            force_exit_reason="risk kill switch" if self.kill_switch.active and position else None,
        )
        signal = self.strategy.evaluate(snapshot, self.strategy_state, position)
        activity = self._waiting_reason(snapshot)
        if signal is not None:
            self.signal_count += 1
            if signal.action in {SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT}:
                activity = await self._enter(signal, account)
            else:
                await self.broker.flatten_all()
                activity = (
                    f"{signal.action.value}: {signal.reason}. Paper position flatten requested"
                )
        self.strategy_state = self.strategy_state.model_copy(update={"last_z": snapshot.z_score})
        self._record(candle, account.equity, position, classified.regime, row, activity, spread)
        await self._save_risk_state()
        await self.persist_kill_switch()
        self._write_report(account.equity)

    async def _enter(self, signal: Signal, account: AccountSnapshot) -> str:
        if await self.broker.get_open_orders(signal.instrument):
            self.rejected_count += 1
            return "Risk rejected: an Alpaca order is already open"
        stop = self._floor_tick(signal.stop_price)
        target = self._floor_tick(signal.target_price)
        if stop is None or target is None:
            self.rejected_count += 1
            return "Risk rejected: bracket prices are unavailable"
        rounded = signal.model_copy(update={"stop_price": stop, "target_price": target})
        costs = estimated_round_trip_cost_per_unit(
            signal.reference_price,
            self.config.backtest.spread_bps,
            self.config.backtest.slippage_bps,
            self.config.backtest.commission_per_unit,
        )
        assert self.risk_state is not None
        decision = self.risk_engine.assess(
            rounded,
            self.config.instrument,
            account,
            self.risk_state,
            costs,
        )
        if not decision.approved:
            self.rejected_count += 1
            return f"Risk rejected: {decision.reason}"
        request = OrderRequest(
            exchange_timestamp=signal.exchange_timestamp,
            received_timestamp=signal.received_timestamp,
            source="paper",
            instrument=signal.instrument,
            sequence_number=signal.sequence_number,
            correlation_id=signal.correlation_id,
            client_order_id=client_order_id(signal, "paper-bracket"),
            side=Side.BUY if signal.action is SignalAction.ENTER_LONG else Side.SELL,
            order_type=OrderType.MARKET,
            quantity=decision.quantity,
            stop_price=stop,
        )
        await self.broker.submit_bracket_entry(request, target)
        return (
            f"Paper bracket submitted · quantity {decision.quantity} · "
            f"stop {stop} · target {target}"
        )

    def _latest_features(self) -> pd.Series:
        timestamps = pd.to_datetime(self.frame["timestamp"], utc=True)
        sessions = timestamps.dt.tz_convert("America/New_York").dt.date.astype(str)
        return build_features(
            self.frame,
            sessions,
            self.config.strategy,
            self.config.instrument.asset_class,
        ).iloc[-1]

    def _session(self, timestamp: datetime) -> tuple[str, datetime, datetime]:
        calendar = mcal.get_calendar("NYSE")
        local_date = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        schedule = calendar.schedule(local_date, local_date)
        if schedule.empty:
            raise ValueError("paper candle is outside an NYSE trading session")
        row = schedule.iloc[0]
        return (
            str(local_date),
            row["market_open"].to_pydatetime(),
            row["market_close"].to_pydatetime(),
        )

    def _market_quality(self) -> tuple[float, bool]:
        quote = self.last_quote
        if quote is None:
            self.kill_switch.trigger(
                KillSwitchCause.STALE_DATA,
                self.clock.now(),
                "no current Alpaca quote",
            )
            return math.inf, False
        age = max(0.0, (self.clock.now() - quote.received_timestamp).total_seconds())
        mid = (quote.bid + quote.ask) / 2
        spread = float((quote.ask - quote.bid) / mid * Decimal("10000"))
        alpaca = self.config.alpaca
        if alpaca is None:
            raise RuntimeError("Alpaca configuration is required")
        if age > alpaca.stale_after_seconds:
            self.kill_switch.trigger(
                KillSwitchCause.STALE_DATA, self.clock.now(), "Alpaca quote is stale"
            )
            return spread, False
        if spread > self.config.strategy.max_spread_bps:
            self.kill_switch.trigger(
                KillSwitchCause.EXTREME_SPREAD, self.clock.now(), "Alpaca spread is too wide"
            )
            return spread, False
        return spread, True

    def _update_risk_state(
        self, equity: Decimal, position: Position | None, session: str, timestamp: datetime
    ) -> None:
        assert self.risk_state is not None
        week = timestamp.isocalendar()[:2]
        self._bars_held = self._bars_held + 1 if position else 0
        self.risk_state = RiskState(
            day_start_equity=(
                equity if session != self._last_session else self.risk_state.day_start_equity
            ),
            week_start_equity=(
                equity if week != self._last_week else self.risk_state.week_start_equity
            ),
            peak_equity=max(equity, self.risk_state.peak_equity),
            open_positions=int(position is not None),
            correlated_positions=int(position is not None),
        )
        self._last_session = session
        self._last_week = week

    def _activate_loss_switch(self, equity: Decimal, timestamp: datetime) -> None:
        assert self.risk_state is not None
        if loss_limit_breached(
            equity, self.risk_state.day_start_equity, self.config.risk.max_daily_loss
        ):
            self.kill_switch.trigger(
                KillSwitchCause.DAILY_LOSS, timestamp, "daily loss limit", close_position=True
            )
        elif (
            drawdown(equity, self.risk_state.peak_equity) >= self.config.risk.max_strategy_drawdown
        ):
            self.kill_switch.trigger(
                KillSwitchCause.DRAWDOWN, timestamp, "drawdown limit", close_position=True
            )

    def _position(self, positions: tuple[Position, ...]) -> Position | None:
        if len(positions) > 1 or any(
            position.instrument != self.config.instrument.symbol for position in positions
        ):
            self.kill_switch.trigger(
                KillSwitchCause.UNKNOWN_POSITION,
                self.clock.now(),
                "unexpected Alpaca paper position",
            )
            raise RuntimeError("paper account position is not reconcilable")
        if not positions:
            return None
        return positions[0].model_copy(update={"bars_held": self._bars_held})

    async def _protective_stop_present(self, position: Position) -> bool:
        orders = await self.broker.get_open_orders(position.instrument)
        closing_side = Side.SELL if position.side is Side.BUY else Side.BUY
        return any(
            order.protective
            and order.side is closing_side
            and order.quantity - order.filled_quantity >= position.quantity
            for order in orders
        )

    async def _load_position(self) -> Position | None:
        if self.state_store is None:
            return None
        payload = await self.state_store.get("paper_position")
        return None if payload in {None, "null"} else Position.model_validate_json(payload)

    async def _save_position(self, position: Position | None) -> None:
        if self.state_store is not None:
            await self.state_store.set(
                "paper_position",
                "null" if position is None else position.model_dump_json(),
            )

    async def _load_risk_state(self, equity: Decimal) -> RiskState:
        if self.state_store is None:
            return RiskState(equity, equity, equity)
        payload = await self.state_store.get("paper_risk_state")
        if payload in {None, "null"}:
            return RiskState(equity, equity, equity)
        stored = json.loads(payload)
        now = self.clock.now()
        day = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        week = list(now.isocalendar()[:2])
        return RiskState(
            day_start_equity=(
                Decimal(stored["day_start_equity"]) if stored.get("day") == day else equity
            ),
            week_start_equity=(
                Decimal(stored["week_start_equity"]) if stored.get("week") == week else equity
            ),
            peak_equity=max(equity, Decimal(stored["peak_equity"])),
        )

    async def _save_risk_state(self) -> None:
        if self.state_store is None or self.risk_state is None:
            return
        now = self.clock.now()
        payload = {
            "day": now.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
            "week": list(now.isocalendar()[:2]),
            "day_start_equity": str(self.risk_state.day_start_equity),
            "week_start_equity": str(self.risk_state.week_start_equity),
            "peak_equity": str(self.risk_state.peak_equity),
        }
        await self.state_store.set(
            "paper_risk_state",
            json.dumps(payload, separators=(",", ":")),
        )

    async def persist_kill_switch(self) -> None:
        if self.state_store is None or self.kill_switch.event is None:
            return
        event = self.kill_switch.event
        await self.state_store.set(
            "paper_kill_switch",
            json.dumps(
                {
                    "cause": event.cause.value,
                    "timestamp": event.timestamp.isoformat(),
                    "details": event.details,
                    "close_position": event.close_position,
                },
                separators=(",", ":"),
            ),
        )

    def _record(
        self,
        candle: Candle,
        equity: Decimal,
        position: Position | None,
        regime: MarketRegime,
        row: pd.Series,
        activity: str,
        spread: float = 0.0,
    ) -> None:
        atr = self._finite(row.get("atr"))
        center = self._decimal(row.get("center"))
        distance = (
            Decimal(str(self.config.strategy.range_multiplier)) * Decimal(str(atr))
            if atr is not None and center is not None
            else None
        )
        self.equity_curve.append(EquityPoint(timestamp=candle.exchange_timestamp, equity=equity))
        self.telemetry.append(
            TelemetryPoint(
                timestamp=candle.exchange_timestamp,
                close=candle.close,
                center=center,
                lower_band=center - distance
                if center is not None and distance is not None
                else None,
                upper_band=center + distance
                if center is not None and distance is not None
                else None,
                atr=atr,
                adx=self._finite(row.get("adx")),
                z_score=self._finite(row.get("z")),
                atr_percentile=self._finite(row.get("atr_percentile")),
                ema_slope=self._finite(row.get("ema_slope")),
                spread_bps=spread,
                regime=regime.value,
                equity=equity,
                position_quantity=position.quantity if position else Decimal("0"),
                activity=activity,
            )
        )

    def _write_report(self, equity: Decimal) -> None:
        fees = sum((fill.commission for fill in self.fills), Decimal("0"))
        slippage = sum((fill.slippage for fill in self.fills), Decimal("0"))
        net = equity - self.initial_equity
        peak = self.initial_equity
        maximum = Decimal("0")
        for point in self.equity_curve:
            peak = max(peak, point.equity)
            maximum = max(maximum, drawdown(point.equity, peak))
        PaperReport(
            instrument=self.config.instrument.symbol,
            timeframe_minutes=self.config.strategy.timeframe_minutes,
            risk_per_trade=self.config.risk.risk_per_trade,
            initial_equity=self.initial_equity,
            final_equity=equity,
            gross_pnl=net + fees + slippage,
            net_pnl=net,
            fees=fees,
            slippage=slippage,
            max_drawdown=maximum,
            fills=tuple(self.fills),
            equity_curve=tuple(self.equity_curve),
            telemetry=tuple(self.telemetry),
            signals=self.signal_count,
            rejected_signals=self.rejected_count,
            kill_switches=int(self.kill_switch.active),
        ).write_json(self.output_path)

    def _floor_tick(self, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        tick = self.config.instrument.tick_size
        return (value / tick).to_integral_value(rounding=ROUND_FLOOR) * tick

    def _waiting_reason(self, snapshot: MarketSnapshot) -> str:
        if self.kill_switch.active:
            return "Kill switch active — new paper orders blocked"
        if snapshot.regime is not MarketRegime.RANGE:
            return f"No paper entry — market regime is {snapshot.regime.value}"
        return f"No paper entry — |z| {abs(snapshot.z_score):.2f} is below threshold"

    @staticmethod
    def _usable(row: pd.Series) -> bool:
        return not row[["atr", "adx", "center", "z", "atr_percentile", "ema_slope"]].isna().any()

    @staticmethod
    def _finite(value: object) -> float | None:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _decimal(cls, value: object) -> Decimal | None:
        number = cls._finite(value)
        return Decimal(str(number)) if number is not None else None


async def run_paper(
    runtime: PaperRuntime,
    market_data: MarketDataProvider,
    trade_updates: AlpacaTradeUpdates,
) -> None:
    await runtime.initialize()
    aggregator = CandleAggregator(runtime.config.strategy.timeframe_minutes)
    queue: asyncio.Queue[object] = asyncio.Queue()

    async def pump_market() -> None:
        try:
            async for event in market_data.stream(runtime.config.instrument.symbol):
                await queue.put(event)
            await queue.put(ConnectionError("Alpaca market-data stream ended"))
        except Exception as error:
            await queue.put(error)

    async def pump_orders() -> None:
        try:
            async for event in trade_updates.stream():
                await queue.put(event)
            await queue.put(ConnectionError("Alpaca trading stream ended"))
        except Exception as error:
            await queue.put(error)

    tasks = [asyncio.create_task(pump_market()), asyncio.create_task(pump_orders())]
    try:
        while True:
            event = await queue.get()
            if isinstance(event, BaseException):
                runtime.kill_switch.trigger(
                    KillSwitchCause.WEBSOCKET_DISCONNECTED,
                    runtime.clock.now(),
                    str(event),
                )
                await runtime.persist_kill_switch()
                raise event
            if isinstance(event, Quote):
                runtime.on_quote(event)
            elif isinstance(event, Fill):
                await runtime.on_fill(event)
            elif isinstance(event, Candle):
                completed = aggregator.add(event)
                if completed is not None:
                    await runtime.process(completed)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
