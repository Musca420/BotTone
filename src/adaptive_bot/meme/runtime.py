from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import pandas as pd

from adaptive_bot.domain.enums import Side
from adaptive_bot.domain.models import Position
from adaptive_bot.meme.config import MemeBotConfig
from adaptive_bot.meme.strategy import (
    MemeDecision,
    MemeMomentumStrategy,
    MemeStrategyState,
    build_meme_features,
)
from adaptive_bot.meme.universe import (
    MarketQuality,
    MemeContract,
    choose_leverage,
    rank_candidates,
)


@dataclass(frozen=True)
class PaperPosition:
    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    leverage: int
    opened_at: datetime
    bars_held: int = 0
    one_r_hit: bool = False
    best_price: Decimal | None = None


@dataclass(frozen=True)
class PendingEntry:
    decision: MemeDecision
    created_at: datetime


@dataclass(frozen=True)
class Sizing:
    quantity: Decimal
    leverage: int
    effective_risk: Decimal
    risk_budget: Decimal


class MemePaperEngine:
    def __init__(self, config: MemeBotConfig, contracts: tuple[MemeContract, ...]) -> None:
        self.config = config
        self.contracts = {contract.symbol: contract for contract in contracts}
        self.strategy = MemeMomentumStrategy(config.strategy)

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        qualities: dict[str, MarketQuality],
        *,
        trade_after: datetime | None = None,
        mode: str = "backtest",
    ) -> dict[str, object]:
        features = {
            symbol: build_meme_features(frame, self.config.strategy).set_index("timestamp")
            for symbol, frame in frames.items()
            if symbol in self.contracts and not frame.empty
        }
        timestamps = sorted({timestamp for frame in features.values() for timestamp in frame.index})
        if trade_after is not None:
            state_warmup = trade_after - timedelta(
                minutes=self.config.strategy.timeframe_minutes
                * (self.config.strategy.retest_bars + 1)
            )
            timestamps = [
                timestamp for timestamp in timestamps if pd.Timestamp(timestamp) >= state_warmup
            ]
        states = {symbol: MemeStrategyState() for symbol in features}
        equity = self.config.initial_equity
        peak = equity
        realized = Decimal("0")
        fees = Decimal("0")
        day_start = equity
        week_start = equity
        current_day = None
        current_week = None
        position: PaperPosition | None = None
        pending: PendingEntry | None = None
        cooldown = 0
        consecutive_losses = 0
        operations: list[dict[str, object]] = []
        audit: deque[dict[str, object]] = deque(maxlen=300)
        equity_curve: list[dict[str, str]] = []
        latest_rows: dict[str, pd.Series] = {}
        active_contracts = tuple(
            contract for symbol, contract in self.contracts.items() if symbol in features
        )
        scanner = rank_candidates(
            active_contracts,
            qualities,
            self.config.universe,
            equity * self.config.risk.max_margin_fraction * self.config.risk.leverage_ceiling,
        )
        ranks = {item.contract.symbol: item.rank for item in scanner if item.eligible}

        for timestamp in timestamps:
            moment = pd.Timestamp(timestamp).to_pydatetime()
            rows: dict[str, pd.Series] = {}
            for symbol, frame in features.items():
                if timestamp not in frame.index:
                    continue
                row = frame.loc[timestamp].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1].copy()
                row["timestamp"] = timestamp
                rows[symbol] = row
            latest_rows.update(rows)
            day = moment.date()
            week = moment.isocalendar()[:2]
            if day != current_day:
                day_start, current_day = self._mark_equity(equity, position, rows), day
            if week != current_week:
                week_start, current_week = self._mark_equity(equity, position, rows), week
            cooldown = max(0, cooldown - 1)

            if (
                pending is not None
                and pending.decision.symbol in rows
                and moment > pending.created_at
            ):
                contract = self.contracts[pending.decision.symbol]
                fill = self._entry_fill(
                    pending.decision, rows[pending.decision.symbol], contract, equity
                )
                if fill is None:
                    audit.append(
                        self._audit(
                            moment, pending.decision.symbol, "REJECT", "risk_or_margin_rejected"
                        )
                    )
                else:
                    price, sizing = fill
                    assert (
                        pending.decision.stop_price is not None
                        and pending.decision.target_price is not None
                    )
                    entry_fee = self._cost(price * sizing.quantity) / 2
                    equity -= entry_fee
                    fees += entry_fee
                    position = PaperPosition(
                        symbol=pending.decision.symbol,
                        side=Side.BUY if pending.decision.action == "enter_long" else Side.SELL,
                        quantity=sizing.quantity,
                        entry_price=price,
                        stop_price=pending.decision.stop_price,
                        target_price=pending.decision.target_price,
                        leverage=sizing.leverage,
                        opened_at=moment,
                        best_price=price,
                    )
                    operations.append(
                        self._operation(
                            moment,
                            position.symbol,
                            "OPEN",
                            position.side,
                            sizing.quantity,
                            price,
                            equity,
                            sizing.leverage,
                            "breakout_retest_confirmed",
                        )
                    )
                pending = None

            if position is not None and position.symbol in rows and moment > position.opened_at:
                before = equity
                position, equity, new_operations = self._process_position(
                    position, rows[position.symbol], moment, equity
                )
                operations.extend(new_operations)
                delta = equity - before
                if position is None and delta != 0:
                    realized += delta
                    if delta < 0:
                        consecutive_losses += 1
                        if consecutive_losses >= self.config.risk.max_consecutive_losses:
                            cooldown = self.config.risk.cooldown_bars
                    else:
                        consecutive_losses = 0
                    states = {symbol: MemeStrategyState() for symbol in features}

            marked = self._mark_equity(equity, position, rows)
            peak = max(peak, marked)
            loss_block = (
                marked <= day_start * (Decimal("1") - self.config.risk.max_daily_loss)
                or marked <= week_start * (Decimal("1") - self.config.risk.max_weekly_loss)
                or marked <= peak * (Decimal("1") - self.config.risk.max_strategy_drawdown)
            )
            can_trade = trade_after is None or moment > trade_after
            entry_options: list[MemeDecision] = []
            for symbol, row in rows.items():
                domain_position = None
                if position is not None and position.symbol == symbol:
                    domain_position = self._domain_position(position)
                decision, states[symbol] = self.strategy.evaluate(
                    row,
                    symbol,
                    states[symbol],
                    domain_position,
                )
                if decision.action in {"watch", "reject"}:
                    audit.append(
                        self._audit(moment, symbol, decision.action.upper(), decision.reason)
                    )
                if decision.action in {"enter_long", "enter_short"}:
                    entry_options.append(decision)
            if (
                position is None
                and pending is None
                and can_trade
                and not loss_block
                and cooldown == 0
                and entry_options
            ):
                eligible = [decision for decision in entry_options if decision.symbol in ranks]
                if eligible:
                    chosen = min(eligible, key=lambda item: ranks[item.symbol] or 10**9)
                    pending = PendingEntry(chosen, moment)
                    audit.append(
                        self._audit(
                            moment, chosen.symbol, "ORDER_CREATED", "fill_not_before_next_candle"
                        )
                    )
            equity_curve.append({"timestamp": moment.isoformat(), "equity": str(marked)})

        final_equity = self._mark_equity(equity, position, latest_rows)
        return {
            "mode": mode,
            "generated_at": datetime.now(UTC).isoformat(),
            "initial_equity": str(self.config.initial_equity),
            "final_equity": str(final_equity),
            "net_pnl": str(final_equity - self.config.initial_equity),
            "realized_pnl": str(realized),
            "fees": str(fees),
            "risk": self.config.risk.model_dump(mode="json"),
            "position": None if position is None else _jsonable(asdict(position)),
            "pending_entry": None if pending is None else pending.decision.action,
            "operations": operations,
            "audit": list(audit),
            "equity_curve": equity_curve[-1000:],
            "scanner": [
                {
                    "symbol": item.contract.symbol,
                    "eligible": item.eligible,
                    "rank": item.rank,
                    "reasons": item.reasons,
                    "momentum_atr": str(item.quality.momentum_atr),
                    "volume_zscore": str(item.quality.volume_zscore),
                    "spread_bps": str(item.quality.spread_bps),
                    "depth": str(item.quality.depth_half_percent),
                }
                for item in scanner
            ],
            "probabilistic": {
                "status": "collecting_data",
                "can_trade": False,
                "p_win": None,
                "expected_value": None,
                "note": "Shadow models cannot affect orders before the validation gate.",
            },
        }

    def _entry_fill(
        self, decision: MemeDecision, row: pd.Series, contract: MemeContract, equity: Decimal
    ) -> tuple[Decimal, Sizing] | None:
        assert decision.stop_price is not None
        opening = Decimal(str(row["open"]))
        direction = Decimal("1") if decision.action == "enter_long" else Decimal("-1")
        price = opening * (Decimal("1") + direction * Decimal("0.0005"))
        risk_per_unit = abs(price - decision.stop_price) + price * (
            self.config.risk.estimated_round_trip_cost_bps / Decimal("10000")
        )
        budget = equity * self.config.risk.risk_per_trade
        if risk_per_unit <= 0 or budget <= 0:
            return None
        maximum_notional = (
            equity
            * self.config.risk.max_margin_fraction
            * Decimal(self.config.risk.leverage_ceiling)
        )
        quantity = min(budget / risk_per_unit, maximum_notional / price)
        quantity = (quantity / contract.lot_size).to_integral_value(
            rounding=ROUND_FLOOR
        ) * contract.lot_size
        notional = quantity * price
        leverage = choose_leverage(
            notional,
            equity,
            self.config.risk.max_margin_fraction,
            min(self.config.risk.leverage_ceiling, int(contract.maximum_leverage)),
        )
        effective = quantity * risk_per_unit
        if (
            leverage is None
            or quantity < contract.minimum_quantity
            or notional < contract.minimum_notional
            or effective > budget
        ):
            return None
        return price, Sizing(quantity, leverage, effective, budget)

    def _process_position(
        self, position: PaperPosition, row: pd.Series, moment: datetime, equity: Decimal
    ) -> tuple[PaperPosition | None, Decimal, list[dict[str, object]]]:
        high, low, opening = (Decimal(str(row[name])) for name in ("high", "low", "open"))
        stop_hit = (
            low <= position.stop_price if position.side is Side.BUY else high >= position.stop_price
        )
        target_hit = (
            high >= position.target_price
            if position.side is Side.BUY
            else low <= position.target_price
        )
        if stop_hit:
            price = (
                min(opening, position.stop_price)
                if position.side is Side.BUY
                else max(opening, position.stop_price)
            )
            return self._close(position, position.quantity, price, moment, equity, "STOP")
        if target_hit and not position.one_r_hit:
            half = (
                position.quantity / 2 / self.contracts[position.symbol].lot_size
            ).to_integral_value(rounding=ROUND_FLOOR) * self.contracts[position.symbol].lot_size
            half = (
                position.quantity
                if half < self.contracts[position.symbol].minimum_quantity
                else half
            )
            remaining, equity, operations = self._close(
                position, half, position.target_price, moment, equity, "TAKE_PROFIT_1R"
            )
            if remaining is None:
                return None, equity, operations
            return (
                replace(remaining, one_r_hit=True, stop_price=remaining.entry_price),
                equity,
                operations,
            )
        bars = position.bars_held + 1
        if bars >= self.config.strategy.time_stop_bars:
            return self._close(
                position, position.quantity, Decimal(str(row["close"])), moment, equity, "TIME_STOP"
            )
        if position.one_r_hit:
            atr_value = Decimal(str(row["atr"]))
            if position.side is Side.BUY:
                best = max(position.best_price or position.entry_price, high)
                stop = max(
                    position.stop_price, best - atr_value * self.config.strategy.trailing_atr
                )
            else:
                best = min(position.best_price or position.entry_price, low)
                stop = min(
                    position.stop_price, best + atr_value * self.config.strategy.trailing_atr
                )
            position = replace(position, best_price=best, stop_price=stop)
        return replace(position, bars_held=bars), equity, []

    def _close(
        self,
        position: PaperPosition,
        quantity: Decimal,
        price: Decimal,
        moment: datetime,
        equity: Decimal,
        reason: str,
    ) -> tuple[PaperPosition | None, Decimal, list[dict[str, object]]]:
        sign = Decimal("1") if position.side is Side.BUY else Decimal("-1")
        fee = self._cost(price * quantity) / 2
        pnl = sign * quantity * (price - position.entry_price) - fee
        equity += pnl
        remaining = position.quantity - quantity
        updated = None if remaining <= 0 else replace(position, quantity=remaining)
        operation = self._operation(
            moment,
            position.symbol,
            reason,
            Side.SELL if position.side is Side.BUY else Side.BUY,
            quantity,
            price,
            equity,
            position.leverage,
            reason.lower(),
        )
        return updated, equity, [operation]

    def _cost(self, notional: Decimal) -> Decimal:
        return notional * self.config.risk.estimated_round_trip_cost_bps / Decimal("10000")

    @staticmethod
    def _mark_equity(
        equity: Decimal, position: PaperPosition | None, rows: dict[str, pd.Series]
    ) -> Decimal:
        if position is None or position.symbol not in rows:
            return equity
        close = Decimal(str(rows[position.symbol]["close"]))
        sign = Decimal("1") if position.side is Side.BUY else Decimal("-1")
        return equity + sign * position.quantity * (close - position.entry_price)

    @staticmethod
    def _domain_position(position: PaperPosition) -> Position:
        return Position(
            instrument=position.symbol,
            quantity=position.quantity,
            side=position.side,
            average_entry_price=position.entry_price,
            stop_price=position.stop_price,
            opened_at=position.opened_at,
            bars_held=position.bars_held,
        )

    @staticmethod
    def _audit(moment: datetime, symbol: str, action: str, reason: str) -> dict[str, object]:
        return {
            "timestamp": moment.isoformat(),
            "symbol": symbol,
            "action": action,
            "reason": reason,
        }

    @staticmethod
    def _operation(
        moment: datetime,
        symbol: str,
        event: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        equity: Decimal,
        leverage: int,
        reason: str,
    ) -> dict[str, object]:
        return {
            "timestamp": moment.isoformat(),
            "symbol": symbol,
            "event": event,
            "side": side.value,
            "quantity": str(quantity),
            "price": str(price),
            "equity": str(equity),
            "leverage": leverage,
            "reason": reason,
        }


async def run_meme_paper(
    config: MemeBotConfig,
    *,
    duration_hours: float,
    poll_seconds: float = 15,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    if duration_hours <= 0 or poll_seconds <= 0:
        raise ValueError("paper duration and poll interval must be positive")
    baseline_path = config.storage.raw_directory / "paper.start"
    baseline = _paper_baseline(baseline_path)
    deadline = monotonic() + duration_hours * 3600
    last_signature: tuple[tuple[str, str, int], ...] | None = None
    while True:
        frames = load_recorded_frames(config.storage.raw_directory / "events.jsonl")
        contracts = load_cached_contracts(config.storage.raw_directory / "universe.json")
        qualities = load_market_qualities(config.storage.raw_directory / "stream.json", frames)
        signature = tuple(
            sorted(
                (
                    symbol,
                    str(frame["timestamp"].iloc[-1]),
                    len(frame),
                )
                for symbol, frame in frames.items()
                if not frame.empty
            )
        )
        if frames and contracts and signature != last_signature:
            report = MemePaperEngine(config, contracts).run(
                frames, qualities, trade_after=baseline, mode="paper"
            )
            _write_json(config.storage.report_path, report)
            last_signature = signature
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(poll_seconds, remaining))


def load_recorded_frames(path: Path) -> dict[str, pd.DataFrame]:
    rows: dict[str, dict[int, dict[str, object]]] = {}
    history_directory = path.parent / "history"
    if history_directory.exists():
        for history_path in history_directory.glob("*_5m.json"):
            try:
                symbol = history_path.stem.removesuffix("_5m").upper()
                history = json.loads(history_path.read_text(encoding="utf-8"))
                for candle in history:
                    timestamp = pd.Timestamp(candle["timestamp"])
                    timestamp_ms = int(timestamp.timestamp() * 1000)
                    rows.setdefault(symbol, {})[timestamp_ms] = {
                        **candle,
                        "timestamp": timestamp.to_pydatetime(),
                    }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if not path.exists():
        return {
            symbol: pd.DataFrame([values[key] for key in sorted(values)])
            for symbol, values in rows.items()
        }
    cutoff_ms = int(datetime.now(UTC).timestamp()) // 300 * 300 * 1000
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            envelope = json.loads(line)
            message = envelope["message"]
            channel = str(message.get("ch", ""))
            data = message.get("data")
            symbol = str(message.get("symbol", "")).upper()
            if channel != "market_kline_5min" or not isinstance(data, dict) or not symbol:
                continue
            timestamp_ms = int(message["ts"])
            timestamp_ms = timestamp_ms // 300_000 * 300_000
            if timestamp_ms >= cutoff_ms:
                continue
            rows.setdefault(symbol, {})[timestamp_ms] = {
                "timestamp": datetime.fromtimestamp(timestamp_ms / 1000, UTC),
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data.get("v", data.get("b", "0")),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return {
        symbol: pd.DataFrame([values[key] for key in sorted(values)])
        for symbol, values in rows.items()
    }


def load_cached_contracts(path: Path) -> tuple[MemeContract, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        MemeContract(
            symbol=str(row["symbol"]),
            base=str(row["base"]),
            tick_size=Decimal(str(row["tick_size"])),
            lot_size=Decimal(str(row["lot_size"])),
            minimum_quantity=Decimal(str(row["minimum_quantity"])),
            minimum_notional=Decimal(str(row["minimum_notional"])),
            maximum_leverage=Decimal(str(row["maximum_leverage"])),
        )
        for row in payload
        if isinstance(row, dict)
    )


def load_market_qualities(path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, MarketQuality]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(payload["generated_at"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    age = Decimal(str(max(0, (datetime.now(UTC) - generated).total_seconds())))
    qualities: dict[str, MarketQuality] = {}
    for symbol, state in payload.get("symbols", {}).items():
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        close = Decimal(str(frame["close"].iloc[-1]))
        mark = Decimal(str(state.get("mark_price") or close))
        fallback_volume = sum(
            (
                Decimal(str(row.close)) * Decimal(str(row.volume))
                for row in frame.tail(288).itertuples()
            ),
            Decimal("0"),
        )
        quote_volume = Decimal(str(state.get("quote_volume_24h") or fallback_volume))
        history = int(
            (
                pd.Timestamp(frame["timestamp"].iloc[-1]) - pd.Timestamp(frame["timestamp"].iloc[0])
            ).total_seconds()
            / 3600
        )
        recent = frame.tail(21)
        volumes = pd.to_numeric(recent["volume"])
        deviation = float(volumes.iloc[:-1].std(ddof=0)) if len(volumes) > 1 else 0
        volume_zscore = (
            Decimal(str((float(volumes.iloc[-1]) - float(volumes.iloc[:-1].mean())) / deviation))
            if deviation > 0
            else Decimal("0")
        )
        price_range = pd.to_numeric(frame.tail(14)["high"]) - pd.to_numeric(frame.tail(14)["low"])
        atr_value = Decimal(str(price_range.mean())) if not price_range.empty else Decimal("0")
        comparison = Decimal(str(frame["close"].iloc[-13])) if len(frame) >= 13 else close
        momentum = (close - comparison) / atr_value if atr_value > 0 else Decimal("0")
        funding_interval = Decimal(str(state.get("funding_interval_hours") or "8"))
        funding_8h = (
            Decimal(str(state.get("funding_rate") or "0")) * Decimal("8") / funding_interval
        )
        qualities[symbol] = MarketQuality(
            quote_volume_24h=quote_volume,
            spread_bps=Decimal(str(state.get("spread_bps") or "999999")),
            depth_half_percent=Decimal(str(state.get("depth_half_percent") or "0")),
            mark_divergence=(mark - close) / close,
            funding_8h=funding_8h,
            history_hours=history,
            momentum_atr=momentum,
            volume_zscore=volume_zscore,
            stream_age_seconds=age,
        )
    return qualities


def _paper_baseline(path: Path) -> datetime:
    if path.exists():
        return datetime.fromisoformat(path.read_text(encoding="utf-8")).astimezone(UTC)
    baseline = datetime.now(UTC)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.isoformat(), encoding="utf-8")
    return baseline


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: object) -> object:
    if isinstance(value, (Decimal, datetime, Side)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
