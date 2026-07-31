from adaptive_bot.domain.models import Candle, Fill, Order, OrderBookSnapshot, Quote, Signal, Trade

MarketEvent = Candle | Quote | Trade | OrderBookSnapshot
TradingEvent = Signal | Order | Fill

__all__ = ["MarketEvent", "TradingEvent"]
