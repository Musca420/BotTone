from enum import StrEnum


class TradingMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FUTURE = "future"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"


class OrderStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class SignalAction(StrEnum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    REDUCE = "reduce"


class MarketRegime(StrEnum):
    RANGE = "range"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SHOCK = "shock"
    UNKNOWN = "unknown"


class HealthLevel(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class KillSwitchCause(StrEnum):
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    STALE_DATA = "stale_data"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"
    REST_UNAVAILABLE = "rest_unavailable"
    CLOCK_DRIFT = "clock_drift"
    RATE_LIMIT = "rate_limit"
    DUPLICATE_ORDER = "duplicate_order"
    UNKNOWN_POSITION = "unknown_position"
    MISSING_PROTECTIVE_STOP = "missing_protective_stop"
    STATE_DIVERGENCE = "state_divergence"
    EXTREME_SLIPPAGE = "extreme_slippage"
    EXTREME_SPREAD = "extreme_spread"
    ABNORMAL_FREQUENCY = "abnormal_frequency"
    DATABASE_ERROR = "database_error"
    LIQUIDATION = "liquidation"
