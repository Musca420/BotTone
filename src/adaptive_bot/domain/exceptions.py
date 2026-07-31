class AdaptiveBotError(Exception):
    """Base application error."""


class DataQualityError(AdaptiveBotError):
    """Raised when market data cannot be trusted."""


class InvalidOrderTransition(AdaptiveBotError):
    """Raised for an illegal order state transition."""


class LiveTradingDisabled(AdaptiveBotError):
    """Raised whenever live trading safeguards are not satisfied."""
