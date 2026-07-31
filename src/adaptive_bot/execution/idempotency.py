from hashlib import sha256

from adaptive_bot.domain.models import Signal


def client_order_id(signal: Signal, purpose: str) -> str:
    payload = ":".join(
        (
            signal.instrument,
            signal.action.value,
            signal.exchange_timestamp.isoformat(),
            str(signal.correlation_id),
            purpose,
        )
    )
    return f"arb-{sha256(payload.encode()).hexdigest()[:24]}"
