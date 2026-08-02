from datetime import UTC, datetime, timedelta

from adaptive_bot.meme.config import MemeBotConfig
from adaptive_bot.meme.luna import (
    LunaAction,
    MarketPolicy,
    PolicyStore,
    SourceEvidence,
    validate_policy,
)


def _policy(now: datetime) -> MarketPolicy:
    return MarketPolicy(
        policy_id="policy-20260802",
        generated_at=now,
        expires_at=now + timedelta(hours=6),
        regime="risk_on_selective",
        regime_confidence=0.8,
        systemic_risk=0.3,
        action=LunaAction.ALLOW_EVALUATION,
        allowed_strategies=("breakout_retest", "momentum_pullback"),
        risk_multiplier=0.75,
        maximum_leverage=2,
        thesis="Selective momentum is permitted only after quantitative validation.",
        invalidation_conditions=("market breadth collapses",),
        sources=(
            SourceEvidence(
                title="Bitunix market data",
                url="https://www.bitunix.com/market",
                observed_at=now,
            ),
        ),
    )


def test_luna_policy_is_validated_and_promoted_atomically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    now = datetime(2026, 8, 2, 10, tzinfo=UTC)
    config = MemeBotConfig(luna={"storage_directory": tmp_path})
    policy = _policy(now)
    assert validate_policy(policy, config, now) == (True, "ok")
    store = PolicyStore(tmp_path)
    store.promote(policy)
    assert store.active_policy() == policy


def test_luna_policy_fails_closed_on_expiry_or_unapproved_source() -> None:
    now = datetime(2026, 8, 2, 10, tzinfo=UTC)
    config = MemeBotConfig()
    assert validate_policy(_policy(now - timedelta(hours=7)), config, now)[1] == "policy_expired"
    invalid = _policy(now).model_copy(
        update={
            "sources": (
                SourceEvidence(
                    title="Unknown blog",
                    url="https://example.com/post",
                    observed_at=now,
                ),
            )
        }
    )
    assert validate_policy(invalid, config, now)[1] == "unapproved_source"
