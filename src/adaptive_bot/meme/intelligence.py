from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from adaptive_bot.meme.config import MemeBotConfig
from adaptive_bot.meme.universe import MarketQuality


@dataclass(frozen=True)
class QuantitativeAssessment:
    liquidity_score: Decimal
    manipulation_risk: Decimal
    uncalibrated_p_win: Decimal
    expected_value_r: Decimal
    max_safe_notional: Decimal


def assess_market(quality: MarketQuality, config: MemeBotConfig) -> QuantitativeAssessment:
    spread = _ratio(quality.spread_bps, config.universe.maximum_spread_bps)
    depth_needed = config.risk.hard_notional_cap * config.universe.minimum_depth_multiple
    depth = min(Decimal("1"), _ratio(quality.depth_half_percent, depth_needed))
    funding = (
        Decimal("1")
        if quality.funding_8h is None
        else _ratio(abs(quality.funding_8h), config.universe.maximum_funding_8h)
    )
    divergence = _ratio(abs(quality.mark_divergence), config.universe.maximum_mark_divergence)
    liquidity = _clamp((Decimal("1") - spread) * Decimal("0.5") + depth * Decimal("0.5"))
    manipulation = _clamp(
        spread * Decimal("0.25")
        + funding * Decimal("0.30")
        + divergence * Decimal("0.30")
        + max(Decimal("0"), abs(quality.momentum_atr) - Decimal("3")) * Decimal("0.05")
    )
    # ponytail: this transparent prior stays shadow-only until the temporal model gate passes.
    p_win = _clamp(
        Decimal("0.50")
        + min(Decimal("0.10"), max(Decimal("0"), quality.volume_zscore) / Decimal("50"))
        + liquidity * Decimal("0.05")
        - manipulation * Decimal("0.10")
    )
    cost_r = config.risk.estimated_round_trip_cost_bps / Decimal("10000")
    expected_value = p_win - (Decimal("1") - p_win) - cost_r
    safe_notional = min(
        config.risk.hard_notional_cap,
        quality.depth_half_percent / config.universe.minimum_depth_multiple,
    )
    return QuantitativeAssessment(
        liquidity,
        manipulation,
        p_win,
        expected_value,
        max(Decimal("0"), safe_notional),
    )


def _ratio(value: Decimal, maximum: Decimal) -> Decimal:
    return Decimal("1") if maximum <= 0 else min(Decimal("1"), value / maximum)


def _clamp(value: Decimal) -> Decimal:
    return min(Decimal("1"), max(Decimal("0"), value))
