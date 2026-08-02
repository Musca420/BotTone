from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_bot.domain.enums import AssetClass
from adaptive_bot.domain.models import Instrument
from adaptive_bot.meme.config import MemeUniverseConfig

JsonGetter = Callable[[str, dict[str, str]], object]


@dataclass(frozen=True)
class MemeContract:
    symbol: str
    base: str
    tick_size: Decimal
    lot_size: Decimal
    minimum_quantity: Decimal
    minimum_notional: Decimal
    maximum_leverage: Decimal

    def instrument(self, leverage: int) -> Instrument:
        return Instrument(
            symbol=self.symbol,
            asset_class=AssetClass.CRYPTO,
            currency="USDT",
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            minimum_quantity=self.minimum_quantity,
            minimum_notional=self.minimum_notional,
            shortable=True,
            max_leverage=min(self.maximum_leverage, Decimal(leverage)),
        )


@dataclass(frozen=True)
class MarketQuality:
    quote_volume_24h: Decimal
    spread_bps: Decimal
    depth_half_percent: Decimal
    mark_divergence: Decimal
    funding_8h: Decimal | None
    history_hours: int
    momentum_atr: Decimal
    volume_zscore: Decimal
    stream_age_seconds: Decimal = Decimal("0")


class Eligibility(StrEnum):
    BLOCKED = "BLOCKED"
    ELIGIBLE_REDUCED = "ELIGIBLE_REDUCED"
    ELIGIBLE = "ELIGIBLE"


@dataclass(frozen=True)
class RankedCandidate:
    contract: MemeContract
    quality: MarketQuality
    eligible: bool
    reasons: tuple[str, ...]
    rank: int | None = None
    status: Eligibility = Eligibility.BLOCKED
    risk_multiplier: Decimal = Decimal("0")


class MemeUniverseClient:
    BITUNIX_PAIRS = "https://fapi.bitunix.com/api/v1/futures/market/trading_pairs"
    COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"

    def __init__(self, api_key: str | None = None, get_json: JsonGetter | None = None) -> None:
        self.api_key = api_key
        self._get_json = get_json or _get_json

    def discover(self, category: str = "meme-token") -> tuple[MemeContract, ...]:
        pairs = self._get_json(self.BITUNIX_PAIRS, {})
        query = urlencode({"vs_currency": "usd", "category": category, "per_page": 250, "page": 1})
        headers = {"x-cg-demo-api-key": self.api_key} if self.api_key else {}
        coins = self._get_json(f"{self.COINGECKO_MARKETS}?{query}", headers)
        return intersect_meme_contracts(pairs, coins)


def intersect_meme_contracts(
    pairs_payload: object, coins_payload: object
) -> tuple[MemeContract, ...]:
    pair_rows = _rows(pairs_payload)
    if not isinstance(coins_payload, list):
        raise ValueError("CoinGecko meme catalog must be a list")
    coin_symbols: dict[str, set[str]] = {}
    for coin in coins_payload:
        if isinstance(coin, dict) and coin.get("id") and coin.get("symbol"):
            coin_symbols.setdefault(str(coin["symbol"]).upper(), set()).add(str(coin["id"]))
    unique_symbols = {symbol for symbol, ids in coin_symbols.items() if len(ids) == 1}
    contracts: list[MemeContract] = []
    for row in pair_rows:
        try:
            symbol = str(row.get("symbol", row.get("symbolName", ""))).upper()
            base = str(row.get("base", row.get("baseCoin", symbol.removesuffix("USDT")))).upper()
            quote = str(row.get("quote", row.get("quoteCoin", "USDT"))).upper()
            status = str(row.get("status", "OPEN")).upper()
            api_supported = row.get("apiSupport", row.get("apiSupported", True))
            if quote != "USDT" or not symbol.endswith("USDT") or base not in unique_symbols:
                continue
            if status not in {"OPEN", "TRADING", "ONLINE"} or api_supported is False:
                continue
            contracts.append(
                MemeContract(
                    symbol=symbol,
                    base=base,
                    tick_size=_decimal(row, "tickSize", "priceStep", default="0.00000001"),
                    lot_size=_decimal(row, "lotSize", "volumeStep", default="1"),
                    minimum_quantity=_decimal(row, "minTradeVolume", "minQty", default="1"),
                    minimum_notional=_decimal(row, "minNotional", default="0"),
                    maximum_leverage=_decimal(row, "maxLeverage", default="1"),
                )
            )
        except (InvalidOperation, ValueError):
            continue
    return tuple(sorted(contracts, key=lambda contract: contract.symbol))


def rank_candidates(
    contracts: Iterable[MemeContract],
    qualities: dict[str, MarketQuality],
    config: MemeUniverseConfig,
    requested_notional: Decimal,
) -> tuple[RankedCandidate, ...]:
    candidates: list[RankedCandidate] = []
    for contract in contracts:
        quality = qualities.get(contract.symbol)
        if quality is None:
            candidates.append(RankedCandidate(contract, _empty_quality(), False, ("missing_data",)))
            continue
        reasons: list[str] = []
        soft_reasons: list[str] = []
        if quality.history_hours < config.minimum_listing_days * 24:
            reasons.append("listing_too_new")
        if quality.quote_volume_24h < config.minimum_quote_volume:
            reasons.append("insufficient_volume")
        if quality.spread_bps > config.maximum_spread_bps:
            reasons.append("spread_too_wide")
        elif quality.spread_bps > config.preferred_spread_bps:
            soft_reasons.append("spread_above_preferred")
        if quality.depth_half_percent < requested_notional * config.minimum_depth_multiple:
            reasons.append("insufficient_depth")
        if abs(quality.mark_divergence) > config.maximum_mark_divergence:
            reasons.append("mark_divergence")
        if quality.funding_8h is None:
            reasons.append("funding_unavailable")
        elif abs(quality.funding_8h) > config.maximum_funding_8h:
            soft_reasons.append("extreme_funding_requires_side_check")
        elif abs(quality.funding_8h) > config.preferred_funding_8h:
            soft_reasons.append("funding_elevated")
        if quality.stream_age_seconds > Decimal("5"):
            reasons.append("stale_stream")
        status = (
            Eligibility.BLOCKED
            if reasons
            else Eligibility.ELIGIBLE_REDUCED
            if soft_reasons
            else Eligibility.ELIGIBLE
        )
        candidates.append(
            RankedCandidate(
                contract,
                quality,
                not reasons,
                tuple((*reasons, *soft_reasons)),
                status=status,
                risk_multiplier=Decimal("0")
                if reasons
                else Decimal("0.6")
                if soft_reasons
                else Decimal("1"),
            )
        )
    eligible = sorted(
        (candidate for candidate in candidates if candidate.eligible),
        key=lambda item: (
            -abs(item.quality.momentum_atr),
            -item.quality.volume_zscore,
            item.quality.spread_bps,
            -item.quality.depth_half_percent,
        ),
    )
    ranks = {candidate.contract.symbol: index + 1 for index, candidate in enumerate(eligible)}
    return tuple(
        RankedCandidate(
            candidate.contract,
            candidate.quality,
            candidate.eligible,
            candidate.reasons,
            ranks.get(candidate.contract.symbol),
            candidate.status,
            candidate.risk_multiplier,
        )
        for candidate in sorted(
            candidates, key=lambda item: (not item.eligible, ranks.get(item.contract.symbol, 10**9))
        )
    )


def choose_leverage(
    notional: Decimal, equity: Decimal, margin_fraction: Decimal, ceiling: int
) -> int | None:
    if notional <= 0 or equity <= 0 or margin_fraction <= 0 or ceiling not in {1, 2, 3, 5}:
        return None
    margin_cap = equity * margin_fraction
    for leverage in (1, 2, 3, 5):
        if leverage <= ceiling and notional / leverage <= margin_cap:
            return leverage
    return None


def _rows(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            data = data.get("list", data.get("items", data.get("symbols", [])))
        if isinstance(data, list):
            return [cast(dict[str, Any], row) for row in data if isinstance(row, dict)]
    raise ValueError("Bitunix trading-pair response is invalid")


def _decimal(row: dict[str, Any], *names: str, default: str) -> Decimal:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            result = Decimal(str(value))
            if result < 0:
                raise ValueError("negative contract metadata")
            return result
    return Decimal(default)


def _empty_quality() -> MarketQuality:
    return MarketQuality(
        quote_volume_24h=Decimal("0"),
        spread_bps=Decimal("0"),
        depth_half_percent=Decimal("0"),
        mark_divergence=Decimal("0"),
        funding_8h=None,
        history_hours=0,
        momentum_atr=Decimal("0"),
        volume_zscore=Decimal("0"),
    )


def _get_json(url: str, headers: dict[str, str]) -> object:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "adaptive-range-bot/0.1", **headers},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)
