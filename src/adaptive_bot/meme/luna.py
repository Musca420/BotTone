from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adaptive_bot.meme.config import MemeBotConfig


class LunaAction(StrEnum):
    ALLOW_EVALUATION = "ALLOW_EVALUATION"
    WATCH = "WATCH"
    REJECT = "REJECT"
    PAUSE_NEW_ENTRIES = "PAUSE_NEW_ENTRIES"


class LunaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceEvidence(LunaModel):
    title: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=8, max_length=500)
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("source timestamp must be timezone-aware")
        return value.astimezone(UTC)


class MarketPolicy(LunaModel):
    schema_version: str
    policy_id: str = Field(min_length=8, max_length=100)
    generated_at: datetime
    expires_at: datetime
    regime: str = Field(min_length=1, max_length=80)
    regime_confidence: float = Field(ge=0, le=1)
    systemic_risk: float = Field(ge=0, le=1)
    action: LunaAction
    allowed_strategies: tuple[str, ...]
    risk_multiplier: float = Field(gt=0, le=1)
    maximum_leverage: int = Field(ge=1, le=5)
    thesis: str = Field(min_length=1, max_length=1200)
    invalidation_conditions: tuple[str, ...]
    sources: tuple[SourceEvidence, ...]

    @model_validator(mode="after")
    def valid_period(self) -> MarketPolicy:
        if self.generated_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("policy timestamps must be timezone-aware")
        generated = self.generated_at.astimezone(UTC)
        expires = self.expires_at.astimezone(UTC)
        if expires <= generated:
            raise ValueError("policy expiry must follow generation")
        return self


class LunaLowRequest(LunaModel):
    request_id: str = Field(min_length=8, max_length=100)
    created_at: datetime
    symbol: str
    strategy: str
    side: str
    entry_price: str
    stop_price: str
    target_price: str
    regime: str
    quantitative_snapshot: dict[str, Any]
    policy_id: str


class LunaLowReview(LunaModel):
    schema_version: str
    request_id: str
    reviewed_at: datetime
    policy_id: str
    action: LunaAction
    confidence: float = Field(ge=0, le=1)
    risk_multiplier: float = Field(gt=0, le=1)
    reason: str = Field(min_length=1, max_length=500)


class PolicyStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests = root / "requests"
        self.reviews = root / "reviews"
        self.policies = root / "policies"

    def active_policy(self) -> MarketPolicy | None:
        try:
            return MarketPolicy.model_validate_json((self.policies / "active.json").read_text())
        except (OSError, ValueError):
            return None

    def promote(self, policy: MarketPolicy) -> None:
        self._write(self.policies / f"{policy.policy_id}.json", policy.model_dump(mode="json"))
        self._write(self.policies / "active.json", policy.model_dump(mode="json"))

    def queue(self, request: LunaLowRequest) -> None:
        path = self.requests / f"{request.request_id}.json"
        if not path.exists():
            self._write(path, request.model_dump(mode="json"))

    def review(self, request_id: str) -> LunaLowReview | None:
        try:
            return LunaLowReview.model_validate_json(
                (self.reviews / f"{request_id}.json").read_text()
            )
        except (OSError, ValueError):
            return None

    def save_review(self, review: LunaLowReview) -> None:
        self._write(self.reviews / f"{review.request_id}.json", review.model_dump(mode="json"))

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)


def validate_policy(policy: MarketPolicy, config: MemeBotConfig, now: datetime) -> tuple[bool, str]:
    current = now.astimezone(UTC)
    if policy.generated_at.astimezone(UTC) > current + timedelta(minutes=5):
        return False, "policy_from_future"
    if current >= policy.expires_at.astimezone(UTC):
        return False, "policy_expired"
    if policy.expires_at - policy.generated_at > timedelta(hours=config.luna.policy_hours):
        return False, "policy_lifetime_exceeds_limit"
    if (
        policy.regime_confidence < float(config.luna.minimum_regime_confidence)
        and policy.action is not LunaAction.PAUSE_NEW_ENTRIES
    ):
        return False, "regime_confidence_too_low"
    if (
        policy.systemic_risk > float(config.luna.maximum_systemic_risk)
        and policy.action is not LunaAction.PAUSE_NEW_ENTRIES
    ):
        return False, "systemic_risk_too_high"
    if policy.maximum_leverage > config.risk.leverage_ceiling:
        return False, "leverage_exceeds_ceiling"
    known_strategies = {
        "adaptive_range",
        "breakout_retest",
        "momentum_pullback",
    }
    if not set(policy.allowed_strategies) <= known_strategies or (
        not policy.allowed_strategies and policy.action is not LunaAction.PAUSE_NEW_ENTRIES
    ):
        return False, "invalid_strategy_allowlist"
    if not policy.sources:
        return False, "missing_sources"
    allowed = config.luna.allowed_source_domains
    for source in policy.sources:
        observed = source.observed_at.astimezone(UTC)
        if observed > current + timedelta(minutes=5) or observed < policy.generated_at.astimezone(
            UTC
        ) - timedelta(hours=24):
            return False, "stale_or_future_source"
        host = (urlparse(source.url).hostname or "").lower()
        if not any(host == domain or host.endswith(f".{domain}") for domain in allowed):
            return False, "unapproved_source"
    return True, "ok"


def codex_login_status(command: str) -> bool:
    try:
        result = subprocess.run(
            _command(command, "login", "status"),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        return result.returncode == 0 and "logged in" in output
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_codex_json(
    command: str,
    prompt: str,
    model: type[LunaModel],
    *,
    web_search: bool,
    timeout: int,
) -> LunaModel:
    with tempfile.TemporaryDirectory(prefix="meme-luna-") as temporary_name:
        directory = Path(temporary_name)
        schema = directory / "schema.json"
        output = directory / "output.json"
        schema.write_text(json.dumps(model.model_json_schema()), encoding="utf-8")
        arguments = _command(
            command,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--color",
            "never",
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(output),
        )
        if web_search:
            arguments.extend(("-c", 'web_search="live"'))
        arguments.append("-")
        result = subprocess.run(
            arguments,
            cwd=directory,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0 or not output.exists():
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Codex Luna failed closed: {detail or 'no output'}")
        return model.model_validate_json(output.read_text(encoding="utf-8"))


def _command(command: str, *arguments: str) -> list[str]:
    executable = shutil.which(command) or command
    result = [executable, *arguments]
    if os.name == "nt" and executable.lower().endswith((".cmd", ".bat")):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", *result]
    return result


def run_luna_max(config: MemeBotConfig, snapshot: dict[str, Any]) -> MarketPolicy:
    now = datetime.now(UTC)
    prompt = (
        "You are Luna Max, a cautious market-policy analyst for a long-and-short meme perpetual "
        "paper bot. Adaptive Range mean reversion is allowed only in liquid sideways regimes; "
        "long strategies are breakout and pullback in bullish regimes; short "
        "strategies are breakdown and failed-retest in distribution or bearish regimes. Positive "
        "funding is adverse to longs but may support shorts; negative funding is adverse to "
        "shorts. Use web search only for current evidence from the allowed domains included "
        "below. Never place or suggest an order. Return only the requested JSON. Fail closed: "
        "PAUSE_NEW_ENTRIES when evidence is missing, contradictory or stale. Risk multiplier may "
        "only reduce risk. Policy lifetime must be at most six hours.\n"
        f"Current UTC: {now.isoformat()}\n"
        f"Allowed domains: {json.dumps(config.luna.allowed_source_domains)}\n"
        f"Sanitized quantitative snapshot: {json.dumps(snapshot, default=str)}"
    )
    policy = run_codex_json(
        config.luna.codex_command, prompt, MarketPolicy, web_search=True, timeout=300
    )
    assert isinstance(policy, MarketPolicy)
    valid, reason = validate_policy(policy, config, now)
    if not valid:
        raise ValueError(f"Luna Max policy rejected: {reason}")
    PolicyStore(config.luna.storage_directory).promote(policy)
    return policy


def run_luna_low(config: MemeBotConfig, request: LunaLowRequest) -> LunaLowReview:
    policy = PolicyStore(config.luna.storage_directory).active_policy()
    if policy is None or request.policy_id != policy.policy_id:
        raise ValueError("Luna Low requires the matching active policy")
    prompt = (
        "You are Luna Low, a deterministic second opinion for a paper-trading setup. Do not use "
        "web search or tools. You cannot create orders. Return only JSON. Your action must be one "
        "of ALLOW_EVALUATION, WATCH, REJECT, PAUSE_NEW_ENTRIES. Reject on inconsistency, missing "
        "protective stop, stale policy, or conflict with the policy. Risk multiplier may only "
        "reduce risk.\n"
        f"Active policy: {policy.model_dump_json()}\n"
        f"Setup request: {request.model_dump_json()}"
    )
    review = run_codex_json(
        config.luna.codex_command,
        prompt,
        LunaLowReview,
        web_search=False,
        timeout=config.luna.low_timeout_seconds,
    )
    assert isinstance(review, LunaLowReview)
    if review.request_id != request.request_id or review.policy_id != policy.policy_id:
        raise ValueError("Luna Low review does not match its request or policy")
    PolicyStore(config.luna.storage_directory).save_review(review)
    return review
