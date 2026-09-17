"""Low-latency deterministic routing with a mockable structured fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agents.subagents import (
    BaseRecoveryAgent,
    IncentiveAgent,
    SafeNoActionAgent,
    SmartRetryAgent,
)

ClassifierIntent = Literal["incentive", "smart_retry"]
AgentIntent = Literal["incentive", "smart_retry", "safe_no_action"]
RoutingPath = Literal["FastPath", "LLMPath"]
AGENT_LABELS: tuple[ClassifierIntent, ...] = ("incentive", "smart_retry")


class IntentClassification(BaseModel):
    """Structured response contract for a future external intent classifier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    agent: ClassifierIntent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=240)


@dataclass(frozen=True)
class RoutingDecision:
    agent: BaseRecoveryAgent
    agent_intent: AgentIntent
    path: RoutingPath
    confidence: float
    rationale: str


class TokenLogitBackend(Protocol):
    """Backend contract for one cached context pass and constrained token scores."""

    def prepare(
        self,
        context: Mapping[str, Any],
        json_schema: Mapping[str, Any],
    ) -> Any: ...

    def score_tokens(
        self,
        prepared_context: Any,
        field_suffix: str,
        candidate_tokens: tuple[str, ...],
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class _PreparedLocalContext:
    error_reason: str


class DeterministicTokenLogitBackend:
    """Local stand-in for a transformer decoder using the same scoring contract."""

    def prepare(
        self,
        context: Mapping[str, Any],
        json_schema: Mapping[str, Any],
    ) -> _PreparedLocalContext:
        del json_schema
        return _PreparedLocalContext(
            error_reason=_normalize_reason(context.get("error_reason"))
        )

    def score_tokens(
        self,
        prepared_context: Any,
        field_suffix: str,
        candidate_tokens: tuple[str, ...],
    ) -> Mapping[str, float]:
        del field_suffix
        if not isinstance(prepared_context, _PreparedLocalContext):
            raise TypeError("Unexpected prepared classifier context.")
        reason = prepared_context.error_reason
        incentive_signal = any(
            token in reason
            for token in ("abort", "abandon", "cancel", "decline", "hesitat")
        )
        scores = {
            "incentive": 2.0 if incentive_signal else -0.5,
            "smart_retry": -0.5 if incentive_signal else 2.0,
        }
        return {token: scores[token] for token in candidate_tokens}


class ConstrainedIntentClassifier:
    """Select an allowed schema value from logits without generating JSON tokens."""

    def __init__(self, backend: TokenLogitBackend | None = None) -> None:
        self._backend = backend or DeterministicTokenLogitBackend()

    def classify(self, reason: str) -> IntentClassification:
        normalized = _normalize_reason(reason)
        prepared = self._backend.prepare(
            {"error_reason": normalized},
            IntentClassification.model_json_schema(),
        )
        scores = self._backend.score_tokens(
            prepared,
            field_suffix='"agent":',
            candidate_tokens=AGENT_LABELS,
        )
        if set(scores) != set(AGENT_LABELS):
            raise ValueError("Classifier must score every permitted agent label exactly once.")
        logits = {label: float(scores[label]) for label in AGENT_LABELS}
        if any(not isfinite(value) for value in logits.values()):
            raise ValueError("Classifier logits must be finite.")
        maximum = max(logits.values())
        weights = {label: exp(value - maximum) for label, value in logits.items()}
        denominator = sum(weights.values())
        probabilities = {
            label: weight / denominator for label, weight in weights.items()
        }
        selected = max(AGENT_LABELS, key=lambda label: probabilities[label])
        rationale = (
            "Ambiguous customer-intent failure routed to causal evaluation."
            if selected == "incentive"
            else "Unknown failure routed to the zero-cost recovery path."
        )
        return IntentClassification(
            agent=selected,
            confidence=probabilities[selected],
            rationale=rationale,
        )


KNOWN_ERROR_REASON_ROUTES: dict[str, ClassifierIntent] = {
    "payment_cancelled": "incentive",
    "payment_cancelled_by_user": "incentive",
    "customer_abandoned": "incentive",
    "checkout_abandoned": "incentive",
    "abandonment": "incentive",
    "payment_declined": "incentive",
    "general_decline": "incentive",
    "card_declined": "incentive",
    "insufficient_funds": "smart_retry",
    "insufficient_balance": "smart_retry",
    "balance_insufficient": "smart_retry",
    "exceed_withdrawal_limit": "smart_retry",
    "insufficient_funds_mandate_block": "smart_retry",
    "bank_downtime": "smart_retry",
    "bank_technical_error": "smart_retry",
    "issuer_down": "smart_retry",
    "gateway_error": "smart_retry",
    "gateway_technical_error": "smart_retry",
    "gateway_timeout": "smart_retry",
    "network_timeout": "smart_retry",
    "service_unavailable": "smart_retry",
    "payment_timed_out": "smart_retry",
    "payment_processing_failed": "smart_retry",
    "incorrect_pin": "smart_retry",
    "pin_attempts_exceeded": "smart_retry",
    "incorrect_otp": "smart_retry",
    "invalid_otp": "smart_retry",
    "otp_expired": "smart_retry",
    "otp_attempts_exceeded": "smart_retry",
    "authentication_failed": "smart_retry",
    "bank_account_inactive": "smart_retry",
    "account_dormant": "smart_retry",
    "invalid_card_number": "smart_retry",
    "restricted_card": "smart_retry",
    "debit_instrument_inactive": "smart_retry",
    "debit_instrument_blocked": "smart_retry",
    "card_expired": "smart_retry",
    "daily_amount_limit_exceeded": "smart_retry",
    "transaction_frequency_limit_exceeded": "smart_retry",
    "transaction_limit_exceeded": "smart_retry",
    "payment_risk_check_failed": "smart_retry",
}


def _normalize_reason(reason: Any) -> str:
    return str(reason or "UNKNOWN").strip().lower().replace(" ", "_")


def _call_llm_intent_classifier(reason: str) -> IntentClassification:
    """Run constrained label scoring through the structured-output contract."""
    return ConstrainedIntentClassifier().classify(reason)


class HybridRouter:
    """Resolve known telemetry locally and escalate only ambiguous intent labels."""

    def __init__(
        self,
        classifier: Callable[[str], IntentClassification | Mapping[str, Any]] | None = None,
    ) -> None:
        self._classifier = classifier or _call_llm_intent_classifier

    @staticmethod
    def _agent(intent: ClassifierIntent) -> BaseRecoveryAgent:
        return IncentiveAgent() if intent == "incentive" else SmartRetryAgent()

    def route(self, payload: Mapping[str, Any]) -> RoutingDecision:
        case = payload.get("recovery_case")
        case_payload = case if isinstance(case, Mapping) else payload
        reason = _normalize_reason(case_payload.get("error_reason"))
        known_intent = KNOWN_ERROR_REASON_ROUTES.get(reason)
        if known_intent is not None:
            return RoutingDecision(
                agent=self._agent(known_intent),
                agent_intent=known_intent,
                path="FastPath",
                confidence=1.0,
                rationale=f"Exact Razorpay error_reason match: {reason}.",
            )

        try:
            classification = IntentClassification.model_validate(
                self._classifier(reason)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return RoutingDecision(
                agent=SafeNoActionAgent(),
                agent_intent="safe_no_action",
                path="LLMPath",
                confidence=0.0,
                rationale=f"Classifier failed closed: {type(exc).__name__}.",
            )
        return RoutingDecision(
            agent=self._agent(classification.agent),
            agent_intent=classification.agent,
            path="LLMPath",
            confidence=classification.confidence,
            rationale=classification.rationale,
        )
