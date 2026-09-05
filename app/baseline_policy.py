"""Research-calibrated simulation of Razorpay's incumbent recovery stack.

This module models only the recovery capabilities relevant to this project. It
does not claim to reproduce Razorpay's private production implementation or
conversion data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.guardrails import MERCHANT_ACTION_VETO_REASON, enforce_merchant_action_veto
from app.models import RecoveryCase


RECOVERY_ACTIONS = {
    "no_action",
    "retry",
    "payment_link",
    "incentive_link",
    "message",
}
CONTACT_ACTIONS = {"payment_link", "incentive_link", "message"}
INCUMBENT_POLICY_VERSION = "razorpay-incumbent-sim-v1"

INCUMBENT_CAPABILITIES = (
    "transient_failure_retry",
    "fixed_subscription_retry_ladder",
    "npci_upi_retry_cap",
    "rbi_15k_customer_authentication",
    "payment_link_recovery",
    "static_multichannel_reminders",
    "abandoned_checkout_recovery",
    "merchant_configured_static_offers",
    "velocity_throttling",
    "idempotent_audited_execution",
)

NPCI_MAX_RETRIES = "REGULATORY_NPCI_MAX_RETRIES_EXCEEDED"
RBI_AFA_REQUIRED = "BASELINE_RBI_AFA_CUSTOMER_AUTH_REQUIRED"
TRANSIENT_RETRY = "BASELINE_TRANSIENT_FAILURE_RETRY"
SUBSCRIPTION_RETRY = "BASELINE_FIXED_SUBSCRIPTION_RETRY_LADDER"
PAYMENT_LINK_REMINDER = "BASELINE_PAYMENT_LINK_REMINDER"
STATIC_OFFER = "BASELINE_STATIC_MERCHANT_OFFER"
GRACE_WINDOW = "BASELINE_GRACE_WINDOW_NO_DISCOUNT"
VELOCITY_WATCH = "BASELINE_VELOCITY_RISK_WATCH"
VELOCITY_THROTTLED = "BASELINE_VELOCITY_THROTTLED"
CONTACT_LIMIT = "BASELINE_CONTACT_LIMIT_REACHED"
ACTION_NOT_ALLOWED = "BASELINE_ACTION_NOT_ALLOWED_FALLBACK"
STATIC_OFFER_BUDGET = "BASELINE_STATIC_OFFER_BUDGET_LIMIT"
LOW_INTENT_MESSAGE = "BASELINE_LOW_INTENT_STATIC_MESSAGE"


@dataclass(frozen=True)
class BaselineTraceStep:
    stage: str
    action: str
    reason_code: str
    description: str


@dataclass(frozen=True)
class BaselineDecision:
    final_action: str
    discount_tier: int
    discount_cost: float
    channel: str | None
    integrity_status: str
    policy_decision: str
    reason_codes: tuple[str, ...]
    action_trace: tuple[BaselineTraceStep, ...]
    next_attempt_hours: float | None = None


def incumbent_capability_coverage() -> dict[str, Any]:
    """Return the explicit functional-coverage contract for tests and demos."""
    implemented = len(INCUMBENT_CAPABILITIES)
    return {
        "implemented": implemented,
        "total": implemented,
        "ratio": implemented / len(INCUMBENT_CAPABILITIES),
        "capabilities": list(INCUMBENT_CAPABILITIES),
    }


def _constraint_values(constraints: Any) -> dict[str, Any]:
    if hasattr(constraints, "model_dump"):
        return constraints.model_dump()
    return dict(constraints) if isinstance(constraints, Mapping) else {}


def _value(case: Any, name: str, default: Any = None) -> Any:
    if isinstance(case, Mapping):
        return case.get(name, default)
    return getattr(case, name, default)


def _preferred_fallback(permitted: set[str], *actions: str) -> str:
    for action in actions:
        if action in permitted:
            return action
    return "no_action"


def enforce_baseline_action_veto(
    decision: BaselineDecision,
    constraints: Any,
) -> BaselineDecision:
    """Make no_action the only fallback when the merchant vetoes an action."""
    final_action = enforce_merchant_action_veto(decision.final_action, constraints)
    if final_action == decision.final_action:
        return decision

    trace = (*decision.action_trace, BaselineTraceStep(
        stage="merchant_action_veto",
        action="no_action",
        reason_code=MERCHANT_ACTION_VETO_REASON,
        description="The merchant did not permit the fallback action, so recovery execution stopped.",
    ))
    return BaselineDecision(
        final_action="no_action",
        discount_tier=0,
        discount_cost=0.0,
        channel=None,
        integrity_status=decision.integrity_status,
        policy_decision="suppress",
        reason_codes=tuple(
            dict.fromkeys(
                (*decision.reason_codes, ACTION_NOT_ALLOWED, MERCHANT_ACTION_VETO_REASON)
            )
        ),
        action_trace=trace,
        next_attempt_hours=None,
    )


def _static_offer_tier(
    scenario_id: str | None,
    constraints: Mapping[str, Any],
) -> int:
    """Return a configured campaign tier, never an optimized tier.

    The two casino scenes explicitly represent an incumbent static-offer
    campaign. Other scenes default to zero because the incumbent has no
    per-customer offer optimizer.
    """
    if scenario_id not in {"recovery_casino", "off_vs_on_casino"}:
        return 0
    max_incentive_percentage = min(
        max(float(constraints.get("max_incentive", 0.0)), 0.0),
        99.0,
    )
    ladder = [
        int(tier)
        for tier in constraints.get("offer_ladder", [0, 3, 5, 8, 10])
        if not isinstance(tier, bool)
        and 0 <= int(tier) <= max_incentive_percentage
    ]
    return max(ladder, default=0)


def decide_incumbent_recovery(
    case: RecoveryCase | Mapping[str, Any] | Any,
    constraints: Any,
    candidate_actions: Sequence[str],
    scenario_id: str | None = None,
) -> BaselineDecision:
    """Select an incumbent action without invoking ML or causal inference."""
    values = _constraint_values(constraints)
    allowed = set(str(action) for action in values.get("allowed_actions", []))
    candidates = set(str(action) for action in candidate_actions)
    permitted = (allowed & candidates & RECOVERY_ACTIONS) | {"no_action"}

    amount = max(float(_value(case, "amount", 0.0) or 0.0), 0.0)
    age = max(float(_value(case, "case_age_hours", 0.0) or 0.0), 0.0)
    attempts = max(int(_value(case, "attempt_count", 1) or 1), 1)
    method = str(_value(case, "payment_method", "card") or "card").lower()
    failure = str(_value(case, "failure_class", "") or "")
    segment = str(_value(case, "customer_segment", "") or "")
    recovery_window = max(float(values.get("recovery_window_hours", 48.0)), 0.0)

    reasons = ["AGENT_DISABLED_RAZORPAY_INCUMBENT_SIMULATION"]
    trace = [
        BaselineTraceStep(
            stage="classification",
            action="inspect",
            reason_code="BASELINE_FAILURE_CLASSIFIED",
            description=(
                f"Classified {failure or 'unknown'} for {segment or 'unknown'} "
                f"on {method}, attempt {attempts}."
            ),
        )
    ]
    integrity_status = "TRUSTED"
    policy_decision = "approve"
    next_attempt_hours: float | None = None

    if method == "upi" and attempts >= 4:
        reasons.append(NPCI_MAX_RETRIES)
        trace.append(
            BaselineTraceStep(
                stage="regulatory_guard",
                action="suppress",
                reason_code=NPCI_MAX_RETRIES,
                description="Stopped automatic UPI recovery after the original attempt and three retries.",
            )
        )
        return BaselineDecision(
            final_action="suppress",
            discount_tier=0,
            discount_cost=0.0,
            channel=None,
            integrity_status=integrity_status,
            policy_decision="suppress",
            reason_codes=tuple(reasons),
            action_trace=tuple(trace),
        )

    if attempts > 20:
        integrity_status = "WATCH"
        reasons.append(VELOCITY_THROTTLED)
        trace.append(
            BaselineTraceStep(
                stage="operational_guard",
                action="suppress",
                reason_code=VELOCITY_THROTTLED,
                description="Throttled an obvious attempt burst without labeling it for causal training.",
            )
        )
        return BaselineDecision(
            final_action="suppress",
            discount_tier=0,
            discount_cost=0.0,
            channel=None,
            integrity_status=integrity_status,
            policy_decision="throttle",
            reason_codes=tuple(reasons),
            action_trace=tuple(trace),
        )

    if attempts > 10 or scenario_id in {"recovery_casino", "off_vs_on_casino"}:
        integrity_status = "WATCH"
        reasons.append(VELOCITY_WATCH)
        trace.append(
            BaselineTraceStep(
                stage="risk_screen",
                action="watch",
                reason_code=VELOCITY_WATCH,
                description="Flagged concentrated velocity for monitoring; no downstream-quality quarantine exists.",
            )
        )

    if amount > 15_000:
        action = _preferred_fallback(permitted, "payment_link", "message")
        reasons.append(RBI_AFA_REQUIRED)
        trace.append(
            BaselineTraceStep(
                stage="regulatory_guard",
                action=action,
                reason_code=RBI_AFA_REQUIRED,
                description="Moved the high-value recovery to a customer-authenticated payment flow.",
            )
        )
    elif segment == "subscription_churn":
        if attempts <= 3 and "retry" in permitted:
            action = "retry"
            next_attempt_hours = float((24, 48, 72)[attempts - 1])
            reasons.append(SUBSCRIPTION_RETRY)
            trace.append(
                BaselineTraceStep(
                    stage="fixed_dunning",
                    action=action,
                    reason_code=SUBSCRIPTION_RETRY,
                    description=f"Scheduled fixed retry T+{attempts} at {next_attempt_hours:.0f} hours.",
                )
            )
        else:
            action = _preferred_fallback(permitted, "payment_link", "message")
            reasons.append(PAYMENT_LINK_REMINDER)
            trace.append(
                BaselineTraceStep(
                    stage="fixed_dunning",
                    action=action,
                    reason_code=PAYMENT_LINK_REMINDER,
                    description="The fixed retry ladder ended; sent a customer-driven recovery reminder.",
                )
            )
    elif failure in {"issuer_down", "network_timeout"} and attempts <= 3:
        action = _preferred_fallback(permitted, "retry", "payment_link")
        reasons.append(TRANSIENT_RETRY if action == "retry" else ACTION_NOT_ALLOWED)
        trace.append(
            BaselineTraceStep(
                stage="in_session_recovery",
                action=action,
                reason_code=reasons[-1],
                description="Used a deterministic transient-failure retry path with payment-link fallback.",
            )
        )
    elif segment == "low_intent":
        action = _preferred_fallback(permitted, "message", "payment_link")
        reasons.append(LOW_INTENT_MESSAGE)
        trace.append(
            BaselineTraceStep(
                stage="static_reminder",
                action=action,
                reason_code=LOW_INTENT_MESSAGE,
                description="Selected a low-cost static reminder without customer-level uplift scoring.",
            )
        )
    else:
        action = _preferred_fallback(permitted, "payment_link", "message", "retry")
        reasons.append(PAYMENT_LINK_REMINDER)
        trace.append(
            BaselineTraceStep(
                stage="static_recovery",
                action=action,
                reason_code=PAYMENT_LINK_REMINDER,
                description="Selected the configured static recovery channel for this failure class.",
            )
        )

    static_tier = _static_offer_tier(scenario_id, values)
    if (
        static_tier > 0
        and segment == "price_sensitive"
        and age >= recovery_window
        and "incentive_link" in permitted
    ):
        discount_cost = amount * static_tier / 100.0
        remaining_budget = max(
            float(values.get("merchant_budget", 0.0))
            - float(values.get("merchant_budget_spent", 0.0)),
            0.0,
        )
        max_incentive_percentage = float(values.get("max_incentive", 0.0))
        if discount_cost <= remaining_budget and static_tier <= max_incentive_percentage:
            action = "incentive_link"
            reasons.append(STATIC_OFFER)
            trace.append(
                BaselineTraceStep(
                    stage="static_campaign",
                    action=action,
                    reason_code=STATIC_OFFER,
                    description=f"Applied the merchant's fixed {static_tier}% campaign offer to the whole cohort.",
                )
            )
        else:
            static_tier = 0
            reasons.append(STATIC_OFFER_BUDGET)
    elif static_tier > 0 and age < recovery_window:
        static_tier = 0
        reasons.append(GRACE_WINDOW)

    max_contacts = int(values.get("max_contacts", 1) or 0)
    contacts_used = int(values.get("contacts_used", 0) or 0)
    if action in CONTACT_ACTIONS and contacts_used >= max_contacts:
        action = "no_action"
        static_tier = 0
        policy_decision = "suppress"
        reasons.append(CONTACT_LIMIT)
        trace.append(
            BaselineTraceStep(
                stage="contact_guard",
                action=action,
                reason_code=CONTACT_LIMIT,
                description="Stopped outreach because the configured contact ceiling was reached.",
            )
        )

    if action not in permitted and action != "suppress":
        action = "no_action"
        static_tier = 0
        policy_decision = "suppress"
        reasons.append(ACTION_NOT_ALLOWED)

    discount_cost = amount * static_tier / 100.0 if action == "incentive_link" else 0.0
    channel = {
        "retry": "api_retry",
        "payment_link": "email",
        "incentive_link": "whatsapp",
        "message": "sms",
        "no_action": None,
        "suppress": None,
    }[action]
    if action == "no_action" and policy_decision == "approve":
        policy_decision = "suppress"

    decision = BaselineDecision(
        final_action=action,
        discount_tier=static_tier,
        discount_cost=round(discount_cost, 2),
        channel=channel,
        integrity_status=integrity_status,
        policy_decision=policy_decision,
        reason_codes=tuple(dict.fromkeys(reasons)),
        action_trace=tuple(trace),
        next_attempt_hours=next_attempt_hours,
    )
    return enforce_baseline_action_veto(decision, values)
