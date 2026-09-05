"""Deterministic merchant-policy vetoes for recovery recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import RecoveryCase


RECOVERY_ACTIONS = {"no_action", "retry", "payment_link", "incentive_link", "message"}
CONTACT_ACTIONS = {"payment_link", "incentive_link", "message"}
NPCI_MAX_UPI_ATTEMPTS_REASON = "REGULATORY_NPCI_MAX_RETRIES_EXCEEDED"
RBI_15K_AFA_REASON = "REGULATORY_RBI_15K_AFA_LIMIT"
GRACE_WINDOW_REASON = "GRACE_WINDOW_ACTIVE_SUPPRESS_DISCOUNT"
INVALID_TIER_REASON = "TIER_NOT_IN_LADDER"
INCENTIVE_NOT_ALLOWED_REASON = "INCENTIVE_LINK_NOT_ALLOWED"
MERCHANT_ACTION_VETO_REASON = "MERCHANT_ACTION_VETO_NO_ACTION"
MAX_INCENTIVE_PERCENTAGE_REASON = "MAX_INCENTIVE_LIMIT_APPLIED"


def _merchant_constraints(context: dict[str, Any]) -> dict[str, Any]:
    constraints = context.get("merchant_constraints", context)
    if hasattr(constraints, "model_dump"):
        return constraints.model_dump()
    return dict(constraints) if isinstance(constraints, dict) else {}


def _permitted_actions(context: dict[str, Any], constraints: dict[str, Any]) -> set[str]:
    candidates = set(context.get("candidate_actions", []))
    allowed = set(constraints.get("allowed_actions", []))
    # No action is always retained as the fail-safe even if a caller omits it.
    return (candidates & allowed & RECOVERY_ACTIONS) | {"no_action"}


def _append_reason_code(context: dict[str, Any], reason_code: str) -> None:
    reason_codes = context.setdefault("guardrail_reason_codes", [])
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def enforce_merchant_action_veto(
    intended_action: str,
    merchant_constraints: Any,
) -> str:
    """Apply the merchant's final action veto after every fallback path.

    ``no_action`` and ``suppress`` are safe terminal states, not merchant-facing
    recovery interventions. Every executable recovery action must appear in the
    merchant-provided allowed-actions list.
    """
    constraints = (
        merchant_constraints.model_dump()
        if hasattr(merchant_constraints, "model_dump")
        else dict(merchant_constraints)
        if isinstance(merchant_constraints, dict)
        else {}
    )
    action = str(intended_action)
    if action in {"no_action", "suppress"}:
        return action
    allowed_actions = set(constraints.get("allowed_actions", []))
    return action if action in allowed_actions else "no_action"


def apply_policy_guard(
    recommended_action: str,
    case: RecoveryCase,
    integrity_status: str,
    context: dict,
) -> str:
    """Return the final action after regulatory, integrity, and budget checks."""
    constraints = _merchant_constraints(context)
    permitted_actions = _permitted_actions(context, constraints)
    action = recommended_action if recommended_action in RECOVERY_ACTIONS else "no_action"

    # NPCI's 1+3 UPI retry cap is a hard stop, independent of model confidence.
    payment_method = (case.payment_method or "card").lower()
    attempt_count = case.attempt_count or 1
    if payment_method == "upi" and attempt_count >= 4:
        _append_reason_code(context, NPCI_MAX_UPI_ATTEMPTS_REASON)
        return "suppress"

    if integrity_status.upper() == "QUARANTINED":
        return "no_action"

    offer_ladder = constraints.get("offer_ladder", [0, 3, 5, 8, 10])
    max_incentive_percentage = float(constraints["max_incentive"])
    recommended_tier = context.get("recommended_tier", 0)
    if (
        isinstance(recommended_tier, bool)
        or not isinstance(recommended_tier, int)
        or recommended_tier not in offer_ladder
    ):
        _append_reason_code(context, INVALID_TIER_REASON)
        return "suppress"

    if recommended_tier > max_incentive_percentage:
        _append_reason_code(context, MAX_INCENTIVE_PERCENTAGE_REASON)
        return "payment_link" if "payment_link" in permitted_actions else "no_action"

    if action == "incentive_link" and action not in permitted_actions:
        _append_reason_code(context, INCENTIVE_NOT_ALLOWED_REASON)
        return "no_action"

    recovery_window_hours = float(
        context.get(
            "recovery_window_hours",
            constraints.get("recovery_window_hours", 48.0),
        )
        or 0.0
    )
    case_age_hours = float(case.case_age_hours)
    if case_age_hours < recovery_window_hours:
        _append_reason_code(context, GRACE_WINDOW_REASON)
        if action not in {"retry", "payment_link"}:
            if "retry" in permitted_actions:
                action = "retry"
            elif "payment_link" in permitted_actions:
                action = "payment_link"
            else:
                return "no_action"

    # Amounts above the AFA threshold require a customer-driven payment flow.
    if case.amount > 15_000 and action == "retry":
        action = "payment_link"
        _append_reason_code(context, RBI_15K_AFA_REASON)

    if action not in permitted_actions:
        return "no_action"

    budget = float(constraints["merchant_budget"])
    budget_spent = float(constraints.get("merchant_budget_spent", 0.0) or 0.0)
    budget_exhausted = bool(constraints.get("merchant_budget_exhausted", False)) or (
        budget_spent >= budget
    )
    discount_cost = float(
        context.get(
            "discount_cost",
            case.amount * (recommended_tier / 100.0),
        )
        or 0.0
    )
    remaining_budget = max(budget - budget_spent, 0.0)

    if action == "incentive_link" and (
        budget_exhausted
        or discount_cost > remaining_budget
    ):
        return "payment_link" if "payment_link" in permitted_actions else "no_action"

    max_contacts = int(constraints.get("max_contacts", 1) or 0)
    contacts_used = int(constraints.get("contacts_used", 0) or 0)
    if action in CONTACT_ACTIONS and contacts_used >= max_contacts:
        return "no_action"

    return action


@dataclass(frozen=True)
class PolicyDecisionResult:
    """Compatibility result for the existing webhook intake endpoint."""

    proposed_action: str
    final_action: str
    policy_decision: str
    reason: str


def apply_guardrails(case: dict[str, Any], integrity_verdict: str) -> PolicyDecisionResult:
    """Keep the original intake helper available while callers migrate."""
    if integrity_verdict.upper() == "QUARANTINED":
        return PolicyDecisionResult(
            proposed_action="review",
            final_action="no_action",
            policy_decision="suppress",
            reason="Case quarantined by integrity checks.",
        )

    failure_class = str(case.get("failure_class") or "")
    segment = str(case.get("customer_segment") or "")
    if failure_class == "issuer_down":
        action = "retry"
    elif segment in {"price_sensitive", "subscription_churn"}:
        action = "incentive_link"
    else:
        action = "payment_link"
    return PolicyDecisionResult(
        proposed_action=action,
        final_action=action,
        policy_decision="approve",
        reason="Legacy intake policy approved the bounded recovery action.",
    )
