from __future__ import annotations

import json
from typing import Any

import pytest

from agents.orchestrator import process_webhook_autonomously
from agents.router import (
    AGENT_LABELS,
    ConstrainedIntentClassifier,
    HybridRouter,
    IntentClassification,
    RoutingDecision,
)
from agents.subagents import BaseRecoveryAgent, IncentiveAgent, SmartRetryAgent


def _payload(error_reason: str) -> dict[str, Any]:
    behavioral_failure = error_reason in {
        "payment_cancelled",
        "customer_abandoned",
        "general_decline",
    }
    return {
        "recovery_case": {
            "case_id": "case-router-001",
            "merchant_id": "merchant-router",
            "transaction_id": "pay-router-001",
            "customer_id": "customer-router-001",
            "amount": 1_000.0,
            "case_age_hours": 72.0,
            "attempt_count": 1,
            "payment_method": "card",
            "error_code": "BAD_REQUEST_ERROR",
            "error_source": "customer",
            "error_reason": error_reason,
            "failure_class": (
                "user_cancelled" if behavioral_failure else "insufficient_funds"
            ),
            "customer_segment": "price_sensitive",
        },
        "merchant_constraints": {
            "merchant_budget": 10_000.0,
            "merchant_budget_spent": 0.0,
            "merchant_budget_exhausted": False,
            "max_incentive": 10.0,
            "max_contacts": 2,
            "contacts_used": 0,
            "recovery_window_hours": 48.0,
            "offer_ladder": [0, 3, 5, 8, 10],
            "allowed_actions": [
                "no_action",
                "retry",
                "payment_link",
                "incentive_link",
            ],
        },
        "candidate_actions": [
            "no_action",
            "retry",
            "payment_link",
            "incentive_link",
        ],
        "integrity_status": "TRUSTED",
    }


def _incentive_estimate(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_action": "incentive_link",
        "recommended_tier": 10,
        "discount_cost": 100.0,
        "actions": {
            "no_action": {
                "causal_lift": 0.0,
                "expected_incremental_revenue": 0.0,
            },
            "payment_link": {
                "causal_lift": 0.05,
                "expected_incremental_revenue": 48.0,
            },
            "incentive_link": {
                "causal_lift": 0.40,
                "expected_incremental_revenue": 300.0,
            },
        },
    }


def test_known_payment_cancellation_uses_fast_incentive_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.subagents.estimate_uplift", _incentive_estimate)

    result = process_webhook_autonomously(_payload("payment_cancelled"))

    assert result["agent"] == "IncentiveAgent"
    assert result["router_path"] == "FastPath"
    assert result["final_action"] == "incentive_link"
    assert result["expected_lift"] == pytest.approx(0.40)
    json.dumps(result, allow_nan=False)


def test_insufficient_funds_bypasses_causal_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Insufficient funds must not invoke EconML")

    monkeypatch.setattr("agents.subagents.estimate_uplift", fail_if_called)
    result = process_webhook_autonomously(_payload("insufficient_funds"))

    assert result["agent"] == "SmartRetryAgent"
    assert result["router_path"] == "FastPath"
    assert result["final_action"] == "retry"
    assert result["discount_cost"] == 0.0


def test_known_bank_downtime_bypasses_causal_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("SmartRetryAgent must not call EconML")

    monkeypatch.setattr("agents.subagents.estimate_uplift", fail_if_called)
    result = process_webhook_autonomously(_payload("bank_downtime"))

    assert result["agent"] == "SmartRetryAgent"
    assert result["router_path"] == "FastPath"
    assert result["final_action"] == "retry"
    assert result["discount_cost"] == 0.0


@pytest.mark.parametrize(
    "reason",
    [
        "payment_cancelled",
        "customer_abandoned",
        "payment_declined",
        "general_decline",
    ],
)
def test_behavioral_failures_route_to_incentive_agent(reason: str) -> None:
    routing = HybridRouter().route(_payload(reason))

    assert isinstance(routing.agent, IncentiveAgent)
    assert routing.path == "FastPath"


@pytest.mark.parametrize(
    "reason",
    [
        "insufficient_funds",
        "exceed_withdrawal_limit",
        "insufficient_funds_mandate_block",
        "bank_technical_error",
        "service_unavailable",
        "payment_timed_out",
        "incorrect_pin",
        "pin_attempts_exceeded",
        "otp_expired",
        "otp_attempts_exceeded",
        "bank_account_inactive",
        "account_dormant",
        "invalid_card_number",
        "restricted_card",
        "daily_amount_limit_exceeded",
        "transaction_frequency_limit_exceeded",
    ],
)
def test_non_behavioral_failures_route_to_smart_retry(reason: str) -> None:
    routing = HybridRouter().route(_payload(reason))

    assert isinstance(routing.agent, SmartRetryAgent)
    assert routing.path == "FastPath"


def test_unseen_reason_uses_mockable_structured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.subagents.estimate_uplift", _incentive_estimate)
    router = HybridRouter(
        classifier=lambda reason: IntentClassification(
            agent="incentive",
            confidence=0.91,
            rationale=f"Structured test classification for {reason}.",
        )
    )

    result = process_webhook_autonomously(
        _payload("customer_aborted_weirdly"),
        router=router,
    )

    assert result["router_path"] == "LLMPath"
    assert result["router_confidence"] == pytest.approx(0.91)
    assert result["agent"] == "IncentiveAgent"


def test_grace_window_blocks_incentive_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.subagents.estimate_uplift", _incentive_estimate)
    payload = _payload("payment_cancelled")
    payload["recovery_case"]["case_age_hours"] = 2.0

    result = IncentiveAgent().execute(payload)

    assert result["final_action"] == "payment_link"
    assert result["recommended_tier"] == 0
    assert "GRACE_WINDOW_ACTIVE_SUPPRESS_DISCOUNT" in result["reason_codes"]


def test_agent_respects_merchant_action_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.subagents.estimate_uplift", _incentive_estimate)
    payload = _payload("payment_cancelled")
    payload["merchant_constraints"]["allowed_actions"] = ["no_action"]
    payload["candidate_actions"] = ["no_action"]

    incentive = IncentiveAgent().execute(payload)
    retry = SmartRetryAgent().execute(payload)

    assert incentive["final_action"] == "no_action"
    assert retry["final_action"] == "no_action"
    assert incentive["expected_incremental_revenue"] == 0.0


def test_constrained_classifier_prepares_once_and_scores_allowed_labels() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.score_calls = 0
            self.candidates: tuple[str, ...] = ()

        def prepare(self, context, json_schema):
            self.prepare_calls += 1
            assert context == {"error_reason": "new_bank_failure"}
            assert json_schema["properties"]["agent"]["enum"] == list(AGENT_LABELS)
            return "cached-key-values"

        def score_tokens(self, prepared_context, field_suffix, candidate_tokens):
            self.score_calls += 1
            self.candidates = candidate_tokens
            assert prepared_context == "cached-key-values"
            assert field_suffix == '"agent":'
            return {"incentive": -2.0, "smart_retry": 3.0}

    backend = RecordingBackend()
    classification = ConstrainedIntentClassifier(backend).classify(
        "new_bank_failure"
    )

    assert backend.prepare_calls == 1
    assert backend.score_calls == 1
    assert backend.candidates == AGENT_LABELS
    assert classification.agent == "smart_retry"
    assert classification.confidence > 0.99
    assert json.loads(classification.model_dump_json())["agent"] == "smart_retry"


def test_classifier_failure_routes_to_safe_no_action() -> None:
    router = HybridRouter(
        classifier=lambda reason: {
            "agent": "unsupported_agent",
            "confidence": 1.0,
            "rationale": reason,
        }
    )

    result = process_webhook_autonomously(
        _payload("unseen_failure"),
        router=router,
    )

    assert result["router_intent"] == "safe_no_action"
    assert result["final_action"] == "no_action"
    assert result["execution_status"] == "not_executed"


def test_subagent_failure_is_contained() -> None:
    class BrokenAgent(BaseRecoveryAgent):
        def execute(self, payload):
            del payload
            raise RuntimeError("isolated failure")

    class FixedRouter:
        def route(self, payload):
            del payload
            return RoutingDecision(
                agent=BrokenAgent(),
                agent_intent="smart_retry",
                path="FastPath",
                confidence=1.0,
                rationale="Test route.",
            )

    result = process_webhook_autonomously(
        _payload("bank_downtime"),
        router=FixedRouter(),
    )

    assert result["final_action"] == "no_action"
    assert result["failed_agent"] == "BrokenAgent"
    assert "SUBAGENT_EXECUTION_FAILED_SAFE_CLOSED" in result["reason_codes"]
