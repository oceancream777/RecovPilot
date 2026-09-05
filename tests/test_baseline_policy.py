from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.baseline_policy import (  # noqa: E402
    decide_incumbent_recovery,
    incumbent_capability_coverage,
)
from app.recovery_simulator import simulate_expected_recovery  # noqa: E402
from app.schemas import MerchantConstraints  # noqa: E402


ACTIONS = ["no_action", "retry", "payment_link", "incentive_link", "message"]


def _constraints(**updates) -> MerchantConstraints:
    values = {
        "merchant_budget": 100_000.0,
        "max_incentive": 10.0,
        "max_contacts": 3,
        "contacts_used": 0,
        "recovery_window_hours": 48.0,
        "offer_ladder": [0, 3, 5, 8, 10],
        "allowed_actions": ACTIONS,
    }
    values.update(updates)
    return MerchantConstraints(**values)


def _case(**updates) -> SimpleNamespace:
    values = {
        "amount": 4_999.0,
        "case_age_hours": 50.0,
        "attempt_count": 1,
        "payment_method": "card",
        "failure_class": "user_cancelled",
        "customer_segment": "price_sensitive",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_incumbent_scope_exceeds_the_agreed_80_percent_threshold() -> None:
    coverage = incumbent_capability_coverage()

    assert coverage["total"] == 10
    assert coverage["implemented"] == 10
    assert coverage["ratio"] >= 0.80
    assert len(set(coverage["capabilities"])) == 10


def test_transient_failure_uses_in_session_retry_without_a_link() -> None:
    decision = decide_incumbent_recovery(
        _case(failure_class="network_timeout"),
        _constraints(),
        ACTIONS,
    )

    assert decision.final_action == "retry"
    assert decision.channel == "api_retry"
    assert "BASELINE_TRANSIENT_FAILURE_RETRY" in decision.reason_codes


def test_subscription_uses_fixed_calendar_retry_ladder() -> None:
    decision = decide_incumbent_recovery(
        _case(customer_segment="subscription_churn", attempt_count=2),
        _constraints(),
        ACTIONS,
    )

    assert decision.final_action == "retry"
    assert decision.next_attempt_hours == 48.0
    assert "BASELINE_FIXED_SUBSCRIPTION_RETRY_LADDER" in decision.reason_codes


def test_npci_attempt_cap_suppresses_additional_upi_retries() -> None:
    decision = decide_incumbent_recovery(
        _case(payment_method="upi", attempt_count=4),
        _constraints(),
        ACTIONS,
    )

    assert decision.final_action == "suppress"
    assert decision.policy_decision == "suppress"
    assert "REGULATORY_NPCI_MAX_RETRIES_EXCEEDED" in decision.reason_codes


def test_rbi_high_value_case_uses_customer_authenticated_payment_link() -> None:
    decision = decide_incumbent_recovery(
        _case(amount=25_000.0, failure_class="issuer_down"),
        _constraints(),
        ACTIONS,
    )

    assert decision.final_action == "payment_link"
    assert "BASELINE_RBI_AFA_CUSTOMER_AUTH_REQUIRED" in decision.reason_codes


def test_static_campaign_is_watched_but_not_given_causal_quarantine() -> None:
    decision = decide_incumbent_recovery(
        _case(),
        _constraints(),
        ACTIONS,
        "recovery_casino",
    )

    assert decision.final_action == "incentive_link"
    assert decision.discount_tier == 10
    assert decision.discount_cost == pytest.approx(499.90)
    assert decision.integrity_status == "WATCH"
    assert "BASELINE_STATIC_MERCHANT_OFFER" in decision.reason_codes


def test_static_offer_falls_back_when_budget_cannot_fund_it() -> None:
    decision = decide_incumbent_recovery(
        _case(),
        _constraints(merchant_budget=100.0, max_incentive=10.0),
        ACTIONS,
        "recovery_casino",
    )

    assert decision.final_action == "payment_link"
    assert decision.discount_tier == 0
    assert "BASELINE_STATIC_OFFER_BUDGET_LIMIT" in decision.reason_codes


def test_expected_value_math_has_no_fake_denominator_for_free_retry() -> None:
    decision = decide_incumbent_recovery(
        _case(failure_class="network_timeout"),
        _constraints(),
        ACTIONS,
    )
    result = simulate_expected_recovery(
        _case(failure_class="network_timeout"),
        decision.final_action,
        decision.discount_tier,
        decision.integrity_status,
        decision.reason_codes,
        decision.action_trace,
        incumbent_capability_coverage()["ratio"],
    )

    assert result.intervention_cost == 0.0
    assert result.recovery_roi is None
    assert result.incremental_recovered_revenue > 0.0


def test_attack_offer_preserves_raw_conversion_but_zeroes_downstream_value() -> None:
    decision = decide_incumbent_recovery(
        _case(),
        _constraints(),
        ACTIONS,
        "off_vs_on_casino",
    )
    result = simulate_expected_recovery(
        _case(),
        decision.final_action,
        decision.discount_tier,
        decision.integrity_status,
        decision.reason_codes,
        decision.action_trace,
        incumbent_capability_coverage()["ratio"],
        "off_vs_on_casino",
    )

    assert result.predicted_probability == 0.98
    assert result.downstream_quality == 0.0
    assert result.net_expected_recovery < 0.0
    assert result.attacks_quarantined == 0
