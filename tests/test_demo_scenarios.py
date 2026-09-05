from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.demo_scenarios import (  # noqa: E402
    build_scenario_metrics,
    estimate_scenario_uplift,
    list_scenarios,
    scenario_integrity_override,
    warmup_scenario,
)
from app.schemas import MerchantConstraints, ScenarioMetrics, ScenarioWarmupResponse  # noqa: E402


SCENARIO_IDS = [
    "discount_wins_trap",
    "recovery_casino",
    "policy_rollback",
    "off_vs_on_casino",
    "predictive_vs_causal",
]


def _constraints() -> MerchantConstraints:
    return MerchantConstraints(
        merchant_budget=250_000.0,
        max_incentive=10.0,
        allowed_actions=["retry", "payment_link", "incentive_link", "no_action"],
    )


def _context(constraints: MerchantConstraints) -> dict:
    return {
        "amount": 4_999.0,
        "failure_class": "user_cancelled",
        "customer_segment": "high_intent_repeat",
        "integrity_status": "TRUSTED",
        "anomaly_score": 0.05,
        "permitted_actions": constraints.allowed_actions,
        "offer_ladder": constraints.offer_ladder,
        "max_incentive": constraints.max_incentive,
        "merchant_constraints": constraints.model_dump(),
    }


def test_catalog_contains_the_five_document_scenarios() -> None:
    catalog = list_scenarios()

    assert [item["scenario_id"] for item in catalog] == SCENARIO_IDS
    assert all(item["event_count"] == 10_000 for item in catalog)
    assert sum(item["attack_count"] for item in catalog) == 1_000


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_every_scenario_warms_to_a_strict_contract(scenario_id: str) -> None:
    result = warmup_scenario(scenario_id)

    validated = ScenarioWarmupResponse.model_validate(result)
    assert validated.status == "ready"
    assert validated.event_count == 10_000


def test_discount_trap_can_choose_payment_link_over_raw_discount_conversion() -> None:
    constraints = _constraints()
    estimate = estimate_scenario_uplift("discount_wins_trap", _context(constraints))

    assert estimate["actions"]["incentive_link"]["pay_probability"] > estimate["actions"][
        "payment_link"
    ]["pay_probability"]
    assert estimate["recommended_action"] == "payment_link"
    assert estimate["recommended_tier"] == 0


def test_recovery_casino_computes_quarantine_from_attack_evidence() -> None:
    verdict = scenario_integrity_override("recovery_casino", "case-attack-test")

    assert verdict is not None
    assert verdict["integrity_status"] == "QUARANTINED"
    assert verdict["signal_type"] == "reward_farming"
    assert "downstream quality" in verdict["rationale"]


def test_fixed_seed_replay_is_deterministic_and_quarantines_all_attacks() -> None:
    constraints = _constraints()
    estimate = estimate_scenario_uplift("off_vs_on_casino", _context(constraints))
    first = build_scenario_metrics(
        "off_vs_on_casino",
        True,
        4_999.0,
        50.0,
        constraints,
        estimate,
        estimate["recommended_action"],
        "QUARANTINED",
    )
    second = build_scenario_metrics(
        "off_vs_on_casino",
        True,
        4_999.0,
        50.0,
        constraints,
        estimate,
        estimate["recommended_action"],
        "QUARANTINED",
    )

    ScenarioMetrics.model_validate(first)
    assert first["cumulative_revenue"] == second["cumulative_revenue"]
    assert first["incremental_recovered_revenue"] == second["incremental_recovered_revenue"]
    assert first["attacks_quarantined"] == 500


def test_drift_scene_executes_a_computed_canary_rollback() -> None:
    constraints = _constraints()
    estimate = estimate_scenario_uplift("policy_rollback", _context(constraints))
    result = build_scenario_metrics(
        "policy_rollback",
        True,
        7_499.0,
        72.0,
        constraints,
        estimate,
        estimate["recommended_action"],
        "TRUSTED",
    )

    assert result["policy_status"] == "ROLLED_BACK"
    assert result["reason_codes"] == [
        "CANARY_LIFT_DEGRADED",
        "PROMOTION_REJECTED",
        "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
    ]
    assert [step["status"] for step in result["policy_trace"]] == [
        "INCUMBENT",
        "CANDIDATE",
        "ROLLED_BACK",
    ]
