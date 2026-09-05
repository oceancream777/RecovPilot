from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.learner import MultiTreatmentTLearner, _ActionModel
from app.schemas import DemoToggleMLMetrics, MerchantConstraints


def _fixed_offer_learner() -> MultiTreatmentTLearner:
    model = MultiTreatmentTLearner()
    model.models = {
        "no_action": _ActionModel(
            encoder=None,
            classifier=None,
            probability=0.10,
            samples=400,
        ),
        "incentive_link": _ActionModel(
            encoder=None,
            classifier=None,
            probability=0.45,
            samples=400,
        ),
    }
    return model


def test_offer_ladder_selects_highest_positive_incremental_net_tier() -> None:
    estimate = _fixed_offer_learner().estimate_uplift(
        {
            "amount": 1000.0,
            "failure_class": "network_timeout",
            "customer_segment": "price_sensitive",
            "integrity_status": "TRUSTED",
            "permitted_actions": ["no_action", "incentive_link"],
            "offer_ladder": [0, 3, 5, 10],
            "max_incentive": 10.0,
            "tier_pay_probabilities": {0: 0.10, 3: 0.12, 5: 0.24, 10: 0.27},
        }
    )

    assert estimate["recommended_action"] == "incentive_link"
    assert estimate["recommended_tier"] == 5
    assert estimate["discount_cost"] == pytest.approx(50.0)
    assert estimate["net_expected_recovery"] == pytest.approx(90.0)
    assert estimate["tier_estimates"][3]["incremental_net_recovery"] == pytest.approx(
        -10.0
    )
    assert estimate["tier_estimates"][10]["incremental_net_recovery"] == pytest.approx(
        70.0
    )


def test_offer_ladder_returns_no_action_when_every_paid_tier_is_non_positive() -> None:
    estimate = _fixed_offer_learner().estimate_uplift(
        {
            "amount": 1000.0,
            "failure_class": "network_timeout",
            "customer_segment": "price_sensitive",
            "integrity_status": "TRUSTED",
            "permitted_actions": ["no_action", "incentive_link"],
            "offer_ladder": [0, 3, 5],
            "max_incentive": 5.0,
            "tier_pay_probabilities": {0: 0.10, 3: 0.10, 5: 0.11},
        }
    )

    assert estimate["recommended_action"] == "no_action"
    assert estimate["recommended_tier"] == 0
    assert estimate["discount_cost"] == 0.0
    assert estimate["net_expected_recovery"] == 0.0


def test_offer_ladder_never_selects_an_unlisted_tier() -> None:
    ladder = [0, 3, 5, 8, 10]
    estimate = _fixed_offer_learner().estimate_uplift(
        {
            "amount": 1000.0,
            "failure_class": "network_timeout",
            "customer_segment": "price_sensitive",
            "integrity_status": "TRUSTED",
            "permitted_actions": ["no_action", "incentive_link"],
            "offer_ladder": ladder,
            "max_incentive": 10.0,
            "tier_pay_probabilities": {
                0: 0.10,
                3: 0.15,
                5: 0.20,
                8: 0.25,
                10: 0.30,
                99: 1.0,
            },
        }
    )

    assert estimate["recommended_tier"] in ladder
    assert estimate["recommended_tier"] != 99
    assert set(estimate["tier_estimates"]) == set(ladder)


def test_offer_ladder_respects_maximum_incentive_percentage() -> None:
    estimate = _fixed_offer_learner().estimate_uplift(
        {
            "amount": 1000.0,
            "failure_class": "network_timeout",
            "customer_segment": "price_sensitive",
            "integrity_status": "TRUSTED",
            "permitted_actions": ["no_action", "incentive_link"],
            "offer_ladder": [0, 3, 5, 10],
            "max_incentive": 5.0,
            "tier_pay_probabilities": {0: 0.10, 3: 0.13, 5: 0.25, 10: 0.50},
        }
    )

    assert estimate["recommended_tier"] == 5
    assert estimate["discount_cost"] == pytest.approx(50.0)
    assert set(estimate["tier_estimates"]) == {0, 3, 5}
    assert 10 not in estimate["tier_estimates"]


def test_offer_ladder_uses_a_dynamic_ceiling_above_ten_percent() -> None:
    estimate = _fixed_offer_learner().estimate_uplift(
        {
            "amount": 1000.0,
            "failure_class": "network_timeout",
            "customer_segment": "price_sensitive",
            "integrity_status": "TRUSTED",
            "permitted_actions": ["no_action", "incentive_link"],
            "offer_ladder": [0, 5, 10, 15, 20],
            "max_incentive": 15.0,
            "tier_pay_probabilities": {
                0: 0.10,
                5: 0.20,
                10: 0.35,
                15: 0.55,
                20: 0.90,
            },
        }
    )

    assert estimate["recommended_tier"] == 15
    assert estimate["discount_cost"] == pytest.approx(150.0)
    assert set(estimate["tier_estimates"]) == {0, 5, 10, 15}


def test_maximum_incentive_percentage_must_be_below_one_hundred() -> None:
    constraints = {
        "merchant_budget": 1000.0,
        "max_incentive": 25.5,
        "allowed_actions": ["no_action", "incentive_link"],
    }

    parsed = MerchantConstraints.model_validate(constraints)
    assert parsed.max_incentive == 25.5

    with pytest.raises(ValueError):
        MerchantConstraints.model_validate({**constraints, "max_incentive": 100.0})


def test_swagger_schema_exposes_safe_offer_defaults() -> None:
    constraints_schema = MerchantConstraints.model_json_schema()
    metrics_schema = DemoToggleMLMetrics.model_json_schema()

    assert constraints_schema["properties"]["offer_ladder"]["default"] == [
        0,
        3,
        5,
        8,
        10,
    ]
    assert metrics_schema["properties"]["recommended_tier"]["default"] == 0
