from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark import action_cost, static_actions  # noqa: E402


def test_benchmark_incumbent_uses_the_same_scenario_aware_policy_engine() -> None:
    frame = pd.DataFrame(
        [
            {
                "amount": 2_500.0,
                "case_age_hours": 2.0,
                "attempt_count": 1,
                "payment_method": "card",
                "failure_class": "network_timeout",
                "customer_segment": "high_intent_repeat",
            },
            {
                "amount": 999.0,
                "case_age_hours": 24.0,
                "attempt_count": 2,
                "payment_method": "card",
                "failure_class": "insufficient_funds",
                "customer_segment": "subscription_churn",
            },
            {
                "amount": 500.0,
                "case_age_hours": 4.0,
                "attempt_count": 4,
                "payment_method": "upi",
                "failure_class": "network_timeout",
                "customer_segment": "low_intent",
            },
            {
                "amount": 25_000.0,
                "case_age_hours": 2.0,
                "attempt_count": 1,
                "payment_method": "card",
                "failure_class": "issuer_down",
                "customer_segment": "high_intent_repeat",
            },
            {
                "amount": 4_999.0,
                "case_age_hours": 72.0,
                "attempt_count": 15,
                "payment_method": "card",
                "failure_class": "user_cancelled",
                "customer_segment": "price_sensitive",
            },
        ]
    )

    selection = static_actions(frame)

    assert selection.actions.tolist() == [
        "retry",
        "retry",
        "suppress",
        "payment_link",
        "incentive_link",
    ]
    assert selection.discount_tiers.tolist() == [0, 0, 0, 0, 10]


def test_benchmark_cost_math_handles_suppression_without_division_workarounds() -> None:
    assert action_cost(500.0, "suppress", 0) == 0.0
    assert action_cost(4_999.0, "incentive_link", 10) == 501.90
