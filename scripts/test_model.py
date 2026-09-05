#!/usr/bin/env python3
"""Load or bootstrap the causal learner and inspect predictions without payments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal  # noqa: E402
from app.learner import ACTIONS, bootstrap_or_load_model  # noqa: E402


SEGMENTS = (
    "high_intent_repeat",
    "price_sensitive",
    "subscription_churn",
    "low_intent",
)
FAILURE_CLASSES = (
    "issuer_down",
    "insufficient_funds",
    "network_timeout",
    "user_cancelled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the persisted T-learner locally. This command never creates "
            "a Razorpay Payment Link."
        )
    )
    parser.add_argument("--amount", type=float, default=2500.0)
    parser.add_argument("--segment", choices=SEGMENTS, default="price_sensitive")
    parser.add_argument(
        "--failure-class",
        choices=FAILURE_CLASSES,
        default="network_timeout",
    )
    parser.add_argument(
        "--integrity-status",
        choices=("TRUSTED", "WATCH", "QUARANTINED"),
        default="TRUSTED",
    )
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="Evaluate one representative case for each customer segment.",
    )
    args = parser.parse_args()
    if args.amount <= 0:
        parser.error("--amount must be greater than zero")
    return args


def evaluate_context(model, context: dict[str, Any]) -> None:
    estimate = model.estimate_uplift(context)
    print()
    print(
        f"Segment={context['customer_segment']} amount=INR {context['amount']:.2f} "
        f"integrity={context['integrity_status']}"
    )
    print(f"Recommended action: {estimate['recommended_action']}")
    print(f"Expected incremental revenue: INR {estimate['expected_incremental_revenue']:.2f}")
    print()
    print(
        f"{'Action':<18} {'P(pay)':>9} {'Lift':>9} "
        f"{'Net INR':>12} {'Samples':>9} {'Allowed':>9}"
    )
    print("-" * 72)
    for action in ACTIONS:
        result = estimate["actions"][action]
        probability = result["pay_probability"]
        lift = result["causal_lift"]
        net_revenue = result["expected_incremental_revenue"]
        print(
            f"{action:<18} "
            f"{probability if probability is not None else float('nan'):>9.3f} "
            f"{lift if lift is not None else float('nan'):>9.3f} "
            f"{net_revenue if net_revenue is not None else float('nan'):>12.2f} "
            f"{result['training_samples']:>9d} "
            f"{str(result['permitted']):>9}"
        )


def main() -> int:
    args = parse_args()
    with SessionLocal() as db:
        model = bootstrap_or_load_model(db)

    if model is None:
        print(
            "Model unavailable: seed the database with "
            "`python scripts/simulate_environment.py` and retry.",
            file=sys.stderr,
        )
        return 1

    segments = SEGMENTS if args.all_segments else (args.segment,)
    for segment in segments:
        evaluate_context(
            model,
            {
                "amount": args.amount,
                "failure_class": args.failure_class,
                "customer_segment": segment,
                "integrity_status": args.integrity_status,
                "anomaly_score": 0.0,
                "permitted_actions": list(ACTIONS),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
