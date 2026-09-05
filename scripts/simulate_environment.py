#!/usr/bin/env python3
"""Seed the recovery-agent database with clean and adversarial ground truth."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


# Support direct execution with `python scripts/simulate_environment.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import (  # noqa: E402
    Assignment,
    AuditLog,
    IntegritySignal,
    Intervention,
    Outcome,
    RecoveryCase,
)


CLEAN_CASE_COUNT = 5_000
ATTACK_CASE_COUNT = 500
POLICY_VERSION = "ground-truth-simulator-v2"

SEGMENTS = (
    "high_intent_repeat",
    "price_sensitive",
    "subscription_churn",
    "low_intent",
)
TREATMENT_ACTIONS = ("retry", "payment_link", "incentive_link", "message")

# Probabilities represent P(pay | segment, assigned action).
# The subscription assumptions retain the project's earlier 60%-with-retry rule.
PAY_PROBABILITY = {
    "high_intent_repeat": {
        "no_action": 0.45,
        "retry": 0.80,
        "payment_link": 0.55,
        "incentive_link": 0.85,
        "message": 0.50,
    },
    "price_sensitive": {
        "no_action": 0.05,
        "retry": 0.06,
        "payment_link": 0.15,
        "incentive_link": 0.45,
        "message": 0.08,
    },
    "subscription_churn": {
        "no_action": 0.30,
        "retry": 0.60,
        "payment_link": 0.40,
        "incentive_link": 0.50,
        "message": 0.35,
    },
    "low_intent": {
        "no_action": 0.02,
        "retry": 0.03,
        "payment_link": 0.03,
        "incentive_link": 0.03,
        "message": 0.03,
    },
}

AMOUNT_RANGES = {
    "high_intent_repeat": (500.0, 15_000.0),
    "price_sensitive": (750.0, 25_000.0),
    "subscription_churn": (199.0, 4_999.0),
    "low_intent": (100.0, 10_000.0),
}

FAILURE_CLASSES = {
    "high_intent_repeat": ("issuer_down", "network_timeout"),
    "price_sensitive": ("insufficient_funds", "user_cancelled"),
    "subscription_churn": ("issuer_down", "insufficient_funds"),
    "low_intent": ("network_timeout", "user_cancelled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed clean and adversarial recovery cases into recovery_agent.db."
    )
    parser.add_argument(
        "--clean-cases",
        type=int,
        default=CLEAN_CASE_COUNT,
        help=f"Number of clean randomized cases (default: {CLEAN_CASE_COUNT}).",
    )
    parser.add_argument(
        "--attack-cases",
        type=int,
        default=ATTACK_CASE_COUNT,
        help=f"Number of attack cases (default: {ATTACK_CASE_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible output (default: 42).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace simulator-owned rows while preserving API and user-created cases.",
    )
    args = parser.parse_args()
    if args.clean_cases < 0 or args.attack_cases < 0:
        parser.error("case counts must be non-negative")
    return args


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def random_event_time(rng: random.Random) -> datetime:
    return utcnow() - timedelta(
        days=rng.randint(1, 90),
        seconds=rng.randint(0, 86_399),
    )


def action_cost(segment: str, action: str) -> float:
    if action == "retry":
        return 0.0
    if action == "payment_link":
        return 2.0
    if action == "incentive_link":
        return 150.0 if segment == "high_intent_repeat" else 100.0
    if action == "message":
        return 1.0
    return 0.0


def action_channel(action: str) -> str:
    return {
        "retry": "api_retry",
        "payment_link": "whatsapp",
        "incentive_link": "whatsapp",
        "message": "sms",
        "no_action": "email",
    }[action]


def make_clean_records(
    index: int,
    segment: str,
    rng: random.Random,
) -> tuple[RecoveryCase, Assignment, Intervention, Outcome, float]:
    case_id = str(uuid4())
    event_time = random_event_time(rng)
    is_treatment = rng.random() < 0.50
    action = rng.choice(TREATMENT_ACTIONS) if is_treatment else "no_action"
    assignment_type = "treatment" if is_treatment else "control"
    amount_low, amount_high = AMOUNT_RANGES[segment]
    amount = round(rng.uniform(amount_low, amount_high), 2)
    paid = rng.random() < PAY_PROBABILITY[segment][action]

    case = RecoveryCase(
        case_id=case_id,
        merchant_id=f"merchant_{rng.randint(1, 50):03d}",
        transaction_id=f"txn_{uuid4().hex[:20]}",
        customer_id=f"customer_{index:06d}",
        amount=amount,
        case_age_hours=round(rng.uniform(0.0, 96.0), 2),
        attempt_count=rng.randint(1, 4 if segment == "subscription_churn" else 3),
        payment_method=rng.choices(["card", "upi"], weights=[72, 28], k=1)[0],
        failure_class=rng.choice(FAILURE_CLASSES[segment]),
        customer_segment=segment,
        event_time=event_time,
    )
    assignment = Assignment(
        id=str(uuid4()),
        case_id=case_id,
        action=action,
        assignment_type=assignment_type,
        policy_version=POLICY_VERSION,
        assigned_at=event_time + timedelta(seconds=rng.randint(1, 30)),
    )

    executed_at = (
        assignment.assigned_at + timedelta(seconds=rng.randint(1, 30))
        if is_treatment
        else None
    )
    intervention = Intervention(
        id=str(uuid4()),
        case_id=case_id,
        action=action,
        incentive_amount=(action_cost(segment, action) if action == "incentive_link" else 0.0),
        channel=action_channel(action),
        status="executed" if is_treatment else "suppressed",
        executed_at=executed_at,
    )

    delay_seconds = rng.randint(60, 3_600) if paid else None
    is_closed = paid or rng.random() < 0.75
    outcome_time = (
        event_time
        + timedelta(seconds=delay_seconds if paid else rng.randint(3_600, 86_400))
        if is_closed
        else None
    )
    outcome = Outcome(
        id=str(uuid4()),
        case_id=case_id,
        status="CLOSED" if is_closed else "PENDING",
        paid=paid,
        amount_paid=amount if paid else 0.0,
        time_to_pay_seconds=delay_seconds,
        downstream_quality=round(rng.uniform(0.85, 1.0), 3) if paid else 1.0,
        outcome_time=outcome_time,
    )
    return case, assignment, intervention, outcome, action_cost(segment, action)


def make_attack_records(
    index: int,
    attack_offset: int,
    paid_attack_count: int,
    device_hashes: tuple[str, ...],
    ip_hashes: tuple[str, ...],
    campaign_start: datetime,
    rng: random.Random,
) -> tuple[RecoveryCase, Assignment, Intervention, Outcome, str, str]:
    case_id = str(uuid4())
    event_time = campaign_start + timedelta(seconds=attack_offset * 10)
    device_hash = rng.choice(device_hashes)
    ip_hash = rng.choice(ip_hashes)
    amount = round(rng.uniform(1_000.0, 30_000.0), 2)
    paid = attack_offset < paid_attack_count
    delay_seconds = rng.randint(5, 45) if paid else None

    # The current RecoveryCase schema has no fingerprint columns. Keeping hashes in
    # IDs makes the concentrated attack pattern queryable without changing the PRD.
    case = RecoveryCase(
        case_id=case_id,
        merchant_id="merchant_attack_target",
        transaction_id=f"attack:{ip_hash}:{uuid4().hex[:10]}",
        customer_id=f"attacker:{device_hash}:{index:06d}",
        amount=amount,
        case_age_hours=72.0,
        attempt_count=15,
        payment_method="card",
        failure_class="user_cancelled",
        customer_segment="price_sensitive",
        event_time=event_time,
    )
    assignment = Assignment(
        id=str(uuid4()),
        case_id=case_id,
        action="incentive_link",
        assignment_type="treatment",
        policy_version=POLICY_VERSION,
        assigned_at=event_time + timedelta(seconds=1),
    )
    intervention = Intervention(
        id=str(uuid4()),
        case_id=case_id,
        action="incentive_link",
        incentive_amount=100.0,
        channel="whatsapp",
        status="executed",
        executed_at=event_time + timedelta(seconds=2),
    )
    outcome = Outcome(
        id=str(uuid4()),
        case_id=case_id,
        status="CLOSED",
        paid=paid,
        amount_paid=amount if paid else 0.0,
        time_to_pay_seconds=delay_seconds,
        downstream_quality=0.0,
        outcome_time=(
            event_time + timedelta(seconds=delay_seconds)
            if delay_seconds is not None
            else event_time + timedelta(hours=1)
        ),
    )
    return case, assignment, intervention, outcome, device_hash, ip_hash


def seed_environment(
    clean_count: int,
    attack_count: int,
    seed: int,
    reset: bool = False,
) -> dict[str, object]:
    rng = random.Random(seed)
    Base.metadata.create_all(bind=engine)

    segment_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    ip_counts: Counter[str] = Counter()
    total_action_cost = 0.0

    # A shuffled round-robin gives all four segments near-equal representation.
    clean_segments = [SEGMENTS[index % len(SEGMENTS)] for index in range(clean_count)]
    rng.shuffle(clean_segments)

    device_hashes = tuple(f"device_hash_{index:02d}" for index in range(10))
    ip_hashes = tuple(f"ip_hash_{index:02d}" for index in range(6))
    campaign_start = utcnow() - timedelta(days=1)
    paid_attack_count = round(attack_count * 0.98)

    session = SessionLocal()
    try:
        simulator_assignments = (
            session.query(Assignment)
            .filter(Assignment.policy_version.like("ground-truth-simulator-%"))
            .all()
        )
        simulator_case_ids = {
            assignment.case_id for assignment in simulator_assignments
        }
        if simulator_case_ids and not reset:
            raise RuntimeError(
                f"Found {len(simulator_case_ids):,} existing simulator cases. "
                "Run again with --reset to replace only simulator-owned rows."
            )
        if simulator_case_ids:
            for model in (AuditLog, IntegritySignal, Intervention, Outcome, Assignment):
                session.query(model).filter(
                    model.case_id.in_(simulator_case_ids)
                ).delete(synchronize_session=False)
            session.query(RecoveryCase).filter(
                RecoveryCase.case_id.in_(simulator_case_ids)
            ).delete(synchronize_session=False)
            session.flush()

        for index, segment in enumerate(clean_segments, start=1):
            case, assignment, intervention, outcome, cost = make_clean_records(
                index, segment, rng
            )
            session.add_all((case, assignment, intervention, outcome))
            segment_counts[segment] += 1
            assignment_counts[assignment.assignment_type] += 1
            action_counts[assignment.action] += 1
            total_action_cost += cost

        for attack_offset in range(attack_count):
            records = make_attack_records(
                clean_count + attack_offset + 1,
                attack_offset,
                paid_attack_count,
                device_hashes,
                ip_hashes,
                campaign_start,
                rng,
            )
            case, assignment, intervention, outcome, device_hash, ip_hash = records
            session.add_all((case, assignment, intervention, outcome))
            segment_counts[case.customer_segment] += 1
            assignment_counts[assignment.assignment_type] += 1
            action_counts[assignment.action] += 1
            device_counts[device_hash] += 1
            ip_counts[ip_hash] += 1
            total_action_cost += action_cost(case.customer_segment, assignment.action)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return {
        "total": clean_count + attack_count,
        "clean": clean_count,
        "attacks": attack_count,
        "paid_attacks": paid_attack_count,
        "segments": segment_counts,
        "assignments": assignment_counts,
        "actions": action_counts,
        "devices": device_counts,
        "ips": ip_counts,
        "action_cost": total_action_cost,
    }


def print_summary(summary: dict[str, object]) -> None:
    total = int(summary["total"])
    attacks = int(summary["attacks"])
    paid_attacks = int(summary["paid_attacks"])
    print(f"Total cases generated: {total:,}")
    print(f"  Clean cases: {int(summary['clean']):,}")
    print(f"  Attack cases: {attacks:,}")
    print("Customer segment distribution:")
    for segment, count in sorted(summary["segments"].items()):
        print(f"  {segment}: {count:,} ({count / total:.1%})")
    print("Assignment split:")
    for assignment_type, count in sorted(summary["assignments"].items()):
        print(f"  {assignment_type}: {count:,} ({count / total:.1%})")
    print("Action distribution:")
    for action, count in sorted(summary["actions"].items()):
        print(f"  {action}: {count:,}")
    print("Attack distribution:")
    print(f"  Labeled attacks: {attacks:,}")
    conversion = paid_attacks / attacks if attacks else 0.0
    print(f"  Immediate fake conversions: {paid_attacks:,} ({conversion:.1%})")
    print(f"  Concentrated device hashes: {len(summary['devices']):,}")
    print(f"  Concentrated IP hashes: {len(summary['ips']):,}")
    print(f"Simulated action cost: Rs {float(summary['action_cost']):,.2f}")


def main() -> None:
    args = parse_args()
    summary = seed_environment(
        args.clean_cases,
        args.attack_cases,
        args.seed,
        reset=args.reset,
    )
    print_summary(summary)
    engine.dispose()


if __name__ == "__main__":
    main()
