"""Seed deterministic cold-start outcomes and train the causal recovery model."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from uuid import uuid4

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "webhooks_log.jsonl"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "app" / "artifacts" / "causal_models.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed 500 causal recovery outcomes and train the EconML artifact."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Append-only JSONL path (default: data/webhooks_log.jsonl).",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Joblib artifact path (default: app/artifacts/causal_models.joblib).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible feature values (default: 2026).",
    )
    return parser.parse_args()


def _outcomes(
    rng: random.Random,
    *,
    treatment_count: int,
    control_count: int,
    treatment_recoveries: int,
    control_recoveries: int,
) -> list[tuple[int, int]]:
    treated = [1] * treatment_recoveries + [0] * (treatment_count - treatment_recoveries)
    controls = [1] * control_recoveries + [0] * (control_count - control_recoveries)
    records = [(1, recovered) for recovered in treated] + [
        (0, recovered) for recovered in controls
    ]
    rng.shuffle(records)
    return records


def build_seed_records(rng: random.Random) -> list[dict[str, object]]:
    """Create 500 CLOSED, TRUSTED control/treatment outcomes with known ITEs."""
    payment_cancelled = _outcomes(
        rng,
        treatment_count=100,
        control_count=100,
        treatment_recoveries=70,
        control_recoveries=20,
    )
    insufficient_funds = _outcomes(
        rng,
        treatment_count=75,
        control_count=75,
        treatment_recoveries=15,
        control_recoveries=15,
    )
    bank_downtime = _outcomes(
        rng,
        treatment_count=75,
        control_count=75,
        treatment_recoveries=0,
        control_recoveries=0,
    )

    records: list[dict[str, object]] = []
    for reason, outcomes in (
        ("payment_cancelled", payment_cancelled),
        ("insufficient_funds", insufficient_funds),
        ("bank_downtime", bank_downtime),
    ):
        for treatment_applied, is_recovered in outcomes:
            bank_failure = reason == "bank_downtime"
            behavioral_failure = reason == "payment_cancelled"
            records.append(
                {
                    "record_type": "cold_start_attributed_outcome",
                    "case_id": f"cold-start-{uuid4()}",
                    "amount": round(rng.uniform(500.0, 12_000.0), 2),
                    "case_age_hours": round(rng.uniform(48.0, 96.0), 2),
                    "merchant_budget": round(rng.uniform(10_000.0, 50_000.0), 2),
                    "error_code": "GATEWAY_ERROR" if bank_failure else "BAD_REQUEST_ERROR",
                    "error_source": "customer" if behavioral_failure else "issuer_bank",
                    "error_reason": reason,
                    "failure_class": (
                        "issuer_down"
                        if bank_failure
                        else "user_cancelled"
                        if behavioral_failure
                        else "insufficient_funds"
                    ),
                    "customer_segment": (
                        "price_sensitive" if behavioral_failure else "high_intent_repeat"
                    ),
                    "treatment_applied": treatment_applied,
                    "is_recovered": is_recovered,
                    "action": "incentive_link" if treatment_applied else "no_action",
                    "incentive_applied": bool(treatment_applied),
                    "status": "CLOSED",
                    "paid": bool(is_recovered),
                    "integrity_status": "TRUSTED",
                }
            )

    rng.shuffle(records)
    return records


async def append_records(records: list[dict[str, object]], log_path: Path) -> None:
    from app.webhook_log import append_webhook_payload

    for record in records:
        await append_webhook_payload({}, normalized_fields=record, path=log_path)


def save_artifact(model: object, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_name(
        f"{artifact_path.name}.{os.getpid()}.tmp"
    )
    try:
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, artifact_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_args()
    log_path = args.log_path.resolve()
    artifact_path = args.artifact_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Set this before importing learner so every project utility resolves the same log.
    os.environ["WEBHOOK_LOG_PATH"] = str(log_path)

    print("\n=== RecovPilot cold-start seeding ===")
    print("Generating 500 CLOSED, TRUSTED causal telemetry records...")
    records = build_seed_records(random.Random(args.seed))
    asyncio.run(append_records(records, log_path))
    print(f"Seeded {len(records)} records into {log_path}")
    print("Signal: payment cancelled 70% treated vs 20% control")
    print("Signal: insufficient funds 20% for treatment and control")
    print("Signal: bank downtime 0% recovered for treatment and control")

    from app.learner import train_t_learner_from_jsonl

    print("\nTraining DuckDB/EconML T-Learner...")
    model = train_t_learner_from_jsonl(log_path, random_state=args.seed)
    save_artifact(model, artifact_path)
    print(f"Training complete. Artifact saved to {artifact_path}")
    print("Restart FastAPI to load the newly trained model artifact.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
