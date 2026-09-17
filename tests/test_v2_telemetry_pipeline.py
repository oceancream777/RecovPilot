from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

import joblib
import pytest
from econml.metalearners import TLearner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import learner
from app.intake import normalize_recovery_event
from app.schemas import RecoveryCaseRead
from app.webhook_log import append_webhook_payload
from scripts.seed_and_train import build_seed_records


def _attributed_rows() -> list[dict]:
    rows = []
    telemetry = (
        ("BAD_REQUEST_ERROR", "bank", "insufficient_balance"),
        ("SERVER_ERROR", "bank", "bank_offline"),
        ("BAD_REQUEST_ERROR", "customer", "incorrect_otp"),
    )
    for index in range(120):
        action = "incentive_link" if index % 2 else "no_action"
        error_code, error_source, error_reason = telemetry[index % len(telemetry)]
        paid = index % 5 in ({0, 1, 2, 3} if action == "incentive_link" else {0, 1})
        rows.append(
            {
                "record_type": "attributed_outcome",
                "case_id": f"case-jsonl-{index:03d}",
                "amount": 1000.0 + index,
                "case_age_hours": float(index % 72),
                "merchant_budget": 25_000.0,
                "error_code": error_code,
                "error_source": error_source,
                "error_reason": error_reason,
                "failure_class": "insufficient_funds",
                "customer_segment": "price_sensitive",
                "action": action,
                "incentive_applied": action == "incentive_link",
                "status": "CLOSED",
                "paid": paid,
                "integrity_status": (
                    "QUARANTINED" if index == 119 else "TRUSTED"
                ),
            }
        )
    rows[0]["merchant_budget"] = None
    rows[1]["error_reason"] = None
    return rows


def test_cold_start_data_teaches_behavioral_lift_not_capital_deficit_lift() -> None:
    records = build_seed_records(random.Random(2026))

    def recovery_rate(reason: str, treatment: int) -> float:
        cohort = [
            row
            for row in records
            if row["error_reason"] == reason
            and row["treatment_applied"] == treatment
        ]
        return sum(int(row["is_recovered"]) for row in cohort) / len(cohort)

    assert len(records) == 500
    assert recovery_rate("payment_cancelled", 1) == pytest.approx(0.70)
    assert recovery_rate("payment_cancelled", 0) == pytest.approx(0.20)
    assert recovery_rate("insufficient_funds", 1) == pytest.approx(0.20)
    assert recovery_rate("insufficient_funds", 0) == pytest.approx(0.20)
    assert recovery_rate("bank_downtime", 1) == 0.0
    assert recovery_rate("bank_downtime", 0) == 0.0


def test_recovery_case_and_abandonment_default_to_no_attempt() -> None:
    parsed = RecoveryCaseRead.model_validate(
        {
            "case_id": "case-no-attempt",
            "merchant_id": "merchant-test",
            "customer_id": "customer-test",
            "amount": 4999.0,
            "failure_class": "user_cancelled",
            "customer_segment": "price_sensitive",
            "event_time": "2026-08-30T07:58:49.099Z",
        }
    )
    abandoned = normalize_recovery_event(
        {
            "event": "checkout.abandoned",
            "payload": {
                "shop_id": "merchant-test",
                "email": "buyer@example.test",
                "cart_token": "cart-test",
                "cart_amount": 4999.0,
                "drop_off_step": "payment_method",
            },
        },
        "checkout.abandoned",
    )

    assert parsed.error_code == "NO_ATTEMPT"
    assert parsed.error_source == "NO_ATTEMPT"
    assert parsed.error_reason == "NO_ATTEMPT"
    assert abandoned["error_code"] == "NO_ATTEMPT"
    assert abandoned["error_source"] == "NO_ATTEMPT"
    assert abandoned["error_reason"] == "NO_ATTEMPT"


def test_append_only_log_flattens_payload_and_duckdb_reads_training_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "webhooks_log.jsonl"

    async def write_rows() -> None:
        await append_webhook_payload(
            {"payload": {"payment": {"entity": {"id": "pay_001"}}}},
            normalized_fields={"record_type": "webhook_received"},
            path=path,
        )
        await asyncio.gather(
            *(
                append_webhook_payload({}, normalized_fields=row, path=path)
                for row in _attributed_rows()
            )
        )

    asyncio.run(write_rows())
    first_record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    records = learner.load_training_records_from_jsonl(path)

    assert first_record["payload.payment.entity.id"] == "pay_001"
    assert len(records) == 119
    assert all(record["integrity_status"] == "TRUSTED" for record in records)


def test_econml_pipeline_handles_unseen_failure_labels_and_round_trips(
    tmp_path: Path,
) -> None:
    path = tmp_path / "webhooks_log.jsonl"

    async def write_rows() -> None:
        await asyncio.gather(
            *(
                append_webhook_payload({}, normalized_fields=row, path=path)
                for row in _attributed_rows()
            )
        )

    asyncio.run(write_rows())
    model = learner.train_t_learner_from_jsonl(path, random_state=7)
    context = {
        "amount": 8000.0,
        "case_age_hours": 50.0,
        "merchant_budget": 20_000.0,
        "error_code": "NEW_PROVIDER_ERROR",
        "error_source": "new_gateway",
        "error_reason": "previously_unseen_reason",
        "customer_segment": "price_sensitive",
        "integrity_status": "TRUSTED",
        "permitted_actions": ["no_action", "incentive_link"],
        "offer_ladder": [0, 3, 5, 8, 10],
        "max_incentive": 10.0,
    }

    estimate = model.estimate_uplift(context)
    artifact = tmp_path / "causal_models.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    restored_estimate = restored.estimate_uplift(context)

    assert isinstance(model.causal_model, TLearner)
    assert estimate["recommended_tier"] in context["offer_ladder"]
    assert -1.0 <= estimate["actions"]["incentive_link"]["causal_lift"] <= 1.0
    assert restored_estimate["baseline_pay_probability"] == pytest.approx(
        estimate["baseline_pay_probability"]
    )
