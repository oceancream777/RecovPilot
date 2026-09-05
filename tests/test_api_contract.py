from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import Base
from app.models import AuditLog, BatchPolicyRun, IntegritySignal, Intervention
from app.schemas import DemoToggleRequest
from app import learner, main, razorpay_client


EXPECTED_DECISION_KEYS = {
    "recommended_action",
    "expected_incremental_revenue",
    "confidence",
    "integrity_status",
    "policy_decision",
    "reason_codes",
    "policy_version",
    "current_policy_version",
}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    artifact_path = tmp_path / "artifacts" / "causal_models.joblib"
    monkeypatch.setattr(main, "SessionLocal", testing_session_local)
    monkeypatch.setattr(main, "MODEL_ARTIFACT_PATH", str(artifact_path))
    monkeypatch.setattr(learner, "MODEL_ARTIFACT_PATH", str(artifact_path))
    main.app.dependency_overrides[main.get_db] = override_get_db
    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def _decision_payload() -> dict:
    return {
        "case_id": "case-test-001",
        "merchant_id": "merchant-test",
        "transaction_id": "pay-test-001",
        "customer_id": "customer-test-001",
        "amount": 2500.0,
        "case_age_hours": 72.0,
        "attempt_count": 1,
        "payment_method": "card",
        "failure_class": "network_timeout",
        "customer_segment": "price_sensitive",
        "event_time": "2026-08-30T07:58:49.099Z",
        "merchant_constraints": {
            "merchant_budget": 10000.0,
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
                "message",
            ],
            "policy_version": "v3.0-test",
        },
        "candidate_actions": [
            "no_action",
            "retry",
            "payment_link",
            "incentive_link",
            "message",
        ],
    }


def _demo_payload(agent_enabled: bool) -> dict:
    decision_payload = _decision_payload()
    recovery_case_fields = {
        key: decision_payload[key]
        for key in (
            "case_id",
            "merchant_id",
            "transaction_id",
            "customer_id",
            "amount",
            "case_age_hours",
            "attempt_count",
            "payment_method",
            "failure_class",
            "customer_segment",
            "event_time",
        )
    }
    return {
        "agent_enabled": agent_enabled,
        "recovery_case": recovery_case_fields,
        "merchant_constraints": decision_payload["merchant_constraints"],
        "candidate_actions": decision_payload["candidate_actions"],
    }


def _trusted_signal(case, _recent_history) -> IntegritySignal:
    return IntegritySignal(
        case_id=case.case_id,
        signal_type="pattern_probe",
        anomaly_score=0.05,
        integrity_status="TRUSTED",
        rationale="test trusted case",
    )


def _quarantined_signal(case, _recent_history) -> IntegritySignal:
    return IntegritySignal(
        case_id=case.case_id,
        signal_type="reward_farming",
        anomaly_score=0.98,
        integrity_status="QUARANTINED",
        rationale="test forced quarantine",
    )


def _watch_signal(case, _recent_history) -> IntegritySignal:
    return IntegritySignal(
        case_id=case.case_id,
        signal_type="velocity_spike",
        anomaly_score=0.65,
        integrity_status="WATCH",
        rationale="test legitimate repeated activity requiring review",
    )


def _incentive_recommendation(_context: dict) -> dict:
    return {
        "recommended_action": "incentive_link",
        "recommended_tier": 5,
        "discount_cost": 125.0,
        "net_expected_recovery": 750.0,
        "baseline_pay_probability": 0.10,
        "expected_incremental_revenue": 750.0,
        "ranked_actions": ["incentive_link", "payment_link", "retry", "message", "no_action"],
        "actions": {
            "no_action": {
                "pay_probability": 0.10,
                "causal_lift": 0.0,
                "action_cost": 0.0,
                "expected_abuse_penalty": 0.0,
                "expected_incremental_revenue": 0.0,
                "permitted": True,
                "training_samples": 300,
            },
            "incentive_link": {
                "pay_probability": 0.45,
                "causal_lift": 0.35,
                "action_cost": 125.0,
                "expected_abuse_penalty": 0.0,
                "expected_incremental_revenue": 750.0,
                "permitted": True,
                "training_samples": 300,
            },
            "payment_link": {
                "pay_probability": 0.15,
                "causal_lift": 0.05,
                "action_cost": 2.0,
                "expected_abuse_penalty": 0.0,
                "expected_incremental_revenue": 123.0,
                "permitted": True,
                "training_samples": 300,
            },
            "retry": {
                "pay_probability": 0.11,
                "causal_lift": 0.01,
                "action_cost": 0.0,
                "expected_abuse_penalty": 0.0,
                "expected_incremental_revenue": 25.0,
                "permitted": True,
                "training_samples": 300,
            },
            "message": {
                "pay_probability": 0.12,
                "causal_lift": 0.02,
                "action_cost": 1.0,
                "expected_abuse_penalty": 0.0,
                "expected_incremental_revenue": 49.0,
                "permitted": True,
                "training_samples": 300,
            },
        },
    }


def _retry_recommendation(_context: dict) -> dict:
    estimate = deepcopy(_incentive_recommendation({}))
    estimate["recommended_action"] = "retry"
    estimate["recommended_tier"] = 0
    estimate["discount_cost"] = 0.0
    estimate["net_expected_recovery"] = 0.0
    estimate["expected_incremental_revenue"] = 25.0
    estimate["ranked_actions"] = [
        "retry",
        "payment_link",
        "incentive_link",
        "message",
        "no_action",
    ]
    return estimate


def test_decision_evaluate_returns_prd_v3_contract_keys(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)

    response = client.post("/api/v1/decision/evaluate", json=_decision_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_DECISION_KEYS
    assert body["recommended_action"] == "incentive_link"
    assert body["policy_decision"] == "approve"


def test_decision_evaluate_suppresses_quarantined_case_regardless_of_ml_recommendation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "evaluate_case_integrity", _quarantined_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    payload = deepcopy(_decision_payload())
    payload["case_id"] = "case-test-quarantined-001"

    response = client.post("/api/v1/decision/evaluate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_DECISION_KEYS
    assert body["integrity_status"] == "QUARANTINED"
    assert body["policy_decision"] == "suppress"
    assert body["recommended_action"] == "no_action"


def test_razorpay_client_applies_discount_and_sends_paise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict = {}

    class FakePaymentLink:
        def create(self, payload: dict) -> dict:
            captured_payload.update(payload)
            return {"short_url": "https://rzp.io/i/test-discount"}

    class FakeClient:
        payment_link = FakePaymentLink()

    monkeypatch.setattr(razorpay_client, "client", FakeClient())

    result = razorpay_client.create_recovery_payment_link(
        2500.0,
        "case-test-sdk-001",
        "incentive_link",
        8.0,
    )

    assert captured_payload == {
        "amount": 230000,
        "currency": "INR",
        "reference_id": "case-test-sdk-001",
        "description": "Discounted Payment Recovery",
    }
    assert result == {
        "short_url": "https://rzp.io/i/test-discount",
        "final_amount": 2300.0,
    }


def test_demo_toggle_baseline_bypasses_ml(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str, str, float]] = []

    def fake_payment_link(
        amount: float,
        case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((amount, case_id, action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/test-baseline",
            "final_amount": amount,
        }

    def fail_if_ml_runs(_context: dict) -> dict:
        raise AssertionError("baseline mode must bypass the learner")

    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)
    monkeypatch.setattr(main, "estimate_uplift", fail_if_ml_runs)

    response = client.post(
        "/api/v1/execute/demo_toggle",
        json=_demo_payload(agent_enabled=False),
    )

    assert response.status_code == 200
    assert calls == []
    body = response.json()
    assert body["short_url"] is None
    assert body["final_amount"] == 2500.0
    assert body["final_action"] == "retry"
    assert body["agent_enabled"] is False
    assert body["ml_metrics"] is None
    assert body["input_amount"] == 2500.0
    assert body["discount_cost"] == 0.0
    assert body["net_expected_recovery"] == 150.0
    assert body["expected_incremental_revenue"] == 25.0
    assert "AGENT_DISABLED_RAZORPAY_INCUMBENT_SIMULATION" in body["reason_codes"]
    assert "BASELINE_TRANSIENT_FAILURE_RETRY" in body["reason_codes"]
    assert body["scenario_metrics"] is None
    assert body["recovery_metrics"]["recovery_roi"] is None
    assert body["recovery_metrics"]["incumbent_capability_coverage"] >= 0.80


def test_demo_toggle_incumbent_static_campaign_does_not_borrow_causal_quarantine(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str, str, float]] = []

    def fake_payment_link(
        amount: float,
        case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((amount, case_id, action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/test-static-campaign",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)
    payload = _demo_payload(agent_enabled=False)
    payload["scenario_id"] = "recovery_casino"
    payload["merchant_constraints"]["max_incentive"] = 10.0

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert calls == [(2500.0, "case-test-001", "incentive_link", 10)]
    assert body["final_action"] == "incentive_link"
    assert body["final_amount"] == 2250.0
    assert body["recovery_metrics"]["integrity_status"] == "WATCH"
    assert body["recovery_metrics"]["attacks_quarantined"] == 0
    assert body["recovery_metrics"]["downstream_quality"] == 0.0
    assert "BASELINE_STATIC_MERCHANT_OFFER" in body["reason_codes"]


def test_dynamic_inputs_are_required_and_preserved_by_pydantic() -> None:
    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_age_hours"] = 19.75
    payload["merchant_constraints"]["merchant_budget"] = 4321.5
    payload["merchant_constraints"]["max_incentive"] = 7.25

    parsed = DemoToggleRequest.model_validate(payload)
    assert parsed.auto_execute is False

    assert parsed.recovery_case.case_age_hours == 19.75
    assert parsed.merchant_constraints.merchant_budget == 4321.5
    assert parsed.merchant_constraints.max_incentive == 7.25
    assert "merchant_constraints" in DemoToggleRequest.model_json_schema()["required"]
    merchant_required = DemoToggleRequest.model_json_schema()["$defs"][
        "MerchantConstraints"
    ]["required"]
    assert "merchant_budget" in merchant_required
    assert "max_incentive" in merchant_required


def test_action_arrays_are_required_and_preserved_exactly() -> None:
    payload = _demo_payload(agent_enabled=True)
    allowed_actions = ["retry", "payment_link", "future_action"]
    candidate_actions = ["payment_link", "retry", "future_action"]
    payload["merchant_constraints"]["allowed_actions"] = allowed_actions
    payload["candidate_actions"] = candidate_actions

    parsed = DemoToggleRequest.model_validate(payload)
    schema = DemoToggleRequest.model_json_schema()

    assert parsed.merchant_constraints.allowed_actions == allowed_actions
    assert parsed.candidate_actions == candidate_actions
    assert "candidate_actions" in schema["required"]
    assert "default" not in schema["properties"]["candidate_actions"]
    merchant_schema = schema["$defs"]["MerchantConstraints"]
    assert "allowed_actions" in merchant_schema["required"]
    assert "default" not in merchant_schema["properties"]["allowed_actions"]


def test_demo_toggle_rejects_missing_dynamic_merchant_constraints(
    client: TestClient,
) -> None:
    payload = _demo_payload(agent_enabled=True)
    payload.pop("merchant_constraints")

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 422


def test_demo_toggle_passes_dynamic_values_to_learner_and_guardrails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[float, float, float]] = {}
    original_guard = main.apply_policy_guard

    def capture_estimate(context: dict) -> dict:
        constraints = context["merchant_constraints"]
        captured["learner"] = (
            context["max_incentive"],
            constraints["merchant_budget"],
            constraints["max_incentive"],
        )
        return _incentive_recommendation(context)

    def capture_guard(action, case, integrity_status, context):
        constraints = context["merchant_constraints"]
        captured["guardrails"] = (
            case.case_age_hours,
            constraints["merchant_budget"],
            constraints["max_incentive"],
        )
        return original_guard(action, case, integrity_status, context)

    def fake_payment_link(
        amount: float,
        _case_id: str,
        _action_type: str,
        _discount_percentage: float,
    ) -> dict:
        return {"short_url": "https://rzp.io/i/dynamic", "final_amount": amount}

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"].update(
        {"case_id": "case-test-dynamic-inputs", "case_age_hours": 19.75}
    )
    payload["merchant_constraints"].update(
        {
            "merchant_budget": 4321.5,
            "max_incentive": 7.25,
            "recovery_window_hours": 12.0,
        }
    )
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", capture_estimate)
    monkeypatch.setattr(main, "apply_policy_guard", capture_guard)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    assert captured["learner"] == (7.25, 4321.5, 7.25)
    assert captured["guardrails"] == (19.75, 4321.5, 7.25)


def test_demo_toggle_suppresses_disallowed_incentive_recommendation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("a disallowed incentive must not create a payment link")

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_id"] = "case-test-incentive-not-allowed"
    payload["merchant_constraints"]["allowed_actions"] = [
        "no_action",
        "retry",
        "payment_link",
    ]
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "no_action"
    assert body["short_url"] is None
    assert body["ml_metrics"]["policy_decision"] == "suppress"
    assert "INCENTIVE_LINK_NOT_ALLOWED" in body["ml_metrics"]["reason_codes"]


def test_demo_toggle_agent_uses_guarded_incentive_action(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str, str, float]] = []

    def fake_payment_link(
        amount: float,
        case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((amount, case_id, action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/test-agent",
            "final_amount": 2375.0,
        }

    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post(
        "/api/v1/execute/demo_toggle",
        json=_demo_payload(agent_enabled=True),
    )

    assert response.status_code == 200
    body = response.json()
    assert calls == []
    assert body["case_id"] == "case-test-001"
    assert body["short_url"] is None
    assert body["final_amount"] == 2375.0
    assert body["final_action"] == "incentive_link"
    assert body["ml_metrics"]["recommended_action"] == "incentive_link"
    assert body["ml_metrics"]["recommended_tier"] == 5
    assert body["ml_metrics"]["discount_cost"] == 125.0
    assert body["ml_metrics"]["net_expected_recovery"] == 750.0
    assert body["ml_metrics"]["integrity_status"] == "TRUSTED"
    assert body["ml_metrics"]["policy_decision"] == "approve"

    db = main.SessionLocal()
    try:
        intervention = (
            db.query(Intervention)
            .filter(Intervention.case_id == body["case_id"])
            .one()
        )
        assert intervention.status == "pending"
        assert intervention.executed_at is None
        assert intervention.incentive_amount == pytest.approx(125.0)
    finally:
        db.close()

    mismatched = client.post(
        "/api/v1/execute/approve_link",
        json={"case_id": body["case_id"], "approved_action": "payment_link"},
    )
    assert mismatched.status_code == 409
    assert calls == []

    approval = client.post(
        "/api/v1/execute/approve_link",
        json={"case_id": body["case_id"], "approved_action": "incentive_link"},
    )

    assert approval.status_code == 200
    assert calls == [(2500.0, "case-test-001", "incentive_link", 5.0)]
    assert approval.json()["short_url"] == "https://rzp.io/i/test-agent"
    assert approval.json()["final_amount"] == 2375.0

    db = main.SessionLocal()
    try:
        intervention = (
            db.query(Intervention)
            .filter(Intervention.case_id == body["case_id"])
            .one()
        )
        assert intervention.status == "executed"
        assert intervention.executed_at is not None
    finally:
        db.close()

    duplicate = client.post(
        "/api/v1/execute/approve_link",
        json={"case_id": body["case_id"], "approved_action": "incentive_link"},
    )
    assert duplicate.status_code == 409
    assert calls == [(2500.0, "case-test-001", "incentive_link", 5.0)]


def test_on_auto_executes_at_exactly_25_percent_of_merchant_budget(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_payment_link(
        amount: float,
        _case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/auto-low-risk",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    payload = _demo_payload(agent_enabled=True)
    payload["auto_execute"] = True
    payload["recovery_case"]["case_id"] = "case-auto-low-risk"
    payload["merchant_constraints"]["max_incentive"] = 10.0
    payload["merchant_constraints"]["merchant_budget"] = 500.0
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "auto_executed"
    assert body["short_url"] == "https://rzp.io/i/auto-low-risk"
    assert calls == [("incentive_link", 5)]

    db = main.SessionLocal()
    try:
        intervention = (
            db.query(Intervention)
            .filter(Intervention.case_id == body["case_id"])
            .one()
        )
        assert intervention.status == "executed"
        assert intervention.executed_at is not None
    finally:
        db.close()


def test_percentage_ceiling_and_budget_concentration_require_approval(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def ten_percent_recommendation(context: dict) -> dict:
        estimate = _incentive_recommendation(context)
        estimate.update(
            {
                "recommended_tier": 10,
                "discount_cost": 800.0,
                "net_expected_recovery": 1000.0,
                "expected_incremental_revenue": 1000.0,
            }
        )
        return estimate

    def fake_payment_link(
        amount: float,
        _case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/approved-budget-review",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    payload = _demo_payload(agent_enabled=True)
    payload["auto_execute"] = True
    payload["recovery_case"].update(
        {"case_id": "case-percentage-budget-review", "amount": 8000.0}
    )
    payload["merchant_constraints"].update(
        {
            "merchant_budget": 2000.0,
            "max_incentive": 10.0,
            "offer_ladder": list(range(11)),
        }
    )
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", ten_percent_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_human_review"
    assert body["short_url"] is None
    assert body["ml_metrics"]["recommended_tier"] == 10
    assert body["ml_metrics"]["discount_cost"] == pytest.approx(800.0)
    assert "CIRCUIT_BREAKER_MERCHANT_BUDGET_CONCENTRATION" in body["reason_codes"]
    assert calls == []

    approval = client.post(
        "/api/v1/execute/approve_link",
        json={
            "case_id": body["case_id"],
            "approved_action": "incentive_link",
        },
    )

    assert approval.status_code == 200
    assert approval.json()["final_amount"] == pytest.approx(7200.0)
    assert calls == [("incentive_link", 10.0)]


def test_approve_link_reports_provider_quota_and_keeps_approval_pending(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_id"] = "case-provider-quota"
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)

    evaluation = client.post("/api/v1/execute/demo_toggle", json=payload)
    assert evaluation.status_code == 200
    assert evaluation.json()["status"] == "pending_human_review"

    def quota_exhausted(*_args, **_kwargs) -> dict:
        raise razorpay_client.RazorpayPaymentLinkQuotaError(
            "ServerError",
            "test mode limit of 30 reached for payment_link",
        )

    monkeypatch.setattr(main, "create_recovery_payment_link", quota_exhausted)
    approval = client.post(
        "/api/v1/execute/approve_link",
        json={
            "case_id": payload["recovery_case"]["case_id"],
            "approved_action": "incentive_link",
        },
    )

    assert approval.status_code == 429
    assert "test mode limit of 30" in approval.json()["detail"].lower()
    assert "retry this pending approval" in approval.json()["detail"].lower()

    db = main.SessionLocal()
    try:
        intervention = (
            db.query(Intervention)
            .filter(Intervention.case_id == payload["recovery_case"]["case_id"])
            .one()
        )
        assert intervention.status == "pending"
        assert intervention.executed_at is None
    finally:
        db.close()


@pytest.mark.parametrize(
    ("case_updates", "merchant_budget", "max_incentive", "expected_reason"),
    [
        (
            {"case_id": "case-auto-high-ticket", "amount": 12_000.0},
            10_000.0,
            10.0,
            "CIRCUIT_BREAKER_HIGH_TICKET_VALUE",
        ),
        (
            {"case_id": "case-auto-high-incentive"},
            400.0,
            10.0,
            "CIRCUIT_BREAKER_MERCHANT_BUDGET_CONCENTRATION",
        ),
        (
            {
                "case_id": "case-auto-fingerprint-review",
                "customer_id": "customer:device_hash_00:legitimate",
                "transaction_id": "pay:ip_hash_00:legitimate",
            },
            10_000.0,
            10.0,
            "CIRCUIT_BREAKER_SUSPICIOUS_FINGERPRINT_REVIEW",
        ),
    ],
)
def test_on_auto_routes_circuit_breakers_to_human_review(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    case_updates: dict,
    merchant_budget: float,
    max_incentive: float,
    expected_reason: str,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_payment_link(
        amount: float,
        _case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/human-approved",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    def context_aware_incentive(context: dict) -> dict:
        estimate = _incentive_recommendation(context)
        discount_cost = float(context["amount"]) * 0.05
        estimate["discount_cost"] = discount_cost
        estimate["net_expected_recovery"] = max(
            float(estimate["net_expected_recovery"]) - discount_cost,
            0.0,
        )
        return estimate

    payload = _demo_payload(agent_enabled=True)
    payload["auto_execute"] = True
    payload["recovery_case"].update(case_updates)
    payload["merchant_constraints"]["merchant_budget"] = merchant_budget
    payload["merchant_constraints"]["max_incentive"] = max_incentive
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", context_aware_incentive)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_human_review"
    assert body["short_url"] is None
    assert expected_reason in body["reason_codes"]
    assert calls == []

    approval = client.post(
        "/api/v1/execute/approve_link",
        json={
            "case_id": body["case_id"],
            "approved_action": "incentive_link",
        },
    )
    assert approval.status_code == 200
    assert approval.json()["status"] == "executed"
    assert calls == [("incentive_link", 5.0)]
    assert approval.json()["short_url"] == "https://rzp.io/i/human-approved"

    db = main.SessionLocal()
    try:
        intervention = (
            db.query(Intervention)
            .filter(Intervention.case_id == body["case_id"])
            .one()
        )
        assert intervention.status == "executed"
        assert intervention.executed_at is not None
    finally:
        db.close()

    duplicate = client.post(
        "/api/v1/execute/approve_link",
        json={"case_id": body["case_id"], "approved_action": "incentive_link"},
    )
    assert duplicate.status_code == 409
    assert calls == [("incentive_link", 5.0)]


def test_on_auto_routes_integrity_watch_to_human_review(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("WATCH activity must wait for human approval")

    payload = _demo_payload(agent_enabled=True)
    payload["auto_execute"] = True
    payload["recovery_case"]["case_id"] = "case-auto-watch-review"
    payload["merchant_constraints"]["max_incentive"] = 10.0
    monkeypatch.setattr(main, "evaluate_case_integrity", _watch_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_human_review"
    assert body["short_url"] is None
    assert "CIRCUIT_BREAKER_INTEGRITY_WATCH_REVIEW" in body["reason_codes"]


def test_demo_toggle_quarantine_does_not_create_payment_link(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("a quarantined case must not create a payment link")

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_id"] = "case-test-demo-quarantined"
    monkeypatch.setattr(main, "evaluate_case_integrity", _quarantined_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["short_url"] is None
    assert body["final_amount"] == 2500.0
    assert body["final_action"] == "no_action"
    assert body["ml_metrics"]["integrity_status"] == "QUARANTINED"
    assert body["ml_metrics"]["policy_decision"] == "suppress"


def test_demo_toggle_suppresses_upi_after_npci_retry_cap(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("a UPI case at the retry cap must be suppressed")

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"].update(
        {
            "case_id": "case-test-npci-cap",
            "payment_method": "upi",
            "attempt_count": 4,
        }
    )
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _retry_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "suppress"
    assert body["short_url"] is None
    assert body["final_amount"] == 2500.0
    assert body["ml_metrics"]["policy_decision"] == "suppress"
    assert (
        "REGULATORY_NPCI_MAX_RETRIES_EXCEEDED"
        in body["ml_metrics"]["reason_codes"]
    )


def test_demo_toggle_redirects_high_value_retry_to_payment_link(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, str, str, float]] = []

    def fake_payment_link(
        amount: float,
        case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((amount, case_id, action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/test-rbi-afa",
            "final_amount": amount,
        }

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"].update(
        {
            "case_id": "case-test-rbi-afa",
            "amount": 15_001.0,
            "payment_method": "card",
            "attempt_count": 1,
        }
    )
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _retry_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert calls == []
    assert body["short_url"] is None
    assert body["final_action"] == "payment_link"
    assert body["ml_metrics"]["recommended_action"] == "retry"
    assert body["ml_metrics"]["policy_decision"] == "modify"
    assert "REGULATORY_RBI_15K_AFA_LIMIT" in body["ml_metrics"]["reason_codes"]

    approval = client.post(
        "/api/v1/execute/approve_link",
        json={"case_id": body["case_id"], "approved_action": "payment_link"},
    )
    assert approval.status_code == 200
    assert calls == [(15_001.0, "case-test-rbi-afa", "payment_link", 0.0)]


def test_demo_toggle_grace_window_removes_discount(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("a retry action must not create a payment link")

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"].update(
        {"case_id": "case-test-grace-window", "case_age_hours": 12.0}
    )
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "retry"
    assert body["short_url"] is None
    assert body["final_amount"] == 2500.0
    assert (
        "GRACE_WINDOW_ACTIVE_SUPPRESS_DISCOUNT"
        in body["ml_metrics"]["reason_codes"]
    )


def test_demo_toggle_suppresses_tier_outside_merchant_ladder(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_tier_recommendation(_context: dict) -> dict:
        estimate = deepcopy(_incentive_recommendation({}))
        estimate["recommended_tier"] = 7
        return estimate

    def fail_if_razorpay_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("an out-of-ladder tier must never reach Razorpay")

    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_id"] = "case-test-invalid-tier"
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", invalid_tier_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_razorpay_runs)

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "suppress"
    assert body["short_url"] is None
    assert body["final_amount"] == 2500.0
    assert body["ml_metrics"]["policy_decision"] == "suppress"
    assert "TIER_NOT_IN_LADDER" in body["ml_metrics"]["reason_codes"]


def test_demo_catalog_and_recovery_casino_are_wired_end_to_end(
    client: TestClient,
) -> None:
    catalog_response = client.get("/api/v1/demo/scenarios")

    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    assert len(catalog) == 5
    assert {item["scenario_id"] for item in catalog} == {
        "discount_wins_trap",
        "recovery_casino",
        "policy_rollback",
        "off_vs_on_casino",
        "predictive_vs_causal",
    }

    payload = _demo_payload(agent_enabled=True)
    payload["scenario_id"] = "recovery_casino"
    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["short_url"] is None
    assert body["final_action"] == "no_action"
    assert body["ml_metrics"]["integrity_status"] == "QUARANTINED"
    assert "POLICY_GUARD_APPROVED" not in body["ml_metrics"]["reason_codes"]
    assert body["scenario_metrics"]["event_count"] == 10_000
    assert body["scenario_metrics"]["attacks_quarantined"] == 500


@pytest.mark.parametrize(
    ("agent_enabled", "expected_action", "expected_tier", "expected_reason"),
    [
        (
            False,
            "incentive_link",
            10.0,
            "NAIVE_ML_OPTIMIZED_FOR_RAW_CONVERSION",
        ),
        (
            True,
            "payment_link",
            0.0,
            "CAUSAL_LIFT_NEGATIVE_MARGIN_SAVED",
        ),
    ],
)
def test_predictive_vs_causal_uses_the_correct_comparator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    agent_enabled: bool,
    expected_action: str,
    expected_tier: float,
    expected_reason: str,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_payment_link(
        amount: float,
        _case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        calls.append((action_type, discount_percentage))
        return {
            "short_url": "https://rzp.io/i/scenario-comparator",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    if not agent_enabled:
        def fail_if_incumbent_runs(*_args, **_kwargs):
            raise AssertionError("Scenario 5 OFF must use Naive ML, not incumbent rules")

        monkeypatch.setattr(main, "decide_incumbent_recovery", fail_if_incumbent_runs)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)
    payload = _demo_payload(agent_enabled=agent_enabled)
    payload["scenario"] = "predictive_vs_causal"
    payload["recovery_case"].update(
        {
            "case_id": f"case-predictive-causal-{agent_enabled}",
            "amount": 4_999.0,
            "customer_segment": "high_intent_repeat",
            "failure_class": "issuer_down",
        }
    )
    payload["merchant_constraints"]["max_incentive"] = 10.0

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == expected_action
    assert expected_reason in body["reason_codes"]
    if agent_enabled:
        assert calls == []
        assert body["short_url"] is None
        assert body["ml_metrics"]["recommended_action"] == "payment_link"
        assert body["ml_metrics"]["recommended_tier"] == 0
        approval = client.post(
            "/api/v1/execute/approve_link",
            json={
                "case_id": body["case_id"],
                "approved_action": expected_action,
            },
        )
        assert approval.status_code == 200
        assert calls == [(expected_action, expected_tier)]
    else:
        assert calls == [(expected_action, expected_tier)]
        assert body["ml_metrics"] is None
        assert body["discount_cost"] == pytest.approx(499.9)


@pytest.mark.parametrize("scenario", ["predictive_vs_causal", "policy_rollback"])
def test_forced_causal_fallback_respects_merchant_action_veto(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    def fail_if_payment_link_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("A merchant-vetoed fallback must not create a link")

    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_payment_link_runs)
    payload = _demo_payload(agent_enabled=True)
    payload["scenario"] = scenario
    payload["recovery_case"]["case_id"] = f"case-veto-{scenario}"
    payload["merchant_constraints"]["allowed_actions"] = [
        "incentive_link",
        "no_action",
    ]
    payload["candidate_actions"] = ["incentive_link", "no_action"]

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "no_action"
    assert body["short_url"] is None
    assert body["discount_cost"] == 0.0
    assert body["net_expected_recovery"] == 0.0
    assert body["expected_incremental_revenue"] == 0.0
    assert body["ml_metrics"]["recommended_tier"] == 0
    assert body["ml_metrics"]["discount_cost"] == 0.0
    assert body["ml_metrics"]["net_expected_recovery"] == 0.0
    assert body["ml_metrics"]["expected_incremental_revenue"] == 0.0
    assert "MERCHANT_ACTION_VETO_NO_ACTION" in body["reason_codes"]


def test_naive_baseline_fixture_respects_the_final_merchant_action_veto(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_payment_link_runs(*_args, **_kwargs) -> dict:
        raise AssertionError("A merchant-vetoed baseline must not create a link")

    monkeypatch.setattr(main, "create_recovery_payment_link", fail_if_payment_link_runs)
    payload = _demo_payload(agent_enabled=False)
    payload["scenario"] = "predictive_vs_causal"
    payload["recovery_case"]["case_id"] = "case-veto-naive-baseline"
    payload["merchant_constraints"]["allowed_actions"] = ["no_action"]
    payload["candidate_actions"] = ["no_action"]

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["final_action"] == "no_action"
    assert body["short_url"] is None
    assert body["discount_cost"] == 0.0
    assert "MERCHANT_ACTION_VETO_NO_ACTION" in body["reason_codes"]


def test_policy_rollback_restores_the_last_known_good_action(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_payment_link(
        amount: float,
        _case_id: str,
        action_type: str,
        discount_percentage: float,
    ) -> dict:
        assert action_type == "payment_link"
        assert discount_percentage == 0.0
        return {
            "short_url": "https://rzp.io/i/rollback",
            "final_amount": amount,
        }

    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)
    payload = _demo_payload(agent_enabled=True)
    payload["scenario"] = "policy_rollback"
    payload["recovery_case"].update(
        {
            "case_id": "case-policy-rollback",
            "amount": 7_499.0,
            "customer_segment": "high_intent_repeat",
            "failure_class": "network_timeout",
            "case_age_hours": 72.0,
        }
    )

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    expected_reasons = [
        "CANARY_LIFT_DEGRADED",
        "PROMOTION_REJECTED",
        "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
    ]
    assert body["final_action"] == "payment_link"
    assert body["reason_codes"] == expected_reasons
    assert body["ml_metrics"]["reason_codes"] == expected_reasons
    assert body["ml_metrics"]["policy_version"] == "v3.5"
    assert body["current_policy_version"] == "v3.5"
    assert body["scenario_metrics"]["policy_status"] == "ROLLED_BACK"


def test_batch_learning_promotes_challenger_and_tags_subsequent_audit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = {"Idempotency-Key": "batch-promote-contract-test"}
    response = client.post(
        "/api/v1/admin/trigger_batch_learning",
        json={"trigger_source": "n8n_orchestrator", "force_scenario": "normal"},
        headers=headers,
    )
    duplicate = client.post(
        "/api/v1/admin/trigger_batch_learning",
        json={"trigger_source": "n8n_orchestrator", "force_scenario": "normal"},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body == duplicate.json()
    assert body["status"] == "success"
    assert body["champion_model"] == "v3.5"
    assert body["challenger_model"] == "v3.6"
    assert body["challenger_roi"] == 5.82
    assert body["incumbent_roi"] == 5.16
    assert body["quarantine_rate"] == 1.0
    assert body["promoted"] is True
    assert body["active_version"] == "v3.6"
    assert main.ACTIVE_POLICY_VERSION == "v3.6"

    db = main.SessionLocal()
    try:
        assert db.query(BatchPolicyRun).count() == 1
        persisted_run = db.get(BatchPolicyRun, body["run_id"])
        assert persisted_run is not None
        assert persisted_run.status == "PROMOTED"
    finally:
        db.close()

    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _retry_recommendation)
    payload = _decision_payload()
    payload["case_id"] = "case-after-v36-promotion"
    decision = client.post("/api/v1/decision/evaluate", json=payload)

    assert decision.status_code == 200
    decision_body = decision.json()
    assert decision_body["current_policy_version"] == "v3.6"
    assert decision_body["policy_version"] == "v3.6"
    assert "CHALLENGER_POLICY_V3.6_EXECUTED" in decision_body["reason_codes"]

    db = main.SessionLocal()
    try:
        audit = (
            db.query(AuditLog)
            .filter(AuditLog.case_id == "case-after-v36-promotion")
            .one()
        )
        assert audit.policy_version == "v3.6"
    finally:
        db.close()


def test_batch_learning_rollback_restores_champion_and_persists_trace(
    client: TestClient,
) -> None:
    client.post(
        "/api/v1/admin/trigger_batch_learning",
        json={"trigger_source": "manual_ui", "force_scenario": "normal"},
    )
    response = client.post(
        "/api/v1/admin/trigger_batch_learning",
        json={"trigger_source": "manual_ui", "force_scenario": "policy_rollback"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rollback"
    assert body["challenger_roi"] == 2.14
    assert body["quarantine_rate"] == 0.85
    assert body["promoted"] is False
    assert body["active_version"] == "v3.5"
    assert body["run_status"] == "REJECTED_ROLLBACK"
    assert body["rejection_reason"] == (
        "Canary lift degraded below margin safety threshold"
    )
    assert main.ACTIVE_POLICY_VERSION == "v3.5"

    state = client.get("/api/v1/admin/policy_state")
    assert state.status_code == 200
    assert state.json()["active_version"] == "v3.5"
    assert state.json()["latest_run"]["run_id"] == body["run_id"]

    db = main.SessionLocal()
    try:
        persisted_run = db.get(BatchPolicyRun, body["run_id"])
        assert persisted_run is not None
        assert persisted_run.status == "REJECTED_ROLLBACK"
        metadata = json.loads(persisted_run.metadata_json)
        assert metadata["reason_codes"] == [
            "CANARY_LIFT_DEGRADED",
            "PROMOTION_REJECTED",
            "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
        ]
    finally:
        db.close()


def test_promoted_challenger_calibrates_demo_offer_and_persists_version(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = client.post(
        "/api/v1/admin/trigger_batch_learning",
        json={"trigger_source": "manual_ui", "force_scenario": "normal"},
    )
    assert promotion.json()["active_version"] == "v3.6"

    link_calls: list[float] = []

    def fake_payment_link(
        amount: float,
        _case_id: str,
        _action_type: str,
        discount_percentage: float,
    ) -> dict:
        link_calls.append(discount_percentage)
        return {
            "short_url": "https://rzp.io/i/v36-calibrated",
            "final_amount": amount * (1.0 - discount_percentage / 100.0),
        }

    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _incentive_recommendation)
    monkeypatch.setattr(main, "create_recovery_payment_link", fake_payment_link)
    payload = _demo_payload(agent_enabled=True)
    payload["recovery_case"]["case_id"] = "case-v36-calibrated-offer"
    payload["recovery_case"]["customer_segment"] = "price_sensitive"

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["current_policy_version"] == "v3.6"
    assert body["ml_metrics"]["policy_version"] == "v3.6"
    assert body["ml_metrics"]["recommended_tier"] == 3
    assert body["ml_metrics"]["discount_cost"] == pytest.approx(75.0)
    assert "CHALLENGER_POLICY_V3.6_EXECUTED" in body["reason_codes"]
    assert body["short_url"] is None
    assert link_calls == []

    approval = client.post(
        "/api/v1/execute/approve_link",
        json={
            "case_id": body["case_id"],
            "approved_action": "incentive_link",
        },
    )
    assert approval.status_code == 200
    assert link_calls == [3.0]

    db = main.SessionLocal()
    try:
        audit = (
            db.query(AuditLog)
            .filter(AuditLog.case_id == "case-v36-calibrated-offer")
            .one()
        )
        assert audit.policy_version == "v3.6"
    finally:
        db.close()


def test_sandbox_scenario_continues_when_live_razorpay_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_link(*_args, **_kwargs) -> dict:
        raise razorpay_client.RazorpayConfigurationError("test credentials unavailable")

    monkeypatch.setattr(main, "create_recovery_payment_link", unavailable_link)
    payload = _demo_payload(agent_enabled=False)
    payload["scenario_id"] = "discount_wins_trap"
    payload["recovery_case"]["failure_class"] = "user_cancelled"

    response = client.post("/api/v1/execute/demo_toggle", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["short_url"] is None
    assert body["final_amount"] == payload["recovery_case"]["amount"]
    assert "RAZORPAY_LINK_UNAVAILABLE_SANDBOX" in body["reason_codes"]


def _signed_webhook_request(
    client: TestClient,
    payload: dict,
    secret: str,
    event_id: str = "evt-test-001",
):
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return client.post(
        "/api/v1/webhooks/razorpay",
        content=raw_body,
        headers={
            "content-type": "application/json",
            "x-razorpay-signature": signature,
            "x-razorpay-event-id": event_id,
        },
    )


def test_live_razorpay_webhook_rejects_invalid_signature(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")

    response = client.post(
        "/api/v1/webhooks/razorpay",
        content=b'{"event":"payment.failed"}',
        headers={
            "content-type": "application/json",
            "x-razorpay-signature": "invalid",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid Razorpay webhook signature."


def test_live_razorpay_webhook_ignores_authenticated_unhandled_event(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-webhook-secret"
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", secret)

    response = _signed_webhook_request(
        client,
        {"event": "payment.captured", "payload": {}},
        secret,
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ignored",
        "reason": "unhandled_event",
        "current_policy_version": "v3.5",
    }


def test_live_razorpay_webhook_evaluates_paise_payload_once(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-webhook-secret"
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", secret)
    monkeypatch.setattr(main, "evaluate_case_integrity", _trusted_signal)
    monkeypatch.setattr(main, "estimate_uplift", _retry_recommendation)

    captured_amounts: list[float] = []
    original_upsert = main._upsert_recovery_case

    def capture_upsert(db, payload):
        captured_amounts.append(payload.amount)
        return original_upsert(db, payload)

    monkeypatch.setattr(main, "_upsert_recovery_case", capture_upsert)
    payload = {
        "event": "payment.failed",
        "account_id": "acc_test_001",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_live_001",
                    "amount": 499900,
                    "contact": "+919999999999",
                    "email": "buyer@example.test",
                    "method": "card",
                    "error_description": "network timeout",
                    "created_at": 1788057529,
                }
            }
        },
    }

    first = _signed_webhook_request(
        client,
        payload,
        secret,
        event_id="evt-payment-failed-001",
    )
    duplicate = _signed_webhook_request(
        client,
        payload,
        secret,
        event_id="evt-payment-failed-001",
    )

    assert first.status_code == 200
    assert first.json() == {
        "status": "received",
        "action_taken": "retry",
        "current_policy_version": "v3.5",
    }
    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert captured_amounts == [4999.0]
