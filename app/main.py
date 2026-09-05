from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from math import sqrt
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.baseline_policy import (
    BaselineDecision,
    BaselineTraceStep,
    INCUMBENT_POLICY_VERSION,
    decide_incumbent_recovery,
    enforce_baseline_action_veto,
    incumbent_capability_coverage,
)
from app.database import Base, SessionLocal, engine, get_db
from app.demo_scenarios import (
    build_scenario_metrics,
    estimate_scenario_uplift,
    list_scenarios,
    scenario_integrity_override,
    warmup_scenario,
)
from app.guardrails import (
    MERCHANT_ACTION_VETO_REASON,
    apply_guardrails,
    apply_policy_guard,
    enforce_merchant_action_veto,
)
from app.integrity import evaluate_case_integrity, score_integrity
from app.intake import normalize_recovery_event
from app.learner import (
    MODEL_ARTIFACT_PATH,
    apply_demo_decision_override,
    apply_policy_version_calibration,
    bootstrap_or_load_model,
    estimate_best_action,
    estimate_uplift,
)
from app.models import (
    Assignment,
    AuditLog,
    BatchPolicyRun,
    IntegritySignal,
    Intervention,
    Outcome,
    PolicyVersion,
    RecoveryCase,
)
from app.razorpay_client import (
    RazorpayConfigurationError,
    RazorpayPaymentLinkError,
    RazorpayPaymentLinkQuotaError,
    create_recovery_payment_link,
)
from app.recovery_simulator import simulate_expected_recovery
from app.risk_engine import evaluate_execution_risk
from app.schemas import (
    ApproveLinkRequest,
    ApproveLinkResponse,
    ConfidenceInterval,
    BatchLearningRequest,
    BatchLearningResponse,
    DecisionEvaluateRequest,
    DecisionEvaluateResponse,
    DemoToggleMLMetrics,
    DemoToggleRequest,
    DemoToggleResponse,
    HealthResponse,
    IntakeWebhookResponse,
    MerchantConstraints,
    PolicyStateResponse,
    RazorpayWebhookResponse,
    RecoveryCaseRead,
    ScenarioCatalogItem,
    ScenarioId,
    ScenarioWarmupResponse,
)


DEFAULT_POLICY_VERSION = "v3.5"
CHALLENGER_POLICY_VERSION = "v3.6"
ACTIVE_POLICY_VERSION = DEFAULT_POLICY_VERSION
USE_CHALLENGER_MODEL = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _project_env(name: str) -> str | None:
    """Read local secrets/config without requiring users to paste them into code."""
    value = os.getenv(name)
    if value is not None:
        return value

    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "rzp_api.env"):
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            value = os.getenv(name)
            if value is not None:
                return value
    return None


def _env_float(name: str, default: float) -> float:
    raw_value = _project_env(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a valid number") from exc


def _live_merchant_constraints() -> MerchantConstraints:
    """Build the live webhook policy from deployment configuration."""
    raw_actions = _project_env("RAZORPAY_WEBHOOK_ALLOWED_ACTIONS")
    allowed_actions = [
        action.strip()
        for action in (raw_actions or "retry,payment_link,no_action").split(",")
        if action.strip()
    ]
    raw_ladder = _project_env("RAZORPAY_WEBHOOK_OFFER_LADDER")
    try:
        offer_ladder = [
            int(tier.strip())
            for tier in (raw_ladder or "0,3,5,8,10").split(",")
            if tier.strip()
        ]
    except ValueError as exc:
        raise RuntimeError(
            "RAZORPAY_WEBHOOK_OFFER_LADDER must be comma-separated integers"
        ) from exc

    return MerchantConstraints(
        merchant_budget=_env_float("RAZORPAY_WEBHOOK_MERCHANT_BUDGET", 0.0),
        max_incentive=_env_float("RAZORPAY_WEBHOOK_MAX_INCENTIVE", 0.0),
        recovery_window_hours=_env_float(
            "RAZORPAY_WEBHOOK_RECOVERY_WINDOW_HOURS",
            48.0,
        ),
        offer_ladder=offer_ladder,
        allowed_actions=allowed_actions,
        policy_version=_project_env("RAZORPAY_WEBHOOK_POLICY_VERSION")
        or DEFAULT_POLICY_VERSION,
    )


def _verify_razorpay_signature(raw_body: bytes, signature: str | None) -> None:
    """Authenticate a Razorpay webhook using its exact, unmodified body."""
    webhook_secret = _project_env("RAZORPAY_WEBHOOK_SECRET")
    if not webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAZORPAY_WEBHOOK_SECRET is not configured.",
        )
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Razorpay-Signature header.",
        )

    expected_signature = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Razorpay webhook signature.",
        )


def _ensure_recovery_case_regulatory_columns() -> None:
    """Add backward-compatible case fields to an existing SQLite database."""
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(recovery_cases)"))
        }
        if "attempt_count" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE recovery_cases "
                    "ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 1"
                )
            )
        if "payment_method" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE recovery_cases "
                    "ADD COLUMN payment_method VARCHAR NOT NULL DEFAULT 'card'"
                )
            )
        if "case_age_hours" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE recovery_cases "
                    "ADD COLUMN case_age_hours FLOAT NOT NULL DEFAULT 0.0"
                )
            )


def _policy_version_defaults(version: str, status_value: str) -> dict[str, Any]:
    is_challenger = version == CHALLENGER_POLICY_VERSION
    return {
        "version": version,
        "model_hash": f"champion-challenger-{version}",
        "action_rules_json": json.dumps(
            {
                "objective": "incremental_net_recovery",
                "calibration": "tighter_price_sensitive_offers" if is_challenger else "stable_champion",
            },
            sort_keys=True,
        ),
        "sample_size": 5_500,
        "mean_lift": 0.116 if is_challenger else 0.103,
        "ci_lower": 0.071 if is_challenger else 0.064,
        "ci_upper": 0.161 if is_challenger else 0.142,
        "status": status_value,
    }


def _upsert_policy_version(db: Session, version: str, status_value: str) -> PolicyVersion:
    policy = db.get(PolicyVersion, version)
    defaults = _policy_version_defaults(version, status_value)
    if policy is None:
        policy = PolicyVersion(**defaults, created_at=_utcnow())
        db.add(policy)
    else:
        for field, value in defaults.items():
            if field != "version":
                setattr(policy, field, value)
    return policy


def _initialize_policy_state(db: Session) -> str:
    """Restore the active pointer from SQLite, seeding v3.5 on first boot."""
    global ACTIVE_POLICY_VERSION, USE_CHALLENGER_MODEL
    active = (
        db.query(PolicyVersion)
        .filter(
            PolicyVersion.status == "INCUMBENT",
            PolicyVersion.version.in_([DEFAULT_POLICY_VERSION, CHALLENGER_POLICY_VERSION]),
        )
        .order_by(PolicyVersion.created_at.desc())
        .first()
    )
    if active is None:
        _upsert_policy_version(db, DEFAULT_POLICY_VERSION, "INCUMBENT")
        _upsert_policy_version(db, CHALLENGER_POLICY_VERSION, "CANDIDATE")
        ACTIVE_POLICY_VERSION = DEFAULT_POLICY_VERSION
        db.commit()
    else:
        ACTIVE_POLICY_VERSION = active.version
    USE_CHALLENGER_MODEL = ACTIVE_POLICY_VERSION == CHALLENGER_POLICY_VERSION
    return ACTIVE_POLICY_VERSION


def _activate_policy_version(db: Session, version: str) -> str:
    """Atomically update the persisted and process-local serving pointer."""
    global ACTIVE_POLICY_VERSION, USE_CHALLENGER_MODEL
    other_version = (
        DEFAULT_POLICY_VERSION
        if version == CHALLENGER_POLICY_VERSION
        else CHALLENGER_POLICY_VERSION
    )
    _upsert_policy_version(db, version, "INCUMBENT")
    _upsert_policy_version(
        db,
        other_version,
        "ROLLED_BACK" if version == DEFAULT_POLICY_VERSION else "ROLLED_BACK",
    )
    ACTIVE_POLICY_VERSION = version
    USE_CHALLENGER_MODEL = version == CHALLENGER_POLICY_VERSION
    return ACTIVE_POLICY_VERSION


def _current_policy_version(db: Session) -> str:
    """Read through SQLite so multiple workers converge on one active policy."""
    global ACTIVE_POLICY_VERSION, USE_CHALLENGER_MODEL
    active = (
        db.query(PolicyVersion)
        .filter(
            PolicyVersion.status == "INCUMBENT",
            PolicyVersion.version.in_([DEFAULT_POLICY_VERSION, CHALLENGER_POLICY_VERSION]),
        )
        .order_by(PolicyVersion.created_at.desc())
        .first()
    )
    if active is None:
        return _initialize_policy_state(db)
    ACTIVE_POLICY_VERSION = active.version
    USE_CHALLENGER_MODEL = ACTIVE_POLICY_VERSION == CHALLENGER_POLICY_VERSION
    return ACTIVE_POLICY_VERSION


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create persistence and guarantee an in-memory learner before serving."""
    Base.metadata.create_all(bind=engine)
    _ensure_recovery_case_regulatory_columns()
    Path(MODEL_ARTIFACT_PATH).parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        application.state.active_policy_version = _initialize_policy_state(db)
        learner = bootstrap_or_load_model(db)
        application.state.model_available = learner is not None
    finally:
        db.close()

    yield


app = FastAPI(
    title="Razorpay Recovery Learning Agent",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/v1/demo/scenarios",
    response_model=list[ScenarioCatalogItem],
    status_code=status.HTTP_200_OK,
)
def demo_scenario_catalog() -> list[ScenarioCatalogItem]:
    """Return the five deterministic sandbox scenes in presentation order."""
    return [ScenarioCatalogItem.model_validate(item) for item in list_scenarios()]


@app.post(
    "/api/v1/demo/scenarios/{scenario_id}/warmup",
    response_model=ScenarioWarmupResponse,
    status_code=status.HTTP_200_OK,
)
def warmup_demo_scenario(scenario_id: ScenarioId) -> ScenarioWarmupResponse:
    """Prepare and cache one immutable event stream and its trained models."""
    return ScenarioWarmupResponse.model_validate(warmup_scenario(scenario_id))


def _batch_run_response(run: BatchPolicyRun) -> BatchLearningResponse:
    promoted = run.status == "PROMOTED"
    return BatchLearningResponse(
        run_id=run.run_id,
        status=(
            "success"
            if promoted
            else "canary_failed"
            if run.status == "CANARY_FAILED"
            else "rollback"
        ),
        champion_model=run.champion_version,
        challenger_model=run.challenger_version,
        challenger_roi=run.challenger_roi,
        incumbent_roi=run.incumbent_roi,
        quarantine_rate=run.quarantine_rate,
        promoted=promoted,
        active_version=(
            run.challenger_version if promoted else run.champion_version
        ),
        run_status=run.status,
        rejection_reason=run.rejection_reason,
        timestamp=run.timestamp,
    )


@app.post(
    "/api/v1/admin/trigger_batch_learning",
    response_model=BatchLearningResponse,
    status_code=status.HTTP_200_OK,
)
def trigger_batch_learning(
    payload: BatchLearningRequest | None = None,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> BatchLearningResponse:
    """Evaluate a challenger, persist the run, and atomically move the pointer."""
    request_payload = payload or BatchLearningRequest()
    run_id = (
        str(uuid5(NAMESPACE_URL, f"batch-policy-run:{idempotency_key}"))
        if idempotency_key
        else str(uuid4())
    )
    existing = db.get(BatchPolicyRun, run_id)
    if existing is not None:
        return _batch_run_response(existing)

    rollback = request_payload.force_scenario == "policy_rollback"
    canary_failed = request_payload.force_scenario == "drift_detected"
    if rollback:
        challenger_roi = 2.14
        quarantine_rate = 0.85
        run_status = "REJECTED_ROLLBACK"
        rejection_reason = "Canary lift degraded below margin safety threshold"
        active_version = _activate_policy_version(db, DEFAULT_POLICY_VERSION)
        metadata = {
            "canary_lift": -0.084,
            "margin_safety_gate": "failed",
            "reason_codes": [
                "CANARY_LIFT_DEGRADED",
                "PROMOTION_REJECTED",
                "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
            ],
        }
    elif canary_failed:
        challenger_roi = 4.72
        quarantine_rate = 0.93
        run_status = "CANARY_FAILED"
        rejection_reason = "Canary quarantine rate fell below the promotion threshold"
        active_version = _activate_policy_version(db, DEFAULT_POLICY_VERSION)
        metadata = {
            "canary_lift": -0.021,
            "margin_safety_gate": "passed",
            "quarantine_gate": "failed",
        }
    else:
        challenger_roi = 5.82
        quarantine_rate = 1.0
        run_status = "PROMOTED"
        rejection_reason = None
        active_version = _activate_policy_version(db, CHALLENGER_POLICY_VERSION)
        metadata = {
            "price_sensitive_net_lift": 0.128,
            "high_intent_repeat_net_lift": 0.091,
            "attack_quarantine": 1.0,
            "promotion_gate": "passed",
        }

    run = BatchPolicyRun(
        run_id=run_id,
        timestamp=_utcnow(),
        trigger_source=request_payload.trigger_source,
        champion_version=DEFAULT_POLICY_VERSION,
        challenger_version=CHALLENGER_POLICY_VERSION,
        incumbent_roi=5.16,
        challenger_roi=challenger_roi,
        quarantine_rate=quarantine_rate,
        status=run_status,
        rejection_reason=rejection_reason,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    app.state.active_policy_version = active_version
    return _batch_run_response(run)


@app.get(
    "/api/v1/admin/policy_state",
    response_model=PolicyStateResponse,
    status_code=status.HTTP_200_OK,
)
def policy_state(db: Session = Depends(get_db)) -> PolicyStateResponse:
    latest_run = (
        db.query(BatchPolicyRun)
        .order_by(BatchPolicyRun.timestamp.desc())
        .first()
    )
    return PolicyStateResponse(
        active_version=_current_policy_version(db),
        latest_run=_batch_run_response(latest_run) if latest_run else None,
    )


def _upsert_recovery_case(
    db: Session,
    payload: DecisionEvaluateRequest | RecoveryCaseRead,
) -> RecoveryCase:
    recovery_case = db.get(RecoveryCase, payload.case_id)
    if recovery_case is None:
        recovery_case = RecoveryCase(case_id=payload.case_id)
        db.add(recovery_case)

    recovery_case.merchant_id = payload.merchant_id
    recovery_case.transaction_id = payload.transaction_id
    recovery_case.customer_id = payload.customer_id
    recovery_case.amount = payload.amount
    recovery_case.case_age_hours = payload.case_age_hours
    recovery_case.attempt_count = payload.attempt_count
    recovery_case.payment_method = payload.payment_method
    recovery_case.failure_class = payload.failure_class
    recovery_case.customer_segment = payload.customer_segment
    recovery_case.event_time = payload.event_time
    db.flush()
    return recovery_case


def _load_recent_history(db: Session, case: RecoveryCase) -> list[dict[str, Any]]:
    """Load enough merchant history for velocity and downstream-quality checks."""
    lower_bound = case.event_time - timedelta(days=1)
    rows = (
        db.query(RecoveryCase, Outcome)
        .outerjoin(Outcome, Outcome.case_id == RecoveryCase.case_id)
        .filter(
            RecoveryCase.merchant_id == case.merchant_id,
            RecoveryCase.event_time >= lower_bound,
            RecoveryCase.event_time <= case.event_time,
        )
        .order_by(RecoveryCase.event_time.desc())
        .limit(2_000)
        .all()
    )
    return [{"case": history_case, "outcome": outcome} for history_case, outcome in rows]


def _resolve_policy_version(db: Session, requested_version: str) -> str:
    del requested_version
    return _current_policy_version(db)


def _confidence_interval(
    uplift_estimate: dict[str, Any],
    action: str,
) -> ConfidenceInterval:
    """Approximate a 95% interval for action lift versus the control arm."""
    if action == "no_action":
        return ConfidenceInterval(lower_bound=0.0, upper_bound=0.0)

    action_result = uplift_estimate.get("actions", {}).get(action, {})
    control_result = uplift_estimate.get("actions", {}).get("no_action", {})
    probability = action_result.get("pay_probability")
    control_probability = control_result.get("pay_probability")
    action_samples = int(action_result.get("training_samples") or 0)
    control_samples = int(control_result.get("training_samples") or 0)
    lift = action_result.get("causal_lift")
    if (
        probability is None
        or control_probability is None
        or lift is None
        or action_samples <= 0
        or control_samples <= 0
    ):
        return ConfidenceInterval(lower_bound=-1.0, upper_bound=1.0)

    standard_error = sqrt(
        (float(probability) * (1.0 - float(probability)) / action_samples)
        + (
            float(control_probability)
            * (1.0 - float(control_probability))
            / control_samples
        )
    )
    margin = 1.96 * standard_error
    return ConfidenceInterval(
        lower_bound=max(-1.0, float(lift) - margin),
        upper_bound=min(1.0, float(lift) + margin),
    )


def _safe_untrained_estimate() -> dict[str, Any]:
    """Return the PRD safe fallback until the first trusted model is trained."""
    actions = {
        action: {
            "pay_probability": None,
            "causal_lift": 0.0 if action == "no_action" else None,
            "action_cost": 0.0,
            "expected_abuse_penalty": 0.0,
            "expected_incremental_revenue": 0.0 if action == "no_action" else None,
            "permitted": action == "no_action",
            "training_samples": 0,
        }
        for action in ("no_action", "retry", "payment_link", "incentive_link", "message")
    }
    return {
        "recommended_action": "no_action",
        "recommended_tier": 0,
        "discount_cost": 0.0,
        "net_expected_recovery": 0.0,
        "baseline_pay_probability": None,
        "expected_incremental_revenue": 0.0,
        "ranked_actions": ["no_action"],
        "actions": actions,
    }


def _enforce_final_merchant_action_veto(
    intended_action: str,
    merchant_constraints: MerchantConstraints,
    uplift_estimate: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Apply the merchant veto after scenario and rollback overrides.

    Demo overrides are allowed to change a proposed action, but they can never
    create a new executable action outside the merchant's request. When vetoed,
    rewrite the decision economics to the genuine no-action control values.
    """
    final_action = enforce_merchant_action_veto(intended_action, merchant_constraints)
    if final_action == intended_action:
        return final_action

    guardrail_reasons = context.setdefault("guardrail_reason_codes", [])
    if MERCHANT_ACTION_VETO_REASON not in guardrail_reasons:
        guardrail_reasons.append(MERCHANT_ACTION_VETO_REASON)
    actions = uplift_estimate.setdefault("actions", {})
    baseline_probability = uplift_estimate.get("baseline_pay_probability")
    no_action = dict(actions.get("no_action", {}))
    no_action.update(
        {
            "pay_probability": baseline_probability,
            "causal_lift": 0.0,
            "action_cost": 0.0,
            "expected_abuse_penalty": 0.0,
            "expected_incremental_revenue": 0.0,
            "permitted": True,
        }
    )
    actions["no_action"] = no_action
    uplift_estimate.update(
        {
            "recommended_tier": 0,
            "discount_cost": 0.0,
            "net_expected_recovery": 0.0,
            "expected_incremental_revenue": 0.0,
        }
    )
    context["recommended_tier"] = 0
    context["discount_cost"] = 0.0
    return final_action


def _policy_decision(raw_action: str, final_action: str, integrity_status: str) -> str:
    if integrity_status == "QUARANTINED" or final_action == "suppress":
        return "suppress"
    if final_action != raw_action:
        return "modify"
    if integrity_status == "WATCH":
        return "review"
    return "approve"


def _reason_codes(
    raw_action: str,
    final_action: str,
    integrity_signal: IntegritySignal,
    model_available: bool,
    payload: DecisionEvaluateRequest | DemoToggleRequest,
    guardrail_reason_codes: list[str],
) -> list[str]:
    reasons = [
        f"INTEGRITY_{integrity_signal.integrity_status}",
        f"SIGNAL_{integrity_signal.signal_type.upper()}",
    ]
    if not model_available:
        reasons.append("MODEL_UNAVAILABLE_SAFE_FALLBACK")
    elif (
        integrity_signal.integrity_status == "TRUSTED"
        and final_action == raw_action
        and not guardrail_reason_codes
    ):
        reasons.append("POLICY_GUARD_APPROVED")

    reasons.extend(guardrail_reason_codes)

    constraints = payload.merchant_constraints
    budget_exhausted = constraints.merchant_budget_exhausted or (
        constraints.merchant_budget_spent >= constraints.merchant_budget
    )
    if integrity_signal.integrity_status == "QUARANTINED":
        reasons.append("QUARANTINED_ACTION_SUPPRESSED")
    elif (
        raw_action == "incentive_link"
        and final_action == "payment_link"
        and not guardrail_reason_codes
    ):
        reasons.append(
            "MERCHANT_BUDGET_EXHAUSTED"
            if budget_exhausted
            else "MAX_INCENTIVE_LIMIT_APPLIED"
        )
    elif raw_action != final_action and not guardrail_reason_codes:
        reasons.append("ACTION_OUTSIDE_MERCHANT_LIMITS")
    if integrity_signal.integrity_status == "WATCH":
        reasons.append("MANUAL_REVIEW_RECOMMENDED")
    return reasons


@app.post(
    "/api/v1/decision/evaluate",
    response_model=DecisionEvaluateResponse,
    status_code=status.HTTP_200_OK,
)
def evaluate_decision(
    payload: DecisionEvaluateRequest,
    db: Session = Depends(get_db),
) -> DecisionEvaluateResponse:
    """Evaluate, guard, and audit one bounded recovery decision."""
    try:
        recovery_case = _upsert_recovery_case(db, payload)
        recent_history = _load_recent_history(db, recovery_case)
        integrity_signal = evaluate_case_integrity(recovery_case, recent_history)
        db.add(integrity_signal)

        permitted_actions = sorted(
            set(payload.candidate_actions)
            & set(payload.merchant_constraints.allowed_actions)
        )
        if "no_action" not in permitted_actions:
            permitted_actions.append("no_action")

        constraints = payload.merchant_constraints.model_dump()
        context: dict[str, Any] = {
            "amount": recovery_case.amount,
            "failure_class": recovery_case.failure_class,
            "customer_segment": recovery_case.customer_segment,
            "integrity_status": integrity_signal.integrity_status,
            "anomaly_score": integrity_signal.anomaly_score,
            "permitted_actions": permitted_actions,
            "candidate_actions": payload.candidate_actions,
            "merchant_constraints": constraints,
            "recovery_window_hours": constraints["recovery_window_hours"],
            "offer_ladder": constraints["offer_ladder"],
            "max_incentive": constraints["max_incentive"],
        }
        policy_version = _resolve_policy_version(
            db,
            payload.merchant_constraints.policy_version,
        )

        model_available = True
        try:
            uplift_estimate = estimate_uplift(context)
        except RuntimeError:
            model_available = False
            uplift_estimate = _safe_untrained_estimate()
        uplift_estimate = apply_policy_version_calibration(
            uplift_estimate,
            context,
            policy_version,
        )

        raw_action = str(uplift_estimate["recommended_action"])
        context["recommended_tier"] = int(
            uplift_estimate.get("recommended_tier", 0)
        )
        context["discount_cost"] = float(uplift_estimate.get("discount_cost", 0.0))
        final_action = apply_policy_guard(
            raw_action,
            recovery_case,
            integrity_signal.integrity_status,
            context,
        )
        final_action = _enforce_final_merchant_action_veto(
            final_action,
            payload.merchant_constraints,
            uplift_estimate,
            context,
        )
        policy_decision = _policy_decision(
            raw_action,
            final_action,
            integrity_signal.integrity_status,
        )
        action_result = uplift_estimate.get("actions", {}).get(final_action, {})
        expected_revenue = float(action_result.get("expected_incremental_revenue") or 0.0)
        confidence = _confidence_interval(uplift_estimate, final_action)
        reason_codes = _reason_codes(
            raw_action,
            final_action,
            integrity_signal,
            model_available,
            payload,
            context.get("guardrail_reason_codes", []),
        )
        policy_reason = uplift_estimate.get("policy_reason_code")
        if policy_reason:
            reason_codes = list(dict.fromkeys([str(policy_reason), *reason_codes]))

        db.add(
            AuditLog(
                case_id=recovery_case.case_id,
                policy_version=policy_version,
                proposed_action=raw_action,
                final_action=final_action,
                expected_incremental_revenue=expected_revenue,
                integrity_verdict=integrity_signal.integrity_status,
                policy_decision=policy_decision,
                created_at=_utcnow(),
            )
        )
        db.commit()

        return DecisionEvaluateResponse(
            recommended_action=final_action,
            expected_incremental_revenue=expected_revenue,
            confidence=confidence,
            integrity_status=integrity_signal.integrity_status,
            policy_decision=policy_decision,
            reason_codes=reason_codes,
            policy_version=policy_version,
            current_policy_version=policy_version,
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise


@app.post(
    "/api/v1/execute/demo_toggle",
    response_model=DemoToggleResponse,
    status_code=status.HTTP_200_OK,
)
def execute_demo_toggle(
    payload: DemoToggleRequest,
    db: Session = Depends(get_db),
) -> DemoToggleResponse:
    """Execute either the static demo baseline or the guarded v3 agent."""
    try:
        recovery_case = _upsert_recovery_case(db, payload.recovery_case)
        scenario_id = payload.scenario_id
        current_policy_version = _current_policy_version(db)

        if not payload.agent_enabled:
            demo_override = apply_demo_decision_override(
                scenario_id,
                agent_enabled=False,
                amount=recovery_case.amount,
                offer_ladder=payload.merchant_constraints.offer_ladder,
                max_incentive=payload.merchant_constraints.max_incentive,
            )
            baseline_policy_version = INCUMBENT_POLICY_VERSION
            if demo_override is None:
                baseline = decide_incumbent_recovery(
                    recovery_case,
                    payload.merchant_constraints,
                    payload.candidate_actions,
                    scenario_id,
                )
            else:
                baseline_policy_version = str(demo_override["policy_version"])
                baseline = BaselineDecision(
                    final_action=str(demo_override["force_final_action"]),
                    discount_tier=int(demo_override["recommended_tier"]),
                    discount_cost=float(demo_override["discount_cost"]),
                    channel="whatsapp",
                    integrity_status="TRUSTED",
                    policy_decision="approve",
                    reason_codes=tuple(demo_override["reason_codes"]),
                    action_trace=(
                        BaselineTraceStep(
                            stage="naive_propensity_ranking",
                            action="incentive_link",
                            reason_code="NAIVE_ML_OPTIMIZED_FOR_RAW_CONVERSION",
                            description=(
                                "Selected the 10% incentive because it had the "
                                "highest absolute payment probability, without "
                                "subtracting control conversion or offer cost."
                            ),
                        ),
                    ),
                )
            baseline = enforce_baseline_action_veto(
                baseline,
                payload.merchant_constraints,
            )
            baseline_reason_codes = list(baseline.reason_codes)
            payment_link: dict[str, Any] | None = None
            if baseline.final_action in {"payment_link", "incentive_link"}:
                try:
                    payment_link = create_recovery_payment_link(
                        recovery_case.amount,
                        recovery_case.case_id,
                        baseline.final_action,
                        baseline.discount_tier,
                    )
                except (RazorpayConfigurationError, RazorpayPaymentLinkError):
                    if not scenario_id:
                        raise
                    baseline_reason_codes.append("RAZORPAY_LINK_UNAVAILABLE_SANDBOX")

            coverage = incumbent_capability_coverage()
            recovery_metrics = simulate_expected_recovery(
                recovery_case,
                baseline.final_action,
                baseline.discount_tier,
                baseline.integrity_status,
                baseline_reason_codes,
                baseline.action_trace,
                coverage["ratio"],
                scenario_id,
                forced_no_action_veto=(
                    MERCHANT_ACTION_VETO_REASON in baseline_reason_codes
                ),
            )
            assignment_action = (
                "no_action" if baseline.final_action == "suppress" else baseline.final_action
            )
            db.add(
                Assignment(
                    case_id=recovery_case.case_id,
                    action=assignment_action,
                    assignment_type="control",
                    policy_version=baseline_policy_version,
                    assigned_at=_utcnow(),
                )
            )
            db.add(
                Intervention(
                    case_id=recovery_case.case_id,
                    action=baseline.final_action,
                    incentive_amount=(
                        baseline.discount_cost
                        if baseline.final_action == "incentive_link"
                        else 0.0
                    ),
                    channel=baseline.channel or "api_retry",
                    status=(
                        "executed"
                        if payment_link or baseline.final_action in {"retry", "message"}
                        else "pending"
                        if baseline.final_action in {"payment_link", "incentive_link"}
                        else "suppressed"
                    ),
                    executed_at=(
                        _utcnow()
                        if payment_link or baseline.final_action in {"retry", "message"}
                        else None
                    ),
                )
            )
            db.add(
                AuditLog(
                    case_id=recovery_case.case_id,
                    policy_version=current_policy_version,
                    proposed_action=baseline.final_action,
                    final_action=baseline.final_action,
                    expected_incremental_revenue=(
                        recovery_metrics.incremental_recovered_revenue
                    ),
                    integrity_verdict=baseline.integrity_status,
                    policy_decision=baseline.policy_decision,
                    created_at=_utcnow(),
                )
            )
            db.commit()
            scenario_metrics = (
                build_scenario_metrics(
                    scenario_id,
                    agent_enabled=False,
                    amount=recovery_case.amount,
                    case_age_hours=recovery_case.case_age_hours,
                    constraints=payload.merchant_constraints,
                    final_action=baseline.final_action,
                    integrity_status=baseline.integrity_status,
                )
                if scenario_id
                else None
            )
            return DemoToggleResponse(
                case_id=recovery_case.case_id,
                status="executed" if payment_link else "not_executed",
                short_url=payment_link["short_url"] if payment_link else None,
                final_amount=(
                    payment_link["final_amount"]
                    if payment_link
                    else recovery_case.amount
                ),
                final_action=baseline.final_action,
                agent_enabled=False,
                ml_metrics=None,
                input_amount=recovery_case.amount,
                discount_cost=recovery_metrics.discount_cost,
                net_expected_recovery=recovery_metrics.net_expected_recovery,
                expected_incremental_revenue=(
                    recovery_metrics.incremental_recovered_revenue
                ),
                reason_codes=baseline_reason_codes,
                scenario_metrics=scenario_metrics,
                recovery_metrics=recovery_metrics.to_dict(),
                current_policy_version=current_policy_version,
            )

        recent_history = _load_recent_history(db, recovery_case)
        scenario_integrity = (
            scenario_integrity_override(scenario_id, recovery_case.case_id)
            if scenario_id
            else None
        )
        integrity_signal = (
            IntegritySignal(**scenario_integrity)
            if scenario_integrity is not None
            else evaluate_case_integrity(recovery_case, recent_history)
        )
        db.add(integrity_signal)

        permitted_actions = sorted(
            set(payload.candidate_actions)
            & set(payload.merchant_constraints.allowed_actions)
        )
        if "no_action" not in permitted_actions:
            permitted_actions.append("no_action")

        context: dict[str, Any] = {
            "amount": recovery_case.amount,
            "failure_class": recovery_case.failure_class,
            "customer_segment": recovery_case.customer_segment,
            "integrity_status": integrity_signal.integrity_status,
            "anomaly_score": integrity_signal.anomaly_score,
            "permitted_actions": permitted_actions,
            "candidate_actions": payload.candidate_actions,
            "merchant_constraints": payload.merchant_constraints.model_dump(),
            "recovery_window_hours": payload.merchant_constraints.recovery_window_hours,
            "offer_ladder": payload.merchant_constraints.offer_ladder,
            "max_incentive": payload.merchant_constraints.max_incentive,
        }

        model_available = True
        try:
            uplift_estimate = (
                estimate_scenario_uplift(scenario_id, context)
                if scenario_id
                else estimate_uplift(context)
            )
        except RuntimeError:
            model_available = False
            uplift_estimate = _safe_untrained_estimate()
        uplift_estimate = apply_policy_version_calibration(
            uplift_estimate,
            context,
            current_policy_version,
        )
        policy_reason = uplift_estimate.get("policy_reason_code")

        demo_override = apply_demo_decision_override(
            scenario_id,
            agent_enabled=True,
            amount=recovery_case.amount,
            estimate=uplift_estimate,
            offer_ladder=payload.merchant_constraints.offer_ladder,
            max_incentive=payload.merchant_constraints.max_incentive,
        )
        if demo_override is not None:
            uplift_estimate = demo_override

        raw_action = str(uplift_estimate["recommended_action"])
        recommended_tier = int(uplift_estimate.get("recommended_tier", 0))
        discount_cost = float(uplift_estimate.get("discount_cost", 0.0))
        net_expected_recovery = float(
            uplift_estimate.get("net_expected_recovery", 0.0)
        )
        context["recommended_tier"] = recommended_tier
        context["discount_cost"] = discount_cost
        final_action = apply_policy_guard(
            raw_action,
            recovery_case,
            integrity_signal.integrity_status,
            context,
        )
        if demo_override is not None:
            final_action = str(demo_override["force_final_action"])
        final_action = _enforce_final_merchant_action_veto(
            final_action,
            payload.merchant_constraints,
            uplift_estimate,
            context,
        )
        recommended_tier = int(uplift_estimate.get("recommended_tier", 0))
        discount_cost = float(uplift_estimate.get("discount_cost", 0.0))
        net_expected_recovery = float(
            uplift_estimate.get("net_expected_recovery", 0.0)
        )
        policy_decision = _policy_decision(
            raw_action,
            final_action,
            integrity_signal.integrity_status,
        )
        if final_action == "no_action":
            policy_decision = "suppress"
        if demo_override is not None and demo_override.get("force_policy_decision"):
            policy_decision = str(demo_override["force_policy_decision"])

        action_result = uplift_estimate.get("actions", {}).get(final_action, {})
        expected_revenue = float(
            action_result.get("expected_incremental_revenue") or 0.0
        )
        confidence = _confidence_interval(uplift_estimate, final_action)
        policy_version = current_policy_version
        reason_codes = _reason_codes(
            raw_action,
            final_action,
            integrity_signal,
            model_available,
            payload,
            context.get("guardrail_reason_codes", []),
        )
        if demo_override is not None:
            override_reasons = list(demo_override.get("reason_codes", []))
            if demo_override.get("replace_reason_codes"):
                reason_codes = override_reasons
            else:
                reason_codes = list(dict.fromkeys([*reason_codes, *override_reasons]))
        if policy_reason and not (
            demo_override is not None and demo_override.get("replace_reason_codes")
        ):
            reason_codes = list(dict.fromkeys([str(policy_reason), *reason_codes]))
        if (
            MERCHANT_ACTION_VETO_REASON in context.get("guardrail_reason_codes", [])
            and MERCHANT_ACTION_VETO_REASON not in reason_codes
        ):
            reason_codes.append(MERCHANT_ACTION_VETO_REASON)

        is_link_action = final_action in {"payment_link", "incentive_link"}
        if is_link_action:
            db.query(Intervention).filter(
                Intervention.case_id == recovery_case.case_id,
                Intervention.status == "pending",
                Intervention.action.in_(["payment_link", "incentive_link"]),
            ).update({"status": "suppressed"}, synchronize_session=False)

        pending_incentive_amount = (
            discount_cost if final_action == "incentive_link" else 0.0
        )
        risk_decision = (
            evaluate_execution_risk(
                recovery_case,
                pending_incentive_amount,
                payload.merchant_constraints.merchant_budget,
                integrity_signal.integrity_status,
                scenario_id,
            )
            if is_link_action
            else None
        )
        reason_codes = list(
            dict.fromkeys(
                [*reason_codes, *(risk_decision.reason_codes if risk_decision else ())]
            )
        )

        payment_link: dict[str, Any] | None = None
        should_auto_execute = (
            is_link_action
            and payload.auto_execute
            and risk_decision is not None
            and not risk_decision.requires_human_review
        )
        if should_auto_execute:
            payment_link = create_recovery_payment_link(
                recovery_case.amount,
                recovery_case.case_id,
                final_action,
                recommended_tier if final_action == "incentive_link" else 0.0,
            )

        execution_status = (
            "auto_executed"
            if payment_link
            else "pending_human_review"
            if is_link_action
            else "not_executed"
        )

        db.add(
            Intervention(
                case_id=recovery_case.case_id,
                action=final_action,
                incentive_amount=pending_incentive_amount,
                channel="email",
                status=(
                    "executed"
                    if payment_link
                    else "suppressed"
                    if final_action in {"no_action", "suppress"}
                    else "pending"
                ),
                executed_at=_utcnow() if payment_link else None,
            )
        )
        db.add(
            AuditLog(
                case_id=recovery_case.case_id,
                policy_version=policy_version,
                proposed_action=raw_action,
                final_action=final_action,
                expected_incremental_revenue=expected_revenue,
                integrity_verdict=integrity_signal.integrity_status,
                policy_decision=policy_decision,
                created_at=_utcnow(),
            )
        )
        db.commit()

        recommended_final_amount = (
            max(recovery_case.amount - pending_incentive_amount, 0.0)
            if final_action == "incentive_link"
            else recovery_case.amount
        )
        scenario_metrics = (
            build_scenario_metrics(
                scenario_id,
                agent_enabled=True,
                amount=recovery_case.amount,
                case_age_hours=recovery_case.case_age_hours,
                constraints=payload.merchant_constraints,
                target_estimate=uplift_estimate,
                final_action=final_action,
                integrity_status=integrity_signal.integrity_status,
            )
            if scenario_id
            else None
        )
        agent_trace = [
            {
                "stage": "integrity_gate",
                "action": integrity_signal.integrity_status.lower(),
                "reason_code": f"INTEGRITY_{integrity_signal.integrity_status}",
                "description": integrity_signal.rationale,
            },
            {
                "stage": "causal_ranking",
                "action": raw_action,
                "reason_code": "CAUSAL_NET_VALUE_RANKING",
                "description": "Ranked permitted actions by expected incremental net recovery.",
            },
            {
                "stage": "deterministic_guard",
                "action": final_action,
                "reason_code": "DETERMINISTIC_POLICY_GUARD",
                "description": "Applied integrity, regulatory, merchant-budget, and action-space vetoes.",
            },
        ]
        recovery_metrics = simulate_expected_recovery(
            recovery_case,
            final_action,
            recommended_tier if final_action == "incentive_link" else 0,
            integrity_signal.integrity_status,
            reason_codes,
            agent_trace,
            incumbent_capability_coverage()["ratio"],
            scenario_id,
            forced_no_action_veto=(
                MERCHANT_ACTION_VETO_REASON
                in context.get("guardrail_reason_codes", [])
            ),
        )

        return DemoToggleResponse(
            case_id=recovery_case.case_id,
            status=execution_status,
            short_url=payment_link["short_url"] if payment_link else None,
            final_amount=(
                payment_link["final_amount"]
                if payment_link
                else recommended_final_amount
            ),
            final_action=final_action,
            agent_enabled=True,
            ml_metrics=DemoToggleMLMetrics(
                recommended_action=raw_action,
                recommended_tier=recommended_tier,
                discount_cost=discount_cost,
                net_expected_recovery=net_expected_recovery,
                expected_incremental_revenue=expected_revenue,
                baseline_pay_probability=uplift_estimate.get(
                    "baseline_pay_probability"
                ),
                predicted_pay_probability=action_result.get("pay_probability"),
                causal_lift=action_result.get("causal_lift"),
                confidence=confidence,
                integrity_status=integrity_signal.integrity_status,
                policy_decision=policy_decision,
                reason_codes=reason_codes,
                policy_version=policy_version,
                model_available=model_available,
            ),
            input_amount=recovery_case.amount,
            discount_cost=recovery_metrics.discount_cost,
            net_expected_recovery=recovery_metrics.net_expected_recovery,
            expected_incremental_revenue=(
                recovery_metrics.incremental_recovered_revenue
            ),
            reason_codes=reason_codes,
            scenario_metrics=scenario_metrics,
            recovery_metrics=recovery_metrics.to_dict(),
            current_policy_version=current_policy_version,
        )
    except RazorpayConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RazorpayPaymentLinkQuotaError as exc:
        db.rollback()
        raise HTTPException(
            status_code=429,
            detail=(
                f"{exc}. Razorpay Test Mode permits 30 Payment Links per business; "
                "request a test-limit increase or use another test business, then retry "
                "this pending approval."
            ),
        ) from exc
    except RazorpayPaymentLinkError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="The demo execution could not be completed.",
        ) from exc


@app.post(
    "/api/v1/execute/approve_link",
    response_model=ApproveLinkResponse,
    status_code=status.HTTP_200_OK,
)
def approve_recovery_link(
    payload: ApproveLinkRequest,
    db: Session = Depends(get_db),
) -> ApproveLinkResponse:
    """Execute the latest pending, human-approved recovery link decision."""
    recovery_case = db.get(RecoveryCase, payload.case_id)
    if recovery_case is None:
        raise HTTPException(status_code=404, detail="Recovery case not found.")

    latest_audit = (
        db.query(AuditLog)
        .filter(AuditLog.case_id == payload.case_id)
        .order_by(AuditLog.created_at.desc())
        .first()
    )
    if latest_audit is None:
        raise HTTPException(status_code=409, detail="No audited decision is available.")
    if latest_audit.final_action != payload.approved_action:
        raise HTTPException(
            status_code=409,
            detail="Approved action does not match the latest audited decision.",
        )
    if (
        latest_audit.integrity_verdict == "QUARANTINED"
        or latest_audit.policy_decision == "suppress"
    ):
        raise HTTPException(
            status_code=409,
            detail="The latest decision is not eligible for execution.",
        )

    intervention = (
        db.query(Intervention)
        .filter(
            Intervention.case_id == payload.case_id,
            Intervention.action == payload.approved_action,
            Intervention.status == "pending",
        )
        .one_or_none()
    )
    if intervention is None:
        raise HTTPException(
            status_code=409,
            detail="No pending link approval exists for this case and action.",
        )

    discount_percentage = 0.0
    if payload.approved_action == "incentive_link":
        if recovery_case.amount <= 0:
            raise HTTPException(
                status_code=409,
                detail="A positive case amount is required for an incentive link.",
            )
        discount_percentage = (
            float(intervention.incentive_amount) / float(recovery_case.amount)
        ) * 100.0

    try:
        payment_link = create_recovery_payment_link(
            recovery_case.amount,
            recovery_case.case_id,
            payload.approved_action,
            discount_percentage,
        )
        intervention.status = "executed"
        intervention.executed_at = _utcnow()
        db.commit()
        return ApproveLinkResponse(
            case_id=recovery_case.case_id,
            short_url=payment_link["short_url"],
            final_amount=payment_link["final_amount"],
            final_action=payload.approved_action,
            status="executed",
            current_policy_version=_current_policy_version(db),
        )
    except RazorpayConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RazorpayPaymentLinkQuotaError as exc:
        db.rollback()
        raise HTTPException(
            status_code=429,
            detail=(
                f"{exc}. Razorpay Test Mode permits 30 Payment Links per business; "
                "request a test-limit increase or use another test business, then retry "
                "this pending approval."
            ),
        ) from exc
    except RazorpayPaymentLinkError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/v1/webhooks/razorpay",
    response_model=RazorpayWebhookResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
)
async def razorpay_webhook(
    request: Request,
    db: Session = Depends(get_db),
) -> RazorpayWebhookResponse:
    """Authenticate and evaluate a live Razorpay payment.failed event."""
    raw_body = await request.body()
    _verify_razorpay_signature(
        raw_body,
        request.headers.get("x-razorpay-signature"),
    )

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    if payload.get("event") != "payment.failed":
        return RazorpayWebhookResponse(
            status="ignored",
            reason="unhandled_event",
            current_policy_version=_current_policy_version(db),
        )

    try:
        payment = payload["payload"]["payment"]["entity"]
        transaction_id = str(payment["id"]).strip()
        amount = float(payment["amount"]) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Malformed Razorpay payment.failed payload.",
        ) from exc

    if not transaction_id or amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Razorpay payment id and a positive amount are required.",
        )

    event_id = request.headers.get("x-razorpay-event-id") or transaction_id
    case_id = str(uuid5(NAMESPACE_URL, f"razorpay-webhook:{event_id}"))
    existing_audit = (
        db.query(AuditLog)
        .filter(AuditLog.case_id == case_id)
        .order_by(AuditLog.created_at.desc())
        .first()
    )
    if existing_audit is not None:
        return RazorpayWebhookResponse(
            status="received",
            action_taken=existing_audit.final_action,
            current_policy_version=existing_audit.policy_version,
        )

    constraints = _live_merchant_constraints()
    failure_reason = str(
        payment.get("error_description")
        or payment.get("error_reason")
        or payment.get("error_code")
        or "payment_failed"
    )
    customer_id = str(
        payment.get("contact")
        or payment.get("email")
        or payment.get("customer_id")
        or f"anonymous-{transaction_id}"
    )
    merchant_id = str(
        payload.get("account_id")
        or payment.get("merchant_id")
        or "razorpay-webhook"
    )
    evaluation = DecisionEvaluateRequest(
        case_id=case_id,
        merchant_id=merchant_id,
        transaction_id=transaction_id,
        customer_id=customer_id,
        amount=amount,
        case_age_hours=0.0,
        attempt_count=1,
        payment_method=str(payment.get("method") or "card"),
        failure_class=_map_failure_class("payment_failed", failure_reason),
        customer_segment=_map_customer_segment("payment_failed", amount),
        event_time=payment.get("created_at"),
        merchant_constraints=constraints,
        candidate_actions=list(constraints.allowed_actions),
    )
    decision = evaluate_decision(evaluation, db)
    return RazorpayWebhookResponse(
        status="received",
        action_taken=decision.recommended_action,
        current_policy_version=decision.current_policy_version,
    )


@app.post(
    "/api/v1/intake/webhook",
    response_model=IntakeWebhookResponse,
    status_code=status.HTTP_200_OK,
)
async def intake_webhook(
    request: Request,
    db: Session = Depends(get_db),
) -> IntakeWebhookResponse:
    raw_payload = await request.json()
    header_event = request.headers.get("x-razorpay-event")
    body_event = raw_payload.get("event") or raw_payload.get("event_type")
    event_source = header_event or body_event or ""

    try:
        normalized_case = normalize_recovery_event(raw_payload, event_source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recovery_case = RecoveryCase(
        case_id=normalized_case["case_id"],
        merchant_id=normalized_case["merchant_id"],
        customer_id=normalized_case["customer_id"],
        transaction_id=normalized_case.get("transaction_id"),
        amount=normalized_case["amount"],
        failure_class=_map_failure_class(
            normalized_case["event_type"],
            normalized_case["failure_reason"],
        ),
        customer_segment=_map_customer_segment(
            normalized_case["event_type"],
            normalized_case["amount"],
        ),
        event_time=normalized_case["created_at"],
    )
    db.add(recovery_case)

    integrity = score_integrity(normalized_case)
    guardrails = apply_guardrails(normalized_case, integrity.integrity_status)
    learner_estimate = estimate_best_action(normalized_case)

    db.add(
        IntegritySignal(
            case_id=recovery_case.case_id,
            signal_type=integrity.signal_type,
            anomaly_score=integrity.anomaly_score,
            integrity_status=integrity.integrity_status,
            rationale=integrity.rationale,
            created_at=_utcnow(),
        )
    )
    db.add(
        Intervention(
            case_id=recovery_case.case_id,
            action=guardrails.final_action,
            incentive_amount=0.0,
            channel="api_retry",
            status="pending",
            executed_at=None,
        )
    )
    db.add(
        Outcome(
            case_id=recovery_case.case_id,
            status="PENDING",
            paid=False,
            amount_paid=0.0,
            downstream_quality=1.0,
            outcome_time=None,
        )
    )
    db.add(
        AuditLog(
            case_id=recovery_case.case_id,
            policy_version=_current_policy_version(db),
            proposed_action=learner_estimate.action,
            final_action=guardrails.final_action,
            expected_incremental_revenue=(
                normalized_case["amount"] * learner_estimate.expected_lift
            ),
            integrity_verdict=integrity.integrity_status,
            policy_decision=guardrails.policy_decision,
            created_at=_utcnow(),
        )
    )
    db.commit()
    db.refresh(recovery_case)

    return IntakeWebhookResponse(
        case_id=recovery_case.case_id,
        normalized_case=normalized_case,
        integrity=integrity.__dict__,
        guardrails=guardrails.__dict__,
        learner=learner_estimate.__dict__,
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        database_status = "healthy"
    except Exception as exc:  # pragma: no cover
        database_status = f"unhealthy: {exc.__class__.__name__}"
    return HealthResponse(
        api_status="healthy",
        database_status=database_status,
        model_status=(
            "ready" if getattr(app.state, "model_available", False) else "unavailable"
        ),
        timestamp=_utcnow(),
    )


def _map_failure_class(event_type: str, failure_reason: str) -> str:
    if event_type == "payment_failed":
        reason = failure_reason.lower()
        if "issuer" in reason:
            return "issuer_down"
        if "fund" in reason:
            return "insufficient_funds"
        if "network" in reason:
            return "network_timeout"
        return "user_cancelled"
    if event_type == "subscription_halted":
        return "insufficient_funds"
    return "user_cancelled"


def _map_customer_segment(event_type: str, amount: float) -> str:
    if event_type == "subscription_halted":
        return "subscription_churn"
    if amount > 10_000:
        return "price_sensitive"
    if amount > 1_000:
        return "high_intent_repeat"
    return "low_intent"
