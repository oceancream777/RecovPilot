from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

FailureClass = Literal["issuer_down", "insufficient_funds", "network_timeout", "user_cancelled"]
CustomerSegment = Literal["high_intent_repeat", "price_sensitive", "subscription_churn", "low_intent"]
AssignmentAction = Literal["no_action", "retry", "payment_link", "incentive_link", "message"]
AssignmentType = Literal["treatment", "control"]
InterventionChannel = Literal["sms", "whatsapp", "email", "api_retry"]
InterventionStatus = Literal["pending", "executed", "throttled", "suppressed"]
OutcomeStatus = Literal["PENDING", "CLOSED"]
IntegritySignalType = Literal["velocity_spike", "ip_concentration", "reward_farming", "pattern_probe"]
IntegrityStatus = Literal["TRUSTED", "WATCH", "QUARANTINED"]
PolicyVersionStatus = Literal["INCUMBENT", "CANDIDATE", "ROLLED_BACK"]
PolicyDecision = Literal["approve", "modify", "throttle", "suppress", "review"]
BatchTriggerSource = Literal["manual_ui", "n8n_orchestrator", "cron_scheduler"]
BatchForceScenario = Literal["normal", "policy_rollback", "drift_detected"]
BatchPolicyStatus = Literal["PROMOTED", "REJECTED_ROLLBACK", "CANARY_FAILED"]
DemoExecutionStatus = Literal[
    "auto_executed",
    "pending_human_review",
    "executed",
    "not_executed",
]
ScenarioId = Literal[
    "discount_wins_trap",
    "recovery_casino",
    "policy_rollback",
    "off_vs_on_casino",
    "predictive_vs_causal",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_datetime_fields(cls, value, info):
        field_info = info.field_name
        if field_info is None:
            return value

        field = cls.model_fields.get(field_info)
        if field is None:
            return value

        annotation = field.annotation
        annotation_str = str(annotation)
        if "datetime" not in annotation_str.lower():
            return value

        return _parse_datetime_value(value)


def _parse_datetime_value(value):
    """Coerce webhook timestamps into a safe datetime value."""

    if isinstance(value, datetime):
        return value

    if value is None:
        return datetime.utcnow()

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 1_000_000_000_000:
            numeric_value /= 1000.0
        try:
            return datetime.utcfromtimestamp(numeric_value)
        except (OverflowError, OSError, ValueError):
            return datetime.utcnow()

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return datetime.utcnow()

        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return datetime.utcnow()

            if numeric_value > 1_000_000_000_000:
                numeric_value /= 1000.0
            try:
                return datetime.utcfromtimestamp(numeric_value)
            except (OverflowError, OSError, ValueError):
                return datetime.utcnow()

        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    return datetime.utcnow()


class RecoveryCaseBase(StrictModel):
    merchant_id: str = Field(min_length=1)
    transaction_id: str | None = None
    customer_id: str = Field(min_length=1)
    amount: float = Field(ge=0)
    case_age_hours: float = 0.0
    attempt_count: int = Field(default=1, ge=1)
    payment_method: str = Field(default="card", min_length=1)
    failure_class: FailureClass
    customer_segment: CustomerSegment
    event_time: datetime

    @field_validator("case_age_hours")
    @classmethod
    def case_age_hours_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("case_age_hours must be non-negative")
        return value

    @field_validator("payment_method")
    @classmethod
    def normalize_payment_method(cls, value: str) -> str:
        return value.lower()


class RecoveryCaseCreate(RecoveryCaseBase):
    case_id: str | None = None


class RecoveryCaseUpdate(StrictModel):
    merchant_id: str | None = Field(default=None, min_length=1)
    transaction_id: str | None = None
    customer_id: str | None = Field(default=None, min_length=1)
    amount: float | None = Field(default=None, ge=0)
    case_age_hours: float | None = Field(default=None, ge=0.0)
    attempt_count: int | None = Field(default=None, ge=1)
    payment_method: str | None = Field(default=None, min_length=1)
    failure_class: FailureClass | None = None
    customer_segment: CustomerSegment | None = None
    event_time: datetime | None = None

    @field_validator("payment_method")
    @classmethod
    def normalize_optional_payment_method(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None


class RecoveryCaseRead(RecoveryCaseBase):
    case_id: str


class AssignmentBase(StrictModel):
    case_id: str = Field(min_length=1)
    action: AssignmentAction
    assignment_type: AssignmentType
    policy_version: str = Field(min_length=1)
    assigned_at: datetime


class AssignmentCreate(AssignmentBase):
    id: str | None = None


class AssignmentUpdate(StrictModel):
    case_id: str | None = Field(default=None, min_length=1)
    action: AssignmentAction | None = None
    assignment_type: AssignmentType | None = None
    policy_version: str | None = Field(default=None, min_length=1)
    assigned_at: datetime | None = None


class AssignmentRead(AssignmentBase):
    id: str


class InterventionBase(StrictModel):
    case_id: str = Field(min_length=1)
    action: AssignmentAction
    incentive_amount: float = Field(default=0.0, ge=0)
    channel: InterventionChannel
    status: InterventionStatus
    executed_at: datetime | None = None


class InterventionCreate(InterventionBase):
    id: str | None = None


class InterventionUpdate(StrictModel):
    case_id: str | None = Field(default=None, min_length=1)
    action: AssignmentAction | None = None
    incentive_amount: float | None = Field(default=None, ge=0)
    channel: InterventionChannel | None = None
    status: InterventionStatus | None = None
    executed_at: datetime | None = None


class InterventionRead(InterventionBase):
    id: str


class OutcomeBase(StrictModel):
    case_id: str = Field(min_length=1)
    status: OutcomeStatus
    paid: bool = False
    amount_paid: float = Field(default=0.0, ge=0)
    time_to_pay_seconds: int | None = Field(default=None, ge=0)
    downstream_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    outcome_time: datetime | None = None


class OutcomeCreate(OutcomeBase):
    id: str | None = None


class OutcomeUpdate(StrictModel):
    case_id: str | None = Field(default=None, min_length=1)
    status: OutcomeStatus | None = None
    paid: bool | None = None
    amount_paid: float | None = Field(default=None, ge=0)
    time_to_pay_seconds: int | None = Field(default=None, ge=0)
    downstream_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    outcome_time: datetime | None = None


class OutcomeRead(OutcomeBase):
    id: str


class IntegritySignalBase(StrictModel):
    case_id: str = Field(min_length=1)
    signal_type: IntegritySignalType
    anomaly_score: float = Field(ge=0.0, le=1.0)
    integrity_status: IntegrityStatus
    rationale: str = Field(min_length=1)
    created_at: datetime


class IntegritySignalCreate(IntegritySignalBase):
    id: str | None = None


class IntegritySignalUpdate(StrictModel):
    case_id: str | None = Field(default=None, min_length=1)
    signal_type: IntegritySignalType | None = None
    anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    integrity_status: IntegrityStatus | None = None
    rationale: str | None = Field(default=None, min_length=1)
    created_at: datetime | None = None


class IntegritySignalRead(IntegritySignalBase):
    id: str


class PolicyVersionBase(StrictModel):
    version: str = Field(min_length=1)
    model_hash: str = Field(min_length=1)
    action_rules_json: str = Field(min_length=1)
    sample_size: int = Field(ge=0)
    mean_lift: float
    ci_lower: float
    ci_upper: float
    status: PolicyVersionStatus
    created_at: datetime


class PolicyVersionCreate(PolicyVersionBase):
    pass


class PolicyVersionUpdate(StrictModel):
    model_hash: str | None = Field(default=None, min_length=1)
    action_rules_json: str | None = Field(default=None, min_length=1)
    sample_size: int | None = Field(default=None, ge=0)
    mean_lift: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    status: PolicyVersionStatus | None = None
    created_at: datetime | None = None


class PolicyVersionRead(PolicyVersionBase):
    pass


class AuditLogBase(StrictModel):
    case_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    proposed_action: str = Field(min_length=1)
    final_action: str = Field(min_length=1)
    expected_incremental_revenue: float = Field(ge=0)
    integrity_verdict: str = Field(min_length=1)
    policy_decision: PolicyDecision
    created_at: datetime


class AuditLogCreate(AuditLogBase):
    id: str | None = None


class AuditLogUpdate(StrictModel):
    case_id: str | None = Field(default=None, min_length=1)
    policy_version: str | None = Field(default=None, min_length=1)
    proposed_action: str | None = Field(default=None, min_length=1)
    final_action: str | None = Field(default=None, min_length=1)
    expected_incremental_revenue: float | None = Field(default=None, ge=0)
    integrity_verdict: str | None = Field(default=None, min_length=1)
    policy_decision: PolicyDecision | None = None
    created_at: datetime | None = None


class AuditLogRead(AuditLogBase):
    id: str


class StrictWebhookPayload(StrictModel):
    event: str = Field(min_length=1)
    payload: dict
    created_at: datetime | None = None


class CreateResponse(StrictModel):
    success: bool = True
    message: str = Field(min_length=1)
    case_id: str | None = None


class MerchantConstraints(StrictModel):
    """Merchant-owned limits that the model is not allowed to override."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        strict=True,
    )

    merchant_budget: float = Field(ge=0.0)
    merchant_budget_spent: float = Field(default=0.0, ge=0)
    merchant_budget_exhausted: bool = False
    max_incentive: float = Field(
        ge=0.0,
        lt=100.0,
        description="Maximum discount percentage of the recovery-case amount.",
    )
    max_contacts: int = Field(default=1, ge=0)
    contacts_used: int = Field(default=0, ge=0)
    recovery_window_hours: float = Field(default=48.0, ge=0.0)
    offer_ladder: list[int] = Field(
        default_factory=lambda: [0, 3, 5, 8, 10],
        min_length=1,
        json_schema_extra={"default": [0, 3, 5, 8, 10]},
    )
    allowed_actions: List[str]
    policy_version: str = Field(default="v3.0", min_length=1)

    @field_validator("allowed_actions")
    @classmethod
    def allowed_actions_must_be_unique(
        cls,
        value: List[str],
    ) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_actions must not contain duplicates")
        return value

    @field_validator("offer_ladder")
    @classmethod
    def offer_ladder_must_be_discrete_and_ordered(cls, value: list[int]) -> list[int]:
        if any(tier < 0 or tier >= 100 for tier in value):
            raise ValueError("offer_ladder tiers must be between 0 and 99 percent")
        if len(value) != len(set(value)):
            raise ValueError("offer_ladder must not contain duplicates")
        if value != sorted(value):
            raise ValueError("offer_ladder must be sorted in ascending order")
        if 0 not in value:
            raise ValueError("offer_ladder must include the 0 percent control tier")
        return value


class DecisionEvaluateRequest(RecoveryCaseRead):
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        strict=True,
    )

    merchant_constraints: MerchantConstraints
    candidate_actions: List[str]

    @field_validator("candidate_actions")
    @classmethod
    def candidate_actions_must_be_unique(
        cls,
        value: List[str],
    ) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("candidate_actions must not contain duplicates")
        return value


class ConfidenceInterval(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lower_bound: float = Field(ge=-1.0, le=1.0)
    upper_bound: float = Field(ge=-1.0, le=1.0)
    confidence_level: Literal[0.95] = 0.95


class DecisionEvaluateResponse(StrictModel):
    """Strict PRD v3 decision contract with no additional top-level fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    recommended_action: Literal[
        "no_action", "retry", "payment_link", "incentive_link", "message", "suppress"
    ]
    expected_incremental_revenue: float
    confidence: ConfidenceInterval
    integrity_status: IntegrityStatus
    policy_decision: PolicyDecision
    reason_codes: list[str] = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    current_policy_version: str = Field(min_length=1)


class DemoToggleRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    agent_enabled: bool
    auto_execute: bool = False
    scenario_id: ScenarioId | None = Field(
        default=None,
        validation_alias=AliasChoices("scenario", "scenario_id"),
        serialization_alias="scenario",
    )
    recovery_case: RecoveryCaseRead
    merchant_constraints: MerchantConstraints
    candidate_actions: List[str]

    @field_validator("candidate_actions")
    @classmethod
    def demo_actions_must_be_unique(
        cls,
        value: List[str],
    ) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("candidate_actions must not contain duplicates")
        return value


class DemoToggleMLMetrics(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    recommended_action: AssignmentAction
    recommended_tier: int = Field(default=0, ge=0, lt=100)
    discount_cost: float = Field(ge=0.0)
    net_expected_recovery: float = Field(ge=0.0)
    expected_incremental_revenue: float
    baseline_pay_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_pay_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    causal_lift: float | None = Field(default=None, ge=-1.0, le=1.0)
    confidence: ConfidenceInterval
    integrity_status: IntegrityStatus
    policy_decision: PolicyDecision
    reason_codes: list[str] = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    model_available: bool


class ScenarioPreset(StrictModel):
    amount: float = Field(ge=0.0)
    case_age_hours: float = Field(ge=0.0)
    merchant_budget: float = Field(ge=0.0)
    max_incentive: float = Field(
        ge=0.0,
        lt=100.0,
        description="Maximum discount percentage of the recovery-case amount.",
    )
    allowed_actions: List[str] = Field(min_length=1)
    customer_segment: CustomerSegment
    failure_class: FailureClass
    payment_method: str = Field(min_length=1)


class ScenarioCatalogItem(StrictModel):
    scenario_id: ScenarioId
    label: str = Field(min_length=1)
    short_label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    primary_proof: str = Field(min_length=1)
    seed: int
    event_count: int = Field(ge=1)
    attack_count: int = Field(ge=0)
    preset: ScenarioPreset
    prerequisites: list[str] = Field(min_length=1)


class ScenarioWarmupResponse(ScenarioCatalogItem):
    status: Literal["ready"]
    cache_hit: bool
    warmup_ms: float = Field(ge=0.0)


class CumulativeRevenuePoint(StrictModel):
    round: int = Field(ge=1, le=5)
    baseline: float
    agent: float
    naive_ml: float


class PolicyTraceStep(StrictModel):
    version: str = Field(min_length=1)
    status: PolicyVersionStatus
    reason: str = Field(min_length=1)


class ScenarioMetrics(StrictModel):
    scenario_id: ScenarioId
    scenario_label: str = Field(min_length=1)
    seed: int
    event_count: int = Field(ge=1)
    attack_count: int = Field(ge=0)
    amount_input: float = Field(ge=0.0)
    integrity_status: IntegrityStatus
    baseline_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    causal_lift: float | None = Field(default=None, ge=-1.0, le=1.0)
    incremental_recovered_revenue: float
    cumulative_net_recovery: float
    recovery_roi: float
    attacks_quarantined: int = Field(ge=0)
    cumulative_revenue: list[CumulativeRevenuePoint] = Field(min_length=5, max_length=5)
    policy_status: PolicyVersionStatus
    policy_version: str = Field(min_length=1)
    policy_trace: list[PolicyTraceStep] = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)
    comparison_ready: bool


class ExecutionTraceStep(StrictModel):
    stage: str = Field(min_length=1)
    action: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RecoveryMetrics(StrictModel):
    """Common, research-calibrated economics for fair OFF/ON comparison."""

    metric_source: Literal["research_calibrated_simulation"]
    simulation_disclaimer: str = Field(min_length=1)
    baseline_probability: float = Field(ge=0.0, le=1.0)
    predicted_probability: float = Field(ge=0.0, le=1.0)
    downstream_quality: float = Field(ge=0.0, le=1.0)
    quality_adjusted_probability: float = Field(ge=0.0, le=1.0)
    incremental_lift_pp: float = Field(ge=-100.0, le=100.0)
    gross_expected_recovery: float
    intervention_cost: float = Field(ge=0.0)
    discount_cost: float = Field(ge=0.0)
    net_expected_recovery: float
    incremental_recovered_revenue: float
    recovery_roi: float | None = None
    integrity_status: IntegrityStatus
    attacks_quarantined: int = Field(ge=0)
    action_trace: list[ExecutionTraceStep] = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)
    incumbent_capability_coverage: float = Field(ge=0.0, le=1.0)


class DemoToggleResponse(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(min_length=1)
    status: DemoExecutionStatus
    short_url: str | None = None
    final_amount: float | None = Field(default=None, ge=0)
    final_action: Literal[
        "no_action", "retry", "payment_link", "incentive_link", "message", "suppress"
    ]
    agent_enabled: bool
    ml_metrics: DemoToggleMLMetrics | None = None
    input_amount: float = Field(default=0.0, ge=0.0)
    discount_cost: float = Field(default=0.0, ge=0.0)
    net_expected_recovery: float = 0.0
    expected_incremental_revenue: float = 0.0
    reason_codes: list[str] = Field(default_factory=list)
    scenario_metrics: ScenarioMetrics | None = None
    recovery_metrics: RecoveryMetrics | None = None
    current_policy_version: str = Field(min_length=1)


class ApproveLinkRequest(StrictModel):
    case_id: str = Field(min_length=1, max_length=40)
    approved_action: Literal["payment_link", "incentive_link"]


class ApproveLinkResponse(StrictModel):
    case_id: str = Field(min_length=1)
    short_url: str = Field(min_length=1)
    final_amount: float = Field(ge=0.0)
    final_action: Literal["payment_link", "incentive_link"]
    status: Literal["executed"]
    current_policy_version: str = Field(min_length=1)


class BatchLearningRequest(StrictModel):
    trigger_source: BatchTriggerSource = "manual_ui"
    force_scenario: BatchForceScenario = "normal"


class BatchLearningResponse(StrictModel):
    run_id: str = Field(min_length=1)
    status: Literal["success", "rollback", "canary_failed"]
    champion_model: str = Field(min_length=1)
    challenger_model: str = Field(min_length=1)
    challenger_roi: float
    incumbent_roi: float
    quarantine_rate: float = Field(ge=0.0, le=1.0)
    promoted: bool
    active_version: str = Field(min_length=1)
    run_status: BatchPolicyStatus
    rejection_reason: str | None = None
    timestamp: datetime


class PolicyStateResponse(StrictModel):
    active_version: str = Field(min_length=1)
    latest_run: BatchLearningResponse | None = None


class IntakeWebhookResponse(StrictModel):
    case_id: str
    normalized_case: dict
    integrity: dict
    guardrails: dict
    learner: dict


class RazorpayWebhookResponse(StrictModel):
    """Minimal acknowledgement contract for Razorpay webhook delivery."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["received", "ignored"]
    action_taken: Literal[
        "no_action", "retry", "payment_link", "incentive_link", "message", "suppress"
    ] | None = None
    reason: str | None = None
    current_policy_version: str = Field(min_length=1)


class HealthResponse(StrictModel):
    api_status: str
    database_status: str
    model_status: Literal["ready", "unavailable"]
    timestamp: datetime
