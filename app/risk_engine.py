from __future__ import annotations

from dataclasses import dataclass

from app.models import RecoveryCase


HIGH_TICKET_THRESHOLD = 10_000.0
MERCHANT_BUDGET_CONCENTRATION_RATIO = 0.25


@dataclass(frozen=True)
class ExecutionRiskDecision:
    requires_human_review: bool
    reason_codes: tuple[str, ...]


def evaluate_execution_risk(
    case: RecoveryCase,
    incentive_amount: float,
    merchant_budget: float,
    integrity_status: str,
    scenario_id: str | None = None,
) -> ExecutionRiskDecision:
    """Route executable links through deterministic enterprise circuit breakers."""
    reasons: list[str] = []

    if float(case.amount) > HIGH_TICKET_THRESHOLD:
        reasons.append("CIRCUIT_BREAKER_HIGH_TICKET_VALUE")

    incentive = max(float(incentive_amount), 0.0)
    budget = max(float(merchant_budget), 0.0)
    if incentive > 0 and (
        budget == 0
        or incentive > budget * MERCHANT_BUDGET_CONCENTRATION_RATIO
    ):
        reasons.append("CIRCUIT_BREAKER_MERCHANT_BUDGET_CONCENTRATION")

    fingerprint_context = " ".join(
        (
            str(case.customer_id or ""),
            str(case.transaction_id or ""),
            str(scenario_id or ""),
        )
    ).lower()
    has_suspicious_fingerprint = any(
        marker in fingerprint_context
        for marker in ("device_hash", "ip_hash", "device_id", "ip_address")
    )
    if has_suspicious_fingerprint and integrity_status.upper() != "QUARANTINED":
        reasons.append("CIRCUIT_BREAKER_SUSPICIOUS_FINGERPRINT_REVIEW")
    if integrity_status.upper() == "WATCH":
        reasons.append("CIRCUIT_BREAKER_INTEGRITY_WATCH_REVIEW")

    return ExecutionRiskDecision(
        requires_human_review=bool(reasons),
        reason_codes=tuple(reasons),
    )
