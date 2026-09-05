"""Transparent expected-value math shared by OFF and ON demo executions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


GROUND_TRUTH_PAY_PROBABILITIES: dict[str, dict[str, float]] = {
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

SIMULATION_BASIS = "research_calibrated_simulation"
SIMULATION_DISCLAIMER = (
    "Synthetic expected values calibrated from the project research and scenario "
    "ground truth; they are not Razorpay production statistics."
)


@dataclass(frozen=True)
class RecoveryExpectedValue:
    metric_source: str
    simulation_disclaimer: str
    baseline_probability: float
    predicted_probability: float
    downstream_quality: float
    quality_adjusted_probability: float
    incremental_lift_pp: float
    gross_expected_recovery: float
    intervention_cost: float
    discount_cost: float
    net_expected_recovery: float
    incremental_recovered_revenue: float
    recovery_roi: float | None
    integrity_status: str
    attacks_quarantined: int
    action_trace: list[dict[str, str]]
    reason_codes: list[str]
    incumbent_capability_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(case: Any, name: str, default: Any = None) -> Any:
    if isinstance(case, Mapping):
        return case.get(name, default)
    return getattr(case, name, default)


def action_cost(amount: float, action: str, discount_tier: int = 0) -> tuple[float, float]:
    """Return total intervention cost and the discount component."""
    discount_cost = amount * max(discount_tier, 0) / 100.0 if action == "incentive_link" else 0.0
    platform_cost = {
        "no_action": 0.0,
        "suppress": 0.0,
        "retry": 0.0,
        "payment_link": 2.0,
        "incentive_link": 2.0,
        "message": 1.0,
    }.get(action, 0.0)
    return platform_cost + discount_cost, discount_cost


def response_probability(segment: str, action: str, discount_tier: int = 0) -> float:
    """Return a bounded scenario-ground-truth response probability."""
    curves = GROUND_TRUTH_PAY_PROBABILITIES.get(
        segment,
        GROUND_TRUTH_PAY_PROBABILITIES["low_intent"],
    )
    effective_action = "no_action" if action in {"no_action", "suppress"} else action
    maximum = float(curves.get(effective_action, curves["no_action"]))
    baseline = float(curves["no_action"])
    if effective_action != "incentive_link":
        return maximum
    if discount_tier <= 0:
        return baseline
    response_share = min(float(discount_tier) / 10.0, 1.0)
    return baseline + (maximum - baseline) * response_share


def simulate_expected_recovery(
    case: Any,
    final_action: str,
    discount_tier: int,
    integrity_status: str,
    reason_codes: Sequence[str],
    action_trace: Sequence[Any],
    incumbent_capability_coverage: float,
    scenario_id: str | None = None,
    forced_no_action_veto: bool = False,
) -> RecoveryExpectedValue:
    """Compute apples-to-apples economics without sampling or hidden divisors."""
    amount = max(float(_read(case, "amount", 0.0) or 0.0), 0.0)
    segment = str(_read(case, "customer_segment", "low_intent") or "low_intent")
    baseline_probability = response_probability(segment, "no_action")
    predicted_probability = response_probability(segment, final_action, discount_tier)

    downstream_quality = 1.0
    if scenario_id in {"recovery_casino", "off_vs_on_casino"}:
        baseline_probability = 0.01
        if final_action == "incentive_link":
            predicted_probability = 0.98
            downstream_quality = 0.0
        else:
            predicted_probability = 0.01
    elif scenario_id == "predictive_vs_causal":
        baseline_probability = 0.74
        predicted_probability = {
            "payment_link": 0.90,
            "incentive_link": 0.93,
        }.get(final_action, baseline_probability)

    quality_adjusted_probability = predicted_probability * downstream_quality
    baseline_recovery = amount * baseline_probability
    gross_expected_recovery = amount * quality_adjusted_probability
    intervention_cost, discount_cost = action_cost(amount, final_action, discount_tier)
    net_expected_recovery = gross_expected_recovery - intervention_cost
    incremental_revenue = net_expected_recovery - baseline_recovery
    recovery_roi = (
        incremental_revenue / intervention_cost if intervention_cost > 0.0 else None
    )
    if forced_no_action_veto:
        baseline_probability = 0.0
        predicted_probability = 0.0
        quality_adjusted_probability = 0.0
        gross_expected_recovery = 0.0
        intervention_cost = 0.0
        discount_cost = 0.0
        net_expected_recovery = 0.0
        incremental_revenue = 0.0
        recovery_roi = None

    serialized_trace: list[dict[str, str]] = []
    for step in action_trace:
        if hasattr(step, "__dataclass_fields__"):
            serialized_trace.append(asdict(step))
        elif isinstance(step, Mapping):
            serialized_trace.append({str(key): str(value) for key, value in step.items()})

    return RecoveryExpectedValue(
        metric_source=SIMULATION_BASIS,
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        baseline_probability=round(baseline_probability, 4),
        predicted_probability=round(predicted_probability, 4),
        downstream_quality=round(downstream_quality, 4),
        quality_adjusted_probability=round(quality_adjusted_probability, 4),
        incremental_lift_pp=round(
            (quality_adjusted_probability - baseline_probability) * 100.0,
            2,
        ),
        gross_expected_recovery=round(gross_expected_recovery, 2),
        intervention_cost=round(intervention_cost, 2),
        discount_cost=round(discount_cost, 2),
        net_expected_recovery=round(net_expected_recovery, 2),
        incremental_recovered_revenue=round(incremental_revenue, 2),
        recovery_roi=round(recovery_roi, 2) if recovery_roi is not None else None,
        integrity_status=integrity_status,
        attacks_quarantined=1 if integrity_status == "QUARANTINED" else 0,
        action_trace=serialized_trace,
        reason_codes=list(dict.fromkeys(reason_codes)),
        incumbent_capability_coverage=round(incumbent_capability_coverage, 4),
    )
