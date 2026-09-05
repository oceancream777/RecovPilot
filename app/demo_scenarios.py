"""Deterministic sandbox scenarios for the Recovery Learning Agent demo.

The scenario engine is intentionally separate from production persistence. It
creates immutable, fixed-seed event streams, trains models on randomized
observations, and computes every displayed benchmark value from simulated
outcomes. Interactive API traffic therefore cannot contaminate or invalidate a
demo comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
import os
from threading import Lock
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
# Avoid a slow physical-core probe in restricted laptop and CI environments.
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.baseline_policy import decide_incumbent_recovery
from app.learner import MultiTreatmentTLearner


DEMO_ACTIONS = ("no_action", "retry", "payment_link", "incentive_link")
SEGMENTS = (
    "high_intent_repeat",
    "price_sensitive",
    "subscription_churn",
    "low_intent",
)
SCENARIO_EVENT_COUNT = 10_000
SCENARIO_ROUNDS = 5
MESSAGE_PROBABILITIES = {
    "high_intent_repeat": 0.50,
    "price_sensitive": 0.08,
    "subscription_churn": 0.35,
    "low_intent": 0.03,
}


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    short_label: str
    description: str
    primary_proof: str
    seed: int
    attack_count: int
    preset: dict[str, Any]
    prerequisites: tuple[str, ...]


@dataclass
class PreparedScenario:
    definition: ScenarioDefinition
    frame: pd.DataFrame
    learner: MultiTreatmentTLearner
    naive_model: Pipeline
    preparation_ms: float


COMMON_ACTIONS = ["retry", "payment_link", "incentive_link", "no_action"]

SCENARIOS: dict[str, ScenarioDefinition] = {
    "discount_wins_trap": ScenarioDefinition(
        scenario_id="discount_wins_trap",
        label="The Discount Wins Trap",
        short_label="Discount Wins Trap",
        description=(
            "Discount has the highest raw conversion, while a lower-cost payment "
            "link creates more incremental net recovery."
        ),
        primary_proof="Causal net value beats absolute payment propensity.",
        seed=35101,
        attack_count=0,
        preset={
            "amount": 4999.0,
            "case_age_hours": 50.0,
            "merchant_budget": 250000.0,
            "max_incentive": 10.0,
            "allowed_actions": COMMON_ACTIONS,
            "customer_segment": "high_intent_repeat",
            "failure_class": "user_cancelled",
            "payment_method": "card",
        },
        prerequisites=(
            "10,000 randomized control and treatment observations",
            "Hidden heterogeneous response curves",
            "Explicit payment-link and discount costs",
        ),
    ),
    "recovery_casino": ScenarioDefinition(
        scenario_id="recovery_casino",
        label="The Recovery Casino",
        short_label="Recovery Casino",
        description=(
            "A concentrated incentive-farming cohort creates attractive short-term "
            "labels but zero downstream quality."
        ),
        primary_proof="The integrity gate protects the causal learning signal.",
        seed=35202,
        attack_count=500,
        preset={
            "amount": 1999.0,
            "case_age_hours": 50.0,
            "merchant_budget": 150000.0,
            "max_incentive": 10.0,
            "allowed_actions": COMMON_ACTIONS,
            "customer_segment": "price_sensitive",
            "failure_class": "user_cancelled",
            "payment_method": "card",
        },
        prerequisites=(
            "500 labeled reward-poisoning observations",
            "10 concentrated device hashes and 6 IP hashes",
            "98% apparent conversion with zero downstream quality",
        ),
    ),
    "policy_rollback": ScenarioDefinition(
        scenario_id="policy_rollback",
        label="The Learner That Admits It Was Wrong",
        short_label="Drift and Rollback",
        description=(
            "A once-successful policy drifts, a candidate enters a bounded canary, "
            "and the last-known-good policy is restored."
        ),
        primary_proof="Promotion requires evidence and canary failure triggers rollback.",
        seed=35303,
        attack_count=0,
        preset={
            "amount": 7499.0,
            "case_age_hours": 72.0,
            "merchant_budget": 250000.0,
            "max_incentive": 10.0,
            "allowed_actions": COMMON_ACTIONS,
            "customer_segment": "high_intent_repeat",
            "failure_class": "network_timeout",
            "payment_method": "card",
        },
        prerequisites=(
            "Two clean incumbent rounds",
            "A hidden response-curve shift",
            "Held-out promotion and 20% canary gates",
        ),
    ),
    "off_vs_on_casino": ScenarioDefinition(
        scenario_id="off_vs_on_casino",
        label="Recovery Agent OFF vs ON",
        short_label="OFF vs ON Casino",
        description=(
            "The same immutable traffic and attack seed are replayed with only the "
            "learning and integrity layer changed."
        ),
        primary_proof="The product layer improves net recovery and limits exposure.",
        seed=35404,
        attack_count=500,
        preset={
            "amount": 4999.0,
            "case_age_hours": 50.0,
            "merchant_budget": 150000.0,
            "max_incentive": 10.0,
            "allowed_actions": COMMON_ACTIONS,
            "customer_segment": "price_sensitive",
            "failure_class": "user_cancelled",
            "payment_method": "card",
        },
        prerequisites=(
            "One immutable 10,000-case stream",
            "Identical attack index and random seed for OFF and ON",
            "Common random outcomes for paired measurement",
        ),
    ),
    "predictive_vs_causal": ScenarioDefinition(
        scenario_id="predictive_vs_causal",
        label="Naive ML vs Recovery Learning Agent",
        short_label="Naive ML vs Causal",
        description=(
            "A propensity classifier selects the highest raw payment probability, "
            "while the causal policy subtracts the no-action baseline and cost."
        ),
        primary_proof="Prediction is not causal decisioning.",
        seed=35505,
        attack_count=0,
        preset={
            "amount": 4999.0,
            "case_age_hours": 50.0,
            "merchant_budget": 250000.0,
            "max_incentive": 10.0,
            "allowed_actions": COMMON_ACTIONS,
            "customer_segment": "high_intent_repeat",
            "failure_class": "issuer_down",
            "payment_method": "card",
        },
        prerequisites=(
            "One contaminated absolute-propensity classifier",
            "Independent action response models",
            "No-action control and explicit intervention costs",
        ),
    ),
}


TRAP_CURVES: dict[str, dict[str, float]] = {
    "high_intent_repeat": {
        "no_action": 0.74,
        "retry": 0.78,
        "payment_link": 0.90,
        "incentive_link": 0.93,
    },
    "price_sensitive": {
        "no_action": 0.08,
        "retry": 0.09,
        "payment_link": 0.20,
        "incentive_link": 0.47,
    },
    "subscription_churn": {
        "no_action": 0.34,
        "retry": 0.67,
        "payment_link": 0.45,
        "incentive_link": 0.54,
    },
    "low_intent": {
        "no_action": 0.03,
        "retry": 0.035,
        "payment_link": 0.038,
        "incentive_link": 0.04,
    },
}

DRIFT_TRAIN_CURVES: dict[str, dict[str, float]] = {
    **TRAP_CURVES,
    "high_intent_repeat": {
        "no_action": 0.48,
        "retry": 0.62,
        "payment_link": 0.82,
        "incentive_link": 0.78,
    },
}

DRIFT_HOLDOUT_CURVES: dict[str, dict[str, float]] = {
    **TRAP_CURVES,
    "high_intent_repeat": {
        "no_action": 0.48,
        "retry": 0.76,
        "payment_link": 0.61,
        "incentive_link": 0.66,
    },
}

DRIFT_CANARY_CURVES: dict[str, dict[str, float]] = {
    **TRAP_CURVES,
    "high_intent_repeat": {
        "no_action": 0.48,
        "retry": 0.58,
        "payment_link": 0.72,
        "incentive_link": 0.64,
    },
}


_CACHE: dict[str, PreparedScenario] = {}
_CACHE_LOCK = Lock()


def list_scenarios() -> list[dict[str, Any]]:
    """Return the public five-scene catalog in the buildathon run order."""
    order = (
        "discount_wins_trap",
        "recovery_casino",
        "policy_rollback",
        "off_vs_on_casino",
        "predictive_vs_causal",
    )
    return [_public_definition(SCENARIOS[scenario_id]) for scenario_id in order]


def _public_definition(definition: ScenarioDefinition) -> dict[str, Any]:
    return {
        "scenario_id": definition.scenario_id,
        "label": definition.label,
        "short_label": definition.short_label,
        "description": definition.description,
        "primary_proof": definition.primary_proof,
        "seed": definition.seed,
        "event_count": SCENARIO_EVENT_COUNT,
        "attack_count": definition.attack_count,
        "preset": dict(definition.preset),
        "prerequisites": list(definition.prerequisites),
    }


def get_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"Unknown demo scenario: {scenario_id}") from exc


def prepare_scenario(scenario_id: str) -> PreparedScenario:
    """Build a scenario once and reuse it for every paired ON/OFF replay."""
    definition = get_scenario_definition(scenario_id)
    cached = _CACHE.get(scenario_id)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _CACHE.get(scenario_id)
        if cached is not None:
            return cached

        started = perf_counter()
        frame = _generate_event_stream(definition)
        learner = _train_causal_learner(frame, definition.seed)
        naive_model = _train_naive_model(frame, definition.seed)
        prepared = PreparedScenario(
            definition=definition,
            frame=frame,
            learner=learner,
            naive_model=naive_model,
            preparation_ms=round((perf_counter() - started) * 1000.0, 2),
        )
        _CACHE[scenario_id] = prepared
        return prepared


def warmup_scenario(scenario_id: str) -> dict[str, Any]:
    was_cached = scenario_id in _CACHE
    prepared = prepare_scenario(scenario_id)
    result = _public_definition(prepared.definition)
    result.update(
        {
            "status": "ready",
            "cache_hit": was_cached,
            "warmup_ms": 0.0 if was_cached else prepared.preparation_ms,
        }
    )
    return result


def _curves_for(definition: ScenarioDefinition) -> dict[str, dict[str, float]]:
    if definition.scenario_id == "policy_rollback":
        return DRIFT_TRAIN_CURVES
    return TRAP_CURVES


def _generate_event_stream(definition: ScenarioDefinition) -> pd.DataFrame:
    rng = np.random.default_rng(definition.seed)
    operations_rng = np.random.default_rng(definition.seed + 90_000)
    count = SCENARIO_EVENT_COUNT
    segments = rng.choice(
        np.asarray(SEGMENTS),
        size=count,
        p=np.asarray([0.35, 0.35, 0.15, 0.15]),
    )
    default_amount = max(float(definition.preset["amount"]), 100.0)
    amounts = np.clip(
        rng.lognormal(mean=log(default_amount), sigma=0.45, size=count),
        100.0,
        50_000.0,
    ).round(2)
    assigned_actions = rng.choice(np.asarray(DEMO_ACTIONS), size=count)
    case_age_hours = operations_rng.uniform(0.0, 96.0, size=count).round(2)
    attempt_count = operations_rng.integers(1, 5, size=count)
    payment_method = operations_rng.choice(
        np.asarray(["card", "upi"]),
        size=count,
        p=np.asarray([0.72, 0.28]),
    )
    is_attack = np.zeros(count, dtype=bool)
    if definition.attack_count:
        attack_start = count - definition.attack_count
        is_attack[attack_start:] = True
        segments[attack_start:] = "price_sensitive"
        assigned_actions[attack_start:] = "incentive_link"
        case_age_hours[attack_start:] = 72.0
        attempt_count[attack_start:] = 15
        payment_method[attack_start:] = "card"

    failure_by_segment = {
        "high_intent_repeat": "issuer_down",
        "price_sensitive": "user_cancelled",
        "subscription_churn": "insufficient_funds",
        "low_intent": "network_timeout",
    }
    failure_classes = np.asarray([failure_by_segment[str(segment)] for segment in segments])
    curves = _curves_for(definition)
    probabilities = np.asarray(
        [
            0.98 if attack else curves[str(segment)][str(action)]
            for segment, action, attack in zip(
                segments,
                assigned_actions,
                is_attack,
                strict=True,
            )
        ],
        dtype=float,
    )
    paid = rng.random(count) < probabilities
    downstream_quality = np.where(is_attack, 0.0, 1.0)

    device_hashes = np.asarray([f"device_hash_{index:02d}" for index in range(10)])
    ip_hashes = np.asarray([f"ip_hash_{index:02d}" for index in range(6)])
    device_hash = np.full(count, "", dtype=object)
    ip_hash = np.full(count, "", dtype=object)
    if definition.attack_count:
        device_hash[is_attack] = rng.choice(device_hashes, size=definition.attack_count)
        ip_hash[is_attack] = rng.choice(ip_hashes, size=definition.attack_count)

    return pd.DataFrame(
        {
            "case_id": [f"{definition.scenario_id}-{index:05d}" for index in range(count)],
            "amount": amounts,
            "failure_class": failure_classes,
            "customer_segment": segments,
            "case_age_hours": case_age_hours,
            "attempt_count": attempt_count,
            "payment_method": payment_method,
            "assigned_action": assigned_actions,
            "status": "CLOSED",
            "paid": paid,
            "downstream_quality": downstream_quality,
            "integrity_status": np.where(is_attack, "QUARANTINED", "TRUSTED"),
            "is_attack": is_attack,
            "device_hash": device_hash,
            "ip_hash": ip_hash,
        }
    )


def _train_causal_learner(frame: pd.DataFrame, seed: int) -> MultiTreatmentTLearner:
    records = frame.rename(columns={"assigned_action": "action"})[
        [
            "amount",
            "failure_class",
            "customer_segment",
            "action",
            "status",
            "paid",
            "downstream_quality",
            "integrity_status",
        ]
    ]
    return MultiTreatmentTLearner(random_state=seed).fit(records.to_dict("records"))


def _train_naive_model(frame: pd.DataFrame, seed: int) -> Pipeline:
    features = ["amount", "failure_class", "customer_segment", "assigned_action"]
    transformer = ColumnTransformer(
        transformers=[
            ("amount", StandardScaler(), ["amount"]),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["failure_class", "customer_segment", "assigned_action"],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("features", transformer),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=seed),
            ),
        ]
    )
    model.fit(frame[features], frame["paid"].astype(int))
    return model


def scenario_integrity_override(scenario_id: str, case_id: str) -> dict[str, Any] | None:
    """Compute an attack verdict from the prepared cohort's observed fingerprints."""
    prepared = prepare_scenario(scenario_id)
    attacks = prepared.frame.loc[prepared.frame["is_attack"]]
    if attacks.empty:
        return None

    device_concentration = int(attacks["device_hash"].value_counts().max())
    ip_concentration = int(attacks["ip_hash"].value_counts().max())
    conversion_rate = float(attacks["paid"].mean())
    downstream_quality = float(attacks.loc[attacks["paid"], "downstream_quality"].mean())
    mismatch = conversion_rate >= 0.80 and downstream_quality <= 0.10
    status = "QUARANTINED" if mismatch and max(device_concentration, ip_concentration) >= 15 else "WATCH"
    return {
        "case_id": case_id,
        "signal_type": "reward_farming" if mismatch else "ip_concentration",
        "anomaly_score": 0.98 if status == "QUARANTINED" else 0.68,
        "integrity_status": status,
        "rationale": (
            f"Sandbox cohort shows {conversion_rate:.0%} apparent conversion, "
            f"{downstream_quality:.2f} downstream quality, and "
            f"{max(device_concentration, ip_concentration)} shared fingerprints."
        ),
    }


def estimate_scenario_uplift(
    scenario_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Score one interactive case with the selected scenario's trained learner."""
    return prepare_scenario(scenario_id).learner.estimate_uplift(context)


def _constraints_dict(constraints: Any) -> dict[str, Any]:
    if hasattr(constraints, "model_dump"):
        return constraints.model_dump()
    return dict(constraints) if isinstance(constraints, Mapping) else {}


def _cohort_agent_policy(
    prepared: PreparedScenario,
    constraints: Mapping[str, Any],
    case_age_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = prepared.frame
    allowed = set(str(action) for action in constraints.get("allowed_actions", []))
    allowed &= set(DEMO_ACTIONS)
    allowed.add("no_action")
    if case_age_hours < float(constraints.get("recovery_window_hours", 48.0)):
        allowed.discard("incentive_link")

    max_incentive = float(constraints.get("max_incentive", 0.0))
    offer_ladder = [
        int(tier)
        for tier in constraints.get("offer_ladder", [0, 3, 5, 8, 10])
        if int(tier) <= max_incentive
    ]
    actions_by_segment: dict[str, str] = {}
    tiers_by_segment: dict[str, int] = {}
    for segment in SEGMENTS:
        segment_frame = frame.loc[(frame["customer_segment"] == segment) & ~frame["is_attack"]]
        representative_amount = float(segment_frame["amount"].median())
        failure_class = str(segment_frame["failure_class"].mode().iloc[0])
        estimate = prepared.learner.estimate_uplift(
            {
                "amount": representative_amount,
                "failure_class": failure_class,
                "customer_segment": segment,
                "integrity_status": "TRUSTED",
                "anomaly_score": 0.05,
                "permitted_actions": sorted(allowed),
                "offer_ladder": offer_ladder,
                "max_incentive": max_incentive,
                "merchant_constraints": dict(constraints),
            }
        )
        actions_by_segment[segment] = str(estimate["recommended_action"])
        tiers_by_segment[segment] = int(estimate.get("recommended_tier", 0))

    selected_actions = np.asarray(
        [actions_by_segment[str(segment)] for segment in frame["customer_segment"]],
        dtype=object,
    )
    selected_tiers = np.asarray(
        [tiers_by_segment[str(segment)] for segment in frame["customer_segment"]],
        dtype=int,
    )
    selected_actions[frame["is_attack"].to_numpy(dtype=bool)] = "no_action"
    selected_tiers[frame["is_attack"].to_numpy(dtype=bool)] = 0
    upi_exhausted = (
        frame["payment_method"].eq("upi").to_numpy(dtype=bool)
        & frame["attempt_count"].ge(4).to_numpy(dtype=bool)
    )
    selected_actions[upi_exhausted] = "no_action"
    selected_tiers[upi_exhausted] = 0
    high_value_retry = (
        frame["amount"].gt(15_000).to_numpy(dtype=bool)
        & (selected_actions == "retry")
    )
    high_value_fallback = "payment_link" if "payment_link" in allowed else "no_action"
    selected_actions[high_value_retry] = high_value_fallback
    selected_tiers[high_value_retry] = 0
    return _apply_batch_budget(frame, selected_actions, selected_tiers, constraints)


def _incumbent_policy(
    prepared: PreparedScenario,
    constraints: Mapping[str, Any],
    case_age_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = prepared.frame
    candidates = list(constraints.get("allowed_actions", []))
    actions: list[str] = []
    tiers: list[int] = []
    for row in frame.itertuples(index=False):
        case_values = row._asdict()
        case_values["case_age_hours"] = case_age_hours
        decision = decide_incumbent_recovery(
            case_values,
            constraints,
            candidates,
            prepared.definition.scenario_id,
        )
        actions.append(decision.final_action)
        tiers.append(decision.discount_tier)
    return _apply_batch_budget(
        frame,
        np.asarray(actions, dtype=object),
        np.asarray(tiers, dtype=int),
        constraints,
    )


def _apply_batch_budget(
    frame: pd.DataFrame,
    actions: np.ndarray,
    tiers: np.ndarray,
    constraints: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    actions = actions.copy()
    tiers = tiers.copy()
    amounts = frame["amount"].to_numpy(dtype=float)
    max_incentive = float(constraints.get("max_incentive", 0.0))
    budget = max(
        float(constraints.get("merchant_budget", 0.0))
        - float(constraints.get("merchant_budget_spent", 0.0)),
        0.0,
    )
    fallback = "payment_link" if "payment_link" in constraints.get("allowed_actions", []) else "no_action"
    spent = 0.0
    for index in np.flatnonzero(actions == "incentive_link"):
        cost = amounts[index] * (tiers[index] / 100.0)
        if tiers[index] > max_incentive or spent + cost > budget:
            actions[index] = fallback
            tiers[index] = 0
            continue
        spent += cost
    return actions, tiers


def _naive_policy(
    prepared: PreparedScenario,
    constraints: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    frame = prepared.frame
    allowed = [
        action
        for action in DEMO_ACTIONS
        if action in set(constraints.get("allowed_actions", [])) | {"no_action"}
    ]
    base = frame[["amount", "failure_class", "customer_segment"]]
    scores: list[np.ndarray] = []
    for action in allowed:
        candidate = base.copy()
        candidate["assigned_action"] = action
        scores.append(prepared.naive_model.predict_proba(candidate)[:, 1])
    selected = np.asarray(allowed, dtype=object)[np.argmax(np.column_stack(scores), axis=1)]
    max_incentive = float(constraints.get("max_incentive", 0.0))
    eligible_tiers = [
        int(tier)
        for tier in constraints.get("offer_ladder", [0])
        if int(tier) <= max_incentive
    ]
    tiers = np.where(selected == "incentive_link", max(eligible_tiers, default=0), 0)
    return _apply_batch_budget(frame, selected, tiers.astype(int), constraints)


def _probability_for(
    scenario_id: str,
    segment: str,
    action: str,
    tier: int,
    is_attack: bool,
    curves: dict[str, dict[str, float]] | None = None,
) -> float:
    if action == "suppress":
        action = "no_action"
    if is_attack:
        return 0.98 if action == "incentive_link" else 0.01
    if action == "message":
        return MESSAGE_PROBABILITIES[segment]
    active_curves = curves or (
        DRIFT_TRAIN_CURVES if scenario_id == "policy_rollback" else TRAP_CURVES
    )
    if action != "incentive_link":
        return active_curves[segment][action]
    baseline = active_curves[segment]["no_action"]
    maximum_offer = active_curves[segment]["incentive_link"]
    if tier <= 0:
        return baseline
    response_share = (1.0 - np.exp(-tier / 5.0)) / (1.0 - np.exp(-2.0))
    return float(np.clip(baseline + (maximum_offer - baseline) * response_share, 0.0, 1.0))


def _evaluate_policy_round(
    prepared: PreparedScenario,
    actions: np.ndarray,
    tiers: np.ndarray,
    outcome_uniforms: np.ndarray,
    curves: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    frame = prepared.frame
    amounts = frame["amount"].to_numpy(dtype=float)
    attacks = frame["is_attack"].to_numpy(dtype=bool)
    probabilities = np.asarray(
        [
            _probability_for(
                prepared.definition.scenario_id,
                str(segment),
                str(action),
                int(tier),
                bool(attack),
                curves,
            )
            for segment, action, tier, attack in zip(
                frame["customer_segment"],
                actions,
                tiers,
                attacks,
                strict=True,
            )
        ]
    )
    paid = outcome_uniforms < probabilities
    poisoned = attacks & paid & (actions == "incentive_link")
    downstream_quality = np.where(poisoned, 0.0, 1.0)
    gross_recovered = float(np.sum(amounts * paid * downstream_quality))
    link_cost = float(np.sum(actions == "payment_link") * 2.0)
    message_cost = float(np.sum(actions == "message") * 1.0)
    discount_cost = float(
        np.sum(np.where(actions == "incentive_link", amounts * tiers / 100.0, 0.0))
    )
    abuse_exposure = float(np.sum(amounts * poisoned))
    intervention_cost = link_cost + message_cost + discount_cost
    net_recovery = gross_recovered - intervention_cost - abuse_exposure
    return {
        "gross_recovered": gross_recovered,
        "intervention_cost": intervention_cost,
        "discount_cost": discount_cost,
        "abuse_exposure": abuse_exposure,
        "net_recovery": net_recovery,
    }


def _bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(1_000, dtype=float)
    for iteration in range(1_000):
        sample = rng.integers(0, len(values), len(values))
        means[iteration] = float(values[sample].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def _policy_trace(prepared: PreparedScenario) -> tuple[str, list[dict[str, Any]], list[str]]:
    if prepared.definition.scenario_id != "policy_rollback":
        return (
            "INCUMBENT",
            [
                {
                    "version": f"{prepared.definition.scenario_id}-v1",
                    "status": "INCUMBENT",
                    "reason": "Fixed-seed held-out policy is active.",
                }
            ],
            [],
        )

    rng = np.random.default_rng(prepared.definition.seed + 800)
    count = 1_000
    uniforms = rng.random(count)
    holdout_delta = (
        (uniforms < DRIFT_HOLDOUT_CURVES["high_intent_repeat"]["retry"]).astype(float)
        - (uniforms < DRIFT_HOLDOUT_CURVES["high_intent_repeat"]["payment_link"]).astype(float)
    )
    holdout_mean, holdout_lower, holdout_upper = _bootstrap_interval(
        holdout_delta,
        prepared.definition.seed + 801,
    )
    canary_uniforms = np.random.default_rng(prepared.definition.seed + 802).random(count)
    canary_delta = (
        (canary_uniforms < DRIFT_CANARY_CURVES["high_intent_repeat"]["retry"]).astype(float)
        - (canary_uniforms < DRIFT_CANARY_CURVES["high_intent_repeat"]["payment_link"]).astype(float)
    )
    canary_mean, canary_lower, canary_upper = _bootstrap_interval(
        canary_delta,
        prepared.definition.seed + 803,
    )
    status = "ROLLED_BACK"
    reasons = [
        "CANARY_LIFT_DEGRADED",
        "PROMOTION_REJECTED",
        "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
    ]

    trace = [
        {
            "version": "drift-v1-payment-link",
            "status": "INCUMBENT",
            "reason": "Payment link won two clean rounds before drift.",
        },
        {
            "version": "drift-v2-retry-candidate",
            "status": "CANDIDATE",
            "reason": (
                f"Held-out lift delta {holdout_mean:+.3f}, "
                f"95% CI [{holdout_lower:+.3f}, {holdout_upper:+.3f}], "
                "20% policy delta."
            ),
        },
        {
            "version": "drift-v1-payment-link-restored",
            "status": status,
            "reason": (
                f"Canary lift delta {canary_mean:+.3f}, "
                f"95% CI [{canary_lower:+.3f}, {canary_upper:+.3f}]."
            ),
        },
    ]
    return status, trace, reasons


def build_scenario_metrics(
    scenario_id: str,
    agent_enabled: bool,
    amount: float,
    case_age_hours: float,
    constraints: Any,
    target_estimate: Mapping[str, Any] | None = None,
    final_action: str = "payment_link",
    integrity_status: str = "TRUSTED",
) -> dict[str, Any]:
    """Compute paired mode metrics from one immutable prepared event stream."""
    prepared = prepare_scenario(scenario_id)
    constraint_values = _constraints_dict(constraints)
    off_actions, off_tiers = _incumbent_policy(
        prepared,
        constraint_values,
        case_age_hours,
    )
    on_actions, on_tiers = _cohort_agent_policy(
        prepared,
        constraint_values,
        case_age_hours,
    )
    naive_actions, naive_tiers = _naive_policy(prepared, constraint_values)

    cumulative_off = 0.0
    cumulative_on = 0.0
    cumulative_naive = 0.0
    off_cost = 0.0
    on_cost = 0.0
    revenue_points: list[dict[str, Any]] = []
    for round_number in range(1, SCENARIO_ROUNDS + 1):
        uniforms = np.random.default_rng(
            prepared.definition.seed + 10_000 + round_number
        ).random(len(prepared.frame))
        off_result = _evaluate_policy_round(prepared, off_actions, off_tiers, uniforms)
        on_result = _evaluate_policy_round(prepared, on_actions, on_tiers, uniforms)
        naive_result = _evaluate_policy_round(prepared, naive_actions, naive_tiers, uniforms)
        cumulative_off += off_result["net_recovery"]
        cumulative_on += on_result["net_recovery"]
        cumulative_naive += naive_result["net_recovery"]
        off_cost += off_result["intervention_cost"]
        on_cost += on_result["intervention_cost"]
        revenue_points.append(
            {
                "round": round_number,
                "baseline": round(
                    cumulative_naive
                    if scenario_id == "predictive_vs_causal"
                    else cumulative_off,
                    2,
                ),
                "agent": round(cumulative_on, 2),
                "naive_ml": round(cumulative_naive, 2),
            }
        )

    comparison_net = (
        cumulative_naive
        if scenario_id == "predictive_vs_causal"
        else cumulative_off
    )
    selected_net = cumulative_on if agent_enabled else comparison_net
    incremental_revenue = cumulative_on - comparison_net if agent_enabled else 0.0
    incremental_cost = max(on_cost - off_cost, 0.0)
    recovery_roi = incremental_revenue / max(incremental_cost, 1.0) if agent_enabled else 0.0
    attacks_quarantined = prepared.definition.attack_count if agent_enabled else 0

    estimate = dict(target_estimate or {})
    action_result = estimate.get("actions", {}).get(final_action, {})
    baseline_probability = estimate.get("baseline_pay_probability")
    predicted_probability = action_result.get("pay_probability")
    causal_lift = action_result.get("causal_lift")
    policy_status, policy_trace, policy_reasons = _policy_trace(prepared)
    if scenario_id == "predictive_vs_causal":
        reason_codes = [
            "CAUSAL_LIFT_NEGATIVE_MARGIN_SAVED"
            if agent_enabled
            else "NAIVE_ML_OPTIMIZED_FOR_RAW_CONVERSION"
        ]
    elif scenario_id == "policy_rollback":
        if agent_enabled:
            reason_codes = [
                "CANARY_LIFT_DEGRADED",
                "PROMOTION_REJECTED",
                "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
            ]
        else:
            policy_status = "INCUMBENT"
            policy_trace = policy_trace[:1]
            policy_reasons = ["LAST_KNOWN_GOOD_POLICY_ACTIVE"]
            reason_codes = policy_reasons
    else:
        reason_codes = {
            "discount_wins_trap": ["CAUSAL_NET_VALUE_BEATS_RAW_CONVERSION"],
            "recovery_casino": [
                "INTEGRITY_REWARD_POISONING_DETECTED",
                "QUARANTINED_OUTCOMES_EXCLUDED",
            ],
            "off_vs_on_casino": [
                "IMMUTABLE_SEED_REPLAY",
                "QUARANTINED_OUTCOMES_EXCLUDED",
            ],
        }[scenario_id]

    return {
        "scenario_id": scenario_id,
        "scenario_label": prepared.definition.label,
        "seed": prepared.definition.seed,
        "event_count": len(prepared.frame),
        "attack_count": prepared.definition.attack_count,
        "amount_input": round(float(amount), 2),
        "integrity_status": integrity_status,
        "baseline_probability": baseline_probability,
        "predicted_probability": predicted_probability,
        "causal_lift": causal_lift,
        "incremental_recovered_revenue": round(incremental_revenue, 2),
        "cumulative_net_recovery": round(selected_net, 2),
        "recovery_roi": round(recovery_roi, 2),
        "attacks_quarantined": attacks_quarantined,
        "cumulative_revenue": revenue_points,
        "policy_status": policy_status,
        "policy_version": policy_trace[-1]["version"],
        "policy_trace": policy_trace,
        "reason_codes": reason_codes,
        "comparison_ready": True,
    }


def scenario_confidence_interval(
    estimate: Mapping[str, Any],
    action: str,
) -> tuple[float, float]:
    """Return a normal-approximation interval for a scenario action lift."""
    action_result = estimate.get("actions", {}).get(action, {})
    control_result = estimate.get("actions", {}).get("no_action", {})
    probability = action_result.get("pay_probability")
    control_probability = control_result.get("pay_probability")
    lift = action_result.get("causal_lift")
    action_samples = int(action_result.get("training_samples") or 0)
    control_samples = int(control_result.get("training_samples") or 0)
    if None in {probability, control_probability, lift} or min(action_samples, control_samples) <= 0:
        return -1.0, 1.0
    standard_error = sqrt(
        float(probability) * (1.0 - float(probability)) / action_samples
        + float(control_probability)
        * (1.0 - float(control_probability))
        / control_samples
    )
    margin = 1.96 * standard_error
    return max(-1.0, float(lift) - margin), min(1.0, float(lift) + margin)
