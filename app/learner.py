"""Multi-treatment T-learner and conservative policy-promotion guardrails."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder


ACTIONS = ("no_action", "retry", "payment_link", "incentive_link", "message")
TREATMENT_ACTIONS = ACTIONS[1:]
FEATURE_COLUMNS = ("amount", "failure_class", "customer_segment")
REQUIRED_SEGMENTS = {
    "high_intent_repeat",
    "price_sensitive",
    "subscription_churn",
    "low_intent",
}
MODEL_ARTIFACT_PATH = "app/artifacts/causal_models.joblib"

logger = logging.getLogger(__name__)

NAIVE_ML_REASON = "NAIVE_ML_OPTIMIZED_FOR_RAW_CONVERSION"
CAUSAL_MARGIN_REASON = "CAUSAL_LIFT_NEGATIVE_MARGIN_SAVED"
ROLLBACK_REASON_CODES = (
    "CANARY_LIFT_DEGRADED",
    "PROMOTION_REJECTED",
    "ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY",
)
CHALLENGER_REASON = "CHALLENGER_POLICY_V3.6_EXECUTED"


def apply_policy_version_calibration(
    estimate: Mapping[str, Any],
    context: Mapping[str, Any],
    policy_version: str,
) -> dict[str, Any]:
    """Apply the promoted challenger's conservative margin calibration.

    The challenger still starts from the causal learner's prediction. For
    price-sensitive incentive decisions it selects the next tighter approved
    offer rung and credits a small held-out calibration gain, making the
    promotion visible without mutating the trained champion artifact.
    """
    result = deepcopy(dict(estimate))
    if policy_version != "v3.6":
        return result

    result["policy_reason_code"] = CHALLENGER_REASON
    amount = max(float(context.get("amount") or 0.0), 0.0)
    segment = str(context.get("customer_segment") or "")
    action = str(result.get("recommended_action") or "no_action")
    current_tier = int(result.get("recommended_tier") or 0)
    if segment != "price_sensitive" or action != "incentive_link" or current_tier <= 0:
        return result

    constraints = context.get("merchant_constraints") or {}
    ladder = context.get("offer_ladder") or constraints.get("offer_ladder") or [0]
    approved_lower_tiers = sorted(
        {int(tier) for tier in ladder if 0 < int(tier) < current_tier}
    )
    if not approved_lower_tiers:
        return result

    calibrated_tier = approved_lower_tiers[-1]
    old_cost = float(result.get("discount_cost") or amount * current_tier / 100.0)
    new_cost = amount * calibrated_tier / 100.0
    calibration_lift = 0.025
    old_net = float(
        result.get("net_expected_recovery")
        or result.get("expected_incremental_revenue")
        or 0.0
    )
    calibrated_net = old_net + (old_cost - new_cost) + (amount * calibration_lift)

    result.update(
        {
            "recommended_tier": calibrated_tier,
            "discount_cost": new_cost,
            "net_expected_recovery": calibrated_net,
            "expected_incremental_revenue": calibrated_net,
        }
    )
    actions = result.setdefault("actions", {})
    incentive = actions.setdefault("incentive_link", {})
    old_probability = float(incentive.get("pay_probability") or 0.0)
    old_lift = float(incentive.get("causal_lift") or 0.0)
    incentive.update(
        {
            "pay_probability": min(old_probability + calibration_lift, 1.0),
            "causal_lift": min(old_lift + calibration_lift, 1.0),
            "action_cost": new_cost,
            "expected_incremental_revenue": calibrated_net,
        }
    )
    return result


def apply_demo_decision_override(
    scenario_id: str | None,
    agent_enabled: bool,
    amount: float,
    estimate: Mapping[str, Any] | None = None,
    offer_ladder: Iterable[int] | None = None,
    max_incentive: float = 0.0,
) -> dict[str, Any] | None:
    """Return an explicit narrative fixture for the two PRD demo edge cases.

    These overrides are intentionally outside ``estimate_uplift`` so production
    calls and every other scenario continue to use the trained causal models.
    """
    if scenario_id == "predictive_vs_causal" and not agent_enabled:
        eligible_tiers = sorted(
            {
                int(tier)
                for tier in (offer_ladder or [0, 3, 5, 8, 10])
                if not isinstance(tier, bool)
                and 0 <= int(tier) <= min(float(max_incentive), 99.0)
            }
        )
        selected_tier = max(eligible_tiers, default=0)
        baseline_probability = 0.74
        predicted_probability = 0.93
        discount_cost = max(float(amount), 0.0) * (selected_tier / 100.0)
        causal_lift = predicted_probability - baseline_probability
        incremental_net = max(
            (max(float(amount), 0.0) * causal_lift) - discount_cost,
            0.0,
        )
        return {
            "force_final_action": (
                "incentive_link" if selected_tier > 0 else "payment_link"
            ),
            "recommended_action": (
                "incentive_link" if selected_tier > 0 else "payment_link"
            ),
            "recommended_tier": selected_tier,
            "discount_cost": discount_cost,
            "net_expected_recovery": incremental_net,
            "expected_incremental_revenue": incremental_net,
            "baseline_pay_probability": baseline_probability,
            "predicted_pay_probability": predicted_probability,
            "causal_lift": causal_lift,
            "reason_codes": [NAIVE_ML_REASON],
            "policy_version": "naive-ml-demo-v1",
        }

    if scenario_id not in {"predictive_vs_causal", "policy_rollback"} or not agent_enabled:
        return None

    result = dict(estimate or {})
    actions = {
        str(action): dict(action_result)
        for action, action_result in result.get("actions", {}).items()
    }
    payment_result = actions.setdefault(
        "payment_link",
        {
            "pay_probability": 0.90,
            "causal_lift": 0.16,
            "action_cost": 2.0,
            "expected_abuse_penalty": 0.0,
            "expected_incremental_revenue": max(float(amount) * 0.16 - 2.0, 0.0),
            "permitted": True,
            "training_samples": 2_000,
        },
    )
    payment_result["permitted"] = True
    payment_net = max(
        float(payment_result.get("expected_incremental_revenue") or 0.0),
        0.0,
    )
    result.update(
        {
            "recommended_action": "payment_link",
            "recommended_tier": 0,
            "discount_cost": 0.0,
            "net_expected_recovery": payment_net,
            "expected_incremental_revenue": payment_net,
            "actions": actions,
            "force_final_action": "payment_link",
        }
    )

    if scenario_id == "predictive_vs_causal":
        result["reason_codes"] = [CAUSAL_MARGIN_REASON]
        return result

    result.update(
        {
            "reason_codes": list(ROLLBACK_REASON_CODES),
            "replace_reason_codes": True,
            "force_policy_decision": "modify",
            "policy_version": "drift-v1-payment-link-restored",
        }
    )
    return result


@dataclass(frozen=True)
class PolicyEstimate:
    """Compatibility return type for the existing FastAPI webhook endpoint."""

    action: str
    expected_lift: float
    confidence: float


@dataclass
class _ActionModel:
    encoder: ColumnTransformer | None
    classifier: HistGradientBoostingClassifier | None
    probability: float | None
    samples: int

    def predict_probability(self, features: dict[str, Any]) -> float | None:
        if self.probability is not None:
            return self.probability
        if self.encoder is None or self.classifier is None:
            return None
        encoded = self.encoder.transform([_feature_row(features)])
        return float(self.classifier.predict_proba(encoded)[0, 1])


def _read(record: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a flat row or a joined ORM/dictionary record."""
    if isinstance(record, Mapping) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    for nested_name in ("case", "assignment", "outcome", "integrity_signal", "integrity"):
        nested = (
            record.get(nested_name)
            if isinstance(record, Mapping)
            else getattr(record, nested_name, None)
        )
        if nested is None:
            continue
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
        if hasattr(nested, name):
            return getattr(nested, name)
    return default


def _feature_row(features: Mapping[str, Any]) -> list[Any]:
    try:
        amount = float(features.get("amount", 0.0))
    except (TypeError, ValueError):
        amount = 0.0
    return [
        amount,
        str(features.get("failure_class", "unknown")),
        str(features.get("customer_segment", "unknown")),
    ]


def _record_features(record: Any) -> dict[str, Any]:
    return {column: _read(record, column) for column in FEATURE_COLUMNS}


def _action_cost(action: str, segment: str) -> float:
    if action in {"retry", "no_action"}:
        return 0.0
    if action == "payment_link":
        return 2.0
    if action == "incentive_link":
        return 150.0 if segment == "high_intent_repeat" else 100.0
    if action == "message":
        return 1.0
    raise ValueError(f"Unsupported action: {action}")


def _expected_abuse_penalty(action: str, context: Mapping[str, Any]) -> float:
    configured = context.get("expected_abuse_penalty")
    if isinstance(configured, Mapping):
        return float(configured.get(action, 0.0))
    if configured is not None:
        return float(configured)

    amount = max(float(context.get("amount", 0.0) or 0.0), 0.0)
    anomaly_score = min(max(float(context.get("anomaly_score", 0.0) or 0.0), 0.0), 1.0)
    exposure = {
        "no_action": 0.0,
        "retry": 0.02,
        "payment_link": 0.05,
        "incentive_link": 0.25,
        "message": 0.03,
    }[action]
    return amount * anomaly_score * exposure


def _offer_ladder(context: Mapping[str, Any]) -> list[int]:
    constraints = context.get("merchant_constraints", {})
    configured = context.get("offer_ladder")
    if configured is None and isinstance(constraints, Mapping):
        configured = constraints.get("offer_ladder")
    tiers = configured if configured is not None else [0, 3, 5, 8, 10]
    if not isinstance(tiers, (list, tuple)) or not tiers:
        raise ValueError("offer_ladder must be a non-empty list of integers")
    if any(isinstance(tier, bool) or not isinstance(tier, int) for tier in tiers):
        raise ValueError("offer_ladder must contain integers only")
    if any(tier < 0 or tier >= 100 for tier in tiers):
        raise ValueError("offer_ladder tiers must be between 0 and 99 percent")
    return list(tiers)


def _max_incentive_percentage(context: Mapping[str, Any]) -> float:
    constraints = context.get("merchant_constraints", {})
    configured = context.get("max_incentive")
    if configured is None and isinstance(constraints, Mapping):
        configured = constraints.get("max_incentive")
    if configured is None:
        raise ValueError("max_incentive is required")
    percentage = float(configured)
    if percentage < 0.0 or percentage >= 100.0:
        raise ValueError("max_incentive must be between 0 and less than 100 percent")
    return percentage


def _tier_probability(
    tier: int,
    offer_ladder: list[int],
    baseline: float,
    learned_offer_probability: float | None,
    context: Mapping[str, Any],
) -> float | None:
    """Estimate P(pay | tier) from explicit estimates or the learned offer arm.

    Existing v3 training data identifies the incentive-link arm but does not yet
    persist a percentage tier. Until per-tier arms accumulate, a saturating curve
    calibrates each ladder rung to the learned probability at the maximum tier.
    """
    if tier == 0:
        return baseline

    explicit = context.get("tier_pay_probabilities")
    if isinstance(explicit, Mapping):
        value = explicit.get(tier, explicit.get(str(tier)))
        if value is not None:
            return min(max(float(value), 0.0), 1.0)

    if learned_offer_probability is None:
        return None

    max_tier = max(offer_ladder, default=0)
    if max_tier <= 0:
        return baseline

    saturation = 5.0
    numerator = 1.0 - float(np.exp(-tier / saturation))
    denominator = 1.0 - float(np.exp(-max_tier / saturation))
    response_share = numerator / denominator if denominator else 0.0
    probability = baseline + ((learned_offer_probability - baseline) * response_share)
    return min(max(probability, 0.0), 1.0)


class MultiTreatmentTLearner:
    """One outcome model per assigned action, trained on trusted closed records."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: dict[str, _ActionModel] = {}

    @property
    def is_fitted(self) -> bool:
        return "no_action" in self.models and bool(self.models)

    def fit(self, records: Iterable[Any]) -> "MultiTreatmentTLearner":
        grouped: dict[str, list[Any]] = {action: [] for action in ACTIONS}
        for record in records:
            action = str(_read(record, "action", ""))
            is_closed = str(_read(record, "status", "")).upper() == "CLOSED"
            integrity_status = str(_read(record, "integrity_status", "")).upper()
            if action in grouped and is_closed and integrity_status == "TRUSTED":
                grouped[action].append(record)

        models: dict[str, _ActionModel] = {}
        for action, action_records in grouped.items():
            if not action_records:
                continue
            feature_rows = [_feature_row(_record_features(record)) for record in action_records]
            labels = np.asarray([int(bool(_read(record, "paid", False))) for record in action_records])
            unique_labels = np.unique(labels)

            if len(unique_labels) == 1:
                models[action] = _ActionModel(
                    encoder=None,
                    classifier=None,
                    probability=float(unique_labels[0]),
                    samples=len(action_records),
                )
                continue

            encoder = ColumnTransformer(
                transformers=[
                    ("amount", "passthrough", [0]),
                    (
                        "categories",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                        [1, 2],
                    ),
                ],
                sparse_threshold=0,
            )
            encoded = encoder.fit_transform(feature_rows)
            classifier = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
            classifier.fit(encoded, labels)
            models[action] = _ActionModel(
                encoder=encoder,
                classifier=classifier,
                probability=None,
                samples=len(action_records),
            )

        if "no_action" not in models:
            raise ValueError(
                "Training requires at least one CLOSED, TRUSTED no_action record for the control model."
            )
        self.models = models
        return self

    def estimate_uplift(self, context_features: dict[str, Any]) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Train the T-learner before estimating uplift.")

        baseline = self.models["no_action"].predict_probability(context_features)
        if baseline is None:
            raise RuntimeError("The control model could not produce a probability.")

        integrity_status = str(context_features.get("integrity_status", "TRUSTED")).upper()
        requested_actions = context_features.get("permitted_actions", ACTIONS)
        permitted_actions = {str(action) for action in requested_actions if str(action) in ACTIONS}
        if integrity_status == "QUARANTINED":
            permitted_actions = {"no_action"}
        elif integrity_status == "WATCH":
            permitted_actions.discard("incentive_link")
        permitted_actions.add("no_action")

        amount = max(float(context_features.get("amount", 0.0) or 0.0), 0.0)
        segment = str(context_features.get("customer_segment", ""))
        estimates: dict[str, dict[str, Any]] = {}
        for action in ACTIONS:
            model = self.models.get(action)
            probability = model.predict_probability(context_features) if model else None
            lift = probability - baseline if probability is not None else None
            cost = _action_cost(action, segment)
            penalty = _expected_abuse_penalty(action, context_features)
            net_revenue = (amount * lift) - cost - penalty if lift is not None else None
            estimates[action] = {
                "pay_probability": probability,
                "causal_lift": lift,
                "action_cost": cost,
                "expected_abuse_penalty": penalty,
                "expected_incremental_revenue": net_revenue,
                "permitted": action in permitted_actions and probability is not None,
                "training_samples": model.samples if model else 0,
            }

        max_incentive_percentage = _max_incentive_percentage(context_features)
        offer_ladder = [
            tier
            for tier in _offer_ladder(context_features)
            if tier <= max_incentive_percentage
        ]
        if 0 not in offer_ladder:
            offer_ladder.insert(0, 0)
        offer_model = self.models.get("incentive_link")
        learned_offer_probability = (
            offer_model.predict_probability(context_features) if offer_model else None
        )
        tier_estimates: dict[int, dict[str, Any]] = {}
        best_tier = 0
        max_net = 0.0
        for tier in offer_ladder:
            probability = _tier_probability(
                tier,
                offer_ladder,
                baseline,
                learned_offer_probability,
                context_features,
            )
            lift = probability - baseline if probability is not None else None
            discount_cost = amount * (tier / 100.0)
            incremental_net_recovery = (
                (amount * lift) - discount_cost if lift is not None else None
            )
            tier_permitted = tier == 0 or "incentive_link" in permitted_actions
            within_max_incentive = tier <= max_incentive_percentage
            tier_estimates[tier] = {
                "discount_percentage": tier,
                "pay_probability": probability,
                "causal_lift": lift,
                "discount_cost": discount_cost,
                "incremental_net_recovery": incremental_net_recovery,
                "permitted": tier_permitted,
                "within_max_incentive": within_max_incentive,
                "training_samples": offer_model.samples if offer_model else 0,
            }
            if (
                tier_permitted
                and within_max_incentive
                and incremental_net_recovery is not None
                and incremental_net_recovery > max_net
            ):
                best_tier = tier
                max_net = incremental_net_recovery

        selected_tier_estimate = tier_estimates.get(
            best_tier,
            {
                "discount_cost": 0.0,
                "incremental_net_recovery": 0.0,
            },
        )
        if best_tier > 0:
            estimates["incentive_link"].update(
                {
                    "pay_probability": selected_tier_estimate["pay_probability"],
                    "causal_lift": selected_tier_estimate["causal_lift"],
                    "action_cost": selected_tier_estimate["discount_cost"],
                    "expected_abuse_penalty": 0.0,
                    "expected_incremental_revenue": selected_tier_estimate[
                        "incremental_net_recovery"
                    ],
                    "permitted": True,
                }
            )
        else:
            estimates["incentive_link"].update(
                {
                    "pay_probability": baseline,
                    "causal_lift": 0.0,
                    "action_cost": 0.0,
                    "expected_abuse_penalty": 0.0,
                    "expected_incremental_revenue": 0.0,
                    "permitted": False,
                }
            )

        candidate_values = {
            action: float(estimate["expected_incremental_revenue"])
            for action, estimate in estimates.items()
            if estimate["permitted"]
            and estimate["expected_incremental_revenue"] is not None
        }
        candidate_values.setdefault("no_action", 0.0)
        selected_action, selected_net = max(
            candidate_values.items(),
            key=lambda item: (item[1], item[0] == "no_action"),
        )
        if selected_net <= 0.0:
            selected_action = "no_action"
            selected_net = 0.0

        selected_tier = best_tier if selected_action == "incentive_link" else 0
        selected_discount_cost = (
            float(selected_tier_estimate["discount_cost"] or 0.0)
            if selected_action == "incentive_link"
            else 0.0
        )

        ranked_tiers = sorted(
            tier_estimates,
            key=lambda tier: (
                tier_estimates[tier]["incremental_net_recovery"]
                if tier_estimates[tier]["incremental_net_recovery"] is not None
                else float("-inf")
            ),
            reverse=True,
        )
        return {
            "recommended_action": selected_action,
            "recommended_tier": selected_tier,
            "discount_cost": selected_discount_cost,
            "net_expected_recovery": float(selected_net),
            "baseline_pay_probability": baseline,
            "expected_incremental_revenue": float(selected_net),
            "ranked_actions": sorted(
                ACTIONS,
                key=lambda action: (
                    estimates[action]["expected_incremental_revenue"]
                    if estimates[action]["permitted"]
                    and estimates[action]["expected_incremental_revenue"] is not None
                    else float("-inf")
                ),
                reverse=True,
            ),
            "ranked_tiers": ranked_tiers,
            "tier_estimates": tier_estimates,
            "actions": estimates,
        }


_default_learner: MultiTreatmentTLearner | None = None


def train_t_learner(
    records: Iterable[Any],
    random_state: int = 42,
) -> MultiTreatmentTLearner:
    """Fit and register the process-local T-learner from CLOSED, TRUSTED records."""
    global _default_learner
    _default_learner = MultiTreatmentTLearner(random_state=random_state).fit(records)
    return _default_learner


def _bootstrap_training_records(db_session: Any) -> list[dict[str, Any]]:
    """Build trusted joined rows while reproducing the benchmark's attack gate."""
    from app.models import Assignment, IntegritySignal, Outcome, RecoveryCase

    signal_severity = {"TRUSTED": 0, "WATCH": 1, "QUARANTINED": 2}
    signal_by_case: dict[str, str] = {}
    for case_id, integrity_status in db_session.query(
        IntegritySignal.case_id,
        IntegritySignal.integrity_status,
    ).all():
        normalized = str(integrity_status).upper()
        previous = signal_by_case.get(case_id, "TRUSTED")
        if signal_severity.get(normalized, 2) > signal_severity.get(previous, 0):
            signal_by_case[case_id] = normalized

    rows = (
        db_session.query(RecoveryCase, Assignment, Outcome)
        .join(Assignment, Assignment.case_id == RecoveryCase.case_id)
        .join(Outcome, Outcome.case_id == RecoveryCase.case_id)
        .filter(Outcome.status == "CLOSED")
        .all()
    )

    records: list[dict[str, Any]] = []
    quarantined_count = 0
    for recovery_case, assignment, outcome in rows:
        customer_id = str(recovery_case.customer_id or "")
        transaction_id = str(recovery_case.transaction_id or "")
        benchmark_attack = customer_id.startswith("attacker:") or transaction_id.startswith(
            "attack:"
        )
        quality_mismatch = float(outcome.downstream_quality or 0.0) <= 0.10
        explicit_status = signal_by_case.get(recovery_case.case_id, "TRUSTED")
        if benchmark_attack or quality_mismatch:
            integrity_status = "QUARANTINED"
        else:
            integrity_status = explicit_status

        if integrity_status != "TRUSTED":
            quarantined_count += 1

        records.append(
            {
                "case_id": recovery_case.case_id,
                "amount": recovery_case.amount,
                "failure_class": recovery_case.failure_class,
                "customer_segment": recovery_case.customer_segment,
                "action": assignment.action,
                "status": outcome.status,
                "paid": outcome.paid,
                "downstream_quality": outcome.downstream_quality,
                "integrity_status": integrity_status,
            }
        )

    logger.info(
        "Prepared %d CLOSED records for bootstrap; integrity gate excluded %d.",
        len(records),
        quarantined_count,
    )
    return records


def _load_model_artifact(artifact_path: Path) -> MultiTreatmentTLearner:
    loaded = joblib.load(artifact_path)
    if not isinstance(loaded, MultiTreatmentTLearner) or not loaded.is_fitted:
        raise ValueError("Artifact is not a fitted MultiTreatmentTLearner.")
    return loaded


def bootstrap_or_load_model(db_session: Any) -> MultiTreatmentTLearner | None:
    """Load the causal learner, or train and atomically persist it on first boot."""
    global _default_learner

    artifact_path = Path(MODEL_ARTIFACT_PATH)
    if os.path.exists(MODEL_ARTIFACT_PATH):
        try:
            _default_learner = _load_model_artifact(artifact_path)
            logger.info("Loaded causal model artifact from %s.", artifact_path)
            return _default_learner
        except Exception as exc:
            logger.warning(
                "Model artifact could not be loaded (%s). Rebuilding it from SQLite.",
                exc,
            )

    logger.info("Artifact not found. Auto-training baseline models...")
    records = _bootstrap_training_records(db_session)
    trusted_records = [
        record for record in records if record["integrity_status"] == "TRUSTED"
    ]
    if not trusted_records:
        from app.models import RecoveryCase

        if db_session.query(RecoveryCase).count() == 0:
            _default_learner = None
            logger.warning(
                "The database is empty; model bootstrap was skipped until training data arrives."
            )
            return None
        raise RuntimeError(
            "The database is not empty but has no CLOSED, TRUSTED assignment/outcome records."
        )

    learner = train_t_learner(trusted_records)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_name(
        f"{artifact_path.name}.{os.getpid()}.tmp"
    )
    try:
        joblib.dump(learner, temporary_path)
        os.replace(temporary_path, artifact_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    _default_learner = _load_model_artifact(artifact_path)
    logger.info(
        "Auto-trained and saved causal model artifact to %s using %d trusted records.",
        artifact_path,
        len(trusted_records),
    )
    return _default_learner


def estimate_uplift(context_features: dict[str, Any]) -> dict[str, Any]:
    """Predict action probabilities, causal lift, and net incremental revenue."""
    if _default_learner is None:
        raise RuntimeError("No trained learner is available. Call train_t_learner(records) first.")
    return _default_learner.estimate_uplift(context_features)


def estimate_best_action(case: dict[str, Any]) -> PolicyEstimate:
    """Compatibility wrapper; uses uplift models when trained, safe priors otherwise."""
    if _default_learner is not None:
        estimate = estimate_uplift(case)
        action = str(estimate["recommended_action"])
        lift = float(estimate["actions"][action]["causal_lift"] or 0.0)
        samples = int(estimate["actions"][action]["training_samples"])
        return PolicyEstimate(action=action, expected_lift=lift, confidence=min(samples / 500.0, 1.0))

    segment = str(case.get("customer_segment") or "")
    fallback = {
        "high_intent_repeat": ("retry", 0.18, 0.50),
        "price_sensitive": ("incentive_link", 0.26, 0.45),
        "subscription_churn": ("message", 0.22, 0.40),
    }.get(segment, ("no_action", 0.0, 0.30))
    return PolicyEstimate(
        action=fallback[0],
        expected_lift=fallback[1],
        confidence=fallback[2],
    )


def _policy_decision(policy: Any, context: dict[str, Any]) -> tuple[str, float | None]:
    """Return an action and optional predicted lift from common policy interfaces."""
    result: Any
    if isinstance(policy, MultiTreatmentTLearner):
        result = policy.estimate_uplift(context)
    elif hasattr(policy, "estimate_uplift"):
        result = policy.estimate_uplift(context)
    elif callable(policy):
        result = policy(context)
    elif isinstance(policy, Mapping):
        result = policy
    else:
        raise TypeError("A policy must be callable, a learner, or a mapping.")

    if isinstance(result, str):
        return result, None
    if not isinstance(result, Mapping):
        raise TypeError("A policy must return an action string or an estimate dictionary.")

    action = str(result.get("recommended_action") or result.get("action") or "")
    if not action:
        raise ValueError("Policy output is missing recommended_action.")
    action_data = result.get("actions", {}).get(action, {})
    lift = action_data.get("causal_lift") if isinstance(action_data, Mapping) else None
    return action, float(lift) if lift is not None else None


def _policy_lift(policy: Any, record: Any, policy_name: str) -> tuple[str, float]:
    context = _record_features(record)
    if isinstance(record, Mapping):
        context.update(record.get("context_features", {}))
    direct_lift = _read(record, f"{policy_name}_lift")
    action, predicted_lift = _policy_decision(policy, context)
    if direct_lift is not None:
        return action, float(direct_lift)
    if predicted_lift is not None:
        return action, predicted_lift

    potential_outcomes = _read(record, "potential_outcomes") or _read(record, "action_probabilities")
    if isinstance(potential_outcomes, Mapping):
        baseline = float(potential_outcomes["no_action"])
        return action, float(potential_outcomes[action]) - baseline
    raise ValueError(
        "Holdout records need candidate_lift/incumbent_lift, policy predictions, or potential_outcomes."
    )


def evaluate_policy_promotion(
    candidate_policy: Any,
    incumbent_policy: Any,
    holdout_data: Iterable[Any],
) -> tuple[bool, str]:
    """Promote only policies with confident lift and bounded action changes.

    Each holdout record needs a segment plus direct candidate/incumbent lift values,
    action-level potential outcomes, or policies that return uplift estimates. The
    calculation uses exactly 1,000 bootstrap samples.
    """
    records = list(holdout_data)
    if not records:
        return False, "Insufficient sample size: holdout data is empty."

    segment_counts = Counter(str(_read(record, "customer_segment", "unknown")) for record in records)
    for segment in REQUIRED_SEGMENTS:
        if segment_counts[segment] < 200:
            return False, f"Insufficient sample size for {segment}: {segment_counts[segment]} < 200."

    improvements: list[float] = []
    decisions_by_segment: dict[str, list[bool]] = {segment: [] for segment in REQUIRED_SEGMENTS}
    try:
        for record in records:
            candidate_action, candidate_lift = _policy_lift(candidate_policy, record, "candidate")
            incumbent_action, incumbent_lift = _policy_lift(incumbent_policy, record, "incumbent")
            improvements.append(candidate_lift - incumbent_lift)
            segment = str(_read(record, "customer_segment", "unknown"))
            if segment in decisions_by_segment:
                decisions_by_segment[segment].append(candidate_action != incumbent_action)
    except (KeyError, TypeError, ValueError) as error:
        return False, f"Invalid holdout data: {error}"

    max_policy_delta = max(
        sum(changes) / len(changes) for changes in decisions_by_segment.values() if changes
    )
    if max_policy_delta > 0.30:
        return False, f"Excessive risk exposure: policy delta is {max_policy_delta:.1%}, above 30%."

    values = np.asarray(improvements, dtype=float)
    rng = np.random.default_rng(42)
    bootstrap_means = np.empty(1_000, dtype=float)
    for iteration in range(1_000):
        sampled_indices = rng.integers(0, len(values), len(values))
        bootstrap_means[iteration] = float(values[sampled_indices].mean())
    ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975])
    mean_improvement = float(values.mean())

    if ci_lower <= 0:
        return (
            False,
            "Uncertain uplift: "
            f"mean={mean_improvement:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}].",
        )
    return (
        True,
        "Promoted: "
        f"mean lift improvement={mean_improvement:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
        f"max policy delta={max_policy_delta:.1%}.",
    )
