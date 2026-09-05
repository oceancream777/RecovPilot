#!/usr/bin/env python3
"""Benchmark static, naive ML, and trusted causal recovery policies.

The simulator's action probabilities are deliberately repeated here as hidden ground
truth for offline evaluation. They are never provided to either learned policy.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# Avoid a slow physical-core probe in restricted desktop environments.
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine


# Support direct execution with `python scripts/run_benchmark.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.baseline_policy import decide_incumbent_recovery  # noqa: E402
from app.learner import ACTIONS, MultiTreatmentTLearner  # noqa: E402


DEFAULT_DATABASE_URL = "sqlite:///./recovery_agent.db"
EXPECTED_CASE_COUNT = 5_500
EXPECTED_POLICY_VERSION = "ground-truth-simulator-v2"
ATTACK_CONVERSION_RATE = 0.98

# These probabilities are the simulator's ground truth, used only after a policy
# selects an action so the benchmark can estimate the counterfactual outcome.
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


@dataclass(frozen=True)
class RoundMetrics:
    agent: str
    incremental_lift_pp: float
    incremental_revenue: float
    intervention_cost: float
    abuse_exposure: float
    net_recovery_roi: float
    poisoning_resistant: bool
    attacker_incentive_assignments: int
    attacks_quarantined: int


@dataclass(frozen=True)
class PolicySelection:
    actions: np.ndarray
    discount_tiers: np.ndarray
    applies_integrity_quarantine: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recovery policies against the synthetic SQLite environment."
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help=f"SQLAlchemy database URL (default: {DEFAULT_DATABASE_URL}).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of seeded 80/20 simulation rounds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    return args


def load_environment(database_url: str) -> pd.DataFrame:
    """Load one simulator population with its randomized action and outcome."""
    engine = create_engine(database_url)
    query = """
        SELECT
            recovery_cases.case_id,
            recovery_cases.merchant_id,
            recovery_cases.transaction_id,
            recovery_cases.customer_id,
            recovery_cases.amount,
            recovery_cases.case_age_hours,
            recovery_cases.attempt_count,
            recovery_cases.payment_method,
            recovery_cases.failure_class,
            recovery_cases.customer_segment,
            recovery_cases.event_time,
            assignments.action AS assigned_action,
            assignments.assignment_type,
            assignments.policy_version,
            outcomes.status AS outcome_status,
            outcomes.paid,
            outcomes.downstream_quality
        FROM recovery_cases
        INNER JOIN assignments ON assignments.case_id = recovery_cases.case_id
        INNER JOIN outcomes ON outcomes.case_id = recovery_cases.case_id
    """
    try:
        data = pd.read_sql_query(query, engine)
    except Exception as error:
        raise RuntimeError(
            "Could not load the v3 simulator tables. Run "
            "`python scripts/simulate_environment.py` against a fresh database first."
        ) from error
    finally:
        engine.dispose()

    if len(data) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_CASE_COUNT:,} simulator cases, found {len(data):,}. "
            "Use a fresh database and run scripts/simulate_environment.py once."
        )

    policy_versions = set(data["policy_version"].dropna().astype(str))
    if policy_versions != {EXPECTED_POLICY_VERSION}:
        raise RuntimeError(
            "The SQLite environment uses an older simulator contract. Run "
            "`python scripts/simulate_environment.py --reset` before benchmarking."
        )

    data["paid"] = data["paid"].astype(bool)
    data["is_attack"] = data["transaction_id"].str.startswith("attack:", na=False) | data[
        "customer_id"
    ].str.startswith("attacker:", na=False)
    attack_count = int(data["is_attack"].sum())
    if attack_count != 500:
        raise RuntimeError(
            f"Expected 500 incentive-farming attacks, found {attack_count}. "
            "Use a fresh database and run scripts/simulate_environment.py once."
        )
    data["integrity_status"] = np.where(data["is_attack"], "QUARANTINED", "TRUSTED")
    data["status"] = data["outcome_status"]
    return data


def static_actions(data: pd.DataFrame) -> PolicySelection:
    """Run the research-calibrated incumbent recovery state machine."""
    constraints = {
        "merchant_budget": 1_000_000_000.0,
        "merchant_budget_spent": 0.0,
        "max_incentive": 10.0,
        "max_contacts": 3,
        "contacts_used": 0,
        "recovery_window_hours": 48.0,
        "offer_ladder": [0, 3, 5, 8, 10],
        "allowed_actions": list(ACTIONS),
    }
    actions: list[str] = []
    tiers: list[int] = []
    for record in data.to_dict("records"):
        decision = decide_incumbent_recovery(
            record,
            constraints,
            list(ACTIONS),
            "off_vs_on_casino",
        )
        actions.append(decision.final_action)
        tiers.append(decision.discount_tier)
    return PolicySelection(
        actions=np.asarray(actions, dtype=object),
        discount_tiers=np.asarray(tiers, dtype=int),
    )


def train_naive_model(train_data: pd.DataFrame) -> Pipeline:
    """Fit a single contaminated payment-propensity model with action as a feature."""
    features = ["amount", "failure_class", "customer_segment", "assigned_action"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount", StandardScaler(), ["amount"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["failure_class", "customer_segment", "assigned_action"],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("features", preprocessor),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    model.fit(train_data[features], train_data["paid"].astype(int))
    return model


def naive_actions(model: Pipeline, data: pd.DataFrame) -> PolicySelection:
    """Choose the action with highest raw payment probability, ignoring economics."""
    candidate_scores: list[np.ndarray] = []
    base_features = data[["amount", "failure_class", "customer_segment"]]
    for action in ACTIONS:
        candidate = base_features.copy()
        candidate["assigned_action"] = action
        candidate_scores.append(model.predict_proba(candidate)[:, 1])
    score_matrix = np.column_stack(candidate_scores)
    actions = np.asarray(ACTIONS)[np.argmax(score_matrix, axis=1)]
    return PolicySelection(
        actions=actions,
        discount_tiers=np.where(actions == "incentive_link", 10, 0),
    )


def train_recovery_learner(train_data: pd.DataFrame, seed: int) -> MultiTreatmentTLearner:
    """Fit one response model per action after quarantining attack records."""
    training_columns = [
        "amount",
        "failure_class",
        "customer_segment",
        "assigned_action",
        "status",
        "integrity_status",
        "paid",
    ]
    records = train_data[training_columns].rename(columns={"assigned_action": "action"})
    return MultiTreatmentTLearner(random_state=seed).fit(records.to_dict("records"))


def recovery_actions(learner: MultiTreatmentTLearner, data: pd.DataFrame) -> PolicySelection:
    """Choose each case's highest net incremental-value permitted action.

    This batch form makes the same T-learner potential-outcome comparison as
    ``estimate_uplift`` while keeping five full simulation rounds fast on a laptop.
    """
    feature_rows = data[["amount", "failure_class", "customer_segment"]].values.tolist()
    amounts = data["amount"].to_numpy(dtype=float)
    attacks = data["is_attack"].to_numpy(dtype=bool)
    probabilities: dict[str, np.ndarray] = {}

    for action in ACTIONS:
        action_model = learner.models.get(action)
        if action_model is None:
            probabilities[action] = np.full(len(data), np.nan)
        elif action_model.probability is not None:
            probabilities[action] = np.full(len(data), action_model.probability)
        else:
            encoded = action_model.encoder.transform(feature_rows)
            probabilities[action] = action_model.classifier.predict_proba(encoded)[:, 1]

    baseline = probabilities["no_action"]
    action_values: list[np.ndarray] = []
    for action in ACTIONS:
        if action == "no_action":
            cost = np.zeros(len(data))
        elif action == "retry":
            cost = np.zeros(len(data))
        elif action == "payment_link":
            cost = np.full(len(data), 2.0)
        elif action == "incentive_link":
            cost = amounts * 0.10
        else:  # message
            cost = np.full(len(data), 1.0)

        values = amounts * (probabilities[action] - baseline) - cost
        # The integrity gate permits only no_action for the attack cohort.
        if action != "no_action":
            values = np.where(attacks, -np.inf, values)
        action_values.append(values)
    actions = np.asarray(ACTIONS)[np.argmax(np.column_stack(action_values), axis=1)]
    upi_exhausted = data["payment_method"].eq("upi").to_numpy() & data[
        "attempt_count"
    ].ge(4).to_numpy()
    actions[upi_exhausted] = "no_action"
    actions[(amounts > 15_000) & (actions == "retry")] = "payment_link"
    return PolicySelection(
        actions=actions,
        discount_tiers=np.where(actions == "incentive_link", 10, 0),
        applies_integrity_quarantine=True,
    )


def ground_truth_probability(
    segment: str,
    action: str,
    is_attack: bool,
    discount_tier: int = 0,
) -> float:
    """Return the simulator's hidden pay probability for a proposed intervention."""
    if action == "suppress":
        action = "no_action"
    if is_attack:
        return ATTACK_CONVERSION_RATE if action == "incentive_link" else 0.0
    probability = GROUND_TRUTH_PAY_PROBABILITIES[segment][action]
    if action != "incentive_link":
        return probability
    baseline = GROUND_TRUTH_PAY_PROBABILITIES[segment]["no_action"]
    response_share = min(max(discount_tier, 0) / 10.0, 1.0)
    return baseline + (probability - baseline) * response_share


def action_cost(amount: float, action: str, discount_tier: int) -> float:
    if action in {"no_action", "suppress", "retry"}:
        return 0.0
    if action == "payment_link":
        return 2.0
    if action == "incentive_link":
        return 2.0 + amount * discount_tier / 100.0
    if action == "message":
        return 1.0
    raise ValueError(f"Unknown action: {action}")


def evaluate_actions(
    agent: str,
    test_data: pd.DataFrame,
    selection: PolicySelection,
    outcome_uniforms: np.ndarray,
) -> RoundMetrics:
    """Simulate actual outcomes with common random numbers for fair comparisons."""
    selected_actions = selection.actions
    selected_tiers = selection.discount_tiers
    action_probabilities = np.asarray(
        [
            ground_truth_probability(segment, action, is_attack, int(tier))
            for segment, action, is_attack, tier in zip(
                test_data["customer_segment"],
                selected_actions,
                test_data["is_attack"],
                selected_tiers,
                strict=True,
            )
        ]
    )
    baseline_probabilities = np.asarray(
        [
            ground_truth_probability(segment, "no_action", is_attack)
            for segment, is_attack in zip(
                test_data["customer_segment"],
                test_data["is_attack"],
                strict=True,
            )
        ]
    )
    action_paid = outcome_uniforms < action_probabilities
    baseline_paid = outcome_uniforms < baseline_probabilities
    attack_incentive = test_data["is_attack"].to_numpy() & (selected_actions == "incentive_link")

    # Attack payments are deliberate label poison: a chargeback/cancellation makes
    # their downstream quality zero, so they produce no recovered merchant revenue.
    downstream_quality = np.where(attack_incentive, 0.0, 1.0)
    action_recovery = action_paid.astype(float) * downstream_quality
    baseline_recovery = baseline_paid.astype(float)
    amounts = test_data["amount"].to_numpy(dtype=float)

    incremental_revenue = float(np.sum(amounts * (action_recovery - baseline_recovery)))
    incremental_lift_pp = float(100 * np.mean(action_recovery - baseline_recovery))
    intervention_cost = float(
        sum(
            action_cost(float(amount), action, int(tier))
            for amount, action, tier in zip(
                test_data["amount"],
                selected_actions,
                selected_tiers,
                strict=True,
            )
        )
    )
    # The full apparent attacker payment is expected to be reversed downstream.
    abuse_exposure = float(np.sum(amounts * action_paid * attack_incentive))
    net_recovery_roi = (
        (incremental_revenue - intervention_cost - abuse_exposure)
        / max(intervention_cost + abuse_exposure, 1.0)
    )
    attacker_incentives = int(np.sum(attack_incentive))
    attacks_quarantined = (
        int(np.sum(test_data["is_attack"].to_numpy()))
        if selection.applies_integrity_quarantine
        else 0
    )
    return RoundMetrics(
        agent=agent,
        incremental_lift_pp=incremental_lift_pp,
        incremental_revenue=incremental_revenue,
        intervention_cost=intervention_cost,
        abuse_exposure=abuse_exposure,
        net_recovery_roi=net_recovery_roi,
        poisoning_resistant=attacker_incentives == 0,
        attacker_incentive_assignments=attacker_incentives,
        attacks_quarantined=attacks_quarantined,
    )


def run_round(data: pd.DataFrame, round_number: int, seed: int) -> list[RoundMetrics]:
    split_labels = data["customer_segment"].astype(str) + ":" + data["is_attack"].astype(str)
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=seed + round_number,
        stratify=split_labels,
    )
    test_data = test_data.reset_index(drop=True)
    outcome_uniforms = np.random.default_rng(seed + 10_000 + round_number).random(len(test_data))

    print(f"  Training Naive ML agent for round {round_number}...")
    naive_model = train_naive_model(train_data)
    print(f"  Training trusted multi-treatment T-Learner for round {round_number}...")
    recovery_learner = train_recovery_learner(train_data, seed + round_number)

    policies = {
        "Razorpay Incumbent Simulation": static_actions(test_data),
        "Naive ML Agent": naive_actions(naive_model, test_data),
        "Recovery Learning Agent (v3.5)": recovery_actions(recovery_learner, test_data),
    }
    return [
        evaluate_actions(agent, test_data, actions, outcome_uniforms)
        for agent, actions in policies.items()
    ]


def summarize_metrics(metrics: list[RoundMetrics]) -> pd.DataFrame:
    rows = pd.DataFrame([metric.__dict__ for metric in metrics])
    summary = rows.groupby("agent", as_index=False).agg(
        incremental_lift_pp=("incremental_lift_pp", "mean"),
        incremental_revenue=("incremental_revenue", "mean"),
        intervention_cost=("intervention_cost", "mean"),
        abuse_exposure=("abuse_exposure", "mean"),
        net_recovery_roi=("net_recovery_roi", "mean"),
        poisoning_resistant=("poisoning_resistant", "all"),
        attacker_incentive_assignments=("attacker_incentive_assignments", "sum"),
        attacks_quarantined=("attacks_quarantined", "sum"),
    )
    ordering = {
        "Razorpay Incumbent Simulation": 0,
        "Naive ML Agent": 1,
        "Recovery Learning Agent (v3.5)": 2,
    }
    return summary.sort_values("agent", key=lambda values: values.map(ordering))


def render_markdown_table(summary: pd.DataFrame) -> str:

    lines = [
        "| Agent | Incremental Recovery Lift | Incremental Recovered Revenue | Net Recovery ROI | Attacks Quarantined | Poisoning Resistance |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        resistance = "Yes" if row.poisoning_resistant else "No"
        lines.append(
            "| "
            f"{row.agent} | {row.incremental_lift_pp:+.2f} pp | "
            f"₹{row.incremental_revenue:,.2f} | {row.net_recovery_roi:+.2f}x | "
            f"{row.attacks_quarantined:,.0f} | {resistance} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print("[1/3] Loading the 5,500-case synthetic environment from SQLite...")
    data = load_environment(args.database_url)
    attack_count = int(data["is_attack"].sum())
    print(f"      Loaded {len(data):,} cases: {len(data) - attack_count:,} clean, {attack_count:,} attacks.")
    print("[2/3] Running static, contaminated naive ML, and trusted causal policies...")

    all_metrics: list[RoundMetrics] = []
    for round_number in range(1, args.rounds + 1):
        print(f"Round {round_number}/{args.rounds}")
        all_metrics.extend(run_round(data, round_number, args.seed))

    print("[3/3] Aggregating held-out results across all simulation rounds...")
    print()
    summary = summarize_metrics(all_metrics)
    print(render_markdown_table(summary))

    static_revenue = float(
        summary.loc[
            summary["agent"] == "Razorpay Incumbent Simulation", "incremental_revenue"
        ].iloc[0]
    )
    recovery_row = summary.loc[summary["agent"] == "Recovery Learning Agent (v3.5)"].iloc[0]
    win_condition_passed = bool(
        recovery_row["incremental_revenue"] > static_revenue
        and recovery_row["poisoning_resistant"]
    )
    result = "PASS" if win_condition_passed else "FAIL"
    print()
    print(
        f"Win Condition: {result} - v3.5 revenue is ₹{recovery_row['incremental_revenue']:,.2f} "
        f"vs ₹{static_revenue:,.2f} for static, and poisoning resistance is "
        f"{'Yes' if recovery_row['poisoning_resistant'] else 'No'}."
    )


if __name__ == "__main__":
    main()
