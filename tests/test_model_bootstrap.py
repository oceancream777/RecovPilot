from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import learner
from app.database import Base
from app.models import Assignment, IntegritySignal, Outcome, RecoveryCase


def _seed_action(
    db,
    action: str,
    paid: bool,
    *,
    attack: bool = False,
    explicitly_quarantined: bool = False,
) -> None:
    case_id = str(uuid4())
    db.add(
        RecoveryCase(
            case_id=case_id,
            merchant_id="merchant-bootstrap",
            transaction_id=(f"attack:ip_hash:{uuid4().hex[:8]}" if attack else f"txn-{uuid4()}"),
            customer_id=(f"attacker:device_hash:{uuid4().hex[:8]}" if attack else f"customer-{uuid4()}"),
            amount=2500.0,
            failure_class="network_timeout",
            customer_segment="price_sensitive",
        )
    )
    db.add(
        Assignment(
            case_id=case_id,
            action=action,
            assignment_type="control" if action == "no_action" else "treatment",
            policy_version="bootstrap-test",
        )
    )
    db.add(
        Outcome(
            case_id=case_id,
            status="CLOSED",
            paid=paid,
            amount_paid=2500.0 if paid else 0.0,
            downstream_quality=0.0 if attack else 1.0,
        )
    )
    if explicitly_quarantined:
        db.add(
            IntegritySignal(
                case_id=case_id,
                signal_type="reward_farming",
                anomaly_score=0.99,
                integrity_status="QUARANTINED",
                rationale="bootstrap test quarantine",
            )
        )


def test_bootstrap_trains_filters_poison_and_then_loads_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    artifact_path = tmp_path / "app" / "artifacts" / "causal_models.joblib"
    monkeypatch.setattr(learner, "MODEL_ARTIFACT_PATH", str(artifact_path))
    monkeypatch.setattr(learner, "_default_learner", None)

    with testing_session() as db:
        for action in learner.ACTIONS:
            _seed_action(db, action, paid=False)
            _seed_action(db, action, paid=True)
        _seed_action(db, "incentive_link", paid=True, attack=True)
        _seed_action(
            db,
            "incentive_link",
            paid=True,
            explicitly_quarantined=True,
        )
        db.commit()

        trained = learner.bootstrap_or_load_model(db)
        assert trained is not None
        assert artifact_path.exists()
        assert trained.models["incentive_link"].samples == 2

        estimate = trained.estimate_uplift(
            {
                "amount": 2500.0,
                "failure_class": "network_timeout",
                "customer_segment": "price_sensitive",
                    "integrity_status": "TRUSTED",
                    "anomaly_score": 0.0,
                    "permitted_actions": list(learner.ACTIONS),
                    "max_incentive": 10.0,
                }
            )
        assert estimate["recommended_action"] in learner.ACTIONS
        assert set(estimate["actions"]) == set(learner.ACTIONS)
        assert estimate["actions"]["no_action"]["causal_lift"] == pytest.approx(0.0)
        for action_result in estimate["actions"].values():
            assert 0.0 <= action_result["pay_probability"] <= 1.0

        monkeypatch.setattr(learner, "_default_learner", None)

        def fail_if_retrained(*_args, **_kwargs):
            raise AssertionError("existing artifact should load without retraining")

        monkeypatch.setattr(learner.MultiTreatmentTLearner, "fit", fail_if_retrained)
        loaded = learner.bootstrap_or_load_model(db)

    assert loaded is not None
    assert loaded.is_fitted
    assert loaded.models["no_action"].samples == 2
