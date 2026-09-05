from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RecoveryCase(Base):
    __tablename__ = "recovery_cases"

    case_id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    merchant_id = Column(String, nullable=False, index=True)
    transaction_id = Column(String, nullable=True, index=True)
    customer_id = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    case_age_hours = Column(Float, nullable=False, default=0.0, server_default="0.0")
    attempt_count = Column(Integer, nullable=False, default=1, server_default="1")
    payment_method = Column(
        String,
        nullable=False,
        default="card",
        server_default="card",
        index=True,
    )
    failure_class = Column(String, nullable=False, index=True)
    customer_segment = Column(String, nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    assignments = relationship(
        "Assignment",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    interventions = relationship(
        "Intervention",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    outcomes = relationship(
        "Outcome",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    integrity_signals = relationship(
        "IntegritySignal",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    audit_logs = relationship(
        "AuditLog",
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String, nullable=False, index=True)
    assignment_type = Column(String, nullable=False, index=True)
    policy_version = Column(String, nullable=False, index=True)
    assigned_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    case = relationship("RecoveryCase", back_populates="assignments")


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String, nullable=False, index=True)
    incentive_amount = Column(Float, nullable=False, default=0.0)
    channel = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)

    case = relationship("RecoveryCase", back_populates="interventions")


class Outcome(Base):
    __tablename__ = "outcomes"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, nullable=False, default="PENDING", index=True)
    paid = Column(Boolean, nullable=False, default=False)
    amount_paid = Column(Float, nullable=False, default=0.0)
    time_to_pay_seconds = Column(Integer, nullable=True)
    downstream_quality = Column(Float, nullable=False, default=1.0)
    outcome_time = Column(DateTime(timezone=True), nullable=True)

    case = relationship("RecoveryCase", back_populates="outcomes")


class IntegritySignal(Base):
    __tablename__ = "integrity_signals"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    signal_type = Column(String, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    integrity_status = Column(String, nullable=False, index=True)
    rationale = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    case = relationship("RecoveryCase", back_populates="integrity_signals")


class PolicyVersion(Base):
    __tablename__ = "policy_versions"

    version = Column(String, primary_key=True)
    model_hash = Column(String, nullable=False)
    action_rules_json = Column(Text, nullable=False)
    sample_size = Column(Integer, nullable=False)
    mean_lift = Column(Float, nullable=False)
    ci_lower = Column(Float, nullable=False)
    ci_upper = Column(Float, nullable=False)
    status = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class BatchPolicyRun(Base):
    __tablename__ = "batch_policy_runs"

    run_id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    trigger_source = Column(String, nullable=False, index=True)
    champion_version = Column(String, nullable=False)
    challenger_version = Column(String, nullable=False)
    incumbent_roi = Column(Float, nullable=False)
    challenger_roi = Column(Float, nullable=False)
    quarantine_rate = Column(Float, nullable=False)
    status = Column(String, nullable=False, index=True)
    rejection_reason = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()), index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    policy_version = Column(String, nullable=False, index=True)
    proposed_action = Column(String, nullable=False, index=True)
    final_action = Column(String, nullable=False, index=True)
    expected_incremental_revenue = Column(Float, nullable=False, default=0.0)
    integrity_verdict = Column(String, nullable=False, index=True)
    policy_decision = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    case = relationship("RecoveryCase", back_populates="audit_logs")
