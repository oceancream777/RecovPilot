"""Integrity checks that prevent adversarial cases from reaching the learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models import IntegritySignal, RecoveryCase


VELOCITY_WATCH_THRESHOLD = 10
VELOCITY_QUARANTINE_THRESHOLD = 20
CONCENTRATION_WATCH_THRESHOLD = 6
CONCENTRATION_QUARANTINE_THRESHOLD = 15
QUALITY_MISMATCH_MINIMUM_CASES = 5


@dataclass(frozen=True)
class IntegrityVerdict:
    """Small compatibility value object used by the existing webhook endpoint."""

    signal_type: str
    anomaly_score: float
    integrity_status: str
    rationale: str


def _read(record: Any, name: str, default: Any = None) -> Any:
    """Read a field from flat or joined ORM/dictionary records."""
    if isinstance(record, Mapping) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)

    for nested_name in ("case", "outcome", "assignment", "intervention"):
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


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _device_hash(record: Any) -> str | None:
    explicit = _read(record, "device_hash") or _read(record, "device_id")
    if explicit:
        return str(explicit)

    customer_id = str(_read(record, "customer_id", ""))
    # The simulator stores fingerprints as attacker:<device-hash>:<unique-id>.
    parts = customer_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "attacker" else None


def _ip_hash(record: Any) -> str | None:
    explicit = _read(record, "ip_hash") or _read(record, "ip_address")
    if explicit:
        return str(explicit)

    transaction_id = str(_read(record, "transaction_id", ""))
    # The simulator stores fingerprints as attack:<ip-hash>:<unique-id>.
    parts = transaction_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "attack" else None


def _same_identity(case: RecoveryCase, record: Any) -> bool:
    if _read(record, "customer_id") == case.customer_id:
        return True
    case_device = _device_hash(case)
    case_ip = _ip_hash(case)
    return bool(
        (case_device and case_device == _device_hash(record))
        or (case_ip and case_ip == _ip_hash(record))
    )


def evaluate_case_integrity(
    case: RecoveryCase,
    recent_history: list[Any],
) -> IntegritySignal:
    """Return a persisted-model-ready integrity decision for one recovery case.

    ``recent_history`` may contain ORM records, dictionaries, or joined records with
    ``case`` and ``outcome`` properties. It should contain historical attempts only;
    the case under evaluation is counted automatically.
    """
    case_time = _as_datetime(case.event_time) or datetime.now(timezone.utc)
    one_minute_ago = case_time - timedelta(minutes=1)
    five_minutes_ago = case_time - timedelta(minutes=5)

    matching_records = [
        record
        for record in recent_history
        if _read(record, "case_id") != case.case_id and _same_identity(case, record)
    ]
    attempts_last_minute = 1 + sum(
        1
        for record in matching_records
        if (record_time := _as_datetime(_read(record, "event_time")))
        and one_minute_ago <= record_time <= case_time
    )

    device_matches = [
        record
        for record in recent_history
        if _read(record, "case_id") != case.case_id
        and _device_hash(case)
        and _device_hash(case) == _device_hash(record)
        and (record_time := _as_datetime(_read(record, "event_time")))
        and five_minutes_ago <= record_time <= case_time
    ]
    ip_matches = [
        record
        for record in recent_history
        if _read(record, "case_id") != case.case_id
        and _ip_hash(case)
        and _ip_hash(case) == _ip_hash(record)
        and (record_time := _as_datetime(_read(record, "event_time")))
        and five_minutes_ago <= record_time <= case_time
    ]
    concentration = 1 + max(len(device_matches), len(ip_matches))

    quality_records = [
        record
        for record in matching_records
        if str(_read(record, "status", "")).upper() == "CLOSED"
    ]
    paid_records = [record for record in quality_records if bool(_read(record, "paid", False))]
    conversion_rate = len(paid_records) / len(quality_records) if quality_records else 0.0
    mean_downstream_quality = (
        sum(float(_read(record, "downstream_quality", 1.0)) for record in paid_records)
        / len(paid_records)
        if paid_records
        else 1.0
    )
    quality_mismatch = (
        len(quality_records) >= QUALITY_MISMATCH_MINIMUM_CASES
        and conversion_rate >= 0.80
        and mean_downstream_quality <= 0.10
    )

    reasons: list[str] = []
    anomaly_score = 0.05
    if attempts_last_minute > VELOCITY_WATCH_THRESHOLD:
        reasons.append(f"{attempts_last_minute} attempts from the same identity in one minute")
        anomaly_score = max(
            anomaly_score,
            0.90 if attempts_last_minute > VELOCITY_QUARANTINE_THRESHOLD else 0.65,
        )
    if concentration >= CONCENTRATION_WATCH_THRESHOLD:
        reasons.append(f"{concentration} attempts share a device or IP fingerprint in five minutes")
        anomaly_score = max(
            anomaly_score,
            0.85 if concentration >= CONCENTRATION_QUARANTINE_THRESHOLD else 0.60,
        )
    if quality_mismatch:
        reasons.append(
            f"{conversion_rate:.0%} conversion with {mean_downstream_quality:.2f} downstream quality"
        )
        anomaly_score = 0.98

    if quality_mismatch:
        status = "QUARANTINED"
        signal_type = "reward_farming"
    elif (
        attempts_last_minute > VELOCITY_QUARANTINE_THRESHOLD
        or concentration >= CONCENTRATION_QUARANTINE_THRESHOLD
    ):
        status = "QUARANTINED"
        signal_type = (
            "velocity_spike"
            if attempts_last_minute > VELOCITY_QUARANTINE_THRESHOLD
            else "ip_concentration"
        )
    elif attempts_last_minute > VELOCITY_WATCH_THRESHOLD:
        status = "WATCH"
        signal_type = "velocity_spike"
    elif concentration >= CONCENTRATION_WATCH_THRESHOLD:
        status = "WATCH"
        signal_type = "ip_concentration"
    else:
        status = "TRUSTED"
        signal_type = "pattern_probe"
        reasons.append("No velocity, concentration, or downstream-quality anomaly detected")

    return IntegritySignal(
        case_id=case.case_id,
        signal_type=signal_type,
        anomaly_score=round(min(anomaly_score, 1.0), 4),
        integrity_status=status,
        rationale="; ".join(reasons),
    )


def score_integrity(event: dict[str, Any]) -> IntegrityVerdict:
    """Preserve the simple helper used by the current FastAPI webhook endpoint."""
    amount = float(event.get("amount") or 0.0)
    failure_reason = str(event.get("failure_reason") or "").lower()
    suspicious = any(token in failure_reason for token in ("synthetic", "probe", "farming"))
    anomaly_score = 0.92 if suspicious else min(0.25 + amount / 100_000.0, 0.9)
    if anomaly_score >= 0.85:
        return IntegrityVerdict(
            signal_type="reward_farming",
            anomaly_score=anomaly_score,
            integrity_status="QUARANTINED",
            rationale="High anomaly score indicates possible reward-farming or synthetic activity.",
        )
    if anomaly_score >= 0.55:
        return IntegrityVerdict(
            signal_type="pattern_probe",
            anomaly_score=anomaly_score,
            integrity_status="WATCH",
            rationale="Medium anomaly score suggests this case should be monitored.",
        )
    return IntegrityVerdict(
        signal_type="pattern_probe",
        anomaly_score=anomaly_score,
        integrity_status="TRUSTED",
        rationale="No strong integrity risk detected.",
    )
