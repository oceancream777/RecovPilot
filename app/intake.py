from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_non_empty(*values: Any, default: str = "unknown") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        else:
            return str(value)
    return default


def _extract_event_source(raw_payload: dict, event_source: str | None) -> str:
    source = _first_non_empty(event_source, raw_payload.get("event"), raw_payload.get("event_type"), default="")
    if source:
        return source.lower()

    payload = raw_payload.get("payload") or {}
    if "subscription" in payload:
        return "subscription.halted"
    if "payment" in payload:
        return "payment.failed"
    if any(key in raw_payload for key in ("cart_token", "token", "line_items", "abandoned_checkout_url")):
        return "checkout.abandoned"
    return ""


def _normalize_payment_failed(raw_payload: dict, case_id: str) -> dict:
    payment_entity = (raw_payload.get("payload") or {}).get("payment", {}).get("entity", {})
    amount = _as_float(payment_entity.get("amount"))
    return {
        "case_id": case_id,
        "merchant_id": _first_non_empty(raw_payload.get("account_id"), payment_entity.get("merchant_id")),
        "customer_id": _first_non_empty(payment_entity.get("customer_id"), payment_entity.get("email"), payment_entity.get("contact")),
        "transaction_id": _first_non_empty(payment_entity.get("id"), payment_entity.get("order_id"), default=None),
        "event_type": "payment_failed",
        "amount": amount,
        "failure_reason": _first_non_empty(payment_entity.get("error_description"), payment_entity.get("error_reason"), payment_entity.get("description"), default="payment_failed"),
        "channel_preference": _first_non_empty(payment_entity.get("method"), payment_entity.get("wallet"), payment_entity.get("bank"), default="payment_gateway"),
        "status": "proposed",
        "created_at": _utcnow(),
    }


def _normalize_subscription_halted(raw_payload: dict, case_id: str) -> dict:
    subscription_entity = (raw_payload.get("payload") or {}).get("subscription", {}).get("entity", {})
    payment_entity = (raw_payload.get("payload") or {}).get("payment", {}).get("entity", {})
    plan = subscription_entity.get("plan") if isinstance(subscription_entity.get("plan"), dict) else {}
    amount = _as_float(plan.get("amount") or subscription_entity.get("plan_amount") or payment_entity.get("amount"))
    return {
        "case_id": case_id,
        "merchant_id": _first_non_empty(raw_payload.get("account_id"), subscription_entity.get("merchant_id")),
        "customer_id": _first_non_empty(subscription_entity.get("customer_id"), payment_entity.get("email")),
        "transaction_id": _first_non_empty(subscription_entity.get("id"), default=None),
        "event_type": "subscription_halted",
        "amount": amount,
        "failure_reason": _first_non_empty(payment_entity.get("error_description"), payment_entity.get("error_reason"), subscription_entity.get("status"), default="subscription_halted"),
        "channel_preference": _first_non_empty(payment_entity.get("method"), default="subscription_recurring"),
        "status": "proposed",
        "created_at": _utcnow(),
    }


def _normalize_checkout_abandoned(raw_payload: dict, case_id: str) -> dict:
    checkout_payload = raw_payload.get("payload") or raw_payload
    line_items = checkout_payload.get("line_items") or []
    amount = _as_float(checkout_payload.get("cart_amount") or checkout_payload.get("line_items_total"))
    if not amount and line_items:
        amount = sum(
            _as_float(item.get("price")) * max(int(item.get("quantity") or 1), 1)
            for item in line_items
            if isinstance(item, dict)
        )
    customer = checkout_payload.get("customer") or {}
    drop_off_step = _first_non_empty(
        checkout_payload.get("drop_off_step"),
        checkout_payload.get("checkout_step"),
        checkout_payload.get("stage"),
        default="checkout_abandoned",
    )
    return {
        "case_id": case_id,
        "merchant_id": _first_non_empty(checkout_payload.get("shop_id"), raw_payload.get("account_id")),
        "customer_id": _first_non_empty(
            checkout_payload.get("email"),
            checkout_payload.get("phone"),
            customer.get("email"),
            customer.get("phone"),
            checkout_payload.get("cart_token"),
            checkout_payload.get("token"),
        ),
        "transaction_id": _first_non_empty(checkout_payload.get("cart_token"), checkout_payload.get("token"), default=None),
        "event_type": "checkout_abandoned",
        "amount": amount,
        "failure_reason": drop_off_step,
        "channel_preference": _first_non_empty(checkout_payload.get("platform"), default="checkout"),
        "status": "proposed",
        "created_at": _utcnow(),
    }


def normalize_recovery_event(raw_payload: dict, event_source: str) -> dict:
    case_id = str(uuid4())
    source = _extract_event_source(raw_payload, event_source)

    if source == "payment.failed" or source.endswith("payment.failed"):
        return _normalize_payment_failed(raw_payload, case_id)
    if source == "subscription.halted" or source.endswith("subscription.halted"):
        return _normalize_subscription_halted(raw_payload, case_id)
    if source == "checkout.abandoned" or source.endswith("checkout.abandoned"):
        return _normalize_checkout_abandoned(raw_payload, case_id)

    payload = raw_payload.get("payload") or {}
    if "subscription" in payload:
        return _normalize_subscription_halted(raw_payload, case_id)
    if "payment" in payload:
        return _normalize_payment_failed(raw_payload, case_id)
    if any(key in raw_payload for key in ("cart_token", "token", "line_items", "abandoned_checkout_url")):
        return _normalize_checkout_abandoned(raw_payload, case_id)

    raise ValueError(f"Unsupported event source: {event_source!r}")

