"""Small, testable wrapper around Razorpay Payment Links."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import razorpay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
client: Any | None = None
logger = logging.getLogger(__name__)


class RazorpayConfigurationError(RuntimeError):
    """Raised when the SDK cannot be configured safely."""


class RazorpayPaymentLinkError(RuntimeError):
    """Raised when Razorpay does not return a usable Payment Link."""

    def __init__(self, provider_error_type: str, provider_reason: str) -> None:
        self.provider_error_type = provider_error_type
        self.provider_reason = provider_reason
        super().__init__(
            f"Razorpay payment-link error ({provider_error_type}): {provider_reason}"
        )


class RazorpayPaymentLinkQuotaError(RazorpayPaymentLinkError):
    """Raised when the Razorpay business has exhausted its Payment Link quota."""


def _credentials() -> tuple[str | None, str | None]:
    """Load standard or legacy local env files only when an API call is made."""
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "rzp_api.env"):
        if key_id and key_secret:
            break
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            key_id = os.getenv("RAZORPAY_KEY_ID")
            key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    return key_id, key_secret


def _get_client() -> Any:
    global client

    if client is not None:
        return client

    key_id, key_secret = _credentials()
    if not key_id or not key_secret:
        raise RazorpayConfigurationError(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set in .env or rzp_api.env."
        )

    client = razorpay.Client(auth=(key_id, key_secret))
    return client


def _money(value: float) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("amount must be a valid number") from exc

    if not amount.is_finite() or amount <= 0:
        raise ValueError("amount must be greater than zero")
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _safe_provider_reason(exc: Exception) -> str:
    reason = str(exc).strip() or "No error description was returned by Razorpay."
    key_id, key_secret = _credentials()
    for credential in (key_id, key_secret):
        if credential:
            reason = reason.replace(credential, "[redacted]")
    reason = re.sub(r"rzp_(?:test|live)_[A-Za-z0-9]+", "[redacted]", reason)
    return reason[:500]


def _recover_existing_link(reference_id: str, amount_in_paise: int) -> Mapping | None:
    response = _get_client().payment_link.all({"reference_id": reference_id})
    links = response.get("payment_links", []) if isinstance(response, Mapping) else []
    for link in links:
        if link.get("reference_id") != reference_id:
            continue
        if int(link.get("amount", -1)) != amount_in_paise:
            raise RazorpayPaymentLinkError(
                "ReferenceConflict",
                "An existing Razorpay Payment Link uses this case ID with a different amount.",
            )
        if link.get("short_url"):
            return link
    return None


def create_recovery_payment_link(
    amount: float,
    case_id: str,
    action_type: str,
    discount_percentage: float = 0.0,
) -> dict:
    """Create a standard INR Payment Link and return rupee-denominated billing data."""
    reference_id = str(case_id).strip()
    if not reference_id:
        raise ValueError("case_id must not be empty")
    if len(reference_id) > 40:
        raise ValueError("case_id must be at most 40 characters for a Razorpay reference_id")

    try:
        discount = Decimal(str(discount_percentage))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("discount_percentage must be a valid number") from exc
    if not discount.is_finite() or discount < 0 or discount >= 100:
        raise ValueError("discount_percentage must be between 0 and 99.99")
    if action_type != "incentive_link" and discount > 0:
        raise ValueError("discount_percentage requires an incentive_link action")

    final_amount = _money(amount)
    discounted = discount > 0
    if discounted:
        multiplier = Decimal("1") - (discount / Decimal("100"))
        final_amount = (final_amount * multiplier).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    amount_in_paise = int(
        (final_amount * Decimal("100")).to_integral_value(rounding=ROUND_HALF_UP)
    )
    if amount_in_paise < 100:
        raise ValueError("final billed amount must be at least INR 1.00")

    request_payload = {
        "amount": amount_in_paise,
        "currency": "INR",
        "reference_id": reference_id,
        "description": (
            "Discounted Payment Recovery" if discounted else "Payment Recovery"
        ),
    }

    try:
        response = _get_client().payment_link.create(request_payload)
    except RazorpayConfigurationError:
        raise
    except Exception as exc:
        provider_error_type = type(exc).__name__
        provider_reason = _safe_provider_reason(exc)
        normalized_reason = provider_reason.lower()
        duplicate_reference = "reference id" in normalized_reason and any(
            marker in normalized_reason
            for marker in ("already attempted", "already exists", "has already been used")
        )
        if duplicate_reference:
            existing_link = _recover_existing_link(reference_id, amount_in_paise)
            if existing_link is not None:
                logger.warning(
                    "Reconciled existing Razorpay payment link after duplicate reference: "
                    "reference_id=%s action=%s amount_paise=%s",
                    reference_id,
                    action_type,
                    amount_in_paise,
                )
                return {
                    "short_url": str(existing_link["short_url"]),
                    "final_amount": float(final_amount),
                }
        logger.error(
            "Razorpay payment-link creation failed: type=%s reference_id=%s "
            "action=%s amount_paise=%s reason=%s",
            provider_error_type,
            reference_id,
            action_type,
            amount_in_paise,
            provider_reason,
        )
        error_class = (
            RazorpayPaymentLinkQuotaError
            if "test mode limit of 30 reached for payment_link"
            in provider_reason.lower()
            else RazorpayPaymentLinkError
        )
        raise error_class(provider_error_type, provider_reason) from exc

    if not isinstance(response, Mapping) or not response.get("short_url"):
        raise RazorpayPaymentLinkError(
            "InvalidResponse",
            "Razorpay returned a payment-link response without a short_url.",
        )

    return {
        "short_url": str(response["short_url"]),
        "final_amount": float(final_amount),
    }
