from __future__ import annotations

import logging
from pathlib import Path
import sys

import pytest
from razorpay.errors import BadRequestError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import razorpay_client


class _RejectingPaymentLinks:
    def create(self, _payload: dict) -> dict:
        raise BadRequestError("Invalid request for rzp_test_visible_token")


class _RejectingClient:
    payment_link = _RejectingPaymentLinks()


def test_payment_link_error_preserves_safe_provider_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_visible_token")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret-never-log-this")
    monkeypatch.setattr(razorpay_client, "client", _RejectingClient())

    with caplog.at_level(logging.ERROR), pytest.raises(
        razorpay_client.RazorpayPaymentLinkError
    ) as raised:
        razorpay_client.create_recovery_payment_link(
            amount=8000.0,
            case_id="case-provider-rejection",
            action_type="incentive_link",
            discount_percentage=10.0,
        )

    error = raised.value
    assert error.provider_error_type == "BadRequestError"
    assert error.provider_reason == "Invalid request for [redacted]"
    assert "BadRequestError" in str(error)
    assert "[redacted]" in caplog.text
    assert "rzp_test_visible_token" not in caplog.text
    assert "secret-never-log-this" not in caplog.text


def test_payment_link_requires_a_short_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyPaymentLinks:
        def create(self, _payload: dict) -> dict:
            return {"id": "plink_without_url"}

    class EmptyClient:
        payment_link = EmptyPaymentLinks()

    monkeypatch.setattr(razorpay_client, "client", EmptyClient())

    with pytest.raises(razorpay_client.RazorpayPaymentLinkError) as raised:
        razorpay_client.create_recovery_payment_link(
            amount=8000.0,
            case_id="case-invalid-response",
            action_type="payment_link",
        )

    assert raised.value.provider_error_type == "InvalidResponse"
    assert "short_url" in raised.value.provider_reason


def test_test_mode_limit_is_classified_as_a_quota_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class QuotaPaymentLinks:
        def create(self, _payload: dict) -> dict:
            raise BadRequestError("test mode limit of 30 reached for payment_link")

    class QuotaClient:
        payment_link = QuotaPaymentLinks()

    monkeypatch.setattr(razorpay_client, "client", QuotaClient())

    with pytest.raises(razorpay_client.RazorpayPaymentLinkQuotaError):
        razorpay_client.create_recovery_payment_link(
            amount=8000.0,
            case_id="case-test-quota",
            action_type="incentive_link",
            discount_percentage=10.0,
        )


def test_duplicate_reference_reconciles_the_existing_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DuplicatePaymentLinks:
        def create(self, _payload: dict) -> dict:
            raise BadRequestError(
                "payment link creation with reference ID already attempted"
            )

        def all(self, data: dict) -> dict:
            assert data == {"reference_id": "case-reconcile"}
            return {
                "payment_links": [
                    {
                        "reference_id": "case-reconcile",
                        "amount": 720000,
                        "short_url": "https://rzp.io/i/reconciled",
                    }
                ]
            }

    class DuplicateClient:
        payment_link = DuplicatePaymentLinks()

    monkeypatch.setattr(razorpay_client, "client", DuplicateClient())

    result = razorpay_client.create_recovery_payment_link(
        amount=8000.0,
        case_id="case-reconcile",
        action_type="incentive_link",
        discount_percentage=10.0,
    )

    assert result == {
        "short_url": "https://rzp.io/i/reconciled",
        "final_amount": 7200.0,
    }
