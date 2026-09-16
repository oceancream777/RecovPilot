import hmac
import hashlib
import json
import requests
import os
from dotenv import load_dotenv

# 1. Load the exact same .env file the server is using
load_dotenv()
SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not SECRET:
    print(" ERROR: Could not read RAZORPAY_WEBHOOK_SECRET from .env")
    exit(1)

print(f"Loaded Secret: {SECRET[:3]}...{SECRET[-3:]}")

URL = "http://127.0.0.1:8000/api/v1/webhooks/razorpay" # Update this path if your route is different

# 2. The Razorpay Payload
payload = {
  "event": "payment.failed",
  "payload": {
    "payment": {
      "entity": {
        "id": "pay_test123",
        "amount": 500000,
        "currency": "INR",
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "customer",
        "error_reason": "insufficient_funds"
      }
    }
  }
}

# 3. Create the exact raw byte string (no extra spaces)
# Razorpay signatures are extremely strict about whitespace
payload_body = json.dumps(payload, separators=(',', ':'))

# 4. Generate the HMAC SHA256 signature
signature = hmac.new(
    SECRET.encode('utf-8'),
    payload_body.encode('utf-8'),
    hashlib.sha256
).hexdigest()

headers = {
    'Content-Type': 'application/json',
    'X-Razorpay-Signature': signature
}

print(f"Firing Webhook with Signature: {signature}")
response = requests.post(URL, data=payload_body, headers=headers)

print(f"Response Status: {response.status_code}")
print(f"Response Body: {response.text}")