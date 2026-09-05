# RecovPilot Setup and Demo Guide

This guide takes a fresh clone from local installation through the five-scenario dashboard, Razorpay Test Mode, and the n8n Champion/Challenger orchestration flow.

The causal decision API and synthetic scenarios run entirely on a local machine. Razorpay, ngrok, and n8n are required only for the external integration demonstration.

## Table of Contents

- [System Prerequisites](#system-prerequisites)
- [Clone the Repository](#clone-the-repository)
- [Backend Initialization](#backend-initialization)
- [Frontend Initialization](#frontend-initialization)
- [Local Demo](#local-demo)
- [Razorpay Test Mode](#razorpay-test-mode)
- [Ngrok Tunnel](#ngrok-tunnel)
- [n8n Batch Orchestration](#n8n-batch-orchestration)
- [End-to-End Demo Sequence](#end-to-end-demo-sequence)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

## System Prerequisites

| Tool | Version / Requirement | Purpose |
| --- | --- | --- |
| **Python** | 3.10+; 3.11 recommended | FastAPI, causal ML, SQLAlchemy, and SQLite state |
| **Node.js** | 20.19+ or 22.12+ | Required by the repository's Vite 8 frontend |
| **npm** | Bundled with a supported Node.js release | Frontend dependency and build commands |
| **Git** | Current stable release | Clone and version-control workflow |
| **Browser** | Current Chrome, Edge, Firefox, or Safari | React dashboard and FastAPI documentation |
| **Razorpay** | Test Mode account; optional | Signed webhook and Payment Link demonstration |
| **ngrok** | Free account and CLI; optional | Public HTTPS tunnel to local FastAPI |
| **n8n** | Cloud account; optional | Asynchronous batch-policy orchestration |

No LLM provider, model API key, Docker runtime, or hosted database is required.

Confirm the local runtimes before continuing:

```bash
python3 --version
node --version
npm --version
git --version
```

If Node reports version 18, upgrade it before running `npm install`; Vite 8 does not support Node 18.

## Clone the Repository

Replace the placeholders with the repository URL used after publication:

```bash
git clone https://github.com/<github-username>/<repository-name>.git
cd <repository-name>
```

All backend commands in this guide must run from the repository root. Running Uvicorn from another directory can cause `ModuleNotFoundError: No module named 'app'`.

## Backend Initialization

### 1. Create an isolated Python environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

The active terminal prompt normally displays `(.venv)`.

### 2. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, SQLAlchemy, Pydantic v2, the Razorpay SDK, Pandas, NumPy, Joblib, and Scikit-Learn.

### 3. Create the local environment file

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The default file is safe for local decision and simulation testing. Razorpay link execution requires real **test-mode** values for the first three variables:

```dotenv
RAZORPAY_KEY_ID=rzp_test_replace_me
RAZORPAY_KEY_SECRET=replace_me
RAZORPAY_WEBHOOK_SECRET=replace_with_dashboard_webhook_secret
RAZORPAY_WEBHOOK_MERCHANT_BUDGET=0
RAZORPAY_WEBHOOK_MAX_INCENTIVE=0
RAZORPAY_WEBHOOK_RECOVERY_WINDOW_HOURS=48
RAZORPAY_WEBHOOK_OFFER_LADDER=0,3,5,8,10
RAZORPAY_WEBHOOK_ALLOWED_ACTIONS=retry,payment_link,no_action
RAZORPAY_WEBHOOK_POLICY_VERSION=v3.5-live-test
```

`RAZORPAY_WEBHOOK_MAX_INCENTIVE` is a percentage. Its safe default of `0` disables incentive tiers for live webhook traffic until you configure a policy explicitly. Simulator requests use the editable percentage sent by the React dashboard instead.

Razorpay Test Mode allows up to 30 Payment Links per business. If the limit is reached, `/api/v1/execute/approve_link` returns `429` and keeps the intervention pending. Request a test-limit increase from Razorpay Support or switch to another test business, restart the API so it reloads the credentials, and retry the same approval.

Do not add quotes around comma-separated values unless your shell or deployment environment requires them.

### 4. Seed SQLite before the first server start

```bash
python scripts/simulate_environment.py --reset
```

Expected summary:

- 5,000 clean randomized cases
- 500 labeled incentive-farming cases
- Treatment and control assignments
- Persisted recovery cases, interventions, outcomes, and integrity signals

The command creates `recovery_agent.db` in the repository root. Seeding before startup allows the lifespan hook to train a baseline artifact when `app/artifacts/causal_models.joblib` is absent.

`--reset` replaces simulator-owned records while preserving API-created cases. It is safe to rerun when a reproducible synthetic dataset is needed.

`recovery_agent.db` and SQLite sidecar files such as `recovery_agent.db-wal` and `recovery_agent.db-shm` are local runtime state. They are ignored by Git and should not be uploaded to GitHub; every evaluator should generate a fresh database with the seed command above.

### 5. Start FastAPI

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal running. The startup hook:

- Creates missing SQLite tables.
- Restores the active policy version from SQLite.
- Loads an existing Joblib model artifact.
- Trains and saves a baseline artifact if eligible records exist and the artifact is absent.

### 6. Verify backend health

Open <http://127.0.0.1:8000/docs> or run:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

The response reports API, database, and model state. If `model_available` is false, see [Model unavailable](#model-unavailable).

## Frontend Initialization

Open a second terminal and keep FastAPI running in the first.

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> in a browser.

The frontend targets `http://127.0.0.1:8000` by default. To use another backend origin:

macOS or Linux:

```bash
VITE_API_BASE_URL=https://your-backend.example npm run dev
```

Windows PowerShell:

```powershell
$env:VITE_API_BASE_URL="https://your-backend.example"
npm run dev
```

The frontend and backend may use different local ports. FastAPI's configured CORS policy permits the local Vite origin and tunnel-based demonstrations.

## Local Demo

External accounts are not required for decision evaluation, scenario warmup, causal metrics, or the paired comparison.

### Five available scenarios

1. **The Discount Wins Trap**: Demonstrates why raw conversion can destroy margin.
2. **The Recovery Casino**: Injects incentive farming and tests quarantine.
3. **The Learner That Admits It Was Wrong**: Demonstrates canary failure and rollback.
4. **Recovery Agent OFF vs ON**: Compares both policies on the same immutable event stream.
5. **Naive ML vs Recovery Learning Agent**: Contrasts propensity scoring with causal net recovery.

### Execute a paired benchmark

1. Select a scenario from the dropdown.
2. Wait until the scenario card reports `READY` or `CACHED`.
3. Review or edit `Amount`, `Case age`, `Merchant budget`, and `Max incentive`. `Max incentive` is a percentage of the case amount, accepts values from 0 up to but not including 100, and is not fixed at 10%.
4. Select the permitted actions. Unchecked actions remain vetoed after model scoring.
5. Leave **ON-AUTO execution** disabled for the default human-approval flow.
6. Click **Execute Paired Benchmark**.
7. Inspect the OFF and ON results, causal probabilities, net recovery, integrity status, discount cost, policy version, and reason codes.

Paired execution sends identical inputs to both branches. Scenario 5 intentionally uses a naive propensity policy for the OFF branch; the remaining scenarios use the incumbent recovery baseline defined by the repository.

### Execution behavior

- `no_action` and `suppress` never create a Razorpay link.
- The UI builds the incentive ladder from 0 through the whole-number portion of the submitted `Max incentive` percentage. The learner selects the highest-value eligible tier within that range; it does not automatically select the ceiling.
- Low-risk `payment_link` or `incentive_link` actions can execute immediately only when ON-AUTO is enabled and Razorpay test credentials are configured.
- An incentive above 25% of the incoming `merchant_budget` returns `pending_human_review`. An incentive exactly equal to 25% does not trip this circuit breaker.
- Example: amount INR 8,000, merchant budget INR 2,000, and `Max incentive = 10%` permit at most an INR 800 discount. If the learner selects that tier, INR 800 exceeds the INR 500 ON-AUTO threshold and therefore requires **Approve & Execute Offer**.
- Repeated or suspicious IP/device activity returns `pending_human_review` when the integrity verdict is WATCH; QUARANTINED activity is suppressed and cannot be approved.
- Cart amounts above INR 10,000 and other WATCH verdicts also require human review.
- **Approve & Execute Offer** calls `/api/v1/execute/approve_link`; only that approval call creates the deferred link.
- Without Razorpay credentials, causal decisions still run, but an attempted link side effect returns a configuration error.

## Razorpay Test Mode

Skip this section if only the local simulator and benchmark are required.

### 1. Add test credentials

In the Razorpay dashboard, enable **Test Mode** and copy the test Key ID and Key Secret into `.env`:

```dotenv
RAZORPAY_KEY_ID=rzp_test_your_key_id
RAZORPAY_KEY_SECRET=your_test_key_secret
```

Restart FastAPI after changing `.env`.

### 2. Configure the webhook secret

Choose a webhook secret in the Razorpay Test Dashboard and set the same value locally:

```dotenv
RAZORPAY_WEBHOOK_SECRET=your_matching_webhook_secret
```

The signed Razorpay listener is:

```text
/api/v1/webhooks/razorpay
```

Do not use `/api/v1/intake/webhook` as the Razorpay dashboard destination. That route is the generic event-normalization endpoint and is not the signed Razorpay listener.

The live listener currently handles `payment.failed`. Other Razorpay event types receive a successful ignored response rather than entering the recovery decision path.

## Ngrok Tunnel

Skip this section when all requests originate on the local machine.

### 1. Install and authenticate ngrok

Follow the ngrok account instructions, then add the account token once:

```bash
ngrok config add-authtoken <your-ngrok-authtoken>
```

### 2. Expose FastAPI

Open a third terminal while FastAPI is running:

```bash
ngrok http 8000
```

Copy the generated HTTPS forwarding origin, for example:

```text
https://example-subdomain.ngrok-free.app
```

Free ngrok domains may change when the tunnel restarts. Update Razorpay and n8n whenever the forwarding URL changes.

### 3. Configure the Razorpay destination

In the Razorpay Test Dashboard, subscribe to `payment.failed` and use:

```text
https://<your-ngrok-domain>/api/v1/webhooks/razorpay
```

The complete URL must contain exactly one `/` between the ngrok domain and `api`.

### 4. Confirm tunnel health

```bash
curl https://<your-ngrok-domain>/api/v1/health
```

## n8n Batch Orchestration

n8n is independent of the React frontend. It sends a scheduled or manual command through ngrok to FastAPI, then evaluates the persisted Champion/Challenger response.

### Four-node workflow

Create this DAG:

```text
Manual Trigger -> HTTP Request -> IF Condition -> Output Log
```

Connect both IF outputs to the Output Log node if the same node should record promotion and rollback responses.

### Node 1: Manual Trigger

Add **Manual Trigger**. After validating the workflow, it can be replaced or supplemented with a Schedule Trigger for the midnight batch.

### Node 2: HTTP Request

Configure **HTTP Request** as follows:

- **Method:** `POST`
- **URL:** `https://<your-ngrok-domain>/api/v1/admin/trigger_batch_learning`
- **Send Headers:** enabled
- **Header `Content-Type`:** `application/json`
- **Header `Idempotency-Key`:** `{{$execution.id}}`
- **Send Body:** enabled
- **Body Content Type:** JSON

Standard promotion body:

```json
{
  "trigger_source": "n8n_orchestrator",
  "force_scenario": "normal"
}
```

The expected standard response includes:

```json
{
  "status": "success",
  "champion_model": "v3.5",
  "challenger_model": "v3.6",
  "challenger_roi": 5.82,
  "incumbent_roi": 5.16,
  "quarantine_rate": 1.0,
  "promoted": true,
  "active_version": "v3.6",
  "run_status": "PROMOTED"
}
```

FastAPI returns additional audit fields, including `run_id` and `timestamp`.

### Node 3: IF Condition

Configure a Boolean condition:

```text
Value 1: {{$json.promoted}}
Operation: is true
```

For a stricter promotion check, add an AND condition:

```text
{{$json.run_status}} equals PROMOTED
```

The true branch represents promotion. The false branch represents rollback or canary rejection.

### Node 4: Output Log

Add an **Edit Fields (Set)** node and rename it `Output Log`. Preserve or map these fields:

- `run_id`
- `status`
- `run_status`
- `champion_model`
- `challenger_model`
- `incumbent_roi`
- `challenger_roi`
- `quarantine_rate`
- `promoted`
- `active_version`
- `rejection_reason`
- `timestamp`

This node can later be replaced with Slack, email, or another approved notification destination.

### Test rollback behavior

Change only the request body:

```json
{
  "trigger_source": "n8n_orchestrator",
  "force_scenario": "policy_rollback"
}
```

The response returns `promoted: false`, `run_status: REJECTED_ROLLBACK`, and keeps `active_version: v3.5`.

### Idempotency behavior

`Idempotency-Key` prevents n8n retries from creating duplicate `batch_policy_runs` entries. Reusing a key returns the stored response for that run. Use a new key for each intentional batch evaluation; `{{$execution.id}}` provides one key per n8n workflow execution.

## End-to-End Demo Sequence

Follow this order for the complete buildathon demonstration:

```text
Clone repository
-> Create Python environment
-> Install backend dependencies
-> Copy .env.example to .env
-> Seed 5,500 SQLite cases
-> Start FastAPI on port 8000
-> Start React/Vite on port 5173
-> Select one of five scenarios
-> Execute Paired Benchmark
-> Inspect v3.5 baseline versus causal decision evidence
-> Start ngrok for external callbacks
-> Run the n8n four-node workflow
-> FastAPI evaluates and persists Champion versus Challenger
-> n8n verifies the promotion response
-> React reruns the same scenario
-> Execution response reports the active v3.6 policy
```

For the live Razorpay Test Mode demonstration:

```text
Configure matching Razorpay webhook secret
-> Point Razorpay to /api/v1/webhooks/razorpay through ngrok
-> Trigger a test payment.failed event
-> FastAPI validates the signature and normalizes paise to rupees
-> Integrity, causal scoring, and deterministic guardrails run
-> Low-risk auto mode executes or high-risk mode waits for HITL approval
-> Razorpay test Payment Link is created only at the permitted execution state
```

## Verification

Run backend tests from the repository root:

```bash
python -m pytest tests/
```

Run the static-versus-naive-versus-causal benchmark after seeding:

```bash
python scripts/run_benchmark.py
```

Run frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

Inspect the active persisted policy:

```bash
curl http://127.0.0.1:8000/api/v1/admin/policy_state
```

Trigger the batch endpoint without n8n:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/trigger_batch_learning \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: local-setup-check-001' \
  -d '{"trigger_source":"manual_ui","force_scenario":"normal"}'
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run Uvicorn from the repository root:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Model unavailable

Stop FastAPI, seed the database, and restart:

```bash
python scripts/simulate_environment.py --reset
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The lifespan bootstrap runs at server startup, not after the seed script modifies a running server's database.

### Frontend rejects the Node version

Install Node `20.19+` or `22.12+`, delete the incomplete local dependency installation if necessary, then reinstall:

```bash
cd frontend
rm -rf node_modules
npm install
```

Do not run the removal command outside the `frontend` directory.

### Port already in use

Start FastAPI on a different port and pass the matching origin to Vite:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8001 npm run dev
```

### Razorpay link creation fails

Confirm that:

- Both credentials are from Razorpay Test Mode.
- `.env` is in the repository root.
- FastAPI was restarted after editing `.env`.
- The approved action is `payment_link` or `incentive_link`.
- The final amount is at least INR 1.00.
- The machine has outbound network access.

### Razorpay webhook is ignored

Confirm that the event is `payment.failed`. Other event types are acknowledged but intentionally ignored by the signed live listener.

### Razorpay webhook signature fails

Confirm that the Razorpay dashboard secret and `RAZORPAY_WEBHOOK_SECRET` are identical. Do not use the API Key Secret as the webhook secret unless that is explicitly how the test webhook was configured.

### n8n receives HTTP 422

Use an accepted request body:

- `trigger_source`: `manual_ui`, `n8n_orchestrator`, or `cron_scheduler`
- `force_scenario`: `normal`, `policy_rollback`, or `drift_detected`

### n8n appears to repeat an old result

The same `Idempotency-Key` returns the stored result by design. Start a new workflow execution or provide a new unique key for an intentional new batch run.

### Active policy remains v3.5

Inspect the batch response. `policy_rollback` and `drift_detected` deliberately reject the Challenger. A standard body with `force_scenario: normal` returns the deterministic promotion demonstration and activates v3.6.

## Security Notes

- Use Razorpay Test Mode only for this repository demonstration.
- Never commit `.env`, `rzp_api.env`, API secrets, webhook secrets, or ngrok authentication tokens.
- Treat an ngrok URL as a public internet endpoint while the tunnel is active.
- The batch endpoint uses idempotency for retry safety, not caller authentication. Keep the demo tunnel short-lived. A production deployment requires authenticated service-to-service requests, secret rotation, rate limiting, and network policy.
- SQLite is suitable for local state and audit demonstration. Multi-instance production serving requires a transactional shared database and managed migrations.
- The implemented payment guardrails are project controls, not a substitute for legal, security, RBI, NPCI, acquiring-bank, or Razorpay production review.
