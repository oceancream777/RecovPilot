import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  ActionList,
  ActionListItem,
  ActivityIcon,
  Alert,
  Badge,
  BladeProvider,
  Button,
  Card,
  CardBody,
  Checkbox,
  CheckCircleIcon,
  ClockIcon,
  Dropdown,
  DropdownOverlay,
  Heading,
  InfoIcon,
  RefreshIcon,
  SelectInput,
  SendIcon,
  ShieldIcon,
  SparklesIcon,
  Switch,
  Text,
  TextInput,
  ToastContainer,
  TrendingUpIcon,
  useToast,
  ZapIcon,
} from '@razorpay/blade/components'
import { bladeTheme } from '@razorpay/blade/tokens'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type ActiveTab = 'triage' | 'delta' | 'explain'
type IntegrityStatus = 'TRUSTED' | 'WATCH' | 'QUARANTINED'
type ScenarioId =
  | 'discount_wins_trap'
  | 'recovery_casino'
  | 'policy_rollback'
  | 'off_vs_on_casino'
  | 'predictive_vs_causal'

type ScenarioPreset = {
  amount: number
  case_age_hours: number
  merchant_budget: number
  max_incentive: number
  allowed_actions: string[]
  customer_segment: 'high_intent_repeat' | 'price_sensitive' | 'subscription_churn' | 'low_intent'
  failure_class: 'issuer_down' | 'insufficient_funds' | 'network_timeout' | 'user_cancelled'
  payment_method: string
}

type ScenarioCatalogItem = {
  scenario_id: ScenarioId
  label: string
  short_label: string
  description: string
  primary_proof: string
  seed: number
  event_count: number
  attack_count: number
  preset: ScenarioPreset
  prerequisites: string[]
}

type WarmupResponse = ScenarioCatalogItem & {
  status: 'ready'
  cache_hit: boolean
  warmup_ms: number
}

type Confidence = {
  lower_bound: number
  upper_bound: number
  confidence_level: number
}

type MLMetrics = {
  recommended_action: string
  recommended_tier: number
  discount_cost: number
  net_expected_recovery: number
  expected_incremental_revenue: number
  baseline_pay_probability: number | null
  predicted_pay_probability: number | null
  causal_lift: number | null
  confidence: Confidence
  integrity_status: IntegrityStatus
  policy_decision: string
  reason_codes: string[]
  policy_version: string
  model_available: boolean
}

type BatchLearningResponse = {
  run_id: string
  status: 'success' | 'rollback' | 'canary_failed'
  champion_model: string
  challenger_model: string
  challenger_roi: number
  incumbent_roi: number
  quarantine_rate: number
  promoted: boolean
  active_version: string
  run_status: 'PROMOTED' | 'REJECTED_ROLLBACK' | 'CANARY_FAILED'
  rejection_reason: string | null
  timestamp: string
}

type PolicyStateResponse = {
  active_version: string
  latest_run: BatchLearningResponse | null
}

type ExecutionTraceStep = {
  stage: string
  action: string
  reason_code: string
  description: string
}

type RecoveryMetrics = {
  metric_source: 'research_calibrated_simulation'
  simulation_disclaimer: string
  baseline_probability: number
  predicted_probability: number
  downstream_quality: number
  quality_adjusted_probability: number
  incremental_lift_pp: number
  gross_expected_recovery: number
  intervention_cost: number
  discount_cost: number
  net_expected_recovery: number
  incremental_recovered_revenue: number
  recovery_roi: number | null
  integrity_status: IntegrityStatus
  attacks_quarantined: number
  action_trace: ExecutionTraceStep[]
  reason_codes: string[]
  incumbent_capability_coverage: number
}

type RevenuePoint = {
  round: number
  baseline: number
  agent: number
  naive_ml: number
}

type PolicyTraceStep = {
  version: string
  status: 'INCUMBENT' | 'CANDIDATE' | 'ROLLED_BACK'
  reason: string
}

type ScenarioMetrics = {
  scenario_id: ScenarioId
  scenario_label: string
  seed: number
  event_count: number
  attack_count: number
  amount_input: number
  integrity_status: IntegrityStatus
  baseline_probability: number | null
  predicted_probability: number | null
  causal_lift: number | null
  incremental_recovered_revenue: number
  cumulative_net_recovery: number
  recovery_roi: number
  attacks_quarantined: number
  cumulative_revenue: RevenuePoint[]
  policy_status: 'INCUMBENT' | 'CANDIDATE' | 'ROLLED_BACK'
  policy_version: string
  policy_trace: PolicyTraceStep[]
  reason_codes: string[]
  comparison_ready: boolean
}

type DemoResponse = {
  case_id: string
  status: 'auto_executed' | 'pending_human_review' | 'executed' | 'not_executed'
  short_url: string | null
  final_amount: number
  final_action: string
  agent_enabled: boolean
  ml_metrics: MLMetrics | null
  input_amount: number
  discount_cost: number
  net_expected_recovery: number
  expected_incremental_revenue: number
  reason_codes: string[]
  scenario_metrics: ScenarioMetrics | null
  recovery_metrics: RecoveryMetrics | null
  current_policy_version: string
}

type ApproveLinkResponse = {
  case_id: string
  short_url: string
  final_amount: number
  final_action: 'payment_link' | 'incentive_link'
  status: 'executed'
  current_policy_version: string
}

type PairedRuns = { signature: string; off?: DemoResponse; on?: DemoResponse }

const API_ORIGIN = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const EXECUTE_URL = import.meta.env.VITE_API_URL ?? `${API_ORIGIN}/api/v1/execute/demo_toggle`
const APPROVE_LINK_URL = `${API_ORIGIN}/api/v1/execute/approve_link`
const SCENARIO_URL = `${API_ORIGIN}/api/v1/demo/scenarios`
const HEALTH_URL = `${API_ORIGIN}/api/v1/health`
const BATCH_LEARNING_URL = `${API_ORIGIN}/api/v1/admin/trigger_batch_learning`
const POLICY_STATE_URL = `${API_ORIGIN}/api/v1/admin/policy_state`

const ACTION_OPTIONS = [
  { value: 'retry', label: 'Retry' },
  { value: 'payment_link', label: 'Payment link' },
  { value: 'incentive_link', label: 'Incentive link' },
  { value: 'no_action', label: 'No action' },
]
const baseActions = ACTION_OPTIONS.map(({ value }) => value)

const FALLBACK_SCENARIOS: ScenarioCatalogItem[] = [
  { scenario_id: 'discount_wins_trap', label: 'The Discount Wins Trap', short_label: 'Discount Wins Trap', description: 'A discount converts best, but the cheaper payment link creates more incremental net recovery.', primary_proof: 'Causal net value beats raw conversion.', seed: 35101, event_count: 10000, attack_count: 0, preset: { amount: 4999, case_age_hours: 50, merchant_budget: 250000, max_incentive: 10, allowed_actions: baseActions, customer_segment: 'high_intent_repeat', failure_class: 'user_cancelled', payment_method: 'card' }, prerequisites: ['10,000 randomized observations', 'Hidden response curves', 'Explicit intervention costs'] },
  { scenario_id: 'recovery_casino', label: 'The Recovery Casino', short_label: 'Recovery Casino', description: 'A concentrated incentive-farming cohort manufactures conversions with zero downstream quality.', primary_proof: 'The integrity gate protects the learning signal.', seed: 35202, event_count: 10000, attack_count: 500, preset: { amount: 1999, case_age_hours: 50, merchant_budget: 150000, max_incentive: 10, allowed_actions: baseActions, customer_segment: 'price_sensitive', failure_class: 'user_cancelled', payment_method: 'card' }, prerequisites: ['500 reward-poisoning cases', 'Concentrated fingerprints', 'Zero downstream quality'] },
  { scenario_id: 'policy_rollback', label: 'The Learner That Admits It Was Wrong', short_label: 'Drift and Rollback', description: 'A candidate degrades in canary and the last-known-good recovery policy is restored.', primary_proof: 'Promotion is evidence-bound and reversible.', seed: 35303, event_count: 10000, attack_count: 0, preset: { amount: 7499, case_age_hours: 72, merchant_budget: 250000, max_incentive: 10, allowed_actions: baseActions, customer_segment: 'high_intent_repeat', failure_class: 'network_timeout', payment_method: 'card' }, prerequisites: ['Two incumbent rounds', 'Hidden response drift', 'Held-out and canary gates'] },
  { scenario_id: 'off_vs_on_casino', label: 'Recovery Agent OFF vs ON', short_label: 'OFF vs ON Casino', description: 'The same immutable traffic is replayed through the incumbent stack and the recovery agent.', primary_proof: 'One controlled replay isolates the product delta.', seed: 35404, event_count: 10000, attack_count: 500, preset: { amount: 4999, case_age_hours: 50, merchant_budget: 150000, max_incentive: 10, allowed_actions: baseActions, customer_segment: 'price_sensitive', failure_class: 'user_cancelled', payment_method: 'card' }, prerequisites: ['Immutable event stream', 'Identical attack index', 'Common random outcomes'] },
  { scenario_id: 'predictive_vs_causal', label: 'Naive ML vs Recovery Learning Agent', short_label: 'Naive vs Causal', description: 'A propensity model maximizes payment probability while the agent optimizes causal net value.', primary_proof: 'Prediction is not decisioning.', seed: 35505, event_count: 10000, attack_count: 0, preset: { amount: 4999, case_age_hours: 50, merchant_budget: 250000, max_incentive: 10, allowed_actions: baseActions, customer_segment: 'high_intent_repeat', failure_class: 'issuer_down', payment_method: 'card' }, prerequisites: ['Contaminated propensity classifier', 'Independent action models', 'No-action control'] },
]

const TABS: { value: ActiveTab; label: string }[] = [
  { value: 'triage', label: '01  Triage' },
  { value: 'delta', label: '02  Policy Delta' },
  { value: 'explain', label: '03  Explainability' },
]

const REASON_EXPLANATIONS: Record<string, string> = {
  AGENT_DISABLED_RAZORPAY_INCUMBENT_SIMULATION: 'The causal agent was bypassed and the researched incumbent recovery state machine handled the case.',
  BASELINE_TRANSIENT_FAILURE_RETRY: 'A temporary network or issuer failure entered the incumbent in-session retry path.',
  BASELINE_FIXED_SUBSCRIPTION_RETRY_LADDER: 'The subscription entered a fixed calendar retry rather than a learned intervention.',
  BASELINE_PAYMENT_LINK_REMINDER: 'The incumbent stack selected a static payment-link reminder.',
  BASELINE_STATIC_MERCHANT_OFFER: 'A merchant-wide fixed offer was applied without customer-level optimization.',
  BASELINE_VELOCITY_RISK_WATCH: 'The incumbent noticed unusual velocity but did not quarantine the learning signal.',
  BASELINE_RBI_AFA_CUSTOMER_AUTH_REQUIRED: 'The high-value payment moved to a customer-authenticated flow.',
  CAUSAL_NET_VALUE_BEATS_RAW_CONVERSION: 'The agent rejected a higher-converting option because its cost reduced incremental value.',
  INTEGRITY_REWARD_POISONING_DETECTED: 'Fast conversions paired with zero downstream quality indicate reward poisoning.',
  QUARANTINED_OUTCOMES_EXCLUDED: 'Suspicious outcomes remain auditable but cannot train the next policy.',
  IMMUTABLE_SEED_REPLAY: 'OFF and ON used the same events, attack timing, and random outcomes.',
  POLICY_DRIFT_DETECTED: 'Recent evidence no longer supports the previously successful response curve.',
  CANARY_LIFT_DEGRADATION_ROLLBACK: 'The candidate degraded in canary, so the last-known-good policy was restored.',
  CANARY_LIFT_DEGRADED: 'The candidate lost its expected lift after entering the bounded canary population.',
  PROMOTION_REJECTED: 'The safety gate refused to promote a policy that failed canary evidence.',
  ROLLBACK_TO_LAST_KNOWN_GOOD_POLICY: 'The service restored the proven incumbent payment-link policy.',
  LAST_KNOWN_GOOD_POLICY_ACTIVE: 'The proven incumbent payment-link policy is the active comparator before the candidate canary runs.',
  NAIVE_ML_OPTIMIZED_FOR_RAW_CONVERSION: 'The predictive comparator chose the 10% offer because it maximized payment probability without subtracting margin.',
  CAUSAL_LIFT_NEGATIVE_MARGIN_SAVED: 'The causal learner rejected the expensive discount and retained more incremental value with a payment link.',
  CONTROL_ADJUSTED_LIFT_SELECTED: 'The selected action was evaluated against what would happen without intervention.',
  NAIVE_ABSOLUTE_PROPENSITY_REJECTED: 'High payment probability alone did not prove the action caused payment.',
  GRACE_WINDOW_ACTIVE_SUPPRESS_DISCOUNT: 'The case is inside the recovery grace period, so discounts are blocked.',
  REGULATORY_RBI_15K_AFA_LIMIT: 'A high-value retry became a customer-authenticated payment link.',
  REGULATORY_NPCI_MAX_RETRIES_EXCEEDED: 'The UPI retry ceiling was reached, so further automatic attempts were suppressed.',
  INCENTIVE_LINK_NOT_ALLOWED: 'The merchant removed incentives from the permitted action space.',
  MERCHANT_BUDGET_EXHAUSTED: 'The merchant recovery budget has no remaining capacity for a discount.',
  MAX_INCENTIVE_LIMIT_APPLIED: 'The proposed discount tier exceeded the maximum incentive percentage entered by the merchant.',
  CIRCUIT_BREAKER_MERCHANT_BUDGET_CONCENTRATION: 'The proposed incentive would consume more than 25% of the merchant budget, so human approval is required.',
  CIRCUIT_BREAKER_SUSPICIOUS_FINGERPRINT_REVIEW: 'Repeated or suspicious IP/device activity requires human review before link execution.',
  CIRCUIT_BREAKER_INTEGRITY_WATCH_REVIEW: 'The integrity engine found repeated identity activity that is concerning but not strong enough to quarantine.',
  CIRCUIT_BREAKER_HIGH_TICKET_VALUE: 'The cart exceeds the automatic high-ticket threshold and requires human approval.',
  QUARANTINED_ACTION_SUPPRESSED: 'The integrity gate blocked execution for this suspicious cohort.',
  RAZORPAY_LINK_UNAVAILABLE_SANDBOX: 'The decision completed, but live Razorpay link execution was unavailable.',
  'CHALLENGER_POLICY_V3.6_EXECUTED': 'The promoted v3.6 challenger used its tighter margin calibration for this decision.',
}

const RESULT_DESCRIPTIONS = {
  amount: 'The original unpaid amount received from the recovery event.',
  finalAmount: 'The amount the customer would pay after the selected offer.',
  finalAction: 'The action that survived regulatory and merchant policy checks.',
  discountCost: 'The literal merchant margin committed to the intervention.',
  netRecovery: 'Expected recovered money after direct intervention cost.',
  incrementalRevenue: 'Additional expected value beyond doing nothing.',
  integrity: 'Whether the traffic looks genuine, concerning, or actively adversarial.',
}

function formatCurrency(value: number | null | undefined, compact = false) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', notation: compact ? 'compact' : 'standard', maximumFractionDigits: compact ? 1 : 2 }).format(value ?? 0)
}

function formatProbability(value: number | null | undefined) {
  return value == null ? 'Not available' : `${(value * 100).toFixed(1)}%`
}

function formatLift(value: number | null | undefined, pointsAlready = false) {
  if (value == null) return 'Not available'
  const points = pointsAlready ? value : value * 100
  return `${points >= 0 ? '+' : ''}${points.toFixed(1)} pp`
}

function formatRoi(value: number | null | undefined) {
  return value == null ? 'No direct spend' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}x`
}

function humanize(value: string | undefined) {
  return value ? value.replaceAll('_', ' ') : 'Awaiting decision'
}

function integrityColor(status: IntegrityStatus | undefined) {
  if (status === 'TRUSTED') return 'positive' as const
  if (status === 'WATCH') return 'notice' as const
  if (status === 'QUARANTINED') return 'negative' as const
  return 'neutral' as const
}

function explainReason(code: string) {
  return REASON_EXPLANATIONS[code] ?? `The deterministic trace recorded ${humanize(code).toLowerCase()}.`
}

function uniqueReasons(response: DemoResponse | null | undefined) {
  return Array.from(new Set([...(response?.reason_codes ?? []), ...(response?.ml_metrics?.reason_codes ?? []), ...(response?.scenario_metrics?.reason_codes ?? []), ...(response?.recovery_metrics?.reason_codes ?? [])]))
}

function ResultCell({ label, value, description }: { label: string; value: string; description: string }) {
  return <div className="result-cell"><span>{label}</span><strong>{value}</strong><p>{description}</p></div>
}

function DecisionSummary({ response, mode, scenarioId }: { response: DemoResponse; mode: 'off' | 'on'; scenarioId: ScenarioId }) {
  const metrics = response.recovery_metrics
  const tier = response.ml_metrics?.recommended_tier ?? (response.final_action === 'incentive_link' && response.input_amount > 0 ? Math.round((response.discount_cost / response.input_amount) * 100) : 0)
  const isPredictiveComparator = mode === 'off' && scenarioId === 'predictive_vs_causal'
  return (
    <div className={`decision-summary mode-${mode}`}>
      <div className="summary-topline"><div><span>{mode === 'on' ? `RECOVERY AGENT ${response.current_policy_version}` : isPredictiveComparator ? 'NAIVE PREDICTIVE ML' : 'RAZORPAY INCUMBENT SIMULATION'}</span><h4>{humanize(response.final_action)}</h4></div><Badge color={integrityColor(metrics?.integrity_status)} emphasis="subtle" size="small">{metrics?.integrity_status ?? 'NOT SCORED'}</Badge></div>
      <p className="summary-copy">{tier > 0 ? `${tier}% merchant offer selected.` : 'No personalized discount applied.'}</p>
      <div className="summary-metrics"><div><span>Net recovery</span><strong>{formatCurrency(response.net_expected_recovery)}</strong></div><div><span>Incremental</span><strong>{formatCurrency(response.expected_incremental_revenue)}</strong></div><div><span>ROI</span><strong>{formatRoi(metrics?.recovery_roi)}</strong></div></div>
      <div className="summary-verdict"><span>{mode === 'on' ? 'Policy verdict' : isPredictiveComparator ? 'Optimization target' : 'Recovery posture'}</span><strong>{mode === 'on' ? humanize(response.ml_metrics?.policy_decision) : isPredictiveComparator ? 'Raw conversion' : humanize(metrics?.integrity_status)}</strong></div>
    </div>
  )
}

function RecoveryDashboard() {
  const initialScenario = FALLBACK_SCENARIOS[0]
  const [scenarioCatalog, setScenarioCatalog] = useState(FALLBACK_SCENARIOS)
  const [selectedScenario, setSelectedScenario] = useState<ScenarioId>(initialScenario.scenario_id)
  const [agentEnabled, setAgentEnabled] = useState(true)
  const [autoExecuteEnabled, setAutoExecuteEnabled] = useState(false)
  const [activeTab, setActiveTab] = useState<ActiveTab>('triage')
  const [offerApproved, setOfferApproved] = useState(false)
  const [apiResponse, setApiResponse] = useState<DemoResponse | null>(null)
  const [pairedRuns, setPairedRuns] = useState<PairedRuns>({ signature: '' })
  const [amount, setAmount] = useState(initialScenario.preset.amount)
  const [caseAgeHours, setCaseAgeHours] = useState(initialScenario.preset.case_age_hours)
  const [merchantBudget, setMerchantBudget] = useState(initialScenario.preset.merchant_budget)
  const [maxIncentive, setMaxIncentive] = useState(initialScenario.preset.max_incentive)
  const [allowedActions, setAllowedActions] = useState([...initialScenario.preset.allowed_actions])
  const [warmup, setWarmup] = useState<WarmupResponse | null>(null)
  const [isWarming, setIsWarming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isApprovingOffer, setIsApprovingOffer] = useState(false)
  const [isPairLoading, setIsPairLoading] = useState(false)
  const [isBatchUpdating, setIsBatchUpdating] = useState(false)
  const [learningStep, setLearningStep] = useState('Idle')
  const [activePolicyVersion, setActivePolicyVersion] = useState('v3.5')
  const [latestBatchRun, setLatestBatchRun] = useState<BatchLearningResponse | null>(null)
  const [serviceOnline, setServiceOnline] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const toast = useToast()

  const scenario = scenarioCatalog.find((item) => item.scenario_id === selectedScenario) ?? initialScenario
  const currentSignature = JSON.stringify({ selectedScenario, amount, caseAgeHours, merchantBudget, maxIncentive, allowedActions: [...allowedActions].sort() })
  const comparisonReady = pairedRuns.signature === currentSignature && Boolean(pairedRuns.off && pairedRuns.on)
  const comparisonMetrics = pairedRuns.on?.scenario_metrics ?? pairedRuns.off?.scenario_metrics
  const activeAgentResponse = pairedRuns.on ?? (apiResponse?.agent_enabled ? apiResponse : null)

  useEffect(() => {
    let active = true
    const initialize = async () => {
      const [catalogResult, healthResult, policyResult] = await Promise.allSettled([axios.get<ScenarioCatalogItem[]>(SCENARIO_URL, { timeout: 15000 }), axios.get(HEALTH_URL, { timeout: 8000 }), axios.get<PolicyStateResponse>(POLICY_STATE_URL, { timeout: 8000 })])
      if (!active) return
      if (catalogResult.status === 'fulfilled' && catalogResult.value.data.length === 5) setScenarioCatalog(catalogResult.value.data)
      setServiceOnline(healthResult.status === 'fulfilled')
      if (policyResult.status === 'fulfilled') { setActivePolicyVersion(policyResult.value.data.active_version); setLatestBatchRun(policyResult.value.data.latest_run) }
    }
    void initialize()
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    const warm = async () => {
      setIsWarming(true)
      setWarmup(null)
      try {
        const response = await axios.post<WarmupResponse>(`${SCENARIO_URL}/${selectedScenario}/warmup`, {}, { timeout: 30000 })
        if (active) { setWarmup(response.data); setServiceOnline(true) }
      } catch { if (active) setErrorMessage('Scenario warm-up failed. Start FastAPI and retry.') }
      finally { if (active) setIsWarming(false) }
    }
    void warm()
    return () => { active = false }
  }, [selectedScenario])

  const resetEvidence = () => { setApiResponse(null); setPairedRuns({ signature: '' }); setOfferApproved(false); setErrorMessage(null) }

  const applyScenario = (scenarioId: ScenarioId) => {
    const next = scenarioCatalog.find((item) => item.scenario_id === scenarioId) ?? FALLBACK_SCENARIOS[0]
    setSelectedScenario(scenarioId); setAmount(next.preset.amount); setCaseAgeHours(next.preset.case_age_hours); setMerchantBudget(next.preset.merchant_budget); setMaxIncentive(next.preset.max_incentive); setAllowedActions([...next.preset.allowed_actions]); setAgentEnabled(true); resetEvidence()
  }

  const updateNumber = (value: string | undefined, setter: (next: number) => void) => { const parsed = Number(value); setter(Number.isFinite(parsed) ? parsed : 0); resetEvidence() }

  const toggleAction = (action: string, isChecked: boolean) => {
    setAllowedActions((current) => isChecked ? ACTION_OPTIONS.map(({ value }) => value).filter((value) => value === action || current.includes(value)) : current.filter((value) => value !== action))
    resetEvidence()
  }

  const validateInputs = () => {
    if ([amount, caseAgeHours, merchantBudget, maxIncentive].some((value) => value < 0)) return 'Amount, case age, budget, and incentive cap must be non-negative.'
    if (maxIncentive >= 100) return 'Max incentive must be between 0% and less than 100% of the amount.'
    if (allowedActions.length === 0) return 'Select at least one allowed action.'
    return null
  }

  const buildPayload = (enabled: boolean, executionId: string) => {
    const isAttackScene = ['recovery_casino', 'off_vs_on_casino'].includes(selectedScenario)
    const executionToken = executionId.replaceAll('-', '').slice(0, 24)
    const effectiveOfferLadder = Array.from(
      { length: Math.floor(maxIncentive) + 1 },
      (_, tier) => tier,
    )
    return {
      agent_enabled: enabled,
      auto_execute: enabled && autoExecuteEnabled,
      scenario: selectedScenario,
      recovery_case: { case_id: `ui-${executionToken}-${enabled ? 'on' : 'off'}`, merchant_id: `merchant-demo-${selectedScenario}`, transaction_id: isAttackScene ? `attack:ip_hash_00:${executionToken}` : `pay-${executionToken}`, customer_id: isAttackScene ? `attacker:device_hash_00:${executionToken}` : `customer-${selectedScenario}-${executionToken}`, amount, case_age_hours: caseAgeHours, attempt_count: isAttackScene ? 15 : 1, payment_method: scenario.preset.payment_method, failure_class: scenario.preset.failure_class, customer_segment: scenario.preset.customer_segment, event_time: new Date().toISOString() },
      merchant_constraints: { merchant_budget: merchantBudget, merchant_budget_spent: 0, merchant_budget_exhausted: false, max_incentive: maxIncentive, max_contacts: 3, contacts_used: 0, recovery_window_hours: 48.0, offer_ladder: effectiveOfferLadder, allowed_actions: allowedActions, policy_version: activePolicyVersion },
      candidate_actions: allowedActions,
    }
  }

  const apiError = (error: unknown) => {
    if (!axios.isAxiosError(error)) return 'Unable to reach the FastAPI recovery service.'
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((item) => item.msg).join(', ')
    return error.code === 'ECONNABORTED' ? 'The request timed out while preparing the scenario.' : 'Unable to reach the FastAPI recovery service.'
  }

  const handleInjectWebhook = async () => {
    setOfferApproved(false); setErrorMessage(null)
    const validationError = validateInputs()
    if (validationError) { setErrorMessage(validationError); return }
    setIsLoading(true)
    try {
      const response = await axios.post<DemoResponse>(EXECUTE_URL, buildPayload(agentEnabled, crypto.randomUUID()), { timeout: 30000 })
      setApiResponse(response.data); setPairedRuns({ signature: currentSignature, ...(agentEnabled ? { on: response.data } : { off: response.data }) }); setActiveTab('triage'); setServiceOnline(true)
      toast.show({ content: `${agentEnabled ? 'Agent' : 'Incumbent'} execution complete.`, color: 'positive', autoDismiss: true })
    } catch (error) { const message = apiError(error); setErrorMessage(message); toast.show({ content: message, color: 'negative', autoDismiss: true }) }
    finally { setIsLoading(false) }
  }

  const handlePairedBenchmark = async () => {
    setOfferApproved(false); setErrorMessage(null)
    const validationError = validateInputs()
    if (validationError) { setErrorMessage(validationError); return }
    setIsPairLoading(true)
    const pairId = crypto.randomUUID()
    try {
      const [offResponse, onResponse] = await Promise.all([
        axios.post<DemoResponse>(EXECUTE_URL, buildPayload(false, pairId), { timeout: 45000 }),
        axios.post<DemoResponse>(EXECUTE_URL, buildPayload(true, pairId), { timeout: 45000 }),
      ])
      setPairedRuns({ signature: currentSignature, off: offResponse.data, on: onResponse.data }); setApiResponse(agentEnabled ? onResponse.data : offResponse.data); setActiveTab('delta'); setServiceOnline(true)
      toast.show({ content: 'Paired benchmark complete. OFF and ON used identical inputs.', color: 'positive', autoDismiss: true })
    } catch (error) { const message = apiError(error); setErrorMessage(message); toast.show({ content: message, color: 'negative', autoDismiss: true }) }
    finally { setIsPairLoading(false) }
  }

  const handleBatchLearning = async () => {
    setIsBatchUpdating(true); setLearningStep('Starting policy iteration'); toast.show({ content: 'Starting Batch Policy Iteration...', color: 'information', autoDismiss: true })
    try {
      const response = await axios.post<BatchLearningResponse>(BATCH_LEARNING_URL, { trigger_source: 'manual_ui', force_scenario: selectedScenario === 'policy_rollback' ? 'policy_rollback' : 'normal' }, { headers: { 'Idempotency-Key': crypto.randomUUID() }, timeout: 30000 })
      setLearningStep('Outcomes labeled'); toast.show({ content: 'Attribution window closed. Outcomes labeled.', color: 'notice', autoDismiss: true })
      await new Promise((resolve) => window.setTimeout(resolve, 650))
      setLatestBatchRun(response.data); setActivePolicyVersion(response.data.active_version); resetEvidence()
      if (response.data.promoted) {
        setLearningStep(`Policy ${response.data.active_version} promoted`)
        toast.show({ content: 'Held-out benchmark passed. New causal policy promoted.', color: 'positive', autoDismiss: true })
      } else {
        setLearningStep(`Rollback retained ${response.data.active_version}`)
        toast.show({ content: 'Canary degraded. Challenger rejected and champion restored.', color: 'negative', autoDismiss: true })
      }
    } catch (error) {
      const message = apiError(error); setLearningStep('Batch failed'); setErrorMessage(message); toast.show({ content: message, color: 'negative', autoDismiss: true })
    } finally { setIsBatchUpdating(false) }
  }

  const approveOffer = async () => {
    if (!apiResponse || !['payment_link', 'incentive_link'].includes(apiResponse.final_action)) return
    setIsApprovingOffer(true); setErrorMessage(null)
    try {
      const response = await axios.post<ApproveLinkResponse>(APPROVE_LINK_URL, {
        case_id: apiResponse.case_id,
        approved_action: apiResponse.final_action,
      }, { timeout: 30000 })
      const executed = { ...apiResponse, status: response.data.status, short_url: response.data.short_url, final_amount: response.data.final_amount }
      setApiResponse(executed)
      setPairedRuns((current) => ({
        ...current,
        on: current.on?.case_id === response.data.case_id
          ? { ...current.on, status: response.data.status, short_url: response.data.short_url, final_amount: response.data.final_amount }
          : current.on,
      }))
      setOfferApproved(true)
      toast.show({ content: 'Offer approved. Link generated on Razorpay Dashboard.', color: 'positive', autoDismiss: true })
    } catch (error) {
      const message = apiError(error)
      setErrorMessage(message)
      toast.show({ content: message, color: 'negative', autoDismiss: true })
    } finally { setIsApprovingOffer(false) }
  }
  const selectedMetrics = apiResponse?.recovery_metrics
  const selectedReasons = uniqueReasons(apiResponse)
  const offReasons = uniqueReasons(pairedRuns.off)
  const onReasons = uniqueReasons(pairedRuns.on)
  const confidence = activeAgentResponse?.ml_metrics?.confidence
  const circuitBreakerTriggered = apiResponse?.reason_codes.some((reason) => reason.startsWith('CIRCUIT_BREAKER_')) ?? false

  return (
    <div className="recovery-shell">
      <aside className="simulator-pane">
        <div className="brand-row"><div className="brand-symbol"><span>R</span></div><div><Text size="small" weight="semibold" color="surface.text.staticWhite.normal">RECOVERY CONTROL</Text><Text size="xsmall" color="surface.text.staticWhite.muted">Active policy · {activePolicyVersion}</Text></div></div>
        <div className="control-intro"><div className="live-row"><span className={`status-light ${serviceOnline ? 'online' : ''}`} /><span>{serviceOnline ? 'FASTAPI ONLINE' : 'WAITING FOR FASTAPI'}</span></div><Text size="small" color="surface.text.staticWhite.muted">Choose a controlled scene, set merchant vetoes, and replay the same evidence through both systems.</Text></div>

        <div className="control-form">
          <div className="scenario-select"><Dropdown selectionType="single"><SelectInput label="Demo scenario" value={selectedScenario} onChange={({ values }) => { const value = values[0] as ScenarioId | undefined; if (value) applyScenario(value) }} /><DropdownOverlay><ActionList>{scenarioCatalog.map((item, index) => <ActionListItem key={item.scenario_id} value={item.scenario_id} title={`${index + 1}. ${item.label}`} description={item.primary_proof} />)}</ActionList></DropdownOverlay></Dropdown></div>
          <div className={`scenario-card ${isWarming ? 'is-warming' : ''}`}><div className="scenario-card-head"><span>{isWarming ? 'PREPARING' : 'SCENE READY'}</span><Badge color={isWarming ? 'notice' : 'positive'} emphasis="subtle" size="small">{isWarming ? 'WARMING' : warmup?.cache_hit ? 'CACHED' : 'READY'}</Badge></div><strong>{scenario.short_label}</strong><p>{scenario.description}</p><small>Seed {scenario.seed} · {scenario.event_count.toLocaleString('en-IN')} cases · {scenario.attack_count} attacks</small></div>
          <div className="numeric-grid"><TextInput label="Amount" type="number" prefix="₹" value={String(amount)} onChange={({ value }) => updateNumber(value, setAmount)} /><TextInput label="Case age" type="number" suffix="hours" value={String(caseAgeHours)} onChange={({ value }) => updateNumber(value, setCaseAgeHours)} /><TextInput label="Merchant budget" type="number" prefix="₹" value={String(merchantBudget)} onChange={({ value }) => updateNumber(value, setMerchantBudget)} /><TextInput label="Max incentive" type="number" suffix="%" helpText="Percentage ceiling of amount" value={String(maxIncentive)} onChange={({ value }) => updateNumber(value, setMaxIncentive)} /></div>
          <div className="action-control"><div><Text size="small" weight="semibold">Allowed actions</Text><Text size="xsmall" color="surface.text.gray.muted">Merchant vetoes are enforced after model scoring.</Text></div><div className="checkbox-grid">{ACTION_OPTIONS.map((action) => <Checkbox key={action.value} value={action.value} isChecked={allowedActions.includes(action.value)} onChange={({ isChecked }) => toggleAction(action.value, isChecked)} size="small">{action.label}</Checkbox>)}</div></div>
          <div className={`agent-toggle ${agentEnabled ? 'agent-on' : 'agent-off'}`}><div><Text size="small" weight="semibold">Selected single-run mode</Text><Text size="xsmall" color="surface.text.gray.muted">{agentEnabled ? 'ON · causal learner + integrity gate' : 'OFF · incumbent recovery stack'}</Text></div><Switch accessibilityLabel="Toggle recovery agent" isChecked={agentEnabled} onChange={({ isChecked }) => { setAgentEnabled(isChecked); if (comparisonReady) setApiResponse(isChecked ? pairedRuns.on ?? null : pairedRuns.off ?? null); else setApiResponse(null); setOfferApproved(false) }} /></div>
          <div className={`agent-toggle auto-toggle ${autoExecuteEnabled && agentEnabled ? 'agent-on' : 'agent-off'}`}><div><Text size="small" weight="semibold">ON-AUTO execution</Text><Text size="xsmall" color="surface.text.gray.muted">{autoExecuteEnabled && agentEnabled ? 'Low-risk links execute automatically' : 'Manual approval is the default'}</Text></div><Switch accessibilityLabel="Toggle automatic low-risk execution" isChecked={autoExecuteEnabled && agentEnabled} isDisabled={!agentEnabled} onChange={({ isChecked }) => { setAutoExecuteEnabled(isChecked); resetEvidence() }} /></div>
          <div className="control-actions"><Button variant="primary" size="large" icon={ZapIcon} isFullWidth isLoading={isPairLoading} isDisabled={isWarming || isLoading} onClick={handlePairedBenchmark}>Execute Paired Benchmark</Button><Button variant="secondary" size="medium" icon={SendIcon} isFullWidth isLoading={isLoading} isDisabled={isWarming || isPairLoading} onClick={handleInjectWebhook}>Run Selected Mode</Button><Button variant="tertiary" size="medium" icon={RefreshIcon} isFullWidth isLoading={isBatchUpdating} onClick={handleBatchLearning}>Trigger Batch Policy Update</Button></div>
          <div className="learning-status"><ActivityIcon size="small" color="surface.icon.gray.subtle" /><span>Learning loop</span><strong>{learningStep}</strong></div>
          {latestBatchRun && <div className={`batch-audit ${latestBatchRun.promoted ? 'promoted' : 'rollback'}`}><div><span>LAST PERSISTED BATCH RUN</span><Badge color={latestBatchRun.promoted ? 'positive' : 'negative'} emphasis="subtle" size="small">{latestBatchRun.run_status}</Badge></div><strong>{latestBatchRun.champion_model} → {latestBatchRun.challenger_model}</strong><p>ROI {latestBatchRun.incumbent_roi.toFixed(2)}x → {latestBatchRun.challenger_roi.toFixed(2)}x · quarantine {(latestBatchRun.quarantine_rate * 100).toFixed(0)}%</p>{latestBatchRun.rejection_reason && <small>{latestBatchRun.rejection_reason}</small>}<code>{latestBatchRun.run_id.slice(0, 18)}</code></div>}
        </div>
        <div className="pane-footnote"><ShieldIcon size="small" color="surface.icon.staticWhite.muted" /></div>
      </aside>

      <main className="workspace-pane">
        <header className="workspace-header"><div><Text size="xsmall" weight="semibold" color="surface.text.gray.muted">DECISION OPERATIONS / {scenario.short_label.toUpperCase()}</Text></div><div className="run-context"><span>{formatCurrency(amount)}</span><span><ClockIcon size="small" color="surface.icon.gray.subtle" /> {caseAgeHours}h old</span><Badge color={comparisonReady ? 'positive' : 'notice'} emphasis="subtle">{comparisonReady ? 'PAIR VERIFIED' : 'AWAITING RUN'}</Badge></div></header>
        <div className="scenario-proof"><div><span>SCENE THESIS</span><strong>{scenario.primary_proof}</strong></div><div className="proof-prerequisites">{scenario.prerequisites.slice(0, 3).map((item) => <small key={item}>{item}</small>)}</div></div>
        <nav className="tab-rail" aria-label="Recovery dashboard views">{TABS.map((tab) => <Button key={tab.value} variant={activeTab === tab.value ? 'primary' : 'tertiary'} size="small" onClick={() => setActiveTab(tab.value)}>{tab.label}</Button>)}</nav>
        {errorMessage && <div className="alert-slot"><Alert title="Execution failed" description={errorMessage} color="negative" isFullWidth isDismissible onDismiss={() => setErrorMessage(null)} /></div>}

        <section className="workspace-content" key={activeTab}>
          {activeTab === 'triage' && <div className="triage-layout"><div className="view-heading"><div><Text size="small" weight="semibold" color="surface.text.primary.normal">TRIAGE DESK</Text><Heading as="h3" size="large" weight="semibold">Inspect the action before releasing value.</Heading></div><Badge color={integrityColor(selectedMetrics?.integrity_status)} emphasis="subtle" size="medium">{selectedMetrics?.integrity_status ?? 'NOT EVALUATED'}</Badge></div>
            {!apiResponse ? <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="empty-state"><div className="decision-icon"><SparklesIcon size="large" color="surface.icon.primary.normal" /></div><Heading as="h4" size="large" weight="semibold">The triage desk is ready.</Heading><Text size="small" color="surface.text.gray.muted">Execute the paired benchmark for immediate incumbent and agent evidence, or run one selected mode for debugging.</Text></div></CardBody></Card> : <>
              <div className={`decision-pair ${comparisonReady ? '' : 'single'}`}>{comparisonReady && pairedRuns.off && <DecisionSummary response={pairedRuns.off} mode="off" scenarioId={selectedScenario} />}{comparisonReady && pairedRuns.on && <DecisionSummary response={pairedRuns.on} mode="on" scenarioId={selectedScenario} />}{!comparisonReady && <DecisionSummary response={apiResponse} mode={apiResponse.agent_enabled ? 'on' : 'off'} scenarioId={selectedScenario} />}</div>
              <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="decision-card"><div className="decision-hero"><div className="decision-icon">{apiResponse.agent_enabled ? <SparklesIcon size="large" color="surface.icon.primary.normal" /> : <ShieldIcon size="large" color="surface.icon.primary.normal" />}</div><div><Text size="xsmall" color="surface.text.gray.muted">{apiResponse.agent_enabled ? 'AGENTIC EXECUTION' : 'INCUMBENT EXECUTION'}</Text><Heading as="h4" size="2xlarge" weight="semibold">{humanize(apiResponse.final_action)}</Heading><Text size="small" color="surface.text.gray.muted">Showing the {apiResponse.agent_enabled ? 'ON' : 'OFF'} decision selected by the control toggle.</Text></div></div>
                <div className="result-grid"><ResultCell label="Amount input" value={formatCurrency(apiResponse.input_amount)} description={RESULT_DESCRIPTIONS.amount} /><ResultCell label="Final amount" value={formatCurrency(apiResponse.final_amount)} description={RESULT_DESCRIPTIONS.finalAmount} /><ResultCell label="Final action" value={humanize(apiResponse.final_action)} description={RESULT_DESCRIPTIONS.finalAction} /><ResultCell label="Integrity" value={selectedMetrics?.integrity_status ?? 'Not scored'} description={RESULT_DESCRIPTIONS.integrity} /><ResultCell label="Discount cost" value={formatCurrency(apiResponse.discount_cost)} description={RESULT_DESCRIPTIONS.discountCost} /><ResultCell label="Net expected recovery" value={formatCurrency(apiResponse.net_expected_recovery)} description={RESULT_DESCRIPTIONS.netRecovery} /><ResultCell label="Incremental revenue" value={formatCurrency(apiResponse.expected_incremental_revenue)} description={RESULT_DESCRIPTIONS.incrementalRevenue} /><ResultCell label="Recovery ROI" value={formatRoi(selectedMetrics?.recovery_roi)} description="Incremental value divided by direct intervention spend." /></div>
                <div className="reason-summary"><div><Text size="small" weight="semibold">Reason codes</Text><Text size="xsmall" color="surface.text.gray.muted">The exact deterministic rules behind the final action.</Text></div><div>{selectedReasons.map((reason) => <code key={reason}>{reason}</code>)}</div></div>
                {apiResponse.agent_enabled && apiResponse.status === 'pending_human_review' && !offerApproved && <div className={`approval-zone ${circuitBreakerTriggered ? 'high-risk' : ''}`}><div><Text size="small" weight="semibold">{circuitBreakerTriggered ? 'Circuit breaker · human review required' : 'Manual approval required'}</Text><Text size="xsmall" color="surface.text.gray.muted">No Razorpay link exists yet. Approval executes the pending recommendation.</Text></div><Button variant="primary" icon={CheckCircleIcon} isLoading={isApprovingOffer} onClick={approveOffer}>Approve & Execute Offer</Button></div>}
                {apiResponse.agent_enabled && apiResponse.status === 'auto_executed' && apiResponse.short_url && <div className="released-link auto-link"><CheckCircleIcon size="medium" color="interactive.icon.positive.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">LOW RISK · AUTO EXECUTED</Text><Text size="small" weight="semibold">Link Generated on Razorpay Dashboard</Text><a href={apiResponse.short_url} target="_blank" rel="noreferrer">Open Razorpay payment link</a></div></div>}
                {apiResponse.agent_enabled && offerApproved && apiResponse.short_url && <div className="released-link"><CheckCircleIcon size="medium" color="interactive.icon.positive.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">HUMAN APPROVAL · EXECUTION CONFIRMED</Text><Text size="small" weight="semibold">Link Generated on Razorpay Dashboard</Text><a href={apiResponse.short_url} target="_blank" rel="noreferrer">Open Razorpay payment link</a></div></div>}
                {!apiResponse.agent_enabled && apiResponse.short_url && <div className="released-link baseline-link"><CheckCircleIcon size="medium" color="interactive.icon.positive.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">INCUMBENT EXECUTION</Text><Text size="small" weight="semibold">Link Generated on Razorpay Dashboard</Text></div></div>}
              </div></CardBody></Card>
            </>}
          </div>}

          {activeTab === 'delta' && <div className="delta-layout"><div className="view-heading"><div><Text size="small" weight="semibold" color="surface.text.primary.normal">PAIRED POLICY DELTA</Text><Heading as="h3" size="large" weight="semibold">Measure the product, not two different samples.</Heading></div><Badge color={comparisonReady ? 'positive' : 'notice'} emphasis="subtle" size="medium">{comparisonReady ? 'COMMON SEED VERIFIED' : 'PAIR REQUIRED'}</Badge></div>
            {!comparisonReady ? <div className="pair-callout"><ZapIcon size="large" color="surface.icon.primary.normal" /><div><Heading as="h4" size="medium" weight="semibold">Run both systems in one click.</Heading><Text size="small" color="surface.text.gray.muted">Two simultaneous requests use identical merchant inputs and open this evidence view automatically.</Text></div><Button variant="primary" icon={ZapIcon} isLoading={isPairLoading} isDisabled={isWarming} onClick={handlePairedBenchmark}>Execute Paired Benchmark</Button></div> : <>
              <div className="win-banner"><div className="win-mark"><TrendingUpIcon size="large" color="surface.icon.staticWhite.normal" /></div><div><span>WIN CONDITION</span><strong>{(comparisonMetrics?.incremental_recovered_revenue ?? 0) > 0 ? 'Agent generates more net recovery and protects the learning signal.' : 'Evidence captured. Review the policy delta.'}</strong></div><Badge color="positive" emphasis="subtle">{pairedRuns.on?.scenario_metrics?.attacks_quarantined ?? 0} ATTACKS QUARANTINED</Badge></div>
              <div className="delta-grid"><Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="panel-card"><div className="panel-title"><TrendingUpIcon size="medium" color="surface.icon.primary.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">SAME STREAM · SEED {comparisonMetrics?.seed}</Text><Heading as="h4" size="medium" weight="semibold">{selectedScenario === 'predictive_vs_causal' ? 'Naive ML vs Recovery Agent' : 'Incumbent vs Recovery Agent'}</Heading></div></div><div className="table-wrap"><table className="comparison-table"><thead><tr><th>Metric</th><th>{selectedScenario === 'predictive_vs_causal' ? 'Naive ML (OFF)' : 'Agent OFF'}</th><th>Agent ON</th></tr></thead><tbody><tr><td>Amount input</td><td>{formatCurrency(pairedRuns.off?.input_amount)}</td><td>{formatCurrency(pairedRuns.on?.input_amount)}</td></tr><tr><td>Final action</td><td>{humanize(pairedRuns.off?.final_action)}</td><td>{humanize(pairedRuns.on?.final_action)}</td></tr><tr><td>Net expected recovery</td><td>{formatCurrency(pairedRuns.off?.net_expected_recovery)}</td><td>{formatCurrency(pairedRuns.on?.net_expected_recovery)}</td></tr><tr><td>Integrity status</td><td>{pairedRuns.off?.recovery_metrics?.integrity_status ?? 'Not scored'}</td><td>{pairedRuns.on?.recovery_metrics?.integrity_status ?? 'Not scored'}</td></tr><tr><td>Discount cost</td><td>{formatCurrency(pairedRuns.off?.discount_cost)}</td><td>{formatCurrency(pairedRuns.on?.discount_cost)}</td></tr><tr><td>Case incremental value</td><td>{formatCurrency(pairedRuns.off?.expected_incremental_revenue)}</td><td>{formatCurrency(pairedRuns.on?.expected_incremental_revenue)}</td></tr><tr><td>Recovery ROI</td><td>{formatRoi(pairedRuns.off?.recovery_metrics?.recovery_roi)}</td><td>{formatRoi(pairedRuns.on?.recovery_metrics?.recovery_roi)}</td></tr><tr><td>Attacks quarantined</td><td>{pairedRuns.off?.scenario_metrics?.attacks_quarantined ?? 0}</td><td>{pairedRuns.on?.scenario_metrics?.attacks_quarantined ?? 0}</td></tr><tr><td>{selectedScenario === 'predictive_vs_causal' ? 'Comparator objective' : 'Incumbent coverage'}</td><td>{selectedScenario === 'predictive_vs_causal' ? 'Absolute pay probability' : `${((pairedRuns.off?.recovery_metrics?.incumbent_capability_coverage ?? 0) * 100).toFixed(0)}%`}</td><td>{selectedScenario === 'predictive_vs_causal' ? 'Incremental net value' : 'Same fair baseline'}</td></tr></tbody></table></div><p className="simulation-note"><InfoIcon size="small" color="surface.icon.gray.subtle" /> {pairedRuns.off?.recovery_metrics?.simulation_disclaimer}</p></div></CardBody></Card>
                <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="panel-card chart-panel"><div><Text size="xsmall" color="surface.text.gray.muted">FIVE POLICY ITERATIONS</Text><Heading as="h4" size="medium" weight="semibold">Cumulative net recovery</Heading><Text size="xsmall" color="surface.text.gray.muted">Common random outcomes isolate policy performance over time.</Text></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={comparisonMetrics?.cumulative_revenue ?? []} margin={{ top: 12, right: 12, left: 2, bottom: 4 }}><CartesianGrid strokeDasharray="4 6" stroke="#dce4df" vertical={false} /><XAxis dataKey="round" tickLine={false} axisLine={false} tick={{ fill: '#66736d', fontSize: 11 }} /><YAxis tickFormatter={(value: number) => `₹${Math.round(value / 100000)}L`} tickLine={false} axisLine={false} tick={{ fill: '#66736d', fontSize: 11 }} /><Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #dce4df', boxShadow: '0 12px 32px rgba(20,42,33,.12)' }} /><Legend iconType="circle" wrapperStyle={{ fontSize: 11, paddingTop: 12 }} /><Line name={selectedScenario === 'predictive_vs_causal' ? 'Naive ML (Agent OFF)' : 'Razorpay incumbent'} type="monotone" dataKey="baseline" stroke="#8e9a95" strokeWidth={2.5} dot={{ r: 3 }} /><Line name="Recovery Agent" type="monotone" dataKey="agent" stroke="#087b5b" strokeWidth={3.5} dot={{ r: 4 }} />{selectedScenario !== 'predictive_vs_causal' && <Line name="Naive ML" type="monotone" dataKey="naive_ml" stroke="#e2765d" strokeWidth={2} strokeDasharray="5 5" dot={false} />}</LineChart></ResponsiveContainer></div></div></CardBody></Card></div>
            </>}
          </div>}

          {activeTab === 'explain' && <div className="explain-layout"><div className="view-heading"><div><Text size="small" weight="semibold" color="surface.text.primary.normal">DECISION EVIDENCE</Text><Heading as="h3" size="large" weight="semibold">From control probability to deterministic veto.</Heading></div><Badge color={integrityColor(activeAgentResponse?.recovery_metrics?.integrity_status)} emphasis="subtle" size="medium">{activeAgentResponse?.recovery_metrics?.integrity_status ?? 'NO AGENT RUN'}</Badge></div>
            {!activeAgentResponse ? <Alert title="Agent evidence is not available yet" description="Execute the paired benchmark or run the selected mode with the agent ON." color="information" isFullWidth isDismissible={false} /> : <div className="explain-grid">
              <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="math-card"><div className="panel-title"><SparklesIcon size="medium" color="surface.icon.primary.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">CAUSAL ATTRIBUTION</Text></div></div><div className="equation-flow"><div><span>Baseline probability</span><strong>{formatProbability(activeAgentResponse.ml_metrics?.baseline_pay_probability)}</strong><small>Would pay without intervention</small></div><b>→</b><div><span>Predicted probability</span><strong>{formatProbability(activeAgentResponse.ml_metrics?.predicted_pay_probability)}</strong><small>Would pay under selected action</small></div><b>−</b><div className="lift-result"><span>Causal lift</span><strong>{formatLift(activeAgentResponse.ml_metrics?.causal_lift)}</strong><small>Increment caused by the action</small></div></div><div className="value-equation"><span>EXPECTED INCREMENTAL NET</span><code>amount × causal lift − intervention cost − abuse penalty</code><strong>{formatCurrency(activeAgentResponse.expected_incremental_revenue)}</strong></div><div className="confidence-band"><div><span>95% confidence interval</span><strong>{confidence ? `${formatLift(confidence.lower_bound)} to ${formatLift(confidence.upper_bound)}` : 'Not available'}</strong></div><Badge color={activeAgentResponse.ml_metrics?.model_available ? 'positive' : 'notice'} emphasis="subtle">{activeAgentResponse.ml_metrics?.model_available ? 'MODEL AVAILABLE' : 'SAFE FALLBACK'}</Badge></div></div></CardBody></Card>
              <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="rules-card"><div className="panel-title"><ShieldIcon size="medium" color="surface.icon.primary.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">DETERMINISTIC LAYER</Text><Heading as="h4" size="medium" weight="semibold">Rules the model cannot override</Heading></div></div><div className="reason-columns"><div><span>{selectedScenario === 'predictive_vs_causal' ? 'NAIVE ML · AGENT OFF' : 'AGENT OFF'}</span>{offReasons.length ? offReasons.map((reason, index) => <div className="reason-row" key={reason}><i>{String(index + 1).padStart(2, '0')}</i><div><code>{reason}</code><p>{explainReason(reason)}</p></div></div>) : <p className="muted-copy">Run the pair to load comparator evidence.</p>}</div><div><span>CAUSAL AGENT · ON</span>{onReasons.map((reason, index) => <div className="reason-row" key={reason}><i>{String(index + 1).padStart(2, '0')}</i><div><code>{reason}</code><p>{explainReason(reason)}</p></div></div>)}</div></div></div></CardBody></Card>
              <Card variant="secondary" padding="spacing.7" width="100%"><CardBody><div className="trace-card"><div className="panel-title"><ActivityIcon size="medium" color="surface.icon.primary.normal" /><div><Text size="xsmall" color="surface.text.gray.muted">EXECUTION PIPELINE</Text><Heading as="h4" size="medium" weight="semibold">Every state transition is inspectable</Heading></div></div><div className="trace-columns"><div><span>{selectedScenario === 'predictive_vs_causal' ? 'NAIVE ML TRACE' : 'INCUMBENT TRACE'}</span>{(pairedRuns.off?.recovery_metrics?.action_trace ?? []).map((step) => <div className="trace-step" key={`${step.stage}-${step.reason_code}`}><i /><div><strong>{humanize(step.stage)} → {humanize(step.action)}</strong><p>{step.description}</p><code>{step.reason_code}</code></div></div>)}</div><div><span>CAUSAL AGENT TRACE</span>{(activeAgentResponse.recovery_metrics?.action_trace ?? []).map((step) => <div className="trace-step" key={`${step.stage}-${step.reason_code}`}><i /><div><strong>{humanize(step.stage)} → {humanize(step.action)}</strong><p>{step.description}</p><code>{step.reason_code}</code></div></div>)}</div></div></div></CardBody></Card>
            </div>}
          </div>}
        </section>
      </main>
    </div>
  )
}

export default function App() {
  return <BladeProvider colorScheme="light" themeTokens={bladeTheme}><RecoveryDashboard /><ToastContainer /></BladeProvider>
}
