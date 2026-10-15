// Automation scheduler kernel (VST-038): the pure decisions for scheduler leases and each
// automation's completion policy, retries and misfires.
// No timers, no I/O — everything takes an injected `now` so the host can drive it from a
// single interval and every decision is deterministically testable.
//
// Four decisions the spec names:
//   1. Single-runner LEASE — only one scheduler fires automations, even with a laptop AND a
//      mini both connected (lease contention).
//   2. Misfire COALESCING — a host that was asleep across several fire times runs ONCE, not
//      N times (missed-schedule coalesce).
//   3. Completion-policy VERDICTS — decide done / continue / needs-review, incl. an
//      AI-evaluated policy whose verdict is injected (the model call lives outside).
//   4. Acknowledged-RISK gate — a run with unacknowledged risks (e.g. `local-checkout`) is
//      refused (risk acknowledgement required).

// ---- 1. single-runner lease ----

export interface SchedulerLease {
  owner: string | null
  acquiredAt: number
  heartbeatAt: number
  expiresAt: number
}

export const EMPTY_LEASE: SchedulerLease = { owner: null, acquiredAt: 0, heartbeatAt: 0, expiresAt: 0 }

// Who holds the lease right now — null if free (never taken, or expired). An expired lease
// is free regardless of its recorded owner, so a crashed scheduler never blocks forever.
export function leaseHeldBy(lease: SchedulerLease, now: number): string | null {
  if (!lease.owner) return null
  return now < lease.expiresAt ? lease.owner : null
}

export interface LeaseResult {
  ok: boolean
  lease: SchedulerLease
  heldBy: string | null
}

// Acquire (or take over an expired) lease. The current holder re-acquiring is a renew. A
// different owner is refused while the lease is live (contention → the loser stands down).
export function acquireLease(lease: SchedulerLease, owner: string, now: number, ttlMs: number): LeaseResult {
  const holder = leaseHeldBy(lease, now)
  if (holder && holder !== owner) return { ok: false, lease, heldBy: holder }
  const next: SchedulerLease = { owner, acquiredAt: holder === owner ? lease.acquiredAt : now, heartbeatAt: now, expiresAt: now + ttlMs }
  return { ok: true, lease: next, heldBy: owner }
}

// Renew only if still the holder; a lost lease (expired + taken by another) fails so the
// caller stops firing.
export function renewLease(lease: SchedulerLease, owner: string, now: number, ttlMs: number): LeaseResult {
  if (leaseHeldBy(lease, now) !== owner) return { ok: false, lease, heldBy: leaseHeldBy(lease, now) }
  return { ok: true, lease: { ...lease, owner, heartbeatAt: now, expiresAt: now + ttlMs }, heldBy: owner }
}

// ---- 2. due calculation + misfire coalescing ----

export type MisfirePolicy = 'coalesce' | 'run_each'

export interface AutomationTiming {
  enabled: boolean
  everyMs: number
  lastRunAt: number | null
  // Host-committed first-fire instant, also shown in the automation UI.
  // Missing legacy instants must be initialized by the host before dispatch.
  nextRunAt?: number | null
  misfire: MisfirePolicy
  // Cap on run_each catch-up so a week-long sleep can't queue thousands of runs.
  maxCatchup?: number
}

// How many fire times have elapsed since the last run (0 if not yet due). A never-run
// automation waits for its committed first fire, rather than running when enabled.
export function missedFireCount(timing: AutomationTiming, now: number): number {
  if (!timing.enabled || timing.everyMs <= 0) return 0
  if (timing.lastRunAt == null) {
    const first = timing.nextRunAt
    if (typeof first !== 'number' || !Number.isFinite(first) || first <= 0 || now < first) return 0
    return 1 + Math.floor((now - first) / timing.everyMs)
  }
  const elapsed = now - timing.lastRunAt
  if (elapsed < timing.everyMs) return 0
  return Math.floor(elapsed / timing.everyMs)
}

// How many runs to actually dispatch now. `coalesce` (default) collapses any number of
// missed fires into ONE; `run_each` dispatches each missed fire up to maxCatchup.
export function dueRuns(timing: AutomationTiming, now: number): number {
  const missed = missedFireCount(timing, now)
  if (missed <= 0) return 0
  if (timing.misfire === 'run_each') return Math.min(missed, timing.maxCatchup ?? 10)
  return 1
}

// Next scheduled fire time from a reference point.
export function nextRunAt(timing: AutomationTiming, now: number): number {
  if (timing.everyMs <= 0) return now
  if (timing.lastRunAt == null) {
    const first = timing.nextRunAt
    return typeof first === 'number' && Number.isFinite(first) && first > 0 ? first : now + timing.everyMs
  }
  const missed = missedFireCount(timing, now)
  return timing.lastRunAt + Math.max(1, missed) * timing.everyMs
}

// ---- 3. retry policy ----

export interface RetryPolicy {
  maxAttempts: number
  backoffMs: number
  factor?: number
  maxBackoffMs?: number
}

// After `attempt` (1-based) failures, should we retry and after how long? Exponential
// backoff, capped. attempt >= maxAttempts → give up.
export function nextRetry(attempt: number, policy: RetryPolicy): { retry: boolean; delayMs: number } {
  if (attempt >= policy.maxAttempts) return { retry: false, delayMs: 0 }
  const factor = policy.factor ?? 2
  const raw = policy.backoffMs * Math.pow(factor, Math.max(0, attempt - 1))
  return { retry: true, delayMs: Math.min(raw, policy.maxBackoffMs ?? Number.MAX_SAFE_INTEGER) }
}

// ---- 4. completion policy (versioned, AI-evaluated) ----

export type CompletionPolicy =
  | { type: 'none'; version: number }
  | { type: 'max_iterations'; version: number; max: number }
  | { type: 'on_signal'; version: number; signal: string }
  | { type: 'ai_evaluated'; version: number; prompt: string }

export type CompletionVerdict = 'complete' | 'continue' | 'needs_review'

export interface CompletionContext {
  iterations: number
  signals: string[]
  // The model's verdict for an 'ai_evaluated' policy (the call is made outside this pure
  // core). Absent → the run isn't done deciding yet ('continue').
  aiVerdict?: CompletionVerdict
  // The policy version the RUN started with — a mismatch means the definition's policy was
  // edited mid-run, which forces human review rather than trusting a changed contract.
  runPolicyVersion?: number
}

export interface CompletionResult {
  verdict: CompletionVerdict
  reason: string
}

export function evaluateCompletion(policy: CompletionPolicy, ctx: CompletionContext): CompletionResult {
  // Versioning guard first: if the run began under a different policy version, don't apply
  // the changed policy silently — surface for review.
  if (ctx.runPolicyVersion != null && ctx.runPolicyVersion !== policy.version) {
    return { verdict: 'needs_review', reason: `completion policy changed mid-run (run v${ctx.runPolicyVersion} → def v${policy.version})` }
  }
  switch (policy.type) {
    case 'none':
      return { verdict: 'complete', reason: 'no completion policy — one pass and done' }
    case 'max_iterations':
      return ctx.iterations >= policy.max
        ? { verdict: 'complete', reason: `reached ${policy.max} iterations` }
        : { verdict: 'continue', reason: `iteration ${ctx.iterations}/${policy.max}` }
    case 'on_signal':
      return ctx.signals.includes(policy.signal)
        ? { verdict: 'complete', reason: `signal "${policy.signal}" observed` }
        : { verdict: 'continue', reason: `awaiting signal "${policy.signal}"` }
    case 'ai_evaluated':
      if (!ctx.aiVerdict) return { verdict: 'continue', reason: 'awaiting AI completion evaluation' }
      return { verdict: ctx.aiVerdict, reason: `AI evaluation: ${ctx.aiVerdict}` }
    default:
      return { verdict: 'needs_review', reason: 'unknown completion policy' }
  }
}

// The evaluation request a host builds for an 'ai_evaluated' policy (fed to a model; its
// answer comes back as CompletionContext.aiVerdict). Null for non-AI policies.
export function completionEvaluationRequest(policy: CompletionPolicy, transcriptExcerpt: string): { prompt: string } | null {
  if (policy.type !== 'ai_evaluated') return null
  return {
    prompt: `${policy.prompt}\n\nRespond with exactly one word — complete, continue, or needs_review — judging whether the automation's objective is met.\n\nRecent work:\n${transcriptExcerpt}`,
  }
}

// ---- 5. acknowledged-risk gate + permission snapshot ----

export interface RiskGateInput {
  requiredRisks: string[]
  acknowledgedRisks: string[]
}

export interface RiskGateResult {
  allowed: boolean
  missing: string[]
}

// A run is refused until every required risk is explicitly acknowledged (e.g. `local-checkout`
// runs edits on your real working copy).
export function gateAutomationRun(input: RiskGateInput): RiskGateResult {
  const acked = new Set(input.acknowledgedRisks)
  const missing = input.requiredRisks.filter((risk) => !acked.has(risk))
  return { allowed: missing.length === 0, missing }
}

// ---- 6. per-tick scheduler decision (composes the above) ----

// The minimal shape the tick needs from an automation (host maps its Automation onto this).
export interface SchedulableAutomation {
  id: string
  enabled: boolean
  manual: boolean
  everyMs: number
  lastRunAt: number | null
  nextRunAt?: number | null
  misfire: MisfirePolicy
  requiredRisks: string[]
  acknowledgedRisks: string[]
}

export interface AutomationTickInput {
  automations: SchedulableAutomation[]
  lease: SchedulerLease
  ownerId: string
  now: number
  leaseTtlMs: number
  // The host's live "already running this id" check — a dispatch never double-fires an
  // in-flight automation.
  isRunning: (id: string) => boolean
}

export interface AutomationTickResult {
  lease: SchedulerLease
  // Did THIS owner hold/take the scheduler lease this tick? false → another scheduler owns
  // it and we dispatch nothing (single-runner).
  acquired: boolean
  heldBy: string | null
  // Automation ids to dispatch now (misfire-coalesced: one entry even if several fires were
  // missed; run_each may repeat an id up to its catch-up cap).
  dispatch: string[]
  // Due automations refused by the risk gate, with the risks still needing acknowledgement.
  blocked: Array<{ id: string; missing: string[] }>
}

// One scheduler tick: acquire/renew the single-runner lease, then for each due automation
// apply the misfire policy and the risk gate. Pure — the host runs it on an interval and
// acts on `dispatch`/`blocked`. Standing down when another owner holds the lease is what
// stops a laptop AND a mini both firing the same automations.
export function automationTick(input: AutomationTickInput): AutomationTickResult {
  const lease = acquireLease(input.lease, input.ownerId, input.now, input.leaseTtlMs)
  if (!lease.ok) return { lease: input.lease, acquired: false, heldBy: lease.heldBy, dispatch: [], blocked: [] }

  const dispatch: string[] = []
  const blocked: Array<{ id: string; missing: string[] }> = []
  for (const automation of input.automations) {
    if (!automation.enabled || automation.manual) continue
    if (input.isRunning(automation.id)) continue
    const due = dueRuns({ enabled: automation.enabled, everyMs: automation.everyMs, lastRunAt: automation.lastRunAt, nextRunAt: automation.nextRunAt, misfire: automation.misfire }, input.now)
    if (due <= 0) continue
    const gate = gateAutomationRun({ requiredRisks: automation.requiredRisks, acknowledgedRisks: automation.acknowledgedRisks })
    if (!gate.allowed) { blocked.push({ id: automation.id, missing: gate.missing }); continue }
    for (let i = 0; i < due; i++) dispatch.push(automation.id)
  }
  return { lease: lease.lease, acquired: true, heldBy: input.ownerId, dispatch, blocked }
}

export interface PermissionSnapshotInput {
  approvalMode: string
  sandbox?: string | null
  worktreeMode?: string | null
  acknowledgedRisks?: string[]
  completionPolicyVersion?: number
}

// Freeze the safety-relevant posture at run start (a permission snapshot), so an
// in-flight run's guarantees can't be changed by a later edit to the definition.
export function snapshotAutomationPermissions(input: PermissionSnapshotInput, now: number) {
  return {
    approvalMode: input.approvalMode,
    sandbox: input.sandbox ?? null,
    worktreeMode: input.worktreeMode ?? 'auto',
    acknowledgedRisks: [...(input.acknowledgedRisks ?? [])].sort(),
    completionPolicyVersion: input.completionPolicyVersion ?? 0,
    capturedAt: now,
  }
}
