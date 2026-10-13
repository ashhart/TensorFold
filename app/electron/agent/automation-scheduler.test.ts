// Run: jiti electron/agent/automation-scheduler.test.ts
//
// Industrial automation scheduler kernel (VST-038): lease contention, misfire coalescing,
// retry backoff, completion-policy verdicts (incl. versioning + AI-evaluated), risk gate.
// Pure — injected `now`, no timers.
import assert from 'node:assert/strict'
import {
  EMPTY_LEASE,
  leaseHeldBy,
  acquireLease,
  renewLease,
  missedFireCount,
  dueRuns,
  nextRunAt,
  nextRetry,
  evaluateCompletion,
  completionEvaluationRequest,
  gateAutomationRun,
  snapshotAutomationPermissions,
  automationTick,
  type AutomationTiming,
  type CompletionPolicy,
  type SchedulableAutomation,
} from './automation-scheduler'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('automation-scheduler tests:')

// --- 1. lease contention (single-runner) ---
{
  const t0 = 1_000_000
  const ttl = 30_000
  const a = acquireLease(EMPTY_LEASE, 'laptop', t0, ttl)
  assert.equal(a.ok, true)
  assert.equal(a.heldBy, 'laptop')

  // The mini tries while the laptop's lease is live → refused (stands down).
  const b = acquireLease(a.lease, 'mini', t0 + 5_000, ttl)
  assert.equal(b.ok, false, 'a second scheduler cannot acquire a live lease')
  assert.equal(b.heldBy, 'laptop')

  // The holder renews.
  const r = renewLease(a.lease, 'laptop', t0 + 10_000, ttl)
  assert.equal(r.ok, true)
  assert.equal(r.lease.expiresAt, t0 + 10_000 + ttl, 'renew extends expiry')

  // The laptop crashes; after expiry the mini takes over.
  const afterExpiry = t0 + 10_000 + ttl + 1
  assert.equal(leaseHeldBy(r.lease, afterExpiry), null, 'an expired lease is free')
  const takeover = acquireLease(r.lease, 'mini', afterExpiry, ttl)
  assert.equal(takeover.ok, true, 'the mini takes over an expired lease')
  assert.equal(takeover.heldBy, 'mini')
  // The laptop (now stale) can no longer renew.
  assert.equal(renewLease(takeover.lease, 'laptop', afterExpiry + 100, ttl).ok, false, 'the crashed scheduler cannot renew after takeover')
  ok('lease: single holder, refusal under contention, expiry takeover, no stale renew')
}

// --- 2. misfire coalescing (missed-schedule) ---
{
  const every = 60_000 // 1 min
  const base: AutomationTiming = { enabled: true, everyMs: every, lastRunAt: 1_000_000, misfire: 'coalesce' }

  assert.equal(missedFireCount(base, 1_000_000 + 30_000), 0, 'not due before the interval')
  assert.equal(dueRuns(base, 1_000_000 + 30_000), 0)

  // Host asleep 5 minutes → 5 fires missed.
  const asleep = 1_000_000 + 5 * every + 1
  assert.equal(missedFireCount(base, asleep), 5, 'counts every missed fire')
  assert.equal(dueRuns(base, asleep), 1, 'coalesce collapses 5 missed fires into ONE run')

  // run_each dispatches each, capped.
  const runEach: AutomationTiming = { ...base, misfire: 'run_each', maxCatchup: 3 }
  assert.equal(dueRuns(runEach, asleep), 3, 'run_each dispatches each missed fire up to maxCatchup')

  // A never-run automation is due once.
  assert.equal(dueRuns({ enabled: true, everyMs: every, lastRunAt: null, misfire: 'coalesce' }, 1_000_000), 1)
  // Disabled never fires.
  assert.equal(dueRuns({ ...base, enabled: false }, asleep), 0)
  // nextRunAt walks forward past the missed window.
  assert.equal(nextRunAt(base, asleep), 1_000_000 + 5 * every)
  ok('misfire: coalesce → 1, run_each → capped N, disabled → 0')
}

// --- 3. retry backoff ---
{
  const policy = { maxAttempts: 4, backoffMs: 1_000, factor: 2, maxBackoffMs: 5_000 }
  assert.deepEqual(nextRetry(1, policy), { retry: true, delayMs: 1_000 })
  assert.deepEqual(nextRetry(2, policy), { retry: true, delayMs: 2_000 })
  assert.deepEqual(nextRetry(3, policy), { retry: true, delayMs: 4_000 })
  assert.deepEqual(nextRetry(4, policy), { retry: false, delayMs: 0 }, 'gives up at maxAttempts')
  assert.equal(nextRetry(10, { maxAttempts: 99, backoffMs: 1_000, factor: 3, maxBackoffMs: 5_000 }).delayMs, 5_000, 'backoff is capped')
  ok('nextRetry: exponential + capped, stops at maxAttempts')
}

// --- 4. completion-policy verdicts (+ versioning + AI-evaluated) ---
{
  assert.equal(evaluateCompletion({ type: 'none', version: 1 }, { iterations: 1, signals: [] }).verdict, 'complete')

  const maxIter: CompletionPolicy = { type: 'max_iterations', version: 1, max: 3 }
  assert.equal(evaluateCompletion(maxIter, { iterations: 2, signals: [] }).verdict, 'continue')
  assert.equal(evaluateCompletion(maxIter, { iterations: 3, signals: [] }).verdict, 'complete')

  const onSignal: CompletionPolicy = { type: 'on_signal', version: 1, signal: 'tests_green' }
  assert.equal(evaluateCompletion(onSignal, { iterations: 9, signals: [] }).verdict, 'continue')
  assert.equal(evaluateCompletion(onSignal, { iterations: 1, signals: ['tests_green'] }).verdict, 'complete')

  const ai: CompletionPolicy = { type: 'ai_evaluated', version: 2, prompt: 'Is the refactor done?' }
  assert.equal(evaluateCompletion(ai, { iterations: 1, signals: [] }).verdict, 'continue', 'no AI verdict yet → continue')
  assert.equal(evaluateCompletion(ai, { iterations: 1, signals: [], aiVerdict: 'complete' }).verdict, 'complete')
  assert.equal(evaluateCompletion(ai, { iterations: 1, signals: [], aiVerdict: 'needs_review' }).verdict, 'needs_review')

  // Versioning: a run that started under v1 must not silently obey a def edited to v2.
  const changed = evaluateCompletion(maxIter, { iterations: 5, signals: [], runPolicyVersion: 99 })
  assert.equal(changed.verdict, 'needs_review', 'a mid-run policy version change forces review')
  assert.match(changed.reason, /changed mid-run/)

  // The AI request is only built for ai_evaluated policies.
  assert.equal(completionEvaluationRequest(maxIter, 'work'), null)
  const req = completionEvaluationRequest(ai, 'edited auth.ts; tests pass')
  assert.ok(req && req.prompt.includes('Is the refactor done?') && req.prompt.includes('complete, continue, or needs_review'))
  ok('completion: none/max/signal/AI verdicts + mid-run version guard + AI request')
}

// --- 5. acknowledged-risk gate ---
{
  const blocked = gateAutomationRun({ requiredRisks: ['local-checkout', 'network'], acknowledgedRisks: ['network'] })
  assert.equal(blocked.allowed, false, 'an unacknowledged risk blocks the run')
  assert.deepEqual(blocked.missing, ['local-checkout'])

  const allowed = gateAutomationRun({ requiredRisks: ['local-checkout'], acknowledgedRisks: ['local-checkout', 'network'] })
  assert.equal(allowed.allowed, true)
  assert.deepEqual(allowed.missing, [])

  assert.equal(gateAutomationRun({ requiredRisks: [], acknowledgedRisks: [] }).allowed, true, 'no required risks → allowed')
  ok('risk gate: refuses until every required risk is acknowledged')
}

// --- permission snapshot freezes posture ---
{
  const snap = snapshotAutomationPermissions({ approvalMode: 'suggest', sandbox: 'workspace-write', worktreeMode: 'create-worktree', acknowledgedRisks: ['network', 'local-checkout'], completionPolicyVersion: 3 }, 5_000)
  assert.equal(snap.approvalMode, 'suggest')
  assert.equal(snap.sandbox, 'workspace-write')
  assert.deepEqual(snap.acknowledgedRisks, ['local-checkout', 'network'], 'risks are sorted for stable comparison')
  assert.equal(snap.completionPolicyVersion, 3)
  assert.equal(snap.capturedAt, 5_000)
  assert.equal(snapshotAutomationPermissions({ approvalMode: 'auto' }, 1).worktreeMode, 'auto', 'worktree mode defaults to auto')
  ok('permission snapshot: freezes approval/sandbox/worktree/risks/policy-version at run start')
}

// --- 6. scheduler tick (composes lease + misfire + risk gate) ---
{
  const every = 60_000
  const auto = (over: Partial<SchedulableAutomation> = {}): SchedulableAutomation => ({
    id: 'a1', enabled: true, manual: false, everyMs: every, lastRunAt: 1_000_000, misfire: 'coalesce', requiredRisks: [], acknowledgedRisks: [], ...over,
  })
  const asleep = 1_000_000 + 5 * every + 1
  const none = () => false

  // Holds the lease → dispatches the due automation once (coalesced), advances the lease.
  const t1 = automationTick({ automations: [auto()], lease: EMPTY_LEASE, ownerId: 'mini', now: asleep, leaseTtlMs: 30_000, isRunning: none })
  assert.equal(t1.acquired, true)
  assert.deepEqual(t1.dispatch, ['a1'], 'a due automation dispatches once despite 5 missed fires (coalesce)')
  assert.equal(t1.heldBy, 'mini')

  // A DIFFERENT owner tries against the live lease → stands down, dispatches nothing.
  const t2 = automationTick({ automations: [auto()], lease: t1.lease, ownerId: 'laptop', now: asleep + 1_000, leaseTtlMs: 30_000, isRunning: none })
  assert.equal(t2.acquired, false, 'the second scheduler does not hold the lease')
  assert.deepEqual(t2.dispatch, [], 'and dispatches nothing (single-runner)')
  assert.equal(t2.heldBy, 'mini')

  // Already-running automation is not re-dispatched.
  const t3 = automationTick({ automations: [auto()], lease: t1.lease, ownerId: 'mini', now: asleep + 1_000, leaseTtlMs: 30_000, isRunning: (id) => id === 'a1' })
  assert.deepEqual(t3.dispatch, [], 'an in-flight automation is never double-fired')

  // Risk gate: a due automation with an unacknowledged required risk is blocked, not run.
  const risky = auto({ id: 'risky', requiredRisks: ['local-checkout'], acknowledgedRisks: [] })
  const t4 = automationTick({ automations: [risky], lease: EMPTY_LEASE, ownerId: 'mini', now: asleep, leaseTtlMs: 30_000, isRunning: none })
  assert.deepEqual(t4.dispatch, [], 'a risk-gated automation is not dispatched')
  assert.deepEqual(t4.blocked, [{ id: 'risky', missing: ['local-checkout'] }])
  // Acknowledge it → now it dispatches.
  const t5 = automationTick({ automations: [auto({ id: 'risky', requiredRisks: ['local-checkout'], acknowledgedRisks: ['local-checkout'] })], lease: EMPTY_LEASE, ownerId: 'mini', now: asleep, leaseTtlMs: 30_000, isRunning: none })
  assert.deepEqual(t5.dispatch, ['risky'], 'acknowledging the risk unblocks dispatch')

  // Not-yet-due + manual + disabled all dispatch nothing.
  const t6 = automationTick({
    automations: [auto({ id: 'early', lastRunAt: asleep }), auto({ id: 'man', manual: true }), auto({ id: 'off', enabled: false })],
    lease: EMPTY_LEASE, ownerId: 'mini', now: asleep, leaseTtlMs: 30_000, isRunning: none,
  })
  assert.deepEqual(t6.dispatch, [], 'not-due / manual / disabled automations are skipped')

  // run_each catches up (capped by the kernel default) when the host was long asleep.
  const t7 = automationTick({ automations: [auto({ id: 'each', misfire: 'run_each' })], lease: EMPTY_LEASE, ownerId: 'mini', now: asleep, leaseTtlMs: 30_000, isRunning: none })
  assert.equal(t7.dispatch.filter((id) => id === 'each').length, 5, 'run_each dispatches each missed fire')
  ok('automationTick: lease stand-down, coalesced dispatch, in-flight skip, risk gate, run_each')
}

console.log(`\n${passed} passed`)
