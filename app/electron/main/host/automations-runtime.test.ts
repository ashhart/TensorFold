// Run: jiti electron/main/host/automations-runtime.test.ts
//
// Scheduler dispatch (VST-038 misfire policies): automationTick returns a repeated id per
// missed fire for `misfire: 'run_each'`; the host dispatch loop must run that whole batch
// (serialized per id so the runAutomation isRunning guard stays the double-fire protection),
// while `coalesce` still fires once no matter how many fires were missed.
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Isolate persistence to a temp dir so we touch neither ~/.vontra nor an Electron app.
process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'vontra-automations-runtime-'))

const { defaultSettings } = await import('../../store')
const { store } = await import('../state')
const { createAutomationRuntime } = await import('./automations')
import type { Automation, Job } from '../../agent/types'

function mkIntervalAutomation(p: Partial<Automation>): Automation {
  return {
    id: p.id || 'auto-1',
    name: p.name || 'Interval automation',
    prompt: p.prompt || 'summarize',
    scheduleKind: 'interval',
    everyMinutes: p.everyMinutes ?? 1,
    atHour: 9,
    atMinute: 0,
    providerId: null,
    providerAccountId: null,
    model: null,
    // Digest mode runs no model call, so the dispatch loop is exercised without a provider.
    mode: 'digest',
    cwd: null,
    deliverTo: 'inbox',
    enabled: true,
    lastRunAt: p.lastRunAt ?? null,
    nextRunAt: null,
    lastStatus: null,
    runs: [],
    misfire: p.misfire,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
}

store.settings = defaultSettings()

let jobSeq = 0
const runtimeDeps = {
  threadJobHeartbeatMs: 30_000,
  claimJob: () => ({ id: `job-${++jobSeq}` }) as unknown as Job,
  patchJob: (job: Job | null) => job,
  heartbeatJob: (job: Job | null) => job,
  finishJob: () => {},
  emitAutomation: () => {},
  effectiveContextLimit: () => 8_192,
  fullToolContext: () => ({}),
  applyEventToThread: () => {},
  timelineErrorSince: () => null,
  timelineError: () => new Error('timeline error'),
  isHighRiskTool: () => false,
  rememberConnectorEvents: () => {},
}
const runtime = createAutomationRuntime(runtimeDeps)

// Capture the scheduler interval callback so the test drives ticks deterministically.
let tickFn: (() => void) | null = null
const realSetInterval = globalThis.setInterval
globalThis.setInterval = ((fn: () => void) => {
  tickFn = fn
  return { unref() {} }
}) as unknown as typeof setInterval
try {
  runtime.startScheduler()
} finally {
  globalThis.setInterval = realSetInterval
}
assert.ok(tickFn, 'startScheduler registers the tick interval')
const tick = tickFn as unknown as () => void

async function waitFor(cond: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 500; i++) {
    if (cond()) return
    await new Promise((resolve) => setTimeout(resolve, 10))
  }
  throw new Error(`timed out waiting for: ${label}`)
}

// run_each: three missed fires -> three sequential catch-up runs in one tick.
const runEach = mkIntervalAutomation({ id: 'auto-run-each', misfire: 'run_each', lastRunAt: Date.now() - 190_000 })
// coalesce: three missed fires -> exactly one run.
const coalesce = mkIntervalAutomation({ id: 'auto-coalesce', misfire: 'coalesce', lastRunAt: Date.now() - 190_000 })
store.automations = [runEach, coalesce]

tick()
await waitFor(
  () => runEach.runs.length === 3 && coalesce.runs.length === 1 && [...runEach.runs, ...coalesce.runs].every((r) => r.status === 'ok'),
  'run_each catch-up batch and single coalesce run',
)
assert.equal(runEach.runs.length, 3, 'run_each dispatches one run per missed fire')
assert.equal(coalesce.runs.length, 1, 'coalesce still collapses missed fires into one run')
assert.ok(runEach.lastRunAt != null && runEach.lastRunAt > Date.now() - 60_000, 'catch-up advances lastRunAt')

// A follow-up tick dispatches nothing: lastRunAt is current, so no fires are missed.
tick()
await new Promise((resolve) => setTimeout(resolve, 100))
assert.equal(runEach.runs.length, 3, 'no further run_each dispatches once caught up')
assert.equal(coalesce.runs.length, 1, 'no further coalesce dispatches once caught up')

// A renderer-selected timezone must survive creation and be interpreted by the scheduler,
// even when the scheduler host uses a different local timezone.
const previousTimeZone = process.env.TZ
process.env.TZ = 'UTC'
try {
  const daily = runtime.mkAutomation({ scheduleKind: 'daily', atHour: 9, atMinute: 0, enabled: true, timeZone: 'America/New_York' } as Partial<Automation>)
  assert.equal((daily as Automation & { timeZone?: string }).timeZone, 'America/New_York', 'creation preserves the explicit timezone')
  assert.equal(runtime.computeNextRun(daily, Date.parse('2026-03-07T12:00:00Z')), Date.parse('2026-03-07T14:00:00Z'), 'the scheduler uses New York wall time rather than UTC host time')
  const invalidDaily = { ...daily, id: 'invalid-daily', timeZone: 'Europe/Unknown', nextRunAt: Date.now() - 86_400_000, runs: [] as Automation['runs'] }
  const overdueDaily = { ...daily, id: 'overdue-daily', timeZone: 'UTC', mode: 'digest' as const, nextRunAt: Date.now() - 86_400_000, runs: [] as Automation['runs'] }
  const committedOverdue = overdueDaily.nextRunAt
  store.automations = [invalidDaily, overdueDaily]
  const recoveredRuntime = createAutomationRuntime(runtimeDeps)
  globalThis.setInterval = ((fn: () => void) => { tickFn = fn; return { unref() {} } }) as unknown as typeof setInterval
  try { recoveredRuntime.startScheduler() } finally { globalThis.setInterval = realSetInterval }
  assert.match(invalidDaily.scheduleError ?? '', /IANA timezone/, 'an invalid persisted zone becomes an actionable diagnostic')
  assert.equal(invalidDaily.nextRunAt, null, 'an invalid persisted zone never falls back to host-local scheduling')
  assert.equal(overdueDaily.nextRunAt, committedOverdue, 'restart preserves an overdue committed instant for one coalesced run')
  const recoveredTick = tickFn as unknown as () => void
  recoveredTick(); recoveredTick()
  await waitFor(() => overdueDaily.runs.length === 1 && overdueDaily.runs[0].status === 'ok', 'one coalesced daily run after restart')
  recoveredTick()
  await new Promise((resolve) => setTimeout(resolve, 20))
  assert.equal(overdueDaily.runs.length, 1, 'repeated ticks cannot dispatch a daily run twice')
  assert.equal(invalidDaily.runs.length, 0, 'an invalid persisted zone is never dispatched')
  assert.ok(overdueDaily.nextRunAt! > Date.now(), 'completion advances the pinned daily schedule')
} finally {
  if (previousTimeZone === undefined) delete process.env.TZ
  else process.env.TZ = previousTimeZone
}

// Never-run interval scheduling must agree with the next-run value displayed by the UI.
// Exercise the real host mapping/tick with inert digest work and isolated persistence.
const realNow = Date.now
let clock = Date.UTC(2026, 9, 14, 9)
Date.now = () => clock
try {
  const first = runtime.mkAutomation({ scheduleKind: 'interval', everyMinutes: 60, enabled: true, mode: 'digest' })
  first.nextRunAt = runtime.computeNextRun(first)
  assert.equal(first.nextRunAt, clock + 3_600_000, 'creation displays one full interval before the first run')
  const legacy = { ...first, id: 'legacy-no-first-instant', nextRunAt: null, runs: [] as Automation['runs'] }
  store.automations = [first, legacy]
  const firstRuntime = createAutomationRuntime(runtimeDeps)
  globalThis.setInterval = ((fn: () => void) => { tickFn = fn; return { unref() {} } }) as unknown as typeof setInterval
  try { firstRuntime.startScheduler() } finally { globalThis.setInterval = realSetInterval }
  const firstTick = tickFn as unknown as () => void
  const committed = first.nextRunAt!
  assert.equal(legacy.nextRunAt, committed, 'legacy missing instants initialize to the same delayed first-run contract')
  firstTick()
  assert.equal(first.runs.length, 0, 'the real scheduler does not dispatch a new interval immediately')
  assert.equal(legacy.runs.length, 0, 'legacy initialization also avoids an immediate implicit run')
  clock = committed - 1
  firstTick()
  assert.equal(first.runs.length, 0, 'the displayed first-fire instant is respected until due')
  clock = committed
  firstTick(); firstTick()
  await waitFor(() => first.runs.length === 1 && first.runs[0].status === 'ok' && legacy.runs.length === 1 && legacy.runs[0].status === 'ok', 'exactly one first run at the displayed instant')
  assert.equal(first.runs.length, 1)
  assert.equal(first.nextRunAt, committed + 3_600_000, 'completion shows the next full interval')

  const overdueFirst = { ...first, id: 'restart-before-first', lastRunAt: null, nextRunAt: clock - 3_600_000, runs: [] as Automation['runs'] }
  const overdueInstant = overdueFirst.nextRunAt
  store.automations = [overdueFirst]
  const restarted = createAutomationRuntime(runtimeDeps)
  globalThis.setInterval = ((fn: () => void) => { tickFn = fn; return { unref() {} } }) as unknown as typeof setInterval
  try { restarted.startScheduler() } finally { globalThis.setInterval = realSetInterval }
  assert.equal(overdueFirst.nextRunAt, overdueInstant, 'restart preserves an overdue committed first instant instead of postponing it')
  const restartTick = tickFn as unknown as () => void
  restartTick(); restartTick()
  await waitFor(() => overdueFirst.runs.length === 1 && overdueFirst.runs[0].status === 'ok', 'one coalesced first run after restart')
  assert.equal(overdueFirst.runs.length, 1)

  const immediate = runtime.mkAutomation({ scheduleKind: 'interval', everyMinutes: 60, enabled: true, mode: 'digest' })
  immediate.nextRunAt = runtime.computeNextRun(immediate)
  await firstRuntime.runAutomation(immediate, { trigger: 'manual' })
  assert.equal(immediate.runs.length, 1, 'explicit Run now remains immediate before the scheduled first fire')
  assert.equal(immediate.runs[0].status, 'ok')
} finally {
  Date.now = realNow
}

console.log('automations runtime dispatch tests passed')
