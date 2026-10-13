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
const runtime = createAutomationRuntime({
  threadJobHeartbeatMs: 30_000,
  claimJob: () => ({ id: `job-${++jobSeq}` }) as unknown as Job,
  patchJob: (job) => job,
  heartbeatJob: (job) => job,
  finishJob: () => {},
  emitAutomation: () => {},
  effectiveContextLimit: () => 8_192,
  fullToolContext: () => ({}),
  applyEventToThread: () => {},
  timelineErrorSince: () => null,
  timelineError: () => new Error('timeline error'),
  isHighRiskTool: () => false,
  rememberConnectorEvents: () => {},
})

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

console.log('automations runtime dispatch tests passed')
