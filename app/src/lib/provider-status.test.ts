// Run: jiti src/lib/provider-status.test.ts
//
// Provider status VIEWS (VST-037), renderer side. Codex percent-window → UsageWindow, pace
// derivation, staleness drop, roll-up tone.
import assert from 'node:assert/strict'
import {
  codexRateLimitsToWindows,
  codexWindowToUsageWindow,
  providerStatusView,
  providerStatusViewFor,
} from './provider-status'
import type { PersistedProviderStatus } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('provider-status view tests:')

const NOW = 1_000_000_000_000

{
  const w = codexWindowToUsageWindow({ usedPercent: 62, resetsAt: NOW + 90 * 60_000, windowDurationMins: 300 }, 'codex:session', 'Session', 'session')
  assert.ok(w)
  assert.equal(w!.used, 62)
  assert.equal(w!.limit, 100)
  assert.equal(w!.resetsAt, NOW + 90 * 60_000)
  // startedAt = resetsAt − duration → gives usage-windows a pace reference.
  assert.equal(w!.startedAt, NOW + 90 * 60_000 - 300 * 60_000)
  assert.equal(codexWindowToUsageWindow({ usedPercent: NaN as unknown as number }, 'x', 'X', 'session'), null)
  assert.equal(codexWindowToUsageWindow(null, 'x', 'X', 'session'), null)
  ok('codexWindowToUsageWindow: percent→count, startedAt from reset−duration')
}

{
  const windows = codexRateLimitsToWindows({
    primary: { usedPercent: 62, resetsAt: NOW + 60_000, windowDurationMins: 300 },
    secondary: { usedPercent: 30, resetsAt: NOW + 4 * 24 * 60 * 60_000, windowDurationMins: 7 * 24 * 60 },
  })
  assert.equal(windows.length, 2)
  assert.equal(windows[0].kind, 'session', 'session orders first')
  assert.equal(windows[1].kind, 'weekly')
  assert.deepEqual(codexRateLimitsToWindows(null), [])
  assert.deepEqual(codexRateLimitsToWindows({}), [])
  ok('codexRateLimitsToWindows: primary=session, secondary=weekly, ordered')
}

{
  // A fresh snapshot with a mostly-spent, nearly-reset session window reads as pressured.
  const view = providerStatusView({
    providerId: 'codex-cli', accountId: null,
    raw: {
      primary: { usedPercent: 95, resetsAt: NOW + 30 * 60_000, windowDurationMins: 300 },
      planType: 'plus',
    },
    updatedAt: NOW,
  }, NOW)
  assert.ok(view)
  assert.equal(view!.planType, 'plus')
  assert.equal(view!.windows.length, 1)
  assert.equal(view!.windows[0].usedPercent, 95)
  assert.ok(view!.windows[0].etaText, 'a reset ETA is rendered')
  assert.ok(['warn', 'critical'].includes(view!.tone), '95% used, well past even pace → warn/critical tone')
  ok('providerStatusView: real pace + ETA + rolled-up tone')
}

{
  // A snapshot older than the staleness window is dropped — not shown as current truth.
  const stale = providerStatusView({
    providerId: 'codex-cli', accountId: null,
    raw: { primary: { usedPercent: 50 } },
    updatedAt: NOW - 7 * 60 * 60_000, // 7h ago, past the 6h default
  }, NOW)
  assert.equal(stale, null)
  // A window-less snapshot yields no view.
  assert.equal(providerStatusView({ providerId: 'x', accountId: null, raw: {}, updatedAt: NOW }, NOW), null)
  ok('providerStatusView: stale + empty snapshots produce no view')
}

{
  const persisted: PersistedProviderStatus = {
    version: 1,
    statuses: {
      'codex-cli:default': {
        providerId: 'codex-cli', accountId: null,
        raw: { primary: { usedPercent: 40, resetsAt: NOW + 60_000, windowDurationMins: 300 } },
        updatedAt: NOW,
      },
    },
  }
  assert.ok(providerStatusViewFor(persisted, 'codex-cli', null, NOW))
  assert.equal(providerStatusViewFor(persisted, 'codex-cli', 'other', NOW), null)
  assert.equal(providerStatusViewFor(null, 'codex-cli', null, NOW), null)
  ok('providerStatusViewFor: keyed lookup from the persisted map')
}

console.log(`  ${passed} passed`)
