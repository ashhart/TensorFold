// Run: jiti electron/agent/provider-status.test.ts
//
// Provider status persistence core (VST-037), main side. Codex rate-limit snapshot
// normalization, merge (replace + empty-refuse), key + lookup.
import assert from 'node:assert/strict'
import {
  codexSnapshotHasWindows,
  mergeProviderStatus,
  normalizeCodexSnapshot,
  providerStatusKey,
  providerStatusSnapshotFor,
  type PersistedProviderStatus,
} from './provider-status'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('provider-status tests:')

{
  assert.equal(providerStatusKey('codex-cli', null), 'codex-cli:default')
  assert.equal(providerStatusKey('codex-cli', 'acct-2'), 'codex-cli:acct-2')
  ok('providerStatusKey: provider + account (default fallback)')
}

{
  assert.equal(codexSnapshotHasWindows(null), false)
  assert.equal(codexSnapshotHasWindows({}), false)
  assert.equal(codexSnapshotHasWindows({ primary: { usedPercent: 40 } }), true)
  assert.equal(codexSnapshotHasWindows({ secondary: { usedPercent: 10 } }), true)
  assert.equal(codexSnapshotHasWindows({ primary: { usedPercent: NaN as unknown as number } }), false)
  ok('codexSnapshotHasWindows: at least one finite usedPercent')
}

{
  // Normalize keeps ONLY the fields we persist — a crafted payload can't smuggle extras.
  const clean = normalizeCodexSnapshot({
    primary: { usedPercent: 62, resetsAt: 1000, windowDurationMins: 300, injected: 'x' },
    secondary: { usedPercent: 30 },
    planType: 'plus',
    limitName: 'ChatGPT Plus',
    secret: 'nope',
  })
  assert.ok(clean)
  assert.deepEqual(clean!.primary, { usedPercent: 62, resetsAt: 1000, windowDurationMins: 300 })
  assert.deepEqual(clean!.secondary, { usedPercent: 30, resetsAt: null, windowDurationMins: null })
  assert.equal(clean!.planType, 'plus')
  assert.equal((clean as unknown as Record<string, unknown>).secret, undefined, 'extra keys are dropped')
  assert.equal(normalizeCodexSnapshot({ planType: 'plus' }), null, 'a snapshot with no windows normalizes to null')
  assert.equal(normalizeCodexSnapshot('garbage'), null)
  ok('normalizeCodexSnapshot: picked fields only, refuses window-less input')
}

{
  let persisted: PersistedProviderStatus | null = null
  persisted = mergeProviderStatus(persisted, {
    providerId: 'codex-cli', accountId: null,
    raw: { primary: { usedPercent: 20 } }, updatedAt: 1,
  })
  assert.equal(Object.keys(persisted.statuses).length, 1)
  assert.equal(providerStatusSnapshotFor(persisted, 'codex-cli', null)?.raw.primary?.usedPercent, 20)

  // A newer snapshot for the same key REPLACES the old one (rate limits are point-in-time).
  persisted = mergeProviderStatus(persisted, {
    providerId: 'codex-cli', accountId: null,
    raw: { primary: { usedPercent: 55 } }, updatedAt: 2,
  })
  assert.equal(providerStatusSnapshotFor(persisted, 'codex-cli', null)?.raw.primary?.usedPercent, 55)

  // An empty snapshot is refused — a transient blank read can't erase a good last-known.
  persisted = mergeProviderStatus(persisted, {
    providerId: 'codex-cli', accountId: null,
    raw: {}, updatedAt: 3,
  })
  assert.equal(providerStatusSnapshotFor(persisted, 'codex-cli', null)?.raw.primary?.usedPercent, 55, 'the good snapshot survives an empty update')

  // A different account is a separate slot.
  persisted = mergeProviderStatus(persisted, {
    providerId: 'codex-cli', accountId: 'work',
    raw: { primary: { usedPercent: 5 } }, updatedAt: 4,
  })
  assert.equal(Object.keys(persisted.statuses).length, 2)
  assert.equal(providerStatusSnapshotFor(persisted, 'codex-cli', 'work')?.raw.primary?.usedPercent, 5)
  assert.equal(providerStatusSnapshotFor(persisted, 'unknown', null), null)
  ok('mergeProviderStatus: replace-by-key, empty-refuse, per-account slots')
}

console.log(`  ${passed} passed`)
