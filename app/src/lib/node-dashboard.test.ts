// Run: jiti src/lib/node-dashboard.test.ts
//
// Node & inference dashboard (VST-014): health matrix for fake nodes (online / degraded /
// offline), token usage aggregation, host disconnect, evidence-gated provider health.
import assert from 'node:assert/strict'
import { buildNodeDashboard, formatEvidenceAge, rollupUsage } from './node-dashboard'
import type { TokenUsageSnapshot } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('node-dashboard tests:')

const NOW = 1_000_000
const WORK = {
  jobs: { total: 3, visible: 3, queued: 1, running: 2, waitingApproval: 0, paused: 0, stalled: 0, failed: 0, deadLetter: 0, retryable: 0 },
  leases: { active: 2, expired: 0, expiringSoon: 0, missingRunning: 0 },
  approvals: { pending: 1, tool: 1, connector: 0, mission: 0 },
  nextWakeAt: null,
  nextWakeJobId: null,
  generatedAt: NOW,
  terminals: { total: 2, running: 1, attention: 1 },
} as never

const LOCAL = { activeJobs: 1, pendingApprovals: 0, terminals: { total: 1, running: 1, attention: 0 } }

{
  const dashboard = buildNodeDashboard({
    now: NOW,
    local: LOCAL,
    remoteStaleAfterMs: 60_000,
    remoteHosts: [
      { hostId: 'mini', name: 'Mac mini', ok: true, generatedAt: NOW - 5_000, latencyMs: 12, work: WORK },
      { hostId: 'attic', name: 'Attic box', ok: true, generatedAt: NOW - 300_000 },
      { hostId: 'gone', name: 'Old laptop', ok: false, error: 'connect ECONNREFUSED' },
    ],
  })
  const byId = new Map(dashboard.nodes.map((node) => [node.id, node]))
  assert.equal(byId.get('node:local')?.health, 'online')

  const fresh = byId.get('node:remote:mini')!
  assert.equal(fresh.health, 'online')
  assert.equal(fresh.latencyMs, 12)
  assert.equal(fresh.activeJobs, 3, 'running + queued')
  assert.equal(fresh.pendingApprovals, 1)
  assert.deepEqual(fresh.terminals, { total: 2, running: 1, attention: 1 })
  assert.match(byId.get('node:local')?.detail ?? '', /1 active of 1 terminal/)

  const stale = byId.get('node:remote:attic')!
  assert.equal(stale.health, 'degraded')
  assert.match(stale.detail, /stale \(300s old\)/)

  const dead = byId.get('node:remote:gone')!
  assert.equal(dead.health, 'offline')
  assert.match(dead.lastError ?? '', /ECONNREFUSED/)
  assert.ok(dashboard.failures.some((f) => f.includes('Old laptop')), 'a disconnected host lands on the failure strip')
  assert.ok(dashboard.failures.some((f) => f.includes('Attic box')), 'stale heartbeats do too')
  ok('hosts: fresh → online with work counts; stale → degraded; unreachable → offline')
}

{
  const dashboard = buildNodeDashboard({
    now: NOW,
    local: LOCAL,
    providers: [
      { id: 'openai', name: 'OpenAI', kind: 'cloud', apiKey: 'k', models: [] } as never,
      { id: 'zai', name: 'z.ai', kind: 'cloud', apiKey: 'k', models: [] } as never,
      { id: 'codex-cli', name: 'Codex CLI', kind: 'cloud', transport: 'codex-cli', models: [] } as never,
    ],
    providerStatus: {
      version: 1,
      statuses: {
        'openai:default': { providerId: 'openai', accountId: null, raw: { primary: { used_percent: 42 } }, updatedAt: NOW - 1000 },
        'zai:default': { providerId: 'zai', accountId: null, raw: { weekly: { used_percent: 100 } }, updatedAt: NOW - 1000 },
      },
    } as never,
    cliTools: [
      { id: 'codex', name: 'Codex', installed: true, version: '0.142.5', providerIds: ['codex-cli'], builtin: true, effectiveCommand: 'codex' } as never,
      { id: 'gemini', name: 'Gemini CLI', installed: false, providerIds: ['gemini-cli'], builtin: true, effectiveCommand: null } as never,
    ],
  })
  const byId = new Map(dashboard.nodes.map((node) => [node.id, node]))
  assert.equal(byId.get('node:provider:openai')?.health, 'online', 'live telemetry → online')
  assert.equal(byId.get('node:provider:zai')?.health, 'degraded', 'an exhausted window degrades the provider')
  assert.equal(byId.get('node:provider:codex-cli'), undefined, 'CLI-transport providers card as CLI nodes, not cloud ones')
  assert.equal(byId.get('node:cli:codex')?.health, 'online')
  assert.match(byId.get('node:cli:codex')?.detail ?? '', /0\.142\.5/)
  assert.equal(byId.get('node:cli:gemini')?.health, 'offline')
  assert.ok(dashboard.failures.some((f) => f.includes('Gemini CLI')))
  ok('providers/CLIs: telemetry-gated health, exhausted windows degrade, missing binaries offline')
}

{
  const usage = rollupUsage({
    generatedAt: NOW,
    latestAt: NOW - 50,
    active: true,
    windows: {} as never,
    daily: [
      { key: 'd1', ts: 1, total: 1000, accounts: { 'openai:default': 600, 'codex:work': 400 } },
      { key: 'd2', ts: 2, total: 500, accounts: { 'openai:default': 500 } },
    ],
    hourly: [],
  } as unknown as TokenUsageSnapshot)
  assert.equal(usage.totalTokens, 1500)
  assert.deepEqual(usage.byAccount, { 'openai:default': 1100, 'codex:work': 400 })
  assert.equal(usage.latestAt, NOW - 50)
  assert.deepEqual(rollupUsage(null), { totalTokens: 0, byAccount: {}, latestAt: null })
  ok('usage: totals and per-account aggregation across daily points')
}

assert.equal(formatEvidenceAge(4_000), 'live now')
assert.equal(formatEvidenceAge(65_000), '1m ago')
assert.equal(formatEvidenceAge(3_600_000), '1h ago')

console.log(`  ${passed} passed`)
