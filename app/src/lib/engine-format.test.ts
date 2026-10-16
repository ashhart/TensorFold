import assert from 'node:assert/strict'
import { VERDICT_BADGES, canDownload, engineStateLabel, formatBytes, pullPercent } from './engine-format'
import type { EngineStatus, ModelPreflight, PullProgress } from './types'

const GIB = 1024 ** 3

assert.equal(formatBytes(1.6 * GIB), '1.6 GB')
assert.equal(formatBytes(512 * 1024 ** 2), '512 MB')
assert.equal(formatBytes(2048), '2 KB')
assert.equal(formatBytes(null), '?')

const pull = { repo: 'Org/A', revision: null, state: 'running', phase: '', receivedBytes: GIB, totalBytes: 4 * GIB, lastLine: null, message: null, startedAt: 0, finishedAt: null } satisfies PullProgress
assert.equal(pullPercent(pull), 25)
assert.equal(pullPercent({ ...pull, totalBytes: null }), null, 'no total means an indeterminate bar')
assert.equal(pullPercent({ ...pull, receivedBytes: 9 * GIB }), 100)

// Only a clean preflight (or a re-pull of an installed model) may start a download.
for (const verdict of Object.keys(VERDICT_BADGES) as ModelPreflight['verdict'][]) {
  assert.equal(canDownload({ verdict } as ModelPreflight), verdict === 'ok' || verdict === 'installed', verdict)
}
assert.equal(canDownload(null), false)
assert.equal(VERDICT_BADGES['too-big'].label, 'Too big for this Mac')
assert.equal(VERDICT_BADGES.gated.label, 'Gated')
assert.equal(VERDICT_BADGES.unsupported.label, 'Unsupported family')

const base: EngineStatus = { available: true, version: '1.0.7', binaryPath: '/x', state: 'stopped', model: null, drafter: null, port: null, startedAt: null, readyAt: null, message: null, logTail: [], totalMemoryBytes: 32 * GIB, memoryLimitBytes: 22 * GIB, contextTokens: null }
assert.equal(engineStateLabel(base), 'Engine 1.0.7 idle')
assert.equal(engineStateLabel({ ...base, state: 'ready', model: 'Org/A' }), 'Running Org/A')
assert.equal(engineStateLabel({ ...base, state: 'starting', model: 'Org/A', message: 'Loading Org/A' }), 'Loading Org/A')
assert.equal(engineStateLabel({ ...base, state: 'failed', message: 'The engine exited (code 1)' }), 'The engine exited (code 1)')
assert.equal(engineStateLabel({ ...base, available: false }), 'Engine missing from this build')
assert.equal(engineStateLabel(null), 'Checking engine')

console.log('engine format tests passed')
