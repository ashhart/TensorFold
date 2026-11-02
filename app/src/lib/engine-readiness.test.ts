import assert from 'node:assert/strict'
import { engineReadiness } from './engine-readiness'
import type { EngineSnapshot } from './types'

const snapshot = {
  status: { available: true, state: 'ready', model: 'fixture/model', contextTokens: 32768, memoryLimitBytes: 100 },
  installed: [{ repo: 'fixture/model', memoryFloorBytes: 40 }],
  performance: { model: 'fixture/model', decodeTokensPerSecond: 55, measuredAt: 123 },
} as EngineSnapshot
assert.deepEqual(engineReadiness(snapshot), {
  ready: true, context: 32768, budget: 100, floor: 40, floorPercent: 40, decode: 55, measuredAt: 123,
})
assert.equal(engineReadiness({ ...snapshot, status: { ...snapshot.status, state: 'starting' } }).decode, null)
assert.equal(engineReadiness({ ...snapshot, status: { ...snapshot.status, state: 'failed' } }).context, null)
assert.equal(engineReadiness({ ...snapshot, status: { ...snapshot.status, available: false } }).ready, false)
assert.equal(engineReadiness({ ...snapshot, performance: { ...snapshot.performance!, model: 'fixture/other' } }).decode, null)
assert.equal(engineReadiness({ ...snapshot, status: { ...snapshot.status, memoryLimitBytes: 0 } }).floorPercent, null)
assert.equal(engineReadiness(null).ready, false)
console.log('Engine readiness never presents stale measurements, unknown budgets or loading context as ready.')
