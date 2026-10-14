import assert from 'node:assert/strict'
import { engineBaseUrl, engineCurl, formatTokenRate } from './engine-workspace'
import type { EngineStatus } from './types'

const ready = { state: 'ready', model: 'Org/Model', port: 52814 } as EngineStatus
assert.equal(engineBaseUrl(ready), 'http://127.0.0.1:52814/v1')
for (const state of ['stopped', 'starting', 'failed'] as const) assert.equal(engineBaseUrl({ ...ready, state }), null)
for (const port of [0, -1, 65536, NaN, 12.5]) assert.equal(engineBaseUrl({ ...ready, port }), null)
assert.equal(engineBaseUrl(null), null)
assert.equal(formatTokenRate(null), '—')
assert.equal(formatTokenRate(NaN), '—')
assert.equal(formatTokenRate(0), '0')
assert.equal(formatTokenRate(212.51), '212.5')
const command = engineCurl({ baseUrl: 'http://127.0.0.1:52814/v1', model: "Org/Model's", apiKey: "fixture'key" })
assert.ok(command.includes('http://127.0.0.1:52814/v1/chat/completions'))
assert.ok(command.includes("fixture'\\''key"), 'quotes in credentials cannot end the shell argument')
assert.ok(command.includes('"stream":true'))
for (const baseUrl of ['https://example.com/v1', 'http://localhost:52814/v1', 'http://127.0.0.1:52814/v1?token=x', 'http://user@127.0.0.1:52814/v1']) {
  assert.throws(() => engineCurl({ baseUrl, model: 'Org/Model', apiKey: 'fixture' }))
}
console.log('engine workspace connection tests passed')
