import assert from 'node:assert/strict'
import { setEngineServeHook } from '../engine/provider'
import { runQualificationProbe } from './probe'

async function main() {
  const saved = globalThis.fetch
  let starts: boolean[] = [], releases = 0, requests = 0
  setEngineServeHook(async (model, options) => { starts.push(options.start); assert.equal(model, 'synthetic-native'); return { model, baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', release: () => { releases++ } } })
  globalThis.fetch = (async (_input, init) => { requests++; assert.deepEqual(JSON.parse(String(init?.body)).messages, [{ role: 'user', content: 'Reply with OK.' }]); return new Response('data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } }) }) as typeof fetch
  try {
    const result = await runQualificationProbe({ id: 'tensorfold', source: 'tensorfold', name: 'Synthetic native', kind: 'local', baseUrl: 'http://127.0.0.1/v1', apiKey: '', models: ['synthetic-native'] }, 'synthetic-native', 'text', { signal: new AbortController().signal, assertCurrent() {}, timeoutMs: 100 })
    assert.equal(result.state, 'observed')
    assert.deepEqual(starts, [false], 'Native qualification must never acquire an automatic-start request')
    assert.equal(releases, 1)
    assert.equal(requests, 1)
    console.log('PASS actual native qualification borrows one loaded-only lease with start:false, sends only the fixed fixture and releases its lease')
  } finally { globalThis.fetch = saved; setEngineServeHook(null) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
