import assert from 'node:assert/strict'
import { EngineService } from './service'
import { makeHarness, seedModel } from './test-fixtures/harness'
import { measureModelPreview } from './model-preview'
import { defaultEngineModelProfile } from '../../../shared/model-profiles'
import type { StreamChunk } from '../openai'

const model = 'Fixture/Preview', h = await makeHarness({ FAKE_REPLY_TIMINGS: '1' })
const generation = { ...defaultEngineModelProfile(8192), temperature: 0.7, maxOutputTokens: 1024, speculativeDecoding: 'off' as const }
const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 15, emit: () => {}, modelSettings: () => generation })
try {
  await seedModel(h.hub, model)
  service.install()
  const result = await service.tryModel(model)
  assert.equal(result.model, model)
  assert.equal(result.completionTokens, 3)
  assert.equal(result.decodeTokensPerSecond, 200)
  assert.equal(result.firstTokenMs, 50)
  assert.equal(result.cachedTokens, 1)
  assert.equal(result.peakMemoryBytes, null, 'no memory measurement is invented')
  assert.equal((await service.snapshot()).performance, null, 'preview does not become chat throughput')
  const request = (await h.records()).find((entry) => entry.kind === 'chat')
  assert.equal(request.max_tokens, 96)
  assert.equal(request.temperature, 0.7)
  assert.equal(request.draft, false)
  assert.deepEqual(request.chat_template_kwargs, { enable_thinking: false })
  assert.equal(request.messages, 1)
  assert.equal(request.tools, undefined)
  assert.equal(service.supervisor.status().contextTokens, 8192, 'the served context reflects the actual profile launch flag')
  const lease = await service.supervisor.acquire(model, { start: false })
  await assert.rejects(service.tryModel(model), /active local requests/)
  lease.release()
  assert.equal(service.supervisor.activeRequestCount, 0)
  const unknown = await measureModelPreview(model, new AbortController().signal, () => null, async function* () {
    yield { type: 'content', text: 'Unmeasured sample' } as StreamChunk
    yield { type: 'finish', reason: 'stop' } as StreamChunk
  })
  assert.equal(unknown.completionTokens, null)
  assert.equal(unknown.decodeTokensPerSecond, null, 'text length is not a token counter')
  await assert.rejects(measureModelPreview(model, new AbortController().signal, () => null, async function* () { yield { type: 'content', text: 'Truncated' } as StreamChunk }), /before a complete response/)
} finally { await service.supervisor.stop(); service.shutdown(); await h.cleanup() }
console.log('Actual preview transport, selected profile, output cap, usage, exclusivity and honest unknown measurements passed')
