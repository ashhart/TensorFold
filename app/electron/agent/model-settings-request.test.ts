import assert from 'node:assert/strict'
import { streamChat } from './openai'
import { builtinEngineProvider, setEnginePerformanceRecorder, setEngineServeHook } from './engine/provider'

const originalFetch = globalThis.fetch
let body: Record<string, unknown> = {}
const generation = { contextTokens: 8192, maxOutputTokens: 512, temperature: 0.7, topP: 0.85, topK: 30, minP: 0.05, speculativeDecoding: 'off' as const }
setEngineServeHook(async (model) => ({ model, baseUrl: 'http://127.0.0.1:24680/v1', apiKey: '', generation }))
globalThis.fetch = async (_url, init) => {
  body = JSON.parse(String(init?.body))
  return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }), { headers: { 'content-type': 'application/json' } })
}
try {
  for await (const _chunk of streamChat({
    provider: builtinEngineProvider(), model: 'example/model-a', messages: [{ role: 'user', content: 'hello' }], temperature: 0.1,
    modelSettings: { ...generation, temperature: 1.5 },
  } as any)) { /* consume the actual transport */ }
  assert.equal(body?.temperature, 0.7, 'the selected model profile must override the global temperature on the actual request')
  assert.equal(body?.max_tokens, 512, 'the selected maximum output must reach the engine')
  assert.equal(body?.top_p, 0.85)
  assert.equal(body?.top_k, 30)
  assert.equal(body?.min_p, 0.05)
  assert.equal(body?.draft, false)
  for (const reasoningEffort of ['none', ' NONE ']) {
    for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/model-a', messages: [], reasoningEffort })) {}
    assert.deepEqual(body.chat_template_kwargs, { enable_thinking: false }, 'explicit no-reasoning must disable native thinking instead of exhausting the reply budget in hidden reasoning')
    assert.equal(body.max_tokens, generation.maxOutputTokens, 'disabling thinking preserves the selected reply cap')
    assert.equal(body.temperature, generation.temperature)
  }
  for (const reasoningEffort of [undefined, 'medium', 'high', 'dynamic']) {
    for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/model-a', messages: [], reasoningEffort })) {}
    assert.equal(body.chat_template_kwargs, undefined, 'normal reasoning choices retain their existing native template behavior')
  }
  // The renderer/caller cannot smuggle local sampler fields into an external provider.
  for await (const _chunk of streamChat({ baseUrl: 'http://127.0.0.1:24680/v1', model: 'external-model', messages: [], temperature: 0.2, modelSettings: generation, greedyOnly: true, reasoningEffort: 'none' })) {}
  assert.equal(body?.temperature, 0.2)
  for (const field of ['max_tokens', 'top_p', 'top_k', 'min_p', 'draft']) assert.equal(body?.[field], undefined, `${field} remains scoped to the built-in engine`)
  assert.equal(body.chat_template_kwargs, undefined, 'native template flags never leak to external providers')
  setEngineServeHook(async (model) => ({ model, baseUrl: 'http://127.0.0.1:24680/v1', apiKey: '', greedyOnly: true }))
  for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/greedy-model', messages: [], temperature: 0.1 })) {}
  assert.equal(body.temperature, 0, 'a greedy-only engine model works without creating a profile')
  assert.equal(body.top_p, undefined)
  setEngineServeHook(async (model) => ({ model, baseUrl: 'http://127.0.0.1:24680/v1', apiKey: '' }))
  for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/sampled-model', messages: [], temperature: 0.3, greedyOnly: true })) {}
  assert.equal(body.temperature, 0.3, 'caller cannot invent a greedy-only engine capability')
  let exclusive = false, performanceWrites = 0
  setEngineServeHook(async (model, options) => { exclusive = options.exclusive === true; return { model, baseUrl: 'http://127.0.0.1:24680/v1', apiKey: '', generation } })
  setEnginePerformanceRecorder(() => { performanceWrites++ })
  globalThis.fetch = async (_url, init) => {
    body = JSON.parse(String(init?.body))
    const frames = [
      { choices: [{ delta: { content: 'sample' }, finish_reason: null }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }], usage: { completion_tokens: 10, prompt_tokens: 20 } },
    ]
    return new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
  }
  const chunks = []
  for await (const chunk of streamChat({ provider: builtinEngineProvider(), model: 'example/model-a', messages: [], modelPreview: true })) chunks.push(chunk)
  assert.equal(exclusive, true, 'preview gets an exclusive real engine request lease')
  assert.equal(body.max_tokens, 96, 'preview bounds large selected reply caps')
  assert.equal(body.temperature, generation.temperature)
  assert.deepEqual(body.chat_template_kwargs, { enable_thinking: false })
  assert.ok(chunks.some((chunk) => chunk.type === 'usage' && chunk.completionTokens === 10), 'preview retains actual usage metrics')
  assert.equal(performanceWrites, 0, 'preview does not replace latest completed chat performance')
  console.log('Selected per-model settings reach the actual engine request; previews are exclusive, bounded and separate from chat metrics')
} finally { globalThis.fetch = originalFetch; setEngineServeHook(null); setEnginePerformanceRecorder(null) }
