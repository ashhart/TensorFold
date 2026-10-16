import assert from 'node:assert/strict'
import { streamChat, quickComplete } from '../openai'
import { EngineService } from './service'
import { builtinEngineProvider, engineRoute, setEnginePerformanceRecorder, setEngineServeHook } from './provider'
import { makeHarness, processAlive, seedModel, waitFor } from './test-fixtures/harness'

async function collect(model: string, signal?: AbortSignal): Promise<{ state: 'ok' | 'error'; text?: string; message?: string }> {
  let text = ''
  try {
    for await (const chunk of streamChat({ provider: builtinEngineProvider(), model, messages: [{ role: 'user', content: 'fixture' }], signal })) {
      if (chunk.type === 'content') text += chunk.text
    }
    return { state: 'ok', text }
  } catch (error) { return { state: 'error', message: (error as Error).message } }
}

const h = await makeHarness({ FAKE_FIRST_TOKEN_MS: '300' })
const service = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 20, emit: () => {} })
try {
  await seedModel(h.hub, 'Org/A')
  await seedModel(h.hub, 'Org/B')
  service.install()
  const first = collect('Org/A')
  await waitFor(async () => (await h.records()).some((entry) => entry.kind === 'chat'), 3000, 'first active request')
  assert.equal(service.supervisor.activeRequestCount, 1)
  const pid = (await h.records()).find((entry) => entry.kind === 'serve').pid
  const other = await collect('Org/B')
  const original = await first
  assert.equal(other.state, 'error', 'a differing-model request is refused while an existing request is active')
  assert.match(other.message ?? '', /active.*requests/i)
  assert.equal(original.state, 'ok')
  assert.equal(original.text, 'Hello from the fake engine')
  assert.equal(service.supervisor.activeRequestCount, 0)
  assert.equal(processAlive(pid), true, 'the original child was not replaced')
  assert.equal((await h.records()).filter((entry) => entry.kind === 'serve').length, 1)

  const priorChats = (await h.records()).filter((entry) => entry.kind === 'chat').length
  let completed = 0
  const parallel = [collect('Org/A').then((value) => { completed++; return value }), collect('Org/A').then((value) => { completed++; return value })]
  await waitFor(async () => (await h.records()).filter((entry) => entry.kind === 'chat').length >= priorChats + 2, 2000, 'both same-model requests reach the child')
  assert.equal(completed, 0, 'same-model requests overlap rather than waiting for each other to finish')
  assert.equal(service.supervisor.activeRequestCount, 2)
  await assert.rejects(() => service.load('Org/B'), /active.*requests/i)
  await assert.rejects(() => service.unload(), /active.*requests/i)
  assert.equal(await quickComplete({ provider: builtinEngineProvider(), model: 'Org/A', messages: [{ role: 'user', content: 'title' }] }), 'Fake title')
  assert.ok((await Promise.all(parallel)).every((value) => value.state === 'ok'))
  await service.load('Org/B')

  // Cancel a real HTTP request before its first token, then confirm its lease is gone.
  const chatCount = (await h.records()).filter((entry) => entry.kind === 'chat').length
  const controller = new AbortController()
  const cancelled = collect('Org/B', controller.signal)
  await waitFor(async () => (await h.records()).filter((entry) => entry.kind === 'chat').length > chatCount, 2000, 'cancelled request reaches its child')
  controller.abort()
  assert.equal((await cancelled).state, 'error')
  assert.equal(service.supervisor.activeRequestCount, 0)
  await service.load('Org/A')
  await service.unload()

  // A generator's consumer can leave on finish without reading the final usage chunk.
  for await (const chunk of streamChat({ provider: builtinEngineProvider(), model: 'Org/A', messages: [] })) if (chunk.type === 'finish') break
  await service.unload()
} finally {
  service.shutdown()
  setEngineServeHook(null)
  setEnginePerformanceRecorder(null)
  await h.cleanup()
}

// Hook completion can race cancellation; an acquired endpoint discarded by the caller is released.
let calls = 0, releases = 0
setEngineServeHook(async () => {
  calls++
  await new Promise((resolve) => setTimeout(resolve, 20))
  return { model: 'Org/A', baseUrl: 'http://127.0.0.1:54321/v1', apiKey: 'fixture-key', release: () => { releases++ } }
})
try {
  const controller = new AbortController()
  const request = engineRoute({ provider: builtinEngineProvider(), model: 'Org/A', messages: [], signal: controller.signal })
  controller.abort()
  await assert.rejects(() => request)
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(releases, 1, 'late hook completion releases a lease the caller never received')
  const alreadyCancelled = new AbortController()
  alreadyCancelled.abort()
  await assert.rejects(() => engineRoute({ provider: builtinEngineProvider(), model: 'Org/A', messages: [], signal: alreadyCancelled.signal }))
  assert.equal(calls, 1, 'pre-aborted requests do not start an engine hook')
} finally { setEngineServeHook(null) }
console.log('Real service/provider engine request lease tests passed')
