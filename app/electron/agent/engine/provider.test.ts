import assert from 'node:assert/strict'
import { quickComplete, streamChat } from '../openai'
import { defaultSettings } from '../../store/defaults'
import type { EngineEvent, Provider } from '../types'
import { ENGINE_FIRST_BYTE_MS, ENGINE_IDLE_MS, ENGINE_PLACEHOLDER_URL, builtinEngineProvider, engineFetch, engineRoute, setEnginePerformanceRecorder, setEngineServeHook, withBuiltinEngineProvider } from './provider'
import { EngineService } from './service'
import { makeHarness, seedModel } from './test-fixtures/harness'

async function collect(stream: AsyncGenerator<any>): Promise<{ text: string; finish: string | null }> {
  let text = ''
  let finish: string | null = null
  for await (const chunk of stream) {
    if (chunk.type === 'content') text += chunk.text
    if (chunk.type === 'finish') finish = chunk.reason
  }
  return { text, finish }
}

async function main() {
  // Migration: the built-in provider is added when missing and cannot be repointed.
  const omlx: Provider = { id: 'omlx', name: 'oMLX', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: ['m'], kind: 'local', source: 'omlx' }
  const seeded = withBuiltinEngineProvider([omlx])
  assert.deepEqual(seeded.map((p) => p.id), ['omlx', 'tensorfold'])
  assert.equal(seeded[1].source, 'tensorfold')
  assert.equal(seeded[1].transport, 'chat-completions')
  assert.equal(seeded[1].kind, 'local')
  const tampered = withBuiltinEngineProvider([{ ...builtinEngineProvider(), baseUrl: 'https://evil.example/v1', apiKey: 'x', source: 'custom', models: ['Org/A'] }, omlx, { ...builtinEngineProvider(), name: 'duplicate' }])
  assert.deepEqual(tampered.map((p) => p.id), ['tensorfold', 'omlx'], 'position kept, duplicates dropped')
  assert.equal(tampered[0].baseUrl, ENGINE_PLACEHOLDER_URL)
  assert.equal(tampered[0].apiKey, '')
  assert.equal(tampered[0].source, 'tensorfold')
  assert.deepEqual(tampered[0].models, ['Org/A'], 'installed models survive the migration')
  assert.equal(withBuiltinEngineProvider(withBuiltinEngineProvider([])).length, 1, 'idempotent')
  assert.deepEqual(withBuiltinEngineProvider([omlx], true), [omlx], 'once seeded, a deletion sticks')
  assert.ok(defaultSettings().providers.some((p) => p.id === 'tensorfold' && p.source === 'tensorfold'), 'fresh profiles ship it')

  // Every other provider passes straight through.
  assert.equal(await engineRoute({ provider: omlx, model: 'm', messages: [] }), null)
  assert.equal(await engineRoute({ baseUrl: 'http://127.0.0.1:1/v1', model: 'm', messages: [] }), null)
  setEngineServeHook(null)
  await assert.rejects(() => engineRoute({ provider: builtinEngineProvider(), model: 'Org/A', messages: [] }), /not available in this process/)

  assert.throws(() => engineFetch('http://10.0.0.5:8080/v1'), /only listens on 127\.0\.0\.1/)
  await assert.rejects(() => engineFetch('http://127.0.0.1:5555/v1')('http://127.0.0.1:5556/v1/models'), /another origin/)

  // End to end through streamChat: the hook starts the fake engine and swaps in its URL and key.
  const h = await makeHarness({ FAKE_FIRST_TOKEN_MS: '50', FAKE_REPLY_TIMINGS: '1' })
  const events: EngineEvent[] = []
  const service = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 64 * 1024 ** 3, pollMs: 20, emit: (event) => events.push(event) })
  try {
    service.install()
    await seedModel(h.hub, 'mlx-community/Qwen3.5-2B-MLX-4bit')
    const provider = builtinEngineProvider()

    // quickComplete never starts or switches the engine for a title.
    await assert.rejects(() => quickComplete({ provider, baseUrl: provider.baseUrl, model: 'mlx-community/Qwen3.5-2B-MLX-4bit', messages: [{ role: 'user', content: 'hi' }] }), /not serving/)
    assert.equal(service.supervisor.status().state, 'stopped')

    const route = await engineRoute({ provider, baseUrl: provider.baseUrl, apiKey: '', model: 'mlx-community/Qwen3.5-2B-MLX-4bit', messages: [] })
    assert.ok(route)
    assert.match(route.params.baseUrl!, /^http:\/\/127\.0\.0\.1:\d+\/v1$/)
    assert.equal(route.params.provider?.baseUrl, route.params.baseUrl, 'the provider endpoint matches the live origin')
    assert.equal(route.firstByteMs, ENGINE_FIRST_BYTE_MS)
    assert.equal(route.idleMs, ENGINE_IDLE_MS)
    assert.ok(ENGINE_FIRST_BYTE_MS > 60_000 && ENGINE_IDLE_MS > 60_000, 'both limits are longer than the 60 s used for other providers')
    route.release?.()

    const result = await collect(streamChat({ provider, baseUrl: provider.baseUrl, apiKey: '', model: 'mlx-community/Qwen3.5-2B-MLX-4bit', messages: [{ role: 'user', content: 'hi' }] }))
    assert.equal(result.text, 'Hello from the fake engine')
    assert.equal(result.finish, 'stop')
    const chats = (await h.records()).filter((r) => r.kind === 'chat')
    assert.equal(chats.at(-1).model, 'mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.equal((await h.records()).filter((r) => r.kind === 'serve').length, 1, 'the engine started once')
    const performance = (await service.snapshot()).performance
    assert.ok(performance, 'the completed built-in request records the latest performance')
    assert.deepEqual({ ...performance, measuredAt: 0 }, { model: 'mlx-community/Qwen3.5-2B-MLX-4bit', measuredAt: 0, decodeTokensPerSecond: 200, prefillTokensPerSecond: 80, firstTokenMs: 50, completionTokens: 3 })
    assert.deepEqual(events.filter((event) => event.kind === 'performance').at(-1), { kind: 'performance', performance })

    // A compatible server with the same wire timings cannot update built-in engine metrics.
    const connection = await service.connection()
    assert.ok(connection)
    const unrelated: Provider = { ...omlx, baseUrl: connection.baseUrl, apiKey: connection.apiKey, models: [connection.model] }
    await collect(streamChat({ provider: unrelated, baseUrl: unrelated.baseUrl, apiKey: unrelated.apiKey, model: connection.model, messages: [{ role: 'user', content: 'hi' }] }))
    assert.deepEqual((await service.snapshot()).performance, performance, 'only the built-in route records engine performance')

    assert.equal(await quickComplete({ provider, baseUrl: provider.baseUrl, model: 'mlx-community/Qwen3.5-2B-MLX-4bit', messages: [{ role: 'user', content: 'title' }] }), 'Fake title')

    // An aborted wait does not leave the caller hanging.
    const controller = new AbortController()
    controller.abort()
    await assert.rejects(() => collect(streamChat({ provider, baseUrl: provider.baseUrl, model: 'mlx-community/Qwen3.5-2B-MLX-4bit', messages: [], signal: controller.signal })))
  } finally {
    service.shutdown()
    setEngineServeHook(null)
    setEnginePerformanceRecorder(null)
    await h.cleanup()
  }
  console.log('engine provider tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
