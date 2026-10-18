import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService, engineChildEnv } from './service'
import { EnginePreferenceStore } from './preferences'
import { EngineSupervisor, probeModels } from './supervisor'
import { CORTHEON_RELEASE_POLICY, cortheonEnabledForRelease } from '../../../shared/release-feature-policy'
import { makeHarness, seedModel } from './test-fixtures/harness'

const GIB = 1024 ** 3
const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'
const LEARNING_MODEL = 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
const paused = /temporarily disabled in this beta/i
const service = (h: Awaited<ReturnType<typeof makeHarness>>) => new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * GIB, pollMs: 20, emit: () => {} })

async function main() {
  assert.equal(Object.isFrozen(CORTHEON_RELEASE_POLICY), true)
  assert.equal(cortheonEnabledForRelease(true), false)
  assert.equal(cortheonEnabledForRelease('1'), false)
  const h = await makeHarness({ FAKE_ENGINE_CORTHEON: '1' })
  const svc = service(h)
  try {
    const before = await h.records()
    await assert.rejects(svc.setCortheon(true), paused, 'beta.6 refuses enable before any CLI, model restart or preference mutation')
    assert.deepEqual(await h.records(), before)
    await assert.rejects(fs.stat(path.join(h.root, 'engine', 'preferences.json')), /ENOENT/)
    await seedModel(h.hub, MODEL)
    await svc.load(MODEL)
    const live = await svc.connection()
    assert.ok(live)
    assert.ok(await probeModels(Number(new URL(live.baseUrl).port), live.apiKey), 'ordinary local inference remains available')
    const served = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(served.length, 1)
    assert.ok(!served[0].args.includes('--cortheon'))
    const lease = await svc.supervisor.acquire(MODEL, { start: false })
    try { await assert.rejects(svc.setCortheon(true), paused) } finally { lease.release() }
    assert.equal((await h.records()).filter((row) => row.kind === 'serve').length, 1, 'refused enable cannot restart an ordinary model')
    assert.equal((await svc.status()).cortheon?.enabled, false)
    assert.match((await svc.status()).cortheon?.disabledReason ?? '', paused)
    await svc.setCortheon(false)
    assert.equal((await h.records()).filter((row) => row.kind === 'serve').length, 1, 'turning an already-off feature off does not restart')
    // Observed state is still authoritative if an old or unmanaged state is reported.
    const original = svc.supervisor.status.bind(svc.supervisor)
    svc.supervisor.status = () => ({ ...original(), cortheonActive: true })
    assert.equal(svc.hasExternalResearchEnabled(), true, 'the release pause must not hardcode the live privacy guard to false')
    assert.equal((await svc.status()).cortheon?.active, true)
    await svc.setCortheon(false)
    assert.equal((await h.records()).filter((row) => row.kind === 'serve').length, 2, 'Off reconfigures an observed legacy active instance even when normalized preference is already false')
    assert.ok((await h.records()).filter((row) => row.kind === 'serve').every((row) => !row.args.includes('--cortheon')))
    svc.supervisor.status = original
  } finally { await svc.unload().catch(() => {}); await h.cleanup() }

  const old = await makeHarness({ FAKE_ENGINE_CORTHEON: '1' })
  const file = path.join(old.root, 'engine', 'preferences.json')
  await fs.mkdir(path.dirname(file), { recursive: true })
  const prior = { version: 1, cortheon: true, learningEnabled: true, learningModel: LEARNING_MODEL }
  await fs.writeFile(file, JSON.stringify(prior))
  const peers = [path.join(old.root, 'settings.json'), path.join(old.root, 'threads.json')]
  const retained = JSON.stringify({ savedProviderKey: 'owned-fictional-fixture', history: 'Synthetic retained question.' })
  await Promise.all(peers.map((peer) => fs.writeFile(peer, retained)))
  const oldService = service(old)
  try {
    const store = new EnginePreferenceStore(old.root)
    assert.deepEqual(store.get(), { cortheon: false, learningEnabled: true, learningModel: LEARNING_MODEL }, 'old preferences load with research off, retaining learning choices')
    const oldBytes = await fs.readFile(file)
    await assert.rejects(store.set({ cortheon: true }), paused)
    assert.deepEqual(await fs.readFile(file), oldBytes, 'an explicit enable cannot mutate stored preferences')
    await store.set({ learningEnabled: false })
    assert.deepEqual(JSON.parse(await fs.readFile(file, 'utf8')), { ...prior, cortheon: false, learningEnabled: false }, 'the next normal save persists the pause without losing the selected model')
    await seedModel(old.hub, MODEL)
    await oldService.load(MODEL)
    assert.ok(!(await old.records()).find((row) => row.kind === 'serve').args.includes('--cortheon'), 'a service created before normalized save also cannot start old true')
    for (const peer of peers) assert.equal(await fs.readFile(peer, 'utf8'), retained, 'engine preference changes never touch provider settings or chat history')
  } finally { await oldService.unload().catch(() => {}); await old.cleanup() }

  const direct = await makeHarness({ FAKE_ENGINE_CORTHEON: '1', TENSORFOLD_CORTHEON: '1', CORTHEON_ENABLED: '1' })
  await seedModel(direct.hub, MODEL)
  const sup = new EngineSupervisor({ binary: () => direct.binary, env: () => direct.env, info: async () => ({ repo: MODEL, modelType: 'qwen3_5', family: 'fixture', format: '4-bit', contextTokens: 32768, weightsBytes: 8192, memoryFloorBytes: 8192 }), drafterFor: () => null, drafterBytes: async () => null, totalMemoryBytes: 32 * GIB, pollMs: 20, cortheon: () => true })
  try {
    const live = await sup.serve(MODEL)
    assert.ok(await probeModels(Number(new URL(live.baseUrl).port), live.apiKey))
    assert.equal(sup.status().cortheonActive, false)
    assert.ok(!(await direct.records()).find((row) => row.kind === 'serve').args.includes('--cortheon'), 'the supervisor independently ignores a stale true callback')
    await sup.restart()
    assert.ok((await direct.records()).filter((row) => row.kind === 'serve').every((row) => !row.args.includes('--cortheon')), 'restart cannot re-enable research')
    const env = engineChildEnv({ HOME: direct.root, TENSORFOLD_CORTHEON: '1', CORTHEON_ENABLED: '1', TENSORFOLD_API_KEY: 'owned-fictional-key' })
    assert.equal(env.TENSORFOLD_CORTHEON, undefined)
    assert.equal(env.CORTHEON_ENABLED, undefined)
    assert.equal(env.TENSORFOLD_API_KEY, undefined)
    await sup.stop()
    const beforePrefix = (await direct.records()).filter((row) => row.kind === 'serve').length
    const prefixed = new EngineSupervisor({ binary: () => ({ ...direct.binary, args: [...direct.binary.args, '--cortheon'] }), env: () => direct.env, info: async () => ({ repo: MODEL, modelType: 'qwen3_5', family: 'fixture', format: '4-bit', contextTokens: 32768, weightsBytes: 8192, memoryFloorBytes: 8192 }), drafterFor: () => null, drafterBytes: async () => null, totalMemoryBytes: 32 * GIB, pollMs: 20 })
    await assert.rejects(prefixed.serve(MODEL), paused, 'the final spawn also rejects a stale binary argument prefix')
    assert.equal((await direct.records()).filter((row) => row.kind === 'serve').length, beforePrefix, 'prefix refusal occurs before spawning a child')
    await prefixed.stop()
  } finally { await sup.stop(); await direct.cleanup() }
  console.log('beta.6 Cortheon pause: API, saved preferences, direct supervisor starts/restarts and live privacy status; ordinary local model still serves')
}
main().catch((error) => { console.error(error); process.exit(1) })
