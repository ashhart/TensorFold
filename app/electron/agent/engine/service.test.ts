import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { EngineEvent } from '../types'
import { HF_ORIGIN } from './hub'
import { EngineService, engineChildEnv } from './service'
import { recordEnginePerformance, setEnginePerformanceRecorder } from './provider'
import { makeHarness, processAlive, seedModel, waitFor } from './test-fixtures/harness'

const GIB = 1024 ** 3
const SHA = 'd'.repeat(40)

function hubFetch(repo: string, bytes: number, modelType = 'qwen3_5') {
  const routes: Record<string, unknown> = {
    [`/api/models/${repo}/revision/main`]: { sha: SHA, gated: false },
    [`/api/models/${repo}/tree/${SHA}`]: [{ type: 'file', path: 'model.safetensors', size: bytes }],
  }
  return (async (input: Parameters<typeof fetch>[0]) => {
    const url = String(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url)
    const pathOnly = url.slice(HF_ORIGIN.length).split('?')[0]
    if (pathOnly === `/${repo}/resolve/${SHA}/config.json`) return new Response(JSON.stringify({ model_type: modelType, quantization: { bits: 4, group_size: 64 } }), { status: 200 })
    if (!(pathOnly in routes)) return new Response('missing', { status: 404 })
    return new Response(JSON.stringify(routes[pathOnly]), { status: 200, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
}

async function main() {
  // Engine processes get OS values and the HF cache settings, never provider credentials.
  const childEnv = engineChildEnv({ HOME: '/u', PATH: '/bin', HF_HOME: '/hf', OPENAI_API_KEY: 'sk-x', ANTHROPIC_API_KEY: 'x', TENSORFOLD_API_KEY: 'stale', NODE_OPTIONS: '--inspect', HF_TOKEN: 'hf_x' })
  assert.deepEqual(childEnv, { HOME: '/u', PATH: '/bin', HF_HOME: '/hf' })

  const h = await makeHarness({ FAKE_PULL_FILES: '2', FAKE_PULL_SIZE: String(64 * 1024), FAKE_PULL_DELAY_MS: '2' })
  const events: EngineEvent[] = []
  let synced: string[] = []
  const service = new EngineService({
    binaryPath: '/fake/tensorfold-native',
    binary: h.binary,
    appHome: h.root,
    childEnv: h.env,
    fetch: hubFetch('mlx-community/Qwen3.5-2B-MLX-4bit', 2 * 64 * 1024),
    totalMemoryBytes: 32 * GIB,
    pollMs: 20,
    emit: (event) => events.push(event),
    onInstalled: (repos) => { synced = repos },
  })
  try {
    service.install()
    const empty = await service.snapshot()
    assert.equal(empty.status.version, '1.0.7')
    assert.equal(empty.status.available, true)
    assert.deepEqual(empty.status.hardware, { platform: os.platform(), architecture: os.arch(), chip: os.cpus()[0]?.model?.trim() || null }, 'hardware reports the actual host, independently of the test engine binary')
    assert.equal(empty.status.state, 'stopped')
    assert.equal(await service.connection(), null, 'a stopped engine has no client connection')
    assert.equal(empty.performance, null, 'no request has been measured')
    assert.equal(empty.status.memoryLimitBytes, Math.floor(32 * GIB * 0.7))
    assert.deepEqual(empty.installed, [])
    assert.equal(empty.hubCache, h.hub)
    assert.ok(empty.suggestions.some((s) => s.repo === 'TensorFold/Qwen3.8-27B-MLX-4bit'))
    assert.equal((await service.readiness(null)).reason, 'needs-model')

    // Preflight then download through the service; the installed list and provider models follow.
    const pre = await service.preflight('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.equal(pre.verdict, 'ok', pre.reasons.join('; '))
    const started = await service.pull('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.equal(started.state, 'running')
    await waitFor(() => events.some((e) => e.kind === 'pull' && e.pull.state === 'done'), 10_000, 'pull to finish')
    await waitFor(() => synced.length === 1, 5000, 'installed list to refresh')
    assert.deepEqual(synced, ['mlx-community/Qwen3.5-2B-MLX-4bit'])
    const installed = await service.installed()
    assert.equal(installed[0].family, 'Qwen3.5-2B or Qwen3.8-27B')
    assert.equal(installed[0].format, '4-bit, group 64')
    assert.equal(installed[0].fitsMemory, true)
    assert.ok((installed[0].memoryFloorBytes ?? 0) > 2 * GIB)
    assert.equal(service.pullList()[0].state, 'done')
    assert.deepEqual(await service.readiness('mlx-community/Qwen3.5-2B-MLX-4bit'), { ready: true, reason: 'ok', message: '' })
    assert.match((await service.readiness('Org/Other')).message, /not downloaded/)

    // HF_HUB_OFFLINE: no capabilities call, no network.
    const offline = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: { command: '/nonexistent', args: [] }, appHome: h.root, childEnv: { ...h.env, HF_HUB_OFFLINE: '1' }, fetch: (async () => { throw new Error('no network in offline mode') }) as typeof fetch, emit: () => {} })
    await assert.rejects(() => offline.preflight('mlx-community/Qwen3.5-2B-MLX-4bit'), /HF_HUB_OFFLINE/)
    await assert.rejects(() => offline.search('qwen'), /HF_HUB_OFFLINE/)

    // Unsupported repos are refused before any pull starts.
    const refusing = new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, fetch: hubFetch('google/gemma-9', 1024, 'gemma4'), totalMemoryBytes: 32 * GIB, emit: () => {} })
    await assert.rejects(() => refusing.pull('google/gemma-9'), /serves no gemma4 family/)
    await assert.rejects(() => refusing.pull('../../etc'), /owner\/name/)

    // Load, then delete is refused until unloaded; delete removes only the cache folder.
    const loading = service.load('mlx-community/Qwen3.5-2B-MLX-4bit')
    await waitFor(() => service.supervisor.status().state === 'starting', 5000, 'engine to start loading')
    assert.equal(await service.connection(), null, 'a loading model is not a working client connection')
    const status = await loading
    assert.equal(status.state, 'ready')
    const connection = await service.connection()
    assert.ok(connection)
    assert.deepEqual(connection, service.supervisor.live(), 'connection comes from the live supervisor')
    assert.equal(connection.model, status.model)
    assert.equal(connection.baseUrl, `http://127.0.0.1:${status.port}/v1`)
    assert.match(connection.apiKey, /^[a-f0-9]{64}$/)
    const unauthorized = await fetch(`${connection.baseUrl}/models`)
    assert.equal(unauthorized.status, 401, 'the endpoint still requires authentication')
    const authorized = await fetch(`${connection.baseUrl}/models`, { headers: { Authorization: `Bearer ${connection.apiKey}` } })
    assert.equal(authorized.status, 200, 'copied connection credentials authenticate to the running engine')
    assert.equal((await authorized.json()).data[0].id, connection.model)
    assert.ok(events.some((e) => e.kind === 'status' && e.status.state === 'starting'))
    await assert.rejects(() => service.remove('mlx-community/Qwen3.5-2B-MLX-4bit'), /Unload/)
    const record = JSON.parse(await fs.readFile(path.join(h.root, 'engine', 'serve.json'), 'utf8'))
    assert.equal(record.owner, process.pid)
    assert.equal(JSON.stringify({ snapshot: await service.snapshot(), events, record }).includes(connection.apiKey), false, 'snapshots, events, logs and process bookkeeping never contain the launch key')

    const sample = {
      requestAt: Date.now(), firstTokenAt: null, lastTokenAt: null, completionTokens: 86, promptTokens: 1200, cachedTokens: 400,
      engine: { decodeTokensPerSecond: 212.5, decodeSeconds: 0.4, prefillSeconds: 0.25, firstTokenSeconds: 0.31 },
    }
    recordEnginePerformance(connection.model, sample)
    const performance = (await service.snapshot()).performance
    assert.ok(performance)
    assert.deepEqual({ ...performance, measuredAt: 0 }, { model: connection.model, measuredAt: 0, decodeTokensPerSecond: 212.5, prefillTokensPerSecond: 3200, firstTokenMs: 310, completionTokens: 86 })
    assert.ok(performance.measuredAt >= sample.requestAt)
    assert.deepEqual(events.at(-1), { kind: 'performance', performance }, 'each completed request updates the renderer')
    recordEnginePerformance('Other/Model', sample)
    recordEnginePerformance(connection.model, { ...sample, requestAt: status.readyAt! - 1 })
    assert.deepEqual((await service.snapshot()).performance, performance, 'wrong-model and previous-launch samples are ignored')
    recordEnginePerformance(connection.model, { ...sample, requestAt: Date.now(), engine: { ...sample.engine, decodeTokensPerSecond: 212.345, decodeSeconds: 0.001 } })
    assert.equal((await service.snapshot()).performance?.decodeTokensPerSecond, 212.345, 'the latest request preserves the engine-reported rate without reconstructing a token count')
    recordEnginePerformance(connection.model, { requestAt: Date.now(), firstTokenAt: null, lastTokenAt: null, completionTokens: null })
    const missing = (await service.snapshot()).performance
    assert.ok(missing)
    assert.deepEqual({ ...missing, measuredAt: 0 }, { model: connection.model, measuredAt: 0, decodeTokensPerSecond: null, prefillTokensPerSecond: null, firstTokenMs: null, completionTokens: null }, 'missing timings and usage remain unknown')
    recordEnginePerformance(connection.model, { requestAt: Date.now(), firstTokenAt: NaN, lastTokenAt: Infinity, completionTokens: -1, promptTokens: Infinity, cachedTokens: -1, engine: { decodeTokensPerSecond: NaN, decodeSeconds: -1, prefillSeconds: Infinity, firstTokenSeconds: -1 } })
    const invalid = (await service.snapshot()).performance
    assert.deepEqual({ ...invalid, measuredAt: 0 }, { ...missing, measuredAt: 0 }, 'invalid measurements cannot create a speed or token estimate')

    // Switching models clears the previous request and rotates the explicit connection key.
    await seedModel(h.hub, 'Other/Kept')
    await service.load('Other/Kept')
    assert.equal((await service.snapshot()).performance, null)
    assert.ok(events.some((e) => e.kind === 'performance' && e.performance === null))
    recordEnginePerformance(connection.model, { ...sample, requestAt: Date.now() })
    assert.equal((await service.snapshot()).performance, null, 'an old model cannot populate the new model performance')
    const switched = await service.connection()
    assert.equal(switched?.model, 'Other/Kept')
    assert.notEqual(switched?.apiKey, connection.apiKey, 'a model switch uses a fresh launch credential')
    recordEnginePerformance('Other/Kept', { ...sample, requestAt: Date.now() })
    assert.equal((await service.snapshot()).performance?.model, 'Other/Kept')
    assert.equal((await service.unload()).state, 'stopped')
    assert.equal(await service.connection(), null, 'unloading discards the working client connection')
    assert.equal((await service.snapshot()).performance, null, 'unloading clears the latest request')
    await waitFor(async () => !(await fs.stat(path.join(h.root, 'engine', 'serve.json')).catch(() => null)), 5000, 'pid record to clear')
    const after = await service.remove('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.deepEqual(after.map((m) => m.repo), ['Other/Kept'])
    await assert.rejects(() => fs.stat(path.join(h.hub, 'models--mlx-community--Qwen3.5-2B-MLX-4bit')), { code: 'ENOENT' })
    await fs.stat(path.join(h.hub, 'models--Other--Kept'))

    // The 27B's drafter goes with it unless another installed model still uses it.
    await seedModel(h.hub, 'TensorFold/Qwen3.8-27B-MLX-4bit')
    await seedModel(h.hub, 'z-lab/Qwen3.8-27B-DFlash2', { dflash: true })
    await service.installed(true)
    await service.remove('TensorFold/Qwen3.8-27B-MLX-4bit')
    await assert.rejects(() => fs.stat(path.join(h.hub, 'models--z-lab--Qwen3.8-27B-DFlash2')), { code: 'ENOENT' })

    // A symlinked repo folder is never followed by delete.
    const outside = path.join(h.root, 'outside')
    await fs.mkdir(outside)
    await fs.symlink(outside, path.join(h.hub, 'models--Org--Linked'))
    await assert.rejects(() => service.remove('Org/Linked'), /not a cache folder/)
    await fs.stat(outside)
  } finally {
    service.shutdown()
    setEnginePerformanceRecorder(null)
  }

  // Stale engine from a crashed app: reaped only when its owner is gone and the pid still runs our binary.
  const stale = spawn(h.binary.command, [...h.binary.args, 'serve', 'Org/Stale', '--port', '0'], { env: h.env, detached: true, stdio: 'ignore' })
  await waitFor(() => stale.pid, 2000, 'stale engine')
  await fs.mkdir(path.join(h.root, 'reap', 'engine'), { recursive: true })
  await fs.writeFile(path.join(h.root, 'reap', 'engine', 'serve.json'), JSON.stringify({ pid: stale.pid, binary: process.execPath, owner: 999_999 }))
  const reaper = new EngineService({ binaryPath: process.execPath, binary: h.binary, appHome: path.join(h.root, 'reap'), childEnv: h.env, emit: () => {} })
  // ps shows `node .../fake-engine.mjs serve ...`, not `<binaryPath> serve`, so this pid is left alone.
  reaper.install()
  await new Promise((resolve) => setTimeout(resolve, 300))
  assert.equal(processAlive(stale.pid!), true, 'a pid running something else is never killed')
  try { process.kill(-stale.pid!, 'SIGKILL') } catch { /* gone */ }

  // The real case: the recorded pid runs `<binaryPath> serve ...` and its owning app is gone.
  const bin = path.join(h.root, 'bin')
  await fs.mkdir(bin)
  const fakeNative = path.join(bin, 'tensorfold-native')
  await fs.symlink(process.execPath, fakeNative)
  await fs.writeFile(path.join(bin, 'serve'), 'setInterval(() => {}, 1000)\n')
  const orphan = spawn(fakeNative, ['serve', 'Org/Orphan'], { cwd: bin, env: h.env, detached: true, stdio: 'ignore' })
  await waitFor(() => orphan.pid, 2000, 'orphan engine')
  await fs.mkdir(path.join(h.root, 'reap2', 'engine'), { recursive: true })
  await fs.writeFile(path.join(h.root, 'reap2', 'engine', 'serve.json'), JSON.stringify({ pid: orphan.pid, binary: fakeNative, owner: 999_999 }))
  new EngineService({ binaryPath: fakeNative, appHome: path.join(h.root, 'reap2'), childEnv: h.env, emit: () => {} }).install()
  await waitFor(() => !processAlive(orphan.pid!), 5000, 'orphaned engine to be reaped')
  await h.cleanup()
  console.log('engine service tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
