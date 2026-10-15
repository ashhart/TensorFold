import assert from 'node:assert/strict'
import { parseInfoOutput, runEngineCli } from './cli'
import { hasCachedSnapshot, repoCachedBytes } from './paths'
import { EngineSupervisor, probeModels } from './supervisor'
import { makeHarness, processAlive, seedModel, waitFor, type Harness } from './test-fixtures/harness'

const GIB = 1024 ** 3

function supervisor(h: Harness, overrides: Partial<ConstructorParameters<typeof EngineSupervisor>[0]> = {}, env: NodeJS.ProcessEnv = {}) {
  let changes = 0
  const sup = new EngineSupervisor({
    binary: () => h.binary,
    env: () => ({ ...h.env, ...env }),
    info: async (repo) => {
      const r = await runEngineCli(h.binary, ['info', repo], { ...h.env, ...env })
      if (r.code !== 0) throw new Error(r.stdout.trim())
      return parseInfoOutput(r.stdout)
    },
    drafterFor: (repo) => (repo === 'TensorFold/Qwen3.8-27B-MLX-4bit' ? 'z-lab/Qwen3.8-27B-DFlash2' : null),
    drafterBytes: async (repo) => (await hasCachedSnapshot(h.hub, repo)) ? repoCachedBytes(h.hub, repo) : null,
    totalMemoryBytes: 64 * GIB,
    pollMs: 20,
    stopGraceMs: 2000,
    readyTimeoutMs: 10_000,
    onChange: () => { changes++ },
    ...overrides,
  })
  return { sup, changes: () => changes }
}

async function servePids(h: Harness): Promise<number[]> {
  return (await h.records()).filter((r) => r.kind === 'serve').map((r) => r.pid)
}

async function main() {
  const h = await makeHarness()
  try {
    await seedModel(h.hub, 'mlx-community/Qwen3.5-2B-MLX-4bit')
    await seedModel(h.hub, 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit', { modelType: 'nemotron_h' })
    await seedModel(h.hub, 'TensorFold/Qwen3.8-27B-MLX-4bit')

    const { sup, changes } = supervisor(h, {}, { OPENAI_API_KEY: 'sk-should-not-matter' })
    const live = await sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.match(live.baseUrl, /^http:\/\/127\.0\.0\.1:\d+\/v1$/)
    assert.equal(live.apiKey.length, 64)
    assert.equal(sup.status().state, 'ready')
    assert.ok(changes() >= 2, 'starting and ready are announced')
    const port = Number(new URL(live.baseUrl).port)
    assert.ok(await probeModels(port, live.apiKey))
    assert.equal(await probeModels(port, 'wrong-key'), null, 'the per-launch key guards the server')
    assert.equal(sup.status().contextTokens, 32768, 'max_model_len from /v1/models')
    assert.equal((await h.records()).some((r) => r.kind === 'tokenize'), false, 'no /tokenize call when /v1/models reports the window')

    const [serveRecord] = (await h.records()).filter((r) => r.kind === 'serve')
    assert.deepEqual(serveRecord.args.slice(0, 7), ['mlx-community/Qwen3.5-2B-MLX-4bit', '--host', '127.0.0.1', '--port', String(port), '--name', 'mlx-community/Qwen3.5-2B-MLX-4bit'])
    assert.deepEqual(serveRecord.args.slice(7, 9), ['--prompt-cache-gib', '8'], 'kept prompt states are capped for a desktop app')
    assert.equal(serveRecord.args.includes(live.apiKey), false, 'the key travels by environment, never argv')
    assert.equal(serveRecord.hasKey, true)
    assert.equal(serveRecord.noUpdateCheck, '1')

    // Same model: same process. Concurrent callers share one start.
    assert.deepEqual(await sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit'), live)
    assert.equal((await servePids(h)).length, 1)

    // Switching stops the old process before the new one starts; only one engine runs.
    const [firstPid] = await servePids(h)
    const [a, b] = await Promise.all([sup.serve('TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'), sup.serve('TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')])
    assert.deepEqual(a, b)
    assert.equal(processAlive(firstPid), false, 'the previous engine is gone')
    assert.equal((await servePids(h)).length, 2)
    assert.equal(sup.status().model, 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')

    // The 27B needs its drafter: refused while missing, passed as --drafter once cached.
    await assert.rejects(() => sup.serve('TensorFold/Qwen3.8-27B-MLX-4bit'), /needs its drafter z-lab\/Qwen3.8-27B-DFlash2/)
    assert.equal(sup.status().state, 'failed')
    await seedModel(h.hub, 'z-lab/Qwen3.8-27B-DFlash2', { dflash: true })
    await sup.serve('TensorFold/Qwen3.8-27B-MLX-4bit')
    const last = (await h.records()).filter((r) => r.kind === 'serve').at(-1)
    assert.deepEqual(last.args.slice(-2), ['--drafter', 'z-lab/Qwen3.8-27B-DFlash2'])

    await sup.stop()
    assert.equal(sup.status().state, 'stopped')
    assert.equal(sup.status().model, null)
    await waitFor(async () => (await servePids(h)).every((pid) => !processAlive(pid)), 5000, 'all engines to exit')

    // Window only from /tokenize, or none at all.
    const viaTokenize = supervisor(h, {}, { FAKE_MODELS_NO_CONTEXT: '1', FAKE_CONTEXT: '65536' })
    await viaTokenize.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.equal(viaTokenize.sup.status().contextTokens, 65536)
    assert.ok((await h.records()).some((r) => r.kind === 'tokenize'))
    await viaTokenize.sup.stop()
    assert.equal(viaTokenize.sup.status().contextTokens, null)
    const unbounded = supervisor(h, {}, { FAKE_CONTEXT: 'none' })
    await unbounded.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    assert.equal(unbounded.sup.status().contextTokens, null)
    await unbounded.sup.stop()

    // Memory floor above 70% of RAM: refused before any process starts.
    const small = supervisor(h, { totalMemoryBytes: 2 * GIB })
    const before = (await servePids(h)).length
    await assert.rejects(() => small.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit'), /needs at least 2\.0 GB of memory, more than the 1\.4 GB limit \(70% of this Mac's 2\.0 GB\)/)
    assert.equal((await servePids(h)).length, before)

    // A load that dies reports the engine's own last line; the next request may retry.
    const broken = supervisor(h, {}, { FAKE_SERVE_FAIL: 'Metal is unavailable' })
    await assert.rejects(() => broken.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit'), /tensorfold: Metal is unavailable/)
    assert.equal(broken.sup.status().state, 'failed')
    assert.ok(broken.sup.status().logTail.some((line) => line.includes('Metal is unavailable')))

    // A crash after ready flips the state to failed.
    const crashy = supervisor(h, {}, { FAKE_SERVE_EXIT_AFTER_MS: '150' })
    await crashy.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    await waitFor(() => crashy.sup.status().state === 'failed', 5000, 'crash to be noticed')
    assert.match(crashy.sup.status().message ?? '', /Metal device lost/)
    assert.equal(crashy.sup.live(), null)

    // Stop during a slow load cancels it; an engine ignoring SIGTERM is killed after the grace period.
    const slow = supervisor(h, { stopGraceMs: 300 }, { FAKE_SERVE_LOAD_MS: '3000', FAKE_SERVE_IGNORE_TERM: '1' })
    const loading = slow.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    await waitFor(() => slow.sup.status().state === 'starting' && slow.sup.status().port, 5000, 'load to begin')
    await slow.sup.stop()
    await assert.rejects(() => loading, /cancelled|exited/)
    assert.equal(slow.sup.status().state, 'stopped')

    const stubborn = supervisor(h, { stopGraceMs: 300 }, { FAKE_SERVE_IGNORE_TERM: '1' })
    await stubborn.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    const stubbornPid = (await servePids(h)).at(-1)!
    await stubborn.sup.stop()
    await waitFor(() => !processAlive(stubbornPid), 5000, 'SIGKILL after the grace period')

    // killNow is synchronous for app exit.
    const quitting = supervisor(h)
    await quitting.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
    const quitPid = (await servePids(h)).at(-1)!
    quitting.sup.killNow()
    await waitFor(() => !processAlive(quitPid), 5000, 'killNow to end the engine')

    // No binary: a clear refusal.
    const missing = supervisor(h, { binary: () => null })
    await assert.rejects(() => missing.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit'), /not bundled/)
    await assert.rejects(() => missing.sup.serve('../escape'), /owner\/name/)
  } finally {
    await h.cleanup()
  }
  console.log('engine supervisor tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
