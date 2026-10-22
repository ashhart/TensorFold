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
    assert.equal(serveRecord.args.includes('--prompt-cache-gib'), false, 'native admission sizes retained states from actual loaded memory')
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
    assert.equal(sup.status().state, 'ready', 'failed selection restores the previously working engine')
    assert.equal(sup.status().model, 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')
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

    const explicitCache = await runEngineCli(h.binary, ['serve', 'mlx-community/Qwen3.5-2B-MLX-4bit', '--prompt-cache-gib', '8'], { ...h.env, FAKE_PROMPT_CACHE_ROOM_GIB: '0' })
    assert.equal(explicitCache.code, 1); assert.match(explicitCache.stderr, /prompt cache exceeds memory left/, 'the fake engine reproduces native rejection of a positive cache beside a fitting model')
    for (const roomGiB of [0, 3]) {
      const adaptive = supervisor(h, { totalMemoryBytes: 48 * GIB }, { FAKE_PROMPT_CACHE_ROOM_GIB: String(roomGiB) })
      try {
        await adaptive.sup.serve('mlx-community/Qwen3.5-2B-MLX-4bit')
        assert.equal(adaptive.sup.status().state, 'ready', 'a fitting native load remains usable even when retained prompt states have no room')
        const launch = (await h.records()).filter((record) => record.kind === 'serve').at(-1)
        assert.equal(launch.args.includes('--prompt-cache-gib'), false, 'the app cannot reserve a positive cache from its weights-only floor')
        assert.equal(launch.args.includes('--prompt-cache-over-cap'), false, 'the app cannot bypass native memory admission')
      } finally { await adaptive.sup.stop() }
    }

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

    // Native load-limit semantics (families/glm/load_plan.zig, native/cache_fit.zig): decimal GB,
    // each variable applied only to its own model family, requested values capped at this RAM.
    const baseLimit = Math.floor(0.7 * 64 * GIB)
    assert.equal(supervisor(h).sup.memoryLimitBytes, baseLimit)
    assert.equal(supervisor(h).sup.memoryLimitFor('glm5_next'), baseLimit)
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitFor('glm5_next'), 60e9, 'the GLM variable raises the GLM family to the requested decimal GB')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitFor('glm_moe_dsa'), 60e9, 'both GLM checkpoints share the GLM variable')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitFor('qwen4_exp'), baseLimit, 'the GLM variable never touches the Flash Next family')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitFor(null), baseLimit, 'a checkpoint with no family keeps the default policy')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitBytes, baseLimit, 'the reported app default policy stays the 70% admission limit')
    assert.equal(supervisor(h, {}, { FLASHNEXT_LOAD_LIMIT_GB: '55' }).sup.memoryLimitFor('qwen4_exp'), 55e9, 'the Flash Next variable raises only its own family')
    assert.equal(supervisor(h, {}, { FLASHNEXT_LOAD_LIMIT_GB: '55', GLM_LOAD_LIMIT_GB: '60' }).sup.memoryLimitFor('glm5_next'), 60e9, 'with both set, each family reads only its own variable')
    assert.equal(supervisor(h, {}, { FLASHNEXT_LOAD_LIMIT_GB: '40' }).sup.memoryLimitFor('qwen4_exp'), baseLimit, 'a limit below the default does not lower it')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: ' 60 ' }).sup.memoryLimitFor('glm5_next'), 60e9, 'surrounding spaces match the native trim')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '0x64' }).sup.memoryLimitFor('glm5_next'), 64 * GIB, 'hex parses exactly like the native parseFloat: 0x64 is 100 GB, capped at this Mac\'s RAM')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '100__0' }).sup.memoryLimitFor('glm5_next'), baseLimit, 'double underscores are refused exactly like the native parseFloat')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: 'about sixty' }).sup.memoryLimitFor('glm5_next'), baseLimit, 'a value the engine would reject changes nothing here')
    assert.equal(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '1e308' }).sup.memoryLimitFor('glm5_next'), 64 * GIB, 'an overflowing request is capped at the machine RAM, never Infinity')
    assert.match(supervisor(h, {}, { GLM_LOAD_LIMIT_GB: '1e308' }).sup.memoryPolicyFor('glm5_next').basis, /capped at this Mac's memory ceiling/)
  } finally {
    for (const pid of await servePids(h)) if (Number.isSafeInteger(pid) && pid > 1 && processAlive(pid)) { try { process.kill(-pid, 'SIGKILL') } catch {} }
    await h.cleanup()
  }
  console.log('engine supervisor tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
