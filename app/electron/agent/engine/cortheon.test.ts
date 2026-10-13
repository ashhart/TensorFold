import assert from 'node:assert/strict'
import { EngineService } from './service'
import { makeHarness, seedModel, waitFor } from './test-fixtures/harness'

const GIB = 1024 ** 3
const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'

function service(h: Awaited<ReturnType<typeof makeHarness>>): EngineService {
  return new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * GIB, pollMs: 20, emit: () => {} })
}

const serveArgs = async (h: Awaited<ReturnType<typeof makeHarness>>) => (await h.records()).filter((r) => r.kind === 'serve').map((r) => r.args as string[])

async function main() {
  {
    // An engine without the flag never gets it, and the switch refuses.
    const h = await makeHarness()
    const svc = service(h)
    try {
      assert.deepEqual((await svc.status()).cortheon, { supported: false, enabled: false, active: false })
      await assert.rejects(() => svc.setCortheon(true), /does not offer Cortheon/)
      assert.equal((await svc.status()).cortheon?.enabled, false, 'a refused switch saves nothing')
    } finally { await svc.unload().catch(() => {}); await h.cleanup() }
  }

  {
    const h = await makeHarness({ FAKE_ENGINE_CORTHEON: '1' })
    await seedModel(h.hub, MODEL)
    const svc = service(h)
    try {
      assert.equal((await svc.status()).cortheon?.supported, true)
      await svc.load(MODEL)
      assert.ok(!(await serveArgs(h))[0].includes('--cortheon'), 'off by default')

      // Turning it on restarts the loaded model with the flag and reports it live.
      const on = await svc.setCortheon(true)
      assert.equal(on.state, 'ready')
      assert.deepEqual(on.cortheon, { supported: true, enabled: true, active: true })
      const runs = await serveArgs(h)
      assert.equal(runs.length, 2, 'one restart')
      assert.equal(runs[1][runs[1].indexOf('--name') + 1], MODEL, 'the same model restarts')
      assert.ok(runs[1].includes('--cortheon'))

      // The choice survives the app restarting.
      const again = service(h)
      assert.equal((await again.status()).cortheon?.enabled, true)

      const off = await svc.setCortheon(false)
      assert.deepEqual(off.cortheon, { supported: true, enabled: false, active: false })
      await waitFor(async () => (await serveArgs(h)).length === 3, 5000, 'restart without the flag')
      assert.ok(!(await serveArgs(h))[2].includes('--cortheon'))

      // With nothing loaded the switch only saves the choice.
      await svc.unload()
      await svc.setCortheon(true)
      assert.equal((await serveArgs(h)).length, 3, 'no model starts')
      await svc.load(MODEL)
      assert.ok((await serveArgs(h))[3].includes('--cortheon'), 'the next load uses it')
    } finally { await svc.unload().catch(() => {}); await h.cleanup() }
  }
  console.log('engine Cortheon switch: flag only when offered, restart on change, saved across launches')
}

main().catch((error) => { console.error(error); process.exit(1) })
