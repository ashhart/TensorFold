import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { EnginePreferenceStore } from './preferences'
import { mergeControlVector, readControlVector } from '../../../shared/control-vector'
import { makeHarness, seedModel } from './test-fixtures/harness'

const GIB = 1024 ** 3
const MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'
const service = (h: Awaited<ReturnType<typeof makeHarness>>) => new EngineService({ binaryPath: '/fake/tensorfold-native', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * GIB, pollMs: 20, emit: () => {} })

async function main() {
  // The strength the engine takes: above zero, and not past the app's range.
  const current = { enabled: true, file: '/tmp/v.gguf', alpha: 2 }
  assert.throws(() => mergeControlVector(current, { enabled: true, alpha: 0 }), /above 0/)
  assert.throws(() => mergeControlVector(current, { enabled: true, alpha: 99 }), /above 0/)
  assert.throws(() => mergeControlVector(current, { enabled: true, alpha: 'strong' as unknown as number }), /above 0/)
  assert.deepEqual(mergeControlVector(current, { enabled: true, alpha: 2.5 }), { enabled: true, alpha: 2.5 })
  // A field the caller did not name keeps its value: a new strength does not switch the vector off.
  assert.deepEqual(mergeControlVector(current, { alpha: 3 }), { enabled: true, alpha: 3 })
  assert.deepEqual(mergeControlVector(current, {}), { enabled: true, alpha: 2 })
  assert.deepEqual(mergeControlVector(current, { enabled: false }), { enabled: false, alpha: 2 })
  // A saved vector that names no file cannot be in force, whatever it says.
  assert.deepEqual(readControlVector({ enabled: true, file: null, alpha: 3 }), { enabled: false, file: null, alpha: 3 })
  assert.deepEqual(readControlVector({ enabled: true, file: '/tmp/v.gguf', alpha: 0 }), { enabled: true, file: '/tmp/v.gguf', alpha: 1 })

  // An engine that does not offer --glp cannot take a vector.
  const plain = await makeHarness()
  const plainSvc = service(plain)
  try {
    await seedModel(plain.hub, MODEL)
    await plainSvc.load(MODEL)
    assert.equal((await plainSvc.controlVectorSnapshot()).supported, false)
    await assert.rejects(plainSvc.setControlVector({ enabled: true, alpha: 1 }), /does not offer control vectors/)
    const served = (await plain.records()).filter((row) => row.kind === 'serve')
    assert.equal(served.length, 1, 'a refused vector cannot restart the model')
    assert.ok(!served[0].args.includes('--glp'))
  } finally { await plainSvc.unload().catch(() => {}); await plain.cleanup() }

  // An engine that offers --glp takes the file the operator chose.
  const h = await makeHarness({ FAKE_ENGINE_GLP: '1' })
  const svc = service(h)
  try {
    await seedModel(h.hub, MODEL)
    await svc.load(MODEL)
    const vector = path.join(h.root, 'refusal.gguf')
    await fs.writeFile(vector, 'glp')
    await assert.rejects(svc.setControlVector({ enabled: true, alpha: 1 }), /Choose a control vector file first/)
    const chosen = await svc.chooseControlVector(vector)
    assert.equal(chosen.supported, true)
    assert.equal(chosen.file, vector)
    assert.equal(chosen.enabled, false, 'choosing a file does not put it in force')
    assert.equal((await h.records()).filter((row) => row.kind === 'serve').length, 1, 'choosing a file does not restart the model')

    const enabled = await svc.setControlVector({ enabled: true, alpha: 2 })
    assert.equal(enabled.enabled, true)
    assert.equal(enabled.alpha, 2)
    const served = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(served.length, 2, 'putting a vector in force restarts the model')
    assert.ok(served[1].args.includes('--glp'))
    assert.equal(served[1].args[served[1].args.indexOf('--glp') + 1], vector)
    assert.equal(served[1].args[served[1].args.indexOf('--glp-alpha') + 1], '2')

    // A new strength on its own leaves the vector in force.
    const stronger = await svc.setControlVector({ alpha: 3 })
    assert.equal(stronger.enabled, true)
    assert.equal(stronger.alpha, 3)
    const restrung = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(restrung.length, 3, 'a new strength restarts the model with it')
    assert.equal(restrung[2].args[restrung[2].args.indexOf('--glp-alpha') + 1], '3')

    // Another file takes effect at once when a vector is already in force.
    const other = path.join(h.root, 'other.gguf')
    await fs.writeFile(other, 'glp')
    const swapped = await svc.chooseControlVector(other)
    assert.equal(swapped.file, other)
    assert.equal(swapped.enabled, true)
    const exchanged = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(exchanged.length, 4, 'swapping the file restarts the model with it')
    assert.equal(exchanged[3].args[exchanged[3].args.indexOf('--glp') + 1], other)

    // The choice survives a restart of the app: it is on disk, not in memory.
    const stored = new EnginePreferenceStore(h.root)
    assert.equal(stored.get().controlVector.file, other)
    assert.equal(stored.get().controlVector.alpha, 3)

    await svc.setControlVector({ enabled: false })
    const off = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(off.length, 5)
    assert.ok(!off[4].args.includes('--glp'), 'turning the vector off restarts the model without it')
    const after = new EnginePreferenceStore(h.root)
    assert.equal(after.get().controlVector.enabled, false)
    assert.equal(after.get().controlVector.alpha, 3, 'turning it off leaves the strength alone')

    // A file that is no longer on this Mac cannot be put back in force.
    await fs.rm(other)
    await assert.rejects(svc.setControlVector({ enabled: true, alpha: 1 }), /no longer on this Mac/)
    const cleared = await svc.clearControlVector()
    assert.equal(cleared.file, null)
    assert.equal(cleared.enabled, false)
  } finally { await svc.unload().catch(() => {}); await h.cleanup() }
}

main().then(() => process.exit(0), (error) => { console.error(error); process.exit(1) })
