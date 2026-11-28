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
  assert.throws(() => mergeControlVector(current, { enabled: 'yes' }), /on or off/)
  // A field the caller did not name keeps its value: a new strength does not switch the vector off.
  assert.deepEqual(mergeControlVector(current, { alpha: 3 }), { enabled: true, alpha: 3 })
  assert.deepEqual(mergeControlVector(current, { enabled: false }), { enabled: false, alpha: 2 })
  // A saved vector that names no file cannot be in force, whatever it says.
  assert.deepEqual(readControlVector({ enabled: true, file: null, alpha: 3 }), { enabled: false, file: null, alpha: 3 })

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
  } finally { await plainSvc.unload().catch(() => {}); await plain.cleanup() }

  // An engine that offers --glp takes the file the operator chose.
  const h = await makeHarness({ FAKE_ENGINE_GLP: '1' })
  const svc = service(h)
  try {
    await seedModel(h.hub, MODEL)
    await svc.load(MODEL)
    const vector = path.join(h.root, 'style-vector.gguf')
    await fs.writeFile(vector, 'glp')
    await assert.rejects(svc.setControlVector({ enabled: true, alpha: 1 }), /Choose a control vector file first/)
    await assert.rejects(svc.setControlVector({ enabled: 'yes' }), /on or off/)
    const chosen = await svc.chooseControlVector(vector)
    assert.equal(chosen.supported, true)
    assert.equal(chosen.enabled, false, 'choosing a file does not put it in force')
    assert.equal((await h.records()).filter((row) => row.kind === 'serve').length, 1, 'choosing a file does not restart the model')

    const enabled = await svc.setControlVector({ enabled: true, alpha: 2 })
    assert.equal(enabled.enabled, true)
    assert.equal(enabled.alpha, 2)
    const served = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(served.length, 2, 'putting a vector in force restarts the model')

    // A new strength on its own leaves the vector in force.
    const stronger = await svc.setControlVector({ alpha: 3 })
    assert.equal(stronger.enabled, true)
    assert.equal(stronger.alpha, 3)
    const restrung = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(restrung.length, 3, 'a new strength restarts the model with it')

    // Another file takes effect at once when a vector is already in force.
    const other = path.join(h.root, 'other.gguf')
    await fs.writeFile(other, 'glp')
    const swapped = await svc.chooseControlVector(other)
    assert.equal(swapped.file, other)
    assert.equal(swapped.enabled, true)
    const exchanged = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(exchanged.length, 4, 'swapping the file restarts the model with it')

    // The choice survives a restart of the app: it is on disk, not in memory.
    const stored = new EnginePreferenceStore(h.root)
    assert.equal(stored.get().controlVector.file, other)
    assert.equal(stored.get().controlVector.alpha, 3)

    await svc.setControlVector({ enabled: false })
    const off = (await h.records()).filter((row) => row.kind === 'serve')
    assert.equal(off.length, 5)
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

  // The picker shares the feature-change queue: a strength or state change still pending when
  // the operator picks cannot be lost to the values the picker read before that change landed.
  const overlap = await makeHarness({ FAKE_ENGINE_GLP: '1' })
  const held = service(overlap)
  try {
    await seedModel(overlap.hub, MODEL)
    await held.load(MODEL)
    const first = path.join(overlap.root, 'first.gguf')
    const second = path.join(overlap.root, 'second.gguf')
    await fs.writeFile(first, 'glp')
    await fs.writeFile(second, 'glp')
    await held.chooseControlVector(first)
    await held.setControlVector({ enabled: true, alpha: 2 })
    // Hold the queued restart so the picker reads the preferences before that change lands.
    const reconfigure = held.supervisor.reconfigure.bind(held.supervisor)
    let release = () => {}
    let gate = Promise.resolve()
    held.supervisor.reconfigure = (apply, rollback) => reconfigure(async () => { await gate; await apply() }, rollback)
    const hold = () => { gate = new Promise<void>((resolve) => { release = resolve }) }
    const drain = () => new Promise<void>((resolve) => setImmediate(resolve))

    hold()
    const alpha = held.setControlVector({ alpha: 5 })
    const picked = held.chooseControlVector(second)
    await drain()
    release()
    const [, pickedState] = await Promise.all([alpha, picked])
    assert.equal(pickedState.alpha, 5, 'the picker sees the pending strength, not the one it replaced')
    assert.deepEqual(new EnginePreferenceStore(overlap.root).get().controlVector, { enabled: true, file: second, alpha: 5 }, 'the pending strength and the picked file both survive')

    hold()
    const off = held.setControlVector({ enabled: false })
    const swapped = held.chooseControlVector(first)
    await drain()
    release()
    const [, swappedState] = await Promise.all([off, swapped])
    assert.equal(swappedState.enabled, false, 'the picker sees the pending state, not the one it replaced')
    assert.deepEqual(new EnginePreferenceStore(overlap.root).get().controlVector, { enabled: false, file: first, alpha: 5 }, 'the pending state and the picked file both survive')
  } finally { await held.unload().catch(() => {}); await overlap.cleanup() }
}

main().then(() => process.exit(0), (error) => { console.error(error); process.exit(1) })
