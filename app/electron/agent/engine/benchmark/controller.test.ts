import assert from 'node:assert/strict'
import { ModelBenchmarks } from './controller'
import { syntheticBenchmarkReceipt } from './fixtures'
import type { SavedModelBenchmark } from './storage'
import type { BenchmarkProcessResult } from './process'
import { BENCHMARK_ENGINE_VERSION } from './version'

const input = { id: 'benchmark_synthetic_0001', model: 'example-org/synthetic-model' }
function fixture() {
  let records: SavedModelBenchmark[] = []
  let valid = true; let missing = false; let saveFails = false; let listFails = false; let restoreFails = false
  let pauses = 0; let restores = 0; let releases = 0; let processes = 0; let cleanups = 0
  let finish!: (result: BenchmarkProcessResult) => void
  const store = { list: async () => { if (listFails) throw new Error('/private/operator/disk'); return structuredClone(records) }, save: async (record: SavedModelBenchmark) => { if (saveFails) throw new Error('/private/operator/disk'); records = [structuredClone(record), ...records.filter(item => item.snapshot.id !== record.snapshot.id)] } }
  const deps = {
    assertAllowed: () => { if (!valid) throw new Error('Synthetic active work') },
    validateModel: async () => {},
    reserve: async () => ({ previousModel: 'example-org/prior-model', pause: async () => { pauses++ }, restore: async () => { restores++; if (restoreFails) throw new Error('/private/operator/restore') }, release: () => { releases++ } }),
    workspace: async () => ({ directory: '/private/synthetic-workspace', receiptPath: '/private/synthetic-workspace/receipt.json', remove: async () => { cleanups++ } }),
    process: () => { processes++; return { done: new Promise<BenchmarkProcessResult>(resolve => { finish = resolve }), cancel: () => finish({ code: 130, cancelled: true, overflow: false, failed: false }) } },
    readReceipt: async () => missing ? null : syntheticBenchmarkReceipt(input.model),
    store,
  }
  const controller = new ModelBenchmarks(deps)
  return { controller, deps, store, setValid: (value: boolean) => { valid = value }, missing: () => { missing = true }, failSave: () => { saveFails = true }, recoverSave: () => { saveFails = false }, failList: () => { listFails = true }, failRestore: () => { restoreFails = true }, finish: (code = 0) => finish({ code, cancelled: false, overflow: false, failed: false }), stats: () => ({ pauses, restores, releases, processes, cleanups, records: records.length }) }
}
async function running(f: ReturnType<typeof fixture>) {
  for (let i = 0; i < 100; i++) { if ((await f.controller.snapshot())?.state === 'running') return; await new Promise(resolve => setTimeout(resolve, 1)) }
  throw new Error('Synthetic benchmark did not reach running')
}
const normal = fixture()
const first = await normal.controller.start(input)
assert.equal(first.state, 'preparing')
await assert.rejects(normal.controller.start({ ...input, id: 'benchmark_synthetic_0002' }), /current local benchmark/)
await running(normal)
assert.equal(normal.controller.cancel('benchmark_not_the_owner'), false)
const drain = normal.controller.settle(input.id)
normal.finish(); await drain
assert.equal((await normal.controller.snapshot(input.id))?.state, 'complete')
assert.equal((await normal.controller.snapshot())?.metadata?.runtime.tensorfold_version, BENCHMARK_ENGINE_VERSION)
assert.equal((await normal.controller.snapshot())?.saved, true)
assert.equal((await normal.controller.receiptForExport(input.id)).samples.length, 20)
assert.deepEqual(normal.stats(), { pauses: 1, restores: 1, releases: 1, processes: 1, cleanups: 1, records: 1 })
const restarted = new ModelBenchmarks(normal.deps)
assert.equal((await restarted.snapshot())?.id, input.id)
await restarted.start(input); await restarted.settle(input.id)
assert.equal((await normal.store.list())[0].receipt?.samples.length, 20, 'duplicate IDs cannot replace saved results')
assert.equal((await restarted.receiptForExport(input.id)).samples.length, 20)
const denied = fixture(); denied.setValid(false)
await assert.rejects(denied.controller.start(input), /active work/)
assert.equal(denied.stats().pauses, 0)
await assert.rejects(denied.controller.start({ ...input, model: '/private/operator/model' }), /linked-folder/)
const cancelled = fixture(); await cancelled.controller.start(input); await running(cancelled)
assert.equal(cancelled.controller.cancel(input.id), true); await cancelled.controller.settle(input.id)
assert.equal((await cancelled.controller.snapshot())?.state, 'cancelled')
assert.equal(cancelled.stats().restores, 1)
const restore = fixture(); restore.failRestore(); await restore.controller.start(input); await running(restore); restore.finish(); await restore.controller.settle()
assert.equal((await restore.controller.snapshot())?.restored, false)
assert.equal((await restore.controller.snapshot())?.state, 'failed')
assert.equal(JSON.stringify(await restore.controller.snapshot()).includes('/private/operator'), false)
const disk = fixture(); disk.failSave(); await disk.controller.start(input); await running(disk); disk.finish(); await disk.controller.settle()
assert.equal((await disk.controller.snapshot())?.saved, false)
assert.ok((await disk.controller.snapshot())?.storageError)
assert.equal((await disk.controller.receiptForExport(input.id)).samples.length, 20)
const nextUnsavedId = 'benchmark_next_unsaved_result'
await disk.controller.start({ ...input, id: nextUnsavedId }); await running(disk); disk.finish(); await disk.controller.settle(nextUnsavedId)
assert.equal((await disk.controller.receiptForExport(input.id)).samples.length, 20, 'starting a later run cannot discard the only copy of an unsaved receipt')
assert.ok((await disk.controller.history()).some(item => item.id === input.id && !item.saved))
const oldReceipt = await disk.controller.receiptForExport(input.id); oldReceipt.samples.length = 0
const oldSnapshot = (await disk.controller.snapshot(input.id))!; oldSnapshot.warnings.push('/private/operator/modified')
assert.equal((await disk.controller.receiptForExport(input.id)).samples.length, 20, 'export never mutates the retained receipt')
assert.equal(JSON.stringify(await disk.controller.snapshot(input.id)).includes('/private/operator'), false)
disk.recoverSave()
await disk.controller.start({ ...input, id: 'benchmark_saved_after_disk_recovery' }); await running(disk); disk.finish(); await disk.controller.settle()
assert.equal((await disk.controller.history()).length, 3, 'saved history merges with earlier unsaved results')
disk.failSave()
for (let index = 2; index < 50; index++) {
  await disk.controller.start({ ...input, id: `benchmark_unsaved_bounded_${index}` }); await running(disk); disk.finish(); await disk.controller.settle()
}
await assert.rejects(disk.controller.start({ ...input, id: 'benchmark_unsaved_would_evict' }), /Unsaved benchmark history is full/)
assert.equal((await disk.controller.receiptForExport(input.id)).samples.length, 20, 'the bounded fallback cannot silently evict receipts')
disk.failList()
const memoryHistory = await disk.controller.history()
assert.equal(memoryHistory.length, 50)
assert.ok(memoryHistory.every(item => !item.saved && item.warnings.some(warning => warning.includes('Saved benchmark history is unavailable'))))
assert.equal((await disk.controller.receiptForExport(input.id)).samples.length, 20, 'old unsaved receipts remain exportable when disk reads fail too')
const absent = fixture(); absent.missing(); await absent.controller.start(input); await running(absent); absent.finish(1); await absent.controller.settle()
assert.equal((await absent.controller.snapshot())?.receiptAvailable, false)
assert.equal((await absent.controller.snapshot())?.state, 'failed')
await assert.rejects(absent.controller.receiptForExport(input.id), /no finished receipt/)
const quitting = fixture(); await quitting.controller.start(input); await running(quitting)
quitting.controller.dispose(false); await quitting.controller.settle(input.id)
assert.equal((await quitting.controller.snapshot())?.state, 'cancelled')
assert.equal((await quitting.controller.snapshot())?.restored, false)
assert.equal(quitting.stats().restores, 0, 'app closing saves cancellation without reloading the previous model')
await assert.rejects(quitting.controller.start({ ...input, id: 'benchmark_after_closing' }), /current local benchmark/)
console.log('Benchmark controller exact-job admission/drain, cancel, restart/export, storage and restoration failures passed')
