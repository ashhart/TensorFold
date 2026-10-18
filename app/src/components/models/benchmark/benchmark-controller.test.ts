import assert from 'node:assert/strict'
import { createBenchmarkController, type BenchmarkPort } from './benchmark-controller'
import { fixtureCell, fixtureRun } from './benchmark-test-fixtures'

function deferred<T>() { let resolve!: (value: T) => void, reject!: (error: unknown) => void; const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail }); return { promise, resolve, reject } }
const calls: any[] = []
const lastCall = (name: string) => calls.filter((call) => call[0] === name).at(-1)
let latest = null as ReturnType<typeof fixtureRun> | null
let history: ReturnType<typeof fixtureRun>[] = []
let statusWait: ReturnType<typeof deferred<ReturnType<typeof fixtureRun> | null>> | null = null
let startWait: ReturnType<typeof deferred<ReturnType<typeof fixtureRun>>> | null = null
let statusFailure = false, historyFailure = false, exportSaved = true
const port: BenchmarkPort = {
  status: async (id) => { calls.push(['status', id]); if (statusFailure) throw Error('private-error-fixture'); return statusWait ? statusWait.promise : latest },
  history: async () => { calls.push(['history']); if (historyFailure) throw Error('private-history-fixture'); return history },
  start: async (input) => { calls.push(['start', input]); return startWait ? startWait.promise : fixtureRun({ id: input.id, model: input.model, serial: !!input.serial, state: 'preparing', revision: 0 }) },
  cancel: async (id) => { calls.push(['cancel', id]); return true },
  export: async (id) => { calls.push(['export', id]); return { saved: exportSaved } },
  copy: async (text) => { calls.push(['copy', text]); return { ok: true } },
}
const controller = createBenchmarkController(port, () => {})
await controller.start({ id: 'not-yet-loaded', model: 'fixture/model' })
assert.equal(calls.length, 0, 'a run cannot start before active status admission is known')
await controller.refresh()
assert.equal(controller.snapshot().selected, null)
assert.equal(controller.snapshot().loading, false)
assert.deepEqual(calls.map((call) => call[0]), ['status', 'history'], 'opening empty results performs only read operations')
startWait = deferred()
const input = { id: 'benchmark_new_fixture_012345', model: 'fixture/cached-model', serial: true }
const starting = controller.start(input)
await controller.start({ ...input, id: 'benchmark_duplicate_fixture' })
assert.equal(calls.filter((call) => call[0] === 'start').length, 1, 'imperative action lock rejects duplicate Run clicks')
startWait.resolve(fixtureRun({ ...input, state: 'preparing', revision: 0 })); await starting
assert.equal(controller.snapshot().active?.id, input.id)
latest = fixtureRun({ ...input, state: 'running', revision: 2 })
statusWait = deferred()
const polling = controller.poll()
await controller.poll()
assert.equal(calls.filter((call) => call[0] === 'status').length, 2, 'concurrent polls issue only one status request')
await controller.cancel(); await controller.cancel()
assert.deepEqual(calls.filter((call) => call[0] === 'cancel'), [['cancel', input.id]])
statusWait.resolve(fixtureRun({ ...input, state: 'preparing', revision: 0 })); await polling; statusWait = null
assert.equal(controller.snapshot().cancellationRequested, true, 'a late pre-cancellation poll cannot reset cancellation state')
controller.dispose()
assert.equal(calls.filter((call) => call[0] === 'cancel').length, 1, 'closing the drawer never cancels the host job')

history = [fixtureRun({ id: 'benchmark_history_fixture' })]
const reopened = createBenchmarkController(port, () => {})
await reopened.refresh()
assert.equal(reopened.snapshot().active?.id, input.id, 'reopening recovers the same active host job')
assert.equal(reopened.snapshot().selected?.id, input.id)
reopened.select(history[0].id)
latest = fixtureRun({ ...input, state: 'running', revision: 3 })
const historyReads = calls.filter((call) => call[0] === 'history').length
await reopened.poll()
assert.equal(reopened.snapshot().selected?.id, history[0].id, 'active polling never replaces the history item the user opened')
assert.equal(calls.filter((call) => call[0] === 'history').length, historyReads, 'ongoing measurements do not repeatedly load encrypted history')
latest = fixtureRun({ ...input, state: 'running', revision: 1 }); await reopened.poll()
assert.equal(reopened.snapshot().active?.revision, 3, 'out-of-order status cannot roll a run back')
reopened.select(input.id)
latest = fixtureRun({ ...input, state: 'failed', revision: 4, restored: false, restoreError: 'restore-fixture', protocolComplete: false, cells: [{ ...fixtureCell(), complete: false, successful: 2, errors: 3 }], saved: false, storageError: 'storage-fixture' })
history = [latest, history[0]]
await reopened.poll()
assert.equal(reopened.snapshot().active, null)
assert.equal(reopened.snapshot().selected?.state, 'failed')
assert.equal(reopened.snapshot().selected?.restored, false)
await reopened.copy()
const copy = lastCall('copy')[1]
assert.match(copy, /incomplete, 2\/5 valid samples/)
assert.doesNotMatch(copy, /25 tokens|restore-fixture|storage-fixture/, 'basic copy omits incomplete medians and diagnostic text')
await reopened.export()
assert.deepEqual(lastCall('export'), ['export', input.id], 'unsaved but available validated receipt can still be recovered by local export')
exportSaved = false; await reopened.export(); assert.match(reopened.snapshot().notice, /cancelled/)
reopened.dispose()

statusFailure = true
const failure = createBenchmarkController(port, () => {})
await failure.refresh()
assert.match(failure.snapshot().error, /status could not be read/)
assert.doesNotMatch(failure.snapshot().error, /private-error/)
const starts = calls.filter((call) => call[0] === 'start').length
await failure.start(input)
assert.equal(calls.filter((call) => call[0] === 'start').length, starts, 'unknown host activity after a read failure never enables Run')
statusFailure = false; latest = fixtureRun({ ...input, state: 'running', revision: 8 }); historyFailure = true
await failure.refresh()
assert.equal(failure.snapshot().active?.revision, 8, 'history failure does not hide an active job')
assert.match(failure.snapshot().error, /history/)
historyFailure = false; failure.dispose()

const oldStatus = deferred<ReturnType<typeof fixtureRun> | null>(), freshStatus = deferred<ReturnType<typeof fixtureRun> | null>()
let statusNumber = 0
const reactivated = createBenchmarkController({ ...port, status: () => (++statusNumber === 1 ? oldStatus.promise : freshStatus.promise) }, () => {})
const oldRead = reactivated.refresh()
reactivated.dispose(); reactivated.activate()
const freshRead = reactivated.refresh()
assert.equal(statusNumber, 2, 'effect replay can start a new owned read instead of remaining stuck behind a disposed request')
freshStatus.resolve(fixtureRun({ revision: 10 })); await freshRead
oldStatus.resolve(fixtureRun({ revision: 1 })); await oldRead
assert.equal(reactivated.snapshot().selected?.revision, 10)
reactivated.dispose()
// A storage-failure session can retain more than20 receipts; none may become unreachable.
latest = null
history = Array.from({ length: 30 }, (_, index) => fixtureRun({ id: `benchmark_retained_fixture_${index}`, saved: false, storageError: 'Synthetic profile write failure' }))
const retained = createBenchmarkController(port, () => {})
await retained.refresh()
assert.equal(retained.snapshot().history.length, 30, 'all host-retained results remain selectable instead of silently truncating the list')
retained.select(history[29].id)
assert.equal(retained.snapshot().selected?.id, history[29].id)
await retained.export()
assert.deepEqual(lastCall('export'), ['export', history[29].id], 'an older unsaved receipt remains exportable through the GUI')
retained.dispose()
console.log('Benchmark controller: empty/error recovery, duplicate Run, serial polls, precise cancel, reopen/history, monotonic revisions, honest failed cells, guarded copy/export and effect replay passed')
