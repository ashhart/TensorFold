import assert from 'node:assert/strict'
import { mkdtemp, readFile, stat, writeFile, chmod, symlink, link, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { syntheticBenchmarkReceipt } from './fixtures'
import { benchmarkCells } from './receipt'
import type { ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-benchmark-storage-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECRET_KEY_BASE64 = Buffer.alloc(32, 17).toString('base64')
const { createModelBenchmarkStore, benchmarkWorkspace, readBenchmarkReceiptFile } = await import('./storage')
const receipt = syntheticBenchmarkReceipt()
const snapshot: ModelBenchmarkSnapshot = { id: 'benchmark_synthetic_stored', model: receipt.model.repo_id, serial: false, state: 'complete', revision: 2, startedAt: 1, finishedAt: 2, progress: { completedRequests: 24, totalRequests: 24, fixture: 'chat', temperature: 0, repeat: 4, phase: 'measured' }, cells: benchmarkCells(receipt), receiptAvailable: true, protocolComplete: true, previousModel: null, restored: true, restoreError: null, error: null, storageError: null, warnings: ['Results stay local.'], saved: true }
try {
  const store = createModelBenchmarkStore(home)
  assert.deepEqual(await store.list(), [])
  await store.save({ snapshot, receipt })
  const file = path.join(home, 'benchmarks', 'history.protected.json')
  const protectedBytes = await readFile(file, 'utf8')
  assert.ok(!protectedBytes.includes(receipt.model.repo_id))
  assert.equal((await stat(file)).mode & 0o777, 0o600)
  const reloaded = await createModelBenchmarkStore(home).list()
  assert.equal(reloaded[0].snapshot.metadata?.hardware.cpu_model, 'Synthetic CPU')
  assert.equal(reloaded[0].snapshot.cells[0].deliveryTokensPerSecond?.median, 100)
  assert.equal(reloaded[0].receipt?.samples.length, 20)
  const historical = structuredClone(receipt); historical.runtime.tensorfold_version = '1.0.12'
  const historicalSnapshot = { ...snapshot, id: 'benchmark_historical_stored' }
  await store.save({ snapshot: historicalSnapshot, receipt: historical })
  const afterUpgrade = await createModelBenchmarkStore(home).list()
  assert.equal(afterUpgrade.find(record => record.snapshot.id === historicalSnapshot.id)?.receipt?.runtime.tensorfold_version, '1.0.12', 'engine upgrades retain validated historical receipts')
  await writeFile(file, '[]', { mode: 0o600 })
  await assert.rejects(store.list(), /protected profile/)
  await assert.rejects(store.save({ snapshot, receipt }), /protected profile/)
  assert.equal(await readFile(file, 'utf8'), '[]', 'corrupt or plaintext history is never overwritten')
  await writeFile(file, protectedBytes, { mode: 0o600 })
  const job = await benchmarkWorkspace(home)
  await writeFile(job.receiptPath, JSON.stringify(receipt), { mode: 0o600 })
  assert.equal((await readBenchmarkReceiptFile(job.receiptPath, { model: snapshot.model, serial: false }))?.samples.length, 20)
  const symbolic = path.join(job.directory, 'symlink.json'); await symlink(job.receiptPath, symbolic)
  await assert.rejects(readBenchmarkReceiptFile(symbolic, { model: snapshot.model, serial: false }), /safely|unsafe/)
  const hard = path.join(job.directory, 'hard.json'); await link(job.receiptPath, hard)
  await assert.rejects(readBenchmarkReceiptFile(hard, { model: snapshot.model, serial: false }), /unsafe/)
  await rm(hard)
  await chmod(job.receiptPath, 0o644)
  await assert.rejects(readBenchmarkReceiptFile(job.receiptPath, { model: snapshot.model, serial: false }), /permissions/)
  await job.remove()
  await assert.rejects(store.save({ snapshot: { ...snapshot, error: '/private/operator/path' }, receipt }), /invalid/)
  assert.equal((await store.list()).length, 1)
} finally { await rm(home, { recursive: true, force: true }) }
console.log('Benchmark protected atomic history/reload, recorded metadata, corrupt-state preservation and unsafe receipt refusal passed')
