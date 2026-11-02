import assert from 'node:assert/strict'
import path from 'node:path'
import { EngineService } from './service'
import { makeHarness, seedModel } from './test-fixtures/harness'
import { syntheticBenchmarkReceipt } from './benchmark/fixtures'
import { parseBenchmarkReceipt } from './benchmark/receipt'
import { assertCompatibleBenchmarkVersion } from './benchmark/version'
const h = await makeHarness()
const repo = 'Fixture/Model'
const service = new EngineService({ binaryPath: '/fixture/unused', binary: h.binary, appHome: h.root,
  childEnv: h.env, totalMemoryBytes: 16 * 1024 ** 3, pollMs: 20, emit: () => {} })
try {
  await seedModel(h.hub, repo)
  await service.load(repo)
  const lease = await service.supervisor.acquire(repo, { start: false })
  let selected = 'bundled'
  await assert.rejects(service.reconfigureRuntime(async () => { selected = 'updated' }, async () => { selected = 'bundled' }), /active TensorFold requests/)
  assert.equal(selected, 'bundled'); assert.ok(service.supervisor.live(repo))
  lease.release()
  const before = service.supervisor.status().startedAt
  await service.reconfigureRuntime(async () => { selected = 'updated' }, async () => { selected = 'bundled' })
  assert.equal(selected, 'updated'); assert.ok(service.supervisor.live(repo))
  assert.notEqual(service.supervisor.status().startedAt, before, 'an idle loaded model restarts on the selected runtime')
  const sequence = await service.supervisor.reserveSequence('fixture comparison')
  await assert.rejects(service.reconfigureRuntime(async () => {}, async () => {}), /fixture comparison/)
  sequence.release()
  await assert.rejects(service.reconfigureRuntime(async () => { selected = 'broken'; throw new Error('activation failed') },
    async () => { selected = 'updated' }), /activation failed/)
  assert.equal(selected, 'updated'); assert.ok(service.supervisor.live(repo), 'failed activation preserves the previous running model')
  await assert.rejects(service.reconfigureRuntime(async () => { h.env.FAKE_SERVE_FAIL = 'Synthetic updated runtime load failure' },
    async () => { delete h.env.FAKE_SERVE_FAIL }), /Synthetic updated runtime load failure/)
  assert.ok(service.supervisor.live(repo), 'a candidate load failure restarts the previous model after rollback')
  assert.equal(assertCompatibleBenchmarkVersion('1.1.7'), '1.1.7')
  const receipt = syntheticBenchmarkReceipt(); receipt.runtime.tensorfold_version = '1.1.7'
  assert.equal(parseBenchmarkReceipt(JSON.stringify(receipt), { model: receipt.model.repo_id, serial: false, engineVersion: '1.1.7' }).runtime.tensorfold_version, '1.1.7')
  receipt.suite.sha256 = 'f'.repeat(64)
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(receipt)), /invalid|unsupported/, 'new engine versions cannot change the frozen benchmark suite')
  assert.throws(() => assertCompatibleBenchmarkVersion('2.0.0'), /protocol 1/)
} finally { await service.supervisor.stop(); await service.shutdown(true); await h.cleanup() }
console.log('Runtime switching preserves leases, model recovery and version-independent benchmark schema checks.')
