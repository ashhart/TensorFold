import assert from 'node:assert/strict'
import path from 'node:path'
import { makeHarness, seedModel, waitFor } from '../test-fixtures/harness'
import type { EngineModelGenerationSettings } from '../../types'
import { BENCHMARK_ENGINE_VERSION } from './version'

const h = await makeHarness()
process.env.VONTRA_HOME = h.root
process.env.VONTRA_SECRET_KEY_BASE64 = Buffer.alloc(32, 23).toString('base64')
const { EngineService } = await import('../service')
const model = 'mlx-community/Qwen3.5-2B-MLX-4bit'
await seedModel(h.hub, model)
let profile: EngineModelGenerationSettings = { contextTokens: 4096, maxOutputTokens: 128, temperature: 0.25, topP: 0.9, topK: 20, minP: 0, speculativeDecoding: 'off' }
const binary = { command: process.execPath, args: [path.join(__dirname, 'fake-benchmark.mjs'), 'scenario:delayed'] }
const service = new EngineService({ binaryPath: '/synthetic/configured-binary', binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, modelSettings: () => profile, emit() {} })
try {
  await service.status()
  await service.load(model)
  const before = service.supervisor.status()
  assert.equal(before.contextTokens, 4096)
  let block = false
  service.setBenchmarkAdmission(() => { if (block) throw new Error('Synthetic active chat') })
  block = true
  await assert.rejects(service.startBenchmark({ id: 'benchmark_service_denied', model }), /active chat/)
  block = false
  const run = { id: 'benchmark_service_positive', model }
  await service.startBenchmark(run)
  await waitFor(async () => (await service.benchmarkStatus(run.id))?.state === 'running', 5000, 'synthetic benchmark process')
  assert.equal(service.isLearningBusy(), true)
  await assert.rejects(service.load(model), /benchmark/)
  await assert.rejects(service.unload(), /benchmark/)
  assert.throws(() => service.tryModel(model), /benchmark/)
  await assert.rejects(service.teach('Synthetic lesson'), /benchmark/)
  await assert.rejects(service.setLearning(false), /benchmark/)
  assert.throws(() => service.compareModels({ id: 'comparison_synthetic_blocked', models: [model, 'example-org/other'], prompt: 'Synthetic' }), /benchmark/)
  profile = { ...profile, contextTokens: 8192, temperature: 0.9 }
  await service.settleBenchmarks(run.id)
  const result = await service.benchmarkStatus(run.id)
  assert.equal(result?.state, 'complete')
  assert.equal(result?.restored, true)
  assert.equal(result?.metadata?.runtime.tensorfold_version, BENCHMARK_ENGINE_VERSION)
  assert.equal(service.supervisor.status().model, model)
  assert.equal(service.supervisor.status().contextTokens, 4096, 'restoration uses the profile actually loaded before reservation, not a subsequently changed callback')
  assert.equal(service.isLearningBusy(), false)
  assert.equal((await service.benchmarkReceiptForExport(run.id)).samples.length, 20)
  const fresh = new EngineService({ binaryPath: '/synthetic/configured-binary', binary, appHome: h.root, childEnv: h.env, emit() {} })
  assert.equal((await fresh.benchmarkHistory())[0].id, run.id)
  assert.equal((await fresh.benchmarkStatus())?.metadata?.hardware.cpu_model, 'Synthetic CPU')
  fresh.shutdown()
  await service.startBenchmark({ id: 'benchmark_service_absent', model: 'example-org/not-installed' })
  await service.settleBenchmarks('benchmark_service_absent')
  assert.equal((await service.benchmarkStatus('benchmark_service_absent'))?.state, 'failed')
  assert.equal(service.supervisor.status().model, model, 'invalid models do not pause or replace the existing runtime')
  for (const scenario of ['old-version', 'unknown-version', 'mismatched-receipt']) {
    const fixture = new EngineService({ binaryPath: '/synthetic/configured-binary', binary: { command: process.execPath, args: [path.join(__dirname, 'fake-benchmark.mjs'), `scenario:${scenario}`] }, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 10, emit() {} })
    const id = `benchmark_service_${scenario.replaceAll('-', '_')}`
    const starts = (await h.records()).filter(record => record.kind === 'serve').length
    try {
      await fixture.startBenchmark({ id, model }); await fixture.settleBenchmarks(id)
      const refused = await fixture.benchmarkStatus(id)
      assert.equal(refused?.state, 'failed'); assert.equal(refused?.receiptAvailable, false); assert.equal(refused?.metadata, undefined)
      if (scenario !== 'mismatched-receipt') assert.equal((await h.records()).filter(record => record.kind === 'serve').length, starts, 'an unapproved engine is rejected before reserving or launching a model')
      else assert.match(refused?.error ?? '', /receipt failed validation/, 'a valid historical receipt cannot replace a current-engine run')
    } finally { fixture.shutdown() }
  }
  await service.prepareToQuit()
} finally { service.shutdown(); await h.cleanup() }
console.log('Actual EngineService synthetic native process: admission, engine locks, exact loaded profile restoration and persistent results passed')
