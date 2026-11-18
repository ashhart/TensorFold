import assert from 'node:assert/strict'
import { syntheticBenchmarkReceipt } from './fixtures'
import { parseBenchmarkReceipt, benchmarkCells, benchmarkProtocolComplete } from './receipt'
import { assertBundledBenchmarkVersion, BENCHMARK_ENGINE_VERSION } from './version'

const receipt = syntheticBenchmarkReceipt()
assert.equal(assertBundledBenchmarkVersion(BENCHMARK_ENGINE_VERSION), BENCHMARK_ENGINE_VERSION)
for (const version of ['1.0.12', '9.9.9', null]) assert.throws(() => assertBundledBenchmarkVersion(version), /bundled TensorFold/, 'live benchmark admission requires the approved pinned engine')
const parsed = parseBenchmarkReceipt(JSON.stringify(receipt), { model: receipt.model.repo_id, serial: false })
assert.equal(benchmarkProtocolComplete(parsed), true)
assert.equal(benchmarkCells(parsed)[0].successful, 5)
assert.equal(benchmarkCells(parsed)[0].deliveryTokensPerSecond?.median, 100)
assert.equal(benchmarkCells(parsed)[0].firstTextSeconds?.median, 0.30000000000000004)
assert.equal(parsed.samples[0].server_decode_seconds, null, 'missing engine duration stays unknown')
const current = structuredClone(receipt); current.runtime.tensorfold_version = '1.1.0'
assert.equal(parseBenchmarkReceipt(JSON.stringify(current), { model: current.model.repo_id, serial: false, engineVersion: '1.1.0' }).runtime.tensorfold_version, '1.1.0')
const historical = structuredClone(receipt); historical.runtime.tensorfold_version = '1.0.12'
assert.equal(parseBenchmarkReceipt(JSON.stringify(historical)).runtime.tensorfold_version, '1.0.12', 'reviewed historical receipts remain readable after the engine update')
assert.throws(() => parseBenchmarkReceipt(JSON.stringify(historical), { model: historical.model.repo_id, serial: false, engineVersion: '1.1.0' }), /invalid|unsupported/, 'a historical receipt cannot substitute for a live run on the new engine')
for (const mutate of [
  (value: any) => { value.prompt = 'SYNTHETIC_PRIVATE_PROMPT' },
  (value: any) => { value.model.repo_id = '/private/operator/model' },
  (value: any) => { value.hardware.cpu_model = 'person@example.com' },
  (value: any) => { value.samples[0].delivery_tps = 1000 },
  (value: any) => { value.samples[0].seed = 999 },
  (value: any) => { value.samples.push(value.samples[0]) },
  (value: any) => { value.samples[0].completion_tokens = 257 },
  (value: any) => { value.created_at = '2026-02-30T00:00:00Z' },
  (value: any) => { value.suite.sha256 = '0'.repeat(64) },
  (value: any) => { value.settings.managed = false },
  (value: any) => { value.settings.context_tokens = 8192 },
  (value: any) => { value.runtime.dependencies.private_path = '/private/operator' },
  (value: any) => { value.runtime.tensorfold_version = '9.9.9' },
  (value: any) => { value.runtime.tensorfold_version = '1.1.0-preview' },
]) {
  const invalid = structuredClone(receipt); mutate(invalid)
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(invalid)), /invalid|unsupported/)
}
assert.throws(() => parseBenchmarkReceipt(JSON.stringify(receipt).replace('"schema_version":1', '"schema_version":1,"schema_version":1')), /invalid/)
assert.throws(() => parseBenchmarkReceipt(JSON.stringify(receipt), { model: 'example-org/other-model', serial: false }), /invalid/)
const early = structuredClone(receipt)
Object.assign(early.samples[0], { status: 'early_eos', completion_tokens: 12, delivery_seconds: 0.11, delivery_tps: 100 })
const earlyParsed = parseBenchmarkReceipt(JSON.stringify(early))
assert.equal(benchmarkProtocolComplete(earlyParsed), false)
assert.equal(benchmarkCells(earlyParsed)[0].earlyEos, 1)
assert.equal(benchmarkCells(earlyParsed)[0].deliveryTokensPerSecond, null, 'failures cannot disappear from a mean or median')
const warmup = structuredClone(receipt)
warmup.samples.push({ ...warmup.samples[0], repeat: -1, status: 'error', error_code: 'timeout', completion_tokens: null, delivery_seconds: null, delivery_tps: null })
assert.equal(benchmarkCells(parseBenchmarkReceipt(JSON.stringify(warmup)))[0].failedWarmups, 1)
assert.equal(benchmarkCells(warmup)[0].complete, false)
const buffered = structuredClone(receipt)
Object.assign(buffered.samples[0], { status: 'unmeasured', delivery_seconds: null, delivery_tps: null, error_code: 'unmeasured_delivery' })
assert.equal(benchmarkCells(parseBenchmarkReceipt(JSON.stringify(buffered)))[0].unmeasured, 1)
const unknown = structuredClone(receipt); unknown.hardware.gpus = []; unknown.hardware.gpu_count = null; unknown.hardware.source = 'unavailable'
assert.equal(benchmarkProtocolComplete(parseBenchmarkReceipt(JSON.stringify(unknown))), false)
console.log('Benchmark strict receipt, privacy, exact samples, complete-only aggregates, earlyEOS, warmup errors and unknown metrics passed')
