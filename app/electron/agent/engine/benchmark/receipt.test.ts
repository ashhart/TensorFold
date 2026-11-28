import assert from 'node:assert/strict'
import { syntheticBenchmarkReceipt } from './fixtures'
import { parseBenchmarkReceipt, benchmarkCells, benchmarkProtocolComplete, parseBenchmarkDetails, checkedBenchmarkDetails } from './receipt'
import { BENCHMARK_SUITES } from '../../../../shared/model-benchmark'
import { assertBundledBenchmarkVersion, BENCHMARK_ENGINE_VERSION } from './version'

const receipt = syntheticBenchmarkReceipt(); receipt.settings.thinking_mode = 'server_default'
assert.equal(assertBundledBenchmarkVersion(BENCHMARK_ENGINE_VERSION), BENCHMARK_ENGINE_VERSION)
for (const version of ['1.0.12', '9.9.9', null]) assert.throws(() => assertBundledBenchmarkVersion(version), /bundled TensorFold/, 'live benchmark admission requires the approved pinned engine')
const parsed = parseBenchmarkReceipt(JSON.stringify(receipt), { model: receipt.model.repo_id, serial: false })
assert.deepEqual(parsed, receipt, 'the current receipt preserves cache, thinking, finish reason and route')
const fp8Variant = structuredClone(receipt); fp8Variant.settings.kv_cache = 'fp8'; fp8Variant.samples[0].finish_reason = 'stop'
assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(fp8Variant)), fp8Variant)
assert.equal(benchmarkProtocolComplete(parsed), true)
assert.equal(benchmarkCells(parsed)[0].successful, 5)
assert.equal(benchmarkCells(parsed)[0].deliveryTokensPerSecond?.median, 100)
assert.equal(benchmarkCells(parsed)[0].firstTextSeconds?.median, 0.30000000000000004)
assert.equal(parsed.samples[0].server_decode_seconds, null, 'missing engine duration stays unknown')
const current = structuredClone(receipt); current.runtime.tensorfold_version = BENCHMARK_ENGINE_VERSION
assert.equal(parseBenchmarkReceipt(JSON.stringify(current), { model: current.model.repo_id, serial: false, engineVersion: BENCHMARK_ENGINE_VERSION }).runtime.tensorfold_version, BENCHMARK_ENGINE_VERSION)
const historical = structuredClone(receipt); historical.runtime.tensorfold_version = '1.0.12'
delete historical.settings.kv_cache; delete historical.settings.thinking_mode; delete historical.group
for (const sample of historical.samples) { delete sample.finish_reason; delete sample.route }
for (const version of ['1.0.12', '1.1.0']) {
  historical.runtime.tensorfold_version = version
  assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(historical)), historical, 'reviewed historical receipts without the new fields remain readable')
  const historicalShort = structuredClone(historical)
  Object.assign(historicalShort.samples[5], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100 })
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(historicalShort)), /invalid|unsupported/, 'historical receipts without finish_reason still require 256 completion tokens for ok samples')
}
assert.throws(() => parseBenchmarkReceipt(JSON.stringify(historical), { model: historical.model.repo_id, serial: false, engineVersion: BENCHMARK_ENGINE_VERSION }), /invalid|unsupported/, 'a historical receipt cannot substitute for a live run on the new engine')
const natural = structuredClone(receipt)
Object.assign(natural.samples[5], { completion_tokens: 177, delivery_seconds: 2, delivery_tps: 88, finish_reason: 'stop' })
const naturalParsed = parseBenchmarkReceipt(JSON.stringify(natural), { model: natural.model.repo_id, serial: false })
assert.deepEqual(naturalParsed, natural, 'a 177-token natural stop with finish_reason stop is accepted')
assert.equal(benchmarkProtocolComplete(naturalParsed), true)
assert.equal(benchmarkCells(naturalParsed)[1].successful, 5)
assert.deepEqual(benchmarkCells(naturalParsed)[1].deliveryTokensPerSecond, { median: 100, min: 88, max: 100, spread: 12 })
const minNatural = structuredClone(receipt)
Object.assign(minNatural.samples[5], { completion_tokens: 128, delivery_seconds: 1.27, delivery_tps: 100, finish_reason: 'stop' })
assert.equal(benchmarkProtocolComplete(parseBenchmarkReceipt(JSON.stringify(minNatural))), true, '128 tokens with finish_reason stop meets the natural-stop floor')
const codeChatRoute = structuredClone(receipt)
for (const sample of codeChatRoute.samples) sample.route = '/v1/chat/completions'
assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(codeChatRoute)), codeChatRoute, 'code samples may consistently use /v1/chat/completions across the whole run')
for (const mutate of [
  (value: any) => { value.prompt = 'SYNTHETIC_PRIVATE_PROMPT' },
  (value: any) => { value.model.repo_id = '/private/operator/model' },
  (value: any) => { value.hardware.cpu_model = 'person@example.com' },
  (value: any) => { value.samples[0].delivery_tps = 1000 },
  (value: any) => { value.samples[0].seed = 999 },
  (value: any) => { value.samples.push(value.samples[0]) },
  (value: any) => { value.samples[0].completion_tokens = 257 },
  (value: any) => { Object.assign(value.samples[5], { completion_tokens: 127, delivery_seconds: 1.26, delivery_tps: 100, finish_reason: 'stop' }) },
  (value: any) => { Object.assign(value.samples[15], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100, finish_reason: 'stop' }) },
  (value: any) => { value.suite = { id: 'community-v1', sha256: BENCHMARK_SUITES['community-v1'] }; Object.assign(value.samples[5], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100, finish_reason: 'stop' }) },
  (value: any) => { Object.assign(value.samples[5], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100, finish_reason: null }) },
  (value: any) => { Object.assign(value.samples[5], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100 }); delete value.samples[5].finish_reason },
  (value: any) => { Object.assign(value.samples[5], { completion_tokens: 177, delivery_seconds: 1.76, delivery_tps: 100, finish_reason: 'length' }) },
  (value: any) => { value.samples[0].finish_reason = null },
  (value: any) => { delete value.samples[0].finish_reason },
  (value: any) => { value.created_at = '2026-02-30T00:00:00Z' },
  (value: any) => { value.suite.sha256 = '0'.repeat(64) },
  (value: any) => { value.settings.managed = false },
  (value: any) => { value.settings.kv_cache = null },
  (value: any) => { delete value.settings.kv_cache },
  (value: any) => { value.settings.kv_cache = 'int8' },
  (value: any) => { value.settings.thinking_mode = 'disabled' },
  (value: any) => { value.settings.extra = true },
  (value: any) => { value.samples[0].finish_reason = 'tool_calls' },
  (value: any) => { value.samples[0].route = '/v1/models' },
  (value: any) => { for (const sample of value.samples) if (sample.fixture_id === 'chat') sample.route = '/v1/completions' },
  (value: any) => { value.samples[1].route = '/v1/chat/completions' },
  (value: any) => { value.samples[10].route = '/v1/chat/completions' },
  (value: any) => { delete value.samples[1].route },
  (value: any) => { for (const sample of value.samples) if (sample.fixture_id === 'chat') delete sample.route },
  (value: any) => { for (const sample of value.samples) delete sample.route },
  (value: any) => { value.samples[0].extra = true },
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

for (const family of [null, 'Mac mini', 'Mac Studio', 'MacBook Pro', 'MacBook Air', 'iMac', 'Mac Pro'] as const) {
  const withFamily = structuredClone(receipt)
  Object.assign(withFamily.hardware, { form_factor: family })
  const result = parseBenchmarkReceipt(JSON.stringify(withFamily), { model: receipt.model.repo_id, serial: false })
  assert.equal(result.hardware.form_factor, family)
  assert.deepEqual(result, withFamily, 'The app preserves the accepted metadata for storage and export')
  assert.deepEqual(benchmarkCells(result), benchmarkCells(parsed), 'A product family cannot alter measured benchmark cells')
  assert.equal(benchmarkProtocolComplete(result), true)
}
assert.equal(Object.hasOwn(parsed.hardware, 'form_factor'), false, 'Old receipts keep the field absent')
for (const family of ['', 'Mac17,16', 'mac mini', 'Mac mini M5 Pro', 'Mac mini\n', true, 7, [], {}, ['Mac mini']]) {
  const invalid = structuredClone(receipt) as any
  invalid.hardware.form_factor = family
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(invalid)), /invalid|unsupported/)
}
for (const formFactor of [undefined, null, 'Mac mini']) {
  const invalid = structuredClone(receipt) as any
  if (formFactor !== undefined) invalid.hardware.form_factor = formFactor
  invalid.hardware.model_identifier = 'Mac17,16'
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(invalid)), /invalid|unsupported/, 'The optional key cannot admit unrelated hardware fields')
}
console.log('Studio accepts absent, null and six coarse product families, preserves metadata and rejects raw IDs, extra keys and invalid values.')

assert.equal(parsed.suite.id, 'community-v2', 'the synthetic receipt is the engine\'s current suite')
const legacy = structuredClone(receipt); legacy.suite = { id: 'community-v1', sha256: BENCHMARK_SUITES['community-v1'] }; delete legacy.group
assert.equal(parseBenchmarkReceipt(JSON.stringify(legacy)).suite.id, 'community-v1', 'older engines\' community-v1 receipts still parse')
const legacyNoRoute = structuredClone(legacy); delete legacyNoRoute.settings.kv_cache; delete legacyNoRoute.settings.thinking_mode
for (const sample of legacyNoRoute.samples) { delete sample.finish_reason; delete sample.route }
assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(legacyNoRoute)), legacyNoRoute, 'historical community-v1 receipts without route or finish_reason remain readable')
for (const suite of [{ id: 'community-v3', sha256: BENCHMARK_SUITES['community-v2'] }, { id: 'community-v2', sha256: BENCHMARK_SUITES['community-v1'] }, { id: 'community-v1', sha256: BENCHMARK_SUITES['community-v2'] }, { id: 'community-v2', sha256: '0'.repeat(64) }]) {
  const wrong = structuredClone(receipt) as any; wrong.suite = suite
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(wrong)), /invalid|unsupported/, `suite ${suite.id} with the wrong or unknown hash is refused`)
}
const broken = structuredClone(receipt) as any; broken.suite.id = 'community-v9'; broken.samples[0].seed = 999
assert.throws(() => parseBenchmarkReceipt(JSON.stringify(broken)), /invalid|unsupported/)
const kept = parseBenchmarkDetails(JSON.stringify(broken), { model: receipt.model.repo_id, serial: false })
assert.deepEqual(kept, { runtime: receipt.runtime, model: receipt.model, hardware: receipt.hardware, settings: receipt.settings }, 'a failed receipt keeps its checked details')
const leaky = structuredClone(broken); leaky.hardware.cpu_model = 'person@example.com'
assert.equal(parseBenchmarkDetails(JSON.stringify(leaky)), null, 'details that fail their privacy checks are not kept')
assert.equal(parseBenchmarkDetails(JSON.stringify(broken), { model: 'example-org/other-model', serial: false }), null)
assert.deepEqual(checkedBenchmarkDetails(kept), kept, 'saved details re-check')
assert.equal(checkedBenchmarkDetails({ ...kept, extra: 1 }), null)
console.log('Benchmark suites: community-v2 and legacy community-v1 accepted, unknown suites and wrong hashes refused; failed runs keep checked details')

assert.deepEqual(parsed.group, { schema_version: 1, source: 'loaded-group', rank_count: 1, category: 'single-node', nodes: [{ id: 0, ranks: [0], hardware: receipt.hardware }] }, 'the pinned engine\'s loaded-group inventory is preserved for storage and export')
const ungrouped = structuredClone(receipt); delete ungrouped.group
assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(ungrouped), { model: receipt.model.repo_id, serial: false }), ungrouped, 'v2 receipts without a group stay readable')
// Shape written by tensorfold-native 1.2.8 on a single Apple Silicon node, with synthetic values.
const engineShaped = structuredClone(receipt)
engineShaped.hardware = { cpu_model: 'Apple M5 Max', gpu_count: 1, gpus: [{ cores: 40, memory_bytes: 68719476736, name: 'Apple M5 Max' }], memory_type: 'unified', source: 'detected', system_memory_bytes: 68719476736 }
engineShaped.group = { category: 'single-node', nodes: [{ hardware: structuredClone(engineShaped.hardware), id: 0, ranks: [0] }], rank_count: 1, schema_version: 1, source: 'loaded-group' }
assert.deepEqual(parseBenchmarkReceipt(JSON.stringify(engineShaped), { model: receipt.model.repo_id, serial: false, engineVersion: BENCHMARK_ENGINE_VERSION }), engineShaped, 'the 1.2.8 managed receipt with a loaded group is accepted')
const nodeFamily = structuredClone(receipt); nodeFamily.group!.nodes[0].hardware = { ...nodeFamily.group!.nodes[0].hardware, form_factor: 'Mac Studio' } as any
assert.equal(parseBenchmarkReceipt(JSON.stringify(nodeFamily)).group?.nodes[0].hardware.cpu_model, 'Synthetic CPU', 'node hardware accepts the same coarse product family as root hardware')
for (const mutate of [
  (value: any) => { value.group = null },
  (value: any) => { value.group = [] },
  (value: any) => { value.group.extra = true },
  (value: any) => { delete value.group.category },
  (value: any) => { value.group.schema_version = 2 },
  (value: any) => { value.group.source = 'client-hint' },
  (value: any) => { value.group.rank_count = 2 },
  (value: any) => { value.group.rank_count = 0 },
  (value: any) => { value.group.rank_count = '1' },
  (value: any) => { value.group.category = 'multi-node' },
  (value: any) => { value.group.nodes = [] },
  (value: any) => { value.group.nodes = {} },
  (value: any) => { value.group.nodes.push(structuredClone(value.group.nodes[0])) },
  (value: any) => { value.group.nodes[0].id = 1 },
  (value: any) => { value.group.nodes[0].ranks = [] },
  (value: any) => { value.group.nodes[0].ranks = [1] },
  (value: any) => { value.group.nodes[0].ranks = [0, 0] },
  (value: any) => { value.group.nodes[0].ranks = [-1] },
  (value: any) => { value.group.nodes[0].ranks = [0.5] },
  (value: any) => { value.group.nodes[0].boot_id = 'private' },
  (value: any) => { delete value.group.nodes[0].hardware },
  (value: any) => { value.group.nodes[0].hardware.cpu_model = 'person@example.com' },
  (value: any) => { value.group.nodes[0].hardware.serial_number = 'SYNTHETIC' },
  (value: any) => { value.group.nodes[0].hardware.gpu_count = 2 },
  (value: any) => { value.group.nodes[0].hardware.source = 'unavailable' },
  (value: any) => { value.group.nodes[0].hardware.gpus = []; value.group.nodes[0].hardware.gpu_count = 0 },
  (value: any) => { value.group.nodes[0].hardware.form_factor = 'Mac17,16' },
  (value: any) => { value.suite = { id: 'community-v1', sha256: BENCHMARK_SUITES['community-v1'] } },
]) {
  const invalid = structuredClone(receipt) as any; mutate(invalid)
  assert.throws(() => parseBenchmarkReceipt(JSON.stringify(invalid)), /invalid|unsupported/)
}
console.log('Benchmark loaded group: optional single-node inventory accepted and preserved; wrong shape, counts, ranks, categories, private node fields and v1 groups refused')
