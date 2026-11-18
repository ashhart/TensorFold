import type { ModelBenchmarkCell, ModelBenchmarkReceipt, ModelBenchmarkSpread } from '../../../../shared/model-benchmark'
import { isRepoId } from '../paths'
import { strictBenchmarkJson } from './strict-json'
import { supportedBenchmarkReceiptVersion } from './version'

export const BENCHMARK_SUITE_SHA = '9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5be'
export const BENCHMARK_RECEIPT_MAX_BYTES = 1024 * 1024
const SAMPLE_KEYS = ['fixture_id', 'temperature', 'repeat', 'seed', 'status', 'prompt_tokens', 'completion_tokens', 'cached_tokens', 'delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds', 'server_decode_tps', 'server_decode_seconds', 'prefill_seconds', 'token_sha', 'error_code']
const ERROR_CODES = new Set(['http_error', 'connection_error', 'timeout', 'cancelled', 'server_error', 'invalid_stream', 'truncated_stream', 'missing_usage', 'invalid_usage', 'empty_output', 'unexpected_finish', 'token_count_mismatch', 'unmeasured_delivery'])
function invalid(): never { throw new Error('The local benchmark receipt is invalid or contains unsupported fields.') }
function object(value: any, keys: string[]): void {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== keys.length || keys.some(key => !Object.hasOwn(value, key))) invalid()
}
function text(value: unknown, nullable = false): void {
  if (value === null && nullable) return
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}$/.test(value)) invalid()
}
function number(value: unknown, nullable = true, integer = false, max = 1e15): void {
  if (value === null && nullable) return
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > max || integer && !Number.isSafeInteger(value)) invalid()
}
function digest(value: unknown, sizes = [64]): void { if (value !== null && (typeof value !== 'string' || !sizes.includes(value.length) || !/^[a-f0-9]+$/.test(value))) invalid() }
export function parseBenchmarkReceipt(raw: string, expected?: { model: string; serial: boolean; engineVersion?: string }, now = Date.now()): ModelBenchmarkReceipt {
  if (Buffer.byteLength(raw) > BENCHMARK_RECEIPT_MAX_BYTES) invalid()
  let value: any
  try { value = strictBenchmarkJson(raw) } catch { invalid() }
  object(value, ['schema_version', 'suite', 'run_id', 'created_at', 'runtime', 'model', 'hardware', 'settings', 'samples'])
  if (value.schema_version !== 1 || typeof value.run_id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value.run_id)) invalid()
  if (typeof value.created_at !== 'string' || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z$/.test(value.created_at) || !Number.isFinite(Date.parse(value.created_at)) || Date.parse(value.created_at) > now + 900_000 || new Date(value.created_at).toISOString().slice(0, 19) !== value.created_at.slice(0, 19)) invalid()
  object(value.suite, ['id', 'sha256'])
  if (value.suite.id !== 'community-v1' || value.suite.sha256 !== BENCHMARK_SUITE_SHA) invalid()
  const runtime = value.runtime
  object(runtime, ['tensorfold_version', 'backend', 'python_version', 'platform', 'dependencies'])
  text(runtime.tensorfold_version); text(runtime.python_version)
  if (!supportedBenchmarkReceiptVersion(runtime.tensorfold_version) || expected?.engineVersion !== undefined && runtime.tensorfold_version !== expected.engineVersion || runtime.python_version !== 'none' || !['mlx', 'cuda', 'rocm'].includes(runtime.backend) || runtime.platform !== (runtime.backend === 'mlx' ? 'macos' : 'linux')) invalid()
  if (!runtime.dependencies || typeof runtime.dependencies !== 'object' || Array.isArray(runtime.dependencies) || Object.keys(runtime.dependencies).some(key => !['mlx', 'mlx_lm', 'torch', 'triton', 'cuda', 'driver'].includes(key))) invalid()
  Object.values(runtime.dependencies).forEach(item => text(item, true))
  const model = value.model
  object(model, ['repo_id', 'revision', 'family', 'config_sha256', 'tokenizer_sha256', 'quantization', 'drafter_repo_id', 'local'])
  if (!isRepoId(model.repo_id) || model.local !== false || model.drafter_repo_id !== null && !isRepoId(model.drafter_repo_id)) invalid()
  digest(model.revision, [40, 64]); digest(model.config_sha256); digest(model.tokenizer_sha256)
  text(model.family, true); text(model.quantization, true)
  const hardware = value.hardware
  object(hardware, ['cpu_model', 'system_memory_bytes', 'memory_type', 'gpus', 'gpu_count', 'source'])
  text(hardware.cpu_model, true); number(hardware.system_memory_bytes, true, true); number(hardware.gpu_count, true, true, 256)
  if (!['unified', 'dedicated', 'unknown'].includes(hardware.memory_type) || !['detected', 'partial', 'unavailable'].includes(hardware.source) || !Array.isArray(hardware.gpus) || hardware.gpus.length > 256) invalid()
  for (const gpu of hardware.gpus) { object(gpu, ['name', 'memory_bytes', 'cores']); text(gpu.name); number(gpu.memory_bytes, true, true); number(gpu.cores, true, true, 100_000) }
  if (hardware.gpu_count !== null && hardware.gpu_count !== hardware.gpus.length) invalid()
  const settings = value.settings
  object(settings, ['tokens', 'repetitions', 'temperatures', 'serial', 'context_tokens', 'managed', 'rank_count'])
  if (settings.tokens !== 256 || settings.repetitions !== 5 || settings.context_tokens !== 4096 || settings.managed !== true || settings.rank_count !== 1 || typeof settings.serial !== 'boolean' || !Array.isArray(settings.temperatures) || settings.temperatures.length !== 2 || settings.temperatures[0] !== 1 || settings.temperatures[1] !== 0) invalid()
  if (expected && (model.repo_id !== expected.model || settings.serial !== expected.serial)) invalid()
  if (!Array.isArray(value.samples) || value.samples.length < 1 || value.samples.length > 24) invalid()
  const seen = new Set<string>()
  for (const sample of value.samples) {
    object(sample, SAMPLE_KEYS)
    if (!['code', 'chat'].includes(sample.fixture_id) || ![0, 1].includes(sample.temperature) || !Number.isInteger(sample.repeat) || sample.repeat < -1 || sample.repeat > 4 || sample.seed !== 1234 + Math.max(0, sample.repeat) || !['ok', 'early_eos', 'error', 'unmeasured'].includes(sample.status) || sample.repeat === -1 && sample.status !== 'error') invalid()
    const key = `${sample.fixture_id}:${sample.temperature}:${sample.repeat}`
    if (seen.has(key)) invalid(); seen.add(key)
    for (const key of ['prompt_tokens', 'completion_tokens', 'cached_tokens']) number(sample[key], true, true, 10_000_000)
    for (const key of ['delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds', 'server_decode_tps', 'server_decode_seconds', 'prefill_seconds']) number(sample[key], true, false, 1e9)
    digest(sample.token_sha, [12, 64])
    if (sample.error_code !== null && !ERROR_CODES.has(sample.error_code)) invalid()
    if (sample.completion_tokens !== null && sample.completion_tokens > 256 || sample.cached_tokens !== null && sample.prompt_tokens !== null && sample.cached_tokens > sample.prompt_tokens) invalid()
    if (sample.delivery_tps !== null) {
      if (sample.completion_tokens === null || sample.completion_tokens < 2 || sample.delivery_seconds === null || sample.delivery_seconds <= 0) invalid()
      const rate = (sample.completion_tokens - 1) / sample.delivery_seconds
      if (Math.abs(sample.delivery_tps - rate) > Math.max(1e-6, rate * 1e-6)) invalid()
    }
    if (sample.status === 'ok' && (sample.completion_tokens !== 256 || sample.delivery_tps === null || sample.error_code !== null)) invalid()
    if (sample.status === 'early_eos' && (sample.completion_tokens === null || sample.completion_tokens >= 256)) invalid()
  }
  return value as ModelBenchmarkReceipt
}
function spread(values: (number | null)[], complete: boolean): ModelBenchmarkSpread | null {
  if (!complete || values.some(value => value === null)) return null
  const sorted = (values as number[]).slice().sort((a, b) => a - b)
  if (!sorted.length) return null
  const mid = Math.floor(sorted.length / 2)
  return { median: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2, min: sorted[0], max: sorted.at(-1)!, spread: sorted.at(-1)! - sorted[0] }
}
export function benchmarkCells(receipt?: ModelBenchmarkReceipt): ModelBenchmarkCell[] {
  return [1, 0].flatMap(temperature => (['code', 'chat'] as const).map(fixture => {
    const rows = receipt?.samples.filter(row => row.fixture_id === fixture && row.temperature === temperature) ?? []
    const measured = rows.filter(row => row.repeat >= 0)
    const successful = measured.filter(row => row.status === 'ok').length
    const complete = rows.length === 5 && successful === 5
    return { fixture, temperature: temperature as 0 | 1, attempted: measured.length, expected: 5, successful, earlyEos: measured.filter(row => row.status === 'early_eos').length, errors: measured.filter(row => row.status === 'error').length, unmeasured: measured.filter(row => row.status === 'unmeasured').length, failedWarmups: rows.filter(row => row.repeat === -1).length, complete,
      deliveryTokensPerSecond: spread(measured.map(row => row.delivery_tps), complete), firstTextSeconds: spread(measured.map(row => row.ttft_seconds), complete), endToEndSeconds: spread(measured.map(row => row.end_to_end_seconds), complete), serverDecodeTokensPerSecond: spread(measured.map(row => row.server_decode_tps), complete), prefillSeconds: spread(measured.map(row => row.prefill_seconds), complete) } as ModelBenchmarkCell
  }))
}
export function benchmarkProtocolComplete(receipt: ModelBenchmarkReceipt): boolean {
  return benchmarkCells(receipt).every(cell => cell.complete) && receipt.model.revision !== null && receipt.model.config_sha256 !== null && receipt.hardware.source !== 'unavailable' && receipt.hardware.gpus.length > 0 && receipt.runtime.backend !== 'rocm'
}
