import engineManifest from '../../../../build/engine.json'

export const BENCHMARK_ENGINE_VERSION = engineManifest.version
// Schema compatibility is reviewed explicitly; a newer binary cannot validate its own receipts by version alone.
const REVIEWED_VERSIONS = new Set(['1.0.12', '1.1.0', '1.1.4'])
export function supportedBenchmarkReceiptVersion(value: unknown): value is string { return typeof value === 'string' && REVIEWED_VERSIONS.has(value) }
export function assertBundledBenchmarkVersion(value: unknown): string {
  if (value !== BENCHMARK_ENGINE_VERSION || !supportedBenchmarkReceiptVersion(value)) throw new Error(`This benchmark requires the bundled TensorFold ${BENCHMARK_ENGINE_VERSION} engine.`)
  return value
}
