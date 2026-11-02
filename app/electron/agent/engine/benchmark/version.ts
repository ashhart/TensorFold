import { parseSemver, compareSemver } from '../../cli-version-advisory'
import engineManifest from '../../../../build/engine.json'

export const BENCHMARK_ENGINE_VERSION = engineManifest.version
// Schema compatibility is reviewed explicitly; a newer binary cannot validate its own receipts by version alone.
const REVIEWED_VERSIONS = new Set(['1.0.12', '1.1.0', '1.1.4', '1.1.6', '1.2.0'])
export function supportedBenchmarkReceiptVersion(value: unknown): value is string { return typeof value === 'string' && (REVIEWED_VERSIONS.has(value) || compatibleBenchmarkV2Version(value)) }
export function assertBundledBenchmarkVersion(value: unknown): string {
  if (value !== BENCHMARK_ENGINE_VERSION || !supportedBenchmarkReceiptVersion(value)) throw new Error(`This benchmark requires the bundled TensorFold ${BENCHMARK_ENGINE_VERSION} engine.`)
  return value
}

// Protocol 1 runtimes still need the frozen benchmark v2 suite and receipt schema.
export function compatibleBenchmarkV2Version(value: unknown): value is string {
  const version = typeof value === 'string' && /^1\.\d{1,5}\.\d{1,5}$/.test(value) ? parseSemver(value) : null
  return !!version && version.major === 1 && compareSemver(version, parseSemver('1.1.4')) >= 0
}
export function assertCompatibleBenchmarkVersion(value: unknown): string {
  if (!compatibleBenchmarkV2Version(value)) throw new Error('This benchmark requires a protocol 1 runtime with the community-v2 suite.')
  return value
}
