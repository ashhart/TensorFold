import type { ModelBenchmarkApi } from '../../../shared/model-benchmark-api'

/** Browser preview never invents measurements or executes a native benchmark. */
export function createMockModelBenchmarkApi(): ModelBenchmarkApi {
  return {
    engineStartBenchmark: async () => { throw new Error('Benchmarks require the desktop app and a cached local model; browser preview does not run measurements.') },
    engineBenchmarkStatus: async () => null,
    engineBenchmarkHistory: async () => [],
    engineCancelBenchmark: async () => false,
    engineExportBenchmark: async () => { throw new Error('Benchmark JSON export requires a completed desktop run.') },
  }
}
