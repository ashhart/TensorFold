import type { ModelBenchmarkApi } from '../../shared/model-benchmark-api'

export function modelBenchmarkPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): ModelBenchmarkApi {
  return {
    engineStartBenchmark: (input) => invoke('engineBenchmark:start', input),
    engineBenchmarkStatus: (id) => invoke('engineBenchmark:status', id),
    engineBenchmarkHistory: () => invoke('engineBenchmark:history'),
    engineCancelBenchmark: (id) => invoke('engineBenchmark:cancel', id),
    engineExportBenchmark: (id) => invoke('engineBenchmark:export', id),
  }
}
