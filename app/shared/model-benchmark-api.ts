import type { ModelBenchmarkInput, ModelBenchmarkSnapshot } from './model-benchmark'

export interface ModelBenchmarkApi {
  engineStartBenchmark(input: ModelBenchmarkInput): Promise<ModelBenchmarkSnapshot>
  engineBenchmarkStatus(id?: string): Promise<ModelBenchmarkSnapshot | null>
  engineBenchmarkHistory(): Promise<ModelBenchmarkSnapshot[]>
  engineCancelBenchmark(id: string): Promise<boolean>
  engineExportBenchmark(id: string): Promise<{ saved: boolean }>
}

export const MODEL_BENCHMARK_API_SCHEMA = {
  engineStartBenchmark: 'method', engineBenchmarkStatus: 'method', engineBenchmarkHistory: 'method',
  engineCancelBenchmark: 'method', engineExportBenchmark: 'method',
} as const
