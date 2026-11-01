import type { EngineInstalledModel } from '../../../lib/types'
import type { ModelBenchmarkCell, ModelBenchmarkSnapshot, ModelBenchmarkSpread } from '../../../../shared/model-benchmark'

export const benchmarkActive = (run: ModelBenchmarkSnapshot | null) => !!run && ['preparing', 'running', 'restoring'].includes(run.state)
export const cachedBenchmarkModels = (models: EngineInstalledModel[]) => models.filter((model) => model.fitsMemory === true && model.source !== 'library' && !model.repo.startsWith('TensorFoldLocal/') && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(model.repo))
export function benchmarkDefaultModel(models: EngineInstalledModel[], preferred?: string | null): string {
  const eligible = cachedBenchmarkModels(models)
  return eligible.find((model) => model.repo === preferred)?.repo ?? eligible[0]?.repo ?? ''
}
export function benchmarkMetric(value: ModelBenchmarkSpread | null, complete: boolean, unit: string): string {
  return complete && value && Number.isFinite(value.median) && value.median >= 0 ? `${Number(value.median.toFixed(2))} ${unit}` : 'Not reported'
}
export function benchmarkRange(value: ModelBenchmarkSpread | null, complete: boolean, unit: string): string | null {
  return complete && value && [value.min, value.max].every((number) => Number.isFinite(number) && number >= 0) ? `${Number(value.min.toFixed(2))}–${Number(value.max.toFixed(2))} ${unit}` : null
}
export function benchmarkCellLabel(cell: ModelBenchmarkCell): string { return `${cell.fixture === 'code' ? 'Code' : 'Chat'} · Temperature ${cell.temperature}` }
export const benchmarkSuite = (run: ModelBenchmarkSnapshot) => run.metadata?.suite?.id ?? 'community-v1'
export function benchmarkBasicSummary(run: ModelBenchmarkSnapshot): string {
  return [`TensorFold · ${benchmarkSuite(run)}`, `Model: ${run.model}`, `Mode: ${run.serial ? 'Serial baseline' : 'Standard'}`, `State: ${run.state}; complete protocol: ${run.protocolComplete ? 'yes' : 'no'}`,
    ...run.cells.map((cell) => `${benchmarkCellLabel(cell)}: ${cell.complete ? `median delivery ${benchmarkMetric(cell.deliveryTokensPerSecond, true, 'tokens/s')}; first text ${benchmarkMetric(cell.firstTextSeconds, true, 's')}` : `incomplete, ${cell.successful}/${cell.expected} valid samples`}`),
    'Measured speed is not a quality ranking.'].join('\n')
}
