import type { EngineInstalledModel } from '../../../lib/types'
import type { ModelBenchmarkCell, ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'

export function fixtureCell(fixture: 'code' | 'chat' = 'code', temperature: 0 | 1 = 0): ModelBenchmarkCell {
  const spread = (median: number) => ({ median, min: median * 0.9, max: median * 1.1, spread: median * 0.2 })
  return { fixture, temperature, attempted: 5, expected: 5, successful: 5, earlyEos: 0, errors: 0, unmeasured: 0, failedWarmups: 0, complete: true, deliveryTokensPerSecond: spread(25), firstTextSeconds: spread(0.2), endToEndSeconds: spread(10), serverDecodeTokensPerSecond: null, prefillSeconds: null }
}
export function fixtureRun(patch: Partial<ModelBenchmarkSnapshot> = {}): ModelBenchmarkSnapshot {
  return { id: 'benchmark_fixture_0123456789', model: 'fixture/cached-model', serial: false, state: 'complete', revision: 5, startedAt: 1_000, finishedAt: 2_000, progress: { completedRequests: 24, totalRequests: 24, fixture: 'chat', temperature: 0, repeat: 4, phase: 'measured' }, cells: [fixtureCell('code', 1), fixtureCell('chat', 1), fixtureCell('code', 0), fixtureCell('chat', 0)], receiptAvailable: true, protocolComplete: true, previousModel: 'fixture/previous', restored: true, restoreError: null, error: null, storageError: null, warnings: [], saved: true, ...patch }
}
export function fixtureModel(repo = 'fixture/cached-model', patch: Partial<EngineInstalledModel> = {}): EngineInstalledModel {
  return { repo, title: repo, family: 'fixture', modelType: 'fixture', format: 'mlx', contextTokens: 32768, weightsBytes: 1024, memoryFloorBytes: 2048, fitsMemory: true, drafter: null, drafterInstalled: null, source: 'cache', ...patch }
}
