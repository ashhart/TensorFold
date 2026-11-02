import type { EngineSnapshot } from './types'

function positive(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

export function engineReadiness(snapshot: EngineSnapshot | null) {
  const status = snapshot?.status
  const ready = !!status?.available && status.state === 'ready' && !!status.model
  const model = snapshot?.installed.find(item => item.repo === status?.model)
  const budget = positive(status?.memoryLimitBytes)
  const floor = status?.model ? positive(model?.memoryFloorBytes) : null
  const performance = ready && snapshot?.performance?.model === status?.model ? snapshot.performance : null
  return {
    ready,
    context: ready ? positive(status?.contextTokens) : null,
    budget,
    floor,
    floorPercent: floor != null && budget != null ? Math.min(100, floor / budget * 100) : null,
    decode: positive(performance?.decodeTokensPerSecond),
    measuredAt: performance?.measuredAt ?? null,
  }
}
