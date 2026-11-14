import type { EngineEvent, EnginePerformance } from '../types'
import { createTurnStats, finalizeTurnStats, readEngineTimings, recordModelCall, type ModelCallSample } from '../turn-stats'
import type { EngineSupervisor } from './supervisor'

// The latest measured speed of the model in memory, shown beside it and dropped when it unloads.

export class EnginePerformanceTracker {
  private latest: EnginePerformance | null = null

  constructor(private readonly deps: { supervisor: Pick<EngineSupervisor, 'status' | 'live'>; emit: (event: EngineEvent) => void }) {}

  get current(): EnginePerformance | null {
    return this.latest
  }

  clear(): void {
    if (!this.latest) return
    this.latest = null
    this.deps.emit({ kind: 'performance', performance: null })
  }

  record(model: string, sample: ModelCallSample): void {
    const status = this.deps.supervisor.status()
    // Reject old requests even when the same model was unloaded and launched again.
    if (!this.deps.supervisor.live(model) || !Number.isFinite(sample.requestAt) || status.readyAt == null || sample.requestAt < status.readyAt) return
    const raw = sample.engine
    const engine = raw ? readEngineTimings({ tokens_per_second: raw.decodeTokensPerSecond, seconds: raw.decodeSeconds, prefill_seconds: raw.prefillSeconds, time_to_first_token: raw.firstTokenSeconds }) : null
    const stats = createTurnStats()
    recordModelCall(stats, { ...sample, engine })
    const latest = finalizeTurnStats(stats, { model, providerId: 'tensorfold' })
    const metric = (value: number | undefined): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
    this.latest = {
      model, measuredAt: Date.now(),
      decodeTokensPerSecond: metric(engine && engine.decodeSeconds > 0 ? engine.decodeTokensPerSecond : latest.decodeTokensPerSecond),
      prefillTokensPerSecond: metric(latest.prefillTokensPerSecond),
      firstTokenMs: metric(latest.firstTokenMs),
      completionTokens: metric(latest.completionTokens),
    }
    this.deps.emit({ kind: 'performance', performance: this.latest })
  }
}
