import type { EngineModelGenerationSettings } from './engine'

export interface EngineModelPreviewResult {
  model: string
  startedAt: number
  finishedAt: number
  loadMs: number
  elapsedMs: number
  firstTokenMs: number | null
  decodeTokensPerSecond: number | null
  prefillTokensPerSecond: number | null
  promptTokens: number | null
  completionTokens: number | null
  cachedTokens: number | null
  peakMemoryBytes: number | null
  finishReason?: string | null
  text: string
  profile: EngineModelGenerationSettings | null
}
