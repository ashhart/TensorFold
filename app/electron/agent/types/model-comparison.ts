import type { EngineModelPreviewResult } from './model-preview'

export interface EngineModelComparisonInput {
  id: string
  models: [string, string]
  prompt: string
}
export interface EngineModelComparisonRow {
  model: string
  state: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'
  result: EngineModelPreviewResult | null
  error: string | null
}
export interface EngineModelComparisonSnapshot {
  id: string
  revision: number
  state: 'running' | 'restoring' | 'complete' | 'cancelled' | 'failed'
  startedAt: number
  finishedAt: number | null
  prompt: string
  maxOutputTokens: 256
  temperature: 0
  rows: [EngineModelComparisonRow, EngineModelComparisonRow]
  previousModel: string | null
  restored: boolean
  restoreError: string | null
}
