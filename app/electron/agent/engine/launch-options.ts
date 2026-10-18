import type { EngineModelGenerationSettings } from '../types'
import { validateEngineModelGeneration } from '../../../shared/model-profiles'

// Only flags supported by the bundled engine; model profiles never contain arbitrary arguments.
export function modelLaunchOptions(value: EngineModelGenerationSettings | null | undefined, modelLimit: number | null): string[] {
  if (!value) return []
  const settings = validateEngineModelGeneration(value)
  if (modelLimit != null && settings.contextTokens > modelLimit) throw new Error(`This model supports up to ${modelLimit.toLocaleString('en-US')} context tokens. Choose a smaller context window.`)
  return ['--context', String(settings.contextTokens), '--max-tokens', String(settings.maxOutputTokens), '--temperature', String(settings.temperature),
    '--top-p', String(settings.topP), '--top-k', String(settings.topK), '--min-p', String(settings.minP), ...(settings.speculativeDecoding === 'off' ? ['--no-drafts'] : [])]
}
