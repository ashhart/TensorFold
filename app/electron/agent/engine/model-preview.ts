import { streamChat } from '../openai'
import { createTurnStats, finalizeTurnStats, recordModelCall } from '../turn-stats'
import type { EngineModelGenerationSettings } from '../types'
import type { EngineModelPreviewResult } from '../types/model-preview'
import { builtinEngineProvider } from './provider'

export const MODEL_PREVIEW_PROMPT = 'Explain in three short sentences how a rainbow forms. Answer directly, without tools or a thinking preamble.'

// This preview has no chat history, files, tools or external provider, and never teaches memory.
export async function measureModelPreview(model: string, signal: AbortSignal, profile: () => EngineModelGenerationSettings | null, stream = streamChat, options: { prompt?: string; owner?: symbol; assertAllowed?: () => void } = {}): Promise<EngineModelPreviewResult> {
  const startedAt = Date.now(), stats = createTurnStats()
  let text = '', requestAt = startedAt, finished = false, reasoning = false, finishReason: string | null = null
  for await (const chunk of stream({ provider: builtinEngineProvider(), model, assertRequestAllowed: options.assertAllowed, modelPreview: true, modelComparisonOwner: options.owner, messages: [{ role: 'user', content: options.prompt ?? MODEL_PREVIEW_PROMPT }], tools: [], reasoningEffort: 'none', signal })) {
    if (signal.aborted) throw signal.reason ?? new Error('Preview cancelled.')
    options.assertAllowed?.()
    if (chunk.type === 'content') text += chunk.text
    else if (chunk.type === 'reasoning') reasoning = true
    else if (chunk.type === 'usage') { requestAt = chunk.requestAt; recordModelCall(stats, chunk) }
    else if (chunk.type === 'finish') { finished = true; finishReason = chunk.reason }
    else if (chunk.type === 'tool_calls') throw new Error('A model preview cannot run tools.')
    if (text.length > 16_384) throw new Error('The preview exceeded its text limit.')
  }
  if (signal.aborted) throw signal.reason ?? new Error('Preview cancelled.')
  if (!finished) throw new Error('The model preview ended before a complete response was received.')
  if (!text.trim() && !reasoning) throw new Error('The model returned no preview text. Try another model or profile.')
  const finishedAt = Date.now(), latest = finalizeTurnStats(stats, { model, providerId: 'tensorfold' })
  const measured = (value: number | undefined): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
  return { model, startedAt, finishedAt, loadMs: Math.max(0, requestAt - startedAt), elapsedMs: Math.max(0, finishedAt - requestAt),
    firstTokenMs: measured(latest.firstTokenMs), decodeTokensPerSecond: measured(latest.decodeTokensPerSecond), prefillTokensPerSecond: measured(latest.prefillTokensPerSecond),
    promptTokens: stats.promptCalls ? stats.promptTokens : null, completionTokens: stats.usageCalls ? stats.completionTokens : null, cachedTokens: stats.cachedCalls ? stats.cachedTokens : null,
    peakMemoryBytes: null, finishReason, text: text.trim() || 'This model used the short preview for reasoning and returned no final text.', profile: profile() }
}
