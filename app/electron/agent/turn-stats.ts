// Per-run model facts for the line under a final answer: real usage and timings only.
// A call that sent no usage event contributes nothing to tokens or tok/s; nothing is estimated.
import type { AssistantTurnStats } from './types'

// The built-in engine's own timings for one reply, from the `tensorfold` object on its final chunk (seconds).
export interface EngineReplyTimings {
  decodeTokensPerSecond: number
  decodeSeconds: number
  prefillSeconds: number | null
  firstTokenSeconds: number | null
}

// Cortheon's report for one reply, from the `cortheon` object on the engine's final chunk.
export interface CortheonReply {
  task: string | null
  searches: number
  pagesRead: number
  sources: string[]
}

// One model call as streamChat's `usage` chunk reports it (epoch ms); the first call with a token sets first-token time.
export interface ModelCallSample {
  requestAt: number
  firstTokenAt: number | null
  lastTokenAt: number | null
  completionTokens: number | null
  promptTokens?: number | null
  cachedTokens?: number | null
  engine?: EngineReplyTimings | null
  cortheon?: CortheonReply | null
}

export interface TurnStatsAccumulator {
  modelCalls: number
  toolCalls: number
  firstTokenMs: number | null
  completionTokens: number
  usageCalls: number
  decodeTokens: number
  decodeMs: number
  promptCalls: number
  promptTokens: number
  cachedCalls: number
  cachedTokens: number
  engineDecodeTokens: number
  engineDecodeSeconds: number
  prefillTokens: number
  prefillSeconds: number
  cortheonCalls: number
  cortheonSearches: number
  cortheonPages: number
  cortheonSources: string[]
  cortheonTask: string | null
}

export function createTurnStats(): TurnStatsAccumulator {
  return {
    modelCalls: 0, toolCalls: 0, firstTokenMs: null, completionTokens: 0, usageCalls: 0, decodeTokens: 0, decodeMs: 0,
    promptCalls: 0, promptTokens: 0, cachedCalls: 0, cachedTokens: 0, engineDecodeTokens: 0, engineDecodeSeconds: 0, prefillTokens: 0, prefillSeconds: 0,
    cortheonCalls: 0, cortheonSearches: 0, cortheonPages: 0, cortheonSources: [], cortheonTask: null,
  }
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const nonNegative = (value: unknown): number | null => finite(value) && value >= 0 ? value : null
const count = (value: unknown): number | null => finite(value) && value >= 0 ? Math.floor(value) : null
const round = (value: number, places: number) => Math.round(value * 10 ** places) / 10 ** places

/** The engine's `tensorfold` object, or null when the chunk has none or its decode figures are unusable. */
export function readEngineTimings(value: unknown): EngineReplyTimings | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const raw = value as Record<string, unknown>
  const decodeTokensPerSecond = nonNegative(raw.tokens_per_second)
  const decodeSeconds = nonNegative(raw.seconds)
  if (decodeTokensPerSecond == null || decodeSeconds == null) return null
  return { decodeTokensPerSecond, decodeSeconds, prefillSeconds: nonNegative(raw.prefill_seconds), firstTokenSeconds: nonNegative(raw.time_to_first_token) }
}

const MAX_SOURCES = 12
const webSource = (value: unknown): value is string => {
  if (typeof value !== 'string' || value.length > 2048) return false
  try { return ['http:', 'https:'].includes(new URL(value).protocol) } catch { return false }
}

/** The engine's `cortheon` object, or null when the reply carries none. */
export function readCortheonReport(value: unknown): CortheonReply | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const raw = value as Record<string, unknown>
  const sources = Array.isArray(raw.sources) ? raw.sources.filter(webSource).slice(0, MAX_SOURCES) : []
  return {
    task: typeof raw.task === 'string' && raw.task.length <= 64 ? raw.task : null,
    searches: count(raw.searches) ?? 0,
    pagesRead: count(raw.pages_read) ?? 0,
    sources,
  }
}

export function recordModelCall(acc: TurnStatsAccumulator, sample: ModelCallSample): void {
  acc.modelCalls++
  if (sample.cortheon) {
    acc.cortheonCalls++
    acc.cortheonSearches += sample.cortheon.searches
    acc.cortheonPages += sample.cortheon.pagesRead
    acc.cortheonTask ??= sample.cortheon.task
    for (const source of sample.cortheon.sources) if (!acc.cortheonSources.includes(source) && acc.cortheonSources.length < MAX_SOURCES) acc.cortheonSources.push(source)
  }
  const { requestAt, firstTokenAt, lastTokenAt } = sample
  const engine = sample.engine ?? null
  if (acc.firstTokenMs == null) {
    if (engine?.firstTokenSeconds != null) acc.firstTokenMs = Math.round(engine.firstTokenSeconds * 1000)
    else if (finite(firstTokenAt) && finite(requestAt) && firstTokenAt >= requestAt) acc.firstTokenMs = firstTokenAt - requestAt
  }
  const prompt = count(sample.promptTokens)
  const cached = prompt == null ? null : Math.min(prompt, count(sample.cachedTokens) ?? 0)
  if (prompt != null) {
    acc.promptCalls++
    acc.promptTokens += prompt
    if (count(sample.cachedTokens) != null) { acc.cachedCalls++; acc.cachedTokens += cached ?? 0 }
  }
  if (engine) {
    // The engine divides its decoded tokens (all but the first) by these seconds.
    if (engine.decodeSeconds > 0) {
      acc.engineDecodeTokens += Math.round(engine.decodeTokensPerSecond * engine.decodeSeconds)
      acc.engineDecodeSeconds += engine.decodeSeconds
    }
    if (engine.prefillSeconds != null && prompt != null) {
      acc.prefillTokens += prompt - (cached ?? 0)
      acc.prefillSeconds += engine.prefillSeconds
    }
  }
  const completionTokens = count(sample.completionTokens)
  if (completionTokens == null) return
  acc.usageCalls++
  acc.completionTokens += completionTokens
  // Decode excludes the first token: n tokens span n - 1 inter-token gaps.
  if (completionTokens >= 2 && finite(firstTokenAt) && finite(lastTokenAt) && lastTokenAt > firstTokenAt) {
    acc.decodeTokens += completionTokens - 1
    acc.decodeMs += lastTokenAt - firstTokenAt
  }
}

export function recordToolCalls(acc: TurnStatsAccumulator, count: number): void {
  if (finite(count) && count > 0) acc.toolCalls += Math.floor(count)
}

export function finalizeTurnStats(acc: TurnStatsAccumulator, route: { model: string; providerId: string | null }): AssistantTurnStats {
  const stats: AssistantTurnStats = {
    model: route.model,
    providerId: route.providerId,
    modelCalls: acc.modelCalls,
    toolCalls: acc.toolCalls,
  }
  if (acc.usageCalls > 0) stats.completionTokens = acc.completionTokens
  // The engine's own decode figures win; the stream's token times are the fallback.
  if (acc.engineDecodeTokens > 0 && acc.engineDecodeSeconds > 0) {
    stats.decodeTokensPerSecond = round(acc.engineDecodeTokens / acc.engineDecodeSeconds, 1)
    stats.decodeSeconds = round(acc.engineDecodeSeconds, 3)
  } else if (acc.decodeTokens > 0 && acc.decodeMs > 0) {
    stats.decodeTokensPerSecond = round((acc.decodeTokens / acc.decodeMs) * 1000, 1)
    stats.decodeSeconds = round(acc.decodeMs / 1000, 3)
  }
  // A fully cached prompt has no new tokens: prefill is left out rather than infinite or zero.
  if (acc.prefillTokens > 0 && acc.prefillSeconds > 0) stats.prefillTokensPerSecond = round(acc.prefillTokens / acc.prefillSeconds, 1)
  if (acc.promptCalls > 0) stats.promptTokens = acc.promptTokens
  if (acc.cachedCalls > 0) stats.cachedTokens = acc.cachedTokens
  if (acc.firstTokenMs != null) stats.firstTokenMs = acc.firstTokenMs
  if (acc.cortheonCalls > 0) stats.cortheon = { task: acc.cortheonTask, searches: acc.cortheonSearches, pagesRead: acc.cortheonPages, sources: acc.cortheonSources.slice() }
  return stats
}
