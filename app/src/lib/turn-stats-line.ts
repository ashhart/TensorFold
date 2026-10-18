// The quiet facts line under a final answer: "Nemotron 3.5 Lightning · 2 tool calls · decode 212 tok/s · prefill 3,140 tok/s · first token 0.4 s".
// Every figure comes from AssistantTurnStats; a figure that was not measured is left out.
import { modelDisplayName } from './provider-accounts'
import { currentUiLanguage } from './preferences/current-locale'
import { authoredText } from './preferences/language-packs'
import { normalizeLanguage } from '../../shared/languages'
import type { AssistantTurnStats, Provider } from './types'

const FORMAT_SUFFIX = /(?:-(?:mlx|gguf|bf16|fp16|fp8|nvfp4|awq|gptq|q\d\w*|\d+bit))+$/i

/** A readable name for a Hugging Face style repo id: "TensorFold/Qwen3.8-27B-MLX-4bit" to "Qwen3.8 27B". */
export function repoDisplayName(repo: string): string {
  if (/^local\/[a-f0-9]{64}$/.test(repo)) return 'Local checkpoint'
  const name = repo.split('/').pop() ?? repo
  return name.replace(FORMAT_SUFFIX, '').replace(/[-_]+/g, ' ').trim() || name
}

export function statsModelLabel(model: string, provider: Provider | null | undefined, titles?: ReadonlyArray<{ repo: string; title: string }>): string {
  const titled = titles?.find((entry) => entry.repo === model)?.title
  if (titled) return titled
  if (provider?.source === 'tensorfold' || model.includes('/')) return repoDisplayName(model)
  return modelDisplayName(provider, model)
}

const whole = (value: number) => Math.round(value).toLocaleString('en-US')
const known = (value: number | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0
const seconds = (value: number) => `${value < 1 ? value.toFixed(2) : value.toFixed(1)} s`

export function turnStatsSegments(stats: AssistantTurnStats, modelLabel: string, locale: unknown = currentUiLanguage()): string[] {
  if (normalizeLanguage(locale) !== 'en') {
    const tx = (source: string, params = {}) => authoredText(locale, source, params), parts = [modelLabel]
    if (stats.toolCalls > 0) parts.push(tx('Tool calls: {count}', { count: stats.toolCalls }))
    if (known(stats.decodeTokensPerSecond) && stats.decodeTokensPerSecond > 0) parts.push(tx('Decode: {speed} tok/s', { speed: Math.round(stats.decodeTokensPerSecond) }))
    if (known(stats.prefillTokensPerSecond) && stats.prefillTokensPerSecond > 0) parts.push(tx('Prefill: {speed} tok/s', { speed: Math.round(stats.prefillTokensPerSecond) }))
    if (known(stats.firstTokenMs)) parts.push(tx('First token: {seconds} s', { seconds: Math.round(stats.firstTokenMs / 100) / 10 }))
    return parts.filter(Boolean)
  }
  const parts = [modelLabel]
  if (stats.toolCalls > 0) parts.push(`${stats.toolCalls} tool call${stats.toolCalls === 1 ? '' : 's'}`)
  if (known(stats.decodeTokensPerSecond) && stats.decodeTokensPerSecond > 0) parts.push(`decode ${whole(stats.decodeTokensPerSecond)} tok/s`)
  if (known(stats.prefillTokensPerSecond) && stats.prefillTokensPerSecond > 0) parts.push(`prefill ${whole(stats.prefillTokensPerSecond)} tok/s`)
  if (known(stats.firstTokenMs)) parts.push(`first token ${(stats.firstTokenMs / 1000).toFixed(1)} s`)
  return parts.filter(Boolean)
}

/** "Cortheon · 2 searches · 2 pages read" when the engine ran Cortheon for the run, else null. */
export function cortheonSummary(stats: AssistantTurnStats, locale: unknown = currentUiLanguage()): string | null {
  if (normalizeLanguage(locale) !== 'en') {
    if (!stats.cortheon) return null
    const tx = (source: string, params = {}) => authoredText(locale, source, params), parts = ['Cortheon']
    if (stats.cortheon.searches > 0) parts.push(tx('Searches: {count}', { count: stats.cortheon.searches }))
    if (stats.cortheon.pagesRead > 0) parts.push(tx('Pages read: {count}', { count: stats.cortheon.pagesRead }))
    if (parts.length === 1) parts.push(tx('No web research needed'))
    return parts.join(' · ')
  }
  const c = stats.cortheon
  if (!c) return null
  const parts = ['Cortheon']
  if (c.searches > 0) parts.push(`${c.searches} search${c.searches === 1 ? '' : 'es'}`)
  if (c.pagesRead > 0) parts.push(`${c.pagesRead} page${c.pagesRead === 1 ? '' : 's'} read`)
  if (parts.length === 1) parts.push('no web research needed')
  return parts.join(' · ')
}

/** Hover text for the line: prompt, cached and new token counts and the decode seconds; undefined when none are known. */
export function turnStatsTitle(stats: AssistantTurnStats, locale: unknown = currentUiLanguage()): string | undefined {
  if (normalizeLanguage(locale) !== 'en') {
    const tx = (source: string, params = {}) => authoredText(locale, source, params), lines: string[] = []
    if (known(stats.promptTokens)) lines.push(tx('Prompt: {count} tokens', { count: Math.round(stats.promptTokens) }))
    if (known(stats.promptTokens) && known(stats.cachedTokens)) {
      lines.push(tx('Cached: {count} tokens', { count: Math.round(stats.cachedTokens) }))
      lines.push(tx('New: {count} tokens', { count: Math.round(Math.max(0, stats.promptTokens - stats.cachedTokens)) }))
    }
    if (known(stats.completionTokens)) lines.push(tx('Output: {count} tokens', { count: Math.round(stats.completionTokens) }))
    if (known(stats.decodeSeconds) && stats.decodeSeconds > 0) lines.push(tx('Decode: {seconds} s', { seconds: Number(stats.decodeSeconds.toFixed(stats.decodeSeconds < 1 ? 2 : 1)) }))
    return lines.length ? lines.join('\n') : undefined
  }
  const lines: string[] = []
  if (known(stats.promptTokens)) lines.push(`Prompt ${whole(stats.promptTokens)} tokens`)
  if (known(stats.promptTokens) && known(stats.cachedTokens)) {
    lines.push(`Cached ${whole(stats.cachedTokens)} tokens`)
    lines.push(`New ${whole(Math.max(0, stats.promptTokens - stats.cachedTokens))} tokens`)
  }
  if (known(stats.completionTokens)) lines.push(`Output ${whole(stats.completionTokens)} tokens`)
  if (known(stats.decodeSeconds) && stats.decodeSeconds > 0) lines.push(`Decode ${seconds(stats.decodeSeconds)}`)
  return lines.length ? lines.join('\n') : undefined
}
