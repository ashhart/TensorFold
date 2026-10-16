// The quiet facts line under a final answer: "Nemotron 3.5 Lightning · 2 tool calls · decode 212 tok/s · prefill 3,140 tok/s · first token 0.4 s".
// Every figure comes from AssistantTurnStats; a figure that was not measured is left out.
import { modelDisplayName } from './provider-accounts'
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

export function turnStatsSegments(stats: AssistantTurnStats, modelLabel: string): string[] {
  const parts = [modelLabel]
  if (stats.toolCalls > 0) parts.push(`${stats.toolCalls} tool call${stats.toolCalls === 1 ? '' : 's'}`)
  if (known(stats.decodeTokensPerSecond) && stats.decodeTokensPerSecond > 0) parts.push(`decode ${whole(stats.decodeTokensPerSecond)} tok/s`)
  if (known(stats.prefillTokensPerSecond) && stats.prefillTokensPerSecond > 0) parts.push(`prefill ${whole(stats.prefillTokensPerSecond)} tok/s`)
  if (known(stats.firstTokenMs)) parts.push(`first token ${(stats.firstTokenMs / 1000).toFixed(1)} s`)
  return parts.filter(Boolean)
}

/** "Cortheon · 2 searches · 2 pages read" when the engine ran Cortheon for the run, else null. */
export function cortheonSummary(stats: AssistantTurnStats): string | null {
  const c = stats.cortheon
  if (!c) return null
  const parts = ['Cortheon']
  if (c.searches > 0) parts.push(`${c.searches} search${c.searches === 1 ? '' : 'es'}`)
  if (c.pagesRead > 0) parts.push(`${c.pagesRead} page${c.pagesRead === 1 ? '' : 's'} read`)
  if (parts.length === 1) parts.push('no web research needed')
  return parts.join(' · ')
}

/** Hover text for the line: prompt, cached and new token counts and the decode seconds; undefined when none are known. */
export function turnStatsTitle(stats: AssistantTurnStats): string | undefined {
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
