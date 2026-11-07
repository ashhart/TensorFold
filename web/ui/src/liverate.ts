// Estimate live tokens from text calibrated by engine counts; a draft round's event may hold several tokens.
export const DEFAULT_CHARS_PER_TOKEN = 4

type Reply = { role: string; content: string; reasoning?: string; stats?: { tokens: number | null; measured: string } }


export function charsPerToken(msgs: Reply[]): number {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    const chars = m.content.length + (m.reasoning?.length ?? 0)
    if (m.role === 'assistant' && m.stats?.measured === 'engine' && m.stats.tokens && m.stats.tokens >= 20 && chars) {
      return Math.min(8, Math.max(1.5, chars / m.stats.tokens))
    }
  }
  return DEFAULT_CHARS_PER_TOKEN
}


export function liveRate(chars: number, seconds: number, ratio: number): number | null {
  return seconds > 0.3 && chars > 0 ? chars / ratio / seconds : null
}


export function prefillRate(rt: { prefill_seconds?: number } | undefined, prompt: number | undefined, cached: number | undefined): number | undefined {
  const read = prompt != null ? prompt - (cached ?? 0) : 0
  return rt?.prefill_seconds && read > 0 ? read / rt.prefill_seconds : undefined
}
