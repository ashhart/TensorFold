// The speed shown while a reply is written. A stream event is not a token: TensorFold sends one per verification
// round, and with drafts a round carries several tokens (1.75 on a GB10, more on a Mac), so counting events reads
// low. The page counts the text instead, at the characters per token the engine counted for the last reply.
export const DEFAULT_CHARS_PER_TOKEN = 4

type Reply = { role: string; content: string; reasoning?: string; stats?: { tokens: number | null; measured: string } }

/** Characters per token of the latest reply the engine counted, or the default before any. */
export function charsPerToken(msgs: Reply[]): number {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    const chars = m.content.length + (m.reasoning?.length ?? 0)
    if (m.role === 'assistant' && m.stats?.measured === 'engine' && m.stats.tokens && m.stats.tokens >= 20 && chars) {
      return Math.min(8, Math.max(1.5, chars / m.stats.tokens))   // bounded: a reply of code or of CJK text is not a ruler
    }
  }
  return DEFAULT_CHARS_PER_TOKEN
}

/** Tokens a second from the text received since the first token: an estimate, which the engine's figure replaces. */
export function liveRate(chars: number, seconds: number, ratio: number): number | null {
  return seconds > 0.3 && chars > 0 ? chars / ratio / seconds : null
}

/** The prompt's reading speed, tokens a second: the tokens not restored from the cache over the engine's prompt pass. */
export function prefillRate(rt: { prefill_seconds?: number } | undefined, prompt: number | undefined, cached: number | undefined): number | undefined {
  const read = prompt != null ? prompt - (cached ?? 0) : 0
  return rt?.prefill_seconds && read > 0 ? read / rt.prefill_seconds : undefined
}
