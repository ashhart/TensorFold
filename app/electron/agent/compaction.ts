// Pure, dependency-free helpers for context compaction. Kept separate from
// loop.ts (which does the network call) so the boundary math is unit-testable.
import type { ApiMessage } from './types'
import { contentText } from './content.ts'

// Rough token estimate (~4 chars/token) — works across every OpenAI-compatible
// endpoint without depending on server-reported usage (Ollama/LM Studio omit it).
export function estimateTokens(messages: ApiMessage[]): number {
  let chars = 0
  for (const m of messages) {
    chars += contentText(m.content).length
    if (m.tool_calls) for (const tc of m.tool_calls) chars += tc.function.name.length + tc.function.arguments.length
    chars += 8 // per-message structural overhead
  }
  return Math.ceil(chars / 4)
}

export const kfmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(n))

// Flatten messages to a bounded transcript for the summarizer.
export function renderTranscript(msgs: ApiMessage[]): string {
  const parts = msgs.map((m) => {
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const calls = m.tool_calls.map((t) => `${t.function.name}(${t.function.arguments})`).join(', ')
      return `ASSISTANT: ${contentText(m.content)}\n[called: ${calls}]`
    }
    if (m.role === 'tool') return `TOOL[${m.name}] → ${contentText(m.content)}`
    return `${m.role.toUpperCase()}: ${contentText(m.content)}`
  })
  let text = parts.join('\n\n')
  const CAP = 48_000
  if (text.length > CAP) text = text.slice(0, 30_000) + '\n\n…[middle elided]…\n\n' + text.slice(-18_000)
  return text
}

// Decide where to cut for compaction. Keeps the leading system prompt and the
// most-recent `keepRecent` messages verbatim; folds the middle. Crucially, it
// never leaves the kept tail starting on an orphan `tool` result (which the API
// rejects) — so an assistant(tool_calls)/tool group is never split. Returns the
// cut index + system-message count, or null when there's nothing safe to fold.
export function planCompaction(msgs: ApiMessage[], keepRecent: number): { cut: number; sysCount: number } | null {
  const sysCount = msgs[0]?.role === 'system' ? 1 : 0
  let cut = Math.max(sysCount, msgs.length - keepRecent)
  // The latest user turn is the current task contract. Keep it verbatim even when a
  // long assistant/tool group would otherwise consume the whole recent-message budget.
  let latestUser = -1
  for (let i = msgs.length - 1; i >= sysCount; i--) {
    if (msgs[i].role === 'user') { latestUser = i; break }
  }
  if (latestUser >= sysCount) cut = Math.min(cut, latestUser)
  // ...but preserving the latest user turn verbatim must never disable folding for a
  // long single-user run (the primary auto-compaction audience): when nothing before
  // it is foldable, fall back to the original boundary — the verified summary is
  // required to preserve the user goal, so folding the original request is safe.
  if (cut <= sysCount || cut - sysCount < 2) cut = Math.max(sysCount, msgs.length - keepRecent)
  while (cut < msgs.length && msgs[cut].role === 'tool') cut++ // keep tool results attached to their assistant turn
  if (cut <= sysCount) return null
  if (cut - sysCount < 2) return null // not worth folding a single message
  return { cut, sysCount }
}

export function deterministicCompactionSummary(messages: ApiMessage[], maxChars = 12_000): string {
  const transcript = renderTranscript(messages)
  if (transcript.length <= maxChars) return transcript
  const selected = messages.length <= 48
    ? messages
    : [...messages.slice(0, 12), ...messages.slice(-36)]
  const omitted = messages.length - selected.length
  const separator = '\n\n'
  const omission = omitted > 0 ? `[${omitted} middle messages omitted deterministically]\n\n` : ''
  const perMessage = Math.max(120, Math.floor((maxChars - omission.length - separator.length * selected.length) / selected.length))
  const rendered = selected.map((message) => {
    const text = renderTranscript([message])
    if (text.length <= perMessage) return text
    const marker = '\n[message middle omitted]\n'
    const head = Math.max(60, Math.floor((perMessage - marker.length) * 0.6))
    const tail = Math.max(40, perMessage - marker.length - head)
    return `${text.slice(0, head)}${marker}${text.slice(-tail)}`
  })
  return `${omission}${rendered.join(separator)}`.slice(0, maxChars)
}

// ---- Anti-thrash (pure decision logic; compactThread in loop.ts owns the I/O) ----
// Once compaction stops meaningfully shrinking a thread, re-running it every turn is
// pure cost with no benefit. These two pure functions decide (a) whether to skip a
// pass given the streak left by past passes, and (b) how the streak should change
// after a pass completes — kept here, not inlined in loop.ts, so the threshold math
// is unit-testable without mocking the network.
export const INEFFECTIVE_SAVINGS_RATIO = 0.10
export const INEFFECTIVE_STREAK_LIMIT = 2

/**
 * True when compaction should be skipped outright: the streak already hit the
 * limit, and the thread hasn't grown substantially past the size it was at when
 * the last (ineffective) pass finished — so there's unlikely to be meaningfully
 * more to fold than last time.
 */
export function shouldSkipCompactionForThrash(streak: number | undefined, tokensNow: number, tokensAfterLastPass: number | undefined): boolean {
  if ((streak ?? 0) < INEFFECTIVE_STREAK_LIMIT) return false
  if (tokensAfterLastPass === undefined) return false // no prior pass to compare against
  const grownEnough = tokensNow > tokensAfterLastPass * (1 + INEFFECTIVE_SAVINGS_RATIO)
  return !grownEnough
}

/**
 * Streak after a completed pass: extends by one when the pass saved well under
 * the worthwhile threshold (relative to the size of the slice it folded), resets
 * to 0 the moment a pass saves a useful amount again.
 */
export function nextIneffectiveStreak(priorStreak: number | undefined, savedChars: number, foldedSliceChars: number): number {
  if (foldedSliceChars <= 0) return priorStreak ?? 0
  const ineffective = savedChars < foldedSliceChars * INEFFECTIVE_SAVINGS_RATIO
  return ineffective ? (priorStreak ?? 0) + 1 : 0
}
