// Context governance (VST-016). Two pure pieces:
//
//  1) Bounded context FRAGMENTS — every variable-size block injected into the model's
//     context (project guide, durable project memory, persona, channel prompt, …) becomes a
//     fragment with provenance (source), a source priority that sets its truncation policy,
//     the context epoch it was built in (how many compactions the thread has survived), an
//     estimated token cost, and an explicit truncated flag. Nothing enters the window
//     unbounded, and truncation is never silent: a cut fragment SAYS so, in-band, so the
//     model and the user both see that content was dropped.
//
//  2) Bounded TOOL OUTPUT with managed spill records — a tool result larger than the cap is
//     cut head+tail (build logs put the error at the end), and when the host provides a
//     managed-output ref the FULL output is preserved outside the workspace, with the
//     in-band note naming the ref and how to page it (read_file offset/limit).
//
// Pure derivations only: loop.ts owns the wiring; the host owns protected spill storage;
// compaction owns folding. The epoch here is simply the compaction ledger length.

export type ContextFragmentSource =
  | 'project_guide'
  | 'project_memory'
  | 'persona'
  | 'goal'
  | 'capabilities'
  | 'channel_prompt'
  | 'tool_output'

export type ContextFragmentPriority = 'pinned' | 'high' | 'normal' | 'low'

// Truncation policy by priority (chars). 'pinned' content (the user's goal, the base system
// prompt) is never cut by this layer — cutting it would change what the agent IS doing.
export const FRAGMENT_CHAR_BUDGETS: Record<ContextFragmentPriority, number> = {
  pinned: Number.POSITIVE_INFINITY,
  high: 24_000,
  normal: 12_000,
  low: 4_000,
}

// Default priority per source; callers may override per fragment.
export const DEFAULT_FRAGMENT_PRIORITY: Record<ContextFragmentSource, ContextFragmentPriority> = {
  project_guide: 'high',
  project_memory: 'high',
  persona: 'normal',
  goal: 'pinned',
  capabilities: 'normal',
  channel_prompt: 'normal',
  tool_output: 'normal',
}

export interface ContextFragment {
  source: ContextFragmentSource
  priority: ContextFragmentPriority
  // Context epoch the fragment was assembled in: the number of compactions the thread had
  // survived at build time. Lets "was this built before or after that fold?" be answered.
  epoch: number
  // Bounded text, including the visible truncation notice when cut.
  text: string
  chars: number
  estTokens: number
  truncated: boolean
  // Chars of the original, pre-bound text (equals chars when not truncated, sans notice).
  originalChars: number
}

// The persistable, text-free view for the thread's fragment ledger (audit, not payload).
export interface ContextFragmentStat {
  source: ContextFragmentSource
  priority: ContextFragmentPriority
  epoch: number
  chars: number
  estTokens: number
  truncated: boolean
  originalChars: number
  note?: string
}

// ~4 chars/token — the same heuristic compaction.estimateTokens uses for messages, applied
// to a bare string so fragments and messages report comparable costs.
export function estimateTextTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

// The thread's current context epoch: how many compaction folds its transcript has survived.
export function contextEpoch(thread: { compactions?: unknown[] | null }): number {
  return thread.compactions?.length ?? 0
}

export function buildContextFragment(input: {
  source: ContextFragmentSource
  text: string
  epoch: number
  priority?: ContextFragmentPriority
  maxChars?: number
}): ContextFragment {
  const priority = input.priority ?? DEFAULT_FRAGMENT_PRIORITY[input.source]
  const budget = input.maxChars ?? FRAGMENT_CHAR_BUDGETS[priority]
  const original = String(input.text ?? '')
  let text = original
  let truncated = false
  if (Number.isFinite(budget) && original.length > budget) {
    truncated = true
    // Head-biased cut: these blocks front-load their important content (guides, memory).
    text = `${original.slice(0, Math.max(0, budget))}\n[Context fragment truncated by TensorFold: showing ${Math.max(0, budget)} of ${original.length} chars from ${input.source} (priority ${priority}).]`
  }
  return {
    source: input.source,
    priority,
    epoch: input.epoch,
    text,
    chars: text.length,
    estTokens: estimateTextTokens(text),
    truncated,
    originalChars: original.length,
  }
}

export function fragmentStat(fragment: ContextFragment, note?: string): ContextFragmentStat {
  const { text: _text, ...rest } = fragment
  return note ? { ...rest, note } : rest
}

// ---------------------------------------------------------------------------
// Bounded tool output + managed spill files

export const TOOL_OUTPUT_MAX_CHARS = 16_000
export const TOOL_OUTPUT_HEAD_CHARS = 12_000
export const TOOL_OUTPUT_TAIL_CHARS = 2_500
export const MANAGED_OUTPUT_REF_PREFIX = 'vontra-output://'

export interface BoundedToolOutput {
  // What goes to the model (and the timeline item): bounded, with the in-band note.
  content: string
  truncated: boolean
  totalChars: number
  // Set when the output was cut AND a spill ref was offered: the host must write the
  // FULL output to protected app state (best-effort; on failure use `content` —
  // the note only names the file when the write succeeded, so never lie about it).
  spill: { relPath: string; totalChars: number; totalLines: number } | null
}

/**
 * Bound a tool result for the context window (VST-016). Under the cap it passes through
 * untouched. Over the cap it keeps the head and the tail (errors live at the end of build
 * logs) and — when `spillRelPath` is provided — directs the full output to a managed file,
 * named in the note together with how to page it. `spillWritten: false` produces the
 * honest fallback note (content dropped, nothing pretends a file exists).
 */
export function boundToolOutput(input: {
  tool: string
  output: string
  spillRelPath?: string | null
  spillWritten?: boolean
  maxChars?: number
}): BoundedToolOutput {
  const output = String(input.output ?? '')
  const maxChars = input.maxChars ?? TOOL_OUTPUT_MAX_CHARS
  if (output.length <= maxChars) {
    return { content: output, truncated: false, totalChars: output.length, spill: null }
  }
  const head = output.slice(0, TOOL_OUTPUT_HEAD_CHARS)
  const tail = output.slice(-TOOL_OUTPUT_TAIL_CHARS)
  const totalLines = output.split('\n').length
  const spilled = !!(input.spillRelPath && input.spillWritten)
  const note = spilled
    ? `[Tool output truncated by TensorFold: ${input.tool} produced ${output.length.toLocaleString()} chars (${totalLines.toLocaleString()} lines); showing the first ${TOOL_OUTPUT_HEAD_CHARS.toLocaleString()} and last ${TOOL_OUTPUT_TAIL_CHARS.toLocaleString()}. Full output saved to ${input.spillRelPath} — use read_file with offset/limit to page through it.]`
    : `[Tool output truncated by TensorFold: ${input.tool} produced ${output.length.toLocaleString()} chars (${totalLines.toLocaleString()} lines); showing the first ${TOOL_OUTPUT_HEAD_CHARS.toLocaleString()} and last ${TOOL_OUTPUT_TAIL_CHARS.toLocaleString()}. The middle was dropped (no spill file available). Re-run with a narrower query if you need the rest.]`
  return {
    content: `${head}\n${note}\n${tail}`,
    truncated: true,
    totalChars: output.length,
    spill: spilled ? { relPath: input.spillRelPath!, totalChars: output.length, totalLines } : null,
  }
}

/** Opaque protected-state ref for the Nth oversized output of a run. */
export function toolOutputSpillPath(runId: string, seq: number, tool: string): string {
  const runPart = String(runId ?? 'run').replace(/[^a-zA-Z0-9-]/g, '').slice(0, 40) || 'run'
  const toolPart = String(tool ?? 'tool').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 32) || 'tool'
  return `${MANAGED_OUTPUT_REF_PREFIX}${runPart}-${String(seq).padStart(3, '0')}-${toolPart}.txt`
}

export function isManagedOutputRef(value: unknown): value is string {
  return typeof value === 'string' && value.startsWith(MANAGED_OUTPUT_REF_PREFIX)
}
