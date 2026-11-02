// Pure repetition, truncation, verification, and progress checks for the agent loop.

import { isOpenCodeProtocol } from './opencode-engine'
import { type CompactionLedgerEntry, type Provider, type TimelineItem } from './types'

export const THRASH_LIMIT = 3

export const REPEAT_LIMIT = 3

export const MAX_TEXT_LOOP_BREAKS = 2

export const LOOP_CHECK_STRIDE = 280

export const LOOP_CHECK_MIN = 400

export function repeatGuardNote(count: number, limit = REPEAT_LIMIT): string | null {
  if (count < limit) return null
  return `\n\n[guardrail] You have already run this exact action ${count} times. Stop repeating it — use the results you already have, or wrap up and report to the user.`
}

// Count normalized executor arguments, retaining array order but ignoring object key order and whitespace.
export function toolCallSignature(name: string, args: Record<string, any>): string {
  return `${name}|${JSON.stringify(args, (_key, value) => value && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, value[key]]))
    : value)}`
}

function tailLineCycle(text: string): boolean {
  const lines = text
    .split('\n')
    .map((l) => l.trim().replace(/\s+/g, ' ').toLowerCase())
    .filter(Boolean)
  if (lines.length < 4) return false
  const tail = lines.slice(-120)

  const requiredRepeats = (p: number) => (p === 1 ? 6 : p <= 3 ? 4 : 3)
  const maxPeriod = Math.min(30, Math.floor(tail.length / 2))
  for (let p = 1; p <= maxPeriod; p++) {
    const reps = requiredRepeats(p)
    if (p * reps > tail.length) continue
    const last = tail.slice(tail.length - p).join('\n')
    let matched = true
    for (let k = 1; k < reps; k++) {
      const block = tail.slice(tail.length - (k + 1) * p, tail.length - k * p).join('\n')
      if (block !== last) { matched = false; break }
    }
    if (matched) return true
  }
  return false
}

function tailCharCycle(text: string, window = 400, maxPeriod = 80, minRepeats = 4, minSpan = 40): boolean {
  const t = text.replace(/\s+/g, ' ').slice(-window)
  for (let p = 1; p <= maxPeriod && p * minRepeats <= t.length; p++) {
    const unit = t.slice(t.length - p)
    if (!unit.trim()) continue
    let reps = 1
    while ((reps + 1) * p <= t.length && t.slice(t.length - (reps + 1) * p, t.length - reps * p) === unit) reps++
    if (reps >= minRepeats && reps * p >= minSpan) return true
  }
  return false
}

export function detectTextLoop(text: string): boolean {
  if (!text || text.length < 24) return false
  return tailLineCycle(text) || tailCharCycle(text)
}

export function providerNeedsTextLoopBreaker(provider: Pick<Provider, 'baseUrl' | 'kind' | 'transport'> | null | undefined): boolean {
  if (!provider) return false
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli' || provider.transport === 'pi-cli' || provider.transport === 'cursor-cli' || provider.transport === 'gemini-cli' || isOpenCodeProtocol(provider.transport)) return false
  if (provider.kind === 'local') return true
  return /(^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$))/.test(provider.baseUrl || '')
}

export type TextTruncationAction = 'none' | 'continue' | 'fail'

export function textTruncationAction(finishReason: string | null, toolCallCount: number, priorContinuations: number, maxContinuations = 2): TextTruncationAction {
  if (finishReason !== 'length' || toolCallCount > 0) return 'none'
  return priorContinuations < maxContinuations ? 'continue' : 'fail'
}

export function compactionMadeProgress(entry: CompactionLedgerEntry | null, beforeTokens: number, afterTokens: number): boolean {
  return Boolean(entry) && Number.isFinite(beforeTokens) && Number.isFinite(afterTokens) && afterTokens < beforeTokens
}

const SEARCH_LOOP_TOOLS = new Set(['web_search', 'web_fetch', 'search', 'retrieval_search', 'browser_read', 'browser_summary', 'browser_list'])

const SEARCH_LOOP_LIMIT = 4

export function searchLoopNote(toolName: string, count: number, limit = SEARCH_LOOP_LIMIT): string | null {
  if (!SEARCH_LOOP_TOOLS.has(toolName)) return null
  if (count < limit) return null
  return `\n\n[guardrail] You have already run ${count} searches in this turn. Stop searching — answer the user with what you already have and name any gaps, instead of rephrasing the query and searching again.`
}

const TOOL_CADENCE_EXEMPT = new Set([
  'read_file', 'list_dir', 'search', 'glob', 'apply_patch', 'edit_file', 'write_file',
  'ast_grep_search', 'ast_grep_rewrite_preview', 'safe_refactor_preview', 'symbol_index',
  'codebase_map', 'code_intelligence', 'semgrep_scan',
])

const TOOL_CADENCE_LIMIT = 6

export function toolCadenceNote(toolName: string, count: number, limit = TOOL_CADENCE_LIMIT): string | null {
  if (TOOL_CADENCE_EXEMPT.has(toolName)) return null
  if (count <= limit) return null
  return `\n\n[guardrail] You have called ${toolName} ${count} times in this turn. Stop calling it — consolidate what these results already tell you and give the user a final answer (or take a clearly different action). Do not run it again with reworded arguments.`
}

export const NO_PROGRESS_TOOLS = new Set([
  'read_file', 'list_dir', 'search', 'glob', 'web_search', 'web_fetch',
  'retrieval_search', 'browser_read', 'browser_summary',
])

const NO_PROGRESS_LIMIT = 2

export function noProgressNote(toolName: string, priorCount: number, limit = NO_PROGRESS_LIMIT): string | null {
  if (!NO_PROGRESS_TOOLS.has(toolName)) return null

  const occurrences = priorCount + 1
  if (occurrences < limit) return null
  return `\n\n[guardrail] ${toolName} has now returned the same result ${occurrences} times for this query — repeating it is not making progress. Do not run it again with the same (or a trivially reworded) input: use what you already have, or take a materially different action.`
}

const VERIFY_WRITE_TOOLS = new Set(['write_file', 'edit_file', 'apply_patch', 'patch_file'])

const VERIFY_TEST_COMMAND_RE =
  /\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|typecheck|lint|build|verify|check)\b|\b(?:vitest|jest|mocha|ava|pytest|ruff|eslint|tsc|cargo\s+test|go\s+test|dotnet\s+test|mvn\s+test|gradle\s+test)\b/i

export interface VerifyOnStopSnapshot {
  changedFiles: string[]
  testCommands: string[]
  needsVerification: boolean
  freshVerification: boolean
  lastChangeIndex: number
  lastVerificationIndex: number
}

export function verifyOnStopSnapshot(timeline: TimelineItem[], startIndex = 0): VerifyOnStopSnapshot {
  const changedFiles: string[] = []
  const testCommands: string[] = []
  let lastChangeIndex = -1
  let lastVerificationIndex = -1

  timeline.slice(startIndex).forEach((item, relativeIndex) => {
    if (item.kind !== 'tool') return
    if ((item.tool && VERIFY_WRITE_TOOLS.has(item.tool)) || item.diff?.length) {
      lastChangeIndex = relativeIndex
      const pathValue = item.path ?? item.args?.path ?? item.args?.file ?? item.args?.target
      if (typeof pathValue === 'string' && pathValue.trim()) changedFiles.push(pathValue.trim())
    }
    if (item.command || item.tool === 'run_command') {
      const command = item.command ?? String(item.args?.command ?? '')
      if (VERIFY_TEST_COMMAND_RE.test(command)) {
        testCommands.push(command)
        if (!item.state || item.state === 'ok') lastVerificationIndex = relativeIndex
      }
    }
  })

  const uniqueChangedFiles = uniqueStrings(changedFiles)
  const freshVerification = uniqueChangedFiles.length > 0 && lastVerificationIndex > lastChangeIndex
  return {
    changedFiles: uniqueChangedFiles,
    testCommands: uniqueStrings(testCommands),
    needsVerification: uniqueChangedFiles.length > 0 && !freshVerification,
    freshVerification,
    lastChangeIndex,
    lastVerificationIndex,
  }
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const clean = value.replace(/\s+/g, ' ').trim()
    if (!clean || seen.has(clean)) continue
    seen.add(clean)
    out.push(clean)
  }
  return out
}

export function isTruncatedToolCallArgs(argsParseFailed: boolean, finishReason: string | null): boolean {
  return argsParseFailed && finishReason === 'length'
}
