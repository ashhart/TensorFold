// Model-only output preparation. Every retained span is copied from the original;
// the caller retains the raw result for the timeline, audit and loop fingerprints.
// No model request, application state, package dependency or tool execution lives here.
export const TOOL_OUTPUT_EXCERPT_MARKER = '[TensorFold tool-output excerpt v1;'
export const MODEL_TOOL_OUTPUT_MAX_CHARS = 16_000
export const MODEL_TOOL_OUTPUT_MAX_BYTES = 48_000
const STRUCTURAL_THRESHOLD = 6_000
const encoder = new TextEncoder()
const preparedMessages = new WeakSet<object>()

type Category = 'test' | 'shell' | 'search' | 'data'
type Span = { start: number; end: number }
type Line = Span & { text: string; diagnostic: boolean; noise: boolean }

export interface ToolOutputCompressionInput {
  tool: string
  output: string
  args?: Record<string, unknown>
  /** Execution metadata supplied by the host, never inferred from output text. */
  ok?: boolean
  meta?: string
  spillRef?: string | null
  maxChars?: number
  maxBytes?: number
}

export interface CompressedToolOutput {
  content: string
  compressed: boolean
  category: Category
  originalChars: number
  originalBytes: number
  omittedChars: number
  omittedDiagnosticLines: number
}

function categoryFor(input: ToolOutputCompressionInput): Category {
  if (input.tool === 'run_command') {
    // Tool identity and requested command select the policy. Output cannot opt
    // itself into a policy that discards arbitrary lines containing a keyword.
    const command = typeof input.args?.command === 'string' ? input.args.command.trim() : ''
    return /^(?:(?:npm|pnpm|yarn|bun)\s+(?:test\b|run\s+test(?:[:\s]|$))|(?:npx\s+)?(?:vitest|jest|pytest)\b|python(?:3)?\s+-m\s+pytest\b|node\s+--test\b|(?:go|cargo|dotnet)\s+test\b)/.test(command) ? 'test' : 'shell'
  }
  return ['search', 'glob', 'list_dir', 'web_search', 'retrieval_search', 'ast_grep_search'].includes(input.tool) ? 'search' : 'data'
}

const diagnosticPattern = /\b(?:error|failed|failure|fatal|traceback|exception|panic|assertion|denied|cancelled|canceled|timeout|timed out|redacted)\b|(?:fatal|error|panic):|\b(?:Type|Reference|Syntax|Range|Assertion)Error\b|^\s*(?:FAIL\b|not ok\b|E\s+|✗|×)/i
const summaryPattern = /^\s*(?:#\s*(?:tests|pass|fail|cancelled|skipped|todo|duration_ms)\b|(?:Test (?:Suites|Files)|Tests|Snapshots)\s*:|[= -]*\d+\s+(?:passed|failed|errors?)\b|\d+\/\d+\s+passed\b|(?:FAILED|PASSED)\s*\()/i
const passingPattern = /^(?:ok[ \t]+\d+(?:[ \t]+-[ \t]?.*)?|PASS[ \t]+\S.*|[✓✔][ \t]+.+|\.+)$/
const progressPattern = /^(?:\d{1,3}%|[|/\\-]|\[[#=>. -]+\][ \t]*\d{1,3}%)$/

function sourceLines(output: string, category: Category): Line[] {
  const lines: Line[] = []
  let start = 0
  let previous = ''
  let failureSection = false
  let diagnosticContinuation = false
  while (start < output.length) {
    const newline = output.indexOf('\n', start)
    const end = newline < 0 ? output.length : newline + 1
    const raw = output.slice(start, end)
    const text = raw.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '').trimEnd()
    const stripped = text.trimStart()
    if (stripped.startsWith('===') && stripped.endsWith('===')) failureSection = /\b(?:FAILURES|ERRORS)\b/.test(stripped)
    const summary = summaryPattern.test(stripped)
    if (summary) failureSection = false
    const continuation: boolean = diagnosticContinuation && /^(?:\s{2,}\S|\s*(?:at\s|File\s|Expected:|Received:|Caused by:|During handling of|The above exception))/i.test(text)
    const diagnostic: boolean = failureSection || diagnosticPattern.test(text) || summary || continuation
    diagnosticContinuation = diagnostic && !summary && !!text.trim()
    const isLog = category === 'test' || category === 'shell'
    // Only known passing/progress syntax and adjacent identical log lines are
    // removable without a size bound. Every other line stays until explicit
    // original-offset excerpts are necessary to meet the hard limits.
    const noise = isLog && !diagnostic && ((category === 'test' && passingPattern.test(stripped)) || progressPattern.test(stripped) || (raw === previous && !!stripped))
    lines.push({ start, end, text, diagnostic, noise })
    previous = raw
    start = end
  }
  return lines
}

function merge(spans: Span[], closeGap = 0): Span[] {
  const sorted = spans.filter((span) => span.end > span.start).sort((a, b) => a.start - b.start)
  const out: Span[] = []
  for (const span of sorted) {
    const last = out[out.length - 1]
    if (last && span.start <= last.end + closeGap) last.end = Math.max(last.end, span.end)
    else out.push({ ...span })
  }
  return out
}

function aligned(output: string, span: Span): Span {
  let { start, end } = span
  // UTF-16 offsets remain useful for exact source access without splitting a
  // supported Unicode code point at either excerpt boundary.
  if (start > 0 && /[\uDC00-\uDFFF]/.test(output[start]) && /[\uD800-\uDBFF]/.test(output[start - 1])) start++
  if (end < output.length && /[\uD800-\uDBFF]/.test(output[end - 1]) && /[\uDC00-\uDFFF]/.test(output[end])) end--
  return { start, end }
}

/** Host-only in-memory bookkeeping; tool text cannot forge this identity. */
export function rememberPreparedToolOutput<T extends object>(message: T): T {
  preparedMessages.add(message)
  return message
}

export function isPreparedToolOutput(message: object): boolean {
  return preparedMessages.has(message)
}

export function compressToolOutput(input: ToolOutputCompressionInput): CompressedToolOutput {
  const output = input.output
  const maxChars = input.maxChars ?? MODEL_TOOL_OUTPUT_MAX_CHARS
  const maxBytes = input.maxBytes ?? MODEL_TOOL_OUTPUT_MAX_BYTES
  if (!Number.isSafeInteger(maxChars) || !Number.isSafeInteger(maxBytes) || maxChars < 512 || maxBytes < 512) throw new RangeError('Tool output budgets must be safe integers of at least 512.')
  const category = categoryFor(input)
  const originalChars = output.length
  const originalBytes = encoder.encode(output).length
  const unchanged: CompressedToolOutput = { content: output, compressed: false, category, originalChars, originalBytes, omittedChars: 0, omittedDiagnosticLines: 0 }
  const fits = (text: string, chars = maxChars, bytes = maxBytes) => text.length <= chars && encoder.encode(text).length <= bytes
  const originalFits = fits(output)
  if (originalFits && (originalChars < STRUCTURAL_THRESHOLD || category === 'data' || category === 'search')) return unchanged

  const lines = sourceLines(output, category)
  const diagnostics = lines.filter((line) => line.diagnostic)
  const status = input.ok === undefined ? 'not provided' : input.ok ? 'tool reported ok=true' : 'tool reported ok=false'
  const exit = /^exit (-?\d{1,10})\b/.exec(input.meta ?? '')?.[1]
  const spill = /^vontra-output:\/\/[a-zA-Z0-9_-]{1,120}\.txt$/.test(input.spillRef ?? '') ? input.spillRef : null
  const lowerBound = (value: number, edge: 'start' | 'end', inclusive: boolean) => {
    let low = 0, high = diagnostics.length
    while (low < high) {
      const mid = Math.floor((low + high) / 2)
      if (diagnostics[mid][edge] < value || (inclusive && diagnostics[mid][edge] === value)) low = mid + 1
      else high = mid
    }
    return low
  }
  const countOmittedDiagnostics = (spans: Span[]) => diagnostics.length - spans.reduce((count, span) => count + Math.max(0, lowerBound(span.end, 'end', true) - lowerBound(span.start, 'start', false)), 0)
  const render = (spans: Span[]) => {
    const kept = spans.reduce((sum, span) => sum + span.end - span.start, 0)
    const omittedDiagnostics = countOmittedDiagnostics(spans)
    const parts = [
      `${TOOL_OUTPUT_EXCERPT_MARKER} original ${originalChars} UTF-16 chars / ${originalBytes} UTF-8 bytes; omitted ${originalChars - kept} chars; category=${category}; ${status}${exit === undefined ? '' : `; exit code ${exit}`}; omitted diagnostic lines=${omittedDiagnostics}. Exact original excerpts follow; omissions are not evidence of success.]`,
      spill && maxChars >= 1_024 && maxBytes >= 1_024 ? `Full output saved to ${spill}; use read_file with offset/limit to page it.` : 'Request narrower tool output for any omitted data.',
    ]
    let cursor = 0
    for (const span of spans) {
      if (span.start > cursor) parts.push(`[omitted original chars ${cursor}..${span.start}: ${span.start - cursor} chars]`)
      parts.push(output.slice(span.start, span.end))
      cursor = span.end
    }
    if (cursor < originalChars) parts.push(`[omitted original chars ${cursor}..${originalChars}: ${originalChars - cursor} chars]`)
    return parts.join('\n')
  }

  const structural = merge(lines.filter((line) => !line.noise), 80)
  const structuralText = render(structural)
  if (fits(structuralText) && structuralText.length < originalChars * 0.95) return {
    content: structuralText, compressed: true, category, originalChars, originalBytes,
    omittedChars: originalChars - structural.reduce((sum, span) => sum + span.end - span.start, 0),
    omittedDiagnosticLines: countOmittedDiagnostics(structural),
  }
  if (originalFits) return unchanged

  let selected: Span[] = []
  const add = (span: Span, chars: number, bytes: number, tail = false) => {
    const candidate = (length: number) => merge([...selected, aligned(output, tail ? { start: span.end - length, end: span.end } : { start: span.start, end: span.start + length })])
    const full = candidate(span.end - span.start)
    if (fits(render(full), chars, bytes)) { selected = full; return }
    let low = 0
    let high = span.end - span.start
    while (low < high) {
      const mid = Math.ceil((low + high) / 2)
      if (fits(render(candidate(mid)), chars, bytes)) low = mid
      else high = mid - 1
    }
    if (low >= 32) selected = candidate(low)
  }

  // Middle-of-log failures outrank the head. Reserve room for the tail's runner
  // summary, then the command/setup head. Excess diagnostics are counted rather
  // than silently implying that the retained failures are the complete set.
  let diagnosticCandidates = 0
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]
    if (!line.diagnostic || selected.some((span) => span.start <= line.start && span.end >= line.end)) continue
    if (diagnosticCandidates++ >= 96) break
    const match = diagnosticPattern.exec(output.slice(line.start, line.end))
    const start = line.end - line.start > 2_000 && match ? line.start + Math.max(0, match.index - 160) : lines[Math.max(0, index - 2)].start
    add({ start, end: Math.min(lines[Math.min(lines.length - 1, index + 12)].end, start + 5_000) }, Math.max(512, Math.floor(maxChars * 0.65)), Math.max(512, Math.floor(maxBytes * 0.65)))
  }
  add({ start: Math.max(0, originalChars - Math.floor(maxChars * 0.22)), end: originalChars }, maxChars, maxBytes, true)
  add({ start: 0, end: Math.min(originalChars, Math.floor(maxChars * 0.22)) }, maxChars, maxBytes)
  // Spend spare budget on exact source spans, including unknown text. The
  // structural path never discards unknown lines merely because of a classifier.
  for (const span of structural.slice(0, 96)) add(span, maxChars, maxBytes)
  const content = render(selected)
  return {
    content, compressed: true, category, originalChars, originalBytes,
    omittedChars: originalChars - selected.reduce((sum, span) => sum + span.end - span.start, 0),
    omittedDiagnosticLines: countOmittedDiagnostics(selected),
  }
}
