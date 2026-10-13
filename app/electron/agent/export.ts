import type {
  Thread,
  TimelineItem,
  RunExportBundle,
  ExportOptions,
  ExportMode,
  RunExportEvent,
  ExportRedactionKind,
  EvidenceBundle,
  RunChronicle,
  RunChronicleEntry,
} from './types'
import { redactText, aggregate } from './export-redaction'
import { fallbackTimelineTrace } from './trace'

// Pure: turn a Thread into a portable, redacted export bundle. No fs/electron deps
// so it stays unit-testable and safe to call from the main-process handler.
// Secret redaction is ON by default; share/eval modes additionally anonymize
// identifiers (thread id, cwd, provider, paths, IPs).

// Stable, non-reversible id hash for anonymized packets (FNV-ish; not for security,
// only to keep referential structure without leaking the real id).
function anonId(s: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) }
  return 'anon-' + (h >>> 0).toString(36)
}

const URL_RE = /\bhttps?:\/\/[^\s<>)"']+/gi
const SCREENSHOT_RE = /\.(?:png|jpe?g|webp|gif)$/i

function compact(input: string | undefined, limit = 220): string | undefined {
  const text = input?.replace(/\s+/g, ' ').trim()
  if (!text) return undefined
  return text.length > limit ? text.slice(0, limit - 1) + '...' : text
}

function firstLine(input: string | undefined): string | undefined {
  return compact(input?.split(/\r?\n/).find((line) => line.trim()) ?? input, 160)
}

function exportedPath(item: TimelineItem, scrub: (s: string | undefined) => string | undefined, anonymize: boolean, redact: boolean): string | undefined {
  if (!item.path) return undefined
  return anonymize || redact ? scrub(item.path) : item.path
}

function addTouchedFile(evidence: EvidenceBundle, path: string | undefined, itemId: string, at?: number): void {
  if (!path || /^https?:\/\//i.test(path)) return
  const found = evidence.touchedFiles.find((file) => file.path === path)
  if (found) {
    if (!found.itemIds.includes(itemId)) found.itemIds.push(itemId)
    if (!found.at || (at && at < found.at)) found.at = at
    return
  }
  evidence.touchedFiles.push({ path, itemIds: [itemId], at })
}

function addCitations(evidence: EvidenceBundle, item: TimelineItem, text: string | undefined, title: string | undefined): void {
  if (!text) return
  for (const match of text.matchAll(URL_RE)) {
    const url = match[0].replace(/[.,;:]+$/, '')
    if (evidence.citations.some((citation) => citation.itemId === item.id && citation.url === url)) continue
    evidence.citations.push({ itemId: item.id, url, title, at: item.createdAt })
  }
}

function addAssistantClaim(evidence: EvidenceBundle, item: TimelineItem, text: string | undefined, sourceItemIds: string[]): void {
  if (item.kind !== 'assistant' || !text?.trim()) return
  evidence.assistantClaims.push({
    itemId: item.id,
    claim: compact(text, 260) ?? text.trim(),
    sourceItemIds: sourceItemIds.slice(-12),
    supported: sourceItemIds.length > 0,
    at: item.createdAt,
  })
}

function evidenceLabel(item: TimelineItem): string {
  return item.title || item.tool || item.kind
}

function exportTrace(thread: Thread, item: TimelineItem, anonymize: boolean) {
  const trace = fallbackTimelineTrace(thread, item)
  return {
    ...trace,
    traceId: anonymize ? anonId(trace.traceId) : trace.traceId,
    parentTraceId: trace.parentTraceId ? (anonymize ? anonId(trace.parentTraceId) : trace.parentTraceId) : null,
    runId: trace.runId ? (anonymize ? anonId(trace.runId) : trace.runId) : null,
    threadId: anonymize ? anonId(trace.threadId || thread.id) : (trace.threadId ?? thread.id),
    itemId: trace.itemId ?? item.id,
    callId: trace.callId ? (anonymize ? anonId(trace.callId) : trace.callId) : null,
  }
}

function buildChronicleSummary(evidence: EvidenceBundle, toolCalls: number, diffs: number, errors: number): string {
  const parts = [
    `${evidence.traceIds.length} events`,
    `${toolCalls} tool calls`,
    `${evidence.touchedFiles.length} touched files`,
    `${diffs} diffs`,
    `${evidence.commandOutputs.length} command outputs`,
  ]
  if (evidence.approvals.length) parts.push(`${evidence.approvals.length} approvals`)
  if (evidence.assistantClaims.length) parts.push(`${evidence.assistantClaims.length} assistant claims`)
  if (evidence.gaps.length) parts.push(`${evidence.gaps.length} evidence gaps`)
  if (errors) parts.push(`${errors} errors`)
  return parts.join(' · ')
}

function buildClaimGaps(thread: Thread, evidence: EvidenceBundle, scrub: (s: string | undefined) => string | undefined): void {
  const hasAnyEvidence = evidence.touchedFiles.length > 0 || evidence.diffs.length > 0 || evidence.commandOutputs.length > 0 || evidence.citations.length > 0
  const hasGreenCommand = evidence.commandOutputs.some((command) => command.state === 'ok')
  const hasChangeEvidence = evidence.diffs.length > 0 || evidence.touchedFiles.length > 0

  for (const item of thread.timeline) {
    if (item.kind !== 'assistant' || !item.text) continue
    const text = item.text.replace(/\s+/g, ' ').trim()
    if (!text) continue
    const lower = text.toLowerCase()
    let reason: string | null = null
    if (/\b(tested|verified|passing|passed|green)\b/.test(lower) && !hasGreenCommand) {
      reason = 'Verification claim found, but no successful command output is present in the bundle.'
    } else if (/\b(fixed|implemented|patched|changed|updated)\b/.test(lower) && !hasChangeEvidence) {
      reason = 'Implementation claim found, but no touched file or diff evidence is present in the bundle.'
    } else if (/\b(done|completed|resolved)\b/.test(lower) && !hasAnyEvidence) {
      reason = 'Completion claim found, but the bundle has no file, command, diff, or citation evidence.'
    }
    if (reason) evidence.gaps.push({ kind: 'claim_without_evidence', itemId: item.id, claim: scrub(compact(text, 260) ?? text) ?? '', reason, at: item.createdAt })
  }
}

function isSourceEvidenceItem(item: TimelineItem): boolean {
  if (item.kind === 'tool') return Boolean(item.output || item.path || item.command || item.diff?.length || item.approval || item.visualization)
  if (item.kind === 'notice' || item.kind === 'cue' || item.kind === 'error') return Boolean(item.text || item.output || item.path)
  return false
}

export function buildRunExport(thread: Thread, opts: ExportOptions = {}, now = Date.now()): RunExportBundle {
  const mode: ExportMode = opts.mode ?? 'full'
  const redact = opts.redact ?? true
  const anonymize = opts.anonymize ?? (mode === 'share' || mode === 'eval')
  const includeReasoning = opts.includeReasoning ?? (mode !== 'eval')
  const hits: ExportRedactionKind[] = []

  const scrub = (s: string | undefined): string | undefined => {
    if (s == null || !redact) return s
    const r = redactText(s, anonymize)
    hits.push(...r.hits)
    return r.text
  }

  const messages = thread.apiMessages
    .filter((m) => typeof m.content === 'string')
    .map((m) => ({ role: String(m.role), content: scrub(String(m.content)) ?? '' }))

  let toolCalls = 0, diffs = 0, denied = 0, errors = 0
  const timeline: RunExportEvent[] = []
  const evidence: EvidenceBundle = { touchedFiles: [], diffs: [], commandOutputs: [], screenshots: [], citations: [], approvals: [], traceIds: [], assistantClaims: [], gaps: [] }
  const chronicleEntries: RunChronicleEntry[] = []
  let startedAt = thread.createdAt
  let finishedAt = thread.updatedAt
  const sourceItemIds: string[] = []
  for (const it of thread.timeline) {
    if (it.kind === 'reasoning' && !includeReasoning) continue
    if (it.kind === 'tool') toolCalls++
    if (it.diff?.length) diffs++
    if (it.state === 'denied') denied++
    if (it.kind === 'error') errors++
    if (it.createdAt) {
      startedAt = Math.min(startedAt || it.createdAt, it.createdAt)
      finishedAt = Math.max(finishedAt || it.createdAt, it.createdAt)
    }

    const title = scrub(it.title)
    const output = scrub(it.output)
    const command = scrub(it.command)
    const path = exportedPath(it, scrub, anonymize, redact)
    const text = mode === 'eval' && it.kind === 'assistant' ? undefined : scrub(it.text)
    const label = scrub(evidenceLabel(it)) ?? evidenceLabel(it)
    const detail = command ?? path ?? firstLine(output) ?? firstLine(text)

    const trace = exportTrace(thread, it, anonymize)
    evidence.traceIds.push({ ...trace, itemId: it.id, kind: it.kind, tool: it.tool, title, at: it.createdAt })
    chronicleEntries.push({ itemId: it.id, kind: it.kind, label, detail, state: it.state, trace, at: it.createdAt })
    addTouchedFile(evidence, path, it.id, it.createdAt)
    if (it.diff?.length) evidence.diffs.push({
      itemId: it.id,
      path,
      summary: scrub(it.meta) ?? title,
      lines: it.diff.map((d) => ({ ...d, text: scrub(d.text) ?? '' })),
      at: it.createdAt,
    })
    if (it.command) evidence.commandOutputs.push({ itemId: it.id, command: command ?? '', output, state: it.state, meta: scrub(it.meta), at: it.createdAt })
    if ((it.tool && /screenshot/i.test(it.tool)) || (it.title && /screenshot/i.test(it.title)) || (path && SCREENSHOT_RE.test(path))) {
      evidence.screenshots.push({ itemId: it.id, path, title, at: it.createdAt })
    }
    if (it.approval || it.state === 'awaiting' || it.state === 'denied') {
      evidence.approvals.push({ itemId: it.id, tool: it.tool, state: it.state, text: scrub(it.approval), at: it.createdAt })
    }
    addCitations(evidence, it, [text, output, path].filter(Boolean).join('\n'), title)
    addAssistantClaim(evidence, it, text, sourceItemIds)
    if (isSourceEvidenceItem(it) && !sourceItemIds.includes(it.id)) sourceItemIds.push(it.id)

    // In eval mode keep the skeleton (what was asked + done + outcome) but drop prose.
    if (mode === 'eval' && (it.kind === 'notice' || it.kind === 'cue')) continue
    timeline.push({
      id: it.id,
      kind: it.kind,
      text,
      tool: it.tool,
      title,
      command,
      path,
      output,
      diff: it.diff?.map((d) => ({ ...d, text: scrub(d.text) ?? '' })),
      state: it.state,
      trace,
      at: it.createdAt,
    })
  }
  buildClaimGaps(thread, evidence, scrub)

  const chronicle: RunChronicle = {
    summary: buildChronicleSummary(evidence, toolCalls, diffs, errors),
    startedAt,
    finishedAt,
    entries: chronicleEntries,
  }

  return {
    schema: 'vontra.run-export/1',
    mode,
    exportedAt: now,
    redacted: redact,
    anonymized: anonymize,
    thread: {
      id: anonymize ? anonId(thread.id) : thread.id,
      title: scrub(thread.title) ?? thread.title,
      kind: thread.kind,
      cwd: anonymize ? undefined : (redact ? scrub(thread.cwd) : thread.cwd),
      provider: anonymize ? undefined : thread.providerId,
      model: thread.model,
      createdAt: thread.createdAt,
      updatedAt: thread.updatedAt,
      status: thread.status,
      goal: scrub(thread.goal),
      goalStatus: thread.goalStatus,
    },
    messages,
    timeline,
    compactions: anonymize
      ? undefined
      : thread.compactions?.map((entry) => ({ ...entry, summaryExcerpt: scrub(entry.summaryExcerpt) ?? '' })),
    stats: { messages: messages.length, toolCalls, diffs, denied, errors },
    redactionReport: redact ? aggregate(hits) : [],
    evidence,
    chronicle,
  }
}
