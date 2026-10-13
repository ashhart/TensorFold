// Pure core for importing an EXISTING external-harness session (a Codex rollout or a
// Claude Code project session) into a TensorFold thread. This module only PARSES: it turns the raw JSONL text of a
// session file into an `ImportableSession` descriptor (id, title, cwd, model, timestamps,
// message count, and the resume cursor to reattach). Reading the filesystem and creating
// the thread live in the host runtime; keeping the parse pure makes it testable with
// fixtures and safe to run over half-written files (a live session's JSONL can be read
// mid-append, so every line is parsed defensively and bad lines are skipped).

import type { ImportableSession, ImportSessionSource } from './types/cli'

export type { ImportableSession, ImportSessionSource }

// Rebuild the engine resume cursor from source + sessionId. The host uses this instead of
// trusting a renderer-supplied cursor object, so a crafted IPC payload can't smuggle an
// arbitrary cursor shape into a persisted thread.
export function resumeCursorFor(source: ImportSessionSource, sessionId: string): { threadId: string } | { sessionId: string } {
  return source === 'codex' ? { threadId: sessionId } : { sessionId }
}

// A single session file can be enormous (thousands of turns). We only need enough to build a
// preview + resume, so cap the scan. Counts past this are a lower bound (truncated=true).
const MAX_SCAN_LINES = 4000
const TITLE_MAX = 80

function isoToEpoch(value: unknown): number | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const ms = Date.parse(value)
  return Number.isFinite(ms) ? ms : null
}

function firstLine(text: string, max = TITLE_MAX): string {
  const line = text.replace(/\s+/g, ' ').trim()
  if (!line) return ''
  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

// Codex encodes message content as an array of typed parts ([{type:'input_text',text},…]);
// pull the human-readable text out, ignoring the developer/permissions preamble parts.
function codexPartsToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  const out: string[] = []
  for (const part of content) {
    if (typeof part === 'string') out.push(part)
    else if (part && typeof part === 'object' && typeof (part as any).text === 'string') out.push((part as any).text)
  }
  return out.join(' ')
}

function claudeContentToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  const out: string[] = []
  for (const part of content) {
    if (typeof part === 'string') out.push(part)
    else if (part && typeof part === 'object' && (part as any).type === 'text' && typeof (part as any).text === 'string') out.push((part as any).text)
  }
  return out.join(' ')
}

// Parse a Codex `rollout-*.jsonl`. Returns null when there is no session_meta (not a rollout,
// or the head is corrupt) — the caller drops it from the import list rather than importing junk.
export function parseCodexRollout(text: string): ImportableSession | null {
  const lines = text.split('\n')
  let sessionId = ''
  let cwd: string | null = null
  let model: string | null = null
  let createdAt: number | null = null
  let updatedAt: number | null = null
  let title = ''
  let messageCount = 0
  let scanned = 0
  let truncated = false

  for (const line of lines) {
    if (!line.trim()) continue
    if (scanned >= MAX_SCAN_LINES) { truncated = true; break }
    scanned++
    let obj: any
    try { obj = JSON.parse(line) } catch { continue }
    const type = obj?.type
    const payload = obj?.payload

    if (type === 'session_meta' && payload) {
      sessionId = String(payload.session_id ?? payload.id ?? sessionId)
      if (typeof payload.cwd === 'string') cwd = payload.cwd
      createdAt = isoToEpoch(payload.timestamp) ?? createdAt
      continue
    }
    // Model is first known from a turn_context line (matches token-usage.ts).
    if (type === 'turn_context' && payload?.model && !model) {
      model = String(payload.model)
      continue
    }
    const ts = isoToEpoch(obj?.timestamp)
    if (ts != null) updatedAt = updatedAt == null ? ts : Math.max(updatedAt, ts)

    if (type === 'event_msg' && payload?.type === 'user_message') {
      messageCount++
      if (!title) title = firstLine(typeof payload.message === 'string' ? payload.message : codexPartsToText(payload.message))
      continue
    }
    if (type === 'response_item' && payload?.type === 'message' && payload?.role === 'assistant') {
      messageCount++
    }
  }

  if (!sessionId) return null
  return {
    sessionId,
    source: 'codex',
    title: title || 'Imported Codex session',
    cwd,
    model,
    createdAt,
    updatedAt: updatedAt ?? createdAt,
    messageCount,
    resumeCursor: { threadId: sessionId },
    truncated,
  }
}

// Parse a Claude Code project session JSONL. Prefers an explicit `custom-title`/`ai-title`
// line for the title, else the first user message. Returns null when no sessionId is found.
export function parseClaudeSession(text: string): ImportableSession | null {
  const lines = text.split('\n')
  let sessionId = ''
  let cwd: string | null = null
  let model: string | null = null
  let createdAt: number | null = null
  let updatedAt: number | null = null
  let explicitTitle = ''
  let firstUserText = ''
  let messageCount = 0
  let scanned = 0
  let truncated = false

  for (const line of lines) {
    if (!line.trim()) continue
    if (scanned >= MAX_SCAN_LINES) { truncated = true; break }
    scanned++
    let obj: any
    try { obj = JSON.parse(line) } catch { continue }
    const type = obj?.type

    if (!sessionId && typeof obj?.sessionId === 'string') sessionId = obj.sessionId
    if (!cwd && typeof obj?.cwd === 'string' && obj.cwd) cwd = obj.cwd
    const ts = isoToEpoch(obj?.timestamp)
    if (ts != null) {
      createdAt = createdAt == null ? ts : Math.min(createdAt, ts)
      updatedAt = updatedAt == null ? ts : Math.max(updatedAt, ts)
    }

    if ((type === 'custom-title' || type === 'ai-title') && typeof obj?.content === 'string' && obj.content.trim()) {
      // A custom (user-set) title wins over an ai-generated one.
      if (type === 'custom-title' || !explicitTitle) explicitTitle = firstLine(obj.content)
      continue
    }
    if (type === 'user' && obj?.message && typeof obj.message === 'object') {
      messageCount++
      if (!firstUserText) firstUserText = firstLine(claudeContentToText(obj.message.content))
      continue
    }
    if (type === 'assistant' && obj?.message && typeof obj.message === 'object') {
      messageCount++
      if (!model && typeof obj.message.model === 'string') model = obj.message.model
    }
  }

  if (!sessionId) return null
  return {
    sessionId,
    source: 'claude',
    title: explicitTitle || firstUserText || 'Imported Claude session',
    cwd,
    model,
    createdAt,
    updatedAt: updatedAt ?? createdAt,
    messageCount,
    resumeCursor: { sessionId },
    truncated,
  }
}

export function parseSessionFile(source: ImportSessionSource, text: string): ImportableSession | null {
  return source === 'codex' ? parseCodexRollout(text) : parseClaudeSession(text)
}

// Merge + rank a batch of parsed sessions for an import picker: drop nulls, drop sessions with
// no real content (0 messages — an empty/aborted session isn't worth importing), de-dupe by
// sessionId keeping the most-recently-updated, and sort newest-first.
export function collectImportableSessions(parsed: Array<ImportableSession | null>): ImportableSession[] {
  const byId = new Map<string, ImportableSession>()
  for (const session of parsed) {
    if (!session || session.messageCount === 0) continue
    const existing = byId.get(session.sessionId)
    if (!existing || (session.updatedAt ?? 0) > (existing.updatedAt ?? 0)) byId.set(session.sessionId, session)
  }
  return [...byId.values()].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
}

// Filesystem collaborators the host injects (real fs) and tests fake (in-memory). Kept as an
// interface so the scan orchestration stays testable without touching ~/.codex or ~/.claude.
export interface SessionFileSystem {
  // List candidate session file paths for a source (host globs ~/.codex/sessions/**/rollout-*.jsonl
  // and ~/.claude/projects/**/*.jsonl).
  listFiles(source: ImportSessionSource): Promise<string[]>
  readFile(path: string): Promise<string>
}

export interface ScanImportableSessionsOptions {
  sources?: ImportSessionSource[]
  // Hard cap on files parsed per source so a machine with thousands of old sessions can't stall
  // the scan; the newest files should be listed first by the host.
  maxFilesPerSource?: number
}

// Orchestrate a full import scan: list → read → parse → merge/rank, across the requested
// sources. Read/parse failures on any one file are swallowed so one corrupt session never
// sinks the whole picker.
export async function scanImportableSessions(fs: SessionFileSystem, options: ScanImportableSessionsOptions = {}): Promise<ImportableSession[]> {
  const sources = options.sources ?? ['codex', 'claude']
  const cap = options.maxFilesPerSource ?? 200
  const parsed: Array<ImportableSession | null> = []
  for (const source of sources) {
    let files: string[] = []
    try { files = await fs.listFiles(source) } catch { files = [] }
    for (const file of files.slice(0, cap)) {
      let text = ''
      try { text = await fs.readFile(file) } catch { continue }
      if (!text) continue
      try { parsed.push(parseSessionFile(source, text)) } catch { /* skip a session that fails to parse */ }
    }
  }
  return collectImportableSessions(parsed)
}
