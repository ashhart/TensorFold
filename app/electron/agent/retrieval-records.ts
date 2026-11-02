import { createHash } from 'node:crypto'
import { RETRIEVAL_KINDS } from './retrieval-types'
import type { RetrievalIndexStats, RetrievalKind, RetrievalRecord, RetrievalSearchOptions } from './retrieval-types'

const MAX_RECORD_TEXT = 4_000

export function normalizeRetrievalRecord(record: Partial<RetrievalRecord> & { kind?: unknown }): RetrievalRecord | null {
  const kind = normalizeKind(record.kind)
  if (!kind) return null
  const title = clean(record.title) || clean(record.path) || kind
  const text = compact(clean(record.text), MAX_RECORD_TEXT)
  if (!text) return null
  const sourceIds = normalizeSourceIds(record.sourceIds, record.id ? [String(record.id)] : undefined)
  const id = clean(record.id) || `${kind}:${hashText(`${title}:${text}`)}`
  return {
    id,
    kind,
    title,
    text,
    sourceIds,
    path: record.path ?? null,
    createdAt: finite(record.createdAt) ?? null,
    metadata: normalizeMetadata(record.metadata),
  }
}

export function mergeRecords(a: RetrievalRecord, b: RetrievalRecord): RetrievalRecord {
  return {
    ...a,
    title: a.title || b.title,
    text: compact(`${a.text}\n${b.text}`, MAX_RECORD_TEXT),
    sourceIds: unique([...a.sourceIds, ...b.sourceIds]),
    path: a.path ?? b.path ?? null,
    createdAt: Math.max(a.createdAt ?? 0, b.createdAt ?? 0) || a.createdAt || b.createdAt || null,
    metadata: { ...(a.metadata ?? {}), ...(b.metadata ?? {}) },
  }
}

export function indexStats(records: RetrievalRecord[]): RetrievalIndexStats {
  const byKind = emptyKindCounts()
  const sourceIds = new Set<string>()
  for (const record of records) {
    byKind[record.kind]++
    for (const sourceId of record.sourceIds) sourceIds.add(sourceId)
  }
  return { total: records.length, byKind, sourceIds: sourceIds.size }
}

export function emptyKindCounts(): Record<RetrievalKind, number> {
  return { code: 0, doc: 0, chat: 0, email: 0, summary: 0, artifact: 0 }
}

export function normalizeKinds(value: RetrievalSearchOptions['kinds']): RetrievalKind[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,;\s]+/)
      : []
  const kinds = unique(raw.map((item) => normalizeKind(item)).filter((item): item is RetrievalKind => !!item))
  return kinds.length ? kinds : [...RETRIEVAL_KINDS]
}

export function normalizeKind(value: unknown): RetrievalKind | null {
  const text = clean(value).toLowerCase()
  if (text === 'docs' || text === 'document' || text === 'documents' || text === 'browser') return 'doc'
  if (text === 'messages' || text === 'message' || text === 'slack') return 'chat'
  if (text === 'emails' || text === 'mail') return 'email'
  if (text === 'summaries' || text === 'memory' || text === 'calendar') return 'summary'
  if (text === 'artifacts' || text === 'evidence') return 'artifact'
  return (RETRIEVAL_KINDS as readonly string[]).includes(text) ? text as RetrievalKind : null
}

export function normalizeSourceIds(value: unknown, fallback: string[] = []): string[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,;\n]/)
      : []
  const ids = raw.map((item) => clean(item)).filter(Boolean)
  for (const id of fallback.map((item) => clean(item)).filter(Boolean)) {
    if (!ids.includes(id)) ids.push(id)
  }
  return unique(ids)
}

export function compact(value: string, max: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`
}

export function clean(value: unknown): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

export function normalizeSlash(value: string): string {
  return value.replace(/\\/g, '/').replace(/^\.\/+/, '')
}

export function finite(value: unknown): number | null {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function normalizeMetadata(value: unknown): Record<string, string | number | boolean | null> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const out: Record<string, string | number | boolean | null> = {}
  for (const [key, raw] of Object.entries(value)) {
    if (raw == null || typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') out[key] = raw ?? null
  }
  return Object.keys(out).length ? out : undefined
}

export function hashText(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 12)
}

export function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}
