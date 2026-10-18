import { aggregate, redactText } from '../../electron/agent/export-redaction'
import { SECRET_MASK } from '../../electron/agent/security'
import type { ExportRedaction, ExportRedactionKind } from '../../electron/agent/types'

export const EVIDENCE_BUNDLE_SCHEMA = 'vontra.evidence-bundle/1'

export const EVIDENCE_BUNDLE_SCOPES = ['mission', 'ticket', 'pr_loop', 'incident', 'assistant_output'] as const

export type EvidenceBundleScope = typeof EVIDENCE_BUNDLE_SCOPES[number]

export type EvidenceBundleMetadataValue =
  | string
  | number
  | boolean
  | null
  | EvidenceBundleMetadataValue[]
  | { [key: string]: EvidenceBundleMetadataValue }

export type EvidenceBundleMetadata = Record<string, EvidenceBundleMetadataValue>

export interface EvidenceBundleRecord {
  schema: typeof EVIDENCE_BUNDLE_SCHEMA
  id: string
  scope: EvidenceBundleScope
  scopeId: string
  title: string
  summary: string | null
  sourceIds: string[]
  artifactIds: string[]
  metadata: EvidenceBundleMetadata
  createdAt: number
  updatedAt: number
}

export interface CreateEvidenceBundleInput {
  id?: string
  scope: EvidenceBundleScope
  scopeId: string
  title?: string | null
  summary?: string | null
  sourceIds?: unknown
  artifactIds?: unknown
  metadata?: unknown
  createdAt?: number
  updatedAt?: number
}

export interface AttachEvidenceBundleRefsInput {
  sourceIds?: unknown
  artifactIds?: unknown
}

export interface EvidenceBundleValidation {
  ok: boolean
  errors: string[]
}

export interface EvidenceBundleExport {
  record: EvidenceBundleRecord
  redacted: boolean
  anonymized: boolean
  redactionReport: ExportRedaction[]
}

export interface EvidenceBundleCountSummary {
  total: number
  byScope: Record<EvidenceBundleScope, number>
  sourceIds: number
  artifactIds: number
  empty: number
  invalid: number
}

const DEFAULT_TITLES: Record<EvidenceBundleScope, string> = {
  mission: 'Mission evidence',
  ticket: 'Room ticket evidence',
  pr_loop: 'Local PR loop evidence',
  incident: 'Incident evidence',
  assistant_output: 'Assistant output evidence',
}

const SENSITIVE_METADATA_KEY_RE = /(?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|bearer)/i
const MAX_METADATA_DEPTH = 8
const MAX_METADATA_ARRAY = 100

export function createEvidenceBundleRecord(input: CreateEvidenceBundleInput, now = Date.now()): EvidenceBundleRecord {
  const validation = validateEvidenceBundleInput(input)
  if (!validation.ok) throw new Error(`Invalid evidence bundle: ${validation.errors.join('; ')}`)
  return normalizeEvidenceBundleRecord(input, now)
}

export function normalizeEvidenceBundleRecord(input: unknown, now = Date.now()): EvidenceBundleRecord {
  const obj = asObject(input)
  const scope = evidenceBundleScope(obj.scope) ?? 'assistant_output'
  const scopeId = cleanString(obj.scopeId)
  const createdAt = finiteNumber(obj.createdAt) ?? now
  const updatedAt = finiteNumber(obj.updatedAt) ?? createdAt
  return {
    schema: EVIDENCE_BUNDLE_SCHEMA,
    id: cleanString(obj.id) || makeEvidenceBundleId(scope, scopeId, createdAt),
    scope,
    scopeId,
    title: cleanString(obj.title) || DEFAULT_TITLES[scope],
    summary: cleanString(obj.summary) || null,
    sourceIds: normalizeIdList(obj.sourceIds),
    artifactIds: normalizeIdList(obj.artifactIds),
    metadata: normalizeEvidenceBundleMetadata(obj.metadata),
    createdAt,
    updatedAt,
  }
}

export function attachEvidenceBundleRefs(record: EvidenceBundleRecord, refs: AttachEvidenceBundleRefsInput, now = Date.now()): EvidenceBundleRecord {
  const sourceIds = mergeIdLists(record.sourceIds, refs.sourceIds)
  const artifactIds = mergeIdLists(record.artifactIds, refs.artifactIds)
  const changed = sourceIds.length !== record.sourceIds.length || artifactIds.length !== record.artifactIds.length
  return {
    ...record,
    sourceIds,
    artifactIds,
    updatedAt: changed ? now : record.updatedAt,
  }
}

export function exportEvidenceBundleRecord(record: EvidenceBundleRecord, opts: { redact?: boolean; anonymize?: boolean } = {}): EvidenceBundleExport {
  const redact = opts.redact ?? true
  const anonymize = opts.anonymize ?? false
  const redacted = redact
    ? redactEvidenceBundleMetadata(record.metadata, { anonymize })
    : { metadata: record.metadata, redactionReport: [] }

  return {
    record: {
      ...record,
      metadata: redacted.metadata,
    },
    redacted: redact,
    anonymized: anonymize,
    redactionReport: redacted.redactionReport,
  }
}

export function redactEvidenceBundleMetadata(input: unknown, opts: { anonymize?: boolean } = {}): { metadata: EvidenceBundleMetadata; redactionReport: ExportRedaction[] } {
  const hits: ExportRedactionKind[] = []
  const metadata = redactMetadataValue(normalizeEvidenceBundleMetadata(input), hits, opts.anonymize ?? false) as EvidenceBundleMetadata
  return { metadata, redactionReport: aggregate(hits) }
}

export function validateEvidenceBundleInput(input: unknown): EvidenceBundleValidation {
  const errors: string[] = []
  const obj = input && typeof input === 'object' ? input as Record<string, unknown> : null
  if (!obj) {
    return { ok: false, errors: ['bundle must be an object'] }
  }
  if (!evidenceBundleScope(obj.scope)) errors.push('scope must be one of mission, ticket, pr_loop, incident, assistant_output')
  if (!cleanString(obj.scopeId)) errors.push('scopeId is required')
  if (obj.id != null && !cleanString(obj.id)) errors.push('id must be a non-empty string when provided')
  if (obj.createdAt != null && finiteNumber(obj.createdAt) == null) errors.push('createdAt must be a finite number when provided')
  if (obj.updatedAt != null && finiteNumber(obj.updatedAt) == null) errors.push('updatedAt must be a finite number when provided')
  return { ok: errors.length === 0, errors }
}

export function validateEvidenceBundleRecord(record: unknown): EvidenceBundleValidation {
  const errors: string[] = []
  const obj = record && typeof record === 'object' ? record as Record<string, unknown> : null
  if (!obj) {
    return { ok: false, errors: ['bundle must be an object'] }
  }
  if (obj.schema !== EVIDENCE_BUNDLE_SCHEMA) errors.push(`schema must be ${EVIDENCE_BUNDLE_SCHEMA}`)
  if (!cleanString(obj.id)) errors.push('id is required')
  if (!evidenceBundleScope(obj.scope)) errors.push('scope must be one of mission, ticket, pr_loop, incident, assistant_output')
  if (!cleanString(obj.scopeId)) errors.push('scopeId is required')
  if (!cleanString(obj.title)) errors.push('title is required')
  if (!Array.isArray(obj.sourceIds)) errors.push('sourceIds must be an array')
  if (!Array.isArray(obj.artifactIds)) errors.push('artifactIds must be an array')
  if (finiteNumber(obj.createdAt) == null) errors.push('createdAt must be a finite number')
  if (finiteNumber(obj.updatedAt) == null) errors.push('updatedAt must be a finite number')
  return { ok: errors.length === 0, errors }
}

export function summarizeEvidenceBundleCounts(input: unknown): EvidenceBundleCountSummary {
  const summary: EvidenceBundleCountSummary = {
    total: 0,
    byScope: emptyScopeCounts(),
    sourceIds: 0,
    artifactIds: 0,
    empty: 0,
    invalid: 0,
  }
  if (!Array.isArray(input)) return summary

  const sourceIds = new Set<string>()
  const artifactIds = new Set<string>()
  for (const item of input) {
    summary.total++
    const record = normalizeEvidenceBundleRecord(item, 0)
    const raw = asObject(item)
    const scope = evidenceBundleScope(raw.scope)
    if (scope) summary.byScope[scope]++
    if (!validateEvidenceBundleInput(item).ok) summary.invalid++
    if (!record.sourceIds.length && !record.artifactIds.length) summary.empty++
    for (const id of record.sourceIds) sourceIds.add(id)
    for (const id of record.artifactIds) artifactIds.add(id)
  }
  summary.sourceIds = sourceIds.size
  summary.artifactIds = artifactIds.size
  return summary
}

export function normalizeEvidenceBundleMetadata(input: unknown): EvidenceBundleMetadata {
  const value = normalizeMetadataValue(input, 0, new WeakSet<object>())
  return isPlainRecord(value) ? value : {}
}

function normalizeMetadataValue(input: unknown, depth: number, seen: WeakSet<object>): EvidenceBundleMetadataValue {
  if (depth > MAX_METADATA_DEPTH) return '[truncated]'
  if (input == null) return null
  if (typeof input === 'string') return input.replace(/\s+/g, ' ').trim()
  if (typeof input === 'boolean') return input
  if (typeof input === 'number') return Number.isFinite(input) ? input : null
  if (Array.isArray(input)) return input.slice(0, MAX_METADATA_ARRAY).map((item) => normalizeMetadataValue(item, depth + 1, seen))
  if (typeof input !== 'object') return null
  if (seen.has(input)) return '[circular]'
  seen.add(input)
  const out: EvidenceBundleMetadata = {}
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    const cleanKey = cleanString(key)
    if (!cleanKey || value === undefined || typeof value === 'function' || typeof value === 'symbol') continue
    out[cleanKey] = normalizeMetadataValue(value, depth + 1, seen)
  }
  seen.delete(input)
  return out
}

function redactMetadataValue(input: EvidenceBundleMetadataValue, hits: ExportRedactionKind[], anonymize: boolean, key = ''): EvidenceBundleMetadataValue {
  if (SENSITIVE_METADATA_KEY_RE.test(key) && input != null) {
    hits.push('assignment')
    return SECRET_MASK
  }
  if (typeof input === 'string') {
    const redacted = redactText(input, anonymize)
    hits.push(...redacted.hits)
    return redacted.text
  }
  if (Array.isArray(input)) return input.map((item) => redactMetadataValue(item, hits, anonymize, key))
  if (input && typeof input === 'object') {
    const out: EvidenceBundleMetadata = {}
    for (const [childKey, childValue] of Object.entries(input)) {
      out[childKey] = redactMetadataValue(childValue, hits, anonymize, childKey)
    }
    return out
  }
  return input
}

function normalizeIdList(input: unknown): string[] {
  const raw = Array.isArray(input)
    ? input
    : typeof input === 'string'
      ? input.split(/\n|,|;/)
      : []
  return [...new Set(raw.map(cleanString).filter(Boolean))].slice(0, 200)
}

function mergeIdLists(current: string[], next: unknown): string[] {
  return [...new Set([...normalizeIdList(current), ...normalizeIdList(next)])].slice(0, 200)
}

function evidenceBundleScope(input: unknown): EvidenceBundleScope | null {
  const text = cleanString(input)
  return EVIDENCE_BUNDLE_SCOPES.includes(text as EvidenceBundleScope) ? text as EvidenceBundleScope : null
}

function cleanString(input: unknown): string {
  return typeof input === 'string' ? input.replace(/\s+/g, ' ').trim() : ''
}

function finiteNumber(input: unknown): number | null {
  return typeof input === 'number' && Number.isFinite(input) ? input : null
}

function asObject(input: unknown): Record<string, unknown> {
  return input && typeof input === 'object' ? input as Record<string, unknown> : {}
}

function isPlainRecord(input: EvidenceBundleMetadataValue): input is EvidenceBundleMetadata {
  return !!input && typeof input === 'object' && !Array.isArray(input)
}

function makeEvidenceBundleId(scope: EvidenceBundleScope, scopeId: string, now: number): string {
  const slug = (scopeId || scope).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40)
  return `bundle-${scope}-${slug || 'record'}-${Math.max(0, Math.floor(now)).toString(36)}`
}

function emptyScopeCounts(): Record<EvidenceBundleScope, number> {
  return {
    mission: 0,
    ticket: 0,
    pr_loop: 0,
    incident: 0,
    assistant_output: 0,
  }
}
