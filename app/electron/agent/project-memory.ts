import type { ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryConfidence, ProjectMemoryEntry, ProjectMemoryKind, ProjectMemoryPrivacyScope, ProjectMemoryRevision, ProjectMemorySourceType } from './types'

// Superseded states kept per entry for rollback (VST-009). Small on purpose: memory is a
// prompt-injected surface, not an archive; older states age out oldest-first.
const MAX_MEMORY_REVISIONS = 5

export type ProjectMemoryWarningCode =
  | 'missing_source_ref'
  | 'source_missing'
  | 'source_newer_than_verification'
  | 'memory_changed_after_verification'

export interface ProjectMemorySourceSnapshot {
  sourceRef: string
  exists: boolean
  updatedAt: number | null
}

export interface ProjectMemoryWarning {
  entryId: string
  code: ProjectMemoryWarningCode
  sourceRef: string | null
  message: string
}

export interface ProjectMemoryCandidateInput {
  id?: string
  targetEntryId?: string | null
  projectId: string
  kind?: ProjectMemoryKind
  text: string
  sourceType?: ProjectMemorySourceType
  sourceRef?: string | null
  sourceIds?: string[]
  confidence?: ProjectMemoryConfidence
  privacyScope?: ProjectMemoryPrivacyScope
  owner?: string | null
  reason?: string | null
  // Provenance to the conversation/turn that noticed the pattern (VST-009).
  sourceThreadId?: string | null
  sourceTurn?: number | null
}

function sourceBacked(entry: ProjectMemoryEntry): boolean {
  return entry.sourceType === 'file' || entry.sourceType === 'connector'
}

function sourceLabel(ref: string | null): string {
  return ref?.trim() || 'source'
}

export function scopeProjectMemoryEntries(entries: ProjectMemoryEntry[], projectId: string): ProjectMemoryEntry[] {
  return entries.filter((entry) => entry.projectId === projectId && isActiveProjectMemoryEntry(entry)).map(normalizeProjectMemoryEntry)
}

export function forgetProjectMemoryEntry(entries: ProjectMemoryEntry[], id: string): ProjectMemoryEntry[] {
  return entries.filter((entry) => entry.id !== id)
}

export function isActiveProjectMemoryEntry(entry: ProjectMemoryEntry): boolean {
  return entry.trashedAt === undefined || entry.trashedAt === null
}

export function scopeProjectMemoryTrash(entries: ProjectMemoryEntry[], projectId: string): ProjectMemoryEntry[] {
  return entries.filter((entry) => entry.projectId === projectId && !isActiveProjectMemoryEntry(entry)).map(normalizeProjectMemoryEntry)
}

export function projectMemoryStateRevision(entry: ProjectMemoryEntry): number {
  const revision = entry.stateRevision === undefined ? 1 : entry.stateRevision
  if (!Number.isSafeInteger(revision) || revision < 1) throw new Error('Invalid memory state revision')
  return revision
}

export function advanceProjectMemoryStateRevision(entry: ProjectMemoryEntry): number {
  const revision = projectMemoryStateRevision(entry)
  if (revision === Number.MAX_SAFE_INTEGER) throw new Error('Memory state revision limit reached')
  return revision + 1
}

export interface ProjectMemoryLifecycleInput {
  projectId: string
  entryId: string
  expectedRevision: number
}

export function changeProjectMemoryTrash(entries: ProjectMemoryEntry[], input: ProjectMemoryLifecycleInput, trash: boolean, now = Date.now()): { entries: ProjectMemoryEntry[]; entry: ProjectMemoryEntry } {
  if (!input || typeof input.projectId !== 'string' || !input.projectId.trim() || typeof input.entryId !== 'string' || !input.entryId.trim()) throw new Error('Memory action requires projectId and entryId')
  const current = entries.find((entry) => entry.id === input.entryId && entry.projectId === input.projectId)
  if (!current) throw new Error('Memory entry not found in this project')
  normalizeProjectMemoryEntry(current)
  if (!Number.isSafeInteger(input.expectedRevision) || input.expectedRevision !== projectMemoryStateRevision(current)) throw new Error('Memory changed; refresh before retrying')
  if (trash !== isActiveProjectMemoryEntry(current)) throw new Error(trash ? 'Memory is already in Trash' : 'Memory is not in Trash')
  if (!Number.isFinite(now) || now <= 0) throw new Error('Invalid memory Trash timestamp')
  const entry = { ...current, trashedAt: trash ? now : null, stateRevision: advanceProjectMemoryStateRevision(current) }
  return { entries: entries.map((item) => item === current ? entry : item), entry }
}

export function normalizeProjectMemoryEntry(entry: ProjectMemoryEntry): ProjectMemoryEntry {
  if (entry.trashedAt !== undefined && entry.trashedAt !== null && (typeof entry.trashedAt !== 'number' || !Number.isFinite(entry.trashedAt) || entry.trashedAt <= 0)) throw new Error('Invalid memory Trash state')
  return {
    ...entry,
    trashedAt: entry.trashedAt ?? null,
    stateRevision: projectMemoryStateRevision(entry),
    kind: normalizeMemoryKind(entry.kind),
    privacyScope: normalizeMemoryPrivacyScope(entry.privacyScope),
    sourceIds: normalizeSourceIds(entry.sourceIds, entry.sourceRef),
  }
}

export function scopeProjectMemoryCandidates(candidates: ProjectMemoryCandidate[], projectId: string): ProjectMemoryCandidate[] {
  return candidates
    .filter((candidate) => candidate.projectId === projectId && candidate.status === 'pending')
    .map((candidate) => ({
      ...candidate,
      kind: normalizeMemoryKind(candidate.kind),
      privacyScope: normalizeMemoryPrivacyScope(candidate.privacyScope),
      sourceIds: normalizeSourceIds(candidate.sourceIds, candidate.sourceRef),
    }))
}

export function createProjectMemoryCandidate(input: ProjectMemoryCandidateInput, now = Date.now(), id = `mc_${Math.random().toString(36).slice(2, 9)}`): ProjectMemoryCandidate {
  if (!input.projectId?.trim()) throw new Error('memory candidate requires projectId')
  const text = String(input.text ?? '').replace(/\s+/g, ' ').trim()
  if (!text) throw new Error('memory candidate requires text')
  return {
    id,
    projectId: input.projectId,
    kind: input.kind ?? 'summary',
    text,
    sourceType: input.sourceType ?? 'manual',
    sourceRef: input.sourceRef?.trim() || null,
    sourceIds: normalizeSourceIds(input.sourceIds, input.sourceRef),
    confidence: input.confidence ?? 'medium',
    privacyScope: normalizeMemoryPrivacyScope(input.privacyScope),
    owner: input.owner ?? null,
    reason: input.reason?.trim() || null,
    status: 'pending',
    targetEntryId: input.targetEntryId ?? null,
    reviewedAt: null,
    reviewedBy: null,
    durableEntryId: null,
    sourceThreadId: input.sourceThreadId ?? null,
    sourceTurn: typeof input.sourceTurn === 'number' && input.sourceTurn >= 0 ? input.sourceTurn : null,
    createdAt: now,
    updatedAt: now,
  }
}

/**
 * What approving `candidateId` would DO to durable memory (VST-009): a create, or an
 * update with per-field before/after. Review must be informed — the UI renders this diff
 * next to Approve so no durable behavior mutates blind.
 */
export function projectMemoryCandidateDiff(entries: ProjectMemoryEntry[], candidate: ProjectMemoryCandidate): ProjectMemoryCandidateDiff {
  const target = candidate.targetEntryId ? entries.find((entry) => entry.id === candidate.targetEntryId) ?? null : null
  if (candidate.targetEntryId && !target) throw new Error('memory candidate target entry not found')
  if (target && !isActiveProjectMemoryEntry(target)) throw new Error('Restore memory from Trash before changing it')
  if (target && target.projectId !== candidate.projectId) throw new Error('memory candidate target belongs to a different project')
  const nextSourceIds = normalizeSourceIds(candidate.sourceIds, candidate.sourceRef)
  const nextSourceThreadId = candidate.sourceThreadId ?? target?.sourceThreadId ?? null
  const nextSourceTurn = candidate.sourceTurn ?? target?.sourceTurn ?? null
  const fields: Array<{ field: ProjectMemoryCandidateDiff['changes'][number]['field']; before: string | null; after: string | null }> = [
    { field: 'text', before: target?.text ?? null, after: candidate.text },
    { field: 'kind', before: target?.kind ?? null, after: candidate.kind },
    { field: 'confidence', before: target?.confidence ?? null, after: candidate.confidence },
    { field: 'privacyScope', before: target ? normalizeMemoryPrivacyScope(target.privacyScope) : null, after: normalizeMemoryPrivacyScope(candidate.privacyScope) },
    { field: 'sourceType', before: target?.sourceType ?? null, after: candidate.sourceType },
    { field: 'sourceRef', before: target?.sourceRef ?? null, after: candidate.sourceRef },
    { field: 'sourceIds', before: stringListValue(target?.sourceIds), after: stringListValue(nextSourceIds) },
    { field: 'owner', before: target?.owner ?? null, after: candidate.owner },
    { field: 'lastVerifiedAt', before: numberValue(target?.lastVerifiedAt), after: candidate.sourceRef ? 'approval time' : numberValue(target?.lastVerifiedAt) },
    { field: 'sourceThreadId', before: target?.sourceThreadId ?? null, after: nextSourceThreadId },
    { field: 'sourceTurn', before: numberValue(target?.sourceTurn), after: numberValue(nextSourceTurn) },
  ]
  return {
    candidateId: candidate.id,
    action: target ? 'update' : 'create',
    targetEntryId: target?.id ?? null,
    // For an update, only genuinely-changing fields; for a create, every set field.
    changes: fields.filter(({ before, after }) => (target ? before !== after : after != null)),
  }
}

export function approveProjectMemoryCandidate(
  entries: ProjectMemoryEntry[],
  candidates: ProjectMemoryCandidate[],
  candidateId: string,
  now = Date.now(),
  reviewedBy = 'user',
  entryId = `m_${Math.random().toString(36).slice(2, 9)}`,
): { entries: ProjectMemoryEntry[]; candidates: ProjectMemoryCandidate[]; candidate: ProjectMemoryCandidate; entry: ProjectMemoryEntry } {
  const candidate = candidates.find((item) => item.id === candidateId)
  if (!candidate) throw new Error('memory candidate not found')
  if (candidate.status !== 'pending') throw new Error('memory candidate has already been reviewed')

  const existingIndex = candidate.targetEntryId ? entries.findIndex((entry) => entry.id === candidate.targetEntryId) : -1
  if (candidate.targetEntryId && existingIndex < 0) throw new Error('memory candidate target entry not found')
  if (existingIndex >= 0 && entries[existingIndex].projectId !== candidate.projectId) {
    throw new Error('memory candidate target belongs to a different project')
  }
  if (existingIndex >= 0 && !isActiveProjectMemoryEntry(entries[existingIndex])) throw new Error('Restore memory from Trash before changing it')
  if (existingIndex < 0 && entries.some((entry) => entry.id === entryId)) throw new Error('Memory identity already exists')
  const sourceIds = normalizeSourceIds(candidate.sourceIds, candidate.sourceRef)
  const entry: ProjectMemoryEntry = existingIndex >= 0
    ? {
      ...entries[existingIndex],
      stateRevision: advanceProjectMemoryStateRevision(entries[existingIndex]),
      kind: candidate.kind,
      text: candidate.text,
      sourceType: candidate.sourceType,
      sourceRef: candidate.sourceRef,
      sourceIds,
      confidence: candidate.confidence,
      privacyScope: normalizeMemoryPrivacyScope(candidate.privacyScope),
      owner: candidate.owner,
      lastVerifiedAt: candidate.sourceRef ? now : entries[existingIndex].lastVerifiedAt,
      // The superseded state becomes a revision (bounded, newest first) so this
      // overwrite is reversible — never a silent mutation of durable behavior (VST-009).
      revisions: [supersededRevision(entries[existingIndex], now, reviewedBy), ...(entries[existingIndex].revisions ?? [])].slice(0, MAX_MEMORY_REVISIONS),
      sourceThreadId: candidate.sourceThreadId ?? entries[existingIndex].sourceThreadId ?? null,
      sourceTurn: candidate.sourceTurn ?? entries[existingIndex].sourceTurn ?? null,
      updatedAt: now,
    }
    : {
      id: entryId,
      trashedAt: null,
      stateRevision: 1,
      projectId: candidate.projectId,
      kind: candidate.kind,
      text: candidate.text,
      sourceType: candidate.sourceType,
      sourceRef: candidate.sourceRef,
      sourceIds,
      confidence: candidate.confidence,
      privacyScope: normalizeMemoryPrivacyScope(candidate.privacyScope),
      owner: candidate.owner,
      lastVerifiedAt: candidate.sourceRef ? now : null,
      sourceThreadId: candidate.sourceThreadId ?? null,
      sourceTurn: candidate.sourceTurn ?? null,
      createdAt: now,
      updatedAt: now,
    }
  const nextEntries = existingIndex >= 0
    ? entries.map((item, index) => index === existingIndex ? entry : item)
    : [entry, ...entries]
  const reviewed = { ...candidate, status: 'approved' as const, reviewedAt: now, reviewedBy, durableEntryId: entry.id, updatedAt: now }
  return {
    entries: nextEntries,
    candidates: candidates.map((item) => item.id === candidate.id ? reviewed : item),
    candidate: reviewed,
    entry,
  }
}

function supersededRevision(entry: ProjectMemoryEntry, now: number, replacedBy: string): ProjectMemoryRevision {
  return {
    text: entry.text,
    kind: entry.kind,
    sourceType: entry.sourceType,
    confidence: entry.confidence,
    privacyScope: normalizeMemoryPrivacyScope(entry.privacyScope),
    sourceRef: entry.sourceRef,
    ...(entry.sourceIds?.length ? { sourceIds: [...entry.sourceIds] } : {}),
    owner: entry.owner,
    lastVerifiedAt: entry.lastVerifiedAt,
    sourceThreadId: entry.sourceThreadId ?? null,
    sourceTurn: entry.sourceTurn ?? null,
    replacedAt: now,
    replacedBy,
  }
}

export interface ProjectMemoryRollbackResult {
  entries: ProjectMemoryEntry[]
  candidates: ProjectMemoryCandidate[]
  // 'restored_previous' when a superseded revision came back; 'removed' when the entry had
  // no prior state (born from an approval/manual create) and rolling back deletes it.
  action: 'restored_previous' | 'removed'
  entry: ProjectMemoryEntry | null
}

/**
 * Roll back the newest change to a durable memory entry (VST-009). An entry with revisions
 * returns to its previous state; an entry with none (its creation WAS the change) is
 * removed. Either way, the newest approved candidate that produced the entry's current
 * state flips to 'rolled_back' so the audit trail shows the reversal — durable behavior
 * mutations are never one-way doors.
 */
export function rollbackProjectMemoryEntry(
  entries: ProjectMemoryEntry[],
  candidates: ProjectMemoryCandidate[],
  entryId: string,
  now = Date.now(),
): ProjectMemoryRollbackResult {
  const index = entries.findIndex((entry) => entry.id === entryId)
  if (index < 0) throw new Error('memory entry not found')
  const current = entries[index]
  if (!isActiveProjectMemoryEntry(current)) throw new Error('Restore memory from Trash before rolling it back')

  // The newest approved candidate pointing at this entry is the change being reverted.
  const linked = candidates
    .filter((candidate) => candidate.durableEntryId === entryId && candidate.status === 'approved')
    .sort((a, b) => (b.reviewedAt ?? 0) - (a.reviewedAt ?? 0))[0] ?? null
  const nextCandidates = linked
    ? candidates.map((candidate) => candidate.id === linked.id
      ? { ...candidate, status: 'rolled_back' as const, reviewedAt: now, updatedAt: now }
      : candidate)
    : candidates

  const [previous, ...olderRevisions] = current.revisions ?? []
  if (!previous) {
    return {
      entries: entries.filter((entry) => entry.id !== entryId),
      candidates: nextCandidates,
      action: 'removed',
      entry: null,
    }
  }
  const restored: ProjectMemoryEntry = {
    ...current,
    stateRevision: advanceProjectMemoryStateRevision(current),
    text: previous.text,
    kind: previous.kind,
    sourceType: previous.sourceType ?? current.sourceType,
    confidence: previous.confidence,
    privacyScope: previous.privacyScope,
    sourceRef: previous.sourceRef,
    sourceIds: previous.sourceIds ? [...previous.sourceIds] : undefined,
    owner: previous.owner !== undefined ? previous.owner : current.owner,
    sourceThreadId: previous.sourceThreadId ?? null,
    sourceTurn: previous.sourceTurn ?? null,
    revisions: olderRevisions,
    // Restore the verification timestamp belonging to this exact superseded state. Live
    // source drift checks still invalidate it when the source changed in the meantime.
    lastVerifiedAt: previous.lastVerifiedAt !== undefined ? previous.lastVerifiedAt : null,
    updatedAt: now,
  }
  return {
    entries: entries.map((entry, at) => at === index ? restored : entry),
    candidates: nextCandidates,
    action: 'restored_previous',
    entry: restored,
  }
}

export function rejectProjectMemoryCandidate(
  candidates: ProjectMemoryCandidate[],
  candidateId: string,
  now = Date.now(),
  reviewedBy = 'user',
): { candidates: ProjectMemoryCandidate[]; candidate: ProjectMemoryCandidate } {
  const candidate = candidates.find((item) => item.id === candidateId)
  if (!candidate) throw new Error('memory candidate not found')
  if (candidate.status !== 'pending') throw new Error('memory candidate has already been reviewed')
  const reviewed = { ...candidate, status: 'rejected' as const, reviewedAt: now, reviewedBy, updatedAt: now }
  return { candidates: candidates.map((item) => item.id === candidate.id ? reviewed : item), candidate: reviewed }
}

function normalizeMemoryKind(kind: unknown): ProjectMemoryKind {
  const value = String(kind || '')
  return ['architecture', 'convention', 'api', 'decision', 'gotcha', 'summary', 'person', 'preference'].includes(value)
    ? value as ProjectMemoryKind
    : 'summary'
}

function normalizeMemoryPrivacyScope(scope: unknown): ProjectMemoryPrivacyScope {
  const value = String(scope || '')
  return value === 'personal' || value === 'team' || value === 'project' ? value : 'project'
}

function normalizeSourceIds(sourceIds: unknown, fallbackRef?: string | null): string[] | undefined {
  const raw = Array.isArray(sourceIds)
    ? sourceIds
    : typeof sourceIds === 'string'
      ? sourceIds.split(/[,;\n]/)
      : []
  const normalized = raw.map((value) => String(value ?? '').trim()).filter(Boolean)
  const fallback = fallbackRef?.trim()
  if (fallback && !normalized.includes(fallback)) normalized.unshift(fallback)
  const unique = [...new Set(normalized)]
  return unique.length ? unique : undefined
}

function stringListValue(value: string[] | undefined): string | null {
  return value?.length ? JSON.stringify(value) : null
}

function numberValue(value: number | null | undefined): string | null {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : null
}

export function formatProjectMemoryEntryForPrompt(entry: ProjectMemoryEntry): string {
  if (!isActiveProjectMemoryEntry(entry)) return ''
  const normalized = normalizeProjectMemoryEntry(entry)
  const text = normalized.text.replace(/\s+/g, ' ').trim()
  const citations = normalized.sourceIds?.length
    ? normalized.sourceIds.join(', ')
    : normalized.sourceRef?.trim() || ''
  return `- [${normalized.kind}, ${normalized.confidence}, scope:${normalized.privacyScope ?? 'project'}] ${text}${citations ? ` (src: ${citations})` : ''}`
}

export function detectProjectMemoryWarnings(
  entries: ProjectMemoryEntry[],
  sources: Record<string, ProjectMemorySourceSnapshot | undefined> = {},
): ProjectMemoryWarning[] {
  const warnings: ProjectMemoryWarning[] = []
  for (const entry of entries) {
    if (!isActiveProjectMemoryEntry(entry)) continue
    const sourceRef = entry.sourceRef?.trim() || null
    if (sourceBacked(entry) && !sourceRef) {
      warnings.push({
        entryId: entry.id,
        code: 'missing_source_ref',
        sourceRef: null,
        message: 'Source-backed memory has no source reference, so live truth cannot be checked.',
      })
    }

    const snapshot = sourceRef ? sources[sourceRef] : undefined
    if (snapshot) {
      if (!snapshot.exists) {
        warnings.push({
          entryId: entry.id,
          code: 'source_missing',
          sourceRef,
          message: `Memory source ${sourceLabel(sourceRef)} is missing; prefer live files and ask for verification.`,
        })
      } else if (entry.lastVerifiedAt != null && snapshot.updatedAt != null && snapshot.updatedAt > entry.lastVerifiedAt) {
        warnings.push({
          entryId: entry.id,
          code: 'source_newer_than_verification',
          sourceRef,
          message: `Memory source ${sourceLabel(sourceRef)} changed after this memory was verified.`,
        })
      }
    }

    if (entry.lastVerifiedAt != null && entry.updatedAt > entry.lastVerifiedAt) {
      warnings.push({
        entryId: entry.id,
        code: 'memory_changed_after_verification',
        sourceRef,
        message: 'Memory text changed after its last verification; treat it as unverified until rechecked.',
      })
    }
  }
  return warnings
}

export function formatProjectMemoryWarnings(warnings: ProjectMemoryWarning[], max = 6): string {
  return warnings.slice(0, max).map((warning) => {
    const ref = warning.sourceRef ? ` src: ${warning.sourceRef};` : ''
    return `- [${warning.code};${ref} memory: ${warning.entryId}] ${warning.message}`
  }).join('\n')
}
