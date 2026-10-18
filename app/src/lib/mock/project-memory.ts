import type { ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryEntry, ProjectMemoryKind, ProjectMemoryPrivacyScope } from '../types'
import type { TensorFoldStudioApi } from '../ipc'
import { mockState, uid } from './state'

type MemoryRevision = NonNullable<ProjectMemoryEntry['revisions']>[number]
type MemoryApi = Pick<TensorFoldStudioApi, 'getProjectMemory' | 'getProjectMemoryTrash' | 'trashProjectMemory' | 'restoreProjectMemory' | 'getProjectMemoryCandidates' | 'proposeProjectMemory' | 'approveProjectMemoryCandidate' | 'memoryCandidateDiff' | 'rollbackMemoryEntry' | 'rejectProjectMemoryCandidate' | 'saveProjectMemory' | 'deleteProjectMemory'>
const active = (entry: ProjectMemoryEntry) => entry.trashedAt === undefined || entry.trashedAt === null
const has = (input: object, field: string) => Object.prototype.hasOwnProperty.call(input, field)
const kinds = ['architecture', 'convention', 'api', 'decision', 'gotcha', 'summary', 'person', 'preference']
const kind = (value: unknown): ProjectMemoryKind => kinds.includes(String(value)) ? value as ProjectMemoryKind : 'summary'
const privacy = (value: unknown): ProjectMemoryPrivacyScope => value === 'personal' || value === 'team' ? value : 'project'
function sourceIds(input: unknown, ref?: string | null): string[] | undefined {
  const raw = Array.isArray(input) ? input : typeof input === 'string' ? input.split(/[,;\n]/) : []
  const ids = raw.map((value) => String(value ?? '').trim()).filter(Boolean)
  if (ref?.trim() && !ids.includes(ref.trim())) ids.unshift(ref.trim())
  const unique = [...new Set(ids)]
  return unique.length ? unique : undefined
}
function revision(entry: ProjectMemoryEntry): number {
  const value = entry.stateRevision === undefined ? 1 : entry.stateRevision
  if (!Number.isSafeInteger(value) || value < 1) throw new Error('Invalid memory state revision')
  return value
}
function advance(entry: ProjectMemoryEntry): number {
  const value = revision(entry)
  if (value === Number.MAX_SAFE_INTEGER) throw new Error('Memory state revision limit reached')
  return value + 1
}
function normalize(entry: ProjectMemoryEntry): ProjectMemoryEntry {
  if (entry.trashedAt !== undefined && entry.trashedAt !== null && (typeof entry.trashedAt !== 'number' || !Number.isFinite(entry.trashedAt) || entry.trashedAt <= 0)) throw new Error('Invalid memory Trash state')
  return { ...entry, kind: kind(entry.kind), privacyScope: privacy(entry.privacyScope), sourceIds: sourceIds(entry.sourceIds, entry.sourceRef), trashedAt: entry.trashedAt ?? null, stateRevision: revision(entry) }
}
function project(id: string): void {
  if (typeof id !== 'string' || !id.trim() || !mockState.projects.some((item) => item.id === id && !item.archivedAt && item.executionTarget !== 'orchestrator')) throw new Error('Choose an active local project before changing memory')
}
function entryFor(id: string): ProjectMemoryEntry {
  const entry = mockState.mockProjectMemory.find((item) => item.id === id)
  if (!entry) throw new Error('Memory entry not found')
  project(entry.projectId)
  return normalize(entry)
}
function editable(entry: ProjectMemoryEntry): void {
  if (!active(entry)) throw new Error('Restore memory from Trash before changing it')
}
function candidateFor(id: string): ProjectMemoryCandidate {
  const item = mockState.mockProjectMemoryCandidates.find((candidate) => candidate.id === id)
  if (!item) throw new Error('Memory candidate not found')
  project(item.projectId)
  return item
}
function candidateTarget(candidate: ProjectMemoryCandidate): ProjectMemoryEntry | null {
  if (!candidate.targetEntryId) return null
  const target = entryFor(candidate.targetEntryId)
  if (target.projectId !== candidate.projectId) throw new Error('Memory candidate target is unavailable')
  editable(target)
  return target
}
function visible(candidate: ProjectMemoryCandidate): boolean {
  const id = candidate.targetEntryId || candidate.durableEntryId
  return !id || mockState.mockProjectMemory.some((entry) => entry.id === id && entry.projectId === candidate.projectId && active(entry))
}
function validate(input: Partial<ProjectMemoryEntry & ProjectMemoryCandidate>): void {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid project memory input')
  for (const key of ['trashedAt', 'stateRevision', 'revisions', 'createdAt', 'updatedAt', 'status', 'reviewedAt', 'reviewedBy', 'durableEntryId']) if (has(input, key)) throw new Error(`${key} is host-owned`)
  for (const field of ['text', 'sourceRef', 'owner', 'reason', 'sourceThreadId'] as const) if (input[field] !== undefined && input[field] !== null && typeof input[field] !== 'string') throw new Error(`Invalid memory ${field}`)
  if (input.kind !== undefined && !kinds.includes(input.kind)) throw new Error('Invalid memory kind')
  if (input.sourceType !== undefined && !['manual', 'file', 'agent', 'connector'].includes(input.sourceType)) throw new Error('Invalid memory source type')
  if (input.confidence !== undefined && !['low', 'medium', 'high'].includes(input.confidence)) throw new Error('Invalid memory confidence')
  if (input.privacyScope !== undefined && !['project', 'personal', 'team'].includes(input.privacyScope)) throw new Error('Invalid memory privacy scope')
  if (input.sourceIds !== undefined && (!Array.isArray(input.sourceIds) || input.sourceIds.some((id) => typeof id !== 'string'))) throw new Error('Invalid memory source IDs')
  if (input.sourceTurn !== undefined && input.sourceTurn !== null && (!Number.isSafeInteger(input.sourceTurn) || input.sourceTurn < 0)) throw new Error('Invalid memory source turn')
}
function createCandidate(input: Partial<ProjectMemoryCandidate>): ProjectMemoryCandidate {
  validate(input); project(input.projectId!)
  if (input.targetEntryId) candidateTarget({ ...input, projectId: input.projectId! } as ProjectMemoryCandidate)
  const id = input.id || uid()
  if (mockState.mockProjectMemoryCandidates.some((item) => item.id === id)) throw new Error('Memory candidate identity already exists')
  const now = Date.now(), text = String(input.text ?? '').replace(/\s+/g, ' ').trim(), ref = input.sourceRef?.trim() || null
  if (!text) throw new Error('Memory text is required')
  return { id, projectId: input.projectId!, kind: input.kind ?? 'summary', text, sourceType: input.sourceType ?? 'manual', sourceRef: ref, sourceIds: sourceIds(input.sourceIds, ref), sourceThreadId: input.sourceThreadId ?? null, sourceTurn: input.sourceTurn ?? null, confidence: input.confidence ?? 'medium', privacyScope: privacy(input.privacyScope), owner: input.owner ?? null, reason: input.reason?.trim() || null, status: 'pending', targetEntryId: input.targetEntryId ?? null, reviewedAt: null, reviewedBy: null, durableEntryId: null, createdAt: now, updatedAt: now }
}
function superseded(entry: ProjectMemoryEntry, now: number): MemoryRevision {
  return { text: entry.text, kind: entry.kind, sourceType: entry.sourceType, confidence: entry.confidence, privacyScope: privacy(entry.privacyScope), sourceRef: entry.sourceRef, ...(entry.sourceIds?.length ? { sourceIds: [...entry.sourceIds] } : {}), owner: entry.owner, lastVerifiedAt: entry.lastVerifiedAt, sourceThreadId: entry.sourceThreadId ?? null, sourceTurn: entry.sourceTurn ?? null, replacedAt: now, replacedBy: 'user' }
}
function approve(candidate: ProjectMemoryCandidate, candidates: ProjectMemoryCandidate[]): ProjectMemoryEntry {
  if (candidate.status !== 'pending') throw new Error('Memory candidate is no longer pending')
  const now = Date.now(), existing = candidateTarget(candidate)
  const entry: ProjectMemoryEntry = {
    ...(existing ?? {}), id: existing?.id ?? uid(), projectId: candidate.projectId,
    kind: candidate.kind, text: candidate.text, sourceType: candidate.sourceType, sourceRef: candidate.sourceRef, sourceIds: sourceIds(candidate.sourceIds, candidate.sourceRef),
    sourceThreadId: candidate.sourceThreadId ?? existing?.sourceThreadId ?? null, sourceTurn: candidate.sourceTurn ?? existing?.sourceTurn ?? null,
    confidence: candidate.confidence, privacyScope: privacy(candidate.privacyScope), owner: candidate.owner,
    lastVerifiedAt: candidate.sourceRef ? now : existing?.lastVerifiedAt ?? null, createdAt: existing?.createdAt ?? now, updatedAt: now,
    trashedAt: existing?.trashedAt ?? null, stateRevision: existing ? advance(existing) : 1,
    ...(existing ? { revisions: [superseded(existing, now), ...existing.revisions ?? []].slice(0, 5) } : {}),
  }
  const reviewed = { ...candidate, status: 'approved' as const, reviewedAt: now, reviewedBy: 'user', durableEntryId: entry.id, updatedAt: now }
  // Publish both collections after validation, including manual save's implicit proposal.
  mockState.mockProjectMemory = existing ? mockState.mockProjectMemory.map((item) => item.id === entry.id ? entry : item) : [entry, ...mockState.mockProjectMemory]
  mockState.mockProjectMemoryCandidates = candidates.map((item) => item.id === candidate.id ? reviewed : item)
  return structuredClone(entry)
}
function forget(id: string): void {
  mockState.mockProjectMemory = mockState.mockProjectMemory.filter((item) => item.id !== id)
  mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.filter((item) => item.targetEntryId !== id && item.durableEntryId !== id)
}
function changeTrash(input: { projectId: string; entryId: string; expectedRevision: number }, trash: boolean): ProjectMemoryEntry {
  if (!input || !Number.isSafeInteger(input.expectedRevision)) throw new Error('Memory action requires an expectedRevision')
  const entry = entryFor(input.entryId)
  if (entry.projectId !== input.projectId || input.expectedRevision !== revision(entry)) throw new Error('Memory changed; refresh before trying again')
  if (trash !== active(entry)) throw new Error(trash ? 'Memory is already in Trash' : 'Memory is not in Trash')
  const now = Date.now()
  if (!Number.isFinite(now) || now <= 0) throw new Error('Invalid memory Trash timestamp')
  const next = { ...entry, trashedAt: trash ? now : null, stateRevision: advance(entry) }
  mockState.mockProjectMemory = mockState.mockProjectMemory.map((item) => item.id === entry.id ? next : item)
  return next
}
const numberValue = (value: number | null | undefined): string | null => typeof value === 'number' && Number.isFinite(value) ? String(value) : null
const listValue = (value: string[] | undefined): string | null => value?.length ? JSON.stringify(value) : null
export function createMockProjectMemoryApi(): MemoryApi {
  return {
    getProjectMemory: async (id) => { project(id); return structuredClone(mockState.mockProjectMemory.filter((item) => item.projectId === id && active(item)).map(normalize)) },
    getProjectMemoryTrash: async (id) => { project(id); return structuredClone(mockState.mockProjectMemory.filter((item) => item.projectId === id && !active(item)).map(normalize)) },
    trashProjectMemory: async (input) => structuredClone(changeTrash(input, true)),
    restoreProjectMemory: async (input) => structuredClone(changeTrash(input, false)),
    getProjectMemoryCandidates: async (id) => { project(id); return structuredClone(mockState.mockProjectMemoryCandidates.filter((item) => item.projectId === id && item.status === 'pending' && visible(item)).map((item) => ({ ...item, kind: kind(item.kind), privacyScope: privacy(item.privacyScope), sourceIds: sourceIds(item.sourceIds, item.sourceRef) }))) },
    proposeProjectMemory: async (input) => { const candidate = createCandidate(input); mockState.mockProjectMemoryCandidates = [candidate, ...mockState.mockProjectMemoryCandidates]; return structuredClone(candidate) },
    approveProjectMemoryCandidate: async (id) => approve(candidateFor(id), mockState.mockProjectMemoryCandidates),
    memoryCandidateDiff: async (id) => {
      const item = candidateFor(id)
      if (!visible(item)) throw new Error('Restore memory from Trash before reviewing changes')
      const target = candidateTarget(item)
      const fields: ProjectMemoryCandidateDiff['changes'] = [
        { field: 'text', before: target?.text ?? null, after: item.text }, { field: 'kind', before: target?.kind ?? null, after: item.kind },
        { field: 'confidence', before: target?.confidence ?? null, after: item.confidence }, { field: 'privacyScope', before: target ? privacy(target.privacyScope) : null, after: privacy(item.privacyScope) },
        { field: 'sourceType', before: target?.sourceType ?? null, after: item.sourceType }, { field: 'sourceRef', before: target?.sourceRef ?? null, after: item.sourceRef },
        { field: 'sourceIds', before: listValue(target?.sourceIds), after: listValue(sourceIds(item.sourceIds, item.sourceRef)) }, { field: 'owner', before: target?.owner ?? null, after: item.owner },
        { field: 'lastVerifiedAt', before: numberValue(target?.lastVerifiedAt), after: item.sourceRef ? 'approval time' : numberValue(target?.lastVerifiedAt) },
        { field: 'sourceThreadId', before: target?.sourceThreadId ?? null, after: item.sourceThreadId ?? target?.sourceThreadId ?? null },
        { field: 'sourceTurn', before: numberValue(target?.sourceTurn), after: numberValue(item.sourceTurn ?? target?.sourceTurn ?? null) },
      ]
      return { candidateId: id, action: target ? 'update' : 'create', targetEntryId: target?.id ?? null, changes: fields.filter(({ before, after }) => target ? before !== after : after !== null) }
    },
    rollbackMemoryEntry: async (id) => {
      const entry = entryFor(id); editable(entry)
      const now = Date.now(), [previous, ...older] = entry.revisions ?? []
      const linked = mockState.mockProjectMemoryCandidates.filter((item) => item.durableEntryId === id && item.status === 'approved').sort((a, b) => (b.reviewedAt ?? 0) - (a.reviewedAt ?? 0))[0]
      const next = previous ? { ...entry, text: previous.text, kind: previous.kind, sourceType: previous.sourceType ?? entry.sourceType, confidence: previous.confidence, privacyScope: previous.privacyScope, sourceRef: previous.sourceRef, sourceIds: previous.sourceIds ? [...previous.sourceIds] : undefined, owner: previous.owner !== undefined ? previous.owner : entry.owner, sourceThreadId: previous.sourceThreadId ?? null, sourceTurn: previous.sourceTurn ?? null, revisions: older, lastVerifiedAt: previous.lastVerifiedAt !== undefined ? previous.lastVerifiedAt : null, updatedAt: now, stateRevision: advance(entry) } : null
      mockState.mockProjectMemory = next ? mockState.mockProjectMemory.map((item) => item.id === id ? next : item) : mockState.mockProjectMemory.filter((item) => item.id !== id)
      mockState.mockProjectMemoryCandidates = linked ? mockState.mockProjectMemoryCandidates.map((item) => item.id === linked.id ? { ...item, status: 'rolled_back', reviewedAt: now, updatedAt: now } : item) : mockState.mockProjectMemoryCandidates
      return { action: next ? 'restored_previous' : 'removed', entry: structuredClone(next) }
    },
    rejectProjectMemoryCandidate: async (id) => {
      const item = candidateFor(id)
      if (item.status !== 'pending') throw new Error('Memory candidate is no longer pending')
      const now = Date.now(), reviewed = { ...item, status: 'rejected' as const, reviewedAt: now, reviewedBy: 'user', updatedAt: now }
      mockState.mockProjectMemoryCandidates = mockState.mockProjectMemoryCandidates.map((candidate) => candidate.id === id ? reviewed : candidate)
      return structuredClone(reviewed)
    },
    saveProjectMemory: async (input) => {
      validate(input)
      const existing = input.id ? entryFor(input.id) : null
      if (existing) { editable(existing); if (input.projectId && input.projectId !== existing.projectId) throw new Error('Memory cannot move to a different project') }
      const candidate = createCandidate({ projectId: existing?.projectId ?? input.projectId, targetEntryId: existing?.id, kind: input.kind ?? existing?.kind, text: input.text ?? existing?.text, confidence: input.confidence ?? existing?.confidence, privacyScope: input.privacyScope ?? existing?.privacyScope, sourceType: input.sourceType ?? existing?.sourceType, sourceRef: has(input, 'sourceRef') ? input.sourceRef ?? null : existing?.sourceRef, sourceIds: has(input, 'sourceIds') ? input.sourceIds : has(input, 'sourceRef') ? undefined : existing?.sourceIds, owner: has(input, 'owner') ? input.owner ?? null : existing?.owner, sourceThreadId: input.sourceThreadId ?? existing?.sourceThreadId, sourceTurn: input.sourceTurn ?? existing?.sourceTurn, reason: 'Manual save reviewed in demo mode.' })
      return approve(candidate, [candidate, ...mockState.mockProjectMemoryCandidates])
    },
    deleteProjectMemory: async (id, guard) => { const entry = entryFor(id); if (guard && (guard.projectId !== entry.projectId || !Number.isSafeInteger(guard.expectedRevision) || guard.expectedRevision !== revision(entry))) throw new Error('Memory changed; refresh before permanently forgetting it'); forget(id) },
  }
}
