import { randomUUID } from 'node:crypto'
import type { Project, ProjectMemoryCandidate, ProjectMemoryEntry, SecurityRole } from '../../agent/types'
import {
  approveProjectMemoryCandidate, changeProjectMemoryTrash, createProjectMemoryCandidate,
  isActiveProjectMemoryEntry, projectMemoryCandidateDiff, rejectProjectMemoryCandidate,
  rollbackProjectMemoryEntry, scopeProjectMemoryCandidates, scopeProjectMemoryEntries,
  scopeProjectMemoryTrash, type ProjectMemoryLifecycleInput,
} from '../../agent/project-memory'
import { commitProjectMemoryMutation, removeProjectMemoryRecoveryCopies, type ProjectMemoryState } from '../../store/project-memory'
import type { handle as typedHandle } from './ipc'

interface ProjectMemoryIpcDeps {
  store: { projects: Project[]; projectMemory: ProjectMemoryEntry[]; projectMemoryCandidates: ProjectMemoryCandidate[] }
  handle: typeof typedHandle
  requireSecurityRole(role: SecurityRole, action: string): void
  auditEvent(kind: string, data?: Record<string, unknown>): Promise<void>
  auditPreview(value: string): string
}

const textFields = ['text', 'sourceRef', 'owner', 'reason', 'sourceThreadId'] as const
function validateInput(input: Partial<ProjectMemoryEntry & ProjectMemoryCandidate>): void {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid project memory input')
  for (const field of ['trashedAt', 'stateRevision', 'createdAt', 'updatedAt', 'revisions', 'reviewedAt', 'reviewedBy', 'durableEntryId', 'status']) if (Object.prototype.hasOwnProperty.call(input, field)) throw new Error(`${field} is host-owned`)
  for (const field of textFields) if (input[field] !== undefined && input[field] !== null && typeof input[field] !== 'string') throw new Error(`Invalid memory ${field}`)
  if (input.kind !== undefined && !['architecture', 'convention', 'api', 'decision', 'gotcha', 'summary', 'person', 'preference'].includes(input.kind)) throw new Error('Invalid memory kind')
  if (input.sourceType !== undefined && !['manual', 'file', 'agent', 'connector'].includes(input.sourceType)) throw new Error('Invalid memory source type')
  if (input.confidence !== undefined && !['low', 'medium', 'high'].includes(input.confidence)) throw new Error('Invalid memory confidence')
  if (input.privacyScope !== undefined && !['project', 'personal', 'team'].includes(input.privacyScope)) throw new Error('Invalid memory privacy scope')
  if (input.sourceIds !== undefined && (!Array.isArray(input.sourceIds) || input.sourceIds.some((id) => typeof id !== 'string'))) throw new Error('Invalid memory source IDs')
  if (input.sourceTurn !== undefined && input.sourceTurn !== null && (!Number.isSafeInteger(input.sourceTurn) || input.sourceTurn < 0)) throw new Error('Invalid memory source turn')
}

export function registerProjectMemoryIpc(deps: ProjectMemoryIpcDeps): void {
  const { store, handle, requireSecurityRole, auditEvent, auditPreview } = deps
  const state = (): ProjectMemoryState => ({ entries: store.projectMemory, candidates: store.projectMemoryCandidates })
  const project = (projectId: string): Project => {
    if (typeof projectId !== 'string' || !projectId.trim()) throw new Error('Choose an active project before changing memory')
    const current = store.projects.find((item) => item.id === projectId)
    if (!current || current.archivedAt || current.executionTarget === 'orchestrator') throw new Error('Choose an active local project before changing memory')
    return current
  }
  const entryFor = (id: string): ProjectMemoryEntry => {
    const entry = store.projectMemory.find((item) => item.id === id)
    if (!entry) throw new Error('Memory entry not found')
    project(entry.projectId)
    return entry
  }
  const candidateFor = (id: string): ProjectMemoryCandidate => {
    const candidate = store.projectMemoryCandidates.find((item) => item.id === id)
    if (!candidate) throw new Error('Memory candidate not found')
    project(candidate.projectId)
    return candidate
  }
  const mutate = <T>(projectId: string, action: string, stage: (value: ProjectMemoryState) => ProjectMemoryState & { result: T }, beforeCommit?: () => Promise<void>): Promise<T> => {
    requireSecurityRole('operator', action)
    const owner = project(projectId)
    return commitProjectMemoryMutation(state, stage, () => {
      requireSecurityRole('operator', action)
      if (project(projectId) !== owner) throw new Error('Project changed before memory save completed')
    }, (next) => { store.projectMemory = next.entries; store.projectMemoryCandidates = next.candidates }, beforeCommit)
  }
  const targetVisible = (candidate: ProjectMemoryCandidate): boolean => {
    const targetId = candidate.targetEntryId || candidate.durableEntryId
    return !targetId || store.projectMemory.some((entry) => entry.id === targetId && entry.projectId === candidate.projectId && isActiveProjectMemoryEntry(entry))
  }
  const audit = async (kind: string, projectId: string) => {
    // Do not put memory text, provenance, or stable entry IDs in the plaintext journal.
    await auditEvent(kind, { projectId: auditPreview(projectId) })
  }
  handle('memory:get', async (_event, projectId: string) => { requireSecurityRole('viewer', 'read project memory'); project(projectId); return scopeProjectMemoryEntries(store.projectMemory, projectId) })
  handle('memory:trashList', async (_event, projectId: string) => { requireSecurityRole('viewer', 'read project memory Trash'); project(projectId); return scopeProjectMemoryTrash(store.projectMemory, projectId) })
  handle('memory:candidates', async (_event, projectId: string) => { requireSecurityRole('viewer', 'read project memory candidates'); project(projectId); return scopeProjectMemoryCandidates(store.projectMemoryCandidates, projectId).filter(targetVisible) })
  handle('memory:candidateDiff', async (_event, id: string) => {
    requireSecurityRole('viewer', 'review project memory candidates')
    const candidate = candidateFor(id)
    if (!targetVisible(candidate)) throw new Error('Restore memory from Trash before reviewing changes')
    return projectMemoryCandidateDiff(store.projectMemory, candidate)
  })
  handle('memory:propose', async (_event, input: Partial<ProjectMemoryCandidate>) => {
    validateInput(input)
    if (!input.projectId) throw new Error('memory:propose requires a projectId')
    const result = await mutate(input.projectId, 'propose project memory', (value) => {
      if (input.targetEntryId) {
        const target = value.entries.find((entry) => entry.id === input.targetEntryId && entry.projectId === input.projectId)
        if (!target || !isActiveProjectMemoryEntry(target)) throw new Error('Memory candidate target is unavailable')
      }
      if (input.id && value.candidates.some((item) => item.id === input.id)) throw new Error('Memory candidate identity already exists')
      const candidate = createProjectMemoryCandidate({ ...input, projectId: input.projectId!, text: input.text || '' }, Date.now(), input.id || `mc_${randomUUID().slice(0, 8)}`)
      return { ...value, candidates: [candidate, ...value.candidates], result: candidate }
    })
    return result
  })
  handle('memory:approveCandidate', async (_event, id: string) => {
    requireSecurityRole('operator', 'approve project memory candidates')
    const candidate = candidateFor(id)
    const entry = await mutate(candidate.projectId, 'approve project memory candidates', (value) => {
      const result = approveProjectMemoryCandidate(value.entries, value.candidates, id, Date.now(), 'user', `m_${randomUUID().slice(0, 8)}`)
      return { ...result, result: result.entry }
    })
    await audit('memory.candidate_approved', entry.projectId)
    return entry
  })
  handle('memory:rejectCandidate', async (_event, id: string) => {
    requireSecurityRole('operator', 'reject project memory candidates')
    const candidate = candidateFor(id)
    const result = await mutate(candidate.projectId, 'reject project memory candidates', (value) => {
      const next = rejectProjectMemoryCandidate(value.candidates, id)
      return { ...value, candidates: next.candidates, result: next.candidate }
    })
    await audit('memory.candidate_rejected', candidate.projectId)
    return result
  })
  handle('memory:rollbackEntry', async (_event, id: string) => {
    requireSecurityRole('operator', 'roll back project memory')
    const entry = entryFor(id)
    const result = await mutate(entry.projectId, 'roll back project memory', (value) => {
      const next = rollbackProjectMemoryEntry(value.entries, value.candidates, id)
      return { ...next, result: { action: next.action, entry: next.entry } }
    })
    await audit('memory.entry_rolled_back', entry.projectId)
    return result
  })
  handle('memory:save', async (_event, input: Partial<ProjectMemoryEntry>) => {
    validateInput(input)
    const existing = input.id ? entryFor(input.id) : null
    if (input.projectId && existing && input.projectId !== existing.projectId) throw new Error('Memory cannot move to a different project')
    const projectId = existing?.projectId ?? input.projectId
    if (!projectId) throw new Error('memory:save requires a projectId')
    return mutate(projectId, 'save project memory', (value) => {
      const current = input.id ? value.entries.find((entry) => entry.id === input.id) : null
      if (input.id && !current) throw new Error('Memory entry not found')
      if (current && !isActiveProjectMemoryEntry(current)) throw new Error('Restore memory from Trash before changing it')
      const hasRef = Object.prototype.hasOwnProperty.call(input, 'sourceRef')
      const candidate = createProjectMemoryCandidate({
        projectId, targetEntryId: current?.id ?? null,
        kind: input.kind ?? current?.kind, text: input.text ?? current?.text ?? '', sourceType: input.sourceType ?? current?.sourceType,
        sourceRef: hasRef ? input.sourceRef ?? null : current?.sourceRef,
        sourceIds: Object.prototype.hasOwnProperty.call(input, 'sourceIds') ? input.sourceIds : hasRef ? undefined : current?.sourceIds,
        confidence: input.confidence ?? current?.confidence, privacyScope: input.privacyScope ?? current?.privacyScope,
        owner: Object.prototype.hasOwnProperty.call(input, 'owner') ? input.owner ?? null : current?.owner,
        sourceThreadId: input.sourceThreadId ?? current?.sourceThreadId, sourceTurn: input.sourceTurn ?? current?.sourceTurn,
        reason: 'Manual save reviewed in settings.',
      }, Date.now(), `mc_${randomUUID().slice(0, 8)}`)
      const result = approveProjectMemoryCandidate(value.entries, [candidate, ...value.candidates], candidate.id, Date.now(), 'user', `m_${randomUUID().slice(0, 8)}`)
      return { ...result, result: result.entry }
    })
  })
  for (const [channel, trash] of [['memory:trash', true], ['memory:restore', false]] as const) handle(channel, async (_event, input: ProjectMemoryLifecycleInput) => {
    if (!input || !Number.isSafeInteger(input.expectedRevision)) throw new Error('Memory action requires an expectedRevision')
    const entry = await mutate(input.projectId, trash ? 'move project memory to Trash' : 'restore project memory', (value) => {
      const next = changeProjectMemoryTrash(value.entries, input, trash)
      return { ...value, entries: next.entries, result: next.entry }
    })
    await audit(trash ? 'memory.entry_trashed' : 'memory.entry_restored', entry.projectId)
    return entry
  })
  handle('memory:delete', async (_event, id: string, guard?: { projectId: string; expectedRevision: number }) => {
    requireSecurityRole('operator', 'permanently forget project memory')
    const entry = entryFor(id)
    await mutate(entry.projectId, 'permanently forget project memory', (value) => {
      const current = value.entries.find((item) => item.id === id)
      if (!current) throw new Error('Memory entry not found')
      if (guard && (guard.projectId !== current.projectId || !Number.isSafeInteger(guard.expectedRevision) || guard.expectedRevision !== (current.stateRevision ?? 1))) throw new Error('Memory changed; refresh before permanently forgetting it')
      return { entries: value.entries.filter((item) => item.id !== id), candidates: value.candidates.filter((item) => item.targetEntryId !== id && item.durableEntryId !== id), result: undefined }
    }, removeProjectMemoryRecoveryCopies)
    await audit('memory.entry_forgotten', entry.projectId)
  })
}
