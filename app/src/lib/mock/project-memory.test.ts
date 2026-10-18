import assert from 'node:assert/strict'
import { createMockProjectMemoryApi } from './project-memory'
import { mockState } from './state'
import type { Project } from '../types'
mockState.projects = ['fixture-A', 'fixture-B'].map((id) => ({ id, name: id, archivedAt: null, executionTarget: 'local' }) as Project)
mockState.mockProjectMemory = []; mockState.mockProjectMemoryCandidates = []
const api = createMockProjectMemoryApi()
const entry = await api.saveProjectMemory({ projectId: 'fixture-A', kind: 'convention', text: 'Synthetic private fixture', confidence: 'high', sourceType: 'manual', sourceRef: 'fixture.txt' })
await api.saveProjectMemory({ id: entry.id, projectId: entry.projectId, text: 'Revised fixture' })
const before = (await api.getProjectMemory(entry.projectId))[0]
const trashed = await api.trashProjectMemory({ projectId: entry.projectId, entryId: entry.id, expectedRevision: 2 })
assert.equal(trashed.stateRevision, 3); assert.equal(trashed.updatedAt, before.updatedAt); assert.deepEqual(trashed.revisions, before.revisions)
assert.deepEqual(await api.getProjectMemory(entry.projectId), [])
assert.equal((await api.getProjectMemoryTrash(entry.projectId)).length, 1)
await assert.rejects(api.restoreProjectMemory({ projectId: 'fixture-B', entryId: entry.id, expectedRevision: 3 }), /Memory changed/)
await assert.rejects(api.restoreProjectMemory({ projectId: entry.projectId, entryId: entry.id, expectedRevision: 2 }), /Memory changed/)
await assert.rejects(api.saveProjectMemory({ id: entry.id, text: 'Resurrect' }), /Restore memory/)
await assert.rejects(api.rollbackMemoryEntry(entry.id), /Restore memory/)
const restored = await api.restoreProjectMemory({ projectId: entry.projectId, entryId: entry.id, expectedRevision: 3 })
assert.equal(restored.stateRevision, 4); assert.equal(restored.updatedAt, before.updatedAt)
await assert.rejects(api.saveProjectMemory({ id: entry.id, projectId: 'fixture-B', text: 'Move' }), /different project/)
await assert.rejects(api.saveProjectMemory({ id: entry.id, stateRevision: 100, text: 'Forge' }), /host-owned/)
await api.deleteProjectMemory(entry.id, { projectId: entry.projectId, expectedRevision: 4 })
assert.equal(mockState.mockProjectMemoryCandidates.length, 0); assert.equal(mockState.mockProjectMemory.length, 0)
console.log('browser demo memory parity: scoped CAS, timestamp/provenance retention, Trash gates and linked record Forget')

const reset = () => { mockState.mockProjectMemory = []; mockState.mockProjectMemoryCandidates = [] }
const state = () => structuredClone({ entries: mockState.mockProjectMemory, candidates: mockState.mockProjectMemoryCandidates })
const provenance = (value: import('../types').ProjectMemoryEntry) => ({ sourceType: value.sourceType, sourceRef: value.sourceRef, sourceIds: value.sourceIds, owner: value.owner, lastVerifiedAt: value.lastVerifiedAt, sourceThreadId: value.sourceThreadId, sourceTurn: value.sourceTurn })
reset()
const legacy = { ...before, id: 'legacy-entry', kind: 'legacy-kind', privacyScope: 'legacy-scope', sourceIds: undefined, trashedAt: undefined, stateRevision: undefined } as unknown as import('../types').ProjectMemoryEntry
mockState.mockProjectMemory = [legacy]
const normalized = (await api.getProjectMemory('fixture-A'))[0]
assert.equal(normalized.kind, 'summary'); assert.equal(normalized.privacyScope, 'project'); assert.equal(normalized.trashedAt, null); assert.equal(normalized.stateRevision, 1)
assert.deepEqual(normalized.sourceIds, ['fixture.txt'])
normalized.text = 'Caller must not mutate the stored fixture'
assert.equal(mockState.mockProjectMemory[0].text, legacy.text)
for (const malformed of [0, -1, NaN, Infinity, '0']) {
  mockState.mockProjectMemory = [{ ...before, trashedAt: malformed } as unknown as import('../types').ProjectMemoryEntry]
  const original = state()
  assert.deepEqual(await api.getProjectMemory('fixture-A'), [], 'Malformed Trash is never active')
  await assert.rejects(api.getProjectMemoryTrash('fixture-A'), /Invalid memory Trash state/)
  await assert.rejects(api.restoreProjectMemory({ projectId: 'fixture-A', entryId: before.id, expectedRevision: before.stateRevision! }), /Invalid memory Trash state/)
  assert.deepEqual(state(), original, 'Invalid lifecycle state cannot mutate either collection')
}
for (const malformed of [null, 0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
  mockState.mockProjectMemory = [{ ...before, stateRevision: malformed } as unknown as import('../types').ProjectMemoryEntry]
  const original = state()
  await assert.rejects(api.getProjectMemory('fixture-A'), /Invalid memory state revision/)
  await assert.rejects(api.trashProjectMemory({ projectId: 'fixture-A', entryId: before.id, expectedRevision: 2 }), /Invalid memory state revision/)
  assert.deepEqual(state(), original)
}
mockState.mockProjectMemory = [{ ...before, stateRevision: Number.MAX_SAFE_INTEGER }]
const overflowOriginal = state()
await assert.rejects(api.trashProjectMemory({ projectId: 'fixture-A', entryId: before.id, expectedRevision: Number.MAX_SAFE_INTEGER }), /revision limit/)
await assert.rejects(api.saveProjectMemory({ id: before.id, text: 'Overflow must remain unapplied' }), /revision limit/)
await assert.rejects(api.rollbackMemoryEntry(before.id), /revision limit/)
assert.deepEqual(state(), overflowOriginal, 'Failed manual save leaves no implicit candidate or partial revision')
const overflowProposal = await api.proposeProjectMemory({ projectId: 'fixture-A', targetEntryId: before.id, text: 'Explicit review survives a failed approval' })
const beforeApproval = state()
await assert.rejects(api.approveProjectMemoryCandidate(overflowProposal.id), /revision limit/)
assert.deepEqual(state(), beforeApproval, 'Failed approval preserves its pending proposal')
console.log('browser demo memory invalid state, legacy normalization, CAS overflow and atomic failed-save gates')

reset()
const first = await api.saveProjectMemory({ projectId: 'fixture-A', text: 'Original fictional decision', sourceType: 'connector', sourceRef: 'connector:synthetic-event', sourceIds: ['connector:synthetic-event', 'synthetic-related'], sourceThreadId: 'synthetic-thread-a', sourceTurn: 4, owner: 'fictional-owner-a', confidence: 'high', privacyScope: 'personal' })
assert.deepEqual(first.sourceIds, ['connector:synthetic-event', 'synthetic-related'])
assert.equal(first.sourceThreadId, 'synthetic-thread-a'); assert.equal(first.sourceTurn, 4)
const second = await api.saveProjectMemory({ id: first.id, text: 'Second fictional decision' })
assert.deepEqual(provenance(second), { ...provenance(first), lastVerifiedAt: second.lastVerifiedAt })
const proposal = await api.proposeProjectMemory({ projectId: 'fixture-A', targetEntryId: first.id, text: 'Third fictional decision', sourceType: 'file', sourceRef: 'synthetic-guide.md', sourceIds: ['synthetic-guide.md', 'synthetic-guide-related'], sourceThreadId: 'synthetic-thread-b', sourceTurn: 9, owner: 'fictional-owner-b', confidence: 'low', privacyScope: 'team' })
const diff = await api.memoryCandidateDiff(proposal.id)
for (const field of ['text', 'sourceType', 'sourceRef', 'sourceIds', 'owner', 'sourceThreadId', 'sourceTurn', 'lastVerifiedAt']) assert.ok(diff.changes.some((change) => change.field === field), `Review exposes ${field}`)
const third = await api.approveProjectMemoryCandidate(proposal.id)
assert.deepEqual(provenance(third), { sourceType: 'file', sourceRef: 'synthetic-guide.md', sourceIds: ['synthetic-guide.md', 'synthetic-guide-related'], owner: 'fictional-owner-b', lastVerifiedAt: third.lastVerifiedAt, sourceThreadId: 'synthetic-thread-b', sourceTurn: 9 })
const [oldSecond, oldFirst] = third.revisions!
assert.deepEqual({ sourceType: oldSecond.sourceType, sourceRef: oldSecond.sourceRef, sourceIds: oldSecond.sourceIds, owner: oldSecond.owner, lastVerifiedAt: oldSecond.lastVerifiedAt, sourceThreadId: oldSecond.sourceThreadId, sourceTurn: oldSecond.sourceTurn }, provenance(second))
assert.deepEqual({ sourceType: oldFirst.sourceType, sourceRef: oldFirst.sourceRef, sourceIds: oldFirst.sourceIds, owner: oldFirst.owner, lastVerifiedAt: oldFirst.lastVerifiedAt, sourceThreadId: oldFirst.sourceThreadId, sourceTurn: oldFirst.sourceTurn }, provenance(first))
const thirdTrashed = await api.trashProjectMemory({ projectId: first.projectId, entryId: first.id, expectedRevision: third.stateRevision! })
assert.deepEqual(provenance(thirdTrashed), provenance(third)); assert.deepEqual(thirdTrashed.revisions, third.revisions)
assert.deepEqual(await api.getProjectMemoryCandidates('fixture-A'), [])
await assert.rejects(api.memoryCandidateDiff(proposal.id), /Restore memory/)
const thirdRestored = await api.restoreProjectMemory({ projectId: first.projectId, entryId: first.id, expectedRevision: thirdTrashed.stateRevision! })
assert.deepEqual(provenance(thirdRestored), provenance(third)); assert.deepEqual(thirdRestored.revisions, third.revisions)
const rolledSecond = await api.rollbackMemoryEntry(first.id)
assert.equal(rolledSecond.action, 'restored_previous'); assert.deepEqual(provenance(rolledSecond.entry!), provenance(second))
assert.equal(mockState.mockProjectMemoryCandidates.find((candidate) => candidate.id === proposal.id)?.status, 'rolled_back')
assert.equal(mockState.mockProjectMemoryCandidates.filter((candidate) => candidate.status === 'approved').length, 2, 'Only the newest linked approval rolls back')
const rolledFirst = await api.rollbackMemoryEntry(first.id)
assert.deepEqual(provenance(rolledFirst.entry!), provenance(first))
assert.equal(mockState.mockProjectMemoryCandidates.filter((candidate) => candidate.status === 'approved').length, 1)
const removed = await api.rollbackMemoryEntry(first.id)
assert.deepEqual(removed, { action: 'removed', entry: null })
assert.equal(mockState.mockProjectMemoryCandidates.length, 3, 'Creation rollback keeps its reviewed history, unlike permanent Forget')
assert.ok(mockState.mockProjectMemoryCandidates.every((candidate) => candidate.status === 'rolled_back'))
console.log('browser demo memory full source provenance, complete review diffs, exact revision rollback and newest-only approval history')

reset()
mockState.mockProjectMemory = [{ ...first, stateRevision: 1, revisions: [{ text: 'Legacy fictional revision', kind: 'summary', confidence: 'medium', privacyScope: 'project', sourceRef: 'synthetic-legacy.md', replacedAt: 1, replacedBy: 'fictional-owner' }] }]
const legacyRollback = (await api.rollbackMemoryEntry(first.id)).entry!
assert.equal(legacyRollback.sourceType, first.sourceType); assert.equal(legacyRollback.owner, first.owner)
assert.equal(legacyRollback.lastVerifiedAt, null); assert.equal(legacyRollback.sourceThreadId, null); assert.equal(legacyRollback.sourceTurn, null)
assert.equal(legacyRollback.sourceIds, undefined)
const pending = await api.proposeProjectMemory({ id: 'synthetic-pending', projectId: 'fixture-A', targetEntryId: first.id, text: 'A pending edit may still be rejected while its target is in Trash' })
await assert.rejects(api.proposeProjectMemory({ id: pending.id, projectId: 'fixture-A', text: 'Duplicate review identity' }), /already exists/)
await api.trashProjectMemory({ projectId: first.projectId, entryId: first.id, expectedRevision: legacyRollback.stateRevision! })
assert.deepEqual(await api.getProjectMemoryCandidates(first.projectId), [])
await assert.rejects(api.approveProjectMemoryCandidate(pending.id), /Restore memory/)
assert.equal((await api.rejectProjectMemoryCandidate(pending.id)).status, 'rejected')
console.log('browser demo legacy revision fallbacks, hidden pending targets, duplicate identity guard and rejection in Trash')
