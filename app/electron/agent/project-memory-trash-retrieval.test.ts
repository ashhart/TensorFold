import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { buildRetrievalIndex, buildWorkspaceRetrievalIndex, filterRetrievalProjectMemorySources, formatRetrievalSearchResult, searchRetrievalIndex } from './retrieval-index'
import { executeTool } from './tools'
import type { ProjectMemoryEntry, Thread } from './types'

const entry = (id: string, projectId: string, text: string, extra: Partial<ProjectMemoryEntry> = {}): ProjectMemoryEntry => ({
  id, projectId, text, kind: 'preference', sourceType: 'manual', sourceRef: null,
  confidence: 'high', privacyScope: 'project', owner: null, lastVerifiedAt: null,
  createdAt: 1, updatedAt: 1, ...extra,
})
const alpha = entry('same-id', 'synthetic-alpha', 'SYNTHETIC_ACTIVE_ALPHA')
const beta = entry('same-id', 'synthetic-beta', 'SYNTHETIC_ACTIVE_BETA')
const fictionalSecret = 'SYNTHETIC_TRASH_TOKEN_EXAMPLE_INVALID'
const hidden = entry('hidden', alpha.projectId, fictionalSecret, { trashedAt: 10, revisions: [{ text: 'SYNTHETIC_OLD_TRASH_EXAMPLE_INVALID' } as any] })
const malformed = [0, -1, NaN, 'damaged', false, {}].map((trashedAt, i) => entry(`malformed-${i}`, alpha.projectId, fictionalSecret, { trashedAt } as any))
const poison = entry('unreadable-trash', alpha.projectId, '', { trashedAt: 0 })
Object.defineProperty(poison, 'text', { get: () => { throw new Error('Trash body was read') } })

const mixed = buildRetrievalIndex({ projectMemory: [alpha, beta, hidden, poison, ...malformed] })
assert.equal(mixed.records.length, 2)
assert.equal(new Set(mixed.records.map((record) => record.id)).size, 2, 'same entry ids in different projects never merge')
assert.doesNotMatch(JSON.stringify(mixed), /SYNTHETIC_(?:TRASH|OLD_TRASH)/)
assert.match(JSON.stringify(mixed), /SYNTHETIC_ACTIVE_ALPHA/)
assert.match(JSON.stringify(mixed), /SYNTHETIC_ACTIVE_BETA/)
assert.ok(buildRetrievalIndex({ projectMemory: [entry('file-source', alpha.projectId, 'SYNTHETIC_FILE_CONTROL', { sourceType: 'file', sourceRef: 'notes.md' })] }).records[0].sourceIds.includes('notes.md'), 'project identity retains the original source citation')

const canonical = 'project-memory:synthetic-alpha:hidden'
const sources = buildRetrievalIndex({
  projectMemory: [alpha, hidden],
  records: [{ id: 'artifact:cached-trash', kind: 'artifact', text: fictionalSecret, sourceIds: [canonical] }],
  threads: (['chat', 'task'] as const).map((kind) => ({ id: `synthetic-${kind}`, kind, title: kind, cwd: '/synthetic-workspace', providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: `SYNTHETIC_${kind.toUpperCase()}_CONTROL` }], timeline: [] } as Thread)),
  rooms: [{ id: 'synthetic-room', name: 'Synthetic room', memory: [{ id: 'room-fact', kind: 'finding', text: 'SYNTHETIC_ROOM_CONTROL' }], events: [] } as any],
  missions: [{ id: 'synthetic-mission', title: 'Synthetic mission', checkpoints: [{ id: 'synthetic-checkpoint', label: 'Control', summary: 'SYNTHETIC_MISSION_CONTROL' }] } as any],
})
const recall = formatRetrievalSearchResult(searchRetrievalIndex(sources, { query: 'SYNTHETIC', limit: 50 }))
for (const marker of ['CHAT', 'TASK', 'ROOM', 'MISSION']) assert.match(recall, new RegExp(`SYNTHETIC_${marker}_CONTROL`))
assert.doesNotMatch(recall, /SYNTHETIC_TRASH/)

const cache = buildRetrievalIndex({ projectMemory: [alpha, beta] })
const trashedAlpha = { ...alpha, trashedAt: 20 }
const whileTrashed = filterRetrievalProjectMemorySources(cache, [trashedAlpha, beta])
assert.equal(whileTrashed.stats.total, 1)
assert.match(JSON.stringify(whileTrashed), /SYNTHETIC_ACTIVE_BETA/)
assert.doesNotMatch(JSON.stringify(whileTrashed), /SYNTHETIC_ACTIVE_ALPHA/)
assert.equal(filterRetrievalProjectMemorySources(cache, [{ ...alpha, trashedAt: null }, beta]).stats.total, 2, 'restoration makes a live cache record eligible again')
const ambiguous = buildRetrievalIndex({ projectMemory: [trashedAlpha, beta], records: [{ id: 'artifact:legacy', kind: 'artifact', text: fictionalSecret, sourceIds: ['memory:same-id'] }] })
assert.doesNotMatch(JSON.stringify(ambiguous), /SYNTHETIC_TRASH/, 'an id-only cached record cannot use another project to bypass Trash')
assert.equal(filterRetrievalProjectMemorySources(cache, [beta]).stats.total, 1, 'permanent removal invalidates old cache provenance')

const cwd = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-trash-retrieval-'))
try {
  await writeFile(path.join(cwd, 'fixture.ts'), 'export const syntheticControl = true\n')
  let live = [alpha]
  const pending = buildWorkspaceRetrievalIndex({ cwd, kinds: ['code', 'summary'], projectMemory: live, recallProjectMemory: () => live })
  live = [trashedAlpha]
  const settled = await pending
  assert.doesNotMatch(JSON.stringify(settled), /SYNTHETIC_ACTIVE_ALPHA/, 'Trash during asynchronous collection invalidates the initial snapshot')
  live = [{ ...alpha, trashedAt: null }]
  assert.match(JSON.stringify(await buildWorkspaceRetrievalIndex({ cwd, kinds: ['summary'], projectMemory: [], recallProjectMemory: () => live })), /SYNTHETIC_ACTIVE_ALPHA/)

  let getterCalls = 0
  const ctx = { cwd, projectMemory: [alpha], recallProjectMemory: () => { getterCalls++; return [alpha] } }
  for (const narrowing of [
    { currentThread: { memoryPolicy: { useExisting: false, contribute: false } } },
    { sensitiveRetrievalAllowed: false },
    { botVm: { executeTool: async () => ({ ok: true, output: 'SYNTHETIC_GUEST_CONTROL' }) } },
    { allowedToolNames: [] },
  ]) {
    const result = await executeTool('retrieval_search', { query: 'SYNTHETIC', kinds: ['summary'] }, { ...ctx, ...narrowing } as any)
    assert.doesNotMatch(result.output, /SYNTHETIC_ACTIVE_ALPHA/)
    assert.equal(getterCalls, 0, 'memory-off, cloud, VM and denied tools cannot invoke the live host reader')
  }
} finally {
  await rm(cwd, { recursive: true, force: true })
}
console.log('Project-memory Trash retrieval, provenance, same-id project isolation, live collection and policy narrowing passed')
