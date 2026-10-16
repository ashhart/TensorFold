import assert from 'node:assert/strict'
import { assertTicketRelations } from './ticket-helpers'
import { assertActiveTicketWorkOwnership } from './ticket-worker-runtime'
import { assertArchiveEntityExists } from './archive'

const project = { id: 'p', archivedAt: null } as any
const otherProject = { id: 'other', archivedAt: null } as any
const room = { id: 'r', projectId: 'p', archivedAt: null } as any
const parent = { id: 'parent', projectId: 'p', parentId: null, archivedAt: null } as any
const child = { id: 'child', projectId: 'p', parentId: 'parent', archivedAt: null } as any
const thread = { id: 'thread', projectId: 'p', archivedAt: null } as any
const state: any = { projects: [project, otherProject], rooms: [room], tickets: [parent, child], threads: [thread] }

assert.doesNotThrow(() => assertTicketRelations(state, 'p', 'r', { selfId: 'child', parentId: 'parent', threadId: 'thread' }))
assert.throws(() => assertTicketRelations(state, 'other', null, { parentId: 'parent' }), /same project/)
assert.throws(() => assertTicketRelations(state, 'other', null, { threadId: 'thread' }), /same project/)
assert.throws(() => assertTicketRelations(state, 'p', null, { selfId: 'parent', parentId: 'child' }), /own parent or descendant/)
thread.archivedAt = 1
assert.throws(() => assertTicketRelations(state, 'p', null, { threadId: 'thread' }), /Restore the ticket thread/)
assert.throws(() => assertActiveTicketWorkOwnership({ ...child, threadId: 'thread' }, project, null, thread), /active ticket thread in the same project/)
thread.archivedAt = null
assert.throws(() => assertActiveTicketWorkOwnership({ ...child, threadId: 'missing' }, project, null, null), /active ticket thread in the same project/)

for (const kind of ['project', 'thread', 'room', 'ticket'] as const) {
  assert.throws(() => assertArchiveEntityExists(kind, 'missing', state), new RegExp(`${kind} not found`, 'i'))
}

console.log('ticket ownership graph and archive existence checks reject cross-project, cycles, archived, and missing relations')
