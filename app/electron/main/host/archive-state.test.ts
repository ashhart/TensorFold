import assert from 'node:assert/strict'
import type { Project, Room, Thread, Ticket } from '../../agent/types'
import {
  archiveProjectState,
  archiveRoomState,
  archiveThreadState,
  archiveTicketState,
  restoreProjectState,
  restoreRoomState,
  restoreThreadState,
  restoreTicketState,
} from './archive-state'

const project = { id: 'p', updatedAt: 1 } as Project
const otherProject = { id: 'other', updatedAt: 1 } as Project
const thread = { id: 't', projectId: 'p', updatedAt: 1 } as Thread
const room = { id: 'r', projectId: 'p', updatedAt: 1 } as Room
const ticket = { id: 'tk', projectId: 'p', roomId: 'r', parentId: 'parent', updatedAt: 1 } as Ticket
const otherTicket = { id: 'other-tk', projectId: 'other', updatedAt: 1 } as Ticket
const state = { projects: [project, otherProject], threads: [thread], rooms: [room], tickets: [ticket, otherTicket] }

const archivedProject = archiveProjectState(state, 'p', 10)
assert.ok(archivedProject)
assert.equal(state.projects.length, 2, 'soft archive retains the project record')
assert.equal(state.threads.length, 1, 'soft archive retains child threads and their journals remain addressable')
assert.equal(state.rooms.length, 1, 'soft archive retains child rooms')
assert.equal(state.tickets.length, 2, 'soft archive retains child tickets')
for (const item of [project, thread, room, ticket]) assert.equal(item.archivedAt, 10)
assert.equal(ticket.parentId, 'parent', 'soft archive preserves ticket relationships')
assert.equal(otherProject.archivedAt, undefined)
assert.equal(otherTicket.archivedAt, undefined)

const restoredProject = restoreProjectState(state, 'p', 20)
assert.ok(restoredProject)
for (const item of [project, thread, room, ticket]) assert.equal(item.archivedAt, null)

const roomDescendant = { id: 'room-descendant', projectId: 'p', roomId: null, parentId: 'tk', updatedAt: 1 } as Ticket
state.tickets.push(roomDescendant)
archiveRoomState(state, 'r', 30)
assert.equal(room.archivedAt, 30)
assert.equal(ticket.archivedAt, 30, 'room-scoped tickets leave active snapshots with their room')
assert.equal(roomDescendant.archivedAt, 30, 'room archive follows recursive ticket descendants even when they do not repeat roomId')
restoreRoomState(state, 'r', 40)
assert.equal(room.archivedAt, null)
assert.equal(ticket.archivedAt, null)
assert.equal(roomDescendant.archivedAt, null)

archiveTicketState(state, 'tk', 50)
assert.equal(ticket.archivedAt, 50)
assert.equal(roomDescendant.archivedAt, 50)
assert.equal(state.tickets.includes(ticket), true, 'ticket delete is an archive, not a splice')
restoreTicketState(state, 'tk', 60)
assert.equal(ticket.archivedAt, null)
assert.equal(roomDescendant.archivedAt, null)

assert.equal(archiveProjectState(state, 'missing'), null)
assert.equal(archiveRoomState(state, 'missing'), null)
assert.equal(archiveTicketState(state, 'missing'), null)

const preArchivedThread = { id: 'pre-thread', projectId: 'batch-project', archivedAt: 70, updatedAt: 1 } as Thread
const preArchivedRoom = { id: 'pre-room', projectId: 'batch-project', archivedAt: 71, updatedAt: 1 } as Room
const preArchivedTicket = { id: 'pre-ticket', projectId: 'batch-project', archivedAt: 72, updatedAt: 1 } as Ticket
const collisionTicket = { id: 'collision-ticket', projectId: 'batch-project', archivedAt: 80, updatedAt: 1 } as Ticket
const batchState = {
  projects: [{ id: 'batch-project', updatedAt: 1 } as Project],
  threads: [preArchivedThread, { id: 'batch-thread', projectId: 'batch-project', updatedAt: 1 } as Thread],
  rooms: [preArchivedRoom, { id: 'batch-room', projectId: 'batch-project', updatedAt: 1 } as Room],
  tickets: [preArchivedTicket, collisionTicket, { id: 'batch-ticket', projectId: 'batch-project', updatedAt: 1 } as Ticket],
}
const batch = archiveProjectState(batchState, 'batch-project', 80)!
assert.equal(batch.project.archivedAt, 81, 'parent batch marker avoids colliding with an independent child archive timestamp')
restoreProjectState(batchState, 'batch-project', 90)
assert.equal(preArchivedThread.archivedAt, 70, 'restoring a project preserves an independently archived thread')
assert.equal(preArchivedRoom.archivedAt, 71, 'restoring a project preserves an independently archived room')
assert.equal(preArchivedTicket.archivedAt, 72, 'restoring a project preserves an independently archived ticket')
assert.equal(collisionTicket.archivedAt, 80, 'timestamp collision cannot confuse an independent ticket with the parent batch')
assert.equal(batchState.threads.find((item) => item.id === 'batch-thread')?.archivedAt, null)
assert.equal(batchState.rooms.find((item) => item.id === 'batch-room')?.archivedAt, null)
assert.equal(batchState.tickets.find((item) => item.id === 'batch-ticket')?.archivedAt, null)

const epic = { id: 'epic', projectId: 'p', updatedAt: 1 } as Ticket
const child = { id: 'child', projectId: 'p', parentId: 'epic', updatedAt: 1 } as Ticket
const independentlyArchivedChild = { id: 'independent-child', projectId: 'p', parentId: 'epic', archivedAt: 95, updatedAt: 1 } as Ticket
const grandchild = { id: 'grandchild', projectId: 'p', parentId: 'independent-child', updatedAt: 1 } as Ticket
const ticketTreeState = { tickets: [epic, child, independentlyArchivedChild, grandchild] }
const epicArchive = archiveTicketState(ticketTreeState, 'epic', 100)!
assert.deepEqual(epicArchive.tickets.map((item) => item.id), ['epic', 'child', 'independent-child', 'grandchild'])
assert.equal(epic.archivedAt, 100)
assert.equal(child.archivedAt, 100, 'archiving an epic recursively archives its active descendants')
assert.equal(independentlyArchivedChild.archivedAt, 95, 'an independently archived subtree root keeps its marker')
assert.equal(grandchild.archivedAt, 95, 'active descendants below an independent archive inherit that archive provenance')
restoreTicketState(ticketTreeState, 'epic', 110)
assert.equal(epic.archivedAt, null)
assert.equal(child.archivedAt, null)
assert.equal(independentlyArchivedChild.archivedAt, 95)
assert.equal(grandchild.archivedAt, 95, 'restoring an epic cannot resurrect a separately archived subtree')
restoreTicketState(ticketTreeState, 'independent-child', 120)
assert.equal(independentlyArchivedChild.archivedAt, null)
assert.equal(grandchild.archivedAt, null, 'restoring the independent subtree restores only its own batch')

const linkedThread = { id: 'linked-thread', projectId: 'p', updatedAt: 1 } as Thread
const linkedTicket = { id: 'linked-ticket', projectId: 'p', threadId: 'linked-thread', updatedAt: 1 } as Ticket
const linkedDescendant = { id: 'linked-descendant', projectId: 'p', parentId: 'linked-ticket', updatedAt: 1 } as Ticket
const threadState = { threads: [linkedThread], tickets: [linkedTicket, linkedDescendant] }
const threadArchive = archiveThreadState(threadState, 'linked-thread', 130)!
assert.equal(linkedThread.archivedAt, 130)
assert.deepEqual(threadArchive.tickets.map((item) => item.id), ['linked-ticket', 'linked-descendant'])
assert.equal(linkedTicket.archivedAt, 130, 'a linked active ticket follows its archived thread')
assert.equal(linkedDescendant.archivedAt, 130, 'the linked ticket descendant closure follows the same batch')
restoreThreadState(threadState, 'linked-thread', 140)
assert.equal(linkedThread.archivedAt, null)
assert.equal(linkedTicket.archivedAt, null)
assert.equal(linkedDescendant.archivedAt, null)

console.log('remote/local archive-state retention and round-trip tests passed')
