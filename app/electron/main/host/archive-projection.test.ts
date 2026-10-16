import assert from 'node:assert/strict'
import { archiveEntity, archiveSnapshot, restoreEntity } from './archive'
import { store } from '../state'

const saved = { projects: store.projects, threads: store.threads, rooms: store.rooms, tickets: store.tickets }
try {
  store.projects = [{ id: 'parent', archivedAt: 100, updatedAt: 1 } as any]
  store.threads = [{ id: 'child-thread', projectId: 'parent', archivedAt: 50, updatedAt: 1 } as any]
  store.rooms = [{ id: 'child-room', projectId: 'parent', archivedAt: 100, updatedAt: 1 } as any]
  store.tickets = [{ id: 'child-ticket', projectId: 'parent', roomId: 'child-room', archivedAt: 51, updatedAt: 1 } as any]

  let snapshot = archiveSnapshot()
  assert.deepEqual(snapshot.projects.map((item) => item.id), ['parent'])
  assert.deepEqual(snapshot.threads, [], 'independently archived child stays hidden while parent is archived')
  assert.deepEqual(snapshot.rooms, [])
  assert.deepEqual(snapshot.tickets, [])
  await assert.rejects(() => restoreEntity('thread', 'child-thread'), /parent project/, 'child restore rejects while parent batch is archived')
  await assert.rejects(() => restoreEntity('ticket', 'child-ticket'), /active parent project/, 'ticket restore cannot create an active orphan')

  store.projects[0].archivedAt = null
  store.rooms[0].archivedAt = null
  store.tickets.push({ id: 'parent-ticket', projectId: 'parent', archivedAt: 75, updatedAt: 1 } as any)
  ;(store.tickets.find((ticket) => ticket.id === 'child-ticket') as any).parentId = 'parent-ticket'
  await assert.rejects(() => restoreEntity('ticket', 'child-ticket'), /active parent ticket/, 'subticket restore cannot create an active orphan under an archived ticket')

  store.projects[0].archivedAt = null
  store.rooms[0].archivedAt = null
  snapshot = archiveSnapshot()
  assert.deepEqual(snapshot.threads.map((item) => item.id), ['child-thread'], 'independent child archive reappears after parent restore')
  assert.deepEqual(snapshot.tickets.map((item) => item.id), ['parent-ticket'], 'archived subtickets stay nested beneath their archived ticket parent')
  ;(store.tickets.find((ticket) => ticket.id === 'parent-ticket') as any).archivedAt = null
  snapshot = archiveSnapshot()
  assert.deepEqual(snapshot.tickets.map((item) => item.id), ['child-ticket'], 'an independently archived subticket reappears after its parent is restored')

  ;(store.tickets.find((ticket) => ticket.id === 'child-ticket') as any).threadId = 'child-thread'
  snapshot = archiveSnapshot()
  assert.deepEqual(snapshot.tickets, [], 'an archived ticket stays nested beneath its archived linked thread')
  await assert.rejects(() => restoreEntity('ticket', 'child-ticket'), /active linked thread/, 'ticket restore cannot reactivate a link to an archived thread')
  await restoreEntity('thread', 'child-thread')
  assert.equal(store.tickets[0].archivedAt, 51, 'restoring a thread preserves a ticket archived in an independent batch')
  snapshot = archiveSnapshot()
  assert.deepEqual(snapshot.tickets.map((item) => item.id), ['child-ticket'], 'the ticket archive reappears when its linked thread is restored')

  store.tickets.push({ id: 'atomic-ticket', projectId: 'parent', archivedAt: null, updatedAt: 1 } as any)
  await assert.rejects(
    () => archiveEntity('ticket', 'atomic-ticket', { stopLoops: async () => { throw new Error('loop did not settle') } }),
    /loop did not settle/,
    'archive fails when an owned runtime cannot be quiesced',
  )
  assert.equal(store.tickets.find((ticket) => ticket.id === 'atomic-ticket')?.archivedAt, null, 'runtime-quiesce failure leaves the archive state untouched')

  store.rooms.push({ id: 'integrity-room', projectId: 'parent', archivedAt: 200, updatedAt: 1 } as any)
  store.tickets.push(
    { id: 'outside-parent', projectId: 'parent', archivedAt: 199, updatedAt: 1 } as any,
    { id: 'room-child', projectId: 'parent', roomId: 'integrity-room', parentId: 'outside-parent', archivedAt: 200, updatedAt: 1 } as any,
  )
  await assert.rejects(() => restoreEntity('room', 'integrity-room'), /active parent ticket/, 'room batch restore cannot reactivate a ticket beneath a separately archived parent')
  assert.equal(store.rooms.find((room) => room.id === 'integrity-room')?.archivedAt, 200, 'failed relationship validation leaves the room batch archived')
} finally {
  store.projects = saved.projects
  store.threads = saved.threads
  store.rooms = saved.rooms
  store.tickets = saved.tickets
}

console.log('archive projection hides parent-owned children and rejects orphaning restores')
