import { writeArchiveRecord, writeTensorFoldStudioProtectedJson } from '../../store'
import type { ArchiveKind, ArchiveSnapshot, Ticket } from '../../agent/types'
import { store } from '../state'
import { stripIncognitoThreads } from '../../agent/incognito'
import { persistProjects, persistRooms, persistThreads, persistTickets } from './persistence'
import { archiveProjectState, archiveRoomState, archiveThreadState, archiveTicketState, restoreProjectState, restoreRoomState, restoreThreadState, restoreTicketState, ticketArchiveClosure } from './archive-state'

export const isActive = <T extends { archivedAt?: number | null }>(item: T) => !item.archivedAt

export interface ArchiveRuntimeControls {
  abortThread?: (id: string) => void
  abortRoom?: (id: string) => void
  abortProjectTerminals?: (projectId: string) => void
  abortMissions?: (target: { projectId?: string | null; threadId?: string | null }) => Promise<unknown> | unknown
  stopLoops?: (kind: ArchiveKind, id: string) => Promise<unknown> | unknown
  publish?: (kind: 'project' | 'thread' | 'room' | 'ticket', data: unknown) => void
}

export function assertArchiveEntityExists(kind: ArchiveKind, id: string, state = store): void {
  const collection = kind === 'project' ? state.projects : kind === 'thread' ? state.threads : kind === 'room' ? state.rooms : state.tickets
  if (!collection.some((item) => item.id === id)) throw new Error(`${kind[0].toUpperCase()}${kind.slice(1)} not found.`)
}

function assertTicketBatchRestorable(tickets: Ticket[], restoring: {
  projectIds?: ReadonlySet<string>
  roomIds?: ReadonlySet<string>
  threadIds?: ReadonlySet<string>
} = {}): void {
  const restoringIds = new Set(tickets.map((ticket) => ticket.id))
  for (const ticket of tickets) {
    const project = store.projects.find((candidate) => candidate.id === ticket.projectId)
    if (!project || (project.archivedAt && !restoring.projectIds?.has(project.id))) throw new Error('Restore the active parent project before restoring linked tickets.')
    if (ticket.roomId) {
      const room = store.rooms.find((candidate) => candidate.id === ticket.roomId)
      if (!room || (room.archivedAt && !restoring.roomIds?.has(room.id)) || room.projectId !== ticket.projectId) throw new Error('Restore the active parent room in the same project before restoring linked tickets.')
    }
    if (ticket.threadId) {
      const thread = store.threads.find((candidate) => candidate.id === ticket.threadId)
      if (!thread || (thread.archivedAt && !restoring.threadIds?.has(thread.id)) || thread.projectId !== ticket.projectId) throw new Error('Restore the active linked thread in the same project before restoring this ticket.')
    }
    if (ticket.parentId) {
      const parent = store.tickets.find((candidate) => candidate.id === ticket.parentId)
      if (!parent || parent.projectId !== ticket.projectId || (parent.archivedAt && !restoringIds.has(parent.id))) {
        throw new Error('Restore the active parent ticket in the same project before restoring this subticket.')
      }
    }
  }
}

export const archiveSnapshot = (): ArchiveSnapshot => {
  const projectArchiveAt = new Map(store.projects.filter((project) => project.archivedAt).map((project) => [project.id, project.archivedAt!]))
  const roomArchiveAt = new Map(store.rooms.filter((room) => room.archivedAt).map((room) => [room.id, room.archivedAt!]))
  const archivedThreadIds = new Set(store.threads.filter((thread) => thread.archivedAt).map((thread) => thread.id))
  const ticketsById = new Map(store.tickets.map((ticket) => [ticket.id, ticket]))
  const hasArchivedTicketAncestor = (ticketId: string): boolean => {
    const seen = new Set<string>([ticketId])
    let parentId = ticketsById.get(ticketId)?.parentId ?? null
    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)
      const parent = ticketsById.get(parentId)
      if (!parent) return false
      if (parent.archivedAt) return true
      parentId = parent.parentId ?? null
    }
    return false
  }
  return {
    projects: store.projects.filter((project) => !!project.archivedAt),
    // Descendants stay hidden while any parent is archived. Independently archived children
    // reappear after parent restore; cascade children are restored by the parent operation.
    threads: store.threads.filter((thread) => !!thread.archivedAt && !projectArchiveAt.has(thread.projectId ?? '')),
    rooms: store.rooms.filter((room) => !!room.archivedAt && !projectArchiveAt.has(room.projectId)),
    tickets: store.tickets.filter((ticket) => {
      if (!ticket.archivedAt) return false
      if (projectArchiveAt.has(ticket.projectId)) return false
      if (ticket.roomId && roomArchiveAt.has(ticket.roomId)) return false
      if (ticket.threadId && archivedThreadIds.has(ticket.threadId)) return false
      if (hasArchivedTicketAncestor(ticket.id)) return false
      return true
    }),
  }
}

const ticketMayPersist = (ticket: Ticket) => !store.threads.some((thread) => thread.id === ticket.threadId && thread.incognito)
export function persistentArchiveSnapshot(): ArchiveSnapshot {
  const snapshot = archiveSnapshot()
  return { ...snapshot, threads: stripIncognitoThreads(snapshot.threads), tickets: snapshot.tickets.filter(ticketMayPersist) }
}

async function snapshotArchive(kind: ArchiveKind, id: string, relatedTickets: readonly Ticket[] = []): Promise<void> {
  const now = Date.now()
  if (kind === 'project') {
    const project = store.projects.find((p) => p.id === id)
    if (project) await writeArchiveRecord('project', id, {
      archivedAt: project.archivedAt ?? now,
      project,
      threads: stripIncognitoThreads(store.threads.filter((t) => t.projectId === id)),
      rooms: store.rooms.filter((r) => r.projectId === id),
      tickets: store.tickets.filter((t) => t.projectId === id && ticketMayPersist(t)),
    }).catch(() => {})
  } else if (kind === 'thread') {
    const thread = store.threads.find((t) => t.id === id)
    if (thread && !thread.incognito) await writeArchiveRecord('thread', id, { archivedAt: thread.archivedAt ?? now, thread, tickets: relatedTickets.filter(ticketMayPersist) }).catch(() => {})
  } else if (kind === 'room') {
    const room = store.rooms.find((r) => r.id === id)
    if (room) await writeArchiveRecord('room', id, {
      archivedAt: room.archivedAt ?? now,
      room,
      tickets: relatedTickets.filter(ticketMayPersist),
    }).catch(() => {})
  } else {
    const ticket = store.tickets.find((t) => t.id === id)
    if (ticket && ticketMayPersist(ticket)) await writeArchiveRecord('ticket', id, { archivedAt: ticket.archivedAt ?? now, ticket, tickets: relatedTickets.filter(ticketMayPersist) }).catch(() => {})
  }
}

export async function archiveEntity(kind: ArchiveKind, id: string, controls: ArchiveRuntimeControls = {}): Promise<ArchiveSnapshot> {
  assertArchiveEntityExists(kind, id)
  const now = Date.now()
  if (kind === 'project') {
    const threads = store.threads.filter((thread) => thread.projectId === id)
    const rooms = store.rooms.filter((room) => room.projectId === id)
    controls.abortProjectTerminals?.(id)
    for (const thread of threads) controls.abortThread?.(thread.id)
    for (const room of rooms) controls.abortRoom?.(room.id)
    await controls.abortMissions?.({ projectId: id })
    await controls.stopLoops?.(kind, id)
    const archived = archiveProjectState(store, id, now)
    if (archived) {
      await snapshotArchive('project', id)
      controls.publish?.('project', archived.project)
      for (const thread of archived.threads) controls.publish?.('thread', thread)
      for (const room of archived.rooms) controls.publish?.('room', room)
      for (const ticket of archived.tickets) controls.publish?.('ticket', ticket)
    }
  } else if (kind === 'thread') {
    const roots = store.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
    const tickets = ticketArchiveClosure(store.tickets, roots)
    controls.abortThread?.(id)
    await controls.abortMissions?.({ threadId: id })
    await controls.stopLoops?.(kind, id)
    for (const ticket of tickets) {
      await controls.stopLoops?.('ticket', ticket.id)
      const worker = ticket.threadId ? store.threads.find((candidate) => candidate.id === ticket.threadId) : null
      if (worker?.projectId === ticket.projectId) {
        controls.abortThread?.(worker.id)
        await controls.abortMissions?.({ threadId: worker.id })
      }
    }
    const archived = archiveThreadState(store, id, now)
    if (archived) {
      await snapshotArchive('thread', id, archived.tickets)
      controls.publish?.('thread', archived.thread)
      for (const ticket of archived.tickets) controls.publish?.('ticket', ticket)
    }
  } else if (kind === 'room') {
    const room = store.rooms.find((candidate) => candidate.id === id)!
    const ticketPool = store.tickets.filter((ticket) => ticket.projectId === room.projectId)
    const roomTicketRoots = ticketPool.filter((ticket) => ticket.roomId === id).map((ticket) => ticket.id)
    const tickets = ticketArchiveClosure(ticketPool, roomTicketRoots)
    controls.abortRoom?.(id)
    await controls.stopLoops?.(kind, id)
    for (const ticket of tickets) {
      await controls.stopLoops?.('ticket', ticket.id)
      const thread = ticket.threadId ? store.threads.find((candidate) => candidate.id === ticket.threadId) : null
      if (thread?.projectId === ticket.projectId) {
        controls.abortThread?.(thread.id)
        await controls.abortMissions?.({ threadId: thread.id })
      }
    }
    const archived = archiveRoomState(store, id, now)
    if (archived) {
      await snapshotArchive('room', id, archived.tickets)
      controls.publish?.('room', archived.room)
      for (const ticket of archived.tickets) controls.publish?.('ticket', ticket)
    }
  } else {
    const tickets = ticketArchiveClosure(store.tickets, [id])
    for (const ticket of tickets) {
      await controls.stopLoops?.('ticket', ticket.id)
      const thread = ticket.threadId ? store.threads.find((candidate) => candidate.id === ticket.threadId) : null
      if (thread?.projectId === ticket.projectId) {
        controls.abortThread?.(thread.id)
        await controls.abortMissions?.({ threadId: thread.id })
      }
    }
    const archived = archiveTicketState(store, id, now)
    if (archived) {
      await snapshotArchive('ticket', id, archived.tickets)
      for (const ticket of archived.tickets) controls.publish?.('ticket', ticket)
    }
  }
  persistProjects(); persistThreads(); persistRooms(); persistTickets()
  await writeTensorFoldStudioProtectedJson(['archives', 'index.json'], persistentArchiveSnapshot()).catch(() => {})
  return archiveSnapshot()
}

export async function restoreEntity(kind: ArchiveKind, id: string, controls: ArchiveRuntimeControls = {}): Promise<ArchiveSnapshot> {
  assertArchiveEntityExists(kind, id)
  const now = Date.now()
  if (kind === 'project') {
    const candidate = store.projects.find((project) => project.id === id)
    const batchAt = candidate?.archivedAt ?? null
    if (candidate && batchAt != null) {
      const restoringThreads = new Set(store.threads.filter((thread) => thread.projectId === id && thread.archivedAt === batchAt).map((thread) => thread.id))
      const restoringRooms = new Set(store.rooms.filter((room) => room.projectId === id && room.archivedAt === batchAt).map((room) => room.id))
      const restoringTickets = store.tickets.filter((ticket) => ticket.projectId === id && ticket.archivedAt === batchAt)
      assertTicketBatchRestorable(restoringTickets, { projectIds: new Set([id]), roomIds: restoringRooms, threadIds: restoringThreads })
    }
    const restored = restoreProjectState(store, id, now)
    if (restored) {
      controls.publish?.('project', restored.project)
      for (const thread of restored.threads) controls.publish?.('thread', thread)
      for (const room of restored.rooms) controls.publish?.('room', room)
      for (const ticket of restored.tickets) controls.publish?.('ticket', ticket)
    }
  } else if (kind === 'thread') {
    const candidate = store.threads.find((thread) => thread.id === id)
    if (candidate) {
      const project = candidate.projectId ? store.projects.find((item) => item.id === candidate.projectId) : null
      if (candidate.projectId && (!project || project.archivedAt)) throw new Error('Restore the parent project before restoring this thread.')
      const roots = store.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
      const closure = ticketArchiveClosure(store.tickets, roots)
      const restoringTickets = candidate.archivedAt == null ? [] : closure.filter((ticket) => ticket.archivedAt === candidate.archivedAt)
      assertTicketBatchRestorable(restoringTickets, { threadIds: new Set([id]) })
      const restored = restoreThreadState(store, id, now)
      if (restored) {
        controls.publish?.('thread', restored.thread)
        for (const ticket of restored.tickets) controls.publish?.('ticket', ticket)
      }
    }
  } else if (kind === 'room') {
    const candidate = store.rooms.find((room) => room.id === id)
    const parentProject = candidate ? store.projects.find((project) => project.id === candidate.projectId) : null
    if (candidate && (!parentProject || parentProject.archivedAt)) throw new Error('Restore the parent project before restoring this room.')
    if (candidate?.archivedAt != null) {
      const ticketPool = store.tickets.filter((ticket) => ticket.projectId === candidate.projectId)
      const roots = ticketPool.filter((ticket) => ticket.roomId === id).map((ticket) => ticket.id)
      const restoringTickets = ticketArchiveClosure(ticketPool, roots).filter((ticket) => ticket.archivedAt === candidate.archivedAt)
      assertTicketBatchRestorable(restoringTickets, { roomIds: new Set([id]) })
    }
    const restored = restoreRoomState(store, id, now)
    if (restored) {
      const room = restored.room
      controls.publish?.('room', room)
      for (const ticket of restored.tickets) controls.publish?.('ticket', ticket)
    }
  } else {
    const candidate = store.tickets.find((ticket) => ticket.id === id)
    if (candidate) {
      const closure = ticketArchiveClosure(store.tickets, [id])
      const restoringTickets = candidate.archivedAt == null ? [] : closure.filter((ticket) => ticket.archivedAt === candidate.archivedAt)
      assertTicketBatchRestorable(restoringTickets)
    }
    const restored = restoreTicketState(store, id, now)
    if (restored) {
      for (const ticket of restored.tickets) controls.publish?.('ticket', ticket)
    }
  }
  persistProjects(); persistThreads(); persistRooms(); persistTickets()
  await writeTensorFoldStudioProtectedJson(['archives', 'index.json'], persistentArchiveSnapshot()).catch(() => {})
  return archiveSnapshot()
}
