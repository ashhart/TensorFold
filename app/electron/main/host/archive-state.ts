import type { Project, Room, Thread, Ticket } from '../../agent/types'

export interface ArchiveState {
  projects: Project[]
  threads: Thread[]
  rooms: Room[]
  tickets: Ticket[]
}

export interface ProjectArchiveMutation {
  project: Project
  threads: Thread[]
  rooms: Room[]
  tickets: Ticket[]
}

export interface RoomArchiveMutation {
  room: Room
  tickets: Ticket[]
}

export interface TicketArchiveMutation {
  ticket: Ticket
  /** Root plus every recursive descendant whose archive projection may have changed. */
  tickets: Ticket[]
}

export interface ThreadArchiveMutation {
  thread: Thread
  /** Tickets linked to the thread plus their recursive descendants. */
  tickets: Ticket[]
}

function stamp<T extends { archivedAt?: number | null; updatedAt: number }>(item: T, archivedAt: number | null, now: number): T {
  item.archivedAt = archivedAt
  item.updatedAt = now
  return item
}

function uniqueArchiveBatchAt(items: Array<{ archivedAt?: number | null }>, requested: number): number {
  const occupied = new Set(items.map((item) => item.archivedAt).filter((value): value is number => typeof value === 'number'))
  let candidate = requested
  while (occupied.has(candidate)) candidate++
  return candidate
}

/** Soft-archive a project and every record whose ownership is the project. */
export function archiveProjectState(state: ArchiveState, id: string, now = Date.now()): ProjectArchiveMutation | null {
  const project = state.projects.find((candidate) => candidate.id === id)
  if (!project) return null
  const threads = state.threads.filter((candidate) => candidate.projectId === id)
  const rooms = state.rooms.filter((candidate) => candidate.projectId === id)
  const tickets = state.tickets.filter((candidate) => candidate.projectId === id)
  if (project.archivedAt) return { project, threads, rooms, tickets }
  // The shared timestamp is a durable batch marker. Existing child archives retain their
  // earlier marker, so restoring the parent cannot resurrect something archived separately.
  const batchAt = uniqueArchiveBatchAt([...threads, ...rooms, ...tickets], now)
  const preexistingRoomArchive = new Map(rooms.filter((room) => room.archivedAt).map((room) => [room.id, room.archivedAt!]))
  const preexistingThreadArchive = new Map(threads.filter((thread) => thread.archivedAt).map((thread) => [thread.id, thread.archivedAt!]))
  stamp(project, batchAt, now)
  for (const item of [...threads, ...rooms]) if (!item.archivedAt) stamp(item, batchAt, now)
  const inheritedMarkers = new Map<string, number>()
  for (const ticket of tickets) {
    const roomBatch = ticket.roomId ? preexistingRoomArchive.get(ticket.roomId) : null
    const threadBatch = ticket.threadId ? preexistingThreadArchive.get(ticket.threadId) : null
    const inherited = roomBatch ?? threadBatch
    if (inherited != null) inheritedMarkers.set(ticket.id, inherited)
  }
  archiveTicketForest(tickets, tickets.map((ticket) => ticket.id), batchAt, now, inheritedMarkers)
  return { project, threads, rooms, tickets }
}

/** Restore the same ownership closure archived by archiveProjectState. */
export function restoreProjectState(state: ArchiveState, id: string, now = Date.now()): ProjectArchiveMutation | null {
  const project = state.projects.find((candidate) => candidate.id === id)
  if (!project) return null
  const threads = state.threads.filter((candidate) => candidate.projectId === id)
  const rooms = state.rooms.filter((candidate) => candidate.projectId === id)
  const tickets = state.tickets.filter((candidate) => candidate.projectId === id)
  const batchAt = project.archivedAt
  if (!batchAt) return { project, threads, rooms, tickets }
  stamp(project, null, now)
  for (const item of [...threads, ...rooms, ...tickets]) if (item.archivedAt === batchAt) stamp(item, null, now)
  return { project, threads, rooms, tickets }
}

/** A room archive also hides its room-scoped tickets without discarding them. */
export function archiveRoomState(state: ArchiveState, id: string, now = Date.now()): RoomArchiveMutation | null {
  const room = state.rooms.find((candidate) => candidate.id === id)
  if (!room) return null
  const ticketPool = state.tickets.filter((candidate) => candidate.projectId === room.projectId)
  const rootIds = ticketPool.filter((candidate) => candidate.roomId === id).map((ticket) => ticket.id)
  const tickets = ticketArchiveClosure(ticketPool, rootIds)
  if (room.archivedAt) return { room, tickets }
  const batchAt = uniqueArchiveBatchAt(tickets, now)
  const preexistingThreadArchive = new Map(state.threads.filter((thread) => thread.archivedAt).map((thread) => [thread.id, thread.archivedAt!]))
  stamp(room, batchAt, now)
  const inheritedMarkers = new Map<string, number>()
  for (const ticket of tickets) {
    const threadBatch = ticket.threadId ? preexistingThreadArchive.get(ticket.threadId) : null
    if (threadBatch != null) inheritedMarkers.set(ticket.id, threadBatch)
  }
  archiveTicketForest(ticketPool, rootIds, batchAt, now, inheritedMarkers)
  return { room, tickets }
}

export function restoreRoomState(state: ArchiveState, id: string, now = Date.now()): RoomArchiveMutation | null {
  const room = state.rooms.find((candidate) => candidate.id === id)
  if (!room) return null
  const ticketPool = state.tickets.filter((candidate) => candidate.projectId === room.projectId)
  const rootIds = ticketPool.filter((candidate) => candidate.roomId === id).map((ticket) => ticket.id)
  const tickets = ticketArchiveClosure(ticketPool, rootIds)
  const batchAt = room.archivedAt
  if (!batchAt) return { room, tickets }
  stamp(room, null, now)
  restoreTicketForest(ticketPool, rootIds, batchAt, now)
  return { room, tickets }
}

export function ticketArchiveClosure(tickets: Ticket[], rootIds: Iterable<string>): Ticket[] {
  const included = new Set([...rootIds])
  let changed = true
  while (changed) {
    changed = false
    for (const ticket of tickets) {
      if (!ticket.parentId || !included.has(ticket.parentId) || included.has(ticket.id)) continue
      included.add(ticket.id)
      changed = true
    }
  }
  return tickets.filter((ticket) => included.has(ticket.id))
}

/**
 * Archive a ticket forest with one new batch marker while preserving nested independent
 * archives. Active descendants inherit their nearest already-archived ancestor's marker,
 * so restoring an epic/thread never resurrects a subtree the operator archived earlier.
 */
function archiveTicketForest(tickets: Ticket[], rootIds: Iterable<string>, batchAt: number, now: number, inheritedMarkers = new Map<string, number>()): Ticket[] {
  const closure = ticketArchiveClosure(tickets, rootIds)
  const closureById = new Map(closure.map((ticket) => [ticket.id, ticket]))
  const priorMarkers = new Map(closure.map((ticket) => [ticket.id, ticket.archivedAt ?? inheritedMarkers.get(ticket.id) ?? null]))
  const markers = new Map<string, number>()
  const markerFor = (ticket: Ticket, visiting = new Set<string>()): number => {
    const known = markers.get(ticket.id)
    if (known != null) return known
    const prior = priorMarkers.get(ticket.id)
    if (prior != null) {
      markers.set(ticket.id, prior)
      return prior
    }
    if (visiting.has(ticket.id)) return batchAt
    const nextVisiting = new Set(visiting).add(ticket.id)
    const parent = ticket.parentId ? closureById.get(ticket.parentId) : null
    const marker = parent ? markerFor(parent, nextVisiting) : batchAt
    markers.set(ticket.id, marker)
    return marker
  }
  for (const ticket of closure) {
    if (ticket.archivedAt == null) stamp(ticket, markerFor(ticket), now)
  }
  return closure
}

function restoreTicketForest(tickets: Ticket[], rootIds: Iterable<string>, batchAt: number | null | undefined, now: number): Ticket[] {
  const closure = ticketArchiveClosure(tickets, rootIds)
  if (batchAt == null) return closure
  for (const ticket of closure) if (ticket.archivedAt === batchAt) stamp(ticket, null, now)
  return closure
}

export function archiveTicketState(state: Pick<ArchiveState, 'tickets'>, id: string, now = Date.now()): TicketArchiveMutation | null {
  const ticket = state.tickets.find((candidate) => candidate.id === id)
  if (!ticket) return null
  const closure = ticketArchiveClosure(state.tickets, [id])
  const batchAt = ticket.archivedAt ?? uniqueArchiveBatchAt(closure, now)
  const tickets = archiveTicketForest(state.tickets, [id], batchAt, now)
  return { ticket, tickets }
}

export function restoreTicketState(state: Pick<ArchiveState, 'tickets'>, id: string, now = Date.now()): TicketArchiveMutation | null {
  const ticket = state.tickets.find((candidate) => candidate.id === id)
  if (!ticket) return null
  const tickets = restoreTicketForest(state.tickets, [id], ticket.archivedAt, now)
  return { ticket, tickets }
}

export function archiveThreadState(state: Pick<ArchiveState, 'threads' | 'tickets'>, id: string, now = Date.now()): ThreadArchiveMutation | null {
  const thread = state.threads.find((candidate) => candidate.id === id)
  if (!thread) return null
  const roots = state.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
  const closure = ticketArchiveClosure(state.tickets, roots)
  const batchAt = thread.archivedAt ?? uniqueArchiveBatchAt(closure, now)
  if (thread.archivedAt == null) stamp(thread, batchAt, now)
  const tickets = archiveTicketForest(state.tickets, roots, batchAt, now)
  return { thread, tickets }
}

export function restoreThreadState(state: Pick<ArchiveState, 'threads' | 'tickets'>, id: string, now = Date.now()): ThreadArchiveMutation | null {
  const thread = state.threads.find((candidate) => candidate.id === id)
  if (!thread) return null
  const batchAt = thread.archivedAt
  const roots = state.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
  if (batchAt != null) stamp(thread, null, now)
  const tickets = restoreTicketForest(state.tickets, roots, batchAt, now)
  return { thread, tickets }
}
