import { api } from '../../ipc'
import { enqueueWorkspaceMutation } from '../shared/queue'
import { projectRecordHostId, remoteHostIdForRoom, remoteHostIdForThread, owningRemoteHostIdForTicket } from '../shared/remote-host'
import { mergeArchiveSnapshots, mergeThreadLists, archiveOriginHostId } from '../shared/merge'
import { mergeOriginBoundRecords, projectedHostId, projectedProjectHostId, projectionOriginsMatch, removeProjectOwnedProxyState, clearPendingAgentEvents } from '../shared/proxy-state'
import { runErrorMessage } from '../shared/model-policy'
import type { Slice, View } from '../types'
import type { ArchiveKind, ArchiveSnapshot, Project, Room, Thread, Ticket } from '../../types'

export const archivesSlice: Slice = (set, get) => ({
  archives: null as ArchiveSnapshot | null,

  async refreshArchives() {
    const local = await api.getArchives()
    const hosts = (get().settings?.remoteOrchestrators ?? []).filter((host) => host.serverUrl && host.authToken)
    const remote = await Promise.allSettled(hosts.map((host) => api.getRemoteOrchestratorArchives(host.id)))
    const previous = get().archives ?? { projects: [], threads: [], rooms: [], tickets: [] }
    const retainedForHost = (hostId: string): ArchiveSnapshot => ({
      projects: previous.projects.filter((item) => archiveOriginHostId(item) === hostId),
      threads: previous.threads.filter((item) => archiveOriginHostId(item) === hostId),
      rooms: previous.rooms.filter((item) => archiveOriginHostId(item) === hostId),
      tickets: previous.tickets.filter((item) => archiveOriginHostId(item) === hostId),
    })
    const archives = mergeArchiveSnapshots([
      local,
      ...remote.map((result, index) => result.status === 'fulfilled' ? result.value : retainedForHost(hosts[index].id)),
    ])
    set({ archives })
  },
  async archiveItem(kind: ArchiveKind, id: string) {
    const lockKeys = kind === 'project'
      ? [`project:${id}`, 'rooms:*', 'tickets:*']
      : kind === 'room'
        ? ['rooms:*', `room:${id}`, 'tickets:*']
        : kind === 'ticket'
          ? ['tickets:*', `ticket:${id}`]
          : ['tickets:*']
    await enqueueWorkspaceMutation(lockKeys, async () => {
    let remoteOrigin: string | null = null
    let remoteRecord: Project | Thread | Room | Ticket | null = null
    if (kind === 'project') {
      const project = get().projects.find((candidate) => candidate.id === id)
      const hostId = projectRecordHostId(project)
      if (hostId) { remoteOrigin = hostId; remoteRecord = await api.deleteRemoteOrchestratorProject(hostId, id) }
    } else if (kind === 'room') {
      const hostId = remoteHostIdForRoom(get(), id)
      if (hostId) { remoteOrigin = hostId; remoteRecord = await api.deleteRemoteOrchestratorRoom(hostId, id) }
    } else if (kind === 'thread') {
      const hostId = remoteHostIdForThread(get(), id)
      if (hostId) { remoteOrigin = hostId; remoteRecord = await api.archiveRemoteOrchestratorThread(hostId, id) }
    } else if (kind === 'ticket') {
      const ticket = get().tickets.find((candidate) => candidate.id === id)
      const hostId = owningRemoteHostIdForTicket(get(), ticket)
      if (hostId) { remoteOrigin = hostId; remoteRecord = await api.deleteRemoteOrchestratorTicket(hostId, id) }
    }
    // Remote projects have a durable local proxy; remote threads/rooms/tickets do not. Avoid
    // mutating an unrelated local record if an adversarial remote host deliberately reuses its id.
    let localArchives: ArchiveSnapshot | null = null
    try {
      if (!remoteOrigin || kind === 'project') localArchives = await api.archiveItem(kind, id)
    } catch (error) {
      if (remoteOrigin) {
        try {
          if (kind === 'project') await api.restoreRemoteOrchestratorProject(remoteOrigin, id)
          else if (kind === 'thread') await api.restoreRemoteOrchestratorThread(remoteOrigin, id)
          else if (kind === 'room') await api.restoreRemoteOrchestratorRoom(remoteOrigin, id)
          else await api.restoreRemoteOrchestratorTicket(remoteOrigin, id)
        } catch {
          throw new Error(`Local archive persistence failed after the Orchestrator archived ${kind} ${id}; automatic rollback also failed. Refresh the archive before retrying. Cause: ${runErrorMessage(error)}`)
        }
      }
      throw error
    }
    const remoteSnapshot: ArchiveSnapshot = { projects: [], threads: [], rooms: [], tickets: [] }
    if (remoteRecord) {
      if (kind === 'project') remoteSnapshot.projects.push(remoteRecord as Project)
      else if (kind === 'thread') remoteSnapshot.threads.push(remoteRecord as Thread)
      else if (kind === 'room') remoteSnapshot.rooms.push(remoteRecord as Room)
      else remoteSnapshot.tickets.push(remoteRecord as Ticket)
    }
    const nextArchives = mergeArchiveSnapshots([get().archives ?? { projects: [], threads: [], rooms: [], tickets: [] }, ...(localArchives ? [localArchives] : []), remoteSnapshot])
    set((st) => {
      const view: View =
        kind === 'project' && st.view.name === 'project' && st.view.id === id ? { name: 'projects' } :
        kind === 'room' && st.view.name === 'room' && st.view.id === id ? { name: 'projects' } :
        kind === 'thread' && st.view.name === 'thread' && st.view.id === id ? { name: 'home' } :
        st.view
      const projectIds = kind === 'project' ? new Set([id]) : new Set<string>()
      // Drop buffered agent events for any thread leaving the list; it will never drain them now.
      for (const t of st.threads) if (t.id === id || projectIds.has(t.projectId ?? '')) clearPendingAgentEvents(t.id, projectedHostId(t))
      if (kind === 'project') {
        return { archives: nextArchives, ...removeProjectOwnedProxyState(st, id, view, remoteOrigin) }
      }
      const status = { ...st.status }
      const approvals = { ...st.approvals }
      const roomStatus = { ...st.roomStatus }
      const roomLive = { ...st.roomLive }
      if (kind === 'thread') {
        delete status[id]
        delete approvals[id]
      }
      if (kind === 'room') {
        delete roomStatus[id]
        delete roomLive[id]
      }
      const matchesArchivedOrigin = (record: { orchestratorHostId?: string | null }) => projectionOriginsMatch(projectedHostId(record), remoteOrigin)
      return {
        archives: nextArchives,
        projects: st.projects,
        threads: st.threads.filter((t) => !(t.id === id && matchesArchivedOrigin(t)) && !projectIds.has(t.projectId ?? '')),
        rooms: st.rooms.filter((r) => !(r.id === id && matchesArchivedOrigin(r)) && !projectIds.has(r.projectId)),
        tickets: kind === 'ticket'
          ? st.tickets.filter((ticket) => ticket.id !== id || !matchesArchivedOrigin(ticket))
          : kind === 'room'
            ? st.tickets.filter((ticket) => ticket.roomId !== id || !matchesArchivedOrigin(ticket))
            : st.tickets,
        status,
        approvals,
        roomStatus,
        roomLive,
        view,
      }
    })
    await get().refreshArchives().catch((error) => {
      set({ notice: { kind: 'error', message: `Archived successfully, but the archive list could not be refreshed: ${runErrorMessage(error)}` } })
    })
    })
  },
  async restoreItem(kind: ArchiveKind, id: string, originHostId?: string | null) {
    const lockKeys = kind === 'project'
      ? [`project:${id}`, 'rooms:*', 'tickets:*']
      : kind === 'room'
        ? ['rooms:*', `room:${id}`, 'tickets:*']
        : kind === 'ticket'
          ? ['tickets:*', `ticket:${id}`]
          : ['tickets:*']
    await enqueueWorkspaceMutation(lockKeys, async () => {
    const archived = get().archives ?? await api.getArchives()
    const matchesOrigin = (item: { id: string; orchestratorHostId?: string | null; orchestratorId?: string | null }) =>
      item.id === id && (originHostId === undefined || archiveOriginHostId(item) === originHostId)
    const archivedProject = kind === 'project' ? archived.projects.find(matchesOrigin) : null
    const archivedThread = kind === 'thread' ? archived.threads.find(matchesOrigin) : null
    const archivedRoom = kind === 'room' ? archived.rooms.find(matchesOrigin) : null
    const archivedTicket = kind === 'ticket' ? archived.tickets.find(matchesOrigin) : null
    const projectId = archivedRoom?.projectId ?? archivedTicket?.projectId ?? archivedThread?.projectId ?? null
    const parentProject = archivedProject
      ?? get().projects.find((item) => item.id === projectId)
      ?? archived.projects.find((item) => item.id === projectId && (originHostId === undefined || archiveOriginHostId(item) === originHostId))
      ?? null
    const hostId = archivedThread?.orchestratorHostId
      ?? archivedRoom?.orchestratorHostId
      ?? archivedTicket?.orchestratorHostId
      ?? (parentProject?.executionTarget === 'orchestrator' ? parentProject.orchestratorId : null)
      ?? null

    // Remote first: if the Orchestrator is offline, leave the local proxy archived rather than
    // claim a restore that will immediately diverge/reappear on the next snapshot.
    const remoteProject = kind === 'project' && hostId ? await api.restoreRemoteOrchestratorProject(hostId, id) : null
    const remoteThread = kind === 'thread' && hostId ? await api.restoreRemoteOrchestratorThread(hostId, id) : null
    const remoteRoom = kind === 'room' && hostId ? await api.restoreRemoteOrchestratorRoom(hostId, id) : null
    const remoteTicket = kind === 'ticket' && hostId ? await api.restoreRemoteOrchestratorTicket(hostId, id) : null
    // Parent restores can reactivate a whole ownership closure. Do not depend on SSE delivery
    // to repopulate those children: fetch an authenticated active snapshot and merge the exact
    // project/room slice below. A refresh failure does not reverse a successful host restore.
    const remoteClosure = hostId && (kind === 'project' || kind === 'room')
      ? await api.getRemoteSnapshot(hostId).catch((error) => ({ ok: false, error: runErrorMessage(error) } as Awaited<ReturnType<typeof api.getRemoteSnapshot>>))
      : null
    try {
      if (!hostId || kind === 'project') await api.restoreItem(kind, id)
    } catch (error) {
      if (hostId) {
        try {
          if (kind === 'project') await api.deleteRemoteOrchestratorProject(hostId, id)
          else if (kind === 'thread') await api.archiveRemoteOrchestratorThread(hostId, id)
          else if (kind === 'room') await api.deleteRemoteOrchestratorRoom(hostId, id)
          else await api.deleteRemoteOrchestratorTicket(hostId, id)
        } catch {
          throw new Error(`Local restore persistence failed after the Orchestrator restored ${kind} ${id}; automatic rollback also failed. Refresh before retrying. Cause: ${runErrorMessage(error)}`)
        }
      }
      throw error
    }
    const current = get()
    const loaded = await Promise.allSettled([api.getProjects(), api.getThreads(), api.getRooms(), api.getTickets()])
    const value = <T,>(result: PromiseSettledResult<T>, fallback: T): T => result.status === 'fulfilled' ? result.value : fallback
    const projects = value(loaded[0] as PromiseSettledResult<Project[]>, current.projects)
    const threads = value(loaded[1] as PromiseSettledResult<Thread[]>, current.threads)
    const rooms = value(loaded[2] as PromiseSettledResult<Room[]>, current.rooms)
    const tickets = value(loaded[3] as PromiseSettledResult<Ticket[]>, current.tickets)
    const targetBatch = archivedProject?.archivedAt ?? archivedRoom?.archivedAt ?? null
    const sameOrigin = (item: { orchestratorHostId?: string | null; orchestratorId?: string | null }) => archiveOriginHostId(item) === hostId
    const nextArchives: ArchiveSnapshot = {
      projects: archived.projects.filter((item) => !(kind === 'project' && item.id === id && sameOrigin(item))),
      threads: archived.threads.filter((item) => !(
        (kind === 'thread' && item.id === id && sameOrigin(item)) ||
        (kind === 'project' && item.projectId === id && item.archivedAt === targetBatch && sameOrigin(item))
      )),
      rooms: archived.rooms.filter((item) => !(
        (kind === 'room' && item.id === id && sameOrigin(item)) ||
        (kind === 'project' && item.projectId === id && item.archivedAt === targetBatch && sameOrigin(item))
      )),
      tickets: archived.tickets.filter((item) => !(
        (kind === 'ticket' && item.id === id && sameOrigin(item)) ||
        (kind === 'room' && item.roomId === id && item.archivedAt === targetBatch && sameOrigin(item)) ||
        (kind === 'project' && item.projectId === id && item.archivedAt === targetBatch && sameOrigin(item))
      )),
    }
    set({
      archives: nextArchives,
      projects: remoteProject ? mergeOriginBoundRecords(projects, [remoteProject], projectedProjectHostId) : projects,
      threads: mergeThreadLists(threads, [
        ...(remoteThread ? [remoteThread] : []),
        ...((remoteClosure?.ok && kind === 'project' ? remoteClosure.threads?.filter((item) => item.projectId === id) : []) ?? []),
      ]),
      rooms: mergeOriginBoundRecords(rooms, [
        ...(remoteRoom ? [remoteRoom] : []),
        ...((remoteClosure?.ok && kind === 'project' ? remoteClosure.rooms?.filter((item) => item.projectId === id) : []) ?? []),
      ], projectedHostId),
      tickets: mergeOriginBoundRecords(tickets, [
        ...(remoteTicket ? [remoteTicket] : []),
        ...((remoteClosure?.ok && kind === 'project'
          ? remoteClosure.tickets?.filter((item) => item.projectId === id)
          : remoteClosure?.ok && kind === 'room'
            ? remoteClosure.tickets?.filter((item) => item.roomId === id)
            : []) ?? []),
      ], projectedHostId),
      ...(remoteClosure && !remoteClosure.ok
        ? { notice: { kind: 'error' as const, message: `Restored successfully, but active child records could not be refreshed: ${remoteClosure.error ?? 'Orchestrator snapshot unavailable.'}` } }
        : {}),
    })
    await get().refreshArchives().catch((error) => {
      set({ notice: { kind: 'error', message: `Restored successfully, but the archive list could not be refreshed: ${runErrorMessage(error)}` } })
    })
    })
  },
})
