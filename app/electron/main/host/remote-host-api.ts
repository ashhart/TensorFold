import path from 'node:path'
import os from 'node:os'
import { promises as fs } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { canonicalHomeDirectory, resolvePairedWorkspace, type RemoteWorkspaceScope } from '../../agent/local-workspace-policy'
import { resolveProviderModel, surfaceModelOverride } from '../../agent/provider-accounts'
import { DEFAULT_ROOM_COLLABORATION_MODE } from '../../agent/room-collaboration'
import { roomModeDefaults } from '../../agent/room-outcomes'
import {
  assertRoomConfigUpdateAllowed,
  assertRoomSeatRoutesConfigured,
  appendRoomSteeringEvent,
  applyRoomManualMemoryCommand,
  pickRoomConfigPatch,
  pickRoomManualMemoryCommand,
  pickTicketCreateInput,
  pickTicketRemoteLifecycleCommand,
  pickTicketUserPatch,
  assertTicketUserUpdateAllowed,
  ticketRunEvidence,
} from './room-ticket-patch-policy'
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
import { sanitizeProjectCreateInput, sanitizeProjectUpdatePatch, validateProjectProviderDefaults } from './project-patch-policy'
import { mkTicket } from './ticket-helpers'
import { normalizeTicketExecution, normalizeStringList } from '../../agent/tickets'
import { selectedProviderAccountId, resolveProjectProvider } from './provider-runtime'
import { killProjectTerminals } from './terminal-runtime'
import type { Attachment, CueRunPayload, CueRunResult, Project, RemoteDirectoryListResult, Room, RoomEventItem, RoomManualMemoryCommand, RoomStreamEvent, StartRunPayload, Thread, Ticket, TicketRemoteLifecycleCommand } from '../../agent/types'
import { ticketArchiveClosure } from './archive-state'
import { store } from '../state'
import { assertTicketDependencyGraph, requireTicketDependenciesReady } from '../../../shared/ticket-dependencies'

/**
 * The `*OnThisHost` layer: every REST mutation the remote-control server forwards to this
 * host. These are pure wiring consumed only by the `remoteServer` config object.
 *
 * The functions close over a number of collaborators that are initialized later in the
 * main-process bootstrap (the abort-controller maps, the loop host, the thread/room run
 * entry points, the remote server itself, …). Those are passed as accessor functions so
 * the factory can be constructed early while the lookups stay lazy — exactly mirroring the
 * module-scope closure semantics the inlined version relied on.
 */
export function createRemoteHostApi(deps: RemoteHostApiDeps) {
  const {
    sendToRenderer,
    persistProjects,
    persistThreads,
    persistRooms,
    persistTickets,
    defaultRoomCollaborationMode,
    defaultRoomWorkerBudget,
  } = deps

  function remoteMutationError(message: string, code = 409): Error {
    return Object.assign(new Error(message), { code })
  }

  function assertArchiveStateNotPatched(input: object): void {
    if (Object.prototype.hasOwnProperty.call(input, 'archivedAt')) throw remoteMutationError('Archive state can only change through the archive/restore endpoints.', 400)
  }

  function assertRecordIdNotPatched(input: object): void {
    for (const field of ['id', 'createdAt', 'updatedAt'] as const) {
      if (Object.prototype.hasOwnProperty.call(input, field)) throw remoteMutationError(`${field} is host-owned and cannot be changed through update endpoints.`, 400)
    }
  }

  function requireActiveProjectOnThisHost(id: string): Project {
    const project = store.projects.find((candidate) => candidate.id === id)
    if (!project) throw remoteMutationError('Project not found.', 404)
    if (project.archivedAt) throw remoteMutationError('Project is archived; restore it before mutating child work.')
    return project
  }

  function requireActiveThreadOnThisHost(id: string): Thread {
    const thread = store.threads.find((candidate) => candidate.id === id)
    if (!thread) throw remoteMutationError('Thread not found.', 404)
    if (thread.archivedAt) throw remoteMutationError('Thread is archived; restore it before starting or guiding it.')
    if (thread.projectId) requireActiveProjectOnThisHost(thread.projectId)
    return thread
  }

  function requireActiveRoomOnThisHost(id: string): Room {
    const room = store.rooms.find((candidate) => candidate.id === id)
    if (!room) throw remoteMutationError('Room not found.', 404)
    if (room.archivedAt) throw remoteMutationError('Room is archived; restore it before mutating it.')
    requireActiveProjectOnThisHost(room.projectId)
    return room
  }

  function validateTicketOwnershipOnThisHost(
    projectId: string,
    roomId?: string | null,
    relations: {
      selfId?: string
      parentId?: string | null
      threadId?: string | null
      requireActiveThread?: boolean
      restoringTicketIds?: ReadonlySet<string>
      restoringProjectIds?: ReadonlySet<string>
      restoringRoomIds?: ReadonlySet<string>
      restoringThreadIds?: ReadonlySet<string>
      dependsOn?: string[]
    } = {},
  ): void {
    const project = store.projects.find((candidate) => candidate.id === projectId)
    if (!project) throw remoteMutationError('Project not found.', 404)
    if (project.archivedAt && !relations.restoringProjectIds?.has(project.id)) throw remoteMutationError('Restore the ticket project before mutating this ticket.')
    if (roomId) {
      const room = store.rooms.find((candidate) => candidate.id === roomId)
      if (!room) throw remoteMutationError('Ticket room not found.', 400)
      if (room.archivedAt && !relations.restoringRoomIds?.has(room.id)) throw remoteMutationError('Restore the ticket room before mutating this ticket.')
      if (room.projectId !== projectId) throw remoteMutationError('Ticket room must belong to the same project.', 400)
    }
    if (relations.parentId) {
      if (relations.parentId === relations.selfId) throw remoteMutationError('A ticket cannot be its own parent.', 400)
      const parent = store.tickets.find((candidate) => candidate.id === relations.parentId)
      if (!parent) throw remoteMutationError('Parent ticket not found.', 400)
      if (parent.archivedAt && !relations.restoringTicketIds?.has(parent.id)) throw remoteMutationError('Restore the parent ticket before assigning or restoring this subticket.')
      if (parent.projectId !== projectId) throw remoteMutationError('Parent ticket must belong to the same project.', 400)
    }
    if (relations.threadId) {
      const thread = store.threads.find((candidate) => candidate.id === relations.threadId)
      if (!thread) throw remoteMutationError('Ticket thread not found.', 400)
      if (thread.projectId !== projectId) throw remoteMutationError('Ticket thread must belong to the same project.', 400)
      if (relations.requireActiveThread !== false && thread.archivedAt && !relations.restoringThreadIds?.has(thread.id)) throw remoteMutationError('Restore the ticket thread before assigning it.', 409)
    }
    if (relations.dependsOn) {
      try {
        assertTicketDependencyGraph(store.tickets, {
          id: relations.selfId ?? '__new_ticket__',
          projectId,
          dependsOn: relations.dependsOn,
        })
      } catch (error) {
        throw remoteMutationError(String((error as Error)?.message ?? error), 400)
      }
    }
  }

  function resolveHostDirectory(raw: string): string {
    const input = String(raw || '').trim()
    const home = os.homedir()
    if (!input || input === '~') return home
    if (input.startsWith('~/')) return path.resolve(home, input.slice(2))
    return path.resolve(input)
  }

  async function listHostDirectories(raw: string, options: { confineToHome?: boolean } = {}): Promise<RemoteDirectoryListResult> {
    let target = resolveHostDirectory(raw)
    const home = options.confineToHome ? await fs.realpath(os.homedir()) : os.homedir()
    // Least-privilege tier (paired-device tokens) may only browse inside the user's home
    // subtree; the owner token keeps unrestricted local browsing.
    const confined = options.confineToHome === true
    if (confined) target = await canonicalHomeDirectory(target)
    const stat = await fs.stat(target)
    if (!stat.isDirectory()) throw Object.assign(new Error('Path is not a directory.'), { code: 400 })
    const entries = await fs.readdir(target, { withFileTypes: true })
    const dirs = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => ({ name: entry.name, path: path.join(target, entry.name), isDirectory: true }))
      .sort((a, b) => {
        const ah = a.name.startsWith('.')
        const bh = b.name.startsWith('.')
        if (ah !== bh) return ah ? 1 : -1
        return a.name.localeCompare(b.name)
      })
    const parent = path.dirname(target)
    // A confined caller at the home root gets no parent link, so it cannot navigate above it.
    return { ok: true, path: target, parent: parent === target || (confined && target === home) ? null : parent, entries: dirs }
  }

  async function createProjectOnThisHost(opts: Partial<Project> & { name?: string; cwd?: string }, scope: RemoteWorkspaceScope = {}): Promise<Project> {
    assertArchiveStateNotPatched(opts)
    let safe: Partial<Project>
    try {
      // A caller-chosen id is accepted solely for idempotent retry. The host owns all timestamps.
      safe = sanitizeProjectCreateInput(opts, { allowId: true, allowCreatedAt: false })
    } catch (error) {
      throw remoteMutationError(error instanceof Error ? error.message : String(error), 400)
    }
    if (safe.executionTarget && safe.executionTarget !== 'local') throw remoteMutationError('The Orchestrator host creates local-owned projects only.', 400)
    if (safe.orchestratorId) throw remoteMutationError('The Orchestrator host cannot persist a controller-local host id.', 400)
    let cwd = safe.cwd ? resolveHostDirectory(String(safe.cwd)) : ''
    if (scope.confineToHome) {
      cwd = await resolvePairedWorkspace(cwd, safe.workspace)
      safe.workspace = { kind: 'local', path: cwd }
    } else if (safe.workspace?.kind === 'local') {
      const effective = await fs.realpath(resolveHostDirectory(safe.workspace.path))
      if (cwd && await fs.realpath(cwd) !== effective) throw remoteMutationError('Project cwd and local workspace path must identify the same directory.', 400)
      cwd = effective
      safe.workspace = { kind: 'local', path: cwd }
    }
    if (cwd) {
      const stat = await fs.stat(cwd)
      if (!stat.isDirectory()) throw Object.assign(new Error('Project folder is not a directory.'), { code: 400 })
    }
    const now = Date.now()
    const p: Project = {
      id: safe.id || randomUUID(),
      name: safe.name || 'New project',
      cwd,
      workspace: safe.workspace ?? null,
      ...(scope.confineToHome ? { workspaceBoundary: 'home' as const } : {}),
      memory: safe.memory ?? '',
      team: safe.team ?? '',
      mode: safe.mode || 'solo',
      localOnly: safe.localOnly ?? false,
      executionTarget: 'local',
      orchestratorId: null,
      defaultProviderId: safe.defaultProviderId ?? null,
      defaultProviderAccountId: safe.defaultProviderAccountId ?? null,
      defaultModel: safe.defaultModel ?? null,
      kanban: safe.kanban ?? { modelMode: 'single' },
      tags: safe.tags ?? [],
      archivedAt: null,
      createdAt: safe.createdAt ?? now,
      updatedAt: now,
    }
    try {
      validateProjectProviderDefaults(p, {
        localOnly: p.localOnly,
        defaultProviderId: p.defaultProviderId,
        defaultProviderAccountId: p.defaultProviderAccountId,
        defaultModel: p.defaultModel,
        kanban: p.kanban,
      }, store.settings.providers)
    } catch (error) {
      throw remoteMutationError(error instanceof Error ? error.message : String(error), 400)
    }
    const existing = safe.id ? store.projects.find((project) => project.id === safe.id) : null
    if (existing) {
      if (existing.archivedAt) throw remoteMutationError('Project is archived; use the restore endpoint before updating it.')
      if (existing.cwd !== p.cwd || JSON.stringify(existing.workspace ?? null) !== JSON.stringify(p.workspace ?? null)) {
        throw remoteMutationError('An existing project cannot be relocated through create; use the validated update endpoint.', 409)
      }
      return existing
    }
    await deps.ensureProjectInstructionFiles(p)
    store.projects.unshift(p)
    persistProjects()
    sendToRenderer('project:update', p)
    deps.remoteServer().publish({ kind: 'project', data: p })
    return p
  }

  async function updateProjectOnThisHost(id: string, patch: Partial<Project>): Promise<Project> {
    assertArchiveStateNotPatched(patch)
    assertRecordIdNotPatched(patch)
    let nextPatch: Partial<Project>
    try { nextPatch = sanitizeProjectUpdatePatch(patch) } catch (error) { throw remoteMutationError(error instanceof Error ? error.message : String(error), 400) }
    const project = requireActiveProjectOnThisHost(id)
    try { validateProjectProviderDefaults(project, nextPatch, store.settings.providers) } catch (error) { throw remoteMutationError(error instanceof Error ? error.message : String(error), 400) }
    Object.assign(project, nextPatch, { updatedAt: Date.now() })
    persistProjects()
    sendToRenderer('project:update', project)
    deps.remoteServer().publish({ kind: 'project', data: project })
    return project
  }

  async function archiveProjectOnThisHost(id: string): Promise<Project | null> {
    const existing = store.projects.find((candidate) => candidate.id === id)
    if (!existing) return null
    killProjectTerminals(id)
    for (const thread of store.threads.filter((candidate) => candidate.projectId === id)) deps.aborters().get(thread.id)?.abort()
    for (const room of store.rooms.filter((candidate) => candidate.projectId === id)) deps.roomAborters().get(room.id)?.abort()
    await deps.abortAndWaitMissionsForArchive()({ projectId: id })
    await deps.loopHost().stopLoopsForArchive('project', id)
    const archived = archiveProjectState(store, id)
    if (!archived) return null
    persistProjects()
    persistThreads()
    persistRooms()
    persistTickets()
    for (const thread of archived.threads) {
      sendToRenderer('thread:update', thread)
      deps.remoteServer().publish({ kind: 'thread', data: thread })
    }
    for (const room of archived.rooms) {
      sendToRenderer('room:update', room)
      deps.remoteServer().publish({ kind: 'room', data: room })
    }
    for (const ticket of archived.tickets) deps.emitTicket(ticket)
    sendToRenderer('project:update', archived.project)
    deps.remoteServer().publish({ kind: 'project', data: archived.project })
    return archived.project
  }

  async function restoreProjectOnThisHost(id: string): Promise<Project | null> {
    const candidate = store.projects.find((project) => project.id === id)
    if (!candidate) return null
    const batchAt = candidate.archivedAt ?? null
    if (batchAt != null) {
      const restoringThreads = new Set(store.threads.filter((thread) => thread.projectId === id && thread.archivedAt === batchAt).map((thread) => thread.id))
      const restoringRooms = new Set(store.rooms.filter((room) => room.projectId === id && room.archivedAt === batchAt).map((room) => room.id))
      const restoringTickets = store.tickets.filter((ticket) => ticket.projectId === id && ticket.archivedAt === batchAt)
      const restoringTicketIds = new Set(restoringTickets.map((ticket) => ticket.id))
      for (const ticket of restoringTickets) validateTicketOwnershipOnThisHost(ticket.projectId, ticket.roomId, {
        selfId: ticket.id, parentId: ticket.parentId, threadId: ticket.threadId,
        restoringTicketIds, restoringProjectIds: new Set([id]), restoringRoomIds: restoringRooms, restoringThreadIds: restoringThreads,
      })
    }
    const restored = restoreProjectState(store, id)
    if (!restored) return null
    persistProjects(); persistThreads(); persistRooms(); persistTickets()
    for (const thread of restored.threads) {
      sendToRenderer('thread:update', thread)
      deps.remoteServer().publish({ kind: 'thread', data: thread })
    }
    for (const room of restored.rooms) {
      sendToRenderer('room:update', room)
      deps.remoteServer().publish({ kind: 'room', data: room })
    }
    for (const ticket of restored.tickets) deps.emitTicket(ticket)
    sendToRenderer('project:update', restored.project)
    deps.remoteServer().publish({ kind: 'project', data: restored.project })
    return restored.project
  }

  async function archiveThreadOnThisHost(id: string): Promise<Thread | null> {
    const existing = store.threads.find((candidate) => candidate.id === id)
    if (!existing) return null
    deps.aborters().get(id)?.abort()
    await deps.abortAndWaitMissionsForArchive()({ threadId: id })
    await deps.loopHost().stopLoopsForArchive('thread', id)
    const linkedRoots = store.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
    const linkedTickets = ticketArchiveClosure(store.tickets, linkedRoots)
    for (const ticket of linkedTickets) {
      await deps.loopHost().stopLoopsForArchive('ticket', ticket.id)
      if (ticket.threadId) {
        const worker = store.threads.find((candidate) => candidate.id === ticket.threadId)
        if (worker?.projectId === ticket.projectId) {
          deps.aborters().get(worker.id)?.abort()
          await deps.abortAndWaitMissionsForArchive()({ threadId: worker.id })
        }
      }
    }
    const archived = archiveThreadState(store, id)
    if (!archived) return null
    persistThreads()
    persistTickets()
    // Deliberately retain the NDJSON journal: this REST operation is a recoverable archive.
    sendToRenderer('thread:update', archived.thread)
    deps.remoteServer().publish({ kind: 'thread', data: archived.thread })
    for (const ticket of archived.tickets) deps.emitTicket(ticket)
    return archived.thread
  }

  async function restoreThreadOnThisHost(id: string): Promise<Thread | null> {
    const thread = store.threads.find((candidate) => candidate.id === id)
    if (!thread) return null
    if (thread.projectId) requireActiveProjectOnThisHost(thread.projectId)
    const roots = store.tickets.filter((ticket) => ticket.threadId === id).map((ticket) => ticket.id)
    const closure = ticketArchiveClosure(store.tickets, roots)
    const restoring = thread.archivedAt == null ? [] : closure.filter((ticket) => ticket.archivedAt === thread.archivedAt)
    const restoringTicketIds = new Set(restoring.map((ticket) => ticket.id))
    for (const ticket of restoring) {
      validateTicketOwnershipOnThisHost(ticket.projectId, ticket.roomId, {
        selfId: ticket.id,
        parentId: ticket.parentId,
        threadId: ticket.threadId,
        requireActiveThread: true,
        restoringTicketIds,
        restoringThreadIds: new Set([id]),
      })
    }
    const restored = restoreThreadState(store, id)
    if (!restored) return null
    persistThreads()
    persistTickets()
    sendToRenderer('thread:update', restored.thread)
    deps.remoteServer().publish({ kind: 'thread', data: restored.thread })
    for (const ticket of restored.tickets) deps.emitTicket(ticket)
    return restored.thread
  }

  async function createThreadOnThisHost(opts: { cwd?: string; kind?: 'task' | 'chat'; projectId?: string | null }, scope: RemoteWorkspaceScope = {}): Promise<Thread> {
    const kind = opts.kind === 'chat' ? 'chat' : 'task'
    const projectId = opts.projectId ?? null
    const project = projectId ? requireActiveProjectOnThisHost(projectId) : null
    let projectWorkspace = project?.workspace ?? null
    const confineToHome = scope.confineToHome || project?.workspaceBoundary === 'home'
    const candidateCwd = opts.cwd || project?.cwd || (projectWorkspace?.kind === 'ssh' ? projectWorkspace.path : '') || store.settings.defaultCwd || ''
    let selectedCwd = ''
    if (kind !== 'chat' && confineToHome) {
      selectedCwd = await resolvePairedWorkspace(candidateCwd, projectWorkspace)
      projectWorkspace = { kind: 'local', path: selectedCwd }
    } else if (kind === 'chat') {
      selectedCwd = ''
    } else if (projectWorkspace?.kind === 'ssh' && candidateCwd === projectWorkspace.path) {
      selectedCwd = candidateCwd
    } else if (candidateCwd) {
      try {
        const stat = await fs.stat(candidateCwd)
        selectedCwd = stat.isDirectory() ? candidateCwd : ''
      } catch {
        selectedCwd = ''
      }
    }
    const projectOwnsRoute = !!project && (!!project.localOnly || !!project.defaultProviderId || !!project.defaultModel)
    const surfaceRoute = projectOwnsRoute ? null : surfaceModelOverride(store.settings, kind === 'chat' ? 'chat' : 'code')
    const resolved = project ? resolveProjectProvider(
      project,
      surfaceRoute?.providerAccountId,
      surfaceRoute?.model,
      surfaceRoute?.providerId,
    ) : null
    const baseProvider = resolved?.baseProvider
      ?? store.settings.providers.find((p) => p.id === surfaceRoute?.providerId)
      ?? store.settings.providers.find((p) => p.id === store.settings.activeProviderId)
    const requestedAccountId = surfaceRoute?.providerAccountId ?? store.settings.activeProviderAccountId ?? undefined
    const providerAccountId = resolved?.providerAccountId ?? (baseProvider ? selectedProviderAccountId(baseProvider, requestedAccountId) : null)
    const model = resolved?.model ?? (baseProvider ? resolveProviderModel(baseProvider, providerAccountId, surfaceRoute?.model, store.settings.activeModel) : (surfaceRoute?.model ?? store.settings.activeModel))
    const now = Date.now()
    const t: Thread = {
      id: randomUUID(),
      title: kind === 'chat' ? 'New chat' : 'New task',
      kind,
      projectId,
      cwd: selectedCwd,
      workspace: projectWorkspace,
      ...(confineToHome ? { workspaceBoundary: 'home' as const } : {}),
      providerId: baseProvider?.id ?? store.settings.activeProviderId,
      providerAccountId,
      model,
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      apiMessages: [],
      timeline: [],
    }
    store.threads.unshift(t)
    persistThreads()
    sendToRenderer('thread:update', t)
    deps.remoteServer().publish({ kind: 'thread', data: t })
    return t
  }

  async function createRoomOnThisHost(input: { projectId: string; name?: string }): Promise<Room> {
    requireActiveProjectOnThisHost(input.projectId)
    const now = Date.now()
    const r: Room = {
      id: randomUUID(),
      projectId: input.projectId,
      name: input.name || 'New room',
      ...roomModeDefaults(),
      seats: [],
      rootCwd: null,
      workspaceCwd: null,
      scratchpadCwd: null,
      collaborationMode: defaultRoomCollaborationMode,
      workerBudget: { ...defaultRoomWorkerBudget },
      memory: [],
      memorySummary: '',
      problem: '',
      state: 'created',
      events: [],
      finalText: '',
      createdAt: now,
      updatedAt: now,
    }
    store.rooms.unshift(r)
    persistRooms()
    sendToRenderer('room:update', r)
    deps.remoteServer().publish({ kind: 'room', data: r })
    return r
  }

  async function updateRoomOnThisHost(id: string, requestedPatch: Partial<Room>): Promise<Room> {
    const room = requireActiveRoomOnThisHost(id)
    let patch: Partial<Room>
    try {
      patch = pickRoomConfigPatch(requestedPatch)
      assertRoomConfigUpdateAllowed(room, patch, deps.roomAborters().has(id))
      if (patch.seats) assertRoomSeatRoutesConfigured(patch.seats, store.settings.providers, store.projects.find((project) => project.id === room.projectId))
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }
    const modeBudget = patch.mode && patch.deliberationEffort === undefined ? roomModeDefaults(patch.mode).deliberationEffort : undefined
    Object.assign(room, patch, modeBudget ? { deliberationEffort: modeBudget } : {}, { updatedAt: Date.now() })
    if (patch.seats && ['idle', 'created', 'ready', 'degraded'].includes(room.state)) room.state = room.seats.length ? 'ready' : 'created'
    persistRooms()
    sendToRenderer('room:update', room)
    deps.remoteServer().publish({ kind: 'room', data: room })
    return room
  }

  async function renameRoomOnThisHost(id: string, name: string): Promise<Room> {
    const room = requireActiveRoomOnThisHost(id)
    room.name = String(name ?? '').trim() || 'New room'
    room.updatedAt = Date.now()
    persistRooms()
    sendToRenderer('room:update', room)
    deps.remoteServer().publish({ kind: 'room', data: room })
    return room
  }

  async function steerRoomOnThisHost(id: string, text: string, scope: RemoteWorkspaceScope = {}): Promise<RoomEventItem> {
    const room = await assertRemoteRoomWorkspace(id, scope)
    let item: RoomEventItem
    try {
      item = appendRoomSteeringEvent(room, text, randomUUID)
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }
    room.updatedAt = Date.now()
    persistRooms()
    deps.emitRoom({ roomId: id, type: 'event', item })
    return item
  }

  async function mutateRoomMemoryOnThisHost(id: string, requestedCommand: RoomManualMemoryCommand): Promise<Room> {
    const room = requireActiveRoomOnThisHost(id)
    try {
      const command = pickRoomManualMemoryCommand(requestedCommand)
      assertRoomConfigUpdateAllowed(room, { memory: room.memory }, deps.roomAborters().has(id))
      room.memory = applyRoomManualMemoryCommand(room, command, randomUUID)
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }
    room.updatedAt = Date.now()
    persistRooms()
    deps.emitRoom({ roomId: id, type: 'memory', memory: room.memory ?? [], memorySummary: room.memorySummary ?? '' })
    return room
  }

  async function archiveRoomOnThisHost(id: string): Promise<Room | null> {
    const existing = store.rooms.find((candidate) => candidate.id === id)
    if (!existing) return null
    deps.roomAborters().get(id)?.abort()
    await deps.loopHost().stopLoopsForArchive('room', id)
    const ticketPool = store.tickets.filter((ticket) => ticket.projectId === existing.projectId)
    const roomTicketRoots = ticketPool.filter((ticket) => ticket.roomId === id).map((ticket) => ticket.id)
    const roomTickets = ticketArchiveClosure(ticketPool, roomTicketRoots)
    for (const ticket of roomTickets) {
      await deps.loopHost().stopLoopsForArchive('ticket', ticket.id)
      const thread = ticket.threadId ? store.threads.find((candidate) => candidate.id === ticket.threadId) : null
      if (thread?.projectId === ticket.projectId) {
        deps.aborters().get(thread.id)?.abort()
        await deps.abortAndWaitMissionsForArchive()({ threadId: thread.id })
      }
    }
    const archived = archiveRoomState(store, id)
    if (!archived) return null
    persistRooms()
    persistTickets()
    for (const ticket of archived.tickets) deps.emitTicket(ticket)
    sendToRenderer('room:update', archived.room)
    deps.remoteServer().publish({ kind: 'room', data: archived.room })
    return archived.room
  }

  async function restoreRoomOnThisHost(id: string): Promise<Room | null> {
    const room = store.rooms.find((candidate) => candidate.id === id)
    if (!room) return null
    requireActiveProjectOnThisHost(room.projectId)
    if (room.archivedAt != null) {
      const ticketPool = store.tickets.filter((ticket) => ticket.projectId === room.projectId)
      const roots = ticketPool.filter((ticket) => ticket.roomId === id).map((ticket) => ticket.id)
      const restoringTickets = ticketArchiveClosure(ticketPool, roots).filter((ticket) => ticket.archivedAt === room.archivedAt)
      const restoringTicketIds = new Set(restoringTickets.map((ticket) => ticket.id))
      for (const ticket of restoringTickets) validateTicketOwnershipOnThisHost(ticket.projectId, ticket.roomId, {
        selfId: ticket.id, parentId: ticket.parentId, threadId: ticket.threadId,
        restoringTicketIds, restoringRoomIds: new Set([id]),
      })
    }
    const restored = restoreRoomState(store, id)
    if (!restored) return null
    persistRooms(); persistTickets()
    for (const ticket of restored.tickets) deps.emitTicket(ticket)
    sendToRenderer('room:update', restored.room)
    deps.remoteServer().publish({ kind: 'room', data: restored.room })
    return restored.room
  }

  async function createTicketOnThisHost(requestedTicket: Partial<Ticket> & { projectId: string }): Promise<Ticket> {
    let input: Partial<Ticket> & { projectId: string }
    try {
      input = pickTicketCreateInput(requestedTicket)
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }
    validateTicketOwnershipOnThisHost(input.projectId, input.roomId, { parentId: input.parentId, threadId: input.threadId, dependsOn: input.dependsOn })
    const ticket = mkTicket(input.projectId, input)
    store.tickets.unshift(ticket)
    persistTickets()
    deps.emitTicket(ticket)
    return ticket
  }

  async function updateTicketOnThisHost(id: string, requestedPatch: Partial<Ticket>): Promise<Ticket> {
    const ticket = store.tickets.find((candidate) => candidate.id === id)
    if (!ticket) throw remoteMutationError('Ticket not found.', 404)
    if (ticket.archivedAt) throw remoteMutationError('Ticket is archived; restore it before updating it.')
    let patch: Partial<Ticket>
    try {
      patch = pickTicketUserPatch(requestedPatch)
      const activeThread = !!ticket.threadId && deps.aborters().has(ticket.threadId)
      assertTicketUserUpdateAllowed(ticket, patch, activeThread || deps.ticketWorkerRuntime().isTicketStarting(id))
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }
    const next = { ...ticket, ...patch }
    if (next.projectId !== ticket.projectId) throw remoteMutationError('Ticket project ownership is immutable.', 400)
    validateTicketOwnershipOnThisHost(next.projectId, next.roomId, { selfId: id, parentId: next.parentId, threadId: next.threadId, dependsOn: next.dependsOn })
    Object.assign(ticket, next, normalizeTicketExecution(next, { roomScoped: !!next.roomId }), {
      tags: normalizeStringList(next.tags, []),
      updatedAt: Date.now(),
    })
    persistTickets()
    deps.emitTicket(ticket)
    return ticket
  }

  async function transitionRemoteTicketOnThisHost(id: string, requestedCommand: TicketRemoteLifecycleCommand): Promise<Ticket> {
    const ticket = store.tickets.find((candidate) => candidate.id === id)
    if (!ticket) throw remoteMutationError('Ticket not found.', 404)
    if (ticket.archivedAt) throw remoteMutationError('Ticket is archived; restore it before changing its run.', 400)
    let command: TicketRemoteLifecycleCommand
    try {
      command = pickTicketRemoteLifecycleCommand(requestedCommand)
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 400)
    }

    const now = Date.now()
    if (command.action === 'begin') {
      throw remoteMutationError('Use the atomic remote ticket start endpoint; standalone begin is not allowed.', 400)
    } else {
      const execution = ticket.execution
      if (!execution || execution.target !== 'remote' || !execution.threadId || ticket.threadId !== execution.threadId) {
        throw remoteMutationError('Ticket has no valid remote execution binding.', 409)
      }
      const thread = requireActiveThreadOnThisHost(execution.threadId)
      if (thread.projectId !== ticket.projectId) throw remoteMutationError('The remote execution thread no longer belongs to the ticket project.', 409)

      if (command.action === 'sync') {
        if (ticket.runState === 'cancelled' || ticket.runState === 'done') return ticket
        if (thread.status === 'running' || deps.aborters().has(thread.id)) {
          ticket.status = 'in_progress'
          ticket.runState = 'running'
        } else {
          const evidence = ticketRunEvidence(thread.timeline, execution)
          if (!evidence.correlated) throw remoteMutationError('Remote execution is missing its host-owned run correlation.', 409)
          if (thread.status !== 'error' && !(thread.status === 'idle' && evidence.ran)) return ticket
          const failed = thread.status === 'error' || evidence.failed
          ticket.status = failed ? 'needs_clarification' : 'review'
          ticket.runState = 'done'
          ticket.execution = {
            ...execution,
            finishedAt: now,
            blockers: failed ? [...execution.blockers.filter((item) => !item.startsWith('remote run')), 'remote run ended with an error'] : execution.blockers,
            reviewVerdict: failed ? null : 'pending',
          }
        }
      } else if (command.action === 'start_failed') {
        if (ticket.runState !== 'queued' || deps.aborters().has(thread.id) || thread.status === 'running') throw remoteMutationError('Only a queued execution that failed to start can be marked failed.', 409)
        const message = typeof command.message === 'string' ? command.message.trim().slice(0, 500) : ''
        ticket.status = 'needs_clarification'
        ticket.runState = null
        ticket.execution = {
          ...execution,
          finishedAt: now,
          blockers: [...execution.blockers, message || 'remote run failed to start'],
          reviewVerdict: null,
        }
      } else if (command.action === 'cancel') {
        if (ticket.runState !== 'done' && ticket.runState !== 'cancelled') {
          ticket.status = 'needs_clarification'
          ticket.runState = 'cancelled'
          ticket.execution = {
            ...execution,
            finishedAt: now,
            blockers: [...execution.blockers.filter((item) => item !== 'remote run cancelled'), 'remote run cancelled'],
            reviewVerdict: null,
          }
          deps.cancelThreadRun(thread.id)
        }
      } else {
        throw remoteMutationError('Unsupported ticket lifecycle action.', 400)
      }
    }

    ticket.updatedAt = now
    persistTickets()
    deps.emitTicket(ticket)
    return ticket
  }

  async function startRemoteTicketExecutionOnThisHost(
    id: string,
    input: { hostId: string; userText: string },
    scope: RemoteWorkspaceScope = {},
  ): Promise<{ ticket: Ticket; thread: Thread; runId: string }> {
    const ticket = store.tickets.find((candidate) => candidate.id === id)
    if (!ticket) throw remoteMutationError('Ticket not found.', 404)
    if (ticket.archivedAt) throw remoteMutationError('Ticket is archived; restore it before starting work.', 400)
    const userText = typeof input?.userText === 'string' ? input.userText.trim() : ''
    if (!userText || userText.length > 120_000) throw remoteMutationError('A valid remote ticket prompt is required.', 400)
    const hostId = typeof input?.hostId === 'string' ? input.hostId.trim() : ''
    if (!hostId || hostId.length > 256) throw remoteMutationError('A valid host id is required.', 400)
    try {
      requireTicketDependenciesReady(ticket, store.tickets)
    } catch (error) {
      throw remoteMutationError(String((error as Error)?.message ?? error), 409)
    }

    const existingExecution = ticket.execution
    if ((ticket.runState === 'queued' || ticket.runState === 'running') && existingExecution?.target === 'remote' && existingExecution.threadId) {
      const existingThread = requireActiveThreadOnThisHost(existingExecution.threadId)
      if (existingThread.projectId !== ticket.projectId) throw remoteMutationError('The bound execution thread no longer belongs to the ticket project.', 409)
      if (ticket.runState === 'running' && existingExecution.runId) return { ticket, thread: existingThread, runId: existingExecution.runId }
      if (ticket.runState === 'queued' && existingThread.status === 'idle' && existingThread.timeline.length === (existingExecution.timelineBaseline ?? 0)) {
        const resumed = await startRemoteThreadOnThisHost({ threadId: existingThread.id, userText }, scope)
        ticket.runState = 'running'
        ticket.execution = { ...existingExecution, runId: resumed.runId }
        ticket.updatedAt = Date.now()
        persistTickets(); deps.emitTicket(ticket)
        return { ticket, thread: existingThread, runId: resumed.runId }
      }
      throw remoteMutationError('Ticket work is already active.', 409)
    }

    const project = requireActiveProjectOnThisHost(ticket.projectId)
    let thread: Thread | null = null
    try {
      thread = await createThreadOnThisHost({ cwd: project.cwd, kind: 'task', projectId: ticket.projectId }, scope)
      if (thread.timeline.length || thread.apiMessages.length || thread.status !== 'idle') throw remoteMutationError('A fresh empty thread is required for remote ticket execution.', 409)
      const startedAt = Date.now()
      ticket.threadId = thread.id
      ticket.status = 'in_progress'
      ticket.runState = 'queued'
      ticket.startedAt = startedAt
      ticket.assignee = 'remote'
      ticket.execution = {
        target: 'remote', hostId, providerId: null, model: null, branch: null, threadId: thread.id,
        runId: null, timelineBaseline: thread.timeline.length, startedAt, finishedAt: null,
        checks: [], blockers: [], reviewVerdict: null,
      }
      ticket.updatedAt = startedAt
      persistTickets(); deps.emitTicket(ticket)

      const started = await startRemoteThreadOnThisHost({ threadId: thread.id, userText }, scope)
      ticket.runState = 'running'
      ticket.execution = { ...ticket.execution, runId: started.runId }
      ticket.updatedAt = Date.now()
      persistTickets(); deps.emitTicket(ticket)
      return { ticket, thread, runId: started.runId }
    } catch (error) {
      const failedExecution = ticket.execution
      if (ticket.runState === 'queued' && failedExecution && thread && failedExecution.threadId === thread.id) {
        ticket.status = 'needs_clarification'
        ticket.runState = null
        ticket.execution = { ...failedExecution, finishedAt: Date.now(), blockers: [...failedExecution.blockers, `remote run failed to start: ${String((error as Error)?.message ?? error).slice(0, 400)}`] }
        ticket.updatedAt = Date.now()
        persistTickets(); deps.emitTicket(ticket)
      } else if (thread && !store.tickets.some((candidate) => candidate.threadId === thread!.id)) {
        await archiveThreadOnThisHost(thread.id).catch(() => null)
      }
      throw error
    }
  }

  async function archiveTicketOnThisHost(id: string): Promise<Ticket | null> {
    const existing = store.tickets.find((candidate) => candidate.id === id)
    if (!existing) return null
    const ticketClosure = ticketArchiveClosure(store.tickets, [id])
    for (const ticket of ticketClosure) {
      await deps.loopHost().stopLoopsForArchive('ticket', ticket.id)
      const thread = ticket.threadId ? store.threads.find((candidate) => candidate.id === ticket.threadId) : null
      if (thread?.projectId === ticket.projectId) {
        deps.aborters().get(thread.id)?.abort()
        await deps.abortAndWaitMissionsForArchive()({ threadId: thread.id })
      }
    }
    const archived = archiveTicketState(store, id)
    if (!archived) return null
    persistTickets()
    for (const ticket of archived.tickets) deps.emitTicket(ticket)
    return archived.ticket
  }

  async function restoreTicketOnThisHost(id: string): Promise<Ticket | null> {
    const existing = store.tickets.find((candidate) => candidate.id === id)
    if (!existing) return null
    const closure = ticketArchiveClosure(store.tickets, [id])
    const restoring = existing.archivedAt == null ? [] : closure.filter((ticket) => ticket.archivedAt === existing.archivedAt)
    const restoringTicketIds = new Set(restoring.map((ticket) => ticket.id))
    for (const ticket of restoring) {
      validateTicketOwnershipOnThisHost(ticket.projectId, ticket.roomId, {
        selfId: ticket.id,
        parentId: ticket.parentId,
        threadId: ticket.threadId,
        restoringTicketIds,
      })
    }
    const restored = restoreTicketState(store, id)
    if (!restored) return null
    persistTickets()
    for (const ticket of restored.tickets) deps.emitTicket(ticket)
    return restored.ticket
  }

  async function assertRemoteThreadWorkspace(threadId: string, scope: RemoteWorkspaceScope): Promise<Thread> {
    const thread = requireActiveThreadOnThisHost(threadId)
    const project = thread.projectId ? requireActiveProjectOnThisHost(thread.projectId) : null
    if (thread.kind !== 'chat' && (scope.confineToHome || thread.workspaceBoundary === 'home' || project?.workspaceBoundary === 'home')) {
      await resolvePairedWorkspace(thread.cwd, thread.workspace ?? project?.workspace)
    }
    return thread
  }

  async function startRemoteThreadOnThisHost(payload: StartRunPayload, scope: RemoteWorkspaceScope = {}): Promise<{ runId: string }> {
    await assertRemoteThreadWorkspace(payload.threadId, scope)
    return deps.startThreadRun({ ...payload, ...(scope.confineToHome ? { workspaceBoundary: 'home' as const } : {}) })
  }

  async function guideRemoteThreadOnThisHost(payload: CueRunPayload, scope: RemoteWorkspaceScope = {}): Promise<CueRunResult> {
    await assertRemoteThreadWorkspace(payload.threadId, scope)
    return deps.guideThreadRun(payload)
  }

  async function assertRemoteRoomWorkspace(roomId: string, scope: RemoteWorkspaceScope): Promise<Room> {
    const room = requireActiveRoomOnThisHost(roomId)
    const project = requireActiveProjectOnThisHost(room.projectId)
    if (scope.confineToHome || project.workspaceBoundary === 'home') {
      await resolvePairedWorkspace(project.cwd, project.workspace)
      for (const cwd of [room.rootCwd, room.workspaceCwd, room.scratchpadCwd]) {
        if (cwd) await resolvePairedWorkspace(cwd)
      }
    }
    return room
  }

  async function startRemoteRoomOnThisHost(roomId: string, problem: string, attachments?: Attachment[], scope: RemoteWorkspaceScope = {}): Promise<{ ok: boolean; room?: Room }> {
    await assertRemoteRoomWorkspace(roomId, scope)
    return deps.startRoomRun(roomId, problem, attachments)
  }

  async function cancelRemoteThreadOnThisHost(threadId: string): Promise<void> {
    requireActiveThreadOnThisHost(threadId)
    await deps.cancelThreadRun(threadId)
  }

  async function cancelRemoteRoomOnThisHost(roomId: string): Promise<void> {
    requireActiveRoomOnThisHost(roomId)
    await deps.cancelRoomRun(roomId)
  }

  return {
    listHostDirectories,
    createProjectOnThisHost,
    updateProjectOnThisHost,
    archiveProjectOnThisHost,
    restoreProjectOnThisHost,
    archiveThreadOnThisHost,
    restoreThreadOnThisHost,
    createThreadOnThisHost,
    createRoomOnThisHost,
    updateRoomOnThisHost,
    renameRoomOnThisHost,
    steerRoomOnThisHost,
    mutateRoomMemoryOnThisHost,
    archiveRoomOnThisHost,
    restoreRoomOnThisHost,
    createTicketOnThisHost,
    updateTicketOnThisHost,
    transitionRemoteTicketOnThisHost,
    startRemoteTicketExecutionOnThisHost,
    archiveTicketOnThisHost,
    restoreTicketOnThisHost,
    startRemoteThreadOnThisHost,
    guideRemoteThreadOnThisHost,
    startRemoteRoomOnThisHost,
    cancelRemoteThreadOnThisHost,
    cancelRemoteRoomOnThisHost,
  }
}

export interface RemoteHostApiDeps {
  sendToRenderer(channel: string, payload: unknown): void
  persistProjects(): void
  persistThreads(): void
  persistRooms(): void
  persistTickets(): void
  defaultRoomCollaborationMode: typeof DEFAULT_ROOM_COLLABORATION_MODE
  defaultRoomWorkerBudget: { maxActive: number; maxStarts: number }
  ensureProjectInstructionFiles(project: Project): Promise<void>
  // Late-bound collaborators (initialized later in the bootstrap): passed as accessors so the
  // factory can be constructed before they exist while lookups stay lazy.
  remoteServer(): { publish(event: { kind: string; data: unknown }): void }
  aborters(): Map<string, AbortController>
  roomAborters(): Map<string, AbortController>
  emitTicket(t: Ticket): void
  emitRoom(e: RoomStreamEvent): void
  abortAndWaitMissionsForArchive(): (target: { projectId?: string | null; threadId?: string | null }) => Promise<void>
  loopHost(): { stopLoopsForArchive(kind: 'project' | 'thread' | 'room' | 'ticket', id: string): Promise<unknown> }
  ticketWorkerRuntime(): { isTicketStarting(ticketId: string): boolean }
  startThreadRun(payload: StartRunPayload): Promise<{ runId: string }>
  guideThreadRun(payload: CueRunPayload): Promise<CueRunResult>
  cancelThreadRun(threadId: string): void
  startRoomRun(roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }>
  cancelRoomRun(roomId: string): void
}
