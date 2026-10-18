import { registerProjectMemoryIpc } from './project-memory-ipc'
import type { RoomSupervisorMetadataPatch } from '../../agent/jobs'
import type { runRoom as typedRunRoom } from '../../agent/rooms'
import { buildUserContent, contentText } from '../../agent/content'
import { normalizeAttachments } from '../../agent/attachment-policy'
import type {
  Attachment,
  Job,
  Project,
  ProjectMemoryCandidate,
  ProjectMemoryEntry,
  Provider,
  RemoteControlEvent,
  Room,
  RoomManualMemoryCommand,
  RoomMergeResult,
  RoomStreamEvent,
  Seat,
  Skill,
  Ticket,
  TicketWorkOptions,
  Thread,
} from '../../agent/types'
import type { store as typedStore } from '../state'
import type { handle as typedHandle } from './ipc'
import { roomModeDefaults } from '../../agent/room-outcomes'
import { resolveDelegationRoute } from '../../agent/delegation-route'
import { createSharedMemory, MEMORY_MAX_BLOCK_CHARS, MEMORY_MAX_BLOCKS, renderSharedMemory, writeBlock } from '../../agent/shared-room-memory'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, type WorkspaceMutationGuard, type WorkspaceMutationLease } from './workspace-mutation-guard'
import {
  applyRoomManualMemoryCommand,
  assertRoomConfigUpdateAllowed,
  assertRoomSeatRoutesConfigured,
  assertTicketUserUpdateAllowed,
  pickRoomConfigPatch,
  pickRoomManualMemoryCommand,
  pickTicketCreateInput,
  pickTicketUserPatch,
} from './room-ticket-patch-policy'
import { sanitizeProjectCreateInput, sanitizeProjectUpdatePatch, validateProjectExecutionTargetInput, validateProjectProviderDefaults } from './project-patch-policy'
import { assertUniqueConfigName, sanitizeSkillSave } from './public-config-policy'
import { scanGitHygiene } from '../../agent/git-hygiene'
import { inspectGitRepository } from './workspace'
import { requireNativeConfirmation } from './sensitive-confirmation'
import { localRoomMergeRunner, mergeRoomWorktreeBranches, roomMergeBranches, roomMergeConflictTicket, roomMergeGateBlockers } from './room-merge'
import { steerExistingRoom } from './room-steering-runtime'

export interface WorkspaceRuntimeIpc {
  ensureProjectInstructionFiles(project: Project): Promise<void>
  upgradeGeneratedClaudeGuidance(project: Project): Promise<boolean>
  cancelRoomRun(roomId: string): void
  startRoomRun(roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }>
}

export function reserveRoomStartup(
  roomId: string,
  roomAborters: Map<string, AbortController>,
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>,
  workspaceKeys: Iterable<string | null | undefined>,
): { controller: AbortController; runLease: WorkspaceMutationLease } | null {
  if (roomAborters.has(roomId)) return null
  const runLease = guard.reserveStart({ workspaceKeys })
  const controller = new AbortController()
  roomAborters.set(roomId, controller)
  return { controller, runLease }
}

export function assertLocalTicketGenerationProject(project: Pick<Project, 'workspace'>): void {
  if (project.workspace?.kind === 'ssh') {
    throw new Error('Ticket generation for SSH projects is unavailable until the planner can run on the remote workspace.')
  }
}

export function requireActiveRoomForCancel(rooms: readonly Room[], roomId: string): Room {
  const room = rooms.find((candidate) => candidate.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before cancelling it.')
  return room
}

export function assertRecordMetadataNotPatched(patch: object, additionalHostFields: readonly string[] = []): void {
  for (const field of ['id', 'createdAt', 'updatedAt', ...additionalHostFields]) {
    if (Object.prototype.hasOwnProperty.call(patch, field)) {
      throw new Error(`${field} is host-owned and cannot be changed through update endpoints.`)
    }
  }
}

type StartTicketWork = (input: {
  ticketId: string
  opts?: TicketWorkOptions
  source?: {
    roomId?: string | null
    seat?: string | null
    onSettled?: (summary: { codeChanged: boolean; testsPassed: boolean; failed: boolean; jobId?: string | null }) => void
  }
}) => Promise<Thread>

export interface WorkspaceRuntimeIpcDeps {
  store: typeof typedStore
  handle: typeof typedHandle
  runRoom: typeof typedRunRoom
  startTicketWork: StartTicketWork
  parkRoomIfWorkRemains(job: Job | null, room: Room, project: Project | null): boolean
  publishRemote?: (event: Pick<RemoteControlEvent, 'kind' | 'data'>) => void
  [key: string]: any
}

export function createWorkspaceRuntimeIpc(ctx: WorkspaceRuntimeIpcDeps): WorkspaceRuntimeIpc {
  const {
    fs,
    path,
    randomUUID,
    store,
    handle,
    spine,
    isActive,
    archiveEntity,
    killProjectTerminals,
    abortMissionsForArchive,
    stopLoopsForArchive,
    workspaceMutationGuard,
    requireSecurityRole,
    auditEvent,
    auditPreview,
    persistProjects,
    persistThreads,
    persistRooms,
    persistTickets,
    persistSubAgents,
    persistSkills,
    persistProjectMemory,
    persistProjectMemoryCandidates,
    aborters,
    roomAborters,
    DEFAULT_ROOM_COLLABORATION_MODE,
    DEFAULT_ROOM_WORKER_BUDGET,
    ensureRoomWorktree,
    emitRoom,
    emitTicket,
    emitJob,
    publishRemote,
    activeJob,
    patchRoomSupervisorJob,
    finishRoomSupervisorMission,
    finishJob,
    ensureRoomSupervisorMission,
    roomWorkerBudget,
    claimJob,
    patchJob,
    buildRoomSupervisorJobMetadata,
    ticketsForRoom,
    runtimeProvider,
    inferSeatLane,
    appendRoomSupervisorEvent,
    ticketNeedsReviewAfterBuilder,
    roomTicketConflictBlockers,
    summarizeRoomLeftToWork,
    describeTicketForRoom,
    mkTicket,
    assertTicketRelations,
    normalizeTicketCapabilities,
    normalizeStringList,
    ticketEvidenceFromArgs,
    ticketFileRefs,
    startTicketWork,
    parkRoomIfWorkRemains,
    createSubAgentFromLaunch,
    parentForRoomSeat,
    persistSubAgents: persistSubAgentsFromCtx,
    startSubAgent,
    runReadOnlySubAgent,
    executeTool,
    modelRouter,
    modelRouteAutoFailoverEnabled,
    effectiveContextLimit,
    roomContextLimit,
    auditScopedModelRoute,
    runtimeProviderWithFreshModels,
    assertProviderEndpoint,
    modelRouteFailoverText,
    roomRouteHandoff,
    modelRouteResetPolicy,
    roomFinalGateBlockers,
    runRoom,
    normalizeTicketExecution,
    ticketWorkOptionsWithProjectRoute,
    isTicketStarting,
    resolveProjectProvider,
    ensureTensorFoldStudioStylePrompt,
    streamChat,
    parseJsonLoose,
    createProjectMemoryCandidate,
    approveProjectMemoryCandidate,
    rejectProjectMemoryCandidate,
    projectMemoryCandidateDiff,
    rollbackProjectMemoryEntry,
    forgetProjectMemoryEntry,
    scopeProjectMemoryEntries,
    scopeProjectMemoryCandidates,
    registerWebhookIpc,
    webhookServer,
    rememberConnectorEvents,
    validateProjectCwd,
    confirmSensitiveAction,
  } = ctx
  const persistSubAgentsSafe = persistSubAgentsFromCtx ?? persistSubAgents

async function writeIfMissing(file: string, content: string): Promise<boolean> {
  try {
    await fs.access(file)
    return false
  } catch {
    await fs.writeFile(file, content, 'utf8')
    return true
  }
}

async function writeClaudeGuidance(file: string, content: string, createIfMissing = true): Promise<boolean> {
  try {
    const existing = await fs.readFile(file, 'utf8')
    const normalized = existing.trim()
    const oldGenerated = new Set([
      '# Claude guidance\n\nUse AGENTS.md as the shared project instruction source.',
      '@agents.md\n\n# Claude guidance\n\nClaude Code imports the shared project instructions above.',
      '@AGENTS.md\n\n# Claude guidance\n\nClaude Code imports the shared project instructions above.',
      '# Claude project context\n\n@AGENTS.md',
    ])
    if (oldGenerated.has(normalized)) {
      await fs.writeFile(file, content, 'utf8')
      return true
    }
    return false
  } catch {
    if (!createIfMissing) return false
    await fs.writeFile(file, content, 'utf8')
    return true
  }
}

function generatedClaudeGuidance(): string {
  return `@AGENTS.md

# Claude project context
`
}

async function withProjectInstructionLease<T>(project: Project, operation: () => Promise<T>): Promise<T> {
  if (!workspaceMutationGuard) throw new Error('Cannot write project guidance because the workspace mutation guard is unavailable.')
  const lease = workspaceMutationGuard.reserveStart({ workspaceKeys: [lexicalLocalWorkspaceKey(project.cwd)] })
  try {
    lease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(project.cwd)])
    return await operation()
  } finally {
    lease.release()
  }
}

async function upgradeGeneratedClaudeGuidance(project: Project): Promise<boolean> {
  if (!project.cwd || project.localOnly || project.workspace?.kind === 'ssh') return false
  return withProjectInstructionLease(project, () => writeClaudeGuidance(path.join(project.cwd, 'CLAUDE.md'), generatedClaudeGuidance(), false))
}

async function ensureProjectInstructionFiles(project: Project): Promise<void> {
  if (!project.cwd || project.workspace?.kind === 'ssh') return
  await withProjectInstructionLease(project, async () => {
    const agentsPath = path.join(project.cwd, 'AGENTS.md')
    const claudePath = path.join(project.cwd, 'CLAUDE.md')
    const agents = `# ${project.name} guidance

This project can contain solo operations, multi-agent rooms, and Kanban boards.
Treat sibling rooms, solo operations, board tickets, and project memory as shared context.
Do not assume another room has completed work unless there is evidence in the project history.
Keep comments concise and avoid bloated narration.
`
    const claude = generatedClaudeGuidance()
    const wroteAgents = await writeIfMissing(agentsPath, agents)
    const wroteClaude = project.localOnly ? false : await writeClaudeGuidance(claudePath, claude)
    if (wroteAgents || wroteClaude) await auditEvent('project.instructions.ensure', { projectId: project.id, agents: wroteAgents, claude: wroteClaude })
  })
}

handle('projects:get', async () => store.projects.filter(isActive))
handle('projects:create', async (_e, opts: Partial<Project> & { name?: string; cwd?: string }) => {
  requireSecurityRole('operator', 'create projects')
  const allowProxyIdentity = opts.executionTarget === 'orchestrator'
  const safe = sanitizeProjectCreateInput(opts, { allowId: allowProxyIdentity, allowCreatedAt: allowProxyIdentity })
  // When a cwd is supplied, validate it before the project is created and its
  // instruction files are written: must be an existing directory and not the home
  // folder (symlink-resolved). A project writes AGENTS.md/CLAUDE.md into its cwd,
  // so this is the authoritative guard against a model or caller pointing it at $HOME.
  const cwd = safe.cwd ? String(safe.cwd).trim() : ''
  const { executionTarget, orchestratorId } = validateProjectExecutionTargetInput(safe.executionTarget ?? 'local', safe.orchestratorId ?? null, store.settings.remoteOrchestrators ?? [])
  const resolvedCwd = cwd && executionTarget === 'local' && safe.workspace?.kind !== 'ssh' ? await validateProjectCwd(cwd) : cwd
  const now = Date.now()
  const p: Project = {
    id: safe.id || randomUUID(), name: safe.name || 'New project', cwd: resolvedCwd, memory: safe.memory ?? '', team: safe.team ?? '',
    workspace: safe.workspace ?? null,
    mode: safe.mode || 'solo', localOnly: safe.localOnly ?? false,
    executionTarget,
    orchestratorId,
    defaultProviderId: safe.defaultProviderId ?? null, defaultProviderAccountId: safe.defaultProviderAccountId ?? null, defaultModel: safe.defaultModel ?? null,
    kanban: safe.kanban ?? { modelMode: 'single' },
    tags: safe.tags ?? [],
    archivedAt: null,
    createdAt: safe.createdAt ?? now, updatedAt: now,
  }
  // A remote proxy was already validated against the remote host's provider catalog; the
  // controller may intentionally have a different catalog. Local projects validate here.
  if (executionTarget === 'local') validateProjectProviderDefaults(p, {
    localOnly: p.localOnly,
    defaultProviderId: p.defaultProviderId,
    defaultProviderAccountId: p.defaultProviderAccountId,
    defaultModel: p.defaultModel,
    kanban: p.kanban,
  }, store.settings.providers)
  const existing = safe.id ? store.projects.find((project) => project.id === safe.id) : null
  if (existing) {
    if (existing.archivedAt) throw new Error('Restore the archived project before updating it.')
    const sameWorkspace = JSON.stringify(existing.workspace ?? null) === JSON.stringify(p.workspace ?? null)
    if (existing.cwd !== p.cwd || !sameWorkspace || existing.executionTarget !== p.executionTarget || existing.orchestratorId !== p.orchestratorId) {
      throw new Error('An existing project cannot be relocated through projects:create; use a dedicated validated relocation operation.')
    }
    // Creation is idempotent for an existing identity. Treating this endpoint as an upsert would
    // let a caller bypass projects:update's strict config allowlist/provider validation simply by
    // replaying projects:create with the victim id. Configuration convergence must use the
    // dedicated validated update/routing endpoints.
    return existing
  }
  if (executionTarget === 'local') await ensureProjectInstructionFiles(p)
  store.projects.unshift(p); persistProjects(); publishRemote?.({ kind: 'project', data: p }); return p
})
handle('projects:update', async (_e, { id, patch }: { id: string; patch: Partial<Project> }) => {
  requireSecurityRole('operator', 'update projects')
  const p = store.projects.find((x) => x.id === id)
  if (!p || p.archivedAt) throw new Error('Restore the archived project before updating it.')
  const safePatch = sanitizeProjectUpdatePatch(patch)
  validateProjectProviderDefaults(p, safePatch, store.settings.providers)
  Object.assign(p, safePatch, { updatedAt: Date.now() }); persistProjects(); publishRemote?.({ kind: 'project', data: p })
  return p
})
handle('projects:setExecutionTarget', async (_e, input: { id: string; executionTarget?: Project['executionTarget']; orchestratorId?: string | null; configPatch?: Partial<Project> }) => {
  requireSecurityRole('operator', 'change project execution target')
  const project = store.projects.find((candidate) => candidate.id === String(input?.id ?? ''))
  if (!project || project.archivedAt) throw new Error('Restore the archived project before changing its execution target.')
  const { executionTarget, orchestratorId } = validateProjectExecutionTargetInput(input?.executionTarget, input?.orchestratorId, store.settings.remoteOrchestrators ?? [])
  const configPatch = sanitizeProjectUpdatePatch(input?.configPatch ?? {})
  if (executionTarget === 'local') validateProjectProviderDefaults(project, configPatch, store.settings.providers)
  Object.assign(project, configPatch, { executionTarget, orchestratorId })
  project.updatedAt = Date.now()
  persistProjects()
  publishRemote?.({ kind: 'project', data: project })
  await auditEvent('project.execution_target', { projectId: auditPreview(project.id), executionTarget, orchestratorId: orchestratorId ? auditPreview(orchestratorId) : null })
  return project
})
handle('projects:delete', async (_e, id: string) => {
  requireSecurityRole('operator', 'delete projects')
  await archiveEntity('project', id, {
    abortProjectTerminals: killProjectTerminals,
    abortMissions: abortMissionsForArchive,
    stopLoops: stopLoopsForArchive,
    abortThread: (threadId: string) => aborters?.get?.(threadId)?.abort(),
    abortRoom: (roomId: string) => roomAborters.get(roomId)?.abort(),
    publish: (kind: 'project' | 'thread' | 'room' | 'ticket', data: unknown) => publishRemote?.({ kind, data }),
  })
})
handle('projects:rename', async (_e, { id, name }: { id: string; name: string }) => {
  requireSecurityRole('operator', 'rename projects')
  const finalName = String(name ?? '').trim() || 'New project'
  const p = store.projects.find((x) => x.id === id)
  if (p?.archivedAt) throw new Error('Restore the archived project before renaming it.')
  if (p) { p.name = finalName; p.updatedAt = Date.now(); persistProjects(); publishRemote?.({ kind: 'project', data: p }) }
  await auditEvent('project.rename', { projectId: id, name: auditPreview(finalName) })
})

handle('rooms:get', async () => store.rooms.filter(isActive))
handle('rooms:create', async (_e, { projectId, name }: { projectId: string; name: string }) => {
  requireSecurityRole('operator', 'create rooms')
  const project = store.projects.find((candidate) => candidate.id === projectId)
  if (!project || project.archivedAt) throw new Error('Choose an active project before creating a room.')
  const r: Room = {
    id: randomUUID(), projectId, name: name || 'New room', seats: [],
    ...roomModeDefaults(),
    rootCwd: null, workspaceCwd: null, scratchpadCwd: null,
    collaborationMode: DEFAULT_ROOM_COLLABORATION_MODE,
    workerBudget: { ...DEFAULT_ROOM_WORKER_BUDGET },
    memory: [], memorySummary: '',
    problem: '', state: 'created', events: [], finalText: '', createdAt: Date.now(), updatedAt: Date.now(),
  }
  store.rooms.unshift(r); persistRooms(); publishRemote?.({ kind: 'room', data: r })
  spine?.()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'room', streamId: r.id, kind: 'room.create', actor: 'user' },
    () => ({ aggregate: 'room', streamId: r.id, type: 'room.created', payload: { projectId } }),
  )
  return r
})
handle('rooms:update', async (_e, { id, patch: requestedPatch }: { id: string; patch: Partial<Room> }) => {
  requireSecurityRole('operator', 'update rooms')
  const r = store.rooms.find((x) => x.id === id)
  if (!r || r.archivedAt) throw new Error('Restore the archived room before updating it.')
  const patch = pickRoomConfigPatch(requestedPatch)
  assertRoomConfigUpdateAllowed(r, patch, roomAborters.has(id))
  if (patch.seats) assertRoomSeatRoutesConfigured(patch.seats, store.settings.providers, store.projects.find((project) => project.id === r.projectId))
  {
    const modeBudget = patch.mode && patch.deliberationEffort === undefined ? roomModeDefaults(patch.mode).deliberationEffort : undefined
    Object.assign(r, patch, modeBudget ? { deliberationEffort: modeBudget } : {}, { updatedAt: Date.now() })
    if (patch.seats && ['idle', 'created', 'ready', 'degraded'].includes(r.state)) r.state = r.seats.length ? 'ready' : 'created'
    persistRooms()
    publishRemote?.({ kind: 'room', data: r })
  }
  return r
})
handle('rooms:delete', async (_e, id: string) => {
  requireSecurityRole('operator', 'delete rooms')
  await archiveEntity('room', id, {
    stopLoops: stopLoopsForArchive,
    abortMissions: abortMissionsForArchive,
    abortThread: (threadId: string) => aborters?.get?.(threadId)?.abort(),
    abortRoom: (roomId: string) => roomAborters.get(roomId)?.abort(),
    publish: (kind: 'project' | 'thread' | 'room' | 'ticket', data: unknown) => publishRemote?.({ kind, data }),
  })
  spine?.()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'room', streamId: id, kind: 'room.archive', actor: 'user' },
    () => ({ aggregate: 'room', streamId: id, type: 'room.archived', payload: {} }),
  )
})
handle('rooms:rename', async (_e, { id, name }: { id: string; name: string }) => {
  requireSecurityRole('operator', 'rename rooms')
  const finalName = String(name ?? '').trim() || 'New room'
  const r = store.rooms.find((x) => x.id === id)
  if (r?.archivedAt) throw new Error('Restore the archived room before renaming it.')
  if (r) { r.name = finalName; r.updatedAt = Date.now(); persistRooms(); publishRemote?.({ kind: 'room', data: r }) }
  await auditEvent('room.rename', { roomId: id, name: auditPreview(finalName) })
})
function cancelRoomRun(roomId: string): void {
  requireSecurityRole('operator', 'cancel rooms')
  const room = requireActiveRoomForCancel(store.rooms, roomId)
  roomAborters.get(roomId)?.abort()
  room.state = 'stopped'
  room.updatedAt = Date.now()
  persistRooms()
  emitRoom({ roomId, type: 'status', state: 'stopped' })
  const job = activeJob('room', roomId)
  if (job) patchRoomSupervisorJob(job, room, store.projects.find((project) => project.id === room.projectId) ?? null)
  finishRoomSupervisorMission(job, 'cancelled', 'Room cancelled by user.')
  finishJob(job, 'cancelled', 'Room cancelled by user.')
}

async function startRoomRun(roomId: string, problem: string, attachments?: Attachment[]): Promise<{ ok: boolean; room?: Room }> {
  requireSecurityRole('operator', 'start rooms')
  const room = store.rooms.find((r) => r.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before starting it.')
  // Don't start a second run for a room that's already running — it would overwrite the live
  // aborter, making both the first run uncancellable and the second un-stoppable.
  if (roomAborters.has(roomId)) return { ok: false, room }
  const project = store.projects.find((p) => p.id === room.projectId)
  if (!project) throw new Error('Project not found')
  if (project.archivedAt) throw new Error('Restore the archived project before starting this room.')
  // Rooms broadcast a single text problem to every seat (many are CLI/text-only), so fold
  // attachments into text: documents become labelled blocks; images are noted (vision-capable
  // seats can't receive pixels through the room channel).
  const safeAttachments = normalizeAttachments(attachments)
  const cleanProblem = contentText(buildUserContent(String(problem || '').trim(), safeAttachments)).trim()
  if (!cleanProblem) throw new Error('Add a room objective before starting.')
  if (room.seats.length < 1) throw new Error('Add at least one agent to the room first.')
  const configuredWorkspace = room.workspace ?? project.workspace ?? null
  if (configuredWorkspace?.kind === 'ssh') {
    throw new Error('Local room execution does not support SSH workspaces; run this room on its remote Orchestrator.')
  }
  const initialPaths = [configuredWorkspace?.path, room.workspaceCwd, room.rootCwd, room.scratchpadCwd, project.cwd]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
  const initialWorkspaceKeys = initialPaths.map((value: string) => lexicalLocalWorkspaceKey(value))
  const reservation = reserveRoomStartup(roomId, roomAborters, workspaceMutationGuard, initialWorkspaceKeys)
  if (!reservation) return { ok: false, room }
  const { controller, runLease } = reservation
  const timeout = setTimeout(() => controller.abort(), 30 * 60_000)
  roomAborters.set(roomId, controller)
  let startupComplete = false
  try {
  const roomWorkspace = await ensureRoomWorktree(project, room)
  const finalWorkspaceKey = await canonicalLocalWorkspaceKey(roomWorkspace)
  runLease.addWorkspaceKeys?.([finalWorkspaceKey])
  if (room.archivedAt || project.archivedAt) throw new Error('Room or project was archived while the room was starting.')
  if (room.workspaceCwd !== roomWorkspace) {
    room.workspaceCwd = roomWorkspace
    persistRooms()
  }
  const roomProject = { ...project, cwd: roomWorkspace }

  room.problem = cleanProblem
  room.events = []
  room.finalText = ''
  room.state = 'queued'
  room.updatedAt = Date.now()
  persistRooms()
  emitRoom({ roomId, type: 'status', state: 'queued' })
  const supervisorMission = ensureRoomSupervisorMission(room, project, roomWorkspace, cleanProblem)
  const supervisorRunId = `room:${room.id}:${Date.now()}`
  const configuredWorkerBudget = roomWorkerBudget(room)
  let job: Job | null = claimJob({
    kind: 'room',
    owner: room.id,
    projectId: room.projectId,
    missionId: supervisorMission.id,
    roomId: room.id,
    currentThreadId: null,
    metadata: buildRoomSupervisorJobMetadata({
      room,
      project,
      tickets: ticketsForRoom(room),
      patch: { missionId: supervisorMission.id, workerBudget: configuredWorkerBudget, workerStarts: 0 },
    }),
  })
  job = patchJob(job, { currentRunId: supervisorRunId })

  let lastCodeChangeAt = 0
  let greenEvidenceAt = 0
  const providerForSeat = (seat: Seat): Provider | undefined => {
    const provider = store.settings.providers.find((p) => p.id === seat.providerId)
    return provider ? runtimeProvider(provider, seat.providerAccountId) : undefined
  }
  if (roomProject.localOnly) {
    const cloudSeat = room.seats.find((seat) => providerForSeat(seat)?.kind !== 'local')
    if (cloudSeat) {
      room.state = 'failed'
      persistRooms()
      emitRoom({ roomId: room.id, type: 'error', message: `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`, fatal: true })
      emitRoom({ roomId: room.id, type: 'status', state: 'failed' })
      emitRoom({ roomId: room.id, type: 'done' })
      clearTimeout(timeout)
      roomAborters.delete(room.id)
      job = patchRoomSupervisorJob(job, room, project)
      finishRoomSupervisorMission(job, 'failed', `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`)
      finishJob(job, 'failed', `Project is local-only, but seat "${cloudSeat.name}" is not using a local provider.`)
      return { ok: false }
    }
  }
  const roomTickets = (): Ticket[] => ticketsForRoom(room)
  // Resolve a ticket from free-form LLM tool input. Precedence: exact id, then exact (trimmed,
  // case-insensitive) title equality, then a UNIQUE case-insensitive substring. A substring that
  // matches multiple tickets is AMBIGUOUS — never silently pick the first (store-insertion-order)
  // match, or claim/complete/start could act on the wrong ticket (e.g. "auth" hitting both
  // "Add auth" and "Add auth tests"). On ambiguity, `error` is set and `ticket` is undefined.
  const resolveTicket = (args: any): { ticket?: Ticket; error?: string } => {
    const board = roomTickets()
    const id = typeof args?.id === 'string' ? args.id : ''
    if (id) {
      const byId = board.find((t) => t.id === id)
      if (byId) return { ticket: byId }
    }
    const rawTitle = typeof args?.title === 'string' ? args.title.trim() : ''
    if (!rawTitle) return { ticket: undefined }
    const wanted = rawTitle.toLowerCase()
    const exact = board.filter((t) => t.title.trim().toLowerCase() === wanted)
    if (exact.length === 1) return { ticket: exact[0] }
    if (exact.length > 1) {
      return { error: `"${rawTitle}" matches ${exact.length} tickets with that exact title; reference the ticket by id instead.` }
    }
    const substr = board.filter((t) => t.title.toLowerCase().includes(wanted))
    if (substr.length === 1) return { ticket: substr[0] }
    if (substr.length > 1) {
      return { error: `"${rawTitle}" is ambiguous — it matches ${substr.length} tickets (${substr.map((t) => `"${t.title}"`).join(', ')}); use a more specific title or the ticket id.` }
    }
    return { ticket: undefined }
  }
  const findTicket = (args: any): Ticket | undefined => resolveTicket(args).ticket
  // Mirrors tickets.ts isReservationEvidence / meaningfulTicketEvidence (which are module-private
  // there): a "Reserved for build/review worker." note placed by start_ticket_work is a placeholder,
  // not real evidence, so it must not satisfy a ticket's required-evidence gate.
  const isReservationEvidenceLocal = (item: { kind?: string; summary?: string; ref?: string | null }): boolean =>
    /\breserved for (?:build|review) worker\b/i.test(`${item.kind ?? ''} ${item.summary ?? ''} ${item.ref ?? ''}`)
  const hasMeaningfulEvidence = (tk: Ticket): boolean =>
    (tk.evidence ?? []).some((item) => !isReservationEvidenceLocal(item))
  const workerBudget = configuredWorkerBudget
  let workerStarts = 0
  const activeTicketWorkerCount = () => roomTickets().filter((ticket) => ticket.runState === 'queued' || ticket.runState === 'running').length
  const touchSupervisor = (patch: RoomSupervisorMetadataPatch = {}) => {
    job = patchRoomSupervisorJob(job, room, project, { workerBudget, workerStarts, ...patch })
  }
  const selectAutomaticReviewSeat = (builderSeat: string): Seat | null => {
    const candidates = room.seats
      .map((seat, index) => {
        if (seat.name === builderSeat) return null
        if (!providerForSeat(seat)) return null
        const reviewRank = inferSeatLane(seat) === 'review' || seat.roleProfileId === 'reviewer' ? 0 : 1
        return { seat, index, reviewRank }
      })
      .filter((item): item is { seat: Seat; index: number; reviewRank: number } => !!item)
      .sort((a, b) => a.reviewRank - b.reviewRank || a.index - b.index)
    return candidates[0]?.seat ?? null
  }
  const blockAutomaticReview = (tk: Ticket, builderSeat: string, paths: string[], summary: string) => {
    touchSupervisor({ phase: { phase: 'review', status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat: builderSeat, paths, summary } })
    appendRoomSupervisorEvent(room, `Automatic review not started for "${tk.title}": ${summary}`, {
      tool: 'auto_review_scheduler',
      args: { ticketId: tk.id, phase: 'review', builderSeat, paths },
    })
  }
  const scheduleAutomaticReviewAfterBuild = async (tk: Ticket, builderSeat: string, paths: string[]) => {
    if (!ticketNeedsReviewAfterBuilder(tk)) return
    const reviewSeat = selectAutomaticReviewSeat(builderSeat)
    if (!reviewSeat) {
      blockAutomaticReview(tk, builderSeat, paths, 'No suitable different room seat is available for review.')
      return
    }
    const blockers: string[] = roomTicketConflictBlockers({ ticket: tk, tickets: roomTickets(), phase: 'review', seat: reviewSeat.name, paths })
    if (blockers.length) {
      blockAutomaticReview(tk, reviewSeat.name, paths, blockers.join(' '))
      return
    }
    const activeWorkers = activeTicketWorkerCount()
    if (activeWorkers >= workerBudget.maxActive) {
      blockAutomaticReview(tk, reviewSeat.name, paths, `Worker budget busy: ${activeWorkers}/${workerBudget.maxActive} active.`)
      return
    }
    if (workerStarts >= workerBudget.maxStarts) {
      blockAutomaticReview(tk, reviewSeat.name, paths, `Worker budget exhausted: ${workerStarts}/${workerBudget.maxStarts} started.`)
      return
    }
    appendRoomSupervisorEvent(room, `Automatic review queued for "${tk.title}" with ${reviewSeat.name}.`, {
      tool: 'auto_review_scheduler',
      args: { ticketId: tk.id, phase: 'review', builderSeat, reviewSeat: reviewSeat.name, paths },
    })
    try {
      await roomTools.start_ticket_work({ id: tk.id, phase: 'review', paths }, reviewSeat.name)
    } catch (err: any) {
      blockAutomaticReview(tk, reviewSeat.name, paths, String(err?.message ?? err))
    }
  }
  const emitRoomProgress = (ev: RoomStreamEvent) => {
    if (ev.type === 'status') {
      room.state = ev.state
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor()
    } else if (ev.type === 'turn') {
      touchSupervisor({ currentTurnSeat: ev.seat })
    } else if (ev.type === 'event') {
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor({ lastEventKind: ev.item.kind, lastEventSeat: ev.item.seat, lastEventAt: ev.item.createdAt })
    } else if (ev.type === 'memory') {
      room.updatedAt = Date.now()
      persistRooms()
      touchSupervisor()
    } else if (ev.type === 'final' || ev.type === 'done' || ev.type === 'error') {
      touchSupervisor()
    }
    emitRoom(ev)
  }
  touchSupervisor()
  const roomTools: Record<string, (args: any, seat: string) => Promise<string>> = {
    list_tickets: async () => {
      const board = roomTickets()
      return board.length ? board.map(describeTicketForRoom).join('\n') : 'The board is empty.'
    },
    left_to_work: async () => summarizeRoomLeftToWork(roomTickets()),
    create_ticket: async (args) => {
      const tk = mkTicket(room.projectId, {
        roomId: room.id,
        title: args?.title || 'Untitled',
        body: args?.body || '',
        priority: args?.priority,
        effort: args?.effort,
        type: args?.type || 'task',
        status: 'ready',
        tags: args?.tags,
        risk: args?.risk,
        allowedCapabilities: args?.allowedCapabilities,
        requiredEvidence: args?.requiredEvidence,
        acceptanceCriteria: args?.acceptanceCriteria,
        evidence: args?.evidence,
      })
      store.tickets.unshift(tk); persistTickets(); emitTicket(tk)
      return `Created ticket "${tk.title}" with ${tk.risk ?? 'low'} risk, allowed capabilities ${normalizeTicketCapabilities(tk.allowedCapabilities, ['read']).join(', ')}, and ${tk.requiredEvidence?.length ?? 0} required evidence item(s).`
    },
    claim_ticket: async (args, seat) => {
      const resolved = resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return `No ticket matches "${args?.title ?? args?.id ?? ''}".`
      if (tk.assignee && tk.assignee !== seat) return `Already claimed by ${tk.assignee}. Pick another ticket.`
      tk.assignee = seat; tk.status = 'in_progress'; tk.updatedAt = Date.now(); persistTickets(); emitTicket(tk)
      const required = normalizeStringList(tk.requiredEvidence, [])
      const caps = normalizeTicketCapabilities(tk.allowedCapabilities, ['read']).join(', ')
      return `You claimed "${tk.title}". Allowed capabilities: ${caps}. Required evidence: ${required.length ? required.join('; ') : 'none'}.`
    },
    add_ticket_evidence: async (args, seat) => {
      const tk = findTicket(args)
      if (!tk) return 'No matching ticket.'
      if (tk.assignee && tk.assignee !== seat) return `Only ${tk.assignee} can add evidence to "${tk.title}".`
      const evidence = ticketEvidenceFromArgs(args, seat)
      if (!evidence.length) return 'No evidence summary was provided.'
      tk.assignee = tk.assignee ?? seat
      tk.evidence = [...(tk.evidence ?? []), ...evidence]
      tk.updatedAt = Date.now()
      persistTickets(); emitTicket(tk)
      return `Added ${evidence.length} evidence item(s) to "${tk.title}".`
    },
    complete_ticket: async (args, seat) => {
      const resolved = resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return 'No matching ticket.'
      if (tk.assignee && tk.assignee !== seat) return `Only ${tk.assignee} can complete "${tk.title}".`
      const evidence = ticketEvidenceFromArgs(args, seat)
      if (evidence.length) tk.evidence = [...(tk.evidence ?? []), ...evidence]
      const required = normalizeStringList(tk.requiredEvidence, [])
      // Only MEANINGFUL evidence satisfies the required-evidence gate — reservation/placeholder notes
      // appended by start_ticket_work ("Reserved for build/review worker.") must not count, or a
      // ticket can be flipped to 'completed' with zero real evidence (the room final gate uses the
      // same meaningful filter, so this keeps the per-ticket gate consistent with it).
      if (required.length && !hasMeaningfulEvidence(tk)) return `Cannot complete "${tk.title}" until evidence is attached: ${required.join('; ')}.`
      tk.assignee = seat; tk.status = 'completed'; tk.updatedAt = Date.now(); persistTickets(); emitTicket(tk)
      return `Completed "${tk.title}" with ${tk.evidence?.length ?? 0} evidence item(s).`
    },
    start_ticket_work: async (args, seat) => {
      const resolved = resolveTicket(args)
      if (resolved.error) return resolved.error
      const tk = resolved.ticket
      if (!tk) return `No ticket matches "${args?.title ?? args?.id ?? ''}".`
      const phase = args?.phase === 'review' ? 'review' : 'build'
      const declaredPaths: string[] = normalizeStringList(args?.paths, [])
      const mergeDecision = typeof args?.mergeDecision === 'string' ? args.mergeDecision.trim() : ''
      const blockers: string[] = roomTicketConflictBlockers({ ticket: tk, tickets: roomTickets(), phase, seat, paths: declaredPaths })
      const overlapBlockers = blockers.filter((blocker) => /\boverlaps with\b/i.test(blocker))
      const hardBlockers = blockers.filter((blocker) => !overlapBlockers.includes(blocker))
      const effectiveBlockers = mergeDecision && overlapBlockers.length ? hardBlockers : blockers
      if (blockers.length) {
        if (effectiveBlockers.length) {
          touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: effectiveBlockers.join(' ') } })
          return `Blocked by room supervisor: ${effectiveBlockers.join(' ')}`
        }
        appendRoomSupervisorEvent(room, `Merge decision accepted for "${tk.title}": ${mergeDecision}`, {
          tool: 'merge_decision',
          args: { ticketId: tk.id, phase, seat, paths: declaredPaths, blockers: overlapBlockers },
        })
        tk.evidence = [
          ...(tk.evidence ?? []),
          {
            id: randomUUID(),
            kind: 'note',
            summary: `Supervisor merge decision: ${mergeDecision}. Waived overlap blocker(s): ${overlapBlockers.join(' ')}`,
            ref: declaredPaths[0] ?? null,
            createdBy: 'room-supervisor',
            createdAt: Date.now(),
          },
        ]
      }
      const activeWorkers = activeTicketWorkerCount()
      if (activeWorkers >= workerBudget.maxActive) {
        touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker budget busy: ${activeWorkers}/${workerBudget.maxActive} active.` } })
        return `Room worker budget busy: ${activeWorkers}/${workerBudget.maxActive} active ticket workers. Use left_to_work and wait for a worker result.`
      }
      if (workerStarts >= workerBudget.maxStarts) {
        touchSupervisor({ phase: { phase, status: 'blocked', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker budget exhausted: ${workerStarts}/${workerBudget.maxStarts} started.` } })
        return `Room worker budget exhausted: ${workerStarts}/${workerBudget.maxStarts} ticket workers started in this room run.`
      }
      const seatConfig = room.seats.find((candidate) => candidate.name === seat)
      if (!seatConfig) return 'Cannot start worker: room seat is no longer configured.'
      // Snapshot before the pre-launch mutation so a failed launch can restore the ticket
      // instead of wedging it (runState 'queued' blocks retries via roomTicketConflictBlockers,
      // counts against the worker budget, and keeps the final gate open).
      const prevAssignee = tk.assignee
      const prevStatus = tk.status
      workerStarts++
      // Only a BUILD-phase start takes ownership of the ticket. A review-phase start must NOT
      // overwrite the builder's assignee with the reviewer — doing so erases the builder's claim and
      // makes a later build-phase conflict check (roomTicketConflictBlockers) report the ticket as
      // claimed by the reviewer, blocking the original builder from their own ticket. The reviewer's
      // identity is tracked separately via the review evidence (createdBy), not tk.assignee.
      if (phase === 'build') tk.assignee = seat
      tk.status = 'in_progress'
      tk.runState = 'queued'
      if (declaredPaths.length) {
        tk.evidence = [
          ...(tk.evidence ?? []),
          ...declaredPaths.map((ref) => ({ id: randomUUID(), kind: 'file' as const, summary: `Reserved for ${phase} worker.`, ref, createdBy: seat, createdAt: Date.now() })),
        ]
      }
      tk.updatedAt = Date.now()
      persistTickets(); emitTicket(tk)
      touchSupervisor({ phase: { phase, status: 'queued', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: 'Ticket worker queued.' } })
      const knownPaths = ticketFileRefs(tk, declaredPaths)
      let workerThreadId: string | null = null
      let thread: Thread
      try {
        thread = await startTicketWork({
          ticketId: tk.id,
          opts: { phase, providerId: seatConfig.providerId, providerAccountId: seatConfig.providerAccountId ?? null, model: seatConfig.model },
          source: {
            roomId: room.id,
            seat,
            onSettled: ({ codeChanged, testsPassed, failed, jobId }) => {
              if (codeChanged) lastCodeChangeAt = Date.now()
              if (testsPassed) greenEvidenceAt = Date.now()
              touchSupervisor({
                phase: {
                  phase,
                  status: tk.status === 'needs_clarification' ? 'failed' : 'completed',
                  ticketId: tk.id,
                  ticketTitle: tk.title,
                  seat,
                  threadId: workerThreadId,
                  jobId: jobId ?? activeJob('ticket', tk.id)?.id ?? null,
                  paths: knownPaths,
                  summary: tk.status === 'needs_clarification' ? 'Ticket worker needs clarification.' : 'Ticket worker finished.',
                },
              })
              if (phase === 'build' && !failed) void scheduleAutomaticReviewAfterBuild(tk, seat, knownPaths)
            },
          },
        })
      } catch (launchError: any) {
        // Launch failed: unwind the pre-launch mutation so the ticket is retryable instead
        // of wedged as 'queued' (which blocks retries, burns budget, and holds the gate).
        workerStarts--
        tk.runState = null
        tk.status = prevStatus
        tk.assignee = prevAssignee
        tk.updatedAt = Date.now()
        persistTickets(); emitTicket(tk)
        touchSupervisor({ phase: { phase, status: 'failed', ticketId: tk.id, ticketTitle: tk.title, seat, paths: declaredPaths, summary: `Worker launch failed: ${launchError?.message ?? launchError}` } })
        return `Cannot start worker: launch failed: ${launchError?.message ?? launchError}`
      }
      workerThreadId = thread.id
      touchSupervisor({
        phase: {
          phase,
          status: 'running',
          ticketId: tk.id,
          ticketTitle: tk.title,
          seat,
          threadId: thread.id,
          jobId: activeJob('ticket', tk.id)?.id ?? null,
          paths: knownPaths,
          summary: 'Ticket worker running.',
        },
      })
      return `Room Supervisor started ${phase} worker thread ${thread.id} for "${tk.title}"${knownPaths.length ? ` on ${knownPaths.slice(0, 4).join(', ')}` : ''}.`
    },
    delegate: async (args, seat) => {
      const s = room.seats.find((x) => x.name === seat)
      if (!s) return 'Cannot delegate: seat misconfigured.'
      // Optional cross-model route: targetSeat adopts that seat's provider/account/model;
      // explicit providerId/accountId/model override it; absent fields inherit this seat.
      // A local-only project may not delegate out to a cloud provider (same guard the
      // thread runtime and room seat config enforce).
      const routed = resolveDelegationRoute(store.settings.providers, {
        providerId: s.providerId,
        providerAccountId: s.providerAccountId,
        model: s.model,
      }, args ?? {}, room.seats, roomProject.localOnly === true)
      if (!routed.route) return `Cannot delegate: ${routed.error}.`
      const routeAccountId = routed.route.providerAccountId ?? null
      const routeProvider = store.settings.providers.find((p) => p.id === routed.route.providerId)
      const prov = routeProvider ? runtimeProvider(routeProvider, routed.route.providerAccountId) : undefined
      if (!prov) return s.providerId === routed.route.providerId ? 'Cannot delegate: seat misconfigured.' : `Cannot delegate: provider "${routed.route.providerId}" is not configured.`
      const task = String(args?.task || '').trim() || 'Focused room subtask'
      const name = typeof args?.name === 'string' ? args.name.trim() : ''
      const runId = `subagent-${randomUUID()}`
      const created = createSubAgentFromLaunch(store.subAgents, {
        source: 'room_delegate',
        parent: parentForRoomSeat(room, s),
        task,
        name,
        providerId: routed.route.providerId,
        providerAccountId: routeAccountId,
        model: routed.route.model,
        runId,
        cwd: roomWorkspace,
      })
      store.subAgents = created.agents
      persistSubAgentsSafe()
      if (args && typeof args === 'object') {
        args.id = created.agent.id
        args.name = created.agent.name
        args.task = task
        args.parentSeat = seat
        args.providerId = routed.route.providerId
        args.providerAccountId = routeAccountId
        args.model = routed.route.model
        if (routed.notice) args.notice = routed.notice
        args.status = 'running'
      }
      const started = startSubAgent(store.subAgents, created.agent.id, runId)
      store.subAgents = started.agents
      persistSubAgentsSafe()
      const launched = await runReadOnlySubAgent(created.agent, prov, routed.route.model, {
        task,
        parentLabel: `room seat "${seat}" in room "${room.name}"`,
        cwd: roomWorkspace,
        projectId: room.projectId,
        providerAccountId: routeAccountId,
        signal: controller.signal,
        project: roomProject,
        maxSteps: 10,
        maxWallMs: 150_000,
      })
      if (args && typeof args === 'object') {
        args.status = launched.agent.status
        args.result = launched.agent.result
        args.error = launched.agent.error
      }
      return routed.notice ? `${routed.notice}\n${launched.output}` : launched.output
    },
    // Room tools are read-only by default. Mutating tools need a future room-level approval UI.
    // memory_write is room state (the shared scratchpad), not workspace mutation, so it is allowed.
    memory_write: async (args, seat) => {
      const name = typeof args?.name === 'string' ? args.name.trim() : ''
      const content = typeof args?.content === 'string' ? args.content.trim() : ''
      if (!name || !content) return 'memory_write needs a "name" (decisions, plan, blockers, or any label) and "content".'
      if (!room.sharedMemory) room.sharedMemory = createSharedMemory()
      const mode = args?.mode === 'append' || args?.mode === 'append-bullet' ? args.mode : 'replace'
      const expectedRevision = typeof args?.expectedRevision === 'number' ? args.expectedRevision : undefined
      const result = writeBlock(room.sharedMemory, { name, content, author: seat, mode, expectedRevision })
      if (!result.ok) {
        if (result.reason === 'conflict') return `Memory conflict on "${name}": current revision is ${result.currentRevision}. Re-read the block and retry.`
        if (result.reason === 'content-too-large') return `Memory block "${name}" is too large: content is capped at ${MEMORY_MAX_BLOCK_CHARS} characters. Summarize it and retry.`
        if (result.reason === 'block-limit') return `Shared memory is full (${MEMORY_MAX_BLOCKS} blocks). Update an existing block instead of adding a new one.`
        return 'Memory write needs a non-empty name.'
      }
      room.updatedAt = Date.now()
      persistRooms()
      // Live push so the renderer reflects the write without waiting for a full snapshot.
      if (result.changed) publishRemote?.({ kind: 'room', data: room })
      return result.changed
        ? `Memory "${result.block.name}" updated (rev ${result.block.revision}).\n${renderSharedMemory(room.sharedMemory)}`
        : `Memory "${result.block.name}" already says that (rev ${result.block.revision}).`
    },
    read_file: async (args) => { const r = await executeTool('read_file', args, { cwd: roomWorkspace }); return r.output.slice(0, 4000) },
    list_dir: async (args) => { const r = await executeTool('list_dir', args, { cwd: roomWorkspace }); return r.output.slice(0, 2000) },
    search: async (args) => { const r = await executeTool('search', args, { cwd: roomWorkspace }); return r.output.slice(0, 2000) },
    write_file: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: room.id, tool: 'write_file', path: auditPreview(args?.path) })
      return 'Blocked: room agents are read-only until a room-level approval gate is available. Create/claim a ticket and run the work in an approved thread.'
    },
    edit_file: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: room.id, tool: 'edit_file', path: auditPreview(args?.path) })
      return 'Blocked: room agents are read-only until a room-level approval gate is available. Create/claim a ticket and run the work in an approved thread.'
    },
    run_command: async (args) => {
      await auditEvent('room.tool_blocked', { roomId: room.id, tool: 'run_command', command: auditPreview(args?.command) })
      return 'Blocked: room agents cannot run shell commands unattended. Ask the user to run an approved thread or terminal command.'
    },
  }

  // Supervisor mode: an ASSIGN auto-starts the addressed seat's ticket worker.
  // The ticket is resolved from the ASSIGN text (a verbatim ticket title inside the
  // text wins; otherwise the text minus the seat name is used as a title query), then
  // routed through the standard start_ticket_work path so budget/conflict gates apply.
  const autoStartAssignedTicket = async ({ target, text }: { seat: string; target: string; text: string }): Promise<string | null> => {
    const board = roomTickets().filter((t) => t.status !== 'completed')
    const hay = text.toLowerCase()
    const inText = board.filter((t) => t.title.trim() && hay.includes(t.title.trim().toLowerCase()))
    let tk = inText.length === 1 ? inText[0] : undefined
    if (!tk && inText.length > 1) {
      const matches = inText.slice(0, 4).map((t) => `"${t.title}"`).join(', ')
      return `ASSIGN auto-start skipped: the text matches ${inText.length} open tickets (${matches}). Name one verbatim to start it.`
    }
    if (!tk && inText.length === 0) {
      const remainder = text.replace(new RegExp(`\\b${target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'ig'), '').trim()
      const resolved = resolveTicket({ title: remainder })
      if (resolved.error) return `ASSIGN auto-start skipped: ${resolved.error}`
      tk = resolved.ticket
    }
    if (!tk) return `ASSIGN to ${target} noted, but no open ticket title matched. Create the ticket with create_ticket first, then assign it.`
    if (tk.assignee && tk.assignee !== target) return `ASSIGN auto-start skipped: "${tk.title}" is already claimed by ${tk.assignee}.`
    return roomTools.start_ticket_work({ id: tk.id, phase: 'build' }, target)
  }

  runRoom({
    room, project: roomProject, providers: store.settings.providers, problem: cleanProblem,
    temperature: store.settings.temperature, reasoningEffort: store.settings.reasoningEffort,
    emit: emitRoomProgress, signal: controller.signal, tools: roomTools,
    onAssign: autoStartAssignedTicket,
    providerForSeat,
    recoverSeatModel: async (seat, error) => {
      const latestBaseProvider = store.settings.providers.find((p) => p.id === seat.providerId)
      if (!latestBaseProvider) return null
      const currentAccountId = seat.providerAccountId ?? null
      const routed = modelRouter.routeAfterError({
        role: 'room_seat',
        provider: latestBaseProvider,
        accountId: currentAccountId,
        model: seat.model,
        error,
        allowFailover: modelRouteAutoFailoverEnabled(latestBaseProvider, currentAccountId),
        localOnly: roomProject.localOnly === true,
        contextLimit: effectiveContextLimit(latestBaseProvider, seat.model),
        reasoningEffort: store.settings.reasoningEffort,
        budget: room.workerBudget?.maxActive ?? null,
        source: 'room_seat_recovery',
      })
      if (routed.signal) {
        for (const audit of routed.audit) {
          await auditScopedModelRoute('room', audit, routed.signal, {
            roomId: room.id,
            seatId: seat.id,
            seatName: seat.name,
            jobId: job?.id ?? null,
            missionId: supervisorMission.id,
          })
        }
      }
      if (!routed.signal || !routed.route.selected || routed.route.selectedAccountId === currentAccountId) return null
      const providerAccountId = routed.route.selectedAccountId
      const provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
      assertProviderEndpoint(provider)
      const liveSeat = room.seats.find((candidate) => candidate.id === seat.id)
      if (liveSeat) {
        liveSeat.providerId = latestBaseProvider.id
        liveSeat.providerAccountId = providerAccountId
        liveSeat.model = seat.model
      }
      room.updatedAt = Date.now()
      persistRooms()
      const failoverText = modelRouteFailoverText(latestBaseProvider, routed.route.audit, routed.signal)
      const notice = `${failoverText}\n${roomRouteHandoff(room, seat, failoverText)}`
      job = patchJob(job, {
        metadata: {
          modelRoute: {
            action: 'room_failover',
            seatId: seat.id,
            seatName: seat.name,
            providerId: latestBaseProvider.id,
            model: seat.model,
            preferredProviderAccountId: currentAccountId,
            failoverProviderAccountId: providerAccountId,
            resetPolicy: modelRouteResetPolicy(latestBaseProvider, currentAccountId),
            resetAt: routed.signal.resetAt,
            reason: routed.route.audit.reason,
          },
        },
      })
      touchSupervisor()
      return { provider, providerAccountId, model: seat.model, notice }
    },
    contextLimitForSeat: (seat) => {
      if (!store.settings.autoCompact) return 0
      const provider = providerForSeat(seat)
      if (!provider) return 0
      // Task #41: in a mixed-model room every seat compacts to the lowest seat's
      // limit so no model silently overflows. Fall back to this seat's own limit
      // if the floor can't be computed (e.g. another seat is misconfigured).
      const floor = roomContextLimit(
        room.seats
          .map((s) => ({ provider: providerForSeat(s), model: s.model }))
          .filter((entry): entry is { provider: Provider; model: string } => !!entry.provider),
        {
          cloudLimit: store.settings.contextLimit,
          localLimit: store.settings.localContextLimit,
          override: store.settings.contextLimitOverride,
          engineModelProfiles: store.settings.engineModelProfiles,
        },
      )
      return floor > 0 ? floor : effectiveContextLimit(provider, seat.model)
    },
    finalGate: () => {
      return roomFinalGateBlockers({ tickets: roomTickets(), lastCodeChangeAt, greenEvidenceAt })
    },
  })
    .catch((err) => emitRoomProgress({ roomId, type: 'error', message: String(err?.message ?? err), fatal: true }))
    .finally(() => {
      clearTimeout(timeout)
      roomAborters.delete(roomId)
      runLease.release()
      const failed = room.state === 'failed' || room.state === 'error'
      const parked = !controller.signal.aborted && !failed && parkRoomIfWorkRemains(job, room, project)
      if (!parked) {
        touchSupervisor({ currentTurnSeat: null })
        const outcome = controller.signal.aborted ? 'paused' : failed ? 'failed' : 'completed'
        const reason = failed ? 'Room ended with an error.' : controller.signal.aborted ? 'Room paused after the active run stopped.' : undefined
        finishRoomSupervisorMission(job, outcome, reason)
        finishJob(job, outcome, reason)
      }
      room.updatedAt = Date.now()
      persistRooms()
    })

  startupComplete = true
  return { ok: true, room }
  } finally {
    if (!startupComplete) {
      clearTimeout(timeout)
      if (roomAborters.get(roomId) === controller) roomAborters.delete(roomId)
      runLease.release()
    }
  }
}

handle('room:cancel', async (_e, roomId: string) => cancelRoomRun(roomId))
handle('room:start', async (_e, { roomId, problem, attachments }: { roomId: string; problem: string; attachments?: Attachment[] }) => startRoomRun(roomId, problem, attachments))
handle('room:steer', async (_e, { roomId, text }: { roomId: string; text: string }) => {
  return steerExistingRoom({ rooms: () => store.rooms, requireOperator: () => requireSecurityRole('operator', 'steer rooms'),
    createId: randomUUID, persistRooms, emitRoom, hasActiveRoom: (id) => roomAborters.has(id) }, roomId, text)
})
handle('room:manualMemory', async (_e, { roomId, command: requestedCommand }: { roomId: string; command: RoomManualMemoryCommand }) => {
  requireSecurityRole('operator', 'change room memory')
  const room = store.rooms.find((candidate) => candidate.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before changing its memory.')
  const command = pickRoomManualMemoryCommand(requestedCommand)
  assertRoomConfigUpdateAllowed(room, { memory: room.memory }, roomAborters.has(roomId))
  room.memory = applyRoomManualMemoryCommand(room, command, randomUUID)
  room.updatedAt = Date.now()
  persistRooms()
  emitRoom({ roomId, type: 'memory', memory: room.memory, memorySummary: room.memorySummary ?? '' })
  return room
})

handle('room:mergeWorktree', async (_e, { roomId }: { roomId: string }): Promise<RoomMergeResult> => {
  requireSecurityRole('operator', 'merge room worktrees')
  const room = store.rooms.find((candidate) => candidate.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before merging its worktree.')
  if (room.orchestratorHostId) throw new Error('Merge this room on its Orchestrator host; the worktree lives there.')
  if (roomAborters.has(room.id)) throw new Error('Stop the running room before merging its worktree.')
  const project = store.projects.find((candidate) => candidate.id === room.projectId)
  if (!project || project.archivedAt) throw new Error('Restore or choose an active project before merging room work.')
  if (project.workspace?.kind === 'ssh') throw new Error('Room worktree merge is unavailable for SSH workspaces.')
  const roomTickets = store.tickets.filter((ticket) => ticket.roomId === room.id && !ticket.archivedAt)
  const gateBlockers = roomMergeGateBlockers(room, roomTickets)
  if (gateBlockers.length) {
    await auditEvent('room.merge_gated', { roomId: room.id, blockers: gateBlockers.length })
    return { ok: false, status: 'gated', baseBranch: null, merged: [], remaining: [], conflict: null, blockers: gateBlockers, message: 'Final gates are not green yet.' }
  }
  const repo = await inspectGitRepository(project.cwd, 'project workspace')
  if (!repo.exists || !repo.topLevel) throw new Error('This project folder is not a Git repo; room worktrees cannot merge back.')
  if (!repo.branch) throw new Error('The project repo is on a detached HEAD; check out a branch before merging room work.')
  // Same audited preflight as workspace:commit, run on the room worktree before merging.
  const hygieneTarget = room.workspaceCwd || repo.topLevel
  const hygiene = await scanGitHygiene(hygieneTarget)
  await auditEvent('room.merge_preflight', { roomId: room.id, cwd: auditPreview(hygieneTarget), blocked: hygiene.blocked, critical: hygiene.summary.critical, high: hygiene.summary.high, medium: hygiene.summary.medium })
  if (hygiene.blocked) {
    const total = hygiene.summary.critical + hygiene.summary.high + hygiene.summary.medium
    return { ok: false, status: 'blocked', baseBranch: repo.branch, merged: [], remaining: [], conflict: null, blockers: [], message: `Git hygiene blocked merge: ${total} blocking finding${total === 1 ? '' : 's'} in the room worktree. Review and sanitize before merging.` }
  }
  const branches = roomMergeBranches(room, roomTickets)
  await requireNativeConfirmation(confirmSensitiveAction, {
    title: 'Merge room worktree?',
    message: `Merge ${branches.length} room branch${branches.length === 1 ? '' : 'es'} into ${repo.branch}?`,
    detail: `Branches: ${branches.join(', ')}. The room worktree stays in place and nothing is pushed.`,
    confirmLabel: 'Merge',
  })
  const result = await mergeRoomWorktreeBranches(localRoomMergeRunner, repo.topLevel, repo.branch, branches)
  if (result.status === 'conflict' && result.conflict) {
    const ticket = mkTicket(room.projectId, roomMergeConflictTicket(room, roomTickets, result.conflict.branch, result.conflict.files, result.merged))
    store.tickets.unshift(ticket); persistTickets(); emitTicket(ticket)
    result.conflict.ticketId = ticket.id
    await auditEvent('room.merge_conflict', { roomId: room.id, branch: result.conflict.branch, files: result.conflict.files.slice(0, 50), ticketId: ticket.id })
  } else {
    await auditEvent('room.merge', { roomId: room.id, baseBranch: repo.branch, merged: result.merged, status: result.status, message: auditPreview(result.message) })
  }
  return result
})

// ---- Kanban tickets ----
handle('tickets:get', async () => store.tickets.filter(isActive))
handle('tickets:create', async (_e, requestedTicket: Partial<Ticket> & { projectId: string }) => {
  requireSecurityRole('operator', 'create tickets')
  const t = pickTicketCreateInput(requestedTicket)
  assertTicketRelations(store, t.projectId, t.roomId, { parentId: t.parentId, threadId: t.threadId, dependsOn: t.dependsOn })
  const ticket = mkTicket(t.projectId, t)
  store.tickets.unshift(ticket); persistTickets(); emitTicket(ticket)
  spine?.()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'ticket', streamId: ticket.id, kind: 'ticket.create', actor: 'user' },
    () => ({ aggregate: 'ticket', streamId: ticket.id, type: 'ticket.created', payload: { projectId: t.projectId } }),
  )
  return ticket
})
handle('tickets:update', async (_e, { id, patch: requestedPatch }: { id: string; patch: Partial<Ticket> }) => {
  requireSecurityRole('operator', 'update tickets')
  const t = store.tickets.find((x) => x.id === id)
  if (!t || t.archivedAt) throw new Error('Restore the archived ticket before updating it.')
  const patch = pickTicketUserPatch(requestedPatch)
  const activeThread = !!t.threadId && aborters?.has?.(t.threadId)
  assertTicketUserUpdateAllowed(t, patch, activeThread || !!isTicketStarting?.(id))
  {
    const next = { ...t, ...patch }
    if (next.projectId !== t.projectId) throw new Error('Ticket project ownership is immutable.')
    assertTicketRelations(store, next.projectId, next.roomId, { selfId: t.id, parentId: next.parentId, threadId: next.threadId, dependsOn: next.dependsOn })
    Object.assign(t, next, normalizeTicketExecution(next, { roomScoped: !!next.roomId }), {
      tags: normalizeStringList(next.tags, []),
      updatedAt: Date.now(),
    })
    persistTickets()
    emitTicket(t)
  }
  return t
})
handle('tickets:delete', async (_e, id: string) => {
  requireSecurityRole('operator', 'delete tickets')
  await archiveEntity('ticket', id, {
    stopLoops: stopLoopsForArchive,
    abortMissions: abortMissionsForArchive,
    abortThread: (threadId: string) => aborters?.get?.(threadId)?.abort(),
    publish: (kind: 'project' | 'thread' | 'room' | 'ticket', data: unknown) => publishRemote?.({ kind, data }),
  })
  spine?.()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'ticket', streamId: id, kind: 'ticket.archive', actor: 'user' },
    () => ({ aggregate: 'ticket', streamId: id, type: 'ticket.archived', payload: {} }),
  )
})
handle('tickets:generate', async (_e, { projectId, idea, opts }: { projectId: string; idea: string; opts?: TicketWorkOptions }) => {
  requireSecurityRole('operator', 'generate tickets')
  const project = store.projects.find((p) => p.id === projectId)
  if (!project || project.archivedAt) throw new Error('Choose an active project before generating tickets.')
  assertLocalTicketGenerationProject(project)
  const routeOpts = ticketWorkOptionsWithProjectRoute(project, opts, 'kanban_draft')
  const { provider, model } = resolveProjectProvider(project, routeOpts?.providerAccountId, routeOpts?.model, routeOpts?.providerId)
  const sys = ensureTensorFoldStudioStylePrompt('You are a senior tech lead planning work on a Kanban board. Decompose the user\'s idea into ONE epic and 3-6 concrete, actionable subtickets. Reply with ONLY minified JSON, no prose and no code fences: {"epic":{"title":"...","body":"...","tags":["..."]},"subtickets":[{"title":"...","body":"...","priority":"low|medium|high|urgent","effort":"small|medium|large","tags":["..."]}]}')
  const ctx = project?.memory ? `Project context:\n${project.memory}\n\n` : ''
  let out = ''
  for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages: [{ role: 'system', content: sys }, { role: 'user', content: ctx + idea }], temperature: 0.4, approvalMode: 'readonly', cwd: project?.cwd || store.settings.defaultCwd, signal: new AbortController().signal })) {
    if (ch.type === 'content') out += ch.text
  }
  const json = parseJsonLoose(out)
  if (!json?.epic) throw new Error('The model did not return parseable tickets.')
  const created: Ticket[] = []
  const epic = mkTicket(projectId, { title: json.epic.title, body: json.epic.body || '', type: 'epic', status: 'todo', tags: json.epic.tags || [] })
  store.tickets.unshift(epic); created.push(epic)
  for (const s of (json.subtickets || [])) {
    const sub = mkTicket(projectId, { title: s.title, body: s.body || '', type: 'subticket', status: 'ready', priority: s.priority, effort: s.effort, tags: s.tags || [], parentId: epic.id })
    store.tickets.unshift(sub); created.push(sub)
  }
  persistTickets()
  return created
})

handle('ticket:work', async (_e, input: string | { ticketId: string; opts?: TicketWorkOptions }) => {
  const ticketId = typeof input === 'string' ? input : input.ticketId
  const opts = typeof input === 'string' ? undefined : input.opts
  return startTicketWork({ ticketId, opts })
})

// ---------------------------------------------------------------- skills & commands
handle('skills:get', async () => store.skills)
handle('skills:save', async (_e, requested: Partial<Skill>) => {
  requireSecurityRole('operator', 'save skills')
  const s = sanitizeSkillSave(requested)
  if (s.id) {
    const ex = store.skills.find((x) => x.id === s.id)
    if (!ex) throw new Error('Skill not found.')
    if (ex.scope === 'project' || ex.projectId) throw new Error('Project skills are repo-owned and must be edited in their source file.')
    if (s.name !== undefined) assertUniqueConfigName(store.skills, s.name, ex.id, 'Skill')
    const { id: _id, ...patch } = s
    Object.assign(ex, patch, { updatedAt: Date.now() })
    persistSkills()
    return ex
  }
  const name = s.name ?? 'skill'
  assertUniqueConfigName(store.skills, name, undefined, 'Skill')
  const now = Date.now()
  const skill: Skill = { id: 's_' + Math.random().toString(36).slice(2, 9), name, description: s.description ?? '', body: s.body ?? '', enabled: s.enabled ?? true, ...(s.trigger ? { trigger: s.trigger } : {}), scope: 'global', projectId: null, createdAt: now, updatedAt: now }
  store.skills.unshift(skill); persistSkills(); return skill
})
handle('skills:delete', async (_e, id: string) => {
  requireSecurityRole('operator', 'delete skills')
  store.skills = store.skills.filter((s) => s.id !== id); persistSkills()
})

// Project memory writes commit protected entries and review history atomically.
registerProjectMemoryIpc({ store, handle, requireSecurityRole, auditEvent, auditPreview })

// ---------------------------------------------------------------- webhooks
registerWebhookIpc({ webhookServer, rememberConnectorEvents })
  return { ensureProjectInstructionFiles, upgradeGeneratedClaudeGuidance, cancelRoomRun, startRoomRun }
}
