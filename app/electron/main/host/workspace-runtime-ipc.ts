import { createRoomRunRuntime } from './room-run-runtime'
import { reserveRoomStartup, assertLocalTicketGenerationProject, requireActiveRoomForCancel, assertRecordMetadataNotPatched } from './workspace-startup-policy'
export { reserveRoomStartup, assertLocalTicketGenerationProject, requireActiveRoomForCancel, assertRecordMetadataNotPatched } from './workspace-startup-policy'
import { registerProjectMemoryIpc } from './project-memory-ipc'

import type { runRoom as typedRunRoom } from '../../agent/rooms'

import type { Attachment, Job, Project, RemoteControlEvent, Room, RoomManualMemoryCommand, RoomMergeResult, Skill, Ticket, TicketWorkOptions, Thread } from '../../agent/types'
import type { store as typedStore } from '../state'
import type { handle as typedHandle } from './ipc'
import { roomModeDefaults } from '../../agent/room-outcomes'

import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey } from './workspace-mutation-guard'
import { applyRoomManualMemoryCommand, assertRoomConfigUpdateAllowed, assertRoomSeatRoutesConfigured, assertTicketUserUpdateAllowed, pickRoomConfigPatch, pickRoomManualMemoryCommand, pickTicketCreateInput, pickTicketUserPatch } from './room-ticket-patch-policy'
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
    persistRooms,
    persistTickets,
    persistSkills,
    aborters,
    roomAborters,
    DEFAULT_ROOM_COLLABORATION_MODE,
    DEFAULT_ROOM_WORKER_BUDGET,
    emitRoom,
    emitTicket,
    publishRemote,
    activeJob,
    patchRoomSupervisorJob,
    finishRoomSupervisorMission,
    finishJob,
    mkTicket,
    assertTicketRelations,
    normalizeStringList,
    startTicketWork,
    parkRoomIfWorkRemains,
    runRoom,
    normalizeTicketExecution,
    ticketWorkOptionsWithProjectRoute,
    isTicketStarting,
    resolveProjectProvider,
    ensureTensorFoldStudioStylePrompt,
    streamChat,
    parseJsonLoose,
    registerWebhookIpc,
    webhookServer,
    rememberConnectorEvents,
    validateProjectCwd,
    confirmSensitiveAction,
  } = ctx

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
  const { startRoomRun } = createRoomRunRuntime(ctx)

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
