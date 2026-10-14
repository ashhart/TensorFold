import type { ArchiveSnapshot, Job, Project, Provider, RedactedProvider, RemoteControlDoctorSnapshot, RemoteControlEvent, RemoteControlSnapshot, RemoteControlWorkSnapshot, Room, Thread, Ticket } from '../../agent/types'
import { store } from '../state'
import { isActive } from './archive'
import { liveTerminalActivity } from './terminal-live-state'

// Providers as a paired device may see them (VST-035): identity + models only. Built by
// PICKING fields, never by deleting from a spread — a new secret field on Provider can
// never leak through here by default.
export function redactProvidersForRemote(providers: Provider[]): RedactedProvider[] {
  return providers.map((provider) => ({
    id: provider.id,
    name: provider.name,
    kind: String(provider.kind ?? 'cloud'),
    transport: provider.transport ?? null,
    models: Array.isArray(provider.models) ? [...provider.models] : [],
    defaultModel: (provider as { defaultModel?: string | null }).defaultModel ?? null,
  }))
}

const REMOTE_VISIBLE_JOB_STATUSES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'])
const REMOTE_RETRYABLE_JOB_STATUSES = new Set<Job['status']>(['failed', 'stalled', 'dead_letter'])
const REMOTE_LEASE_EXPIRING_SOON_MS = 2 * 60_000

export function isIncognitoThreadId(id: unknown): boolean {
  return typeof id === 'string' && store.threads.some((thread) => thread.id === id && thread.incognito)
}

export function assertRemoteThreadVisible(id: string | null | undefined): void {
  if (isIncognitoThreadId(id)) throw Object.assign(new Error('not_found'), { code: 404 })
}

export function assertRemoteTicketVisible(id: string): void {
  const ticket = store.tickets.find((candidate) => candidate.id === id)
  assertRemoteThreadVisible(ticket?.threadId)
  assertRemoteThreadVisible(ticket?.execution?.threadId)
}

export function assertRemoteApprovalVisible(callId: string): void {
  const job = store.jobs.find((candidate) => candidate.pendingApprovals.some((approval) => approval.callId === callId && !approval.resolvedAt))
  if (job && isPrivateJob(job)) throw Object.assign(new Error('not_found'), { code: 404 })
}

function isPrivateJob(job: Job): boolean {
  return job.incognito === true || isIncognitoThreadId(job.threadId) || isIncognitoThreadId(job.currentThreadId) || job.kind === 'thread' && isIncognitoThreadId(job.owner)
}

function isPrivateTicket(ticket: Ticket): boolean {
  return isIncognitoThreadId(ticket.threadId) || isIncognitoThreadId(ticket.execution?.threadId)
}

/** Gates direct fork/archive publishers before the remote server's replay buffer. */
export function isRemoteEventVisible(event: Pick<RemoteControlEvent, 'kind' | 'data'>): boolean {
  const data = event.data as any
  if (!data) return true
  if (event.kind === 'agent') return !isIncognitoThreadId(data.threadId)
  if (event.kind === 'thread') return !data.incognito && !isIncognitoThreadId(data.id)
  if (event.kind === 'job') return !isPrivateJob(data as Job)
  if (event.kind === 'ticket') return !isPrivateTicket(data as Ticket)
  return true
}

export function remoteArchiveSnapshot(snapshot: ArchiveSnapshot): ArchiveSnapshot {
  return { ...snapshot, threads: snapshot.threads.filter((thread) => !thread.incognito), tickets: snapshot.tickets.filter((ticket) => !isPrivateTicket(ticket)) }
}

export interface RemoteSnapshotRuntime {
  appVersion: string
  headless: boolean
  activeThreadIds: string[]
  activeRoomIds: string[]
  activeMissionIds: string[]
}

export function remoteWorkSnapshot(now = Date.now()): RemoteControlWorkSnapshot {
  const visibleJobs = store.jobs.filter((job) => !isPrivateJob(job))
  const jobCounts = {
    total: visibleJobs.length,
    visible: 0,
    queued: 0,
    running: 0,
    waitingApproval: 0,
    paused: 0,
    stalled: 0,
    failed: 0,
    deadLetter: 0,
    retryable: 0,
  }
  const leases = { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 }
  const approvals = {
    pending: 0,
    tool: 0,
    connector: store.connectorEvents.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
    mission: store.missions.reduce((sum, mission) => sum + mission.pendingApprovals.length, 0),
  }
  let nextWakeAt: number | null = null
  let nextWakeJobId: string | null = null

  for (const job of visibleJobs) {
    if (REMOTE_VISIBLE_JOB_STATUSES.has(job.status)) jobCounts.visible += 1
    if (job.status === 'queued') jobCounts.queued += 1
    if (job.status === 'running') jobCounts.running += 1
    if (job.status === 'waiting_approval') jobCounts.waitingApproval += 1
    if (job.status === 'paused') jobCounts.paused += 1
    if (job.status === 'stalled') jobCounts.stalled += 1
    if (job.status === 'failed') jobCounts.failed += 1
    if (job.status === 'dead_letter') jobCounts.deadLetter += 1
    if (REMOTE_RETRYABLE_JOB_STATUSES.has(job.status)) jobCounts.retryable += 1
    approvals.tool += (job.pendingApprovals ?? []).filter((approval) => !approval.resolvedAt).length

    if (job.nextWakeAt && (nextWakeAt == null || job.nextWakeAt < nextWakeAt)) {
      nextWakeAt = job.nextWakeAt
      nextWakeJobId = job.id
    }

    if (job.leaseExpiresAt) {
      if (job.leaseExpiresAt <= now) leases.expired += 1
      else {
        leases.active += 1
        if (job.leaseExpiresAt - now <= REMOTE_LEASE_EXPIRING_SOON_MS) leases.expiringSoon += 1
      }
    } else if (job.status === 'running') {
      leases.missingRunning += 1
    }
  }

  approvals.pending = approvals.tool + approvals.connector + approvals.mission
  const terminals = liveTerminalActivity()
  return { jobs: jobCounts, leases, approvals, nextWakeAt, nextWakeJobId, generatedAt: now, terminals }
}

export function remoteControlSnapshot(runtime: RemoteSnapshotRuntime): RemoteControlSnapshot {
  return {
    generatedAt: Date.now(),
    serverName: store.settings.remoteControl.serverName,
    appVersion: runtime.appVersion,
    headless: runtime.headless,
    threads: store.threads.filter((thread) => isActive(thread) && !thread.incognito),
    projects: store.projects.filter(isActive),
    rooms: store.rooms.filter(isActive),
    tickets: store.tickets.filter((ticket) => isActive(ticket) && !isPrivateTicket(ticket)),
    automations: store.automations,
    missions: store.missions,
    jobs: store.jobs.filter((job) => !isPrivateJob(job)),
    connectorEvents: store.connectorEvents.slice(0, 200),
    work: remoteWorkSnapshot(),
    activeThreadIds: runtime.activeThreadIds.filter((id) => !isIncognitoThreadId(id)),
    activeRoomIds: runtime.activeRoomIds,
    activeMissionIds: runtime.activeMissionIds,
    providers: redactProvidersForRemote(store.settings.providers ?? []),
  }
}

// Revocable paired-device sessions receive an operational projection, not a duplicate of
// every prompt, transcript, tool output, project memory, room discussion, and filesystem
// path held by the owner process. The shared owner token retains the full snapshot.
export function pairedRemoteControlSnapshot(runtime: RemoteSnapshotRuntime): RemoteControlSnapshot {
  return minimizeRemoteSnapshot(remoteControlSnapshot(runtime))
}

function minimizeRemoteSnapshot(snapshot: RemoteControlSnapshot): RemoteControlSnapshot {
  return {
    ...snapshot,
    projects: snapshot.projects.map(redactProjectForPaired),
    threads: snapshot.threads.map(redactThreadForPaired),
    rooms: snapshot.rooms.map(redactRoomForPaired),
    tickets: snapshot.tickets.map(redactTicketForPaired),
    automations: [],
    missions: [],
    jobs: snapshot.jobs.map(redactJobForPaired),
    connectorEvents: [],
  }
}

function redactProjectForPaired(project: Project): Project {
  return { ...project, cwd: '', workspace: null, memory: '', team: '', defaultProviderAccountId: null }
}

function redactThreadForPaired(thread: Thread): Thread {
  return {
    ...thread,
    cwd: '',
    workspace: null,
    providerAccountId: null,
    apiMessages: [],
    timeline: (thread.timeline ?? []).map((item) => ({
      id: item.id,
      kind: item.kind,
      tool: item.tool,
      state: item.state,
      status: item.status,
      createdAt: item.createdAt,
    })),
    codexResumeCursor: null,
    claudeResumeCursor: null,
    openCodeResumeCursor: null,
    piResumeCursor: null,
    cursorResumeCursor: null,
    ompResumeCursor: null,
    systemPromptSnapshot: null,
    activeRunContract: null,
    lastRunReceipt: null,
    goal: undefined,
    plan: [],
    notepad: undefined,
    annotations: undefined,
    compactions: [],
    contextFragments: [],
  }
}

function redactRoomForPaired(room: Room): Room {
  return {
    ...room,
    rootCwd: null,
    workspaceCwd: null,
    scratchpadCwd: null,
    workspace: null,
    memory: [],
    memorySummary: '',
    problem: '',
    events: (room.events ?? []).map((event) => ({ ...event, text: '', args: undefined, path: undefined })),
    finalText: '',
  }
}

function redactTicketForPaired(ticket: Ticket): Ticket {
  return {
    ...ticket,
    body: '',
    acceptanceCriteria: [],
    requiredEvidence: [],
    evidence: [],
    execution: ticket.execution ? { ...ticket.execution, checks: [], blockers: [] } : ticket.execution,
  }
}

export function pairedRemoteArchiveSnapshot(snapshot: ArchiveSnapshot): ArchiveSnapshot {
  snapshot = remoteArchiveSnapshot(snapshot)
  return {
    projects: snapshot.projects.map(redactProjectForPaired),
    threads: snapshot.threads.map(redactThreadForPaired),
    rooms: snapshot.rooms.map(redactRoomForPaired),
    tickets: snapshot.tickets.map(redactTicketForPaired),
  }
}

function redactJobForPaired(job: Job): Job {
  return { ...job, metadata: {}, pendingApprovals: [] }
}

export function redactRemoteControlEventForPaired(event: Pick<RemoteControlEvent, 'kind' | 'data'>): Pick<RemoteControlEvent, 'kind' | 'data'> {
  if (!isRemoteEventVisible(event)) return { kind: event.kind, data: {} }
  const data: any = event.data
  if (!data) return event
  if (event.kind === 'snapshot') return { kind: event.kind, data: minimizeRemoteSnapshot(data as RemoteControlSnapshot) }
  if (event.kind === 'project') return { kind: event.kind, data: redactProjectForPaired(data as Project) }
  if (event.kind === 'thread') return { kind: event.kind, data: redactThreadForPaired(data as Thread) }
  if (event.kind === 'room') {
    if (typeof data.type === 'string' && typeof data.roomId === 'string') {
      return { kind: event.kind, data: { roomId: data.roomId, type: data.type, state: data.state, seat: data.seat, item: data.item ? { id: data.item.id, kind: data.item.kind, seat: data.item.seat, status: data.item.status, text: '' } : undefined, chunk: data.chunk ? '' : undefined, text: data.text ? '' : undefined } }
    }
    return { kind: event.kind, data: redactRoomForPaired(data as Room) }
  }
  if (event.kind === 'ticket') return { kind: event.kind, data: redactTicketForPaired(data as Ticket) }
  if (event.kind === 'job') return { kind: event.kind, data: redactJobForPaired(data as Job) }
  if (event.kind === 'agent') {
    const safe: Record<string, unknown> = { type: data.type, runId: data.runId, threadId: data.threadId, status: data.status }
    if (data.item) safe.item = { id: data.item.id, kind: data.item.kind, tool: data.item.tool, state: data.item.state, createdAt: data.item.createdAt }
    if (data.id) safe.id = data.id
    if (data.field) safe.field = data.field
    if (data.patch) safe.patch = { kind: data.patch.kind, state: data.patch.state, status: data.patch.status }
    if (data.plan) safe.plan = []
    return { kind: event.kind, data: safe }
  }
  if (event.kind === 'terminal') return { kind: event.kind, data: { type: data.type, sessionId: data.sessionId, projectId: data.projectId, status: data.status, data: data.data ? '' : undefined } }
  if (event.kind === 'mission' || event.kind === 'automation' || event.kind === 'connector') {
    return { kind: event.kind, data: { id: data.id, status: data.status, state: data.state, updatedAt: data.updatedAt, createdAt: data.createdAt } }
  }
  return { kind: event.kind, data: {} }
}

export function remoteControlDoctorSnapshot(runtime: RemoteSnapshotRuntime): RemoteControlDoctorSnapshot {
  const activeThreads = new Set(runtime.activeThreadIds.filter((id) => !isIncognitoThreadId(id)))
  const activeRooms = new Set(runtime.activeRoomIds)
  const activeMissions = new Set(runtime.activeMissionIds)
  return {
    generatedAt: Date.now(),
    serverName: store.settings.remoteControl.serverName,
    appVersion: runtime.appVersion,
    headless: runtime.headless,
    counts: {
      threads: store.threads.filter((thread) => isActive(thread) && !thread.incognito).length,
      projects: store.projects.filter(isActive).length,
      rooms: store.rooms.filter(isActive).length,
      tickets: store.tickets.filter((ticket) => isActive(ticket) && !isPrivateTicket(ticket)).length,
      automations: store.automations.length,
      missions: store.missions.length,
      jobs: store.jobs.filter((job) => !isPrivateJob(job)).length,
      connectorEvents: store.connectorEvents.length,
      activeThreads: activeThreads.size,
      activeRooms: activeRooms.size,
      activeMissions: activeMissions.size,
    },
    work: remoteWorkSnapshot(),
  }
}
