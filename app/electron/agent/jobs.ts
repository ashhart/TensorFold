import { randomUUID } from 'node:crypto'
import type { ApprovalMode, Job, JobApprovalDecision, JobExecutionVerification, JobPendingApproval, JobResumeAfterRestart, JobKind, JobStatus, Project, Room, RoomSupervisorJobMetadata, RoomSupervisorPhaseSnapshot, RoomSupervisorTicketPhase, SeatHeartbeat, Ticket } from './types'

const DEFAULT_LEASE_MS = 5 * 60_000
const DEFAULT_STUCK_PROGRESS_MS = 20 * 60_000
const DEFAULT_STUCK_LEASE_GRACE_MS = 60_000
const DEFAULT_REPEATED_NO_PROGRESS_THRESHOLD = 3
const TERMINAL = new Set<JobStatus>(['completed', 'failed', 'cancelled', 'dead_letter'])
const AUTO_RESUME_AFTER_RESTART = new Set<JobResumeAfterRestart>(['read_only', 'low_risk', 'full_policy'])

export interface JobRef {
  kind: JobKind
  incognito?: boolean
  owner: string
  projectId?: string | null
  missionId?: string | null
  roomId?: string | null
  threadId?: string | null
  ticketId?: string | null
  automationId?: string | null
  connectorEventId?: string | null
  currentThreadId?: string | null
  executionHostId?: string | null
  resumeAfterRestart?: JobResumeAfterRestart
  metadata?: Record<string, unknown>
}

export interface JobPatch {
  status?: JobStatus
  attempt?: number
  projectId?: string | null
  missionId?: string | null
  roomId?: string | null
  threadId?: string | null
  ticketId?: string | null
  automationId?: string | null
  connectorEventId?: string | null
  currentRunId?: string | null
  currentThreadId?: string | null
  nextWakeAt?: number | null
  lastUsefulProgressAt?: number | null
  blockedReason?: string | null
  resumePlan?: string | null
  resumeAfterRestart?: JobResumeAfterRestart
  pendingApprovals?: JobPendingApproval[]
  leaseOwner?: string | null
  leaseExpiresAt?: number | null
  executionHostId?: string | null
  executionVerification?: JobExecutionVerification | null
  executionVerificationReason?: string | null
  executionVerifiedAt?: number | null
  metadata?: Record<string, unknown>
}

export interface JobRetryOptions {
  reason?: string
  nextWakeAt?: number | null
  resumePlan?: string | null
}

export interface JobStuckDetectionOptions {
  progressTimeoutMs?: number
  leaseGraceMs?: number
  repeatedNoProgressThreshold?: number
}

export interface RoomSupervisorMetadataPatch {
  missionId?: string | null
  currentTurnSeat?: string | null
  lastEventKind?: string | null
  lastEventSeat?: string | null
  lastEventAt?: number | null
  workerStarts?: number
  workerBudget?: { maxActive: number; maxStarts: number }
  phase?: Partial<RoomSupervisorPhaseSnapshot> & { phase: RoomSupervisorTicketPhase }
}

export interface JobApprovalRecordInput {
  callId: string
  threadId?: string | null
  itemId?: string | null
  tool: string
  policy: JobPendingApproval['policy']
}

export function normalizeJobs(input: Job[], now = Date.now()): Job[] {
  return input
    .filter((job) => job && typeof job.id === 'string' && typeof job.kind === 'string')
    .map((job) => ({
      ...job,
      status: job.status ?? 'queued',
      attempt: Number.isFinite(job.attempt) ? job.attempt : 0,
      owner: job.owner || ownerFromJob(job),
      projectId: job.projectId ?? null,
      missionId: job.missionId ?? null,
      roomId: job.roomId ?? null,
      threadId: job.threadId ?? null,
      ticketId: job.ticketId ?? null,
      automationId: job.automationId ?? null,
      connectorEventId: job.connectorEventId ?? null,
      currentRunId: job.currentRunId ?? null,
      currentThreadId: job.currentThreadId ?? job.threadId ?? null,
      nextWakeAt: job.nextWakeAt ?? null,
      lastUsefulProgressAt: job.lastUsefulProgressAt ?? job.startedAt ?? job.updatedAt ?? now,
      blockedReason: job.blockedReason ?? null,
      resumePlan: job.resumePlan ?? null,
      resumeAfterRestart: job.incognito === true ? 'never' as const : normalizeResumeAfterRestart((job as Partial<Job>).resumeAfterRestart, job.status),
      pendingApprovals: normalizePendingApprovals((job as Partial<Job>).pendingApprovals, now),
      leaseOwner: job.leaseOwner ?? null,
      leaseExpiresAt: job.leaseExpiresAt ?? null,
      executionHostId: job.executionHostId ?? null,
      executionVerification: normalizeExecutionVerification(job.executionVerification),
      executionVerificationReason: job.executionVerificationReason ?? null,
      executionVerifiedAt: job.executionVerifiedAt ?? null,
      startedAt: job.startedAt ?? null,
      finishedAt: job.finishedAt ?? null,
      createdAt: job.createdAt ?? now,
      updatedAt: job.updatedAt ?? now,
    }))
    .sort((a, b) => b.updatedAt - a.updatedAt)
}

export function buildRoomSupervisorJobMetadata(
  input: {
    room: Room
    project?: Pick<Project, 'id' | 'name'> | null
    tickets?: Ticket[]
    previous?: Record<string, unknown> | null
    patch?: RoomSupervisorMetadataPatch
  },
  now = Date.now(),
): RoomSupervisorJobMetadata {
  const previous = input.previous ?? {}
  const patch = input.patch ?? {}
  const tickets = input.tickets ?? []
  const workerBudget = patch.workerBudget ?? readWorkerBudget(previous)
  const workerStarts = patch.workerStarts ?? readNumber(previous.workerBudget, 'started') ?? 0
  const activeWorkers = tickets.filter((ticket) => ticket.runState === 'queued' || ticket.runState === 'running').length
  const phases = readPhases(previous, now)
  if (patch.phase) {
    const current = phases[patch.phase.phase]
    phases[patch.phase.phase] = {
      ...current,
      ...patch.phase,
      phase: patch.phase.phase,
      ticketId: patch.phase.ticketId !== undefined ? patch.phase.ticketId : current.ticketId,
      updatedAt: patch.phase.updatedAt !== undefined ? patch.phase.updatedAt : now,
    }
  }
  const lastEvent = input.room.events.at(-1)
  return {
    schema: 'vontra.room-supervisor/1',
    supervisor: 'room',
    missionId: patch.missionId !== undefined ? patch.missionId : readString(previous, 'missionId'),
    roomId: input.room.id,
    roomName: input.room.name,
    projectId: input.room.projectId,
    projectName: input.project?.name ?? null,
    problem: input.room.problem,
    roomState: input.room.state,
    workspaceCwd: input.room.workspaceCwd ?? null,
    scratchpadCwd: input.room.scratchpadCwd ?? null,
    rootCwd: input.room.rootCwd ?? null,
    seatCount: input.room.seats.length,
    eventCount: input.room.events.length,
    ticketCounts: countTicketStatuses(tickets),
    currentTurnSeat: patch.currentTurnSeat !== undefined ? patch.currentTurnSeat : readString(previous, 'currentTurnSeat'),
    lastEventKind: patch.lastEventKind !== undefined ? patch.lastEventKind : lastEvent?.kind ?? readString(previous, 'lastEventKind'),
    lastEventSeat: patch.lastEventSeat !== undefined ? patch.lastEventSeat : lastEvent?.seat ?? readString(previous, 'lastEventSeat'),
    lastEventAt: patch.lastEventAt !== undefined ? patch.lastEventAt : lastEvent?.createdAt ?? readNumber(previous, 'lastEventAt'),
    lastHeartbeatAt: now,
    seatHeartbeats: readSeatHeartbeats(input.room, previous),
    workerBudget: { maxActive: workerBudget.maxActive, maxStarts: workerBudget.maxStarts, started: workerStarts, active: activeWorkers },
    phases,
  }
}

export function enqueueJob(jobs: Job[], ref: JobRef, now = Date.now()): { jobs: Job[]; job: Job; created: boolean } {
  const owner = ref.owner || ownerFromRef(ref)
  const existing = jobs.find((job) => !TERMINAL.has(job.status) && job.kind === ref.kind && job.owner === owner && (job.incognito === true) === (ref.incognito === true))
  if (existing) {
    const patched = updateJob(jobs, existing.id, {
      status: existing.status === 'paused' || existing.status === 'stalled' ? 'queued' : existing.status,
      ...refPatch(ref),
      lastUsefulProgressAt: existing.lastUsefulProgressAt ?? now,
    }, now)
    return { jobs: patched.jobs, job: patched.job!, created: false }
  }
  const job: Job = {
    id: randomUUID(),
    ...(ref.incognito === true ? { incognito: true } : {}),
    kind: ref.kind,
    status: 'queued',
    attempt: 0,
    owner,
    projectId: ref.projectId ?? null,
    missionId: ref.missionId ?? null,
    roomId: ref.roomId ?? null,
    threadId: ref.threadId ?? null,
    ticketId: ref.ticketId ?? null,
    automationId: ref.automationId ?? null,
    connectorEventId: ref.connectorEventId ?? null,
    currentRunId: null,
    currentThreadId: ref.currentThreadId ?? ref.threadId ?? null,
    nextWakeAt: null,
    lastUsefulProgressAt: now,
    blockedReason: null,
    resumePlan: null,
    resumeAfterRestart: ref.incognito === true ? 'never' : ref.resumeAfterRestart ?? defaultResumeAfterRestart(ref.kind),
    pendingApprovals: [],
    leaseOwner: null,
    leaseExpiresAt: null,
    executionHostId: ref.executionHostId ?? null,
    executionVerification: null,
    executionVerificationReason: null,
    executionVerifiedAt: null,
    startedAt: null,
    finishedAt: null,
    metadata: ref.metadata,
    createdAt: now,
    updatedAt: now,
  }
  return { jobs: [job, ...jobs], job, created: true }
}

export function updateJob(jobs: Job[], id: string, patch: JobPatch, now = Date.now()): { jobs: Job[]; job: Job | null } {
  let updated: Job | null = null
  const next = jobs.map((job) => {
    if (job.id !== id) return job
    const status = patch.status ?? job.status
    const remoteExited = !!job.executionHostId && TERMINAL.has(status)
    updated = {
      ...job,
      ...patch,
      ...(job.incognito === true ? { incognito: true, resumeAfterRestart: 'never' as const } : {}),
      pendingApprovals: patch.pendingApprovals ? normalizePendingApprovals(patch.pendingApprovals, now) : job.pendingApprovals ?? [],
      metadata: patch.metadata ? { ...(job.metadata ?? {}), ...patch.metadata } : job.metadata,
      executionVerification: patch.executionVerification !== undefined
        ? patch.executionVerification
        : remoteExited ? 'exited' : job.executionVerification ?? null,
      executionVerificationReason: patch.executionVerificationReason !== undefined
        ? patch.executionVerificationReason
        : remoteExited ? `Remote process ${status}.` : job.executionVerificationReason ?? null,
      executionVerifiedAt: patch.executionVerifiedAt !== undefined
        ? patch.executionVerifiedAt
        : remoteExited ? now : job.executionVerifiedAt ?? null,
      updatedAt: now,
      startedAt: patch.status === 'running' ? job.startedAt ?? now : job.startedAt ?? null,
      finishedAt: patch.status && TERMINAL.has(patch.status) ? now : job.finishedAt ?? null,
    }
    return updated
  })
  return { jobs: next, job: updated }
}

export function claimJobLease(jobs: Job[], id: string, leaseOwner: string, now = Date.now(), leaseMs = DEFAULT_LEASE_MS): { jobs: Job[]; job: Job | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || TERMINAL.has(current.status)) return { jobs, job: null }
  if (current.leaseExpiresAt && current.leaseExpiresAt > now && current.leaseOwner && current.leaseOwner !== leaseOwner) return { jobs, job: null }
  return updateJob(jobs, id, {
    status: 'running',
    attempt: current.attempt + 1,
    leaseOwner,
    leaseExpiresAt: now + leaseMs,
    lastUsefulProgressAt: current.lastUsefulProgressAt ?? now,
  }, now)
}

export function heartbeatJobLease(jobs: Job[], id: string, leaseOwner: string, now = Date.now(), leaseMs = DEFAULT_LEASE_MS): { jobs: Job[]; job: Job | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || current.leaseOwner !== leaseOwner || TERMINAL.has(current.status)) return { jobs, job: null }
  return updateJob(jobs, id, {
    leaseExpiresAt: now + leaseMs,
    lastUsefulProgressAt: now,
    ...(current.executionHostId ? {
      executionVerification: 'live' as const,
      executionVerificationReason: 'Remote execution heartbeat received.',
      executionVerifiedAt: now,
    } : {}),
  }, now)
}

/**
 * Mark active jobs on a remote host as unverifiable after transport loss. Their lifecycle
 * status is deliberately preserved: a disconnected observer cannot prove process exit.
 */
export function markRemoteJobsUnverifiable(
  jobs: Job[],
  hostId: string,
  reason = 'Remote transport disconnected; process state cannot be verified.',
  now = Date.now(),
): { jobs: Job[]; affected: Job[]; changed: boolean } {
  const affected: Job[] = []
  const next = jobs.map((job) => {
    if (job.executionHostId !== hostId || (job.status !== 'running' && job.status !== 'waiting_approval')) return job
    if (job.executionVerification === 'unverifiable' && job.executionVerificationReason === reason) return job
    const updated: Job = {
      ...job,
      executionVerification: 'unverifiable',
      executionVerificationReason: reason.slice(0, 600),
      executionVerifiedAt: now,
      updatedAt: now,
    }
    affected.push(updated)
    return updated
  })
  return { jobs: next, affected, changed: affected.length > 0 }
}

export function markRemoteJobLive(jobs: Job[], id: string, hostId: string, reason = 'Remote execution is reporting live state.', now = Date.now()) {
  return updateJob(jobs, id, {
    executionHostId: hostId,
    executionVerification: 'live',
    executionVerificationReason: reason,
    executionVerifiedAt: now,
  }, now)
}

export function releaseJobLease(jobs: Job[], id: string, status: JobStatus = 'paused', blockedReason: string | null = null, now = Date.now()): { jobs: Job[]; job: Job | null } {
  // Stopping a job also drops any scheduled wake — a parked room's nextWakeAt must
  // not fire after the job was paused, failed, cancelled, or completed.
  return updateJob(jobs, id, { status, leaseOwner: null, leaseExpiresAt: null, blockedReason, nextWakeAt: null }, now)
}

export function retryJob(jobs: Job[], id: string, options: JobRetryOptions = {}, now = Date.now()): { jobs: Job[]; job: Job | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || current.status === 'completed' || current.status === 'cancelled') return { jobs, job: null }
  return updateJob(jobs, id, {
    status: 'queued',
    nextWakeAt: options.nextWakeAt ?? null,
    blockedReason: options.reason ?? null,
    resumePlan: options.resumePlan ?? current.resumePlan ?? 'Retry from the last safe checkpoint.',
    pendingApprovals: [],
    leaseOwner: null,
    leaseExpiresAt: null,
    currentRunId: null,
  }, now)
}

export function resumeJob(jobs: Job[], id: string, resumePlan: string | null = null, now = Date.now()): { jobs: Job[]; job: Job | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || TERMINAL.has(current.status)) return { jobs, job: null }
  return updateJob(jobs, id, {
    status: 'queued',
    nextWakeAt: null,
    blockedReason: null,
    resumePlan: resumePlan ?? current.resumePlan ?? 'Resume deliberately from the current state.',
    leaseOwner: null,
    leaseExpiresAt: null,
    currentRunId: null,
  }, now)
}

export function deadLetterJob(jobs: Job[], id: string, reason: string, now = Date.now()): { jobs: Job[]; job: Job | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || current.status === 'completed' || current.status === 'cancelled') return { jobs, job: null }
  return releaseJobLease(jobs, id, 'dead_letter', reason || 'Moved to dead letter.', now)
}

export function recordJobApproval(jobs: Job[], id: string, approval: JobApprovalRecordInput, now = Date.now()): { jobs: Job[]; job: Job | null; approval: JobPendingApproval | null } {
  const current = jobs.find((job) => job.id === id)
  if (!current || TERMINAL.has(current.status)) return { jobs, job: null, approval: null }
  const existing = (current.pendingApprovals ?? []).find((item) => item.callId === approval.callId)
  const pending: JobPendingApproval = existing
    ? {
        ...existing,
        threadId: approval.threadId ?? existing.threadId ?? null,
        itemId: approval.itemId ?? existing.itemId ?? null,
        tool: approval.tool || existing.tool,
        policy: approval.policy,
        resolvedAt: null,
        decision: null,
      }
    : {
        id: randomUUID(),
        callId: approval.callId,
        threadId: approval.threadId ?? null,
        itemId: approval.itemId ?? null,
        tool: approval.tool,
        policy: approval.policy,
        requestedAt: now,
        resolvedAt: null,
        decision: null,
      }
  const nextApprovals = existing
    ? (current.pendingApprovals ?? []).map((item) => item.callId === approval.callId ? pending : item)
    : [pending, ...(current.pendingApprovals ?? [])]
  const result = updateJob(jobs, id, {
    status: 'waiting_approval',
    blockedReason: `Approval pending for ${pending.policy.label || pending.tool}.`,
    resumeAfterRestart: 'pending_approval',
    pendingApprovals: nextApprovals,
    metadata: {
      restartResumePolicyBeforeApproval: approvalResumePolicyBeforePending(current),
    },
  }, now)
  return { ...result, approval: result.job ? pending : null }
}

export function resolveJobApproval(jobs: Job[], callId: string, decision: JobApprovalDecision, now = Date.now()): { jobs: Job[]; job: Job | null; approval: JobPendingApproval | null } {
  const matches = jobs.flatMap((job) => (job.pendingApprovals ?? [])
    .filter((approval) => !approval.resolvedAt && [approval.callId, approval.id, approval.itemId].includes(callId))
    .map((approval) => ({ job, approval })))
  if (matches.length > 1) throw new Error('Approval address is ambiguous; resolve it from its worker or restored job.')
  const match = matches[0]
  if (!match) return { jobs, job: null, approval: null }
  const current = match.job
  let resolved: JobPendingApproval | null = null
  const pendingApprovals = (current.pendingApprovals ?? []).map((approval) => {
    if (approval !== match.approval) return approval
    resolved = { ...approval, decision, resolvedAt: now }
    return resolved
  })
  const stillWaiting = pendingApprovals.some((approval) => !approval.resolvedAt)
  const result = updateJob(jobs, current.id, {
    status: stillWaiting ? 'waiting_approval' : 'queued',
    blockedReason: stillWaiting ? current.blockedReason : null,
    resumePlan: stillWaiting ? current.resumePlan : 'Continue after the restored approval was resolved.',
    resumeAfterRestart: stillWaiting ? 'pending_approval' : current.resumeAfterRestart === 'pending_approval' ? restoredApprovalResumePolicy(current) : current.resumeAfterRestart,
    leaseOwner: null,
    leaseExpiresAt: null,
    currentRunId: null,
    pendingApprovals,
  }, now)
  return { ...result, approval: resolved }
}

export function reconcileJobsAfterRestart(jobs: Job[], now = Date.now()): { jobs: Job[]; changed: boolean } {
  let changed = false
  const next: Job[] = jobs.map((job): Job => {
    if (TERMINAL.has(job.status)) return job
    const hasPendingApproval = (job.pendingApprovals ?? []).some((approval) => !approval.resolvedAt)
    if (job.status === 'waiting_approval' || hasPendingApproval || job.resumeAfterRestart === 'pending_approval') {
      if (
        job.status === 'waiting_approval' &&
        job.leaseOwner === null &&
        job.leaseExpiresAt === null &&
        job.resumeAfterRestart === 'pending_approval'
      ) return job
      changed = true
      return {
        ...job,
        status: 'waiting_approval',
        resumeAfterRestart: 'pending_approval',
        leaseOwner: null,
        leaseExpiresAt: null,
        blockedReason: job.blockedReason ?? 'Approval pending from a previous run.',
        resumePlan: job.resumePlan ?? 'Resolve the pending approval, then continue deliberately.',
        updatedAt: now,
      }
    }
    if (job.status !== 'running' && job.status !== 'stalled') return job
    if (restartPolicyAutoResumes(job.resumeAfterRestart)) {
      changed = true
      return {
        ...job,
        status: 'queued',
        currentRunId: null,
        leaseOwner: null,
        leaseExpiresAt: null,
        blockedReason: null,
        resumePlan: job.resumePlan ?? restartResumePlan(job.resumeAfterRestart),
        updatedAt: now,
      }
    }
    changed = true
    return {
      ...job,
      status: 'paused',
      currentRunId: null,
      leaseOwner: null,
      leaseExpiresAt: null,
      blockedReason: restartPauseReason(job.resumeAfterRestart),
      resumePlan: job.resumePlan ?? restartPausedPlan(job),
      updatedAt: now,
    }
  })
  return { jobs: next, changed }
}

export function restartPolicyAutoResumes(policy: JobResumeAfterRestart | null | undefined): boolean {
  return AUTO_RESUME_AFTER_RESTART.has(policy ?? 'ask')
}

export function restartResumeApprovalMode(policy: JobResumeAfterRestart | null | undefined, fallback: ApprovalMode): ApprovalMode {
  if (policy === 'read_only') return 'readonly'
  // `auto` is Full Auto in the agent loop: it suppresses approvals for the safe command/edit
  // allowlist. A low-risk restart must therefore use the explicit suggest posture regardless
  // of the saved fallback, so resumed mutations remain approval-gated. Shell stays sandboxed
  // in every mode.
  if (policy === 'low_risk') return 'suggest'
  return fallback
}

export function detectStuckJobs(jobs: Job[], now = Date.now(), options: JobStuckDetectionOptions = {}): { jobs: Job[]; stalled: Job[]; changed: boolean } {
  const progressTimeoutMs = positiveNumber(options.progressTimeoutMs, DEFAULT_STUCK_PROGRESS_MS)
  const leaseGraceMs = positiveNumber(options.leaseGraceMs, DEFAULT_STUCK_LEASE_GRACE_MS)
  const repeatedNoProgressThreshold = positiveNumber(options.repeatedNoProgressThreshold, DEFAULT_REPEATED_NO_PROGRESS_THRESHOLD)
  const stalled: Job[] = []
  let changed = false
  const next = jobs.map((job) => {
    if (TERMINAL.has(job.status) || job.status === 'paused' || job.status === 'waiting_approval' || job.status === 'stalled') return job
    const reason = stuckReason(job, now, progressTimeoutMs, leaseGraceMs, repeatedNoProgressThreshold)
    if (!reason) return job
    changed = true
    const updated: Job = {
      ...job,
      status: 'stalled',
      leaseOwner: null,
      leaseExpiresAt: null,
      blockedReason: reason,
      resumePlan: job.resumePlan ?? 'Inspect the latest evidence, then resume or retry deliberately.',
      metadata: {
        ...(job.metadata ?? {}),
        stuckDetectedAt: now,
        stuckReason: reason,
      },
      updatedAt: now,
    }
    stalled.push(updated)
    return updated
  })
  return { jobs: next, stalled, changed }
}

export function completeJob(jobs: Job[], id: string, now = Date.now()) {
  return releaseJobLease(jobs, id, 'completed', null, now)
}

export function failJob(jobs: Job[], id: string, reason: string, now = Date.now()) {
  return releaseJobLease(jobs, id, 'failed', reason, now)
}

export function cancelJob(jobs: Job[], id: string, reason = 'Cancelled by user.', now = Date.now()) {
  return releaseJobLease(jobs, id, 'cancelled', reason, now)
}

function refPatch(ref: JobRef): JobPatch {
  const patch: JobPatch = {
    currentThreadId: ref.currentThreadId ?? ref.threadId ?? null,
    metadata: ref.metadata,
  }
  if (ref.projectId !== undefined) patch.projectId = ref.projectId
  if (ref.missionId !== undefined) patch.missionId = ref.missionId
  if (ref.roomId !== undefined) patch.roomId = ref.roomId
  if (ref.threadId !== undefined) patch.threadId = ref.threadId
  if (ref.ticketId !== undefined) patch.ticketId = ref.ticketId
  if (ref.automationId !== undefined) patch.automationId = ref.automationId
  if (ref.connectorEventId !== undefined) patch.connectorEventId = ref.connectorEventId
  return patch
}

function ownerFromRef(ref: JobRef): string {
  return ref.missionId ?? ref.roomId ?? ref.ticketId ?? ref.automationId ?? ref.threadId ?? ref.connectorEventId ?? ref.owner
}

function ownerFromJob(job: Pick<Job, 'kind' | 'missionId' | 'roomId' | 'ticketId' | 'automationId' | 'threadId' | 'connectorEventId' | 'id'>): string {
  return job.missionId ?? job.roomId ?? job.ticketId ?? job.automationId ?? job.threadId ?? job.connectorEventId ?? `${job.kind}:${job.id}`
}

function stuckReason(job: Job, now: number, progressTimeoutMs: number, leaseGraceMs: number, repeatedNoProgressThreshold: number): string | null {
  if (job.status !== 'running') return null
  if (job.leaseExpiresAt && job.leaseExpiresAt + leaseGraceMs < now) return 'Worker lease expired without a heartbeat.'
  const lastProgress = job.lastUsefulProgressAt ?? job.startedAt ?? job.updatedAt ?? job.createdAt
  if (lastProgress && now - lastProgress > progressTimeoutMs) return 'No useful progress heartbeat within the stuck-job window.'
  const repeated = metadataNumber(job, 'sameOutputCount') ?? metadataNumber(job, 'noProgressCount') ?? metadataNumber(job, 'repeatedNoProgressCount')
  if (repeated !== null && repeated >= repeatedNoProgressThreshold) return 'Repeated same-output/no-progress signal reached the stuck-job threshold.'
  return null
}

function metadataNumber(job: Job, key: string): number | null {
  const value = job.metadata?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function positiveNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

function defaultResumeAfterRestart(kind: JobKind): JobResumeAfterRestart {
  return kind === 'automation' ? 'full_policy' : 'ask'
}

function restartResumePlan(policy: JobResumeAfterRestart): string {
  if (policy === 'read_only') return 'Automatically requeued after restart in read-only mode.'
  if (policy === 'low_risk') return 'Automatically requeued after restart with low-risk tool policy.'
  return 'Automatically requeued after restart with the saved approval policy.'
}

function restartPauseReason(policy: JobResumeAfterRestart): string {
  if (policy === 'never') return 'Paused after restart - restart resume is disabled for this job.'
  return 'Paused after restart - user confirmation required.'
}

function restartPausedPlan(job: Job): string {
  const policy = job.resumeAfterRestart
  if (job.kind === 'room' && activeWorkerCount(job.metadata) > 0) {
    return 'A room worker was active during restart. Inspect the worker thread/evidence, then wake or resume the room from the remaining tickets.'
  }
  if (policy === 'never') return 'Restart resume is disabled. Start a fresh run or explicitly resume only after review.'
  return 'Inspect the work, then resume deliberately.'
}

function approvalResumePolicyBeforePending(job: Job): JobResumeAfterRestart {
  if (job.resumeAfterRestart !== 'pending_approval') return job.resumeAfterRestart
  return restoredApprovalResumePolicy(job)
}

function restoredApprovalResumePolicy(job: Job): JobResumeAfterRestart {
  const restored = normalizeResumeAfterRestart(job.metadata?.restartResumePolicyBeforeApproval, job.status)
  return restored === 'pending_approval' ? 'ask' : restored
}

function activeWorkerCount(metadata: Record<string, unknown> | undefined): number {
  const workerBudget = metadata && typeof metadata === 'object' ? metadata.workerBudget : null
  if (!workerBudget || typeof workerBudget !== 'object') return 0
  const active = (workerBudget as Record<string, unknown>).active
  return Number.isFinite(Number(active)) ? Math.max(0, Math.floor(Number(active))) : 0
}

function countTicketStatuses(tickets: Ticket[]): Record<string, number> {
  const counts: Record<string, number> = { total: tickets.length }
  for (const ticket of tickets) counts[ticket.status] = (counts[ticket.status] ?? 0) + 1
  return counts
}

function readPhases(previous: Record<string, unknown>, now: number): Record<RoomSupervisorTicketPhase, RoomSupervisorPhaseSnapshot> {
  const raw = previous.phases && typeof previous.phases === 'object' ? previous.phases as Record<string, Partial<RoomSupervisorPhaseSnapshot>> : {}
  return {
    build: readPhase(raw.build, 'build', now),
    review: readPhase(raw.review, 'review', now),
  }
}

function readPhase(input: Partial<RoomSupervisorPhaseSnapshot> | undefined, phase: RoomSupervisorTicketPhase, now: number): RoomSupervisorPhaseSnapshot {
  return {
    phase,
    status: input?.status ?? 'idle',
    ticketId: input?.ticketId ?? null,
    ticketTitle: input?.ticketTitle ?? null,
    seat: input?.seat ?? null,
    threadId: input?.threadId ?? null,
    jobId: input?.jobId ?? null,
    paths: Array.isArray(input?.paths) ? input.paths.filter((item): item is string => typeof item === 'string') : [],
    summary: input?.summary ?? null,
    updatedAt: typeof input?.updatedAt === 'number' ? input.updatedAt : null,
  }
}

// Prefer the room's live seat heartbeats; fall back to the previously persisted
// snapshot so the field survives metadata patches that do not refresh the room (BUG #53).
function readSeatHeartbeats(room: Room, previous: Record<string, unknown>): Record<string, SeatHeartbeat> | undefined {
  const live = room.seatHeartbeats
  if (live && Object.keys(live).length) return live
  const prior = previous.seatHeartbeats
  if (prior && typeof prior === 'object') return prior as Record<string, SeatHeartbeat>
  return undefined
}

function readWorkerBudget(previous: Record<string, unknown>): { maxActive: number; maxStarts: number } {
  const maxActive = readNumber(previous.workerBudget, 'maxActive') ?? 2
  const maxStarts = readNumber(previous.workerBudget, 'maxStarts') ?? 4
  return { maxActive, maxStarts }
}

function readString(record: unknown, key: string): string | null {
  if (!record || typeof record !== 'object') return null
  const value = (record as Record<string, unknown>)[key]
  return typeof value === 'string' ? value : null
}

function readNumber(record: unknown, key: string): number | null {
  if (!record || typeof record !== 'object') return null
  const value = (record as Record<string, unknown>)[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function normalizeResumeAfterRestart(value: unknown, status?: JobStatus): JobResumeAfterRestart {
  if (value === 'never' || value === 'ask' || value === 'read_only' || value === 'low_risk' || value === 'full_policy' || value === 'pending_approval') return value
  if (value === 'manual') return 'ask'
  if (value === 'auto') return 'full_policy'
  return status === 'waiting_approval' ? 'pending_approval' : 'ask'
}

function normalizeExecutionVerification(value: unknown): JobExecutionVerification | null {
  return value === 'live' || value === 'unverifiable' || value === 'exited' ? value : null
}

function normalizePendingApprovals(input: unknown, now: number): JobPendingApproval[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((approval) => approval && typeof approval === 'object' && typeof (approval as JobPendingApproval).callId === 'string')
    .map((approval) => {
      const item = approval as Partial<JobPendingApproval>
      return {
        id: typeof item.id === 'string' && item.id ? item.id : randomUUID(),
        callId: item.callId!,
        threadId: item.threadId ?? null,
        itemId: item.itemId ?? null,
        tool: typeof item.tool === 'string' && item.tool ? item.tool : 'tool',
        policy: {
          scope: item.policy?.scope === 'connector' ? 'connector' : 'tool',
          key: typeof item.policy?.key === 'string' ? item.policy.key : (typeof item.tool === 'string' ? item.tool : 'tool'),
          label: typeof item.policy?.label === 'string' ? item.policy.label : (typeof item.tool === 'string' ? item.tool : 'Tool approval'),
          explanation: typeof item.policy?.explanation === 'string' ? item.policy.explanation : 'Approval restored from a durable job record.',
        },
        requestedAt: typeof item.requestedAt === 'number' && Number.isFinite(item.requestedAt) ? item.requestedAt : now,
        resolvedAt: typeof item.resolvedAt === 'number' && Number.isFinite(item.resolvedAt) ? item.resolvedAt : null,
        decision: item.decision === 'allow' || item.decision === 'allow_always' || item.decision === 'deny' ? item.decision : null,
      }
    })
}
