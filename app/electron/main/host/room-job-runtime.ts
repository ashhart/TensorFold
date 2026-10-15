import { addMissionCheckpoint, cancelMission as cancelMissionState, completeMission, createMission, failMission, pauseMission, recordMissionHeartbeat, startMission } from '../../agent/missions'
import { cancelJob, claimJobLease, completeJob, detectStuckJobs, enqueueJob, failJob, heartbeatJobLease, releaseJobLease } from '../../agent/jobs'
import type { JobPatch, RoomSupervisorMetadataPatch } from '../../agent/jobs'
import type { Job, Mission, Project, Room, RoomEventItem, RoomStreamEvent, Ticket } from '../../agent/types'
import { store } from '../state'
import { createRoomSupervisorHelpers } from './room-supervisor-helpers'

export interface RoomJobRuntime {
  activeJob(kind: Job['kind'], owner: string): Job | null
  appendRoomSupervisorEvent(room: Room, text: string, extra?: Partial<RoomEventItem>): RoomEventItem
  claimJob(ref: Parameters<typeof enqueueJob>[1]): Job
  finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  finishRoomSupervisorMission(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  heartbeatJob(job: Job | null): Job | null
  ensureRoomSupervisorMission(room: Room, project: Project, workspaceCwd: string, problem: string): Mission
  parkRoomIfWorkRemains(job: Job | null, room: Room, project: Project | null): boolean
  patchRoomSupervisorJob(job: Job | null, room: Room, project: Project | null, patch?: RoomSupervisorMetadataPatch): Job | null
  reconcileStuckJobs(): void
  rememberJob(ref: Parameters<typeof enqueueJob>[1]): Job
  roomWorkerBudget(room: Room): { maxActive: number; maxStarts: number }
  ticketsForRoom(room: Room): Ticket[]
}

interface RoomJobRuntimeDeps {
  defaultRoomWorkerBudget: { maxActive: number; maxStarts: number }
  jobLeaseOwner: string
  roomWorkerSleepWakeMs: number
  emitRoom(event: RoomStreamEvent): void
  emitJob(job: Job): void
  emitMission(mission: Mission): void
  persistJobs(): void
  persistMissions(): void
  patchJob(job: Job | null, patch: JobPatch): Job | null
  replaceMission(next: Mission): Mission
}

const ACTIVE_JOB_STATES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])
const TERMINAL_JOB_STATES = new Set<Job['status']>(['completed', 'failed', 'cancelled', 'dead_letter'])
const ROOM_SUPERVISOR_MISSION_WORKFLOW = 'room_supervisor'
const TERMINAL_MISSION_STATES = new Set(['completed', 'failed', 'cancelled'])

export function createRoomJobRuntime(deps: RoomJobRuntimeDeps): RoomJobRuntime {
  const {
    defaultRoomWorkerBudget,
    jobLeaseOwner,
    roomWorkerSleepWakeMs,
    emitRoom,
    emitJob,
    emitMission,
    persistJobs,
    persistMissions,
    patchJob,
    replaceMission,
  } = deps

  const roomSupervisorHelpers = createRoomSupervisorHelpers({
    terminalJobStates: TERMINAL_JOB_STATES,
    jobLeaseOwner,
    roomWorkerSleepWakeMs,
    emitRoom,
    emitJob,
    activeRoomTicketWorkers,
    patchJob,
    touchRoomSupervisorMission,
    finishRoomSupervisorMission,
  })

  function appendRoomSupervisorEvent(room: Room, text: string, extra: Partial<RoomEventItem> = {}): RoomEventItem {
    return roomSupervisorHelpers.appendRoomSupervisorEvent(room, text, extra)
  }

  function ticketsForRoom(room: Room): Ticket[] {
    return roomSupervisorHelpers.ticketsForRoom(room)
  }

  function patchRoomSupervisorJob(job: Job | null, room: Room, project: Project | null, patch: RoomSupervisorMetadataPatch = {}): Job | null {
    return roomSupervisorHelpers.patchRoomSupervisorJob(job, room, project, patch)
  }

  function parkRoomIfWorkRemains(job: Job | null, room: Room, project: Project | null): boolean {
    return roomSupervisorHelpers.parkRoomIfWorkRemains(job, room, project)
  }

  function rememberJob(ref: Parameters<typeof enqueueJob>[1]): Job {
    const result = enqueueJob(store.jobs, ref)
    store.jobs = result.jobs
    persistJobs()
    emitJob(result.job)
    return result.job
  }

  function activeJob(kind: Job['kind'], owner: string): Job | null {
    return store.jobs.find((job) => job.kind === kind && job.owner === owner && ACTIVE_JOB_STATES.has(job.status)) ?? null
  }

  function roomWorkerBudget(room: Room): { maxActive: number; maxStarts: number } {
    const configured = room.workerBudget ?? defaultRoomWorkerBudget
    const maxActive = Math.max(1, Math.min(8, Math.floor(Number(configured.maxActive) || defaultRoomWorkerBudget.maxActive)))
    const maxStarts = Math.max(maxActive, Math.min(32, Math.floor(Number(configured.maxStarts) || defaultRoomWorkerBudget.maxStarts)))
    return { maxActive, maxStarts }
  }

  function activeRoomTicketWorkers(tickets: Ticket[]): number {
    return tickets.filter((ticket) => ticket.runState === 'queued' || ticket.runState === 'running').length
  }

  function roomSupervisorMissionFor(roomId: string): Mission | null {
    return store.missions.find((mission) =>
      mission.metadata?.workflow === ROOM_SUPERVISOR_MISSION_WORKFLOW &&
      mission.metadata?.roomId === roomId &&
      !TERMINAL_MISSION_STATES.has(mission.status)
    ) ?? null
  }

  function roomSupervisorMissionMetadata(room: Room, project: Project, workspaceCwd: string, problem: string): Record<string, unknown> {
    return {
      workflow: ROOM_SUPERVISOR_MISSION_WORKFLOW,
      roomId: room.id,
      roomName: room.name,
      roomJobOwner: room.id,
      projectId: room.projectId,
      projectName: project.name,
      workspaceCwd,
      rootCwd: room.rootCwd ?? project.cwd ?? null,
      problem,
      workerBudget: roomWorkerBudget(room),
      seatCount: room.seats.length,
    }
  }

  function ensureRoomSupervisorMission(room: Room, project: Project, workspaceCwd: string, problem: string): Mission {
    const now = Date.now()
    const title = `Room Supervisor: ${room.name || 'Room'}`
    const objective = `Coordinate room "${room.name || room.id}" for: ${problem}`
    const stopCondition = 'All room tickets are closed or non-blocking, required evidence is attached, reviews/tests are resolved, and the room final gate has no blockers.'
    const metadata = roomSupervisorMissionMetadata(room, project, workspaceCwd, problem)
    const existing = roomSupervisorMissionFor(room.id)
    if (existing) {
      const refreshed: Mission = {
        ...existing,
        title,
        objective,
        stopCondition,
        projectId: room.projectId,
        cwd: workspaceCwd,
        metadata: { ...(existing.metadata ?? {}), ...metadata },
        updatedAt: now,
        finishedAt: null,
      }
      const next = refreshed.status === 'running' || refreshed.status === 'waiting_approval'
        ? recordMissionHeartbeat(refreshed, 'Room supervisor mission heartbeat.', now)
        : startMission(refreshed, now)
      return replaceMission(next)
    }

    let mission = createMission({
      title,
      objective,
      stopCondition,
      repeatMode: 'until_condition',
      contextMode: 'build_on_previous',
      roleProfileId: 'operator',
      fanOutBudget: 'council',
      maxRuns: null,
      costCapUsd: null,
      destructiveApprovalRequired: true,
      runBudgetClass: 'mission',
      projectId: room.projectId,
      cwd: workspaceCwd,
      providerId: null,
      providerAccountId: null,
      model: null,
      metadata,
    }, now)
    mission = startMission(mission, now)
    mission = addMissionCheckpoint(mission, {
      label: 'Room supervisor started',
      summary: `Supervising room "${room.name || room.id}" in ${workspaceCwd}.`,
      data: metadata,
    }, now)
    store.missions.unshift(mission)
    persistMissions()
    emitMission(mission)
    return mission
  }

  function roomSupervisorMissionIdFromJob(job: Job | null): string | null {
    if (!job) return null
    if (job.missionId) return job.missionId
    const metadataMissionId = job.metadata?.missionId
    return typeof metadataMissionId === 'string' && metadataMissionId ? metadataMissionId : null
  }

  function touchRoomSupervisorMission(job: Job | null, message = 'Room supervisor mission heartbeat.'): void {
    const missionId = roomSupervisorMissionIdFromJob(job)
    if (!missionId) return
    const mission = store.missions.find((candidate) => candidate.id === missionId)
    if (!mission || TERMINAL_MISSION_STATES.has(mission.status)) return
    replaceMission(recordMissionHeartbeat(mission, message))
  }

  function finishRoomSupervisorMission(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void {
    const missionId = roomSupervisorMissionIdFromJob(job)
    if (!missionId) return
    const mission = store.missions.find((candidate) => candidate.id === missionId)
    if (!mission || TERMINAL_MISSION_STATES.has(mission.status)) return
    const next = outcome === 'completed'
      ? completeMission(mission, reason ?? 'Room supervisor completed.')
      : outcome === 'failed'
        ? failMission(mission, reason ?? 'Room supervisor failed.')
        : outcome === 'cancelled'
          ? cancelMissionState(mission, reason ?? 'Room supervisor cancelled.')
          : pauseMission(mission, reason ?? 'Room supervisor paused.')
    replaceMission(next)
  }

  function claimJob(ref: Parameters<typeof enqueueJob>[1]): Job {
    const pending = rememberJob(ref)
    const result = claimJobLease(store.jobs, pending.id, jobLeaseOwner)
    if (!result.job) return pending
    store.jobs = result.jobs
    persistJobs()
    emitJob(result.job)
    return result.job
  }

  function heartbeatJob(job: Job | null): Job | null {
    if (!job) return null
    const result = heartbeatJobLease(store.jobs, job.id, jobLeaseOwner)
    if (!result.job) return job
    store.jobs = result.jobs
    persistJobs()
    emitJob(result.job)
    return result.job
  }

  function finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void {
    if (!job) return
    const current = store.jobs.find((candidate) => candidate.id === job.id)
    if (current && TERMINAL_JOB_STATES.has(current.status)) return
    const result = outcome === 'completed'
      ? completeJob(store.jobs, job.id)
      : outcome === 'failed'
        ? failJob(store.jobs, job.id, reason ?? 'Job failed.')
        : outcome === 'cancelled'
          ? cancelJob(store.jobs, job.id, reason ?? 'Job cancelled.')
          : releaseJobLease(store.jobs, job.id, 'paused', reason ?? 'Job paused.')
    if (!result.job) return
    store.jobs = result.jobs
    persistJobs()
    emitJob(result.job)
  }

  function reconcileStuckJobs(): void {
    const result = detectStuckJobs(store.jobs)
    if (!result.changed) return
    store.jobs = result.jobs
    persistJobs()
    for (const job of result.stalled) emitJob(job)
  }

  return {
    activeJob,
    appendRoomSupervisorEvent,
    claimJob,
    finishJob,
    finishRoomSupervisorMission,
    heartbeatJob,
    ensureRoomSupervisorMission,
    parkRoomIfWorkRemains,
    patchRoomSupervisorJob,
    reconcileStuckJobs,
    rememberJob,
    roomWorkerBudget,
    ticketsForRoom,
  }
}
