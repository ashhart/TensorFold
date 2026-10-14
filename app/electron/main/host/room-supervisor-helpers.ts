import { randomUUID } from 'node:crypto'
import type { JobPatch, RoomSupervisorMetadataPatch } from '../../agent/jobs'
import { buildRoomSupervisorJobMetadata, heartbeatJobLease, updateJob } from '../../agent/jobs'
import type { Job, Project, Room, RoomEventItem, RoomStreamEvent, Ticket } from '../../agent/types'
import { roomLeftToWork } from '../../agent/tickets'
import { store } from '../state'
import { persistJobs, persistRooms } from './persistence'

interface RoomSupervisorHelperDeps {
  terminalJobStates: Set<Job['status']>
  jobLeaseOwner: string
  roomWorkerSleepWakeMs: number
  emitRoom(event: RoomStreamEvent): void
  emitJob(job: Job): void
  activeRoomTicketWorkers(tickets: Ticket[]): number
  patchJob(job: Job | null, patch: JobPatch): Job | null
  touchRoomSupervisorMission(job: Job | null, message?: string): void
  finishRoomSupervisorMission(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
}

export interface RoomSupervisorHelpers {
  appendRoomSupervisorEvent(room: Room, text: string, extra?: Partial<RoomEventItem>): RoomEventItem
  ticketsForRoom(room: Room): Ticket[]
  patchRoomSupervisorJob(job: Job | null, room: Room, project: Project | null, patch?: RoomSupervisorMetadataPatch): Job | null
  parkRoomIfWorkRemains(job: Job | null, room: Room, project: Project | null): boolean
}

export function createRoomSupervisorHelpers(deps: RoomSupervisorHelperDeps): RoomSupervisorHelpers {
  function appendRoomSupervisorEvent(room: Room, text: string, extra: Partial<RoomEventItem> = {}): RoomEventItem {
    const tick = Math.max(0, ...room.events.map((item) => item.tick || 0)) + 1
    const item: RoomEventItem = {
      id: randomUUID(),
      tick,
      kind: extra.kind ?? 'notice',
      seat: extra.seat ?? 'room-supervisor',
      target: extra.target ?? 'all',
      text,
      tool: extra.tool,
      args: extra.args,
      path: extra.path,
      createdAt: Date.now(),
    }
    room.events.push(item)
    room.updatedAt = Date.now()
    persistRooms()
    deps.emitRoom({ roomId: room.id, type: 'event', item })
    return item
  }

  function ticketsForRoom(room: Room): Ticket[] {
    return store.tickets.filter((ticket) => ticket.projectId === room.projectId && ticket.roomId === room.id)
  }

  function patchRoomSupervisorJob(job: Job | null, room: Room, project: Project | null, patch: RoomSupervisorMetadataPatch = {}): Job | null {
    if (!job) return null
    const current = store.jobs.find((candidate) => candidate.id === job.id)
    if (current && deps.terminalJobStates.has(current.status)) return current
    const now = Date.now()
    const heartbeat = heartbeatJobLease(store.jobs, job.id, deps.jobLeaseOwner, now)
    const baseJobs = heartbeat.job ? heartbeat.jobs : store.jobs
    const currentJob = heartbeat.job ?? current ?? job
    const metadata = buildRoomSupervisorJobMetadata({
      room,
      project,
      tickets: ticketsForRoom(room),
      previous: currentJob.metadata ?? null,
      patch,
    }, now)
    const updated = updateJob(baseJobs, currentJob.id, { lastUsefulProgressAt: now, metadata }, now)
    if (!updated.job) return currentJob
    store.jobs = updated.jobs
    persistJobs()
    deps.emitJob(updated.job)
    deps.touchRoomSupervisorMission(updated.job)
    return updated.job
  }

  function parkRoomIfWorkRemains(job: Job | null, room: Room, project: Project | null): boolean {
    const currentTickets = ticketsForRoom(room)
    const left = roomLeftToWork(currentTickets)
    if (!left.length) return false
    const activeWorkers = deps.activeRoomTicketWorkers(currentTickets)
    const nextWakeAt = activeWorkers > 0 ? Date.now() + deps.roomWorkerSleepWakeMs : null
    const state: Room['state'] = activeWorkers > 0 ? 'waiting_external' : 'waiting_user'
    const reason = activeWorkers > 0
      ? `Sleeping while ${activeWorkers} ticket worker${activeWorkers === 1 ? '' : 's'} finish. Wake the room after worker evidence lands.`
      : `Sleeping with ${left.length} room ticket${left.length === 1 ? '' : 's'} left to work. Press Wake to continue the room loop.`
    room.state = state
    room.updatedAt = Date.now()
    persistRooms()
    deps.emitRoom({ roomId: room.id, type: 'status', state })
    appendRoomSupervisorEvent(room, reason, {
      tool: 'sleep_policy',
      args: { left: left.length, activeWorkers, nextWakeAt },
    })
    const nextJob = patchRoomSupervisorJob(job, room, project, { currentTurnSeat: null })
    deps.patchJob(nextJob ?? job, {
      status: 'paused',
      blockedReason: reason,
      nextWakeAt,
      resumePlan: activeWorkers > 0
        ? 'Review the latest worker evidence, then wake the room to continue from the remaining tickets.'
        : 'Wake the room to continue from the remaining tickets.',
      currentRunId: null,
      leaseOwner: null,
      leaseExpiresAt: null,
    })
    deps.finishRoomSupervisorMission(nextJob ?? job, 'paused', reason)
    return true
  }

  return {
    appendRoomSupervisorEvent,
    ticketsForRoom,
    patchRoomSupervisorJob,
    parkRoomIfWorkRemains,
  }
}
