import type { AppState, View } from './store/types'
import { threadPendingApproval } from './thread-approval'

export interface FullWorkspaceActivity { key: string; title: string; state: 'Working' | 'Queued' | 'Waiting' | 'Needs you' | 'Needs attention'; needsYou: boolean; updatedAt: number; view: View }
type ActivityState = Pick<AppState, 'threads' | 'status' | 'approvals' | 'jobs' | 'rooms' | 'roomStatus' | 'automations'>
const busy = new Set(['thinking', 'streaming', 'tool', 'awaiting', 'running'])
const activeRooms = new Set(['queued', 'running', 'waiting_user', 'waiting_external'])

/** Recorded activity only; this neither polls nor starts/stops any underlying work. */
export function fullWorkspaceActivity(input: ActivityState): FullWorkspaceActivity[] {
  const rows = new Map<string, FullWorkspaceActivity>()
  const threadsById = new Map(input.threads.filter((thread) => !thread.archivedAt).map((thread) => [thread.id, thread]))
  const roomsById = new Map(input.rooms.filter((room) => !room.archivedAt).map((room) => [room.id, room]))
  const routinesById = new Map(input.automations.map((routine) => [routine.id, routine]))
  const approvalThreads = new Set(input.jobs.flatMap((job) => (job.pendingApprovals ?? []).filter((item) => !item.resolvedAt).map((item) => item.threadId ?? job.currentThreadId ?? job.threadId).filter((id): id is string => !!id)))
  const add = (row: FullWorkspaceActivity) => {
    const old = rows.get(row.key)
    rows.set(row.key, old ? { ...old, ...(row.needsYou ? { state: row.state } : {}), needsYou: old.needsYou || row.needsYou, updatedAt: Math.max(old.updatedAt, row.updatedAt) } : row)
  }
  for (const thread of input.threads) {
    if (thread.archivedAt) continue
    const status = input.status[thread.id] ?? thread.status
    const liveApprovals = input.approvals[thread.id] ?? []
    const approval = (liveApprovals.length || approvalThreads.has(thread.id)) ? threadPendingApproval(thread.id, thread.timeline, liveApprovals, input.jobs) : null
    if (!busy.has(status) && !approval) continue
    const needsYou = status === 'awaiting' || !!approval
    add({ key: `thread:${thread.id}`, title: thread.title || 'Conversation', state: needsYou ? 'Needs you' : 'Working', needsYou, updatedAt: thread.updatedAt, view: { name: 'thread', id: thread.id } })
  }
  for (const room of input.rooms) {
    if (room.archivedAt) continue
    const state = input.roomStatus[room.id] ?? room.state
    if (!activeRooms.has(state)) continue
    const needsYou = state === 'waiting_user'
    add({ key: `room:${room.id}`, title: room.name || 'Room', state: needsYou ? 'Needs you' : state === 'waiting_external' ? 'Waiting' : state === 'queued' ? 'Queued' : 'Working', needsYou, updatedAt: room.updatedAt, view: { name: 'room', id: room.id } })
  }
  for (const routine of input.automations) {
    if (routine.runs[0]?.status !== 'running') continue
    add({ key: `automation:${routine.id}`, title: routine.name || 'Routine', state: 'Working', needsYou: false, updatedAt: routine.updatedAt, view: { name: 'automations' } })
  }
  for (const job of input.jobs) {
    if (!['queued', 'running', 'waiting_approval', 'stalled'].includes(job.status)) continue
    const threadId = job.threadId ?? job.currentThreadId
    const thread = threadId ? threadsById.get(threadId) : null
    const room = job.roomId ? roomsById.get(job.roomId) : null
    const routine = job.automationId ? routinesById.get(job.automationId) : null
    const key = job.kind === 'automation' && routine ? `automation:${routine.id}` : job.kind === 'room' && room ? `room:${room.id}` : thread ? `thread:${thread.id}` : `job:${job.id}`
    const needsYou = job.status === 'waiting_approval' || job.status === 'stalled'
    add({ key, title: routine?.name || room?.name || thread?.title || 'Background work', state: job.status === 'waiting_approval' ? 'Needs you' : job.status === 'stalled' ? 'Needs attention' : job.status === 'queued' ? 'Queued' : 'Working', needsYou, updatedAt: job.updatedAt, view: thread ? { name: 'thread', id: thread.id } : room ? { name: 'room', id: room.id } : routine ? { name: 'automations' } : { name: 'commandCentre' } })
  }
  return [...rows.values()].sort((a, b) => Number(b.needsYou) - Number(a.needsYou) || b.updatedAt - a.updatedAt)
}
