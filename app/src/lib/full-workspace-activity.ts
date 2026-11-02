import type { AppState, View } from './store/types'
import { threadPendingApproval } from './thread-approval'

export type WorkActivityReason = 'approval' | 'question' | 'input' | 'recovery' | 'stalled' | 'error' | 'running' | 'queued' | 'waiting'
export interface FullWorkspaceActivity {
  key: string
  title: string
  state: 'Working' | 'Queued' | 'Waiting' | 'Needs you' | 'Needs attention'
  reason: WorkActivityReason
  kind: 'task' | 'chat' | 'room' | 'automation' | 'job'
  projectName?: string
  needsYou: boolean
  updatedAt: number
  view: View
}
type ActivityState = Pick<AppState, 'threads' | 'status' | 'approvals' | 'jobs' | 'rooms' | 'roomStatus' | 'automations'> & Partial<Pick<AppState, 'projects'>>
const busy = new Set(['thinking', 'streaming', 'tool', 'awaiting', 'running'])
const activeRooms = new Set(['queued', 'running', 'waiting_user', 'waiting_external'])
const priority: Record<WorkActivityReason, number> = { approval: 8, question: 7, recovery: 6, stalled: 5, error: 4, input: 3, waiting: 2, running: 1, queued: 0 }

// Recorded activity only; this neither polls nor starts/stops any underlying work.
export function fullWorkspaceActivity(input: ActivityState): FullWorkspaceActivity[] {
  const rows = new Map<string, FullWorkspaceActivity>()
  const projectNames = new Map((input.projects ?? []).map((project) => [project.id, project.name]))
  const threadsById = new Map(input.threads.filter((thread) => !thread.archivedAt).map((thread) => [thread.id, thread]))
  const roomsById = new Map(input.rooms.filter((room) => !room.archivedAt).map((room) => [room.id, room]))
  const routinesById = new Map(input.automations.map((routine) => [routine.id, routine]))
  const jobs = input.jobs.filter((job) => !job.archivedAt)
  const approvalThreads = new Set(jobs.flatMap((job) => (job.pendingApprovals ?? []).filter((item) => !item.resolvedAt).map((item) => item.threadId ?? job.currentThreadId ?? job.threadId).filter((id): id is string => !!id)))
  const add = (row: FullWorkspaceActivity) => {
    const old = rows.get(row.key)
    const stronger = old && priority[old.reason] >= priority[row.reason] ? old : row
    rows.set(row.key, old ? { ...stronger, updatedAt: Math.max(old.updatedAt, row.updatedAt) } : row)
  }
  for (const thread of input.threads) {
    if (thread.archivedAt) continue
    const status = input.status[thread.id] ?? thread.status
    const liveApprovals = input.approvals[thread.id] ?? []
    const approval = (liveApprovals.length || approvalThreads.has(thread.id)) ? threadPendingApproval(thread.id, thread.timeline, liveApprovals, jobs) : null
    const question = thread.timeline.some((item) => item.question?.state === 'pending' && !item.question.resolvedAt)
    const recovery = thread.runRecovery?.canStartNewRun && thread.runRecovery.retry === 'requires_review' && thread.runRecoveryReviewedId !== thread.runRecovery.runId
    const reason: WorkActivityReason = approval ? 'approval' : question ? 'question' : recovery ? 'recovery' : status === 'awaiting' ? 'input' : status === 'error' ? 'error' : 'running'
    if (!busy.has(status) && reason === 'running') continue
    const needsYou = reason !== 'running'
    add({ key: `thread:${thread.id}`, title: thread.title || 'Conversation', state: needsYou ? 'Needs you' : 'Working', reason, kind: thread.kind === 'task' ? 'task' : 'chat', projectName: projectNames.get(thread.projectId ?? ''), needsYou, updatedAt: thread.updatedAt, view: { name: 'thread', id: thread.id } })
  }
  for (const room of input.rooms) {
    if (room.archivedAt) continue
    const state = input.roomStatus[room.id] ?? room.state
    if (!activeRooms.has(state)) continue
    const needsYou = state === 'waiting_user'
    const reason = needsYou ? 'input' : state === 'waiting_external' ? 'waiting' : state === 'queued' ? 'queued' : 'running'
    add({ key: `room:${room.id}`, title: room.name || 'Room', state: needsYou ? 'Needs you' : state === 'waiting_external' ? 'Waiting' : state === 'queued' ? 'Queued' : 'Working', reason, kind: 'room', projectName: projectNames.get(room.projectId), needsYou, updatedAt: room.updatedAt, view: { name: 'room', id: room.id } })
  }
  for (const routine of input.automations) {
    if (routine.runs[0]?.status !== 'running') continue
    add({ key: `automation:${routine.id}`, title: routine.name || 'Routine', state: 'Working', reason: 'running', kind: 'automation', needsYou: false, updatedAt: routine.updatedAt, view: { name: 'automations' } })
  }
  for (const job of jobs) {
    if (!['queued', 'running', 'waiting_approval', 'stalled'].includes(job.status)) continue
    const threadId = job.currentThreadId ?? job.threadId
    const thread = threadId ? threadsById.get(threadId) : null
    const room = job.roomId ? roomsById.get(job.roomId) : null
    const routine = job.automationId ? routinesById.get(job.automationId) : null
    const key = job.kind === 'automation' && routine ? `automation:${routine.id}` : job.kind === 'room' && room ? `room:${room.id}` : thread ? `thread:${thread.id}` : `job:${job.id}`
    const needsYou = job.status === 'waiting_approval' || job.status === 'stalled'
    const reason = job.status === 'waiting_approval' ? 'approval' : job.status === 'stalled' ? 'stalled' : job.status === 'queued' ? 'queued' : 'running'
    const kind = key.startsWith('automation:') ? 'automation' : key.startsWith('room:') ? 'room' : thread ? thread.kind === 'task' ? 'task' : 'chat' : 'job'
    const title = kind === 'automation' ? routine!.name : kind === 'room' ? room!.name : thread?.title
    const view: View = job.status === 'waiting_approval' && thread ? { name: 'thread', id: thread.id } : kind === 'automation' ? { name: 'automations' } : kind === 'room' ? { name: 'room', id: room!.id } : thread ? { name: 'thread', id: thread.id } : { name: 'commandCentre' }
    add({ key, title: title || 'Background work', state: job.status === 'waiting_approval' ? 'Needs you' : job.status === 'stalled' ? 'Needs attention' : job.status === 'queued' ? 'Queued' : 'Working', reason, kind, projectName: projectNames.get(job.projectId ?? thread?.projectId ?? room?.projectId ?? ''), needsYou, updatedAt: job.updatedAt, view })
  }
  return [...rows.values()].sort((a, b) => Number(b.needsYou) - Number(a.needsYou) || b.updatedAt - a.updatedAt)
}
