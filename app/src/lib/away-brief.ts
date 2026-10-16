import type { Automation, AutomationRun, ConnectorEvent, Job, Mission, Room } from './types'

export type AwayBriefKind = 'connector' | 'room' | 'mission' | 'automation' | 'pr_loop'
export type AwayBriefTone = 'critical' | 'high' | 'normal' | 'low' | 'good'

export interface AwayBriefItem {
  id: string
  kind: AwayBriefKind
  subjectId: string
  title: string
  detail: string
  tone: AwayBriefTone
  createdAt: number
  sourceIds: string[]
  status?: string
  connectorId?: ConnectorEvent['connectorId']
  roomId?: string
  missionId?: string
  automationId?: string
  jobId?: string
}

export interface AwayBriefInput {
  now?: number
  since?: number
  windowMs?: number
  limit?: number
  connectorEvents?: ConnectorEvent[]
  rooms?: Room[]
  roomStatus?: Record<string, Room['state'] | undefined>
  missions?: Mission[]
  automations?: Automation[]
  jobs?: Job[]
}

const DEFAULT_WINDOW_MS = 24 * 60 * 60_000
const ACTIVE_ROOM_STATES = new Set<Room['state']>(['queued', 'running'])
const BLOCKED_ROOM_STATES = new Set<Room['state']>(['failed', 'waiting_user', 'waiting_external', 'error'])
const ACTIVE_MISSION_STATUSES = new Set<Mission['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])
const ACTIVE_JOB_STATUSES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])
const BLOCKED_JOB_STATUSES = new Set<Job['status']>(['failed', 'dead_letter'])

export function buildAwayBriefItems(input: AwayBriefInput): AwayBriefItem[] {
  const now = input.now ?? Date.now()
  const since = input.since ?? now - (input.windowMs ?? DEFAULT_WINDOW_MS)
  const rows: AwayBriefItem[] = []

  for (const event of input.connectorEvents ?? []) {
    if (event.createdAt < since && !pendingConnectorApproval(event)) continue
    rows.push({
      id: `connector:${event.id}`,
      kind: 'connector',
      subjectId: event.id,
      title: event.title,
      detail: event.summary,
      tone: event.requiresApproval ? 'critical' : priorityTone(event.priority),
      createdAt: event.createdAt,
      sourceIds: connectorSourceIds(event),
      status: event.requiresApproval ? (event.approvalStatus ?? 'pending') ?? 'pending' : event.priority,
      connectorId: event.connectorId,
    })
  }

  for (const room of input.rooms ?? []) {
    const state = input.roomStatus?.[room.id] ?? room.state
    if (room.updatedAt < since && !ACTIVE_ROOM_STATES.has(state) && !BLOCKED_ROOM_STATES.has(state)) continue
    rows.push({
      id: `room:${room.id}`,
      kind: 'room',
      subjectId: room.id,
      title: `${roomStateLabel(state)}: ${room.name}`,
      detail: room.problem || room.finalText || 'Room changed state.',
      tone: roomTone(state),
      createdAt: room.updatedAt,
      sourceIds: [`room:${room.id}`],
      status: state,
      roomId: room.id,
    })
  }

  for (const mission of input.missions ?? []) {
    if (isLocalPrMission(mission)) continue
    if (mission.updatedAt < since && !ACTIVE_MISSION_STATUSES.has(mission.status)) continue
    rows.push(missionRow(mission))
  }

  for (const automation of input.automations ?? []) {
    for (const run of automation.runs.slice(0, 3)) {
      const at = run.finishedAt ?? run.startedAt
      if (at < since && run.status !== 'running') continue
      rows.push(automationRow(automation, run, at))
    }
  }

  const prLoopMissionIds = new Set((input.jobs ?? []).filter((job) => job.kind === 'pr_loop' && job.missionId).map((job) => job.missionId!))
  for (const job of input.jobs ?? []) {
    if (job.kind !== 'pr_loop') continue
    const at = job.finishedAt ?? job.updatedAt
    if (at < since && !ACTIVE_JOB_STATUSES.has(job.status) && !BLOCKED_JOB_STATUSES.has(job.status)) continue
    rows.push(prLoopJobRow(job, at))
  }

  for (const mission of input.missions ?? []) {
    if (!isLocalPrMission(mission) || prLoopMissionIds.has(mission.id)) continue
    if (mission.updatedAt < since && !ACTIVE_MISSION_STATUSES.has(mission.status)) continue
    rows.push(prLoopMissionRow(mission))
  }

  return rows.sort((a, b) => b.createdAt - a.createdAt || a.id.localeCompare(b.id)).slice(0, input.limit ?? 8)
}

function pendingConnectorApproval(event: ConnectorEvent): boolean {
  return event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending'
}

function connectorSourceIds(event: ConnectorEvent): string[] {
  const ids = [...new Set((event.sources ?? []).map((source) => source.id).filter(Boolean))]
  return ids.length ? ids : [`connector:${event.id}`]
}

function priorityTone(priority: ConnectorEvent['priority']): AwayBriefTone {
  if (priority === 'critical') return 'critical'
  if (priority === 'high') return 'high'
  if (priority === 'low') return 'low'
  return 'normal'
}

function roomTone(state: Room['state']): AwayBriefTone {
  if (state === 'failed' || state === 'error') return 'critical'
  if (state === 'waiting_user' || state === 'waiting_external') return 'high'
  if (ACTIVE_ROOM_STATES.has(state)) return 'normal'
  if (state === 'complete') return 'good'
  return 'low'
}

function roomStateLabel(state: Room['state']): string {
  return state.replace(/_/g, ' ')
}

function missionRow(mission: Mission): AwayBriefItem {
  return {
    id: `mission:${mission.id}`,
    kind: 'mission',
    subjectId: mission.id,
    title: `${mission.status.replace(/_/g, ' ')}: ${mission.title}`,
    detail: mission.checkpoints.at(-1)?.summary || mission.objective,
    tone: missionTone(mission.status),
    createdAt: mission.updatedAt,
    sourceIds: [`mission:${mission.id}`],
    status: mission.status,
    missionId: mission.id,
  }
}

function missionTone(status: Mission['status']): AwayBriefTone {
  if (status === 'failed') return 'critical'
  if (status === 'waiting_approval' || status === 'stalled') return 'high'
  if (status === 'completed') return 'good'
  if (status === 'paused' || status === 'queued' || status === 'cancelled' || status === 'draft') return 'low'
  return 'normal'
}

function automationRow(automation: Automation, run: AutomationRun, at: number): AwayBriefItem {
  return {
    id: `automation:${automation.id}:${run.id}`,
    kind: 'automation',
    subjectId: automation.id,
    title: `${automation.name} ${run.status === 'ok' ? 'completed' : run.status}`,
    detail: run.error || firstLine(run.output) || automation.prompt,
    tone: run.status === 'error' ? 'critical' : run.status === 'running' ? 'normal' : 'good',
    createdAt: at,
    sourceIds: [`automation:${automation.id}`, `automation-run:${run.id}`],
    status: run.status,
    automationId: automation.id,
  }
}

function prLoopJobRow(job: Job, at: number): AwayBriefItem {
  const localPrId = stringMeta(job, 'localPrId')
  const branch = stringMeta(job, 'branch')
  return {
    id: `pr_loop:${job.id}`,
    kind: 'pr_loop',
    subjectId: job.id,
    title: `${job.status.replace(/_/g, ' ')}: ${jobTitle(job)}`,
    detail: job.blockedReason || job.resumePlan || branch || localPrId || 'Local PR loop changed state.',
    tone: jobTone(job.status),
    createdAt: at,
    sourceIds: [localPrId, branch, job.missionId ? `mission:${job.missionId}` : null, `job:${job.id}`].filter((value): value is string => Boolean(value)),
    status: job.status,
    jobId: job.id,
    missionId: job.missionId ?? undefined,
  }
}

function prLoopMissionRow(mission: Mission): AwayBriefItem {
  const localPrId = stringMeta(mission, 'localPrId')
  const branch = stringMeta(mission, 'branch')
  return {
    id: `pr_loop:${mission.id}`,
    kind: 'pr_loop',
    subjectId: mission.id,
    title: `${mission.status.replace(/_/g, ' ')}: ${mission.title}`,
    detail: mission.checkpoints.at(-1)?.summary || branch || localPrId || mission.objective,
    tone: missionTone(mission.status),
    createdAt: mission.updatedAt,
    sourceIds: [localPrId, branch, `mission:${mission.id}`].filter((value): value is string => Boolean(value)),
    status: mission.status,
    missionId: mission.id,
  }
}

function isLocalPrMission(mission: Mission): boolean {
  return mission.metadata?.workflow === 'local_pr_loop'
}

function jobTitle(job: Job): string {
  return stringMeta(job, 'title') || stringMeta(job, 'localPrId') || stringMeta(job, 'branch') || 'Local PR loop'
}

function jobTone(status: Job['status']): AwayBriefTone {
  if (status === 'failed' || status === 'dead_letter') return 'critical'
  if (status === 'stalled' || status === 'waiting_approval') return 'high'
  if (status === 'completed') return 'good'
  if (status === 'queued' || status === 'paused' || status === 'cancelled') return 'low'
  return 'normal'
}

function stringMeta(item: { metadata?: Record<string, unknown> }, key: string): string | null {
  const value = item.metadata?.[key]
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function firstLine(value: string): string {
  return value.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? ''
}
