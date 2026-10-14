import type { AgentEvent, Project } from '../../types'
import type { AppState, View } from '../types'

const MAX_PENDING_AGENT_EVENTS_PER_THREAD = 400
const pendingAgentEventsByThread = new Map<string, AgentEvent[]>()
const MAX_PROJECTION_COLLISIONS = 256
const projectionCollisionQuarantine = new Map<string, number>()

export function projectedHostId(record: { orchestratorHostId?: string | null }): string | null {
  return record.orchestratorHostId ?? null
}

export function projectedProjectHostId(project: Pick<Project, 'executionTarget' | 'orchestratorId'>): string | null {
  return project.executionTarget === 'orchestrator' ? project.orchestratorId ?? null : null
}

export function projectionEventKey(id: string, hostId: string | null | undefined): string {
  return `${hostId ?? 'local'}:${id}`
}

export function projectionOriginsMatch(existingHostId: string | null | undefined, incomingHostId: string | null | undefined): boolean {
  return (existingHostId ?? null) === (incomingHostId ?? null)
}

export function quarantineProjectionCollision(kind: 'project' | 'thread' | 'room' | 'ticket' | 'agent event' | 'room event', id: string, existingHostId: string | null, incomingHostId: string | null): string | null {
  const key = `${kind}:${id}:${existingHostId ?? 'local'}:${incomingHostId ?? 'local'}`
  if (projectionCollisionQuarantine.has(key)) return null
  while (projectionCollisionQuarantine.size >= MAX_PROJECTION_COLLISIONS) {
    const oldest = projectionCollisionQuarantine.keys().next().value as string | undefined
    if (!oldest) break
    projectionCollisionQuarantine.delete(oldest)
  }
  projectionCollisionQuarantine.set(key, Date.now())
  return `Ignored a ${kind} id collision for ${id}: ${incomingHostId ?? 'local'} cannot replace ${existingHostId ?? 'local'}.`
}

export function queuePendingAgentEvent(e: AgentEvent) {
  const key = projectionEventKey(e.threadId, e.orchestratorHostId)
  const events = pendingAgentEventsByThread.get(key) ?? []
  events.push(e)
  if (events.length > MAX_PENDING_AGENT_EVENTS_PER_THREAD) events.splice(0, events.length - MAX_PENDING_AGENT_EVENTS_PER_THREAD)
  pendingAgentEventsByThread.set(key, events)
}

export function drainPendingAgentEvents(threadId: string, hostId: string | null = null): AgentEvent[] {
  const key = projectionEventKey(threadId, hostId)
  const events = pendingAgentEventsByThread.get(key) ?? []
  pendingAgentEventsByThread.delete(key)
  return events
}

export function clearPendingAgentEvents(threadId: string, hostId?: string | null): void {
  if (hostId !== undefined) {
    pendingAgentEventsByThread.delete(projectionEventKey(threadId, hostId))
    return
  }
  for (const key of pendingAgentEventsByThread.keys()) if (key.endsWith(`:${threadId}`)) pendingAgentEventsByThread.delete(key)
}

export function removeProjectOwnedProxyState(st: AppState, projectId: string, fallbackView: View = { name: 'projects' }, originHostId?: string | null): Partial<AppState> {
  const matchesOrigin = (record: { orchestratorHostId?: string | null }) => originHostId === undefined || projectionOriginsMatch(projectedHostId(record), originHostId)
  const threadIds = new Set(st.threads.filter((thread) => thread.projectId === projectId && matchesOrigin(thread)).map((thread) => thread.id))
  const roomIds = new Set(st.rooms.filter((room) => room.projectId === projectId && matchesOrigin(room)).map((room) => room.id))
  for (const threadId of threadIds) clearPendingAgentEvents(threadId)

  const status = { ...st.status }
  const approvals = { ...st.approvals }
  for (const threadId of threadIds) {
    delete status[threadId]
    delete approvals[threadId]
  }

  const roomStatus = { ...st.roomStatus }
  const roomLive = { ...st.roomLive }
  for (const roomId of roomIds) {
    delete roomStatus[roomId]
    delete roomLive[roomId]
  }

  const terminals = { ...st.terminals }
  const activeTerminalId = { ...st.activeTerminalId }
  delete terminals[projectId]
  delete activeTerminalId[projectId]

  const view =
    st.view.name === 'project' && st.view.id === projectId ? fallbackView :
    st.view.name === 'thread' && threadIds.has(st.view.id) ? fallbackView :
    st.view.name === 'room' && roomIds.has(st.view.id) ? fallbackView :
    st.view

  return {
    projects: st.projects.filter((project) => project.id !== projectId || (originHostId !== undefined && !projectionOriginsMatch(projectedProjectHostId(project), originHostId))),
    threads: st.threads.filter((thread) => thread.projectId !== projectId || !matchesOrigin(thread)),
    rooms: st.rooms.filter((room) => room.projectId !== projectId || !matchesOrigin(room)),
    tickets: st.tickets.filter((ticket) => ticket.projectId !== projectId || !matchesOrigin(ticket)),
    status,
    approvals,
    roomStatus,
    roomLive,
    terminals,
    activeTerminalId,
    view,
  }
}

export function mergeOriginBoundRecords<T extends { id: string; updatedAt?: number | null }>(loaded: T[], live: T[], hostId: (item: T) => string | null): T[] {
  const byId = new Map<string, T>()
  for (const item of loaded) byId.set(item.id, item)
  for (const item of live) {
    const existing = byId.get(item.id)
    if (existing && !projectionOriginsMatch(hostId(existing), hostId(item))) continue
    byId.set(item.id, existing && (existing.updatedAt ?? 0) > (item.updatedAt ?? 0) ? existing : item)
  }
  return [...byId.values()].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
}

export function mergeRecordsByFreshest<T extends { id: string; updatedAt?: number | null }>(loaded: T[], live: T[]): T[] {
  const byId = new Map<string, T>()
  for (const item of loaded) byId.set(item.id, item)
  for (const item of live) {
    const existing = byId.get(item.id)
    byId.set(item.id, existing && (existing.updatedAt ?? 0) > (item.updatedAt ?? 0) ? existing : item)
  }
  return [...byId.values()].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
}

export function mutableProjectProxyPatch(project: Project): Partial<Project> {
  return {
    name: project.name,
    tags: project.tags,
    kanban: project.kanban,
    localOnly: project.localOnly,
    defaultProviderId: project.defaultProviderId,
    defaultProviderAccountId: project.defaultProviderAccountId,
    defaultModel: project.defaultModel,
    memory: project.memory,
    team: project.team,
  }
}

export function mutableProjectConfigPatch(patch: Partial<Project>): Partial<Project> {
  const config: Partial<Project> = {}
  for (const field of ['name', 'tags', 'kanban', 'localOnly', 'defaultProviderId', 'defaultProviderAccountId', 'defaultModel', 'memory', 'team'] as const) {
    if (Object.prototype.hasOwnProperty.call(patch, field)) (config as any)[field] = patch[field]
  }
  return config
}
