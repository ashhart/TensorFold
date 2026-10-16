import type {
  ConnectorAttentionItem,
  ConnectorBriefing,
  ConnectorEvent,
  ConnectorFeedAction,
  ConnectorMemoryCandidateDraft,
  ConnectorMissionTrigger,
  ConnectorSourceRef,
  Mission,
  MissionStatus,
  Project,
  ProjectMemoryCandidate,
} from './types'
import type { ProjectMemoryCandidateInput } from './project-memory'
import { coalesceThreadBursts } from './thread-address'

type ProjectRef = Pick<Project, 'id' | 'cwd' | 'archivedAt'>
type ExistingMemoryCandidate = Pick<ProjectMemoryCandidate, 'projectId' | 'sourceRef' | 'status'> & Partial<Pick<ProjectMemoryCandidate, 'text'>>

export interface ConnectorFeedOptions {
  projects?: ProjectRef[]
  missions?: Mission[]
  existingCandidates?: ExistingMemoryCandidate[]
  defaultProjectId?: string | null
  now?: number
  limit?: number
}

const ACTIVE_MISSION_STATUSES = new Set<MissionStatus>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])

function attentionItem(event: ConnectorEvent, count = 1, burst = false): ConnectorAttentionItem {
  return {
    id: `attention:${event.id}`,
    eventId: event.id,
    connectorId: event.connectorId,
    threadKey: event.threadKey,
    connectorAction: event.action,
    title: event.title,
    summary: event.summary,
    actor: event.actor || event.pluginName,
    action: actionFor(event),
    priority: event.priority,
    requiresApproval: event.requiresApproval,
    sourceIds: sourceIdsFor(event),
    sources: sourceRefsFor(event),
    createdAt: event.createdAt,
    count,
    burst,
  }
}

/**
 * Build the attention feed, COALESCING a chatty conversation into one "N messages" signal
 * via coalesceThreadBursts (thread-address.ts) instead of one row per message. Events that
 * REQUIRE APPROVAL are kept as individual rows — each is its own actionable decision and
 * must never be hidden behind a burst count. Everything else is grouped per conversation;
 * the row carries the latest event plus the conversation's message `count`/`burst` flag.
 */
export function connectorAttentionItems(events: ConnectorEvent[], limit = 24, windowMs = 60_000): ConnectorAttentionItem[] {
  const ordered = latest(events)
  const approvals = ordered.filter((event) => event.requiresApproval)
  const rest = ordered.filter((event) => !event.requiresApproval)
  const bursts = coalesceThreadBursts(rest, windowMs)
  const items = [
    ...approvals.map((event) => attentionItem(event)),
    ...bursts.map((burst) => attentionItem(burst.latest, burst.count, burst.burst)),
  ].sort((a, b) => b.createdAt - a.createdAt)
  return items.slice(0, limit)
}

export function buildConnectorBriefing(events: ConnectorEvent[], now = Date.now(), limit = 8): ConnectorBriefing {
  const items = latest(events).slice(0, limit)
  const pendingApprovalCount = items.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length
  const sourceIds = unique(items.flatMap(sourceIdsFor))
  const sources = uniqueSources(items.flatMap(sourceRefsFor))
  const lines = items.map((event) => {
    const citations = sourceIdsFor(event).map((id) => `[${id}]`).join(' ')
    return `${citations} ${event.title}: ${event.summary}`.trim()
  })
  return {
    generatedAt: now,
    summary: `${items.length} connector signal${items.length === 1 ? '' : 's'}${pendingApprovalCount ? `, ${pendingApprovalCount} requiring approval` : ''}.`,
    lines,
    sourceIds,
    sources,
    eventCount: items.length,
    pendingApprovalCount,
    latestEventAt: items[0]?.createdAt ?? null,
  }
}

export function connectorMemoryCandidateDrafts(events: ConnectorEvent[], options: ConnectorFeedOptions = {}): ConnectorMemoryCandidateDraft[] {
  const projects = activeProjects(options.projects ?? [])
  const existing = options.existingCandidates ?? []
  const drafts: ConnectorMemoryCandidateDraft[] = []
  const max = options.limit ?? 8
  for (const event of latest(events)) {
    const match = projectForEvent(event, projects, options.defaultProjectId)
    if (!match || !isMemoryWorthy(event, match.byPath)) continue
    const sourceIds = sourceIdsFor(event)
    const sourceRef = sourceIds[0]
    if (!sourceRef || existing.some((candidate) => candidate.projectId === match.project.id && candidate.sourceRef === sourceRef && candidate.status !== 'rejected')) continue
    drafts.push({
      id: `connector-memory:${match.project.id}:${event.id}`,
      eventId: event.id,
      connectorId: event.connectorId,
      projectId: match.project.id,
      kind: event.classification === 'schedule' ? 'decision' : event.classification === 'document' ? 'summary' : 'gotcha',
      text: compactText(`${connectorLabel(event)} ${event.classification}: ${event.title}. ${event.summary}`, 320),
      sourceType: 'connector',
      sourceRef,
      sourceIds,
      confidence: match.byPath ? 'high' : 'medium',
      privacyScope: 'project',
      reason: `Source-backed connector event from ${connectorLabel(event)}${match.byPath ? ' matched this project path' : ' matched the active project'}.`,
      createdAt: options.now ?? event.createdAt,
    })
    if (drafts.length >= max) break
  }
  return drafts
}

export function connectorProjectMemoryCandidateInputs(events: ConnectorEvent[], options: ConnectorFeedOptions = {}): ProjectMemoryCandidateInput[] {
  return connectorMemoryCandidateDrafts(events, options).map((draft) => ({
    projectId: draft.projectId,
    kind: draft.kind,
    text: draft.text,
    sourceType: draft.sourceType,
    sourceRef: draft.sourceRef,
    sourceIds: draft.sourceIds,
    confidence: draft.confidence,
    privacyScope: draft.privacyScope,
    reason: draft.reason,
  }))
}

export function connectorMissionTriggers(events: ConnectorEvent[], missions: Mission[] = [], limit = 12): ConnectorMissionTrigger[] {
  const activeMissions = missions.filter((mission) => ACTIVE_MISSION_STATUSES.has(mission.status))
  const triggers: ConnectorMissionTrigger[] = []
  for (const event of latest(events)) {
    if (!isTriggerWorthy(event)) continue
    for (const mission of activeMissions) {
      if (!missionMatchesEvent(mission, event)) continue
      triggers.push({
        id: `mission-trigger:${mission.id}:${event.id}`,
        missionId: mission.id,
        missionTitle: mission.title,
        eventId: event.id,
        connectorId: event.connectorId,
        title: event.title,
        summary: `${event.title}: ${event.summary}`,
        action: event.requiresApproval ? 'Review' : 'Attach',
        sourceIds: sourceIdsFor(event),
        sources: sourceRefsFor(event),
        createdAt: event.createdAt,
      })
      if (triggers.length >= limit) return triggers
    }
  }
  return triggers
}

function latest(events: ConnectorEvent[]): ConnectorEvent[] {
  return [...events].sort((a, b) => b.createdAt - a.createdAt)
}

function actionFor(event: ConnectorEvent): ConnectorFeedAction {
  if (event.requiresApproval) return 'Review'
  if (event.classification === 'document') return 'Compare'
  if (event.classification === 'schedule') return 'Plan'
  return 'Summarize'
}

function connectorLabel(event: ConnectorEvent): string {
  return event.pluginName || event.connectorId
}

function sourceRefsFor(event: ConnectorEvent): ConnectorSourceRef[] {
  return uniqueSources(event.sources ?? [])
}

function sourceIdsFor(event: ConnectorEvent): string[] {
  const ids = sourceRefsFor(event).map((source) => source.id).filter(Boolean)
  return ids.length ? unique(ids) : [`connector:${event.id}`]
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))]
}

function uniqueSources(sources: ConnectorSourceRef[]): ConnectorSourceRef[] {
  const seen = new Set<string>()
  const result: ConnectorSourceRef[] = []
  for (const source of sources) {
    if (!source.id || seen.has(source.id)) continue
    seen.add(source.id)
    result.push(source)
  }
  return result
}

function activeProjects(projects: ProjectRef[]): ProjectRef[] {
  return projects.filter((project) => !project.archivedAt && project.cwd.trim())
}

function projectForEvent(event: ConnectorEvent, projects: ProjectRef[], defaultProjectId?: string | null): { project: ProjectRef; byPath: boolean } | null {
  const paths = eventPaths(event)
  for (const project of projects) {
    if (paths.some((item) => isPathInside(item, project.cwd))) return { project, byPath: true }
  }
  const defaultProject = defaultProjectId ? projects.find((project) => project.id === defaultProjectId) : null
  if (defaultProject) return { project: defaultProject, byPath: false }
  return null
}

function eventPaths(event: ConnectorEvent): string[] {
  const paths = (event.sources ?? []).flatMap((source) => source.path ? [source.path] : [])
  const payloadPath = typeof event.payload?.path === 'string' ? event.payload.path : null
  return unique([...paths, ...(payloadPath ? [payloadPath] : [])])
}

function isPathInside(candidate: string, cwd: string): boolean {
  const root = trimTrailingSlash(cwd)
  const value = trimTrailingSlash(candidate)
  return value === root || value.startsWith(`${root}/`)
}

function trimTrailingSlash(value: string): string {
  return value.trim().replace(/\/+$/, '')
}

function isMemoryWorthy(event: ConnectorEvent, byPath: boolean): boolean {
  if (event.requiresApproval || event.priority === 'critical' || event.priority === 'high') return true
  if (byPath && (event.classification === 'document' || event.classification === 'schedule')) return true
  return false
}

function isTriggerWorthy(event: ConnectorEvent): boolean {
  return event.requiresApproval
    || event.priority === 'critical'
    || event.priority === 'high'
    || event.classification === 'schedule'
    || event.classification === 'document'
}

function missionMatchesEvent(mission: Mission, event: ConnectorEvent): boolean {
  // Connector titles, summaries, actors, and thread keys are attacker-controlled for
  // inbound email/chat/document sources. They must never become an implicit mission
  // trigger merely because they repeat words from a mission objective. A mission opts
  // into the whole connector feed explicitly, or an event proves a structural path
  // relationship to the mission workspace.
  if (mission.cwd && eventPaths(event).some((item) => isPathInside(item, mission.cwd!))) return true
  return mission.metadata?.connectorEventFeed === true
}

function compactText(value: string, max: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`
}
