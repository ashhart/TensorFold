import type { CliSession } from './types'

// The renderer's view of live terminal sessions, keyed by project id. The main process owns the
// real PTYs; this map is a renderer cache kept in sync by the `terminal:sessions` broadcast and the
// store actions. Pure helpers so the reconciliation is unit-tested independently of the store.
export type TerminalMap = Record<string, CliSession[]>

export interface TerminalActivitySummary {
  total: number
  running: number
  attention: number
}

export type TerminalSessionRoute =
  | { kind: 'local' }
  | { kind: 'orchestrator'; hostId: string }
  | { kind: 'unresolved' }

/**
 * Resolve an EXISTING terminal by its immutable session origin. Never consult project/global
 * execution defaults here: those settings only select the host for a newly-created terminal.
 */
export function terminalSessionRoute(session: CliSession | null | undefined): TerminalSessionRoute {
  if (!session) return { kind: 'unresolved' }
  if (session.hostLocation !== 'orchestrator') return { kind: 'local' }
  const hostId = session.orchestratorHostId?.trim()
  return hostId ? { kind: 'orchestrator', hostId } : { kind: 'unresolved' }
}

export function stampOrchestratorSession(session: CliSession, hostId: string): CliSession {
  return { ...session, hostLocation: 'orchestrator', managedBy: 'local', orchestratorHostId: hostId }
}

export function terminalEventMatchesSession(session: CliSession | null | undefined, orchestratorHostId: string | null): boolean {
  const route = terminalSessionRoute(session)
  return orchestratorHostId
    ? route.kind === 'orchestrator' && route.hostId === orchestratorHostId
    : route.kind === 'local'
}

export function terminalSessionKey(session: CliSession): string {
  const route = terminalSessionRoute(session)
  if (route.kind === 'local') return `local:${session.id}`
  if (route.kind === 'orchestrator') return `orchestrator:${route.hostId}:${session.id}`
  return `unresolved:${session.id}`
}

export function findTerminalSession(sessions: CliSession[], keyOrId: string): CliSession | undefined {
  const exact = sessions.find((session) => terminalSessionKey(session) === keyOrId)
  if (exact) return exact
  const byRawId = sessions.filter((session) => session.id === keyOrId)
  return byRawId.length === 1 ? byRawId[0] : undefined
}

/**
 * Replace one origin's authoritative list while retaining sessions owned by every other origin.
 * This prevents a local `terminal:sessions` broadcast from erasing host A/B sessions (and vice
 * versa). Legacy remote rows without a host id cannot be attributed safely, so no source is
 * allowed to delete them implicitly.
 */
export function reconcileSessionsForOrigin(
  map: TerminalMap,
  projectId: string,
  sessions: CliSession[],
  orchestratorHostId: string | null,
): TerminalMap {
  const incoming = orchestratorHostId
    ? sessions.map((session) => stampOrchestratorSession(session, orchestratorHostId))
    : sessions.map((session) => ({ ...session, hostLocation: 'local' as const, orchestratorHostId: undefined }))
  const retained = (map[projectId] ?? []).filter((session) => {
    const route = terminalSessionRoute(session)
    if (orchestratorHostId) return route.kind !== 'orchestrator' || route.hostId !== orchestratorHostId
    return route.kind !== 'local'
  })
  return { ...map, [projectId]: [...retained, ...incoming] }
}

export function summarizeTerminalActivity(sessions: CliSession[]): TerminalActivitySummary {
  const live = [...new Map(sessions.filter((session) => session.status !== 'done').map((session) => [terminalSessionKey(session), session])).values()]
  return {
    total: live.length,
    running: live.filter((session) => session.status === 'busy').length,
    attention: live.filter((session) => session.status === 'attention').length,
  }
}

export function setSessions(map: TerminalMap, projectId: string, sessions: CliSession[]): TerminalMap {
  return { ...map, [projectId]: sessions }
}

export function removeSession(map: TerminalMap, projectId: string, sessionId: string): TerminalMap {
  const list = map[projectId]
  if (!list) return map
  return { ...map, [projectId]: list.filter((s) => s.id !== sessionId) }
}

export function removeSessionByKey(map: TerminalMap, projectId: string, keyOrId: string): TerminalMap {
  const list = map[projectId]
  if (!list) return map
  const target = findTerminalSession(list, keyOrId)
  if (!target) return map
  const key = terminalSessionKey(target)
  return { ...map, [projectId]: list.filter((session) => terminalSessionKey(session) !== key) }
}

export function renameSession(map: TerminalMap, projectId: string, sessionId: string, title: string): TerminalMap {
  const list = map[projectId]
  if (!list) return map
  return { ...map, [projectId]: list.map((s) => (s.id === sessionId ? { ...s, title } : s)) }
}

export function renameSessionByKey(map: TerminalMap, projectId: string, keyOrId: string, title: string): TerminalMap {
  const list = map[projectId]
  if (!list) return map
  const target = findTerminalSession(list, keyOrId)
  if (!target) return map
  const key = terminalSessionKey(target)
  return { ...map, [projectId]: list.map((session) => terminalSessionKey(session) === key ? { ...session, title } : session) }
}

export function nextActiveKeyAfterClose(sessions: CliSession[], closedKeyOrId: string): string | null {
  const target = findTerminalSession(sessions, closedKeyOrId)
  if (!target) return null
  const closedKey = terminalSessionKey(target)
  const idx = sessions.findIndex((session) => terminalSessionKey(session) === closedKey)
  const next = sessions.filter((session) => terminalSessionKey(session) !== closedKey)
  if (next.length === 0) return null
  return terminalSessionKey(next[idx] ?? next[idx - 1] ?? next[0])
}

export function updateSessionStatus(
  map: TerminalMap,
  sessionId: string,
  status: CliSession['status'],
  orchestratorHostId: string | null = null,
): TerminalMap {
  let changed = false
  const next = Object.fromEntries(Object.entries(map).map(([projectId, sessions]) => [
    projectId,
    sessions.map((session) => session.id === sessionId && terminalEventMatchesSession(session, orchestratorHostId) && session.status !== status
      ? (changed = true, { ...session, status })
      : session),
  ]))
  return changed ? next : map
}

// When the active terminal is closed, fall back to the session that slides into its slot, then the
// previous one, then the first remaining — mirroring a tabbed editor's close behaviour.
export function nextActiveAfterClose(sessions: CliSession[], closedId: string): string | null {
  const idx = sessions.findIndex((s) => s.id === closedId)
  const next = sessions.filter((s) => s.id !== closedId)
  if (next.length === 0) return null
  return (next[idx] ?? next[idx - 1] ?? next[0]).id
}
