import {
  saveAutomations,
  saveBotControlSessions,
  saveCommands,
  saveConnectorEvents,
  saveContactDirectory,
  saveIncidents,
  saveJobs,
  saveMcpServers,
  saveMissions,
  saveArtifacts, saveProjectMemory,
  saveProjectMemoryCandidates,
  saveProjects,
  saveRooms,
  saveSettings,
  saveSkills,
  saveSuggestions,
  saveSubAgents,
  saveThreads,
  saveTickets,
  saveWebhooks,
  saveCliSessions,
  writeChains,
} from '../../store'
import { auditPreview } from '../../agent/security'
import type { PersistenceFailure } from '../../agent/types'
import { store } from '../state'
import { sendToRenderer } from './renderer-events'
import { flushAuditEvents, getLastAuditError } from './audit'

// Fire-and-forget persistence of each in-memory collection to its on-disk store. A failed write must
// not crash the main process or interrupt an agent run, but it must NOT be silent either — a swallowed
// write error is silent data loss. Report a redacted, bounded failure to the renderer and retain it
// in memory for readiness/security snapshots. This event deliberately does not persist itself: doing
// so would recurse when the local store is the thing that failed.
export const PERSISTENCE_FAILURE_CHANNEL = 'host:persistence-failure'

export function createPersistenceFailureReporter(deps: {
  now?: () => number
  publish?: (failure: PersistenceFailure) => void
  warn?: (message: string) => void
} = {}) {
  let lastFailure: PersistenceFailure | null = null

  return {
    lastFailure(): PersistenceFailure | null {
      return lastFailure ? { ...lastFailure } : null
    },
    record(label: string, error: unknown): PersistenceFailure {
      const safeLabel = auditPreview(label, 80) || 'local data'
      const safeMessage = auditPreview((error as { message?: unknown } | null)?.message ?? error, 240) || 'Unknown write failure.'
      const failure = { label: safeLabel, message: safeMessage, at: deps.now?.() ?? Date.now() }
      lastFailure = failure
      try { deps.warn?.(`[vontra] persist failed for ${failure.label}: ${failure.message}`) } catch { /* diagnostics cannot break callers */ }
      try { deps.publish?.({ ...failure }) } catch { /* renderer teardown cannot break persistence */ }
      return { ...failure }
    },
  }
}

const persistenceFailures = createPersistenceFailureReporter({
  publish: (failure) => { sendToRenderer(PERSISTENCE_FAILURE_CHANNEL, failure) },
  warn: (message) => console.warn(message),
})

export function lastPersistFailure() { return persistenceFailures.lastFailure() }

export function recordPersistenceFailure(label: string, error: unknown): PersistenceFailure {
  return persistenceFailures.record(label, error)
}

export async function persistSettingsCritical(label: string, settings = store.settings): Promise<void> {
  try {
    await saveSettings(settings)
  } catch (error) {
    persistenceFailures.record(label, error)
    throw error
  }
}

export function persistSettings(label = 'settings', settings = store.settings): void {
  saveSettings(settings).catch(onPersistError(label))
}

function onPersistError(label: string): (e: unknown) => void {
  return (e) => { persistenceFailures.record(label, e) }
}

export function persistThreads() { saveThreads(store.threads).catch(onPersistError('threads')) }
export function persistProjects() { saveProjects(store.projects).catch(onPersistError('projects')) }
export function persistRooms() { saveRooms(store.rooms).catch(onPersistError('rooms')) }
export function persistTickets() { saveTickets(store.tickets).catch(onPersistError('tickets')) }
export function persistAutomations() { saveAutomations(store.automations).catch(onPersistError('automations')) }
export function persistConnectorEvents() { saveConnectorEvents(store.connectorEvents).catch(onPersistError('connectorEvents')) }
export function persistContacts() { saveContactDirectory(store.contacts).catch(onPersistError('contacts')) }
export function persistMissions() { saveMissions(store.missions).catch(onPersistError('missions')) }
export function persistJobs() { saveJobs(store.jobs).catch(onPersistError('jobs')) }
export function persistSubAgents() { saveSubAgents(store.subAgents).catch(onPersistError('subAgents')) }
export function persistMcpServers() { saveMcpServers(store.mcpServers).catch(onPersistError('mcpServers')) }
export function persistBotSessions() { saveBotControlSessions(store.botSessions).catch(onPersistError('botSessions')) }
export function persistSkills() { saveSkills(store.skills).catch(onPersistError('skills')) }
export function persistCommands() { saveCommands(store.commands).catch(onPersistError('commands')) }
export function persistSuggestions() { saveSuggestions(store.suggestions).catch(onPersistError('suggestions')) }
export function persistProjectMemory() { saveProjectMemory(store.projectMemory).catch(onPersistError('projectMemory')) }
export function persistArtifacts() { saveArtifacts(store.artifacts).catch(onPersistError('artifacts')) }
export function persistProjectMemoryCandidates() { saveProjectMemoryCandidates(store.projectMemoryCandidates).catch(onPersistError('projectMemoryCandidates')) }
export function persistWebhooks() { saveWebhooks(store.webhooks).catch(onPersistError('webhooks')) }
export function persistIncidents() { saveIncidents(store.incidents).catch(onPersistError('incidents')) }

// Update restart is allowed only after the admission gate is closed and these writes succeed.
// Unlike routine persist* calls, errors propagate so the current app stays open on any failure.
export async function preserveStateForUpdate(): Promise<void> {
  await Promise.all([
    saveSettings(store.settings), saveThreads(store.threads), saveProjects(store.projects),
    saveRooms(store.rooms), saveTickets(store.tickets), saveAutomations(store.automations),
    saveConnectorEvents(store.connectorEvents), saveContactDirectory(store.contacts),
    saveMissions(store.missions), saveJobs(store.jobs), saveSubAgents(store.subAgents),
    saveMcpServers(store.mcpServers), saveBotControlSessions(store.botSessions),
    saveSkills(store.skills), saveCommands(store.commands), saveSuggestions(store.suggestions),
    saveProjectMemory(store.projectMemory), saveArtifacts(store.artifacts),
    saveProjectMemoryCandidates(store.projectMemoryCandidates), saveWebhooks(store.webhooks),
    saveIncidents(store.incidents), saveCliSessions(store.cliSessions),
  ])
  await Promise.all([...writeChains.values()])
  await flushAuditEvents()
  if (getLastAuditError()) throw new Error('The security journal could not be saved; the app will stay open instead of restarting.')
}
