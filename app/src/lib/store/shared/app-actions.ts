import { api } from '../../ipc'

// #42 — app-control actions already auto-applied (so a persisted timeline that
// replays on reload doesn't re-create projects/rooms/tickets). Backed by
// localStorage so the guard survives a reload — the persisted timeline keeps the
// original (state-less) item, so an in-memory-only Set would re-run on restart.
export const APPLIED_APP_ACTIONS_KEY = 'vontra.appliedAppActions'

const appliedAppActions = new Set<string>((() => {
  try { return JSON.parse(localStorage.getItem(APPLIED_APP_ACTIONS_KEY) || '[]') } catch { return [] }
})())
export function markAppActionApplied(id: string) {
  appliedAppActions.add(id)
  try { localStorage.setItem(APPLIED_APP_ACTIONS_KEY, JSON.stringify([...appliedAppActions].slice(-500))) } catch { /* ignore quota */ }
}
export function isAppActionApplied(id: string): boolean {
  return appliedAppActions.has(id)
}

// In-flight app-control creates: dedups concurrent fires of the same item (e.g. a duplicated
// item_add event) within a session WITHOUT persisting to localStorage, so a create that ends up
// failing isn't permanently blocked from retrying on the next reload.
const inflightAppActions = new Set<string>()
export function isInflightAppAction(id: string): boolean {
  return inflightAppActions.has(id)
}
export function addInflightAppAction(id: string): void {
  inflightAppActions.add(id)
}
export function deleteInflightAppAction(id: string): void {
  inflightAppActions.delete(id)
}

// Re-exported so slices keep a single import surface; also guards init() against React
// StrictMode's double effect-invocation in dev, which would otherwise subscribe the IPC
// listeners twice and double every streamed item_delta (garbled output).
let storeInitStarted = false
let storeSubscriptionsStarted = false
export function acquireInitStarted(): boolean {
  if (storeInitStarted) return false
  storeInitStarted = true
  return true
}
export function acquireSubscriptionsStarted(): boolean {
  if (storeSubscriptionsStarted) return false
  storeSubscriptionsStarted = true
  return true
}

// Projects whose persisted terminals have already been auto-restored this app session. Module-level
// (resets on app reload) so opening a project's terminal view re-restores at most once — re-opening
// after the user kills a restored terminal must NOT bring it back.
export const restoredProjects = new Set<string>()

// Thread-create / start / guide / cancel helpers that route through either the local api or the
// owning Orchestrator depending on whether the record carries an origin stamp.
export async function createThreadForProject(get: () => import('../types').AppState, cwd: string, kind: 'task' | 'chat', projectId: string | null, incognito = false): Promise<{ thread: import('../../types').Thread; remoteHostId: string | null }> {
  const state = get()
  const project = projectId ? state.projects.find((candidate) => candidate.id === projectId) ?? null : null
  const remoteHostId = (await import('./remote-host')).remoteHostIdForProject(state, project)
  if (remoteHostId) {
    if (incognito) throw new Error('Incognito tasks run in this desktop app; choose This Mac before starting a private task.')
    const created = await api.createRemoteOrchestratorThread(remoteHostId, { cwd, kind, projectId })
    // Stamp the origin immediately — the snapshot mirror stamps too, but the first run can
    // start before the next snapshot lands.
    return { thread: { ...created, orchestratorHostId: remoteHostId }, remoteHostId }
  }
  // One command id per user action: if the IPC reply is lost and this retries, the host's
  // spine receipt returns the SAME thread instead of minting a duplicate.
  const commandId = crypto.randomUUID()
  return { thread: incognito ? await api.createThread(cwd, kind, projectId, commandId, true) : await api.createThread(cwd, kind, projectId, commandId), remoteHostId: null }
}

export function startThreadRunApi(hostId: string | null, payload: Parameters<typeof api.startRun>[0]): Promise<{ runId: string }> {
  return hostId ? api.startRemoteOrchestratorThread(hostId, payload) : api.startRun(payload)
}

export function guideThreadRunApi(hostId: string | null, payload: Parameters<typeof api.cueRun>[0]) {
  return hostId ? api.guideRemoteOrchestratorThread(hostId, payload) : api.cueRun(payload)
}

export function cancelThreadRunApi(hostId: string | null, threadId: string): Promise<void> {
  return hostId ? api.cancelRemoteOrchestratorThread(hostId, threadId) : api.cancelRun(threadId)
}
