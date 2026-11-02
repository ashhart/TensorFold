import { assertGmailInboxRoute } from '../../../../shared/gmail-inbox'
import { api } from '../../ipc'

export const APPLIED_APP_ACTIONS_KEY = 'vontra.appliedAppActions'

const appliedAppActions = new Set<string>((() => {
  try { return JSON.parse(localStorage.getItem(APPLIED_APP_ACTIONS_KEY) || '[]') } catch { return [] }
})())
export function markAppActionApplied(id: string) {
  appliedAppActions.add(id)
  try { localStorage.setItem(APPLIED_APP_ACTIONS_KEY, JSON.stringify([...appliedAppActions].slice(-500))) } catch { }
}
export function isAppActionApplied(id: string): boolean {
  return appliedAppActions.has(id)
}

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

export const restoredProjects = new Set<string>()

export async function createThreadForProject(get: () => import('../types').AppState, cwd: string, kind: 'task' | 'chat', projectId: string | null, incognito = false): Promise<{ thread: import('../../types').Thread; remoteHostId: string | null }> {
  const state = get()
  const project = projectId ? state.projects.find((candidate) => candidate.id === projectId) ?? null : null
  const remoteHostId = (await import('./remote-host')).remoteHostIdForProject(state, project)
  if (remoteHostId) {
    if (incognito) throw new Error('Incognito tasks run in this desktop app; choose This Mac before starting a private task.')
    const created = await api.createRemoteOrchestratorThread(remoteHostId, { cwd, kind, projectId })

    return { thread: { ...created, orchestratorHostId: remoteHostId }, remoteHostId }
  }

  const commandId = crypto.randomUUID()
  return { thread: incognito ? await api.createThread(cwd, kind, projectId, commandId, true) : await api.createThread(cwd, kind, projectId, commandId), remoteHostId: null }
}

export function startThreadRunApi(hostId: string | null, payload: Parameters<typeof api.startRun>[0]): Promise<{ runId: string }> {
  assertGmailInboxRoute(payload.sourceThreadKey, !!hostId || api.platform === 'web-remote')
  return hostId ? api.startRemoteOrchestratorThread(hostId, payload) : api.startRun(payload)
}

export function guideThreadRunApi(hostId: string | null, payload: Parameters<typeof api.cueRun>[0]) {
  return hostId ? api.guideRemoteOrchestratorThread(hostId, payload) : api.cueRun(payload)
}

export function cancelThreadRunApi(hostId: string | null, threadId: string): Promise<void> {
  return hostId ? api.cancelRemoteOrchestratorThread(hostId, threadId) : api.cancelRun(threadId)
}
