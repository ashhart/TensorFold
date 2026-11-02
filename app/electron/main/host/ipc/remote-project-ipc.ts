import type { ArchiveSnapshot, Project, RemoteDirectoryListResult, Thread } from '../../../agent/types'
import { store } from '../../state'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import type { RemoteIpcContext } from './remote-ipc-context'

export function registerRemoteProjectIpc(context: RemoteIpcContext): void {
  const { remoteHostForId, ensureRemoteEventStream, fetchRemoteOrchestratorJson, remoteOrchestratorConnectionError, withRemoteProjectHost, withRemoteTicketHost } = context
  handle('remote:listDir', async (_e, input?: { hostId?: string | null; dir?: string | null }): Promise<RemoteDirectoryListResult> => {
    requireSecurityRole('operator', 'browse Orchestrator folders')
    const host = remoteHostForId(input?.hostId)
    if (!host) return { ok: false, path: '', parent: null, entries: [], error: 'No Orchestrator selected.' }
    try {
      const path = `/v1/files?dir=${encodeURIComponent(input?.dir || '')}`
      const { data } = await fetchRemoteOrchestratorJson<RemoteDirectoryListResult>(host, path)
      return data
    } catch (e) {
      return { ok: false, path: input?.dir || '', parent: null, entries: [], error: remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl) }
    }
  })

  handle('remote:archives', async (_e, input?: { hostId?: string | null }): Promise<ArchiveSnapshot> => {
    requireSecurityRole('operator', 'list Orchestrator archives')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      const { data } = await fetchRemoteOrchestratorJson<ArchiveSnapshot>(host, '/v1/archives', { signal: AbortSignal.timeout(12000) })
      return {
        projects: (data.projects ?? []).map((project) => ({ ...project, executionTarget: 'orchestrator' as const, orchestratorId: host.id })),
        threads: (data.threads ?? []).map((thread) => ({ ...thread, orchestratorHostId: host.id })),
        rooms: (data.rooms ?? []).map((room) => ({ ...room, orchestratorHostId: host.id })),
        tickets: (data.tickets ?? []).map((ticket) => withRemoteTicketHost(host, ticket)),
      }
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:createProject', async (_e, input: { hostId?: string | null; project: Partial<Project> & { name?: string; cwd?: string } }): Promise<Project> => {
    requireSecurityRole('operator', 'create Orchestrator project')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)

      const { executionTarget: _executionTarget, orchestratorId: _orchestratorId, archivedAt: _archivedAt, createdAt: _createdAt, updatedAt: _updatedAt, ...serverProject } = input.project ?? {}
      const { data } = await fetchRemoteOrchestratorJson<Project>(host, '/v1/projects', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ ...serverProject, executionTarget: 'local', orchestratorId: null }),
        signal: AbortSignal.timeout(12000),
      })
      return { ...data, executionTarget: 'orchestrator', orchestratorId: host.id }
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:updateProject', async (_e, input: { hostId?: string | null; id: string; patch: Partial<Project> }): Promise<Project> => {
    requireSecurityRole('operator', 'update Orchestrator project')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Project>(host, `/v1/projects/${encodeURIComponent(input.id)}/update`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.patch ?? {}),
        signal: AbortSignal.timeout(12000),
      })
      const project = { ...data, executionTarget: 'orchestrator' as const, orchestratorId: host.id }
      store.win?.webContents.send('project:update', project)
      return project
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:deleteProject', async (_e, input: { hostId?: string | null; id: string }): Promise<Project | null> => {
    requireSecurityRole('operator', 'archive Orchestrator project')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Project | null>(host, `/v1/projects/${encodeURIComponent(input.id)}/delete`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(12000),
      })
      if (data) store.win?.webContents.send('project:update', { ...data, executionTarget: 'orchestrator', orchestratorId: host.id })
      return data ? { ...data, executionTarget: 'orchestrator', orchestratorId: host.id } : null
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:restoreProject', async (_e, input: { hostId?: string | null; id: string }): Promise<Project | null> => {
    requireSecurityRole('operator', 'restore Orchestrator project')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Project | null>(host, `/v1/projects/${encodeURIComponent(input.id)}/restore`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(12000),
      })
      const project = data ? { ...data, executionTarget: 'orchestrator' as const, orchestratorId: host.id } : null
      if (project) store.win?.webContents.send('project:update', project)
      return project
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:createThread', async (_e, input: { hostId?: string | null; cwd?: string | null; kind?: 'task' | 'chat'; projectId?: string | null }): Promise<Thread> => {
    requireSecurityRole('operator', 'create Orchestrator thread')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Thread>(host, '/v1/threads', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ cwd: input?.cwd ?? '', kind: input?.kind === 'chat' ? 'chat' : 'task', projectId: input?.projectId ?? null }),
        signal: AbortSignal.timeout(12000),
      })
      const thread = { ...data, orchestratorHostId: host.id }
      store.win?.webContents.send('thread:update', thread)
      return thread
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:archiveThread', async (_e, input: { hostId?: string | null; id: string }): Promise<Thread | null> => {
    requireSecurityRole('operator', 'archive Orchestrator thread')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Thread | null>(host, `/v1/threads/${encodeURIComponent(input.id)}/archive`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(12000),
      })
      const thread = data ? { ...data, orchestratorHostId: host.id } : null
      if (thread) store.win?.webContents.send('thread:update', thread)
      return thread
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:restoreThread', async (_e, input: { hostId?: string | null; id: string }): Promise<Thread | null> => {
    requireSecurityRole('operator', 'restore Orchestrator thread')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Thread | null>(host, `/v1/threads/${encodeURIComponent(input.id)}/restore`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(12000),
      })
      const thread = data ? { ...data, orchestratorHostId: host.id } : null
      if (thread) store.win?.webContents.send('thread:update', thread)
      return thread
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

}
