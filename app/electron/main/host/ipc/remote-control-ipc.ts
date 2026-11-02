import { readBoundedConnectorJson } from '../../../agent/connector-http'
import { discardResponseBody } from '../../../agent/pinned-fetch'
import { fetchRemoteControlSse, readRemoteControlSse, REMOTE_SSE_CONNECT_TIMEOUT_MS, REMOTE_SSE_FRAME_MAX_CHARS, REMOTE_SSE_IDLE_TIMEOUT_MS } from '../../../agent/remote-sse'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { createRemoteOrchestratorFetch } from '../../../agent/secure-fetch'
import { assertRemoteOrchestratorUrl, auditPreview, isSecretMask, maskedSecretUrlError, normalizeRemoteOrchestratorUrl } from '../../../agent/security'
import { clientVisibleRemoteSessions, remoteTerminalEventToRenderer, type TerminalEvent } from '../../../agent/terminal-forward'
import type { AgentEvent, CliSession, Project, RemoteConnectionTestResult, RemoteControlEvent, RemoteControlSnapshot, RemoteOrchestratorProfile, RemoteOrchestratorSnapshotResult, Settings, Thread, Ticket } from '../../../agent/types'
import { makeRemoteControlToken, saveSettings } from '../../../store'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { persistSettings } from '../persistence'
import { normalizeRemoteTicketHost } from '../room-ticket-patch-policy'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from '../sensitive-confirmation'
import { registerRemoteProjectIpc } from './remote-project-ipc'
import { registerRemoteRoomTicketIpc } from './remote-room-ticket-ipc'
import { registerRemoteRunIpc } from './remote-run-ipc'

interface RemoteServerLike {
  status(): { url: string | null }
  syncFromSettings(): Promise<void>
  publish(event: Pick<RemoteControlEvent, 'kind' | 'data'>): void
  // Per-device pairing (VST-035).
  issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }>
  listDeviceSessions(): Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
  revokeDeviceSession(id: string): Promise<boolean>
}

export interface RemoteControlIpcDeps {
  remoteServer: RemoteServerLike
  confirmSensitiveAction: ConfirmSensitiveAction
  ensureRemoteControlToken: () => void
}

function remoteOrchestratorHttpError(status: number, base: string): Error {
  if (status === 401) {
    return new Error(`Token rejected by the Orchestrator at ${base}. If you are using SSH port forwarding, make sure this URL points at the forwarded local port for the host that generated the token.`)
  }
  return new Error(`HTTP ${status}`)
}

function remoteOrchestratorConnectionError(e: unknown, base: string, rawBase: string): string {
  const error = e as { message?: string; code?: string; cause?: { code?: string } }
  const message = String(error?.message || e || 'Remote connection failed.')
  if (/invalid url/i.test(message)) {
    return 'Remote Orchestrator URL is invalid. Use a full URL like http://127.0.0.1:19856.'
  }
  const causeCode = error?.cause?.code || error?.code || ''
  if (message === 'fetch failed' || causeCode) {
    const target = base || rawBase || 'the configured URL'
    const reason = causeCode ? ` (${causeCode})` : ''
    return `Could not reach the Orchestrator at ${target}${reason}. If you are using SSH port forwarding, leave the tunnel running and connect to its local port, for example http://127.0.0.1:19856.`
  }
  return message
}

function remoteHostForId(hostId?: string | null): RemoteOrchestratorProfile | null {
  return hostId
    ? store.settings.remoteOrchestrators.find((item) => item.id === hostId) ?? null
    : store.settings.remoteOrchestrators.find((item) => item.enabled) ?? null
}

async function fetchRemoteOrchestratorJson<T>(host: RemoteOrchestratorProfile, path: string, init: RequestInit = {}): Promise<{ base: string; data: T }> {
  const base = await assertRemoteOrchestratorUrl(host.serverUrl)
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${host.authToken}`)
  const res = await createRemoteOrchestratorFetch(base)(`${base}${path}`, {
    ...init,
    headers,
    signal: init.signal ?? AbortSignal.timeout(8000),
  })
  if (!res.ok) {
    await discardResponseBody(res)
    throw remoteOrchestratorHttpError(res.status, base)
  }
  return { base, data: await readBoundedConnectorJson(res, 'Remote Orchestrator', 16 * 1024 * 1024) as T }
}

const remoteEventStreams = new Map<string, { controller: AbortController; key: string }>()

function remoteEventStreamKey(host: RemoteOrchestratorProfile): string {
  return `${host.serverUrl.trim()}|${host.authToken}`
}

function withRemoteProjectHost(host: RemoteOrchestratorProfile, project: Project): Project {
  return { ...project, executionTarget: 'orchestrator', orchestratorId: host.id }
}

function withRemoteTicketHost(host: RemoteOrchestratorProfile, ticket: Ticket): Ticket {
  return normalizeRemoteTicketHost(ticket, host.id)
}

async function refreshRemoteTerminalSessions(host: RemoteOrchestratorProfile, projects: Project[]): Promise<void> {
  for (const project of projects) {
    try {
      const { data } = await fetchRemoteOrchestratorJson<CliSession[]>(host, `/v1/terminals?projectId=${encodeURIComponent(project.id)}`, { signal: AbortSignal.timeout(8000) })
      store.win?.webContents.send('terminal:sessions', {
        projectId: project.id,
        sessions: clientVisibleRemoteSessions(data ?? [], host.id),
        orchestratorHostId: host.id,
      })
    } catch {
      // Terminal refresh is best-effort during snapshot recovery; project/thread state should still land.
    }
  }
}

function dispatchRemoteSnapshot(host: RemoteOrchestratorProfile, snapshot: RemoteControlSnapshot): void {
  for (const project of snapshot.projects ?? []) store.win?.webContents.send('project:update', withRemoteProjectHost(host, project))

  for (const thread of snapshot.threads ?? []) store.win?.webContents.send('thread:update', { ...thread, orchestratorHostId: host.id })
  for (const room of snapshot.rooms ?? []) store.win?.webContents.send('room:update', { ...room, orchestratorHostId: host.id })
  for (const ticket of snapshot.tickets ?? []) store.win?.webContents.send('ticket:update', withRemoteTicketHost(host, ticket))

  for (const mission of snapshot.missions ?? []) store.win?.webContents.send('mission:update', mission)
  void refreshRemoteTerminalSessions(host, snapshot.projects ?? [])
}

function dispatchRemoteControlEvent(host: RemoteOrchestratorProfile, event: RemoteControlEvent): void {
  const data = event.data as any
  if (!data) return
  if (event.kind === 'snapshot') {
    dispatchRemoteSnapshot(host, data as RemoteControlSnapshot)
  } else if (event.kind === 'agent') {
    const agent = data as AgentEvent
    store.win?.webContents.send('agent:event', {
      ...agent,
      orchestratorHostId: host.id,
      trace: { ...(agent.trace ?? {}), surface: agent.trace?.surface ?? 'remote' },
    })
  } else if (event.kind === 'project') {
    store.win?.webContents.send('project:update', withRemoteProjectHost(host, data as Project))
  } else if (event.kind === 'thread') {
    store.win?.webContents.send('thread:update', { ...(data as Thread), orchestratorHostId: host.id })
  } else if (event.kind === 'room') {
    if (typeof data.type === 'string' && typeof data.roomId === 'string') store.win?.webContents.send('room:event', { ...data, orchestratorHostId: host.id })
    else store.win?.webContents.send('room:update', { ...data, orchestratorHostId: host.id })
  } else if (event.kind === 'ticket') {
    store.win?.webContents.send('ticket:update', withRemoteTicketHost(host, data as Ticket))
  } else if (event.kind === 'job' || event.kind === 'automation') {
    // Origin-aware control forwarding is not implemented; keep these remote rows quarantined.
  } else if (event.kind === 'mission') {
    store.win?.webContents.send('mission:update', data)
  } else if (event.kind === 'connector') {
    store.win?.webContents.send('connector:update', data)
  } else if (event.kind === 'terminal') {

    const mapped = remoteTerminalEventToRenderer(data as TerminalEvent, host.id)
    if (mapped) store.win?.webContents.send(mapped.channel, mapped.payload)
  }
}

function sleepRemoteStream(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => { clearTimeout(t); resolve() }, { once: true })
  })
}

function updateRemoteHostConnection(hostId: string, patch: Partial<RemoteOrchestratorProfile>): void {
  store.settings = {
    ...store.settings,
    remoteOrchestrators: store.settings.remoteOrchestrators.map((item) => item.id === hostId ? { ...item, ...patch, updatedAt: Date.now() } : item),
  }
  persistSettings('remoteOrchestrator.connection')
}

function ensureRemoteEventStream(host: RemoteOrchestratorProfile): void {
  if (!host.enabled || !host.serverUrl.trim() || !host.authToken) return
  const key = remoteEventStreamKey(host)
  const existing = remoteEventStreams.get(host.id)
  if (existing?.key === key) return
  existing?.controller.abort()
  const controller = new AbortController()
  remoteEventStreams.set(host.id, { controller, key })
  void (async () => {
    let lastSeen: string | null = null
    while (!controller.signal.aborted) {
      let base = ''
      try {

        base = normalizeRemoteOrchestratorUrl(host.serverUrl)
        const since = lastSeen ? `?since=${encodeURIComponent(lastSeen)}` : ''
        const res = await fetchRemoteControlSse(createRemoteOrchestratorFetch(base), `${base}/v1/events${since}`, {
          headers: { Authorization: `Bearer ${host.authToken}` },
        }, REMOTE_SSE_CONNECT_TIMEOUT_MS, controller.signal)
        if (!res.ok) {
          await discardResponseBody(res)
          throw remoteOrchestratorHttpError(res.status, base)
        }
        updateRemoteHostConnection(host.id, { lastConnectedAt: Date.now(), lastError: null })
        await readRemoteControlSse(res, (event) => {
          lastSeen = event.id || lastSeen
          dispatchRemoteControlEvent(host, event)
        }, REMOTE_SSE_FRAME_MAX_CHARS, REMOTE_SSE_IDLE_TIMEOUT_MS, controller.signal)
      } catch (e) {
        if (controller.signal.aborted) break
        const error = remoteOrchestratorConnectionError(e, base, host.serverUrl)
        updateRemoteHostConnection(host.id, { lastError: error })
        await sleepRemoteStream(2000, controller.signal)
      }
    }
    if (remoteEventStreams.get(host.id)?.controller === controller) remoteEventStreams.delete(host.id)
  })()
}

export function syncRemoteEventStreams(): void {
  const active = new Set<string>()
  for (const host of store.settings.remoteOrchestrators ?? []) {
    if (!host.enabled) continue
    active.add(host.id)
    ensureRemoteEventStream(host)
  }
  for (const [id, stream] of remoteEventStreams) {
    if (!active.has(id)) {
      stream.controller.abort()
      remoteEventStreams.delete(id)
    }
  }
}

export function registerRemoteControlIpc(deps: RemoteControlIpcDeps) {
  const { remoteServer, confirmSensitiveAction, ensureRemoteControlToken } = deps

  handle('remote:status', async () => remoteServer.status())
  handle('remote:pairingInfo', async () => {
    requireSecurityRole('owner', 'reveal Orchestrator pairing token')
    await requireNativeConfirmation(confirmSensitiveAction, {
      title: 'Reveal Orchestrator token?',
      message: 'The token grants control-plane access to this TensorFold instance.',
      detail: 'Only reveal it while pairing a client you trust. Treat it like a password.',
      confirmLabel: 'Reveal token',
    })
    ensureRemoteControlToken()
    await saveSettings(store.settings)
    await remoteServer.syncFromSettings()
    return { url: remoteServer.status().url, token: store.settings.remoteControl.authToken, status: remoteServer.status() }
  })
  handle('remote:regenerateToken', async () => {
    requireSecurityRole('owner', 'regenerate Orchestrator token')
    await requireNativeConfirmation(confirmSensitiveAction, {
      title: 'Rotate Orchestrator token?',
      message: 'Existing clients using the current host token will stop authenticating.',
      detail: 'Paired per-device sessions are managed separately.',
      confirmLabel: 'Rotate token',
    })
    const current = store.settings.remoteControl
    const next = { ...current, authToken: makeRemoteControlToken() }
    store.settings = { ...store.settings, remoteControl: stampSecretRotations(next, current, { authToken: [next.authToken, current.authToken] }) }
    await saveSettings(store.settings)
    await remoteServer.syncFromSettings()
    await auditEvent('remote_control.regenerate_token', { enabled: store.settings.remoteControl.enabled })
    return { url: remoteServer.status().url, token: store.settings.remoteControl.authToken, status: remoteServer.status() }
  })

  handle('remote:issuePairing', async (_e, label?: string): Promise<{ credential: string; expiresAt: number; url: string | null }> => {
    requireSecurityRole('owner', 'issue a device pairing link')
    await requireNativeConfirmation(confirmSensitiveAction, {
      title: 'Create a device pairing credential?',
      message: 'Anyone who receives the one-time credential before it expires can pair a device.',
      detail: `Requested label: ${typeof label === 'string' && label.trim() ? label.trim().slice(0, 120) : 'Paired device'}`,
      confirmLabel: 'Create pairing credential',
    })
    const link = await remoteServer.issuePairingLink(typeof label === 'string' ? label : 'Paired device')
    await auditEvent('remote_control.issue_pairing', { label: typeof label === 'string' ? label : 'Paired device', expiresAt: link.expiresAt })
    return { ...link, url: remoteServer.status().url }
  })
  handle('remote:listDeviceSessions', async () => {
    requireSecurityRole('owner', 'list paired device sessions')
    return remoteServer.listDeviceSessions()
  })
  handle('remote:revokeDeviceSession', async (_e, sessionId: string) => {
    requireSecurityRole('owner', 'revoke a paired device session')
    const revoked = await remoteServer.revokeDeviceSession(String(sessionId ?? ''))
    await auditEvent('remote_control.revoke_session', { sessionId: String(sessionId ?? ''), revoked })
    return { revoked }
  })
  handle('remote:testClient', async (_e, input?: Partial<Settings['remoteClient']>): Promise<RemoteConnectionTestResult> => {
    requireSecurityRole('operator', 'test remote Orchestrator client')
    const client = {
      ...store.settings.remoteClient,
      ...(input ?? {}),
      authToken: isSecretMask(input?.authToken) ? store.settings.remoteClient.authToken : (input?.authToken ?? store.settings.remoteClient.authToken),
    }
    const rawBase = client.serverUrl.trim()
    let base = ''
    if (!rawBase || !client.authToken) return { ok: false, error: 'Server URL and token are required.' }
    // A stored (masked) Orchestrator token must never be sent to a renderer-chosen URL.
    const maskedConflict = maskedSecretUrlError(isSecretMask(input?.authToken), input?.serverUrl, store.settings.remoteClient.serverUrl)
    if (maskedConflict) return { ok: false, error: maskedConflict }
    try {
      base = await assertRemoteOrchestratorUrl(rawBase)
      client.serverUrl = base
      const res = await createRemoteOrchestratorFetch(base)(`${base}/v1/snapshot`, {
        headers: { Authorization: `Bearer ${client.authToken}` },
        signal: AbortSignal.timeout(8000),
      })
      if (!res.ok) {
        await discardResponseBody(res)
        throw remoteOrchestratorHttpError(res.status, base)
      }
      const snapshot = await readBoundedConnectorJson(res, 'Remote Orchestrator', 16 * 1024 * 1024) as RemoteControlSnapshot
      store.settings = { ...store.settings, remoteClient: { ...client, lastConnectedAt: Date.now(), lastError: null } }
      await saveSettings(store.settings)
      await auditEvent('remote_client.test', { serverUrl: auditPreview(base), ok: true })
      return {
        ok: true,
        snapshot: {
          serverName: snapshot.serverName,
          generatedAt: snapshot.generatedAt,
          activeThreadIds: snapshot.activeThreadIds,
          activeRoomIds: snapshot.activeRoomIds,
          activeMissionIds: snapshot.activeMissionIds,
          work: snapshot.work,
        },
      }
    } catch (e: any) {
      const error = remoteOrchestratorConnectionError(e, base, rawBase)
      store.settings = { ...store.settings, remoteClient: { ...store.settings.remoteClient, lastError: error } }
      await auditEvent('remote_client.test', { serverUrl: auditPreview(base || rawBase), ok: false, error: auditPreview(error) })
      return { ok: false, error }
    }
  })
  handle('remote:snapshot', async (_e, hostId?: string | null): Promise<RemoteOrchestratorSnapshotResult> => {
    requireSecurityRole('operator', 'fetch remote Orchestrator snapshot')
    const host = hostId ? store.settings.remoteOrchestrators.find((item) => item.id === hostId) : null
    const authToken = host?.authToken || store.settings.remoteClient.authToken || ''
    const name = host?.name || store.settings.remoteClient.serverUrl || 'Orchestrator'
    const rawServerUrl = host?.serverUrl || store.settings.remoteClient.serverUrl || ''
    let serverUrl = ''
    if (!rawServerUrl.trim() || !authToken) return { ok: false, hostId: host?.id ?? null, name, serverUrl: rawServerUrl, error: 'Server URL and token are required.' }
    try {
      serverUrl = await assertRemoteOrchestratorUrl(rawServerUrl)
      const res = await createRemoteOrchestratorFetch(serverUrl)(`${serverUrl}/v1/snapshot`, {
        headers: { Authorization: `Bearer ${authToken}` },
        signal: AbortSignal.timeout(8000),
      })
      if (!res.ok) {
        await discardResponseBody(res)
        throw remoteOrchestratorHttpError(res.status, serverUrl)
      }
      const snapshot = await readBoundedConnectorJson(res, 'Remote Orchestrator', 16 * 1024 * 1024) as RemoteControlSnapshot
      if (host) {
        dispatchRemoteSnapshot(host, snapshot)
        ensureRemoteEventStream(host)
      }
      if (host) {
        store.settings = {
          ...store.settings,
          remoteOrchestrators: store.settings.remoteOrchestrators.map((item) => item.id === host.id ? { ...item, name: snapshot.serverName || item.name, lastConnectedAt: Date.now(), lastError: null, updatedAt: Date.now() } : item),
        }
      } else {
        store.settings = { ...store.settings, remoteClient: { ...store.settings.remoteClient, lastConnectedAt: Date.now(), lastError: null } }
      }
      await saveSettings(store.settings)
      await auditEvent('remote_client.snapshot', { serverUrl: auditPreview(serverUrl), ok: true })
      return {
        ok: true,
        hostId: host?.id ?? null,
        name: snapshot.serverName || name,
        serverUrl,
        generatedAt: snapshot.generatedAt,
        activeThreadIds: snapshot.activeThreadIds,
        activeRoomIds: snapshot.activeRoomIds,
        activeMissionIds: snapshot.activeMissionIds,
        work: snapshot.work,
        totalProjects: snapshot.projects.length,
        totalThreads: snapshot.threads.length,
        totalRooms: snapshot.rooms.length,
        totalMissions: snapshot.missions.length,
        projects: snapshot.projects,
        threads: snapshot.threads.map((thread) => ({ ...thread, orchestratorHostId: host?.id ?? null })),
        rooms: snapshot.rooms.map((room) => ({ ...room, orchestratorHostId: host?.id ?? null })),
        tickets: snapshot.tickets.map((ticket) => host ? withRemoteTicketHost(host, ticket) : { ...ticket, orchestratorHostId: null }),
        jobs: snapshot.jobs,
      }
    } catch (e: any) {
      const error = remoteOrchestratorConnectionError(e, serverUrl, rawServerUrl)
      if (host) {
        store.settings = {
          ...store.settings,
          remoteOrchestrators: store.settings.remoteOrchestrators.map((item) => item.id === host.id ? { ...item, lastError: error, updatedAt: Date.now() } : item),
        }
      } else {
        store.settings = { ...store.settings, remoteClient: { ...store.settings.remoteClient, lastError: error } }
      }
      persistSettings('remoteClient.snapshotError')
      await auditEvent('remote_client.snapshot', { serverUrl: auditPreview(serverUrl), ok: false, error: auditPreview(error) })
      return { ok: false, hostId: host?.id ?? null, name, serverUrl, error }
    }
  })

  const context = { remoteHostForId, ensureRemoteEventStream, fetchRemoteOrchestratorJson, remoteOrchestratorConnectionError, withRemoteProjectHost, withRemoteTicketHost }
  registerRemoteProjectIpc(context)
  registerRemoteRunIpc(context)
  registerRemoteRoomTicketIpc(context)
}
