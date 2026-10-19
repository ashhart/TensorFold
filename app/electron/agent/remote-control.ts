import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { currentSecurityRoleFromEnv } from './security'
import { remoteDeviceStatus } from './device-status'
import { loadStaticFile } from './remote-static'
import { DEFAULT_REMOTE_STREAM_BUFFER_BYTES, KeepTailRemoteEventBuffer } from './remote-stream-buffer'
import { PairingAttemptLimiter, authToken, tokenMatches, securityRoleAllows, requiredRoleForRemoteMutation } from './remote-control-auth'
import { readJson, writeJson, writeRemoteStatic } from './remote-control-http'
import { remoteControlCommandCatalog } from './remote-control-catalog'
import { handleRemoteMutation } from './remote-control-mutations'
import type { RemoteWorkspaceScope } from './local-workspace-policy'
import type { RemoteControlServer, RemoteControlServerDeps } from './remote-control-types'
import type { RemoteControlEvent, RemoteControlSettings, RemoteControlStatus, SpineAggregate } from './types'

export { PairingAttemptLimiter } from './remote-control-auth'
export type { PairingAttemptLimiterOptions } from './remote-control-auth'
export { isHomeConfinedPath } from './remote-control-mutations'
export type { RemoteControlServer, RemoteControlServerDeps } from './remote-control-types'

function sseWire(event: RemoteControlEvent): string {
  return `id: ${event.id}\nevent: ${event.kind}\ndata: ${JSON.stringify(event)}\n\n`
}

function writeSse(res: ServerResponse, event: RemoteControlEvent): boolean {
  return res.write(sseWire(event))
}

export const REMOTE_CONTROL_SSE_HEARTBEAT_MS = 15_000
const REMOTE_CONTROL_SSE_HEARTBEAT = ': heartbeat\n\n'

export function createRemoteControlServer(deps: RemoteControlServerDeps): RemoteControlServer {
  let server: Server | null = null
  let host: RemoteControlSettings['bindHost'] | null = null
  let port: number | null = null
  let lastStartedAt: number | null = null
  let lastClientAt: number | null = null
  let lastError: string | null = null
  interface SseClientState {
    scope: 'owner' | 'paired'
    authorize: () => boolean
    buffer: KeepTailRemoteEventBuffer
    blocked: boolean
    closed: boolean
  }
  const clients = new Map<ServerResponse, SseClientState>()
  const pendingClients = new Map<ServerResponse, SseClientState>()
  const sseResponses = new Set<ServerResponse>()
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  // Monotonic event sequence + a bounded replay buffer so a dropped client can gap-fill via
  // Last-Event-ID instead of losing every event in the disconnect window.
  const REPLAY_LIMIT = 1000
  let seq = 0
  const recent: RemoteControlEvent[] = []
  const pairingAttempts = new PairingAttemptLimiter()

  function displayUrl(): string | null {
    if (!server || port == null || !host) return null
    return `http://${host}:${port}`
  }

  function settingsHost(settings: RemoteControlSettings): RemoteControlSettings['bindHost'] {
    // The built-in server is plain HTTP. Keep it loopback-only; cross-machine access must
    // arrive through SSH/Tailscale serve/a TLS reverse proxy terminating on loopback.
    return '127.0.0.1'
  }

  const snapshotFor = (owner: boolean) => owner || !deps.getPairedSnapshot ? deps.getSnapshot() : deps.getPairedSnapshot()
  const archivesFor = (owner: boolean) => owner ? deps.getArchives() : deps.getPairedArchives()
  const publishVisible = (event: Pick<RemoteControlEvent, 'kind' | 'data'>): boolean => {
    try { return deps.isPublishVisible?.(event) !== false } catch { return false }
  }
  const eventFor = (event: RemoteControlEvent, owner: boolean): RemoteControlEvent => {
    if (owner || !deps.redactPairedEvent) return event
    return { ...event, ...deps.redactPairedEvent(event) }
  }

  const newClientState = (scope: 'owner' | 'paired', authorize: () => boolean): SseClientState => ({
    scope,
    authorize,
    buffer: new KeepTailRemoteEventBuffer(Math.max(1_024, deps.sseClientBufferBytes ?? DEFAULT_REMOTE_STREAM_BUFFER_BYTES)),
    blocked: false,
    closed: false,
  })

  function removeClient(client: ServerResponse, state?: SseClientState): void {
    const current = state ?? clients.get(client) ?? pendingClients.get(client)
    if (current) {
      current.closed = true
      current.buffer.clear()
    }
    sseResponses.delete(client)
    clients.delete(client)
    pendingClients.delete(client)
  }

  function clientAuthorized(client: ServerResponse, state: SseClientState): boolean {
    if (state.closed) return false
    try { if (deps.getSettings().enabled && state.authorize()) return true } catch {}
    removeClient(client, state)
    try { client.end() } catch { client.destroy() }
    return false
  }

  function closeRevokedClients(): void {
    for (const [client, state] of [...clients, ...pendingClients]) clientAuthorized(client, state)
  }

  function closeForRebaseline(client: ServerResponse, state: SseClientState, event: RemoteControlEvent): void {
    if (state.closed) return
    const droppedEvents = state.buffer.length + 1
    const droppedBytes = state.buffer.byteLength
    state.closed = true
    state.buffer.clear()
    sseResponses.delete(client)
    clients.delete(client)
    pendingClients.delete(client)
    const gap: RemoteControlEvent = {
      id: event.id,
      kind: 'stream_gap',
      createdAt: Date.now(),
      data: {
        reason: 'backpressure',
        droppedEvents,
        droppedBytes,
        fromEventId: event.id,
        toEventId: event.id,
        recovery: 'snapshot',
      },
    }
    try { client.end(sseWire(gap)) } catch { try { client.destroy() } catch { /* already gone */ } }
  }

  function markClientBlocked(client: ServerResponse, state: SseClientState): void {
    if (state.blocked || state.closed) return
    state.blocked = true
    client.once('drain', () => {
      if (state.closed) return
      state.blocked = false
      flushClient(client, state)
    })
  }

  function flushClient(client: ServerResponse, state: SseClientState): void {
    if (!clientAuthorized(client, state) || state.blocked) return
    if (client.destroyed || client.writableEnded) {
      removeClient(client, state)
      return
    }
    try {
      const gap = state.buffer.takeGapEvent()
      if (gap && !writeSse(client, gap)) {
        markClientBlocked(client, state)
        return
      }
      while (!state.blocked) {
        const event = state.buffer.shift()
        if (!event) return
        if (!writeSse(client, event)) markClientBlocked(client, state)
      }
    } catch {
      removeClient(client, state)
      try { client.destroy() } catch { /* already gone */ }
    }
  }

  function deliverClient(client: ServerResponse, state: SseClientState, event: RemoteControlEvent): void {
    if (!clientAuthorized(client, state)) return
    if (client.destroyed || client.writableEnded) {
      removeClient(client, state)
      return
    }
    if (state.blocked) {
      const queued = state.buffer.enqueue(event)
      if (queued.overflow) closeForRebaseline(client, state, event)
      return
    }
    try {
      if (!writeSse(client, event)) markClientBlocked(client, state)
    } catch {
      removeClient(client, state)
      try { client.destroy() } catch { /* already gone */ }
    }
  }

  function markClient(): void {
    lastClientAt = Date.now()
    deps.onClientSeen(lastClientAt)
  }

  function stopHeartbeats(): void {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }

  function startHeartbeats(): void {
    stopHeartbeats()
    const intervalMs = Math.max(1, deps.heartbeatIntervalMs ?? REMOTE_CONTROL_SSE_HEARTBEAT_MS)
    heartbeatTimer = setInterval(() => {
      for (const client of sseResponses) {
        if (client.destroyed || client.writableEnded) {
          sseResponses.delete(client)
          clients.delete(client)
          pendingClients.delete(client)
          continue
        }
        try {
          // SSE comments are intentionally static: they keep intermediaries and readers alive
          // without consuming an event id or disclosing timestamps, state, or credentials.
          const state = clients.get(client) ?? pendingClients.get(client)
          if (state && !clientAuthorized(client, state)) continue
          if (!state?.blocked && !client.write(REMOTE_CONTROL_SSE_HEARTBEAT) && state) markClientBlocked(client, state)
        } catch {
          sseResponses.delete(client)
          clients.delete(client)
          pendingClients.delete(client)
          try { client.destroy() } catch { /* already gone */ }
        }
      }
    }, intervalMs)
    ;(heartbeatTimer as NodeJS.Timeout).unref?.()
  }

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url || '/', `http://${host || '127.0.0.1'}`)
    if (url.pathname === '/health') {
      // /health is reachable before the auth check below, so it must not disclose identifying
      // details (the user-chosen serverName) to unauthenticated callers -- that aids fingerprinting
      // of an exposed host. Return only an opaque liveness response.
      writeJson(res, 200, { ok: true, listening: !!server })
      return
    }

    // Static renderer shell (VST-035): every GET outside /v1 serves the built web client so
    // a paired browser/phone loads TensorFold straight from this host. Pre-auth BY DESIGN — the
    // shell is public app code with no user data (like any hosted web app); every data
    // endpoint below stays behind the token gate, and a fresh browser sits on the pairing
    // screen until it redeems a credential.
    if (req.method === 'GET' && !url.pathname.startsWith('/v1/')) {
      const root = deps.staticRoot?.() ?? null
      const file = root ? await loadStaticFile(root, url.pathname) : null
      if (file) {
        writeRemoteStatic(res, file.content, file.contentType, file.cacheControl)
        return
      }
      writeJson(res, 404, { error: root ? 'not_found' : 'web_client_not_served' })
      return
    }

    // Redeem a one-time pairing credential for a device session token (VST-035). Handled
    // BEFORE the bearer-token gate: a fresh device has no session/owner token yet — the
    // credential IS its authorization, verified constant-time inside redeemPairingLink. This
    // is the only pre-gate mutation, and it mints nothing without a valid, unconsumed link.
    if (req.method === 'POST' && url.pathname === '/v1/auth/pair') {
      if (!deps.pairing) { writeJson(res, 501, { error: 'pairing_unavailable' }); return }
      if (!securityRoleAllows('operator')) {
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'insufficient_role', requiredRole: 'operator' })
        writeJson(res, 403, { error: 'insufficient_role', requiredRole: 'operator' })
        return
      }
      const source = req.socket.remoteAddress || 'unknown'
      const rate = pairingAttempts.check(source)
      if (rate.blocked) {
        res.setHeader('retry-after', String(Math.max(1, Math.ceil(rate.retryAfterMs / 1_000))))
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'pairing_rate_limited' })
        writeJson(res, 429, { error: 'pairing_rate_limited' })
        return
      }
      const body = await readJson(req)
      const credential = String(body.credential ?? authToken(req) ?? '')
      const result = await deps.pairing.redeemPairingLink(credential, typeof body.clientInfo === 'string' ? body.clientInfo : null)
      if (!result.ok || !result.session) {
        pairingAttempts.recordFailure(source)
        writeJson(res, 400, { error: result.error ?? 'pair_failed' })
        return
      }
      pairingAttempts.clear(source)
      deps.audit('remote_control.pair', { label: result.session.label, clientInfo: result.session.clientInfo ?? null })
      markClient()
      writeJson(res, 200, { sessionId: result.session.id, token: result.session.token, label: result.session.label })
      return
    }

    const settings = deps.getSettings()
    // Authentication is header-only, including for the fetch-backed browser SSE client.
    // Credentials in query strings leak through browser history, proxy access logs, crash
    // reports, copied links, and referrers, so no remote-control route accepts them.
    const presented = authToken(req)
    // Accept EITHER the shared owner token (the legacy single secret, still the owner's key)
    // OR a live per-device session token (VST-035). A device session authorizes only the
    // data API — issuing/redeeming pairing links itself stays owner-token-only (below), so a
    // paired phone can never mint new devices.
    const ownerAuthed = tokenMatches(settings.authToken, presented)
    const sessionAuth = !ownerAuthed && deps.pairing ? deps.pairing.authorizeSessionToken(presented) : { ok: false }
    if (!ownerAuthed && !sessionAuth.ok) {
      deps.audit('remote_control.reject', { path: url.pathname, reason: 'unauthorized' })
      writeJson(res, 401, { error: 'unauthorized' })
      return
    }
    const stillAuthorized = (): boolean => {
      if (!deps.getSettings().enabled) return false
      if (ownerAuthed) return tokenMatches(deps.getSettings().authToken, presented)
      const current = deps.pairing?.authorizeSessionToken(presented)
      return current?.ok === true && current.sessionId === sessionAuth.sessionId
    }
    const reply = async (operation: unknown): Promise<void> => {
      const value = await operation
      if (!stillAuthorized()) { writeJson(res, 401, { error: 'unauthorized' }); return }
      writeJson(res, 200, value)
    }
    markClient()
    const workspaceScope: RemoteWorkspaceScope = { confineToHome: !ownerAuthed }

    // Bearer authentication identifies the caller; the process security role caps what
    // this host is willing to do. Viewer hosts remain fully usable for authenticated reads
    // but cannot be mutated through REST. Terminal control and session revocation retain
    // their owner tier, matching the local IPC boundary.
    if (req.method === 'POST') {
      const requiredRole = requiredRoleForRemoteMutation(url.pathname)
      if (!securityRoleAllows(requiredRole)) {
        deps.audit('remote_control.reject', { path: url.pathname, reason: 'insufficient_role', requiredRole })
        writeJson(res, 403, { error: 'insufficient_role', requiredRole })
        return
      }
      if (requiredRole === 'owner' && !ownerAuthed) {
        writeJson(res, 403, { error: 'owner_token_required' })
        return
      }
    }

    // Listing/revoking device sessions is an OWNER action — a paired device can't enumerate
    // or evict its siblings.
    if (req.method === 'GET' && url.pathname === '/v1/auth/sessions') {
      if (!ownerAuthed || !securityRoleAllows('owner')) { writeJson(res, 403, { error: 'owner_role_and_token_required' }); return }
      writeJson(res, 200, { sessions: deps.pairing?.listSessions() ?? [] })
      return
    }
    if (req.method === 'POST' && url.pathname === '/v1/auth/sessions/revoke') {
      if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
      if (!deps.pairing) { writeJson(res, 501, { error: 'pairing_unavailable' }); return }
      const body = await readJson(req)
      const id = String(body.sessionId ?? '')
      const revoked = await revokeDeviceSession(id)
      deps.audit('remote_control.revoke_session', { sessionId: id, revoked })
      writeJson(res, 200, { revoked })
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/snapshot') {
      await reply(snapshotFor(ownerAuthed))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/archives') {
      await reply(archivesFor(ownerAuthed))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/files') {
      // Paired-device sessions are the least-privilege tier: confine their filesystem
      // browsing to the home subtree. The owner token browses unrestricted.
      await reply(deps.listDirectories(url.searchParams.get('dir') || '', { confineToHome: !ownerAuthed }))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/control-plane') {
      writeJson(res, 200, remoteControlCommandCatalog())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/doctor') {
      await reply(deps.getDoctor())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/device-status') {
      const engine = await deps.getDeviceEngine?.()
      await reply(remoteDeviceStatus(ownerAuthed ? 'owner' : 'paired', currentSecurityRoleFromEnv(process.env), engine?.status ?? null, engine?.performance ?? null))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/clis') {
      if (!deps.listClis) { writeJson(res, 501, { error: 'clis_unavailable' }); return }
      await reply(deps.listClis())
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/terminals') {
      if (!deps.listTerminals) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
      writeJson(res, 200, deps.listTerminals(url.searchParams.get('projectId') || ''))
      return
    }

    // Ordered event spine replay (VST-034): durable, cursor-based gap-fill for remote
    // clients. The SSE ring above stays the live full-fidelity push; this is the survives-
    // restarts ordering log — a reconnecting client reads what changed since its cursor
    // (lean payloads: ids/states, never transcript text) and refetches precisely.
    if (req.method === 'GET' && url.pathname === '/v1/spine/events') {
      if (!deps.spineReadSince) { writeJson(res, 501, { error: 'spine_unavailable' }); return }
      const sinceRaw = Number(url.searchParams.get('since') ?? 0)
      const limitRaw = Number(url.searchParams.get('limit') ?? 500)
      const since = Number.isFinite(sinceRaw) && sinceRaw >= 0 ? Math.floor(sinceRaw) : 0
      const limit = Math.min(Math.max(1, Number.isFinite(limitRaw) ? Math.floor(limitRaw) : 500), 2000)
      const aggregate = url.searchParams.get('aggregate') || undefined
      const streamId = url.searchParams.get('streamId') || undefined
      writeJson(res, 200, deps.spineReadSince(since, { aggregate: aggregate as SpineAggregate | undefined, streamId, limit }))
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/spine/receipt') {
      if (!deps.spineReceipt) { writeJson(res, 501, { error: 'spine_unavailable' }); return }
      const commandId = String(url.searchParams.get('commandId') ?? '').trim()
      if (!commandId) { writeJson(res, 400, { error: 'commandId_required' }); return }
      writeJson(res, 200, { receipt: deps.spineReceipt(commandId) })
      return
    }

    if (req.method === 'GET' && url.pathname === '/v1/events') {
      const state = newClientState(ownerAuthed ? 'owner' : 'paired', stillAuthorized)
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-store',
        connection: 'keep-alive',
      })
      // Track the response before the snapshot is assembled. Heartbeat comments can flow
      // during a slow baseline without exposing the client to live events out of order.
      sseResponses.add(res)
      pendingClients.set(res, state)
      res.flushHeaders()
      const remove = () => removeClient(res, state)
      req.on('close', remove)
      res.on('error', remove)
      // Reconnect with Last-Event-ID (EventSource sends it automatically) / ?since: gap-fill from the
      // buffer instead of resending a full snapshot — as long as the gap is still buffered.
      const lastSeen = Number(req.headers['last-event-id'] ?? url.searchParams.get('since') ?? NaN)
      const oldestBuffered = recent.length ? Number(recent[0].id) : seq + 1
      const canReplay = Number.isFinite(lastSeen) && lastSeen <= seq && oldestBuffered <= lastSeen + 1
      if (canReplay) {
        for (const ev of recent) if (Number(ev.id) > lastSeen && publishVisible(ev)) deliverClient(res, state, eventFor(ev, ownerAuthed))
      } else {
        // Fresh client, or the gap is larger than the buffer: send a snapshot baselined at the current
        // seq so the client's subsequent Last-Event-ID lines up with the live stream.
        const snapshot = await snapshotFor(ownerAuthed)
        deliverClient(res, state, { id: String(seq), kind: 'snapshot', createdAt: Date.now(), data: snapshot })
      }
      pendingClients.delete(res)
      if (!state.closed && !res.destroyed && !res.writableEnded) clients.set(res, state)
      return
    }

    if (req.method !== 'POST') {
      writeJson(res, 405, { error: 'method_not_allowed' })
      return
    }

    await handleRemoteMutation({ deps, req, res, url, ownerAuthed, workspaceScope })
  }

  async function start(nextHost: RemoteControlSettings['bindHost'], nextPort: number): Promise<void> {
    await stop()
    await new Promise<void>((resolve) => {
      const s = createServer((req, res) => {
        handle(req, res).catch((err: any) => {
          // If a handler already started the response (e.g. an SSE writeHead before getSnapshot()
          // rejected), a second writeHead throws ERR_HTTP_HEADERS_SENT and crashes the process.
          if (res.headersSent || res.writableEnded) { try { res.end() } catch { /* already closing */ } return }
          const code = typeof err?.code === 'number' ? err.code : 500
          writeJson(res, code, { error: code === 500 ? 'remote_control_error' : String(err?.message || err) })
        })
      })
      s.on('error', (err: any) => {
        stopHeartbeats()
        server = null
        host = null
        port = null
        lastError = String(err?.message || err || 'Remote control server failed.')
        deps.onError(lastError)
        resolve()
      })
      s.listen(nextPort, nextHost, () => {
        server = s
        host = nextHost
        const addr = s.address()
        port = typeof addr === 'object' && addr ? addr.port : nextPort
        lastStartedAt = Date.now()
        lastError = null
        deps.onStarted(lastStartedAt, null)
        deps.audit('remote_control.start', { host: nextHost, port })
        startHeartbeats()
        resolve()
      })
    })
  }

  async function syncFromSettings(): Promise<void> {
    const settings = deps.getSettings()
    if (!settings.enabled) { await stop(); return }
    if (!settings.authToken) {
      lastError = 'Remote control token is missing.'
      deps.onError(lastError)
      await stop()
      return
    }
    const nextHost = settingsHost(settings)
    const nextPort = Number.isFinite(settings.port) ? Math.max(1, Math.min(65535, Math.round(settings.port))) : 8765
    if (settings.allowLan || settings.bindHost !== '127.0.0.1') deps.audit('remote_control.bind_normalized', { requestedHost: settings.bindHost, host: nextHost, reason: 'plaintext_server_loopback_only' })
    if (server && host === nextHost && port === nextPort) { closeRevokedClients(); return }
    await start(nextHost, nextPort)
  }

  async function stop(): Promise<void> {
    stopHeartbeats()
    for (const res of sseResponses) {
      try { res.end() } catch { /* ignore */ }
    }
    sseResponses.clear()
    clients.clear()
    pendingClients.clear()
    const s = server
    server = null
    host = null
    port = null
    if (s) await new Promise<void>((resolve) => s.close(() => resolve()))
  }

  function status(): RemoteControlStatus {
    const settings = deps.getSettings()
    return {
      enabled: settings.enabled,
      listening: !!server,
      serverName: settings.serverName,
      bindHost: settingsHost(settings),
      port,
      url: displayUrl(),
      clients: clients.size,
      lastStartedAt: settings.lastStartedAt ?? lastStartedAt,
      lastClientAt: settings.lastClientAt ?? lastClientAt,
      lastError: settings.lastError ?? lastError,
    }
  }

  function publish(event: Omit<RemoteControlEvent, 'id' | 'createdAt'>): void {
    if (!publishVisible(event)) return
    try { deps.observePublish?.(event.kind) } catch { /* observers are fail-open */ }
    // Buffer even with no clients connected, so a client that reconnects can still replay the gap.
    const full: RemoteControlEvent = { id: String(++seq), createdAt: Date.now(), ...event }
    recent.push(full)
    if (recent.length > REPLAY_LIMIT) recent.shift()
    for (const [client, state] of clients) deliverClient(client, state, eventFor(full, state.scope === 'owner'))
  }

  async function issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }> {
    return await (deps.pairing?.issuePairingLink(label) ?? { credential: '', expiresAt: 0 })
  }
  function listDeviceSessions() {
    return deps.pairing?.listSessions() ?? []
  }
  async function revokeDeviceSession(id: string): Promise<boolean> {
    try { return await (deps.pairing?.revokeSession(id) ?? false) }
    finally { closeRevokedClients() }
  }

  return { syncFromSettings, stop, status, publish, issuePairingLink, listDeviceSessions, revokeDeviceSession }
}
