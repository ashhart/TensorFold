import { createServer, type Server } from 'node:https'
import { request as loopbackRequest, type IncomingMessage, type ServerResponse } from 'node:http'
import { networkInterfaces } from 'node:os'
import { isIP } from 'node:net'
import type { Duplex } from 'node:stream'
import { timingSafeEqual } from 'node:crypto'
import { PairingAttemptLimiter } from './remote-control'
import type { MobileBridgeCertificate } from './mobile-bridge-certificate'
import type { WatchSpeechBroker } from './watch-speech'
import type { WatchSnapshot } from '../../shared/watch-relay'
import { createHostWatchCommandRelay } from './watch-command-relay'
import type { MobileBridgeConfig, MobileBridgeInterface, MobileBridgePairing, MobileBridgeStatus } from './types/mobile-bridge'

export const MOBILE_BRIDGE_REQUEST_BYTES = 64 * 1024
export const MOBILE_BRIDGE_RESPONSE_BYTES = 8 * 1024 * 1024
const REQUEST_TIMEOUT_MS = 15_000
const MAX_CLIENTS = 16

export function isMobilePrivateAddress(value: string): boolean {
  if (isIP(value) !== 4) return false
  const [a, b] = value.split('.').map(Number)
  return a === 10 || a === 192 && b === 168 || a === 172 && b >= 16 && b <= 31
}

export function mobileBridgeInterfaces(): MobileBridgeInterface[] {
  return Object.entries(networkInterfaces()).flatMap(([name, values]) => (values ?? []).filter(item => !item.internal && item.family === 'IPv4' && isMobilePrivateAddress(item.address)).map(item => ({ name, address: item.address })))
}

export function validateMobileBridgeConfig(value: MobileBridgeConfig, interfaces: MobileBridgeInterface[]): void {
  if (value.version !== 1 || typeof value.enabled !== 'boolean' || !isMobilePrivateAddress(value.host) || !interfaces.some(item => item.address === value.host) || !Number.isInteger(value.port) || value.port < 1024 || value.port > 65535) throw new Error('Select an available private network interface and a port from 1024 to 65535.')
}

export function mobileBridgeRouteAllowed(method: string, pathname: string): boolean {
  if (method === 'GET') return pathname === '/v1/snapshot' || pathname === '/v1/device-status' || pathname === '/v1/watch/speech/status' || /^\/v1\/watch\/speech\/[a-zA-Z0-9-]{8,80}\/chunks\/\d{1,5}$/.test(pathname)
  if (method !== 'POST') return false
  return pathname === '/v1/auth/pair' || pathname === '/v1/watch/command' || pathname === '/v1/watch/speech/start' || /^\/v1\/watch\/speech\/[a-zA-Z0-9-]{8,80}\/cancel$/.test(pathname) || /^\/v1\/threads\/[^/]{1,1024}\/(?:start|guide|cancel)$/.test(pathname)
}

function constantEqual(expected: string, presented: string): boolean {
  const left = Buffer.from(expected), right = Buffer.from(presented)
  return left.length > 0 && left.length === right.length && timingSafeEqual(left, right)
}

function fail(code: number, error: string): never { throw Object.assign(new Error(error), { code }) }
export interface MobileBridgeRequestMetadata {
  target: string
  method: string
  expectedOrigin: string
  hostHeader: string | undefined
  origin?: string
  remoteAddress: string
  authorization?: string
  hasCookie?: boolean
  hasLegacyToken?: boolean
}

/** The listener and deterministic security tests use the same policy gate. */
export function authorizeMobileBridgeRequest(metadata: MobileBridgeRequestMetadata, ownerToken: string, authorizeSessionToken: MobileBridgeDeps['authorizeSessionToken']): { pathname: string; claim: boolean; token: string } {
  if (!metadata.target.startsWith('/') || metadata.target.startsWith('//')) fail(400, 'invalid_request_target')
  const expected = new URL(metadata.expectedOrigin), url = new URL(metadata.target, metadata.expectedOrigin)
  if (metadata.hostHeader !== expected.host || metadata.origin && metadata.origin !== metadata.expectedOrigin || url.origin !== metadata.expectedOrigin || url.search || url.hash) fail(403, 'origin_denied')
  if (!isMobilePrivateAddress(metadata.remoteAddress)) fail(403, 'private_client_required')
  if (!mobileBridgeRouteAllowed(metadata.method, url.pathname)) fail(404, 'route_not_available')
  if (metadata.hasCookie || metadata.hasLegacyToken) fail(401, 'bearer_header_required')
  const raw = metadata.authorization || '', token = /^Bearer ([A-Za-z0-9_-]{20,256})$/.exec(raw)?.[1] || ''
  if (token && constantEqual(ownerToken, token)) fail(403, 'owner_token_not_allowed')
  const claim = metadata.method === 'POST' && url.pathname === '/v1/auth/pair'
  if (!claim) {
    const auth = token ? authorizeSessionToken(token) : { ok: false }
    if (!auth.ok || !auth.sessionId || auth.scope !== undefined && auth.scope !== 'paired') fail(401, 'paired_session_required')
  } else if (raw) fail(401, 'pairing_body_required')
  return { pathname: url.pathname, claim, token }
}
function json(res: ServerResponse, status: number, value: unknown): void {
  if (res.writableEnded || res.destroyed) return
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer' })
  res.end(JSON.stringify(value))
}

async function requestBody(req: IncomingMessage): Promise<Buffer> {
  if (req.method !== 'POST') {
    if (req.headers['content-length'] && req.headers['content-length'] !== '0' || req.headers['transfer-encoding']) fail(400, 'unexpected_request_body')
    return Buffer.alloc(0)
  }
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '') || req.headers['content-encoding'] && req.headers['content-encoding'] !== 'identity') fail(415, 'json_required')
  const length = req.headers['content-length']
  if (length && (!/^\d+$/.test(length) || Number(length) > MOBILE_BRIDGE_REQUEST_BYTES)) fail(413, 'request_too_large')
  const chunks: Buffer[] = []; let bytes = 0
  for await (const part of req) {
    const chunk = Buffer.isBuffer(part) ? part : Buffer.from(part)
    bytes += chunk.length
    if (bytes > MOBILE_BRIDGE_REQUEST_BYTES) fail(413, 'request_too_large')
    chunks.push(chunk)
  }
  return Buffer.concat(chunks)
}

export interface MobileBridgeDeps {
  interfaces?: () => MobileBridgeInterface[]
  requireOwner: () => void
  loopbackUrl: () => string | null
  ownerToken: () => string
  authorizeSessionToken: (token: string) => { ok: boolean; sessionId?: string; scope?: 'paired' | 'owner' }
  issuePairingLink: (label: string) => Promise<{ credential: string; expiresAt: number }>
  loadCertificate: (host: string) => Promise<MobileBridgeCertificate>
  audit?: (kind: string, data: Record<string, unknown>) => void
  reserveRequestOperation?: () => { release(): void }
  watchSpeech?: WatchSpeechBroker
  projectWatchSnapshot?: (snapshot: WatchSnapshot, sessionId: string) => WatchSnapshot
}

export function createMobileBridge(deps: MobileBridgeDeps) {
  let server: Server | null = null, certificate: MobileBridgeCertificate | null = null
  let config: MobileBridgeConfig = { version: 1, enabled: false, host: '', port: 8766 }
  let lastError: string | null = null
  const sockets = new Set<Duplex>(), inflight = new Set<ReturnType<typeof loopbackRequest>>()
  const nonces = new Map<string, { expiresAt: number; consumed: boolean }>()
  const pairingAttempts = new PairingAttemptLimiter()
  const interfaces = deps.interfaces ?? mobileBridgeInterfaces
  let pendingRequests = 0, activeMutations = 0, lifecycle = 0
  let pendingServer: Server | null = null
  function upstream(): URL {
    const raw = deps.loopbackUrl()
    if (!raw) fail(503, 'local_orchestrator_unavailable')
    const url = new URL(raw)
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.pathname !== '/' || url.search || url.hash || url.username || url.password) fail(503, 'invalid_local_orchestrator')
    return url
  }

  async function forward(method: string, pathname: string, body: Buffer, token: string, signal: AbortSignal): Promise<{ status: number; value: unknown }> {
    const target = upstream()
    return new Promise((resolve, reject) => {
      const req = loopbackRequest({ hostname: '127.0.0.1', port: Number(target.port), path: pathname, method, signal,
        headers: { accept: 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}), ...(method === 'POST' ? { 'content-type': 'application/json', 'content-length': body.length } : {}) } }, async response => {
        try {
          if (!response.statusCode || response.statusCode >= 300 && response.statusCode < 400 || !/^application\/json(?:\s*;|$)/i.test(response.headers['content-type'] || '')) fail(502, 'invalid_orchestrator_response')
          const declared = response.headers['content-length']
          if (declared && Number(declared) > MOBILE_BRIDGE_RESPONSE_BYTES) fail(502, 'orchestrator_response_too_large')
          const chunks: Buffer[] = []; let bytes = 0
          for await (const part of response) {
            const chunk = Buffer.isBuffer(part) ? part : Buffer.from(part)
            bytes += chunk.length
            if (bytes > MOBILE_BRIDGE_RESPONSE_BYTES) fail(502, 'orchestrator_response_too_large')
            chunks.push(chunk)
          }
          resolve({ status: response.statusCode, value: JSON.parse(Buffer.concat(chunks).toString('utf8')) })
        } catch (error) { response.destroy(); reject(error) }
      })
      inflight.add(req)
      req.on('close', () => inflight.delete(req))
      req.on('error', () => reject(Object.assign(new Error(signal.aborted ? 'request_timeout' : 'local_orchestrator_unavailable'), { code: signal.aborted ? 504 : 503 })))
      req.end(body)
    })
  }

  let watchCommands: ReturnType<typeof createHostWatchCommandRelay> | null = null
  function watchCommandRelay() {
    if (!server && !pendingServer) fail(503, 'mobile_bridge_stopped')
    return watchCommands ??= createHostWatchCommandRelay({
      authorize: deps.authorizeSessionToken, speech: deps.watchSpeech,
      request: (method, pathname, body, token, signal) => forward(method, pathname, body === undefined ? Buffer.alloc(0) : Buffer.from(JSON.stringify(body)), token, signal),
    })
  }

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const origin = `https://${config.host}:${config.port}`
    const gate = authorizeMobileBridgeRequest({ target: req.url || '', method: req.method || '', expectedOrigin: origin, hostHeader: req.headers.host, origin: req.headers.origin, remoteAddress: req.socket.remoteAddress || '', authorization: req.headers.authorization, hasCookie: !!req.headers.cookie, hasLegacyToken: !!req.headers['x-vontra-token'] }, deps.ownerToken(), deps.authorizeSessionToken)
    if (!interfaces().some(item => item.address === config.host)) fail(503, 'private_interface_unavailable')
    const { claim, token: presented, pathname } = gate
    const controller = new AbortController(), deadline = setTimeout(() => { controller.abort(); json(res, 504, { error: 'request_timeout' }); req.destroy() }, REQUEST_TIMEOUT_MS)
    const abort = () => { if (!res.writableEnded) controller.abort() }
    res.once('close', abort)
    let reservation: { release(): void } | undefined
    let countedMutation = false
    try {
      const body = await requestBody(req)
      let payload: Record<string, unknown> = {}
      if (req.method === 'POST') {
        try { const parsed = JSON.parse(body.toString('utf8')); if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) fail(400, 'invalid_json'); payload = parsed }
        catch { fail(400, 'invalid_json') }
      }
      if (req.method === 'POST') {
        try { reservation = deps.reserveRequestOperation?.() }
        catch { fail(409, 'host_operation_in_progress') }
        activeMutations++
        countedMutation = true
      }
      if (req.method === 'POST' && pathname === '/v1/watch/command') {
        if (body.length > 12_000) fail(413, 'watch_request_too_large')
        let value = await watchCommandRelay().handle(presented, payload, controller.signal)
        const current = deps.authorizeSessionToken(presented)
        if (!current.ok || !current.sessionId || current.scope !== undefined && current.scope !== 'paired' || controller.signal.aborted) fail(401, 'paired_session_required')
        if (value.snapshot) {
          if (!deps.projectWatchSnapshot) fail(503, 'watch_projection_unavailable')
          if (value.snapshot.session !== current.sessionId) fail(401, 'paired_session_required')
          value = {...value, snapshot: deps.projectWatchSnapshot(value.snapshot, current.sessionId)}
        }
        if (typeof payload.action === 'string' && payload.action.startsWith('audio-')) {
          const speechId = payload.action === 'audio-stop' ? undefined : value.audio?.speechId ?? value.speech?.speechId ?? (typeof payload.speechId === 'string' ? payload.speechId : undefined)
          if (!deps.watchSpeech?.isAllowed(current.sessionId, speechId)) fail(403, 'watch_speech_permission_required')
        }
        // No awaits after current privacy, exact speech, role and identity fences.
        json(res, 200, value); return
      }
      if (pathname.startsWith('/v1/watch/speech/')) {
        const session = deps.authorizeSessionToken(presented)
        if (!session.ok || !session.sessionId) fail(401, 'paired_session_required')
        if (!deps.watchSpeech) fail(501, 'kokoro_watch_unavailable')
        let value: unknown
        if (req.method === 'GET' && pathname === '/v1/watch/speech/status') value = await deps.watchSpeech.status(session.sessionId)
        else if (req.method === 'POST' && pathname === '/v1/watch/speech/start') {
          if (Object.keys(payload).some(key => !['threadId', 'itemId'].includes(key)) || typeof payload.threadId !== 'string' || typeof payload.itemId !== 'string') fail(400, 'invalid_watch_speech_request')
          value = await deps.watchSpeech.start(session.sessionId, {threadId: payload.threadId, itemId: payload.itemId})
        } else {
          const chunk = /^\/v1\/watch\/speech\/([a-zA-Z0-9-]{8,80})\/chunks\/(\d{1,5})$/.exec(pathname)
          const cancel = /^\/v1\/watch\/speech\/([a-zA-Z0-9-]{8,80})\/cancel$/.exec(pathname)
          if (req.method === 'GET' && chunk) value = await deps.watchSpeech.chunk(session.sessionId, chunk[1], Number(chunk[2]))
          else if (req.method === 'POST' && cancel && !Object.keys(payload).length) value = deps.watchSpeech.cancel(session.sessionId, cancel[1])
          else fail(400, 'invalid_watch_speech_request')
        }
        if (pathname !== '/v1/watch/speech/status') {
          const speechId = pathname.endsWith('/cancel') ? undefined
            : value && typeof value === 'object' && 'speechId' in value && typeof value.speechId === 'string' ? value.speechId : undefined
          if (!deps.watchSpeech.isAllowed(session.sessionId, speechId)) fail(403, 'watch_speech_permission_required')
        }
        if (!deps.authorizeSessionToken(presented).ok || controller.signal.aborted) fail(401, 'paired_session_required')
        json(res, 200, value); return
      }
      let nonce: { expiresAt: number; consumed: boolean } | undefined
      if (claim) {
        const source = req.socket.remoteAddress || 'unknown', rate = pairingAttempts.check(source)
        if (rate.blocked) { res.setHeader('retry-after', String(Math.ceil(rate.retryAfterMs / 1000))); fail(429, 'pairing_rate_limited') }
        if (typeof payload.credential !== 'string' || Object.keys(payload).some(key => !['credential', 'clientInfo'].includes(key)) || payload.clientInfo !== undefined && (typeof payload.clientInfo !== 'string' || payload.clientInfo.length > 120)) fail(400, 'invalid_pairing_body')
        nonce = nonces.get(payload.credential)
        if (!nonce || nonce.consumed || nonce.expiresAt <= Date.now()) {
          pairingAttempts.recordFailure(source)
          fail(400, !nonce ? 'unknown_credential' : nonce.consumed ? 'already_consumed' : 'expired')
        }
        nonce.consumed = true
      } else if (req.method === 'POST') {
        const cancel = pathname.endsWith('/cancel')
        if (cancel ? Object.keys(payload).length !== 0 : Object.keys(payload).some(key => key !== 'userText') || typeof payload.userText !== 'string' || !payload.userText.trim() || payload.userText.length > 8000) fail(400, 'invalid_task_request')
      }
      const result = await forward(req.method!, pathname, body, claim ? '' : presented, controller.signal)
      if (claim && result.status === 200) pairingAttempts.clear(req.socket.remoteAddress || 'unknown')
      json(res, result.status, result.value)
    } finally {
      reservation?.release()
      if (countedMutation) activeMutations--
      clearTimeout(deadline); res.removeListener('close', abort)
    }
  }

  async function closeListener(): Promise<void> {
    const current = server || pendingServer; server = null; pendingServer = null; certificate = null
    nonces.clear()
    const previousWatchCommands = watchCommands; watchCommands = null
    await previousWatchCommands?.stop()
    deps.watchSpeech?.stop()
    for (const req of inflight) req.destroy()
    for (const socket of sockets) socket.destroy()
    sockets.clear()
    if (current) await new Promise<void>(resolve => current.close(() => resolve()))
  }

  async function stop(): Promise<void> {
    lifecycle++
    await closeListener()
  }

  async function configure(next: MobileBridgeConfig, restoring = false): Promise<MobileBridgeStatus> {
    deps.requireOwner()
    // Claim this choice before yielding: two same-tick requests must have distinct owners.
    const generation = ++lifecycle
    config = { version: 1, enabled: false, host: next.host, port: next.port }; lastError = null
    await closeListener()
    if (generation !== lifecycle) throw new Error('This mobile connection request was replaced by a newer owner choice.')
    if (!next.enabled) return status()
    try {
      validateMobileBridgeConfig(next, interfaces()); upstream()
      const tls = await deps.loadCertificate(next.host)
      if (generation !== lifecycle) throw new Error('The mobile connection was disabled while preparing its certificate.')
      const candidate = createServer({ cert: tls.cert, key: tls.key, minVersion: 'TLSv1.2', maxHeaderSize: 8192 }, (req, res) => {
        if (pendingRequests >= MAX_CLIENTS) { json(res, 503, { error: 'mobile_bridge_busy' }); return }
        pendingRequests++
        void handle(req, res).catch((error: unknown) => {
          const code = typeof (error as { code?: unknown })?.code === 'number' ? (error as { code: number }).code : 500
          json(res, code, { error: code === 500 ? 'mobile_bridge_error' : (error as Error).message })
        }).finally(() => { pendingRequests-- })
      })
      candidate.maxConnections = MAX_CLIENTS
      candidate.requestTimeout = REQUEST_TIMEOUT_MS; candidate.headersTimeout = 10_000; candidate.keepAliveTimeout = 2000
      candidate.setTimeout(REQUEST_TIMEOUT_MS, socket => socket.destroy())
      candidate.on('connection', socket => { sockets.add(socket); socket.on('close', () => sockets.delete(socket)) })
      pendingServer = candidate
      await new Promise<void>((resolve, reject) => { candidate.once('error', reject); candidate.listen(next.port, next.host, () => { candidate.removeListener('error', reject); resolve() }) })
      if (generation !== lifecycle) { candidate.close(); throw new Error('The mobile connection was disabled while starting.') }
      candidate.on('error', () => { if (server === candidate) { lastError = 'The mobile connection stopped listening.'; void stop() } })
      pendingServer = null; server = candidate; certificate = tls; config = { ...next }
      deps.audit?.('mobile_bridge.enabled', { restoring, port: next.port })
      return status()
    } catch (error) {
      // An older certificate/listener attempt no longer owns the current connection.
      if (generation !== lifecycle) throw new Error('This mobile connection request was replaced by a newer owner choice.')
      lastError = error instanceof Error && error.message === 'local_orchestrator_unavailable' ? 'TensorFold\'s local Orchestrator is not running.' : error instanceof Error && /Select an available|private|certificate|disabled/.test(error.message) ? error.message : 'The mobile connection could not start on the selected interface and port.'
      const message = lastError
      await stop()
      throw new Error(message)
    }
  }

  function status(): MobileBridgeStatus {
    return { enabled: config.enabled, listening: !!server, url: server ? `https://${config.host}:${config.port}` : null, host: config.host || null, port: server ? config.port : null, certificateSha256: certificate?.sha256 ?? null, certificateExpiresAt: certificate?.expiresAt ?? null, interfaces: interfaces(), clients: sockets.size, lastError }
  }

  async function issuePairing(label: string): Promise<MobileBridgePairing> {
    deps.requireOwner()
    if (!server || !certificate) throw new Error('Enable the private mobile connection before pairing a phone.')
    for (const [credential, nonce] of nonces) if (nonce.expiresAt <= Date.now() || nonce.consumed) nonces.delete(credential)
    if (nonces.size >= 32) throw new Error('Too many active mobile pairing codes; wait for a code to expire.')
    const link = await deps.issuePairingLink(label.trim().slice(0, 120) || 'Paired phone')
    if (!/^[A-Za-z0-9_-]{20,256}$/.test(link.credential) || !Number.isSafeInteger(link.expiresAt) || link.expiresAt <= Date.now() || link.expiresAt > Date.now() + 10 * 60_000 + 1000) throw new Error('The mobile pairing credential is invalid.')
    nonces.set(link.credential, { expiresAt: link.expiresAt, consumed: false })
    return { link: `${status().url}/#pair=${encodeURIComponent(link.credential)}&pin=${certificate.sha256}`, expiresAt: link.expiresAt, certificateSha256: certificate.sha256 }
  }

  return { configure, status, issuePairing, stop, activeOperations: () => activeMutations }
}
