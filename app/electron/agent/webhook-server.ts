// Inbound webhook listener (build.md Phase 2, inbound-first).
//
// A loopback-only (127.0.0.1) HTTP server that is DISABLED BY DEFAULT — it only
// listens while at least one inbound endpoint is enabled. Every delivery passes,
// in order: method → route → content-type → size → timestamp/replay → HMAC →
// idempotency dedupe → JSON parse → normalize to a ConnectorEvent (redacted via
// connector-spine) → route policy. Payload values are NEVER shelled or eval'd;
// 'run_automation' only runs an existing, user-authored automation by id.
import { createServer, type Server } from 'node:http'
import { createHmac, timingSafeEqual } from 'node:crypto'
import { recordConnectorInteraction } from './connector-spine'
import { isBotChatAllowed, parseAllowedBotChatIds } from './bot-control'
import { WHATSAPP_CLOUD_WEBHOOK_PATH, parseWhatsAppCloudWebhook, verifyWhatsAppCloudChallenge, verifyWhatsAppCloudSignature } from './whatsapp-cloud-webhook'
import { createSerialTaskQueue } from './serial-task-queue'
import type { BotControlIngestInput, ConnectorEvent, WebhookEndpoint, WebhookServerStatus, WebhookCapturePolicy, WhatsAppBotControlSettings } from './types'

const HOST = '127.0.0.1'
const MAX_BODY_BYTES = 1_000_000 // 1 MB
const ALLOWED_CONTENT_TYPES = ['application/json']
const SIG_HEADER = 'x-vontra-signature'
const TS_HEADER = 'x-vontra-timestamp'
const ID_HEADER = 'x-vontra-idempotency'

export interface WebhookServerDeps {
  getEndpoints: () => WebhookEndpoint[]
  onConnectorEvent: (e: ConnectorEvent) => void
  runAutomationById: (id: string, source?: { eventId: string; connectorId: ConnectorEvent['connectorId']; action: string; threadKey: string }) => void
  audit: (kind: string, data: Record<string, unknown>) => void
  /** Optional signed Meta Cloud bot ingress. It is main-process only, never renderer IPC. */
  getWhatsAppSettings?: () => WhatsAppBotControlSettings
  onWhatsAppBotInput?: (input: BotControlIngestInput) => void
}

export interface WebhookDelivery {
  pathname: string
  method: string
  contentType: string
  timestamp: string | undefined
  signature: string | undefined
  idempotency: string | undefined
  rawBody: Buffer
}

export interface WebhookServer {
  syncFromEndpoints(): Promise<void>
  status(): WebhookServerStatus
  stop(): Promise<void>
  handleDelivery(opts: WebhookDelivery): { code: number; body: unknown } // exported for tests
  handleWhatsAppChallenge(query: URLSearchParams): HttpResult
  handleWhatsAppDelivery(opts: Pick<WebhookDelivery, 'method' | 'contentType' | 'signature' | 'rawBody'>): HttpResult
}

interface HttpResult {
  code: number
  body: unknown
  contentType?: string
  raw?: boolean
}

// Canonical signed material: `${timestamp}.` + rawBody — binds the timestamp to the body.
export function signWebhook(secret: string, timestamp: string, rawBody: Buffer | string): string {
  return createHmac('sha256', secret).update(`${timestamp}.`).update(rawBody).digest('hex')
}

function header(v: string | string[] | undefined): string | undefined { return Array.isArray(v) ? v[0] : v }

export const DEFAULT_WEBHOOK_REPLAY_WINDOW_SECONDS = 300
export const MAX_WEBHOOK_REPLAY_WINDOW_SECONDS = 86_400

/** Canonical replay window used at save time and again at the enforcement boundary. */
export function normalizeWebhookReplayWindowSeconds(value: unknown, fallback = DEFAULT_WEBHOOK_REPLAY_WINDOW_SECONDS): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return fallback
  return Math.max(1, Math.min(MAX_WEBHOOK_REPLAY_WINDOW_SECONDS, Math.floor(n)))
}

export function createWebhookServer(deps: WebhookServerDeps): WebhookServer {
  let server: Server | null = null
  let port: number | null = null
  let lastDeliveryAt: number | null = null
  const seen = new Map<string, number>() // dedupe key → expiry (epoch ms)
  const lifecycleQueue = createSerialTaskQueue()

  const inboundEnabled = () => deps.getEndpoints().filter((e) => e.direction === 'inbound' && e.enabled && !!e.path)
  const whatsappSettings = () => deps.getWhatsAppSettings?.() ?? null
  const whatsappEnabled = () => {
    const settings = whatsappSettings()
    return !!(deps.onWhatsAppBotInput && settings?.enabled && settings.accessToken && settings.phoneNumberId && settings.verifyToken && settings.appSecret && parseAllowedBotChatIds(settings.allowedChatIds).size)
  }

  function verify(endpoint: WebhookEndpoint, timestamp: string | undefined, rawBody: Buffer, signature: string | undefined): boolean {
    const ts = Number(timestamp)
    if (!timestamp || !Number.isFinite(ts)) return false
    const replayWindowSeconds = normalizeWebhookReplayWindowSeconds(endpoint.replayWindowSeconds)
    if (Math.abs(Date.now() / 1000 - ts) > replayWindowSeconds) return false
    if (!signature || !endpoint.secretRef) return false
    let provided: Buffer
    try { provided = Buffer.from(signature, 'hex') } catch { return false }
    const expected = Buffer.from(signWebhook(endpoint.secretRef, timestamp, rawBody), 'hex')
    if (provided.length !== expected.length || provided.length === 0) return false
    return timingSafeEqual(provided, expected)
  }

  function capture(parsed: unknown, policy: WebhookCapturePolicy): Record<string, unknown> {
    const obj = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : { value: parsed }
    if (policy === 'metadata') {
      const out: Record<string, unknown> = {}
      if (typeof obj.title === 'string') out.title = obj.title
      else if (typeof obj.subject === 'string') out.title = obj.subject
      if (typeof obj.summary === 'string') out.summary = obj.summary
      else if (typeof obj.message === 'string') out.summary = obj.message
      return out
    }
    return obj // 'redacted_payload'/'ttl_payload': connector-spine's safeKeys keeps only key NAMES, never values
  }

  // Pure delivery handler — no socket, fully unit-testable. Fails closed with generic 4xx.
  function handleDelivery(opts: WebhookDelivery): { code: number; body: unknown } {
    const now = Date.now()
    if (opts.method !== 'POST') return { code: 405, body: { error: 'method_not_allowed' } }
    const endpoint = inboundEnabled().find((e) => e.path === opts.pathname)
    if (!endpoint) return { code: 404, body: { error: 'not_found' } } // don't reveal which paths exist
    const ct = (opts.contentType || '').split(';')[0].trim().toLowerCase()
    if (!ALLOWED_CONTENT_TYPES.includes(ct)) return { code: 415, body: { error: 'unsupported_media_type' } }
    if (opts.rawBody.length > MAX_BODY_BYTES) return { code: 413, body: { error: 'payload_too_large' } }
    if (!verify(endpoint, opts.timestamp, opts.rawBody, opts.signature)) {
      deps.audit('webhook.reject', { endpoint: endpoint.id, reason: 'signature_or_timestamp' })
      return { code: 401, body: { error: 'unauthorized' } }
    }
    // Replay dedupe keyed on the SIGNED material (endpoint + timestamp + body) — never on
    // the unsigned client idempotency header, which an attacker can vary to replay a
    // captured-valid delivery within the window (re-firing run_automation each time).
    for (const [k, exp] of seen) if (exp <= now) seen.delete(k)
    const dedupeKey = `${endpoint.id}:${createHmac('sha256', 'vontra-dedupe').update(`${opts.timestamp}.`).update(opts.rawBody).digest('hex')}`
    if (seen.has(dedupeKey)) return { code: 200, body: { ok: true, deduped: true } }
    const lastAcceptedAt = (Number(opts.timestamp) + normalizeWebhookReplayWindowSeconds(endpoint.replayWindowSeconds)) * 1000
    seen.set(dedupeKey, Math.ceil(lastAcceptedAt) + 1)
    let parsed: unknown
    try { parsed = JSON.parse(opts.rawBody.toString('utf8')) } catch { return { code: 400, body: { error: 'invalid_json' } } }
    // The ONE safe path: connector-spine redacts sensitive keys + caps + classifies.
    const event = recordConnectorInteraction({
      connectorId: 'webhook', pluginName: endpoint.name, action: 'webhook_received', direction: 'inbound',
      target: endpoint.path, input: capture(parsed, endpoint.capturePolicy),
    })
    deps.onConnectorEvent(event)
    lastDeliveryAt = now
    deps.audit('webhook.received', { endpoint: endpoint.id, policy: endpoint.triggerPolicy, eventId: event.id })
    // route policy — notify/suggest/approval are carried by the event's trust/classification.
    if (endpoint.triggerPolicy === 'run_automation' && endpoint.runAutomationId) {
      deps.runAutomationById(endpoint.runAutomationId, { eventId: event.id, connectorId: event.connectorId, action: event.action, threadKey: event.threadKey })
    }
    return { code: 202, body: { ok: true, eventId: event.id } }
  }

  function handleWhatsAppChallenge(query: URLSearchParams): HttpResult {
    const settings = whatsappSettings()
    if (!settings || !whatsappEnabled()) return { code: 404, body: { error: 'not_found' } }
    const challenge = verifyWhatsAppCloudChallenge(query, settings.verifyToken)
    if (challenge == null) {
      deps.audit('whatsapp.webhook_reject', { reason: 'verification_token' })
      return { code: 401, body: { error: 'unauthorized' } }
    }
    deps.audit('whatsapp.webhook_verified', {})
    return { code: 200, body: challenge, contentType: 'text/plain; charset=utf-8', raw: true }
  }

  function handleWhatsAppDelivery(opts: Pick<WebhookDelivery, 'method' | 'contentType' | 'signature' | 'rawBody'>): HttpResult {
    const settings = whatsappSettings()
    if (!settings || !whatsappEnabled()) return { code: 404, body: { error: 'not_found' } }
    if (opts.method !== 'POST') return { code: 405, body: { error: 'method_not_allowed' } }
    const ct = (opts.contentType || '').split(';')[0].trim().toLowerCase()
    if (!ALLOWED_CONTENT_TYPES.includes(ct)) return { code: 415, body: { error: 'unsupported_media_type' } }
    if (opts.rawBody.length > MAX_BODY_BYTES) return { code: 413, body: { error: 'payload_too_large' } }
    if (!verifyWhatsAppCloudSignature(opts.rawBody, opts.signature, settings.appSecret)) {
      deps.audit('whatsapp.webhook_reject', { reason: 'signature' })
      return { code: 401, body: { error: 'unauthorized' } }
    }
    let inputs: BotControlIngestInput[]
    try { inputs = parseWhatsAppCloudWebhook(opts.rawBody, settings.phoneNumberId) } catch {
      deps.audit('whatsapp.webhook_reject', { reason: 'payload' })
      return { code: 400, body: { error: 'invalid_payload' } }
    }
    let accepted = 0
    let failed = 0
    for (const input of inputs) {
      if (!input.senderId || !isBotChatAllowed(settings.allowedChatIds, input.senderId)) {
        deps.audit('whatsapp.webhook_reject', { reason: 'principal' })
        continue
      }
      try {
        deps.onWhatsAppBotInput?.(input)
        accepted++
      } catch {
        failed++
        deps.audit('whatsapp.webhook_error', { reason: 'dispatch' })
      }
    }
    if (accepted) lastDeliveryAt = Date.now()
    deps.audit('whatsapp.webhook_received', { delivered: inputs.length, accepted, failed })
    return failed
      ? { code: 500, body: { error: 'dispatch_failed' } }
      : { code: 200, body: { ok: true, accepted } }
  }

  async function syncFromEndpointsNow(): Promise<void> {
    const need = inboundEnabled().length > 0 || whatsappEnabled()
    if (need && !server) {
      await new Promise<void>((resolve) => {
        const s = createServer((req, res) => {
          const chunks: Buffer[] = []
          let size = 0, responded = false
          const respond = (result: HttpResult, closeAfter = false) => {
            if (responded) return
            responded = true
            try {
              // For an oversized/aborted request we still owe the client a real HTTP response,
              // but the client may keep sending the (rejected) body — close the connection after
              // flushing the response so we neither read the rest nor reset the socket mid-write.
              const headers: Record<string, string> = { 'content-type': result.contentType ?? 'application/json' }
              if (closeAfter) {
                headers['connection'] = 'close'
                // Drain any further inbound bytes to nowhere so they don't buffer; do NOT
                // destroy the request here — that would abort the response before it flushes.
                req.resume()
              }
              res.writeHead(result.code, headers)
              res.end(result.raw ? String(result.body) : JSON.stringify(result.body))
            } catch { /* ignore: connection already torn down */ }
          }
          req.on('data', (c: Buffer) => {
            if (responded) return
            size += c.length
            if (size > MAX_BODY_BYTES) {
              // Send the 413 now: the old code called req.destroy() and deferred the response to
              // the 'end' handler, but destroy() never emits 'end', so the client saw a socket
              // reset (ECONNRESET) with no status instead of the documented 413.
              respond({ code: 413, body: { error: 'payload_too_large' } }, true)
            } else {
              chunks.push(c)
            }
          })
          req.on('end', () => {
            if (responded) return
            let result: HttpResult
            try {
              const url = new URL(req.url || '/', `http://${HOST}`)
              if (url.pathname === WHATSAPP_CLOUD_WEBHOOK_PATH && req.method === 'GET') {
                result = handleWhatsAppChallenge(url.searchParams)
              } else if (url.pathname === WHATSAPP_CLOUD_WEBHOOK_PATH) {
                result = handleWhatsAppDelivery({ method: req.method || 'GET', contentType: String(req.headers['content-type'] || ''), signature: header(req.headers['x-hub-signature-256']), rawBody: Buffer.concat(chunks) })
              } else {
                result = handleDelivery({ pathname: url.pathname, method: req.method || 'GET', contentType: String(req.headers['content-type'] || ''), timestamp: header(req.headers[TS_HEADER]), signature: header(req.headers[SIG_HEADER]), idempotency: header(req.headers[ID_HEADER]), rawBody: Buffer.concat(chunks) })
              }
            } catch { result = { code: 400, body: { error: 'bad_request' } } }
            respond(result)
          })
          req.on('error', () => { if (!responded) { responded = true; try { res.writeHead(400); res.end() } catch { /* ignore */ } } })
        })
        s.once('error', () => {
          // An old listener can report an error after a stop/restart. Never let it null
          // the newer winning server reference.
          if (server === s) { server = null; port = null }
          resolve()
        })
        s.listen(0, HOST, () => { const addr = s.address(); port = typeof addr === 'object' && addr ? addr.port : null; server = s; resolve() })
      })
    } else if (!need && server) {
      await stopNow()
    }
  }

  function status(): WebhookServerStatus { return { listening: !!server, port, routes: inboundEnabled().length + (whatsappEnabled() ? 1 : 0), lastDeliveryAt } }
  async function stopNow(): Promise<void> { const s = server; server = null; port = null; if (s) await new Promise<void>((r) => s.close(() => r())) }
  function syncFromEndpoints(): Promise<void> { return lifecycleQueue.run('listener', syncFromEndpointsNow) }
  function stop(): Promise<void> { return lifecycleQueue.run('listener', stopNow) }

  return { syncFromEndpoints, status, stop, handleDelivery, handleWhatsAppChallenge, handleWhatsAppDelivery }
}
