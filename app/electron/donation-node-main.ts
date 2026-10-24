/**
 * The donor side of a donation card: a tiny authenticated proxy in front of a local
 * OpenAI-compatible engine. It enforces the card's window, concurrency, body and output-token
 * caps, forwards the three routes a renter may use, and prints one JSON receipt per request.
 *
 * Run it next to the engine it lends:
 *   node donation-node.js --config donation.json
 *
 * POC: it keeps no ledger and proves nothing about the computation. It exists so a renter can
 * be held to a window and a ceiling, and so the donor can see what left the machine.
 */
import { createHash, timingSafeEqual } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { createServer as createHttpServer, request as httpRequest, type IncomingMessage, type ServerResponse } from 'node:http'
import { createServer as createHttpsServer, request as httpsRequest } from 'node:https'
import { parseDonationCard, donationPhase, type DonationCard } from '../shared/donation'

export interface DonationNodeConfig {
  card: unknown
  upstream: string
  listen: { host: string; port: number }
  upstreamToken?: string
  /** How long one proxied inference request may take before it is abandoned (ms). */
  upstreamTimeoutMs?: number
  tls?: { certFile: string; keyFile: string }
}

const MAX_CAPTURE = 256 * 1024
const UPSTREAM_TIMEOUT_MS = 120_000
/** setTimeout overflows above this; the window deadline is clamped to it. */
const MAX_TIMER_MS = 2_147_483_647
/** Framing, hop-by-hop and donor-only headers: never forwarded upstream. */
const DROP_HEADERS = new Set(['host', 'authorization', 'x-api-key', 'content-length', 'connection', 'keep-alive', 'transfer-encoding', 'te', 'trailer', 'upgrade', 'proxy-authorization', 'proxy-authenticate', 'http2-settings'])
const receiptLog = (line: Record<string, unknown>) => process.stdout.write(`receipt ${JSON.stringify(line)}\n`)

function readConfig(): DonationNodeConfig {
  const index = process.argv.indexOf('--config')
  const path = index >= 0 ? process.argv[index + 1] : process.env.TF_DONATION_CONFIG
  if (!path) throw new Error('Pass --config donation.json or set TF_DONATION_CONFIG.')
  const raw = JSON.parse(readFileSync(path, 'utf8')) as unknown
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new Error('The config must be a JSON object.')
  const config = raw as Partial<DonationNodeConfig>
  for (const key of Object.keys(config)) if (!['card', 'upstream', 'listen', 'upstreamToken', 'upstreamTimeoutMs', 'tls'].includes(key)) throw new Error(`Unknown config field: ${key}.`)
  if (typeof config.upstream !== 'string') throw new Error('The config needs `upstream`, e.g. http://127.0.0.1:8000.')
  const upstream = new URL(config.upstream)
  if (upstream.protocol !== 'http:' && upstream.protocol !== 'https:') throw new Error('The upstream must be http or https.')
  if (upstream.username || upstream.password || upstream.search || upstream.hash || (upstream.pathname !== '/' && upstream.pathname !== '')) throw new Error('The upstream must be an origin.')
  if (config.upstreamTimeoutMs !== undefined && (!Number.isSafeInteger(config.upstreamTimeoutMs) || config.upstreamTimeoutMs < 1_000 || config.upstreamTimeoutMs > 600_000)) throw new Error('`upstreamTimeoutMs` must be an integer from 1000 to 600000.')
  const listen = config.listen
  if (!listen || typeof listen.host !== 'string' || !Number.isSafeInteger(listen.port) || listen.port < 1 || listen.port > 65535) throw new Error('The config needs `listen: {host, port}`.')
  return { ...(config as DonationNodeConfig), upstream: upstream.origin }
}

function constantTimeEqual(a: string, b: string): boolean {
  const left = createHash('sha256').update(a).digest(), right = createHash('sha256').update(b).digest()
  return timingSafeEqual(left, right)
}

function bearer(req: IncomingMessage): string {
  const header = String(req.headers.authorization ?? '')
  if (header.toLowerCase().startsWith('bearer ')) return header.slice(7).trim()
  return String(req.headers['x-api-key'] ?? '').trim()
}

function loopback(host: string): boolean {
  return host === '127.0.0.1' || host === '::1' || host === 'localhost'
}

function fail(res: ServerResponse, status: number, message: string, type: string): void {
  if (res.headersSent || res.writableEnded) return
  const body = JSON.stringify({ error: { message, type } })
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) })
  res.end(body)
}

/**
 * The output-token limit a request asks for. Absent means the card's cap (never the engine's
 * default, which could exceed it); anything present must be a whole number within the cap.
 */
function readTokenLimit(body: Record<string, unknown>, field: string, cap: number): { value: number } | { error: string } {
  const raw = body[field]
  if (raw === undefined || raw === null) return { value: cap }
  if (typeof raw !== 'number' || !Number.isSafeInteger(raw) || raw <= 0 || raw > cap) return { error: `\`${field}\` must be a whole number from 1 to ${cap}.` }
  return { value: raw }
}

export interface DonationNodeOptions {
  /** Called once per request with its receipt; the CLI prints these to stdout. */
  onReceipt?: (receipt: Record<string, unknown>) => void
}

export interface DonationNodeHandle {
  card: DonationCard
  server: { close(callback?: (error?: Error) => void): unknown }
  close(): Promise<void>
}

export async function startDonationNode(config: DonationNodeConfig, options: DonationNodeOptions = {}): Promise<DonationNodeHandle> {
  const parsed = parseDonationCard(config.card)
  if ('error' in parsed) throw new Error(`The card is invalid: ${parsed.error}`)
  const card: DonationCard = parsed.card
  if (!loopback(config.listen.host) && card.security !== 'poc-insecure-lan') throw new Error('A non-loopback listener needs a `poc-insecure-lan` card. Use a TLS proxy for anything else.')
  if (card.security === 'tls' && !config.tls) throw new Error('A `tls` card needs `tls: {certFile, keyFile}` in the config.')
  const emit = (line: Record<string, unknown>) => { if (options.onReceipt) options.onReceipt(line); else receiptLog(line) }
  const forward = config.upstream.startsWith('https:') ? httpsRequest : httpRequest
  const timeoutMs = config.upstreamTimeoutMs ?? UPSTREAM_TIMEOUT_MS
  let active = 0

  /** Any synchronous throw inside a handler becomes a 500 for that request, not a dead donor. */
  const handler = (req: IncomingMessage, res: ServerResponse): void => {
    try { handle(req, res) } catch { fail(res, 500, 'The donor could not process this request.', 'donation_internal') }
  }

  const handle = (req: IncomingMessage, res: ServerResponse): void => {
    let url: URL
    try { url = new URL(req.url ?? '/', 'http://donation.invalid') } catch { return fail(res, 400, 'The request target is not a URL.', 'donation_target') }
    if (url.pathname === '/health') {
      const body = JSON.stringify({ status: 'ok', phase: donationPhase(card), until: card.window.until, model: card.model, active, maxConcurrent: card.caps.maxConcurrent, ...(card.spec ? { spec: card.spec } : {}) })
      res.writeHead(200, { 'content-type': 'application/json' }); res.end(body); return
    }
    if (!constantTimeEqual(bearer(req), card.token)) {
      if (process.env.TF_DONATION_DEBUG) process.stderr.write(`auth mismatch presented=${createHash('sha256').update(bearer(req)).digest('hex').slice(0, 8)} expected=${createHash('sha256').update(card.token).digest('hex').slice(0, 8)}\n`)
      return fail(res, 401, 'The donation token is missing or wrong.', 'donation_auth')
    }
    const phase = donationPhase(card)
    if (phase !== 'open') return fail(res, 403, phase === 'before' ? 'The donation window has not opened.' : 'The donation window has closed.', 'donation_window')
    if (url.pathname !== '/v1/models' && url.pathname !== '/v1/chat/completions' && url.pathname !== '/v1/responses') return fail(res, 404, 'Only models, chat completions and responses are lent.', 'donation_route')
    if (req.method !== (url.pathname === '/v1/models' ? 'GET' : 'POST')) return fail(res, 405, 'Method not allowed.', 'donation_method')
    // The slot is reserved before the body is read so concurrent uploads cannot all pass a
    // stale check and then reach the engine together. Every exit path releases it once.
    if (active >= card.caps.maxConcurrent) return fail(res, 429, 'The donor is at its concurrency cap.', 'donation_busy')
    active += 1
    let released = false
    const release = () => { if (released) return; released = true; active -= 1 }

    const chunks: Buffer[] = []
    let bytes = 0, refused = false
    req.on('data', (chunk: Buffer) => {
      if (refused) return
      bytes += chunk.length
      if (bytes > card.caps.maxRequestBytes) { refused = true; release(); fail(res, 413, 'The request body is larger than the card allows.', 'donation_bytes'); req.destroy(); return }
      chunks.push(chunk)
    })
    req.on('error', () => { release() })
    req.on('end', () => {
      if (refused) return
      // The window is rechecked at the moment the request would reach the engine, not only
      // when its headers arrived.
      if (donationPhase(card) !== 'open') { release(); return fail(res, 403, 'The donation window closed while this request uploaded.', 'donation_window') }
      const body = Buffer.concat(chunks)
      if (url.pathname === '/v1/models') return forwardTo(url.pathname, body)
      let parsedBody: unknown
      try { parsedBody = JSON.parse(body.toString('utf8') || '{}') } catch { release(); return fail(res, 400, 'The request body is not JSON.', 'donation_body') }
      if (typeof parsedBody !== 'object' || parsedBody === null || Array.isArray(parsedBody)) { release(); return fail(res, 400, 'The request body must be a JSON object.', 'donation_body') }
      const request = parsedBody as Record<string, unknown>
      if (url.pathname === '/v1/chat/completions') {
        if (request.model !== card.model) { release(); return fail(res, 400, `This donor serves only \`${card.model}\`.`, 'donation_model') }
        for (const field of ['n', 'best_of'] as const) if (request[field] !== undefined && request[field] !== 1) { release(); return fail(res, 400, 'This donor serves one completion per request.', 'donation_choices') }
        const limit = readTokenLimit(request, 'max_tokens', card.caps.maxTokensPerRequest)
        if ('error' in limit) { release(); return fail(res, 400, limit.error, 'donation_tokens') }
        const alternate = readTokenLimit(request, 'max_completion_tokens', card.caps.maxTokensPerRequest)
        if ('error' in alternate) { release(); return fail(res, 400, alternate.error, 'donation_tokens') }
        if (request.max_tokens !== undefined && request.max_completion_tokens !== undefined && request.max_tokens !== request.max_completion_tokens) { release(); return fail(res, 400, 'Send one output-token limit, not two different ones.', 'donation_tokens') }
        request.max_tokens = request.max_completion_tokens === undefined ? limit.value : alternate.value
        delete request.max_completion_tokens
      } else {
        if (request.model !== undefined && request.model !== card.model) { release(); return fail(res, 400, `This donor serves only \`${card.model}\`.`, 'donation_model') }
        const limit = readTokenLimit(request, 'max_output_tokens', card.caps.maxTokensPerRequest)
        if ('error' in limit) { release(); return fail(res, 400, limit.error, 'donation_tokens') }
        request.max_output_tokens = limit.value
      }
      forwardTo(url.pathname, Buffer.from(JSON.stringify(request)))
    })

    function forwardTo(pathname: string, payload: Buffer): void {
      const started = Date.now()
      const headers: Record<string, string | string[]> = {}
      const nominated = new Set(String(req.headers.connection ?? '').split(',').map((name) => name.trim().toLowerCase()).filter(Boolean))
      for (const [name, value] of Object.entries(req.headers)) {
        if (value === undefined || DROP_HEADERS.has(name) || nominated.has(name)) continue
        headers[name] = value
      }
      // The donor decompresses nothing: ask for identity bytes so stripping Content-Encoding
      // on the way back stays truthful.
      headers['accept-encoding'] = 'identity'
      if (config.upstreamToken) headers.authorization = `Bearer ${config.upstreamToken}`
      headers['content-length'] = String(payload.length)
      const upstream = new URL(pathname, config.upstream)
      const proxied = forward(upstream, { method: req.method, headers }, (answer) => {
        const status = answer.statusCode ?? 502
        const outHeaders: Record<string, string | string[]> = {}
        for (const [name, value] of Object.entries(answer.headers)) if (value !== undefined && !DROP_HEADERS.has(name) && name !== 'content-encoding') outHeaders[name] = value
        res.writeHead(status, { ...outHeaders, 'cache-control': 'no-store' })
        const captured: Buffer[] = []
        let capturedBytes = 0
        answer.on('data', (chunk: Buffer) => {
          if (capturedBytes < MAX_CAPTURE) {
            const piece = chunk.length > MAX_CAPTURE - capturedBytes ? chunk.subarray(0, MAX_CAPTURE - capturedBytes) : chunk
            captured.push(piece); capturedBytes += piece.length
          }
          // A slow renter must slow the engine down, not fill this process's memory.
          if (!res.write(chunk)) {
            answer.pause()
            res.once('drain', () => answer.resume())
          }
        })
        answer.on('end', () => {
          if (!res.writableEnded) res.end()
          release()
          const receipt: Record<string, unknown> = { at: new Date().toISOString(), card: card.id, route: pathname, status, ms: Date.now() - started, bytes: payload.length }
          if (capturedBytes <= MAX_CAPTURE) {
            try {
              const parsedAnswer = JSON.parse(Buffer.concat(captured).toString('utf8'))
              if (parsedAnswer?.usage) receipt.usage = parsedAnswer.usage
              if (parsedAnswer?.tensorfold?.token_sha) receipt.tokenSha = parsedAnswer.tensorfold.token_sha
              if (parsedAnswer?.model) receipt.model = parsedAnswer.model
            } catch { /* streamed or non-JSON: the receipt keeps timing only */ }
          }
          emit(receipt)
        })
        answer.on('error', () => { if (!res.writableEnded) res.end(); release() })
      })
      // An idle timeout, and no work past the card's own window.
      proxied.setTimeout(Math.min(timeoutMs, Math.max(1_000, card.window.until - Date.now()), MAX_TIMER_MS), () => proxied.destroy(new Error('The donor engine did not answer in time.')))
      // A renter that walks away must not keep holding a slot or an upstream request.
      res.on('close', () => { if (!res.writableEnded) proxied.destroy(new Error('The renter closed the connection.')) })
      let finished = false
      proxied.on('error', (error) => {
        const first = !finished
        finished = true
        if (first) {
          if (!res.headersSent) fail(res, 502, `The donor engine refused the request: ${error.message}`, 'donation_upstream')
          else if (!res.writableEnded) res.end()
          emit({ at: new Date().toISOString(), card: card.id, route: pathname, status: 502, ms: Date.now() - started, error: error.message })
        }
        release()
      })
      proxied.end(payload)
    }
  }

  const server = config.tls ? createHttpsServer({ cert: readFileSync(config.tls.certFile), key: readFileSync(config.tls.keyFile) }, handler) : createHttpServer(handler)
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(config.listen.port, config.listen.host, () => { server.off('error', reject); resolve() })
  })
  const close = () => new Promise<void>((resolve) => server.close(() => resolve()))
  return { card, server, close }
}

// The CLI entry: `node donation-node.js --config donation.json` (tests and remote donors).
if (typeof require !== 'undefined' && require.main === module) {
  startDonationNode(readConfig()).then((handle) => {
    const address = handle.server as unknown as { address(): { address: string; port: number } | string | null }
    const bound = address.address()
    process.stdout.write(`donation node: card ${handle.card.id} (${handle.card.model}) listening on ${typeof bound === 'object' && bound ? `${bound.address}:${bound.port}` : 'the configured address'} until ${new Date(handle.card.window.until).toISOString()}\n`)
    for (const signal of ['SIGTERM', 'SIGINT'] as const) process.on(signal, () => { void handle.close().then(() => process.exit(0)) })
  }).catch((error: Error) => { process.stderr.write(`donation node: ${error.message}\n`); process.exit(1) })
}
