// OpenCode engine (VST-033): drive an `opencode serve` HTTP+SSE server as a first-class
// thread engine, mirroring codex-app-server.ts. Schemas were taken from the installed
// opencode 1.17's OpenAPI spec (`GET /doc`) and a live SSE probe — NOT from memory.
//
// Wire facts this module encodes:
// - Create a session:   POST /session            { title }            -> { id: "ses_…" }
// - Prompt (async):     POST /session/:id/prompt_async  { parts, model, agent?, system? } -> 204
// - Abort:              POST /session/:id/abort
// - Events (SSE):       GET  /event   — `data: {json}\n\n`; first frame `server.connected`.
//   Streaming turn events (the `session.next.*` family):
//     session.next.text.delta       { sessionID, assistantMessageID, textID, delta }
//     session.next.reasoning.delta  { sessionID, assistantMessageID, reasoningID, delta }
//     session.next.step.failed      { sessionID, error }
//     session.idle                  { sessionID }              -> turn complete
//     session.error                 { sessionID, error }       -> turn failed
// - Model ids are `providerID/modelID`; the prompt takes model as { providerID, modelID }.

import { createLocalServiceFetch } from './secure-fetch'
import { readBoundedJsonResponse } from './bounded-response'
import { discardResponseBody } from './pinned-fetch'

export const OPENCODE_SSE_FRAME_MAX_CHARS = 512 * 1024
export const OPENCODE_REQUEST_TIMEOUT_MS = 30_000
export const OPENCODE_ABORT_TIMEOUT_MS = 5_000
export const OPENCODE_SSE_IDLE_TIMEOUT_MS = 60_000

// Transports that speak the OpenCode HTTP+SSE server protocol. Kilo CLI is an OpenCode fork
// ("kilo serve") sharing the identical engine — same /session, /prompt_async, /event, and the
// same event families — so both route through this one engine. One helper keeps that
// relationship explicit instead of scattering `=== 'opencode' || === 'kilo'` across the code.
export function isOpenCodeProtocol(transport: string | null | undefined): boolean {
  return transport === 'opencode' || transport === 'kilo'
}

export type OpenCodeChunk =
  | { type: 'reasoning'; text: string }
  | { type: 'content'; text: string }

// A parsed SSE event: `{ type, properties }`. Kept loose — we only read the fields the
// translator needs, and unknown event types pass through as null chunks.
export interface OpenCodeEvent {
  type: string
  properties?: Record<string, any>
}

export interface OpenCodeModelField {
  providerID: string
  modelID: string
}

// Split `anthropic/claude-3-5-sonnet` → { providerID:'anthropic', modelID:'claude-3-5-sonnet' }.
// Only the FIRST slash splits (model ids can contain slashes); an id with no slash is sent
// as its own modelID with an empty providerID (the server resolves the default provider).
export function openCodeModelField(model: string): OpenCodeModelField {
  const trimmed = (model || '').trim()
  const slash = trimmed.indexOf('/')
  if (slash <= 0) return { providerID: '', modelID: trimmed }
  return { providerID: trimmed.slice(0, slash), modelID: trimmed.slice(slash + 1) }
}

export interface OpenCodePromptBody {
  parts: Array<{ type: 'text'; text: string }>
  model?: OpenCodeModelField
  agent?: string
  system?: string
}

export function openCodePromptBody(prompt: string, opts: { model?: string; agent?: string; system?: string } = {}): OpenCodePromptBody {
  const body: OpenCodePromptBody = { parts: [{ type: 'text', text: prompt }] }
  if (opts.model && opts.model.trim() && opts.model !== 'opencode-default') body.model = openCodeModelField(opts.model)
  if (opts.agent?.trim()) body.agent = opts.agent.trim()
  if (opts.system?.trim()) body.system = opts.system.trim()
  return body
}

// Parse one SSE `data:` line into an event, or null for comments/keep-alives/blank lines.
export function parseOpenCodeSseLine(line: string): OpenCodeEvent | null {
  const text = String(line || '')
  if (!text.startsWith('data:')) return null
  const json = text.slice(5).trim()
  if (!json) return null
  try {
    const parsed = JSON.parse(json)
    if (parsed && typeof parsed.type === 'string') return { type: parsed.type, properties: parsed.properties ?? {} }
    return null
  } catch {
    return null
  }
}

// Translate a streaming event into a chunk for THIS session, or null. `sessionID` scopes it:
// the global `/event` stream carries every session's events, so a driver must ignore deltas
// belonging to other sessions.
export function chunkFromOpenCodeEvent(event: OpenCodeEvent, sessionID: string): OpenCodeChunk | null {
  if (event.properties?.sessionID && event.properties.sessionID !== sessionID) return null
  switch (event.type) {
    case 'session.next.text.delta': {
      const text = typeof event.properties?.delta === 'string' ? event.properties.delta : ''
      return text ? { type: 'content', text } : null
    }
    case 'session.next.reasoning.delta': {
      const text = typeof event.properties?.delta === 'string' ? event.properties.delta : ''
      return text ? { type: 'reasoning', text } : null
    }
    default:
      return null
  }
}

// Classify a terminal event for THIS session: 'idle' (turn done), 'error' (turn failed), or
// null (not terminal / other session). Error message is pulled from the event's error shape.
export function openCodeTerminalOutcome(event: OpenCodeEvent, sessionID: string): { kind: 'idle' } | { kind: 'error'; message: string } | null {
  if (event.properties?.sessionID && event.properties.sessionID !== sessionID) return null
  if (event.type === 'session.idle') return { kind: 'idle' }
  if (event.type === 'session.error' || event.type === 'session.next.step.failed') {
    const err = event.properties?.error
    const message = (err && (err.data?.message || err.message || err.name)) || 'opencode session error'
    return { kind: 'error', message: String(message) }
  }
  return null
}

// Transport the driver needs — injected so tests use fakes and never spawn a server.
export interface OpenCodeTransport {
  createSession(title: string, signal?: AbortSignal): Promise<string>
  postPromptAsync(sessionId: string, body: OpenCodePromptBody, signal?: AbortSignal): Promise<void>
  abort(sessionId: string, signal?: AbortSignal): Promise<void>
  // The SSE event stream (`GET /event`), already parsed into events. Must yield until the
  // returned generator is closed (the driver closes it when the turn ends).
  events(signal?: AbortSignal): AsyncGenerator<OpenCodeEvent>
}

export interface OpenCodeResumeCursor {
  sessionId: string
}

export interface OpenCodeRunParams {
  transport: OpenCodeTransport
  prompt: string
  model?: string
  agent?: string
  system?: string
  // Reuse an existing session (resume) instead of creating one.
  resumeCursor?: OpenCodeResumeCursor | null
  signal?: AbortSignal
  onError?: (message: string) => void
}

// Terminal chunk carrying the session cursor once the turn is set up, so the caller can
// persist it (mirrors codex-app-server's resume_cursor chunk).
export type OpenCodeStreamChunk = OpenCodeChunk | { type: 'resume_cursor'; cursor: OpenCodeResumeCursor }

// ---- production transport: real HTTP + SSE against `opencode serve` ----

function opencodeBase(serverUrl?: string): string {
  return (serverUrl?.trim() || 'http://127.0.0.1:4096').replace(/\/+$/, '')
}

function opencodeAuthHeader(password?: string, username?: string): Record<string, string> {
  if (!password?.trim()) return {}
  const token = Buffer.from(`${username?.trim() || 'opencode'}:${password}`).toString('base64')
  return { Authorization: `Basic ${token}` }
}

function openCodeAbortError(label: string): Error {
  const error = new Error(`${label} was aborted.`)
  error.name = 'AbortError'
  return error
}

/** Race any OpenCode protocol operation against the run cancellation and a hard deadline. */
export function withOpenCodeDeadline<T>(
  pending: Promise<T>,
  options: {
    label: string
    timeoutMs: number
    signal?: AbortSignal
    onStop?: () => void | Promise<void>
  },
): Promise<T> {
  const bounded = Math.max(1, options.timeoutMs)
  const stop = () => {
    try {
      const result = options.onStop?.()
      if (result && typeof (result as Promise<void>).catch === 'function') void (result as Promise<void>).catch(() => {})
    } catch { /* best-effort cancellation */ }
  }
  if (options.signal?.aborted) {
    void pending.catch(() => {})
    stop()
    return Promise.reject(openCodeAbortError(options.label))
  }
  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      options.signal?.removeEventListener('abort', onAbort)
      fn()
    }
    const onAbort = () => {
      finish(() => reject(openCodeAbortError(options.label)))
      stop()
    }
    const timer = setTimeout(() => {
      finish(() => reject(new Error(`${options.label} timed out after ${bounded}ms.`)))
      stop()
    }, bounded)
    options.signal?.addEventListener('abort', onAbort, { once: true })
    pending.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(error)),
    )
  })
}

async function fetchOpenCodeWithDeadline(
  fetchImpl: typeof fetch,
  input: Parameters<typeof fetch>[0],
  init: RequestInit,
  options: { label: string; timeoutMs: number; signal?: AbortSignal },
): Promise<Response> {
  const controller = new AbortController()
  return await withOpenCodeDeadline(
    Promise.resolve().then(() => fetchImpl(input, { ...init, signal: controller.signal })),
    {
      ...options,
      onStop: () => controller.abort(),
    },
  )
}

/** Bound a single SSE read; comments/heartbeats count as activity because they yield bytes. */
export function readOpenCodeSseChunk(
  reader: Pick<ReadableStreamDefaultReader<Uint8Array>, 'read'>,
  idleTimeoutMs = OPENCODE_SSE_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  return withOpenCodeDeadline(reader.read(), {
    label: 'OpenCode event stream',
    timeoutMs: idleTimeoutMs,
    signal,
  }).catch((error) => {
    if (error instanceof Error && /timed out after/.test(error.message)) {
      throw new Error(`OpenCode event stream was idle for ${Math.max(1, idleTimeoutMs)}ms.`)
    }
    throw error
  })
}

// Build a live transport. Every request resolves and pins the configured server immediately
// before connecting; loopback/LAN HTTP is allowed, public servers require HTTPS, and
// redirects cannot pivot Basic credentials to another origin.
export function createOpenCodeHttpTransport(input: {
  serverUrl?: string
  serverPassword?: string
  serverUsername?: string
  cwd?: string | null
  fetchImpl?: typeof fetch
  requestTimeoutMs?: number
  abortTimeoutMs?: number
  idleTimeoutMs?: number
}): OpenCodeTransport {
  const base = opencodeBase(input.serverUrl)
  const fetchImpl = createLocalServiceFetch(base, 'OpenCode server', input.fetchImpl ?? fetch)
  const headers = { 'Content-Type': 'application/json', ...opencodeAuthHeader(input.serverPassword, input.serverUsername) }
  const requestTimeoutMs = Math.max(1, input.requestTimeoutMs ?? OPENCODE_REQUEST_TIMEOUT_MS)
  const abortTimeoutMs = Math.max(1, input.abortTimeoutMs ?? OPENCODE_ABORT_TIMEOUT_MS)
  const idleTimeoutMs = Math.max(1, input.idleTimeoutMs ?? OPENCODE_SSE_IDLE_TIMEOUT_MS)
  return {
    async createSession(title: string, signal?: AbortSignal): Promise<string> {
      const label = 'OpenCode create session request'
      const res = await fetchOpenCodeWithDeadline(fetchImpl, `${base}/session`, {
        method: 'POST', headers, body: JSON.stringify({ title, ...(input.cwd ? { directory: input.cwd } : {}) }),
      }, { label, timeoutMs: requestTimeoutMs, signal })
      if (!res.ok) {
        void discardResponseBody(res)
        throw new Error(`opencode create session failed: HTTP ${res.status}`)
      }
      const json = await withOpenCodeDeadline(
        readBoundedJsonResponse(res, 'OpenCode create session') as Promise<{ id?: string }>,
        { label: 'OpenCode create session response', timeoutMs: requestTimeoutMs, signal, onStop: () => discardResponseBody(res) },
      )
      if (!json?.id) throw new Error('opencode create session returned no id')
      return json.id
    },
    async postPromptAsync(sessionId: string, body: OpenCodePromptBody, signal?: AbortSignal): Promise<void> {
      const res = await fetchOpenCodeWithDeadline(fetchImpl, `${base}/session/${encodeURIComponent(sessionId)}/prompt_async`, {
        method: 'POST', headers, body: JSON.stringify(body),
      }, { label: 'OpenCode prompt request', timeoutMs: requestTimeoutMs, signal })
      const status = res.status
      void discardResponseBody(res)
      if (!res.ok && status !== 204) throw new Error(`opencode prompt failed: HTTP ${status}`)
    },
    async abort(sessionId: string, signal?: AbortSignal): Promise<void> {
      const res = await fetchOpenCodeWithDeadline(fetchImpl, `${base}/session/${encodeURIComponent(sessionId)}/abort`, {
        method: 'POST', headers,
      }, { label: 'OpenCode abort request', timeoutMs: abortTimeoutMs, signal })
      const status = res.status
      void discardResponseBody(res)
      if (!res.ok && status !== 204) throw new Error(`opencode abort failed: HTTP ${status}`)
    },
    async *events(signal?: AbortSignal): AsyncGenerator<OpenCodeEvent> {
      const res = await fetchOpenCodeWithDeadline(fetchImpl, `${base}/event`, {
        headers: opencodeAuthHeader(input.serverPassword, input.serverUsername),
      }, { label: 'OpenCode event stream connection', timeoutMs: requestTimeoutMs, signal })
      if (!res.ok) {
        void discardResponseBody(res)
        throw new Error(`opencode event stream failed: HTTP ${res.status}`)
      }
      if (!String(res.headers.get('content-type') ?? '').toLowerCase().startsWith('text/event-stream')) {
        void discardResponseBody(res)
        throw new Error('opencode event stream returned an invalid content type')
      }
      if (!res.body) throw new Error('opencode event stream returned an empty body')
      const reader = (res.body as any).getReader ? (res.body as ReadableStream<Uint8Array>).getReader() : null
      if (!reader) return
      const decoder = new TextDecoder()
      let buffer = ''
      let complete = false
      try {
        while (true) {
          const { done, value } = await readOpenCodeSseChunk(reader, idleTimeoutMs, signal)
          if (done) { complete = true; break }
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
          if (buffer.length > OPENCODE_SSE_FRAME_MAX_CHARS && !buffer.includes('\n\n')) {
            throw new Error('opencode event stream frame exceeded the safety limit')
          }
          // SSE frames are separated by a blank line; a frame may hold multiple `data:` lines.
          let sep: number
          while ((sep = buffer.indexOf('\n\n')) >= 0) {
            const frame = buffer.slice(0, sep)
            buffer = buffer.slice(sep + 2)
            if (frame.length > OPENCODE_SSE_FRAME_MAX_CHARS) {
              throw new Error('opencode event stream frame exceeded the safety limit')
            }
            for (const line of frame.split('\n')) {
              const event = parseOpenCodeSseLine(line.trim())
              if (event) yield event
            }
          }
          if (buffer.length > OPENCODE_SSE_FRAME_MAX_CHARS) {
            throw new Error('opencode event stream frame exceeded the safety limit')
          }
        }
      } finally {
        if (!complete) try { await reader.cancel() } catch { /* best-effort */ }
        reader.releaseLock()
      }
    },
  }
}

// Drive one turn: subscribe FIRST (so no delta is missed between prompt and stream), ensure
// a session, fire the async prompt, then relay chunks until this session goes idle or errors.
export async function* runOpenCodeTurn(params: OpenCodeRunParams): AsyncGenerator<OpenCodeStreamChunk> {
  const { transport } = params
  // A distinct controller lets prompt/session failures close a primed subscription even
  // when the caller did not cancel the whole run. Calling and AWAITING first next() is
  // essential: async-generator bodies are lazy, and the production generator only opens,
  // MIME-checks, and starts reading GET /event when advanced.
  const eventController = new AbortController()
  const relayAbort = () => eventController.abort()
  if (params.signal?.aborted) eventController.abort()
  else params.signal?.addEventListener('abort', relayAbort, { once: true })
  const events = transport.events(eventController.signal)
  let sessionId = params.resumeCursor?.sessionId ?? ''
  let promptSubmitted = false
  let terminal = false
  try {
    const ready = await events.next()
    if (ready.done) throw new Error('OpenCode event stream ended before the subscription became ready.')
    if (!sessionId) sessionId = await transport.createSession('TensorFold', params.signal)
    yield { type: 'resume_cursor', cursor: { sessionId } }
    promptSubmitted = true
    await transport.postPromptAsync(sessionId, openCodePromptBody(params.prompt, { model: params.model, agent: params.agent, system: params.system }), params.signal)

    let result: IteratorResult<OpenCodeEvent> = ready
    while (!result.done) {
      if (params.signal?.aborted) break
      const event = result.value
      const chunk = chunkFromOpenCodeEvent(event, sessionId)
      if (chunk) {
        yield chunk
      } else {
        const outcome = openCodeTerminalOutcome(event, sessionId)
        if (outcome?.kind === 'idle') { terminal = true; break }
        if (outcome?.kind === 'error') { terminal = true; throw new Error(outcome.message) }
      }
      result = await events.next()
    }
  } catch (error) {
    // User cancellation is a normal terminal state. Timeouts and protocol failures remain
    // explicit errors so callers cannot mistake a half-open server for a completed turn.
    if (!(params.signal?.aborted && error instanceof Error && error.name === 'AbortError')) throw error
  } finally {
    params.signal?.removeEventListener('abort', relayAbort)
    eventController.abort()
    // Consumer return(), cancellation, transport failure, or a broken stream can all stop
    // iteration before a terminal event. Once prompt submission began, interrupt that turn
    // so it cannot keep running after TensorFold has stopped reading it.
    if (promptSubmitted && !terminal && sessionId) {
      try { await transport.abort(sessionId) } catch { /* best-effort */ }
    }
    // Close the SSE generator so the underlying connection is released.
    try { await events.return?.(undefined) } catch { /* best-effort */ }
  }
}
