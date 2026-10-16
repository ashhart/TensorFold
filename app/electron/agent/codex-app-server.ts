// JSON-RPC-over-stdio client for a PERSISTENT `codex app-server` session.
//
// This replaces the one-shot `codex exec` transport (`codex-cli.ts`) with a
// long-lived process that keeps a thread alive across turns: initialize once,
// start (or resume) a thread, then drive many `turn/start` calls — each yielding
// TensorFold StreamChunk-compatible items translated from the codex notification
// stream until `turn/completed`.
//
// The stdio transport is INJECTED (a duplex {write,onLine,kill} or a spawn
// factory that produces one), mirroring how `ssh-connection.ts` takes a
// `clientFactory` and `workspace-git.ts` takes a `GitRunner`. Tests script
// JSON-RPC responses + notifications through a fake transport with NO real
// codex binary, subprocess, or network.

import { spawn } from 'node:child_process'
import readline from 'node:readline'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { cliSpawnEnv } from './cli-spawn-env'
import { binary, codexExecApprovalPolicy, codexModelAndEffort, flattenMessages, sandbox } from './codex-cli'
import { planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { EngineSteeringChannel } from './steering'
import type { ApiMessage, ApprovalMode, Provider } from './types'

// StreamChunk shape mirrors openai.ts / codex-cli.ts. We only emit the subset
// codex produces (reasoning + content); the integrator appends `finish`.
export type CodexStreamChunk =
  | { type: 'reasoning'; text: string }
  | { type: 'content'; text: string }
  // Real account rate-limit snapshot (VST-037), from the `account/rateLimits/updated`
  // notification (installed codex 0.142.5 protocol). Carried through the turn so the host
  // can persist a provider-status snapshot for the picker + doctor.
  | { type: 'rate_limits'; snapshot: CodexRateLimitSnapshotWire }

// The RateLimitSnapshot shape codex emits (subset we use): primary=session, secondary=weekly.
export interface CodexRateLimitWindowWire {
  usedPercent: number
  resetsAt?: number | null
  windowDurationMins?: number | null
}
export interface CodexRateLimitSnapshotWire {
  primary?: CodexRateLimitWindowWire | null
  secondary?: CodexRateLimitWindowWire | null
  planType?: string | null
  limitName?: string | null
}

// A line-oriented duplex over the codex stdio. `write` sends one framed JSON-RPC
// message (no trailing newline expected — the transport adds framing); `onLine`
// delivers each inbound newline-delimited JSON line; `kill` tears the process down.
// `onClose` (optional for older fakes) reports the underlying process dying so an
// active turn fails fast instead of waiting forever on a notification that will
// never arrive.
export interface CodexStdioTransport {
  write(line: string): void
  onLine(cb: (line: string) => void): void
  kill(): void
  onClose?(cb: (reason: string) => void): void
}

// Either inject a ready transport, or a factory that spawns one. The factory form
// lets production code lazily spawn `codex app-server` while tests pass a fake.
export interface CodexAppServerDeps {
  transport?: CodexStdioTransport
  spawnTransport?: () => CodexStdioTransport
  // Optional sinks for diagnostics; kept injectable so tests stay silent.
  onError?: (message: string) => void
  onNotification?: (method: string, params: any) => void
}

export interface CodexInitializeParams {
  clientInfo?: { name: string; version: string }
  [key: string]: unknown
}

export interface CodexThreadConfig {
  cwd?: string
  model?: string
  approvalPolicy?: string
  sandbox?: string
  [key: string]: unknown
}

export interface CodexRunTurnOptions {
  model?: string
  effort?: string
  // Extra params merged into turn/start (sandboxPolicy, attachments, etc.).
  extra?: Record<string, unknown>
  // Live mid-turn steering (VST-033): while this turn is active, queued user cues are
  // forwarded via the protocol's `turn/steer` request.
  steering?: EngineSteeringChannel
  /** Maximum silence between protocol events before the poisoned session is killed. */
  idleTimeoutMs?: number
}

export const CODEX_APP_SERVER_IDLE_TIMEOUT_MS = 120_000
export const CODEX_APP_SERVER_INTERRUPT_GRACE_MS = 1_500

export function withProtocolIdleTimeout<T>(promise: Promise<T>, timeoutMs: number, label = 'codex app-server turn'): Promise<T> {
  const bounded = Math.max(1, timeoutMs)
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} was idle for ${bounded}ms.`)), bounded)
    promise.then(
      (value) => { clearTimeout(timer); resolve(value) },
      (error) => { clearTimeout(timer); reject(error) },
    )
  })
}

// A persisted resume cursor. Stored on a Thread so a later session can call
// `resumeThread(cursor.threadId)` instead of starting fresh.
export interface CodexResumeCursor {
  threadId: string
}

// Snippets that mark a thread/resume failure as "thread is gone, start fresh"
// rather than a hard protocol error.
const RECOVERABLE_RESUME_SNIPPETS = [
  'not found',
  'missing thread',
  'no such thread',
  'unknown thread',
  'does not exist',
]

export function isRecoverableThreadResumeError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase()
  if (!message.includes('thread')) return false
  return RECOVERABLE_RESUME_SNIPPETS.some((snippet) => message.includes(snippet))
}

interface PendingRequest {
  resolve: (result: any) => void
  reject: (error: Error) => void
}

type NotificationHandler = (params: any) => void

// Newline-delimited JSON-RPC 2.0 client over an injected duplex. Correlates
// responses by id and routes notifications (no id) to per-method handlers.
export class JsonRpcStdio {
  private nextId = 1
  private readonly pending = new Map<number, PendingRequest>()
  private readonly handlers = new Map<string, Set<NotificationHandler>>()
  private closed = false

  constructor(
    private readonly transport: CodexStdioTransport,
    private readonly onError?: (message: string) => void,
  ) {
    this.transport.onLine((line) => this.handleLine(line))
    // Process death must wake everything: reject in-flight requests AND dispatch a
    // synthetic terminal `error` notification so a suspended runTurn finishes with a
    // failure instead of hanging on a wake signal that would never resolve.
    this.transport.onClose?.((reason) => this.handleTransportClose(reason))
  }

  get isClosed(): boolean {
    return this.closed
  }

  private handleTransportClose(reason: string): void {
    if (this.closed) return
    this.closed = true
    const message = reason || 'codex app-server process exited'
    this.onError?.(message)
    this.rejectAll(new Error(message))
    const set = this.handlers.get('error')
    if (set) for (const handler of set) handler({ error: { message }, willRetry: false })
  }

  private handleLine(raw: string): void {
    const text = String(raw || '').trim()
    if (!text) return
    let message: any
    try {
      message = JSON.parse(text)
    } catch {
      this.onError?.(`codex app-server: non-JSON line: ${text.slice(0, 200)}`)
      return
    }
    // A response correlates by id; a notification has a method but no id.
    if (message.id != null && (message.result !== undefined || message.error !== undefined)) {
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) {
        const err = new Error(message.error?.message || 'codex app-server request failed')
        ;(err as any).code = message.error?.code
        ;(err as any).data = message.error?.data
        pending.reject(err)
      } else {
        pending.resolve(message.result)
      }
      return
    }
    if (typeof message.method === 'string') {
      const set = this.handlers.get(message.method)
      if (set) for (const handler of set) handler(message.params)
    }
  }

  // Send a request and resolve with its `result` (or reject on its `error`).
  request<T = any>(method: string, params?: unknown): Promise<T> {
    if (this.closed) return Promise.reject(new Error('codex app-server transport is closed'))
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      try {
        this.transport.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
      } catch (e) {
        this.pending.delete(id)
        reject(e instanceof Error ? e : new Error(String(e)))
      }
    })
  }

  // Fire-and-forget notification (no id, no response expected).
  notify(method: string, params?: unknown): void {
    if (this.closed) return
    this.transport.write(JSON.stringify({ jsonrpc: '2.0', method, params }))
  }

  // Register a handler for an inbound notification method. Returns an unsubscribe.
  on(method: string, handler: NotificationHandler): () => void {
    let set = this.handlers.get(method)
    if (!set) {
      set = new Set()
      this.handlers.set(method, set)
    }
    set.add(handler)
    return () => set?.delete(handler)
  }

  // Reject every in-flight request — used when the transport dies mid-turn.
  rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }

  close(): void {
    if (this.closed) return
    // Route through the same terminal path as process death so an ACTIVE turn is woken
    // with a failure (synthetic `error` notification) instead of hanging on its wake
    // signal — e.g. app-quit closing the pool while a turn is still streaming.
    this.handleTransportClose('codex app-server transport closed')
    this.transport.kill()
  }
}

// Translate one codex notification into a StreamChunk, or null if it carries no
// renderable text. Covers agent-message deltas (content) and reasoning deltas.
export function chunkFromNotification(method: string, params: any): CodexStreamChunk | null {
  switch (method) {
    case 'item/agentMessage/delta': {
      const text = typeof params?.delta === 'string' ? params.delta : ''
      return text ? { type: 'content', text } : null
    }
    case 'item/reasoning/textDelta':
    case 'item/reasoning/summaryTextDelta': {
      const text = typeof params?.delta === 'string' ? params.delta : ''
      return text ? { type: 'reasoning', text } : null
    }
    default:
      return null
  }
}

// Deferred used to bridge the push-based notification stream into the
// async-generator pull model of `runTurn`.
class Signal<T> {
  promise: Promise<T>
  resolve!: (value: T) => void
  reject!: (error: Error) => void
  constructor() {
    this.promise = new Promise<T>((res, rej) => {
      this.resolve = res
      this.reject = rej
    })
  }
}

export class CodexAppServerSession {
  private readonly rpc: JsonRpcStdio
  private threadId: string | null = null
  private activeTurnId: string | null = null
  private readonly onNotification?: (method: string, params: any) => void

  constructor(private readonly deps: CodexAppServerDeps) {
    const transport = deps.transport ?? deps.spawnTransport?.()
    if (!transport) throw new Error('CodexAppServerSession requires a transport or spawnTransport')
    this.rpc = new JsonRpcStdio(transport, deps.onError)
    this.onNotification = deps.onNotification
    if (this.onNotification) {
      // Mirror every notification to the diagnostics sink without consuming it.
      for (const method of [
        'turn/started',
        'turn/completed',
        'item/agentMessage/delta',
        'item/reasoning/textDelta',
        'error',
      ]) {
        this.rpc.on(method, (params) => this.onNotification?.(method, params))
      }
    }
  }

  get currentThreadId(): string | null {
    return this.threadId
  }

  // True once the underlying transport is gone (process death or explicit close).
  // The pool uses this to rebuild instead of reusing a poisoned session.
  get isClosed(): boolean {
    return this.rpc.isClosed
  }

  get hasActiveTurn(): boolean {
    return this.activeTurnId != null
  }

  // Resume cursor to persist on the Thread once a thread is open.
  get resumeCursor(): CodexResumeCursor | null {
    return this.threadId ? { threadId: this.threadId } : null
  }

  // Handshake: initialize then send the `initialized` notification.
  async initialize(params: CodexInitializeParams = {}): Promise<any> {
    const result = await this.rpc.request('initialize', {
      clientInfo: { name: 'vontra', version: '1' },
      ...params,
    })
    this.rpc.notify('initialized', undefined)
    return result
  }

  // Open a brand-new thread. Returns the codex-assigned thread id.
  async startThread(config: CodexThreadConfig = {}): Promise<string> {
    const result = await this.rpc.request('thread/start', config)
    this.threadId = result?.thread?.id ?? result?.threadId ?? null
    if (!this.threadId) throw new Error('codex app-server thread/start returned no thread id')
    return this.threadId
  }

  // Resume an existing thread by id. On a recoverable "thread not found" error,
  // transparently fall back to a fresh `thread/start`.
  async resumeThread(threadId: string, config: CodexThreadConfig = {}): Promise<string> {
    try {
      const result = await this.rpc.request('thread/resume', { threadId, ...config })
      this.threadId = result?.thread?.id ?? result?.threadId ?? threadId
      return this.threadId!
    } catch (error) {
      if (isRecoverableThreadResumeError(error)) {
        this.deps.onError?.(`codex app-server resume fell back to fresh start: ${String(error)}`)
        return this.startThread(config)
      }
      throw error
    }
  }

  // Drive one turn: send `turn/start`, then yield StreamChunks translated from the
  // notification stream until `turn/completed` (resolve) or `error` (throw).
  async *runTurn(prompt: string, opts: CodexRunTurnOptions = {}): AsyncGenerator<CodexStreamChunk> {
    if (!this.threadId) throw new Error('codex app-server: start or resume a thread before runTurn')

    const queue: CodexStreamChunk[] = []
    let wake = new Signal<void>()
    let done = false
    let failure: Error | null = null
    let disconnectSteering: (() => void) | undefined
    const idleTimeoutMs = Math.max(1, opts.idleTimeoutMs ?? CODEX_APP_SERVER_IDLE_TIMEOUT_MS)

    const push = (chunk: CodexStreamChunk | null) => {
      if (!chunk) return
      queue.push(chunk)
      wake.resolve()
    }
    const finish = (err?: Error) => {
      if (done) return
      done = true
      if (err) failure = err
      wake.resolve()
    }

    const offDelta = this.rpc.on('item/agentMessage/delta', (params) => {
      if (this.matchesTurn(params)) push(chunkFromNotification('item/agentMessage/delta', params))
    })
    const offReasoning = this.rpc.on('item/reasoning/textDelta', (params) => {
      if (this.matchesTurn(params)) push(chunkFromNotification('item/reasoning/textDelta', params))
    })
    const offReasoningSummary = this.rpc.on('item/reasoning/summaryTextDelta', (params) => {
      if (this.matchesTurn(params)) push(chunkFromNotification('item/reasoning/summaryTextDelta', params))
    })
    const offStarted = this.rpc.on('turn/started', (params) => {
      const id = params?.turn?.id
      if (id && !this.activeTurnId) this.activeTurnId = id
    })
    const offCompleted = this.rpc.on('turn/completed', (params) => {
      if (this.activeTurnId && params?.turn?.id && params.turn.id !== this.activeTurnId) return
      const status = params?.turn?.status
      if (status === 'failed') {
        const message = params?.turn?.error?.message || 'codex turn failed'
        finish(new Error(message))
      } else {
        finish()
      }
    })
    const offError = this.rpc.on('error', (params) => {
      if (params?.willRetry) return
      finish(new Error(params?.error?.message || 'codex app-server error'))
    })
    // Account rate limits (VST-037): a sparse rolling update arriving during the turn. Not
    // turn-scoped in the protocol, but it piggy-backs on turn activity — capture it and
    // surface as a chunk so the host persists a provider-status snapshot.
    const offRateLimits = this.rpc.on('account/rateLimits/updated', (params) => {
      const snapshot = params?.rateLimits
      if (snapshot && typeof snapshot === 'object') push({ type: 'rate_limits', snapshot })
    })

    const cleanup = () => {
      offDelta()
      offReasoning()
      offReasoningSummary()
      offStarted()
      offCompleted()
      offError()
      offRateLimits()
    }

    try {
      const params: Record<string, unknown> = {
        threadId: this.threadId,
        input: [{ type: 'text', text: prompt }],
        ...(opts.model ? { model: opts.model } : {}),
        ...(opts.effort ? { effort: opts.effort } : {}),
        ...(opts.extra ?? {}),
      }
      let response: any
      try {
        response = await withProtocolIdleTimeout(this.rpc.request('turn/start', params), idleTimeoutMs, 'codex app-server turn/start')
      } catch (error) {
        this.close()
        throw error
      }
      const turnId = response?.turn?.id ?? response?.turnId
      if (turnId) this.activeTurnId = turnId

      // Live steering joins only once the turn exists: `turn/steer` requires the active
      // turn id as a precondition, so readiness is exactly "this turn is still running".
      disconnectSteering = opts.steering?.connect(
        (input) => this.steer(input.text),
        () => this.activeTurnId != null && !this.rpc.isClosed,
      )

      while (true) {
        while (queue.length) {
          const chunk = queue.shift()!
          yield chunk
        }
        if (done) break
        try {
          await withProtocolIdleTimeout(wake.promise, idleTimeoutMs)
        } catch (error) {
          // A live-but-silent process is just as poisoned as a dead one. Kill it so
          // the pool cannot reuse a wedged session and all pending RPCs are rejected.
          this.close()
          throw error
        }
        wake = new Signal<void>()
      }
      // Drain any chunks queued alongside the terminal notification.
      while (queue.length) yield queue.shift()!
      if (failure) throw failure
    } finally {
      disconnectSteering?.()
      cleanup()
      this.activeTurnId = null
    }
  }

  // Forward a queued user cue into the ACTIVE turn via `turn/steer` (verified against the
  // installed codex 0.142.5 protocol schema). The expectedTurnId precondition means a steer
  // racing the turn's completion is refused server-side — surfaced here as `false`, so the
  // caller can mark the cue "not applied" instead of pretending it landed.
  async steer(text: string): Promise<boolean> {
    const turnId = this.activeTurnId
    if (!this.threadId || !turnId || this.rpc.isClosed) return false
    try {
      await this.rpc.request('turn/steer', {
        threadId: this.threadId,
        expectedTurnId: turnId,
        input: [{ type: 'text', text }],
      })
      return true
    } catch {
      return false
    }
  }

  // Interrupt the active (or a specific) turn.
  async interrupt(turnId?: string): Promise<void> {
    if (!this.threadId) return
    const effectiveTurnId = turnId ?? this.activeTurnId
    if (!effectiveTurnId) return
    await this.rpc.request('turn/interrupt', {
      threadId: this.threadId,
      turnId: effectiveTurnId,
    })
  }

  close(): void {
    this.rpc.close()
  }

  // A notification belongs to this turn if it carries no turn id, or a matching one.
  private matchesTurn(params: any): boolean {
    const tid = params?.turnId
    if (!tid || !this.activeTurnId) return true
    return tid === this.activeTurnId
  }
}

// Convenience factory mirroring the `make*`/`create*` style elsewhere.
export function createCodexAppServerSession(deps: CodexAppServerDeps): CodexAppServerSession {
  return new CodexAppServerSession(deps)
}

// --- production transport (lazy spawn keeps the binary out of pure tests) ---

// Wrap a spawned `codex app-server` child process as a CodexStdioTransport.
// Not exercised by tests; tests pass a fake transport instead.
export function spawnCodexAppServerTransport(input: {
  binary: string
  args?: string[]
  // Full argv, replacing `app-server ...args` (a planning-mode seatbelt wrapper).
  argv?: string[]
  cwd?: string
  env?: NodeJS.ProcessEnv
}): CodexStdioTransport {
  const child = spawn(input.binary, input.argv ?? ['app-server', ...(input.args ?? [])], {
    cwd: input.cwd,
    detached: detachedProcessGroup(),
    env: input.env ?? process.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  const lines = readline.createInterface({ input: child.stdout! })
  let lineCb: ((line: string) => void) | null = null
  lines.on('line', (line) => lineCb?.(line))
  // Keep the tail of stderr so a crash reason ("codex: unknown command app-server",
  // a panic, an auth failure) survives into the close reason instead of vanishing.
  let stderrTail = ''
  child.stderr?.on('data', (data: Buffer) => {
    stderrTail = (stderrTail + data.toString()).slice(-400)
  })
  let closeCb: ((reason: string) => void) | null = null
  let closeReported = false
  const reportClose = (reason: string) => {
    if (closeReported) return
    closeReported = true
    closeCb?.(reason)
  }
  child.on('error', (error) => reportClose(`codex app-server failed to start: ${error.message}`))
  child.on('exit', (code, signalName) => {
    const suffix = stderrTail.trim() ? ` — ${stderrTail.trim().split('\n').slice(-2).join(' / ')}` : ''
    reportClose(`codex app-server exited (${signalName ?? code ?? 'unknown'})${suffix}`)
  })
  return {
    write(line: string) {
      child.stdin!.write(line + '\n')
    },
    onLine(cb) {
      lineCb = cb
    },
    kill() {
      terminateChildProcess(child, { processGroup: true })
    },
    onClose(cb) {
      closeCb = cb
    },
  }
}

// ---- session pool: keep a codex app-server thread alive across turns -------------------
//
// A one-shot `codex exec` (codex-cli.ts) re-flattens the WHOLE conversation into a fresh
// process every turn. This pool keeps the underlying `codex app-server` process (and its
// codex-side thread) ALIVE across turns, keyed by a stable session key (the TensorFold Thread
// id) — so a turn only needs to send what's NEW since the thread was opened, not the whole
// history again. Idle-evicted so a long-abandoned thread doesn't hold a process open forever.

const POOL_IDLE_MS = 30 * 60_000

interface PooledSession {
  session: CodexAppServerSession
  idleTimer: ReturnType<typeof setTimeout>
  // The sandbox/approval/cwd the pooled codex thread was opened with. A later turn that
  // changes any of these (e.g. the user switches a conversation to read-only) must NOT be
  // served by a thread still holding the old, looser policy — so we rebuild on a mismatch.
  configKey: string
}

// A stable identity for the safety-relevant thread config. Two turns with the same
// sandbox + approvalPolicy + cwd may share a pooled thread; a change in any of them
// means the live thread's policy no longer matches what the caller asked for, so the
// pooled thread must be torn down and rebuilt rather than silently reused.
function threadConfigKey(config: CodexThreadConfig): string {
  return JSON.stringify({
    cwd: config.cwd ?? null,
    approvalPolicy: config.approvalPolicy ?? null,
    sandbox: config.sandbox ?? null,
  })
}

const pool = new Map<string, PooledSession>()

function touchIdleTimer(key: string, entry: PooledSession): void {
  clearTimeout(entry.idleTimer)
  entry.idleTimer = setTimeout(() => closeCodexAppServerSession(key), POOL_IDLE_MS)
}

// Evict + close one pooled session (e.g. the thread was deleted, or it idled out).
export function closeCodexAppServerSession(key: string): void {
  const entry = pool.get(key)
  if (!entry) return
  clearTimeout(entry.idleTimer)
  pool.delete(key)
  try { entry.session.close() } catch { /* best-effort */ }
}

// Close every pooled session — called on app quit, mirroring killAllTerminals().
export function closeAllCodexAppServerSessions(): void {
  for (const key of [...pool.keys()]) closeCodexAppServerSession(key)
}

export function pooledCodexAppServerSessionKeys(): string[] {
  return [...pool.keys()]
}

// Reuse a live pooled session for `key`, or spawn+initialize a fresh one and either resume
// the persisted cursor (if given) or start a brand-new thread. `needsFullHistory` tells the
// caller whether THIS call's turn needs the full flattened conversation as its prompt: true
// only the very first time a key starts a genuinely fresh thread (no resume cursor, no pool
// entry) — codex has no memory of anything yet. A resumed thread (codex already has its
// history server-side, possibly from a PRIOR app run) or a reused pooled session (already
// has everything from earlier turns THIS run) both get false: sending the full history again
// would duplicate context codex already remembers.
export async function getOrCreateCodexAppServerSession(
  key: string,
  spawnTransport: () => CodexStdioTransport,
  threadConfig: CodexThreadConfig,
  resumeCursor: CodexResumeCursor | null,
  onError?: (message: string) => void,
  signal?: AbortSignal,
): Promise<{ session: CodexAppServerSession; needsFullHistory: boolean }> {
  const configKey = threadConfigKey(threadConfig)
  const existing = pool.get(key)
  if (existing) {
    // A pooled session whose process died (transport closed) is poisoned — every request on
    // it rejects immediately. Evict + rebuild instead of reusing it.
    if (existing.session.isClosed) {
      closeCodexAppServerSession(key)
    } else if (existing.configKey === configKey) {
      // Reuse only when the sandbox/approval/cwd are unchanged. If they changed (e.g. the user
      // switched the conversation to read-only), the live thread still holds the OLD policy — so
      // evict + rebuild a fresh thread under the new config instead of silently reusing it.
      touchIdleTimer(key, existing)
      return { session: existing.session, needsFullHistory: false }
    } else {
      closeCodexAppServerSession(key)
    }
  }
  const session = createCodexAppServerSession({ spawnTransport, onError })
  // A failure during initialize/start/resume leaves the freshly-spawned child neither pooled
  // nor closed — close it before rethrowing so the subprocess isn't orphaned.
  let needsFullHistory: boolean
  const abortHandshake = () => session.close()
  signal?.addEventListener('abort', abortHandshake, { once: true })
  try {
    if (signal?.aborted) abortHandshake()
    await withProtocolIdleTimeout(session.initialize(), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server initialize')
    needsFullHistory = !resumeCursor
    if (resumeCursor) {
      await withProtocolIdleTimeout(session.resumeThread(resumeCursor.threadId, threadConfig), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server thread/resume')
    } else {
      await withProtocolIdleTimeout(session.startThread(threadConfig), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server thread/start')
    }
  } catch (error) {
    try { session.close() } catch { /* best-effort */ }
    throw error
  } finally {
    signal?.removeEventListener('abort', abortHandshake)
  }
  const entry: PooledSession = { session, idleTimer: setTimeout(() => {}, 0), configKey }
  pool.set(key, entry)
  touchIdleTimer(key, entry)
  return { session, needsFullHistory }
}

// The newest user-role message's text content. Used for a turn on an already-resumed or
// pooled codex thread, which already remembers everything before it server-side — sending
// the full flattened history again would duplicate context the thread already has.
export function latestUserMessageText(messages: ApiMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return contentText(messages[i].content).trim()
  }
  return ''
}

// ---- streaming entry point: resume-aware codex app-server transport --------------------

export interface CodexAppServerStreamParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  reasoningEffort?: string
  approvalMode?: ApprovalMode
  cwd?: string | null
  // Stable per-conversation key for the session pool — the TensorFold Thread id.
  sessionKey: string
  // The Thread's persisted resume cursor, if any (from a prior app run).
  resumeCursor?: CodexResumeCursor | null
  signal?: AbortSignal
  /** Cooperative interrupt grace before the pooled process is forcibly reaped. */
  interruptGraceMs?: number
  // Live mid-turn steering channel (VST-033); connected for the duration of the turn.
  steering?: EngineSteeringChannel
  // Injectable for tests; production omits this and a real `codex app-server` process spawns.
  spawnTransport?: () => CodexStdioTransport
  onError?: (message: string) => void
}

// A terminal chunk carrying the resume cursor once the turn completes, so the caller can
// persist it onto its own Thread record for next time.
export type CodexAppServerChunk = CodexStreamChunk | { type: 'resume_cursor'; cursor: CodexResumeCursor }

export async function* streamCodexAppServer(params: CodexAppServerStreamParams): AsyncGenerator<CodexAppServerChunk> {
  if (params.signal?.aborted) return
  const { model, effort } = codexModelAndEffort(params.model, params.reasoningEffort)
  const threadConfig: CodexThreadConfig = {
    cwd: params.cwd || undefined,
    model,
    approvalPolicy: codexExecApprovalPolicy(params.provider, params.approvalMode),
    sandbox: sandbox(params.provider, params.approvalMode),
  }
  const spawnTransport = params.spawnTransport ?? (() => {
    const run = planningCommand(params.provider, binary(params.provider), ['app-server'], {
      cwd: params.cwd || undefined,
      env: cliSpawnEnv(params.provider.cliEnv),
    })
    return spawnCodexAppServerTransport({ binary: run.command, argv: run.args, cwd: run.cwd, env: run.env })
  })
  let acquired: { session: CodexAppServerSession; needsFullHistory: boolean }
  try {
    acquired = await getOrCreateCodexAppServerSession(
      params.sessionKey,
      spawnTransport,
      threadConfig,
      params.resumeCursor ?? null,
      params.onError,
      params.signal,
    )
  } catch (error) {
    if (params.signal?.aborted) return
    throw error
  }
  const { session, needsFullHistory } = acquired
  // The signal can race an immediately-resolved pooled acquisition. Never start
  // a fresh turn after cancellation, even if no handshake needed aborting.
  if (params.signal?.aborted) return
  const prompt = needsFullHistory ? flattenMessages(params.messages) : latestUserMessageText(params.messages)
  if (!prompt) return

  // Send turn/interrupt to codex as soon as the run is aborted. The cancel path returns the
  // generator while it's suspended at `yield chunk`, so a top-of-loop abort check never fires
  // again — fire the interrupt from an abort listener AND from the finally so the codex turn is
  // actually stopped server-side instead of running on (possibly mutating files) after cancel.
  let interrupted = false
  let forcedCloseTimer: ReturnType<typeof setTimeout> | null = null
  const interrupt = () => {
    if (interrupted) return
    interrupted = true
    void session.interrupt().catch(() => {})
    // A wedged app-server can ignore turn/interrupt (or never answer the RPC).
    // Give cooperative cancellation a short grace period, then evict and reap
    // the process group so cancellation cannot leave mutating descendants alive.
    forcedCloseTimer = setTimeout(() => {
      forcedCloseTimer = null
      if (session.hasActiveTurn) closeCodexAppServerSession(params.sessionKey)
    }, Math.max(1, params.interruptGraceMs ?? CODEX_APP_SERVER_INTERRUPT_GRACE_MS))
    forcedCloseTimer.unref?.()
  }
  const signal = params.signal
  if (signal?.aborted) interrupt()
  const onAbort = () => interrupt()
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    for await (const chunk of session.runTurn(prompt, { model, effort, steering: params.steering })) {
      if (signal?.aborted) {
        interrupt()
        break
      }
      yield chunk
    }
  } catch (error) {
    // Forced transport closure rejects the suspended protocol request. Once the
    // caller has cancelled, that is successful cancellation rather than a new
    // provider error; preserve genuine failures while the signal is live.
    if (!signal?.aborted) throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
    if (signal?.aborted) interrupt()
    if (forcedCloseTimer && !session.hasActiveTurn) {
      clearTimeout(forcedCloseTimer)
      forcedCloseTimer = null
    }
  }
  if (signal?.aborted) return
  if (session.resumeCursor) yield { type: 'resume_cursor', cursor: session.resumeCursor }
}
