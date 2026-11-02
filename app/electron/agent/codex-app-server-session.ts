import type { EngineSteeringChannel } from './steering'

export type CodexStreamChunk =
  | { type: 'reasoning'; text: string }
  | { type: 'content'; text: string }

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

export interface CodexStdioTransport {
  write(line: string): void
  onLine(cb: (line: string) => void): void
  kill(): void
  onClose?(cb: (reason: string) => void): void
}

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

  steering?: EngineSteeringChannel
  // Maximum silence between protocol events before the poisoned session is killed.
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

export interface CodexResumeCursor {
  threadId: string
}

export const RECOVERABLE_RESUME_SNIPPETS = [
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

export interface PendingRequest {
  resolve: (result: any) => void
  reject: (error: Error) => void
}

export type NotificationHandler = (params: any) => void

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
          ; (err as any).code = message.error?.code
          ; (err as any).data = message.error?.data
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

    this.handleTransportClose('codex app-server transport closed')
    this.transport.kill()
  }
}

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

export class Signal<T> {
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
