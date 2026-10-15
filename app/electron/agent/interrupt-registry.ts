// Programmatic mid-run interrupt + edit-args-then-resume — the primitive approval-policy.ts
// (a pure yes/no scope classifier) deliberately stops short of. approval-policy.ts answers
// "does THIS tool call, with THIS argument signature, need a human?"; it has no notion of a
// call that is PARKED in flight while a human (or another agent) decides, possibly EDITS the
// arguments, and then lets it RESUME. This module is that missing piece.
//
// A tool/plugin raises a typed `InterruptRequest`; the registry parks it (handing back a
// promise that stays pending) and remembers it under a deterministic id. Out of band, a
// resolver looks the parked request up and resolves it exactly once — `allow` (with the
// original OR edited args), `deny` (with a reason), or `answer` (free-form payload, e.g. a
// reply to a question the tool asked). The parked promise then settles deterministically.
//
// Pure + fully injectable: no timers, no globals, no I/O. The id source and clock are injected
// (same idiom as ssh-connection's clientFactory, connector-retry's fetchImpl, pr-publish's
// runner) so behaviour is byte-for-byte reproducible in tests. Nothing here touches the live
// loop; wiring is reported separately.

// ── Request kinds ────────────────────────────────────────────────────────────
// 'approval' — a tool call is parked awaiting allow/deny; the resolver MAY edit the args.
// 'question' — the tool needs a free-form answer before it can proceed (e.g. "which file?").
// 'choice'   — the tool needs one option picked from a fixed set.
export type InterruptKind = 'approval' | 'question' | 'choice'

export interface InterruptRequest<Args = Record<string, unknown>> {
  kind: InterruptKind
  // Human/telemetry label for what is being interrupted.
  label: string
  // The tool (or plugin action) that raised the interrupt.
  tool: string
  // The arguments the tool was called with. For 'approval' these may be replaced on resume.
  args?: Args
  // For 'question'/'choice': the prompt shown to whoever resolves.
  prompt?: string
  // For 'choice': the allowed option values. A resolve must pick one of these.
  options?: string[]
  // Free-form context carried through untouched (mission/room ids, risk score, etc.).
  meta?: Record<string, unknown>
}

// ── Resolutions ──────────────────────────────────────────────────────────────
export type InterruptResolution<Args = Record<string, unknown>> =
  | { type: 'allow'; args?: Args } // args omitted → keep the original; present → edited args win
  | { type: 'deny'; reason?: string }
  | { type: 'answer'; value: unknown }

// What the parked promise settles TO. `allow` resolves with the FINAL args (edited or original)
// and an `edited` flag so the caller can tell whether a human changed them. `deny` REJECTS with
// an InterruptDeniedError. `answer` resolves with the value.
export interface AllowOutcome<Args = Record<string, unknown>> {
  type: 'allow'
  args: Args | undefined
  edited: boolean
}
export interface AnswerOutcome {
  type: 'answer'
  value: unknown
}
export type InterruptOutcome<Args = Record<string, unknown>> = AllowOutcome<Args> | AnswerOutcome

export class InterruptDeniedError extends Error {
  readonly interruptId: string
  readonly reason: string
  constructor(interruptId: string, reason: string) {
    super(reason || 'Interrupt denied')
    this.name = 'InterruptDeniedError'
    this.interruptId = interruptId
    this.reason = reason || 'Interrupt denied'
  }
}

// A parked interrupt as seen from the outside (the resolver side / a UI list).
export interface ParkedInterrupt<Args = Record<string, unknown>> {
  id: string
  request: InterruptRequest<Args>
  createdAt: number
}

export interface InterruptRegistryOptions {
  // Injected id source — MUST be deterministic in tests. Default: monotonic counter.
  nextId?: () => string
  // Injected clock (ms). Default: 0-based monotonic counter (NOT Date.now — keep it pure).
  now?: () => number
  // Observe lifecycle transitions (park/resolve/deny) for logging/telemetry. Never throws back.
  onEvent?: (event: InterruptEvent) => void
}

export type InterruptEvent =
  | { type: 'parked'; id: string; kind: InterruptKind; tool: string; at: number }
  | { type: 'resolved'; id: string; resolution: 'allow' | 'deny' | 'answer'; edited: boolean; at: number }

const OPTION_UNKNOWN = (opt: string, options: string[]) =>
  `Resolution option "${opt}" is not one of [${options.join(', ')}]`

/**
 * A registry that parks interrupt requests and resolves them deterministically.
 *
 * Contract:
 *  - `park(request)` returns `{ id, promise }`. The promise stays pending until `resolve(id, …)`.
 *  - `resolve(id, resolution)` settles the parked promise EXACTLY once and forgets the entry.
 *  - A second `resolve` for the same id throws (double-resolution is a bug, surfaced loudly).
 *  - `resolve` for an unknown id throws.
 *  - `answer` on an 'approval' request, or `allow` on a 'question'/'choice', throws —
 *    the resolution kind must match the request kind.
 *  - A 'choice' answer whose value isn't in `options` throws.
 */
export class InterruptRegistry {
  private readonly parked = new Map<string, {
    request: InterruptRequest<any>
    createdAt: number
    settle: (outcome: InterruptOutcome<any>) => void
    fail: (err: Error) => void
  }>()
  private readonly nextId: () => string
  private readonly now: () => number
  private readonly onEvent?: (event: InterruptEvent) => void
  private counter = 0
  private clock = 0

  constructor(options: InterruptRegistryOptions = {}) {
    this.nextId = options.nextId ?? (() => `int_${++this.counter}`)
    this.now = options.now ?? (() => this.clock++)
    this.onEvent = options.onEvent
  }

  /** Park a request. Returns its id and a promise that settles when it is resolved. */
  park<Args = Record<string, unknown>>(
    request: InterruptRequest<Args>,
  ): { id: string; promise: Promise<InterruptOutcome<Args>> } {
    validateRequest(request)
    const id = this.nextId()
    if (this.parked.has(id)) throw new Error(`Interrupt id collision: ${id}`)
    const createdAt = this.now()
    let settle!: (outcome: InterruptOutcome<Args>) => void
    let fail!: (err: Error) => void
    const promise = new Promise<InterruptOutcome<Args>>((res, rej) => {
      settle = res
      fail = rej
    })
    this.parked.set(id, { request, createdAt, settle: settle as any, fail })
    this.emit({ type: 'parked', id, kind: request.kind, tool: request.tool, at: createdAt })
    return { id, promise }
  }

  /** True if an interrupt with this id is currently parked (not yet resolved). */
  has(id: string): boolean {
    return this.parked.has(id)
  }

  /** How many interrupts are currently parked. */
  get size(): number {
    return this.parked.size
  }

  /** Snapshot of all parked interrupts, oldest first — for a resolver UI / list. */
  list(): ParkedInterrupt[] {
    return [...this.parked.entries()]
      .sort((a, b) => a[1].createdAt - b[1].createdAt || a[0].localeCompare(b[0]))
      .map(([id, entry]) => ({ id, request: entry.request, createdAt: entry.createdAt }))
  }

  /** Peek at a single parked request without resolving it. */
  peek(id: string): ParkedInterrupt | undefined {
    const entry = this.parked.get(id)
    return entry ? { id, request: entry.request, createdAt: entry.createdAt } : undefined
  }

  /**
   * Resolve a parked interrupt exactly once. Settles its promise and removes the entry.
   * Throws (WITHOUT settling anything) if the id is unknown or the resolution is invalid for
   * the request kind — an invalid resolve must not consume the parked request.
   */
  resolve<Args = Record<string, unknown>>(id: string, resolution: InterruptResolution<Args>): void {
    const entry = this.parked.get(id)
    if (!entry) throw new Error(`No parked interrupt with id: ${id}`)
    // Validate BEFORE mutating so a bad resolve leaves the request parked & retry-able.
    validateResolution(entry.request, resolution)

    this.parked.delete(id)
    const at = this.now()

    if (resolution.type === 'deny') {
      this.emit({ type: 'resolved', id, resolution: 'deny', edited: false, at })
      entry.fail(new InterruptDeniedError(id, resolution.reason ?? 'Interrupt denied'))
      return
    }
    if (resolution.type === 'answer') {
      this.emit({ type: 'resolved', id, resolution: 'answer', edited: false, at })
      entry.settle({ type: 'answer', value: resolution.value })
      return
    }
    // allow — edited args win when present; otherwise keep the original.
    const edited = Object.prototype.hasOwnProperty.call(resolution, 'args') && resolution.args !== undefined
    const finalArgs = (edited ? resolution.args : entry.request.args) as Args | undefined
    this.emit({ type: 'resolved', id, resolution: 'allow', edited, at })
    entry.settle({ type: 'allow', args: finalArgs, edited })
  }

  /**
   * Deny every parked interrupt (e.g. the run was cancelled). Settles all their promises with
   * an InterruptDeniedError and clears the registry. Returns the ids that were cancelled.
   */
  cancelAll(reason = 'Run cancelled'): string[] {
    const ids = [...this.parked.keys()]
    for (const id of ids) this.resolve(id, { type: 'deny', reason })
    return ids
  }

  private emit(event: InterruptEvent): void {
    if (!this.onEvent) return
    try {
      this.onEvent(event)
    } catch {
      /* observers never break the registry */
    }
  }
}

// ── Validation (pure) ────────────────────────────────────────────────────────

function validateRequest(request: InterruptRequest<any>): void {
  if (!request || typeof request !== 'object') throw new Error('Interrupt request must be an object')
  if (request.kind !== 'approval' && request.kind !== 'question' && request.kind !== 'choice') {
    throw new Error(`Unknown interrupt kind: ${String((request as { kind?: unknown }).kind)}`)
  }
  if (!request.tool || typeof request.tool !== 'string') throw new Error('Interrupt request needs a tool')
  if (request.kind === 'choice') {
    if (!Array.isArray(request.options) || request.options.length === 0) {
      throw new Error("Interrupt kind 'choice' requires a non-empty options array")
    }
  }
}

function validateResolution(request: InterruptRequest<any>, resolution: InterruptResolution<any>): void {
  if (!resolution || typeof resolution !== 'object') throw new Error('Resolution must be an object')

  if (request.kind === 'approval') {
    if (resolution.type !== 'allow' && resolution.type !== 'deny') {
      throw new Error(`Approval interrupts resolve with 'allow' or 'deny', got '${resolution.type}'`)
    }
    return
  }

  // question / choice — expect an 'answer'. (deny is allowed to reject a question too.)
  if (resolution.type === 'allow') {
    throw new Error(`'${request.kind}' interrupts do not take 'allow'; use 'answer' or 'deny'`)
  }
  if (resolution.type === 'answer' && request.kind === 'choice') {
    const value = String(resolution.value)
    if (!request.options!.includes(value)) throw new Error(OPTION_UNKNOWN(value, request.options!))
  }
}
