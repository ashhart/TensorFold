// Ordered event spine v2 (VST-034): the pure core of an event-sourced store with ordered
// events, command receipts and projection cursors.
//
// Semantics:
// - Every event gets a GLOBAL monotonic `sequence` and a per-stream `streamVersion`.
// - Appends are idempotent by `eventId`: a duplicate delivery returns the already-stored
//   event instead of double-recording it.
// - Mutations run as COMMANDS: the handler returns the events to record and they append
//   atomically after it succeeds; a throw yields a rejected receipt and records nothing.
//   Re-executing a commandId returns the cached receipt WITHOUT re-running the handler
//   (exactly-once across retries/reconnects).
// - Projectors keep cursors (last applied sequence) and catch up via `project()`; cursors
//   never move backwards, so crash-resume just projects again.
// - `compactThrough()` drops events every cursor has passed, keeping memory bounded; reads
//   below the compaction watermark say so instead of silently returning a gap.
//
// This module is pure and dependency-injected: persistence (disk journal), transports
// (IPC/SSE), and the existing AgentEvent bridge wire up in the host layer.

// Type definitions live in ./types/spine (the shared type layer) so the renderer can
// consume them type-only; re-exported here so runtime callers keep one import site.
import type { CommandReceipt, ReadSinceOptions, ReadSinceResult, SpineAggregate, SpineCommand, SpineEvent, SpineEventInput, SpineRestoreState } from './types/spine'

export type { CommandReceipt, ReadSinceOptions, ReadSinceResult, SpineAggregate, SpineCommand, SpineEvent, SpineEventInput, SpineRestoreState }

export interface EventSpineOptions {
  now?: () => number
  newEventId?: () => string
}

let fallbackIdCounter = 0

export class EventSpine {
  private events: SpineEvent[] = []
  private byEventId = new Map<string, SpineEvent>()
  private receipts = new Map<string, CommandReceipt>()
  private streamVersions = new Map<string, number>()
  private cursors = new Map<string, number>()
  private subscribers = new Set<(event: SpineEvent) => void>()
  private head = 0
  private compactedThrough = 0
  private readonly now: () => number
  private readonly newEventId: () => string

  constructor(options: EventSpineOptions = {}, restore: SpineRestoreState = {}) {
    this.now = options.now ?? (() => Date.now())
    this.newEventId = options.newEventId ?? (() => `evt-${++fallbackIdCounter}-${this.now().toString(36)}`)
    for (const event of restore.events ?? []) {
      this.events.push(event)
      this.byEventId.set(event.eventId, event)
      this.head = Math.max(this.head, event.sequence)
      const key = this.streamKey(event.aggregate, event.streamId)
      this.streamVersions.set(key, Math.max(this.streamVersions.get(key) ?? 0, event.streamVersion))
    }
    this.events.sort((a, b) => a.sequence - b.sequence)
    for (const receipt of restore.receipts ?? []) this.receipts.set(receipt.commandId, receipt)
    for (const [projector, sequence] of Object.entries(restore.cursors ?? {})) this.cursors.set(projector, sequence)
    this.compactedThrough = restore.compactedThrough ?? 0
  }

  private streamKey(aggregate: SpineAggregate, streamId: string): string {
    return `${aggregate}:${streamId}`
  }

  get headSequence(): number {
    return this.head
  }

  get compactionWatermark(): number {
    return this.compactedThrough
  }

  /** Number of replay/idempotency records retained in memory (for purge accounting). */
  recordCount(): number {
    return this.events.length + this.receipts.size + this.cursors.size
  }

  /**
   * Forget replay events, command receipts, projector cursors, and stream versions while
   * preserving the global sequence watermark. Keeping `head` as the compaction watermark
   * means an already-connected client with an old cursor is explicitly told to re-baseline,
   * and new events never reuse a pre-purge sequence number.
   */
  clearAll(): number {
    const removed = this.recordCount()
    this.events = []
    this.byEventId.clear()
    this.receipts.clear()
    this.streamVersions.clear()
    this.cursors.clear()
    this.compactedThrough = Math.max(this.compactedThrough, this.head)
    return removed
  }

  append<TPayload>(input: SpineEventInput<TPayload>): SpineEvent<TPayload> {
    const eventId = input.eventId ?? this.newEventId()
    const existing = this.byEventId.get(eventId)
    // Idempotent redelivery: same eventId → the already-recorded fact, no new sequence.
    if (existing) return existing as SpineEvent<TPayload>
    const key = this.streamKey(input.aggregate, input.streamId)
    const streamVersion = (this.streamVersions.get(key) ?? 0) + 1
    const event: SpineEvent<TPayload> = {
      sequence: ++this.head,
      eventId,
      aggregate: input.aggregate,
      streamId: input.streamId,
      streamVersion,
      type: input.type,
      occurredAt: input.occurredAt ?? this.now(),
      commandId: input.commandId ?? null,
      causationEventId: input.causationEventId ?? null,
      correlationId: input.correlationId ?? null,
      actor: input.actor ?? null,
      payload: input.payload,
    }
    this.streamVersions.set(key, streamVersion)
    this.events.push(event)
    this.byEventId.set(eventId, event)
    for (const subscriber of [...this.subscribers]) {
      try { subscriber(event) } catch { /* one bad subscriber must not break the fan-out */ }
    }
    return event
  }

  // Execute a mutation exactly once. The handler RETURNS the events to record; they are
  // appended atomically only after it succeeds. A repeated commandId (retry, reconnect,
  // double-click) returns the cached receipt without re-running the handler.
  execute(command: SpineCommand, handler: () => Array<SpineEventInput> | SpineEventInput | void): CommandReceipt {
    const cached = this.receipts.get(command.commandId)
    if (cached) return cached
    let receipt: CommandReceipt
    try {
      const produced = handler()
      const inputs = produced == null ? [] : Array.isArray(produced) ? produced : [produced]
      let last: SpineEvent | null = null
      for (const input of inputs) {
        last = this.append({
          ...input,
          commandId: command.commandId,
          correlationId: input.correlationId ?? command.correlationId ?? null,
          actor: input.actor ?? command.actor ?? null,
        })
      }
      receipt = {
        commandId: command.commandId,
        aggregate: command.aggregate,
        streamId: command.streamId,
        kind: command.kind,
        status: 'accepted',
        resultSequence: last?.sequence ?? null,
        error: null,
        at: this.now(),
      }
    } catch (error) {
      receipt = {
        commandId: command.commandId,
        aggregate: command.aggregate,
        streamId: command.streamId,
        kind: command.kind,
        status: 'rejected',
        resultSequence: null,
        error: error instanceof Error ? error.message : String(error),
        at: this.now(),
      }
    }
    this.receipts.set(command.commandId, receipt)
    return receipt
  }

  receipt(commandId: string): CommandReceipt | null {
    return this.receipts.get(commandId) ?? null
  }

  readSince(since: number, options: ReadSinceOptions = {}): ReadSinceResult {
    const snapshotRequired = since < this.compactedThrough
    const from = Math.max(since, this.compactedThrough)
    const matches = this.events.filter((event) => (
      event.sequence > from &&
      (!options.aggregate || event.aggregate === options.aggregate) &&
      (!options.streamId || event.streamId === options.streamId)
    ))
    const limit = options.limit ?? Infinity
    const events = matches.slice(0, limit)
    return { events, hasMore: matches.length > events.length, snapshotRequired }
  }

  subscribe(fn: (event: SpineEvent) => void): () => void {
    this.subscribers.add(fn)
    return () => { this.subscribers.delete(fn) }
  }

  cursor(projector: string): number {
    return this.cursors.get(projector) ?? 0
  }

  // Cursors only move forward — a stale/duplicate advance is a no-op, so replays and
  // out-of-order acks can never rewind a projector into double-applying.
  advance(projector: string, sequence: number): number {
    const current = this.cursor(projector)
    if (sequence > current) this.cursors.set(projector, sequence)
    return this.cursor(projector)
  }

  // Catch a projector up: apply every event past its cursor in order, advancing as it goes.
  // An apply() throw stops the catch-up AT that event (cursor stays before it) so the next
  // project() retries from the failure point instead of skipping it.
  project(projector: string, apply: (event: SpineEvent) => void, options: ReadSinceOptions = {}): { applied: number; cursor: number; snapshotRequired: boolean } {
    const { events, snapshotRequired } = this.readSince(this.cursor(projector), options)
    let applied = 0
    for (const event of events) {
      try {
        apply(event)
      } catch {
        break
      }
      this.advance(projector, event.sequence)
      applied++
    }
    return { applied, cursor: this.cursor(projector), snapshotRequired }
  }

  // Drop events every registered cursor has passed (bounded memory). Returns the new
  // watermark. Events at or below the watermark are gone; readSince below it flags
  // snapshotRequired so late consumers re-baseline instead of silently missing history.
  compactThrough(sequence: number): number {
    const minCursor = this.cursors.size ? Math.min(...this.cursors.values()) : sequence
    const target = Math.min(sequence, minCursor)
    if (target <= this.compactedThrough) return this.compactedThrough
    this.events = this.events.filter((event) => {
      if (event.sequence > target) return true
      this.byEventId.delete(event.eventId)
      return false
    })
    this.compactedThrough = target
    return this.compactedThrough
  }

  snapshotState(): Required<SpineRestoreState> {
    return {
      events: [...this.events],
      receipts: [...this.receipts.values()],
      cursors: Object.fromEntries(this.cursors),
      compactedThrough: this.compactedThrough,
    }
  }
}

export function createEventSpine(options: EventSpineOptions = {}, restore: SpineRestoreState = {}): EventSpine {
  return new EventSpine(options, restore)
}
