// Additive, in-process event broker for orchestration lifecycle (missions / rooms / jobs).
//
// Today the Command Centre POLLS snapshots (missions:snapshot, jobs:get, …) to learn
// when a mission/room/job changes state. That works but is laggy and wasteful. This
// broker lets the supervisors PUSH a typed lifecycle event the instant a transition
// happens, so the renderer can update live — while the existing poll/snapshot path
// stays exactly as it was as the reconciling fallback (so a missed event never leaves
// the UI stale).
//
// Design (mirrors the rest of this codebase):
//   * The broker itself is a tiny, dependency-free pub/sub — no Electron, no disk — so
//     it is unit-testable under node/jiti and importable from the pure agent modules
//     (mission-supervisor.ts, rooms.ts) without dragging `electron` into the agent test
//     runner. The Electron host bridges it to the renderer separately
//     (wireOrchestrationEventsToWindow), the same split orchestration-journal.ts uses.
//   * Emitting is fire-and-forget and error-tolerant: a throwing subscriber can never
//     break a live mission/room run (the supervisors call this alongside their existing
//     persistence + snapshot updates, never instead of them).
//   * The event shapes are derived from the lifecycle event types the supervisors
//     already produce for the durable journal (orchestration-journal.ts), so this adds
//     no new transition vocabulary — it just makes those transitions observable live.

import type {
  JobKind,
  JobStatus,
  JobLifecycleEvent,
  MissionLifecycleEvent,
  MissionLifecyclePhase,
  MissionRepeatMode,
  MissionStatus,
  OrchestrationEvent,
  RoomLifecycleEvent,
  RoomLifecyclePhase,
  RoomState,
} from '../../agent/types'

// Event shapes (MissionLifecycleEvent / RoomLifecycleEvent / JobLifecycleEvent /
// OrchestrationEvent + the *Phase unions) live in the shared type layer
// (electron/agent/types/jobs.ts) so the renderer can consume them type-only over the
// push channel. Re-export them here so existing host-side importers of this module keep
// working unchanged.
export type {
  JobLifecycleEvent,
  MissionLifecycleEvent,
  MissionLifecyclePhase,
  OrchestrationEvent,
  RoomLifecycleEvent,
  RoomLifecyclePhase,
} from '../../agent/types'

export type OrchestrationEventListener = (event: OrchestrationEvent) => void

// ---------------------------------------------------------------------------
// Pure event shapers — build a well-formed event, dropping undefined optionals so the
// payload stays compact and stable (easy to assert in tests, cheap to serialize over
// IPC). Exported for unit testing the shaping in isolation from the broker.
// ---------------------------------------------------------------------------

export function missionLifecycleEvent(
  phase: MissionLifecyclePhase,
  missionId: string,
  fields: { status?: MissionStatus; title?: string; repeatMode?: MissionRepeatMode; runIndex?: number; reason?: string; ts?: number } = {},
): MissionLifecycleEvent {
  const event: MissionLifecycleEvent = { scope: 'mission', phase, missionId, ts: fields.ts ?? Date.now() }
  if (fields.status !== undefined) event.status = fields.status
  if (fields.title !== undefined) event.title = fields.title
  if (fields.repeatMode !== undefined) event.repeatMode = fields.repeatMode
  if (typeof fields.runIndex === 'number' && Number.isFinite(fields.runIndex)) event.runIndex = Math.max(0, Math.floor(fields.runIndex))
  if (fields.reason !== undefined && fields.reason !== '') event.reason = fields.reason.slice(0, 400)
  return event
}

export function roomLifecycleEvent(
  phase: RoomLifecyclePhase,
  roomId: string,
  fields: { state?: RoomState; seat?: string; ts?: number } = {},
): RoomLifecycleEvent {
  const event: RoomLifecycleEvent = { scope: 'room', phase, roomId, ts: fields.ts ?? Date.now() }
  if (fields.state !== undefined) event.state = fields.state
  if (fields.seat !== undefined && fields.seat !== '') event.seat = fields.seat
  return event
}

export function jobLifecycleEvent(
  jobId: string,
  status: JobStatus,
  kind: JobKind,
  fields: { missionId?: string | null; roomId?: string | null; ts?: number } = {},
): JobLifecycleEvent {
  const event: JobLifecycleEvent = { scope: 'job', phase: 'status_changed', jobId, status, kind, ts: fields.ts ?? Date.now() }
  if (fields.missionId !== undefined) event.missionId = fields.missionId
  if (fields.roomId !== undefined) event.roomId = fields.roomId
  return event
}

// ---------------------------------------------------------------------------
// The broker — a minimal synchronous pub/sub. Kept deliberately tiny (no node
// EventEmitter) so it carries no listener-cap warnings and stays trivially testable.
// ---------------------------------------------------------------------------

export class OrchestrationEventBroker {
  private listeners = new Set<OrchestrationEventListener>()

  /** Subscribe to every orchestration event. Returns an idempotent unsubscribe. */
  subscribe(listener: OrchestrationEventListener): () => void {
    this.listeners.add(listener)
    let active = true
    return () => {
      if (!active) return
      active = false
      this.listeners.delete(listener)
    }
  }

  /**
   * Deliver an event to all current subscribers. A subscriber that throws is isolated:
   * its error is swallowed so one bad listener can neither break a live run nor stop the
   * remaining subscribers from receiving the event. Iterates a snapshot so a listener
   * that subscribes/unsubscribes during delivery does not perturb this dispatch.
   */
  emit(event: OrchestrationEvent): void {
    if (this.listeners.size === 0) return
    for (const listener of [...this.listeners]) {
      try {
        listener(event)
      } catch {
        // isolate a faulty subscriber — never propagate into the supervisor
      }
    }
  }

  /** Current subscriber count (test/observability helper). */
  get size(): number {
    return this.listeners.size
  }

  /** Drop all subscribers (test-only / teardown). */
  clear(): void {
    this.listeners.clear()
  }
}

// Process-wide singleton: the supervisors emit into it, the Electron host subscribes to
// forward to the renderer. A module singleton (not a passed dependency) so the pure
// agent modules can import-and-emit without threading the broker through every call.
export const orchestrationEvents = new OrchestrationEventBroker()

// ---------------------------------------------------------------------------
// Convenience emitters used at the supervisor transition points. Each builds the event
// with the pure shaper above and pushes it through the singleton. Error-tolerant by way
// of the broker's per-listener isolation; the shapers never throw on well-typed input.
// ---------------------------------------------------------------------------

export function emitMissionLifecycle(
  phase: MissionLifecyclePhase,
  missionId: string,
  fields: Parameters<typeof missionLifecycleEvent>[2] = {},
): void {
  if (!missionId) return
  orchestrationEvents.emit(missionLifecycleEvent(phase, missionId, fields))
}

export function emitRoomLifecycle(
  phase: RoomLifecyclePhase,
  roomId: string,
  fields: Parameters<typeof roomLifecycleEvent>[2] = {},
): void {
  if (!roomId) return
  orchestrationEvents.emit(roomLifecycleEvent(phase, roomId, fields))
}

export function emitJobLifecycle(
  jobId: string,
  status: JobStatus,
  kind: JobKind,
  fields: Parameters<typeof jobLifecycleEvent>[3] = {},
): void {
  if (!jobId) return
  orchestrationEvents.emit(jobLifecycleEvent(jobId, status, kind, fields))
}

// ---------------------------------------------------------------------------
// Electron host bridge. Mirrors the onLoopUpdate push (loop-host.ts emits
// store.win.webContents.send('loop:update', …)): subscribe to the broker and forward
// each event to the renderer over the 'orchestration:event' channel. Takes a `send`
// thunk (not a BrowserWindow) so it stays Electron-free and unit-testable, and so a
// torn-down / not-yet-created window is simply skipped. Returns the unsubscribe.
// ---------------------------------------------------------------------------

export function wireOrchestrationEventsToWindow(
  send: (event: OrchestrationEvent) => void,
  broker: OrchestrationEventBroker = orchestrationEvents,
): () => void {
  return broker.subscribe((event) => {
    try {
      send(event)
    } catch {
      // a destroyed webContents / serialization hiccup must not break the run
    }
  })
}
