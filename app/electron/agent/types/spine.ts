// Ordered event spine types (VST-034). Kept in the shared type layer so the renderer can
// consume spine events/receipts type-only; the store logic lives in agent/event-spine.ts.

export type SpineAggregate =
  | 'thread'
  | 'room'
  | 'ticket'
  | 'job'
  | 'terminal'
  | 'mission'
  | 'automation'
  | 'connector'
  | 'settings'

export interface SpineEventInput<TPayload = unknown> {
  aggregate: SpineAggregate
  streamId: string
  type: string
  payload: TPayload
  eventId?: string
  occurredAt?: number
  commandId?: string | null
  causationEventId?: string | null
  correlationId?: string | null
  actor?: string | null
}

export interface SpineEvent<TPayload = unknown> {
  sequence: number
  eventId: string
  aggregate: SpineAggregate
  streamId: string
  streamVersion: number
  type: string
  occurredAt: number
  commandId: string | null
  causationEventId: string | null
  correlationId: string | null
  actor: string | null
  payload: TPayload
}

export interface SpineCommand {
  commandId: string
  aggregate: SpineAggregate
  streamId: string
  kind: string
  actor?: string | null
  correlationId?: string | null
}

export interface CommandReceipt {
  commandId: string
  aggregate: SpineAggregate
  streamId: string
  kind: string
  status: 'accepted' | 'rejected'
  // Sequence of the LAST event the command recorded (null when rejected or event-less).
  resultSequence: number | null
  error: string | null
  at: number
}

export interface ReadSinceOptions {
  aggregate?: SpineAggregate
  streamId?: string
  limit?: number
}

export interface ReadSinceResult {
  events: SpineEvent[]
  // True when more events exist past the returned page.
  hasMore: boolean
  // True when `since` fell below the compaction watermark: the caller missed events that no
  // longer exist and must re-baseline from a snapshot instead of trusting this page.
  snapshotRequired: boolean
}

export interface SpineRestoreState {
  events?: SpineEvent[]
  receipts?: CommandReceipt[]
  cursors?: Record<string, number>
  compactedThrough?: number
}
