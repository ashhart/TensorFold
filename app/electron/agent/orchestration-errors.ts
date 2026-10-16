// Typed error hierarchy for mission/room orchestration.
//
// These carry a structured { command, issue, detail } payload so a guard can fail
// an illegal state transition with an actionable, machine-inspectable reason rather
// than a bare string. They are surfaced via the existing error event paths
// (mission journal `error` events; room `send({ type: 'error' })`) — nothing here
// changes control flow for legal transitions; the invariant modules return `null`
// in that case and only mint one of these when an invariant is actually violated.

export interface OrchestrationErrorFields {
  // The operation that was attempted, e.g. 'mission:transition' or 'room:state'.
  command: string
  // A short, stable summary of what is wrong (suitable for a one-line surface).
  issue: string
  // Optional longer explanation with the specifics (states involved, counts, …).
  detail?: string
}

// Base class for every orchestration invariant failure. Extends Error so it flows
// through normal try/catch + the string-based error event surfaces unchanged, while
// also exposing the structured fields for callers that want them.
export class OrchestrationError extends Error {
  readonly command: string
  readonly issue: string
  readonly detail: string

  constructor(fields: OrchestrationErrorFields) {
    const detail = fields.detail?.trim() ?? ''
    super(detail ? `${fields.issue}: ${detail}` : fields.issue)
    this.name = 'OrchestrationError'
    this.command = fields.command
    this.issue = fields.issue
    this.detail = detail
    // Preserve the prototype chain when compiled down to ES5-style targets so
    // `instanceof` keeps working across the subclasses.
    Object.setPrototypeOf(this, new.target.prototype)
  }

  // Stable structured view for logging / journalling / IPC payloads.
  toFields(): Required<OrchestrationErrorFields> {
    return { command: this.command, issue: this.issue, detail: this.detail }
  }
}

// A mission attempted an illegal status transition (e.g. resuming a completed
// mission, or completing a cancelled one).
export class MissionInvariantError extends OrchestrationError {
  constructor(fields: OrchestrationErrorFields) {
    super(fields)
    this.name = 'MissionInvariantError'
  }
}

// A room attempted an illegal state transition, or its configuration is internally
// inconsistent (seat count vs collaboration mode, effort compatibility).
export class RoomStateError extends OrchestrationError {
  constructor(fields: OrchestrationErrorFields) {
    super(fields)
    this.name = 'RoomStateError'
  }
}

// An approval gate was violated — e.g. resolving an approval that does not exist,
// or proceeding past a gate that still has pending approvals.
export class ApprovalGateError extends OrchestrationError {
  constructor(fields: OrchestrationErrorFields) {
    super(fields)
    this.name = 'ApprovalGateError'
  }
}

export function isOrchestrationError(value: unknown): value is OrchestrationError {
  return value instanceof OrchestrationError
}
