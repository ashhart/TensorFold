// Pure, tested invariants for multi-agent rooms.
//
// Three independent guards, each returning a typed RoomStateError or null:
//   1. validateRoomTransition(from, to) — legal RoomState moves.
//   2. validateRoomSeating(seatCount, mode) — seat count vs collaboration mode.
//   3. validateRoomEffortCompatibility(seatCount, effort) — deliberation effort fit.
//
// The room runtime (`rooms.ts`) drives `room.state` through a known set of moves and
// surfaces failures over its error event channel. These functions make the rule set
// explicit and testable without changing behavior for legal transitions (they return
// null on the happy path).

import { RoomStateError } from './orchestration-errors'
import type { RoomCollaborationMode, RoomDeliberationEffort, RoomState } from './types'

const TRANSITION_COMMAND = 'room:state'
const SEATING_COMMAND = 'room:seating'
const EFFORT_COMMAND = 'room:effort'

export const ROOM_STATES: readonly RoomState[] = [
  'idle',
  'created',
  'ready',
  'degraded',
  'queued',
  'running',
  'waiting_user',
  'waiting_external',
  'complete',
  'stopped',
  'failed',
  'archived',
  'error',
]

// Absorbing end states. `archived` is reachable from anywhere (a user can archive a
// room in any state), so it is handled specially in the transition check.
export const TERMINAL_ROOM_STATE_SET: ReadonlySet<RoomState> = new Set<RoomState>([
  'complete',
  'stopped',
  'failed',
  'archived',
])

// States from which a room can (re)enter an active run.
const STARTABLE: readonly RoomState[] = ['idle', 'created', 'ready', 'degraded', 'queued', 'waiting_user', 'waiting_external', 'error']

// Allowed forward transitions per source state (self-edge allowed implicitly).
// Mirrors the real runtime in `rooms.ts`, where a run moves the room to 'running'
// and then resolves to one of complete / waiting_user / waiting_external / stopped /
// failed, plus the pre-run setup states.
const RUN_OUTCOMES: readonly RoomState[] = ['complete', 'waiting_user', 'waiting_external', 'stopped', 'failed']

const ROOM_TRANSITIONS: Record<RoomState, readonly RoomState[]> = {
  idle: ['created', 'ready', 'queued', 'running', 'degraded', 'archived'],
  created: ['ready', 'queued', 'running', 'degraded', 'archived'],
  ready: ['queued', 'running', 'degraded', 'archived'],
  degraded: ['ready', 'queued', 'running', 'archived', 'failed'],
  queued: ['running', 'ready', 'degraded', 'archived', 'failed', 'stopped'],
  running: [...RUN_OUTCOMES, 'archived'],
  waiting_user: ['queued', 'running', ...RUN_OUTCOMES, 'archived'],
  waiting_external: ['queued', 'running', ...RUN_OUTCOMES, 'archived'],
  error: ['queued', 'running', 'degraded', 'ready', 'archived', 'failed', 'stopped'],
  // Absorbing end states: only re-archival / idempotent self-edge (and a user may
  // re-open complete/stopped/failed rooms back to ready via the UI's reset path).
  complete: ['ready', 'queued', 'running', 'archived'],
  stopped: ['ready', 'queued', 'running', 'archived'],
  failed: ['ready', 'queued', 'running', 'archived'],
  archived: ['ready', 'idle'],
}

// Compile-time exhaustiveness guard against new RoomState members.
const EXHAUSTIVE: Record<RoomState, readonly RoomState[]> = ROOM_TRANSITIONS
void EXHAUSTIVE

export function isTerminalRoomState(state: RoomState): boolean {
  return TERMINAL_ROOM_STATE_SET.has(state)
}

export function isStartableRoomState(state: RoomState): boolean {
  return STARTABLE.includes(state)
}

export function isLegalRoomTransition(from: RoomState, to: RoomState): boolean {
  if (from === to) return true // idempotent re-emit
  if (to === 'archived') return true // a room can be archived from any state
  return ROOM_TRANSITIONS[from]?.includes(to) ?? false
}

export function validateRoomTransition(from: RoomState, to: RoomState): RoomStateError | null {
  if (isLegalRoomTransition(from, to)) return null

  const reason = isTerminalRoomState(from)
    ? `room is in end state "${from}" and cannot transition to "${to}"`
    : `room cannot transition from "${from}" to "${to}"`

  return new RoomStateError({
    command: TRANSITION_COMMAND,
    issue: `Illegal room transition ${from} → ${to}`,
    detail: reason,
  })
}

// Minimum seats a collaboration mode needs to be meaningful. A room with fewer
// seats than this can technically run, but the mode's premise (council debate,
// frontier-plans/local-builds split) is broken, so we guard it.
const MODE_MIN_SEATS: Record<RoomCollaborationMode, number> = {
  // A single seat cannot deliberate as a council nor split planning across seats.
  frontier_council: 2,
  frontier_plans_local_builds: 2,
  // A supervisor needs at least one other seat to delegate to.
  supervisor: 2,
  // Balanced + local_private are coherent with a single seat (it just plays every
  // lane), so they only require at least one seat.
  balanced: 1,
  local_private: 1,
}

export function minSeatsForCollaborationMode(mode: RoomCollaborationMode): number {
  return MODE_MIN_SEATS[mode] ?? 1
}

// Validate the seat roster against the collaboration mode. Returns a typed error
// when there are too few seats for the mode (including the zero-seat case, which is
// never runnable), or null when the roster is adequate.
export function validateRoomSeating(seatCount: number, mode: RoomCollaborationMode): RoomStateError | null {
  const seats = Number.isFinite(seatCount) ? Math.max(0, Math.floor(seatCount)) : 0
  if (seats <= 0) {
    return new RoomStateError({
      command: SEATING_COMMAND,
      issue: 'Room has no seats',
      detail: 'a room needs at least one seat to run',
    })
  }
  const min = minSeatsForCollaborationMode(mode)
  if (seats < min) {
    return new RoomStateError({
      command: SEATING_COMMAND,
      issue: `Collaboration mode "${mode}" needs at least ${min} seats`,
      detail: `mode "${mode}" requires ${min} seats but the room has ${seats}`,
    })
  }
  return null
}

// Deep / max deliberation efforts are expensive multi-turn councils — they only pay
// off with more than one seat. A single-seat room on a high effort just burns turns
// talking to itself, so we flag it (callers may downgrade or warn).
const HIGH_EFFORTS: ReadonlySet<RoomDeliberationEffort> = new Set<RoomDeliberationEffort>(['deep', 'max'])

export function validateRoomEffortCompatibility(seatCount: number, effort: RoomDeliberationEffort): RoomStateError | null {
  const seats = Number.isFinite(seatCount) ? Math.max(0, Math.floor(seatCount)) : 0
  if (seats <= 1 && HIGH_EFFORTS.has(effort)) {
    return new RoomStateError({
      command: EFFORT_COMMAND,
      issue: `Deliberation effort "${effort}" needs more than one seat`,
      detail: `effort "${effort}" runs a long multi-seat council but the room has ${seats} seat${seats === 1 ? '' : 's'}`,
    })
  }
  return null
}

// Convenience: run every configuration invariant and return the first violation
// (or null). Order: seating before effort, since a zero-seat room is the more
// fundamental problem.
export function validateRoomConfiguration(
  seatCount: number,
  mode: RoomCollaborationMode,
  effort: RoomDeliberationEffort,
): RoomStateError | null {
  return validateRoomSeating(seatCount, mode) ?? validateRoomEffortCompatibility(seatCount, effort)
}
