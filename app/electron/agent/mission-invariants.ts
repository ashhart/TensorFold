// Pure, tested state-transition invariants for durable missions.
//
// The single source of truth for which `MissionStatus` transitions are legal. The
// runtime lifecycle helpers in `missions.ts` already short-circuit some illegal
// moves (terminal statuses are absorbing); this module makes the full rule set
// explicit and testable, and lets transition points guard with a typed error
// (`MissionInvariantError`) instead of silently no-op'ing or proceeding.
//
// Returns `null` for a legal transition so callers can stay on the happy path.

import { MissionInvariantError } from './orchestration-errors'
import type { MissionStatus } from './types'

const TRANSITION_COMMAND = 'mission:transition'

// Every status in the real union, so the adjacency map below is exhaustive and a
// future status addition surfaces as a compile error here (see EXHAUSTIVE check).
export const MISSION_STATUSES: readonly MissionStatus[] = [
  'draft',
  'queued',
  'running',
  'waiting_approval',
  'paused',
  'stalled',
  'completed',
  'failed',
  'cancelled',
]

export const TERMINAL_MISSION_STATUS_SET: ReadonlySet<MissionStatus> = new Set<MissionStatus>([
  'completed',
  'failed',
  'cancelled',
])

export const ACTIVE_MISSION_STATUS_SET: ReadonlySet<MissionStatus> = new Set<MissionStatus>([
  'queued',
  'running',
  'waiting_approval',
  'paused',
  'stalled',
])

// Allowed forward transitions per source status. A status may always transition to
// itself (idempotent re-emit of the same status is not a violation), so self-edges
// are added implicitly by `isLegalMissionTransition` and omitted here for clarity.
// Mirrors the real lifecycle in `missions.ts`:
//   createMission        -> queued
//   startMission         queued|paused|stalled|waiting_approval|draft -> running
//   pauseMission         (non-terminal) -> paused
//   requestMissionApproval (non-terminal) -> waiting_approval
//   resolveMissionApproval waiting_approval|stalled -> running (or stays waiting)
//   evaluateMissionHeartbeat running|waiting_approval -> stalled
//   recordMissionHeartbeat stalled -> running
//   finishMission        (non-terminal) -> completed|failed|cancelled
const TERMINALS: readonly MissionStatus[] = ['completed', 'failed', 'cancelled']

const MISSION_TRANSITIONS: Record<MissionStatus, readonly MissionStatus[]> = {
  draft: ['queued', 'running', 'paused', ...TERMINALS],
  queued: ['running', 'paused', 'waiting_approval', ...TERMINALS],
  running: ['waiting_approval', 'paused', 'stalled', ...TERMINALS],
  waiting_approval: ['running', 'paused', 'stalled', ...TERMINALS],
  paused: ['queued', 'running', ...TERMINALS],
  stalled: ['running', 'waiting_approval', 'paused', ...TERMINALS],
  // Terminal statuses are absorbing — no outgoing transitions (self-edge allowed).
  completed: [],
  failed: [],
  cancelled: [],
}

// Compile-time exhaustiveness guard: if `MissionStatus` gains a member that is not
// a key of MISSION_TRANSITIONS, this assignment fails to typecheck.
const EXHAUSTIVE: Record<MissionStatus, readonly MissionStatus[]> = MISSION_TRANSITIONS
void EXHAUSTIVE

export function isTerminalMissionStatus(status: MissionStatus): boolean {
  return TERMINAL_MISSION_STATUS_SET.has(status)
}

export function isActiveMissionStatus(status: MissionStatus): boolean {
  return ACTIVE_MISSION_STATUS_SET.has(status)
}

export function isLegalMissionTransition(from: MissionStatus, to: MissionStatus): boolean {
  if (from === to) return true // idempotent re-emit of the same status
  return MISSION_TRANSITIONS[from]?.includes(to) ?? false
}

// Validate a mission status transition. Returns a typed MissionInvariantError when
// the move is illegal, or null when it is legal (including the no-op self-edge).
export function validateMissionTransition(from: MissionStatus, to: MissionStatus): MissionInvariantError | null {
  if (isLegalMissionTransition(from, to)) return null

  const reason = isTerminalMissionStatus(from)
    ? `mission is already in terminal status "${from}" and cannot transition to "${to}"`
    : `mission cannot transition from "${from}" to "${to}"`

  return new MissionInvariantError({
    command: TRANSITION_COMMAND,
    issue: `Illegal mission transition ${from} → ${to}`,
    detail: reason,
  })
}

// Convenience for the supervisor: the abstract step action it decides
// ('continue' | 'complete' | 'pause') maps to a concrete target MissionStatus.
export type MissionStepAction = 'continue' | 'complete' | 'pause'

export function missionStatusForStepAction(action: MissionStepAction): MissionStatus {
  switch (action) {
    case 'complete':
      return 'completed'
    case 'pause':
      return 'paused'
    case 'continue':
    default:
      return 'running'
  }
}

// Validate that the supervisor's next-step decision is a legal transition from the
// mission's current status. Returns null on the happy path.
export function validateMissionStep(from: MissionStatus, action: MissionStepAction): MissionInvariantError | null {
  return validateMissionTransition(from, missionStatusForStepAction(action))
}
