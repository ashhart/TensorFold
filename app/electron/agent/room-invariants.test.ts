// Run: jiti electron/agent/room-invariants.test.ts
// Pure invariants for multi-agent rooms: state transitions, seating, effort fit.
import assert from 'node:assert/strict'
import {
  isLegalRoomTransition,
  isStartableRoomState,
  isTerminalRoomState,
  minSeatsForCollaborationMode,
  validateRoomConfiguration,
  validateRoomEffortCompatibility,
  validateRoomSeating,
  validateRoomTransition,
} from './room-invariants'
import { OrchestrationError, RoomStateError } from './orchestration-errors'
import type { RoomState } from './types'

let passed = 0
const ok = (msg: string) => { passed++; console.log(`ok - ${msg}`) }

// 1: legal run-lifecycle transitions return null.
{
  const legal: Array<[RoomState, RoomState]> = [
    ['queued', 'running'],
    ['ready', 'running'],
    ['created', 'running'],
    ['idle', 'running'],
    ['running', 'complete'],
    ['running', 'waiting_user'],
    ['running', 'waiting_external'],
    ['running', 'stopped'],
    ['running', 'failed'],
    ['waiting_user', 'running'],
    ['error', 'running'],
  ]
  for (const [from, to] of legal) {
    assert.equal(validateRoomTransition(from, to), null, `${from} -> ${to} should be legal`)
    assert.equal(isLegalRoomTransition(from, to), true)
  }
  ok('legal room transitions return null')
}

// 2: self-transition and archival-from-anywhere are always legal.
{
  const states: RoomState[] = ['idle', 'running', 'complete', 'failed', 'stopped', 'waiting_user', 'error']
  for (const s of states) {
    assert.equal(validateRoomTransition(s, s), null, `${s} -> ${s} should be a no-op`)
    assert.equal(validateRoomTransition(s, 'archived'), null, `${s} -> archived should be allowed`)
  }
  ok('self-transition and archive-from-any-state are legal')
}

// 3: illegal transition returns a typed RoomStateError with structured fields.
{
  const err = validateRoomTransition('complete', 'running')
  // complete -> running IS allowed (re-open path), so pick a truly illegal one:
  assert.equal(err, null)
  const bad = validateRoomTransition('archived', 'running')
  assert.ok(bad instanceof RoomStateError)
  assert.ok(bad instanceof OrchestrationError)
  assert.equal(bad!.command, 'room:state')
  assert.match(bad!.issue, /archived → running/)
  assert.deepEqual(bad!.toFields(), { command: 'room:state', issue: bad!.issue, detail: bad!.detail })
  ok('illegal room transition returns a typed RoomStateError')
}

// 4: terminal/startable classification helpers.
{
  assert.equal(isTerminalRoomState('complete'), true)
  assert.equal(isTerminalRoomState('archived'), true)
  assert.equal(isTerminalRoomState('running'), false)
  assert.equal(isStartableRoomState('queued'), true)
  assert.equal(isStartableRoomState('ready'), true)
  assert.equal(isStartableRoomState('running'), false)
  ok('terminal/startable classification helpers behave')
}

// 5: seating invariant — zero seats never runs; council modes need >= 2.
{
  // Zero seats is always a violation, regardless of mode.
  const none = validateRoomSeating(0, 'balanced')
  assert.ok(none instanceof RoomStateError)
  assert.equal(none!.command, 'room:seating')
  assert.match(none!.issue, /no seats/i)

  // Single seat is fine for balanced / local_private.
  assert.equal(validateRoomSeating(1, 'balanced'), null)
  assert.equal(validateRoomSeating(1, 'local_private'), null)
  assert.equal(minSeatsForCollaborationMode('balanced'), 1)

  // Council + frontier-split need at least 2 seats.
  assert.equal(minSeatsForCollaborationMode('frontier_council'), 2)
  assert.equal(minSeatsForCollaborationMode('frontier_plans_local_builds'), 2)
  const council = validateRoomSeating(1, 'frontier_council')
  assert.ok(council instanceof RoomStateError)
  assert.match(council!.issue, /at least 2 seats/)
  assert.equal(validateRoomSeating(2, 'frontier_council'), null)
  assert.equal(validateRoomSeating(3, 'frontier_plans_local_builds'), null)

  // Non-finite / negative counts are treated as zero.
  assert.ok(validateRoomSeating(Number.NaN, 'balanced') instanceof RoomStateError)
  assert.ok(validateRoomSeating(-2, 'balanced') instanceof RoomStateError)
  ok('seating invariant enforces mode minimums and the zero-seat floor')
}

// 6: deliberation-effort compatibility — deep/max need more than one seat.
{
  assert.ok(validateRoomEffortCompatibility(1, 'deep') instanceof RoomStateError)
  assert.ok(validateRoomEffortCompatibility(1, 'max') instanceof RoomStateError)
  assert.ok(validateRoomEffortCompatibility(0, 'deep') instanceof RoomStateError)
  const err = validateRoomEffortCompatibility(1, 'max')
  assert.equal(err!.command, 'room:effort')
  assert.match(err!.issue, /more than one seat/)

  // Quick/normal are fine with a single seat.
  assert.equal(validateRoomEffortCompatibility(1, 'quick'), null)
  assert.equal(validateRoomEffortCompatibility(1, 'normal'), null)
  // deep/max with 2+ seats is fine.
  assert.equal(validateRoomEffortCompatibility(2, 'deep'), null)
  assert.equal(validateRoomEffortCompatibility(4, 'max'), null)
  ok('effort compatibility flags single-seat deep/max councils')
}

// 7: combined configuration check returns the first (most fundamental) violation.
{
  // Zero seats surfaces the seating error before the effort error.
  const zero = validateRoomConfiguration(0, 'frontier_council', 'max')
  assert.ok(zero instanceof RoomStateError)
  assert.equal(zero!.command, 'room:seating')

  // Single seat + valid single-seat mode but deep effort -> effort error.
  const effort = validateRoomConfiguration(1, 'balanced', 'deep')
  assert.ok(effort instanceof RoomStateError)
  assert.equal(effort!.command, 'room:effort')

  // A well-formed council passes everything.
  assert.equal(validateRoomConfiguration(3, 'frontier_council', 'deep'), null)
  ok('validateRoomConfiguration returns the first violation in priority order')
}

console.log(`\n${passed} room-invariant checks passed.`)
