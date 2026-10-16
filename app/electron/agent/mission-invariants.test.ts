// Run: jiti electron/agent/mission-invariants.test.ts
// Pure state-transition invariants for durable missions.
import assert from 'node:assert/strict'
import {
  ACTIVE_MISSION_STATUS_SET,
  TERMINAL_MISSION_STATUS_SET,
  isActiveMissionStatus,
  isLegalMissionTransition,
  isTerminalMissionStatus,
  missionStatusForStepAction,
  validateMissionStep,
  validateMissionTransition,
} from './mission-invariants'
import { MissionInvariantError, OrchestrationError } from './orchestration-errors'
import type { MissionStatus } from './types'

let passed = 0
const ok = (msg: string) => { passed++; console.log(`ok - ${msg}`) }

// 1: legal lifecycle transitions return null (happy path unchanged).
{
  const legal: Array<[MissionStatus, MissionStatus]> = [
    ['queued', 'running'],
    ['running', 'waiting_approval'],
    ['running', 'paused'],
    ['running', 'stalled'],
    ['running', 'completed'],
    ['running', 'failed'],
    ['running', 'cancelled'],
    ['waiting_approval', 'running'],
    ['stalled', 'running'],
    ['paused', 'running'],
    ['paused', 'queued'],
    ['draft', 'queued'],
  ]
  for (const [from, to] of legal) {
    assert.equal(validateMissionTransition(from, to), null, `${from} -> ${to} should be legal`)
    assert.equal(isLegalMissionTransition(from, to), true)
  }
  ok('legal mission transitions return null')
}

// 2: idempotent self-transition is always legal (re-emit of same status).
{
  for (const status of [...ACTIVE_MISSION_STATUS_SET, ...TERMINAL_MISSION_STATUS_SET] as MissionStatus[]) {
    assert.equal(validateMissionTransition(status, status), null, `${status} -> ${status} should be a no-op`)
  }
  ok('self-transition is always legal')
}

// 3: illegal transitions return a typed MissionInvariantError with structured fields.
{
  const err = validateMissionTransition('queued', 'stalled')
  assert.ok(err instanceof MissionInvariantError)
  assert.ok(err instanceof OrchestrationError)
  assert.ok(err instanceof Error)
  assert.equal(err!.command, 'mission:transition')
  assert.match(err!.issue, /queued → stalled/)
  assert.deepEqual(err!.toFields(), { command: 'mission:transition', issue: err!.issue, detail: err!.detail })
  ok('illegal transition returns a typed MissionInvariantError with fields')
}

// 4: terminal states are absorbing — no outgoing transition is legal.
{
  const terminals: MissionStatus[] = ['completed', 'failed', 'cancelled']
  const targets: MissionStatus[] = ['running', 'queued', 'paused', 'waiting_approval', 'stalled']
  for (const from of terminals) {
    assert.equal(isTerminalMissionStatus(from), true)
    for (const to of targets) {
      const err = validateMissionTransition(from, to)
      assert.ok(err instanceof MissionInvariantError, `${from} -> ${to} should be illegal`)
      assert.match(err!.detail, /terminal status/)
    }
    // Cross-terminal moves are also illegal (cannot complete a cancelled mission).
    for (const to of terminals) {
      if (to === from) continue
      assert.ok(validateMissionTransition(from, to) instanceof MissionInvariantError, `${from} -> ${to} should be illegal`)
    }
  }
  ok('terminal mission states reject all outgoing transitions')
}

// 5: active/terminal classification helpers.
{
  assert.equal(isActiveMissionStatus('running'), true)
  assert.equal(isActiveMissionStatus('queued'), true)
  assert.equal(isActiveMissionStatus('completed'), false)
  assert.equal(isTerminalMissionStatus('cancelled'), true)
  assert.equal(isTerminalMissionStatus('running'), false)
  // 'draft' is neither active nor terminal.
  assert.equal(isActiveMissionStatus('draft'), false)
  assert.equal(isTerminalMissionStatus('draft'), false)
  ok('active/terminal classification helpers agree with the union')
}

// 6: step-action mapping + validateMissionStep guard.
{
  assert.equal(missionStatusForStepAction('complete'), 'completed')
  assert.equal(missionStatusForStepAction('pause'), 'paused')
  assert.equal(missionStatusForStepAction('continue'), 'running')

  // From a running mission every step action is legal.
  assert.equal(validateMissionStep('running', 'continue'), null)
  assert.equal(validateMissionStep('running', 'complete'), null)
  assert.equal(validateMissionStep('running', 'pause'), null)

  // From a completed mission, resuming ('continue' -> running) or pausing
  // ('pause' -> paused) is illegal — the supervisor must not reactivate a terminal
  // mission. Re-asserting completion ('complete' -> completed) is the idempotent
  // self-edge and is allowed (harmless no-op).
  assert.ok(validateMissionStep('completed', 'continue') instanceof MissionInvariantError, 'continue from completed should be illegal')
  assert.ok(validateMissionStep('completed', 'pause') instanceof MissionInvariantError, 'pause from completed should be illegal')
  assert.equal(validateMissionStep('completed', 'complete'), null, 're-completing a completed mission is a no-op')

  // From cancelled/failed, completing is also illegal (cannot complete a cancelled
  // mission — that is a cross-terminal move, not a self-edge).
  assert.ok(validateMissionStep('cancelled', 'complete') instanceof MissionInvariantError, 'complete from cancelled should be illegal')
  assert.ok(validateMissionStep('failed', 'continue') instanceof MissionInvariantError, 'continue from failed should be illegal')
  ok('validateMissionStep maps actions and guards terminal missions')
}

console.log(`\n${passed} mission-invariant checks passed.`)
