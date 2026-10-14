import assert from 'node:assert/strict'
import { automationRunError, interruptedMissionSettlement, readOnlyChildRunError } from './run-settlement'

assert.deepEqual(interruptedMissionSettlement(true, 'running'), {
  action: 'fail', reason: 'Mission stalled — no agent activity within the heartbeat window.',
})
assert.equal(interruptedMissionSettlement(false, 'running').action, 'pause')
assert.equal(interruptedMissionSettlement(false, 'cancelled').action, 'preserve')

assert.deepEqual(automationRunError(new Error('AbortError'), true), {
  error: 'Automation cancelled before completion.', jobOutcome: 'cancelled',
})
assert.deepEqual(automationRunError(new Error('provider down'), false), {
  error: 'provider down', jobOutcome: 'failed',
})

assert.equal(readOnlyChildRunError([], true), 'Sub-agent cancelled before completion.')
assert.equal(readOnlyChildRunError([{ id: 'err', kind: 'error', text: 'Provider failed.', createdAt: 1 }], false), 'Provider failed.')
assert.equal(readOnlyChildRunError([{ id: 'ok', kind: 'assistant', text: 'Done.', createdAt: 1 }], false), null)

console.log('runtime settlement tests passed')
