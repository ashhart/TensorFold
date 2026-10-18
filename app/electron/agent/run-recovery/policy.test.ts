import assert from 'node:assert/strict'
import { runRetryVerdict, interruptedRunRecovery } from '../../../shared/run-recovery'
import { streamRetryDecision } from '../loop'
import { failedRequestDelivery } from './policy'
import { RunRecoveryTracker } from './tracker'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

const base = { compactRetries: 0, canCompact: true, random: () => 0 }
for (const status of [429, 503, 529]) {
  const error = Object.assign(new Error(`HTTP ${status}`), { status })
  const delivery = failedRequestDelivery(error, false)
  assert.equal(delivery, 'refused')
  assert.equal(streamRetryDecision(error, 1, { ...base, acceptance: { delivery } }).kind, 'retry')
  assert.equal(streamRetryDecision(error, 4, { ...base, acceptance: { delivery } }).kind, 'fail', 'Even known refusal retries remain capped')
}
for (const error of [Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' }), Object.assign(new Error('HTTP 504'), { status: 504 }), Object.assign(new Error('HTTP 500'), { status: 500 }), new Error('fetch failed')]) {
  const delivery = failedRequestDelivery(error, false)
  assert.equal(delivery, 'unknown')
  assert.equal(streamRetryDecision(error, 1, { ...base, acceptance: { delivery } }).kind, 'fail')
  assert.equal(streamRetryDecision(error, 1, { ...base, acceptance: { delivery: 'partial', visibleOutput: true } }).kind, 'fail')
}
for (const code of ['ECONNREFUSED', 'ENOTFOUND', 'UND_ERR_CONNECT_TIMEOUT']) {
  const error = Object.assign(new Error('connect failed'), { cause: { code } })
  assert.equal(failedRequestDelivery(error, false), 'not_sent')
  assert.equal(streamRetryDecision(error, 1, { ...base, acceptance: { delivery: 'not_sent' } }).kind, 'retry')
}
assert.equal(runRetryVerdict({ delivery: 'refused', uncertainEffects: true }).allowed, false)
assert.equal(runRetryVerdict({ delivery: 'not_sent', delegated: true }).allowed, false)
const context = Object.assign(new Error('context_length_exceeded'), { status: 413 })
assert.equal(streamRetryDecision(context, 1, { ...base, acceptance: { delivery: failedRequestDelivery(context, false) } }).kind, 'compact')
assert.equal(streamRetryDecision(context, 1, { ...base, acceptance: { delivery: 'partial' } }).kind, 'fail')
assert.equal(streamRetryDecision(context, 1, { ...base, compactRetries: 2, acceptance: { delivery: 'refused' } }).kind, 'fail')
const snapshots: RunRecoverySnapshot[] = []
const tracker = new RunRecoveryTracker('synthetic', snapshot => snapshots.push(snapshot), () => 123)
tracker.start(); tracker.begin('mutation', true); tracker.requestComplete(); tracker.stopRequested()
assert.equal(snapshots.at(-1)?.canStartNewRun, false)
assert.throws(() => tracker.begin('new'), /after Stop/)
assert.throws(() => tracker.settle(true), /still active/)
tracker.toolCompleted(true, false); tracker.end('mutation'); const outcome = tracker.settle(false)
assert.equal(outcome.state, 'needs_review'); assert.equal(outcome.reason, 'tool_uncertain'); assert.equal(outcome.retry, 'requires_review')
const interrupted = interruptedRunRecovery({ ...outcome, state: 'stopping', pendingOperations: 1, pendingMutations: 1 })
assert.equal(interrupted.reason, 'restart_interrupted'); assert.equal(interrupted.retry, 'requires_review'); assert.equal(interrupted.state, 'needs_review')
const observer = new RunRecoveryTracker('safe-observer', () => { throw new Error('Observer must not affect settlement') })
observer.start(); observer.requestComplete(); assert.equal(observer.settle(true).state, 'completed')
console.log('PASS run recovery policy: bounded refusals, pre-dispatch safe retry, partial/unknown/delegated replay denial, compaction cap, mutation drain, uncertain effects, restart and observer isolation')
