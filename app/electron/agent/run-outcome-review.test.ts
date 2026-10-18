import assert from 'node:assert/strict'
import type { Thread } from './types'
import { assertRecoveryAllowsNewRun, reviewedRunOutcome } from './run-outcome-review'
const thread = { id: 'one', runRecovery: { runId: 'run-one', canStartNewRun: true, retry: 'requires_review' } } as Thread
assert.throws(() => assertRecoveryAllowsNewRun(thread), /Review/)
assert.throws(() => reviewedRunOutcome(thread, { threadId: 'one', runId: 'old' }, false), /changed/)
assert.throws(() => reviewedRunOutcome(thread, { threadId: 'one', runId: 'run-one' }, true), /draining/)
const reviewed = reviewedRunOutcome(thread, { threadId: 'one', runId: 'run-one' }, false)
assert.doesNotThrow(() => assertRecoveryAllowsNewRun(reviewed))
assert.equal(thread.runRecoveryReviewedId, undefined, 'preview does not mutate host state')
assert.throws(() => assertRecoveryAllowsNewRun({ ...reviewed, runRecovery: { ...reviewed.runRecovery!, runId: 'next' } }), /Review/)
assert.throws(() => assertRecoveryAllowsNewRun({ ...reviewed, runRecovery: { ...reviewed.runRecovery!, canStartNewRun: false } }), /settle/)
console.log('Outcome review binds the exact settled run and cannot authorize a later run or bypass physical drain')
