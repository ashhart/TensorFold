// Run: jiti src/lib/workbench-verdict.test.ts
//
// Workbench done-gate (VST-017): "done means tests passed + diff reviewed" — evidence-gated,
// freshness-gated, and never silently healthy.
import assert from 'node:assert/strict'
import { workbenchDoneVerdict, canMarkDiffsReviewed, workbenchPrReviewState } from './workbench-verdict'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('workbench-verdict tests:')

function receipt(verdict: string, diffCount = 2, changedFiles = ['a.ts']) {
  return { verdict, diffCount, changedFiles } as never
}

{
  // The full path to done: verified checks + a fresh review mark.
  const notReviewed = workbenchDoneVerdict({ surface: 'code', receipt: receipt('verified'), receiptFinishedAt: 1000 })
  assert.equal(notReviewed.applicable, true)
  assert.equal(notReviewed.testsPassed, true)
  assert.equal(notReviewed.done, false, 'passing tests alone are NOT done')
  assert.deepEqual(notReviewed.missing, ['diff review (approve in the review loop, or mark the diffs reviewed)'])
  assert.ok(canMarkDiffsReviewed(notReviewed))

  const done = workbenchDoneVerdict({ surface: 'code', receipt: receipt('verified'), receiptFinishedAt: 1000, diffReviewedAt: 1500 })
  assert.equal(done.done, true)
  assert.match(done.detail, /^Done: checks passed and the diff was reviewed/)
  assert.equal(canMarkDiffsReviewed(done), false)
  ok('done requires BOTH: verified checks and a diff review')
}

{
  // Freshness: a review mark older than the run it should cover does not count.
  const stale = workbenchDoneVerdict({ surface: 'code', receipt: receipt('verified'), receiptFinishedAt: 2000, diffReviewedAt: 1500 })
  assert.equal(stale.diffReviewed, false, 'a mark from before the run finished is stale')
  assert.equal(stale.done, false)
  ok('a new run invalidates an old "reviewed" mark')
}

{
  // Failed verification (spec test): unverified changes are not done, with the fix named.
  const unverified = workbenchDoneVerdict({ surface: 'code', receipt: receipt('unverified'), receiptFinishedAt: 1000, diffReviewedAt: 1500 })
  assert.equal(unverified.testsPassed, false)
  assert.equal(unverified.done, false)
  assert.match(unverified.missing[0], /fresh passing checks/)

  const failed = workbenchDoneVerdict({ surface: 'code', receipt: receipt('failed'), receiptFinishedAt: 1000 })
  assert.equal(failed.done, false)
  assert.deepEqual(failed.missing, ['a clean run (last run failed)'])
  const blocked = workbenchDoneVerdict({ surface: 'code', receipt: receipt('blocked'), receiptFinishedAt: 1000 })
  assert.match(blocked.detail, /blocked \(approval pending or cancelled\)/)
  ok('failed verification / failed / blocked runs are never done, and say why')
}

{
  // Review-loop signal: a PR-loop approval satisfies the review half without a manual mark.
  const viaReview = workbenchDoneVerdict({ surface: 'ticket', receipt: receipt('verified'), receiptFinishedAt: 1000, reviewApproved: true })
  assert.equal(viaReview.done, true)
  assert.match(viaReview.detail, /review approved/)
  ok('a review-loop approval satisfies the diff-review half')
}

{
  const awaiting = workbenchDoneVerdict({
    surface: 'code', receipt: receipt('verified'), receiptFinishedAt: 1000,
    diffReviewedAt: 1500, reviewRequired: true, reviewApproved: false,
  })
  assert.equal(awaiting.done, false, 'a manual review mark cannot bypass required PR approval')
  assert.deepEqual(awaiting.missing, ['required PR review approval'])
  assert.equal(canMarkDiffsReviewed(awaiting), false, 'the UI must not offer an ineffective manual mark')

  const jobs: any[] = [{
    kind: 'pr_loop', updatedAt: 10,
    metadata: {
      implementationThreadId: 'thread_impl', requiredApprovals: 2,
      localPrApprovalEvidence: { approved: false, requiredApprovals: 2, recordedApprovals: 1, missingApprovals: 1 },
    },
  }]
  const review = workbenchPrReviewState('thread_impl', jobs)
  assert.deepEqual(review, { required: true, approved: false, requiredApprovals: 2, recordedApprovals: 1 })
  const approvedJobs: any[] = [{ ...jobs[0], updatedAt: 20, metadata: { ...jobs[0].metadata, localPrApprovalEvidence: { approved: true, requiredApprovals: 2, recordedApprovals: 2, missingApprovals: 0 } } }]
  assert.equal(workbenchPrReviewState('thread_impl', approvedJobs).approved, true)
  assert.deepEqual(workbenchPrReviewState('other', approvedJobs), { required: false, approved: false, requiredApprovals: 0, recordedApprovals: 0 })
  ok('local PR job metadata wires required reviewer counts into the Workbench gate')
}

{
  // No changes → done, but the detail SAYS it is vacuous (never silently healthy).
  const noChanges = workbenchDoneVerdict({ surface: 'code', receipt: receipt('not_applicable', 0, []), receiptFinishedAt: 1000 })
  assert.equal(noChanges.done, true)
  assert.match(noChanges.detail, /no code changes recorded/)
  // Changes WITHOUT verify-on-stop evidence ('not_applicable' + diffs) are NOT done.
  const unproven = workbenchDoneVerdict({ surface: 'code', receipt: receipt('not_applicable'), receiptFinishedAt: 1000, diffReviewedAt: 1500 })
  assert.equal(unproven.done, false, 'changes without check evidence are not done')
  ok('vacuously done is stated; changes without evidence never are')
}

{
  const chat = workbenchDoneVerdict({ surface: 'chat', receipt: receipt('verified'), receiptFinishedAt: 1000 })
  assert.equal(chat.applicable, false)
  assert.equal(canMarkDiffsReviewed(chat), false)
  ok('the gate does not apply to chat/assistant surfaces')
}

console.log(`  ${passed} passed`)
