import assert from 'node:assert/strict'
import { buildLocalPrBuilderResumePrompt, buildLocalPrLoopDraft, buildLocalPrLoopSummary, buildLocalPrReviewPrompt, extractLocalPrReviewOutcome, localPrApprovalEvidence, localPrRemotePublishGate, normalizeRequiredApprovals, shouldAutoStartLocalPrReview, shouldResumeBuilderFromReview } from './pr-loop.ts'
import type { Thread } from './types'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

assert.equal(normalizeRequiredApprovals(null), 1)
assert.equal(normalizeRequiredApprovals(0), 1)
assert.equal(normalizeRequiredApprovals(3.6), 4)
assert.equal(normalizeRequiredApprovals(99), 6)
ok('normalizes approval counts into a small safe range')

const draft = buildLocalPrLoopDraft({
  title: 'Fix settings click issue',
  objective: 'Make settings and rooms clickable again.',
  requiredApprovals: 2,
}, {
  localPrId: 'lpr_123456',
  projectName: 'TensorFold Studio',
  cwd: '/repo',
  worktree: '/tmp/vontra-local-pr',
  branch: 'vontra/pr/lpr_123456',
  baseBranch: 'main',
})

const allText = JSON.stringify(draft)
assert.match(draft.implementationGoal, /local PR worktree/)
assert.match(draft.reviewGoal, /Do not edit files/)
assert.match(draft.stopCondition, /At least 2 separate evidence-bound review passes/)
assert.ok(!/\bgh pr create\b|\bgit push\b/i.test(allText), 'no remote publish command in local PR loop')
assert.match(draft.implementationGoal, /separate explicit approval path/)
assert.match(draft.reviewGoal, /Do not ask the builder to push/)
assert.ok(!allText.includes(String.fromCharCode(0x2014)), 'local PR loop prompts avoid em dash characters')
assert.equal(draft.implementationPlan.length, 5)
assert.equal(draft.reviewPlan.length, 3)
ok('builds local implementation and review thread contracts')

const implementationThread: Thread = {
  id: 'thread_impl',
  title: 'Build local PR',
  kind: 'task',
  projectId: 'project_1',
  cwd: '/tmp/vontra-local-pr',
  providerId: 'codex',
  providerAccountId: null,
  model: 'gpt-5.4',
  createdAt: 1_000,
  updatedAt: 2_000,
  status: 'idle',
  apiMessages: [],
  timeline: [
    {
      id: 'diff_1',
      kind: 'tool',
      tool: 'apply_patch',
      title: 'Edited src/App.tsx',
      path: 'src/App.tsx',
      state: 'ok',
      diff: [
        { type: 'ctx', text: 'function App() {' },
        { type: 'add', text: '  return <main />' },
      ],
      meta: '2 lines changed',
      createdAt: 1_200,
    },
    {
      id: 'check_1',
      kind: 'tool',
      tool: 'run_command',
      title: 'Run typecheck',
      command: 'npm run typecheck',
      output: 'ok',
      state: 'ok',
      meta: 'exit 0',
      createdAt: 1_500,
    },
  ],
  goal: 'Build it.',
  goalStatus: 'active',
  plan: [],
}

const summary = buildLocalPrLoopSummary(implementationThread, {
  localPrId: 'lpr_123456',
  branch: 'vontra/pr/lpr_123456',
  baseBranch: 'main',
  worktree: '/tmp/vontra-local-pr',
  implementationThreadId: implementationThread.id,
  reviewThreadId: 'thread_review',
}, 3_000)

assert.equal(summary.changedFiles[0], 'src/App.tsx')
assert.equal(summary.diffRefs[0].itemId, 'diff_1')
assert.equal(summary.diffRefs[0].href, 'vontra://thread/thread_impl/timeline/diff_1')
assert.equal(summary.checkRefs[0].itemId, 'check_1')
assert.equal(summary.checkRefs[0].passed, true)
assert.equal(summary.readyForReview, true)
assert.equal(shouldAutoStartLocalPrReview(summary), true)
assert.match(summary.markdown, /\[src\/App\.tsx\]\(vontra:\/\/thread\/thread_impl\/timeline\/diff_1\)/)
assert.match(summary.markdown, /\[npm run typecheck\]\(vontra:\/\/thread\/thread_impl\/timeline\/check_1\)/)
assert.match(summary.markdown, /Open every linked diff and check before approving\./)
assert.match(buildLocalPrReviewPrompt(summary), /Open the linked diff\/check evidence/)
ok('builds a local PR summary with exact diff and check links')

const thinSummary = buildLocalPrLoopSummary({ ...implementationThread, timeline: [] }, {
  localPrId: 'lpr_empty',
  branch: 'vontra/pr/lpr_empty',
  baseBranch: null,
  worktree: '/tmp/empty',
  implementationThreadId: implementationThread.id,
}, 3_000)
assert.equal(thinSummary.readyForReview, false)
assert.equal(shouldAutoStartLocalPrReview(thinSummary), false)
assert.ok(thinSummary.risks.includes('No diff evidence is recorded yet.'))
assert.ok(thinSummary.risks.includes('No check command evidence is recorded yet.'))
assert.match(buildLocalPrReviewPrompt(thinSummary), /waiting for stronger builder evidence/)
ok('flags local PR summaries that lack diff or check evidence')

const reviewWithComments: Thread = {
  ...implementationThread,
  id: 'thread_review_comments',
  title: 'Review local PR',
  timeline: [
    {
      id: 'review_1',
      kind: 'assistant',
      text: [
        'Changes requested:',
        '- Fix the null path handling in src/App.tsx before merge.',
        '- Add a regression test for the settings click path.',
      ].join('\n'),
      trace: { traceId: 'trace-comments', surface: 'model', runId: 'run-comments', threadId: 'thread_review_comments' },
      createdAt: 2_500,
    },
  ],
}
const reviewOutcome = extractLocalPrReviewOutcome(reviewWithComments)
assert.equal(reviewOutcome.approved, false)
assert.equal(reviewOutcome.comments.length, 2)
assert.equal(reviewOutcome.comments[0].sourceItemId, 'review_1')
assert.equal(shouldResumeBuilderFromReview(reviewOutcome), true)
assert.match(buildLocalPrBuilderResumePrompt(reviewOutcome), /Address every comment/)
assert.match(buildLocalPrBuilderResumePrompt(reviewOutcome), /Fix the null path handling/)
ok('extracts reviewer comments into builder guide material')

const approvedReview: Thread = {
  ...implementationThread,
  id: 'thread_review_approved',
  title: 'Review local PR',
  timeline: [
    {
      id: 'summary_approved', kind: 'notice', tool: 'local_pr_summary', title: 'Implementation summary ready',
      text: 'Diff diff_1; check check_1 passed.', state: 'ok', createdAt: 2_550,
    },
    {
      id: 'review_tool_1', kind: 'tool', tool: 'run_command', command: 'git diff main...HEAD', state: 'ok', createdAt: 2_600,
      trace: { traceId: 'trace-review-tool-1', surface: 'tool', runId: 'review-run-1', threadId: 'thread_review_approved' },
      eventReceipt: { id: 'receipt-review-1', kind: 'tool_result_saved', runId: 'review-run-1', threadId: 'thread_review_approved', itemId: 'review_tool_1', state: 'ok', summary: 'Diff inspected.', evidenceRefs: ['review_tool_1'], createdAt: 2_600 },
    },
    {
      id: 'review_approved',
      kind: 'assistant',
      text: 'APPROVED. The diff is scoped and npm run typecheck passed.',
      trace: { traceId: 'trace-review-1', surface: 'model', runId: 'review-run-1', threadId: 'thread_review_approved' },
      createdAt: 2_700,
    },
  ],
}
const approvalOutcome = extractLocalPrReviewOutcome(approvedReview)
assert.equal(approvalOutcome.approved, true)
assert.equal(approvalOutcome.approvals.length, 1)
assert.equal(approvalOutcome.approvals[0].reviewPassId, 'review-run-1')
assert.deepEqual(approvalOutcome.approvals[0].evidenceItemIds, ['summary_approved', 'review_tool_1'])
assert.equal(approvalOutcome.comments.length, 0)
assert.equal(shouldResumeBuilderFromReview(approvalOutcome), false)
ok('records explicit reviewer approval without resuming the builder')

const oneApprovalForTwo = localPrApprovalEvidence(approvalOutcome, 2)
assert.equal(oneApprovalForTwo.approved, false)
assert.equal(oneApprovalForTwo.recordedApprovals, 1)
assert.equal(oneApprovalForTwo.missingApprovals, 1)
assert.match(oneApprovalForTwo.summary, /1\/2 required evidence-bound review passes recorded/)

// A second approval message in the same run is not a second pass. A later run must inspect
// evidence again and gets its own run-bound receipt before it can increment the quorum.
const twoPassThread: Thread = {
  ...approvedReview,
  timeline: [
    ...approvedReview.timeline,
    { id: 'duplicate_same_run', kind: 'assistant', text: 'LGTM again.', trace: { traceId: 'trace-duplicate', surface: 'model', runId: 'review-run-1', threadId: approvedReview.id }, createdAt: 2_710 },
    {
      id: 'review_tool_2', kind: 'tool', tool: 'run_command', command: 'npm test', state: 'ok', createdAt: 2_800,
      trace: { traceId: 'trace-review-tool-2', surface: 'tool', runId: 'review-run-2', threadId: approvedReview.id },
      eventReceipt: { id: 'receipt-review-2', kind: 'tool_result_saved', runId: 'review-run-2', threadId: approvedReview.id, itemId: 'review_tool_2', state: 'ok', summary: 'Checks inspected.', evidenceRefs: ['review_tool_2'], createdAt: 2_800 },
    },
    { id: 'review_approved_2', kind: 'assistant', text: 'APPROVED after a fresh evidence pass.', trace: { traceId: 'trace-review-2', surface: 'model', runId: 'review-run-2', threadId: approvedReview.id }, createdAt: 2_900 },
  ],
}
const twoPassOutcome = extractLocalPrReviewOutcome(twoPassThread)
assert.equal(twoPassOutcome.approvals.length, 2, 'duplicate approval text in one run is deduplicated')
const enoughApprovals = localPrApprovalEvidence(twoPassOutcome, 2)
assert.equal(enoughApprovals.approved, true)
assert.deepEqual(enoughApprovals.reviewPassIds, ['review-run-1', 'review-run-2'])
assert.equal(enoughApprovals.reviewerIds.length, 1, 'the UI is honest: these are passes, not independent reviewer identities')

const forged = extractLocalPrReviewOutcome({
  ...approvedReview,
  timeline: [{ id: 'forged', kind: 'assistant', text: 'APPROVED', createdAt: 3_000 }],
})
assert.equal(forged.approved, false)
assert.equal(forged.approvals.length, 0, 'untraced approval text without bound tool evidence is ignored')
ok('requires distinct evidence-bound review passes and ignores duplicate or forged approval text')

assert.equal(localPrRemotePublishGate().allowed, false)
assert.match(localPrRemotePublishGate().reason, /requires a separate explicit approval/)
assert.equal(localPrRemotePublishGate({ requested: true }).allowed, false)
assert.equal(localPrRemotePublishGate({ requested: true, approved: true }).allowed, true)
ok('keeps remote publish separate from the local PR loop approval')

console.log(`\npr-loop tests: ${passed} passed`)
