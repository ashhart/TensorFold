import assert from 'node:assert/strict'
import { summarizeTicketWorkerOutcome } from '../main/host/ticket-helpers'
import { roomFinalGateBlockers, ticketNeedsReviewAfterBuilder, ticketReviewGateState } from './tickets'
import type { Thread, Ticket, TicketEvidence } from './types'

const build: TicketEvidence = { id: 'build', kind: 'file', summary: 'Inspected src/parser.ts.', createdBy: 'Builder', createdAt: 10 }
const approval: TicketEvidence = { id: 'approval', kind: 'review', summary: 'Reviewer approved.', createdBy: 'Reviewer', createdAt: 20 }
const rebuild: TicketEvidence = { ...build, id: 'rebuild', summary: 'Inspected the follow-up implementation.', createdAt: 30 }
const waiver: TicketEvidence = { ...approval, id: 'waiver', kind: 'note', summary: 'Review skipped: read-only source lookup, no code changes.' }

function ticket(evidence: TicketEvidence[]): Ticket {
  return {
    id: 'ticket-fixture', projectId: 'project-fixture', roomId: 'room-fixture', title: 'Follow-up implementation', body: '',
    type: 'task', status: 'completed', priority: 'medium', effort: 'medium', tags: [], assignee: 'Builder',
    parentId: null, threadId: null, runState: null, startedAt: null, order: 0, createdAt: 1, updatedAt: 90,
    risk: 'high', allowedCapabilities: ['read'], requiredEvidence: ['Findings'], acceptanceCriteria: [], evidence,
  }
}

let passed = 0
function check(name: string, evidence: TicketEvidence[], allowed: boolean, builder = 'Builder') {
  const current = ticket(evidence)
  assert.equal(ticketReviewGateState(current).hasReviewOrSkipAfterBuilder, allowed, name)
  assert.equal(ticketReviewGateState(current).builderEvidenceBy, builder, name)
  assert.equal(ticketNeedsReviewAfterBuilder(current), !allowed, name)
  assert.deepEqual(roomFinalGateBlockers({ tickets: [current], greenEvidenceAt: 100 }), allowed ? []
    : ['review evidence after builder or justified skip missing on: Follow-up implementation.'], name)
  passed++
}

function worker(phase: 'build' | 'review', at: number, createdBy: string | null, changes = false): TicketEvidence[] {
  const now = Date.now
  Date.now = () => at
  try {
    const thread = {
      id: `${phase}-thread-${at}`,
      timeline: [
        { id: 'read', kind: 'tool', path: 'src/parser.ts', state: 'ok' },
        ...(changes ? [{ id: 'edit', kind: 'tool', path: 'src/parser.ts', diff: [{ type: 'add', text: 'const value = 1' }], state: 'ok' }] : []),
        { id: 'status', kind: 'tool', command: 'git status --short', output: '', state: 'ok' },
        { id: 'test', kind: 'tool', command: 'npm test', output: '2 passed', state: 'ok' },
      ],
    } as unknown as Thread
    return summarizeTicketWorkerOutcome(ticket([]), thread, phase, false, createdBy).evidence
  } finally { Date.now = now }
}

check('an approval does not cover later builder findings', [build, approval, rebuild], false)
check('timestamp order wins over insertion order', [rebuild, build, approval], false)
check('a fresh independent approval covers the rebuild', [build, approval, rebuild, { ...approval, id: 'fresh', createdAt: 40 }], true)
check('a waiver does not cover later builder findings', [build, waiver, rebuild], false)
check('a fresh justified waiver covers read-only work', [build, waiver, rebuild, { ...waiver, id: 'fresh-waiver', createdAt: 40 }], true)
check('a new builder is reported as the current builder', [build, approval, { ...rebuild, createdBy: 'Second builder' }], false, 'Second builder')
check('a review cannot cover a build at the same timestamp', [build, approval, { ...rebuild, createdAt: 20 }], false)

const reviewBlocker: TicketEvidence = {
  id: 'review-blocker', kind: 'note', summary: 'Blocker: parser validation failed.',
  createdBy: 'Reviewer', createdAt: 20, ref: 'thread:review-fixture',
}
const blockedBatch = [{ ...approval, ref: reviewBlocker.ref }, reviewBlocker]
check('a reviewer blocker preserves the original builder identity', [build, ...blockedBatch], false)
check('reviewer blocker batches are independent of insertion order', [build, ...[...blockedBatch].reverse()], false)
check('a fresh approval resolves the reviewer blocker without a new build', [build, ...blockedBatch, {
  ...approval, id: 'resolved', createdAt: 30,
}], true)
check('an unrelated note cannot inherit a review blocker batch', [build, ...blockedBatch.slice(0, 1), {
  ...reviewBlocker, ref: 'thread:other-review',
}], true)
check('another actor cannot attach a review blocker batch', [build, blockedBatch[0], {
  ...reviewBlocker, createdBy: 'Another actor',
}], true)
check('another timestamp cannot attach a review blocker batch', [build, blockedBatch[0], {
  ...reviewBlocker, createdAt: 21,
}], true)
check('a blocker describing changed code is still a review finding', [build, blockedBatch[0], {
  ...reviewBlocker, summary: 'Blocker: changed parser behavior fails validation.',
}], false)
check('an explicit empty blocker does not reject a review', [build, blockedBatch[0], {
  ...reviewBlocker, summary: 'Blocker: none',
}], true)
check('a later builder run still invalidates approval without file events', [build, approval, {
  ...build, id: 'builder-run', kind: 'note', summary: 'Builder worker finished the follow-up.', createdAt: 30,
}], false)

const reviewRun = worker('review', 20, 'Reviewer')
const reviewOnly = ticket(reviewRun)
assert.deepEqual(ticketReviewGateState(reviewOnly), { hasBuilderEvidence: false, builderEvidenceBy: null, hasReviewOrSkipAfterBuilder: false })
assert.equal(ticketNeedsReviewAfterBuilder(reviewOnly), false, 'a review cannot create builder evidence for another review')
assert.deepEqual(roomFinalGateBlockers({ tickets: [reviewOnly] }), ['review evidence after builder or justified skip missing on: Follow-up implementation.'])
passed++
check('real reviewer verification is not a new build', [build, ...reviewRun], true)
check('review batch order does not alter builder ownership', [build, ...[...reviewRun].reverse()], true)
check('anonymous review batches keep the same verification behavior', [build, ...worker('review', 20, null)], true)
check('an actual later build worker invalidates the review', [build, ...reviewRun, ...worker('build', 30, 'Builder')], false)
check('a new review worker covers the actual rebuilt output', [build, ...reviewRun, ...worker('build', 30, 'Builder'), ...worker('review', 40, 'Reviewer')], true)

const testResult = reviewRun.find((item) => item.kind === 'test')!
assert.ok(testResult, 'the host fixture must produce a verification record')
check('another thread cannot inherit a review batch exemption', [build, ...reviewRun, { ...testResult, id: 'other-thread', ref: 'thread:other-run' }], false, 'Reviewer')
check('another actor cannot inherit a review batch exemption', [build, ...reviewRun, { ...testResult, id: 'other-actor', createdBy: 'Second builder' }], false, 'Second builder')
check('a missing thread reference cannot inherit a review batch exemption', [build, ...reviewRun, { ...testResult, id: 'missing-ref', ref: null }], false, 'Reviewer')
check('a later run cannot reuse an earlier batch timestamp', [build, ...reviewRun, { ...testResult, id: 'later-run', createdAt: 30 }], false, 'Reviewer')
check('code changes made during a review still need independent review', [build, ...worker('review', 20, 'Reviewer', true)], false, 'Reviewer')
check('a different reviewer can approve the reviewer-made changes', [build, ...worker('review', 20, 'Reviewer', true), { ...approval, id: 'independent', createdBy: 'Second reviewer', createdAt: 30 }], true, 'Reviewer')

console.log(`ticket review build freshness: ${passed} cases passed`)
