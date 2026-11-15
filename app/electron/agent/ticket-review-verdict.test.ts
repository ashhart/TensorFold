import assert from 'node:assert/strict'
import { roomFinalGateBlockers, ticketNeedsReviewAfterBuilder, ticketReviewGateState } from './tickets'
import type { Ticket, TicketEvidence } from './types'

const finding: TicketEvidence = { id: 'finding', kind: 'file', summary: 'Inspected src/parser.ts.', createdBy: 'Builder', createdAt: 10 }
const approved: TicketEvidence = { id: 'approved', kind: 'review', summary: 'Reviewer approved; no blockers.', createdBy: 'Reviewer', createdAt: 20 }
const blocked: TicketEvidence = { id: 'blocked', kind: 'review', summary: 'Changes requested: regression in empty input handling.', createdBy: 'Reviewer', createdAt: 30 }
const waived: TicketEvidence = { id: 'waived', kind: 'note', summary: 'Review skipped: read-only source lookup, no code changes.', createdBy: 'Reviewer', createdAt: 20 }

function ticket(evidence: TicketEvidence[], patch: Partial<Ticket> = {}): Ticket {
  return {
    id: 'ticket-fixture', projectId: 'project-fixture', roomId: 'room-fixture', title: 'Review the parser', body: '',
    type: 'task', status: 'completed', priority: 'medium', effort: 'medium', tags: [], assignee: 'Builder',
    parentId: null, threadId: null, runState: null, startedAt: null, order: 0, createdAt: 1, updatedAt: 40,
    risk: 'high', allowedCapabilities: ['read'], requiredEvidence: ['Findings'], acceptanceCriteria: [], evidence, ...patch,
  }
}

let passed = 0
function check(name: string, evidence: TicketEvidence[], allowed: boolean) {
  const current = ticket(evidence)
  assert.equal(ticketReviewGateState(current).hasReviewOrSkipAfterBuilder, allowed, name)
  assert.equal(ticketNeedsReviewAfterBuilder(current), !allowed, name)
  assert.deepEqual(roomFinalGateBlockers({ tickets: [current] }), allowed ? []
    : ['review evidence after builder or justified skip missing on: Review the parser.'], name)
  passed++
}

check('an independent approval completes the gate', [finding, approved], true)
check('a later blocking verdict supersedes approval', [finding, approved, blocked], false)
check('another reviewer can reopen a ticket', [finding, approved, { ...blocked, createdBy: 'Second reviewer' }], false)
check('a fresh approval resolves the blocking verdict', [finding, approved, blocked, { ...approved, id: 'fresh', createdAt: 40 }], true)
check('an older blocking verdict does not override a newer approval', [finding, blocked, { ...approved, createdAt: 40 }], true)
check('timestamps take precedence over array order', [finding, blocked, approved], false)
check('the last verdict wins at equal timestamps', [finding, approved, { ...blocked, createdAt: 20 }], false)
check('a same-timestamp follow-up approval can resolve the verdict', [finding, blocked, { ...approved, createdAt: 30 }], true)
check('informational review notes do not revoke approval', [finding, approved, { ...blocked, summary: 'Reviewer inspected the examples.' }], true)
check('user approval records are not reviewer verdicts', [finding, approved, { ...blocked, kind: 'note', summary: 'User approval rejected for a separate action.' }], true)
check('a later blocking verdict supersedes a waiver', [finding, waived, blocked], false)
check('a justified later waiver can resolve read-only review', [finding, blocked, { ...waived, createdAt: 40 }], true)
check('a later self-review cannot reuse an older independent approval', [finding, approved, { ...approved, id: 'self', createdBy: 'Builder', createdAt: 40 }], false)

const changed: TicketEvidence = { ...finding, kind: 'diff', summary: 'Changed src/parser.ts.' }
const green: TicketEvidence = { id: 'green', kind: 'test', summary: 'Parser tests passed.', createdBy: 'Builder', createdAt: 11 }
check('an invalid code-change waiver cannot bypass a blocking verdict', [changed, green, approved, blocked, { ...waived, createdAt: 40 }], false)

console.log(`ticket review verdict: ${passed} cases passed`)
