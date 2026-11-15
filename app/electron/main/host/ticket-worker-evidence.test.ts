import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { settleTicketExecution, startTicketExecution, ticketReviewGateState, ticketStateAfterRun } from '../../agent/tickets'
import { summarizeTicketWorkerOutcome } from './ticket-helpers'
import type { Thread, Ticket, TimelineItem } from '../../agent/types'

const source = readFileSync(path.resolve('electron/main/host/ticket-worker-runtime.ts'), 'utf8')
const start = source.indexOf('.finally(() => {') + '.finally(() => {'.length
const end = source.indexOf('\n    })\n  launched = true', start)
assert.ok(start > 0 && end > start, 'the production worker settlement must be present')
const settlement = source.slice(start, end)
const capture = source.match(/const timelineBaseline = ([^\n]+)/)?.[1]

const oldDiff = { id: 'old-diff', kind: 'tool', path: 'src/parser.ts', state: 'ok', diff: [{ type: 'add', text: 'const value = 1' }] } as TimelineItem
const oldTest = { id: 'old-test', kind: 'tool', command: 'npm test', output: '2 passed', state: 'ok' } as TimelineItem
const currentRead = { id: 'current-read', kind: 'tool', path: 'src/parser.ts', state: 'ok' } as TimelineItem
const currentTest = { id: 'current-test', kind: 'tool', command: 'npm test', output: '3 passed', state: 'ok' } as TimelineItem

function settle(previous: TimelineItem[], current: TimelineItem[], phase: 'build' | 'review' = 'review') {
  const t = { id: 'reused-thread', status: 'running', timeline: [...previous] } as Thread
  const timelineBaseline = capture ? new Function('thread', `return ${capture}`)(t) : previous.length
  t.timeline.push(...current)
  const ticket = {
    id: 'ticket-fixture', title: 'Inspect parser', status: 'in_progress', risk: 'high', allowedCapabilities: ['read'],
    evidence: [{ id: 'builder', kind: 'file', summary: 'Inspected src/parser.ts.', createdBy: 'Builder', createdAt: 10 }],
    execution: startTicketExecution({ target: 'room', threadId: t.id, startedAt: 20 }),
  } as Ticket
  let outcome: string | null = null
  let released = false
  const scope = {
    t, ticket, timelineBaseline, phase, model: 'fixture', room: null, job: null,
    input: { source: { seat: phase === 'review' ? 'Reviewer' : 'Builder' } }, controller: new AbortController(), aborters: new Map(),
    runLease: { release() { released = true } }, markUndrainedCues() {}, activeJob() { return null },
    appendRoomSupervisorEvent() {}, finishJob(_job: unknown, result: string) { outcome = result },
    persistThreads() {}, persistTickets() {}, emitTicket() {},
    summarizeTicketWorkerOutcome, ticketStateAfterRun, settleTicketExecution,
  }
  new Function(...Object.keys(scope), settlement)(...Object.values(scope))
  assert.ok(released, 'the real settlement still releases its run reservation')
  assert.deepEqual(t.timeline, [...previous, ...current], 'scoping evidence retains the complete thread history')
  return { ticket, outcome }
}

const clean = settle([oldDiff, oldTest], [currentRead, currentTest])
assert.equal(clean.ticket.evidence!.filter((item) => item.kind === 'diff').length, 0, 'old build diffs must not become reviewer-made changes')
assert.equal(clean.ticket.status, 'completed')
assert.equal(clean.outcome, 'completed')
assert.equal(ticketReviewGateState(clean.ticket).hasReviewOrSkipAfterBuilder, true)
assert.equal(ticketReviewGateState(clean.ticket).builderEvidenceBy, 'Builder')

const recovered = settle([{ id: 'old-error', kind: 'error', text: 'Previous run failed.' } as TimelineItem], [currentRead, currentTest])
assert.equal(recovered.ticket.status, 'completed', 'an earlier run error does not fail this review')
assert.equal(recovered.outcome, 'completed')

const noCurrentTest = settle([oldTest], [currentRead])
assert.ok(!noCurrentTest.ticket.evidence!.some((item) => item.kind === 'test'), 'old passing tests are not new verification')
assert.deepEqual(noCurrentTest.ticket.execution!.checks, [])

const changed = settle([], [{ ...oldDiff, id: 'current-diff' }, currentTest])
assert.ok(changed.ticket.evidence!.some((item) => item.kind === 'diff'), 'current reviewer changes remain recorded')
assert.equal(ticketReviewGateState(changed.ticket).hasReviewOrSkipAfterBuilder, false, 'reviewer-made changes still need independent review')

const failed = settle([], [{ id: 'current-error', kind: 'error', text: 'This review failed.' } as TimelineItem])
assert.equal(failed.ticket.status, 'needs_clarification')
assert.equal(failed.outcome, 'failed')
assert.ok(failed.ticket.evidence!.some((item) => item.kind === 'review' && item.summary.includes('blocked')))
const rebuilt = settle([oldDiff, oldTest, { id: 'old-error', kind: 'error', text: 'Previous run failed.' } as TimelineItem], [currentRead], 'build')
assert.equal(rebuilt.ticket.status, 'review', 'a clean rebuild is not failed by an older run')
assert.equal(rebuilt.outcome, 'completed')
assert.ok(!rebuilt.ticket.evidence!.some((item) => item.kind === 'test' || item.kind === 'diff'), 'a rebuild only records its own output')
if (capture) assert.ok(source.indexOf('const timelineBaseline = ') < source.indexOf('thread.timeline.push(userItem)'), 'the boundary is frozen before this run appends events')
console.log('ticket worker evidence: 6 production-settlement scenarios passed')
