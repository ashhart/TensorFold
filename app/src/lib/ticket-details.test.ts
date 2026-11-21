import assert from 'node:assert/strict'
import { ticketDetails, ticketDetailsPatch } from './ticket-details'
import type { Ticket } from './types'

const original = { id: 'ticket', projectId: 'project', title: 'Original', body: 'Full text\n'.repeat(100), type: 'task', priority: 'medium', effort: 'small', status: 'ready', runState: null, assignee: 'Builder', threadId: 'thread', updatedAt: 1 } as Ticket
const unchanged = structuredClone(original)
const draft = { title: '  Revised  ', body: 'Full edited text\n'.repeat(100), type: 'bug', priority: 'urgent', effort: 'large' } as const
assert.deepEqual(ticketDetailsPatch(original, original, draft), { ...draft, title: 'Revised' })
assert.deepEqual(ticketDetailsPatch(original, { ...original, updatedAt: 2 }, ticketDetails(original)), {})
assert.deepEqual(ticketDetailsPatch(original, original, { ...ticketDetails(original), body: '' }), { body: '' })
assert.deepEqual(original, unchanged, 'editing never mutates the original snapshot or lifecycle fields')
for (const field of ['title', 'body', 'type', 'priority', 'effort'] as const) {
  assert.throws(() => ticketDetailsPatch(original, { ...original, [field]: 'externally changed' }, draft), /changed in another window/)
}
for (const runState of ['queued', 'running'] as const) {
  assert.throws(() => ticketDetailsPatch(original, { ...original, runState }, draft), /Stop the active ticket/)
}
assert.throws(() => ticketDetailsPatch(original, undefined, draft), /no longer active/)
assert.throws(() => ticketDetailsPatch(original, { ...original, archivedAt: 2 }, draft), /no longer active/)
assert.throws(() => ticketDetailsPatch(original, { ...original, id: 'other' }, draft), /ownership changed/)
assert.throws(() => ticketDetailsPatch(original, { ...original, projectId: 'other-project' }, draft), /ownership changed/)
assert.throws(() => ticketDetailsPatch(original, { ...original, orchestratorHostId: 'other-host' }, draft), /ownership changed/)
assert.throws(() => ticketDetailsPatch(original, original, { ...draft, title: '   ' }), /title/)
assert.throws(() => ticketDetailsPatch(original, original, { ...draft, title: 'x'.repeat(501) }), /title/)
assert.throws(() => ticketDetailsPatch(original, original, { ...draft, body: 'x'.repeat(64_001) }), /description/)
assert.equal(ticketDetailsPatch(original, original, { ...draft, title: 'x'.repeat(500), body: 'x'.repeat(64_000) }).body?.length, 64_000)
console.log('Ticket edits preserve full text, patch only editable fields, reject stale/active/archived/cross-host records and enforce input bounds')
