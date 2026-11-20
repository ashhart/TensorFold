import assert from 'node:assert/strict'
import { activeRoomWorkerEffort } from './room-worker-effort'
import type { Ticket } from './types'

const ticket: Ticket = {
  id: 'ticket', projectId: 'project', title: 'Synthetic work', body: '', type: 'task', status: 'in_progress',
  priority: 'medium', effort: 'small', tags: [], parentId: null, assignee: 'Builder', threadId: 'worker',
  runState: 'running', order: 1, createdAt: 1, updatedAt: 1,
  execution: { target: 'room', threadId: 'worker', reasoningEffort: 'low', startedAt: 1, finishedAt: null,
    checks: [], blockers: [], reviewVerdict: null },
}
assert.equal(activeRoomWorkerEffort([ticket], 'worker'), 'low')
assert.equal(activeRoomWorkerEffort([ticket], 'other-thread'), undefined, 'effort is scoped to the actual thread')
assert.equal(activeRoomWorkerEffort([{ ...ticket, runState: null }], 'worker'), undefined, 'parked runs clear the override')
assert.equal(activeRoomWorkerEffort([{ ...ticket, runState: 'done' }], 'worker'), undefined, 'settled runs restore manual defaults')
assert.equal(activeRoomWorkerEffort([{ ...ticket, execution: { ...ticket.execution!, finishedAt: 2 } }], 'worker'), undefined)
assert.equal(activeRoomWorkerEffort([{ ...ticket, execution: { ...ticket.execution!, threadId: 'old-worker' } }], 'worker'), undefined)
assert.equal(activeRoomWorkerEffort([{ ...ticket, execution: { ...ticket.execution!, target: 'solo' } }], 'worker'), undefined)
assert.equal(activeRoomWorkerEffort([{ ...ticket, execution: { ...ticket.execution!, reasoningEffort: undefined } }], 'worker'), undefined)
console.log('Room worker effort display checks passed: explicit thread ownership, active lifecycle and legacy/manual fallback.')
