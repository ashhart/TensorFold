import assert from 'node:assert/strict'
import type { Ticket } from './types'
import {
  assertTicketDependencyGraph,
  normalizeTicketDependencyIds,
  requireTicketDependenciesReady,
  resolveTicketDependencies,
} from '../../shared/ticket-dependencies'

const ticket = (id: string, status: Ticket['status'] = 'todo', over: Partial<Ticket> = {}): Ticket => ({
  id,
  projectId: 'project',
  title: id,
  body: '',
  type: 'task',
  status,
  priority: 'medium',
  effort: 'small',
  tags: [],
  parentId: null,
  assignee: null,
  threadId: null,
  runState: null,
  order: 1,
  createdAt: 1,
  updatedAt: 1,
  ...over,
})

assert.deepEqual(normalizeTicketDependencyIds([' a ', 'a', '', 3, 'b']), ['a', 'b'])

const done = ticket('done', 'completed')
const active = ticket('active', 'in_progress')
const failed = ticket('failed', 'needs_clarification')
const dependent = ticket('dependent', 'todo', { dependsOn: [done.id, active.id] })
assert.deepEqual(resolveTicketDependencies(dependent, [done, active, failed, dependent]), {
  state: 'waiting', blockers: [], waitingOn: ['active'],
})

const blocked = ticket('blocked', 'ready', { dependsOn: [failed.id] })
assert.equal(resolveTicketDependencies(blocked, [failed, blocked]).state, 'blocked')
assert.equal(resolveTicketDependencies({ ...blocked, dependencyFailurePolicy: 'continue' }, [failed, blocked]).state, 'ready')
assert.throws(() => requireTicketDependenciesReady(dependent, [done, active, dependent]), /waiting on/i)

const a = ticket('a')
const b = ticket('b', 'todo', { dependsOn: ['a'] })
assert.doesNotThrow(() => assertTicketDependencyGraph([a, b], { ...a, dependsOn: [] }))
assert.throws(() => assertTicketDependencyGraph([a, b], { ...a, dependsOn: ['b'] }), /cycle/i)
assert.throws(() => assertTicketDependencyGraph([a], { ...a, dependsOn: ['a'] }), /itself/i)
assert.throws(() => assertTicketDependencyGraph([a], { ...a, dependsOn: ['missing'] }), /not found/i)

console.log('ticket dependency DAG validation and readiness tests passed')
