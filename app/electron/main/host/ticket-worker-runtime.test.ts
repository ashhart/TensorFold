import assert from 'node:assert/strict'
import type { Ticket } from '../../agent/types'
import { ticketStateAfterRun } from '../../agent/tickets'
import { assertActiveTicketWorkOwnership, awaitTicketStartupStep } from './ticket-worker-runtime'

// The parking decision under test used to live only inside the run's `.finally` block
// (a closure over ~20 injected deps), replicated here by hand. VST-018 extracted it as
// ticketStateAfterRun (electron/agent/tickets.ts) — the runtime and this test now share
// the REAL implementation, so they cannot drift.
//
// Original bug context: ticket.runState was always set to 'done' and ticket.status was
// computed from `failed` alone — so an aborted/paused run still marked the ticket as
// 'review'/'completed' with runState 'done', contradicting the `paused` job and showing
// the kanban ticket as finished while it was parked.
const resolveParkedState = (current: Ticket, opts: { failed: boolean; interrupted: boolean; phase: 'build' | 'review' }) =>
  ticketStateAfterRun(current, opts)

function inProgressTicket(): Ticket {
  return {
    id: 't1', projectId: 'p1', title: 'x', body: '', type: 'task',
    status: 'in_progress', priority: 'medium', effort: 'medium', tags: [],
    parentId: null, assignee: null, threadId: 'th1', runState: 'running',
    order: 1, createdAt: 1, updatedAt: 1,
  }
}

// 1. Aborted/paused build run: must stay in_progress with no active runState.
{
  const r = resolveParkedState(inProgressTicket(), { failed: false, interrupted: true, phase: 'build' })
  assert.equal(r.status, 'in_progress', 'aborted run must NOT mark ticket as review')
  assert.equal(r.runState, null, 'aborted run must clear runState (job is paused)')
  console.log('ok: aborted build run parks ticket as in_progress / runState=null')
}

// 2. Aborted/paused review run: must NOT be marked completed.
{
  const r = resolveParkedState(inProgressTicket(), { failed: false, interrupted: true, phase: 'review' })
  assert.equal(r.status, 'in_progress', 'aborted review run must NOT mark ticket as completed')
  assert.equal(r.runState, null)
  console.log('ok: aborted review run parks ticket without completing it')
}

// 3. Clean build completion (not interrupted) -> review / done (unchanged).
{
  const r = resolveParkedState(inProgressTicket(), { failed: false, interrupted: false, phase: 'build' })
  assert.equal(r.status, 'review')
  assert.equal(r.runState, 'done')
  console.log('ok: clean build run -> review / done')
}

// 4. Clean review completion -> completed / done (unchanged).
{
  const r = resolveParkedState(inProgressTicket(), { failed: false, interrupted: false, phase: 'review' })
  assert.equal(r.status, 'completed')
  assert.equal(r.runState, 'done')
  console.log('ok: clean review run -> completed / done')
}

// 5. Errored (not interrupted) -> needs_clarification / done (unchanged).
{
  const r = resolveParkedState(inProgressTicket(), { failed: true, interrupted: false, phase: 'build' })
  assert.equal(r.status, 'needs_clarification')
  assert.equal(r.runState, 'done')
  console.log('ok: errored run -> needs_clarification / done')
}

// 6. Interrupt takes precedence even if an error item was also recorded.
{
  const r = resolveParkedState(inProgressTicket(), { failed: true, interrupted: true, phase: 'build' })
  assert.equal(r.status, 'in_progress', 'interrupt should not be reported as needs_clarification')
  assert.equal(r.runState, null)
  console.log('ok: interrupt takes precedence over recorded error')
}

console.log('all ticket-worker-runtime parking tests passed')

{
  const project = { id: 'p1', updatedAt: 1 } as any
  const room = { id: 'r1', projectId: 'p1', updatedAt: 1 } as any
  const thread = { id: 'th1', projectId: 'p1', updatedAt: 1 } as any
  const ticket = { ...inProgressTicket(), roomId: 'r1' }
  assert.doesNotThrow(() => assertActiveTicketWorkOwnership(ticket, project, room, thread))
  ticket.archivedAt = 10
  assert.throws(() => assertActiveTicketWorkOwnership(ticket, project, room, thread), /archived ticket/, 'archive during paused startup is rejected by the final ownership check')
  ticket.archivedAt = null
  project.archivedAt = 11
  assert.throws(() => assertActiveTicketWorkOwnership(ticket, project, room, thread), /active project/, 'project archive during startup prevents launch')
  project.archivedAt = null
  room.archivedAt = 12
  assert.throws(() => assertActiveTicketWorkOwnership(ticket, project, room, thread), /active room/, 'room archive during startup prevents launch')
  room.archivedAt = null
  ticket.projectId = 'moved-project'
  assert.throws(() => assertActiveTicketWorkOwnership(ticket, project, room, thread), /active project/, 'project reassignment during startup invalidates captured ownership')
  console.log('ok: ticket startup ownership is revalidated after asynchronous preparation')
}

{
  const project = { id: 'p1', updatedAt: 1 } as any
  const ticket = { ...inProgressTicket(), threadId: null }
  let release!: () => void
  const pausedInspect = new Promise<string | null>((resolve) => { release = () => resolve('main') })
  let committed = false
  const startup = awaitTicketStartupStep(pausedInspect, () => assertActiveTicketWorkOwnership(ticket, project, null))
    .then(() => { committed = true })
  ticket.archivedAt = 99
  release()
  await assert.rejects(startup, /archived ticket/)
  assert.equal(committed, false, 'paused Git inspection cannot commit a thread/run after the ticket is archived')
  console.log('ok: deferred ticket startup step revalidates before commit')
}
