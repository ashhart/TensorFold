// Run: jiti src/lib/ticket-remote.test.ts
//
// Remote ticket lifecycle (VST-018 spec test) + prompt parity with the local worker.
import assert from 'node:assert/strict'
import { beginRemoteTicketExecution, buildRemoteTicketPrompt, cancelRemoteTicketExecution, remoteTicketLifecycle } from './ticket-remote'
import { startTicketExecution, ticketWorkPrompt } from '../../electron/agent/tickets'
import type { Thread, Ticket } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('ticket-remote tests:')

function baseTicket(): Ticket {
  return {
    id: 't1', projectId: 'p1', title: 'Add caching', body: 'Cache the model list.', type: 'task',
    status: 'ready', priority: 'medium', effort: 'small', tags: [], risk: 'medium',
    allowedCapabilities: ['read', 'write'], requiredEvidence: ['tests pass'], acceptanceCriteria: ['list cached for 5m'],
    parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1,
  }
}

function remoteThread(status: Thread['status'], timeline: Thread['timeline'] = []): Pick<Thread, 'id' | 'status' | 'timeline'> & { orchestratorHostId?: string | null } {
  return { id: 'rt1', status, timeline, orchestratorHostId: 'mini' }
}

{
  // Prompt parity: the renderer's remote prompt must equal the local worker's prompt.
  // (Store tickets are main-normalized, so capabilities/evidence arrive already clean.)
  const t = baseTicket()
  assert.equal(buildRemoteTicketPrompt(t, 'build'), ticketWorkPrompt(t, 'build'))
  assert.equal(buildRemoteTicketPrompt(t, 'review'), ticketWorkPrompt(t, 'review'))
  const bare: Ticket = { ...t, body: '', risk: undefined, allowedCapabilities: undefined, requiredEvidence: undefined, acceptanceCriteria: undefined }
  assert.equal(buildRemoteTicketPrompt(bare, 'build'), ticketWorkPrompt(bare, 'build'), 'defaults match too')
  ok('remote prompt is byte-identical to the local worker prompt')
}

{
  // Record parity: the renderer's remote execution record equals the main-side builder's.
  assert.deepEqual(
    beginRemoteTicketExecution({ hostId: 'mini', threadId: 'rt1', startedAt: 500 }),
    startTicketExecution({ target: 'remote', hostId: 'mini', threadId: 'rt1', startedAt: 500 }),
  )
  ok('remote execution record matches the main-side startTicketExecution shape')
}

{
  // Full remote lifecycle: assigned → running → clean finish lands in review with a
  // pending verdict; a later mirror of the same settled thread is a no-op.
  const ticket = baseTicket()
  ticket.executionMode = 'remote'
  ticket.execution = startTicketExecution({ target: 'remote', hostId: 'mini', threadId: 'rt1', startedAt: 500 })
  ticket.status = 'in_progress'
  ticket.runState = 'queued'

  // Freshly-created remote thread (idle, empty timeline) must NOT settle the ticket.
  assert.equal(remoteTicketLifecycle(ticket, remoteThread('idle'), 1000), null, 'pre-start idle is not a finish')

  const running = remoteTicketLifecycle(ticket, remoteThread('running'), 1000)
  assert.deepEqual(running, { status: 'in_progress', runState: 'running' })
  Object.assign(ticket, running)

  const finished = remoteTicketLifecycle(ticket, remoteThread('idle', [{ id: 'a', kind: 'assistant', text: 'done', createdAt: 900 } as never]), 2000)
  assert.equal(finished?.status, 'review')
  assert.equal(finished?.runState, 'done')
  assert.equal(finished?.execution?.reviewVerdict, 'pending')
  assert.equal(finished?.execution?.finishedAt, 2000)
  Object.assign(ticket, finished)

  assert.equal(remoteTicketLifecycle(ticket, remoteThread('idle', [{ id: 'a', kind: 'assistant', text: 'done', createdAt: 900 } as never]), 3000), null, 'settled tickets ignore later mirrors')
  ok('remote lifecycle: queued → running → review(pending verdict); idempotent after settle')
}

{
  const ticket = baseTicket()
  ticket.execution = startTicketExecution({ target: 'remote', hostId: 'mini', threadId: 'rt1', startedAt: 500 })
  ticket.status = 'in_progress'
  ticket.runState = 'running'
  const cancelled = cancelRemoteTicketExecution(ticket, 1500)
  assert.equal(cancelled?.status, 'needs_clarification')
  assert.equal(cancelled?.runState, 'cancelled')
  assert.equal(cancelled?.execution?.finishedAt, 1500)
  assert.deepEqual(cancelled?.execution?.blockers, ['remote run cancelled'])
  Object.assign(ticket, cancelled)
  assert.equal(
    remoteTicketLifecycle(ticket, remoteThread('idle', [{ id: 'a', kind: 'assistant', text: 'partial output', createdAt: 900 } as never]), 2000),
    null,
    'an idle mirror after cancellation cannot become review/done',
  )
  ok('remote cancellation is explicit and cannot project as successful completion')
}

{
  // A remote failure lands in needs_clarification with the blocker named, verdict cleared.
  const ticket = baseTicket()
  ticket.execution = startTicketExecution({ target: 'remote', hostId: 'mini', threadId: 'rt1', startedAt: 500 })
  ticket.status = 'in_progress'
  ticket.runState = 'running'
  const failed = remoteTicketLifecycle(ticket, remoteThread('error', [{ id: 'e', kind: 'error', text: 'boom', createdAt: 900 } as never]), 2000)
  assert.equal(failed?.status, 'needs_clarification')
  assert.deepEqual(failed?.execution?.blockers, ['remote run ended with an error'])
  assert.equal(failed?.execution?.reviewVerdict, null)
  ok('remote failure: needs_clarification with the blocker named')
}

{
  // Guards: wrong thread, wrong host, or a non-remote ticket → never touched.
  const ticket = baseTicket()
  ticket.execution = startTicketExecution({ target: 'remote', hostId: 'mini', threadId: 'OTHER', startedAt: 500 })
  assert.equal(remoteTicketLifecycle(ticket, remoteThread('running'), 1000), null)
  ticket.execution = startTicketExecution({ target: 'remote', hostId: 'attic', threadId: 'rt1', startedAt: 500 })
  assert.equal(remoteTicketLifecycle(ticket, remoteThread('running'), 1000), null)
  ticket.execution = startTicketExecution({ target: 'solo', threadId: 'rt1', startedAt: 500 })
  assert.equal(remoteTicketLifecycle(ticket, remoteThread('running'), 1000), null)
  ok('projector only touches the ticket its execution record points at')
}

console.log(`  ${passed} passed`)
