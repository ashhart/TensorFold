// Remote ticket execution (VST-018): a kanban ticket assigned to the project's remote
// Orchestrator runs as a thread ON THAT HOST; the local ticket tracks the remote thread's
// lifecycle. Two pure pieces:
//  1) buildRemoteTicketPrompt — the worker prompt sent to the remote thread. Mirrors the
//     local worker's ticketWorkPrompt (electron/agent/tickets.ts) — the renderer cannot
//     import that runtime module across the boundary, so a parity test pins the two.
//  2) remoteTicketLifecycle — projects a mirrored remote thread's status onto the ticket
//     (same settle semantics as the local worker: running → in_progress; clean finish →
//     review awaiting verdict; error → needs_clarification; unknown/parked → unchanged).

import type { Thread, Ticket, TicketExecutionRecord, TicketStatus } from './types'

/**
 * A fresh remote execution record at assignment time. Renderer-side twin of the local
 * worker's startTicketExecution (electron/agent/tickets.ts — runtime code cannot cross
 * the boundary); the parity test pins the two shapes together.
 */
export function beginRemoteTicketExecution(input: { hostId: string; threadId: string; startedAt: number }): TicketExecutionRecord {
  return {
    target: 'remote',
    hostId: input.hostId,
    providerId: null,
    model: null,
    branch: null,
    threadId: input.threadId,
    startedAt: input.startedAt,
    finishedAt: null,
    checks: [],
    blockers: [],
    reviewVerdict: null,
  }
}

export function buildRemoteTicketPrompt(ticket: Ticket, phase: 'build' | 'review' = 'build'): string {
  const phaseInstruction = phase === 'review'
    ? 'Adversarial review pass. Inspect the ticket and any completed work. Find concrete blockers, missing tests, unsafe assumptions, and either list what must change or say it is ready.'
    : 'Build pass. Implement this ticket in the workspace and keep the result grounded in the project files.'
  const allowedCapabilities = (ticket.allowedCapabilities?.length ? ticket.allowedCapabilities : ['read'])
  const requiredEvidence = ticket.requiredEvidence ?? []
  const acceptanceCriteria = ticket.acceptanceCriteria ?? []
  return [
    phaseInstruction,
    '',
    'Execution policy:',
    `- Risk: ${ticket.risk ?? 'low'}`,
    `- Allowed capabilities: ${allowedCapabilities.join(', ')}`,
    requiredEvidence.length ? `- Required evidence:\n${requiredEvidence.map((item) => `  - ${item}`).join('\n')}` : '- Required evidence: none specified',
    acceptanceCriteria.length ? `- Acceptance criteria:\n${acceptanceCriteria.map((item) => `  - ${item}`).join('\n')}` : '- Acceptance criteria: none specified',
    '',
    'Ticket:',
    ticket.body ? `${ticket.title}\n\n${ticket.body}` : ticket.title,
  ].join('\n')
}

export interface RemoteTicketPatch {
  status?: TicketStatus
  runState?: Ticket['runState']
  execution?: Ticket['execution']
}

/**
 * Mark an in-flight remote execution as explicitly cancelled. Cancellation is not a
 * successful run result: it must never be inferred as review-ready merely because the
 * remote thread returns to idle with earlier assistant output still in its timeline.
 */
export function cancelRemoteTicketExecution(ticket: Ticket, now = Date.now()): RemoteTicketPatch | null {
  const execution = ticket.execution
  if (!execution || execution.target !== 'remote') return null
  if (ticket.runState === 'done' || ticket.runState === 'cancelled') return null
  return {
    status: 'needs_clarification',
    runState: 'cancelled',
    execution: {
      ...execution,
      finishedAt: now,
      blockers: [...execution.blockers.filter((blocker) => blocker !== 'remote run cancelled'), 'remote run cancelled'],
      reviewVerdict: null,
    },
  }
}

/**
 * Project a mirrored remote thread onto its ticket. Returns the ticket patch to apply,
 * or null when nothing changed (idempotent — safe to call on every thread update).
 * Only applies to tickets whose execution record points at this remote thread.
 */
export function remoteTicketLifecycle(
  ticket: Ticket,
  remoteThread: Pick<Thread, 'id' | 'status' | 'timeline'> & { orchestratorHostId?: string | null },
  now = Date.now(),
): RemoteTicketPatch | null {
  const execution = ticket.execution
  if (!execution || execution.target !== 'remote' || execution.threadId !== remoteThread.id) return null
  if (!remoteThread.orchestratorHostId || remoteThread.orchestratorHostId !== execution.hostId) return null
  if (ticket.runState === 'done' || ticket.runState === 'cancelled') return null // already settled — later mirrors are no-ops

  if (remoteThread.status === 'running') {
    if (ticket.runState === 'running' && ticket.status === 'in_progress') return null
    return { status: 'in_progress', runState: 'running' }
  }

  // Settle only when something actually ran: a freshly-created remote thread is also
  // 'idle', so idle alone must not settle the ticket. Run evidence = any non-user
  // timeline item (assistant text, tool call, error) in the mirrored thread.
  const ranEvidence = remoteThread.timeline.some((item) => item.kind !== 'user')
  if (remoteThread.status === 'error' || (remoteThread.status === 'idle' && ranEvidence)) {
    const failed = remoteThread.status === 'error' || remoteThread.timeline.some((item) => item.kind === 'error')
    return {
      status: failed ? 'needs_clarification' : 'review',
      runState: 'done',
      execution: {
        ...execution,
        finishedAt: now,
        blockers: failed ? [...execution.blockers.filter((b) => !b.startsWith('remote run')), 'remote run ended with an error'] : execution.blockers,
        reviewVerdict: failed ? null : 'pending',
      },
    }
  }

  return null // not started yet, or a transitional status — leave the ticket as-is
}
