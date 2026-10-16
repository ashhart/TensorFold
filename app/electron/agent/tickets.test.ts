// Run: npx jiti electron/agent/tickets.test.ts
import assert from 'node:assert/strict'
import {
  defaultRequiredEvidence,
  inferTicketRisk,
  normalizeStringList,
  normalizeTicketCapabilities,
  normalizeTicketEvidence,
  normalizeTicketExecution,
  normalizeTickets,
  roomFinalGateBlockers,
  roomLeftToWork,
  roomTicketConflictBlockers,
  summarizeRoomLeftToWork,
  ticketNeedsReviewAfterBuilder,
  ticketFileRefs,
} from './tickets'
import { settleTicketExecution, startTicketExecution, ticketStateAfterRun } from './tickets'
import type { Ticket } from './types'

function ticket(patch: Partial<Ticket>): Ticket {
  return {
    id: patch.id ?? `ticket-${Math.random().toString(36).slice(2, 8)}`,
    projectId: patch.projectId ?? 'project-1',
    roomId: patch.roomId ?? 'room-1',
    title: patch.title ?? 'Ticket',
    body: patch.body ?? '',
    type: patch.type ?? 'task',
    status: patch.status ?? 'completed',
    priority: patch.priority ?? 'medium',
    effort: patch.effort ?? 'medium',
    tags: patch.tags ?? [],
    risk: patch.risk,
    allowedCapabilities: patch.allowedCapabilities,
    requiredEvidence: patch.requiredEvidence,
    acceptanceCriteria: patch.acceptanceCriteria,
    evidence: patch.evidence,
    parentId: patch.parentId ?? null,
    assignee: patch.assignee ?? null,
    threadId: patch.threadId ?? null,
    runState: patch.runState ?? null,
    startedAt: patch.startedAt ?? null,
    order: patch.order ?? 1,
    createdAt: patch.createdAt ?? 1,
    updatedAt: patch.updatedAt ?? 1,
  }
}

{
  assert.deepEqual(normalizeTicketCapabilities('read, write shell nope'), ['read', 'write', 'shell'])
  assert.deepEqual(normalizeTicketCapabilities(['read', 'read', 'mcp', 'bad']), ['read', 'mcp'])
  assert.deepEqual(normalizeTicketCapabilities([], ['read']), ['read'])
}

{
  assert.deepEqual(normalizeStringList('Inspect files; Run tests\nSummarize blockers'), ['Inspect files', 'Run tests', 'Summarize blockers'])
  assert.deepEqual(normalizeStringList([' one ', 'one', 'two  spaces']), ['one', 'two spaces'])
}

{
  assert.equal(inferTicketRisk({ priority: 'urgent' } as Partial<Ticket>), 'high')
  assert.equal(inferTicketRisk({ allowedCapabilities: ['network'] } as Partial<Ticket>), 'medium')
  assert.equal(inferTicketRisk({ allowedCapabilities: ['read'] } as Partial<Ticket>), 'low')
  assert.equal(inferTicketRisk({ risk: 'high' } as Partial<Ticket>), 'high')
}

{
  const execution = normalizeTicketExecution({ type: 'task', priority: 'medium' }, { roomScoped: true })
  assert.equal(execution.risk, 'low')
  assert.deepEqual(execution.allowedCapabilities, ['read'])
  assert.deepEqual(execution.requiredEvidence, ['Relevant files or sources inspected', 'Implementation or investigation summary', 'Verification result or blocker'])
  assert.deepEqual(execution.acceptanceCriteria, [])
  assert.deepEqual(execution.evidence, [])
}

{
  const execution = normalizeTicketExecution({
    type: 'bug',
    allowedCapabilities: ['read', 'write', 'shell'],
    requiredEvidence: ['Reproduced locally', 'Patch verified'],
    acceptanceCriteria: 'No regression; Clear reviewer note',
    evidence: [{ kind: 'test', summary: 'npx tsx ticket test passed', ref: 'electron/agent/tickets.test.ts', createdBy: 'Reviewer', createdAt: 100 }],
  } as any)
  assert.equal(execution.risk, 'high')
  assert.deepEqual(execution.allowedCapabilities, ['read', 'write', 'shell'])
  assert.deepEqual(execution.requiredEvidence, ['Reproduced locally', 'Patch verified'])
  assert.deepEqual(execution.acceptanceCriteria, ['No regression', 'Clear reviewer note'])
  assert.ok(execution.evidence)
  assert.equal(execution.evidence[0].kind, 'test')
  assert.equal(execution.evidence[0].createdAt, 100)
}

{
  assert.deepEqual(defaultRequiredEvidence({ type: 'epic' } as Partial<Ticket>), ['Child tickets created or linked', 'Completion summary with remaining gaps'])
  assert.deepEqual(defaultRequiredEvidence({ type: 'bug' } as Partial<Ticket>), ['Reproduction or observed failure', 'Fix or mitigation summary', 'Verification result or blocker'])
  assert.deepEqual(defaultRequiredEvidence({ status: 'review' } as Partial<Ticket>), ['Reviewed files or evidence', 'Reviewer verdict', 'Blocking issues or explicit approval'])
}

{
  const evidence = normalizeTicketEvidence([
    null,
    { kind: 'unknown', summary: '  Checked the docs  ', ref: '' },
    { kind: 'file', summary: 'Read source', ref: 'src/App.tsx' },
  ])
  assert.equal(evidence.length, 2)
  assert.equal(evidence[0].kind, 'note')
  assert.equal(evidence[0].summary, 'Checked the docs')
  assert.equal(evidence[1].kind, 'file')
  assert.equal(evidence[1].ref, 'src/App.tsx')
}

{
  const normalized = normalizeTickets([
    { id: 'ticket-1', projectId: 'project-1', roomId: 'room-1', title: 'Room ticket', body: '', type: 'task', status: 'ready', priority: 'medium', effort: 'medium', tags: [], parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1 },
    { id: 'ticket-2', projectId: 'project-1', title: 'Build ticket', body: '', type: 'task', status: 'todo', priority: 'high', effort: 'medium', tags: [], parentId: null, assignee: null, threadId: null, runState: null, order: 2, createdAt: 2, updatedAt: 2 },
  ] as Ticket[])
  assert.deepEqual(normalized[0].allowedCapabilities, ['read'])
  assert.equal(normalized[0].risk, 'low')
  assert.deepEqual(normalized[1].allowedCapabilities, ['read', 'write', 'shell'])
  assert.equal(normalized[1].risk, 'high')
}

{
  const refs = ticketFileRefs(ticket({
    title: 'Patch src/App.tsx',
    body: 'Also touch electron/main.ts and ./src/lib/store.ts.',
    evidence: [{ id: 'e1', kind: 'file', summary: 'Read src/App.tsx.', ref: 'src/App.tsx', createdAt: 1 }],
  }))
  assert.deepEqual(refs, ['src/App.tsx', 'electron/main.ts', 'src/lib/store.ts'])
}

{
  const active = ticket({
    id: 'active',
    title: 'Active patch',
    status: 'in_progress',
    runState: 'running',
    assignee: 'builder',
    evidence: [{ id: 'e1', kind: 'file', summary: 'Working in src/App.tsx', ref: 'src/App.tsx', createdAt: 1 }],
  })
  const candidate = ticket({ id: 'candidate', title: 'Next patch', body: 'Change src/App.tsx', status: 'ready' })
  const blockers = roomTicketConflictBlockers({ ticket: candidate, tickets: [active, candidate], seat: 'reviewer' })
  assert.equal(blockers.length, 1)
  assert.match(blockers[0], /overlaps with "Active patch" on src\/App\.tsx/)
}

{
  const claimed = ticket({ title: 'Claimed build', assignee: 'builder', status: 'in_progress' })
  assert.deepEqual(roomTicketConflictBlockers({ ticket: claimed, tickets: [claimed], seat: 'reviewer', phase: 'build' }), ['"Claimed build" is claimed by builder.'])
  assert.deepEqual(roomTicketConflictBlockers({ ticket: claimed, tickets: [claimed], seat: 'reviewer', phase: 'review' }), [
    '"Claimed build" is not ready for review; run a build worker first.',
    '"Claimed build" has no builder evidence to review.',
  ])
}

{
  const built = ticket({
    title: 'Built patch',
    status: 'review',
    assignee: 'builder',
    evidence: [{ id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 10 }],
  })
  assert.deepEqual(roomTicketConflictBlockers({ ticket: built, tickets: [built], seat: 'builder', phase: 'review' }), ['"Built patch" needs a reviewer different from builder builder.'])
  assert.deepEqual(roomTicketConflictBlockers({ ticket: built, tickets: [built], seat: 'reviewer', phase: 'review' }), [])
}

{
  const left = roomLeftToWork([
    ticket({ title: 'Done', status: 'completed', order: 4 }),
    ticket({ title: 'Ready', status: 'ready', order: 3 }),
    ticket({ title: 'Review', status: 'review', order: 2 }),
    ticket({ title: 'Blocked', status: 'needs_clarification', order: 1 }),
  ])
  assert.deepEqual(left.map((item) => item.title), ['Blocked', 'Review', 'Ready'])
  const summary = summarizeRoomLeftToWork(left)
  assert.match(summary, /Blocked/)
  assert.match(summary, /Review/)
  assert.doesNotMatch(summary, /Done/)
}

{
  assert.equal(ticketNeedsReviewAfterBuilder(ticket({
    title: 'Needs review',
    status: 'review',
    risk: 'high',
    allowedCapabilities: ['read', 'write'],
    evidence: [{ id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 10 }],
  })), true)
  assert.equal(ticketNeedsReviewAfterBuilder(ticket({
    title: 'Self review is not enough',
    status: 'completed',
    risk: 'high',
    allowedCapabilities: ['read', 'write'],
    evidence: [
      { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 10 },
      { id: 'e2', kind: 'review', summary: 'Reviewer approved.', createdBy: 'builder', createdAt: 11 },
    ],
  })), true)
  assert.equal(ticketNeedsReviewAfterBuilder(ticket({
    title: 'Reviewed by another seat',
    status: 'completed',
    risk: 'high',
    allowedCapabilities: ['read', 'write'],
    evidence: [
      { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 10 },
      { id: 'e2', kind: 'review', summary: 'Reviewer approved.', createdBy: 'reviewer', createdAt: 11 },
    ],
  })), false)
  assert.equal(ticketNeedsReviewAfterBuilder(ticket({
    title: 'Low risk note',
    status: 'review',
    risk: 'low',
    allowedCapabilities: ['read'],
    evidence: [{ id: 'e1', kind: 'note', summary: 'Read docs and made no code changes.', createdBy: 'builder', createdAt: 10 }],
  })), false)
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({ title: 'Plan migration', status: 'ready' }),
      ticket({
        title: 'Patch risky code',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Files changed', 'Tests passed'],
        evidence: [{ id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', createdAt: 10 }],
      }),
    ],
  })
  assert.equal(blockers.length, 4)
  assert.match(blockers[0], /still open: Plan migration/)
  assert.match(blockers[1], /required evidence missing on: Patch risky code/)
  assert.match(blockers[2], /review evidence after builder or justified skip missing on: Patch risky code/)
  assert.match(blockers[3], /code changed without a later passing command/)
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Reserved only',
        status: 'completed',
        requiredEvidence: ['Implementation summary'],
        evidence: [{ id: 'e1', kind: 'file', summary: 'Reserved for build worker.', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 10 }],
      }),
    ],
  })
  assert.deepEqual(blockers, ['required evidence missing on: Reserved only.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Stale review',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'review', summary: 'Reviewer approved before the patch.', createdBy: 'reviewer', createdAt: 10 },
          { id: 'e2', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e3', kind: 'test', summary: 'npx tsx electron/agent/tickets.test.ts passed', createdBy: 'builder', createdAt: 21 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, ['review evidence after builder or justified skip missing on: Stale review.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Self reviewed',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e2', kind: 'test', summary: 'npx tsx electron/agent/tickets.test.ts passed', createdBy: 'builder', createdAt: 21 },
          { id: 'e3', kind: 'review', summary: 'Reviewer approved the change.', createdBy: 'builder', createdAt: 22 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, ['review evidence after builder or justified skip missing on: Self reviewed.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Blocking review',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Tests passed', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e2', kind: 'test', summary: 'npx tsx electron/agent/tickets.test.ts passed', createdBy: 'builder', createdAt: 21 },
          { id: 'e3', kind: 'review', summary: 'Reviewer found blocking issues: missing edge-case coverage.', createdBy: 'reviewer', createdAt: 22 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, ['review evidence after builder or justified skip missing on: Blocking review.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Blocking review resolved',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Tests passed', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e2', kind: 'test', summary: 'npx tsx electron/agent/tickets.test.ts passed', createdBy: 'builder', createdAt: 21 },
          { id: 'e3', kind: 'review', summary: 'Reviewer found blockers: missing edge-case coverage.', createdBy: 'reviewer', createdAt: 22 },
          { id: 'e4', kind: 'review', summary: 'Reviewer approved the follow-up; no blockers.', createdBy: 'reviewer', createdAt: 23 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, [])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'User approval is not review',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Tests passed', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e2', kind: 'test', summary: 'npx tsx electron/agent/tickets.test.ts passed', createdBy: 'builder', createdAt: 21 },
          { id: 'e3', kind: 'note', summary: 'User approval resolved: requester approved the change.', createdBy: 'user', createdAt: 22 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, ['review evidence after builder or justified skip missing on: User approval is not review.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Reviewer checked same timestamp',
        status: 'completed',
        risk: 'high',
        allowedCapabilities: ['read', 'write'],
        requiredEvidence: ['Implementation summary', 'Tests passed', 'Reviewer verdict'],
        evidence: [
          { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', ref: 'src/App.tsx', createdBy: 'builder', createdAt: 20 },
          { id: 'e2', kind: 'review', summary: 'Reviewer worker finished "Reviewer checked same timestamp". Thread: thread-1.', ref: 'thread:thread-1', createdBy: 'reviewer', createdAt: 30 },
          { id: 'e3', kind: 'test', summary: 'Test passed: npx tsx electron/agent/tickets.test.ts', ref: 'thread:thread-1', createdBy: 'reviewer', createdAt: 30 },
        ],
      }),
    ],
  })
  assert.deepEqual(blockers, [])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [
      ticket({
        title: 'Local copy edit',
        risk: 'low',
        allowedCapabilities: ['read'],
        requiredEvidence: ['Summary'],
        evidence: [{ id: 'e1', kind: 'note', summary: 'Read the file and summarized the issue.', createdAt: 10 }],
      }),
      ticket({
        title: 'Shell verification',
        risk: 'high',
        allowedCapabilities: ['read', 'shell'],
        requiredEvidence: ['Command output', 'Reviewer verdict'],
        evidence: [
          { id: 'e2', kind: 'command', summary: 'npx tsx test passed.', createdBy: 'builder', createdAt: 11 },
          { id: 'e3', kind: 'review', summary: 'Reviewer approved the change.', createdBy: 'reviewer', createdAt: 12 },
        ],
      }),
      ticket({
        title: 'Skipped reviewer',
        risk: 'medium',
        allowedCapabilities: ['network'],
        requiredEvidence: ['Source checked'],
        evidence: [
          { id: 'e4', kind: 'source', summary: 'Fetched source and documented result.', createdBy: 'builder', createdAt: 13 },
          { id: 'e5', kind: 'note', summary: 'Review skipped: read-only source lookup, no code change.', createdBy: 'reviewer', createdAt: 14 },
        ],
      }),
    ],
    lastCodeChangeAt: 100,
    greenEvidenceAt: 120,
  })
  assert.deepEqual(blockers, [])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Patch without green command',
      requiredEvidence: ['Patch summary'],
      evidence: [{ id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', createdAt: 20 }],
    })],
  })
  assert.deepEqual(blockers, ['code changed without a later passing command.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Patch with failed check',
      requiredEvidence: ['Patch summary', 'Verification result'],
      evidence: [
        { id: 'e1', kind: 'diff', summary: 'Changed src/App.tsx', createdAt: 20 },
        { id: 'e2', kind: 'test', summary: 'Vitest exited with code 1.', createdAt: 21 },
      ],
    })],
  })
  assert.deepEqual(blockers, ['code changed without a later passing command.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Patch with old test',
      requiredEvidence: ['Reviewer approved'],
      evidence: [{ id: 'e1', kind: 'review', summary: 'Reviewer approved.', createdAt: 1 }],
    })],
    lastCodeChangeAt: 200,
    greenEvidenceAt: 150,
  })
  assert.deepEqual(blockers, ['code changed without a later passing command.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Send Telegram update',
      allowedCapabilities: ['connector'],
      requiredEvidence: ['Delivery summary', 'Reviewer verdict'],
      evidence: [
        { id: 'e1', kind: 'note', summary: 'Sent Telegram update to an external channel.', createdBy: 'builder', createdAt: 10 },
        { id: 'e2', kind: 'review', summary: 'Reviewer approved the delivery evidence.', createdBy: 'reviewer', createdAt: 11 },
      ],
    })],
  })
  assert.deepEqual(blockers, ['external/destructive approval unresolved on: Send Telegram update.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Stale approval before external send',
      allowedCapabilities: ['connector'],
      requiredEvidence: ['Delivery summary', 'Reviewer verdict', 'Approval status'],
      evidence: [
        { id: 'e1', kind: 'note', summary: 'User approval resolved before final content was ready.', createdBy: 'user', createdAt: 9 },
        { id: 'e2', kind: 'note', summary: 'Sent email to the external recipient.', createdBy: 'builder', createdAt: 10 },
        { id: 'e3', kind: 'review', summary: 'Reviewer approved the delivery evidence.', createdBy: 'reviewer', createdAt: 11 },
      ],
    })],
  })
  assert.deepEqual(blockers, ['external/destructive approval unresolved on: Stale approval before external send.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Approved external send',
      allowedCapabilities: ['connector'],
      requiredEvidence: ['Delivery summary', 'Reviewer verdict', 'Approval status'],
      evidence: [
        { id: 'e1', kind: 'note', summary: 'Sent email to the external recipient.', createdBy: 'builder', createdAt: 10 },
        { id: 'e2', kind: 'review', summary: 'Reviewer approved the delivery evidence.', createdBy: 'reviewer', createdAt: 11 },
        { id: 'e3', kind: 'note', summary: 'User approval resolved: requester approved the external email send.', createdBy: 'user', createdAt: 12 },
      ],
    })],
  })
  assert.deepEqual(blockers, [])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Read external source only',
      allowedCapabilities: ['network'],
      requiredEvidence: ['Source checked'],
      evidence: [
        { id: 'e1', kind: 'source', summary: 'Fetched source and documented result.', createdBy: 'builder', createdAt: 10 },
        { id: 'e2', kind: 'note', summary: 'Review skipped: read-only source lookup, no code change.', createdBy: 'reviewer', createdAt: 11 },
      ],
    })],
  })
  assert.deepEqual(blockers, [])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Destructive cleanup',
      allowedCapabilities: ['shell'],
      requiredEvidence: ['Cleanup summary', 'Reviewer verdict'],
      evidence: [
        { id: 'e1', kind: 'command', summary: 'Deleted generated cache with rm -rf /tmp/vontra-cache.', createdBy: 'builder', createdAt: 10 },
        { id: 'e2', kind: 'review', summary: 'Reviewer approved the cleanup evidence.', createdBy: 'reviewer', createdAt: 11 },
      ],
    })],
  })
  assert.deepEqual(blockers, ['external/destructive approval unresolved on: Destructive cleanup.'])
}

{
  const blockers = roomFinalGateBlockers({
    tickets: [ticket({
      title: 'Approved destructive cleanup',
      allowedCapabilities: ['shell'],
      requiredEvidence: ['Cleanup summary', 'Reviewer verdict', 'Approval status'],
      evidence: [
        { id: 'e1', kind: 'command', summary: 'Deleted generated cache with rm -rf /tmp/vontra-cache.', createdBy: 'builder', createdAt: 10 },
        { id: 'e2', kind: 'review', summary: 'Reviewer approved the cleanup evidence.', createdBy: 'reviewer', createdAt: 11 },
        { id: 'e3', kind: 'note', summary: 'Destructive approval resolved: user approved the cleanup command.', createdBy: 'user', createdAt: 12 },
      ],
    })],
  })
  assert.deepEqual(blockers, [])
}

// ---- Ticket execution records (VST-018): build, review, blocked ----

{
  // TICKET BUILD: start stamps engine/branch/run; a clean settle proves checks and
  // lands the verdict at 'pending' (built, awaiting review).
  const record = startTicketExecution({ target: 'solo', providerId: 'openai', model: 'gpt-5.5', branch: 'vontra/room-abc123', threadId: 'th1', startedAt: 100 })
  assert.equal(record.reviewVerdict, null)
  assert.equal(record.finishedAt, null)
  const built = settleTicketExecution(record, {
    outcome: { phase: 'build', failed: false, interrupted: false },
    finishedAt: 900,
    commands: [{ command: 'npm test', state: 'ok' }, { command: 'npm run lint', state: 'error' }],
    blockers: [],
  })
  assert.equal(built.reviewVerdict, 'pending')
  assert.equal(built.finishedAt, 900)
  assert.deepEqual(built.checks, [{ command: 'npm test', passed: true }, { command: 'npm run lint', passed: false }])
  assert.deepEqual(ticketStateAfterRun({ status: 'in_progress' }, { phase: 'build', failed: false, interrupted: false }), { runState: 'done', status: 'review' })
}

{
  // TICKET REVIEW: a passing review approves; a failing one requests changes.
  const record = startTicketExecution({ target: 'solo', startedAt: 100 })
  const approved = settleTicketExecution(record, { outcome: { phase: 'review', failed: false, interrupted: false }, finishedAt: 500 })
  assert.equal(approved.reviewVerdict, 'approved')
  const changes = settleTicketExecution(record, { outcome: { phase: 'review', failed: true, interrupted: false }, finishedAt: 500 })
  assert.equal(changes.reviewVerdict, 'changes_requested')
  assert.deepEqual(ticketStateAfterRun({ status: 'in_progress' }, { phase: 'review', failed: false, interrupted: false }), { runState: 'done', status: 'completed' })
  assert.deepEqual(ticketStateAfterRun({ status: 'in_progress' }, { phase: 'review', failed: true, interrupted: false }), { runState: 'done', status: 'needs_clarification' })
}

{
  // TICKET BLOCKED: an interrupted run parks the ticket (status unchanged, re-dispatchable),
  // names the blocker, keeps finishedAt open, and never invents a verdict.
  const record = startTicketExecution({ target: 'cli', providerId: 'codex-cli', model: 'gpt-5.5-codex', startedAt: 100 })
  const parked = settleTicketExecution(record, {
    outcome: { phase: 'build', failed: false, interrupted: true },
    finishedAt: 700,
    blockers: ['approval pending: run_command'],
  })
  assert.equal(parked.finishedAt, null, 'a parked run is not finished')
  assert.equal(parked.reviewVerdict, null)
  assert.deepEqual(parked.blockers, ['approval pending: run_command', 'run parked before completion (cancelled or waiting on approval)'])
  assert.deepEqual(ticketStateAfterRun({ status: 'in_progress' }, { phase: 'build', failed: false, interrupted: true }), { runState: null, status: 'in_progress' })
  // A failed (not interrupted) build clears the verdict — nothing reviewable was produced.
  const failed = settleTicketExecution(record, { outcome: { phase: 'build', failed: true, interrupted: false }, finishedAt: 700 })
  assert.equal(failed.reviewVerdict, null)
}

console.log('tickets tests passed')
