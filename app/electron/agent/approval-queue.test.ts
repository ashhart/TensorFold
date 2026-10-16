// Run: jiti electron/agent/approval-queue.test.ts
import assert from 'node:assert/strict'
import {
  createApprovalQueue,
  normalizeApprovalQueue,
  enqueueApproval,
  ageApprovalQueue,
  entryUrgency,
  resolveApproval,
  resolveByScopeKey,
  resolveByJob,
  batchResolveApprovals,
  pendingApprovals,
  orderedPending,
  summarizeApprovalQueue,
  pruneApprovalQueue,
  type ApprovalQueue,
  type ApprovalQueueEntryInput,
} from './approval-queue.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('approval-queue tests:')

const scope = (key: string, classification?: string) => ({
  scope: 'tool' as const,
  key,
  label: `label ${key}`,
  classification,
})

const input = (over: Partial<ApprovalQueueEntryInput> & { idempotencyKey: string }): ApprovalQueueEntryInput => ({
  jobId: 'job-1',
  callId: 'call-1',
  tool: 'run_command',
  scope: scope('run_command:test:abc'),
  ...over,
})

// ---- exactly-once enqueue (idempotency) ----
{
  const q0 = createApprovalQueue({ slaMs: 1000, agingMs: 400 }, 0)
  const a = enqueueApproval(q0, input({ idempotencyKey: 'k1' }), 100)
  assert.equal(a.created, true, 'first enqueue creates')
  assert.equal(a.queue.entries.length, 1)

  // Same key while pending => duplicate, no new row, requeueCount bumped.
  const b = enqueueApproval(a.queue, input({ idempotencyKey: 'k1' }), 200)
  assert.equal(b.created, false, 'duplicate not created')
  assert.equal(b.queue.entries.length, 1, 'no second row')
  assert.equal(b.entry.requeueCount, 1, 'requeueCount bumped')

  // Input queue was not mutated (purity).
  assert.equal(a.queue.entries[0].requeueCount, 0, 'original queue untouched')
  ok('duplicate idempotency key coalesces (exactly-once, pure)')
}

// ---- resolved slot never reopens (exactly-once across a decision) ----
{
  let q = createApprovalQueue({ slaMs: 1000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'k1' }), 0).queue
  q = resolveApproval(q, 'k1', 'allow', 'user', 50).queue
  const again = enqueueApproval(q, input({ idempotencyKey: 'k1' }), 100)
  assert.equal(again.created, false, 're-enqueue of resolved key does not create')
  assert.equal(again.entry.state, 'resolved', 'stays resolved')
  assert.equal(pendingApprovals(again.queue).length, 0, 'no pending resurrected')
  ok('resolved slot is terminal — replay after restart cannot double-prompt')
}

// ---- aging + SLA escalation, idempotent ----
{
  let q = createApprovalQueue({ slaMs: 1000, agingMs: 400 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'k1' }), 0).queue
  const e = q.entries[0]

  assert.equal(entryUrgency(q, e, 100), 'normal', 'fresh = normal')
  assert.equal(entryUrgency(q, e, 500), 'aging', 'past aging threshold')
  assert.equal(entryUrgency(q, e, 1000), 'escalated', 'past SLA = escalated')

  // ageApprovalQueue stamps escalatedAt exactly once.
  const first = ageApprovalQueue(q, 1200)
  assert.equal(first.changed, true)
  assert.equal(first.newlyEscalated.length, 1, 'escalated once')
  assert.equal(first.queue.entries[0].escalatedAt, 1200)
  const second = ageApprovalQueue(first.queue, 1300)
  assert.equal(second.changed, false, 'no re-escalation')
  assert.equal(second.newlyEscalated.length, 0)
  ok('aging thresholds + SLA escalation fire once')
}

// ---- per-entry slaMs override ----
{
  let q = createApprovalQueue({ slaMs: 10_000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'fast', slaMs: 100 }), 0).queue
  const r = ageApprovalQueue(q, 200)
  assert.equal(r.newlyEscalated.length, 1, 'per-entry slaMs escalates before queue SLA')
  ok('per-entry SLA override')
}

// ---- hard expiry ----
{
  let q = createApprovalQueue({ slaMs: 10_000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'exp', expiresAt: 500 }), 0).queue
  assert.equal(entryUrgency(q, q.entries[0], 600), 'expired')
  const r = ageApprovalQueue(q, 600)
  assert.equal(r.newlyExpired.length, 1, 'expired at deadline')
  assert.equal(r.queue.entries[0].state, 'expired')
  // A resolve after expiry is a no-op (already terminal).
  const after = resolveApproval(r.queue, 'exp', 'allow', 'user', 700)
  assert.equal(after.resolved.length, 0, 'cannot resolve expired')
  ok('hard expiry deadline auto-expires and blocks late resolve')
}

// ---- batch-resolve by scope key ----
{
  let q = createApprovalQueue({ slaMs: 1000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'a', callId: 'c-a', scope: scope('same:key') }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'b', callId: 'c-b', scope: scope('same:key') }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'c', callId: 'c-c', scope: scope('other:key') }), 0).queue

  const r = resolveByScopeKey(q, 'same:key', 'allow_always', 'user', 10)
  assert.equal(r.resolved.length, 2, 'both same-scope entries resolved')
  assert.ok(r.resolved.every((e) => e.decision === 'allow_always'))
  assert.equal(pendingApprovals(r.queue).length, 1, 'other-scope entry still pending')
  ok('batch-resolve by policy scope key ("approve all like this")')
}

// ---- batch-resolve by job + custom predicate ----
{
  let q = createApprovalQueue({ slaMs: 1000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'a', jobId: 'jX' }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'b', jobId: 'jX' }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'c', jobId: 'jY' }), 0).queue

  const byJob = resolveByJob(q, 'jX', 'deny', 'system', 10)
  assert.equal(byJob.resolved.length, 2, 'all jX entries denied')
  assert.equal(pendingApprovals(byJob.queue).length, 1)

  const byPred = batchResolveApprovals(byJob.queue, (e) => e.jobId === 'jY', 'allow', null, 20)
  assert.equal(byPred.resolved.length, 1)
  assert.equal(pendingApprovals(byPred.queue).length, 0)
  ok('batch-resolve by job id and arbitrary predicate')
}

// ---- ordered pending: most-urgent-first, then oldest-first ----
{
  let q = createApprovalQueue({ slaMs: 1000, agingMs: 400 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'old', slaMs: 100 }), 0).queue      // will be escalated
  q = enqueueApproval(q, input({ idempotencyKey: 'newA' }), 500).queue                // aging at now=500
  q = enqueueApproval(q, input({ idempotencyKey: 'newB' }), 900).queue                // normal at now=1000

  const order = orderedPending(q, 1000).map((e) => e.idempotencyKey)
  assert.deepEqual(order, ['old', 'newA', 'newB'], 'escalated, then aging, then normal')
  ok('orderedPending sorts by urgency then age')
}

// ---- summary ----
{
  let q = createApprovalQueue({ slaMs: 1000, agingMs: 400 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'esc', slaMs: 100 }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'age' }), 500).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'norm' }), 900).queue
  q = resolveApproval(q, 'norm', 'allow', 'user', 950).queue

  const s = summarizeApprovalQueue(q, 1000)
  assert.equal(s.escalated, 1, 'one escalated')
  assert.equal(s.aging, 1, 'one aging')
  assert.equal(s.resolved, 1, 'one resolved')
  assert.equal(s.pending, 2, 'two still pending')
  assert.equal(s.oldestPendingAgeMs, 1000, 'oldest pending age')
  ok('summarizeApprovalQueue counts by urgency')
}

// ---- normalize/rehydrate drops dupes + preserves terminal state ----
{
  const persisted = {
    slaMs: 2000,
    agingMs: 800,
    entries: [
      { idempotencyKey: 'k1', jobId: 'j', callId: 'c', tool: 'run_command', scope: scope('s'), state: 'resolved', enqueuedAt: 0, resolvedAt: 10, decision: 'allow', requeueCount: 0 },
      { idempotencyKey: 'k1', jobId: 'j', callId: 'c', tool: 'run_command', scope: scope('s'), state: 'pending', enqueuedAt: 5, requeueCount: 0 }, // dup key => dropped
      { idempotencyKey: '', jobId: 'j', callId: 'c', tool: 't', scope: scope('s'), state: 'pending', enqueuedAt: 0 }, // empty key => dropped
      { not: 'an entry' },
    ],
  }
  const q: ApprovalQueue = normalizeApprovalQueue(persisted, 100)
  assert.equal(q.entries.length, 1, 'dupes and junk dropped')
  assert.equal(q.entries[0].state, 'resolved', 'terminal state preserved (exactly-once)')
  assert.equal(q.slaMs, 2000)
  // Re-enqueue after restart of a resolved key is still a no-op.
  const re = enqueueApproval(q, input({ idempotencyKey: 'k1' }), 200)
  assert.equal(re.created, false)
  assert.equal(pendingApprovals(re.queue).length, 0)
  ok('normalizeApprovalQueue rehydrates exactly-once (dedup + terminal preserved)')
}

// ---- prune keeps pending, drops old terminal ----
{
  let q = createApprovalQueue({ slaMs: 1000 }, 0)
  q = enqueueApproval(q, input({ idempotencyKey: 'keep' }), 0).queue
  q = enqueueApproval(q, input({ idempotencyKey: 'old' }), 0).queue
  q = resolveApproval(q, 'old', 'allow', 'user', 10).queue

  const pruned = pruneApprovalQueue(q, 100, 1000) // cutoff = 900; resolvedAt=10 < 900 => drop
  assert.equal(pruned.entries.length, 1, 'old resolved pruned')
  assert.equal(pruned.entries[0].idempotencyKey, 'keep', 'pending retained')
  ok('pruneApprovalQueue drops aged terminal rows, keeps pending')
}

// ---- unknown-key resolve is a safe no-op ----
{
  const q = enqueueApproval(createApprovalQueue({}, 0), input({ idempotencyKey: 'k' }), 0).queue
  const r = resolveApproval(q, 'nope', 'allow', 'user', 10)
  assert.equal(r.resolved.length, 0, 'no match => no-op')
  assert.strictEqual(r.queue, q, 'same queue reference on no-op')
  ok('resolve of unknown key is a safe no-op')
}

console.log(`\napproval-queue: ${passed} checks passed`)
