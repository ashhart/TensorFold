// Run: jiti electron/agent/attention.test.ts
import assert from 'node:assert/strict'
import { computeNeedsMe, needsMeCount, newAttention, pruneSeen, attentionNotification } from './attention.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('attention tests:')

{
  // Pending tool approvals + unsent connector drafts both count as "your turn".
  const items = computeNeedsMe({
    jobs: [{ id: 'j1', pendingApprovals: [{ callId: 'c1', tool: 'run_command', threadId: 't1' }] }],
    connectorEvents: [
      { id: 'e1', requiresApproval: true, approvalStatus: null, title: 'Reply to Maya', threadKey: 'email:maya' },
      { id: 'e2', requiresApproval: true, approvalStatus: 'approved' }, // already sent → not needs-me
      { id: 'e3', requiresApproval: false }, // not a draft
    ],
  })
  assert.equal(items.length, 2)
  assert.deepEqual(items.map((i) => i.kind).sort(), ['approval', 'connector_draft'])
  assert.equal(items.find((i) => i.kind === 'approval')?.detail, 'Run run_command?')
  assert.equal(items.find((i) => i.kind === 'connector_draft')?.threadId, 'email:maya')
  assert.equal(needsMeCount({ jobs: [], connectorEvents: [] }), 0)
  ok('computeNeedsMe: pending approvals + unsent drafts; resolved/none excluded')
}

{
  // newAttention fires once per item; pruneSeen lets a resolved id re-notify later.
  const items = computeNeedsMe({ jobs: [{ id: 'j', pendingApprovals: [{ callId: 'c1' }, { callId: 'c2' }] }] })
  const seen = new Set<string>()
  const firstFire = newAttention(items, seen)
  assert.equal(firstFire.length, 2, 'both new on first pass')
  firstFire.forEach((i) => seen.add(i.id))
  assert.equal(newAttention(items, seen).length, 0, 'nothing new on second pass (no re-ping)')

  // c1 resolves; pruneSeen drops it so if it ever recurs it notifies again.
  const after = computeNeedsMe({ jobs: [{ id: 'j', pendingApprovals: [{ callId: 'c2' }] }] })
  const pruned = pruneSeen(seen, after)
  assert.ok(!pruned.has('approval:c1'), 'resolved id pruned from seen')
  assert.ok(pruned.has('approval:c2'), 'still-pending id kept')
  ok('newAttention/pruneSeen: notify once, re-notify only after resolve+recur')
}

{
  const n = attentionNotification({ id: 'x', kind: 'approval', title: 'Approval needed', detail: 'Run run_command?' })
  assert.equal(n.title, 'TensorFold · Approval needed')
  assert.equal(n.body, 'Run run_command?')
  ok('attentionNotification: title + short body')
}

console.log(`\n${passed} passed`)
