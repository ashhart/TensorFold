// Run: jiti src/components/assistant/pending-approvals.test.ts
import assert from 'node:assert/strict'
import { isAwaitingApproval, selectPendingApprovals } from './pending-approvals.ts'
import type { ConnectorEvent } from '../../lib/types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('pending-approvals tests:')

function evt(over: Partial<ConnectorEvent>): ConnectorEvent {
  return {
    id: 'e', connectorId: 'slack', pluginName: 'Messages', action: 'send', direction: 'outbound',
    classification: 'communication', priority: 'normal', trustLevel: 'suggest',
    requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: 't', summary: 's', actor: 'a', threadKey: 'slack:thread', payloadKeys: [], createdAt: 1,
    ...over,
  } as ConnectorEvent
}

{
  assert.equal(isAwaitingApproval(evt({ requiresApproval: false })), false, 'no approval needed')
  assert.equal(isAwaitingApproval(evt({ requiresApproval: true, approvalStatus: 'pending' })), true, 'pending')
  assert.equal(isAwaitingApproval(evt({ requiresApproval: true, approvalStatus: null })), true, 'unset status still counts')
  assert.equal(isAwaitingApproval(evt({ requiresApproval: true, approvalStatus: undefined })), true, 'undefined status still counts')
  assert.equal(isAwaitingApproval(evt({ requiresApproval: true, approvalStatus: 'approved' })), false, 'already approved drops out')
  assert.equal(isAwaitingApproval(evt({ requiresApproval: true, approvalStatus: 'denied' })), false, 'already denied drops out')
  ok('isAwaitingApproval: needs requiresApproval + an unresolved status')
}

{
  const events: ConnectorEvent[] = [
    evt({ id: 'resolved', requiresApproval: true, approvalStatus: 'approved' }),
    evt({ id: 'normal-old', requiresApproval: true, approvalStatus: 'pending', priority: 'normal', createdAt: 10 }),
    evt({ id: 'urgent', requiresApproval: true, approvalStatus: 'pending', priority: 'high', createdAt: 5 }),
    evt({ id: 'normal-new', requiresApproval: true, approvalStatus: 'pending', priority: 'normal', createdAt: 20 }),
    evt({ id: 'fyi', requiresApproval: false }),
  ]
  const queue = selectPendingApprovals(events)
  assert.deepEqual(queue.map((q) => q.id), ['urgent', 'normal-new', 'normal-old'], 'urgent first, then newest first; resolved + fyi excluded')
  assert.equal(queue[0].urgent, true, 'high priority flagged urgent')
  assert.equal(queue[1].urgent, false, 'normal priority not urgent')
  ok('selectPendingApprovals: queues unresolved approvals urgent-first then newest-first')
}

console.log(`\n${passed} checks passed.`)
