import assert from 'node:assert/strict'
import { threadPendingApproval } from './thread-approval'
const item = (id: string) => ({ id, kind: 'tool', callId: 'raw-shared', state: 'awaiting', tool: 'write_file' } as any)
const first = item('item-first'), second = item('item-second')
const jobs = ['first', 'second'].map((id) => ({ id: `job-${id}`, threadId: id, currentThreadId: id, pendingApprovals: [{ id: `pending-${id}`, threadId: id, itemId: `item-${id}`, callId: 'raw-shared', resolvedAt: null }] } as any))
assert.equal(threadPendingApproval('first', [first], [], jobs)?.callId, 'item-first', 'A durable approval must survive without transient events after restart')
assert.equal(threadPendingApproval('second', [second], [], jobs)?.callId, 'item-second', 'Another worker sharing the raw model ID must retain its own prompt')
assert.equal(threadPendingApproval('second', [second], [], [jobs[0]]), null, 'A foreign durable job cannot populate this worker prompt')
assert.equal(threadPendingApproval('first', [], [], jobs), null, 'No decision may be offered for an absent review item')
assert.equal(threadPendingApproval('first', [first], [], [{ ...jobs[0], pendingApprovals: [{ ...jobs[0].pendingApprovals[0], resolvedAt: 123 }] }]), null)
assert.equal(threadPendingApproval('first', [{ ...first, approvalId: 'live-opaque' }], [{ itemId: first.id, callId: 'live-opaque' }], jobs)?.callId, 'live-opaque', 'Current live opaque routing remains unchanged')
console.log('Restored thread prompt ownership, absence, settlement and live routing pass')
