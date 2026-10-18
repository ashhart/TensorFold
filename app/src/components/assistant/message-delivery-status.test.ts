import assert from 'node:assert/strict'
import { eventToBubble } from './conversation'

const approved = { id: 'draft-fixture', actor: 'TensorFold', action: 'message_send_requested', direction: 'outbound', approvalStatus: 'approved', summary: 'A reviewed message', delivery: { status: 'sending' } }
assert.notEqual(eventToBubble(approved as any).status, 'sent', 'Approval and a pending network attempt do not prove delivery')
assert.equal(eventToBubble({ ...approved, delivery: { status: 'delivered' } } as any).status, 'sent')
assert.equal(eventToBubble({ ...approved, delivery: { status: 'uncertain' } } as any).status, 'uncertain', 'Ambiguous completion must not invite duplicate retry')
assert.equal(eventToBubble({ ...approved, delivery: { status: 'failed' } } as any).status, 'failed')
assert.notEqual(eventToBubble({ ...approved, delivery: undefined } as any).status, 'sent', 'Legacy approved drafts without receipts are not labelled sent')
