import assert from 'node:assert/strict'
import { mergeGmailInboxSnapshot } from './gmail-inbox-view'
import { eventToBubble } from './conversation'

const key = 'gmail:account-a:1:dd11'
const anchor: any = { id: 'gmail-a-aa11', threadKey: key, connectorId: 'email', direction: 'inbound', action: 'message_received', actor: 'sender@example.com', createdAt: 10, sources: [{ id: 'source-aa11', itemId: 'aa11', path: 'https://mail.google.com/mail/u/?authuser=reader%40example.com#all/aa11' }], payload: { gmailInbox: true } }
const base: any = { events: [{ id: 'ordinary', createdAt: 1 }], connectors: [{ id: 'email', status: 'disconnected', permissionStatus: 'denied' }] }
const messages = [{ id: 'aa11', threadId: 'dd11', from: 'Sender <sender@example.com>', subject: 'Received', snippet: 'Preview', text: 'Synthetic full body', date: '', internalDate: 10 }, { id: 'aa12', threadId: 'dd11', from: 'Reader <reader@example.com>', subject: 'Sent', snippet: 'My reply', text: 'Synthetic sent body', date: '', internalDate: 20 }]
const gmail: any = { snapshot: { accounts: [{ id: 'account-a', email: 'reader@example.com', enabled: true, connected: true }], events: [anchor] }, opened: { key, thread: { id: 'dd11', messages } } }
const result = mergeGmailInboxSnapshot(base, gmail)!
assert.equal(result.events.length, 3)
assert.equal(base.events.length, 1, 'Protected Gmail DTOs never mutate the general connector snapshot')
const received = result.events.find(event => event.sources?.[0].itemId === 'aa11')!, sent = result.events.find(event => event.sources?.[0].itemId === 'aa12')!
assert.equal(received.summary, 'Synthetic full body'); assert.equal(received.direction, 'inbound')
assert.equal(sent.direction, 'outbound'); assert.equal(eventToBubble({ ...sent, approvalStatus: sent.approvalStatus ?? undefined }).status, undefined, 'Reading a historical sent message cannot invent a delivery receipt')
assert.ok(sent.sources![0].path?.endsWith('#all/aa12')); assert.equal(sent.payload?.messageId, 'aa12')
assert.equal(result.connectors[0].status, 'connected')
assert.equal(mergeGmailInboxSnapshot(null, gmail), null)
assert.equal(eventToBubble({ id: 'review', direction: 'outbound', approvalStatus: 'approved', delivery: { transport: 'gmail', status: 'queued' } }).status, 'approved', 'Gmail acceptance does not become recipient delivery')
console.log('Gmail full reads retain native source IDs, sent/received identity and private UI-only merging without claiming delivery.')
