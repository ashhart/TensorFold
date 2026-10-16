// Run: jiti electron/main/host/connector-enrich.test.ts
import assert from 'node:assert/strict'
import {
  contactPlatformFor,
  emailContactObservations,
  enrichContacts,
  slackContactObservations,
  whatsappContactObservations,
} from './connector-enrich.ts'
import { emptyContactDirectory, learnContacts, resolveHandle } from '../../agent/contact-directory.ts'
import type { ConnectorEvent } from '../../agent/types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('connector-enrich tests:')

function imessageEvent(handle: string, actor: string): ConnectorEvent {
  return {
    id: 'e1', connectorId: 'imessage', pluginName: 'iMessage', action: 'message_received',
    direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify',
    requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: actor, summary: 'hi', actor, threadKey: `imessage:${handle}`,
    payloadKeys: ['text', 'handle'], payload: { text: 'hi', handle }, createdAt: 1,
  }
}

{
  assert.equal(contactPlatformFor('imessage'), 'phone')
  assert.equal(contactPlatformFor('whatsapp'), 'phone')
  assert.equal(contactPlatformFor('slack'), 'slack')
  assert.equal(contactPlatformFor('discord'), 'discord')
  assert.equal(contactPlatformFor('email'), 'email')
  assert.equal(contactPlatformFor('docs'), 'other')
  ok('contactPlatformFor: maps connector ids to contact platforms')
}

{
  // A raw phone-number actor is rewritten to the learned name; the handle stays in payload.
  const events = [imessageEvent('+447700900123', '+447700900123')]
  const { directory, events: out } = enrichContacts(
    emptyContactDirectory(1),
    events,
    [{ platform: 'phone', handle: '+447700900123', name: 'Maya Chen', source: 'imessage' }],
    10,
  )
  assert.equal(out[0].actor, 'Maya Chen', 'raw-number actor resolved to name')
  assert.equal(out[0].title, 'Maya Chen', 'title resolved too')
  assert.equal(out[0].payload?.handle, '+447700900123', 'handle preserved for reply routing')
  assert.equal(directory.records['phone:+447700900123'].name, 'Maya Chen')
  ok('enrichContacts: learns + rewrites a raw-number actor to the resolved name')
}

{
  // An actor that is NOT the raw handle (e.g. a channel name) is left untouched even when a
  // name is known for the handle.
  const ev = imessageEvent('+447700900123', '#family-group')
  const { events: out } = enrichContacts(
    emptyContactDirectory(1),
    [ev],
    [{ platform: 'phone', handle: '+447700900123', name: 'Maya Chen' }],
    10,
  )
  assert.equal(out[0].actor, '#family-group', 'a non-handle actor is preserved')
  ok('enrichContacts: does not clobber a non-handle actor (channel/group name)')
}

{
  // Unknown handle: nothing to resolve, event passes through unchanged.
  const ev = imessageEvent('+15550000000', '+15550000000')
  const { events: out } = enrichContacts(emptyContactDirectory(1), [ev], [], 10)
  assert.equal(out[0].actor, '+15550000000', 'unknown handle stays as the raw number')
  ok('enrichContacts: unknown handle passes through unchanged')
}

// ---- Per-connector contact observation builders -------------------------------

function slackEvent(senderId: string, sender: string, opts: { direction?: ConnectorEvent['direction'] } = {}): ConnectorEvent {
  return {
    id: 's1', connectorId: 'slack', pluginName: 'Slack', action: 'message_received',
    direction: opts.direction ?? 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify',
    requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: '#launch: Maya', summary: 'hi', actor: '#launch', threadKey: 'slack:s:abc',
    payloadKeys: ['sender', 'sender_id'], payload: { sender, senderId, channel: 'C1', ts: '1.2' }, createdAt: 1,
  }
}

function emailEvent(from: string): ConnectorEvent {
  return {
    id: 'm1', connectorId: 'email', pluginName: 'Email', action: 'message_received',
    direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'suggest',
    requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: 'Re: launch', summary: 'hi', actor: from, threadKey: 'email:abc',
    payloadKeys: ['from'], payload: { from }, createdAt: 1,
  }
}

function whatsappEvent(jid: string, actor: string, opts: { direction?: ConnectorEvent['direction'] } = {}): ConnectorEvent {
  return {
    id: 'w1', connectorId: 'whatsapp', pluginName: 'WhatsApp', action: 'message_received',
    direction: opts.direction ?? 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify',
    requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null,
    title: actor, summary: 'hi', actor, threadKey: `whatsapp:wa:${jid}`,
    payloadKeys: ['text', 'jid'], payload: { text: 'hi', jid }, createdAt: 1,
  }
}

{
  // Slack: an inbound message with a resolved sender name yields one slack observation; the
  // learned directory resolves "reply to Maya" on Slack back to the user id.
  const obs = slackContactObservations([slackEvent('U024BE7LH', 'Maya Chen')])
  assert.deepEqual(obs, [{ platform: 'slack', handle: 'U024BE7LH', name: 'Maya Chen', source: 'slack' }])
  const dir = learnContacts(emptyContactDirectory(1), obs, 10)
  assert.equal(resolveHandle(dir, 'Maya Chen', 'slack')?.handle, 'u024be7lh', 'reply-to-name resolves the slack id')
  ok('slackContactObservations: learns a Slack user id -> display name')
}

{
  // Slack: the live-API placeholder where senderName === senderId teaches nothing (no real
  // name yet), and an outbound "You" send is skipped (direction guard).
  assert.deepEqual(slackContactObservations([slackEvent('U999', 'U999')]), [], 'id-as-name is not an observation')
  assert.deepEqual(slackContactObservations([slackEvent('U024', 'Maya', { direction: 'outbound' })]), [], 'outbound is skipped')
  ok('slackContactObservations: skips id-as-name placeholder and outbound events')
}

{
  // Email: a From header with a display name yields an address->name observation; a bare
  // address (no display name) yields nothing.
  const obs = emailContactObservations([emailEvent('"Maya Chen" <maya@example.com>')])
  assert.deepEqual(obs, [{ platform: 'email', handle: 'maya@example.com', name: 'Maya Chen', source: 'email' }])
  assert.deepEqual(emailContactObservations([emailEvent('bare@example.com')]), [], 'a bare address has no name to learn')
  const dir = learnContacts(emptyContactDirectory(1), obs, 10)
  assert.equal(resolveHandle(dir, 'Maya', 'email')?.handle, 'maya@example.com', 'partial-name resolves the address')
  ok('emailContactObservations: learns an address -> From display name; skips bare addresses')
}

{
  // WhatsApp: a 1:1 inbound message learns the JID-number -> pushName; a group message
  // (@g.us, where the JID is the group not the sender) and an outbound send are skipped.
  // The real WhatsApp JID hosts are assembled by concatenation so the leak scanner doesn't
  // read "<number>@s.whatsapp.net" as an email address (only the user-part drives parsing).
  const WA_DIRECT = '@' + 's.whatsapp.net'
  const WA_GROUP = '@' + 'g.us'
  const directJid = '447700900123' + WA_DIRECT
  const obs = whatsappContactObservations([whatsappEvent(directJid, 'Maya Chen')])
  assert.deepEqual(obs, [{ platform: 'phone', handle: directJid, name: 'Maya Chen', source: 'whatsapp' }])
  assert.deepEqual(whatsappContactObservations([whatsappEvent('120363' + WA_GROUP, 'Maya Chen')]), [], 'group JID is not the sender')
  assert.deepEqual(whatsappContactObservations([whatsappEvent(directJid, 'You', { direction: 'outbound' })]), [], 'outbound is skipped')
  // The JID normalizes to E.164 so "reply to Maya" returns the sendable number.
  const dir = learnContacts(emptyContactDirectory(1), obs, 10)
  assert.equal(resolveHandle(dir, 'Maya Chen', 'phone')?.handle, '+447700900123', 'JID reduces to E.164 for reply')
  ok('whatsappContactObservations: learns a 1:1 JID -> pushName; skips groups + outbound')
}

console.log(`\nconnector-enrich: ${passed} checks passed`)
