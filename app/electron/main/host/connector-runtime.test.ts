// Run: jiti electron/main/host/connector-runtime.test.ts
import assert from 'node:assert/strict'
import { dedupeDiscordEvents, reconcileTelegramDraft, resolveSavedConnectorApprovalPolicy } from './connector-runtime.ts'
import type { ConnectorEvent, DiscordReadResult } from '../../agent/types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function msg(id: string, text = 'hi'): DiscordReadResult {
  return {
    id,
    sourceId: 'discord_abc',
    sourceLabel: 'Server',
    channelId: 'chan1',
    channelName: 'general',
    threadId: null,
    threadName: null,
    guildName: 'Guild',
    senderId: 'u1',
    senderName: 'Alice',
    isBot: false,
    date: 1700000000000,
    text,
    snippet: text,
    replyTo: null,
    threadKey: 'discord:discord_abc:chan1',
    raw: {},
  }
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((res) => { resolve = res })
  return { promise, resolve }
}

// 1) First sync of fresh messages emits one event per message and stamps the id.
{
  const messages = [msg('100'), msg('101'), msg('102')]
  const { events, cursor } = dedupeDiscordEvents(messages, [])
  assert.equal(events.length, 3, 'all three new messages become events')
  assert.deepEqual(events.map((e) => e.payload?.messageId).sort(), ['100', '101', '102'])
  for (const e of events) assert.ok(e.payloadKeys.includes('messageId'), 'messageId in payloadKeys')
  // The cursor advances to the newest message's position (date) + id.
  assert.ok(cursor, 'a cursor is returned')
  assert.equal(cursor!.position, 1700000000000)
  ok('first sync emits one event per message and stamps messageId')
}

// 2) THE BUG: a re-poll returning the same ~50 messages must NOT re-emit them.
//    Discord returns the most recent messages every time; this is the regression guard.
{
  const messages = [msg('100'), msg('101'), msg('102')]
  const firstPass = dedupeDiscordEvents(messages, [])
  // firstPass.events is now what lives in store.connectorEvents.
  const secondPass = dedupeDiscordEvents(messages, firstPass.events)
  assert.equal(secondPass.events.length, 0, 'second sync of identical messages emits zero new events')
  ok('re-poll of identical messages produces no duplicates')
}

// 2b) INCREMENTAL: with a cursor at the last sync, only strictly-newer messages pass —
//     before the id-set dedup even runs. A newer-dated message advances the cursor.
{
  const first = dedupeDiscordEvents([msg('100')], [])
  // A later message (newer date) in the same/next poll, passing the prior cursor.
  const newer = { ...msg('200', 'later'), date: 1700000005000 }
  const second = dedupeDiscordEvents([msg('100'), newer], [], first.cursor)
  assert.equal(second.events.length, 1, 'only the newer-than-cursor message is ingested')
  assert.equal(second.events[0]!.payload?.messageId, '200')
  assert.equal(second.cursor!.position, 1700000005000, 'cursor advanced to the newest date')
  ok('cursor filters to strictly-newer messages and advances forward')
}

// 3) An ongoing conversation: new messages in an ALREADY-seen channel still come through.
//    (Guards against the per-channel/threadKey collapse seen in the iMessage bug.)
{
  const existing = dedupeDiscordEvents([msg('100'), msg('101')], [])
  // Same channel/threadKey, but two genuinely new message ids arrive.
  const next = dedupeDiscordEvents([msg('100'), msg('101'), msg('102'), msg('103')], existing.events)
  assert.equal(next.events.length, 2, 'only the two new messages are ingested')
  assert.deepEqual(next.events.map((e) => e.payload?.messageId).sort(), ['102', '103'])
  ok('new messages in a previously-seen channel are not collapsed away')
}

// 3b) Reply context: a Discord reply carries the quoted author + text into summary + reply.
{
  const reply: DiscordReadResult = { ...msg('300', 'on it'), replyTo: '299', replyToAuthor: 'Bob', replyToText: 'can you deploy?' }
  const { events } = dedupeDiscordEvents([reply], [])
  assert.equal(events.length, 1)
  assert.equal(events[0]!.reply?.author, 'Bob')
  assert.equal(events[0]!.reply?.text, 'can you deploy?')
  assert.match(events[0]!.summary, /In reply to Bob: "can you deploy\?"/)
  assert.match(events[0]!.summary, /on it/)
  ok('a reply folds the quoted author + text into summary and the reply field')
}

// 4) Non-discord events in the store are ignored when building the known set.
{
  const foreign: ConnectorEvent = {
    id: 'x', connectorId: 'slack', pluginName: 'Slack', action: 'message_received',
    direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify',
    requiresApproval: false, approvalId: null, title: 't', summary: 's', actor: 'a',
    threadKey: 'slack:s:100', payloadKeys: ['messageId'], payload: { messageId: '100' }, createdAt: 1,
  }
  // A slack event also carrying messageId '100' must not suppress a discord message '100'.
  const { events } = dedupeDiscordEvents([msg('100')], [foreign])
  assert.equal(events.length, 1, 'foreign-connector messageId does not block discord ingest')
  ok('dedup is scoped to discord events only')
}

{
  const calls: string[] = []
  const result = await reconcileTelegramDraft('10', 'final', {
    edit: async () => { calls.push('edit'); return false },
    remove: async () => { calls.push('delete'); return true },
    send: async () => { calls.push('send'); return '11' },
  })
  assert.deepEqual(calls, ['edit', 'delete', 'send'], 'failed draft edit is removed before the fallback send')
  assert.equal(result.sentId, '11')

  let sent = false
  const unreconciled = await reconcileTelegramDraft('12', 'final', {
    edit: async () => false,
    remove: async () => false,
    send: async () => { sent = true; return '13' },
  })
  assert.equal(sent, false, 'a fresh final is not posted beside an undeletable placeholder')
  assert.match(unreconciled.error ?? '', /no duplicate/i)
  ok('Telegram finalization reconciles or fails without placeholder duplication')
}

// Saved connector decisions are asynchronous because bounded-rule consumption is a critical
// disk write. The event must remain pending until that promise returns a durable match.
{
  const gate = deferred<{ decision: 'deny' } | null>()
  const event: ConnectorEvent = {
    id: 'approval-1', connectorId: 'calendar', pluginName: 'Calendar', action: 'event_delete',
    direction: 'outbound', classification: 'approval', priority: 'normal', trustLevel: 'approve',
    requiresApproval: true, approvalId: 'approval-1', approvalStatus: 'pending', title: 'Delete event',
    summary: 'Delete standup', actor: 'user@example.test', threadKey: 'calendar:primary',
    payloadKeys: ['eventId'], payload: { eventId: 'evt-1' }, createdAt: 1,
  }
  let settled = false
  const resolution = resolveSavedConnectorApprovalPolicy(event, {
    findApprovalRule: () => gate.promise,
    now: () => 2,
  }).then((value) => { settled = true; return value })
  await Promise.resolve()
  assert.equal(settled, false, 'connector event does not apply a rule before durable lookup completes')
  assert.equal(event.approvalStatus, 'pending', 'the caller-owned event is not mutated while waiting')
  gate.resolve({ decision: 'deny' })
  const denied = await resolution
  assert.equal(denied.approvalStatus, 'denied')
  assert.equal(denied.requiresApproval, false)
  assert.equal(denied.resolvedBy, 'saved-policy')
  ok('saved connector policy is awaited before it is applied')
}

console.log(`\nconnector-runtime: ${passed} checks passed`)
