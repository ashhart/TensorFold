// Run: jiti electron/agent/imessage-connector.test.ts
import assert from 'node:assert/strict'
import { appleDateToMs, imessageEventFromMessage, imessageMessageKey, imessageConversationKey, imessageEventDedupKey, imessageAccessState, buildIMessageSendScript, isAllowedMessage, isIMessageGroupConversation, parseAssociatedGuid, imessageEventTimestamp } from './imessage-connector.ts'
import type { ConnectorEvent } from './types.ts'
import type { IMessageItem } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function message(partial: Partial<IMessageItem>): IMessageItem {
  return {
    id: '101', handle: '+15551234567', isFromMe: false, text: 'hello',
    date: Date.now(), hasAttachments: false, chatId: 'chatA', chatName: 'Family',
    ...partial,
  }
}

console.log('imessage-connector tests:')

{
  // A message you sent -> outbound event (blue bubble).
  const sent = imessageEventFromMessage(message({ isFromMe: true, text: 'see you at 3' }), 1000)
  assert.equal(sent.direction, 'outbound')
  assert.equal(sent.action, 'message_sent')
  assert.equal(sent.connectorId, 'imessage')
  assert.equal(sent.requiresApproval, false) // reading is never approval-gated
  // threadKey is CONVERSATION-scoped (the Assistant groups a chat's events by it and
  // replies target it); per-message identity lives in payload.messageId for dedup.
  assert.equal(sent.threadKey, 'imessage:chatA')
  assert.equal((sent.payload as { messageId?: unknown }).messageId, '101')
  assert.equal(imessageEventDedupKey(sent), 'imessage:msg:101')
  // Conversation key falls back to the handle when there is no chat id.
  assert.equal(imessageConversationKey({ chatId: '', handle: '+15550000000' }), 'imessage:+15550000000')

  // A message from someone else -> inbound event (gray bubble). The actor prefers
  // the chat/contact name ("Family") over the raw handle now that contact names
  // are resolved (Bug 2 fix).
  const recv = imessageEventFromMessage(message({ isFromMe: false, text: 'are we still on?' }), 2000)
  assert.equal(recv.direction, 'inbound')
  assert.equal(recv.action, 'message_received')
  assert.equal(recv.actor, 'Family')
  // With no chat name, the actor falls back to the raw handle.
  const noName = imessageEventFromMessage(message({ isFromMe: false, chatName: '', handle: '+15559998888' }), 2000)
  assert.equal(noName.actor, '+15559998888')
  ok('imessageEventFromMessage: isFromMe drives outbound/inbound direction; actor prefers contact name')
}

{
  // Two messages in the SAME conversation share ONE conversation threadKey (so the
  // Assistant groups them) but have DISTINCT per-message dedup keys.
  const first = imessageEventFromMessage(message({ id: '201', chatId: 'chatA' }))
  const second = imessageEventFromMessage(message({ id: '202', chatId: 'chatA' }))
  assert.equal(first.threadKey, 'imessage:chatA')
  assert.equal(second.threadKey, 'imessage:chatA')
  assert.equal(first.threadKey, second.threadKey, 'same conversation -> same threadKey (grouping)')
  assert.equal(imessageEventDedupKey(first), 'imessage:msg:201')
  assert.equal(imessageEventDedupKey(second), 'imessage:msg:202')
  assert.notEqual(imessageEventDedupKey(first), imessageEventDedupKey(second), 'distinct per-message dedup keys')
  ok('imessageEventFromMessage: conversation-scoped threadKey, per-message dedup key')
}

{
  // Regression for the "ongoing conversation drops every follow-up" bug: replicate
  // syncIMessageConnector's per-message dedup against `known` (which the runtime now
  // builds via imessageEventDedupKey). A chat that already surfaced ONE event must
  // still ingest its later messages, while all events keep the SAME conversation key.
  const dedup = (msgs: ReturnType<typeof message>[], known: Set<string>) => {
    const seen = new Set(known)
    const out: ConnectorEvent[] = []
    for (const m of msgs) {
      const key = imessageMessageKey(m)
      if (seen.has(key)) continue
      seen.add(key)
      out.push(imessageEventFromMessage(m))
    }
    return out
  }

  // First sync: one inbound message from chatA -> one event.
  const firstSync = dedup([message({ id: '301', chatId: 'chatA' })], new Set())
  assert.equal(firstSync.length, 1)
  // The runtime rebuilds `known` from the per-message dedup key, NOT threadKey.
  const known = new Set(firstSync.map(imessageEventDedupKey))
  assert.deepEqual([...known], ['imessage:msg:301'])

  // Contact sends 3 MORE messages in the same chat. Second sync returns only the 3
  // genuinely-new ones (the scan re-returns 301, which is deduped).
  const secondSync = dedup([
    message({ id: '301', chatId: 'chatA' }),
    message({ id: '302', chatId: 'chatA' }),
    message({ id: '303', chatId: 'chatA' }),
    message({ id: '304', chatId: 'chatA' }),
  ], known)
  // Before the fix this was 0 (the whole conversation was skipped on a conversation
  // key); now the 3 new messages are ingested and the already-seen one is not.
  assert.equal(secondSync.length, 3)
  assert.deepEqual(secondSync.map(imessageEventDedupKey), ['imessage:msg:302', 'imessage:msg:303', 'imessage:msg:304'])
  // Grouping preserved: every event in the conversation shares one threadKey.
  assert.ok(secondSync.every((e) => e.threadKey === 'imessage:chatA'), 'all events keep the conversation threadKey')
  ok('ongoing conversation: later messages ingested (per-message dedup), grouping preserved')
}

{
  // Access state on non-darwin is reported as unavailable (the unit-test env).
  const state = await imessageAccessState()
  // On macOS CI/this machine this can be 'needs_full_disk_access' or 'ok'; on
  // non-mac it's 'no_macos'. Either way it never throws.
  assert.ok(['ok', 'needs_full_disk_access', 'missing_db', 'no_macos', 'error'].includes(state.state))
  assert.equal(typeof state.detail, 'string')
  ok(`imessageAccessState returns a valid state without throwing (got: ${state.state})`)
}

{
  // normal send: handle + text appear as AppleScript string literals
  const s = buildIMessageSendScript('+15551234567', 'Hi there')
  assert.ok(s.includes('participant "+15551234567" of targetService'))
  assert.ok(s.includes('send "Hi there" to targetBuddy'))
  ok('buildIMessageSendScript builds a Messages send for a plain handle + text')
}
{
  // Group conversations are chat.db chat identifiers, not participant handles.
  // Route to the existing text chat or Messages.app cannot resolve the target.
  const groupId = 'chat641900287271320631'
  assert.equal(isIMessageGroupConversation(groupId), true)
  assert.equal(isIMessageGroupConversation('iMessage;+;chat641900287271320631'), true)
  assert.equal(isIMessageGroupConversation('+15551234567'), false)
  assert.equal(isIMessageGroupConversation('person@example.com'), false)
  const s = buildIMessageSendScript(groupId, 'Hello group')
  assert.ok(s.includes('set targetChat to text chat id "chat641900287271320631"'))
  assert.ok(s.includes('send "Hello group" to targetChat'))
  assert.ok(!s.includes('participant'), 'group chat id must never be treated as a buddy handle')
  ok('buildIMessageSendScript routes group chat identifiers to an existing text chat')
}
{
  // injection defense: a body trying to break out of the string is fully escaped
  const evil = 'bye"\nend tell\ndo shell script "rm -rf ~"'
  const s = buildIMessageSendScript('+1555', evil)
  // the raw breakout sequence must NOT appear; the quote/newline are escaped
  assert.ok(!s.includes('"\nend tell'), 'raw quote+newline breakout is neutralised')
  assert.ok(s.includes('\\"') && s.includes('\\n'), 'quote and newline are escaped inside the literal')
  // exactly one `send ... to targetBuddy` line — the payload cannot inject extra statements
  assert.equal((s.match(/send .* to targetBuddy/g) || []).length, 1)
  ok('buildIMessageSendScript escapes quotes/newlines so a message body cannot inject AppleScript')
}
{
  // backslashes are doubled
  assert.ok(buildIMessageSendScript('h', 'a\\b').includes('a\\\\b'))
  ok('buildIMessageSendScript escapes backslashes')
}

{
  // Allowlist gate (scanIMessageMessages line 169). Before the fix, the filter was
  // `!m.chatId || allowed.has(m.chatId)`, so ANY message with an empty chat_identifier
  // (chat_message_join is a LEFT JOIN, so missing chat rows -> chatId '') bypassed the
  // allowlist unconditionally. It also only matched the chat id, never the sender handle.
  const allowed = new Set(['chatA', '+15551234567'])

  // No allowlist configured -> everything passes.
  assert.equal(isAllowedMessage({ chatId: '', handle: '+19999999999' }, new Set()), true)

  // Listed chat id passes.
  assert.equal(isAllowedMessage({ chatId: 'chatA', handle: '+10000000000' }, allowed), true)
  // Listed sender handle passes even when the chat id is not listed (gate by contact).
  assert.equal(isAllowedMessage({ chatId: 'chatZ', handle: '+15551234567' }, allowed), true)
  // Unlisted chat id AND unlisted handle -> blocked.
  assert.equal(isAllowedMessage({ chatId: 'chatZ', handle: '+19999999999' }, allowed), false)

  // THE BUG: empty chatId must NOT auto-pass when an allowlist is configured.
  assert.equal(isAllowedMessage({ chatId: '', handle: '+19999999999' }, allowed), false, 'empty chatId no longer bypasses the allowlist')
  // An empty-chatId message whose handle IS listed still passes (it is genuinely allowed).
  assert.equal(isAllowedMessage({ chatId: '', handle: '+15551234567' }, allowed), true)
  // Both empty -> blocked.
  assert.equal(isAllowedMessage({ chatId: '', handle: '' }, allowed), false)
  ok('isAllowedMessage: empty chatId no longer bypasses the allowlist; gates by chat id OR sender handle')
}

{
  // parseAssociatedGuid pulls the bare GUID out of chat.db's `bp:`/`p:0/` reply prefixes
  // (and a bare guid), and returns null for non-reply / empty values.
  const guid = 'ABCDEF01-2345-6789-ABCD-EF0123456789'
  assert.equal(parseAssociatedGuid(`bp:${guid}`), guid)
  assert.equal(parseAssociatedGuid(`p:0/${guid}`), guid)
  assert.equal(parseAssociatedGuid(guid), guid)
  assert.equal(parseAssociatedGuid(''), null)
  assert.equal(parseAssociatedGuid(null), null)
  ok('parseAssociatedGuid extracts the replied-to GUID from chat.db prefixes')
}

{
  // REPLY-TO CONTEXT: a reply event folds the quoted author + text into summary + reply.
  const reply = imessageEventFromMessage(
    message({ id: '501', text: 'sounds good', isFromMe: false }),
    9000,
    { id: 'guid-parent', author: 'Maya Chen', text: 'are we still on for 3?' },
  )
  assert.equal(reply.reply?.author, 'Maya Chen')
  assert.equal(reply.reply?.text, 'are we still on for 3?')
  assert.match(reply.summary, /In reply to Maya Chen: "are we still on for 3\?"/)
  assert.match(reply.summary, /sounds good/)
  assert.ok(reply.payloadKeys.includes('reply'))
  // No reply context -> summary is just the message text, no reply field.
  const plain = imessageEventFromMessage(message({ id: '502', text: 'hello' }), 9001)
  assert.equal(plain.reply, undefined)
  assert.equal(plain.summary, 'hello')
  ok('imessageEventFromMessage injects quoted reply context into summary + reply field')
}

{
  const appleEpochMs = 978_307_200_000
  assert.equal(appleDateToMs(800_000_000), appleEpochMs + 800_000_000_000, 'legacy seconds convert to Unix milliseconds')
  assert.equal(appleDateToMs(800_000_000_000_000_000), appleEpochMs + 800_000_000_000, 'modern nanoseconds convert to Unix milliseconds')
  assert.equal(appleDateToMs(Number.NaN), null)
  ok('appleDateToMs handles legacy seconds and modern chat.db nanoseconds')
}

{
  // NEWEST-FIRST: the event's createdAt is the real MESSAGE time, not the ingest time, so the
  // Assistant thread list (sorted by createdAt desc) shows the newest conversation at the top.
  const ingest = 5_000_000
  const older = imessageEventFromMessage(message({ id: '601', date: 1_000_000 }), ingest)
  const newer = imessageEventFromMessage(message({ id: '602', date: 2_000_000 }), ingest)
  assert.equal(older.createdAt, 1_000_000, 'createdAt is the message date, not the shared ingest time')
  assert.equal(newer.createdAt, 2_000_000)
  // Sorting a batch (all ingested at the same Date.now()) by createdAt desc puts the newest first.
  const sorted = [older, newer].sort((a, b) => b.createdAt - a.createdAt)
  assert.deepEqual(sorted.map((e) => e.payload?.messageId), ['602', '601'], 'newest message floats to the top')

  // imessageEventTimestamp: real date preferred; missing/garbage/zero falls back to `now`;
  // a future date is clamped to now so a bad clock can't pin a row to the top forever.
  const now = 10_000
  assert.equal(imessageEventTimestamp({ date: 4_000 }, now), 4_000, 'uses the message date')
  assert.equal(imessageEventTimestamp({ date: null }, now), now, 'null date falls back to now')
  assert.equal(imessageEventTimestamp({ date: 0 }, now), now, 'zero date falls back to now')
  assert.equal(imessageEventTimestamp({ date: Number.NaN }, now), now, 'NaN date falls back to now')
  assert.equal(imessageEventTimestamp({ date: now + 9_999 }, now), now, 'future date is clamped to now')
  ok('imessageEventTimestamp / imessageEventFromMessage: createdAt is the message time so the thread list is newest-first')
}

console.log(`\n${passed} checks passed.`)
