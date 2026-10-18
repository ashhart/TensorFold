// Run: jiti src/components/assistant/conversation.test.ts
import assert from 'node:assert/strict'
import { eventDirection, eventToBubble, timelineItemToBubble, mergeBubbles, groupEventsIntoThreads, combineItemsByThread } from './conversation.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('assistant conversation bubble tests:')

{
  // iMessage `is_from_me` is the strongest signal: overrides direction/action.
  assert.equal(eventDirection({ isFromMe: true, direction: 'inbound' }), 'out')
  assert.equal(eventDirection({ isFromMe: false, direction: 'inbound' }), 'in')
  ok('isFromMe wins over direction (iMessage chat.db semantics)')
}

{
  // Outbound direction / send/reply/draft action -> blue (out).
  assert.equal(eventDirection({ direction: 'outbound' }), 'out')
  assert.equal(eventDirection({ action: 'reply_requested' }), 'out')
  assert.equal(eventDirection({ action: 'send_requested' }), 'out')
  assert.equal(eventDirection({ action: 'draft_created' }), 'out')
  // Plain inbound message with no outbound signal -> gray (in).
  assert.equal(eventDirection({ direction: 'inbound', action: 'message_received' }), 'in')
  assert.equal(eventDirection({}), 'in')
  ok('eventDirection classifies outbound vs inbound from direction/action')
}

{
  // An inbound email message -> gray bubble (in).
  const inb = eventToBubble({ id: 'e1', actor: 'Maya', summary: 'Can we move the call?', action: 'message_received', direction: 'inbound', createdAt: 1000 })
  assert.equal(inb.direction, 'in')
  assert.equal(inb.author, 'Maya')
  assert.equal(inb.text, 'Can we move the call?')
  assert.equal(inb.status, undefined)

  // A reply the agent drafted, not yet approved -> blue (out), status draft.
  const draft = eventToBubble({ id: 'e2', actor: 'You', summary: '3pm works.', action: 'reply_requested', direction: 'outbound', approvalStatus: 'pending', createdAt: 2000 })
  assert.equal(draft.direction, 'out')
  assert.equal(draft.status, 'draft')

  // A real transport receipt, rather than approval alone, flips it to 'sent'.
  const sent = eventToBubble({ id: 'e2', actor: 'You', summary: '3pm works.', action: 'send_requested', direction: 'outbound', approvalStatus: 'approved', delivery: { transport: 'fixture', status: 'delivered' }, createdAt: 2000 })
  assert.equal(sent.direction, 'out')
  assert.equal(sent.status, 'sent')
  ok('eventToBubble maps inbound->gray, agent drafts->blue/draft, delivered->blue/sent')
}

{
  // Agent chat timeline: user/cue -> blue out, assistant -> gray in, tool rows dropped.
  const user = timelineItemToBubble({ id: 'u1', kind: 'user', text: 'draft a reply', createdAt: 3000 })
  assert.equal(user?.direction, 'out')
  assert.equal(user?.author, 'You')
  const asst = timelineItemToBubble({ id: 'a1', kind: 'assistant', text: 'Here is a draft.', createdAt: 3100 })
  assert.equal(asst?.direction, 'in')
  assert.equal(asst?.author, 'TensorFold')
  assert.equal(timelineItemToBubble({ id: 't1', kind: 'tool', text: 'read_file' }), null)
  assert.equal(timelineItemToBubble({ id: 'n1', kind: 'notice', text: 'compacting' }), null)
  ok('timelineItemToBubble: user/cue out, assistant in, non-chat rows dropped')
}

{
  // Merge keeps source messages and agent chat in one time-ordered column.
  const merged = mergeBubbles([
    { id: 'a', direction: 'out', author: 'You', text: 'late', sort: 3000 },
    { id: 'b', direction: 'in', author: 'Maya', text: 'first', sort: 1000 },
    { id: 'c', direction: 'in', author: 'TensorFold', text: 'middle', sort: 2000 },
  ])
  assert.deepEqual(merged.map((m) => m.id), ['b', 'c', 'a'])
  ok('mergeBubbles time-orders the unified column')
}

{
  // groupEventsIntoThreads collapses a conversation (same threadKey) into ONE row —
  // the newest message — with a count, and orders threads newest-first. This is the
  // fix for "the same iMessage number shows as 24 separate threads".
  const ev = (id: string, threadKey: string, createdAt: number) => ({ id, threadKey, createdAt })
  const events = [
    ev('m1', 'imessage:+447700900500', 100),
    ev('m2', 'imessage:+447700900500', 300), // newest in this chat
    ev('m3', 'imessage:+447700900500', 200),
    ev('e1', 'email:renewal', 250),
    ev('x1', '', 400), // no threadKey -> its own thread (keyed by id)
  ]
  const threads = groupEventsIntoThreads(events)
  // 3 threads: the imessage chat (1 row, count 3), the email thread, and the keyless event.
  assert.equal(threads.length, 3)
  // newest-first by representative time: x1(400) > imessage rep m2(300) > email(250).
  assert.deepEqual(threads.map((t) => t.event.id), ['x1', 'm2', 'e1'])
  const chat = threads.find((t) => t.event.threadKey === 'imessage:+447700900500')!
  assert.equal(chat.count, 3, 'all 3 messages from the number collapse into one thread')
  assert.equal(chat.event.id, 'm2', 'representative is the newest message')
  assert.equal(threads.find((t) => t.event.id === 'x1')!.count, 1)
  ok('groupEventsIntoThreads: one row per conversation (newest + count), newest-first')
}

{
  // The Assistant "attention" list emits one item per message; without combining,
  // an iMessage conversation shows as many separate thread rows (the reported bug).
  const items: Array<{ id: string; threadKey?: string; count?: number; from: string }> = [
    { id: 'a3', threadKey: 'imessage:+447700900502', from: 'You' },
    { id: 'a2', threadKey: 'imessage:+447700900502', from: '+447700900502' },
    { id: 'a1', threadKey: 'imessage:+447700900502', from: '+447700900502' },
    { id: 'e1', threadKey: 'email:42', from: 'finance@example.com' },
    { id: 'd1', from: 'no-thread' }, // passes through untouched
  ]
  const combined = combineItemsByThread(items)
  assert.equal(combined.length, 3, 'three iMessages collapse to one row; email + no-thread stay separate')
  const chat = combined.find((c) => c.threadKey === 'imessage:+447700900502')!
  assert.equal(chat.count, 3, 'all three messages fold into the thread count')
  assert.equal(chat.id, 'imessage:+447700900502', 'row id is the stable threadKey')
  assert.equal(chat.from, 'You', 'keeps the first (newest) item as the preview')
  assert.equal(combined.find((c) => c.id === 'd1')!.from, 'no-thread', 'items without a threadKey pass through')
  ok('combineItemsByThread: one row per conversation (folds count), threadKey-less pass through')
}

console.log(`\n${passed} checks passed.`)
