// Run: jiti electron/agent/connector-sync.test.ts
import assert from 'node:assert/strict'
import {
  advanceImapCursor,
  advanceSyncCursor,
  formatReplyContext,
  hasReplyContext,
  imapUidFloor,
  injectReplyContext,
  isNewerThanCursor,
  itemsNewerThanCursor,
  makeReplyContext,
} from './connector-sync.ts'
import type { SyncCursor } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('connector-sync tests:')

// ---- 1. Incremental sync cursors ----------------------------------------------

{
  // A null cursor admits everything (full resync).
  assert.equal(isNewerThanCursor({ position: 5, id: 'a' }, null), true)
  assert.equal(isNewerThanCursor({ position: 0, id: 'a' }, undefined), true)
  // Strictly-greater position is new; smaller is not.
  const cursor: SyncCursor = { position: 100, lastId: 'm100', updatedAt: 1 }
  assert.equal(isNewerThanCursor({ position: 101, id: 'm101' }, cursor), true)
  assert.equal(isNewerThanCursor({ position: 99, id: 'm99' }, cursor), false)
  // Boundary: same position is skipped ONLY for the exact cursor item; a same-position
  // sibling (different id) is still admitted so same-second messages aren't lost.
  assert.equal(isNewerThanCursor({ position: 100, id: 'm100' }, cursor), false, 'the cursor item itself is not re-ingested')
  assert.equal(isNewerThanCursor({ position: 100, id: 'm100b' }, cursor), true, 'a same-position sibling is still new')
  // A dateless (null position) item is treated as 0 -> never newer than a real cursor.
  assert.equal(isNewerThanCursor({ position: null, id: 'x' }, cursor), false)
  ok('isNewerThanCursor: strict >, boundary tie-break by id, null cursor admits all')
}

{
  // itemsNewerThanCursor filters and preserves order; null cursor passes all through.
  const items = [
    { position: 50, id: 'a' },
    { position: 100, id: 'm100' }, // the cursor item
    { position: 150, id: 'c' },
    { position: 100, id: 'sibling' }, // same second, different message
  ]
  const cursor: SyncCursor = { position: 100, lastId: 'm100', updatedAt: 1 }
  assert.deepEqual(itemsNewerThanCursor(items, cursor).map((i) => i.id), ['c', 'sibling'])
  assert.equal(itemsNewerThanCursor(items, null).length, 4)
  ok('itemsNewerThanCursor: filters newer-than, preserves order, null = all')
}

{
  // advanceSyncCursor moves the high-water mark to the batch max and never rewinds.
  const start = advanceSyncCursor(null, [
    { position: 10, id: 'a' },
    { position: 30, id: 'c' },
    { position: 20, id: 'b' },
  ], 999)
  assert.ok(start)
  assert.equal(start!.position, 30)
  assert.equal(start!.lastId, 'c', 'lastId is the newest item id')
  assert.equal(start!.updatedAt, 999)

  // A later sync that only returns OLDER items must NOT rewind the cursor.
  const noRewind = advanceSyncCursor(start, [{ position: 5, id: 'old' }], 1000)
  assert.equal(noRewind!.position, 30, 'cursor never moves backward')
  assert.equal(noRewind!.lastId, 'c')

  // A newer item advances it forward.
  const forward = advanceSyncCursor(start, [{ position: 45, id: 'd' }], 1001)
  assert.equal(forward!.position, 45)
  assert.equal(forward!.lastId, 'd')

  // An empty batch returns the SAME reference (callers skip a needless persist).
  assert.equal(advanceSyncCursor(start, [], 1002), start)
  // An empty batch with no prior cursor stays null.
  assert.equal(advanceSyncCursor(null, [], 1002), null)
  ok('advanceSyncCursor: forward-only high-water mark; empty batch is a no-op')
}

{
  // A batch whose only items are dateless (position 0) still sets a 0 cursor from null,
  // and keeps a higher existing cursor (forward-only).
  const fromNull = advanceSyncCursor(null, [{ position: null, id: 'z' }], 1)
  assert.equal(fromNull!.position, 0)
  assert.equal(fromNull!.lastId, 'z')
  const existing: SyncCursor = { position: 100, lastId: 'm', updatedAt: 1 }
  assert.equal(advanceSyncCursor(existing, [{ position: null, id: 'z' }], 2)!.position, 100)
  ok('advanceSyncCursor: dateless items clamp to 0 but never rewind a real cursor')
}

{
  // IMAP UID floor: 0 means full-resync window; a positive uid is the fetch floor.
  assert.equal(imapUidFloor(null), 0)
  assert.equal(imapUidFloor({ position: 0, lastId: null, updatedAt: 1 }), 0)
  assert.equal(imapUidFloor({ position: 0, lastId: null, lastImapUid: 42, updatedAt: 1 }), 42)
  // advanceImapCursor tracks the max UID seen and is forward-only.
  const c1 = advanceImapCursor(null, [10, 30, 20], 5)
  assert.equal(c1!.lastImapUid, 30)
  assert.equal(c1!.position, 30)
  const c2 = advanceImapCursor(c1, [25], 6) // nothing newer than 30
  assert.equal(c2!.lastImapUid, 30, 'older UIDs do not rewind the floor')
  const c3 = advanceImapCursor(c1, [55], 7)
  assert.equal(c3!.lastImapUid, 55)
  ok('imapUidFloor / advanceImapCursor: UID floor is forward-only')
}

// ---- 2. Reply-to context injection --------------------------------------------

{
  // hasReplyContext is true when there's any author or text.
  assert.equal(hasReplyContext(null), false)
  assert.equal(hasReplyContext({ id: 'x', author: null, text: null }), false)
  assert.equal(hasReplyContext({ id: null, author: 'Maya', text: null }), true)
  assert.equal(hasReplyContext({ id: null, author: null, text: 'hi' }), true)
  ok('hasReplyContext: needs author or text')
}

{
  // formatReplyContext renders author + quoted text; degrades to author-only / text-only.
  assert.equal(
    formatReplyContext({ id: '1', author: 'Maya Chen', text: 'can you confirm pricing?' }),
    'In reply to Maya Chen: "can you confirm pricing?"',
  )
  assert.equal(formatReplyContext({ id: '1', author: 'Maya', text: null }), 'In reply to Maya.')
  assert.equal(formatReplyContext({ id: '1', author: null, text: 'just text' }), 'In reply to: "just text"')
  assert.equal(formatReplyContext(null), '')
  // Multi-line quoted text is collapsed to a single line.
  assert.equal(
    formatReplyContext({ id: '1', author: 'A', text: 'line one\n\nline   two' }),
    'In reply to A: "line one line two"',
  )
  // Long quoted text is clipped with an ellipsis.
  const long = formatReplyContext({ id: '1', author: 'A', text: 'x'.repeat(500) }, 20)
  assert.ok(long.length < 60, 'long quote is clipped')
  assert.ok(long.includes('…'), 'clip adds an ellipsis')
  ok('formatReplyContext: author+text, graceful degradation, collapse + clip')
}

{
  // injectReplyContext PREPENDS the quote line above the message body.
  const out = injectReplyContext('I will send details today.', { id: '1', author: 'Maya', text: 'pricing?' })
  assert.equal(out, 'In reply to Maya: "pricing?"\nI will send details today.')
  // No reply context -> summary unchanged.
  assert.equal(injectReplyContext('body', null), 'body')
  assert.equal(injectReplyContext('body', { id: null, author: null, text: null }), 'body')
  // Idempotent: a summary that already starts with the marker is not double-injected.
  assert.equal(injectReplyContext(out, { id: '1', author: 'Maya', text: 'pricing?' }), out)
  // Empty body -> just the quote line.
  assert.equal(injectReplyContext('', { id: '1', author: 'Maya', text: null }), 'In reply to Maya.')
  ok('injectReplyContext: prepends, idempotent, no-op without reply')
}

{
  // makeReplyContext normalizes loose input and rejects content-free quotes.
  assert.equal(makeReplyContext({ id: '99', author: null, text: null }), null, 'id-only quote is not worth showing')
  assert.equal(makeReplyContext({}), null)
  const made = makeReplyContext({ id: '  99  ', author: '  Maya  ', text: '  hello\nthere  ' })
  assert.deepEqual(made, { id: '99', author: 'Maya', text: 'hello there' })
  // Author-only and text-only are both usable.
  assert.deepEqual(makeReplyContext({ author: 'Maya' }), { id: null, author: 'Maya', text: null })
  assert.deepEqual(makeReplyContext({ text: 'hi' }), { id: null, author: null, text: 'hi' })
  // Untrusted long text is clipped before it can reach an event.
  const clipped = makeReplyContext({ author: 'A', text: 'y'.repeat(5000) }, 50)
  assert.ok((clipped!.text ?? '').length <= 50)
  ok('makeReplyContext: trims/clips, drops content-free quotes')
}

console.log(`\n${passed} checks passed.`)
