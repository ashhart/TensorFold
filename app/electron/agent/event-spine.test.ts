// Run: jiti electron/agent/event-spine.test.ts
//
// Event spine v2 (VST-034): global ordering, per-stream versions, idempotent appends,
// exactly-once command receipts, cursor-based replay, crash-resume, compaction watermark.
import assert from 'node:assert/strict'
import { createEventSpine, type SpineEvent } from './event-spine'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('event-spine tests:')

const fixedNow = () => 1_700_000_000_000
let idCounter = 0
const newSpine = () => { idCounter = 0; return createEventSpine({ now: fixedNow, newEventId: () => `e${++idCounter}` }) }

// --- ordering: global sequence + per-stream versions ---
{
  const spine = newSpine()
  const a1 = spine.append({ aggregate: 'thread', streamId: 'A', type: 'thread.created', payload: {} })
  const b1 = spine.append({ aggregate: 'thread', streamId: 'B', type: 'thread.created', payload: {} })
  const a2 = spine.append({ aggregate: 'thread', streamId: 'A', type: 'thread.turn-started', payload: {} })
  const t1 = spine.append({ aggregate: 'terminal', streamId: 'A', type: 'terminal.opened', payload: {} })
  assert.deepEqual([a1.sequence, b1.sequence, a2.sequence, t1.sequence], [1, 2, 3, 4], 'global sequence is monotonic')
  assert.deepEqual([a1.streamVersion, a2.streamVersion], [1, 2], 'stream versions count per stream')
  assert.equal(b1.streamVersion, 1)
  assert.equal(t1.streamVersion, 1, 'streams are scoped by aggregate+id, not id alone')
  assert.equal(spine.headSequence, 4)
  ok('append: global sequence + per-(aggregate,stream) versions')
}

// --- idempotent redelivery ---
{
  const spine = newSpine()
  const first = spine.append({ aggregate: 'job', streamId: 'j1', type: 'job.queued', payload: { n: 1 }, eventId: 'fixed-id' })
  const dup = spine.append({ aggregate: 'job', streamId: 'j1', type: 'job.queued', payload: { n: 2 }, eventId: 'fixed-id' })
  assert.equal(dup, first, 'duplicate eventId returns the stored fact')
  assert.equal(spine.headSequence, 1, 'no second sequence burned')
  assert.equal((dup.payload as any).n, 1, 'the original payload wins')
  ok('append: idempotent by eventId')
}

// --- command receipts: exactly-once, atomic ---
{
  const spine = newSpine()
  let ran = 0
  const receipt = spine.execute({ commandId: 'cmd-1', aggregate: 'thread', streamId: 'A', kind: 'thread.start-turn' }, () => {
    ran++
    return [
      { aggregate: 'thread', streamId: 'A', type: 'thread.turn-started', payload: {} },
      { aggregate: 'thread', streamId: 'A', type: 'thread.message-sent', payload: {} },
    ]
  })
  assert.equal(receipt.status, 'accepted')
  assert.equal(receipt.resultSequence, 2, 'receipt points at the last recorded event')
  const retry = spine.execute({ commandId: 'cmd-1', aggregate: 'thread', streamId: 'A', kind: 'thread.start-turn' }, () => {
    ran++
    return [{ aggregate: 'thread', streamId: 'A', type: 'thread.turn-started', payload: {} }]
  })
  assert.equal(ran, 1, 'the handler never re-runs for a repeated commandId')
  assert.equal(retry, receipt, 'the cached receipt is returned verbatim')
  assert.equal(spine.headSequence, 2, 'no duplicate events from the retry')
  ok('execute: exactly-once per commandId with cached receipt')
}

{
  const spine = newSpine()
  const receipt = spine.execute({ commandId: 'boom', aggregate: 'job', streamId: 'j1', kind: 'job.run' }, () => {
    throw new Error('validation failed: no provider')
  })
  assert.equal(receipt.status, 'rejected')
  assert.equal(receipt.error, 'validation failed: no provider')
  assert.equal(receipt.resultSequence, null)
  assert.equal(spine.headSequence, 0, 'a rejected command records NO events (atomic)')
  assert.equal(spine.receipt('boom')?.status, 'rejected', 'rejection is remembered — a retry with the same id stays rejected')
  ok('execute: a throwing handler yields a rejected receipt and no events')
}

{
  const spine = newSpine()
  spine.execute({ commandId: 'c-actor', aggregate: 'room', streamId: 'r1', kind: 'room.note', actor: 'user', correlationId: 'corr-9' }, () => (
    { aggregate: 'room', streamId: 'r1', type: 'room.note-added', payload: {} }
  ))
  const event = spine.readSince(0).events[0]
  assert.equal(event.commandId, 'c-actor', 'events carry their causing command')
  assert.equal(event.actor, 'user', 'command actor flows onto events')
  assert.equal(event.correlationId, 'corr-9', 'command correlation flows onto events')
  ok('execute: command id/actor/correlation stamp the recorded events')
}

// --- replay: readSince + filters + paging ---
{
  const spine = newSpine()
  for (let i = 0; i < 5; i++) spine.append({ aggregate: 'thread', streamId: 'A', type: 't', payload: { i } })
  spine.append({ aggregate: 'job', streamId: 'j', type: 'j', payload: {} })
  const all = spine.readSince(2)
  assert.deepEqual(all.events.map((event) => event.sequence), [3, 4, 5, 6], 'replay is exclusive of the cursor')
  const paged = spine.readSince(0, { limit: 3 })
  assert.equal(paged.events.length, 3)
  assert.equal(paged.hasMore, true, 'paging reports more past the page')
  const filtered = spine.readSince(0, { aggregate: 'job' })
  assert.deepEqual(filtered.events.map((event) => event.type), ['j'], 'aggregate filter')
  assert.equal(spine.readSince(6).events.length, 0)
  ok('readSince: cursor-exclusive, filtered, paged replay')
}

// --- projector cursors: catch-up, retry-at-failure, monotonic ---
{
  const spine = newSpine()
  for (let i = 1; i <= 4; i++) spine.append({ aggregate: 'thread', streamId: 'A', type: `t${i}`, payload: {} })
  const seen: string[] = []
  let failOnce = true
  const first = spine.project('ui', (event) => {
    if (event.type === 't3' && failOnce) { failOnce = false; throw new Error('transient') }
    seen.push(event.type)
  })
  assert.equal(first.applied, 2, 'catch-up stops AT the failing event')
  assert.equal(first.cursor, 2, 'cursor stays before the failure for retry')
  const second = spine.project('ui', (event) => seen.push(event.type))
  assert.equal(second.applied, 2, 'retry resumes from the failure point')
  assert.deepEqual(seen, ['t1', 't2', 't3', 't4'], 'every event applied exactly once, in order')
  assert.equal(spine.advance('ui', 1), 4, 'cursors never move backwards')
  ok('project: cursor catch-up with retry-at-failure, monotonic advance')
}

// --- live subscribe: bad subscriber isolation ---
{
  const spine = newSpine()
  const seen: number[] = []
  spine.subscribe(() => { throw new Error('bad subscriber') })
  spine.subscribe((event) => seen.push(event.sequence))
  spine.append({ aggregate: 'thread', streamId: 'A', type: 't', payload: {} })
  assert.deepEqual(seen, [1], 'a throwing subscriber never breaks the fan-out')
  ok('subscribe: subscriber isolation')
}

// --- crash-resume: snapshot + restore ---
{
  const spine = newSpine()
  for (let i = 1; i <= 3; i++) spine.append({ aggregate: 'thread', streamId: 'A', type: `t${i}`, payload: {} })
  spine.execute({ commandId: 'c1', aggregate: 'thread', streamId: 'A', kind: 'k' }, () => (
    { aggregate: 'thread', streamId: 'A', type: 't4', payload: {} }
  ))
  spine.advance('ui', 2)
  const state = spine.snapshotState()

  const restored = createEventSpine({ now: fixedNow, newEventId: () => `r${++idCounter}` }, state)
  assert.equal(restored.headSequence, 4, 'head restores')
  assert.equal(restored.cursor('ui'), 2, 'cursors restore')
  assert.equal(restored.receipt('c1')?.status, 'accepted', 'receipts restore (retries stay exactly-once across restarts)')
  const next = restored.append({ aggregate: 'thread', streamId: 'A', type: 't5', payload: {} })
  assert.equal(next.sequence, 5, 'sequences continue, never reused')
  assert.equal(next.streamVersion, 5, 'stream versions continue')
  const caught = restored.readSince(restored.cursor('ui'))
  assert.deepEqual(caught.events.map((event) => event.type), ['t3', 't4', 't5'], 'a restored projector catches up from its cursor')
  ok('restore: head/cursors/receipts survive; replay resumes cleanly')
}

// --- compaction watermark ---
{
  const spine = newSpine()
  for (let i = 1; i <= 6; i++) spine.append({ aggregate: 'thread', streamId: 'A', type: `t${i}`, payload: {} })
  spine.advance('ui', 4)
  spine.advance('remote', 3)
  const watermark = spine.compactThrough(5)
  assert.equal(watermark, 3, 'compaction never passes the slowest cursor')
  assert.equal(spine.readSince(3).snapshotRequired, false, 'reading from the watermark is fine')
  const stale = spine.readSince(0)
  assert.equal(stale.snapshotRequired, true, 'reading below the watermark demands a snapshot re-baseline')
  assert.deepEqual(stale.events.map((event) => event.sequence), [4, 5, 6], 'only surviving events return')
  const noCursorRegression = spine.compactThrough(2)
  assert.equal(noCursorRegression, 3, 'the watermark never moves backwards')
  ok('compactThrough: bounded by the slowest cursor, watermark flags stale readers')
}

console.log(`\n${passed} passed`)
