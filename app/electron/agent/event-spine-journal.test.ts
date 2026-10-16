// Run: jiti electron/agent/event-spine-journal.test.ts
import assert from 'node:assert/strict'
import { createEventSpine, type SpineEvent } from './event-spine'
import { encodeSpineLine, parseSpineJournal, readSpineJournal, rewriteSpineJournal, spineJournalText, spineJournalWriter } from './event-spine-journal'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('event-spine-journal tests:')

const fixedNow = () => 1_700_000_000_000
let idCounter = 0
const newSpine = () => { idCounter = 0; return createEventSpine({ now: fixedNow, newEventId: () => `e${++idCounter}` }) }

// In-memory fs with append/read/write/rename, plus injectable append failures.
function memFs(failAppends = 0) {
  const files = new Map<string, string>()
  let failures = failAppends
  return {
    files,
    async appendFile(file: string, data: string) {
      if (failures > 0) { failures--; throw new Error('disk full') }
      files.set(file, (files.get(file) ?? '') + data)
    },
    async readFile(file: string) {
      const content = files.get(file)
      if (content == null) throw new Error('ENOENT')
      return content
    },
    async writeFile(file: string, data: string) { files.set(file, data) },
    async rename(from: string, to: string) {
      const content = files.get(from)
      if (content == null) throw new Error('ENOENT')
      files.set(to, content)
      files.delete(from)
    },
  }
}

// --- live tail → journal → restore roundtrip ---
{
  const spine = newSpine()
  const fs = memFs()
  const writer = spineJournalWriter(fs, '/spine.jsonl')
  spine.subscribe(writer.onEvent)

  spine.append({ aggregate: 'thread', streamId: 'A', type: 'thread.created', payload: { x: 1 } })
  const receipt = spine.execute({ commandId: 'c1', aggregate: 'thread', streamId: 'A', kind: 'k' }, () => (
    { aggregate: 'thread', streamId: 'A', type: 'thread.turn-started', payload: {} }
  ))
  writer.onReceipt(receipt)
  spine.advance('ui', 2)
  writer.onCursor('ui', 2)
  await writer.flush()

  const restoredState = await readSpineJournal(fs, '/spine.jsonl')
  const restored = createEventSpine({ now: fixedNow, newEventId: () => `r${++idCounter}` }, restoredState)
  assert.equal(restored.headSequence, 2, 'events restored in order')
  assert.equal(restored.cursor('ui'), 2, 'cursor restored')
  assert.equal(restored.receipt('c1')?.status, 'accepted', 'receipt restored — retries stay exactly-once across restarts')
  assert.equal(restored.append({ aggregate: 'thread', streamId: 'A', type: 't3', payload: {} }).sequence, 3)
  ok('roundtrip: tail a live spine, restore an equivalent one')
}

// --- torn tail line from a crash mid-append is skipped ---
{
  const spine = newSpine()
  const good = spine.append({ aggregate: 'job', streamId: 'j', type: 'job.queued', payload: {} })
  const raw = encodeSpineLine({ t: 'event', e: good as SpineEvent }) + '{"t":"event","e":{"sequence":2,"eventId":"torn'
  const state = parseSpineJournal(raw)
  assert.equal(state.events?.length, 1, 'the torn line is skipped, the good one survives')
  assert.equal(state.events?.[0].eventId, 'e1')
  ok('parse: torn tail line never sinks the restore')
}

// --- cursor last-wins-monotonic, receipt first-wins, event dedup ---
{
  const raw = [
    '{"t":"cursor","p":"ui","s":5}',
    '{"t":"cursor","p":"ui","s":3}',
    '{"t":"cursor","p":"ui","s":9}',
    '{"t":"receipt","r":{"commandId":"c1","aggregate":"thread","streamId":"A","kind":"k","status":"accepted","resultSequence":1,"error":null,"at":1}}',
    '{"t":"receipt","r":{"commandId":"c1","aggregate":"thread","streamId":"A","kind":"k","status":"rejected","resultSequence":null,"error":"late dupe","at":2}}',
    '{"t":"event","e":{"sequence":1,"eventId":"dup","aggregate":"thread","streamId":"A","streamVersion":1,"type":"t","occurredAt":1,"commandId":null,"causationEventId":null,"correlationId":null,"actor":null,"payload":{"n":1}}}',
    '{"t":"event","e":{"sequence":1,"eventId":"dup","aggregate":"thread","streamId":"A","streamVersion":1,"type":"t","occurredAt":1,"commandId":null,"causationEventId":null,"correlationId":null,"actor":null,"payload":{"n":2}}}',
    '{"t":"compacted","s":4}',
    '{"t":"???","junk":true}',
  ].join('\n')
  const state = parseSpineJournal(raw)
  assert.equal(state.cursors?.ui, 9, 'cursor folds monotonically (never rewinds to 3)')
  assert.equal(state.receipts?.length, 1)
  assert.equal(state.receipts?.[0].status, 'accepted', 'the FIRST receipt for a commandId is immutable')
  assert.equal(state.events?.length, 1)
  assert.equal((state.events?.[0].payload as any).n, 1, 'duplicate eventIds keep the first payload')
  assert.equal(state.compactedThrough, 4)
  ok('parse: cursor monotonic fold, receipt first-wins, event dedup, junk skipped')
}

// --- atomic compaction rewrite ---
{
  const spine = newSpine()
  for (let i = 1; i <= 5; i++) spine.append({ aggregate: 'thread', streamId: 'A', type: `t${i}`, payload: {} })
  spine.advance('ui', 4)
  spine.compactThrough(4)
  const fs = memFs()
  await rewriteSpineJournal(fs, '/spine.jsonl', spine.snapshotState())
  assert.ok(!fs.files.has('/spine.jsonl.tmp'), 'temp file renamed away')
  const state = await readSpineJournal(fs, '/spine.jsonl')
  assert.deepEqual(state.events?.map((event) => event.sequence), [5], 'compacted events are gone from the rewrite')
  assert.equal(state.compactedThrough, 4, 'watermark persists')
  const restored = createEventSpine({ now: fixedNow, newEventId: () => `r${++idCounter}` }, state)
  assert.equal(restored.readSince(0).snapshotRequired, true, 'restored spine still flags pre-watermark readers')
  ok('rewrite: atomic temp+rename, watermark + surviving events persist')
}

// --- writer: append failures surface via onError and later writes continue ---
{
  const fs = memFs(1)
  const errors: string[] = []
  const writer = spineJournalWriter(fs, '/spine.jsonl', (message) => errors.push(message))
  writer.onCursor('ui', 1)   // fails ('disk full')
  writer.onCursor('ui', 2)   // succeeds
  await writer.flush()
  assert.equal(errors.length, 1)
  assert.ok(errors[0].includes('disk full'))
  assert.equal(parseSpineJournal(fs.files.get('/spine.jsonl') ?? '').cursors?.ui, 2, 'the chain survives a failed append')
  ok('writer: failed append reported, chain continues')
}

// --- missing journal restores an empty state ---
{
  const state = await readSpineJournal(memFs(), '/missing.jsonl')
  assert.deepEqual(state.events, [])
  assert.deepEqual(state.cursors, {})
  ok('read: missing journal → empty state, no throw')
}

// --- journal text shape: events precede markers ---
{
  const spine = newSpine()
  spine.append({ aggregate: 'thread', streamId: 'A', type: 't', payload: {} })
  spine.advance('ui', 1)
  const text = spineJournalText(spine.snapshotState())
  const kinds = text.trim().split('\n').map((line) => JSON.parse(line).t)
  assert.deepEqual(kinds, ['event', 'cursor'], 'events serialize before trailing markers')
  ok('spineJournalText: deterministic line ordering')
}

console.log(`\n${passed} passed`)
