import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  journalPath,
  appendThreadEvent,
  readThreadJournal,
  compactThreadJournal,
  _resetThreadJournalState,
} from './thread-journal'

async function main() {
  const baseDir = await fs.mkdtemp(path.join(os.tmpdir(), 'thread-journal-'))

  // --- journalPath shape + traversal guard ---
  assert.equal(journalPath('/base', 'abc'), path.join('/base', 'abc.ndjson'))
  assert.throws(() => journalPath('/base', '../escape'), /unsafe threadId/)
  assert.throws(() => journalPath('/base', 'a/b'), /unsafe threadId/)

  // --- append N events, read back in order, seq strictly monotonic from 1 ---
  _resetThreadJournalState()
  const N = 12
  for (let i = 0; i < N; i++) {
    const { seq } = await appendThreadEvent(baseDir, 't1', { kind: 'turn', i })
    assert.equal(seq, i + 1, 'sequential appends count up from 1')
  }
  let entries = await readThreadJournal(baseDir, 't1')
  assert.equal(entries.length, N)
  for (let i = 0; i < N; i++) {
    assert.equal(entries[i].seq, i + 1)
    assert.equal((entries[i].event as any).i, i)
    assert.equal(typeof entries[i].ts, 'number')
    if (i > 0) assert.ok(entries[i].seq > entries[i - 1].seq, 'seq strictly monotonic')
  }

  // --- concurrent appends to the same thread do not interleave / corrupt seq ---
  _resetThreadJournalState()
  const M = 50
  const results = await Promise.all(
    Array.from({ length: M }, (_, i) => appendThreadEvent(baseDir, 'concurrent', { i })),
  )
  const seqs = results.map((r) => r.seq).sort((a, b) => a - b)
  assert.deepEqual(seqs, Array.from({ length: M }, (_, i) => i + 1), 'every seq 1..M assigned exactly once')
  const cEntries = await readThreadJournal(baseDir, 'concurrent')
  assert.equal(cEntries.length, M, 'no lost or duplicated lines under concurrency')
  const cSeqs = new Set(cEntries.map((e) => e.seq))
  assert.equal(cSeqs.size, M, 'all seqs unique on disk')

  // separate thread keeps an independent counter
  const otherFirst = await appendThreadEvent(baseDir, 'other', { x: 1 })
  assert.equal(otherFirst.seq, 1, 'distinct threads have independent seq counters')

  // --- a corrupt line is skipped on read ---
  const file = journalPath(baseDir, 'corrupt')
  await fs.writeFile(
    file,
    [
      JSON.stringify({ seq: 1, ts: 1, event: { ok: 'a' } }),
      '{ this is not json',
      JSON.stringify({ seq: 2, ts: 2, event: { ok: 'b' } }),
      '',
    ].join('\n'),
    'utf8',
  )
  entries = await readThreadJournal(baseDir, 'corrupt')
  assert.equal(entries.length, 2, 'corrupt and blank lines skipped')
  assert.deepEqual(entries.map((e) => (e.event as any).ok), ['a', 'b'])

  // appending after a corrupt file still derives seq from the max valid seq on disk
  _resetThreadJournalState()
  const afterCorrupt = await appendThreadEvent(baseDir, 'corrupt', { ok: 'c' })
  assert.equal(afterCorrupt.seq, 3, 'seq derived from highest valid seq, not line count')

  // --- compaction drops lines below keepFromSeq, atomically ---
  _resetThreadJournalState()
  for (let i = 0; i < 6; i++) await appendThreadEvent(baseDir, 'compact', { i })
  await compactThreadJournal(baseDir, 'compact', 4)
  const compacted = await readThreadJournal(baseDir, 'compact')
  assert.deepEqual(compacted.map((e) => e.seq), [4, 5, 6], 'lines below keepFromSeq dropped, tail kept')
  // appending after compaction continues from the prior high-water mark (cache still warm)
  const post = await appendThreadEvent(baseDir, 'compact', { i: 99 })
  assert.equal(post.seq, 7, 'append after compaction continues monotonic seq')

  // compaction with keepFromSeq above everything yields an empty journal that reads clean
  await compactThreadJournal(baseDir, 'compact', 1000)
  assert.deepEqual(await readThreadJournal(baseDir, 'compact'), [])

  // reading an absent journal is empty, not an error
  assert.deepEqual(await readThreadJournal(baseDir, 'never-existed'), [])

  await fs.rm(baseDir, { recursive: true, force: true })
  console.log('thread-journal: all assertions passed')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
