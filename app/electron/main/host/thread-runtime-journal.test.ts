// Run: ./node_modules/.bin/jiti electron/main/host/thread-runtime-journal.test.ts
// Per-thread NDJSON journals must be encrypted at rest (matching threads.json's
// writeProtectedJson posture) instead of writing the full plaintext transcript to disk.
// These tests pin encodeJournalEvent/decodeJournalEvent round-tripping and prove the
// on-disk line is ciphertext, survives compaction, and that legacy plaintext still reads.
import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { encodeJournalEvent, decodeJournalEvent } from './thread-runtime'
import {
  appendThreadEvent,
  readThreadJournal,
  compactThreadJournal,
  _resetThreadJournalState,
  type JournalFsDeps,
} from '../../agent/thread-journal'

let passed = 0
const ok = (msg: string) => { passed++; console.log(`ok - ${msg}`) }

// In-memory fs so the test never touches disk.
function memFs(): { fs: JournalFsDeps; files: Map<string, string> } {
  const files = new Map<string, string>()
  const fs: JournalFsDeps = {
    mkdir: async () => undefined,
    appendFile: async (file, data) => { files.set(file, (files.get(file) ?? '') + data) },
    readFile: async (file) => { const v = files.get(file); if (v === undefined) throw new Error('ENOENT'); return v },
    writeFile: async (file, data) => { files.set(file, data) },
    rename: async (from, to) => { const v = files.get(from); files.delete(from); if (v !== undefined) files.set(to, v) },
  }
  return { fs, files }
}

const key = randomBytes(32)

// 1: encode/decode round-trips an event with a key.
{
  const event = { type: 'message', role: 'assistant', content: 'secret diff: -old\n+new', tool: 'edit_file' }
  const encoded = encodeJournalEvent(event, key)
  // The encoded envelope must NOT contain the plaintext.
  assert.ok(!JSON.stringify(encoded).includes('secret diff'), 'encoded journal event leaks plaintext')
  assert.deepEqual(decodeJournalEvent(encoded, key), event)
  ok('encodeJournalEvent encrypts and decodeJournalEvent restores the original event')
}

// 2: a wrong key cannot decrypt the envelope (returns null, never the plaintext).
{
  const event = { type: 'tool_result', text: 'API_KEY=abcdef' }
  const encoded = encodeJournalEvent(event, key)
  const wrong = randomBytes(32)
  assert.equal(decodeJournalEvent(encoded, wrong), null)
  // And a missing key likewise refuses to surface ciphertext.
  assert.equal(decodeJournalEvent(encoded, null), null)
  ok('decodeJournalEvent refuses to reveal content with the wrong/missing key')
}

// 3: with no key, the optional journal is skipped rather than leaking plaintext;
// legacy plaintext lines written by older builds remain readable.
{
  const event = { type: 'reasoning', text: 'thinking' }
  const encoded = encodeJournalEvent(event, null)
  assert.equal(encoded, null)
  assert.deepEqual(decodeJournalEvent(event, key), event) // legacy plaintext line still reads
  ok('no-key failure skips the write while legacy plaintext lines still decode')
}

// 4: integration — appended journal LINE on disk is ciphertext, reads back, and survives compaction.
{
  _resetThreadJournalState()
  const { fs, files } = memFs()
  const base = '/journals'
  const tid = 'thread-1'
  const events = [
    { type: 'message', content: 'line one PLAINTEXT-MARKER-1' },
    { type: 'message', content: 'line two PLAINTEXT-MARKER-2' },
    { type: 'message', content: 'line three PLAINTEXT-MARKER-3' },
  ]
  ;(async () => {
    for (const e of events) await appendThreadEvent(base, tid, encodeJournalEvent(e, key), fs)
    // The raw on-disk file must not contain any plaintext marker.
    const raw = [...files.values()].join('')
    assert.ok(raw.length > 0, 'journal file was written')
    for (const marker of ['PLAINTEXT-MARKER-1', 'PLAINTEXT-MARKER-2', 'PLAINTEXT-MARKER-3']) {
      assert.ok(!raw.includes(marker), `journal leaks plaintext marker ${marker}`)
    }
    // seq/ts envelope stays in the clear so compaction works without the key.
    assert.ok(raw.includes('"seq"') && raw.includes('"ts"'), 'seq/ts envelope must stay plaintext')

    // Read back and decrypt each event.
    const entries = await readThreadJournal(base, tid, fs)
    assert.equal(entries.length, 3)
    assert.deepEqual(entries.map((en) => decodeJournalEvent(en.event, key)), events)

    // Compaction keeps the encrypted envelope intact (it only re-serializes seq/ts/event).
    await compactThreadJournal(base, tid, entries[1].seq, fs)
    const afterCompact = await readThreadJournal(base, tid, fs)
    assert.equal(afterCompact.length, 2)
    assert.deepEqual(afterCompact.map((en) => decodeJournalEvent(en.event, key)), events.slice(1))
    const rawAfter = [...files.values()].join('')
    for (const marker of ['PLAINTEXT-MARKER-2', 'PLAINTEXT-MARKER-3']) {
      assert.ok(!rawAfter.includes(marker), `compacted journal leaks plaintext ${marker}`)
    }
    ok('appended journal lines are ciphertext on disk, decrypt on read, and survive compaction')
    console.log(`\n${passed} thread-runtime journal assertions passed.`)
  })().catch((e) => { console.error(e); process.exit(1) })
}
