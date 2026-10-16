import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  appendThreadEvent,
  compactThreadJournal,
  readThreadJournal,
  _resetThreadJournalState,
  type JournalFsDeps,
} from './thread-journal'

let ok = 0
function check(cond: boolean, msg: string) {
  assert.ok(cond, msg)
  ok++
}

// Verifies the data-loss bug fix: compactThreadJournal must participate in the same per-thread
// append lock as appendThreadEvent. If it reads/renames OUTSIDE the lock, an append that lands
// after compaction's read but before its rename is silently overwritten and lost.
//
// We provoke the exact interleaving by wrapping the real fs so that the FIRST readFile issued by
// compaction (its "read the journal" step) fires an appendThreadEvent for the same thread before
// resolving. With the bug, that append would race the rename and be lost; with the fix, the append
// is queued behind compaction on the shared chain and survives.
async function main() {
  const baseDir = await fs.mkdtemp(path.join(os.tmpdir(), 'thread-journal-lock-'))

  _resetThreadJournalState()

  // Seed a journal large enough to be a realistic compaction target.
  for (let i = 0; i < 6; i++) await appendThreadEvent(baseDir, 't', { i })

  let injected = false
  let injectedAppend: Promise<{ seq: number }> | undefined

  const racingFs: JournalFsDeps = {
    mkdir: (dir, options) => realMkdir(dir, options),
    appendFile: (file, data, options) => realAppend(file, data, options),
    writeFile: (file, data, options) => realWrite(file, data, options),
    rename: (from, to) => realRename(from, to),
    readFile: async (file, encoding) => {
      const out = await realRead(file, encoding)
      // The first read is compaction's snapshot of the journal. Right after it resolves (but before
      // compaction writes its temp + renames), kick off a concurrent append for the same thread.
      // If compaction is correctly serialized on the tails chain, this append runs AFTER the rename.
      if (!injected) {
        injected = true
        injectedAppend = appendThreadEvent(baseDir, 't', { late: true })
      }
      return out
    },
  }

  await compactThreadJournal(baseDir, 't', 4, racingFs)
  // Make sure the racing append has settled before we read the final state.
  const lateRes = await injectedAppend!

  const entries = await readThreadJournal(baseDir, 't')
  const events = entries.map((e) => e.event as { i?: number; late?: boolean })

  // The compaction kept seqs >= 4 (the i=3,4,5 events), and the late append must NOT have been lost.
  check(
    events.some((e) => e.late === true),
    'concurrent append during compaction is preserved (not overwritten by rename)',
  )
  // Surviving compacted tail (i=3,4,5) is still present.
  for (const i of [3, 4, 5]) {
    check(events.some((e) => e.i === i), `compacted tail event i=${i} survives`)
  }
  // The late append got a fresh monotonic seq above the compacted high-water mark (6 -> 7).
  check(lateRes.seq === 7, 'late append continues monotonic seq after compaction (got ' + lateRes.seq + ')')
  // And it is the highest seq on disk.
  const maxSeq = Math.max(...entries.map((e) => e.seq))
  check(maxSeq === 7, 'late append is the newest line on disk')

  await fs.rm(baseDir, { recursive: true, force: true })
  console.log(`thread-journal-compact-lock: ${ok} assertions passed`)
}

// Pass-throughs to the real fs (kept separate so the instrumented readFile above is the only hook).
const realMkdir: JournalFsDeps['mkdir'] = (dir, options) => fs.mkdir(dir, options)
const realAppend: JournalFsDeps['appendFile'] = (file, data, options) => fs.appendFile(file, data, options)
const realWrite: JournalFsDeps['writeFile'] = (file, data, options) => fs.writeFile(file, data, options)
const realRename: JournalFsDeps['rename'] = (from, to) => fs.rename(from, to)
const realRead: JournalFsDeps['readFile'] = (file, encoding) => fs.readFile(file, encoding)

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
