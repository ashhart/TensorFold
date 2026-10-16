// Append-only NDJSON event journal, one file per thread. Instead of a database, each thread
// gets a {threadId}.ndjson file
// under baseDir, with one JSON line {seq, ts, event} per emitted AgentEvent. Mirrors the
// fs.appendFile + serialize-through-a-promise-chain pattern from main/host/audit.ts, but
// keyed per thread so unrelated threads never block each other.
//
// Pure + fs-injected: every operation takes a configurable baseDir so tests run against a
// temp dir, and the fs surface can be overridden via JournalFsDeps for no-disk unit tests.

import { promises as realFs } from 'node:fs'
import path from 'node:path'

// One journal line, as written and read back.
export interface ThreadJournalEntry {
  seq: number
  ts: number
  event: unknown
}

// Minimal fs surface the journal depends on. Defaults to node:fs/promises so callers can
// pass just (baseDir, threadId, event); tests inject a fake to avoid touching disk.
export interface JournalFsDeps {
  mkdir(dir: string, options: { recursive: boolean }): Promise<unknown>
  appendFile(file: string, data: string, options: { encoding: 'utf8' }): Promise<void>
  readFile(file: string, encoding: 'utf8'): Promise<string>
  writeFile(file: string, data: string, options: { encoding: 'utf8' }): Promise<void>
  rename(from: string, to: string): Promise<void>
}

// Adapt node:fs/promises to JournalFsDeps. node's signatures are wider, so wrap to pin them.
const defaultFs: JournalFsDeps = {
  mkdir: (dir, options) => realFs.mkdir(dir, options),
  appendFile: (file, data, options) => realFs.appendFile(file, data, options),
  readFile: (file, encoding) => realFs.readFile(file, encoding),
  writeFile: (file, data, options) => realFs.writeFile(file, data, options),
  rename: (from, to) => realFs.rename(from, to),
}

// A thread id is part of a filename, so refuse separators / traversal that would escape baseDir.
function safeThreadId(threadId: string): string {
  if (!threadId || /[\\/]/.test(threadId) || threadId === '.' || threadId === '..') {
    throw new Error(`thread-journal: unsafe threadId ${JSON.stringify(threadId)}`)
  }
  return threadId
}

/** Absolute path of a thread's NDJSON journal under baseDir. */
export function journalPath(baseDir: string, threadId: string): string {
  return path.join(baseDir, `${safeThreadId(threadId)}.ndjson`)
}

// Per-thread tail of the append chain. Concurrent appendThreadEvent calls for the same thread
// queue behind one another so the read-last-seq -> write step never interleaves (which would
// fork the seq counter and corrupt the file). Different threads have independent tails.
const tails = new Map<string, Promise<{ seq: number }>>()
// Cached next seq per thread, so the common (hot) path does not re-read the whole file each append.
const nextSeq = new Map<string, number>()

/** Highest seq already on disk, or 0 if the journal is absent/empty. Skips corrupt lines. */
async function lastSeqOnDisk(fs: JournalFsDeps, file: string): Promise<number> {
  let raw: string
  try {
    raw = await fs.readFile(file, 'utf8')
  } catch {
    return 0 // no file yet
  }
  let max = 0
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue
    try {
      const parsed = JSON.parse(line)
      if (typeof parsed?.seq === 'number' && parsed.seq > max) max = parsed.seq
    } catch {
      // corrupt line, ignore for seq derivation
    }
  }
  return max
}

/**
 * Append one event to a thread's journal and return its assigned seq.
 * seq is a monotonic per-thread counter; appends for the same thread are serialized through a
 * promise chain so concurrent callers each get a distinct, strictly increasing seq.
 */
export function appendThreadEvent(
  baseDir: string,
  threadId: string,
  event: unknown,
  fs: JournalFsDeps = defaultFs,
): Promise<{ seq: number }> {
  const file = journalPath(baseDir, threadId)
  const key = file // path uniquely identifies the thread+baseDir
  const prior = tails.get(key) ?? Promise.resolve({ seq: 0 })

  const run = prior.then(async () => {
    // Derive the next seq from the cache if present, otherwise scan the file once.
    let seq = nextSeq.get(key)
    if (seq === undefined) seq = (await lastSeqOnDisk(fs, file)) + 1
    const line = JSON.stringify({ seq, ts: Date.now(), event }) + '\n'
    await fs.mkdir(path.dirname(file), { recursive: true })
    await fs.appendFile(file, line, { encoding: 'utf8' })
    nextSeq.set(key, seq + 1)
    return { seq }
  })

  // Keep the chain alive even if one append throws, but drop the cache so the next append
  // re-derives seq from disk rather than trusting a counter that may not have been written.
  tails.set(key, run.catch(() => { nextSeq.delete(key); return { seq: 0 } }))
  return run
}

/** Read and parse every journal line in seq order. Corrupt lines are skipped, not fatal. */
export async function readThreadJournal(
  baseDir: string,
  threadId: string,
  fs: JournalFsDeps = defaultFs,
): Promise<ThreadJournalEntry[]> {
  const file = journalPath(baseDir, threadId)
  let raw: string
  try {
    raw = await fs.readFile(file, 'utf8')
  } catch {
    return [] // absent journal reads as empty
  }
  const out: ThreadJournalEntry[] = []
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue
    try {
      const parsed = JSON.parse(line)
      if (typeof parsed?.seq === 'number' && typeof parsed?.ts === 'number' && 'event' in parsed) {
        out.push({ seq: parsed.seq, ts: parsed.ts, event: parsed.event })
      }
    } catch {
      // skip corrupt line
    }
  }
  out.sort((a, b) => a.seq - b.seq)
  return out
}

/**
 * Drop journal lines below keepFromSeq, keeping the tail. Used at run-end to compact a long
 * journal down to the events still worth replaying. Writes a temp file then renames over the
 * original so a crash mid-compaction never leaves a half-written journal.
 *
 * Compaction is serialized through the same per-thread `tails` chain as appendThreadEvent, so a
 * concurrent append can never land between the read and the rename (which would silently overwrite
 * that just-appended event). The cached seq counter is kept consistent with the surviving tail so
 * the next append continues monotonically; on failure the cache is dropped to re-derive from disk.
 */
export function compactThreadJournal(
  baseDir: string,
  threadId: string,
  keepFromSeq: number,
  fs: JournalFsDeps = defaultFs,
): Promise<void> {
  const file = journalPath(baseDir, threadId)
  const key = file
  const prior = tails.get(key) ?? Promise.resolve({ seq: 0 })

  const run = prior.then(async () => {
    // Read inside the chain so no append can interleave between the read and the rename below.
    const entries = await readThreadJournal(baseDir, threadId, fs)
    const kept = entries.filter((e) => e.seq >= keepFromSeq)
    const body = kept.map((e) => JSON.stringify({ seq: e.seq, ts: e.ts, event: e.event })).join('\n')
    const tmp = `${file}.tmp-${process.pid}-${Date.now()}`
    await fs.mkdir(path.dirname(file), { recursive: true })
    await fs.writeFile(tmp, body ? body + '\n' : '', { encoding: 'utf8' })
    await fs.rename(tmp, file)
    // Highest surviving seq (0 when the journal was fully emptied) so the next append continues
    // from here rather than re-scanning or trusting a now-stale cached counter.
    return { seq: kept.length ? kept[kept.length - 1].seq : 0 }
  })

  // Advance the chain and refresh the cached next-seq; on failure drop the cache so the next op
  // re-derives seq from disk rather than trusting a counter that may not reflect the file.
  tails.set(
    key,
    run
      .then((r) => {
        nextSeq.set(key, r.seq + 1)
        return r
      })
      .catch(() => {
        nextSeq.delete(key)
        return { seq: 0 }
      }),
  )
  return run.then(() => undefined)
}

/** Test-only: clear the in-memory seq cache and append chains (e.g. between cases). */
export function _resetThreadJournalState(): void {
  tails.clear()
  nextSeq.clear()
}
