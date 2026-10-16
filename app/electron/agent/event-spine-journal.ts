// Crash-safe persistence for the event spine (VST-034): ONE append-only JSONL journal
// holding typed lines — events, command receipts, and cursor advances. Appends are the hot
// path (one line per fact, no rewrite); restore folds the whole journal back into a
// SpineRestoreState (last cursor per projector wins, events dedup by eventId, receipts by
// commandId, a torn tail line from a crash mid-write is skipped). Compaction rewrites the
// journal from a live snapshot via the injected fs (temp + rename → atomic).

import type { CommandReceipt, SpineEvent, SpineRestoreState } from './event-spine'

export type SpineJournalLine =
  | { t: 'event'; e: SpineEvent }
  | { t: 'receipt'; r: CommandReceipt }
  | { t: 'cursor'; p: string; s: number }
  | { t: 'compacted'; s: number }

export interface SpineJournalFileHandle {
  writeFile(data: string): Promise<void>
  datasync(): Promise<void>
  close(): Promise<void>
}

export interface SpineJournalFs {
  appendFile(file: string, data: string): Promise<void>
  readFile(file: string): Promise<string>
  writeFile(file: string, data: string): Promise<void>
  rename(from: string, to: string): Promise<void>
  /** Durability seam for rewriteSpineJournal. Production wires node:fs/promises, whose
   *  datasync flushes the staged temp to disk before the atomic rename. Test doubles omit
   *  it — an in-memory map has no OS page cache to lose on crash. */
  open?(file: string, flags: string): Promise<SpineJournalFileHandle>
}

export function encodeSpineLine(line: SpineJournalLine): string {
  return `${JSON.stringify(line)}\n`
}

// Fold raw journal text into a restore state. Defensive: non-JSON lines (torn tail from a
// crash mid-append) and unknown line types are skipped, never fatal.
export function parseSpineJournal(raw: string): SpineRestoreState {
  const events = new Map<string, SpineEvent>()
  const receipts = new Map<string, CommandReceipt>()
  const cursors: Record<string, number> = {}
  let compactedThrough = 0
  for (const line of raw.split('\n')) {
    const text = line.trim()
    if (!text) continue
    let parsed: any
    try { parsed = JSON.parse(text) } catch { continue }
    if (parsed?.t === 'event' && parsed.e?.eventId != null && typeof parsed.e.sequence === 'number') {
      if (!events.has(parsed.e.eventId)) events.set(parsed.e.eventId, parsed.e)
    } else if (parsed?.t === 'receipt' && parsed.r?.commandId != null) {
      // First write wins: a receipt is immutable once recorded (exactly-once semantics).
      if (!receipts.has(parsed.r.commandId)) receipts.set(parsed.r.commandId, parsed.r)
    } else if (parsed?.t === 'cursor' && typeof parsed.p === 'string' && typeof parsed.s === 'number') {
      // Last write wins, but never backwards (cursors are monotonic).
      cursors[parsed.p] = Math.max(cursors[parsed.p] ?? 0, parsed.s)
    } else if (parsed?.t === 'compacted' && typeof parsed.s === 'number') {
      compactedThrough = Math.max(compactedThrough, parsed.s)
    }
  }
  return {
    events: [...events.values()].sort((a, b) => a.sequence - b.sequence),
    receipts: [...receipts.values()],
    cursors,
    compactedThrough,
  }
}

export async function readSpineJournal(fs: Pick<SpineJournalFs, 'readFile'>, file: string): Promise<SpineRestoreState> {
  let raw = ''
  try { raw = await fs.readFile(file) } catch { return { events: [], receipts: [], cursors: {}, compactedThrough: 0 } }
  return parseSpineJournal(raw)
}

// Serialize a full snapshot back to journal lines (compaction rewrite). Events first so a
// partially-written rewrite that crashes before the trailing markers still restores sanely.
export function spineJournalText(state: Required<SpineRestoreState>): string {
  const lines: string[] = []
  for (const event of state.events) lines.push(encodeSpineLine({ t: 'event', e: event }))
  for (const receipt of state.receipts) lines.push(encodeSpineLine({ t: 'receipt', r: receipt }))
  for (const [projector, sequence] of Object.entries(state.cursors)) lines.push(encodeSpineLine({ t: 'cursor', p: projector, s: sequence }))
  if (state.compactedThrough > 0) lines.push(encodeSpineLine({ t: 'compacted', s: state.compactedThrough }))
  return lines.join('')
}

// Atomic rewrite: stage to a sibling temp file, fsync/datasync it, then rename over the
// journal — a crash at any point leaves either the old journal or the new one, never a torn
// hybrid. The datasync before the rename closes the window where a crash after rename
// "succeeds" could otherwise lose the still-cached bytes (mirrors atomicWriteJson in store.ts
// and atomicWriteWorkspaceFile in atomic-file-write.ts). When the injected fs provides no
// open/datasync (e.g. an in-memory test double), we fall back to writeFile+rename, which
// remains atomic but without the page-cache durability guarantee that only real disk needs.
export async function rewriteSpineJournal(fs: SpineJournalFs, file: string, state: Required<SpineRestoreState>): Promise<void> {
  const temp = `${file}.tmp`
  if (fs.open) {
    const handle = await fs.open(temp, 'w')
    try {
      await handle.writeFile(spineJournalText(state))
      // Flush BEFORE the rename commits the bytes atomically; a flush failure must abort the
      // rename (the finally closes the handle, leaving the old journal intact) rather than
      // renaming possibly-unflushed data over good state.
      await handle.datasync()
    } finally {
      await handle.close()
    }
    await fs.rename(temp, file)
  } else {
    await fs.writeFile(temp, spineJournalText(state))
    await fs.rename(temp, file)
  }
}

// A tiny persister that tails a live spine into the journal: call the returned hooks from
// spine wiring (subscribe → onEvent, after execute → onReceipt, after advance → onCursor).
// Writes are fire-and-forget serialized through a chain so appends never interleave.
export function spineJournalWriter(fs: Pick<SpineJournalFs, 'appendFile'>, file: string, onError?: (message: string) => void) {
  let chain: Promise<void> = Promise.resolve()
  const enqueue = (line: SpineJournalLine) => {
    chain = chain
      .then(() => fs.appendFile(file, encodeSpineLine(line)))
      .catch((error) => onError?.(`event-spine journal append failed: ${error instanceof Error ? error.message : String(error)}`))
  }
  return {
    onEvent: (event: SpineEvent) => enqueue({ t: 'event', e: event }),
    onReceipt: (receipt: CommandReceipt) => enqueue({ t: 'receipt', r: receipt }),
    onCursor: (projector: string, sequence: number) => enqueue({ t: 'cursor', p: projector, s: sequence }),
    onCompacted: (sequence: number) => enqueue({ t: 'compacted', s: sequence }),
    // Serialize arbitrary journal work (e.g. a compaction rewrite) through the SAME chain as
    // appends — a rewrite that ran concurrently would lose any append that landed on the old
    // file between writeFile(temp) and rename.
    run: (fn: () => Promise<void>) => {
      chain = chain
        .then(fn)
        .catch((error) => onError?.(`event-spine journal task failed: ${error instanceof Error ? error.message : String(error)}`))
    },
    // Await everything queued so far (tests + graceful shutdown).
    flush: () => chain,
  }
}
