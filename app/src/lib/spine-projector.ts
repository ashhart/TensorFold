// Renderer spine projector (VST-034/VST-041): the renderer is a real projection consumer
// with a PERSISTED cursor over the host's ordered event spine. On boot and on reconnect it
// gap-fills from its last cursor and produces a compact "while you were away" digest of the
// structural facts it missed — runs finished/failed, approvals resolved, jobs moved,
// threads appearing — instead of silently re-baselining. The live stream (IPC push or SSE)
// stays the realtime path; this is the durable ordering log underneath it.

import type { ReadSinceResult, SpineEvent } from './types'

export interface SpineCatchUp {
  fromSequence: number
  toSequence: number
  eventCount: number
  // Human summary of notable misses; null when nothing worth surfacing happened.
  summary: string | null
  // The gap predates the host's compaction watermark — state was re-baselined from the
  // snapshot and the digest below only covers what is still replayable.
  truncated: boolean
}

export interface SpineProjectorDeps {
  readSince: (input: { since: number; limit?: number }) => Promise<ReadSinceResult>
  // Server-side cursor (renderer-namespaced by the host) — best-effort.
  advanceCursor?: (input: { projector: string; sequence: number }) => Promise<number>
  loadCursor: () => number
  saveCursor: (sequence: number) => void
  pageLimit?: number
  maxPages?: number
}

// Digest the structural events into the few lines a returning user actually wants.
export function summarizeSpineEvents(events: SpineEvent[]): string | null {
  let done = 0
  let errored = 0
  let approvals = 0
  let newThreads = 0
  let jobMoves = 0
  for (const event of events) {
    if (event.type === 'agent.run-done') done++
    else if (event.type === 'agent.run-errored') errored++
    else if (event.type === 'approval.resolved') approvals++
    else if (event.type === 'thread.created' || event.type === 'thread.imported' || event.type === 'thread.forked' || event.type === 'thread.handoff') newThreads++
    else if (event.type === 'job.retried' || event.type === 'job.resumed' || event.type === 'job.dead_lettered') jobMoves++
  }
  const parts: string[] = []
  if (done) parts.push(`${done} run${done === 1 ? '' : 's'} finished`)
  if (errored) parts.push(`${errored} failed`)
  if (approvals) parts.push(`${approvals} approval${approvals === 1 ? '' : 's'} resolved`)
  if (newThreads) parts.push(`${newThreads} new thread${newThreads === 1 ? '' : 's'}`)
  if (jobMoves) parts.push(`${jobMoves} job change${jobMoves === 1 ? '' : 's'}`)
  return parts.length ? `While disconnected: ${parts.join(', ')}.` : null
}

/**
 * Page the spine from the persisted cursor, advance it, and return the digest. A FIRST run
 * (cursor 0) only establishes the baseline — replaying a host's whole history as "news"
 * would be noise — and returns null. Returns null too when nothing happened.
 */
export async function catchUpFromSpine(deps: SpineProjectorDeps): Promise<SpineCatchUp | null> {
  const pageLimit = deps.pageLimit ?? 500
  const maxPages = deps.maxPages ?? 40
  const from = Math.max(0, deps.loadCursor())
  const baseline = from === 0

  let cursor = from
  let truncated = false
  const events: SpineEvent[] = []
  for (let page = 0; page < maxPages; page++) {
    let result: ReadSinceResult
    try {
      result = await deps.readSince({ since: cursor, limit: pageLimit })
    } catch {
      // The host is unreachable or has no spine: keep the cursor untouched and stay quiet —
      // the next boot/reconnect retries from the same place.
      return null
    }
    if (result.snapshotRequired) truncated = true
    for (const event of result.events) {
      if (event.sequence > cursor) cursor = event.sequence
      events.push(event)
    }
    if (!result.hasMore) break
    // Defensive: a page that advances nothing must not spin.
    if (result.events.length === 0) break
  }

  if (cursor !== from) {
    deps.saveCursor(cursor)
    try { await deps.advanceCursor?.({ projector: 'ui', sequence: cursor }) } catch { /* best-effort */ }
  }
  if (baseline || cursor === from) return null

  return {
    fromSequence: from,
    toSequence: cursor,
    eventCount: events.length,
    summary: summarizeSpineEvents(events),
    truncated,
  }
}
