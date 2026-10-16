// Shared, PURE connector-sync helpers adapted from competitive comms analysis:
//
//   1. INCREMENTAL SYNC CURSORS — every connector re-scanned its whole source on
//      each sync, then deduped events by id/threadKey after the fact. That is O(all
//      history) per poll. A per-source cursor lets a sync skip items it has already
//      seen: keep the high-water mark of the newest item (IMAP UID, slack/imessage
//      ts, discord snowflake), only emit items strictly NEWER than it, then advance
//      + persist the cursor. A null cursor (or a smaller one after a reset/account
//      swap) falls back to a FULL resync — correctness never depends on the cursor.
//
//   2. REPLY-TO CONTEXT INJECTION — inbound messages already carried reply/quoted
//      identifiers (email In-Reply-To, slack thread_ts, discord referenced_message)
//      but the model only ever saw the new message. When a message is a reply, fold
//      the quoted author + text into the event SUMMARY (the text the assistant reads)
//      so it understands what is being replied to without a tool round-trip.
//
// Both are pure and unit-tested (connector-sync.test.ts). The connectors and the
// main host wire these in; nothing here does I/O.

import type { ConnectorReplyContext, SyncCursor } from './types'

// ---- 1. Incremental sync cursors ----------------------------------------------

/** A scannable item carries a monotonic position and a stable id for the cursor. */
export interface CursoredItem {
  // Monotonic position within the source. Higher = newer. For IMAP this is the UID,
  // for slack/imessage it is the message timestamp (ms or the raw "<sec>.<seq>" as a
  // number), for discord the snowflake id. null/NaN positions are treated as 0 so a
  // dateless item is never considered "newer" than a real cursor.
  position: number | null
  // Stable per-item id, used only as a tie-breaker for the cursor's lastId and to
  // dedup an item sitting exactly ON the cursor boundary (position === cursor.position).
  id: string
}

/** Read a numeric position defensively (NaN/Infinity/null -> 0). */
function pos(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

/**
 * Should an item be ingested given the current cursor? An item is NEW iff its
 * position is strictly greater than the cursor's high-water position. Items at the
 * exact boundary (same position, e.g. two messages sharing one second) are skipped
 * ONLY when their id matches the cursor's lastId — otherwise a same-second sibling
 * would be lost. A null cursor admits everything (full resync).
 */
export function isNewerThanCursor(item: CursoredItem, cursor: SyncCursor | null | undefined): boolean {
  if (!cursor) return true
  const p = pos(item.position)
  const c = pos(cursor.position)
  if (p > c) return true
  if (p < c) return false
  // Equal position: keep it unless it is the exact item the cursor already points at.
  return item.id !== cursor.lastId
}

/** Filter a batch of items down to the ones newer than the cursor (preserves order). */
export function itemsNewerThanCursor<T extends CursoredItem>(items: T[], cursor: SyncCursor | null | undefined): T[] {
  if (!cursor) return [...items]
  return items.filter((item) => isNewerThanCursor(item, cursor))
}

/**
 * Advance a cursor to the newest item in a batch. The cursor only ever moves
 * FORWARD (a sync that happens to return older items must not rewind the high-water
 * mark), so the result is the max of the existing cursor and the batch's max. When
 * the batch is empty the cursor is returned unchanged. `now` stamps updatedAt for
 * observability. Returns the SAME reference when nothing changed so callers can skip
 * a needless persist.
 */
export function advanceSyncCursor(cursor: SyncCursor | null | undefined, items: CursoredItem[], now = Date.now()): SyncCursor | null {
  const hadCursor = !!cursor
  let bestPos = cursor ? pos(cursor.position) : Number.NEGATIVE_INFINITY
  let bestId = cursor?.lastId ?? null
  let moved = false
  for (const item of items) {
    const p = pos(item.position)
    if (p > bestPos || (!hadCursor && !moved)) {
      bestPos = p
      bestId = item.id
      moved = true
    }
  }
  if (!moved) return cursor ?? null
  return { position: bestPos === Number.NEGATIVE_INFINITY ? 0 : bestPos, lastId: bestId, updatedAt: now }
}

/**
 * The IMAP UID floor to fetch from: only messages with UID > floor are new. Returns
 * 0 (fetch the recent window) when there is no cursor or it carries no uid yet, which
 * is the full-resync fallback. Never returns a negative floor.
 */
export function imapUidFloor(cursor: SyncCursor | null | undefined): number {
  const uid = cursor?.lastImapUid
  return typeof uid === 'number' && Number.isFinite(uid) && uid > 0 ? uid : 0
}

/** Build/advance an IMAP cursor from the max UID seen this fetch (forward-only). */
export function advanceImapCursor(cursor: SyncCursor | null | undefined, uids: number[], now = Date.now()): SyncCursor | null {
  const floor = imapUidFloor(cursor)
  let maxUid = floor
  for (const uid of uids) {
    if (Number.isFinite(uid) && uid > maxUid) maxUid = uid
  }
  if (maxUid <= floor && cursor) return cursor // nothing newer; leave the cursor (and its position) alone
  return {
    position: maxUid,
    lastId: maxUid > 0 ? String(maxUid) : (cursor?.lastId ?? null),
    lastImapUid: maxUid,
    updatedAt: now,
  }
}

// ---- 2. Reply-to context injection --------------------------------------------

/** True when there is any quoted content worth showing the assistant. */
export function hasReplyContext(reply: ConnectorReplyContext | null | undefined): reply is ConnectorReplyContext {
  return !!reply && (!!reply.text?.trim() || !!reply.author?.trim())
}

/**
 * One-line, human/agent-readable rendering of a quoted message, e.g.
 *   In reply to Maya Chen: "can you confirm pricing?"
 * Author-only or text-only replies degrade gracefully. The quoted text is collapsed
 * to a single line and clipped so a long quoted body can't blow up the prompt. Returns
 * '' when there is nothing to show.
 */
export function formatReplyContext(reply: ConnectorReplyContext | null | undefined, maxChars = 200): string {
  if (!hasReplyContext(reply)) return ''
  const author = reply.author?.trim()
  const text = clip(collapse(reply.text ?? ''), maxChars)
  if (author && text) return `${REPLY_MARKER} ${author}: "${text}"`
  if (author) return `${REPLY_MARKER} ${author}.`
  return `In reply to: "${text}"`
}

// Prefix every quoted line so injection is idempotent (a re-derived event won't stack
// duplicate quote lines) and so a downstream reader can strip it if needed.
const REPLY_MARKER = 'In reply to'

/**
 * Fold the quoted context into a message summary so the assistant sees what's being
 * replied to. The reply line is PREPENDED (the model reads the context before the new
 * message). Idempotent: a summary that already begins with the reply marker is left
 * alone so re-deriving an event doesn't stack duplicate quote lines.
 */
export function injectReplyContext(summary: string, reply: ConnectorReplyContext | null | undefined, maxChars = 200): string {
  const line = formatReplyContext(reply, maxChars)
  if (!line) return summary
  const body = (summary ?? '').trim()
  if (body.startsWith(REPLY_MARKER)) return summary // already injected
  return body ? `${line}\n${body}` : line
}

/**
 * Build a normalized ConnectorReplyContext from loosely-typed source fields (a raw
 * provider payload, a quoted-message lookup, etc.). Returns null when there is no
 * usable quoted content, so callers can simply do `reply ?? undefined`. Trims and
 * clips so untrusted quoted text is bounded before it reaches an event.
 */
export function makeReplyContext(input: { id?: string | null; author?: string | null; text?: string | null }, maxChars = 2_000): ConnectorReplyContext | null {
  const id = input.id?.toString().trim() || null
  const author = input.author?.toString().trim() || null
  const text = input.text != null ? clip(collapse(String(input.text)), maxChars) : null
  if (!author && !text) return null // an id with no content isn't worth quoting to the model
  return { id, author, text }
}

function collapse(value: string): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function clip(value: string, max: number): string {
  const v = collapse(value)
  return v.length <= max ? v : `${v.slice(0, Math.max(0, max - 1)).trimEnd()}…`
}
