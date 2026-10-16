// Rate-limit for streaming-draft edits. Telegram's editMessageText is rate-limited (a burst of
// edits trips "Too Many Requests"), so when a reply streams in token-by-token we must NOT fire an
// edit per token. This is a min-interval gate: an intermediate animation frame is emitted only if
// at least `minIntervalMs` has elapsed since the last emitted edit; frames that arrive too soon are
// dropped (the next allowed frame carries the latest text, so nothing is lost visually). The FINAL
// commit is never throttled — it goes through finalizeDraft, which bypasses this gate entirely — so
// the user always ends on the complete message regardless of how many intermediate frames were skipped.
//
// Pure core + a tiny stateful wrapper, both tested. No timers, no I/O: callers pass the current time.

// Default minimum gap between two emitted draft edits, in milliseconds. >= 1.5s keeps a comfortable
// margin under Telegram's per-chat edit rate so a fast token stream can't trip a 429.
export const DEFAULT_DRAFT_EDIT_INTERVAL_MS = 1500

// Per-platform minimum gap between two emitted draft edits (intermediate animation frames), in
// milliseconds. Each editing channel has its OWN edit rate limit, so a single global spacing would
// be either too aggressive for one or too slow for another. These are deliberately conservative
// margins under each platform's documented per-target edit ceiling:
//   telegram  1500  editMessageText is rate-limited per chat; a burst trips "Too Many Requests".
//   slack     1200  chat.update is Tier-3 (~50/min/workspace); 1.2s keeps a margin under that.
//   discord   1000  message edits (PATCH) share the channel bucket (~5 req / 5s); 1s stays under it.
// A platform without an entry falls back to DEFAULT_DRAFT_EDIT_INTERVAL_MS. The FINAL commit is
// never throttled (finalizeDraft bypasses the gate), so the user always ends on the full message.
export const DRAFT_EDIT_INTERVALS_MS: Record<string, number> = {
  telegram: 1500,
  slack: 1200,
  discord: 1000,
}

/**
 * The min draft-edit interval for a platform, in ms. Unknown/absent platform -> the default 1.5s.
 * Pure lookup so the host can build a per-platform throttle (makeDraftEditThrottle(interval)) for a
 * streaming reply, and tests can assert the per-platform selection without any I/O.
 */
export function draftEditIntervalForPlatform(platform: string | null | undefined): number {
  if (!platform) return DEFAULT_DRAFT_EDIT_INTERVAL_MS
  return DRAFT_EDIT_INTERVALS_MS[platform] ?? DEFAULT_DRAFT_EDIT_INTERVAL_MS
}

/**
 * Pure decision: may an intermediate draft edit be emitted at `now`, given the time the last edit
 * was emitted (`lastEmitAt`, or null if none yet) and the minimum interval? The first edit (no prior)
 * is always allowed; afterwards an edit is allowed once `minIntervalMs` has elapsed. A non-positive
 * interval disables throttling (every edit allowed). Clock going backwards is treated conservatively
 * as "not yet" (we only allow when the elapsed gap is provably large enough).
 */
export function canEmitDraftEdit(lastEmitAt: number | null, now: number, minIntervalMs: number): boolean {
  if (minIntervalMs <= 0) return true
  if (lastEmitAt == null) return true
  return now - lastEmitAt >= minIntervalMs
}

export interface DraftEditThrottle {
  // Should an intermediate edit be emitted now? Records `now` as the last-emit time when it returns
  // true, so the NEXT call is gated off this one. Returns false to drop this frame (too soon).
  shouldEmit(now: number): boolean
  // Note that the final commit happened at `now` (it bypasses the gate), so a throttle reused across
  // drafts starts its interval from the final rather than a stale intermediate. Always-allowed.
  markFinal(now: number): void
}

/**
 * Build a stateful throttle around the pure `canEmitDraftEdit` core. One instance per streaming draft:
 * the first intermediate frame fires immediately, subsequent ones are spaced at least `minIntervalMs`
 * apart. The final commit is NOT routed through `shouldEmit` (the caller finalizes unconditionally);
 * `markFinal` only updates the internal clock so the instance can be reused safely.
 */
export function makeDraftEditThrottle(minIntervalMs: number = DEFAULT_DRAFT_EDIT_INTERVAL_MS): DraftEditThrottle {
  let lastEmitAt: number | null = null
  return {
    shouldEmit(now: number): boolean {
      if (!canEmitDraftEdit(lastEmitAt, now, minIntervalMs)) return false
      lastEmitAt = now
      return true
    },
    markFinal(now: number): void {
      lastEmitAt = now
    },
  }
}
