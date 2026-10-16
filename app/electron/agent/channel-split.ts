// Pure, UTF-16-aware message splitting for outbound channel replies. Every transport caps
// a single message (Discord 2000, Slack ~4000, Telegram 4096 UTF-16 units, WhatsApp ~1024,
// email unlimited); a longer reply must go out as several messages. This splits at the
// SAFEST boundary it can find under the cap — paragraph break > line break > sentence end >
// word break > hard cut — and never severs a UTF-16 surrogate pair (an emoji/astral char),
// so a chunk is always valid text the platform will accept. No I/O, no platform calls: the
// channel adapter calls splitForPlatform() then sends the chunks in order.

import type { MessageLengthLimit, MessageLengthLimits } from './types/connectors'

// Per-platform single-message caps, in UTF-16 code units (what `String.length` and the
// platforms themselves count). Deliberately conservative — a margin under the documented
// hard cap absorbs any per-platform counting quirk so a send is never rejected for length.
//   discord   2000  (hard API limit)
//   slack     ~4000 (blocks truncate ~3000; 3900 keeps a margin under the text limit)
//   telegram  4096  (documented UTF-16-unit cap; sendMessage rejects above it)
//   whatsapp  ~1024 (conservative; body text caps vary by template/session)
//   email     null  (no practical per-message cap)
//   imessage  ~8000 (no hard API cap; chunk very long bodies so a bubble stays sane)
export const PLATFORM_MESSAGE_LIMITS: MessageLengthLimits = {
  discord: { maxUnits: 2000 },
  slack: { maxUnits: 3900 },
  telegram: { maxUnits: 4096 },
  whatsapp: { maxUnits: 1024 },
  imessage: { maxUnits: 8000 },
  email: { maxUnits: null },
}

/** The configured limit for a platform, or a permissive default (null = unlimited) when unknown. */
export function limitForPlatform(platform: string | null | undefined): MessageLengthLimit {
  if (!platform) return { maxUnits: null }
  return PLATFORM_MESSAGE_LIMITS[platform] ?? { maxUnits: null }
}

// A high-surrogate occupies 0xD800..0xDBFF; the matching low-surrogate 0xDC00..0xDFFF. A cut
// BETWEEN a high and its following low would split one code point (e.g. an emoji) into two
// invalid halves. isHighSurrogate(text, i) is true when the unit at i begins a surrogate pair.
function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff
}
function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff
}

// Pull `cut` back by one unit if it would land in the middle of a surrogate pair, so a chunk
// boundary never severs an astral code point. `cut` is an index into a UTF-16 string.
function avoidSurrogateSplit(text: string, cut: number): number {
  if (cut <= 0 || cut >= text.length) return cut
  if (isLowSurrogate(text.charCodeAt(cut)) && isHighSurrogate(text.charCodeAt(cut - 1))) return cut - 1
  return cut
}

// Candidate break characters/sequences in preference order, applied to the window [start, hardEnd).
// Each finds the LAST occurrence at/under the hard end; we take the most-preferred one that
// leaves a non-trivial chunk (so we don't repeatedly shave a single trailing char).
function preferredBreak(text: string, start: number, hardEnd: number): number {
  const minUseful = start + Math.floor((hardEnd - start) * 0.35)
  // 1) Paragraph break (blank line). Break AFTER the newline pair so the blank line is consumed.
  const para = lastIndexOfBreak(text, start, hardEnd, ['\n\n', '\r\n\r\n'])
  if (para > minUseful) return para
  // 2) Single line break.
  const line = lastIndexOfBreak(text, start, hardEnd, ['\n', '\r'])
  if (line > minUseful) return line
  // 3) Sentence end followed by a space (". ", "! ", "? ", and CJK 。！？).
  const sentence = lastSentenceBreak(text, start, hardEnd)
  if (sentence > minUseful) return sentence
  // 4) Word break: the last whitespace run at/under the hard end.
  const word = lastWhitespace(text, start, hardEnd)
  if (word > minUseful) return word
  // 5) No good boundary — hard cut at the limit (surrogate-safe, applied by the caller).
  return hardEnd
}

// Last index (exclusive end of chunk) of any of `seqs` within [start, hardEnd], i.e. the
// position just AFTER the matched sequence, so the break characters end the current chunk.
function lastIndexOfBreak(text: string, start: number, hardEnd: number, seqs: string[]): number {
  let best = -1
  for (const seq of seqs) {
    // The sequence must END at or before hardEnd to fit the current chunk.
    const from = hardEnd - seq.length
    if (from < start) continue
    const idx = text.lastIndexOf(seq, from)
    if (idx >= start) best = Math.max(best, idx + seq.length)
  }
  return best
}

function lastSentenceBreak(text: string, start: number, hardEnd: number): number {
  let best = -1
  for (let i = Math.min(hardEnd, text.length) - 1; i > start; i--) {
    const ch = text[i]
    const prev = text[i - 1]
    // ASCII sentence punctuation followed by whitespace -> break after the whitespace.
    if ((ch === ' ' || ch === '\t') && (prev === '.' || prev === '!' || prev === '?')) {
      if (i + 1 <= hardEnd) best = Math.max(best, i + 1)
    }
    // CJK full-stop / bang / question mark are sentence ends on their own.
    if (prev === '。' || prev === '！' || prev === '？') {
      if (i <= hardEnd) best = Math.max(best, i)
    }
    if (best > 0) break
  }
  return best
}

function lastWhitespace(text: string, start: number, hardEnd: number): number {
  for (let i = Math.min(hardEnd, text.length); i > start; i--) {
    if (/\s/.test(text[i - 1])) return i
  }
  return -1
}

export interface SplitOptions {
  // Hard cap in UTF-16 code units. null/undefined/<=0 means "no cap" -> the whole text as one chunk.
  maxUnits?: number | null
  // Safety stop so a pathological input can't loop forever (defaults high enough for any real reply).
  maxChunks?: number
}

/**
 * Split `text` into chunks each at most `maxUnits` UTF-16 code units, preferring paragraph >
 * line > sentence > word boundaries and never splitting a surrogate pair. Pure.
 *
 * - No cap (null/<=0) or text already within the cap -> a single-element array (the trimmed
 *   text), or [] for empty/whitespace input (the caller's silence guard owns that case too).
 * - Leading/trailing whitespace of each emitted chunk is trimmed so a break character (the
 *   newline we split on) doesn't start the next message with blank space.
 */
export function splitMessage(text: string, options: SplitOptions = {}): string[] {
  const max = options.maxUnits ?? null
  const trimmed = (text ?? '').trim()
  if (!trimmed) return []
  if (max == null || max <= 0 || trimmed.length <= max) return [trimmed]
  const maxChunks = options.maxChunks ?? 10_000
  const chunks: string[] = []
  let start = 0
  while (start < trimmed.length && chunks.length < maxChunks) {
    const remaining = trimmed.length - start
    if (remaining <= max) {
      const tail = trimmed.slice(start).trim()
      if (tail) chunks.push(tail)
      break
    }
    let cut = preferredBreak(trimmed, start, start + max)
    cut = avoidSurrogateSplit(trimmed, cut)
    // Guard against a non-advancing cut (e.g. break landed at `start`): force progress with a
    // surrogate-safe hard cut so we never spin.
    if (cut <= start) cut = avoidSurrogateSplit(trimmed, start + max)
    if (cut <= start) cut = start + max // last resort (max==1 inside a pair is not a real case)
    const piece = trimmed.slice(start, cut).trim()
    if (piece) chunks.push(piece)
    start = cut
    // Skip whitespace we broke on so the next chunk doesn't lead with it.
    while (start < trimmed.length && /\s/.test(trimmed[start])) start++
  }
  return chunks.length ? chunks : [trimmed.slice(0, max)]
}

/** Split a reply for a named platform using its configured limit (convenience wrapper). */
export function splitForPlatform(platform: string | null | undefined, text: string): string[] {
  return splitMessage(text, { maxUnits: limitForPlatform(platform).maxUnits })
}
