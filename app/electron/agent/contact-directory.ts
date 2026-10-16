// Contact directory. iMessage and WhatsApp surface raw phone numbers ("+447700900123"),
// Slack/Discord surface opaque user ids ("U024BE7LH"), and email surfaces bare addresses —
// none of which a human (or the assistant deciding "reply to Maya") can read. This module
// maps a platform HANDLE -> a DISPLAY NAME, learned from what each connector already sees
// (an email From's display-name, Slack/Discord member info, an iMessage/WhatsApp contact
// field), so the feed can show the NAME instead of the number, and a "reply to <name>"
// lookup can turn a name back into the handle to send to.
//
// Two directions:
//   resolveName(handle)   -> the best known display name for a handle ("+447…": "Maya Chen")
//   resolveHandle(name)   -> the handle to reply to for a spoken/typed name ("reply to Maya")
//
// Learned mappings persist as a small JSON in ~/.vontra (the HOST owns the file I/O; this
// module is pure: it takes the current directory in and returns the next one out). The
// normalization (E.164 phone, id casing) + the bidirectional resolution + dedupe are
// unit-tested (contact-directory.test.ts). Nothing here does I/O.

export type ContactPlatform = 'phone' | 'slack' | 'discord' | 'email' | 'telegram' | 'other'

// One learned mapping. `handle` is the NORMALIZED handle (the lookup key); `rawHandle`
// preserves what the connector originally sent (for display/debugging). `name` is the
// display name. `confidence` lets a high-signal source (an address-book contact) win over
// a low-signal one (a username guessed from an id). `count` is how many times we've seen
// this pairing — a tie-break and a freshness signal.
export interface ContactRecord {
  handle: string
  rawHandle: string
  platform: ContactPlatform
  name: string
  source: string // which connector taught us ('imessage' | 'whatsapp' | 'email' | 'slack' | 'discord' | ...)
  confidence: ContactConfidence
  count: number
  updatedAt: number
}

export type ContactConfidence = 'high' | 'medium' | 'low'

export interface ContactDirectory {
  // Keyed by `${platform}:${handle}` so the same number on two platforms stays distinct
  // and a lookup is O(1). Treat as opaque; build/mutate via the helpers below.
  records: Record<string, ContactRecord>
  updatedAt: number
}

// A single observation a connector feeds in. The connector knows the platform + handle and
// (when it has one) the display name; everything else has a sane default.
export interface ContactObservation {
  platform: ContactPlatform
  handle: string
  name?: string | null
  source?: string
  confidence?: ContactConfidence
}

const CONFIDENCE_RANK: Record<ContactConfidence, number> = { high: 3, medium: 2, low: 1 }

export function emptyContactDirectory(now = Date.now()): ContactDirectory {
  return { records: {}, updatedAt: now }
}

// ---- Normalization ------------------------------------------------------------

/**
 * Normalize a phone number toward E.164 ("+<country><subscriber>", digits only after the
 * plus). Strips spaces, dashes, dots, parentheses; maps a leading "00" international prefix
 * to "+". A number that already starts with "+" keeps its country code. A bare national
 * number (no "+", no "00") can't be made unambiguously international without a region, so it
 * is returned digits-only WITHOUT a plus — still a stable key, just not E.164. Returns ''
 * for input with no digits.
 */
export function normalizePhone(raw: string | null | undefined): string {
  const s = String(raw ?? '').trim()
  if (!s) return ''
  const hasPlus = s.startsWith('+')
  let digits = s.replace(/[^\d]/g, '')
  if (!digits) return ''
  if (hasPlus) return `+${digits}`
  if (digits.startsWith('00')) {
    digits = digits.slice(2)
    return digits ? `+${digits}` : ''
  }
  return digits
}

/** True when a string looks like a phone number / phone-ish handle (digits, +, spacing). */
export function looksLikePhone(raw: string | null | undefined): boolean {
  const s = String(raw ?? '').trim()
  if (!s) return false
  // An email is never a phone, and a string with letters (a real name) isn't either.
  if (s.includes('@')) return false
  if (/[a-z]/i.test(s)) return false
  // Phone-shaped: only +, digits, and the usual separators ( ) - . space, with >=5 digits.
  if (!/^[+(\d][\d\s().+-]*$/.test(s)) return false
  return s.replace(/[^\d]/g, '').length >= 5
}

/**
 * Normalize any platform handle to its stable lookup form:
 *   phone  -> E.164-ish (normalizePhone)
 *   email  -> lower-cased, trimmed, mailto: stripped
 *   slack/discord/telegram ids -> trimmed, leading '@' stripped, lower-cased
 * A WhatsApp JID ("447700900123@example.net", the "<number>@<host>" form) is reduced to its
 * phone. Returns '' for empty input.
 */
export function normalizeHandle(platform: ContactPlatform, raw: string | null | undefined): string {
  const s = String(raw ?? '').trim()
  if (!s) return ''
  if (platform === 'phone') {
    // WhatsApp JID or "number@host" → take the user part. A JID's user part is ALWAYS the
    // full international number (country code first), so treat it as E.164 by forcing a
    // leading '+' before normalizing; a plain handle is normalized as-is.
    if (s.includes('@')) {
      const userPart = s.split('@')[0].replace(/[^\d]/g, '')
      return userPart ? normalizePhone(`+${userPart}`) : ''
    }
    return normalizePhone(s)
  }
  if (platform === 'email') {
    return s.replace(/^mailto:/i, '').trim().toLowerCase()
  }
  // ids: strip a leading @, lower-case for case-insensitive match.
  return s.replace(/^@+/, '').trim().toLowerCase()
}

/** Stable map key for a normalized (platform, handle) pair. */
export function contactKey(platform: ContactPlatform, normalizedHandle: string): string {
  return `${platform}:${normalizedHandle}`
}

/**
 * Pull a display name out of an email From-style header value, e.g.
 *   '"Maya Chen" <maya@example.com>'  -> 'Maya Chen'
 *   'Maya Chen <maya@example.com>'    -> 'Maya Chen'
 *   'maya@example.com'                -> ''   (no display name, just an address)
 * Returns '' when there is no human name distinct from the address.
 */
export function displayNameFromEmailFrom(from: string | null | undefined): string {
  const s = String(from ?? '').trim()
  if (!s) return ''
  const m = /^\s*"?([^"<]*?)"?\s*<([^>]+)>\s*$/.exec(s)
  if (m) {
    const name = m[1].trim()
    return name && name.toLowerCase() !== m[2].trim().toLowerCase() ? name : ''
  }
  // No angle brackets: it's a bare address (or a bare name). If it looks like an address,
  // there's no display name to learn.
  return s.includes('@') ? '' : s
}

/** The bare email address out of a From-style value ('"Maya" <maya@x>' -> 'maya@x'). */
export function emailAddressFromFrom(from: string | null | undefined): string {
  const s = String(from ?? '').trim()
  const m = /<([^>]+)>/.exec(s)
  const addr = m ? m[1] : s
  return addr.replace(/^mailto:/i, '').trim().toLowerCase()
}

// ---- Learning -----------------------------------------------------------------

const GENERIC_NAMES = new Set(['unknown', 'unknown sender', 'you', 'me', 'unknown contact', 'unknown user', ''])

/** A name worth storing: non-empty, not a generic placeholder, and not equal to the handle. */
export function isUsableName(name: string | null | undefined, normalizedHandle: string): boolean {
  const n = String(name ?? '').trim()
  if (!n) return false
  if (GENERIC_NAMES.has(n.toLowerCase())) return false
  if (n.toLowerCase() === normalizedHandle.toLowerCase()) return false
  // A name that's just the raw handle with formatting (e.g. "+44 7700 900123") is not a name.
  if (looksLikePhone(n) && normalizePhone(n) === normalizedHandle) return false
  return true
}

/**
 * Fold one observation into the directory, returning the NEXT directory (immutably — the
 * input is never mutated). A new pairing is inserted; an existing one is updated only when
 * the incoming name is at least as confident (a high-signal address-book name is never
 * clobbered by a low-signal username) — but its `count` always increments so freshness is
 * tracked. Observations with no usable name still register the handle (count/confidence)
 * so resolveHandle can later surface "handle seen, name unknown" without inventing a name.
 */
export function learnContact(dir: ContactDirectory, obs: ContactObservation, now = Date.now()): ContactDirectory {
  const handle = normalizeHandle(obs.platform, obs.handle)
  if (!handle) return dir
  const key = contactKey(obs.platform, handle)
  const incomingName = String(obs.name ?? '').trim()
  const usable = isUsableName(incomingName, handle)
  const existing = dir.records[key]
  const incomingConf = obs.confidence ?? defaultConfidence(obs.platform, usable)

  if (!existing) {
    const record: ContactRecord = {
      handle,
      rawHandle: String(obs.handle ?? '').trim(),
      platform: obs.platform,
      name: usable ? incomingName : '',
      source: obs.source ?? obs.platform,
      confidence: incomingConf,
      count: 1,
      updatedAt: now,
    }
    return { records: { ...dir.records, [key]: record }, updatedAt: now }
  }

  // Update in place (immutably). The name is replaced only when the incoming one is usable
  // AND at least as confident as what we already have (so we don't downgrade a good name).
  const keepExistingName = !usable || (existing.name && CONFIDENCE_RANK[incomingConf] < CONFIDENCE_RANK[existing.confidence])
  const nextName = keepExistingName ? existing.name : incomingName
  const nextConf = keepExistingName ? existing.confidence : incomingConf
  const next: ContactRecord = {
    ...existing,
    name: nextName,
    confidence: nextName ? nextConf : existing.confidence,
    source: keepExistingName ? existing.source : (obs.source ?? obs.platform),
    count: existing.count + 1,
    updatedAt: now,
  }
  return { records: { ...dir.records, [key]: next }, updatedAt: now }
}

/** Fold a batch of observations in one pass (left-to-right). */
export function learnContacts(dir: ContactDirectory, observations: ContactObservation[], now = Date.now()): ContactDirectory {
  let next = dir
  for (const obs of observations ?? []) next = learnContact(next, obs, now)
  return next
}

function defaultConfidence(platform: ContactPlatform, usable: boolean): ContactConfidence {
  if (!usable) return 'low'
  // Address-book-backed platforms (the user's own phone contacts) are high signal; an email
  // display-name is medium (self-asserted); ids resolved to a username are medium.
  if (platform === 'phone') return 'high'
  return 'medium'
}

// ---- Resolution ---------------------------------------------------------------

/**
 * Resolve a handle to its best known display name. Returns null when the handle is unknown
 * or only its number/id is on file (no usable name). Normalizes the handle first so a
 * caller can pass the raw connector value ("+44 7700 900123", "@maya", a JID).
 */
export function resolveName(dir: ContactDirectory, platform: ContactPlatform, rawHandle: string | null | undefined): string | null {
  const handle = normalizeHandle(platform, rawHandle)
  if (!handle) return null
  const rec = dir.records[contactKey(platform, handle)]
  return rec && rec.name ? rec.name : null
}

/**
 * The display string to SHOW for a handle: the resolved name when known, else the raw
 * handle unchanged (so the feed degrades to the number/id rather than a blank). Pure
 * convenience used by the sync wiring.
 */
export function displayFor(dir: ContactDirectory, platform: ContactPlatform, rawHandle: string | null | undefined): string {
  return resolveName(dir, platform, rawHandle) ?? String(rawHandle ?? '').trim()
}

/**
 * Resolve a NAME back to a handle for "reply to <name>". Case/space-insensitive. When
 * `platform` is given the search is scoped to it (you reply on the channel you're on);
 * otherwise the whole directory is searched. Ties (two people called "Alex") break toward
 * the most-confident, then most-recently-seen record — and an exact (case-insensitive)
 * full-name match always beats a partial/substring match. Returns null when no name matches.
 */
export function resolveHandle(
  dir: ContactDirectory,
  name: string | null | undefined,
  platform?: ContactPlatform,
): ContactRecord | null {
  const q = String(name ?? '').trim().toLowerCase()
  if (!q) return null
  const pool = Object.values(dir.records).filter((r) => r.name && (!platform || r.platform === platform))
  if (!pool.length) return null
  const exact = pool.filter((r) => r.name.trim().toLowerCase() === q)
  const partial = pool.filter((r) => {
    const n = r.name.trim().toLowerCase()
    return n !== q && (n.includes(q) || q.includes(n))
  })
  const ranked = (exact.length ? exact : partial).sort(rankRecords)
  return ranked[0] ?? null
}

/** Best-first ordering: higher confidence, then more sightings, then more recent. */
function rankRecords(a: ContactRecord, b: ContactRecord): number {
  if (CONFIDENCE_RANK[b.confidence] !== CONFIDENCE_RANK[a.confidence]) return CONFIDENCE_RANK[b.confidence] - CONFIDENCE_RANK[a.confidence]
  if (b.count !== a.count) return b.count - a.count
  return b.updatedAt - a.updatedAt
}

/**
 * All distinct handles that match a name (every "Alex" across platforms), best-first. Lets
 * the reply flow disambiguate when more than one person shares a name instead of silently
 * picking one.
 */
export function resolveHandles(dir: ContactDirectory, name: string | null | undefined, platform?: ContactPlatform): ContactRecord[] {
  const q = String(name ?? '').trim().toLowerCase()
  if (!q) return []
  return Object.values(dir.records)
    .filter((r) => r.name && (!platform || r.platform === platform) && r.name.trim().toLowerCase().includes(q))
    .sort(rankRecords)
}

// ---- Maintenance --------------------------------------------------------------

/**
 * Collapse duplicate records that normalize to the same (platform, handle). A directory
 * built from mixed-format inputs over time can hold near-dupes (the same number stored once
 * as "+447…" and once national); re-keying everything through normalizeHandle merges them,
 * keeping the most-confident/most-seen name and SUMMING the counts. Returns a fresh,
 * canonical directory. Idempotent.
 */
export function dedupeContacts(dir: ContactDirectory, now = Date.now()): ContactDirectory {
  const merged: Record<string, ContactRecord> = {}
  for (const rec of Object.values(dir.records)) {
    const handle = normalizeHandle(rec.platform, rec.rawHandle || rec.handle)
    if (!handle) continue
    const key = contactKey(rec.platform, handle)
    const canonical: ContactRecord = { ...rec, handle }
    const prev = merged[key]
    if (!prev) {
      merged[key] = canonical
      continue
    }
    // Pick the better-named record; sum sightings; keep the freshest timestamp.
    const better = pickBetterName(prev, canonical)
    merged[key] = {
      ...better,
      count: prev.count + canonical.count,
      updatedAt: Math.max(prev.updatedAt, canonical.updatedAt),
    }
  }
  return { records: merged, updatedAt: now }
}

function pickBetterName(a: ContactRecord, b: ContactRecord): ContactRecord {
  if (!!a.name !== !!b.name) return a.name ? a : b // a named record beats an unnamed one
  return rankRecords(a, b) <= 0 ? a : b
}

/** Number of records currently in the directory (for status/observability). */
export function contactCount(dir: ContactDirectory): number {
  return Object.keys(dir.records).length
}
