// Sleep-time memory consolidation (Wave 7+). A PURE consolidation pass over a set of durable
// ProjectMemory entries — the counterpart to memory-harvest.ts, which only ever PROPOSES appends.
//
// Where harvest grows memory, consolidation reconciles it:
//   1) DEDUPE     — collapse near-identical entries into a survivor (keeps the most trustworthy).
//   2) SUPERSEDE  — when two entries about the same subject contradict, the newer/higher-confidence
//                   one wins; the loser is retired (superseded), not silently dropped.
//   3) INVALIDATE — flag entries whose backing source is gone or that have gone stale past a TTL.
//
// The pass is a pure function: it takes entries (+ optional source snapshots + config) and returns
// a ConsolidationPlan describing every operation and the resulting reduced entry set. It performs
// NO I/O and mutates nothing — the caller (a background job) applies the plan to the real store.
// Clock and the similarity function are injectable so the whole thing is unit-testable.

import type { ProjectMemoryConfidence, ProjectMemoryEntry } from './types'

// ── tokenization (mirrors memory-harvest's approach; kept local so this module stands alone) ──
const STOP = new Set(
  'the a an of to and or in on for with is are be it this that you i we my your our as at by from not no do does did has have had will would should can could than then them they he she his her its'.split(
    ' ',
  ),
)

function tokenSet(text: string): Set<string> {
  return new Set(
    (String(text ?? '').toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length > 2 && !STOP.has(t)),
  )
}

function intersectionSize(ta: Set<string>, tb: Set<string>): number {
  let inter = 0
  for (const t of ta) if (tb.has(t)) inter++
  return inter
}

/** Jaccard similarity of the significant tokens of two texts (0..1) — whole-text near-identity. */
export function memorySimilarity(a: string, b: string): number {
  const ta = tokenSet(a)
  const tb = tokenSet(b)
  if (!ta.size || !tb.size) return 0
  const inter = intersectionSize(ta, tb)
  const union = ta.size + tb.size - inter
  return union === 0 ? 0 : inter / union
}

/**
 * Overlap coefficient (intersection / smaller set) of two texts (0..1) — SUBJECT match. Used for
 * contradiction detection: a superseding entry often restates the subject and ADDS new content
 * (e.g. "…we use Vercel instead"), which dilutes Jaccard but not the overlap coefficient.
 */
export function memorySubjectOverlap(a: string, b: string): number {
  const ta = tokenSet(a)
  const tb = tokenSet(b)
  if (!ta.size || !tb.size) return 0
  return intersectionSize(ta, tb) / Math.min(ta.size, tb.size)
}

// Negation / contradiction markers: entries that share a subject but flip polarity contradict.
const NEGATION = /\b(no longer|not|never|don'?t|do not|doesn'?t|isn'?t|aren'?t|deprecated|dropped|removed|stopped|instead of|replaced|migrat(?:e|ed|ing)\s+(?:away|off)|avoid)\b/i

function polarity(text: string): boolean {
  return !NEGATION.test(String(text ?? ''))
}

const CONFIDENCE_RANK: Record<ProjectMemoryConfidence, number> = { low: 0, medium: 1, high: 2 }

export interface MemorySourceState {
  /** Whether the backing source still exists (file present, connector reachable, …). */
  exists: boolean
  /** Last-modified time of the source, if known. */
  updatedAt?: number | null
}

export type ConsolidationOpKind = 'dedupe' | 'supersede' | 'invalidate'

export interface ConsolidationOp {
  kind: ConsolidationOpKind
  /** The entry being retired/flagged. */
  entryId: string
  /** For dedupe/supersede: the surviving entry that absorbed or replaced this one. */
  survivorId?: string
  /** Machine-readable reason. */
  reason:
    | 'near_identical'
    | 'contradicted_by_newer'
    | 'contradicted_by_higher_confidence'
    | 'source_missing'
    | 'stale_ttl'
  /** Human-readable explanation for surfacing in review UI / logs. */
  detail: string
}

export interface ConsolidationPlan {
  /** Entries that remain after dedupe + supersede (invalidation does NOT remove — it flags). */
  survivors: ProjectMemoryEntry[]
  /** Entry ids to drop from the store (dedupe losers + superseded losers). */
  removeIds: string[]
  /** Entry ids that should be marked invalid/needs-recheck but kept for the audit trail. */
  invalidateIds: string[]
  /** Full, ordered list of operations for logging / review. */
  ops: ConsolidationOp[]
}

export interface ConsolidateOptions {
  /** Jaccard threshold at/above which two entries are treated as near-identical. Default 0.72. */
  dedupeThreshold?: number
  /** Subject-overlap threshold at/above which two polarity-flipped entries are treated as the same
   *  subject (and thus a contradiction). Default 0.5. */
  contradictionThreshold?: number
  /** Entries older than this many ms since lastVerifiedAt (or createdAt) are stale. Default: off. */
  staleAfterMs?: number
  /** Source states keyed by sourceRef, for invalidation of source-backed entries. */
  sources?: Record<string, MemorySourceState | undefined>
  /** Injectable clock. */
  now?: number
  /** Injectable near-identity function for DEDUPE (defaults to token Jaccard). */
  similarity?: (a: string, b: string) => number
  /** Injectable subject-match function for SUPERSEDE (defaults to overlap coefficient). */
  subjectOverlap?: (a: string, b: string) => number
}

// A survivor is preferred when it is more trustworthy: higher confidence, else more-recently
// verified, else more-recently updated, else lexicographically-stable id (deterministic tiebreak).
function moreTrustworthy(a: ProjectMemoryEntry, b: ProjectMemoryEntry): boolean {
  const ca = CONFIDENCE_RANK[a.confidence] ?? 1
  const cb = CONFIDENCE_RANK[b.confidence] ?? 1
  if (ca !== cb) return ca > cb
  const va = a.lastVerifiedAt ?? 0
  const vb = b.lastVerifiedAt ?? 0
  if (va !== vb) return va > vb
  if (a.updatedAt !== b.updatedAt) return a.updatedAt > b.updatedAt
  return a.id < b.id
}

// Newer-wins for contradictions: verified-then-updated recency, confidence as final tiebreak.
function newerWins(
  a: ProjectMemoryEntry,
  b: ProjectMemoryEntry,
): { winner: ProjectMemoryEntry; loser: ProjectMemoryEntry; byConfidence: boolean } {
  const va = a.lastVerifiedAt ?? a.updatedAt
  const vb = b.lastVerifiedAt ?? b.updatedAt
  if (va !== vb) {
    return va > vb ? { winner: a, loser: b, byConfidence: false } : { winner: b, loser: a, byConfidence: false }
  }
  const ca = CONFIDENCE_RANK[a.confidence] ?? 1
  const cb = CONFIDENCE_RANK[b.confidence] ?? 1
  if (ca !== cb) return ca > cb ? { winner: a, loser: b, byConfidence: true } : { winner: b, loser: a, byConfidence: true }
  // Deterministic final tiebreak.
  return a.id <= b.id ? { winner: a, loser: b, byConfidence: false } : { winner: b, loser: a, byConfidence: false }
}

function snippet(text: string, max = 60): string {
  const t = String(text ?? '').replace(/\s+/g, ' ').trim()
  return t.length > max ? `${t.slice(0, max - 1)}…` : t
}

/**
 * Consolidate a set of memory entries. Pure — returns a plan, mutates nothing.
 *
 * Order of operations (later steps only see survivors of earlier ones):
 *   dedupe → supersede contradictions → invalidate stale/orphaned.
 * Dedupe/supersede are scoped to same projectId + same kind; cross-kind facts are legitimately
 * distinct and never merged.
 */
export function consolidateMemory(
  entries: ProjectMemoryEntry[],
  options: ConsolidateOptions = {},
): ConsolidationPlan {
  const sim = options.similarity ?? memorySimilarity
  const subjectSim = options.subjectOverlap ?? memorySubjectOverlap
  const dedupeThreshold = options.dedupeThreshold ?? 0.72
  const contradictionThreshold = options.contradictionThreshold ?? 0.5
  const now = options.now ?? Date.now()
  const sources = options.sources ?? {}

  const ops: ConsolidationOp[] = []
  const removed = new Set<string>()

  // Stable working copy.
  let living = entries.slice()

  const sameGroup = (a: ProjectMemoryEntry, b: ProjectMemoryEntry) =>
    a.projectId === b.projectId && a.kind === b.kind

  // ── 1) DEDUPE: collapse near-identical same-group entries onto the most trustworthy survivor. ──
  for (let i = 0; i < living.length; i++) {
    const a = living[i]
    if (removed.has(a.id)) continue
    for (let j = i + 1; j < living.length; j++) {
      const b = living[j]
      if (removed.has(b.id)) continue
      if (!sameGroup(a, b)) continue
      if (sim(a.text, b.text) < dedupeThreshold) continue
      // Same polarity required — a flipped near-duplicate is a contradiction, handled below.
      if (polarity(a.text) !== polarity(b.text)) continue
      const survivor = moreTrustworthy(a, b) ? a : b
      const loser = survivor === a ? b : a
      removed.add(loser.id)
      ops.push({
        kind: 'dedupe',
        entryId: loser.id,
        survivorId: survivor.id,
        reason: 'near_identical',
        detail: `Near-duplicate of "${snippet(survivor.text)}" — kept the ${survivor.confidence}-confidence copy.`,
      })
    }
  }
  living = living.filter((e) => !removed.has(e.id))

  // ── 2) SUPERSEDE: contradictory same-group entries — newer / higher-confidence wins. ──
  for (let i = 0; i < living.length; i++) {
    const a = living[i]
    if (removed.has(a.id)) continue
    for (let j = i + 1; j < living.length; j++) {
      const b = living[j]
      if (removed.has(b.id)) continue
      if (!sameGroup(a, b)) continue
      // Same subject (enough token overlap) but opposite polarity ⇒ contradiction.
      if (polarity(a.text) === polarity(b.text)) continue
      if (subjectSim(a.text, b.text) < contradictionThreshold) continue
      const { winner, loser, byConfidence } = newerWins(a, b)
      removed.add(loser.id)
      ops.push({
        kind: 'supersede',
        entryId: loser.id,
        survivorId: winner.id,
        reason: byConfidence ? 'contradicted_by_higher_confidence' : 'contradicted_by_newer',
        detail: `Contradicted by "${snippet(winner.text)}" — retired the ${byConfidence ? 'lower-confidence' : 'older'} claim.`,
      })
    }
  }
  living = living.filter((e) => !removed.has(e.id))

  // ── 3) INVALIDATE: flag (don't remove) survivors whose source vanished or that went stale. ──
  const invalidate = new Set<string>()
  for (const e of living) {
    const ref = e.sourceRef?.trim() || null
    const state = ref ? sources[ref] : undefined
    if (state && !state.exists) {
      invalidate.add(e.id)
      ops.push({
        kind: 'invalidate',
        entryId: e.id,
        reason: 'source_missing',
        detail: `Backing source ${ref} no longer exists — needs recheck before reuse.`,
      })
      continue
    }
    if (options.staleAfterMs != null) {
      const anchor = e.lastVerifiedAt ?? e.createdAt
      if (anchor != null && now - anchor > options.staleAfterMs) {
        invalidate.add(e.id)
        ops.push({
          kind: 'invalidate',
          entryId: e.id,
          reason: 'stale_ttl',
          detail: `Not verified in ${Math.round((now - anchor) / 86_400_000)}d — past the freshness window.`,
        })
      }
    }
  }

  return {
    survivors: living,
    removeIds: [...removed],
    invalidateIds: [...invalidate],
    ops,
  }
}

/**
 * Apply an invalidation flag to an entry immutably: clears lastVerifiedAt and bumps updatedAt so
 * project-memory's detectProjectMemoryWarnings surfaces it as "unverified". Provided so the caller
 * doesn't have to know the shape; still pure.
 */
export function applyInvalidation(entry: ProjectMemoryEntry, now = Date.now()): ProjectMemoryEntry {
  return { ...entry, lastVerifiedAt: null, updatedAt: now }
}
