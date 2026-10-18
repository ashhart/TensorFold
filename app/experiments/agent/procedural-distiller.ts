// Distill reusable, repo-specific procedural notes from recorded run trajectories.
//
// The orchestration journal (orchestration-journal.ts) records one NDJSON line per
// mission/room lifecycle transition. Over many runs those journals accumulate the raw
// evidence of *how work actually goes wrong here*: a room finalizes 'failed' with an
// error message, a seat stalls with a note, a mission requeues with a reason. This
// module is the pure lens that reads those trajectories — especially the failures — and
// distills them into short, reusable "how-to" notes a future run can be primed with
// (e.g. "tests need env X", "last migration broke because Y", "the deploy seat keeps
// stalling on network").
//
// It is deliberately dependency-free and side-effect-free: it takes already-parsed
// journal events (from parseJournal/readJournal) plus a clock, and returns plain note
// objects. The caller decides whether to turn a note into a project-memory candidate
// (project-memory.ts#createProjectMemoryCandidate) — this module never touches disk,
// storage, or the network. That mirrors memory-harvest.ts (harvest, don't persist).
//
// Design goals:
//  - Signal, not noise: only failure / stall / requeue events with an extractable
//    reason become notes; clean successes never mint a note.
//  - Recurrence = confidence: a reason seen once is 'low', a few times 'medium', many
//    times 'high'. Frequency is the whole point — a one-off flake is weaker evidence
//    than a pattern that bites every run.
//  - Stable, deduped output: reasons are normalized (volatile ids/paths/hashes/numbers
//    stripped) so "run 7 failed: exit 1 in /tmp/abc" and "run 9 failed: exit 1 in
//    /tmp/def" collapse to one note with occurrences: 2.
//  - Citations preserved: each note keeps the seq of its first/last supporting event so
//    a reader can trace it back into the journal.

import type { OrchestrationJournalEvent } from '../../electron/agent/orchestration-journal.ts'

// The kinds of trajectory signal a note can come from. Kept coarse on purpose so the
// prompt/UI can group by it.
export type ProceduralNoteCategory =
  | 'failure' // a run/room ended failed, or a step errored
  | 'stall' // a seat/agent stopped making progress
  | 'retry' // work had to be requeued / re-run
  | 'incomplete' // a run finalized without a real final answer

// Confidence derived purely from how often the same normalized reason recurred.
export type ProceduralNoteConfidence = 'low' | 'medium' | 'high'

// One distilled procedural note. `text` is the ready-to-surface how-to line; the rest is
// provenance so a caller can rank, dedupe against existing memory, or cite it.
export interface ProceduralNote {
  // Stable id derived from category+normalized reason, so the same lesson keeps the same
  // id across distills (useful for dedupe/upsert against stored notes).
  id: string
  category: ProceduralNoteCategory
  // Human-facing note, e.g. "Runs repeatedly fail: migration timed out — check before retrying."
  text: string
  // The normalized reason the note clusters on (volatile tokens stripped).
  reason: string
  // How many trajectory events supported this note.
  occurrences: number
  confidence: ProceduralNoteConfidence
  // Which subjects (mission/room/seat ids) exhibited this, deduped + capped.
  subjects: string[]
  // seq of the first and last supporting event, for tracing back into the journal.
  firstSeq: number
  lastSeq: number
  // ts of the most recent supporting event.
  lastTs: number
}

// A trajectory is one subject's ordered event stream (a mission or a room). The caller
// groups journals per id; this module does not care which kind.
export interface Trajectory {
  // mission/room id — used as the subject label on notes.
  subject: string
  events: OrchestrationJournalEvent[]
}

export interface DistillOptions {
  // Max notes returned (highest-confidence / most-recurrent first). Default 12.
  maxNotes?: number
  // Drop notes whose reason recurred fewer than this many times. Default 1 (keep all).
  minOccurrences?: number
  // Cap on subjects listed per note. Default 5.
  maxSubjectsPerNote?: number
}

// A raw signal pulled from one event before clustering.
interface Signal {
  category: ProceduralNoteCategory
  reason: string
  subject: string
  seq: number
  ts: number
}

const strFrom = (payload: Record<string, unknown> | undefined, ...keys: string[]): string | undefined => {
  if (!payload) return undefined
  for (const key of keys) {
    const value = payload[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return undefined
}

const boolFrom = (payload: Record<string, unknown> | undefined, key: string): boolean => {
  return payload ? payload[key] === true : false
}

// Strip the volatile parts of a reason so recurrences of the *same* underlying problem
// cluster together: uuids/hashes, absolute paths, quoted strings, standalone numbers,
// hex/ports, "run N"/"attempt N", and collapse whitespace. Case-folded, length-bounded.
export function normalizeReason(raw: string): string {
  let s = String(raw ?? '')
    .replace(/\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi, ' ') // uuids
    .replace(/\b[0-9a-f]{7,}\b/gi, ' ') // git shas / long hex ids
    .replace(/([a-zA-Z]:)?[\/~][\w./~-]+/g, ' ') // unix/windows-ish paths
    .replace(/["'`][^"'`]*["'`]/g, ' ') // quoted literals (filenames, values)
    .replace(/\b(?:run|attempt|try|pass|tick|turn|seq|line|pid|port)\s*#?\s*\d+\b/gi, ' ') // "run 7", "line 42"
    .replace(/:\d+(?::\d+)?\b/g, ' ') // trailing :line:col
    .replace(/0x[0-9a-f]+/gi, ' ') // hex addrs
    // numbers, incl. ones glued to a short unit suffix (3000ms, 200MB, 5s, exit1's "1")
    .replace(/\b\d[\d,._]*(ms|s|m|h|mb|kb|gb|b|px|%)?\b/gi, ' ')
    .replace(/[^\w\s.:;/+-]+/g, ' ') // stray punctuation
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
  // Guard against reasons that normalized to nothing (e.g. was just an id/number).
  if (!s) s = String(raw ?? '').replace(/\s+/g, ' ').trim().toLowerCase()
  return s.slice(0, 160)
}

// A short, stable id for a note from its category + normalized reason.
function noteId(category: ProceduralNoteCategory, reason: string): string {
  let h = 5381
  const basis = `${category}:${reason}`
  for (let i = 0; i < basis.length; i++) h = ((h << 5) + h + basis.charCodeAt(i)) | 0
  return `pn_${(h >>> 0).toString(36)}`
}

function confidenceFor(occurrences: number): ProceduralNoteConfidence {
  if (occurrences >= 4) return 'high'
  if (occurrences >= 2) return 'medium'
  return 'low'
}

// Turn one journal event into at most one procedural signal. Only failure-shaped events
// yield a signal; clean lifecycle transitions (created/started/checkpoint/turn) do not.
function signalFromEvent(subject: string, event: OrchestrationJournalEvent): Signal | null {
  const p = event.payload
  const base = { subject, seq: event.seq, ts: event.ts }

  switch (event.type) {
    case 'seat_stalled': {
      const reason = strFrom(p, 'note', 'reason') ?? 'seat stalled without a note'
      const seat = strFrom(p, 'seat')
      return { ...base, category: 'stall', reason: seat ? `${seat}: ${reason}` : reason, subject: seat ? `${subject}#${seat}` : subject }
    }
    case 'requeued': {
      const reason = strFrom(p, 'reason')
      if (!reason) return null // a plain requeue with no reason is not a lesson
      return { ...base, category: 'retry', reason }
    }
    case 'finalized': {
      const state = strFrom(p, 'state', 'status', 'to')
      const failed = state === 'failed' || state === 'error' || strFrom(p, 'error') !== undefined
      if (failed) {
        const reason = strFrom(p, 'error', 'reason') ?? `finalized ${state ?? 'failed'}`
        return { ...base, category: 'failure', reason }
      }
      // Finished but with no real final answer (room 'no_final', aborted).
      const noFinal = strFrom(p, 'reason') === 'no_final' || boolFrom(p, 'aborted')
      if (noFinal) {
        const reason = boolFrom(p, 'aborted') ? 'aborted before producing a final answer' : 'finished without a final answer'
        return { ...base, category: 'incomplete', reason }
      }
      return null
    }
    case 'status_changed': {
      const to = strFrom(p, 'to', 'status')
      if (to === 'failed' || to === 'error') {
        const reason = strFrom(p, 'error', 'reason') ?? `status changed to ${to}`
        return { ...base, category: 'failure', reason }
      }
      return null
    }
    default:
      return null
  }
}

// Compose the surfaced note text from category + reason + recurrence. Recurring problems
// get a "repeatedly" framing so the reader treats them as patterns, not flukes.
function noteText(category: ProceduralNoteCategory, reason: string, occurrences: number): string {
  const recurring = occurrences >= 2
  const cleaned = reason.replace(/\s+/g, ' ').trim()
  const lead = cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
  switch (category) {
    case 'failure':
      return recurring
        ? `Runs repeatedly fail here: ${cleaned} — check this before retrying.`
        : `A run failed: ${cleaned}.`
    case 'stall':
      return recurring
        ? `Work repeatedly stalls: ${cleaned} — expect this to need a nudge or timeout.`
        : `Work stalled: ${cleaned}.`
    case 'retry':
      return recurring
        ? `Work repeatedly had to be re-run: ${cleaned} — fix the root cause, not just retry.`
        : `Work was re-run because: ${cleaned}.`
    case 'incomplete':
      return recurring
        ? `Runs repeatedly finish without a real result: ${cleaned}.`
        : `A run finished without a real result: ${lead}.`
  }
}

/**
 * Distill procedural how-to notes from recorded trajectories. Pure: reads events, emits
 * notes. Failures/stalls/retries with an extractable reason are normalized, clustered by
 * (category + normalized reason), and ranked by confidence (recurrence) then recency.
 */
export function distillProceduralNotes(trajectories: Trajectory[], options: DistillOptions = {}): ProceduralNote[] {
  const maxNotes = Math.max(1, options.maxNotes ?? 12)
  const minOccurrences = Math.max(1, options.minOccurrences ?? 1)
  const maxSubjects = Math.max(1, options.maxSubjectsPerNote ?? 5)

  // Cluster by category + normalized reason.
  interface Cluster {
    category: ProceduralNoteCategory
    reason: string
    occurrences: number
    subjects: string[]
    subjectSet: Set<string>
    firstSeq: number
    lastSeq: number
    lastTs: number
    // Keep the most recent raw (un-normalized) reason for the surfaced text — it reads
    // better than the token-stripped form, while clustering still happens on normalized.
    displayReason: string
    displayTs: number
  }
  const clusters = new Map<string, Cluster>()

  for (const trajectory of trajectories) {
    if (!trajectory?.subject) continue
    for (const event of trajectory.events ?? []) {
      const signal = signalFromEvent(trajectory.subject, event)
      if (!signal) continue
      const normalized = normalizeReason(signal.reason)
      if (!normalized) continue
      const key = `${signal.category}::${normalized}`
      let cluster = clusters.get(key)
      if (!cluster) {
        cluster = {
          category: signal.category,
          reason: normalized,
          occurrences: 0,
          subjects: [],
          subjectSet: new Set(),
          firstSeq: signal.seq,
          lastSeq: signal.seq,
          lastTs: signal.ts,
          displayReason: signal.reason.replace(/\s+/g, ' ').trim(),
          displayTs: signal.ts,
        }
        clusters.set(key, cluster)
      }
      cluster.occurrences += 1
      if (!cluster.subjectSet.has(signal.subject)) {
        cluster.subjectSet.add(signal.subject)
        cluster.subjects.push(signal.subject)
      }
      if (signal.seq < cluster.firstSeq) cluster.firstSeq = signal.seq
      if (signal.seq > cluster.lastSeq) cluster.lastSeq = signal.seq
      if (signal.ts > cluster.lastTs) cluster.lastTs = signal.ts
      // Prefer the most recent raw reason for display.
      if (signal.ts >= cluster.displayTs) {
        cluster.displayReason = signal.reason.replace(/\s+/g, ' ').trim()
        cluster.displayTs = signal.ts
      }
    }
  }

  const notes: ProceduralNote[] = []
  for (const cluster of clusters.values()) {
    if (cluster.occurrences < minOccurrences) continue
    notes.push({
      id: noteId(cluster.category, cluster.reason),
      category: cluster.category,
      text: noteText(cluster.category, cluster.displayReason, cluster.occurrences),
      reason: cluster.reason,
      occurrences: cluster.occurrences,
      confidence: confidenceFor(cluster.occurrences),
      subjects: cluster.subjects.slice(0, maxSubjects),
      firstSeq: cluster.firstSeq,
      lastSeq: cluster.lastSeq,
      lastTs: cluster.lastTs,
    })
  }

  // Rank: most-recurrent first, then most-recent, then stable by id.
  notes.sort((a, b) =>
    b.occurrences - a.occurrences ||
    b.lastTs - a.lastTs ||
    (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  )
  return notes.slice(0, maxNotes)
}

/** Format distilled notes as a compact prompt block for priming a future run. */
export function formatProceduralNotesForPrompt(notes: ProceduralNote[], max = 8): string {
  return notes.slice(0, max).map((note) => {
    const cite = note.occurrences > 1 ? ` (seen ${note.occurrences}×)` : ''
    return `- [${note.category}, ${note.confidence}]${cite} ${note.text}`
  }).join('\n')
}
