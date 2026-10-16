// Bi-temporal knowledge-graph memory (contained core).
//
// The existing memory substrate (electron/agent/project-memory.ts + memory-harvest.ts) is
// FLAT: a bag of text entries. It cannot answer "what did we believe about X *as of* last
// Tuesday?" nor "when did this fact become true / stop being true?", and it has no notion of
// one fact SUPERSEDING (contradicting) another over time.
//
// This module adds the CONTAINED core of a bi-temporal entity/relation store:
//
//   * Two independent time axes per fact ("bi-temporal"):
//       - VALID time   (validAt .. invalidAt)   — when the fact is true in the world.
//       - SYSTEM time  (recordedAt .. redactedAt) — when the store knew/believed the fact.
//     This lets you ask both "what was true then?" (valid axis) and "what did we know then?"
//     (system axis), and to see corrections without losing history.
//
//   * insertFact with automatic SUPERSEDE (contradiction handling): a new fact for the same
//     (subject, predicate) tuple closes out the previously-open conflicting fact by setting
//     its invalidAt to the new fact's validAt — instead of deleting it. Old belief stays on
//     the record (append-only), it's just bounded in valid-time.
//
//   * Point-in-time queries: resolve the graph "as of" a given valid instant and/or a given
//     system instant. History is never mutated destructively, so past snapshots stay stable.
//
//   * Connected recall: breadth-first walk of the relation graph from a seed entity, honoring
//     the same bi-temporal filter, so recall follows edges that were live at the query time.
//
// PURE + deterministic: no I/O, no Date.now() inside logic — every mutating call takes an
// explicit `now`. Ids are injectable. This mirrors the repo's dependency-injection test style
// (ssh clientFactory, connector-retry fetchImpl). Integration (persistence into ~/.vontra,
// wiring into the agent loop / recall prompt) is DEFERRED and reported in `wires`.

/** Sentinel meaning "still open / not yet ended" on either time axis. */
export const OPEN = Number.POSITIVE_INFINITY

/** A node in the graph — a person, project, tool, preference target, etc. */
export interface Entity {
  id: string
  type: string
  name: string
  /** free-form attributes (not time-versioned themselves; version by re-asserting facts) */
  attrs?: Record<string, string | number | boolean>
}

/**
 * A bi-temporal fact / edge: subject --predicate--> object(+value).
 * `object` is optional so a fact can be a unary attribute ("Alex prefers dark-mode = true").
 */
export interface Fact {
  id: string
  subject: string // Entity id
  predicate: string
  object?: string | null // Entity id (relation) or null (attribute-style)
  value?: string | number | boolean | null // literal payload for attribute-style facts
  // Valid-time axis: when the fact is true in the world.
  validAt: number
  invalidAt: number // OPEN while still true
  // System-time axis: when the store recorded / retracted its belief in this fact.
  recordedAt: number
  redactedAt: number // OPEN while still believed
  /** provenance / confidence carried through from the flat memory layer, optional. */
  source?: string | null
  confidence?: number | null
}

export interface KnowledgeGraph {
  entities: Map<string, Entity>
  facts: Fact[]
}

export function createGraph(): KnowledgeGraph {
  return { entities: new Map(), facts: [] }
}

let __seq = 0
function autoId(prefix: string): string {
  __seq += 1
  return `${prefix}_${__seq.toString(36)}`
}

// ── Entities ────────────────────────────────────────────────────────────────

export interface UpsertEntityInput {
  id?: string
  type: string
  name: string
  attrs?: Record<string, string | number | boolean>
}

/** Insert or merge an entity (pure: returns the entity, mutates the passed graph's map). */
export function upsertEntity(graph: KnowledgeGraph, input: UpsertEntityInput): Entity {
  if (!input.type?.trim()) throw new Error('entity requires a type')
  if (!input.name?.trim()) throw new Error('entity requires a name')
  const id = input.id?.trim() || autoId('e')
  const existing = graph.entities.get(id)
  const entity: Entity = {
    id,
    type: input.type,
    name: input.name,
    attrs: { ...(existing?.attrs ?? {}), ...(input.attrs ?? {}) },
  }
  graph.entities.set(id, entity)
  return entity
}

// ── Facts (insert-with-supersede) ─────────────────────────────────────────────

export interface InsertFactInput {
  id?: string
  subject: string
  predicate: string
  object?: string | null
  value?: string | number | boolean | null
  /** valid-time start; defaults to `now`. */
  validAt?: number
  /** valid-time end; defaults to OPEN (still true). */
  invalidAt?: number
  source?: string | null
  confidence?: number | null
}

/** True when two facts describe the SAME slot (same subject+predicate, and same object if relational). */
function sameSlot(a: { subject: string; predicate: string; object?: string | null }, b: Fact): boolean {
  if (a.subject !== b.subject || a.predicate !== b.predicate) return false
  const ao = a.object ?? null
  const bo = b.object ?? null
  // Attribute-style facts (object null) collide on subject+predicate regardless of value —
  // asserting a new value supersedes the old one. Relational facts collide only when the
  // object matches (different objects are independent edges, e.g. "member_of" two teams).
  if (ao === null && bo === null) return true
  return ao === bo
}

function factsEquivalent(input: InsertFactInput, f: Fact): boolean {
  return (
    (input.object ?? null) === (f.object ?? null) &&
    (input.value ?? null) === (f.value ?? null)
  )
}

/**
 * Insert a fact with bi-temporal supersede (contradiction handling).
 *
 * If an OPEN (still-valid, still-believed) fact occupies the same slot and CONTRADICTS the
 * new one (different object/value), it is closed: its `invalidAt` is set to the new fact's
 * `validAt` (bounded in valid-time — the belief was true until now). The old fact is NOT
 * deleted, preserving history for point-in-time queries.
 *
 * If an OPEN fact in the same slot is EQUIVALENT (same object+value), we treat the new insert
 * as a no-op re-assertion and return the existing fact — no duplicate rows.
 *
 * @param now system-time instant of this write (recordedAt for the new fact).
 */
export function insertFact(graph: KnowledgeGraph, input: InsertFactInput, now: number, id = input.id): Fact {
  if (!input.subject?.trim()) throw new Error('fact requires a subject')
  if (!input.predicate?.trim()) throw new Error('fact requires a predicate')
  if (input.object == null && input.value == null) {
    throw new Error('fact requires an object (relation) or a value (attribute)')
  }
  const validAt = input.validAt ?? now
  const invalidAt = input.invalidAt ?? OPEN
  if (invalidAt < validAt) throw new Error('invalidAt must be >= validAt')

  // Find the currently-open conflicting fact in this slot (believed now, valid-open).
  for (const f of graph.facts) {
    if (f.redactedAt !== OPEN) continue // not currently believed
    if (f.invalidAt !== OPEN) continue // already closed in valid-time
    if (!sameSlot(input, f)) continue
    if (factsEquivalent(input, f)) {
      return f // idempotent re-assertion
    }
    // Contradiction → bound the old fact's valid-time at the new fact's start.
    f.invalidAt = validAt
  }

  const fact: Fact = {
    id: id?.trim() || autoId('f'),
    subject: input.subject,
    predicate: input.predicate,
    object: input.object ?? null,
    value: input.value ?? null,
    validAt,
    invalidAt,
    recordedAt: now,
    redactedAt: OPEN,
    source: input.source ?? null,
    confidence: input.confidence ?? null,
  }
  graph.facts.push(fact)
  return fact
}

/**
 * Retract a fact in SYSTEM-time (we no longer believe we ever knew it — e.g. it was entered
 * in error). Sets redactedAt; valid-time is untouched. Point-in-time queries whose system
 * instant is before `now` still see it (we did believe it then). Returns true if found.
 */
export function retractFact(graph: KnowledgeGraph, factId: string, now: number): boolean {
  const f = graph.facts.find((x) => x.id === factId)
  if (!f || f.redactedAt !== OPEN) return false
  f.redactedAt = now
  return true
}

// ── Point-in-time queries ─────────────────────────────────────────────────────

export interface AsOf {
  /** valid-time instant: "what was true in the world at T?" Defaults to now-open (OPEN-safe). */
  validAt?: number
  /** system-time instant: "what did we believe at S?" Defaults to latest belief. */
  systemAt?: number
}

/** A fact is live at (validAt, systemAt) if the query instant falls in both half-open intervals. */
export function factLiveAt(f: Fact, at: AsOf): boolean {
  const v = at.validAt
  const s = at.systemAt
  if (v != null) {
    if (v < f.validAt || v >= f.invalidAt) return false
  } else {
    // Default valid instant = "currently true" → require open valid-time.
    if (f.invalidAt !== OPEN) return false
  }
  if (s != null) {
    if (s < f.recordedAt || s >= f.redactedAt) return false
  } else {
    // Default system instant = "currently believed" → require open system-time.
    if (f.redactedAt !== OPEN) return false
  }
  return true
}

export interface QueryFactsInput extends AsOf {
  subject?: string
  predicate?: string
  object?: string
}

/** All facts live at the given bi-temporal instant, optionally filtered by slot fields. */
export function queryFacts(graph: KnowledgeGraph, q: QueryFactsInput = {}): Fact[] {
  return graph.facts.filter((f) => {
    if (q.subject != null && f.subject !== q.subject) return false
    if (q.predicate != null && f.predicate !== q.predicate) return false
    if (q.object != null && (f.object ?? null) !== q.object) return false
    return factLiveAt(f, q)
  })
}

/**
 * Resolve a single slot (subject+predicate[+object]) to its winning fact at an instant.
 * When multiple live facts share the slot (shouldn't happen for attributes, can for relations
 * across objects if object omitted), the most-recently-recorded one wins. undefined if none.
 */
export function resolve(
  graph: KnowledgeGraph,
  slot: { subject: string; predicate: string; object?: string },
  at: AsOf = {},
): Fact | undefined {
  const matches = queryFacts(graph, { ...at, ...slot })
  if (matches.length === 0) return undefined
  return matches.reduce((best, f) => (f.recordedAt >= best.recordedAt ? f : best))
}

// ── Connected recall ──────────────────────────────────────────────────────────

export interface ConnectedRecallInput extends AsOf {
  /** seed entity id to walk out from. */
  from: string
  /** max graph distance (edges) to traverse. */
  depth?: number
  /** only traverse edges whose predicate is in this set (all if omitted). */
  predicates?: string[]
}

export interface ConnectedRecallResult {
  /** entity ids reached, with their BFS distance from the seed (seed = 0). */
  entities: Map<string, number>
  /** the live edges traversed, in discovery order. */
  facts: Fact[]
}

/**
 * Breadth-first connected recall over the bi-temporal graph: starting at `from`, follow live
 * relational edges (facts with a non-null object) up to `depth` hops, honoring the same
 * (validAt, systemAt) filter. Edges are undirected for traversal (subject<->object) so recall
 * reaches neighbors regardless of edge direction, but each returned fact keeps its direction.
 */
export function connectedRecall(graph: KnowledgeGraph, input: ConnectedRecallInput): ConnectedRecallResult {
  const depth = input.depth ?? 2
  const predSet = input.predicates ? new Set(input.predicates) : null
  const at: AsOf = { validAt: input.validAt, systemAt: input.systemAt }

  const dist = new Map<string, number>([[input.from, 0]])
  const facts: Fact[] = []
  const seenFacts = new Set<string>()
  let frontier = [input.from]

  for (let d = 0; d < depth && frontier.length > 0; d++) {
    const next: string[] = []
    for (const node of frontier) {
      for (const f of graph.facts) {
        if (f.object == null) continue // attribute facts don't connect entities
        if (predSet && !predSet.has(f.predicate)) continue
        if (f.subject !== node && f.object !== node) continue
        if (!factLiveAt(f, at)) continue
        if (!seenFacts.has(f.id)) {
          seenFacts.add(f.id)
          facts.push(f)
        }
        const neighbor = f.subject === node ? f.object : f.subject
        if (neighbor != null && !dist.has(neighbor)) {
          dist.set(neighbor, d + 1)
          next.push(neighbor)
        }
      }
    }
    frontier = next
  }

  return { entities: dist, facts }
}

/** Full valid-time history for a slot (every version, newest first), ignoring the system axis
 *  by default so you can render a timeline. Redacted facts are excluded unless includeRedacted. */
export function slotHistory(
  graph: KnowledgeGraph,
  slot: { subject: string; predicate: string; object?: string },
  opts: { includeRedacted?: boolean } = {},
): Fact[] {
  return graph.facts
    .filter((f) => {
      if (f.subject !== slot.subject || f.predicate !== slot.predicate) return false
      if (slot.object != null && (f.object ?? null) !== slot.object) return false
      if (!opts.includeRedacted && f.redactedAt !== OPEN) return false
      return true
    })
    .sort((a, b) => b.validAt - a.validAt)
}
