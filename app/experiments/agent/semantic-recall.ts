// Embedding-based memory recall — replaces literal token/phrase overlap
// (retrieval-index.ts `scoreRecord`) with cosine similarity over vector
// embeddings, so paraphrases and synonyms match ("fix the bug" ↔ "resolve the
// defect"). Pure + dependency-injected: the `Embedder` is supplied by the
// caller (a local-embeddings function, an HTTP embeddings endpoint, or a fake
// in tests), mirroring the ssh-connection clientFactory / connector-retry
// fetchImpl pattern. Nothing here performs I/O on its own.
//
// The parent wires this alongside the existing lexical `searchRetrievalIndex`
// (see the wire report): embed the records once, cache the vectors on an
// `EmbeddedRetrievalIndex`, then rank queries by cosine similarity — optionally
// blended with the recency weight the lexical path already uses.

import type { RetrievalIndex, RetrievalKind, RetrievalRecord } from '../../electron/agent/retrieval-index.ts'

/**
 * A pluggable embedder. Given a batch of texts, returns one dense vector per
 * input, in order. Implementations may be sync or async. Vectors need not be
 * pre-normalized — cosine similarity here normalizes internally.
 */
export interface Embedder {
  /** Optional stable identifier so cached vectors can be invalidated when the
   *  embedding model changes. */
  readonly id?: string
  /** Optional fixed dimensionality; when set, mismatched vectors are rejected. */
  readonly dimensions?: number
  embed(texts: string[]): Promise<number[][]> | number[][]
}

export interface EmbeddedRecord {
  record: RetrievalRecord
  /** L2-normalized embedding of the record's indexed text. */
  vector: number[]
}

export interface EmbeddedRetrievalIndex {
  embedderId: string
  generatedAt: number
  records: EmbeddedRecord[]
}

export interface SemanticSearchOptions {
  /** Cap on returned hits (default 12, clamped 1..100). */
  limit?: number | null
  /** Restrict to these kinds; empty/absent = all kinds. */
  kinds?: RetrievalKind[] | null
  /** Drop hits whose cosine similarity is below this (default 0 = keep all
   *  non-negative). Range -1..1. */
  minScore?: number | null
  /** When set, blend cosine similarity with a caller-supplied recency weight
   *  in [0,1]: final = cosine * (recencyFloor + (1-recencyFloor)*weight).
   *  Omit to rank on pure similarity. */
  recencyWeight?: ((record: RetrievalRecord) => number) | null
  /** Floor applied to the recency blend so an old-but-relevant record still
   *  scores (default 0.5). Ignored when `recencyWeight` is absent. */
  recencyFloor?: number | null
}

export interface SemanticSearchHit {
  record: RetrievalRecord
  /** Blended rank score used for ordering. */
  score: number
  /** Raw cosine similarity in [-1, 1] before any recency blend. */
  similarity: number
}

export interface SemanticSearchResult {
  query: string
  kinds: RetrievalKind[]
  embedderId: string
  total: number
  hits: SemanticSearchHit[]
}

const DEFAULT_LIMIT = 12
const MAX_LIMIT = 100

/** Text that gets embedded for a record: title carries strong signal, so it is
 *  prepended to the body. */
export function recordEmbeddingText(record: RetrievalRecord): string {
  const title = (record.title ?? '').trim()
  const text = (record.text ?? '').trim()
  if (title && text) return `${title}\n${text}`
  return title || text
}

/** L2-normalize a vector. A zero vector normalizes to itself (all zeros), which
 *  yields cosine 0 against anything — a safe "no signal" result. */
export function normalizeVector(vector: number[]): number[] {
  let sumSq = 0
  for (const value of vector) sumSq += value * value
  const norm = Math.sqrt(sumSq)
  if (!(norm > 0) || !Number.isFinite(norm)) return vector.map(() => 0)
  return vector.map((value) => value / norm)
}

/** Cosine similarity of two vectors. Returns 0 for mismatched lengths or a
 *  zero-magnitude operand rather than NaN. */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i]
    const y = b[i]
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  if (!(denom > 0) || !Number.isFinite(denom)) return 0
  const cos = dot / denom
  // Guard floating-point drift just past the [-1, 1] range.
  if (cos > 1) return 1
  if (cos < -1) return -1
  return cos
}

function assertDimensions(vectors: number[][], expected: number | undefined, label: string): void {
  for (const [i, vector] of vectors.entries()) {
    if (!Array.isArray(vector)) throw new Error(`Embedder returned a non-array vector at ${label}[${i}].`)
    if (expected != null && vector.length !== expected) {
      throw new Error(`Embedder returned a ${vector.length}-dim vector at ${label}[${i}], expected ${expected}.`)
    }
  }
}

/**
 * Embed every record in a retrieval index once, producing a reusable
 * `EmbeddedRetrievalIndex`. Records with empty embedding text are skipped
 * (they can never match semantically). Vectors are L2-normalized up front so
 * query-time scoring is a plain dot product.
 */
export async function embedRetrievalIndex(
  index: Pick<RetrievalIndex, 'records' | 'generatedAt'>,
  embedder: Embedder,
): Promise<EmbeddedRetrievalIndex> {
  const usable: RetrievalRecord[] = []
  const texts: string[] = []
  for (const record of index.records) {
    const text = recordEmbeddingText(record)
    if (!text) continue
    usable.push(record)
    texts.push(text)
  }

  const vectors = texts.length ? await embedder.embed(texts) : []
  if (vectors.length !== texts.length) {
    throw new Error(`Embedder returned ${vectors.length} vectors for ${texts.length} texts.`)
  }
  assertDimensions(vectors, embedder.dimensions, 'records')

  const records: EmbeddedRecord[] = usable.map((record, i) => ({
    record,
    vector: normalizeVector(vectors[i]),
  }))

  return {
    embedderId: embedder.id ?? 'anonymous',
    generatedAt: index.generatedAt ?? Date.now(),
    records,
  }
}

/**
 * Rank an embedded index against a natural-language query by cosine similarity.
 * The query is embedded through the same `embedder` (so it lands in the same
 * space). Pure aside from the injected embedder — deterministic given a
 * deterministic embedder.
 */
export async function semanticSearch(
  embedded: EmbeddedRetrievalIndex,
  query: string,
  embedder: Embedder,
  options: SemanticSearchOptions = {},
): Promise<SemanticSearchResult> {
  const cleanQuery = String(query ?? '').trim()
  const kinds = options.kinds && options.kinds.length ? [...new Set(options.kinds)] : []
  const limit = clampLimit(options.limit)
  const minScore = clampScore(options.minScore)
  const recencyFloor = clampFloor(options.recencyFloor)
  const recencyWeight = options.recencyWeight ?? null

  const candidates = kinds.length
    ? embedded.records.filter((entry) => kinds.includes(entry.record.kind))
    : embedded.records

  if (!cleanQuery) {
    return { query: '', kinds, embedderId: embedded.embedderId, total: embedded.records.length, hits: [] }
  }

  const queryVectors = await embedder.embed([cleanQuery])
  if (!queryVectors.length || !Array.isArray(queryVectors[0])) {
    throw new Error('Embedder returned no vector for the query.')
  }
  assertDimensions(queryVectors, embedder.dimensions, 'query')
  const queryVector = normalizeVector(queryVectors[0])

  const hits: SemanticSearchHit[] = []
  for (const entry of candidates) {
    const similarity = cosineSimilarity(queryVector, entry.vector)
    if (similarity < minScore) continue
    let score = similarity
    if (recencyWeight) {
      const raw = recencyWeight(entry.record)
      const weight = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0
      score = similarity * (recencyFloor + (1 - recencyFloor) * weight)
    }
    hits.push({ record: entry.record, score, similarity })
  }

  hits.sort(
    (a, b) =>
      b.score - a.score ||
      (b.record.createdAt ?? 0) - (a.record.createdAt ?? 0) ||
      a.record.title.localeCompare(b.record.title),
  )

  return {
    query: cleanQuery,
    kinds,
    embedderId: embedded.embedderId,
    total: embedded.records.length,
    hits: hits.slice(0, limit),
  }
}

function clampLimit(value: number | null | undefined): number {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_LIMIT
  return Math.max(1, Math.min(Math.floor(n), MAX_LIMIT))
}

function clampScore(value: number | null | undefined): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 0
  return Math.max(-1, Math.min(1, n))
}

function clampFloor(value: number | null | undefined): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return 0.5
  return Math.max(0, Math.min(1, n))
}
