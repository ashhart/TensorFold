// Run: jiti electron/agent/semantic-recall.test.ts
import assert from 'node:assert/strict'
import {
  cosineSimilarity,
  normalizeVector,
  recordEmbeddingText,
  embedRetrievalIndex,
  semanticSearch,
  type Embedder,
} from './semantic-recall.ts'
import type { RetrievalIndex, RetrievalRecord } from './retrieval-index.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('semantic-recall tests:')

function record(partial: Partial<RetrievalRecord>): RetrievalRecord {
  return {
    id: partial.id ?? 't',
    kind: partial.kind ?? 'doc',
    title: partial.title ?? '',
    text: partial.text ?? '',
    sourceIds: partial.sourceIds ?? [],
    path: partial.path ?? null,
    createdAt: partial.createdAt ?? null,
    metadata: partial.metadata,
  }
}

function index(records: RetrievalRecord[]): RetrievalIndex {
  return {
    schema: 'vontra.retrieval-index/1',
    generatedAt: 1000,
    records,
    stats: { total: records.length, byKind: { code: 0, doc: 0, chat: 0, email: 0, summary: 0, artifact: 0 }, sourceIds: 0 },
  }
}

// A fake, DETERMINISTIC embedder: bag-of-concepts over a small synonym-aware
// vocabulary. Paraphrases that map to the same concept land on the same axis,
// which is exactly what token-overlap CANNOT do. Purely for testing — no I/O.
const CONCEPTS: Record<string, string[]> = {
  fix: ['fix', 'resolve', 'repair', 'patch', 'squash'],
  bug: ['bug', 'defect', 'issue', 'error', 'fault'],
  auth: ['auth', 'login', 'signin', 'authentication', 'credentials'],
  db: ['db', 'database', 'postgres', 'sql', 'query'],
}
const CONCEPT_KEYS = Object.keys(CONCEPTS)
function fakeVector(text: string): number[] {
  const words = text.toLowerCase().split(/[^a-z]+/).filter(Boolean)
  const vec = CONCEPT_KEYS.map((key) => {
    const syns = CONCEPTS[key]
    return words.reduce((acc, w) => acc + (syns.includes(w) ? 1 : 0), 0)
  })
  return vec
}
const fakeEmbedder: Embedder = {
  id: 'fake-concepts',
  dimensions: CONCEPT_KEYS.length,
  embed: (texts) => texts.map(fakeVector),
}

// --- pure math ---------------------------------------------------------------
{
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1)
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0)
  assert.equal(Math.round(cosineSimilarity([1, 1], [1, 0]) * 1000) / 1000, 0.707)
  ok('cosineSimilarity: identical=1, orthogonal=0, 45deg≈0.707')
}
{
  // Degenerate inputs return 0, never NaN.
  assert.equal(cosineSimilarity([0, 0], [1, 1]), 0)
  assert.equal(cosineSimilarity([1, 2, 3], [1, 2]), 0)
  assert.equal(cosineSimilarity([], []), 0)
  ok('cosineSimilarity: zero-vector / length-mismatch / empty → 0 (no NaN)')
}
{
  const n = normalizeVector([3, 4])
  assert.equal(Math.round((n[0] * n[0] + n[1] * n[1]) * 1e6) / 1e6, 1)
  assert.deepEqual(normalizeVector([0, 0, 0]), [0, 0, 0])
  ok('normalizeVector: unit length; zero vector stays zero')
}
{
  assert.equal(recordEmbeddingText(record({ title: 'A', text: 'B' })), 'A\nB')
  assert.equal(recordEmbeddingText(record({ title: '', text: 'B' })), 'B')
  assert.equal(recordEmbeddingText(record({ title: 'A', text: '' })), 'A')
  ok('recordEmbeddingText: title prepended; falls back cleanly')
}

// --- the core promise: paraphrase / synonym recall ---------------------------
{
  const idx = index([
    record({ id: 'r1', title: 'Resolve the login defect', text: 'The authentication flow throws an error', createdAt: 5 }),
    record({ id: 'r2', title: 'Postgres query timeout', text: 'Slow database call in reports', createdAt: 5 }),
  ])
  const embedded = await embedRetrievalIndex(idx, fakeEmbedder)
  assert.equal(embedded.records.length, 2)
  assert.equal(embedded.embedderId, 'fake-concepts')

  // Query uses DIFFERENT words than the record: "fix the auth bug".
  // Token overlap would score r1 ≈ 0 ("fix"/"auth"/"bug" don't literally
  // appear); the embedder maps them onto the fix/bug/auth axes → strong match.
  const res = await semanticSearch(embedded, 'fix the auth bug', fakeEmbedder)
  assert.equal(res.hits[0].record.id, 'r1', 'paraphrased auth-bug query should surface r1')
  assert.ok(res.hits[0].similarity > 0.5, `expected strong similarity, got ${res.hits[0].similarity}`)
  ok('semanticSearch: synonym query "fix the auth bug" matches "resolve login defect"')
}

// --- kind filter + limit -----------------------------------------------------
{
  const idx = index([
    record({ id: 'c1', kind: 'code', title: 'fix bug', text: 'patch defect' }),
    record({ id: 'd1', kind: 'doc', title: 'resolve issue', text: 'repair fault' }),
  ])
  const embedded = await embedRetrievalIndex(idx, fakeEmbedder)
  const res = await semanticSearch(embedded, 'fix the bug', fakeEmbedder, { kinds: ['doc'] })
  assert.deepEqual(res.hits.map((h) => h.record.id), ['d1'], 'kind filter should exclude code record')
  assert.deepEqual(res.kinds, ['doc'])

  const limited = await semanticSearch(embedded, 'fix the bug', fakeEmbedder, { limit: 1 })
  assert.equal(limited.hits.length, 1)
  ok('semanticSearch: kind filter and limit respected')
}

// --- minScore drops weak / orthogonal hits -----------------------------------
{
  const idx = index([
    record({ id: 'match', title: 'fix bug', text: '' }),
    record({ id: 'unrelated', title: 'database query', text: '' }),
  ])
  const embedded = await embedRetrievalIndex(idx, fakeEmbedder)
  const res = await semanticSearch(embedded, 'resolve the defect', fakeEmbedder, { minScore: 0.1 })
  assert.deepEqual(res.hits.map((h) => h.record.id), ['match'], 'orthogonal db record filtered by minScore')
  ok('semanticSearch: minScore drops orthogonal records')
}

// --- recency blend re-ranks equal-similarity records -------------------------
{
  const idx = index([
    record({ id: 'old', title: 'fix bug', text: '', createdAt: 1 }),
    record({ id: 'new', title: 'fix bug', text: '', createdAt: 2 }),
  ])
  const embedded = await embedRetrievalIndex(idx, fakeEmbedder)
  // Both have identical similarity to the query. Recency weight favours 'new'.
  const res = await semanticSearch(embedded, 'resolve the defect', fakeEmbedder, {
    recencyWeight: (r) => (r.id === 'new' ? 1 : 0),
    recencyFloor: 0.5,
  })
  assert.equal(res.hits[0].record.id, 'new', 'recency blend should lift the fresher record')
  assert.ok(res.hits[0].score > res.hits[1].score)
  // similarity is preserved untouched alongside the blended score.
  assert.ok(res.hits[0].similarity > 0.9)
  ok('semanticSearch: recencyWeight blend re-ranks equal-similarity ties')
}

// --- empty query and empty index ---------------------------------------------
{
  const embedded = await embedRetrievalIndex(index([record({ id: 'x', title: 'fix bug' })]), fakeEmbedder)
  const res = await semanticSearch(embedded, '   ', fakeEmbedder)
  assert.deepEqual(res.hits, [])
  assert.equal(res.query, '')

  const emptyEmbedded = await embedRetrievalIndex(index([]), fakeEmbedder)
  assert.equal(emptyEmbedded.records.length, 0)
  const emptyRes = await semanticSearch(emptyEmbedded, 'fix bug', fakeEmbedder)
  assert.deepEqual(emptyRes.hits, [])
  ok('semanticSearch: empty query and empty index return no hits')
}

// --- records with no embeddable text are skipped -----------------------------
{
  const idx = index([
    record({ id: 'blank', title: '', text: '' }),
    record({ id: 'real', title: 'fix bug', text: '' }),
  ])
  const embedded = await embedRetrievalIndex(idx, fakeEmbedder)
  assert.deepEqual(embedded.records.map((r) => r.record.id), ['real'])
  ok('embedRetrievalIndex: records with empty title+text are skipped')
}

// --- dimension mismatch is rejected ------------------------------------------
{
  const badEmbedder: Embedder = { id: 'bad', dimensions: 4, embed: () => [[1, 2, 3]] }
  await assert.rejects(
    () => embedRetrievalIndex(index([record({ id: 'r', title: 'fix' })]), badEmbedder),
    /3-dim vector .* expected 4/,
  )
  ok('embedRetrievalIndex: rejects vectors that violate declared dimensions')
}

// --- sync embedder (non-Promise return) works too ----------------------------
{
  const syncEmbedder: Embedder = { id: 'sync', dimensions: 2, embed: (texts) => texts.map(() => [1, 0]) }
  const embedded = await embedRetrievalIndex(index([record({ id: 'r', title: 'anything' })]), syncEmbedder)
  const res = await semanticSearch(embedded, 'query', syncEmbedder)
  assert.equal(res.hits[0].similarity, 1)
  ok('embedder may return a plain array (not just a Promise)')
}

console.log(`\n${passed} passed`)
