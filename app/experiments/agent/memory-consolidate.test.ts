// Run: jiti electron/agent/memory-consolidate.test.ts
import assert from 'node:assert/strict'
import { applyInvalidation, consolidateMemory, memorySimilarity, memorySubjectOverlap, type MemorySourceState } from './memory-consolidate.ts'
import type { ProjectMemoryConfidence, ProjectMemoryEntry } from '../../electron/agent/types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('memory-consolidate tests:')

const DAY = 86_400_000

function entry(over: Partial<ProjectMemoryEntry> & { id: string; text: string }): ProjectMemoryEntry {
  return {
    projectId: 'p1',
    kind: 'convention',
    sourceType: 'manual',
    sourceRef: null,
    confidence: 'medium' as ProjectMemoryConfidence,
    privacyScope: 'project',
    owner: null,
    lastVerifiedAt: null,
    createdAt: 1_000,
    updatedAt: 1_000,
    ...over,
  }
}

// ── similarity sanity ──
{
  assert.ok(memorySimilarity('we deploy with wrangler', 'deploy using wrangler') > 0.3, 'overlap scores > 0')
  assert.equal(memorySimilarity('the and of', 'the and of'), 0, 'stopword-only texts score 0')
  assert.equal(memorySimilarity('alpha beta', 'gamma delta'), 0, 'disjoint texts score 0')
  ok('memorySimilarity: token Jaccard behaves')
}

// ── subject overlap tolerates an added clause that Jaccard would miss ──
{
  const a = 'We deploy to Cloudflare Pages with wrangler.'
  const b = 'We no longer deploy to Cloudflare Pages; we use Vercel instead.'
  assert.ok(memorySimilarity(a, b) < 0.5, 'Jaccard is diluted by the added Vercel clause')
  assert.ok(memorySubjectOverlap(a, b) >= 0.5, 'subject overlap still catches the shared subject')
  assert.equal(memorySubjectOverlap('standup at 9am', 'deploy with wrangler'), 0, 'disjoint subjects → 0')
  ok('memorySubjectOverlap: overlap coefficient catches restated-plus-extended subjects')
}

// ── 1) DEDUPE near-identical, keeping the most trustworthy survivor ──
{
  const entries = [
    entry({ id: 'a', text: 'We deploy the app to Cloudflare Pages using wrangler.', confidence: 'low' }),
    entry({ id: 'b', text: 'We deploy the app to Cloudflare Pages with wrangler.', confidence: 'high' }),
    entry({ id: 'c', text: 'The team standup is at 9am on weekdays.' }),
  ]
  const plan = consolidateMemory(entries)
  assert.deepEqual(plan.removeIds, ['a'], 'the low-confidence duplicate is removed')
  assert.deepEqual(plan.survivors.map((e) => e.id).sort(), ['b', 'c'], 'survivors are the high-conf copy + unrelated entry')
  const op = plan.ops.find((o) => o.kind === 'dedupe')
  assert.ok(op && op.entryId === 'a' && op.survivorId === 'b', 'dedupe op points loser→survivor')
  ok('dedupe: collapses near-identical, keeps highest confidence')
}

// ── dedupe does NOT cross kind boundaries ──
{
  const entries = [
    entry({ id: 'a', text: 'We deploy to Cloudflare Pages with wrangler.', kind: 'convention' }),
    entry({ id: 'b', text: 'We deploy to Cloudflare Pages with wrangler.', kind: 'architecture' }),
  ]
  const plan = consolidateMemory(entries)
  assert.equal(plan.removeIds.length, 0, 'same text but different kind is left alone')
  ok('dedupe: scoped to same kind')
}

// ── 2) SUPERSEDE contradiction — newer wins ──
{
  const entries = [
    entry({ id: 'old', text: 'We deploy to Cloudflare Pages with wrangler.', updatedAt: 1_000, lastVerifiedAt: 1_000 }),
    entry({ id: 'new', text: 'We no longer deploy to Cloudflare Pages; we use Vercel instead.', updatedAt: 5_000, lastVerifiedAt: 5_000 }),
  ]
  const plan = consolidateMemory(entries)
  assert.deepEqual(plan.removeIds, ['old'], 'stale contradicted claim retired')
  assert.deepEqual(plan.survivors.map((e) => e.id), ['new'], 'newer claim survives')
  const op = plan.ops.find((o) => o.kind === 'supersede')
  assert.ok(op && op.reason === 'contradicted_by_newer' && op.survivorId === 'new', 'supersede op reason=newer')
  ok('supersede: newer contradicting entry wins')
}

// ── supersede by higher confidence when timestamps tie ──
{
  const entries = [
    entry({ id: 'weak', text: 'We deploy to Cloudflare with wrangler.', confidence: 'low', updatedAt: 2_000, lastVerifiedAt: 2_000 }),
    entry({ id: 'strong', text: 'We do not deploy to Cloudflare with wrangler anymore.', confidence: 'high', updatedAt: 2_000, lastVerifiedAt: 2_000 }),
  ]
  const plan = consolidateMemory(entries)
  assert.deepEqual(plan.removeIds, ['weak'], 'lower-confidence contradicted claim retired on timestamp tie')
  const op = plan.ops.find((o) => o.kind === 'supersede')
  assert.ok(op && op.reason === 'contradicted_by_higher_confidence', 'reason=higher_confidence')
  ok('supersede: higher confidence breaks a timestamp tie')
}

// ── 3) INVALIDATE: source vanished ──
{
  const entries = [
    entry({ id: 'x', text: 'Zone id lives in the CF_ZONE env var.', sourceType: 'file', sourceRef: 'infra/env.md', lastVerifiedAt: 3_000 }),
    entry({ id: 'y', text: 'Something unrelated about the API.', kind: 'api' }),
  ]
  const sources: Record<string, MemorySourceState> = { 'infra/env.md': { exists: false } }
  const plan = consolidateMemory(entries, { sources })
  assert.deepEqual(plan.invalidateIds, ['x'], 'orphaned-source entry flagged')
  assert.equal(plan.removeIds.length, 0, 'invalidation flags but does not remove')
  assert.ok(plan.survivors.some((e) => e.id === 'x'), 'flagged entry stays in survivors')
  const op = plan.ops.find((o) => o.kind === 'invalidate')
  assert.ok(op && op.reason === 'source_missing', 'invalidate reason=source_missing')
  ok('invalidate: flags orphaned-source entries, keeps them for audit')
}

// ── 3b) INVALIDATE: stale past TTL ──
{
  const now = 100 * DAY
  const entries = [
    entry({ id: 'fresh', text: 'Recently verified fact about builds.', lastVerifiedAt: now - 2 * DAY }),
    entry({ id: 'stale', text: 'A fact verified long ago about releases.', kind: 'decision', lastVerifiedAt: now - 40 * DAY }),
  ]
  const plan = consolidateMemory(entries, { now, staleAfterMs: 30 * DAY })
  assert.deepEqual(plan.invalidateIds, ['stale'], 'only the past-TTL entry is flagged stale')
  const op = plan.ops.find((o) => o.reason === 'stale_ttl')
  assert.ok(op && op.entryId === 'stale', 'stale op targets the old entry')
  ok('invalidate: TTL freshness window')
}

// ── injected similarity is honored (dependency injection) ──
{
  const entries = [entry({ id: 'a', text: 'alpha' }), entry({ id: 'b', text: 'beta' })]
  const plan = consolidateMemory(entries, { similarity: () => 1 }) // force everything "identical"
  // Equal confidence/timestamps → deterministic id tiebreak keeps 'a', removes 'b'.
  assert.deepEqual(plan.removeIds, ['b'], 'injected similarity drives dedupe')
  ok('consolidateMemory: similarity function is injectable')
}

// ── determinism: same input → same plan regardless of run ──
{
  const build = () => [
    entry({ id: 'a', text: 'We deploy with wrangler to pages.', confidence: 'medium', updatedAt: 2_000 }),
    entry({ id: 'b', text: 'We deploy with wrangler to pages now.', confidence: 'medium', updatedAt: 2_000 }),
  ]
  const p1 = consolidateMemory(build())
  const p2 = consolidateMemory(build())
  assert.deepEqual(p1.removeIds, p2.removeIds, 'stable tiebreak makes plans deterministic')
  ok('consolidateMemory: deterministic on ties')
}

// ── empty input is a no-op ──
{
  const plan = consolidateMemory([])
  assert.deepEqual(plan, { survivors: [], removeIds: [], invalidateIds: [], ops: [] }, 'empty in, empty plan out')
  ok('consolidateMemory: empty input is a no-op')
}

// ── applyInvalidation is pure + clears verification ──
{
  const e = entry({ id: 'a', text: 'x', lastVerifiedAt: 5_000, updatedAt: 5_000 })
  const out = applyInvalidation(e, 9_000)
  assert.equal(out.lastVerifiedAt, null, 'verification cleared')
  assert.equal(out.updatedAt, 9_000, 'updatedAt bumped to now')
  assert.equal(e.lastVerifiedAt, 5_000, 'original entry untouched (pure)')
  ok('applyInvalidation: immutable, clears lastVerifiedAt')
}

console.log(`\n${passed} passed`)
