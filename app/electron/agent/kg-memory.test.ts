// Run: jiti electron/agent/kg-memory.test.ts
import assert from 'node:assert/strict'
import {
  OPEN,
  createGraph,
  upsertEntity,
  insertFact,
  retractFact,
  queryFacts,
  resolve,
  connectedRecall,
  slotHistory,
} from './kg-memory.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('kg-memory tests:')

// Time constants (ms-ish, but any monotonic ints work).
const T1 = 1000
const T2 = 2000
const T3 = 3000

{
  // Entities: upsert merges attrs, requires type+name.
  const g = createGraph()
  const alex = upsertEntity(g, { id: 'alex', type: 'person', name: 'Alex', attrs: { role: 'owner' } })
  upsertEntity(g, { id: 'alex', type: 'person', name: 'Alex', attrs: { tz: 'GMT' } })
  const merged = g.entities.get('alex')!
  assert.equal(alex.id, 'alex')
  assert.deepEqual(merged.attrs, { role: 'owner', tz: 'GMT' }, 'attrs merge on upsert')
  assert.throws(() => upsertEntity(g, { type: '', name: 'x' }), /type/)
  ok('upsertEntity: merges attrs, validates type/name')
}

{
  // insertFact supersede: contradicting attribute closes the old fact in valid-time.
  const g = createGraph()
  const f1 = insertFact(g, { subject: 'alex', predicate: 'editor', value: 'vim', validAt: T1 }, T1)
  const f2 = insertFact(g, { subject: 'alex', predicate: 'editor', value: 'zed', validAt: T2 }, T2)
  assert.equal(g.facts.length, 2, 'both versions retained (append-only)')
  assert.equal(f1.invalidAt, T2, 'old fact bounded at new fact validAt')
  assert.equal(f2.invalidAt, OPEN, 'new fact still open')
  ok('insertFact: contradiction supersedes old fact in valid-time (history kept)')
}

{
  // Idempotent re-assertion: same value = no duplicate.
  const g = createGraph()
  const a = insertFact(g, { subject: 'alex', predicate: 'editor', value: 'zed' }, T1)
  const b = insertFact(g, { subject: 'alex', predicate: 'editor', value: 'zed' }, T2)
  assert.equal(a.id, b.id, 'same fact returned')
  assert.equal(g.facts.length, 1, 'no duplicate row')
  ok('insertFact: equivalent re-assertion is idempotent')
}

{
  // Point-in-time valid-axis: query "as of" different instants sees different truths.
  const g = createGraph()
  insertFact(g, { subject: 'alex', predicate: 'editor', value: 'vim', validAt: T1 }, T1)
  insertFact(g, { subject: 'alex', predicate: 'editor', value: 'zed', validAt: T2 }, T2)

  const atT1 = queryFacts(g, { subject: 'alex', predicate: 'editor', validAt: T1 })
  const atT2 = queryFacts(g, { subject: 'alex', predicate: 'editor', validAt: T2 })
  const now = queryFacts(g, { subject: 'alex', predicate: 'editor' }) // default: currently true
  assert.equal(atT1.length, 1)
  assert.equal(atT1[0].value, 'vim', 'as-of T1 → vim')
  assert.equal(atT2[0].value, 'zed', 'as-of T2 → zed')
  assert.equal(now[0].value, 'zed', 'current → zed')

  // resolve() gives the single winning fact.
  assert.equal(resolve(g, { subject: 'alex', predicate: 'editor' }, { validAt: T1 })!.value, 'vim')
  assert.equal(resolve(g, { subject: 'alex', predicate: 'editor' })!.value, 'zed')
  ok('queryFacts/resolve: bi-temporal point-in-time on valid axis')
}

{
  // System-axis: retract a fact; "as we believed at T2" still sees it, "now" does not.
  const g = createGraph()
  const f = insertFact(g, { subject: 'alex', predicate: 'likes', value: 'tabs', validAt: T1 }, T2)
  const retracted = retractFact(g, f.id, T3)
  assert.equal(retracted, true)
  const believedAtT2 = queryFacts(g, { subject: 'alex', validAt: T1, systemAt: T2 })
  const believedNow = queryFacts(g, { subject: 'alex', validAt: T1 })
  assert.equal(believedAtT2.length, 1, 'we believed it at T2')
  assert.equal(believedNow.length, 0, 'no longer believed')
  assert.equal(retractFact(g, f.id, T3), false, 'double-retract is a no-op')
  ok('retractFact: system-axis retraction preserves past beliefs')
}

{
  // Relational supersede is per-object: two different objects are independent edges.
  const g = createGraph()
  insertFact(g, { subject: 'alex', predicate: 'member_of', object: 'team_a', validAt: T1 }, T1)
  insertFact(g, { subject: 'alex', predicate: 'member_of', object: 'team_b', validAt: T1 }, T1)
  const live = queryFacts(g, { subject: 'alex', predicate: 'member_of' })
  assert.equal(live.length, 2, 'different objects coexist')
  // Re-asserting same object supersedes only that edge.
  insertFact(g, { subject: 'alex', predicate: 'member_of', object: 'team_a', invalidAt: T2, validAt: T1 }, T2)
  ok('insertFact: relational facts collide per-object, not per-predicate')
}

{
  // Connected recall: BFS honoring valid-time. Edge that expired before query is not traversed.
  const g = createGraph()
  // alex -knows-> bo (T1..open); bo -knows-> cy (T1..open); alex -knows-> dee (T1..T2, expired).
  insertFact(g, { subject: 'alex', predicate: 'knows', object: 'bo', validAt: T1 }, T1)
  insertFact(g, { subject: 'bo', predicate: 'knows', object: 'cy', validAt: T1 }, T1)
  insertFact(g, { subject: 'alex', predicate: 'knows', object: 'dee', validAt: T1, invalidAt: T2 }, T1)

  const nowR = connectedRecall(g, { from: 'alex', depth: 2 }) // current valid-time
  assert.equal(nowR.entities.get('alex'), 0)
  assert.equal(nowR.entities.get('bo'), 1)
  assert.equal(nowR.entities.get('cy'), 2, 'reaches cy at depth 2')
  assert.equal(nowR.entities.has('dee'), false, 'expired edge to dee not traversed now')

  const pastR = connectedRecall(g, { from: 'alex', depth: 1, validAt: T1 })
  assert.equal(pastR.entities.get('dee'), 1, 'as-of T1, dee edge was live')

  // depth cap: depth 1 does not reach cy.
  const shallow = connectedRecall(g, { from: 'alex', depth: 1 })
  assert.equal(shallow.entities.has('cy'), false, 'depth cap respected')
  ok('connectedRecall: bi-temporal BFS respects depth + valid-time')
}

{
  // Undirected traversal: recall reaches an entity via an incoming edge too.
  const g = createGraph()
  insertFact(g, { subject: 'bo', predicate: 'reports_to', object: 'alex', validAt: T1 }, T1)
  const r = connectedRecall(g, { from: 'alex', depth: 1 })
  assert.equal(r.entities.get('bo'), 1, 'reached bo via incoming edge')
  // predicate filter excludes it.
  const filtered = connectedRecall(g, { from: 'alex', depth: 1, predicates: ['knows'] })
  assert.equal(filtered.entities.has('bo'), false, 'predicate filter drops non-matching edges')
  ok('connectedRecall: undirected reach + predicate filter')
}

{
  // slotHistory: newest-first timeline of a slot.
  const g = createGraph()
  insertFact(g, { subject: 'alex', predicate: 'editor', value: 'vim', validAt: T1 }, T1)
  insertFact(g, { subject: 'alex', predicate: 'editor', value: 'zed', validAt: T2 }, T2)
  const hist = slotHistory(g, { subject: 'alex', predicate: 'editor' })
  assert.equal(hist.length, 2)
  assert.equal(hist[0].value, 'zed', 'newest first')
  assert.equal(hist[1].value, 'vim')
  ok('slotHistory: valid-time timeline newest-first')
}

{
  // Validation guards.
  const g = createGraph()
  assert.throws(() => insertFact(g, { subject: '', predicate: 'p', value: 1 }, T1), /subject/)
  assert.throws(() => insertFact(g, { subject: 's', predicate: '', value: 1 }, T1), /predicate/)
  assert.throws(() => insertFact(g, { subject: 's', predicate: 'p' }, T1), /object.*value|value/)
  assert.throws(() => insertFact(g, { subject: 's', predicate: 'p', value: 1, validAt: T2, invalidAt: T1 }, T2), /invalidAt/)
  ok('insertFact: input validation guards')
}

console.log(`\n${passed} kg-memory test groups passed.`)
