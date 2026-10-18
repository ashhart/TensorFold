import assert from 'node:assert/strict'
import type { EngineLearningFact } from './types'
import { buildLearningLayout, filterLearningFacts, summarizeLearning } from './learning-graph'
const fact = (id: number, state: EngineLearningFact['state'], links: number[] = [], recalled?: boolean): EngineLearningFact => ({ id, state, links, recalled, text: `Cedar fact ${id}`, source: 'manual', at: id })
const facts = [fact(1, 'learned', [2, 2, 1, 999], true), fact(2, 'missed', [1], false), fact(3, 'queued'), fact(4, 'learning'), fact(5, 'replaced')]
const layout = buildLearningLayout(facts)
assert.equal(layout.edges.length, 1, 'deduplicate reciprocal edges and ignore missing/self links')
assert.equal(layout.nodes.length, 5)
assert.equal(new Set(layout.nodes.map((n) => `${n.x},${n.y}`)).size, 5)
assert.deepEqual(summarizeLearning(facts), { total: 5, learned: 1, recalled: 1, missed: 1, pending: 2 }, 'missed/replaced are never counted as learned')
assert.equal(summarizeLearning([fact(6, 'learned')]).recalled, 0, 'missing recall evidence is not a passed check')
assert.deepEqual(filterLearningFacts(facts, 'CEDAR missed').map((f) => f.id), [2])
assert.deepEqual(filterLearningFacts(facts, 'manual').map((f) => f.id), [5, 4, 3, 2, 1])
const bounded = buildLearningLayout(Array.from({ length: 100 }, (_, i) => fact(i + 1, 'learned')), 60)
assert.equal(bounded.nodes.length, 60)
assert.equal(bounded.hidden, 40)
assert.equal(bounded.nodes[0].fact.id, 41, 'large graphs show most recent facts while preserving full list externally')
assert.equal(buildLearningLayout(facts, 0).nodes.length, 0)
const withDeleted = [...facts, fact(6, 'deleting', [], true), { ...fact(7, 'deleted', [], true), deleted: 1234 }]
assert.deepEqual(summarizeLearning(withDeleted), { total: 7, learned: 1, recalled: 1, missed: 1, pending: 3 }, 'deleting remains in progress and stale recall results never count deletion states as learned or recalled')
assert.deepEqual(filterLearningFacts(withDeleted, 'deleted').map((entry) => entry.id), [7])
assert.deepEqual(filterLearningFacts(withDeleted, 'deleting').map((entry) => entry.id), [6])
const deletionLayout = buildLearningLayout(withDeleted)
assert.equal(new Set(deletionLayout.nodes.map((node) => node.x)).size, 7, 'both native deletion states have distinct columns')
assert.ok(deletionLayout.nodes.every((node) => node.x + 22 < deletionLayout.width), 'every column remains inside the SVG viewBox')
console.log('Living Weights graph evidence, links, search and bounds passed.')
