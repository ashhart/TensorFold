// Run: jiti src/lib/thread-annotations.test.ts
//
// Thread annotations, renderer side (VST-040): parse/serialize round-trip, idempotent pin,
// toggle-done, prune-on-revert, garbage safety.
import assert from 'node:assert/strict'
import { emptyAnnotations, isPinned, parseAnnotations, pinMessage, pruneAnnotations, serializeAnnotations, togglePinnedDone, unpinMessage } from './thread-annotations'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-annotations (renderer) tests:')

let seq = 0
const newId = () => `pin-${++seq}`

{
  let a = emptyAnnotations()
  a = pinMessage(a, 'item-1', 'important', 1000, newId)
  a = pinMessage(a, 'item-2', null, 1001, newId)
  assert.equal(a.pinned.length, 2)
  assert.equal(isPinned(a, 'item-1'), true)
  // Re-pinning the same item updates the label, never duplicates.
  a = pinMessage(a, 'item-1', 'renamed', 1002, newId)
  assert.equal(a.pinned.length, 2)
  assert.equal(a.pinned.find((p) => p.itemId === 'item-1')?.label, 'renamed')
  ok('pinMessage: idempotent by itemId, relabels on repin')
}

{
  let a = emptyAnnotations()
  a = pinMessage(a, 'x', null, 1, newId)
  a = togglePinnedDone(a, 'x')
  assert.equal(a.pinned[0].done, true)
  a = togglePinnedDone(a, 'x')
  assert.equal(a.pinned[0].done, false)
  a = unpinMessage(a, 'x')
  assert.equal(a.pinned.length, 0)
  ok('togglePinnedDone + unpinMessage')
}

{
  let a = emptyAnnotations()
  a = pinMessage(a, 'live', null, 1, newId)
  a = pinMessage(a, 'gone', null, 2, newId)
  const pruned = pruneAnnotations(a, ['live'])
  assert.deepEqual(pruned.pinned.map((p) => p.itemId), ['live'], 'a revert-removed item is dropped')
  // No-op when nothing changed (same reference is fine either way).
  assert.equal(pruneAnnotations(pruned, ['live']).pinned.length, 1)
  ok('pruneAnnotations: drops pins whose item a revert removed')
}

{
  const a = pinMessage(emptyAnnotations(), 'z', 'note', 5, newId)
  const round = parseAnnotations(serializeAnnotations({ ...a, notes: 'hello' }))
  assert.equal(round.pinned.length, 1)
  assert.equal(round.pinned[0].itemId, 'z')
  assert.equal(round.notes, 'hello')
  // Garbage in → empty, never throws.
  assert.deepEqual(parseAnnotations('not json'), emptyAnnotations())
  assert.deepEqual(parseAnnotations(null), emptyAnnotations())
  assert.deepEqual(parseAnnotations('{"pinned":[{"bogus":1},{"itemId":"ok"}]}').pinned.map((p) => p.itemId), ['ok'], 'malformed pins filtered')
  ok('parse/serialize: round-trip + garbage-safe')
}

console.log(`  ${passed} passed`)
