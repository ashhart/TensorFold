// Run: jiti electron/agent/thread-annotations.test.ts
//
// Thread annotations (VST-040): pinned messages (labels + done), markers, notes — pure
// reducers with persistence + prune-on-timeline-truncation.
import assert from 'node:assert/strict'
import {
  emptyAnnotations,
  pinMessage,
  unpinMessage,
  setPinnedLabel,
  setPinnedDone,
  addMarker,
  setMarkerDone,
  removeMarker,
  setNotes,
  pruneAnnotations,
  serializeAnnotations,
  restoreAnnotations,
  type AnnotationDeps,
} from './thread-annotations'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('thread-annotations tests:')

function harness() {
  let clock = 1_000
  let n = 0
  const deps: AnnotationDeps = { now: () => (clock += 10), newId: () => `an_${++n}` }
  return { deps }
}

// --- pinned messages: add / label / done / dedupe / unpin ---
{
  const { deps } = harness()
  let s = emptyAnnotations()
  s = pinMessage(s, 'item-1', 'Key decision', deps)
  assert.equal(s.pinned.length, 1)
  assert.equal(s.pinned[0].itemId, 'item-1')
  assert.equal(s.pinned[0].label, 'Key decision')
  assert.equal(s.pinned[0].done, false)

  // Pinning the same item again updates the label, not a duplicate.
  s = pinMessage(s, 'item-1', 'Revised', deps)
  assert.equal(s.pinned.length, 1, 'a message pins once')
  assert.equal(s.pinned[0].label, 'Revised')

  const pinId = s.pinned[0].id
  s = setPinnedDone(s, pinId, true)
  assert.equal(s.pinned[0].done, true, 'pinned messages carry a done state')
  s = setPinnedLabel(s, pinId, null)
  assert.equal(s.pinned[0].label, null)
  s = unpinMessage(s, pinId)
  assert.equal(s.pinned.length, 0)
  ok('pinned: add, dedupe-by-item, label, done, unpin')
}

// --- markers ---
{
  const { deps } = harness()
  let s = emptyAnnotations()
  s = addMarker(s, 'item-9', 'revisit this', deps)
  s = addMarker(s, 'item-9', '', deps) // blank label → default
  assert.equal(s.markers.length, 2, 'markers are not deduped (multiple per item allowed)')
  assert.equal(s.markers[1].label, 'Marker')
  const id = s.markers[0].id
  s = setMarkerDone(s, id, true)
  assert.equal(s.markers[0].done, true)
  s = removeMarker(s, id)
  assert.equal(s.markers.length, 1)
  ok('markers: add (default label), done, remove')
}

// --- notes ---
{
  let s = setNotes(emptyAnnotations(), 'remember the auth edge case')
  assert.equal(s.notes, 'remember the auth edge case')
  s = setNotes(s, 123 as any)
  assert.equal(s.notes, '', 'non-string notes coerce to empty')
  ok('notes: set + coerce')
}

// --- prune on timeline truncation (e.g. a turn revert dropped later items) ---
{
  const { deps } = harness()
  let s = emptyAnnotations()
  s = pinMessage(s, 'keep', 'a', deps)
  s = pinMessage(s, 'gone', 'b', deps)
  s = addMarker(s, 'keep', 'm1', deps)
  s = addMarker(s, 'gone', 'm2', deps)
  const pruned = pruneAnnotations(s, ['keep'])
  assert.equal(pruned.pinned.length, 1, 'pins over deleted items are pruned')
  assert.equal(pruned.pinned[0].itemId, 'keep')
  assert.equal(pruned.markers.length, 1, 'markers over deleted items are pruned')
  assert.equal(pruned.markers[0].itemId, 'keep')
  ok('prune: drops pins/markers whose item was removed (revert-safe)')
}

// --- persistence round-trip + defensive restore ---
{
  const { deps } = harness()
  let s = emptyAnnotations()
  s = pinMessage(s, 'item-1', 'pinned', deps)
  s = setPinnedDone(s, s.pinned[0].id, true)
  s = addMarker(s, 'item-2', 'marker', deps)
  s = setNotes(s, 'note text')
  const raw = serializeAnnotations(s)

  const restored = restoreAnnotations(raw)
  assert.deepEqual(restored, s, 'serialize/restore round-trips exactly')

  // Defensive: corrupt / absent / partial blobs never throw.
  assert.deepEqual(restoreAnnotations(null), emptyAnnotations())
  assert.deepEqual(restoreAnnotations('not json'), emptyAnnotations())
  const partial = restoreAnnotations(JSON.stringify({ pinned: [{ id: 'x' /* missing itemId */ }], notes: 5 }))
  assert.equal(partial.pinned.length, 0, 'malformed pin entries are dropped')
  assert.equal(partial.notes, '', 'non-string notes normalized')
  ok('persistence: exact round-trip; corrupt/partial → safe defaults')
}

console.log(`\n${passed} passed`)
