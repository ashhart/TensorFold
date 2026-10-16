// Run: jiti src/lib/diff-words.test.ts
import assert from 'node:assert/strict'
import { diffWordsInLine, pairWordDiffs } from './diff-words'
import type { DiffLine } from './types'

// --- identical line -> null (no intra-line change) ---
{
  const wd = diffWordsInLine('const x = 1', 'const x = 1')
  assert.equal(wd, null, 'identical line has no word diff')
  console.log('ok: identical line -> null')
}

// --- single value change ---
{
  const wd = diffWordsInLine('const x = 1', 'const x = 2')
  assert.ok(wd, 'value change produces a diff')
  const removedChanged = wd!.removed.filter((s) => s.changed).map((s) => s.text)
  const addedChanged = wd!.added.filter((s) => s.changed).map((s) => s.text)
  assert.deepEqual(removedChanged, ['1'], 'removed changed span is "1"')
  assert.deepEqual(addedChanged, ['2'], 'added changed span is "2"')
  // Context spans present and unchanged on both sides.
  assert.ok(wd!.removed.some((s) => !s.changed && s.text === 'const x = '), 'shared prefix kept as context')
  console.log('ok: single value change')
}

// --- multi-word change collapses to changed phrase ---
{
  const wd = diffWordsInLine('return foo(bar, baz)', 'return foo(bar, qux)')
  assert.ok(wd, 'arg change produces a diff')
  const removedChanged = wd!.removed.filter((s) => s.changed).map((s) => s.text).join('')
  const addedChanged = wd!.added.filter((s) => s.changed).map((s) => s.text).join('')
  assert.equal(removedChanged, 'baz', 'removed is baz')
  assert.equal(addedChanged, 'qux', 'added is qux')
  console.log('ok: multi-word change')
}

// --- whitespace-only change is detected but not inflated ---
{
  const wd = diffWordsInLine('a  b', 'a b')
  // Either there's a change (whitespace) or jsdiff collapses it; either way
  // the function must not throw and must return a sensible shape or null.
  if (wd) {
    const hasChange = wd.removed.some((s) => s.changed) || wd.added.some((s) => s.changed)
    assert.ok(hasChange, 'if returned, must have a real changed span')
  }
  console.log('ok: whitespace-only handled cleanly')
}

// --- long line guard -> null (fall back to plain rendering) ---
{
  const longA = 'x'.repeat(5000)
  const longB = 'y'.repeat(5000)
  const wd = diffWordsInLine(longA, longB)
  assert.equal(wd, null, 'oversized line returns null')
  console.log('ok: long-line guard')
}

// --- empty lines ---
{
  const wd = diffWordsInLine('', 'new')
  assert.ok(wd, 'empty -> content is a change')
  assert.equal(wd!.added.filter((s) => s.changed).map((s) => s.text).join(''), 'new', 'added is "new"')
  console.log('ok: empty -> content')
}

// --- pairWordDiffs walks DiffLine[] and pairs adjacent del+add ---
{
  const diff: DiffLine[] = [
    { type: 'hunk', text: '@@ -1,0 +1,0 @@' },
    { type: 'ctx', text: 'const x = 1', oldNo: 1, newNo: 1 },
    { type: 'del', text: 'return x', oldNo: 2 },
    { type: 'add', text: 'return y', newNo: 2 },
    { type: 'add', text: '// done', newNo: 3 },
  ]
  const pairs = pairWordDiffs(diff)
  // The del at index 2 and add at index 3 are paired; the lone add at 4 is not.
  assert.ok(pairs.has(2), 'del line paired')
  assert.ok(pairs.has(3), 'add line paired')
  assert.ok(!pairs.has(4), 'lone add not paired')
  // The pair shares the same LineWordDiff object.
  assert.equal(pairs.get(2), pairs.get(3), 'paired lines share one diff')
  // The hunk (0) and ctx (1) are not paired.
  assert.ok(!pairs.has(0) && !pairs.has(1), 'non del/add lines not paired')
  console.log('ok: pairWordDiffs adjacency')
}

console.log('ALL TESTS PASSED')
