// Run: jiti electron/agent/incognito.test.ts
//
// Incognito threads: memory-only conversations. The persistence seam strips them
// from threads.json; the spine seam drops their events before the journal.
import assert from 'node:assert/strict'
import { stripIncognitoThreads, spineEligible } from './incognito'

// stripIncognitoThreads drops only incognito threads and keeps order
{
  const threads = [
    { id: 'a' },
    { id: 'b', incognito: true },
    { id: 'c', incognito: false },
    { id: 'd', incognito: true },
    { id: 'e' },
  ]
  assert.deepEqual(stripIncognitoThreads(threads).map((t) => t.id), ['a', 'c', 'e'])
}

// empty list passes through untouched
assert.deepEqual(stripIncognitoThreads([]), [])

// spineEligible: incognito threads never reach the journal
assert.equal(spineEligible({ incognito: true }), false)
assert.equal(spineEligible({ incognito: false }), true)
assert.equal(spineEligible({}), true)

// An event for a thread the host can't resolve (e.g. already deleted) must not be
// silently dropped — only a positive incognito flag suppresses journaling.
assert.equal(spineEligible(undefined), true)
assert.equal(spineEligible(null), true)

console.log('incognito tests passed')
