import assert from 'node:assert/strict'
import { sanitizeRendererThreadPatch } from './core-ipc'

const safe = sanitizeRendererThreadPatch({
  goal: 'Ship safely',
  goalStatus: 'active',
  plan: [{ id: 'p1', text: 'Verify', done: false, changedPaths: ['forged.ts'] }],
  notepad: 'user note',
  diffReviewedAt: 123,
})
assert.deepEqual(safe, {
  goal: 'Ship safely',
  goalStatus: 'active',
  plan: [{ id: 'p1', text: 'Verify', done: false }],
  notepad: 'user note',
  diffReviewedAt: 123,
})

for (const key of ['cwd', 'workspace', 'timeline', 'apiMessages', 'status', 'id', 'kind', 'changedPaths']) {
  assert.throws(
    () => sanitizeRendererThreadPatch({ [key]: key === 'timeline' ? [{ kind: 'tool', state: 'ok', changedPaths: ['victim.ts'] }] : 'forged' }),
    /host-owned fields/,
    `${key} must not cross the generic renderer patch boundary`,
  )
}

assert.throws(() => sanitizeRendererThreadPatch({ plan: [{ id: 'p1', text: 'x', done: 'yes' }] }), /Invalid plan step/)
assert.throws(() => sanitizeRendererThreadPatch({ diffReviewedAt: Number.NaN }), /Invalid diff review timestamp/)

console.log('thread patch boundary tests passed')
