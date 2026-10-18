import assert from 'node:assert/strict'
import { editQueuedFollowUp, capturedDraftMatches } from './queued-followups'
import type { Steer } from '../components/composer/SteerQueueList'
const rows: Steer[] = [{ id: 'first', text: 'first', status: 'queued' }, { id: 'edit', text: 'original', status: 'queued' }, { id: 'last', text: 'last', status: 'queued' }]
const result = editQueuedFollowUp(rows, { id: 'edit', expectedText: 'original', text: 'edited\n  preserved indentation' })
assert.equal(result.ok, true); assert.deepEqual(result.queued.map(row => row.id), rows.map(row => row.id)); assert.equal(result.queued[1].status, 'queued')
assert.equal(rows[1].text, 'original'); assert.equal(result.queued[0], rows[0]); assert.equal(result.queued[2], rows[2])
for (const [input, reason] of [[{ id: 'gone', expectedText: 'original', text: 'edited' }, 'missing'], [{ id: 'edit', expectedText: 'stale', text: 'edited' }, 'changed'], [{ id: 'edit', expectedText: 'original', text: '   ' }, 'empty'], [{ id: 'edit', expectedText: 'original', text: 'x'.repeat(1_000_001) }, 'too-long']] as const) {
  const refused = editQueuedFollowUp(rows, input); assert.equal(refused.ok, false); if (!refused.ok) assert.equal(refused.reason, reason); assert.equal(refused.queued, rows)
}
for (const status of ['guiding', 'guided'] as const) {
  const started = [{ ...rows[1], status }]
  assert.equal(editQueuedFollowUp(started, { id: 'edit', expectedText: 'original', text: 'edited' }).ok, false)
}
const original = { key: 'chat-one', version: 1, text: 'saved original', attachments: [] }
assert.equal(capturedDraftMatches(original, { ...original }), true)
assert.equal(capturedDraftMatches(original, { ...original, version: 2, text: 'new typing' }), false)
assert.equal(capturedDraftMatches(original, { ...original, key: 'chat-two' }), false)
assert.equal(capturedDraftMatches(undefined, original), false)
console.log('Queue edit CAS preserves position/status and unchanged rows, refuses started/stale/removed/blank/oversized edits and protects a newer or different saved draft')
