import assert from 'node:assert/strict'
import { mergeThreadRecord } from './merge'
import type { Thread } from '../../types'
const saved: Thread = { id: 'fixture', providerId: null, model: null, title: 'Synthetic', kind: 'chat', cwd: '', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [], memoryPolicyRevision: 4, memoryPolicy: { useExisting: false, contribute: false } }
const stale: Thread = { ...saved, updatedAt: 999, memoryPolicyRevision: 3, memoryPolicy: { useExisting: true, contribute: true }, timeline: [{ id: 'streaming', kind: 'assistant', text: 'New visible text' }] }
for (const result of [mergeThreadRecord(saved, stale), mergeThreadRecord(stale, saved)]) {
  assert.deepEqual(result.memoryPolicy, saved.memoryPolicy, 'later unrelated snapshot timestamps cannot roll back durably accepted memory flags')
  assert.equal(result.memoryPolicyRevision, 4)
  assert.equal(result.timeline[0].text, 'New visible text', 'memory saves preserve concurrently streamed history')
}
console.log('Renderer merges preserve monotonic host memory choices and concurrent visible history')

const damaged = { ...saved, updatedAt: 1000, memoryPolicyRevision: 'damaged' } as unknown as Thread
assert.equal(mergeThreadRecord(saved, damaged).memoryPolicyRevision, 'damaged', 'damaged metadata remains visible for review; a display merge cannot claim it repaired')

for (const invalid of [null, false, 0, 'damaged']) {
  const damagedPolicy = { ...saved, updatedAt: 1001, memoryPolicyRevision: 5, memoryPolicy: invalid } as unknown as Thread
  assert.equal(mergeThreadRecord(saved, damagedPolicy).memoryPolicy, invalid, 'a malformed present policy must never turn into absent legacy defaults')
}
