import assert from 'node:assert/strict'
import { mergeThreadRecord } from './merge'
import type { Thread } from '../../types'
const live: Thread = { id: 'fixture', title: 'Synthetic', kind: 'chat', cwd: '', providerId: null, model: null, createdAt: 1, updatedAt: 9999, status: 'idle', conversationRevision: 0, timeline: [{ id: 'u', kind: 'user', text: 'Long original question' }, { id: 'a', kind: 'assistant', text: 'Old later answer' }], apiMessages: [{ role: 'user', content: 'Long original question' }, { role: 'assistant', content: 'Old later answer' }] }
const accepted: Thread = { ...live, updatedAt: 2, conversationRevision: 1, timeline: [{ id: 'u', kind: 'user', text: 'Short edit' }, { id: 'n', kind: 'notice', text: 'Message edited' }], apiMessages: [{ role: 'user', content: 'Short edit' }], messageEditRevisions: [{ id: 'r', itemId: 'u', editedAt: 2, conversationRevision: 0, timeline: structuredClone(live.timeline), apiMessages: structuredClone(live.apiMessages) }] }
for (const result of [mergeThreadRecord(accepted, live), mergeThreadRecord(live, accepted)]) {
  assert.deepEqual(result.timeline, accepted.timeline, 'the authoritative shorter edited timeline wins in both arrival orders')
  assert.deepEqual(result.apiMessages, accepted.apiMessages, 'model context follows exactly the same edit epoch')
  assert.deepEqual(result.messageEditRevisions, accepted.messageEditRevisions, 'superseded evidence is retained even if an old renderer timestamp is later')
  assert.equal(result.conversationRevision, 1)
}
const branch = { ...accepted, id: 'new-task', kind: 'task' as const, messageEditSource: { threadId: 'source-task', itemId: 'u', editedAt: 2 } }
const staleBranch = { ...branch, conversationRevision: 0, updatedAt: 9999, messageEditSource: undefined, messageEditRevisions: undefined }
assert.deepEqual(mergeThreadRecord(branch, staleBranch).messageEditSource, branch.messageEditSource)
console.log('Renderer edit epochs keep shorter visible/API histories and superseded/branch evidence together across stale snapshots')
