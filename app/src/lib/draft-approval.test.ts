import assert from 'node:assert/strict'
import { discardPendingDraft } from './draft-approval'

let refreshed = 0
await discardPendingDraft(async (id, decision) => {
  assert.equal(id, 'draft-1')
  assert.equal(decision, 'deny')
}, async () => { refreshed++ }, 'draft-1')
assert.equal(refreshed, 1, 'successful denial refreshes once')

refreshed = 0
await assert.rejects(
  discardPendingDraft(async () => { throw new Error('deny failed') }, async () => { refreshed++ }, 'draft-2'),
  /deny failed/,
)
assert.equal(refreshed, 0, 'failed denial must not refresh away the still-pending draft')

console.log('draft approval tests passed')
