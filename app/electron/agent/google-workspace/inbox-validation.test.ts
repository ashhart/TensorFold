import assert from 'node:assert/strict'
import { validInboxCache } from './inbox-validation'
import { GoogleState } from './private-state'

const accountIds = new Set(['account-a'])
const message = { id: 'aa11', threadId: 'dd11', subject: 'Fixture', from: 'sender@example.com', to: 'reader@example.com', date: '', snippet: 'Synthetic preview', untrustedContent: true as const, attachments: [] }
const row = { accountId: 'account-a', revision: 1, enabled: true, scopeId: 'synthetic-scope', syncedAt: 1, messages: [message] }
assert.ok(validInboxCache([row], accountIds))
for (const value of [[row, row], [{ ...row, revision: 0 }], [{ ...row, accountId: 'unknown' }], [{ ...row, messages: [message, message] }], [{ ...row, messages: [{ ...message, text: 'Full bodies must not enter the metadata cache' }] }], [{ ...row, messages: [{ ...message, snippet: 'x'.repeat(2049) }] }], [{ ...row, messages: [null] }]]) assert.equal(validInboxCache(value as any, accountIds), false)
const draft = { accountId: 'account-a', threadId: 'dd11', draft: { text: 'Synthetic reply', revision: 1, updatedAt: 1, attachments: [] } }
const state: any = { version: 1, accounts: [{ id: 'account-a', email: 'reader@example.com', subject: 'account-a', revision: 1, status: 'disconnected', capabilities: [] }], drafts: [], actions: [], inboxDrafts: [draft, draft] }
const store = new GoogleState({ read: async () => state, write: async () => { throw new Error('No writes expected') } })
await assert.rejects(store.read(() => true), /storage is invalid/)
state.inboxDrafts = [{ ...draft, draft: { ...draft.draft, text: 'x'.repeat(128001) } }]
await assert.rejects(store.read(() => true), /storage is invalid/)
console.log('Protected Gmail cache rejects oversized or duplicate metadata, full bodies, unrelated accounts and malformed saved drafts.')
