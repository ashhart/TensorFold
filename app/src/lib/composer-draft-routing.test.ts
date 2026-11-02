import assert from 'node:assert/strict'
import { createComposerDraftRouting } from './composer-draft-routing'
import { emptyComposerDraft } from '../../shared/composer-drafts'

const calls: unknown[][] = [], draft = emptyComposerDraft()
const port = createComposerDraftRouting({
  getComposerDraft: async input => { calls.push(['ordinary-read', input]); return draft },
  saveComposerDraft: async input => { calls.push(['ordinary-save', input]); return { saved: true, draft } },
  clearComposerDraft: async input => { calls.push(['ordinary-clear', input]); return { saved: true, draft } },
  gmailInboxDraft: async key => { calls.push(['gmail-read', key]); return draft },
  gmailInboxSaveDraft: async input => { calls.push(['gmail-save', input]); return { saved: true, draft } },
})
const scope = { surface: 'assistant' as const, conversationId: 'gmail-draft:account-a:dd11', hostId: null, incognito: false }
await port.getComposerDraft(scope)
assert.deepEqual(calls.at(-1), ['gmail-read', 'gmail:account-a:1:dd11'])
await port.saveComposerDraft({ scope, expectedRevision: 3, text: 'A reply', attachments: [] })
assert.deepEqual(calls.at(-1), ['gmail-save', { key: 'gmail:account-a:1:dd11', expectedRevision: 3, text: 'A reply' }])
await port.clearComposerDraft({ scope, expectedRevision: 4 })
assert.deepEqual(calls.at(-1), ['gmail-save', { key: 'gmail:account-a:1:dd11', expectedRevision: 4, text: '' }])
await port.getComposerDraft({ ...scope, surface: 'chat', conversationId: 'ordinary' })
assert.equal(calls.at(-1)?.[0], 'ordinary-read')
assert.throws(() => port.getComposerDraft({ ...scope, hostId: 'remote' }), /local Gmail draft/)
assert.throws(() => port.getComposerDraft({ ...scope, incognito: true }), /local Gmail draft/)
const before = calls.length
await assert.rejects(port.saveComposerDraft({ scope, expectedRevision: 4, text: 'A reply', attachments: [{ id: 'file', name: 'file.pdf', kind: 'document', mime: 'application/pdf', status: 'needs-reselection' }] }), /text only/)
assert.equal(calls.length, before)
console.log('The single desktop draft port routes Gmail to protected owner APIs while retaining global flush ownership and ordinary drafts.')
