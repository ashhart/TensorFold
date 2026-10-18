import assert from 'node:assert/strict'
import { ConversationDrafts } from './conversation-drafts'
import { composerDraftKey, emptyComposerDraft, validateComposerDraftWrite } from '../../shared/composer-drafts'
import type { ComposerDraftPort, ComposerDraftSnapshot } from '../../shared/composer-drafts'
const a = { surface: 'chat' as const, conversationId: 'quote-a', hostId: null, incognito: false }, b = { ...a, conversationId: 'quote-b' }
const quote = { id: 'q', sourceItemId: 'answer', sourceRevision: 'a'.repeat(64), sourceText: 'Synthetic source', text: 'User edited text' }
const saved = new Map<string, ComposerDraftSnapshot>()
const port: ComposerDraftPort = {
  async getComposerDraft(scope) { return structuredClone(saved.get(composerDraftKey(scope)) ?? emptyComposerDraft()) },
  async saveComposerDraft(input) { const valid = validateComposerDraftWrite(input), old = saved.get(composerDraftKey(input.scope)) ?? emptyComposerDraft(); if (old.revision !== input.expectedRevision) return { saved: false, draft: structuredClone(old) }; const draft = { revision: old.revision + 1, text: valid.text, attachments: valid.attachments, quotes: valid.quotes, updatedAt: 1 }; saved.set(composerDraftKey(input.scope), structuredClone(draft)); return { saved: true, draft } },
  async clearComposerDraft(input) { return this.saveComposerDraft({ ...input, text: '', attachments: [] }) },
}
const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve() }
let drafts = new ConversationDrafts(port, 60_000)
drafts.get(a); drafts.get(b); await settle()
drafts.update(a, { text: 'A draft', quotes: [quote] }); drafts.update(b, { text: 'B draft' }); await drafts.flushAll(); drafts.dispose()
drafts = new ConversationDrafts(port, 60_000); drafts.get(a); drafts.get(b); await settle()
assert.deepEqual(drafts.get(a).quotes, [quote]); assert.deepEqual(drafts.get(b).quotes, []); assert.equal(drafts.get(b).text, 'B draft')
const token = drafts.capture(a); drafts.update(a, { quotes: [{ ...quote, text: 'Newer edited quotation' }] }); await drafts.clearAccepted(token)
assert.equal(drafts.get(a).quotes![0].text, 'Newer edited quotation', 'acceptance of an earlier send cannot clear a newer quote edit')
await drafts.clearAccepted(drafts.capture(a)); assert.deepEqual(drafts.get(a).quotes, [])
for (const scope of [{ ...a, surface: 'home' as const, conversationId: null }, { ...a, surface: 'assistant' as const }, { ...a, hostId: 'remote' }]) assert.throws(() => validateComposerDraftWrite({ scope, expectedRevision: 0, text: '', attachments: [], quotes: [quote] }), /local conversation/)
drafts.update(a, { quotes: [quote] }); drafts.makeIncognito({ ...a, incognito: true }); assert.deepEqual(drafts.get({ ...a, incognito: true }).quotes, [quote]); assert.deepEqual(drafts.get(a).quotes, [])
drafts.dispose()
console.log('Quotation drafts survive restart per conversation, preserve newer edits and stay out of other hosts and assistant scopes')
