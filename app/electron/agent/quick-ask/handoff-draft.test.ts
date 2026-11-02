import assert from 'node:assert/strict'
import { importAskDraft } from '../../../shared/quick-ask-handoff'
import { emptyAskSession } from './validation'
import { emptyComposerDraft } from '../../../shared/composer-drafts'
import type { ComposerDraftSnapshot, ComposerDraftPort } from '../../../shared/composer-drafts'

const scope = { surface: 'chat' as const, conversationId: 'example', hostId: null, incognito: false }
let saved: ComposerDraftSnapshot = emptyComposerDraft(), writes = 0, conflict = false
const port: ComposerDraftPort = {
  getComposerDraft: async () => saved,
  saveComposerDraft: async input => { writes++; if (conflict) return { saved: false, draft: saved }; saved = { revision: saved.revision + 1, text: input.text, attachments: input.attachments, updatedAt: 1 }; return { saved: true, draft: saved } },
  clearComposerDraft: async () => ({ saved: false, draft: saved }),
}
const draft = { ...emptyAskSession().draft, text: 'Unfinished question', attachments: [{ id: 'image', kind: 'image' as const, name: 'Example', mime: 'image/png', dataUrl: 'data:image/png;base64,AAAA' }] }
assert.equal(await importAskDraft(port, scope, draft), true)
assert.equal(saved.text, draft.text)
assert.equal('dataUrl' in saved.attachments[0], false, 'Durable composer references retain no raw file payload')
assert.equal(saved.attachments[0].status, 'needs-reselection')
saved.text = 'A newer full-app draft'
assert.equal(await importAskDraft(port, scope, draft), false)
assert.equal(saved.text, 'A newer full-app draft', 'A retry cannot overwrite later work in the full app')
assert.equal(writes, 1)
saved = emptyComposerDraft(); conflict = true
await assert.rejects(importAskDraft(port, scope, draft), /changed during handoff/)
console.log('Quick Ask durable draft handoff, retry ownership and attachment-reference checks passed')
