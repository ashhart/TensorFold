import { gmailDraftKey } from '../../shared/gmail-inbox'
import type { GmailInboxApi } from '../../shared/gmail-inbox'
import type { ComposerDraftPort } from '../../shared/composer-drafts'

export function createComposerDraftRouting(port: ComposerDraftPort & Pick<GmailInboxApi, 'gmailInboxDraft' | 'gmailInboxSaveDraft'>): ComposerDraftPort {
  return {
    getComposerDraft: scope => { const key = gmailDraftKey(scope); return key ? port.gmailInboxDraft(key) : port.getComposerDraft(scope) },
    saveComposerDraft: input => {
      const key = gmailDraftKey(input.scope)
      if (!key) return port.saveComposerDraft(input)
      if (input.attachments.length || input.quotes?.length) return Promise.reject(new Error('Gmail Inbox drafts currently support text only.'))
      return port.gmailInboxSaveDraft({ key, expectedRevision: input.expectedRevision, text: input.text })
    },
    clearComposerDraft: input => {
      const key = gmailDraftKey(input.scope)
      return key ? port.gmailInboxSaveDraft({ key, expectedRevision: input.expectedRevision, text: '' }) : port.clearComposerDraft(input)
    },
  }
}
