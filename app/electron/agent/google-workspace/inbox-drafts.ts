import { emptyComposerDraft } from '../../../shared/composer-drafts'
import type { ComposerDraftSnapshot, ComposerDraftWriteResult } from '../../../shared/composer-drafts'
import { parseGmailInboxKey } from '../../../shared/gmail-inbox'
import { GoogleState, type GoogleWorkspacePrivateState } from './private-state'

export class GoogleInboxDrafts {
  constructor(private readonly state: GoogleState, private readonly now: () => number) { }
  private owner(state: GoogleWorkspacePrivateState, key: string) {
    const target = parseGmailInboxKey(key)
    if (!target || !state.accounts.some(account => account.id === target.accountId && account.status !== 'disconnected')) throw new Error('The Gmail draft account is no longer available.')
    return target
  }
  read(key: string): Promise<ComposerDraftSnapshot> {
    return this.state.read(state => {
      const owner = this.owner(state, key)
      return structuredClone(state.inboxDrafts?.find(row => row.accountId === owner.accountId && row.threadId === owner.threadId)?.draft ?? emptyComposerDraft())
    })
  }
  write(input: { key: string; expectedRevision: number; text: string }): Promise<ComposerDraftWriteResult> {
    if (!input || !Number.isSafeInteger(input.expectedRevision) || input.expectedRevision < 0 || typeof input.text !== 'string' || input.text.length > 128_000 || input.text.includes('\0')) return Promise.reject(new Error('Provide a bounded Gmail draft and its saved revision.'))
    return this.state.transaction(state => {
      const owner = this.owner(state, input.key)
      const row = state.inboxDrafts?.find(item => item.accountId === owner.accountId && item.threadId === owner.threadId)
      const previous = row?.draft ?? emptyComposerDraft()
      if (previous.revision !== input.expectedRevision) return { saved: false, draft: structuredClone(previous) }
      const draft: ComposerDraftSnapshot = { revision: previous.revision + 1, text: input.text, attachments: [], updatedAt: this.now() }
      if (row) row.draft = draft
      else {
        if ((state.inboxDrafts?.length ?? 0) >= 100) throw new Error('There are too many saved Gmail drafts. Clear an unused draft before saving another.')
        state.inboxDrafts = [...(state.inboxDrafts ?? []), { accountId: owner.accountId, threadId: owner.threadId, draft }]
      }
      return { saved: true, draft: structuredClone(draft) }
    })
  }
}
