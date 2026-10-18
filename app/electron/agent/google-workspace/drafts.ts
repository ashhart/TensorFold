import { randomUUID } from 'node:crypto'
import type { GoogleWorkspaceEmailContent, GoogleWorkspaceLocalDraft } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { validatedEmail } from './action-validation'

export class GoogleDrafts {
  constructor(private readonly accounts: GoogleAccounts, private readonly now: () => number) {}
  async save(accountId: string, email: GoogleWorkspaceEmailContent, id?: string, assertAllowed?: () => void): Promise<GoogleWorkspaceLocalDraft> {
    const content = validatedEmail(email)
    const account = await this.accounts.current(accountId)
    return this.accounts.state.transaction(state => {
      assertAllowed?.()
      const live = state.accounts.find(item => item.id === accountId)
      if (!live || live.revision !== account.revision || live.status !== 'connected') throw new Error('Google account changed. Try saving this draft again.')
      const previous = id ? state.drafts.find(draft => draft.id === id && draft.accountId === accountId) : undefined
      if (id && !previous) throw new Error('This draft does not belong to the selected Google account.')
      if (!previous && state.drafts.length >= 100) throw new Error('Remove an unused local draft before creating another.')
      const draft: GoogleWorkspaceLocalDraft = { ...content, id: previous?.id ?? randomUUID(), accountId, updatedAt: this.now(), localOnly: true }
      if (previous) Object.assign(previous, draft)
      else state.drafts.push(draft)
      return structuredClone(draft)
    }, assertAllowed)
  }
  async list(accountId: string): Promise<GoogleWorkspaceLocalDraft[]> {
    await this.accounts.current(accountId)
    return this.accounts.state.read(state => state.drafts.filter(draft => draft.accountId === accountId).sort((a, b) => b.updatedAt - a.updatedAt))
  }
  async delete(accountId: string, id: string): Promise<void> {
    await this.accounts.current(accountId)
    await this.accounts.state.transaction(state => {
      const index = state.drafts.findIndex(draft => draft.id === id && draft.accountId === accountId)
      if (index < 0) throw new Error('This draft does not belong to the selected Google account.')
      state.drafts.splice(index, 1)
    })
  }
}
