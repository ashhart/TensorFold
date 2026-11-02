import type { GmailInboxSnapshot } from '../../../shared/gmail-inbox'
import { randomUUID } from 'node:crypto'
import { parseGmailInboxKey } from '../../../shared/gmail-inbox'
import { googleRequestError } from '../../../shared/google-workspace-errors'
import type { GoogleWorkspaceThread } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { GoogleGmail } from './gmail'
import { GoogleActions } from './actions'
import { assertCapability } from './scopes'
import { gmailMessageEvent, gmailReviewEvent } from './inbox-events'

export class GoogleInbox {
  private readonly syncing = new Map<string, Promise<void>>()
  private readonly controllers = new Map<string, AbortController>()
  private readonly epochs = new Map<string, number>()
  constructor(private readonly accounts: GoogleAccounts, private readonly gmail: GoogleGmail, private readonly actions: GoogleActions, private readonly now: () => number) { }

  async snapshot(): Promise<GmailInboxSnapshot> {
    return this.accounts.state.read(state => {
      const events: GmailInboxSnapshot['events'] = []
      const accounts = state.accounts.map(account => {
        const cache = state.inbox?.find(item => item.accountId === account.id)
        const current = cache?.revision === account.revision, connected = account.status === 'connected' && account.capabilities.includes('gmail.read')
        if (cache?.enabled && current && connected) {
          events.push(...cache.messages.map(message => gmailMessageEvent(account, message)))
          for (const action of state.actions.filter(item => item.review.accountId === account.id && item.review.revision === account.revision && (item.result || item.review.expiresAt > this.now()))) {
            const event = gmailReviewEvent(account, action)
            if (event && (action.result || action.review.action.kind !== 'gmail.send' || action.review.action.email.reply?.scopeId === cache.scopeId)) events.push(event)
          }
        }
        return {
          id: account.id, email: account.email, revision: account.revision, enabled: cache?.enabled ?? false, connected,
          scopeId: cache?.scopeId, syncedAt: current && connected ? cache?.syncedAt ?? null : null, hasMore: current && cache?.hasMore === true,
          ...(current && cache?.error ? { error: cache.error } : {})
        }
      })
      return { accounts, events: events.sort((a, b) => b.createdAt - a.createdAt) }
    })
  }

  async enable(input: { accountId: string; revision: number; enabled: boolean }): Promise<GmailInboxSnapshot> {
    if (!input || typeof input.accountId !== 'string' || !Number.isSafeInteger(input.revision) || typeof input.enabled !== 'boolean') throw new Error('Choose a current Gmail account and Inbox setting.')
    this.controllers.get(input.accountId)?.abort()
    this.epochs.set(input.accountId, (this.epochs.get(input.accountId) ?? 0) + 1)
    await this.accounts.state.transaction(state => {
      const account = state.accounts.find(item => item.id === input.accountId && item.revision === input.revision)
      if (!account) throw new Error('Google account changed. Refresh its connection status.')
      if (input.enabled) {
        if (account.status !== 'connected') throw new Error('Reconnect this Google account before adding it to Inbox.')
        assertCapability(account.capabilities, 'gmail.read')
      }
      state.inbox = [...(state.inbox ?? []).filter(item => item.accountId !== input.accountId), { accountId: account.id, revision: account.revision, enabled: input.enabled, scopeId: randomUUID(), messages: [], syncedAt: null }]
      if (!input.enabled) state.actions = state.actions.filter(item => item.result || item.review.accountId !== account.id || item.review.action.kind !== 'gmail.send' || !item.review.action.email.reply)
    })
    return this.snapshot()
  }

  async assertSelected(accountId: string, revision: number): Promise<void> {
    const account = await this.accounts.current(accountId, revision)
    assertCapability(account.capabilities, 'gmail.read')
    const enabled = await this.accounts.state.read(state => state.inbox?.some(item => item.accountId === accountId && item.revision === revision && item.enabled))
    if (!enabled) throw new Error('This Gmail account is not selected for Inbox. Refresh its connection status.')
  }

  async sync(): Promise<GmailInboxSnapshot> {
    const snapshot = await this.snapshot()
    await Promise.all(snapshot.accounts.filter(account => account.enabled && account.connected).map(account => this.syncAccount(account.id, account.revision)))
    return this.snapshot()
  }

  private syncAccount(accountId: string, revision: number): Promise<void> {
    const epoch = this.epochs.get(accountId) ?? 0
    const key = `${accountId}:${revision}:${epoch}`, existing = this.syncing.get(key)
    if (existing) return existing
    const controller = new AbortController()
    this.controllers.set(accountId, controller)
    const job = (async () => {
      await this.assertSelected(accountId, revision)
      try {
        const page = await this.gmail.listMessages(accountId, { limit: 20, query: 'in:inbox -in:sent -in:drafts' }, revision, AbortSignal.any([controller.signal, AbortSignal.timeout(45_000)]))
        await this.assertSelected(accountId, revision)
        await this.accounts.state.transaction(state => {
          const cache = state.inbox?.find(item => item.accountId === accountId && item.revision === revision && item.enabled)
          const account = state.accounts.find(item => item.id === accountId && item.revision === revision && item.status === 'connected')
          if (!cache || !account || controller.signal.aborted) throw new Error('Gmail Inbox scope changed during synchronization.')
          cache.messages = page.items.filter(item => item.id && item.threadId).map(({ text: _text, ...metadata }) => metadata); cache.syncedAt = this.now(); cache.hasMore = Boolean(page.nextPageToken); delete cache.error
        })
      } catch (reason) {
        await this.accounts.state.transaction(state => {
          const cache = state.inbox?.find(item => item.accountId === accountId && item.revision === revision && item.enabled)
          if (cache && (this.epochs.get(accountId) ?? 0) === epoch) cache.error = googleRequestError(reason)
        })
      }
    })().finally(() => {
      if (this.syncing.get(key) === job) this.syncing.delete(key)
      if (this.controllers.get(accountId) === controller) this.controllers.delete(accountId)
    })
    this.syncing.set(key, job)
    return job
  }

  async thread(key: string): Promise<GoogleWorkspaceThread> {
    const target = parseGmailInboxKey(key)
    if (!target) throw new Error('Choose a Gmail conversation.')
    const epoch = this.epochs.get(target.accountId) ?? 0
    await this.assertSelected(target.accountId, target.revision)
    const thread = await this.gmail.readThread(target.accountId, target.threadId, target.revision)
    await this.assertSelected(target.accountId, target.revision)
    if ((this.epochs.get(target.accountId) ?? 0) !== epoch || thread.id !== target.threadId || thread.messages.some(message => message.threadId !== target.threadId)) throw new Error('The Gmail conversation or Inbox scope changed. Read it again.')
    return thread
  }

  async reply(key: string, text: string) {
    const target = parseGmailInboxKey(key)
    if (!target) throw new Error('Choose a Gmail conversation.')
    const epoch = this.epochs.get(target.accountId) ?? 0
    await this.assertSelected(target.accountId, target.revision)
    const message = await this.accounts.state.read(state => state.inbox?.find(item => item.accountId === target.accountId && item.revision === target.revision)?.messages.find(item => item.threadId === target.threadId))
    if (!message) throw new Error('Sync this Gmail conversation again before preparing a reply.')
    const review = await this.actions.prepare(target.accountId, { kind: 'gmail.send', email: { to: [], subject: message.subject, text, reply: { threadId: message.threadId, messageId: message.id } } }, undefined, target.revision)
    await this.assertSelected(target.accountId, target.revision)
    if ((this.epochs.get(target.accountId) ?? 0) !== epoch) throw new Error('Gmail Inbox scope changed while preparing this reply. Review it again.')
    return review
  }

  dispose(): void { for (const controller of this.controllers.values()) controller.abort() }
}
