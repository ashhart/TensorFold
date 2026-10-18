import { randomUUID } from 'node:crypto'
import type { GoogleWorkspaceAccount, GoogleWorkspaceCapability } from '../../../shared/google-workspace'
import { GoogleState, type PrivateGoogleAccount } from './private-state'
import { assertCapability, grantedCapabilities } from './scopes'
import { GOOGLE_REVOKE_URL, GOOGLE_TOKEN_URL } from './client-config'
import { parseGoogleTokens, tokenForm } from './oauth-tokens'
import { GoogleApiError, GoogleTransport } from './transport'

export function publicGoogleAccount(account: PrivateGoogleAccount): GoogleWorkspaceAccount {
  return { id: account.id, email: account.email, revision: account.revision, capabilities: [...account.capabilities], status: account.status, connectedAt: account.connectedAt, ...(account.lastError ? { lastError: account.lastError } : {}) }
}
export class GoogleAccounts {
  private readonly refreshing = new Map<string, Promise<void>>()
  private readonly controllers = new Map<string, AbortController>()
  constructor(readonly state: GoogleState, readonly transport: GoogleTransport, private readonly now: () => number) {}
  dispose(): void { for (const controller of this.controllers.values()) controller.abort() }
  async current(id: string, revision?: number): Promise<PrivateGoogleAccount> {
    return this.state.read(state => {
      const account = state.accounts.find(item => item.id === id)
      if (!account || account.status !== 'connected' || !account.tokens) throw new Error('Google account is disconnected. Reconnect it in Connections.')
      if (revision !== undefined && account.revision !== revision) throw new Error('Google account changed. Review this operation again.')
      return account
    })
  }
  private controller(id: string): AbortController {
    let value = this.controllers.get(id)
    if (value?.signal.aborted) throw new Error('Google account changed. Try this operation again.')
    if (!value) { value = new AbortController(); this.controllers.set(id, value) }
    return value
  }
  async withAccess<T>(id: string, capability: GoogleWorkspaceCapability, run: (token: string, revision: number, signal: AbortSignal) => Promise<T>, revision?: number): Promise<T> {
    let account = await this.current(id, revision)
    assertCapability(account.capabilities, capability)
    if (account.tokens!.expiresAt <= this.now() + 60_000) {
      await this.refresh(id, account.revision)
      account = await this.current(id, revision)
      assertCapability(account.capabilities, capability)
    }
    const signal = this.controller(id).signal
    try {
      const result = await run(account.tokens!.accessToken, account.revision, signal)
      await this.current(id, account.revision)
      return result
    } catch (error) {
      if (error instanceof GoogleApiError && error.status === 401) await this.needsAuthorization(id, account.revision)
      throw error
    }
  }
  private refresh(id: string, revision: number): Promise<void> {
    const existing = this.refreshing.get(id)
    if (existing) return existing
    const job = this.doRefresh(id, revision).finally(() => { if (this.refreshing.get(id) === job) this.refreshing.delete(id) })
    this.refreshing.set(id, job)
    return job
  }
  private async doRefresh(id: string, revision: number): Promise<void> {
    const account = await this.current(id, revision)
    const client = await this.state.read(state => state.client)
    if (!client || !account.tokens?.refreshToken) { await this.needsAuthorization(id, revision); throw new Error('Google sign-in must be renewed. Reconnect this account.') }
    try {
      const data = await this.transport.form(GOOGLE_TOKEN_URL, tokenForm(client, { grant_type: 'refresh_token', refresh_token: account.tokens.refreshToken }), this.controller(id).signal)
      const tokens = parseGoogleTokens(data, this.now(), account.tokens)
      const capabilities = grantedCapabilities(tokens.scopes).filter(capability => account.capabilities.includes(capability))
      await this.state.transaction(state => {
        const live = state.accounts.find(item => item.id === id)
        if (!live || live.revision !== revision || live.status !== 'connected') throw new Error('Google account changed during token refresh.')
        live.tokens = tokens
        if (live.capabilities.some(capability => !capabilities.includes(capability))) live.revision++
        live.capabilities = capabilities
      })
    } catch (error) {
      if (error instanceof GoogleApiError && (error.status === 400 || error.status === 401)) await this.needsAuthorization(id, revision)
      throw error
    }
  }
  async needsAuthorization(id: string, revision: number): Promise<void> {
    await this.state.transaction(state => {
      const live = state.accounts.find(item => item.id === id)
      if (!live || live.revision !== revision) return
      // A late failure from an older session must not cancel a newer sign-in's controller.
      const controller = this.controllers.get(id) ?? new AbortController(); controller.abort(); this.controllers.set(id, controller)
      live.status = 'needs-reauthorization'; live.lastError = 'Google sign-in expired or was revoked. Reconnect this account.'; live.revision++
      live.tokens = undefined
    })
  }
  async connect(subject: string, email: string, tokens: PrivateGoogleAccount['tokens'], capabilities: GoogleWorkspaceCapability[], expected?: { id: string; revision: number }, assertValid?: () => void): Promise<void> {
    await this.state.transaction(state => {
      assertValid?.()
      const previous = state.accounts.find(account => account.subject === subject)
      if (expected && (!previous || previous.id !== expected.id || previous.revision !== expected.revision)) throw new Error('A different Google account was selected or the connection changed. Try again.')
      if (!previous && state.accounts.length >= 12) throw new Error('Remove an unused Google connection before adding another.')
      const account: PrivateGoogleAccount = { id: previous?.id ?? randomUUID(), subject, email, tokens, capabilities, status: 'connected', connectedAt: this.now(), revision: (previous?.revision ?? 0) + 1 }
      if (previous) { this.controllers.get(previous.id)?.abort(); this.controllers.set(previous.id, new AbortController()); Object.assign(previous, account); delete previous.lastError }
      else state.accounts.push(account)
    })
  }
  async disconnect(id: string): Promise<{ revoked: boolean; detail: string }> {
    const controller = this.controllers.get(id) ?? new AbortController(); controller.abort(); this.controllers.set(id, controller)
    const disconnected = await this.state.transaction(state => {
      const account = state.accounts.find(item => item.id === id)
      if (!account) throw new Error('Google account does not exist.')
      const previous = account.tokens
      account.tokens = undefined; account.capabilities = []; account.status = 'disconnected'; account.revision++; delete account.lastError
      state.drafts = state.drafts.filter(draft => draft.accountId !== id)
      state.actions = state.actions.filter(action => action.review.accountId !== id)
      return { tokens: previous, revision: account.revision }
    })
    const warn = async () => {
      const detail = 'Disconnected locally; Google revocation could not be confirmed. Remove access in your Google Account settings.'
      await this.state.transaction(state => {
        const account = state.accounts.find(item => item.id === id)
        if (account?.status === 'disconnected' && account.revision === disconnected.revision) account.lastError = detail
      })
      return { revoked: false, detail }
    }
    const tokens = disconnected.tokens
    if (!tokens) return warn()
    try {
      await this.transport.form(GOOGLE_REVOKE_URL, new URLSearchParams({ token: tokens.refreshToken ?? tokens.accessToken }))
      return { revoked: true, detail: 'Google account disconnected and access revoked.' }
    } catch { return warn() }
  }
}
