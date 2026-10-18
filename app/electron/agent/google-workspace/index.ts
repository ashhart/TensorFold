import { randomUUID } from 'node:crypto'
import type { GoogleWorkspaceAction, GoogleWorkspaceAuthRequest, GoogleWorkspaceAuthorization, GoogleWorkspaceEmailContent, GoogleWorkspaceEventQuery, GoogleWorkspaceMessageQuery, GoogleWorkspaceReviewedAction, GoogleWorkspaceSnapshot } from '../../../shared/google-workspace'
import { GoogleAccounts, publicGoogleAccount } from './accounts'
import { GoogleActions } from './actions'
import { GoogleCalendar } from './calendar'
import { parseGoogleDesktopClient } from './client-config'
import { GoogleDrafts } from './drafts'
import { GoogleGmail } from './gmail'
import { createGoogleOAuthLoopback, type GoogleOAuthLoopback } from './oauth-loopback'
import { exchangeGoogleCode, googleIdentity } from './oauth-tokens'
import { GoogleState, type GoogleDesktopClient, type GoogleWorkspaceStore } from './private-state'
import { grantedCapabilities, requestedScopes, validateCapabilities } from './scopes'
import { GoogleTransport, type GoogleTransportDeps } from './transport'

export type { GoogleWorkspaceStore, GoogleWorkspacePrivateState } from './private-state'
export interface GoogleWorkspaceServiceDeps extends GoogleTransportDeps {
  store: GoogleWorkspaceStore
  /** Host must use the system browser, never a WebView or an embedded login. */
  openExternal(url: string): Promise<void>
  now?: () => number
  onChange?: () => void
  authorizationTimeoutMs?: number
}
interface AuthJob {
  public: GoogleWorkspaceAuthorization
  loopback: GoogleOAuthLoopback
  controller: AbortController
  client: GoogleDesktopClient
  generation: number
  expected?: { id: string; revision: number }
}
export class GoogleWorkspaceService {
  private readonly state: GoogleState
  private readonly transport: GoogleTransport
  private readonly accounts: GoogleAccounts
  private readonly gmail: GoogleGmail
  private readonly calendar: GoogleCalendar
  private readonly drafts: GoogleDrafts
  private readonly actions: GoogleActions
  private readonly now: () => number
  private active?: AuthJob
  private authorization?: GoogleWorkspaceAuthorization
  private generation = 0
  private starting = false
  constructor(private readonly deps: GoogleWorkspaceServiceDeps) {
    this.now = deps.now ?? Date.now
    this.state = new GoogleState(deps.store)
    this.transport = new GoogleTransport(deps)
    this.accounts = new GoogleAccounts(this.state, this.transport, this.now)
    this.gmail = new GoogleGmail(this.accounts)
    this.calendar = new GoogleCalendar(this.accounts)
    this.drafts = new GoogleDrafts(this.accounts, this.now)
    this.actions = new GoogleActions(this.accounts, this.now)
  }
  async snapshot(): Promise<GoogleWorkspaceSnapshot> {
    const reviews = await this.actions.listReviews()
    const recentActions = await this.actions.recentActions()
    return this.state.read(state => ({ clientConfigured: Boolean(state.client), accounts: state.accounts.map(publicGoogleAccount), reviews, recentActions, ...(this.authorization ? { authorization: structuredClone(this.authorization) } : {}) }))
  }
  async configureClient(json: string): Promise<GoogleWorkspaceSnapshot> {
    const client = parseGoogleDesktopClient(json)
    this.cancelAuthorization()
    this.generation++
    await this.state.transaction(state => {
      if (state.client && (state.client.clientId !== client.clientId || state.client.clientSecret !== client.clientSecret) && state.accounts.some(account => account.status !== 'disconnected')) throw new Error('Disconnect Google accounts before changing the desktop client.')
      state.client = client
    })
    this.changed()
    return this.snapshot()
  }
  async startAuthorization(request: GoogleWorkspaceAuthRequest): Promise<GoogleWorkspaceAuthorization> {
    if (this.starting || this.active) throw new Error('Google sign-in is already in progress. Finish or cancel it first.')
    const capabilities = validateCapabilities(request?.capabilities)
    this.starting = true
    const generation = this.generation
    let loopback: GoogleOAuthLoopback | undefined
    try {
      const { client, expected } = await this.state.read(state => {
        if (!state.client) throw new Error('Import your Google desktop OAuth client JSON first.')
        const account = request.accountId ? state.accounts.find(item => item.id === request.accountId) : undefined
        if (request.accountId && !account) throw new Error('Google account does not exist.')
        return { client: state.client, expected: account ? { id: account.id, revision: account.revision } : undefined }
      })
      loopback = await createGoogleOAuthLoopback(client, requestedScopes(capabilities), this.deps.authorizationTimeoutMs)
      if (generation !== this.generation) { loopback.cancel(); throw new Error('Google sign-in was cancelled.') }
      const job: AuthJob = { public: { id: randomUUID(), expiresAt: this.now() + (this.deps.authorizationTimeoutMs ?? 180_000), capabilities, status: 'waiting' }, loopback, controller: new AbortController(), client, generation, expected }
      this.active = job; this.authorization = job.public
      void this.finishAuthorization(job)
      this.changed()
      try { await this.deps.openExternal(loopback.authorizationUrl) } catch { this.cancelAuthorization(job.public.id); throw new Error('Could not open the system browser for Google sign-in.') }
      return structuredClone(job.public)
    } catch (error) { loopback?.cancel(); throw error }
    finally { this.starting = false }
  }
  cancelAuthorization(id?: string): void {
    if (id && this.active?.public.id !== id) return
    this.generation++
    const job = this.active
    if (!job) return
    this.active = undefined
    job.controller.abort(); job.loopback.cancel(); job.public.status = 'cancelled'
    this.authorization = job.public
    this.changed()
  }
  private async finishAuthorization(job: AuthJob): Promise<void> {
    const assertValid = () => { if (this.active !== job || job.generation !== this.generation || job.controller.signal.aborted) throw new Error('Google sign-in was cancelled.') }
    try {
      const code = await job.loopback.completion
      assertValid()
      const tokens = await exchangeGoogleCode(this.transport, job.client, code, job.loopback.redirectUri, job.loopback.verifier, this.now(), job.controller.signal)
      assertValid()
      const identity = await googleIdentity(this.transport, tokens, job.controller.signal)
      assertValid()
      const capabilities = grantedCapabilities(tokens.scopes).filter(capability => job.public.capabilities.includes(capability))
      if (!capabilities.length) throw new Error('Google did not grant any of the requested Gmail or Calendar permissions.')
      await this.accounts.connect(identity.subject, identity.email, tokens, capabilities, job.expected, assertValid)
      assertValid()
      job.public.capabilities = capabilities
      job.public.status = 'complete'
      this.authorization = job.public
    } catch (error) {
      if (this.active !== job) return
      job.public.status = 'failed'
      job.public.error = error instanceof Error ? error.message : 'Google sign-in failed.'
      this.authorization = job.public
    } finally {
      if (this.active === job) this.active = undefined
      this.changed()
    }
  }
  async disconnect(accountId: string) {
    // Cancel every pending sign-in before disconnect so an unselected callback cannot reconnect it.
    this.cancelAuthorization()
    const result = await this.accounts.disconnect(accountId)
    this.changed()
    return result
  }
  listMessages(accountId: string, options?: GoogleWorkspaceMessageQuery) { return this.gmail.listMessages(accountId, options) }
  readMessage(accountId: string, id: string) { return this.gmail.readMessage(accountId, id) }
  readThread(accountId: string, id: string) { return this.gmail.readThread(accountId, id) }
  listCalendars(accountId: string, options?: { limit?: number; pageToken?: string }) { return this.calendar.listCalendars(accountId, options) }
  listEvents(accountId: string, options: GoogleWorkspaceEventQuery) { return this.calendar.listEvents(accountId, options) }
  freeBusy(accountId: string, options: { calendarIds: string[]; timeMin: string; timeMax: string; timeZone?: string }) { return this.calendar.freeBusy(accountId, options) }
  saveDraft(accountId: string, email: GoogleWorkspaceEmailContent, id?: string, assertAllowed?: () => void) { return this.drafts.save(accountId, email, id, assertAllowed) }
  listDrafts(accountId: string) { return this.drafts.list(accountId) }
  deleteDraft(accountId: string, id: string) { return this.drafts.delete(accountId, id) }
  async prepareAction(accountId: string, action: GoogleWorkspaceAction, assertAllowed?: () => void) { const review = await this.actions.prepare(accountId, action, assertAllowed); this.changed(); return review }
  getReview(receipt: GoogleWorkspaceReviewedAction) { return this.actions.getReview(receipt) }
  async cancelAction(receipt: GoogleWorkspaceReviewedAction) { const result = await this.actions.cancel(receipt); this.changed(); return result }
  async executeReviewed(receipt: GoogleWorkspaceReviewedAction) { const result = await this.actions.execute(receipt); this.changed(); return result }
  dispose(): void { this.cancelAuthorization(); this.accounts.dispose() }
  private changed(): void { try { this.deps.onChange?.() } catch { /* Host notifications cannot invalidate a completed account operation. */ } }
}
