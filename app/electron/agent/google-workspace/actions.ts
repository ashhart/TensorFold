import { createHash, randomUUID, timingSafeEqual } from 'node:crypto'
import type { GoogleWorkspaceAction, GoogleWorkspaceActionResult, GoogleWorkspaceActionReview, GoogleWorkspaceReviewedAction, GoogleWorkspaceSettledAction } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { emailMime, googleEventBody, validatedGoogleAction } from './action-validation'
import { assertCapability } from './scopes'
import { GoogleApiError } from './transport'
import { googleResourceId, responseText } from './validation'
import { parseGoogleEvent } from './calendar'

const REVIEW_TTL = 10 * 60_000
function digestReview(review: Omit<GoogleWorkspaceActionReview, 'digest'>): string { return createHash('sha256').update(JSON.stringify(review)).digest('hex') }
function sameDigest(actual: string, expected: string): boolean { return /^[a-f0-9]{64}$/.test(actual) && timingSafeEqual(Buffer.from(actual), Buffer.from(expected)) }
export class GoogleActions {
  constructor(private readonly accounts: GoogleAccounts, private readonly now: () => number) {}
  listReviews() {
    return this.accounts.state.read(state => state.actions.filter(item => !item.result && item.review.expiresAt >= this.now() && state.accounts.some(account => account.id === item.review.accountId && account.revision === item.review.revision && account.status === 'connected')).map(item => structuredClone(item.review)))
  }
  recentActions(): Promise<GoogleWorkspaceSettledAction[]> {
    return this.accounts.state.read(state => state.actions.filter(item => item.result && state.accounts.some(account => account.id === item.review.accountId && account.status !== 'disconnected')).slice(-20).reverse().map(item => ({ review: structuredClone(item.review), result: structuredClone(item.result!) })))
  }
  async getReview(receipt: GoogleWorkspaceReviewedAction): Promise<GoogleWorkspaceActionReview> {
    const reviews = await this.listReviews()
    const review = reviews.find(item => item.id === receipt.id && item.accountId === receipt.accountId && item.revision === receipt.revision && typeof receipt.digest === 'string' && sameDigest(receipt.digest, item.digest))
    if (!review) throw new Error('This Google review expired or the connection changed.')
    return review
  }
  async cancel(receipt: GoogleWorkspaceReviewedAction): Promise<GoogleWorkspaceActionResult> {
    await this.getReview(receipt)
    return this.accounts.state.transaction(state => {
      const action = state.actions.find(item => item.review.id === receipt.id)
      if (!action || action.result || action.review.digest !== receipt.digest) throw new Error('This Google action already started or is no longer pending.')
      action.result = { id: receipt.id, status: 'failed', cancelled: true, error: 'Cancelled by you. Nothing was sent.' }
      return structuredClone(action.result)
    })
  }
  async prepare(accountId: string, input: GoogleWorkspaceAction, assertAllowed?: () => void): Promise<GoogleWorkspaceActionReview> {
    const action = validatedGoogleAction(input)
    const account = await this.accounts.current(accountId)
    assertCapability(account.capabilities, action.kind === 'gmail.send' ? 'gmail.send' : 'calendar.write')
    let previousEvent: GoogleWorkspaceActionReview['previousEvent']
    if (action.kind === 'calendar.update' || action.kind === 'calendar.delete') {
      const calendar = googleResourceId(action.calendarId, 'calendar ID'); const event = googleResourceId(action.eventId, 'event ID')
      const existing = await this.accounts.withAccess(accountId, 'calendar.write', (token, _revision, signal) => this.accounts.transport.json(`https://www.googleapis.com/calendar/v3/calendars/${calendar}/events/${event}`, token, undefined, 'GET', signal), account.revision)
      if (existing?.etag !== action.etag || existing?.id !== action.eventId) throw new Error('This calendar event changed. Read it again before preparing an edit.')
      previousEvent = parseGoogleEvent(existing, action.calendarId)
      if (action.kind === 'calendar.delete') action.title = previousEvent.title || 'Untitled event'
    }
    const content = { id: randomUUID(), accountId, accountEmail: account.email, revision: account.revision, expiresAt: this.now() + REVIEW_TTL, action, ...(previousEvent ? { previousEvent } : {}) }
    const review = { ...content, digest: digestReview(content) }
    await this.accounts.state.transaction(state => {
      assertAllowed?.()
      const live = state.accounts.find(item => item.id === accountId)
      if (!live || live.revision !== account.revision || live.status !== 'connected') throw new Error('Google account changed. Review this operation again.')
      state.actions = state.actions.filter(item => item.result || item.review.expiresAt >= this.now())
      // Missing old IDs still fail closed, so definitive receipts can be pruned without allowing replay.
      while (state.actions.length >= 200) {
        const settled = state.actions.findIndex(item => item.result && item.result.status !== 'uncertain')
        if (settled < 0) throw new Error('Google action history is full of pending or uncertain operations. Check Google and cancel unused pending reviews first.')
        state.actions.splice(settled, 1)
      }
      state.actions.push({ review })
    }, assertAllowed)
    return structuredClone(review)
  }
  /** Host-owned UI review only. Models cannot approve an action, alter its payload, or retry it. */
  async execute(receipt: GoogleWorkspaceReviewedAction): Promise<GoogleWorkspaceActionResult> {
    if (!receipt || typeof receipt.digest !== 'string') throw new Error('An owner review is required.')
    const stored = await this.accounts.state.read(state => state.actions.find(item => item.review.id === receipt.id))
    if (!stored || stored.review.accountId !== receipt.accountId || stored.review.revision !== receipt.revision || !sameDigest(receipt.digest, stored.review.digest)) throw new Error('The Google review no longer matches this operation.')
    if (stored.result) return stored.result
    if (stored.review.expiresAt < this.now()) throw new Error('This Google review expired. Review the operation again.')
    const review = stored.review
    const capability = review.action.kind === 'gmail.send' ? 'gmail.send' : 'calendar.write'
    let started = false
    try {
      return await this.accounts.withAccess(receipt.accountId, capability, async (token, revision, signal) => {
        const prior = await this.accounts.state.transaction(state => {
          const live = state.accounts.find(item => item.id === receipt.accountId)
          const action = state.actions.find(item => item.review.id === receipt.id)
          if (!live || live.status !== 'connected' || live.revision !== revision || revision !== receipt.revision || !action || action.review.expiresAt < this.now()) throw new Error('Google account or review changed. Review this operation again.')
          if (action.result) return action.result
          // Persist an uncertain receipt BEFORE a network call. Crashes and timeouts cannot resend it.
          action.result = { id: receipt.id, status: 'uncertain', error: 'The operation started but its outcome has not been confirmed. Check Google before preparing another action.' }
          return undefined
        })
        if (prior) return prior
        // Storage rejection above means the request was never admitted or sent.
        started = true
        if (signal.aborted) return this.record(receipt.id, { id: receipt.id, status: 'failed', error: 'Google account disconnected before the request was sent.' })
        try {
          const data = await this.send(review, token, signal)
          const remoteId = responseText(data?.id, 512)
          return await this.record(receipt.id, { id: receipt.id, status: 'succeeded', ...(remoteId ? { remoteId } : {}) })
        } catch (error) {
          const definite = error instanceof GoogleApiError && error.status < 500
          if (error instanceof GoogleApiError && error.status === 401) await this.accounts.needsAuthorization(receipt.accountId, revision)
          return this.record(receipt.id, { id: receipt.id, status: definite ? 'failed' : 'uncertain', error: definite ? error.message : 'Google may have applied this operation, but confirmation was lost. Check Google before preparing another action.' })
        }
      }, receipt.revision)
    } catch (error) {
      if (started) {
        // A 401 changes the account revision after recording a definite refusal.
        const recorded = await this.accounts.state.read(state => state.actions.find(item => item.review.id === receipt.id && item.review.accountId === receipt.accountId && item.review.digest === receipt.digest)?.result).catch(() => undefined)
        if (recorded?.status === 'failed') return recorded
        return { id: receipt.id, status: 'uncertain', error: 'The connection changed after this operation started. Check Google before preparing another action.' }
      }
      throw error
    }
  }
  private async record(id: string, result: GoogleWorkspaceActionResult): Promise<GoogleWorkspaceActionResult> {
    await this.accounts.state.transaction(state => {
      const action = state.actions.find(item => item.review.id === id)
      if (action) action.result = result
    })
    return result
  }
  private send(review: GoogleWorkspaceActionReview, token: string, signal: AbortSignal): Promise<any> {
    const action = review.action
    if (action.kind === 'gmail.send') return this.accounts.transport.json('https://gmail.googleapis.com/gmail/v1/users/me/messages/send', token, { raw: emailMime(action.email, review.accountEmail) }, 'POST', signal)
    const calendar = googleResourceId(action.calendarId, 'calendar ID')
    const endpoint = `https://www.googleapis.com/calendar/v3/calendars/${calendar}/events`
    if (action.kind === 'calendar.create') return this.accounts.transport.json(`${endpoint}?sendUpdates=${action.event.sendUpdates}`, token, googleEventBody(action.event), 'POST', signal)
    const eventId = googleResourceId(action.eventId, 'event ID')
    const headers = { 'If-Match': action.etag }
    if (action.kind === 'calendar.update') return this.accounts.transport.json(`${endpoint}/${eventId}?sendUpdates=${action.event.sendUpdates}`, token, googleEventBody(action.event), 'PATCH', signal, headers)
    return this.accounts.transport.json(`${endpoint}/${eventId}?sendUpdates=${action.sendUpdates}`, token, undefined, 'DELETE', signal, headers)
  }
}
