import type { GoogleWorkspaceApi } from '../../shared/google-workspace-api'
import type { GoogleWorkspaceAccount, GoogleWorkspaceAuthRequest, GoogleWorkspaceLocalDraft, GoogleWorkspaceMessage, GoogleWorkspaceSnapshot } from '../../shared/google-workspace'

export type Scenario = 'multiple' | 'empty' | 'disconnected' | 'expired' | 'discovery-error' | 'first-setup'
export const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: 'multiple', label: 'Multiple connected accounts' },
  { value: 'empty', label: 'No accounts — client ready' },
  { value: 'disconnected', label: 'Disconnected account' },
  { value: 'expired', label: 'Expired on next Gmail read' },
  { value: 'discovery-error', label: 'Discovery error — Retry recovers' },
  { value: 'first-setup', label: 'First setup — import client' },
]
const clone = <T,>(value: T): T => structuredClone(value)
const firstAccount = (status: GoogleWorkspaceAccount['status'] = 'connected'): GoogleWorkspaceAccount => ({ id: 'synthetic-first', email: 'alex@example.com', revision: 1, capabilities: ['gmail.read', 'calendar.read'], status, connectedAt: 1 })
const secondAccount = (): GoogleWorkspaceAccount => ({ ...firstAccount(), id: 'synthetic-second', email: 'casey@example.com' })

export function createRecoveryFixture() {
  let state: GoogleWorkspaceSnapshot = { clientConfigured: true, accounts: [], reviews: [], recentActions: [] }
  let drafts: GoogleWorkspaceLocalDraft[] = []
  let pending: GoogleWorkspaceAuthRequest | undefined
  let failDiscovery = false, expireNextRead = false, sequence = 0
  let notice = 'Only synthetic fixture data is present.'
  const listeners = new Set<() => void>()
  const changed = () => listeners.forEach(listener => listener())
  const uid = () => `synthetic-${++sequence}`
  const summary = () => ({ notice, expireNextRead, waiting: Boolean(pending), accounts: state.accounts.map(account => `${account.email}: ${account.status} (revision ${account.revision})`).join(' · ') || 'No saved synthetic accounts', drafts: drafts.length })
  const account = (id: string) => {
    const value = state.accounts.find(item => item.id === id)
    if (!value || value.status !== 'connected') throw new Error('Synthetic Google account needs reconnection.')
    return value
  }
  const read = (id: string) => {
    const value = account(id)
    if (expireNextRead) {
      expireNextRead = false
      value.status = 'needs-reauthorization'; value.revision++
      value.lastError = 'Synthetic Google sign-in expired or was revoked. Reconnect this account.'
      notice = 'A synthetic Gmail read expired the selected account; the real request hook must refresh its status.'
      changed()
      throw new Error('Synthetic Google sign-in expired.')
    }
    return value
  }
  const mail = (id: string): GoogleWorkspaceMessage => ({ id: 'synthetic-message', threadId: 'synthetic-thread', subject: 'Synthetic recovery check', from: 'sender@example.com', to: state.accounts.find(item => item.id === id)?.email ?? 'alex@example.com', date: '2026-10-22T12:00:00Z', snippet: 'This message is generated locally by the fixture.', text: 'Synthetic fixture content only; no mailbox was accessed.', attachments: [], untrustedContent: true })
  const api: GoogleWorkspaceApi & { platform: string; openExternal(url: string): Promise<void> } = {
    platform: 'darwin',
    openExternal: async () => { notice = 'External navigation was intercepted by the synthetic fixture.'; changed() },
    googleWorkspaceStatus: async () => {
      if (failDiscovery) { failDiscovery = false; notice = 'Synthetic discovery failure returned; the next Retry succeeds.'; changed(); throw new Error('Synthetic discovery storage is temporarily unavailable.') }
      return clone(state)
    },
    googleWorkspaceImportClient: async () => { state.clientConfigured = true; notice = 'Synthetic Desktop client configured; no file or credential was read.'; changed(); return clone(state) },
    googleWorkspaceAuthorize: async request => {
      if (!state.clientConfigured) throw new Error('Configure the synthetic client first.')
      pending = clone(request)
      state.authorization = { id: uid(), expiresAt: Date.now() + 180_000, capabilities: clone(request.capabilities), status: 'waiting' }
      notice = 'Synthetic sign-in is waiting; use Finish fake sign-in in the fixture controls.'
      changed(); return clone(state.authorization)
    },
    googleWorkspaceCancelAuth: async () => { pending = undefined; if (state.authorization) state.authorization.status = 'cancelled'; notice = 'Synthetic sign-in cancelled.'; changed(); return clone(state) },
    googleWorkspaceDisconnect: async id => {
      const value = account(id)
      value.status = 'disconnected'; value.capabilities = []; value.revision++
      drafts = drafts.filter(item => item.accountId !== id)
      notice = 'Synthetic account disconnected locally.'; changed(); return clone(state)
    },
    googleWorkspaceMessages: async id => { read(id); return { items: [mail(id)] } },
    googleWorkspaceMessage: async id => { read(id); return mail(id) },
    googleWorkspaceThread: async id => { read(id); return { id: 'synthetic-thread', messages: [mail(id)] } },
    googleWorkspaceCalendars: async id => { account(id); return { items: [{ id: 'primary', summary: 'Synthetic calendar', timeZone: 'Europe/London', accessRole: 'owner' }] } },
    googleWorkspaceEvents: async id => { account(id); return { items: [] } },
    googleWorkspaceAvailability: async (id, query) => { account(id); return { timeMin: query.timeMin, timeMax: query.timeMax, calendars: query.calendarIds.map(calendarId => ({ id: calendarId, busy: [], errors: [] })) } },
    googleWorkspaceDrafts: async id => { account(id); return clone(drafts.filter(item => item.accountId === id)) },
    googleWorkspaceSaveDraft: async (id, email, draftId) => {
      account(id)
      const draft: GoogleWorkspaceLocalDraft = { ...clone(email), id: draftId ?? uid(), accountId: id, updatedAt: Date.now(), localOnly: true }
      drafts = [draft, ...drafts.filter(item => item.id !== draft.id)]
      notice = `Saved a synthetic local draft for ${id}; no external action occurred.`; changed(); return clone(draft)
    },
    googleWorkspaceDeleteDraft: async (id, draftId) => { account(id); drafts = drafts.filter(item => item.id !== draftId || item.accountId !== id); changed() },
    googleWorkspacePrepareAction: async (id, action) => {
      const owner = account(id)
      const review = { id: uid(), accountId: id, accountEmail: owner.email, revision: owner.revision, digest: '0'.repeat(64), expiresAt: Date.now() + 300_000, action: clone(action) }
      state.reviews.push(review); changed(); return clone(review)
    },
    googleWorkspaceCancelAction: async receipt => { state.reviews = state.reviews.filter(item => item.id !== receipt.id); changed(); return { id: receipt.id, status: 'failed', cancelled: true } },
    googleWorkspaceExecuteAction: async receipt => {
      const review = state.reviews.find(item => item.id === receipt.id)
      if (!review) throw new Error('Synthetic review unavailable.')
      const result = { id: receipt.id, status: 'succeeded' as const, remoteId: 'synthetic-only' }
      state.reviews = state.reviews.filter(item => item.id !== receipt.id); state.recentActions.push({ review, result }); changed(); return result
    },
  }
  const reset = (scenario: Scenario) => {
    state = { clientConfigured: scenario !== 'first-setup', accounts: [], reviews: [], recentActions: [] }
    drafts = []; pending = undefined; failDiscovery = scenario === 'discovery-error'; expireNextRead = scenario === 'expired'
    if (scenario !== 'empty' && scenario !== 'first-setup') state.accounts = [firstAccount(scenario === 'disconnected' ? 'disconnected' : 'connected')]
    if (scenario === 'multiple') state.accounts.push(secondAccount())
    notice = scenario === 'expired' ? 'Type an unsaved email, then Search Gmail; the next read will expire auth.' : 'Scenario reset to synthetic data; changing scenarios intentionally remounts the connector.'
    changed()
  }
  const finishSignIn = () => {
    if (!pending) { notice = 'Choose Sign in with Google or Reconnect with selected access in the real connector first.'; changed(); return }
    let value = pending.accountId ? state.accounts.find(item => item.id === pending!.accountId) : undefined
    if (!value) { value = state.accounts.length ? secondAccount() : firstAccount(); if (state.accounts.some(item => item.id === value!.id)) value = { ...value, id: uid(), email: `account-${sequence}@example.com` }; state.accounts.push(value) }
    value.status = 'connected'; value.revision++; value.capabilities = clone(pending.capabilities); delete value.lastError
    if (state.authorization) state.authorization.status = 'complete'
    pending = undefined; notice = 'Synthetic sign-in completed for the same selected account; the real hook polls for completion.'; changed()
  }
  return { api, reset, summary, finishSignIn, armExpiry: () => { expireNextRead = true; notice = 'Next Gmail search or read will expire that synthetic account.'; changed() }, subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } } }
}
