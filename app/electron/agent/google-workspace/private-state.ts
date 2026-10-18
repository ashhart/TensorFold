import type { GoogleWorkspaceAccount, GoogleWorkspaceActionResult, GoogleWorkspaceActionReview, GoogleWorkspaceLocalDraft } from '../../../shared/google-workspace'
import { validEmail } from './validation'
import { validateCapabilities } from './scopes'

export interface GoogleDesktopClient { clientId: string; clientSecret?: string }
export interface GoogleWorkspaceTokens { accessToken: string; refreshToken?: string; expiresAt: number; scopes: string[] }
export interface PrivateGoogleAccount extends GoogleWorkspaceAccount { subject: string; tokens?: GoogleWorkspaceTokens }
export interface PrivateGoogleAction { review: GoogleWorkspaceActionReview; result?: GoogleWorkspaceActionResult }
export interface GoogleWorkspacePrivateState {
  version: 1
  client?: GoogleDesktopClient
  accounts: PrivateGoogleAccount[]
  drafts: GoogleWorkspaceLocalDraft[]
  actions: PrivateGoogleAction[]
}
/** The host must implement this with encrypted, account-bound, atomic storage. */
export interface GoogleWorkspaceStore {
  read(): Promise<GoogleWorkspacePrivateState | null>
  /** Production host must honor validateCommit immediately before its atomic rename. */
  write(state: GoogleWorkspacePrivateState, validateCommit?: () => void): Promise<void>
}
function checkedState(value: GoogleWorkspacePrivateState | null): GoogleWorkspacePrivateState {
  const state = structuredClone(value ?? { version: 1 as const, accounts: [], drafts: [], actions: [] })
  const invalid = () => { throw new Error('Google connection storage is invalid. Restore or remove this local connection data.') }
  if (state.version !== 1 || !Array.isArray(state.accounts) || state.accounts.length > 12 || !Array.isArray(state.drafts) || state.drafts.length > 100 || !Array.isArray(state.actions) || state.actions.length > 200) invalid()
  const ids = new Set<string>()
  for (const account of state.accounts) {
    if (!account || typeof account.id !== 'string' || !account.id.length || account.id.length > 128 || ids.has(account.id) || typeof account.subject !== 'string' || account.subject.length > 128 || !Number.isSafeInteger(account.revision) || account.revision < 1 || !['connected', 'disconnected', 'needs-reauthorization'].includes(account.status) || !Array.isArray(account.capabilities)) invalid()
    try { validEmail(account.email); if (account.capabilities.length) validateCapabilities(account.capabilities) } catch { invalid() }
    const tokens = account.tokens
    if (account.status === 'connected' && !tokens) invalid()
    if (tokens && (typeof tokens.accessToken !== 'string' || !tokens.accessToken || /[\s\0]/.test(tokens.accessToken) || tokens.accessToken.length > 16_384 || tokens.refreshToken !== undefined && (typeof tokens.refreshToken !== 'string' || !tokens.refreshToken || tokens.refreshToken.length > 16_384 || /[\s\0]/.test(tokens.refreshToken)) || !Number.isFinite(tokens.expiresAt) || !Array.isArray(tokens.scopes) || !tokens.scopes.length || tokens.scopes.length > 64 || tokens.scopes.some(scope => typeof scope !== 'string' || scope.length > 2048 || /\s/.test(scope)))) invalid()
    ids.add(account.id)
  }
  for (const draft of state.drafts) if (!draft || typeof draft.id !== 'string' || typeof draft.accountId !== 'string' || !ids.has(draft.accountId) || draft.localOnly !== true || !Array.isArray(draft.to) || typeof draft.subject !== 'string' || typeof draft.text !== 'string') invalid()
  for (const action of state.actions) if (!action || !action.review || typeof action.review.id !== 'string' || !ids.has(action.review.accountId) || !Number.isSafeInteger(action.review.revision) || typeof action.review.digest !== 'string' || !/^[a-f0-9]{64}$/.test(action.review.digest) || !Number.isFinite(action.review.expiresAt) || !action.review.action || !['gmail.send', 'calendar.create', 'calendar.update', 'calendar.delete'].includes(action.review.action.kind) || action.result && !['succeeded', 'uncertain', 'failed'].includes(action.result.status)) invalid()
  if (state.client && (typeof state.client.clientId !== 'string' || typeof state.client.clientSecret !== 'string' || !state.client.clientSecret)) invalid()
  return state
}
export class GoogleState {
  private queue: Promise<unknown> = Promise.resolve()
  constructor(private readonly store: GoogleWorkspaceStore) {}
  transaction<T>(run: (state: GoogleWorkspacePrivateState) => Promise<T> | T, validateCommit?: () => void): Promise<T> {
    const job = this.queue.then(async () => {
      const state = checkedState(await this.store.read())
      const result = await run(state)
      validateCommit?.()
      await this.store.write(state, validateCommit)
      return result
    })
    this.queue = job.catch(() => {})
    return job
  }
  read<T>(run: (state: GoogleWorkspacePrivateState) => T): Promise<T> {
    return this.queue.then(async () => run(checkedState(await this.store.read())))
  }
}
