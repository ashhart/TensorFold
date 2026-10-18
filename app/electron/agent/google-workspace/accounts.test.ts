import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import { requestedScopes } from './scopes'
import type { GoogleWorkspacePrivateState, GoogleWorkspaceStore } from './private-state'

const now = 100_000
const lookup = async () => [{ address: '8.8.8.8', family: 4 }]
function initial(): GoogleWorkspacePrivateState {
  return { version: 1, client: { clientId: '123456789012-synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' }, accounts: [{ id: 'account-a', subject: 'synthetic_subject_a', email: 'person-a@example.com', revision: 1, status: 'connected', capabilities: ['gmail.read', 'calendar.read'], connectedAt: now, tokens: { accessToken: 'synthetic-expired-access', refreshToken: 'synthetic-refresh', expiresAt: now, scopes: requestedScopes(['gmail.read', 'calendar.read']) } }], drafts: [], actions: [] }
}
function memory(value: GoogleWorkspacePrivateState): GoogleWorkspaceStore {
  return { read: async () => structuredClone(value), write: async state => { value = structuredClone(state) } }
}
const store = memory(initial())
let refreshCalls = 0; let readCalls = 0
let resolveRefresh!: (value: Response) => void
let refreshStarted!: () => void
const starting = new Promise<void>(resolve => { refreshStarted = resolve })
const service = new GoogleWorkspaceService({ store, openExternal: async () => {}, now: () => now, lookup, fetchImpl: (async (url: any, init: RequestInit) => {
  if (String(url).endsWith('/token')) { refreshCalls++; refreshStarted(); return new Promise<Response>(resolve => { resolveRefresh = resolve }) }
  readCalls++
  assert.equal(new Headers(init.headers).get('authorization'), 'Bearer synthetic-new-access')
  return Response.json({ id: 'message-one', threadId: 'thread-one', payload: { mimeType: 'text/plain', body: { data: Buffer.from('Synthetic message').toString('base64url') } } })
}) as typeof fetch })
const readers = Array.from({ length: 8 }, () => service.readMessage('account-a', 'message-one'))
await starting
assert.equal(refreshCalls, 1)
resolveRefresh(Response.json({ token_type: 'Bearer', access_token: 'synthetic-new-access', expires_in: 3600 }))
const messages = await Promise.all(readers)
assert.equal(messages.length, 8)
assert.equal(refreshCalls, 1, 'parallel expired requests share one refresh')
assert.equal(readCalls, 8)
assert.deepEqual((await store.read())!.accounts[0].tokens?.scopes, requestedScopes(['gmail.read', 'calendar.read']), 'omitted refresh scopes preserve the previously verified exact grant')
assert.equal((await store.read())!.accounts[0].tokens?.refreshToken, 'synthetic-refresh')

const downgradedStore = memory(initial())
let calendarReads = 0
const downgrade = new GoogleWorkspaceService({ store: downgradedStore, openExternal: async () => {}, lookup, now: () => now, fetchImpl: (async (url: any) => {
  if (String(url).endsWith('/token')) return Response.json({ token_type: 'Bearer', access_token: 'synthetic-new-access', expires_in: 3600, scope: requestedScopes(['gmail.read']).join(' ') })
  calendarReads++; return Response.json({ items: [] })
}) as typeof fetch })
await assert.rejects(downgrade.listCalendars('account-a'), /permission/)
assert.equal(calendarReads, 0, 'lost scopes refuse API access after refresh')
assert.deepEqual((await downgrade.snapshot()).accounts[0].capabilities, ['gmail.read'])
assert.equal((await downgrade.snapshot()).accounts[0].revision, 2, 'grant changes invalidate pending owner reviews')

const revokedStore = memory(initial())
let revokedCalls = 0
const revoked = new GoogleWorkspaceService({ store: revokedStore, openExternal: async () => {}, lookup, now: () => now, fetchImpl: (async () => { revokedCalls++; return Response.json({ error: 'invalid_grant', private: 'must not leak' }, { status: 400 }) }) as typeof fetch })
await assert.rejects(revoked.readMessage('account-a', 'message-one'), error => error instanceof Error && !error.message.includes('must not leak'))
assert.equal(revokedCalls, 1)
assert.equal((await revoked.snapshot()).accounts[0].status, 'needs-reauthorization')
assert.equal((await revokedStore.read())!.accounts[0].tokens, undefined)
await assert.rejects(revoked.readMessage('account-a', 'message-one'), /disconnected/)
assert.equal(revokedCalls, 1)

// An outstanding refresh cannot write tokens back after local disconnect.
let refreshResolve!: (value: Response) => void
let refreshBegin!: () => void
const refreshPending = new Promise<void>(resolve => { refreshBegin = resolve })
const racingStore = memory(initial())
const racing = new GoogleWorkspaceService({ store: racingStore, openExternal: async () => {}, lookup, now: () => now, fetchImpl: (async (url: any) => {
  if (String(url).endsWith('/token')) { refreshBegin(); return new Promise<Response>(resolve => { refreshResolve = resolve }) }
  if (String(url).endsWith('/revoke')) return Response.json({})
  throw new Error('Read must not follow a disconnected refresh')
}) as typeof fetch })
const pendingRead = racing.readMessage('account-a', 'message-one')
const pendingOutcome = assert.rejects(pendingRead, /changed|disconnected/)
await refreshPending
await racing.disconnect('account-a')
refreshResolve(Response.json({ token_type: 'Bearer', access_token: 'synthetic-new-access', expires_in: 3600 }))
await pendingOutcome
assert.equal((await racingStore.read())!.accounts[0].tokens, undefined)
assert.equal((await racing.snapshot()).accounts[0].status, 'disconnected')

const corrupt = new GoogleWorkspaceService({ store: { read: async () => ({ version: 99, accounts: null } as any), write: async () => {} }, openExternal: async () => {} })
await assert.rejects(corrupt.snapshot(), /storage is invalid/)
await assert.rejects(corrupt.listMessages('account-a'), /storage is invalid/)
service.dispose(); downgrade.dispose(); revoked.dispose(); racing.dispose(); corrupt.dispose()
console.log('Google single-flight refresh, grant downgrade, revocation, disconnect race and corrupted storage tests passed')
