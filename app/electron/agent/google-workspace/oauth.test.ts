import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { request as httpRequest } from 'node:http'
import { GoogleWorkspaceService } from './index'
import { createGoogleOAuthLoopback } from './oauth-loopback'
import { parseGoogleDesktopClient } from './client-config'
import { requestedScopes } from './scopes'
import { parseGoogleTokens } from './oauth-tokens'
import type { GoogleWorkspacePrivateState, GoogleWorkspaceStore } from './private-state'

const client = { clientId: '123456789012-synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' }
const clientJson = JSON.stringify({ installed: { client_id: client.clientId, client_secret: client.clientSecret, auth_uri: 'https://accounts.google.com/o/oauth2/auth', token_uri: 'https://oauth2.googleapis.com/token', redirect_uris: ['http://localhost'] } })
assert.deepEqual(parseGoogleDesktopClient(clientJson), client)
assert.throws(() => parseGoogleDesktopClient(JSON.stringify({ web: {} })), /Desktop/)
assert.throws(() => parseGoogleDesktopClient(clientJson.replace('oauth2.googleapis.com', 'tokens.example.com')), /endpoint/)
assert.throws(() => parseGoogleDesktopClient(clientJson.replace('accounts.google.com', 'accounts.example.com')), /endpoint/)
assert.throws(() => parseGoogleDesktopClient(clientJson.replace('http://localhost', 'https://callback.example.com')), /callback/)
assert.throws(() => parseGoogleDesktopClient(JSON.stringify({ installed: { client_id: client.clientId, auth_uri: 'https://accounts.google.com/o/oauth2/auth', token_uri: 'https://oauth2.googleapis.com/token' } })), /client secret/)
assert.throws(() => parseGoogleDesktopClient({ clientId: client.clientId } as any), /JSON/)
assert.throws(() => parseGoogleTokens({ token_type: 'Bearer', access_token: 'synthetic', expires_in: 3600 }, 0), /permissions/)

const loop = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const url = new URL(loop.authorizationUrl)
assert.equal(url.origin, 'https://accounts.google.com')
assert.equal(url.searchParams.get('code_challenge_method'), 'S256')
assert.equal(url.searchParams.get('code_challenge'), createHash('sha256').update(loop.verifier).digest('base64url'))
assert.equal(url.searchParams.has('include_granted_scopes'), false)
assert.equal(url.searchParams.has('client_secret'), false)
assert.equal(new URL(loop.redirectUri).hostname, '127.0.0.1')
assert.notEqual(new URL(loop.redirectUri).port, '0')
assert.equal((await fetch(`${loop.redirectUri}?state=wrong&code=synthetic-code`)).status, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${'é'.repeat(43)}&code=synthetic-code`)).status, 400, 'Unicode state cannot throw in timingSafeEqual')
const state = url.searchParams.get('state')!
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&state=${state}&code=synthetic-code`)).status, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code&error=`)).status, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`, { method: 'POST' })).status, 400)
const spoofedHostStatus = await new Promise<number | undefined>((resolve, reject) => {
  const request = httpRequest(`${loop.redirectUri}?state=${state}&code=synthetic-code`, { headers: { Host: 'attacker.example.com' } }, response => { response.resume(); resolve(response.statusCode) })
  request.on('error', reject); request.end()
})
assert.equal(spoofedHostStatus, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code&target=https://evil.example.com`)).status, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`)).status, 200)
assert.equal(await loop.completion, 'synthetic-code')
await assert.rejects(fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`), /fetch failed/)
const cancelled = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
cancelled.cancel()
await assert.rejects(cancelled.completion, /cancelled/)
const timeout = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 20)
await assert.rejects(timeout.completion, /timed out/)

function memory(initial: GoogleWorkspacePrivateState): GoogleWorkspaceStore {
  let value = structuredClone(initial)
  return { read: async () => structuredClone(value), write: async next => { value = structuredClone(next) } }
}
const lookup = async () => [{ address: '8.8.8.8', family: 4 }]
const now = 100_000
const baseline: GoogleWorkspacePrivateState = { version: 1, accounts: [], drafts: [], actions: [] }
const calls: { url: string; init: RequestInit }[] = []
const scope = requestedScopes(['gmail.read']).join(' ')
let opened = ''
const synthetic = (async (url: any, init: RequestInit) => {
  calls.push({ url: String(url), init })
  if (String(url).endsWith('/token')) return Response.json({ token_type: 'Bearer', access_token: 'synthetic-access', refresh_token: 'synthetic-refresh', expires_in: 3600, scope })
  if (String(url).endsWith('/userinfo')) return Response.json({ sub: 'synthetic_subject_a', email: 'person-a@example.com', email_verified: true })
  if (String(url).endsWith('/revoke')) return Response.json({})
  throw new Error('Unexpected synthetic request')
}) as typeof fetch
const store = memory(baseline)
const service = new GoogleWorkspaceService({ store, fetchImpl: synthetic, lookup, now: () => now, openExternal: async url => { opened = url } })
await service.configureClient(clientJson)
const auth = await service.startAuthorization({ capabilities: ['gmail.read'] })
await assert.rejects(service.startAuthorization({ capabilities: ['calendar.read'] }), /in progress/)
const browser = new URL(opened)
assert.equal(browser.searchParams.has('client_secret'), false)
await fetch(`${browser.searchParams.get('redirect_uri')}?state=${browser.searchParams.get('state')}&code=synthetic-code`)
for (let i = 0; i < 100 && (await service.snapshot()).authorization?.status === 'waiting'; i++) await new Promise(resolve => setTimeout(resolve, 5))
const result = await service.snapshot()
assert.equal(result.authorization?.id, auth.id)
assert.equal(result.authorization?.status, 'complete')
assert.deepEqual(result.accounts[0].capabilities, ['gmail.read'])
assert.equal(new URLSearchParams(String(calls[0].init.body)).get('code_verifier')?.length, 64)
assert.equal(new URLSearchParams(String(calls[0].init.body)).get('client_secret'), 'synthetic-secret')
assert.ok(!JSON.stringify(result).includes('synthetic-refresh'))
const accountId = result.accounts[0].id

// A reconnect response arriving after disconnect cannot resurrect the account.
let resolveIdentity!: (value: Response) => void
let identityStarted!: () => void
const identityPending = new Promise<void>(resolve => { identityStarted = resolve })
const racing = new GoogleWorkspaceService({ store, lookup, now: () => now, openExternal: async url => { opened = url }, fetchImpl: (async (url: any, init: RequestInit) => {
  if (String(url).endsWith('/userinfo')) { identityStarted(); return new Promise<Response>(resolve => { resolveIdentity = resolve }) }
  return synthetic(url, init)
}) as typeof fetch })
await racing.startAuthorization({ accountId, capabilities: ['gmail.read', 'calendar.read'] })
const racingBrowser = new URL(opened)
assert.ok(racingBrowser.searchParams.get('scope')?.includes('calendar.events.readonly'), 'desktop reauthorization explicitly requests the full desired scope set')
await fetch(`${racingBrowser.searchParams.get('redirect_uri')}?state=${racingBrowser.searchParams.get('state')}&code=synthetic-code`)
await identityPending
await racing.disconnect(accountId)
resolveIdentity(Response.json({ sub: 'synthetic_subject_a', email: 'person-a@example.com', email_verified: true }))
await new Promise(resolve => setTimeout(resolve, 15))
assert.equal((await racing.snapshot()).accounts[0].status, 'disconnected')
assert.equal((await store.read())!.accounts[0].tokens, undefined)
assert.equal((await racing.snapshot()).authorization?.status, 'cancelled')
racing.dispose(); service.dispose()
console.log('Google desktop OAuth PKCE/state/loopback/configuration/scope and disconnect race tests passed')
