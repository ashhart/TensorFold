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

// A sign-in that never settles is a failure: the waiting state must clear on every path.
async function failure(promise: Promise<unknown>, timeoutMs = 2_000): Promise<Error> {
  const settled = await Promise.race([
    promise.then(() => null, (error: unknown) => (error instanceof Error ? error : new Error(String(error)))),
    new Promise<null>(resolve => setTimeout(() => resolve(null), timeoutMs)),
  ])
  assert.ok(settled, 'the sign-in settled as failed instead of waiting')
  return settled
}

const loop = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const url = new URL(loop.authorizationUrl)
assert.equal(url.origin, 'https://accounts.google.com')
assert.equal(url.searchParams.get('code_challenge_method'), 'S256')
assert.equal(url.searchParams.get('code_challenge'), createHash('sha256').update(loop.verifier).digest('base64url'))
assert.equal(url.searchParams.has('include_granted_scopes'), false)
assert.equal(url.searchParams.has('client_secret'), false)
assert.equal(new URL(loop.redirectUri).hostname, '127.0.0.1')
assert.notEqual(new URL(loop.redirectUri).port, '0')
// A request that cannot be ours is answered and forgotten: it never settles the sign-in.
assert.equal((await fetch(`${loop.redirectUri}?state=wrong&code=synthetic-code`)).status, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${'é'.repeat(43)}&code=synthetic-code`)).status, 400, 'Unicode state cannot throw in timingSafeEqual')
const state = url.searchParams.get('state')!
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`, { method: 'POST' })).status, 400)
const spoofedHostStatus = await new Promise<number | undefined>((resolve, reject) => {
  const request = httpRequest(`${loop.redirectUri}?state=${state}&code=synthetic-code`, { headers: { Host: 'attacker.example.com' } }, response => { response.resume(); resolve(response.statusCode) })
  request.on('error', reject); request.end()
})
assert.equal(spoofedHostStatus, 400)
assert.equal((await fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`)).status, 200)
assert.equal(await loop.completion, 'synthetic-code')
await assert.rejects(fetch(`${loop.redirectUri}?state=${state}&code=synthetic-code`), /fetch failed/)

// Google's own redirect carries iss (RFC 9207): the discovery document declares
// authorization_response_iss_parameter_supported, so the parameter rides along.
const google = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const googleState = new URL(google.authorizationUrl).searchParams.get('state')!
const googleCallback = new URL(google.redirectUri)
googleCallback.searchParams.set('code', 'synthetic-code')
googleCallback.searchParams.set('state', googleState)
googleCallback.searchParams.set('iss', 'https://accounts.google.com')
googleCallback.searchParams.set('scope', 'https://www.googleapis.com/auth/gmail.readonly')
assert.equal((await fetch(googleCallback)).status, 200)
assert.equal(await google.completion, 'synthetic-code')

// An issuer that is present must be Google's exactly.
const wrongIssuer = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const wrongIssuerUrl = new URL(wrongIssuer.redirectUri)
wrongIssuerUrl.searchParams.set('code', 'synthetic-code')
wrongIssuerUrl.searchParams.set('state', new URL(wrongIssuer.authorizationUrl).searchParams.get('state')!)
wrongIssuerUrl.searchParams.set('iss', 'https://accounts.example.com')
assert.equal((await fetch(wrongIssuerUrl)).status, 400)
assert.match((await failure(wrongIssuer.completion)).message, /issuer/)

// A malformed callback of ours settles as failed and names the problem, never a value.
const extraParameter = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const extraUrl = new URL(extraParameter.redirectUri)
extraUrl.searchParams.set('code', 'synthetic-code')
extraUrl.searchParams.set('state', new URL(extraParameter.authorizationUrl).searchParams.get('state')!)
extraUrl.searchParams.set('target', 'https://evil.example.com')
const logged: string[] = []
const originalLog = console.log
const originalError = console.error
console.log = (...parts: unknown[]) => { logged.push(parts.map(String).join(' ')) }
console.error = (...parts: unknown[]) => { logged.push(parts.map(String).join(' ')) }
assert.equal((await fetch(extraUrl)).status, 400)
console.log = originalLog
console.error = originalError
const extraFailure = await failure(extraParameter.completion)
assert.match(extraFailure.message, /target/)
assert.ok(!extraFailure.message.includes('evil.example.com'), 'the message names the parameter, never its value')
assert.ok(!extraFailure.message.includes('synthetic-code'), 'the message carries no authorization code')
assert.ok(!extraFailure.message.includes(extraParameter.verifier), 'the message carries no verifier')
assert.deepEqual(logged, [], 'a failed callback writes no log line')

// A parameter name that is not short and plain is described without it.
const longName = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const longUrl = new URL(longName.redirectUri)
longUrl.searchParams.set('state', new URL(longName.authorizationUrl).searchParams.get('state')!)
longUrl.searchParams.set('code', 'synthetic-code')
longUrl.searchParams.set('x'.repeat(300), 'synthetic-value')
assert.equal((await fetch(longUrl)).status, 400)
const longFailure = await failure(longName.completion)
assert.match(longFailure.message, /an unexpected parameter/)
assert.ok(!longFailure.message.includes('x'.repeat(300)), 'a long parameter name is not shown')
assert.ok(!/x{20,}/.test(longFailure.message), 'a long parameter name leaves no trace in the message')

// A name with a control character is described without it too.
const controlName = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const controlState = new URL(controlName.authorizationUrl).searchParams.get('state')!
assert.equal((await fetch(`${controlName.redirectUri}?state=${controlState}&code=synthetic-code&bad%0Aname=synthetic-value`)).status, 400)
const controlFailure = await failure(controlName.completion)
assert.match(controlFailure.message, /an unexpected parameter/)
assert.ok(!controlFailure.message.includes('bad'), 'a control character in the name is not shown')
assert.ok(!/[\u0000-\u001f\u007f]/.test(controlFailure.message), 'no control character reaches the message')

// The repeated parameter's name is bounded the same way.
const repeatedControl = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const repeatedControlState = new URL(repeatedControl.authorizationUrl).searchParams.get('state')!
assert.equal((await fetch(`${repeatedControl.redirectUri}?state=${repeatedControlState}&state=${repeatedControlState}&bad%00name=synthetic-value`)).status, 400)
const repeatedControlFailure = await failure(repeatedControl.completion)
assert.match(repeatedControlFailure.message, /an unexpected parameter/)
assert.ok(!repeatedControlFailure.message.includes('bad'), 'a control character in a repeated name is not shown')
assert.ok(!/[\u0000-\u001f\u007f]/.test(repeatedControlFailure.message), 'no control character reaches the repeated-parameter message')

// A repeated parameter is malformed too.
const repeated = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const repeatedState = new URL(repeated.authorizationUrl).searchParams.get('state')!
assert.equal((await fetch(`${repeated.redirectUri}?state=${repeatedState}&state=${repeatedState}&code=synthetic-code`)).status, 400)
assert.match((await failure(repeated.completion)).message, /state/)

// A callback with both a code and an error is malformed.
const both = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const bothUrl = new URL(both.redirectUri)
bothUrl.searchParams.set('state', new URL(both.authorizationUrl).searchParams.get('state')!)
bothUrl.searchParams.set('code', 'synthetic-code')
bothUrl.searchParams.set('error', '')
assert.equal((await fetch(bothUrl)).status, 400)
assert.match((await failure(both.completion)).message, /neither a usable authorization code nor an error/)

// A decline says what to check.
const declined = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const declinedUrl = new URL(declined.redirectUri)
declinedUrl.searchParams.set('state', new URL(declined.authorizationUrl).searchParams.get('state')!)
declinedUrl.searchParams.set('error', 'access_denied')
declinedUrl.searchParams.set('error_description', 'Access blocked by the administrator')
assert.equal((await fetch(declinedUrl)).status, 200)
const declinedFailure = await failure(declined.completion)
assert.match(declinedFailure.message, /declined/)
assert.match(declinedFailure.message, /test user/)

// A stranger's request never settles our sign-in.
const stranger = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
const strangerState = new URL(stranger.authorizationUrl).searchParams.get('state')!
assert.equal((await fetch(`${stranger.redirectUri}?code=synthetic-code`)).status, 400, 'a callback without a state is not ours')
assert.equal((await fetch(`${stranger.redirectUri}/other?state=${strangerState}&code=synthetic-code`)).status, 400, 'another path is not ours')
let strangerSettled = false
void stranger.completion.then(() => { strangerSettled = true }, () => { strangerSettled = true })
await new Promise(resolve => setTimeout(resolve, 50))
assert.equal(strangerSettled, false, 'a stranger never settles the sign-in')
stranger.cancel()
await assert.rejects(stranger.completion, /cancelled/)

const cancelled = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 5_000)
cancelled.cancel()
assert.match((await failure(cancelled.completion)).message, /cancelled/)
const timeout = await createGoogleOAuthLoopback(client, requestedScopes(['gmail.read']), 20)
assert.match((await failure(timeout.completion)).message, /timed out/)

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
// A callback that cannot be ours settles the sign-in as failed, says why, and never leaks a secret.
const failing = new GoogleWorkspaceService({ store: memory(baseline), lookup, now: () => now, openExternal: async url => { opened = url }, fetchImpl: synthetic })
await failing.configureClient(clientJson)
const failingAuth = await failing.startAuthorization({ capabilities: ['gmail.read'] })
const failingBrowser = new URL(opened)
const failingState = failingBrowser.searchParams.get('state')!
const failingRedirect = failingBrowser.searchParams.get('redirect_uri')!
await fetch(`${failingRedirect}?state=${failingState}&code=synthetic-code&iss=https://accounts.example.com`)
for (let i = 0; i < 100 && (await failing.snapshot()).authorization?.status === 'waiting'; i++) await new Promise(resolve => setTimeout(resolve, 5))
const failed = await failing.snapshot()
assert.equal(failed.authorization?.id, failingAuth.id)
assert.equal(failed.authorization?.status, 'failed', 'a callback that cannot be ours ends the waiting state')
assert.match(failed.authorization?.error ?? '', /issuer/)
for (const secret of ['synthetic-code', 'synthetic-secret', 'synthetic-access', 'synthetic-refresh']) {
  assert.ok(!(failed.authorization?.error ?? '').includes(secret), 'a failed sign-in names no secret')
  assert.ok(!JSON.stringify(failed).includes(secret), 'a failed sign-in snapshot carries no secret')
}
// A retry after a failure starts a new sign-in and succeeds.
const retryAuth = await failing.startAuthorization({ capabilities: ['gmail.read'] })
assert.notEqual(retryAuth.id, failingAuth.id)
const retryBrowser = new URL(opened)
await fetch(`${retryBrowser.searchParams.get('redirect_uri')}?state=${retryBrowser.searchParams.get('state')}&code=synthetic-code&iss=https://accounts.google.com`)
for (let i = 0; i < 100 && (await failing.snapshot()).authorization?.status === 'waiting'; i++) await new Promise(resolve => setTimeout(resolve, 5))
const retried = await failing.snapshot()
assert.equal(retried.authorization?.id, retryAuth.id)
assert.equal(retried.authorization?.status, 'complete', 'a retry after a failure signs in again')
failing.dispose()
racing.dispose(); service.dispose()
console.log('Google desktop OAuth PKCE/state/loopback/configuration/scope and disconnect race tests passed')
