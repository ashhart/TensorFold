import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import { GoogleAccounts } from './accounts'
import { GoogleState, type GoogleWorkspacePrivateState, type GoogleWorkspaceStore } from './private-state'
import { GoogleTransport } from './transport'
import { requestedScopes } from './scopes'

const now = 100_000
const lookup = async () => [{ address: '8.8.8.8', family: 4 }]
function initial(): GoogleWorkspacePrivateState {
  return { version: 1, accounts: [{ id: 'synthetic-account', subject: 'synthetic-subject', email: 'person-a@example.com', revision: 1, capabilities: ['gmail.read', 'gmail.send'], status: 'connected', connectedAt: now, tokens: { accessToken: 'synthetic-old-token', refreshToken: 'synthetic-refresh', expiresAt: now + 3_600_000, scopes: requestedScopes(['gmail.read', 'gmail.send']) } }], drafts: [], actions: [] }
}
function memory(value: GoogleWorkspacePrivateState) {
  let nextWriteFails = false
  const store: GoogleWorkspaceStore = {
    read: async () => structuredClone(value),
    write: async (state, validateCommit) => { validateCommit?.(); if (nextWriteFails) { nextWriteFails = false; throw new Error('Synthetic receipt disk failure') }; value = structuredClone(state) },
  }
  return { store, failNextWrite: () => { nextWriteFails = true } }
}

// The HTTP result belongs to revision 1; reconnect creates a separate revision 2 controller.
{
  const fixture = memory(initial())
  let finishOld!: (response: Response) => void; let beginOld!: () => void
  const oldStarted = new Promise<void>(resolve => { beginOld = resolve })
  const transport = new GoogleTransport({ lookup, fetchImpl: (async (_url: any, init: RequestInit) => {
    const token = new Headers(init.headers).get('authorization')
    if (token === 'Bearer synthetic-old-token') { beginOld(); return new Promise<Response>(resolve => { finishOld = resolve }) }
    assert.equal(token, 'Bearer synthetic-new-token')
    return Response.json({ id: 'synthetic-message' })
  }) as typeof fetch })
  const accounts = new GoogleAccounts(new GoogleState(fixture.store), transport, () => now)
  const read = () => accounts.withAccess('synthetic-account', 'gmail.read', (token, _revision, signal) => transport.json('https://gmail.googleapis.com/gmail/v1/users/me/messages/synthetic-message', token, undefined, 'GET', signal))
  const oldRead = read()
  const oldFailure = assert.rejects(oldRead, /expired|changed/)
  await oldStarted
  await accounts.connect('synthetic-subject', 'person-a@example.com', { accessToken: 'synthetic-new-token', refreshToken: 'synthetic-refresh', expiresAt: now + 3_600_000, scopes: requestedScopes(['gmail.read', 'gmail.send']) }, ['gmail.read', 'gmail.send'], { id: 'synthetic-account', revision: 1 })
  finishOld(Response.json({ error: 'invalid_token' }, { status: 401 }))
  await oldFailure
  assert.equal((await accounts.current('synthetic-account')).revision, 2)
  assert.equal((await read()).id, 'synthetic-message', 'a late 401 for the replaced session must not abort the freshly connected session')
  accounts.dispose()
}

// Failing the durable admission receipt means nothing was sent, rather than an uncertain send.
{
  const fixture = memory(initial())
  let sends = 0
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => { sends++; return Response.json({ id: 'synthetic-sent-message' }) }) as typeof fetch })
  const review = await service.prepareAction('synthetic-account', { kind: 'gmail.send', email: { to: ['person-b@example.com'], subject: 'Example', text: 'Synthetic body' } })
  const receipt = { id: review.id, digest: review.digest, accountId: review.accountId, revision: review.revision }
  fixture.failNextWrite()
  await assert.rejects(service.executeReviewed(receipt), /receipt disk failure/)
  assert.equal(sends, 0)
  assert.equal((await service.snapshot()).reviews.length, 1, 'the never-started review remains pending after a failed receipt write')
  assert.equal((await service.executeReviewed(receipt)).status, 'succeeded')
  assert.equal(sends, 1, 'a later explicit approval can execute the previously unsent operation once')
  await service.executeReviewed(receipt)
  assert.equal(sends, 1)
  service.dispose()
}

// A definite authentication rejection stays failed even while the account needs reconnection.
{
  const fixture = memory(initial())
  let sends = 0
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => { sends++; return Response.json({ error: 'invalid_token' }, { status: 401 }) }) as typeof fetch })
  const review = await service.prepareAction('synthetic-account', { kind: 'gmail.send', email: { to: ['person-b@example.com'], subject: 'Example', text: 'Synthetic body' } })
  const receipt = { id: review.id, digest: review.digest, accountId: review.accountId, revision: review.revision }
  const result = await service.executeReviewed(receipt)
  assert.equal((await fixture.store.read())!.actions[0].result?.status, 'failed', 'the definite refusal is already recorded durably')
  assert.equal((await fixture.store.read())!.accounts[0].status, 'needs-reauthorization')
  assert.equal(result.status, 'failed', 'HTTP 401 is a known refusal, not an uncertain remote outcome')
  assert.equal((await service.snapshot()).accounts[0].status, 'needs-reauthorization')
  assert.equal((await service.executeReviewed(receipt)).status, 'failed')
  assert.equal(sends, 1, 'the rejected send is never automatically retried')
  service.dispose()
}

// Losing the final confirmation on disk never permits an automatic second send.
{
  const fixture = memory(initial())
  let sends = 0
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => {
    sends++; fixture.failNextWrite()
    return Response.json({ id: 'synthetic-sent-message' })
  }) as typeof fetch })
  const review = await service.prepareAction('synthetic-account', { kind: 'gmail.send', email: { to: ['person-b@example.com'], subject: 'Example', text: 'Synthetic body' } })
  const receipt = { id: review.id, digest: review.digest, accountId: review.accountId, revision: review.revision }
  assert.equal((await service.executeReviewed(receipt)).status, 'uncertain')
  assert.equal((await service.executeReviewed(receipt)).status, 'uncertain')
  assert.equal(sends, 1)
  assert.equal((await service.snapshot()).recentActions[0].result.status, 'uncertain')
  service.dispose()
}

// Calendar writes preserve literal all-day dates and honor an event's conditional version.
{
  const state = initial()
  state.accounts[0].capabilities = ['calendar.read', 'calendar.write']
  state.accounts[0].tokens!.scopes = requestedScopes(['calendar.read', 'calendar.write'])
  const fixture = memory(state)
  const calls: { url: string; init: RequestInit }[] = []
  let conflict = false; let etag = '"synthetic-version"'
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async (url: any, init: RequestInit) => {
    calls.push({ url: String(url), init })
    if (init.method === 'GET') return Response.json({ id: 'synthetic-event', etag, summary: 'Verified previous event', start: { date: '2028-02-29' }, end: { date: '2028-03-01' } })
    if (init.method === 'PATCH') {
      if (conflict) return Response.json({ error: { code: 412 } }, { status: 412 })
      etag = '"synthetic-next-version"'
      return Response.json({ id: 'synthetic-event', etag })
    }
    return Response.json({ id: 'synthetic-created-event' })
  }) as typeof fetch })
  const event = { title: 'Example all-day event', start: '2028-02-29', end: '2028-03-01', timeZone: 'Europe/London', sendUpdates: 'none' as const }
  const created = await service.prepareAction('synthetic-account', { kind: 'calendar.create', calendarId: 'primary', event })
  assert.equal((await service.executeReviewed({ id: created.id, digest: created.digest, accountId: created.accountId, revision: created.revision })).status, 'succeeded')
  assert.deepEqual(JSON.parse(String(calls[0].init.body)).start, { date: '2028-02-29' })
  assert.deepEqual(JSON.parse(String(calls[0].init.body)).end, { date: '2028-03-01' })
  const edited = await service.prepareAction('synthetic-account', { kind: 'calendar.update', calendarId: 'primary', eventId: 'synthetic-event', etag: '"synthetic-version"', event })
  assert.equal(edited.previousEvent?.title, 'Verified previous event')
  assert.equal((await service.executeReviewed({ id: edited.id, digest: edited.digest, accountId: edited.accountId, revision: edited.revision })).status, 'succeeded')
  assert.deepEqual(JSON.parse(String(calls.at(-1)!.init.body)).start, { date: '2028-02-29' })
  assert.equal(new Headers(calls.at(-1)!.init.headers).get('if-match'), '"synthetic-version"')
  const conflicting = await service.prepareAction('synthetic-account', { kind: 'calendar.update', calendarId: 'primary', eventId: 'synthetic-event', etag, event })
  const receipt = { id: conflicting.id, digest: conflicting.digest, accountId: conflicting.accountId, revision: conflicting.revision }
  conflict = true
  assert.equal((await service.executeReviewed(receipt)).status, 'failed', 'ETag conflicts do not overwrite an event changed elsewhere')
  assert.equal(new Headers(calls.at(-1)!.init.headers).get('if-match'), '"synthetic-next-version"')
  const before = calls.length
  assert.equal((await service.executeReviewed(receipt)).status, 'failed')
  assert.equal(calls.length, before)
  service.dispose()
}

// Thread reads retain every supported plain-text message but never fetch remote attachments.
{
  const fixture = memory(initial())
  let oversized = false; let requests = 0
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async (url: any) => {
    requests++
    assert.ok(String(url).endsWith('/threads/synthetic-thread?format=full'))
    return Response.json({ id: 'synthetic-thread', messages: Array.from({ length: oversized ? 101 : 2 }, (_, index) => ({ id: `synthetic-message-${index}`, threadId: 'synthetic-thread', payload: { mimeType: 'multipart/mixed', parts: [{ mimeType: 'text/plain', body: { data: Buffer.from(`Example plain text ${index}`).toString('base64url') } }, { filename: 'example.pdf', mimeType: 'application/pdf', body: { attachmentId: 'synthetic-attachment', size: 100 } }] } })) })
  }) as typeof fetch })
  const thread = await service.readThread('synthetic-account', 'synthetic-thread')
  assert.equal(thread.messages.length, 2)
  assert.equal(thread.messages[0].text, 'Example plain text 0')
  assert.equal(thread.messages[1].text, 'Example plain text 1')
  assert.ok(thread.messages.every(message => message.untrustedContent && message.attachments[0].filename === 'example.pdf'))
  assert.equal(requests, 1)
  oversized = true
  await assert.rejects(service.readThread('synthetic-account', 'synthetic-thread'), /too large/)
  assert.equal(requests, 2, 'oversized thread refusal makes no attachment or per-message retry requests')
  service.dispose()
}

// A local disconnect must retain a durable warning when remote revocation was not confirmed.
{
  const fixture = memory(initial())
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => { throw new Error('Synthetic offline revocation') }) as typeof fetch })
  const result = await service.disconnect('synthetic-account')
  assert.equal(result.revoked, false)
  assert.equal((await service.snapshot()).accounts[0].status, 'disconnected')
  assert.match((await service.snapshot()).accounts[0].lastError ?? '', /revocation could not be confirmed/)
  const restarted = new GoogleWorkspaceService({ store: fixture.store, now: () => now, openExternal: async () => {} })
  assert.match((await restarted.snapshot()).accounts[0].lastError ?? '', /revocation could not be confirmed/)
  assert.equal((await fixture.store.read())!.accounts[0].tokens, undefined)
  service.dispose(); restarted.dispose()
}

// A warning that could not be persisted must not be reported as a completed disconnect flow.
{
  const fixture = memory(initial())
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => {
    fixture.failNextWrite()
    throw new Error('Synthetic offline revocation')
  }) as typeof fetch })
  await assert.rejects(service.disconnect('synthetic-account'), /receipt disk failure/)
  assert.equal((await fixture.store.read())!.accounts[0].status, 'disconnected')
  assert.equal((await fixture.store.read())!.accounts[0].tokens, undefined, 'a failed warning write does not restore disconnected credentials')
  service.dispose()
}

// Missing credentials cannot establish that the remote grant was revoked.
{
  const state = initial()
  state.accounts[0].status = 'needs-reauthorization'; state.accounts[0].tokens = undefined
  const fixture = memory(state)
  let requests = 0
  const service = new GoogleWorkspaceService({ store: fixture.store, now: () => now, lookup, openExternal: async () => {}, fetchImpl: (async () => { requests++; return Response.json({}) }) as typeof fetch })
  assert.equal((await service.disconnect('synthetic-account')).revoked, false)
  assert.match((await service.snapshot()).accounts[0].lastError ?? '', /revocation could not be confirmed/)
  assert.equal(requests, 0)
  service.dispose()
}

// An older disconnect's late revocation failure cannot stain a newly reconnected account.
{
  const fixture = memory(initial())
  let finishRevocation!: (response: Response) => void; let beginRevocation!: () => void
  const revocationStarted = new Promise<void>(resolve => { beginRevocation = resolve })
  const transport = new GoogleTransport({ lookup, fetchImpl: (async () => { beginRevocation(); return new Promise<Response>(resolve => { finishRevocation = resolve }) }) as typeof fetch })
  const accounts = new GoogleAccounts(new GoogleState(fixture.store), transport, () => now)
  const disconnect = accounts.disconnect('synthetic-account')
  await revocationStarted
  const revision = (await fixture.store.read())!.accounts[0].revision
  await accounts.connect('synthetic-subject', 'person-a@example.com', { accessToken: 'synthetic-new-token', expiresAt: now + 3_600_000, scopes: requestedScopes(['gmail.read']) }, ['gmail.read'], { id: 'synthetic-account', revision })
  finishRevocation(Response.json({ error: 'service_unavailable' }, { status: 503 }))
  assert.equal((await disconnect).revoked, false)
  assert.equal((await accounts.current('synthetic-account')).lastError, undefined)
  assert.equal((await accounts.current('synthetic-account')).tokens?.accessToken, 'synthetic-new-token')
  accounts.dispose()
}
console.log('Google qualification covers stale authorization replies, failed durable admission and definite rejection semantics')
