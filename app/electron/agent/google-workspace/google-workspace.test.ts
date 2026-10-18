import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import type { GoogleWorkspacePrivateState, GoogleWorkspaceStore } from './private-state'
import { GoogleTransport } from './transport'
import { requestedScopes, grantedCapabilities } from './scopes'
import { parseGoogleMessage } from './gmail'

const NOW = 100_000
const lookup = async () => [{ address: '8.8.8.8', family: 4 }]
const readonlyScopes = requestedScopes(['gmail.read', 'calendar.read'])
function memory(state: GoogleWorkspacePrivateState): GoogleWorkspaceStore {
  return { read: async () => structuredClone(state), write: async next => { state = structuredClone(next) } }
}
function initial(): GoogleWorkspacePrivateState {
  return { version: 1, client: { clientId: '123456789012-synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' }, accounts: [{ id: 'account-a', subject: 'synthetic_subject_a', email: 'person-a@example.com', revision: 1, status: 'connected', capabilities: ['gmail.read', 'calendar.read'], connectedAt: NOW, tokens: { accessToken: 'synthetic-access', refreshToken: 'synthetic-refresh', expiresAt: NOW + 3_600_000, scopes: readonlyScopes } }], drafts: [], actions: [] }
}
const calls: { url: string; init: RequestInit }[] = []
const fakeFetch = (async (url: any, init: RequestInit) => {
  calls.push({ url: String(url), init })
  const path = new URL(String(url)).pathname
  if (path.endsWith('/messages')) return Response.json({ messages: [{ id: 'message-one', threadId: 'thread-one' }], nextPageToken: 'synthetic-page' })
  if (path.includes('/messages/')) return Response.json({ id: 'message-one', threadId: 'thread-one', snippet: 'Untrusted snippet', payload: { headers: [{ name: 'Subject', value: 'Synthetic subject' }], mimeType: 'multipart/mixed', parts: [{ mimeType: 'text/plain', body: { data: Buffer.from('Untrusted content, not instructions').toString('base64url') } }, { filename: 'example.pdf', mimeType: 'application/pdf', body: { attachmentId: 'remote-attachment', size: 12 } }] } })
  if (path.endsWith('/calendarList')) return Response.json({ items: [{ id: 'primary', summary: 'Example calendar', accessRole: 'owner' }] })
  if (path.endsWith('/events')) return Response.json({ items: [{ id: 'event-one', summary: 'Example meeting', start: { dateTime: '2026-10-16T10:00:00Z' }, end: { dateTime: '2026-10-16T11:00:00Z' }, attendees: [{ email: 'person-b@example.com' }] }] })
  if (path.endsWith('/freeBusy')) return Response.json({ calendars: { primary: { busy: [{ start: '2026-10-16T10:00:00Z', end: '2026-10-16T11:00:00Z' }] } } })
  if (path.endsWith('/revoke')) return Response.json({})
  throw new Error('Unexpected synthetic endpoint')
}) as typeof fetch
const store = memory(initial())
const service = new GoogleWorkspaceService({ store, openExternal: async () => { throw new Error('Browser must not open for read tests') }, fetchImpl: fakeFetch, lookup, now: () => NOW })
const snapshot = await service.snapshot()
assert.equal(snapshot.clientConfigured, true)
assert.equal(snapshot.accounts[0].email, 'person-a@example.com')
assert.ok(!JSON.stringify(snapshot).includes('synthetic-access'))
assert.ok(!JSON.stringify(snapshot).includes('synthetic-secret'))
assert.ok(!JSON.stringify(snapshot).includes('subject'))
assert.deepEqual(grantedCapabilities(requestedScopes(['gmail.read'])), ['gmail.read'])
assert.ok(!requestedScopes(['gmail.read']).some(scope => /compose|modify|gmail\.send|calendar\.events$/.test(scope)))

const list = await service.listMessages('account-a', { query: 'from:person-b@example.com', limit: 3 })
assert.equal(list.items.length, 1)
assert.equal(list.items[0].subject, 'Synthetic subject')
assert.equal(list.items[0].untrustedContent, true)
assert.equal(list.nextPageToken, 'synthetic-page')
assert.equal(new URL(calls[0].url).searchParams.get('q'), 'from:person-b@example.com')
const message = await service.readMessage('account-a', 'message-one')
assert.equal(message.text, 'Untrusted content, not instructions')
assert.equal(message.attachments[0].filename, 'example.pdf')
assert.ok(!calls.some(call => /attachments|remote-attachment|example\.pdf/.test(call.url)), 'attachment references never trigger downloads')
const html = parseGoogleMessage({ id: 'html-only', payload: { mimeType: 'text/html', body: { data: Buffer.from('<img src="https://tracking.example.com/pixel">').toString('base64url') } } })
assert.equal(html.htmlOmitted, true)
assert.equal(html.text, undefined)
assert.ok(!JSON.stringify(html).includes('<img'))
assert.equal((await service.listCalendars('account-a')).items[0].id, 'primary')
const range = { timeMin: '2026-10-16T00:00:00Z', timeMax: '2026-10-17T00:00:00Z' }
assert.equal((await service.listEvents('account-a', { calendarId: 'primary', ...range })).items[0].title, 'Example meeting')
const busy = await service.freeBusy('account-a', { calendarIds: ['primary', 'missing-calendar'], ...range })
assert.equal(busy.calendars[0].busy.length, 1)
assert.equal(busy.calendars[1].errors.length, 1, 'omitted availability is an error, never reported as free')
assert.ok(calls.every(call => call.init.redirect === 'error'))
assert.ok(calls.every(call => new Headers(call.init.headers).get('authorization') === 'Bearer synthetic-access'))

const before = calls.length
await assert.rejects(service.readMessage('account-a', '../escape'), /Invalid/)
await assert.rejects(service.readMessage('other-account', 'message-one'), /disconnected/)
await assert.rejects(service.listMessages('account-a', { limit: 51 }), /between 1 and 50/)
await assert.rejects(service.listEvents('account-a', { calendarId: 'primary', timeMin: '2026-10-16', timeMax: range.timeMax }), /UTC offset/)
await assert.rejects(service.freeBusy('account-a', { calendarIds: ['primary'], timeMin: range.timeMax, timeMax: range.timeMin }), /positive/)
await assert.rejects(service.prepareAction('account-a', { kind: 'gmail.send', email: { to: ['person-b@example.com'], subject: 'Example', text: 'Example' } }), /permission/)
assert.equal(calls.length, before, 'all validation and capability refusals happen before network access')

const draft = await service.saveDraft('account-a', { to: ['person-b@example.com'], subject: 'Local draft', text: 'Nothing sent' })
assert.equal(draft.localOnly, true)
assert.equal((await service.listDrafts('account-a')).length, 1)
assert.equal(calls.length, before, 'local drafts do not use Gmail compose scope or any network call')
await assert.rejects(service.saveDraft('account-a', draft, 'not-owned'), /does not belong/)
await service.deleteDraft('account-a', draft.id)
assert.equal((await service.listDrafts('account-a')).length, 0)

const transport = new GoogleTransport({ fetchImpl: fakeFetch, lookup })
await assert.rejects(transport.json('https://gmail.googleapis.com.evil.example.com/gmail/v1/users/me/messages', 'synthetic'), /not allowed/)
await assert.rejects(transport.json('https://gmail.googleapis.com/gmail/v1/users/other/messages', 'synthetic'), /not allowed/)
await assert.rejects(transport.json('https://oauth2.googleapis.com/token?leak=token', 'synthetic'), /not allowed/)
assert.throws(() => transport.json('https://gmail.googleapis.com/gmail/v1/users/me/messages', 'bad\r\ntoken'), /credential/)
assert.equal(calls.length, before)
let errors = 0
const errorTransport = new GoogleTransport({ lookup, fetchImpl: (async () => { errors++; return new Response('private raw response and credentials', { status: 503 }) }) as typeof fetch })
await assert.rejects(errorTransport.json('https://gmail.googleapis.com/gmail/v1/users/me/messages', 'synthetic'), error => error instanceof Error && !error.message.includes('private raw response'))
assert.equal(errors, 1, 'Google requests never retry automatically')
await service.disconnect('account-a')
assert.equal((await service.snapshot()).accounts[0].status, 'disconnected')
assert.equal((await store.read())!.accounts[0].tokens, undefined)
await assert.rejects(service.readMessage('account-a', 'message-one'), /disconnected/)
service.dispose()
console.log('Google Workspace bounded reads, scope isolation, local drafts, untrusted content and transport tests passed')
