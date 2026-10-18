import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import { emailMime, googleEventBody, validatedEmail, validatedEvent } from './action-validation'
import { requestedScopes } from './scopes'
import type { GoogleWorkspacePrivateState, GoogleWorkspaceStore } from './private-state'
import type { GoogleWorkspaceAction, GoogleWorkspaceActionReview } from '../../../shared/google-workspace'

let now = 100_000
const lookup = async () => [{ address: '8.8.8.8', family: 4 }]
let state: GoogleWorkspacePrivateState = { version: 1, accounts: [{ id: 'account-a', subject: 'synthetic_subject_a', email: 'person-a@example.com', revision: 1, status: 'connected', capabilities: ['gmail.read', 'gmail.send', 'calendar.read', 'calendar.write'], connectedAt: now, tokens: { accessToken: 'synthetic-access', refreshToken: 'synthetic-refresh', expiresAt: now + 3_600_000, scopes: requestedScopes(['gmail.read', 'gmail.send', 'calendar.read', 'calendar.write']) } }], drafts: [], actions: [] }
const store: GoogleWorkspaceStore = { read: async () => structuredClone(state), write: async next => { state = structuredClone(next) } }
const calls: { url: string; init: RequestInit }[] = []
let failNext = false
const fakeFetch = (async (url: any, init: RequestInit) => {
  calls.push({ url: String(url), init })
  if (failNext) { failNext = false; throw new Error('Synthetic lost connection') }
  if (init.method === 'GET') return Response.json({ id: 'event-one', etag: '"version-one"', summary: 'Actual server title', start: { dateTime: '2026-10-16T10:00:00Z' }, end: { dateTime: '2026-10-16T11:00:00Z' }, attendees: [{ email: 'person-b@example.com' }] })
  if (init.method === 'DELETE') return new Response(null, { status: 204 })
  return Response.json({ id: 'synthetic-remote-id' })
}) as typeof fetch
const service = new GoogleWorkspaceService({ store, openExternal: async () => {}, fetchImpl: fakeFetch, lookup, now: () => now })
const email = { to: ['person-b@example.com'], cc: ['person-c@example.com'], bcc: ['person-d@example.com'], subject: 'Synthetic subject', text: 'Exact reviewed body' }
const action: GoogleWorkspaceAction = { kind: 'gmail.send', email }
const receipt = (review: GoogleWorkspaceActionReview) => ({ id: review.id, accountId: review.accountId, revision: review.revision, digest: review.digest })
const review = await service.prepareAction('account-a', action)
assert.equal(calls.length, 0, 'preparing an email never sends anything')
assert.deepEqual(review.action, action)
assert.equal((await service.snapshot()).reviews[0].id, review.id)
assert.deepEqual(await service.getReview(receipt(review)), review)
review.action.kind === 'gmail.send' && (review.action.email.text = 'Renderer tampering')
assert.equal((await service.getReview(receipt(review))).action.kind, 'gmail.send')
assert.equal(((await service.getReview(receipt(review))).action as typeof action).email.text, 'Exact reviewed body')
await assert.rejects(service.executeReviewed({ ...receipt(review), digest: 'a'.repeat(64) }), /no longer matches/)
await assert.rejects(service.executeReviewed({ ...receipt(review), accountId: 'account-b' }), /no longer matches/)
assert.equal(calls.length, 0)
const outcomes = await Promise.all([service.executeReviewed(receipt(review)), service.executeReviewed(receipt(review))])
assert.equal(calls.length, 1, 'parallel approvals cannot send twice')
assert.ok(outcomes.some(result => result.status === 'succeeded'))
const mime = Buffer.from(JSON.parse(String(calls[0].init.body)).raw, 'base64url').toString('utf8')
assert.ok(mime.includes('To: person-b@example.com'))
assert.ok(mime.includes('Cc: person-c@example.com'))
assert.ok(mime.includes('Bcc: person-d@example.com'))
assert.ok(mime.includes('From: person-a@example.com'))
assert.ok(!mime.includes('Renderer tampering'))
assert.ok(mime.includes(Buffer.from('Exact reviewed body').toString('base64')))
assert.equal((await service.snapshot()).reviews.length, 0)
await service.executeReviewed(receipt(review))
assert.equal(calls.length, 1)

const uncertain = await service.prepareAction('account-a', action)
failNext = true
assert.equal((await service.executeReviewed(receipt(uncertain))).status, 'uncertain')
const afterUncertain = calls.length
const restarted = new GoogleWorkspaceService({ store, openExternal: async () => {}, fetchImpl: fakeFetch, lookup, now: () => now })
assert.equal((await restarted.executeReviewed(receipt(uncertain))).status, 'uncertain')
assert.equal(calls.length, afterUncertain, 'an uncertain send survives restart and is never automatically retried')
assert.equal((await restarted.snapshot()).recentActions[0].result.status, 'uncertain', 'uncertain outcomes stay visible after restart')
const cancelled = await service.prepareAction('account-a', action)
assert.equal((await service.cancelAction(receipt(cancelled))).status, 'failed')
assert.match((await service.executeReviewed(receipt(cancelled))).error!, /Cancelled by you/)
assert.equal(calls.length, afterUncertain, 'cancelling a persisted review prevents later execution')
const expired = await service.prepareAction('account-a', action)
now += 11 * 60_000
await assert.rejects(service.executeReviewed(receipt(expired)), /expired/)
assert.equal((await service.snapshot()).reviews.length, 0)
const stale = await service.prepareAction('account-a', action)
state.accounts[0].revision++
await assert.rejects(service.executeReviewed(receipt(stale)), /changed/)
await assert.rejects(service.getReview(receipt(stale)), /expired|changed/)

assert.throws(() => validatedEmail({ ...email, subject: 'Subject\r\nBcc: hidden@example.com' }), /line breaks/)
assert.throws(() => validatedEmail({ ...email, to: ['person-b@example.com\r\nBcc: hidden@example.com'] }), /email address/)
assert.throws(() => validatedEmail({ ...email, to: ['Display Name <person-b@example.com>'] }), /plain email/)
const unicodeMime = Buffer.from(emailMime(validatedEmail({ ...email, subject: 'こんにちは'.repeat(30) }), 'person-a@example.com'), 'base64url').toString('utf8')
assert.ok(unicodeMime.split('\r\n').filter(line => line.startsWith('Subject:') || line.startsWith(' ')).every(line => line.length < 80))
const event = { title: 'Example event', start: '2026-10-16T10:00:00Z', end: '2026-10-16T11:00:00Z', timeZone: 'Europe/London', attendees: ['person-b@example.com'], sendUpdates: 'all' as const }
assert.throws(() => validatedEvent({ ...event, timeZone: 'Invalid/Zone' }), /time zone/)
assert.throws(() => validatedEvent({ ...event, sendUpdates: undefined as any }), /notify/)
assert.throws(() => validatedEvent({ ...event, end: event.start }), /positive/)
assert.throws(() => validatedEvent({ ...event, start: '2026-02-30T10:00:00Z' }), /valid calendar date/)
assert.throws(() => validatedEvent({ ...event, start: '2026-02-29', end: '2026-03-01' }), /valid all-day/)
assert.throws(() => validatedEvent({ ...event, start: '2028-02-29', end: '2028-02-29' }), /exclusive/)
assert.throws(() => validatedEvent({ ...event, start: '2028-02-29', end: '2028-03-01T00:00:00Z' }), /mixture/)
const allDay = validatedEvent({ ...event, start: '2028-02-29', end: '2028-03-01' })
assert.deepEqual(googleEventBody(allDay).start, { date: '2028-02-29' })
assert.deepEqual(googleEventBody(allDay).end, { date: '2028-03-01' })
assert.equal(allDay.timeZone, 'Europe/London', 'all-day timezone is validated but never used to shift literal dates')
assert.throws(() => validatedEvent({ ...allDay, timeZone: 'Invalid/Zone' }), /time zone/)
const calendarReview = await service.prepareAction('account-a', { kind: 'calendar.create', calendarId: 'primary', event })
await service.executeReviewed(receipt(calendarReview))
const creation = calls.at(-1)!
assert.equal(new URL(creation.url).searchParams.get('sendUpdates'), 'all')
assert.deepEqual(JSON.parse(String(creation.init.body)).attendees, [{ email: 'person-b@example.com' }])
assert.equal(JSON.parse(String(creation.init.body)).start.timeZone, 'Europe/London')
const deletion = await service.prepareAction('account-a', { kind: 'calendar.delete', calendarId: 'primary', eventId: 'event-one', etag: '"version-one"', title: 'Misleading model title', sendUpdates: 'none' })
assert.equal(deletion.previousEvent?.title, 'Actual server title')
assert.deepEqual(deletion.previousEvent?.attendees, ['person-b@example.com'])
assert.equal(deletion.action.kind === 'calendar.delete' && deletion.action.title, 'Actual server title')
assert.equal((await service.executeReviewed(receipt(deletion))).status, 'succeeded')
assert.equal(calls.at(-1)!.init.method, 'DELETE')
assert.equal(new Headers(calls.at(-1)!.init.headers).get('if-match'), '"version-one"')
await assert.rejects(service.prepareAction('account-a', { kind: 'calendar.update', calendarId: 'primary', eventId: 'event-one', etag: '"old-version"', event }), /changed/)
const beforePrune = calls.length
const completed = state.actions.find(item => item.review.id === review.id)!
const retainedUncertainId = uncertain.id
while (state.actions.length < 200) state.actions.push({ review: { ...structuredClone(completed.review), id: `synthetic-old-${state.actions.length}` }, result: { id: `synthetic-old-${state.actions.length}`, status: 'succeeded' } })
await service.prepareAction('account-a', action)
assert.ok(state.actions.some(item => item.review.id === retainedUncertainId), 'uncertain receipts are not pruned')
assert.ok(!state.actions.some(item => item.review.id === review.id), 'old definitive receipts can be pruned')
await assert.rejects(service.executeReviewed(receipt(review)), /no longer matches/)
assert.equal(calls.length, beforePrune, 'missing pruned receipts cannot recreate a network action')
service.dispose(); restarted.dispose()
console.log('Google exact owner review, at-most-once send, restart uncertainty, recipient and conditional calendar mutation tests passed')
