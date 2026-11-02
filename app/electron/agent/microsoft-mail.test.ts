import assert from 'node:assert/strict'
import { MicrosoftMailService, type MicrosoftMailState } from './microsoft-mail'
import { setMicrosoftMailRuntime } from './email-provider-runtime'
import { readEmailMessage, scanEmailMessages } from './email-connector'

let now = 1_800_000_000_000
let state: MicrosoftMailState | null = null
const requests: { url: URL; init: RequestInit; body: any }[] = []
let tokenError = ''
let sourceAllowed = true
let sendStatus = 202
let continuation = ''
let delayedToken: (() => Promise<void>) | null = null
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
const fetchImpl: typeof fetch = async (input, init) => {
  const url = new URL(String(input))
  const body = typeof init?.body === 'string' ? (String(init.headers && (init.headers as any)['Content-Type']).includes('json') ? JSON.parse(init.body) : Object.fromEntries(new URLSearchParams(init.body))) : null
  requests.push({ url, init: init ?? {}, body })
  assert.equal(init?.redirect, 'error')
  if (url.pathname.endsWith('/devicecode')) return json({ device_code: 'private-device-code', user_code: 'ABCDEF', expires_in: 900, interval: 5, verification_uri: 'https://attacker.example.com/' })
  if (url.pathname.endsWith('/token')) {
    await delayedToken?.()
    if (tokenError) return json({ error: tokenError, error_description: 'PRIVATE_ERROR_CANARY' }, 400)
    return json({ access_token: 'private-access-token', refresh_token: 'private-refresh-token', expires_in: 3600, scope: 'User.Read Mail.Read Mail.Send' })
  }
  if (url.pathname === '/v1.0/me') return json({ id: 'fixture-microsoft-user', mail: 'account@example.com' })
  if (init?.method === 'POST') return sendStatus === 202 ? new Response(null, { status: 202 }) : json({ message: 'PRIVATE_ERROR_CANARY' }, sendStatus)
  const message = { id: 'fixture-message', subject: 'Synthetic mail', from: { emailAddress: { address: 'sender@example.com' } }, toRecipients: [{ emailAddress: { address: 'account@example.com' } }], receivedDateTime: '2026-10-28T10:00:00Z', isRead: false, bodyPreview: 'Preview', body: { contentType: 'Text', content: 'Exact full body' }, internetMessageId: '<fixture@example.com>' }
  return json(url.pathname.endsWith('/messages') ? { value: [message], ...(continuation ? { '@odata.nextLink': continuation } : {}) } : message)
}
const service = new MicrosoftMailService({ now: () => now, fetchImpl, lookup: async () => [{ address: '1.1.1.1', family: 4 }],
  assertSourceCurrent: () => { if (!sourceAllowed) throw new Error('Synthetic account disabled before dispatch') },
  store: { read: async () => state, write: async (next, validate) => { validate(); state = structuredClone(next) } } })
const client = '11111111-2222-3333-4444-555555555555'
await assert.rejects(() => service.begin('not-a-client', false), /application ID/)
const session = await service.begin(client, true)
assert.equal(session.verificationUri, 'https://microsoft.com/devicelogin', 'untrusted verification URLs never enter the UI')
assert.doesNotMatch(JSON.stringify(session), /private-device-code|private-access-token/)
assert.equal((await service.poll(session.id)).status, 'pending')
assert.equal(requests.length, 1, 'device code poll interval is enforced on the host')
now += 5000; tokenError = 'slow_down'
const slow = await service.poll(session.id)
assert.equal(slow.status, 'pending'); if (slow.status === 'pending') assert.equal(slow.intervalMs, 10_000)
now += 10_000; tokenError = ''
const connected = await service.poll(session.id)
const source = connected.source!
assert.equal(connected.status, 'connected')
assert.ok(state!.accounts[0].refreshToken)
assert.doesNotMatch(JSON.stringify(connected), /private-access-token|private-refresh-token/)
setMicrosoftMailRuntime(service)
const settings = { sources: [source], maxMessages: 50, lastSyncAt: null }
const list = await scanEmailMessages(settings)
assert.equal(list.messages[0].id, `${source.id}:fixture-message`, 'actual Email connector dispatch includes Microsoft inbox messages')
assert.match(requests.at(-1)!.url.pathname, /mailFolders\/inbox\/messages$/)
assert.equal((await readEmailMessage(settings, list.messages[0].id)).body, 'Exact full body')
continuation = 'https://attacker.example.com/'
const beforePage = requests.length
await assert.rejects(() => service.list(source, 10), /invalid mail continuation/)
assert.equal(requests.length, beforePage + 1, 'a continuation cannot forward tokens off the fixed Graph inbox endpoint')
continuation = ''
assert.equal(await service.read({ ...source, id: 'wrong-account' }, list.messages[0].id), null)
await assert.rejects(() => service.list({ ...source, microsoftRevision: 999 }, 10), /changed or was disconnected/)
const snapshot = { version: 1, connectorId: 'email', accountId: source.id, recipient: 'recipient@example.com', subject: 'Exact subject', body: ' Exact approved body\n ', threadKey: 'email:new', accountRevision: 'fixture' }
const event: any = { outbound: snapshot, reviewDigest: 'fixture-digest', delivery: { status: 'sending', reviewDigest: 'fixture-digest' }, approvalStatus: 'approved', requiresApproval: false }
await assert.rejects(() => service.send(source, { ...event, approvalStatus: 'pending' }), /durable exact-content/)
const delivery = await service.send(source, event)
assert.equal(delivery.status, 'queued', 'Graph acceptance is not claimed as recipient delivery')
assert.equal(requests.at(-1)!.body.message.body.content, snapshot.body)
await service.send(source, { ...event, outbound: { ...snapshot, inReplyTo: `${source.id}:fixture-message` } })
assert.match(requests.at(-1)!.url.pathname, /fixture-message\/reply$/)
await assert.rejects(() => service.send(source, { ...event, outbound: { ...snapshot, inReplyTo: 'other:fixture-message' } }), /another Microsoft account/)
sendStatus = 503
const before = requests.length
await assert.rejects(() => service.send(source, event), error => !String(error).includes('PRIVATE_ERROR_CANARY'))
assert.equal(requests.length, before + 1, 'ambiguous Microsoft sends never retry')
sendStatus = 202
sourceAllowed = false
const beforeDisabled = requests.length
await assert.rejects(() => service.send(source, event), /disabled before dispatch/)
assert.equal(requests.length, beforeDisabled, 'a disabled host source cannot send using previously captured account settings')
sourceAllowed = true
now += 3_600_000
const refreshedBefore = requests.filter(request => request.url.pathname.endsWith('/token')).length
await Promise.all([service.list(source, 10), service.list(source, 10)])
assert.equal(requests.filter(request => request.url.pathname.endsWith('/token')).length, refreshedBefore + 1, 'concurrent reads share a serialized token refresh')
const canceled = await service.begin(client, false)
now += 5000
let entered!: () => void, release!: () => void
const started = new Promise<void>(resolve => { entered = resolve })
const barrier = new Promise<void>(resolve => { release = resolve })
delayedToken = async () => { entered(); await barrier }
const pending = service.poll(canceled.id)
await started; service.cancel(canceled.id); release()
await assert.rejects(() => pending, /canceled/)
delayedToken = null
assert.equal(state!.accounts.length, 1, 'canceled sign-in cannot commit an account')
const readonly = await service.begin(client, false); now += 5000
const readonlyResult = await service.poll(readonly.id)
await assert.rejects(() => service.send(readonlyResult.source!, { ...event, outbound: { ...snapshot, accountId: readonlyResult.source!.id } }), /sending permission/)
now += 3_600_000
let refreshEntered!: () => void, refreshRelease!: () => void
const refreshStarted = new Promise<void>(resolve => { refreshEntered = resolve })
const refreshBarrier = new Promise<void>(resolve => { refreshRelease = resolve })
delayedToken = async () => { refreshEntered(); await refreshBarrier }
const staleRead = service.list(readonlyResult.source!, 10)
await refreshStarted
const removal = service.remove(source.id)
refreshRelease()
await assert.rejects(() => staleRead, /connection changed/)
await removal
assert.equal(state!.accounts.length, 0, 'disconnect during token refresh cannot resurrect saved credentials')
await assert.rejects(() => service.list(source, 10), /changed or was disconnected/)
console.log('Microsoft device flow, host polling, protected-token boundary, actual Email dispatch, refresh serialization, read-only grants, replies, exact sends, no retries and cancellation passed')
