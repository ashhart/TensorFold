import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import type { GoogleWorkspacePrivateState } from './private-state'
import { requestedScopes } from './scopes'
import { gmailInboxKey } from '../../../shared/gmail-inbox'

const now = Date.UTC(2026, 9, 28, 12)
let state: GoogleWorkspacePrivateState = {
  version: 1, client: { clientId: 'synthetic.apps.googleusercontent.com', clientSecret: 'synthetic-secret' }, drafts: [], actions: [], accounts: ['account-a', 'account-b'].map(id => ({
    id, subject: id, email: `${id}@example.com`, revision: 1, status: 'connected', connectedAt: now,
    capabilities: ['gmail.read', 'gmail.send'], tokens: { accessToken: `synthetic-${id}`, refreshToken: `synthetic-refresh-${id}`, expiresAt: now + 3_600_000, scopes: requestedScopes(['gmail.read', 'gmail.send']) }
  }))
}
let writes = 0, calls = 0, failSend = false, failStore = false, revokeAtDns = false, service: GoogleWorkspaceService
const message = (id: string, account: string, full = false) => ({
  id, threadId: id === 'bb11' ? 'ee11' : 'dd11', internalDate: String(now + (id === 'aa12' ? 10 : 0)), labelIds: ['INBOX', 'UNREAD'], snippet: `Synthetic preview ${id}`, payload: {
    mimeType: 'text/plain', headers: [{ name: 'Subject', value: 'Same subject' }, { name: 'From', value: `Sender <sender-${account}@example.com>` }, { name: 'To', value: `${account}@example.com` }, { name: 'Date', value: new Date(now).toUTCString() }, { name: 'Message-ID', value: `<${id}@example.com>` }, { name: 'References', value: '<earlier@example.com>' }],
    ...(full ? { body: { data: Buffer.from(`Untrusted synthetic body ${id}`).toString('base64url') } } : {})
  }
})
const deps = {
  store: { read: async () => structuredClone(state), write: async (next: GoogleWorkspacePrivateState, validate?: () => void) => { validate?.(); if (failStore) throw new Error('Synthetic storage failure'); state = structuredClone(next) } },
  now: () => now,
  openExternal: async () => { throw new Error('No sign-in in this fixture') },
  lookup: async () => {
    if (revokeAtDns) { revokeAtDns = false; await service.enableInbox({ accountId: 'account-a', revision: state.accounts[0].revision, enabled: false }) }
    return [{ address: '142.250.1.1', family: 4 as const }]
  },
  fetchImpl: (async (input: any, init: RequestInit = {}) => {
    calls++
    const url = new URL(String(input)), account = new Headers(init.headers).get('Authorization')?.endsWith('account-b') ? 'account-b' : 'account-a'
    if (url.pathname.endsWith('/messages/send')) {
      writes++
      const body = JSON.parse(String(init.body)), mime = Buffer.from(body.raw, 'base64url').toString('utf8')
      assert.equal(body.threadId, 'dd11')
      assert.match(mime, /To: sender-account-a@example.com\r\n/)
      assert.match(mime, /In-Reply-To: <aa11@example.com>\r\n/)
      assert.match(mime, /References: <earlier@example.com> <aa11@example.com>\r\n/)
      assert.ok(mime.endsWith(Buffer.from('Exact reviewed reply').toString('base64') + '\r\n'))
      if (failSend) throw new TypeError('Synthetic private network failure')
      return Response.json({ id: 'sent11', threadId: body.threadId })
    }
    assert.equal(init.method, 'GET')
    if (url.pathname.endsWith('/messages')) return Response.json({ messages: [{ id: 'aa11' }, { id: 'aa12' }, { id: 'bb11' }], nextPageToken: 'more-synthetic' })
    if (url.pathname.includes('/messages/')) return Response.json(message(url.pathname.split('/').at(-1)!, account, url.searchParams.get('format') === 'full'))
    if (url.pathname.includes('/threads/')) return Response.json({ id: 'dd11', messages: [message('aa11', account, true), message('aa12', account, true)] })
    throw new Error('Unexpected synthetic endpoint')
  }) as typeof fetch,
}
service = new GoogleWorkspaceService(deps)
assert.equal((await service.inboxSnapshot()).events.length, 0); assert.equal(calls, 0)
await service.syncInbox(); assert.equal(calls, 0, 'Sync cannot opt accounts into Inbox')
await assert.rejects(service.enableInbox({ accountId: 'account-a', revision: 7, enabled: true }), /account changed/)
await service.enableInbox({ accountId: 'account-a', revision: 1, enabled: true })
await service.enableInbox({ accountId: 'account-b', revision: 1, enabled: true })
const first = await service.syncInbox()
assert.equal(first.events.length, 6)
assert.equal(new Set(first.events.map(event => event.id)).size, 6, 'Same remote IDs in two accounts remain separate')
assert.equal(new Set(first.events.map(event => event.threadKey)).size, 4, 'Native thread IDs group related mail without subject heuristics')
assert.deepEqual((await service.syncInbox()).events.map(event => event.id), first.events.map(event => event.id), 'Repeated sync has stable identities')
assert.equal(writes, 0, 'Status, opt-in and sync never send mail')
assert.ok(!JSON.stringify(first).includes('synthetic-secret')); assert.ok(!JSON.stringify(first).includes('synthetic-refresh'))
const key = gmailInboxKey('account-a', 1, 'dd11')
assert.equal((await service.inboxThread(key)).messages.length, 2)
await assert.rejects(service.inboxThread(gmailInboxKey('account-a', 7, 'dd11')), /account changed/)
assert.equal((await service.saveInboxDraft({ key, expectedRevision: 0, text: 'Keep this unsaved reply' })).saved, true)
assert.equal((await service.inboxDraft(gmailInboxKey('account-b', 1, 'dd11'))).text, '')
const review = await service.inboxReply(key, 'Exact reviewed reply')
assert.equal(writes, 0)
assert.equal(review.action.kind, 'gmail.send')
if (review.action.kind !== 'gmail.send') throw new Error('Wrong action')
assert.deepEqual(review.action.email.to, ['sender-account-a@example.com'])
assert.equal(review.action.email.reply?.rfcMessageId, '<aa11@example.com>')
const receipt = { id: review.id, accountId: review.accountId, revision: review.revision, digest: review.digest }
await assert.rejects(service.executeReviewed({ ...receipt, digest: '0'.repeat(64) }), /review no longer matches/)
assert.equal(writes, 0)
assert.equal((await service.executeReviewed(receipt)).status, 'succeeded')
assert.equal((await service.executeReviewed(receipt)).status, 'succeeded'); assert.equal(writes, 1, 'An accepted reply cannot be replayed')
const uncertain = await service.inboxReply(key, 'Exact reviewed reply'); failSend = true
const uncertainReceipt = { id: uncertain.id, accountId: uncertain.accountId, revision: uncertain.revision, digest: uncertain.digest }
assert.equal((await service.executeReviewed(uncertainReceipt)).status, 'uncertain')
assert.equal((await service.executeReviewed(uncertainReceipt)).status, 'uncertain'); assert.equal(writes, 2)
failSend = false
const interrupted = await service.inboxReply(key, 'Exact reviewed reply'); revokeAtDns = true
assert.equal((await service.executeReviewed({ id: interrupted.id, accountId: interrupted.accountId, revision: interrupted.revision, digest: interrupted.digest })).status, 'uncertain')
assert.equal(writes, 2, 'Removing Inbox scope during DNS prevents socket dispatch')
assert.equal((await service.inboxSnapshot()).events.filter(event => event.threadKey === key).length, 0)
await service.enableInbox({ accountId: 'account-a', revision: 1, enabled: true }); await service.syncInbox()
state.accounts[0].revision++
assert.equal((await service.inboxSnapshot()).events.some(event => event.threadKey === key), false, 'Authorization changes hide old cached mail and approvals')
const renewed = gmailInboxKey('account-a', 2, 'dd11')
assert.equal((await service.inboxDraft(renewed)).text, 'Keep this unsaved reply', 'Same-account reconnect preserves the protected draft')
assert.equal((await service.saveInboxDraft({ key: renewed, expectedRevision: 0, text: 'A stale edit' })).saved, false, 'Draft CAS refuses stale writers')
failStore = true
await assert.rejects(service.saveInboxDraft({ key: renewed, expectedRevision: 1, text: 'A failed edit' }), /storage failure/)
failStore = false
assert.equal((await service.inboxDraft(renewed)).text, 'Keep this unsaved reply')
service.dispose()
const restarted = new GoogleWorkspaceService(deps)
assert.equal((await restarted.inboxDraft(renewed)).text, 'Keep this unsaved reply', 'Draft survives service restart')
state.accounts[0].status = 'disconnected'; state.accounts[0].tokens = undefined
await assert.rejects(restarted.inboxDraft(renewed), /no longer available/)
restarted.dispose()
console.log('Protected Gmail Inbox opt-in, account isolation, native threads, reviewed one-shot replies, dispatch revocation and reconnect-safe drafts passed.')
