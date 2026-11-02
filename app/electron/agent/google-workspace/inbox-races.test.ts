import assert from 'node:assert/strict'
import { GoogleWorkspaceService } from './index'
import { requestedScopes } from './scopes'
import type { GoogleWorkspacePrivateState } from './private-state'

const now = Date.UTC(2026, 9, 28), key = 'gmail:account-a:1:dd11'
let state: GoogleWorkspacePrivateState = { version: 1, client: { clientId: 'synthetic.apps.googleusercontent.com', clientSecret: 'synthetic' }, drafts: [], actions: [], accounts: [{ id: 'account-a', subject: 'synthetic', email: 'reader@example.com', revision: 1, connectedAt: now, status: 'connected', capabilities: ['gmail.read', 'gmail.send'], tokens: { accessToken: 'synthetic-token', expiresAt: now + 3600000, scopes: requestedScopes(['gmail.read', 'gmail.send']) } }] }
let release = () => { }, entered = () => { }, endpoint = '', fail = false
const deferred = () => { let resolve!: () => void; const promise = new Promise<void>(done => { resolve = done }); return { promise, resolve } }
const message = { id: 'aa11', threadId: 'dd11', snippet: 'Synthetic metadata', payload: { headers: [{ name: 'Subject', value: 'Fixture' }, { name: 'From', value: 'sender@example.com' }, { name: 'Message-ID', value: '<aa11@example.com>' }] } }
const service = new GoogleWorkspaceService({
  now: () => now, openExternal: async () => { }, lookup: async () => [{ address: '142.250.1.1', family: 4 }],
  store: { read: async () => structuredClone(state), write: async (next, validate) => { validate?.(); state = structuredClone(next) } },
  fetchImpl: (async input => {
    const url = new URL(String(input))
    if (url.pathname.endsWith(endpoint) && endpoint) { const wait = deferred(); release = wait.resolve; entered(); await wait.promise }
    if (fail) throw new Error('Synthetic provider error with private details')
    return Response.json(url.pathname.endsWith('/messages') ? { messages: [{ id: 'aa11' }] } : url.pathname.endsWith('/threads/dd11') ? { id: 'dd11', messages: [message] } : message)
  }) as typeof fetch,
})
const select = (enabled: boolean) => service.enableInbox({ accountId: 'account-a', revision: 1, enabled })
await select(true); await service.syncInbox()
assert.equal((await service.inboxSnapshot()).events.length, 1)
fail = true; await service.syncInbox(); fail = false
let snapshot = await service.inboxSnapshot()
assert.equal(snapshot.events.length, 1, 'Transient failures preserve the last successful metadata cache')
assert.ok(snapshot.accounts[0].error); assert.doesNotMatch(snapshot.accounts[0].error!, /private details/)
endpoint = '/messages'
let signal = deferred(); entered = signal.resolve
const pendingSync = service.syncInbox(); await signal.promise
await select(false); await select(true); release(); await pendingSync
snapshot = await service.inboxSnapshot()
assert.equal(snapshot.events.length, 0, 'A previous selection cannot repopulate the new Inbox cache')
assert.equal(snapshot.accounts[0].error, undefined, 'Cancelled previous selection cannot overwrite the new source status')
endpoint = ''; await service.syncInbox()
endpoint = '/threads/dd11'; signal = deferred(); entered = signal.resolve
const pendingRead = service.inboxThread(key); const rejected = assert.rejects(pendingRead, /scope changed/)
await signal.promise; await select(false); await select(true); release(); await rejected
endpoint = ''; await service.syncInbox()
assert.equal((await service.inboxThread(key)).id, 'dd11', 'A fresh explicit read after selection change succeeds')
endpoint = '/messages/aa11'; signal = deferred(); entered = signal.resolve
const pendingReply = service.inboxReply(key, 'Keep this draft')
const refusedReply = assert.rejects(pendingReply, /scope changed|no longer selected/)
await signal.promise; await select(false); await select(true); release(); await refusedReply
assert.equal(state.actions.length, 0, 'A reply prepared under an obsolete selection cannot create a usable review')
service.dispose()
console.log('Gmail failures preserve last good metadata; late syncs and full reads cannot cross an Inbox selection change.')
