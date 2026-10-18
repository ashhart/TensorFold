import assert from 'node:assert/strict'
import { createMessagingConnectionState, messagingChannelEntries, messagingChannelScopeError, telegramAccessError } from './messaging-connection-state'

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((done) => { resolve = done }); return { promise, resolve } }
const first = createMessagingConnectionState(() => {})
const second = createMessagingConnectionState(() => {})
const saving = deferred<void>()
let verificationCalls = 0
const pending = first.verify(() => saving.promise, async () => { verificationCalls++; return { ok: true, identity: 'Workspace A bot' } })
assert.equal(verificationCalls, 0, 'verification waits for durable settings save')
assert.equal(first.snapshot().phase, 'verifying')
assert.equal(second.snapshot().phase, 'idle', 'another account never inherits pending status')
saving.resolve()
await pending
assert.equal(first.snapshot().verified, true)
assert.equal(first.snapshot().message, 'Verified as Workspace A bot')
assert.equal(second.snapshot().verified, false, 'verification is per account')
first.invalidate(true)
assert.equal(first.snapshot().verified, true, 'channel changes retain token identity verification')
first.invalidate()
assert.equal(first.snapshot().verified, false, 'credential changes invalidate verification')
let syncCalls = 0
assert.equal(await first.action('syncing', async () => { syncCalls++; return 'synced' }), false)
assert.equal(syncCalls, 0, 'unverified account cannot invoke a sync action')

const stale = deferred<{ ok: boolean; identity: string }>()
const stalePending = first.verify(async () => {}, () => stale.promise)
await Promise.resolve()
first.invalidate()
stale.resolve({ ok: true, identity: 'Old token bot' })
await stalePending
assert.equal(first.snapshot().verified, false, 'late old-token completion cannot reauthorize the account')
assert.equal(first.snapshot().message, '')

await first.verify(async () => { throw new Error('secret-sentinel-never-display') }, async () => { throw new Error('must not reach verify') })
assert.equal(first.snapshot().phase, 'error')
assert.doesNotMatch(first.snapshot().message, /secret-sentinel/, 'save exceptions cannot reveal credentials in status')
await first.verify(async () => {}, async () => { throw new Error('token-in-http-error') })
assert.doesNotMatch(first.snapshot().message, /token-in-http-error/)
await first.verify(async () => {}, async () => ({ ok: false, error: 'The token was revoked.' }))
assert.equal(first.snapshot().verified, false)

const removedResult = deferred<{ ok: boolean }>()
const removed = second.verify(async () => {}, () => removedResult.promise)
await Promise.resolve()
second.dispose()
removedResult.resolve({ ok: true })
await removed
assert.equal(second.snapshot().verified, false, 'removed account cannot accept late verification')

assert.equal(messagingChannelScopeError(''), 'Choose at least one channel before connecting; an empty list grants no access.')
assert.equal(messagingChannelScopeError('*'), null)
assert.match(messagingChannelScopeError('*, C123')!, /alone/)
assert.deepEqual(messagingChannelEntries('#general, C123; C123'), ['general', 'C123'])
assert.equal(telegramAccessError('-100123, 12345', '67890'), null)
for (const chats of ['', '*', 'group', '0', '1.1']) assert.ok(telegramAccessError(chats, '67890'))
for (const users of ['', '*', '-12', '0', 'person']) assert.ok(telegramAccessError('12345', users))
console.log('Messaging account isolation, durable save ordering, stale verification, secret-safe failure and explicit access regressions passed')
