import assert from 'node:assert/strict'
import { MailConnectionHealthService, type MailHealthRecord } from './mail-connection-health'
import type { MailConnectionAccount } from '../../shared/mail-connections'

let accounts: MailConnectionAccount[] = [{ provider: 'gmail', id: 'account-a', email: 'reader-a@example.com', revision: '1', authorization: 'connected', read: true, send: false, inbox: false }]
let records: MailHealthRecord[] = [], reads = 0, now = 100, reason: unknown, gate: (() => Promise<void>) | undefined, failWrite = false
const service = new MailConnectionHealthService({
  accounts: async () => ({ googleConfigured: true, accounts: structuredClone(accounts) }),
  probe: async account => { assert.equal(account.id, 'account-a'); reads++; await gate?.(); if (reason) throw reason },
  store: { read: async () => structuredClone(records), write: async next => { if (failWrite) throw new Error('Synthetic protected write failure'); records = structuredClone(next) } },
  now: () => now,
})
const target = () => ({ provider: 'gmail' as const, id: 'account-a', revision: accounts[0].revision })
assert.equal((await service.snapshot()).accounts[0].state, 'untested')
assert.equal(reads, 0, 'Reading status must never read a mailbox')
const first = await service.test(target())
assert.equal(first.state, 'healthy'); assert.equal(first.lastSuccessAt, 100); assert.equal(first.send, false)
reason = Object.assign(new Error('synthetic-private-response'), { status: 403 }); now = 200
const denied = await service.test(target())
assert.equal(denied.state, 'permission'); assert.equal(denied.lastSuccessAt, 100); assert.equal(denied.checkedAt, 200)
assert.ok(!JSON.stringify(denied).includes('synthetic-private-response'))
for (const [status, expected] of [[401, 'reconnect'], [429, 'rate-limited'], [503, 'error']] as const) {
  reason = Object.assign(new Error('synthetic-private-response'), { status })
  assert.equal((await service.test(target())).state, expected)
}
reason = new TypeError('synthetic-private-network-detail')
assert.equal((await service.test(target())).state, 'unreachable')
reason = undefined
let release!: () => void
gate = () => new Promise<void>(resolve => { release = resolve })
const before = reads, pending = service.test(target()), duplicate = service.test(target())
await new Promise(resolve => setImmediate(resolve))
assert.equal(reads, before + 1); assert.equal(pending, duplicate, 'Repeated checks share one bounded read')
release(); await pending; gate = undefined
const old = target(); accounts[0].revision = '2'
assert.equal((await service.snapshot()).accounts[0].state, 'untested', 'Reauthorization cannot inherit a prior healthy verdict')
await assert.rejects(service.test(old), /account changed/)
const saved = structuredClone(records)
gate = () => new Promise<void>(resolve => { release = resolve })
const interrupted = service.test(target())
await new Promise(resolve => setImmediate(resolve))
accounts[0].revision = '3'; release()
await assert.rejects(interrupted, /account changed/)
assert.deepEqual(records, saved, 'A check crossing reauthorization cannot persist a healthy result')
gate = undefined; accounts[0].authorization = 'disconnected'
const count = reads
assert.equal((await service.test(target())).state, 'disconnected'); assert.equal(reads, count)
accounts[0].authorization = 'connected'; accounts[0].read = false
assert.equal((await service.test(target())).state, 'permission'); assert.equal(reads, count)
accounts[0].read = true; failWrite = true
await assert.rejects(service.test(target()), /protected write failure/)
assert.equal((await service.snapshot()).accounts[0].state, 'untested', 'Unpersisted checks must not become a durable success')
failWrite = false
accounts = []
await assert.rejects(service.test(old), /account changed/)
console.log('Connection status is read-free; explicit checks coalesce, redact errors and fence account changes and failed persistence.')
