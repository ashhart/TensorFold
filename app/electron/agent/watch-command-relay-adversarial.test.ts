import assert from 'node:assert/strict'
import { createHostWatchCommandRelay } from './watch-command-relay'
import { createWatchRelay, type WatchRelayHost, type WatchRelayClient } from '../../shared/watch-relay'
import type { WatchSpeechBroker } from './watch-speech'

const caps = {taskStart: true, taskGuide: true, taskStop: true, approvalResolve: false, transcriptReview: true, changesReview: false, engineControl: false}
const host: WatchRelayHost = {snapshot: {serverName: 'Synthetic host', activeThreadIds: ['job-one'], work: {jobs: {running: 1, queued: 0}, approvals: {pending: 0}},
  threads: [{id: 'job-one', title: 'Synthetic task', kind: 'task', providerId: 'example', model: 'example', status: 'idle', updatedAt: 1, timeline: [{id: 'assistant-one', kind: 'assistant', text: 'Allowed synthetic reply'}]}]}, status: {capabilities: caps, engine: null}}
const base = {version: 1, id: 'request-one', session: 'paired-session', generatedAt: 100, action: 'reply', threadId: 'job-one', text: 'Continue'} as const
let now = 100, sends = 0, currentStatus: unknown = {scope: 'paired', capabilities: caps, engine: null}
const token = 'fixture-token'
const speech = {isAllowed: () => true, status: async () => ({available: true, provider: 'kokoro', reason: null}),
  start: async () => ({speechId: 'speech-one', provider: 'kokoro', sampleRate: 24_000}),
  chunk: async () => ({speechId: 'speech-one', sequence: 0, state: 'audio', audioBase64: 'A'.repeat(70_000), sampleRate: 24_000}),
  cancel: () => ({cancelled: true}), revoke() {}, stop() {}} as WatchSpeechBroker
const relay = createHostWatchCommandRelay({now: () => now, authorize: value => ({ok: value === token, sessionId: base.session, scope: 'paired'}), speech,
  request: async (method, path) => {
    if (path === '/v1/device-status') return {status: 200, value: currentStatus}
    if (method === 'GET') return {status: 200, value: structuredClone(host.snapshot)}
    sends++; return {status: 202, value: {queued: true}}
  }})
const call = (raw: unknown = base, signal = new AbortController().signal) => relay.handle(token, raw, signal)
const first = await call(); assert.equal(first.ok, true); assert.equal(first.snapshot, undefined, 'reply receipts do not retain a pre-dispatch projection')
host.snapshot.threads[0].incognito = true
const cached = await call(); assert.equal(cached.ok, true); assert.equal(cached.snapshot, undefined)
assert.doesNotMatch(JSON.stringify(cached), /Allowed synthetic reply|Synthetic task/)
assert.equal(sends, 1)
currentStatus = {scope: 'owner', capabilities: caps, engine: null}
assert.equal((await call()).ok, false, 'cached receipt never accepts owner-status instead of paired status')
currentStatus = {scope: 'paired', capabilities: {...caps, taskStart: false, taskGuide: false}, engine: null}
assert.equal((await call()).ok, false, 'role downgrade rejects duplicate')
currentStatus = {scope: 'paired', capabilities: caps, engine: null}
const oversized = await call({...base, action: 'audio-next', id: 'oversized-audio', speechId: 'speech-one', sequence: 0})
assert.equal(oversized.ok, false); assert.equal(oversized.audio, undefined)
const cancelled = new AbortController(); cancelled.abort()
assert.equal((await call({...base, id: 'cancelled-command'}, cancelled.signal)).ok, false)
assert.equal(sends, 1)
await relay.stop()

// Future-clock commands remain remembered until generatedAt + max age, not receive time + age.
let clock = 0, deliveries = 0
const client = {send: async () => {deliveries++}} as unknown as WatchRelayClient
const shared = createWatchRelay({now: () => clock, session: () => base.session, foreground: () => true, client: () => client, host: async () => {host.snapshot.threads[0].incognito = false; return host}, publish() {}, ledgerMaxAgeMs: 45_000})
const future = {...base, generatedAt: 45_000}
assert.equal((await shared.handle(future)).ok, true)
clock = 80_000
assert.equal((await shared.handle({...future, id: 'fresh-other-command', generatedAt: clock})).ok, true)
assert.equal((await shared.handle(future)).ok, true)
assert.equal(deliveries, 2, 'valid future-skew duplicate retained past receive-time age')
clock = 90_001
assert.equal((await shared.handle(future)).ok, false, 'original command stale before its cached receipt is read')
assert.equal(deliveries, 2)

// No more than four actionable requests execute concurrently; slow operations are never evicted.
let resolve!: () => void
const waiting = new Promise<void>(yes => {resolve = yes})
const delayedClient = {send: async () => {await waiting}} as unknown as WatchRelayClient
clock = 100
const slow = createWatchRelay({now: () => clock, session: () => base.session, foreground: () => true, client: () => delayedClient, host: async () => host, publish() {}, ledgerMaxAgeMs: 45_000})
const pending = Array.from({length: 4}, (_, i) => slow.handle({...base, id: `pending-request-${i}`}))
for (let i = 0; i < 15; i++) await Promise.resolve()
clock = 100_000
const crowded = await slow.handle({...base, id: 'fifth-request', generatedAt: clock})
assert.equal(crowded.ok, false); assert.match(crowded.message, /Too many/)
resolve(); await Promise.all(pending)
clock = 150_000
assert.equal((await slow.handle({...base, id: 'after-expiry', generatedAt: clock})).ok, true, 'settled expired entries release bounded ledger capacity')
console.log('Watch host adversarial tests passed current-role/privacy duplicate fences, response size, cancellation, future skew expiry and inflight bounds')
