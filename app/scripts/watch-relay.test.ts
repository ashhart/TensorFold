import assert from 'node:assert/strict'
import { createWatchRelay, parseWatchCommand, watchSnapshot, type WatchCommand } from '../mobile/src/lib/watch-relay'
import { RESTRICTED_CAPABILITIES, type MobileHost, type StudioClient } from '../mobile/src/lib/client'

const host: MobileHost = {snapshot: {serverName: 'Example workspace', appVersion: '1', generatedAt: 1, activeThreadIds: ['job-one'], projects: [], work: {approvals: {pending: 1}, jobs: {running: 1, queued: 2}},
  threads: [{id: 'job-one', title: 'Review example task', kind: 'task', providerId: 'example', model: 'example-model', status: 'idle', updatedAt: 1, timeline: [{id: 'reply-one', kind: 'assistant', text: 'Private permitted reply', output: 'private output', path: '/private/example'}]},
    {id: 'chat-one', title: 'Private chat', kind: 'chat', providerId: 'example', model: 'example-model', status: 'idle', updatedAt: 2, timeline: []}]},
  status: {scope: 'paired', role: 'operator', capabilities: {...RESTRICTED_CAPABILITIES, taskGuide: true, taskStart: true}, engine: null}}
const snapshot = watchSnapshot(host, 'session-one', 100)
assert.equal(snapshot.jobs.length, 1)
assert.equal(snapshot.jobs[0].latestReply, null, 'paired transcript stays private')
assert.equal(snapshot.jobs[0].latestReplyId, 'reply-one', 'audio uses an existing item identity without revealing transcript')
assert.equal(snapshot.jobs[0].canReply, true)
assert.ok(!JSON.stringify(snapshot).includes('private output'))
assert.ok(!JSON.stringify(snapshot).includes('/private'))
assert.equal(watchSnapshot(host, 'session-one', 100, 'Offline').jobs[0].canReply, false)
assert.equal(watchSnapshot(null, '', 100).connected, false)
const raw: WatchCommand = {version: 1, id: 'request-one', action: 'reply', session: 'session-one', threadId: 'job-one', text: 'Please continue', generatedAt: 100}
assert.throws(() => parseWatchCommand({...raw, text: 'x'.repeat(2_001)}))
assert.throws(() => parseWatchCommand({...raw, action: 'shell'}))
assert.throws(() => parseWatchCommand({...raw, action: 'audio-next', speechId: 'bad/id', sequence: 0}))
let sent = 0, speech = 0, foreground = true, identity = 'session-one', now = 100
const client = {send: async (id: string, text: string, active: boolean) => {assert.equal(id, 'job-one'); assert.equal(text, raw.text); assert.equal(active, true); sent++},
  watchSpeechStatus: async () => ({available: true, provider: 'kokoro', reason: null}),
  startWatchSpeech: async (id: string, item: string) => {assert.equal(id, 'job-one'); assert.equal(item, 'reply-one'); speech++; return {speechId: 'speech-one', provider: 'kokoro', sampleRate: 24_000}},
  readWatchSpeech: async () => ({speechId: 'speech-one', sequence: 0, state: 'done'}), cancelWatchSpeech: async () => ({cancelled: true})} as unknown as StudioClient
const relay = createWatchRelay({now: () => now, session: () => identity, foreground: () => foreground, client: () => client, host: async () => structuredClone(host), publish: () => {}})
assert.equal((await relay.handle(raw)).ok, true)
assert.equal((await relay.handle(raw)).ok, true)
assert.equal(sent, 1, 'duplicate command IDs cannot resend a reply')
foreground = false
assert.equal((await relay.handle({...raw, id: 'request-two'})).ok, false)
foreground = true; now = 50_000
assert.equal((await relay.handle({...raw, id: 'request-three'})).ok, false)
now = 100; identity = 'session-two'
assert.equal((await relay.handle({...raw, id: 'request-four'})).ok, false)
identity = 'session-one'
assert.equal((await relay.handle({...raw, id: 'audio-start-one', action: 'audio-start', itemId: 'reply-one'})).ok, true)
assert.equal(speech, 1)
assert.equal((await relay.handle({...raw, id: 'audio-start-two', action: 'audio-start', itemId: 'missing'})).ok, false)
let release!: (host: MobileHost) => void
const switched = createWatchRelay({now: () => 100, session: () => identity, foreground: () => true, client: () => client, host: () => new Promise(resolve => {release = resolve}), publish: () => {throw new Error('Must not publish after session switch')}})
const request = switched.handle({...raw, id: 'switch-session'})
identity = 'different-session'; release(host)
assert.equal((await request).ok, false)
assert.equal(sent, 1, 'session changes during refresh must not send to the old account')
console.log('Watch relay privacy, role/freshness/session gates, at-most-once replies and exact-item audio checks passed')

// A late audio fetch must never complete into the next paired session.
identity = 'session-one'
let audioResolve!: (value: unknown) => void
const delayedClient = {...client, readWatchSpeech: () => new Promise(resolve => {audioResolve = resolve})} as unknown as StudioClient
const audioRelay = createWatchRelay({now: () => 100, session: () => identity, foreground: () => true, client: () => delayedClient, host: async () => host, publish: () => {}})
const audioRequest = audioRelay.handle({...raw, id: 'delayed-audio', action: 'audio-next', speechId: 'speech-one', sequence: 0})
identity = 'different-session'; audioResolve({speechId: 'speech-one', sequence: 0, state: 'audio', audioBase64: 'AAAA', sampleRate: 24_000})
const dropped = await audioRequest
assert.equal(dropped.ok, false); assert.equal(dropped.audio, undefined)
