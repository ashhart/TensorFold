import assert from 'node:assert/strict'
import { createHostWatchCommandRelay } from './watch-command-relay'
import type { WatchSpeechBroker } from './watch-speech'

const capabilities = {taskStart: true, taskGuide: true, taskStop: true, approvalResolve: false, transcriptReview: true, changesReview: false, engineControl: false}
const session = 'paired-session', token = 'phone_' + 'x'.repeat(40)
let authorized = true, now = 100, mutations = 0, revokedOnRead = false, failDispatch = false
const snapshot = {serverName: 'Synthetic Mac', activeThreadIds: ['job-one'], work: {jobs: {running: 1, queued: 0}, approvals: {pending: 0}}, threads: [
  {id: 'job-one', title: 'Synthetic job', kind: 'task', providerId: 'tensorfold', model: 'example-model', status: 'idle', updatedAt: 1, timeline: [{id: 'assistant-one', kind: 'assistant', text: 'Synthetic reply', output: 'PRIVATE_TOOL_OUTPUT', path: '/private/example'}]},
  {id: 'secret-job', title: 'PRIVATE_INCOGNITO_TITLE', kind: 'task', incognito: true, providerId: 'example', model: 'example-model', status: 'idle', updatedAt: 1, timeline: []},
  {id: 'phone-local-job', title: 'PRIVATE_PHONE_LOCAL_TITLE', kind: 'task', phoneLocalOnly: true, providerId: 'example', model: 'example-model', status: 'idle', updatedAt: 1, timeline: []},
]}
let status: unknown = {scope: 'paired', role: 'operator', capabilities, engine: {state: 'ready', model: 'example-model'}}
let audioStarts = 0
const broker = {isAllowed: () => authorized, status: async () => ({available: authorized, provider: 'kokoro', reason: null}),
  start: async () => {audioStarts++; return {speechId: 'speech-one', provider: 'kokoro', sampleRate: 24_000}},
  chunk: async () => ({speechId: 'speech-one', sequence: 0, state: 'audio', audioBase64: 'AAAA', sampleRate: 24_000}),
  cancel: () => ({cancelled: true}), revoke() {}, stop() {}} as WatchSpeechBroker
const service = createHostWatchCommandRelay({now: () => now,
  authorize: value => ({ok: authorized && value === token, sessionId: session, scope: 'paired'}), speech: broker,
  request: async (method, path, _body, presented) => {
    assert.equal(presented, token)
    if (path === '/v1/snapshot') { if (revokedOnRead) authorized = false; return {status: 200, value: structuredClone(snapshot)} }
    if (path === '/v1/device-status') return {status: 200, value: status}
    assert.equal(method, 'POST'); assert.equal(path, '/v1/threads/job-one/guide'); mutations++
    if (failDispatch) throw new Error('PRIVATE_INTERNAL_ERROR')
    return {status: 202, value: {queued: true}}
  },
})
const command = {version: 1, id: 'request-one', session, generatedAt: now, action: 'reply', threadId: 'job-one', text: 'Please continue'}
const invoke = (input: unknown = command, bearer = token) => service.handle(bearer, input, new AbortController().signal)
assert.equal((await invoke()).ok, true, 'host command works without React/phone foreground')
assert.equal((await invoke()).ok, true); assert.equal(mutations, 1, 'cold phone duplicate cannot resend')
assert.equal((await invoke({...command, text: 'Different payload'})).ok, false); assert.equal(mutations, 1)
assert.equal((await invoke({...command, session: 'other-session', id: 'wrong-session'})).ok, false)
assert.equal((await invoke({...command, id: 'wrong-token'}, 'other-token')).ok, false)
assert.equal((await invoke({...command, id: 'hidden-job', threadId: 'secret-job'})).ok, false)
assert.equal((await invoke({...command, id: 'phone-local', threadId: 'phone-local-job'})).ok, false)
assert.equal((await invoke({...command, id: 'extra-field', url: 'https://untrusted.example.test'})).ok, false)
const refreshed = await invoke({...command, id: 'refresh-one', action: 'refresh'})
assert.equal(refreshed.ok, true); assert.equal(refreshed.snapshot?.jobs.length, 1)
assert.doesNotMatch(JSON.stringify(refreshed), /PRIVATE_|other-token|phone_x/)
assert.equal((await invoke({...command, id: 'audio-start-one', action: 'audio-start', itemId: 'assistant-one'})).ok, true)
assert.equal(audioStarts, 1)
assert.equal((await invoke({...command, id: 'audio-start-one', action: 'audio-start', itemId: 'assistant-one'})).ok, true)
assert.equal(audioStarts, 1)
authorized = false
assert.equal((await invoke()).ok, false, 'revoked bearer cannot replay cached result')
assert.equal((await invoke({...command, action: 'audio-next', id: 'audio-next-one', speechId: 'speech-one', sequence: 0})).audio, undefined)
authorized = true; now = 50_000
assert.equal((await invoke()).ok, false, 'stale duplicate cannot replay cached response')
now = 100; status = null
assert.equal((await invoke({...command, id: 'missing-caps'})).ok, false, 'missing capabilities default deny')
status = {scope: 'paired', role: 'operator', capabilities, engine: {state: 'loading', model: 'example-model'}}
snapshot.activeThreadIds = []
assert.equal((await invoke({...command, id: 'model-not-ready'})).ok, false)
snapshot.activeThreadIds = ['job-one']; status = {scope: 'paired', role: 'operator', capabilities, engine: null}
revokedOnRead = true
assert.equal((await invoke({...command, id: 'revoke-during-read'})).ok, false); assert.equal(mutations, 1)
revokedOnRead = false; authorized = true; failDispatch = true
const uncertainCommand = {...command, id: 'uncertain-send'}
assert.equal((await invoke(uncertainCommand)).uncertain, true)
assert.equal((await invoke(uncertainCommand)).uncertain, true); assert.equal(mutations, 2, 'uncertain dispatch never automatically retries')
await service.stop()
assert.equal((await invoke({...command, id: 'after-stop'})).ok, false)
console.log('Host watch route: no React dependency, authenticated scope/session/freshness, redaction, per-session at-most-once and uncertain mutation fences passed')
