import assert from 'node:assert/strict'
import { createWatchRelay, watchSnapshot, type WatchCommand } from '../mobile/src/lib/watch-relay'
import { RESTRICTED_CAPABILITIES, type MobileHost, type StudioClient } from '../mobile/src/lib/client'

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((yes) => { resolve = yes }); return {promise, resolve} }
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve() }
const host: MobileHost = {snapshot: {serverName: 'Synthetic host', appVersion: '1', generatedAt: 1, activeThreadIds: [], projects: [], work: {approvals: {pending: 0}, jobs: {running: 0, queued: 0}}, threads: [
  {id: 'visible-job', title: 'Visible', kind: 'task', providerId: 'local', model: 'fixture', status: 'idle', updatedAt: 1, timeline: [{id: 'assistant-reply', kind: 'assistant', text: 'Allowed reply'}]},
  {id: 'private-job', title: 'INCOGNITO_TITLE_PRIVATE', kind: 'task', providerId: 'local', model: 'fixture', status: 'idle', updatedAt: 1, timeline: [], incognito: true},
  {id: 'local-only-job', title: 'PHONE_LOCAL_TITLE_PRIVATE', kind: 'task', providerId: 'local', model: 'fixture', status: 'idle', updatedAt: 1, timeline: [], phoneLocalOnly: true},
]}, status: {scope: 'paired', role: 'operator', capabilities: {...RESTRICTED_CAPABILITIES, taskStart: true, taskGuide: true}, engine: null}}
const projected = JSON.stringify(watchSnapshot(host, 'paired-session', 100))
assert.ok(!projected.includes('INCOGNITO_TITLE_PRIVATE'))
assert.ok(!projected.includes('PHONE_LOCAL_TITLE_PRIVATE'))

for (const action of ['audio-next', 'audio-start', 'audio-stop', 'reply'] as const) {
  for (const invalidation of ['session', 'client', 'background', 'disable'] as const) {
    const gate = deferred<any>(); let entered = false, identity = 'paired-session', foreground = true
    const cancellations: string[] = []
    const client = {
      watchSpeechStatus: async () => ({available: true, provider: 'kokoro', reason: null}),
      readWatchSpeech: async () => {entered = true; return gate.promise},
      startWatchSpeech: async () => {entered = true; return gate.promise},
      cancelWatchSpeech: async (id: string) => {cancellations.push(id); if (action === 'audio-stop') { entered = true; return gate.promise } return {cancelled: true}},
      send: async () => {entered = true; return gate.promise},
    } as unknown as StudioClient
    let currentClient = client
    const relay = createWatchRelay({now: () => 100, session: () => identity, foreground: () => foreground, client: () => currentClient, host: async () => host, publish() {}})
    const command: WatchCommand = {version: 1, id: `adversarial-${action}-${invalidation}`, action, session: identity, generatedAt: 100,
      threadId: 'visible-job', itemId: 'assistant-reply', speechId: 'speech-session', sequence: 0, text: 'Please continue'}
    const pending = relay.handle(command); await settle(); assert.equal(entered, true)
    if (invalidation === 'session') identity = 'different-session'
    if (invalidation === 'disable') identity = ''
    if (invalidation === 'client') currentClient = {} as StudioClient
    if (invalidation === 'background') foreground = false
    gate.resolve(action === 'audio-next' ? {speechId: 'speech-session', sequence: 0, state: 'audio', sampleRate: 24_000, audioBase64: 'PRIVATE_PCM'}
      : action === 'audio-start' ? {speechId: 'speech-session', provider: 'kokoro', sampleRate: 24_000} : undefined)
    const result = await pending
    assert.equal(result.ok, false, `${action} must be fenced after ${invalidation} during host I/O`)
    assert.equal(result.audio, undefined); assert.equal(result.speech, undefined); assert.equal(result.snapshot, undefined)
    assert.ok(!JSON.stringify(result).includes('PRIVATE_PCM'))
    if (action === 'audio-start') assert.deepEqual(cancellations, ['speech-session'], 'old-owner worker is cancelled through its captured client')
    if (action === 'reply') assert.equal(result.uncertain, true, 'dispatch may have been accepted and must never silently retry')
  }
}
console.log('Watch relay independently rejects sixteen late account/client/background/disable completions and excludes private task metadata')
