import assert from 'node:assert/strict'
import type { Thread } from '../../types'
const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
await (await import('../../browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../../ipc')
const { useStore } = await import('../index')
const settings = await api.getSettings()
const provider: any = { id: 'synthetic-local', kind: 'local', name: 'Synthetic local', transport: 'chat-completions', models: ['fixture'], baseUrl: 'http://127.0.0.1:9/v1', apiKey: '' }
function thread(id: string, hostId?: string): Thread { return { id, title: id, kind: 'chat', cwd: '', providerId: provider.id, model: 'fixture', status: 'idle', createdAt: 1, updatedAt: 1, timeline: [], apiMessages: [], memoryPolicy: { useExisting: false, contribute: false }, ...(hostId ? { orchestratorHostId: hostId } : {}) } }
const a = thread('source-a'), b = thread('visible-b'), remote = { ...thread('owned-remote', 'remote-host'), memoryPolicy: { useExisting: true, contribute: true } }
const original = { startRun: api.startRun, cueRun: api.cueRun, startRemoteOrchestratorThread: api.startRemoteOrchestratorThread, guideRemoteOrchestratorThread: api.guideRemoteOrchestratorThread }
let starts: any[] = [], cues: any[] = [], remoteStarts: any[] = [], remoteCues: any[] = []
let startFailure = false, cueFailure = false, cueQueued = true
api.startRun = async (payload) => { starts.push(payload); if (startFailure) throw new Error('Synthetic startup refused'); return { runId: 'run-fixture' } }
api.cueRun = async (payload) => { cues.push(payload); if (cueFailure) throw new Error('Synthetic guide refused'); return { queued: cueQueued, queueDepth: cueQueued ? 1 : 0 } }
api.startRemoteOrchestratorThread = async (hostId, payload) => { remoteStarts.push({ hostId, payload }); return { runId: 'remote-run-fixture' } }
api.guideRemoteOrchestratorThread = async (hostId, payload) => { remoteCues.push({ hostId, payload }); return { queued: true, queueDepth: 1 } }
function reset(status: Record<string, string> = {}, view: any = { name: 'thread', id: b.id }) {
  starts = []; cues = []; remoteStarts = []; remoteCues = []; startFailure = false; cueFailure = false; cueQueued = true
  useStore.setState({ settings: { ...settings, providers: [provider], activeProviderId: provider.id, activeModel: 'fixture', defaultExecutionTarget: 'local', remoteOrchestrators: [{ id: 'remote-host', enabled: true, name: 'Synthetic', baseUrl: 'https://host.example.invalid' } as any] }, projects: [], threads: [a, b, remote], status: status as any, view, notice: null })
}
try {
  reset()
  assert.equal(await useStore.getState().send('Target A while viewing B', undefined, a.id), true)
  assert.equal(starts[0].threadId, a.id)
  assert.equal(useStore.getState().status[b.id], undefined)
  const shown = useStore.getState().view
  assert.equal(shown.name === 'thread' && shown.id, b.id, 'an explicit send does not navigate another chat')
  reset({}, { name: 'home' })
  assert.equal(await useStore.getState().send('No implicit target'), false)
  assert.equal(starts.length, 0)
  assert.equal(await useStore.getState().send('Explicit target offscreen', undefined, a.id), true)
  reset()
  assert.equal(await useStore.getState().send('Implicit visible B'), true)
  assert.equal(starts[0].threadId, b.id)
  assert.equal(await useStore.getState().send('Unknown target', undefined, 'missing'), false)
  assert.equal(starts.length, 1)
  reset(); startFailure = true
  assert.equal(await useStore.getState().send('Retain rejected draft', undefined, a.id), false)
  assert.equal(useStore.getState().status[a.id], 'error')
  assert.match(useStore.getState().notice?.message ?? '', /Synthetic startup refused/)
  reset({ [a.id]: 'streaming' })
  assert.equal(await useStore.getState().send('Accepted guidance', undefined, a.id), true)
  assert.equal(cues[0].threadId, a.id); assert.equal(starts.length, 0)
  reset({ [a.id]: 'streaming' }); cueFailure = true
  assert.equal(await useStore.getState().send('Rejected guidance', undefined, a.id), false, 'busy-guide rejection is a boolean refusal, never an uncaught promise')
  assert.equal(starts.length, 0)
  assert.equal(useStore.getState().status[a.id], 'streaming', 'a failed cue must not claim the still-active model stopped')
  reset({ [a.id]: 'streaming' }); cueQueued = false
  assert.equal(await useStore.getState().send('Run ended before guide', undefined, a.id), true)
  assert.equal(starts[0].threadId, a.id)
  reset({}, { name: 'thread', id: remote.id })
  assert.equal(await useStore.getState().send('Local owner despite visible remote', undefined, a.id), true)
  assert.equal(starts[0].threadId, a.id); assert.equal(remoteStarts.length, 0)
  reset()
  assert.equal(await useStore.getState().send('Remote owner despite visible local', undefined, remote.id), true)
  assert.equal(remoteStarts[0].hostId, 'remote-host'); assert.equal(remoteStarts[0].payload.threadId, remote.id); assert.equal(starts.length, 0)
  reset({ [remote.id]: 'tool' })
  assert.equal(await useStore.getState().send('Remote guidance', undefined, remote.id), true)
  assert.equal(remoteCues[0].hostId, 'remote-host'); assert.equal(remoteCues[0].payload.threadId, remote.id)
  reset({ [a.id]: 'streaming' })
  const image = { id: 'image-fixture', kind: 'image' as const, name: 'sample.png', mime: 'image/png', dataUrl: 'data:image/png;base64,synthetic', status: 'ready' as const }
  assert.equal(await useStore.getState().send('Image must stay in draft', [image], a.id), false, 'a text-only live cue is not acceptance of an image file')
  assert.equal(cues.length, 0); assert.equal(starts.length, 0)
  assert.match(useStore.getState().notice?.message ?? '', /file|finish|draft/i)
  const quote = { id: 'quote', sourceItemId: 'answer', sourceRevision: 'a'.repeat(64), sourceText: 'Synthetic source', text: 'Edited quote' }
  reset(); assert.equal(await useStore.getState().send('Quote reply', undefined, a.id, [quote]), true); assert.deepEqual(starts[0].quotes, [quote])
  reset({ [a.id]: 'streaming' }); assert.equal(await useStore.getState().send('Quotes stay saved', undefined, a.id, [quote]), false); assert.equal(cues.length, 0); assert.equal(starts.length, 0)
  reset(); assert.equal(await useStore.getState().send('No remote quote omission', undefined, remote.id, [quote]), false); assert.equal(remoteStarts.length, 0)
  console.log('Actual send slice: captured target identity, local/remote ownership, acceptance booleans, startup and guide refusal, fallback after ended run, and busy file retention passed')
} finally { Object.assign(api, original) }
