import assert from 'node:assert/strict'

// Use the real store, run-routing helpers and browser transport, replacing only host I/O.
const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem() {}, removeItem() {} }
globals.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
await (await import('../../browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../../ipc')
const { useStore } = await import('../index')
const settings = await api.getSettings()
const provider: any = { id: 'synthetic-send-local', kind: 'local', name: 'Synthetic local API', models: ['fixture'], baseUrl: 'http://127.0.0.1:9896/v1', apiKey: '' }
function thread(id: string, hostId?: string): any { return { id, title: id, kind: 'chat', cwd: '', status: 'idle', providerId: provider.id, model: 'fixture', timeline: [], apiMessages: [], createdAt: 1, updatedAt: 1, ...(hostId ? { orchestratorHostId: hostId } : {}) } }
function reset(hostId?: string) { useStore.setState({ settings: { ...settings, providers: [provider], activeProviderId: provider.id, activeModel: 'fixture', defaultExecutionTarget: 'local' }, threads: [thread('a', hostId), thread('b')], projects: [], status: {}, view: { name: 'thread', id: 'a' }, notice: null }) }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((yes) => { resolve = yes }); return { promise, resolve } }
const original = { startRun: api.startRun, cueRun: api.cueRun, startRemoteOrchestratorThread: api.startRemoteOrchestratorThread, guideRemoteOrchestratorThread: api.guideRemoteOrchestratorThread }
const starts: any[] = [], guides: any[] = []
const file: any = { id: 'file', name: 'reviewed.txt', mime: 'text/plain', kind: 'document', text: 'Reviewed content', status: 'ready' }
try {
  reset()
  let admitted = deferred<{ runId: string }>()
  api.startRun = async (payload) => { starts.push({ host: null, payload }); return admitted.promise }
  const pending = useStore.getState().send('Captured A', [file], 'a')
  useStore.setState({ view: { name: 'thread', id: 'b' } })
  admitted.resolve({ runId: 'synthetic-a' })
  assert.equal(await pending, true)
  assert.equal(starts[0].payload.threadId, 'a')
  assert.equal(starts[0].payload.userText, 'Captured A')
  assert.deepEqual(starts[0].payload.attachments, [file])
  assert.equal(useStore.getState().status.a, 'thinking')
  assert.equal(useStore.getState().status.b, undefined)
  assert.deepEqual(useStore.getState().view, { name: 'thread', id: 'b' })

  reset('owning-host')
  api.startRemoteOrchestratorThread = async (host, payload) => { starts.push({ host, payload }); return { runId: 'synthetic-remote' } }
  useStore.setState({ view: { name: 'home' } })
  assert.equal(await useStore.getState().send('Remote A despite navigation', undefined, 'a'), true)
  assert.equal(starts.at(-1).host, 'owning-host')
  assert.equal(starts.at(-1).payload.threadId, 'a')
  assert.equal(await useStore.getState().send('Missing target', undefined, 'missing'), false)
  assert.equal(await useStore.getState().send('No captured target from home'), false)

  reset()
  api.startRun = async () => { throw new Error('Synthetic host admission failed') }
  assert.equal(await useStore.getState().send('Keep failed draft', undefined, 'a'), false)
  assert.equal(useStore.getState().status.a, 'error')
  assert.match(useStore.getState().notice!.message, /Synthetic host admission failed/)

  reset(); useStore.setState({ status: { a: 'streaming' } })
  api.cueRun = async (payload) => { guides.push(payload); return { queued: true } as any }
  const before = starts.length
  assert.equal(await useStore.getState().send('Accepted steering', undefined, 'a'), true)
  assert.equal(guides.at(-1).threadId, 'a')
  assert.equal(starts.length, before, 'accepted steering must not also start a new run')
  const guidedBeforeFile = guides.length
  assert.equal(await useStore.getState().send('Keep files until idle', [file], 'a'), false)
  assert.equal(guides.length, guidedBeforeFile, 'text-only steering must never acknowledge or omit a pending file')
  assert.match(useStore.getState().notice!.message, /Files stay in your draft/)

  reset(); useStore.setState({ status: { a: 'streaming' } })
  api.cueRun = async () => { throw new Error('Synthetic steering admission failed') }
  assert.equal(await useStore.getState().send('Keep rejected steering draft', undefined, 'a'), false, 'every failed host admission returns false to its caller')
  assert.match(useStore.getState().notice!.message, /Synthetic steering admission failed/)
  console.log('Actual send routing returns durable acceptance, captures owning chat/host through navigation, preserves failure drafts and handles steering admission')
} finally { Object.assign(api, original) }
