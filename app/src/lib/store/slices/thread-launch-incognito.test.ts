import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'
const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
const { api } = await import('../../ipc')
const { useStore } = await import('../index')
const { createRemoteBrowserApi } = await import('../../remote-transport')
const settings = await api.getSettings()
const provider: any = { id: 'synthetic-local', kind: 'local', name: 'Synthetic local API', models: ['synthetic'], baseUrl: 'http://127.0.0.1:9896/v1', apiKey: '' }
const original = { createThread: api.createThread, patchThread: api.patchThread, configureThread: api.configureThread, startRun: api.startRun, createRemoteOrchestratorThread: api.createRemoteOrchestratorThread }
const creates: any[][] = [], starts: any[] = [], sequence: string[] = []
const createdById = new Map<string, any>()
let patches = 0, remoteCreates = 0
useStore.setState({ settings: { ...settings, providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', defaultExecutionTarget: 'local' }, projects: [], threads: [], status: {}, view: { name: 'home' } })
api.createThread = async (...args) => {
  creates.push(args)
  const thread = await original.createThread(...args)
  createdById.set(thread.id, thread); sequence.push(`create:${thread.incognito === true}`)
  return thread
}
api.patchThread = async () => { patches++; throw new Error('Private launch must not rely on a post-create privacy patch.') }
api.configureThread = async (id) => { sequence.push(`configure:${createdById.get(id)?.incognito === true}`) }
api.startRun = async (payload) => { starts.push(payload); sequence.push(`start:${useStore.getState().threads.find((thread) => thread.id === payload.threadId)?.incognito === true}`); return { runId: 'synthetic' } }
api.createRemoteOrchestratorThread = async () => { remoteCreates++; throw new Error('A private launch must not reach a remote host.') }
try {
  const privateLaunch = await useStore.getState().launch({ text: 'Synthetic private request', cwd: '/synthetic', providerId: provider.id, model: 'synthetic', approvalMode: 'suggest', persistActiveModel: false, incognito: true })
  assert.equal(privateLaunch.ok, true)
  assert.equal(creates[0][4], true, 'the actual store/helper forwards privacy on the first create call')
  assert.equal(typeof creates[0][3], 'string', 'the launch carries an idempotency command before IPC')
  assert.equal(useStore.getState().threads.find((thread) => thread.id === privateLaunch.threadId)?.incognito, true, 'actual mock API creates the record private, with no later patch')
  assert.equal(patches, 0)
  assert.equal(starts.length, 1)
  assert.deepEqual(sequence, ['create:true', 'configure:true', 'start:true'], 'all later model setup/admission sees a thread born private')
  const ordinary = await useStore.getState().launch({ text: 'Synthetic ordinary request', cwd: '/synthetic', providerId: provider.id, model: 'synthetic', approvalMode: 'suggest', persistActiveModel: false })
  assert.equal(ordinary.ok, true)
  assert.notEqual(creates[1][4], true)
  assert.notEqual(useStore.getState().threads.find((thread) => thread.id === ordinary.threadId)?.incognito, true)
  assert.equal(patches, 0)
  assert.deepEqual(sequence.slice(3), ['create:false', 'configure:false', 'start:false'])

  const state = useStore.getState()
  useStore.setState({ settings: { ...state.settings!, defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'synthetic-remote', remoteOrchestrators: [{ id: 'synthetic-remote', enabled: true, name: 'Synthetic remote' } as any] } })
  await assert.rejects(() => useStore.getState().launch({ text: 'Private draft retained by caller', cwd: '/synthetic', providerId: provider.id, model: 'synthetic', approvalMode: 'suggest', persistActiveModel: false, incognito: true }), /desktop app|This Mac/)
  assert.equal(remoteCreates, 0, 'remote selection is refused before transmitting the private creation')
  assert.equal(creates.length, 2)

  // Execute the exact preload arrow to verify the native wire payload without Electron UI.
  const preload = await readFile(new URL('../../../../electron/preload.ts', import.meta.url), 'utf8')
  const tree = ts.createSourceFile('preload.ts', preload, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS)
  let arrow: ts.ArrowFunction | undefined
  const visit = (node: ts.Node) => { if (ts.isPropertyAssignment(node) && ts.isIdentifier(node.name) && node.name.text === 'createThread' && ts.isArrowFunction(node.initializer)) arrow = node.initializer; ts.forEachChild(node, visit) }
  visit(tree); assert.ok(arrow)
  const code = ts.transpileModule(`const create = ${ts.createPrinter().printNode(ts.EmitHint.Expression, arrow, tree)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  const nativeCalls: any[] = []
  const nativeCreate = new Function('ipcRenderer', `${code}\nreturn create;`)({ invoke: (channel: string, payload: unknown) => { nativeCalls.push({ channel, payload }); return Promise.resolve({}) } })
  await nativeCreate('/synthetic', 'task', null, 'synthetic-command', true)
  assert.deepEqual(nativeCalls[0], { channel: 'threads:create', payload: { cwd: '/synthetic', kind: 'task', projectId: null, commandId: 'synthetic-command', incognito: true } })

  const browserRequests: any[] = []
  const remote = createRemoteBrowserApi({ token: 'synthetic-device', fetchImpl: (async (url, init) => { browserRequests.push({ url, body: JSON.parse(String(init?.body ?? '{}')) }); return Response.json({ id: 'ordinary-remote' }) }) as typeof fetch, openEventStream: () => ({ close() {}, onEvent() {} }) })
  await assert.rejects(() => remote.createThread('/synthetic', 'task', null, 'synthetic-private-remote', true), /local desktop app/)
  assert.equal(browserRequests.length, 0, 'paired browser cannot serialize a private creation into the ordinary host API')
  await remote.createThread('/synthetic', 'task', null, 'synthetic-normal-remote')
  assert.equal(browserRequests.length, 1, 'normal remote creation remains available')
  assert.deepEqual(browserRequests[0].body, { cwd: '/synthetic', kind: 'task', projectId: null, commandId: 'synthetic-normal-remote' })
  console.log('Actual store/helper/mock and extracted native preload: first-call privacy with no patch; ordinary launch positive control; remote private creation refuses before network while normal remote creation remains functional.')
} finally {
  Object.assign(api, original)
}
