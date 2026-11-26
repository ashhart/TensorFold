import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { comparisonId, comparisonInput } from '../../../agent/engine/model-comparison'
import { isRepoId, isRevision } from '../../../agent/engine/paths'
import { assertCortheonReleaseAllowed } from '../../../../shared/release-feature-policy'
import { normalizeEngineEnvOverrides } from '../../../../shared/engine-advanced-env'
import { enginePreload } from '../../../preload/engine'

// The registered engine handlers run for real against a recording service, audit, role check and native picker.
const secrets = { key: 'sk-synthetic-secret', vector: '/synthetic/private/steering.gguf', folder: '/synthetic/private/models', lesson: 'Synthetic private lesson text.', env: 'synthetic-private-value' }
type Handler = (...args: unknown[]) => Promise<unknown>
const handlers = new Map<string, Handler>()
const roles: string[] = [], events: { name: string; value: unknown }[] = [], calls: string[] = [], dialogs: unknown[][] = []
let allow = true
let pick: { canceled: boolean; filePaths: string[] } = { canceled: false, filePaths: [] }
const record = (name: string, value: unknown) => { calls.push(name); return value }
const vector = (extra: object = {}) => ({ enabled: true, alpha: 0.5, path: secrets.vector, ...extra })
const service = {
  snapshot: () => record('snapshot', {}),
  connection: () => record('connection', { baseUrl: 'http://127.0.0.1:1', apiKey: secrets.key }),
  installed: () => record('installed', []),
  search: () => record('search', []),
  preflight: () => record('preflight', {}),
  pull: () => record('pull', {}),
  cancelPull: () => record('cancelPull', true),
  load: () => record('load', {}),
  setComparisonAdmission: () => record('setComparisonAdmission', undefined),
  compareModels: () => record('compareModels', Promise.resolve({})),
  comparisonStatus: () => record('comparisonStatus', null),
  cancelComparison: () => record('cancelComparison', false),
  tryModel: () => record('tryModel', {}),
  cancelTryModel: () => record('cancelTryModel', false),
  setCortheon: () => record('setCortheon', {}),
  controlVectorSnapshot: () => record('controlVectorSnapshot', vector()),
  chooseControlVector: (file: string) => record(`chooseControlVector:${file}`, vector()),
  setControlVector: () => record('setControlVector', vector()),
  clearControlVector: () => record('clearControlVector', vector({ enabled: false, path: null })),
  learningSnapshot: () => record('learningSnapshot', {}),
  setLearning: () => record('setLearning', { model: 'Fixture/A' }),
  envOverrides: () => record('envOverrides', {}),
  setEnvOverrides: () => record('setEnvOverrides', {}),
  teach: () => record('teach', { model: 'Fixture/A', job: { id: 'job_1' } }),
  cancelTeach: () => record('cancelTeach', {}),
  unload: () => record('unload', {}),
  remove: () => record('remove', []),
  librarySnapshot: () => record('librarySnapshot', {}),
  scanLibrary: () => record('scanLibrary', {}),
  addLibraryFolder: (folder: string) => record(`addLibraryFolder:${folder}`, { folders: [] }),
  removeLibraryFolder: () => record('removeLibraryFolder', {}),
  cancelLibraryScan: () => record('cancelLibraryScan', {}),
}
const imports: Record<string, unknown> = {
  '../../../../shared/release-feature-policy': { assertCortheonReleaseAllowed },
  '../../../../shared/engine-advanced-env': { normalizeEngineEnvOverrides },
  electron: { BrowserWindow: { fromWebContents: () => null }, dialog: { showOpenDialog: async (...args: unknown[]) => { dialogs.push(args); return pick } } },
  '../../../agent/engine/paths': { isRepoId, isRevision },
  '../../../agent/engine/model-comparison': { comparisonId, comparisonInput },
  '../audit': {
    requireSecurityRole(role: string) { roles.push(role); if (!allow) throw new Error('Synthetic role denied') },
    auditEvent: async (name: string, value: unknown) => { events.push({ name, value }) },
  },
  '../ipc': { handle: (channel: string, handler: Handler) => handlers.set(channel, handler) },
}
const loaded = { exports: {} as { registerEngineIpc(engine: () => unknown): void } }
const source = ts.transpileModule(readFileSync(path.join(__dirname, 'engine-ipc.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText
new Function('require', 'module', 'exports', source)((id: string) => { assert.ok(id in imports, `Reviewed main import ${id}`); return imports[id] }, loaded, loaded.exports)
loaded.exports.registerEngineIpc(() => service)
const invoke = (channel: string, ...args: unknown[]) => handlers.get(channel)!({ sender: {} }, ...args)
const reset = () => { roles.length = 0; events.length = 0; calls.length = 0; dialogs.length = 0 }

const repo = 'Fixture/A'
const comparison = { id: 'comparison_ipc_test_1', models: ['Fixture/A', 'Fixture/B'], prompt: 'Synthetic prompt' }
const valid: Record<string, unknown[]> = {
  'engine:snapshot': [], 'engine:connection': [], 'engine:refresh': [], 'engine:search': ['query'],
  'engine:preflight': [{ repo }], 'engine:pull': [{ repo }], 'engine:cancelPull': [repo], 'engine:load': [repo],
  'engine:compareModels': [comparison], 'engine:comparisonStatus': [comparison.id], 'engine:cancelComparison': [comparison.id],
  'engine:tryModel': [repo], 'engine:cancelTryModel': [], 'engine:setCortheon': [false],
  'engine:controlVector': [], 'engine:chooseControlVector': [], 'engine:setControlVector': [{ enabled: true, alpha: 0.5 }], 'engine:clearControlVector': [],
  'engine:learning': [], 'engine:setLearning': [{ enabled: true }], 'engine:envOverrides': [], 'engine:setEnvOverrides': [{ TF_SYNTHETIC_KNOB: secrets.env }],
  'engine:teach': [secrets.lesson], 'engine:cancelTeach': [], 'engine:unload': [], 'engine:delete': [repo],
  'engine:library': [], 'engine:scanLibrary': [undefined], 'engine:addLibraryFolder': [], 'engine:removeLibraryFolder': ['a'.repeat(64)], 'engine:cancelLibraryScan': [],
}
const channels = [...handlers.keys()].sort()
assert.deepEqual(channels, Object.keys(valid).sort(), 'every registered engine channel has a case here, so a new channel cannot skip the checks below')
for (const channel of ['engine:controlVector', 'engine:chooseControlVector', 'engine:setControlVector', 'engine:clearControlVector']) assert.ok(handlers.has(channel), `${channel} is registered`)

// Every channel checks the caller's role before it touches the service, picker or audit; delete needs the owner.
for (const channel of channels) {
  reset(); allow = false
  await assert.rejects(invoke(channel, ...valid[channel]), /role denied/, `${channel} refuses a caller below its role`)
  assert.deepEqual(roles, [channel === 'engine:delete' ? 'owner' : 'operator'], `${channel} asks for exactly its role`)
  assert.deepEqual([calls, events, dialogs], [[], [], []], `${channel} does nothing before the role check passes`)
}
allow = true

// Allowed callers: nothing private reaches an audit event, and the connection and the picked paths are never audited.
pick = { canceled: false, filePaths: [secrets.vector] }
for (const channel of channels) {
  reset()
  if (channel === 'engine:addLibraryFolder') pick = { canceled: false, filePaths: [secrets.folder] }
  await invoke(channel, ...valid[channel])
  assert.equal(roles.length, 1, `${channel} checks the role once`)
  assert.ok(!JSON.stringify(events).match(new RegExp(Object.values(secrets).map((v) => v.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')).join('|'))), `${channel} keeps keys, paths, lessons and values out of the audit`)
  if (['engine:connection', 'engine:chooseControlVector', 'engine:addLibraryFolder'].includes(channel)) assert.deepEqual(events, [], `${channel} writes no audit record`)
}
reset(); await invoke('engine:setControlVector', { enabled: true, alpha: 0.5 })
assert.deepEqual(events, [{ name: 'engine.controlVector', value: { enabled: true, alpha: 0.5 } }], 'the control vector audit holds the state and strength, never the file')
reset(); await invoke('engine:clearControlVector')
assert.deepEqual(events, [{ name: 'engine.controlVector', value: { enabled: false, cleared: true } }], 'clearing audits the state alone')
reset(); await invoke('engine:setEnvOverrides', { TF_SYNTHETIC_KNOB: secrets.env })
assert.deepEqual(events, [{ name: 'engine.envOverrides', value: { keys: ['TF_SYNTHETIC_KNOB'] } }], 'only variable names are audited')
reset(); await invoke('engine:teach', secrets.lesson)
assert.deepEqual(events, [{ name: 'engine.teach', value: { model: 'Fixture/A', jobId: 'job_1' } }], 'lessons are never audited')

// The native picker is the only route by which a path reaches the service; renderer arguments are ignored.
const evil = '/renderer/supplied/path'
for (const [channel, method, pickPath] of [['engine:chooseControlVector', 'chooseControlVector', secrets.vector], ['engine:addLibraryFolder', 'addLibraryFolder', secrets.folder]] as const) {
  reset(); pick = { canceled: false, filePaths: [pickPath] }
  await invoke(channel, { path: evil }, evil)
  assert.deepEqual(calls, [`${method}:${pickPath}`], `${channel} hands the service the picked path only`)
  assert.equal(dialogs.length, 1, `${channel} opens the native picker`)
  assert.ok(!JSON.stringify(calls).includes(evil), `${channel} never forwards a renderer path`)
  for (const refused of [{ canceled: true, filePaths: [pickPath] }, { canceled: false, filePaths: [] }, { canceled: false, filePaths: [pickPath, evil] }]) {
    reset(); pick = refused
    await invoke(channel, evil)
    assert.ok(!calls.some((c) => c.startsWith(`${method}:`)), `${channel} admits no path unless the picker returns exactly one`)
  }
}

// Renderer input is re-validated in the main process.
reset()
await assert.rejects(invoke('engine:setCortheon', 'yes'), /on or off/)
await assert.rejects(invoke('engine:setLearning', { enabled: 'yes' }), /on or off/)
await assert.rejects(invoke('engine:teach', 'x'.repeat(8193)), /up to 8 KB/)
await assert.rejects(invoke('engine:teach', '   '), /up to 8 KB/)
await assert.rejects(invoke('engine:removeLibraryFolder', '/synthetic/path'), /saved model folder/)
await assert.rejects(invoke('engine:scanLibrary', { path: '/synthetic/path' }), /scan options/)
assert.deepEqual([calls, events], [[], []], 'refused input reaches neither the service nor the audit')

// The preload bridge reaches every registered channel.
const reached = new Set<string>()
const bridge = enginePreload(async (channel) => { reached.add(channel) }, (channel) => { reached.add(channel); return () => {} })
for (const [name, fn] of Object.entries(bridge)) if (name !== 'onEngineEvent') await (fn as (...args: unknown[]) => Promise<unknown>)(repo, repo)
assert.equal(typeof bridge.onEngineEvent(() => {}), 'function')
for (const channel of channels) assert.ok(reached.has(channel), `preload exposes ${channel}`)

console.log('Actual engine IPC handlers: role gates, audit privacy, picker-only paths, input refusal and preload coverage passed.')
