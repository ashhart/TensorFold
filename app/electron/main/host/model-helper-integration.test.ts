import * as documentCollections from '../../../shared/document-collections'
import { selectedResponseLocale } from '../../agent/response-language'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import * as fit from '../../../shared/model-fit'
import { isExternalProvider, classifyProvider } from '../../../shared/provider-trust'
import { runModelHelper, validateModelHelperRequest } from '../../agent/model-helper'
import { normalizeStoredHelperDefaults } from '../../store/model-helper-defaults'
import * as credentialPrivacy from '../../agent/provider-credential-privacy'
import * as boundedResponse from '../../agent/bounded-response'
import * as turnStats from '../../agent/turn-stats'

function load(file: string, overrides: Record<string, any>) {
  const source = readFileSync(file, 'utf8'), ast = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true), imports: Record<string, any> = {}
  for (const node of ast.statements) if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
    const name = node.moduleSpecifier.text
    imports[name] = new Proxy({ __esModule: true }, { get(object, key) { if (key in object) return (object as any)[key]; return () => { throw new Error(`Unexpected synthetic helper adapter: ${name}.${String(key)}`) } } })
  }
  Object.assign(imports, overrides)
  const module = { exports: {} as any }, output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports); return imports[id] }, module, module.exports)
  return module.exports
}
const provider: any = { id: 'helper-local', name: 'Synthetic local helper', kind: 'local', transport: 'chat-completions', source: 'custom', baseUrl: 'http://127.0.0.1:12345/v1', apiKey: '', models: ['helper-model'] }
const route = { providerId: provider.id, accountId: null, model: 'helper-model' }
const input: any = { requestId: 'synthetic-helper-1', threadId: 'helper-thread', role: 'naming', text: 'Only supplied synthetic source.' }
assert.deepEqual(normalizeStoredHelperDefaults({ naming: route, document_extraction: { ...route, apiKey: 'forbidden' }, search: { ...route, model: '\u0000bad' } }), { naming: route })
for (const bad of [{ ...input, role: 'unknown' }, { ...input, apiKey: 'forbidden' }, { ...input, text: 'x'.repeat(20_001) }, { ...input, requestId: 'short' }]) assert.throws(() => validateModelHelperRequest(bad))

let mode = 'normal', requests: any[] = [], nativeStarts: boolean[] = [], nativeReleases = 0, cancelCalls = 0, finishCancel: (() => void) | null = null, beforeResponse: (() => void) | null = null
const frame = (delta: any, finish?: string) => `data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish ?? null }] })}\n\n`
const fetchFixture = async (_url: string, init: RequestInit) => {
  requests.push(JSON.parse(String(init.body)))
  beforeResponse?.()
  if (mode.startsWith('json-')) return new Response(JSON.stringify({ choices: [{ message: { content: 'Synthetic JSON helper result.', ...(mode.includes('tools') ? { tool_calls: [{ id: 'unexpected-json', type: 'function', function: { name: 'write_file', arguments: '{}' } }] } : {}) }, ...(mode === 'json-stop' ? { finish_reason: 'stop' } : mode === 'json-length' ? { finish_reason: 'length' } : {}) }] }), { headers: { 'content-type': 'application/json' } })
  if (mode === 'drain') return new Response(new ReadableStream<Uint8Array>({ cancel() { cancelCalls++; return new Promise<void>((resolve) => { finishCancel = resolve }) } }), { headers: { 'content-type': 'text/event-stream' } })
  const body = mode === 'unfinished' ? frame({ content: 'Unfinished synthetic output.' }) : mode === 'length' ? frame({ content: 'Truncated synthetic output.' }, 'length') : mode === 'oversize' ? frame({ content: 'x'.repeat(12_001) }, 'stop') : mode === 'empty' ? frame({}, 'stop') : mode === 'tools' ? frame({ content: 'A synthetic answer.', tool_calls: [{ index: 0, id: 'unexpected', type: 'function', function: { name: 'write_file', arguments: '{}' } }] }, 'stop') : frame({ content: 'A concise synthetic title.' }, 'stop')
  return new Response(body + (mode === 'unfinished' ? '' : 'data: [DONE]\n\n'), { headers: { 'content-type': 'text/event-stream' } })
}
const engineRoute = async (params: any, options?: { start?: boolean }) => {
  if (params.provider?.source !== 'tensorfold') return null
  nativeStarts.push(options?.start !== false)
  return { params: { ...params, baseUrl: provider.baseUrl, apiKey: '' }, release: () => nativeReleases++ }
}
const { streamChat } = load('electron/agent/openai.ts', {
  './secure-fetch': { createProviderFetch: () => fetchFixture, createLocalProviderFetch: () => fetchFixture }, './engine/provider': { engineRoute, recordEnginePerformance() {} },
  './planning-guard': { planningGuard: () => null, needsPlanningGuard: () => false }, './opencode-engine': { isOpenCodeProtocol: () => false },
  './provider-credential-privacy': credentialPrivacy, './bounded-response': boundedResponse, './turn-stats': turnStats, '../../shared/provider-trust': { classifyProvider },
})
const { qualificationStream } = load('electron/agent/model-qualification/transport.ts', { '../openai': { streamChat }, '../engine/provider': { engineRoute } })

let profile = '/synthetic/helper-one', role = 'operator', leased = 0, qualified = true, reportsGate: Promise<void> | null = null, sourceEmpty = false
const thread: any = { id: input.threadId, title: 'Original title', timeline: [{ kind: 'user', text: 'Never automatically copy whole conversation history.' }] }
const store: any = { settings: { documents: { sources: [{ id: 'one', path: '/synthetic/source', enabled: true, kind: 'dir' }] }, modelHelperDefaults: { naming: route, document_extraction: route, search: route } }, threads: [thread] }
const handlers = new Map<string, any>(), sources = [{ documentId: 'synthetic-doc', passageId: 'synthetic-passage', revision: 'a'.repeat(64) }]
const { registerModelHelperRuntime, resolveConfiguredHelper } = load('electron/main/host/model-helper-runtime.ts', {
  '../../../shared/document-collections': documentCollections,
  '../../agent/response-language': { selectedResponseLocale },
  '../../../shared/model-fit': fit, '../../../shared/provider-trust': { isExternalProvider }, '../../agent/model-helper': { validateModelHelperRequest, runModelHelper }, '../../agent/model-qualification/transport': { qualificationStream }, '../state': { store },
  './document-library-runtime': { documentLibrary: () => ({ search: async (_query: any, allowed: string[] | undefined) => ({ passages: sourceEmpty || allowed?.length === 0 ? [] : [{ ...sources[0], relativePath: 'source.txt', text: 'Host-verified synthetic source facts.' }] }) }) },
  './model-qualification-runtime': { resolveQualificationRoute: () => { const capturedProfile = profile, captured = JSON.stringify(provider); return { provider, model: route.model, accountId: null, assertCurrent() { if (profile !== capturedProfile || JSON.stringify(provider) !== captured) throw new Error('Synthetic qualification route changed.') } } }, modelQualificationService: () => ({ reports: async () => { await reportsGate; return qualified ? [{ ...route, key: fit.modelQualificationKey(route), probeRevision: fit.MODEL_PROBE_REVISION, revision: 1, checkedAt: 1, checks: { ...fit.emptyModelChecks(), text: { state: 'observed', message: 'Synthetic qualification' } } }] : [] } }) },
  './ipc': { handle: (name: string, callback: any) => handlers.set(name, callback) }, './audit': { requireSecurityRole: () => { if (role !== 'operator') throw new Error('Synthetic helper role revoked.') } },
})
registerModelHelperRuntime({ reserveOperation: () => { leased++; return { release: () => leased-- } }, assertIdle() {} })
const run = (request = input) => handlers.get('modelHelper:run')({}, request)
let result = await run()
assert.equal(result.complete, true); assert.equal(result.route.model, route.model); assert.equal(thread.title, 'Original title', 'A naming result never renames a conversation automatically')
assert.equal(requests[0].max_tokens, 768); assert.equal(requests[0].temperature, 0); assert.equal(requests[0].tools, undefined); assert.equal(requests[0].tool_choice, undefined)
assert.equal(requests[0].messages.length, 2); assert.equal(requests[0].messages[1].content, input.text); assert.ok(!JSON.stringify(requests[0]).includes(thread.timeline[0].text)); assert.equal(leased, 0)
sourceEmpty = true
const beforeEmpty = requests.length
await assert.rejects(run({ ...input, role: 'search' }), /No current document/); assert.equal(requests.length, beforeEmpty); assert.equal(leased, 0)
sourceEmpty = false
result = await run({ ...input, role: 'search' }); assert.deepEqual(result.sources, sources); assert.match(requests.at(-1).messages[1].content, /untrusted data/)
const beforeEmptyPolicy = requests.length
thread.documentSourceIds = []
await assert.rejects(run({ ...input, role: 'search' }), /No current document/)
assert.equal(requests.length, beforeEmptyPolicy, 'An explicit empty document collection policy cannot call the model with global library passages')
thread.documentSourceIds = ['one']
const documentResolved = await resolveConfiguredHelper('search', thread.id)
thread.documentSourceIds = []
assert.throws(documentResolved.assertCurrent, /boundary changed/)
thread.documentSourceIds = ['one']
const sourceResolved = await resolveConfiguredHelper('search', thread.id)
store.settings.documents.sources[0].enabled = false
assert.throws(sourceResolved.assertCurrent, /boundary changed/)
store.settings.documents.sources[0].enabled = true
delete thread.documentSourceIds
for (const failure of ['length', 'empty', 'oversize', 'json-unfinished', 'json-length']) { mode = failure; await assert.rejects(run(), /finish|missing|bound|oversized/); assert.equal(leased, 0) }
mode = 'json-stop'; assert.equal((await run()).complete, true)
mode = 'normal'; qualified = false; const beforeQualification = requests.length
await assert.rejects(run(), /capability checks/); assert.equal(requests.length, beforeQualification); qualified = true
const previousDefaults = store.settings.modelHelperDefaults
store.settings.modelHelperDefaults = {}; await assert.rejects(run(), /helper model/); store.settings.modelHelperDefaults = previousDefaults
for (const mutation of [{ kind: 'cloud' }, { baseUrl: 'http://192.0.2.1:12345/v1' }, { transport: 'codex-cli' }]) { const captured = { ...provider }, count = requests.length; Object.assign(provider, mutation); await assert.rejects(run(), /external provider/); assert.equal(requests.length, count); Object.assign(provider, captured) }
for (const key of ['bot', 'phoneLocalOnly', 'googleLocalOnly', 'orchestratorHostId']) { thread[key] = key === 'bot' ? {} : true; await assert.rejects(run(), /ordinary local conversation/); delete thread[key] }
let resolveReports!: () => void
reportsGate = new Promise<void>((resolve) => { resolveReports = resolve })
const changing = run(); store.settings.modelHelperDefaults = { ...previousDefaults, naming: { ...route, model: 'changed-default' } }; resolveReports()
await assert.rejects(changing, /boundary changed/); store.settings.modelHelperDefaults = previousDefaults; reportsGate = null
const resolved = await resolveConfiguredHelper('naming', thread.id)
profile = '/synthetic/helper-two'; assert.throws(resolved.assertCurrent, /route changed/); profile = '/synthetic/helper-one'
role = 'viewer'; assert.throws(resolved.assertCurrent, /role revoked/); role = 'operator'

reportsGate = new Promise<void>((resolve) => { resolveReports = resolve })
const beforeProfile = requests.length, preflight = run(); profile = '/synthetic/helper-drift'; resolveReports()
await assert.rejects(preflight, /route changed/); assert.equal(requests.length, beforeProfile); profile = '/synthetic/helper-one'; reportsGate = null
reportsGate = new Promise<void>((resolve) => { resolveReports = resolve })
const beforeRole = requests.length, unauthorized = run(); role = 'viewer'; resolveReports()
await assert.rejects(unauthorized, /role revoked/); assert.equal(requests.length, beforeRole); role = 'operator'; reportsGate = null

for (const drift of ['role', 'profile', 'default', 'provider', 'thread']) {
  const count = requests.length, oldUrl = provider.baseUrl
  beforeResponse = () => {
    if (drift === 'role') role = 'viewer'
    if (drift === 'profile') profile = '/synthetic/helper-stream-drift'
    if (drift === 'default') store.settings.modelHelperDefaults = { ...previousDefaults, naming: { ...route, model: 'changed-stream-default' } }
    if (drift === 'provider') provider.baseUrl = 'http://127.0.0.1:12346/v1'
    if (drift === 'thread') thread.phoneLocalOnly = true
  }
  await assert.rejects(run(), /changed|role revoked/)
  assert.equal(requests.length, count + 1, 'Late drift rejects the response without a second request or fallback')
  assert.equal(leased, 0)
  beforeResponse = null; role = 'operator'; profile = '/synthetic/helper-one'; store.settings.modelHelperDefaults = previousDefaults; provider.baseUrl = oldUrl; delete thread.phoneLocalOnly
}
mode = 'drain'
let settled = false
const draining = run(); draining.then(() => { settled = true }, () => { settled = true })
for (let i = 0; i < 30 && !leased; i++) await Promise.resolve()
for (let i = 0; i < 40; i++) await Promise.resolve()
await handlers.get('modelHelper:cancel')({}, input.requestId)
for (let i = 0; i < 40 && !finishCancel; i++) await Promise.resolve()
assert.equal(cancelCalls, 1); assert.equal(settled, false); assert.equal(leased, 1, 'Cancellation keeps admission closed until the actual transport cleanup drains')
await assert.rejects(run({ ...input, requestId: 'different-synthetic-helper' }), /already running/)
assert.ok(finishCancel); (finishCancel as () => void)()
await assert.rejects(draining, /abort|cancel/i); assert.equal(leased, 0)
mode = 'normal'
provider.source = 'tensorfold'; result = await run(); assert.deepEqual(nativeStarts, [false]); assert.equal(nativeReleases, 1); provider.source = 'custom'
mode = 'unfinished'
await assert.rejects(run(), /complete finish|unfinished/, 'A transport EOF without an explicit provider finish must not become a finished helper result')
assert.equal(leased, 0)
mode = 'tools'
await assert.rejects(run(), /tool/i, 'An unsolicited tool call cannot be accepted as a finished tool-free helper response')
assert.equal(leased, 0)
console.log('Actual helper runtime and OpenAI request path pass local qualification/default/privacy boundaries, no tools/fallback/history, 768-token request limit and explicit completion bounds.')

mode = 'json-tools-unfinished'
let unconfirmedTools = 0
await assert.rejects(async () => {
  for await (const chunk of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: '', model: route.model, messages: [{ role: 'user', content: 'Synthetic strict response.' }], tools: [], maxOutputTokens: 768, requireProviderFinish: true, signal: new AbortController().signal })) if (chunk.type === 'tool_calls') unconfirmedTools++
}, /unfinished/)
assert.equal(unconfirmedTools, 0, 'Strict JSON rejects missing finish before exposing any tool calls')
mode = 'unfinished'
const legacyChunks: any[] = []
for await (const chunk of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: '', model: route.model, messages: [{ role: 'user', content: 'Synthetic legacy response.' }], signal: new AbortController().signal })) legacyChunks.push(chunk)
assert.ok(legacyChunks.some((chunk) => chunk.type === 'finish' && chunk.reason === 'stop'), 'Strict completion remains opt-in for legacy callers')
mode = 'normal'

const { runAgent } = await import('../../agent/loop')
const previousFetch = globalThis.fetch
const nativeRecovery: any[] = [], nativeEvents: any[] = []
let nativeRequests = 0
const partial = 'A preserved partial native response must not be replayed. '.repeat(6)
const nativeThread: any = { id: 'synthetic-strict-native', title: 'Synthetic native chat', kind: 'chat', incognito: true, cwd: '/synthetic/unused-chat', providerId: provider.id, model: route.model, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Answer once with the supplied fixture.' }], timeline: [] }
globalThis.fetch = (async (destination: any) => { assert.equal(String(destination?.url ?? destination), provider.baseUrl + '/chat/completions'); nativeRequests++; return new Response(frame({ content: nativeRequests === 1 ? partial : 'An invisible replay would be a failure.' }, nativeRequests === 1 ? undefined : 'stop') + (nativeRequests === 1 ? '' : 'data: [DONE]\n\n'), { headers: { 'content-type': 'text/event-stream' } }) }) as typeof fetch
try {
  await runAgent({ thread: nativeThread, provider, model: route.model, runId: 'native-current', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'none', maxSteps: 1, autoCompact: false, signal: new AbortController().signal, persist() {}, requestApproval: async () => 'deny', onRecovery: (snapshot: any) => nativeRecovery.push(snapshot), emit(event: any) {
    nativeEvents.push(event)
    if (event.type === 'item_add') nativeThread.timeline.push(event.item)
    if (event.type === 'item_update') Object.assign(nativeThread.timeline.find((item: any) => item.id === event.id) ?? {}, event.patch)
    if (event.type === 'item_delta') { const item = nativeThread.timeline.find((item: any) => item.id === event.id); if (item) item.text = (item.text ?? '') + event.chunk }
  } })
  assert.equal(nativeRequests, 1, 'An actual native EOF after partial output never starts a hidden second request')
  assert.ok(nativeEvents.some((event) => event.type === 'item_delta')); assert.ok(nativeThread.timeline.some((item: any) => item.kind === 'assistant' && item.text?.includes('preserved partial native response')))
  assert.equal(nativeRecovery.at(-1)?.state, 'needs_review'); assert.equal(nativeRecovery.at(-1)?.reason, 'partial_response'); assert.equal(nativeRecovery.at(-1)?.retry, 'requires_review'); assert.equal(nativeRecovery.at(-1)?.pendingOperations, 0)
  console.log('Actual native runAgent plus fake fetch preserves unfinished EOF text, settles its tracker and refuses hidden replay.')
} finally { globalThis.fetch = previousFetch }

const { createRemoteBrowserApi, PairedSessionError } = await import('../../../src/lib/remote-transport')
let helperRemoteCalls = 0
const paired = createRemoteBrowserApi({ token: 'synthetic-helper-token', fetchImpl: (async () => { helperRemoteCalls++; throw new Error('No helper network route is permitted.') }) as typeof fetch, openEventStream: () => ({ onEvent() {}, close() {} }) })
await assert.rejects(paired.modelHelperRun(input), PairedSessionError)
await assert.rejects(paired.modelHelperCancel(input.requestId), PairedSessionError)
assert.equal(helperRemoteCalls, 0)
console.log('Actual paired transport refuses both helper methods before fetch or demo fallback.')
