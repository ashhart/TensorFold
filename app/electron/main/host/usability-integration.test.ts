import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

// Execute the production asynchronous recovery callback with inert host adapters.
// This isolates the routing authority seam without starting an app/model/runtime.
function recoveryHandler(environment: Record<string, unknown>) {
  const source = readFileSync('electron/main/host/thread-runtime.ts', 'utf8')
  const file = ts.createSourceFile('thread-runtime.ts', source, ts.ScriptTarget.ES2022, true)
  let callback = ''
  const visit = (node: ts.Node) => {
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'catch' && node.expression.expression.getText(file) === 'runWithRoute(activeRoute)') callback = node.arguments[0].getText(file)
    ts.forEachChild(node, visit)
  }
  visit(file); assert.ok(callback, 'The actual route-recovery callback is reviewed')
  const output = ts.transpileModule(`const handler = ${callback}; return handler;`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  return new Function(...Object.keys(environment), output)(...Object.values(environment)) as (error: Error) => Promise<void>
}
function fixture(recovery: any, overrides: Record<string, unknown> = {}) {
  const calls: string[] = [], emitted: any[] = []
  const provider = { id: 'synthetic-provider', name: 'Synthetic provider', models: ['original', 'alternative'] }
  const thread: any = { id: 'synthetic-thread', title: 'Synthetic thread', timeline: [], runRecovery: recovery }
  const environment: Record<string, unknown> = {
    thread, runId: 'current-run', activeRoute: { provider, providerAccountId: 'account-one', model: 'original' }, baseProvider: provider,
    controller: new AbortController(), availableModelsFromProviderError: () => ['original', 'alternative'],
    markRouteErrorRecovered: () => calls.push('recovered'), rememberProviderModels() {}, persistThreads() {}, emitToRenderer: (event: any) => emitted.push(event), randomUUID: () => 'synthetic-notice', auditModelRetry: async () => {}, runWithRoute: async () => { calls.push('retry') },
    store: { settings: { providers: [provider] } }, modelRouter: { routeAfterError: () => ({ signal: null, route: {}, audit: [] }) },
    threadRouteRole: 'chat', modelRouteAutoFailoverEnabled: () => true, project: null, effectiveContextLimit: () => 8192, budget: 'chat', auditModelRoute: async () => {}, runtimeProviderWithFreshModels: async () => provider, assertProviderEndpointResolved: async () => {}, providerAccountId: 'account-one', modelRouteAccountLabel: () => 'Synthetic account', modelRouteResetPolicy: () => 'stay', addThreadNotice() {}, modelRouteFailoverText: () => 'Synthetic failover', injectThreadRouteHandoff() {},
    ...overrides,
  }
  return { calls, emitted, thread, handle: recoveryHandler(environment) }
}
const stale = fixture({ runId: 'prior-run', retry: 'safe' })
await stale.handle(new Error('Synthetic delegated request acceptance is unknown.'))
assert.equal(stale.calls.filter((call) => call === 'retry').length, 0, 'A previous run safe-retry snapshot must not authorize replay of the current run')
const refused = fixture({ runId: 'current-run', retry: 'safe' })
await refused.handle(new Error('Synthetic explicit pre-dispatch refusal.'))
assert.equal(refused.calls.filter((call) => call === 'retry').length, 1, 'The exact current run pre-dispatch refusal may select a valid model alternative')
for (const retry of ['requires_review', 'unavailable']) {
  const uncertain = fixture({ runId: 'current-run', retry })
  await uncertain.handle(new Error('Synthetic uncertain acceptance.'))
  assert.equal(uncertain.calls.includes('retry'), false)
}
console.log('Actual route-recovery callback binds retry authority to current run evidence and refuses stale/uncertain replay.')

const { RunRecoveryTracker } = await import('../../agent/run-recovery/tracker')
function delegatedHandler(environment: Record<string, unknown>) {
  const source = readFileSync('electron/main/host/thread-runtime.ts', 'utf8'), ast = ts.createSourceFile('runtime.ts', source, ts.ScriptTarget.ES2022, true)
  let body = ''
  const visit = (node: ts.Node) => { if (ts.isIfStatement(node) && node.expression.getText(ast) === "thread.engine === 'omp'") body = node.thenStatement.getText(ast); ts.forEachChild(node, visit) }
  visit(ast); assert.ok(body)
  const output = ts.transpileModule(`return async function runDelegatedFixture() ${body}`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  return new Function(...Object.keys(environment), output)(...Object.values(environment)) as () => Promise<void>
}
for (const fail of [false, true]) {
  const thread: any = { id: 'synthetic-delegated', engine: 'omp', runRecovery: { runId: 'prior-run', retry: 'safe' } }, controller = new AbortController()
  let bridge: any, finishDelegated!: () => void, finishTool!: (value: any) => void, calls = 0, ended = false
  const toolResult = new Promise((resolve) => { finishTool = resolve })
  let bridgeResult!: Promise<any>
  const run = delegatedHandler({ thread, controller, runId: 'delegated-current', aborters: new Map([[thread.id, controller]]), RunRecoveryTracker, assertOmpEngineAllowed() {}, persistThreads() {}, emitToRenderer() {},
    runOmpThread: async (input: any) => { bridge = input; bridgeResult = input.executeTool('write_file', { path: 'synthetic.txt' }); bridgeResult.catch(() => {}); await new Promise<void>((resolve) => { finishDelegated = resolve }); if (fail) throw new Error('Synthetic delegated acceptance unknown.') },
    store: { settings: { reasoningEffort: 'low' }, subAgents: [] }, emit() {}, beatThreadJob() {}, payload: { userText: 'Synthetic task' }, runApprovalMode: 'suggest', drainThreadCues: () => [], approvalResolver: () => async () => 'allow', jobHandle: {},
    randomUUID: () => `synthetic-operation-${calls}`, isReadOnlyTool: () => false, executeTool: () => { calls++; return toolResult }, toolContext: {},
  })
  const pending = run(); pending.then(() => { ended = true }, () => { ended = true })
  await Promise.resolve(); assert.equal(thread.runRecovery.runId, 'delegated-current'); assert.equal(thread.runRecovery.canStartNewRun, false); assert.equal(thread.runRecovery.pendingMutations, 1)
  if (!fail) controller.abort()
  finishDelegated(); for (let i = 0; i < 8; i++) await Promise.resolve()
  assert.equal(ended, false, 'Delegated transport completion never bypasses a bridge mutation drain')
  assert.equal(thread.runRecovery.canStartNewRun, false)
  await assert.rejects(bridge.executeTool('write_file', {}), /stopping|Stop/)
  assert.equal(calls, 1, 'No tool can begin after delegated admission closes')
  finishTool({ ok: true }); await bridgeResult
  if (fail) await assert.rejects(pending, /acceptance unknown/); else await pending
  assert.equal(thread.runRecovery.canStartNewRun, true); assert.equal(thread.runRecovery.pendingOperations, 0)
  assert.equal(thread.runRecovery.state, fail ? 'needs_review' : 'stopped')
  assert.notEqual(thread.runRecovery.retry, 'safe', 'A delegated accepted/unknown request never inherits the prior safe retry')
}
console.log('Actual delegated branch drains in-flight host mutations, closes late admission and replaces stale safe evidence on Stop or uncertain transport failure.')

function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }, output = ts.transpileModule(readFileSync(file, 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, `Reviewed integration import ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
const { createReadableExportService } = await import('../../readable-export/service')
let owner = true, scope = '/synthetic/profile-one', held = 0, committed = 0, atomicCalls = 0, beforeCommit: (() => void) | null = null
let choose!: (value: any) => void
const live: any = { id: 'local', title: 'Synthetic local', updatedAt: 1, timeline: [{ id: 'u1', kind: 'user', text: 'Synthetic reviewed question.' }] }
const runtimeStore: any = { threads: [live, { ...live, id: 'remote', orchestratorHostId: 'remote-host' }], settings: {} }
const handlers = new Map<string, any>(), event = { sender: { id: 41 } }
const { registerReadableExportRuntime } = load('electron/main/host/readable-export-runtime.ts', {
  electron: { dialog: { showSaveDialog: () => new Promise((resolve) => { choose = resolve }) } },
  'node:os': { __esModule: true, default: { homedir: () => '/synthetic/home', userInfo: () => ({ username: 'fixture' }), hostname: () => 'fixture-host' } },
  '../../readable-export/service': { createReadableExportService }, '../../store': { vontraHomeDir: () => scope }, '../state': { store: runtimeStore }, './ipc': { handle: (name: string, callback: any) => handlers.set(name, callback) },
  './audit': { requireSecurityRole: (role: string) => { assert.equal(role, 'owner'); if (!owner) throw new Error('Synthetic owner role revoked.') } },
  '../../store/atomic-export': { atomicExport: async (_destination: string, bytes: Uint8Array, bound: number, guard: () => void) => { atomicCalls++; assert.ok(bytes.length <= bound); beforeCommit?.(); guard(); committed++ } },
})
registerReadableExportRuntime({ reserveOperation: () => { held++; return { release: () => held-- } } })
const preview = () => handlers.get('readableExport:preview')(event, { threadId: 'local', format: 'text', scope: 'conversation', branch: 'active' })
const save = (approvalId: string, sender = event) => handlers.get('readableExport:save')(sender, { approvalId })
assert.throws(() => handlers.get('readableExport:preview')(event, { threadId: 'remote', format: 'text', scope: 'conversation', branch: 'active' }), /not found/)
let reviewed = preview(), saving = save(reviewed.approvalId)
choose({ canceled: true }); assert.equal((await saving).canceled, true); assert.equal(held, 0); assert.equal(committed, 0)
await assert.rejects(save(reviewed.approvalId), /already used/); assert.equal(held, 0)
reviewed = preview(); saving = save(reviewed.approvalId); live.timeline[0].text += ' Changed while chooser open.'; choose({ filePath: '/synthetic/output.txt' })
await assert.rejects(saving, /changed/); assert.equal(atomicCalls, 0); assert.equal(held, 0)
reviewed = preview(); saving = save(reviewed.approvalId); scope = '/synthetic/profile-two'; choose({ filePath: '/synthetic/output.txt' })
await assert.rejects(saving, /another session/); assert.equal(held, 0); scope = '/synthetic/profile-one'
reviewed = preview(); saving = save(reviewed.approvalId); owner = false; choose({ filePath: '/synthetic/output.txt' })
await assert.rejects(saving, /role revoked/); assert.equal(committed, 0); assert.equal(held, 0); owner = true
reviewed = preview(); await assert.rejects(save(reviewed.approvalId, { sender: { id: 99 } }), /another session/); assert.equal(held, 0)
reviewed = preview(); beforeCommit = () => { live.updatedAt++ }; saving = save(reviewed.approvalId); choose({ filePath: '/synthetic/output.txt' })
await assert.rejects(saving, /changed/); assert.equal(committed, 0); assert.equal(held, 0); beforeCommit = null
reviewed = preview(); saving = save(reviewed.approvalId); choose({ filePath: '/synthetic/output.txt' }); assert.equal((await saving).saved, true); assert.equal(committed, 1); assert.equal(held, 0)
console.log('Actual readable-export registration plus real approval service pass local-only owner/sender/profile/revision fences, canceled/reused approvals and pre-commit drift without native chooser or filesystem.')

let role = 'owner', documentRoot = '/synthetic/doc-one', leases = 0, libraryCreations = 0, lists = 0, resolveList!: (value: any) => void
const documentHandlers = new Map<string, any>()
class LibraryFixture {
  constructor(_options: any) { libraryCreations++ }
  list() { lists++; return new Promise((resolve) => { resolveList = resolve }) }
  search() { return Promise.resolve({ passages: [] }) }
  readEvidence() { return Promise.resolve({ status: 'refused', message: 'Synthetic refused source.' }) }
  purge() { return Promise.resolve({ files: [] }) }
  quiesce() { return Promise.resolve() }
}
const { registerDocumentLibraryRuntime } = load('electron/main/host/document-library-runtime.ts', {
  '../../document-library': { DocumentLibrary: LibraryFixture }, '../../store': { vontraHomeDir: () => documentRoot, vontraPath: () => documentRoot + '/state' }, '../state': { store: { settings: { documents: { sources: [] } } } },
  './ipc': { handle: (name: string, callback: any) => documentHandlers.set(name, callback) }, './audit': { requireSecurityRole: (minimum: string) => { if (role === 'viewer' || minimum === 'owner' && role !== 'owner') throw new Error('Synthetic document role refused.') } },
})
registerDocumentLibraryRuntime({ reserveOperation: () => { leases++; return { release: () => leases-- } } })
let listing = documentHandlers.get('documentLibrary:list')(); documentRoot = '/synthetic/doc-two'; resolveList({ sources: ['old-profile-sensitive-source'] })
await assert.rejects(listing, /profile changed/); assert.equal(leases, 0)
listing = documentHandlers.get('documentLibrary:list')(); role = 'viewer'; resolveList({ sources: ['revoked-source'] })
await assert.rejects(listing, /role refused/); assert.equal(leases, 0); role = 'operator'
await assert.rejects(documentHandlers.get('documentLibrary:purge')({}, 'source'), /role refused/); assert.equal(leases, 0)
const before = lists
await assert.rejects(documentHandlers.get('documentLibrary:search')({}, { query: 'x'.repeat(257) }), /valid document query/)
assert.equal(lists, before); assert.equal(leases, 0); assert.equal(libraryCreations, 2)
console.log('Actual document runtime rejects profile/role drift after an asynchronous result, bounds queries, restricts purge to owner and releases each operation lease.')

function mainBackupAdapter(environment: Record<string, unknown>) {
  const source = readFileSync('electron/main/composition/workspace-services.ts', 'utf8'), ast = ts.createSourceFile('main.ts', source, ts.ScriptTarget.ES2022, true)
  let expression = '', helper = ''
  const visit = (node: ts.Node) => { if (ts.isCallExpression(node) && node.expression.getText(ast) === 'registerWorkspaceBackupRuntime') expression = node.arguments[0].getText(ast); if (ts.isFunctionDeclaration(node) && node.name?.text === 'persistWorkspaceSnapshot') helper = node.getText(ast); ts.forEachChild(node, visit) }
  visit(ast); assert.ok(expression)
  const compiled = ts.transpileModule(`${helper}\nreturn (${expression});`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  return new Function('exports', ...Object.keys(environment), compiled)({}, ...Object.values(environment))
}
const backupCalls: string[] = [], backupStore: any = { settings: { agentPresets: [] }, threads: [], projects: [], unrelated: 'keep', win: { webContents: { reload: () => backupCalls.push('reload') } } }
let persistFailure = false
const adapter = mainBackupAdapter({
  handle() {}, requireSecurityRole: (minimum: string) => { backupCalls.push(`role:${minimum}`) }, vontraHomeDir: () => '/synthetic/backup-profile', store: backupStore,
  normalizeAgentPresets: (presets: any, strict: boolean) => { assert.equal(strict, true); return presets }, saveSettings: async () => { backupCalls.push('settings'); if (persistFailure) throw new Error('Synthetic storage failure.') }, saveThreads: async () => { backupCalls.push('threads') }, saveProjects: async () => { backupCalls.push('projects') },
  readProtectedJson() {}, writeProtectedJson() {}, requestRendererDraftFlush: (sender: any) => { assert.equal(sender, event.sender); backupCalls.push('flush') }, beginComposerDraftPurge: async () => { backupCalls.push('pause') }, endComposerDraftPurge: () => backupCalls.push('resume'), workspaceMutationGuard: { acquireGlobal: async (reason: string) => { assert.match(reason, /backup or restore/); return { release: () => backupCalls.push('release') } } }, assertWorkspaceIdle: () => backupCalls.push('idle'),
  dialog: { showOpenDialog: async () => ({ canceled: true }), showSaveDialog: async () => ({ canceled: true }) }, sendToRenderer: (channel: string) => backupCalls.push(channel), setImmediate: (callback: () => void) => callback(),
})
adapter.authorize(event); await adapter.flushDrafts(event); await adapter.pauseDraftWrites(); adapter.resumeDraftWrites(); const lease = await adapter.acquireLease(); lease.release(); adapter.publishRestored()
assert.ok(backupCalls.includes('role:owner')); assert.ok(backupCalls.includes('flush')); assert.ok(backupCalls.includes('composer-drafts:reset')); assert.ok(backupCalls.includes('reload'))
persistFailure = true; await assert.rejects(adapter.persist({ settings: { agentPresets: [] }, threads: [], projects: [] }), /storage could not complete/)
assert.ok(backupCalls.includes('settings') && backupCalls.includes('threads') && backupCalls.includes('projects'), 'All protected stores settle before a partial failure triggers rollback')
assert.equal(await adapter.chooseOpen(event), null); assert.equal(await adapter.chooseSave(event), null)
adapter.setState({ settings: {}, threads: [{ id: 'restored' }], projects: [] }); assert.equal(backupStore.unrelated, 'keep')
console.log('Actual main-app backup adapter preserves owner/global lease, renderer draft flush/pause/reset, protected all-settled storage failure and canceled chooser semantics with inert collaborators.')

const { bootAppState } = await import('./boot-sequence')
const nodePath = await import('node:path')
const bootCalls: string[] = []
let finishRecovery!: () => void
const bootDeps: any = new Proxy({
  HEADLESS_MODE: false, store: {}, path: nodePath, fs: {}, vontraHomeDir: () => '/synthetic/boot',
  recoverWorkspaceBackup: () => new Promise<void>((resolve) => { bootCalls.push('recovery-start'); finishRecovery = () => { bootCalls.push('recovery-end'); resolve() } }),
  createEventSpineHost: async () => ({}), loadSettings: async () => { bootCalls.push('load-settings'); return {} }, loadThreads: async () => { bootCalls.push('load-threads'); throw new Error('Synthetic boot seam reached.') },
}, { get(object, key) { return key in object ? (object as any)[key] : () => undefined } })
const booting = bootAppState(bootDeps)
for (let i = 0; i < 8; i++) await Promise.resolve()
assert.deepEqual(bootCalls, ['recovery-start'], 'No settings/thread loader crosses an unfinished backup recovery barrier')
finishRecovery(); await assert.rejects(booting, /boot seam reached/)
assert.deepEqual(bootCalls, ['recovery-start', 'recovery-end', 'load-settings', 'load-threads'])
bootCalls.length = 0
await assert.rejects(bootAppState(new Proxy({ ...bootDeps, recoverWorkspaceBackup: async () => { throw new Error('Synthetic recovery failure.') } }, { get(object, key) { return key in object ? (object as any)[key] : () => undefined } })), /recovery failure/)
assert.deepEqual(bootCalls, [], 'A failed recovery never loads partially restored stores')
console.log('Actual startup coordinator waits for backup recovery before settings/threads load and fails closed if recovery rejects.')

const { createRemoteBrowserApi, PairedSessionError } = await import('../../../src/lib/remote-transport')
let remoteRequests = 0
const remote: any = createRemoteBrowserApi({ token: 'synthetic-paired-token', fetchImpl: (async () => { remoteRequests++; throw new Error('No network may occur in this integration fixture.') }) as typeof fetch, openEventStream: () => ({ onEvent() {}, close() {} }) })
for (const method of ['conversationSearch', 'conversationSearchSetActiveThread', 'documentLibraryList', 'documentLibrarySearch', 'documentLibraryReadEvidence', 'documentLibraryPurge', 'readableExportPreview', 'readableExportSave', 'workspaceBackupList', 'workspaceBackupPreviewExport', 'workspaceBackupExport', 'workspaceBackupPreviewRestore', 'workspaceBackupApplyRestore', 'workspaceBackupCancel', 'modelQualificationReports', 'modelQualificationCheck', 'modelQualificationCancel', 'reviewRunOutcome']) {
  assert.equal(typeof remote[method], 'function', `${method} has an explicit fail-closed API implementation`)
  await assert.rejects(remote[method]({ synthetic: true }), PairedSessionError)
}
assert.equal(remoteRequests, 0, 'All new host-only methods reject before a paired REST request or demo fallback')
console.log('Actual paired transport rejects all new local search, document, readable export, backup, model-check and outcome-review APIs without a network request.')

const { reviewedRunOutcome, assertRecoveryAllowsNewRun } = await import('../../agent/run-outcome-review')
function reviewCallback(environment: Record<string, unknown>) {
  const source = readFileSync('electron/main/composition/workspace-content.ts', 'utf8'), ast = ts.createSourceFile('main.ts', source, ts.ScriptTarget.ES2022, true)
  let callback = ''
  const visit = (node: ts.Node) => { if (ts.isCallExpression(node) && node.expression.getText(ast) === 'handle' && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === 'runRecovery:review') callback = node.arguments[1].getText(ast); ts.forEachChild(node, visit) }
  visit(ast); assert.ok(callback)
  const compiled = ts.transpileModule(`return (${callback});`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  return new Function(...Object.keys(environment), compiled)(...Object.values(environment))
}
const reviewThread: any = { id: 'review-thread', timeline: [{ id: 'unchanged', kind: 'assistant', text: 'Retained partial answer.' }], runRecovery: { runId: 'uncertain-run', canStartNewRun: true, retry: 'requires_review' } }
const reviewAborters = new Map(), reviewStore = { threads: [reviewThread] }
let reviewWrites = 0, reviewReleased = 0, finishReview!: () => void
const review = reviewCallback({ requireSecurityRole() {}, workspaceMutationGuard: { reserveStart: () => ({ release: () => reviewReleased++ }) }, store: reviewStore, aborters: reviewAborters, reviewedRunOutcome, saveThreads: async () => { reviewWrites++; await new Promise<void>((resolve) => { finishReview = resolve }) }, sendToRenderer() {} })
await assert.rejects(review({}, { threadId: reviewThread.id, runId: 'old-run' }), /changed/); assert.equal(reviewWrites, 0)
reviewAborters.set(reviewThread.id, {}); await assert.rejects(review({}, { threadId: reviewThread.id, runId: 'uncertain-run' }), /draining/); reviewAborters.clear()
reviewThread.runRecovery.canStartNewRun = false; await assert.rejects(review({}, { threadId: reviewThread.id, runId: 'uncertain-run' }), /draining/); reviewThread.runRecovery.canStartNewRun = true
let reviewing = review({}, { threadId: reviewThread.id, runId: 'uncertain-run' }); finishReview(); await reviewing
assert.equal(reviewThread.runRecoveryReviewedId, 'uncertain-run'); assert.doesNotThrow(() => assertRecoveryAllowsNewRun(reviewThread)); assert.equal(reviewThread.timeline[0].text, 'Retained partial answer.')
reviewThread.runRecovery = { ...reviewThread.runRecovery, runId: 'next-uncertain' }
reviewing = review({}, { threadId: reviewThread.id, runId: 'next-uncertain' }); reviewThread.runRecovery = { ...reviewThread.runRecovery, runId: 'newer-uncertain' }; finishReview()
await assert.rejects(reviewing, /changed/); assert.throws(() => assertRecoveryAllowsNewRun(reviewThread), /Review/); assert.equal(reviewReleased, 5)
console.log('Actual main outcome-review callback refuses wrong/active/draining runs and stale persistence completion, preserves history and cannot authorize a newer uncertain run.')

const { buildAgentRunContract, buildAgentRunReceipt, buildEventReceipt } = await import('../../agent/run-contract')
const runtimeSource = readFileSync('electron/main/host/thread-runtime.ts', 'utf8')
const receiptStart = runtimeSource.indexOf('const recoveryPaused = thread.runRecovery'), receiptEnd = runtimeSource.indexOf('receipt.eventReceipts.push(turnReceipt, jobReceipt, savedReceipt)', receiptStart)
assert.ok(receiptStart > 0 && receiptEnd > receiptStart)
const receiptFragment = runtimeSource.slice(receiptStart, receiptEnd) + 'return { receipt, turnReceipt, jobReceipt, savedReceipt };'
const compiledReceipt = ts.transpileModule(receiptFragment, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const emitReceipts = new Function('thread', 'runId', 'controller', 'failed', 'buildAgentRunReceipt', 'buildEventReceipt', 'runContract', 'runTimelineStart', compiledReceipt)
const receiptThread: any = { id: 'receipt-thread', kind: 'chat', timeline: [], cwd: '', providerId: null, model: null }
const runContract = buildAgentRunContract({ thread: receiptThread, runId: 'receipt-current', objective: 'Synthetic read-only work', provider: null, model: 'synthetic', approvalMode: 'suggest' })
for (const state of ['paused', 'needs_review']) {
  receiptThread.runRecovery = { runId: 'receipt-current', state }
  const result = emitReceipts(receiptThread, 'receipt-current', new AbortController(), false, buildAgentRunReceipt, buildEventReceipt, runContract, 0)
  for (const key of ['turnReceipt', 'jobReceipt', 'savedReceipt']) assert.equal(result[key].state, 'blocked', 'Current paused or uncertain runs never emit an okay completion receipt')
}
receiptThread.runRecovery = { runId: 'old-run', state: 'needs_review' }
const oldReceipt = emitReceipts(receiptThread, 'receipt-current', new AbortController(), false, buildAgentRunReceipt, buildEventReceipt, runContract, 0)
assert.equal(oldReceipt.turnReceipt.state, 'ok', 'A prior run recovery state cannot contaminate current completion authority')
console.log('Actual runtime receipt-emission block marks current paused/uncertain completions blocked and ignores stale prior-run state.')

const { createHash } = await import('node:crypto')
let qualificationRole = 'operator', qualificationRoot = '/synthetic/qualification-one', qualificationLeases = 0, qualificationDispatches = 0, endpointReady!: () => void
const qualificationHandlers = new Map<string, any>()
const qualificationProvider: any = { id: 'qualification-local', name: 'Synthetic local API', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:12345', models: ['fixture-model'] }
const qualificationStore: any = { settings: { providers: [qualificationProvider] }, threads: [] }
class QualificationFixture { reports() { return Promise.resolve([]) }; check() { qualificationDispatches++; return Promise.resolve({ state: 'complete' }) }; cancel() { return Promise.resolve() }; dispose() { return Promise.resolve() } }
const { registerModelQualificationRuntime, resolveQualificationRoute } = load('electron/main/host/model-qualification-runtime.ts', {
  'node:crypto': { createHash }, 'node:fs': { promises: {} }, '../../agent/model-qualification': { ModelQualificationService: QualificationFixture }, '../../agent/provider-accounts': { providerWithAccount: (provider: any) => structuredClone(provider) }, '../../agent/engine/provider': { isEngineProvider: () => false },
  '../../agent/security': { assertProviderEndpointResolved: () => new Promise<void>((resolve) => { endpointReady = resolve }) }, '../../../shared/provider-trust': { isExternalProvider: (provider: any) => provider.kind !== 'local' },
  '../../store': { vontraHomeDir: () => qualificationRoot, vontraPath: () => '/synthetic/qualification-file', readProtectedJson: async () => [], writeProtectedJson: async () => {} }, '../state': { store: qualificationStore },
  './ipc': { handle: (name: string, callback: any) => qualificationHandlers.set(name, callback) }, './audit': { requireSecurityRole: () => { if (qualificationRole === 'viewer') throw new Error('Synthetic qualification role revoked.') } }, './engine-runtime': { engineService: () => null },
})
registerModelQualificationRuntime({ reserveOperation: () => { qualificationLeases++; return { release: () => qualificationLeases-- } }, assertIdle() {} })
const qualificationInput = { requestId: 'synthetic-request', providerId: qualificationProvider.id, model: 'fixture-model', accountId: null, checks: ['text'] }
const route = resolveQualificationRoute(qualificationInput)
qualificationRoot = '/synthetic/qualification-two'; assert.throws(route.assertCurrent, /route changed/); qualificationRoot = '/synthetic/qualification-one'
const changedRoute = resolveQualificationRoute(qualificationInput); qualificationProvider.baseUrl = 'http://127.0.0.1:12346'; assert.throws(changedRoute.assertCurrent, /route changed/); qualificationProvider.baseUrl = 'http://127.0.0.1:12345'
const checking = qualificationHandlers.get('modelQualification:check')({}, qualificationInput)
qualificationRole = 'viewer'; endpointReady()
await assert.rejects(checking, /role revoked/, 'Revoking operator authority during asynchronous endpoint qualification must prevent model-check dispatch')
assert.equal(qualificationDispatches, 0); assert.equal(qualificationLeases, 0)
console.log('Actual model-qualification runtime refuses route/profile/role drift before a synthetic probe service can dispatch.')
