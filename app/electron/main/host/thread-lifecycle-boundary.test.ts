import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { store } from '../state'
import { createThreadForkFactory, assertThreadForkRestrictions } from './thread-revert-fork'
import { captureBotConfig, botRunApprovalMode } from '../../../shared/bot-policy'
import { captureThreadMemoryPolicy, assertThreadMemoryTransport } from '../../../shared/thread-memory-policy'
import { classifyProvider } from '../../../shared/provider-trust'
import { executeTool } from '../../agent/tools'
import { runAgent } from '../../agent/loop'
import { buildHandoff } from '../../agent/turn-checkpoints'
import { latestPlanText, runPlanPrompt } from '../../agent/planning-guard'
import { providerWithAccount, resolveProviderModel, selectProviderAccount } from '../../agent/provider-accounts'
import { spineEligible } from '../../agent/incognito'
import type { Provider, Thread } from '../../agent/types'

// Exact production handlers execute with inert IPC/filesystem/audit collaborators;
// no native app, phone, provider endpoint or real profile is used.
function compile(file: string, select: (tree: ts.SourceFile) => ts.Node): string {
  const tree = ts.createSourceFile(file, readFileSync(path.join(process.cwd(), file), 'utf8'), ts.ScriptTarget.Latest, true)
  const node = select(tree)
  const source = ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, tree).replace(/\bexport\s+function\s+/g, 'function ')
  return ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
}
function find(tree: ts.SourceFile, predicate: (node: ts.Node) => boolean): ts.Node {
  let result: ts.Node | undefined
  const visit = (node: ts.Node) => { if (predicate(node)) result = node; ts.forEachChild(node, visit) }
  visit(tree); assert.ok(result); return result
}
const provider: Provider = { id: 'lifecycle-local', name: 'Synthetic local API', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
const target: Provider = { ...provider, id: 'lifecycle-other-local', name: 'Synthetic second local API' }
const cloud: Provider = { ...provider, id: 'lifecycle-cloud', name: 'Synthetic external API', kind: 'cloud', baseUrl: 'https://provider.example.invalid/v1' }
const cli: Provider = { ...provider, id: 'lifecycle-cli', transport: 'codex-cli' }
const lan: Provider = { ...provider, id: 'lifecycle-lan', baseUrl: 'http://192.0.2.12:8000/v1' }
const preset = { id: 'bot_lifecycle', name: 'Synthetic limited bot', role: 'Inspect synthetic data', lane: 'review' as const, instructions: 'Use no tools.', providerId: provider.id, model: 'fixture', approvalScope: 'readonly' as const, executionTarget: 'workspace' as const, toolScope: { mode: 'selected' as const, names: [] as string[] }, createdAt: 1, updatedAt: 1 }
const base: Thread = { id: 'lifecycle-source', title: 'Synthetic source', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'fixture', status: 'idle', createdAt: 1, updatedAt: 1, memoryPolicy: { useExisting: false, contribute: false }, timeline: [], apiMessages: [{ role: 'user', content: 'SYNTHETIC_PHONE_CONTEXT_AMBER' }] }
const forkPoint: any = { turn: 1, branch: 'synthetic', conversation: { apiMessages: base.apiMessages, timeline: [] } }
const saved = { threads: store.threads, projects: store.projects, settings: store.settings }
const originalFetch = globalThis.fetch
let modelCalls = 0, copiedMarkerSent = false, persisted = 0, projected = 0
globalThis.fetch = (async (url: any, init: any) => {
  assert.match(String(url), /^http:\/\/127\.0\.0\.1:9\//)
  modelCalls++; copiedMarkerSent ||= String(init.body).includes('SYNTHETIC_PHONE_CONTEXT_AMBER')
  return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic response.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
const createFork = createThreadForkFactory({ persistThreads() { persisted++ }, remoteServer: () => ({ publish() { projected++ } }), spineHost: () => null })
const run = (thread: Thread, extra: Record<string, unknown> = {}) => runAgent({ thread, provider, model: 'fixture', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 1, signal: new AbortController().signal, persist() {}, emit() {}, requestApproval: async () => 'deny', ...extra })
try {
  store.settings = { ...saved.settings, providers: [provider, target, cloud, cli, lan], agentPresets: [preset], activeProviderAccountId: null, activeModel: 'fixture' }
  store.projects = []
  const source = { ...base, bot: captureBotConfig(preset, [provider]) }
  store.threads = [source]
  const fork = createFork(source.id, forkPoint)!
  assert.deepEqual(fork.bot, source.bot)
  assert.notEqual(fork.bot, source.bot); assert.notEqual(fork.bot!.toolScope, source.bot!.toolScope)
  assert.ok(Object.isFrozen(fork.bot)); assert.ok(Object.isFrozen(fork.bot!.toolScope)); assert.ok(Object.isFrozen(fork.bot!.toolScope!.names))
  assert.deepEqual(fork.memoryPolicy, base.memoryPolicy); assert.notEqual(fork.memoryPolicy, source.memoryPolicy)
  source.bot!.toolScope!.names.push('read_file')
  assert.deepEqual(fork.bot!.toolScope!.names, [], 'the independent fork snapshot cannot expand through mutation of its source')
  assert.equal(botRunApprovalMode(fork, [preset], [provider], 'auto'), 'readonly')
  assert.equal((await executeTool('read_file', { path: 'never-read.txt' }, { cwd: '', currentThread: fork })).ok, false)
  await assert.rejects(run(fork, { approvalMode: 'auto' }), /cannot be elevated/)
  assert.equal(modelCalls, 0)
  await run(fork, { toolContext: { cwd: '', currentThread: fork, botToolScopeGuard() {} } })
  assert.equal(modelCalls, 1, 'normal bot forks still use their captured API model with no broadened tools')
  const before = { persisted, projected }
  assert.throws(() => createFork(source.id, forkPoint, { cwd: '/synthetic-host-worktree' }), /cannot fork into a host worktree/)
  assert.deepEqual({ persisted, projected }, before)
  const phone = { ...base, id: 'lifecycle-phone', phoneLocalOnly: true }
  store.threads = [phone]
  const phoneFork = createFork(phone.id, forkPoint)!
  assert.equal(phoneFork.phoneLocalOnly, true)
  assert.equal(Object.hasOwn(phoneFork, 'phoneGrant'), false)
  const start = modelCalls
  await assert.rejects(run(phoneFork), /phone privacy guard is unavailable/)
  assert.equal(modelCalls, start, 'a fork cannot send copied phone context without a local-processing guard')
  await run(phoneFork, { toolContext: { cwd: '', currentThread: phoneFork, phonePrivacyGuard: () => null } })
  assert.equal(modelCalls, start + 1); assert.equal(copiedMarkerSent, true)
  const ordinary = { ...base, id: 'lifecycle-ordinary', memoryPolicy: { useExisting: true, contribute: true } }
  store.threads = [ordinary]
  assert.equal(createFork(ordinary.id, forkPoint, { cwd: '/synthetic-host-worktree' })?.cwd, '/synthetic-host-worktree', 'ordinary worktree forks retain their normal behavior')

  const handlerSource = compile('electron/main/host/workspace-ipc.ts', (tree) => find(tree, (node) => ts.isCallExpression(node) && node.expression.getText(tree) === 'handle' && node.arguments[0]?.getText(tree) === "'checkpoint:forkTurn'"))
  let forkHandler: any, contexts = 0, checkpointCalls = 0, reservations = 0, releases = 0
  const inertDeps = { workspaceMutationGuard: { reserveStart() { reservations++; return { addWorkspaceKeys() {}, release() { releases++ } } } }, createForkThread: () => ({ id: 'ordinary-fork' }) }
  new Function('handle', 'requireSecurityRole', 'store', 'assertThreadForkRestrictions', 'target', 'deps', 'checkpointInitialWorkspaceKey', 'checkpointWorkspaceKey', 'sameCheckpointTarget', 'checkpoints', 'auditEvent', 'auditPreview', handlerSource)(
    (_name: string, callback: any) => { forkHandler = callback }, () => {}, store, assertThreadForkRestrictions,
    () => { contexts++; return { kind: 'chat', cwd: '' } }, inertDeps, () => null, async () => null, () => true,
    { forkTurn: async () => { checkpointCalls++; return { ok: true, fork: forkPoint, worktreePath: '/synthetic-worktree' } }, adoptFork() {} }, async () => {}, String,
  )
  store.threads = [source]
  await assert.rejects(forkHandler({}, { threadId: source.id, turn: 1, worktree: true }), /cannot fork into a host worktree/)
  assert.equal(contexts, 0); assert.equal(checkpointCalls, 0); assert.equal(reservations, 0, 'real IPC refuses before context, reservation or filesystem checkpoint work')
  store.threads = [ordinary]
  assert.equal((await forkHandler({}, { threadId: ordinary.id, turn: 1, worktree: true })).ok, true)
  assert.equal(checkpointCalls, 1); assert.equal(reservations, releases)

  let handoff: any, transcriptBuilds = 0, handoffPersisted = 0, handoffStarted = 0
  const registration = compile('electron/main/host/handoff-ipc.ts', (tree) => find(tree, (node) => ts.isFunctionDeclaration(node) && node.name?.text === 'registerThreadHandoffIpc'))
  const register = new Function('handle', 'requireSecurityRole', 'store', 'captureThreadMemoryPolicy', 'assertThreadMemoryTransport', 'classifyProvider', 'runtimeProvider', 'selectedProviderAccountId', 'latestPlanText', 'runPlanPrompt', 'assertSshThreadTransportSupported', 'buildHandoff', 'threadTurnNumber', 'resolveProviderModel', 'spineEligible', 'auditEvent', 'auditPreview', 'randomUUID', registration.replace('export function', 'function') + '\nreturn registerThreadHandoffIpc;')(
    (_channel: string, callback: any) => { handoff = callback }, () => {}, store, captureThreadMemoryPolicy, assertThreadMemoryTransport, classifyProvider, providerWithAccount,
    (p: Provider, accountId: string | null) => selectProviderAccount(p, accountId)?.id ?? null, latestPlanText, runPlanPrompt, () => {},
    (input: any) => { transcriptBuilds++; return buildHandoff(input) }, () => 1, resolveProviderModel, spineEligible, async () => {}, String, () => `handoff-${handoffPersisted}-${handoffStarted}`,
  )
  register({ spine: () => null, sendToRenderer() {}, remoteServer: { publish() {} }, persistThreads() { handoffPersisted++ }, startThreadRun: async () => { handoffStarted++; return { runId: 'synthetic-run' } } })
  store.threads = [source]
  await assert.rejects(handoff({}, { threadId: source.id, toProviderId: target.id, autoStart: false }), /captured model and permissions/)
  assert.equal(transcriptBuilds, 0); assert.equal(handoffPersisted, 0)
  for (const bot of [null, false, 0]) {
    const damaged: any = { ...base, bot }
    store.threads = [damaged]
    assert.throws(() => createFork(damaged.id, forkPoint), /invalid captured bot/)
    assert.throws(() => createFork(damaged.id, forkPoint, { cwd: '/synthetic-worktree' }), /invalid captured bot/)
    assert.throws(() => botRunApprovalMode(damaged, [], [provider], 'auto'), /invalid captured bot/)
    await assert.rejects(run(damaged), /invalid captured bot/)
    await assert.rejects(handoff({}, { threadId: damaged.id, toProviderId: cloud.id, autoStart: false }), /captured model and permissions/)
    assert.equal(transcriptBuilds, 0); assert.equal(handoffPersisted, 0)
  }
  for (const toProvider of [cloud, cli, lan]) {
    store.threads = [phone]
    await assert.rejects(handoff({}, { threadId: phone.id, toProviderId: toProvider.id, autoStart: false }), /Phone conversations can only hand off/)
    assert.equal(transcriptBuilds, 0); assert.equal(handoffPersisted, 0)
  }
  store.threads = [{ ...phone, orchestratorHostId: 'synthetic-remote' }]
  await assert.rejects(handoff({}, { threadId: phone.id, toProviderId: target.id, autoStart: false }), /Phone conversations can only hand off/)
  store.threads = [{ ...phone, workspace: { kind: 'ssh', hostId: 'synthetic-host', path: '/synthetic' } }]
  await assert.rejects(handoff({}, { threadId: phone.id, toProviderId: target.id, autoStart: false }), /SSH workspace/)
  assert.equal(transcriptBuilds, 0)
  store.threads = [phone]
  const phoneHandoff = await handoff({}, { threadId: phone.id, toProviderId: target.id, autoStart: false })
  assert.equal(phoneHandoff.phoneLocalOnly, true); assert.deepEqual(phoneHandoff.memoryPolicy, phone.memoryPolicy)
  assert.equal(handoffStarted, 0); assert.equal(Object.hasOwn(phoneHandoff, 'phoneGrant'), false)
  const callsBefore = modelCalls
  await assert.rejects(run(phoneHandoff, { provider: target }), /phone privacy guard is unavailable/)
  assert.equal(modelCalls, callsBefore)
  store.threads = [ordinary]
  const ordinaryHandoff = await handoff({}, { threadId: ordinary.id, toProviderId: cloud.id, autoStart: false })
  assert.equal(ordinaryHandoff.providerId, cloud.id); assert.equal(ordinaryHandoff.phoneLocalOnly, undefined)
  assert.deepEqual(ordinaryHandoff.memoryPolicy, ordinary.memoryPolicy)
  store.threads = [ordinary]
  await handoff({}, { threadId: ordinary.id, toProviderId: target.id })
  assert.equal(handoffStarted, 1, 'ordinary unrestricted handoffs still support automatic continuation')
  console.log('Actual lifecycle boundaries passed: independent immutable bot forks, sticky phone/memory restrictions without grants, pre-context bot worktree/engine refusals, pre-transcript phone CLI/cloud/LAN/remote refusals, local handoff markers and ordinary fork/handoff positives.')
} finally { globalThis.fetch = originalFetch; Object.assign(store, saved) }
