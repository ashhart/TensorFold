import { mergeThreadEventPatch } from '../../../src/lib/store/shared/merge'
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import type { AgentEvent, Thread } from '../../agent/types'
import { projectedHostId, projectionOriginsMatch } from '../../../src/lib/store/shared/proxy-state'
import { threadStatusFromRun } from '../../../src/lib/store/types'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-memory-publication-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { createThreadMemoryPolicyHost } = await import('./thread-memory-policy-host')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const oldFetch = globalThis.fetch, rename = fs.rename
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, jobs: store.jobs, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, skills: store.skills, suggestions: store.suggestions }
const provider = { id: 'synthetic-memory-publication', name: 'Synthetic API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9896/v1', apiKey: '', models: ['synthetic'] }
let modelRequests = 0, holdModel: (() => Promise<void>) | undefined
const sse = () => new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic generation finished.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'all provider requests use the inert fixture')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  modelRequests++
  if (holdModel) await holdModel()
  return sse()
}) as typeof fetch

// Execute the production renderer reducer, not a duplicate patch implementation.
// Its collaborators below are only origin/status helpers and an in-memory store.
const rendererSource = await fs.readFile(path.join(process.cwd(), 'src/lib/store/slices/threadsSlice.ts'), 'utf8')
const tree = ts.createSourceFile('threadsSlice.ts', rendererSource, ts.ScriptTarget.Latest, true)
let applySource = ''
function visit(node: ts.Node) { if (ts.isMethodDeclaration(node) && node.name.getText(tree) === 'applyEvent') applySource = node.getText(tree); ts.forEachChild(node, visit) }
visit(tree); assert.ok(applySource)
const reducerJs = ts.transpileModule(`function ${applySource}`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const makeReducer = new Function('get', 'set', 'projectedHostId', 'projectionOriginsMatch', 'threadStatusFromRun', 'mergeThreadEventPatch', `${reducerJs}\nreturn applyEvent;`)
async function waitFor(predicate: () => boolean, label: string) { for (let i = 0; i < 800; i++) { if (predicate()) return; await new Promise((r) => setTimeout(r, 5)) }; throw new Error(`Timed out: ${label}`) }
async function fixture(id: string, extra: Partial<Thread> = {}) {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', approvalMode: 'readonly', skillSelfImprovement: false, autoCompact: false }
  store.projects = []; store.jobs = []; store.projectMemory = []; store.projectMemoryCandidates = []; store.skills = []; store.suggestions = []
  const thread: Thread = { id, title: 'Synthetic retained chat', kind: 'chat', cwd: '/synthetic-workspace', workspace: { kind: 'ssh', hostId: 'inert-workspace', path: '/synthetic-workspace' }, providerId: provider.id, providerAccountId: null, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'SYNTHETIC-RETAINED-HISTORY' }], memoryPolicy: { useExisting: true, contribute: true }, memoryPolicyRevision: 0, ...extra }
  store.threads = [thread]
  if (!thread.incognito) await storage.saveThreads(store.threads)
  let rendered: any = { threads: [structuredClone(thread)], status: {}, approvals: {}, view: { name: 'thread', id } }
  const apply = makeReducer(() => rendered, (patch: any) => { rendered = { ...rendered, ...(typeof patch === 'function' ? patch(rendered) : patch) } }, projectedHostId, projectionOriginsMatch, threadStatusFromRun, mergeThreadEventPatch)
  const events: AgentEvent[] = [], rawMemoryAliases: boolean[] = []
  const publish = (event: AgentEvent) => {
    if (event.type === 'thread_update' && event.patch.memoryPolicyRevision !== undefined) rawMemoryAliases.push(event.patch.memoryPolicy === thread.memoryPolicy)
    const wire = structuredClone(event); events.push(wire); apply(wire)
  }
  const aborters = new Map<string, AbortController>(), guard = new WorkspaceMutationGuard()
  let settled = 0
  const runtime = createThreadRuntime({ aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer: publish, persistThreads() {},
    fullToolContext: (run) => ({ cwd: '/synthetic-workspace', backend: { kind: 'ssh' }, currentThread: thread, memoryPolicy: run?.memoryPolicy }), approvalResolver: () => async () => 'allow', isHighRiskTool: () => false,
    claimJob: () => ({ id: `job-${id}`, status: 'running', metadata: {}, pendingApprovals: [] } as any), patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob() { settled++ }, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 128000,
    captureCheckpoint: async () => ({ ok: true, sha: 'synthetic' }), recordTurnCheckpoint() {},
  })
  const change = createThreadMemoryPolicyHost({ threads: () => store.threads, provider: () => provider, aborters, guard, publish: (accepted) => publish({ type: 'thread_update', runId: '', threadId: id, patch: { memoryPolicy: { ...accepted.memoryPolicy! }, memoryPolicyRevision: accepted.memoryPolicyRevision } }) })
  return { thread, runtime, aborters, guard, change, events, rawMemoryAliases, rendered: () => rendered.threads[0] as Thread, settled: () => settled,
    memoryEvents: () => events.filter((event) => event.type === 'thread_update' && event.patch.memoryPolicyRevision !== undefined) }
}
try {
  const first = await fixture('synthetic-first-turn')
  await first.runtime.startThreadRun({ threadId: first.thread.id, userText: 'Generate one synthetic local response.' })
  await waitFor(() => first.settled() === 1 && !first.aborters.size, 'first run settlement')
  assert.equal(first.thread.memoryPolicyRevision, 1)
  await assert.doesNotReject(() => first.change(first.thread.id, { useExisting: false, contribute: false }, first.rendered().memoryPolicyRevision ?? 0), 'the sole renderer must receive the durably committed startup revision before Save choices')
  assert.equal(first.rendered().memoryPolicyRevision, 2)
  assert.deepEqual(first.rendered().memoryPolicy, { useExisting: false, contribute: false })
  assert.deepEqual((await storage.loadThreads())[0].memoryPolicy, { useExisting: false, contribute: false })
  assert.ok(first.rendered().timeline.some((item) => item.text === 'Synthetic generation finished.'))
  assert.ok(first.thread.apiMessages.some((message) => message.content === 'SYNTHETIC-RETAINED-HISTORY'), 'startup publication preserves existing user history')
  const startup = first.memoryEvents()[0]
  const firstUser = first.events.findIndex((event) => event.type === 'item_add' && event.item.kind === 'user')
  assert.ok(first.events.indexOf(startup) < firstUser, 'committed choices publish before the first user timeline item')
  assert.equal(first.rawMemoryAliases[0], false, 'the publication is a detached value, not a mutable live-thread reference')
  await assert.rejects(first.change(first.thread.id, { useExisting: true, contribute: true }, 1), /another window/, 'a real stale revision is still refused')
  await first.runtime.startThreadRun({ threadId: first.thread.id, userText: 'Generate a second synthetic response.' })
  await waitFor(() => first.settled() === 2 && !first.aborters.size, 'second run settlement')
  assert.equal(first.rendered().memoryPolicyRevision, 3)
  assert.deepEqual(first.rendered().memoryPolicy, { useExisting: false, contribute: false })

  const blocked = await fixture('synthetic-blocked-commit')
  let entered!: () => void, release!: () => void
  const enteredPromise = new Promise<void>((resolve) => { entered = resolve })
  const gate = new Promise<void>((resolve) => { release = resolve })
  let once = true
  fs.rename = async (from, to) => { if (String(to) === storage.threadsPath() && once) { once = false; entered(); await gate }; return rename(from, to) }
  const beforeBlockedRequests = modelRequests
  const starting = blocked.runtime.startThreadRun({ threadId: blocked.thread.id, userText: 'Must wait for durable policy.' })
  await enteredPromise
  assert.equal(blocked.thread.memoryPolicyRevision, 0)
  assert.equal(blocked.rendered().memoryPolicyRevision, 0)
  assert.equal(blocked.memoryEvents().length, 0)
  assert.equal(modelRequests, beforeBlockedRequests)
  await assert.rejects(blocked.change(blocked.thread.id, { useExisting: false, contribute: false }, 0), /Stop this run/)
  blocked.aborters.get(blocked.thread.id)!.abort()
  release(); await assert.rejects(starting, /cancelled after memory choices were accepted/)
  fs.rename = rename
  assert.equal(blocked.thread.memoryPolicyRevision, 1)
  assert.equal(blocked.rendered().memoryPolicyRevision, 1, 'Stop after a successful commit still publishes the accepted revision')
  assert.equal(blocked.memoryEvents().length, 1)
  assert.equal(modelRequests, beforeBlockedRequests)
  assert.equal(blocked.thread.timeline.length, 0)
  assert.equal(blocked.aborters.size, 0)
  blocked.guard.reserveStart({ threadId: blocked.thread.id }).release()
  await blocked.change(blocked.thread.id, { useExisting: false, contribute: false }, blocked.rendered().memoryPolicyRevision!)

  const failed = await fixture('synthetic-failed-commit')
  const beforeFailure = modelRequests
  fs.rename = async (from, to) => { if (String(to) === storage.threadsPath()) throw new Error('Synthetic durable rename failure'); return rename(from, to) }
  await assert.rejects(failed.runtime.startThreadRun({ threadId: failed.thread.id, userText: 'Do not publish uncommitted memory choices.' }), /Synthetic durable rename failure/)
  fs.rename = rename
  assert.equal(failed.memoryEvents().length, 0)
  assert.equal(failed.rendered().memoryPolicyRevision, 0)
  assert.equal(failed.thread.memoryPolicyRevision, 0)
  assert.equal(modelRequests, beforeFailure)
  assert.equal(failed.thread.timeline.length, 0)
  assert.equal(failed.aborters.size, 0)

  const active = await fixture('synthetic-active-run')
  let finish!: () => void
  const modelGate = new Promise<void>((resolve) => { finish = resolve })
  holdModel = () => modelGate
  const beforeActive = modelRequests
  await active.runtime.startThreadRun({ threadId: active.thread.id, userText: 'Hold the synthetic response.' })
  await waitFor(() => modelRequests > beforeActive, 'active model call')
  assert.equal(active.rendered().memoryPolicyRevision, 1)
  await assert.rejects(active.change(active.thread.id, { useExisting: false, contribute: false }, 1), /Stop this run/)
  assert.deepEqual(active.rendered().memoryPolicy, { useExisting: true, contribute: true })
  holdModel = undefined; finish()
  await waitFor(() => active.settled() === 1 && !active.aborters.size, 'held model settlement')

  for (const extra of [{ memoryPolicy: { useExisting: 'damaged', contribute: true } }, { memoryPolicyRevision: 'damaged' }]) {
    const damaged = await fixture('synthetic-damaged', extra as any)
    const beforeDamaged = modelRequests
    await assert.rejects(damaged.runtime.startThreadRun({ threadId: damaged.thread.id, userText: 'Never normalize damaged choices into acceptance.' }), /boolean|revision/)
    assert.equal(damaged.memoryEvents().length, 0)
    assert.equal(modelRequests, beforeDamaged)
  }
  const privateRun = await fixture('synthetic-private-choice', { incognito: true })
  const diskBeforePrivate = await fs.readFile(storage.threadsPath())
  await privateRun.runtime.startThreadRun({ threadId: privateRun.thread.id, userText: 'Private synthetic response.' })
  await waitFor(() => privateRun.settled() === 1 && !privateRun.aborters.size, 'private settlement')
  assert.equal(privateRun.rendered().memoryPolicyRevision, 1)
  assert.deepEqual(privateRun.rendered().memoryPolicy, { useExisting: true, contribute: false })
  assert.deepEqual(await fs.readFile(storage.threadsPath()), diskBeforePrivate)
  console.log('Actual startup/encrypted commit/renderer reducer/host CAS: first and second turn, blocked/failed commit, Stop after commit, active/stale/damaged guards and Incognito publication passed')
} finally {
  fs.rename = rename; holdModel = undefined; globalThis.fetch = oldFetch
  await new Promise((r) => setTimeout(r, 100)); await Promise.allSettled([...storage.writeChains.values()])
  Object.assign(store, saved); storage.closeSessionSearchIndexes()
  await fs.rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 })
}
