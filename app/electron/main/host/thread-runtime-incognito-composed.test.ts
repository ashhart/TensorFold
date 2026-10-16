import assert from 'node:assert/strict'
import { mkdtemp, readFile, writeFile, readdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-private-composed-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { createUserQuestionPersistence } = await import('./user-question-persistence')
const { initializeUserQuestionHost } = await import('./user-question-host')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const { enqueueJob, claimJobLease, updateJob } = await import('../../agent/jobs')
const { hostPluginRuntime } = await import('../../agent/plugin-runtime')
const sentinel = 'PRIVATE-LAVENDER-FIXTURE-5842'
const provider = { id: 'synthetic-private', name: 'Synthetic API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9897/v1', apiKey: '', models: ['synthetic'] }
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, jobs: store.jobs, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, suggestions: store.suggestions, subAgents: store.subAgents }
const originalFetch = globalThis.fetch
const originalPluginEvent = hostPluginRuntime.emitAgentEvent.bind(hostPluginRuntime)
const writes: Promise<unknown>[] = []
const aborters = new Map<string, AbortController>()
let observerCalls = 0, pluginCalls = 0, managedWrites = 0, checkpointCalls = 0, approvals = 0, settled = false, responseIndex = 0
const requestBodies: any[] = []
const approvedOutput = path.join(home, 'owner-approved-output.txt')
const seed: any = { id: 'ordinary-control', title: 'Ordinary retained control', kind: 'chat', cwd: home, providerId: null, model: null, status: 'idle', createdAt: 1, updatedAt: 1, timeline: [], apiMessages: [{ role: 'user', content: 'Keep this existing ordinary fixture history.' }] }
const thread: any = { ...seed, id: 'private-composed', title: sentinel, kind: 'task', projectId: 'synthetic-project', incognito: true, cwd: '/synthetic-workspace', workspace: { kind: 'ssh', hostId: 'inert-fixture', path: '/synthetic-workspace' }, providerId: provider.id, model: 'synthetic', timeline: [], apiMessages: Array.from({ length: 48 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: i === 0 ? `I always prefer ${sentinel} for this synthetic project.` : `Synthetic bounded history ${i} ${'context '.repeat(40)}` })) }
store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', autoCompact: true, skillSelfImprovement: true, approvalMode: 'suggest' }
store.threads = [seed, thread]; store.projects = [{ id: 'synthetic-project', name: 'Synthetic project', cwd: '/synthetic-workspace', workspace: thread.workspace, memory: '', team: '', createdAt: 1, updatedAt: 1 }]; store.jobs = []; store.projectMemory = []; store.projectMemoryCandidates = []; store.suggestions = []; store.subAgents = []
await storage.saveThreads([seed])
const questionPersistence = createUserQuestionPersistence({ getThread: (id) => store.threads.find((candidate) => candidate.id === id) ?? null, save: (run, patch) => storage.saveThreadQuestion(() => store.threads, run, patch), emit() {} })
const questions = initializeUserQuestionHost({ getThread: (id) => store.threads.find((candidate) => candidate.id === id) ?? null, isRunActive: (id, runId) => aborters.has(id) && thread.activeRunContract?.runId === runId, commitQuestion: questionPersistence.commit })
hostPluginRuntime.emitAgentEvent = async (event) => { pluginCalls++; return originalPluginEvent(event) }
const calls = [
  { name: 'ask_user', arguments: JSON.stringify({ question: `Which synthetic choice ${sentinel}?`, choices: ['Amber', 'Blue'] }) },
  { name: 'read_file', arguments: JSON.stringify({ path: 'large.txt' }) },
  { name: 'write_file', arguments: JSON.stringify({ path: 'owner-approved-output.txt', content: sentinel }) },
  { name: 'read_file', arguments: JSON.stringify({ path: 'owner-approved-output.txt' }) },
  { name: 'read_file', arguments: JSON.stringify({ path: 'missing.txt' }) },
]
const sse = (delta: any, finish = 'stop') => new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'composed fixture refuses any unplanned network route')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  const body = JSON.parse(String(init.body)); requestBodies.push(body)
  if (!body.tools?.length) return sse({ content: `Synthetic history summary ${sentinel}.` })
  const call = calls[responseIndex++]
  return call ? sse({ tool_calls: [{ index: 0, id: `synthetic-${responseIndex}`, type: 'function', function: call }] }, 'tool_calls') : sse({ content: `Synthetic result ${sentinel}: approved output is available; missing file remains an error.` })
}) as typeof fetch
const backend: any = {
  kind: 'ssh',
  readFile: async (file: string) => { if (file === '/synthetic-workspace/large.txt') return `${sentinel} ${'bounded data '.repeat(4000)}`; if (file === '/synthetic-workspace/owner-approved-output.txt') return readFile(approvedOutput, 'utf8'); throw Object.assign(new Error(`Synthetic missing-file error ${sentinel}`), { code: 'ENOENT' }) },
  exists: async (file: string) => file === '/synthetic-workspace/owner-approved-output.txt' && !!await readFile(approvedOutput).catch(() => null),
  writeFile: async (file: string, content: string) => { assert.equal(file, '/synthetic-workspace/owner-approved-output.txt'); await writeFile(approvedOutput, content) },
}
const runtime = createThreadRuntime({
  aborters, workspaceMutationGuard: new WorkspaceMutationGuard(), pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000, sshManager: { ensureConnected: async () => {} }, emitToRenderer() {},
  persistThreads() { writes.push(storage.saveThreads(store.threads)) }, observe: () => { observerCalls++ },
  fullToolContext: () => ({ cwd: '/synthetic-workspace', backend, currentThread: thread, writeManagedOutput: async () => { managedWrites++; throw new Error('Injected writer must never be reached.') } }),
  approvalResolver: () => async () => { approvals++; return 'allow' }, isHighRiskTool: () => true,
  claimJob: (ref) => { const queued = enqueueJob(store.jobs, ref); const claimed = claimJobLease(queued.jobs, queued.job.id, 'synthetic'); store.jobs = claimed.jobs; writes.push(storage.saveJobs(store.jobs)); return claimed.job! },
  patchJob: (job, patch) => { if (!job) return null; const next = updateJob(store.jobs, job.id, patch); store.jobs = next.jobs; writes.push(storage.saveJobs(store.jobs)); return next.job },
  replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
  settleThreadRunJob(job) { if (job) { const next = updateJob(store.jobs, job.id, { status: 'completed', metadata: { privateResult: sentinel } }); store.jobs = next.jobs; writes.push(storage.saveJobs(store.jobs)) }; settled = true },
  renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 1500,
  captureCheckpoint: async () => { checkpointCalls++; return { ok: true, sha: 'synthetic' } }, recordTurnCheckpoint: () => { checkpointCalls++ },
})
async function waitFor(predicate: () => boolean, label: string) { for (let i = 0; i < 600; i++) { if (predicate()) return; await new Promise((resolve) => setTimeout(resolve, 5)) }; throw new Error(`Timed out: ${label}`) }
try {
  await runtime.startThreadRun({ threadId: thread.id, userText: `I always prefer ${sentinel}. Ask the synthetic choice, inspect, and create the explicitly approved workspace output.` })
  await waitFor(() => thread.timeline.some((item: any) => item.question && questions.isWaiting(item.question.id)), 'actual question parked')
  const question = thread.timeline.find((item: any) => item.question).question
  await questions.answer({ questionId: question.id, threadId: thread.id, runId: question.runId, action: 'answer', answer: `Amber ${sentinel}` })
  await waitFor(() => settled && !aborters.has(thread.id), 'composed actual host run settlement')
  await questionPersistence.drain(); await Promise.all(writes); await new Promise((resolve) => setTimeout(resolve, 50))
  assert.equal(thread.timeline.find((item: any) => item.question)?.question.state, 'answered')
  assert.ok(thread.compactions?.length, 'the real native loop compacted the private history')
  assert.ok(thread.timeline.some((item: any) => item.kind === 'tool' && item.state === 'error'), 'a private tool error was actually exercised')
  assert.ok(thread.timeline.some((item: any) => item.kind === 'assistant' && item.text?.includes(sentinel)), 'ordinary assistant text remains visible in this private conversation')
  assert.ok(requestBodies.some((body) => JSON.stringify(body.messages).includes(`Amber ${sentinel}`)), 'the intended local API model receives the explicit question answer')
  assert.equal(await readFile(approvedOutput, 'utf8'), sentinel, 'explicitly approved workspace output remains usable')
  assert.ok(approvals > 0)
  assert.equal(observerCalls + pluginCalls + managedWrites + checkpointCalls, 0)
  assert.equal(store.projectMemoryCandidates.length + store.suggestions.length, 0)
  assert.equal((await storage.loadJobs()).length, 0, 'actual protected job persistence excludes private derived receipts and metadata')
  const loadedThreads = await storage.loadThreads()
  assert.deepEqual(loadedThreads.map((candidate) => candidate.id), [seed.id], 'ordinary history survives while the private conversation never persists')
  assert.doesNotMatch(JSON.stringify(loadedThreads), new RegExp(sentinel))
  const journals = await readdir(path.join(home, 'thread-journals')).catch(() => [] as string[])
  assert.equal(journals.length, 0)
  const automaticFiles: string[] = []
  async function inspect(dir: string) { for (const item of await readdir(dir, { withFileTypes: true })) { const file = path.join(dir, item.name); if (item.isDirectory()) await inspect(file); else if (file !== approvedOutput) { automaticFiles.push(file); const raw = await readFile(file, 'utf8'); assert.ok(!raw.includes(sentinel), 'automatic app files contain no plaintext private sentinel') } } }
  await inspect(home)
  assert.ok(automaticFiles.length > 0, 'the actual protected store and structural audit were inspected')
  console.log('Composed actual runtime + protected store: private assistant/question answer/tool error/compaction retained only in live chat; no journal/checkpoint/job/memory/suggestion/observer/plugin/overflow sentinel; explicit approved workspace output and ordinary saved history survive.')
} finally {
  await questions.shutdown(); await questionPersistence.drain(); await Promise.allSettled(writes)
  hostPluginRuntime.emitAgentEvent = originalPluginEvent; globalThis.fetch = originalFetch; Object.assign(store, saved)
  await rm(home, { recursive: true, force: true })
}
