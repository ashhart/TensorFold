import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-quote-runtime-'))
process.env.VONTRA_HOME = profile; process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const { prepareMessageQuote } = await import('../../agent/conversation-quotation')
const provider = { id: 'synthetic-quotes', name: 'Synthetic API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9876/v1', apiKey: '', models: ['synthetic'] }
const originalFetch = globalThis.fetch
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, jobs: store.jobs, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, skills: store.skills, suggestions: store.suggestions }
const requests: any[] = []
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'never contact an external service')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  requests.push(JSON.parse(String(init.body)))
  return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic reply.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
try {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', skillSelfImprovement: false, autoCompact: false, approvalMode: 'readonly' }
  store.projects = []; store.jobs = []; store.projectMemory = []; store.projectMemoryCandidates = []; store.skills = []; store.suggestions = []
  const thread: any = { id: 'synthetic-quote-chat', title: 'Synthetic quote chat', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'synthetic', status: 'idle', createdAt: 1, updatedAt: 1, memoryPolicy: { useExisting: false, contribute: false }, apiMessages: [{ role: 'assistant', content: 'Quoted amber evidence.' }], timeline: [{ id: 'answer', kind: 'assistant', text: 'Quoted amber evidence.', createdAt: 1 }] }
  store.threads = [thread]
  const quote = { ...prepareMessageQuote(thread, { threadId: thread.id, itemId: 'answer' }), text: 'Edited amber quotation.' }
  const aborters = new Map<string, AbortController>(), guard = new WorkspaceMutationGuard()
  let settled = false
  const runtime = createThreadRuntime({ aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {},
    fullToolContext: run => ({ cwd: '', currentThread: thread, memoryPolicy: run?.memoryPolicy }), approvalResolver: () => async () => 'allow', isHighRiskTool: () => false,
    claimJob: () => ({ id: 'synthetic-job', status: 'running', metadata: {}, pendingApprovals: [] } as any), patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: result => result.job!, heartbeatJob: job => job,
    settleThreadRunJob() { settled = true }, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 128000,
    captureCheckpoint: async () => ({ ok: true, sha: 'synthetic' }), recordTurnCheckpoint() {},
  })
  await runtime.startThreadRun({ threadId: thread.id, userText: 'Explain the passage.', quotes: [quote] })
  for (let i = 0; i < 800 && (!settled || aborters.has(thread.id)); i++) await new Promise(done => setTimeout(done, 5))
  assert.equal(settled, true); assert.equal(aborters.has(thread.id), false)
  const user = thread.timeline.find((row: any) => row.kind === 'user')
  assert.equal(user.text, 'Explain the passage.'); assert.deepEqual(user.quotedMessages, [quote])
  assert.ok(requests.some(request => request.messages.some((row: any) => row.role === 'user' && String(row.content).includes('Edited amber quotation.') && String(row.content).includes('Explain the passage.'))))
  const count = requests.length, timeline = structuredClone(thread.timeline)
  thread.timeline[0].text = 'Source was edited later.'
  await assert.rejects(runtime.startThreadRun({ threadId: thread.id, userText: 'Do not send stale context', quotes: [quote] }), /source changed/)
  assert.equal(requests.length, count); assert.equal(thread.timeline.length, timeline.length)
  await assert.rejects(runtime.guideThreadRun({ threadId: thread.id, userText: 'Do not drop quotes', quotes: [quote] } as any), /Quotes stay/)
} finally { await new Promise(done => setTimeout(done, 100)); await Promise.allSettled([...storage.writeChains.values()]); globalThis.fetch = originalFetch; Object.assign(store, saved); storage.closeSessionSearchIndexes(); await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 }) }
console.log('Actual host run emits and sends the same edited quote, refuses changed sources before generation and never drops quotes into live guidance')
