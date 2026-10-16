import assert from 'node:assert/strict'
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

// Actual host run, isolated app data and a fixed in-process HTTP transport fixture.
// No user history, real provider or external endpoint is read or contacted.
const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-incognito-runtime-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const { defaultSettings } = await import('../../store')
const { store } = await import('../state')
const { hostPluginRuntime } = await import('../../agent/plugin-runtime')
const originalPluginEvent = hostPluginRuntime.emitAgentEvent.bind(hostPluginRuntime)
let hostPluginEvents = 0
hostPluginRuntime.emitAgentEvent = async (event) => { hostPluginEvents++; return originalPluginEvent(event) }
const { createThreadRuntime, assertIncognitoThreadTransportSupported } = await import('./thread-runtime')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const provider = { id: 'incognito-fixture', name: 'Synthetic local API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9899/v1', apiKey: '', models: ['synthetic'] }
const skill = { id: 'synthetic-skill', name: 'synthetic-review', description: 'Synthetic fixture', body: 'Inspect the bounded fixture. SYNTHETIC-SKILL-CONTENT.', enabled: true, createdAt: 1, updatedAt: 1 }
const originalFetch = globalThis.fetch
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, skills: store.skills, suggestions: store.suggestions, jobs: store.jobs, subAgents: store.subAgents }
let reviewRequests = 0
let modelRequests = 0
let nextTool = false
let compactionRequests = 0
const sse = (delta: unknown, finish = 'stop') => new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  const url = String(input?.url ?? input)
  assert.ok(url.startsWith(provider.baseUrl), 'fixture refuses any unexpected network route')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  const request = JSON.parse(String(init.body))
  const messages = request.messages ?? []
  const text = messages.map((message: any) => String(message.content ?? '')).join('\n')
  if (/You refine reusable skills/.test(text)) {
    reviewRequests++
    return sse({ content: JSON.stringify({ action: 'improve', body: 'Inspect the bounded fixture and report evidence.', rationale: 'Synthetic acceptance.' }) })
  }
  if (/summariz(?:e|ing)|compress.*transcript/i.test(String(messages[0]?.content ?? ''))) {
    compactionRequests++
    return sse({ content: 'The user prefers the synthetic lavenderwidget convention.' })
  }
  modelRequests++
  if (nextTool) {
    nextTool = false
    return sse({ tool_calls: [{ index: 0, id: 'synthetic-skill-call', type: 'function', function: { name: 'skill_view', arguments: JSON.stringify({ name: skill.name }) } }] }, 'tool_calls')
  }
  return sse({ content: 'The synthetic inspection is complete.' })
}) as typeof fetch

async function waitFor(predicate: () => boolean, label: string) {
  for (let i = 0; i < 400; i++) {
    if (predicate()) return
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error(`Timed out: ${label}`)
}
async function scenario(incognito: boolean, compact = false) {
  store.settings = { ...defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', autoCompact: compact, skillSelfImprovement: true, approvalMode: 'suggest' }
  store.projects = [{ id: 'synthetic-project', name: 'Synthetic project', cwd: home, memory: '', team: '', createdAt: 1, updatedAt: 1 }]
  store.projectMemory = []; store.projectMemoryCandidates = []; store.skills = [skill as any]; store.suggestions = []; store.jobs = []; store.subAgents = []
  const history = compact ? Array.from({ length: 48 }, (_, i) => ({ role: i % 2 ? 'assistant' as const : 'user' as const, content: i === 0 ? 'I always prefer lavenderwidget for synthetic demos.' : `Synthetic history ${i}: ${'bounded context '.repeat(25)}` })) : []
  const thread: any = { id: `synthetic-${incognito ? 'private' : 'normal'}-${compact ? 'compact' : 'post'}`, title: 'Synthetic fixture', kind: 'task', projectId: 'synthetic-project', cwd: '/synthetic-workspace', workspace: { kind: 'ssh', hostId: 'inert-fixture', path: '/synthetic-workspace' }, incognito, providerId: provider.id, providerAccountId: null, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: history, timeline: [] }
  store.threads = [thread]
  const aborters = new Map<string, AbortController>()
  const checkpoints: string[] = []
  const claims: any[] = []
  let settled = false
  let observed = 0
  const startHooks = hostPluginEvents
  const startReviews = reviewRequests
  const startCompactions = compactionRequests
  nextTool = !compact
  const runtime = createThreadRuntime({
    aborters, workspaceMutationGuard: new WorkspaceMutationGuard(), pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {}, observe: () => { observed++ },
    fullToolContext: () => ({ cwd: '/synthetic-workspace', backend: { kind: 'ssh' }, currentThread: thread, skills: [skill], packagedPlugins: [{ id: 'fixture-disclosure', name: 'Fixture disclosure', manifestPath: '/synthetic/manifest', enabled: true, trusted: true, checksumSha256: 'a'.repeat(64), toolNames: ['skill_view'] }] }), approvalResolver: () => async () => 'allow', isHighRiskTool: (name) => name === 'skill_view',
    claimJob: (ref) => { claims.push(ref); return { id: `job-${thread.id}`, status: 'running', incognito: ref.incognito, metadata: {}, pendingApprovals: [] } as any },
    patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob() { settled = true }, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => compact ? 1500 : 128000,
    captureCheckpoint: async () => { checkpoints.push('files'); return { ok: true, sha: 'synthetic' } }, recordTurnCheckpoint: () => { checkpoints.push('conversation') },
  })
  await runtime.startThreadRun({ threadId: thread.id, userText: compact ? 'Summarize this bounded history.' : 'I always prefer lavenderwidget for synthetic demos. Inspect using the synthetic-review skill.' })
  await waitFor(() => settled && !aborters.has(thread.id), 'actual host run settlement')
  await new Promise((resolve) => setTimeout(resolve, 80)) // detached normal skill review/journal writes settle
  const journals = await readdir(path.join(home, 'thread-journals')).catch(() => [] as string[])
  const auditRows = (await readFile(path.join(home, 'security-events.jsonl'), 'utf8').catch(() => '')).split('\n').filter(Boolean).map((line) => JSON.parse(line)).filter((event) => JSON.stringify(event).includes(thread.id))
  return { runtime, thread, checkpoints, claims, observed, hostHooks: hostPluginEvents - startHooks, auditRows, candidates: store.projectMemoryCandidates.slice(), suggestions: store.suggestions.slice(), reviews: reviewRequests - startReviews, compactions: compactionRequests - startCompactions, journal: journals.some((name) => name.includes(thread.id)) }
}
try {
  const privateRun = await scenario(true)
  assert.equal(privateRun.claims[0].incognito, true, 'private job classification is set before the first claim/persistence seam')
  assert.equal(privateRun.observed, 0, 'Incognito bypasses automatic API observer retention')
  assert.equal(privateRun.hostHooks, 0, 'Incognito never invokes the process-wide plugin event observer')
  assert.doesNotMatch(JSON.stringify(privateRun.auditRows), /SYNTHETIC-SKILL-CONTENT|synthetic-workspace|lavenderwidget/, 'private tool audits retain structure only, not args/output/cwd')
  assert.equal(privateRun.candidates.length, 0, 'Incognito success must not file a durable post-run memory candidate')
  assert.equal(privateRun.reviews, 0, 'Incognito must not send an automatic skill-review request containing its task')
  assert.equal(privateRun.suggestions.length, 0)
  assert.deepEqual(privateRun.checkpoints, [], 'Incognito must not capture files or conversation checkpoint snapshots')
  assert.equal(privateRun.journal, false, 'Incognito must not write an encrypted transcript journal either')
  const normalRun = await scenario(false)
  assert.equal(normalRun.claims[0].incognito, false)
  assert.ok(normalRun.observed > 0 && normalRun.hostHooks > 0, 'ordinary observer and plugin callback behavior remains enabled')
  assert.match(JSON.stringify(normalRun.auditRows), /SYNTHETIC-SKILL-CONTENT/, 'positive ordinary audit still records the tool evidence preview')
  assert.ok(normalRun.candidates.length > 0, 'ordinary saved runs retain reviewed memory proposals')
  assert.equal(normalRun.reviews, 1, 'ordinary opt-in skill review remains functional')
  assert.ok(normalRun.suggestions.length > 0)
  assert.deepEqual(normalRun.checkpoints, ['files', 'conversation'])
  assert.equal(normalRun.journal, true)
  const privateCompaction = await scenario(true, true)
  assert.ok(privateCompaction.thread.compactions?.length, 'fixture actually compacts the Incognito history')
  assert.equal(privateCompaction.candidates.length, 0, 'automatic compaction must not harvest Incognito user statements')
  assert.equal(privateCompaction.journal, false)
  const beforeRefusal = privateCompaction.thread.timeline.length
  const beforeRequests = modelRequests
  store.settings.providers = [{ ...provider, transport: 'codex-cli' }]
  await assert.rejects(() => privateCompaction.runtime.startThreadRun({ threadId: privateCompaction.thread.id, userText: 'Synthetic unsupported retry.' }), /keeps its own session history/)
  store.settings.providers = [provider]
  privateCompaction.thread.engine = 'omp'
  await assert.rejects(() => privateCompaction.runtime.startThreadRun({ threadId: privateCompaction.thread.id, userText: 'Synthetic unsupported retry.' }), /keeps its own session history/)
  delete privateCompaction.thread.engine
  await assert.rejects(() => privateCompaction.runtime.startThreadRun({ threadId: privateCompaction.thread.id, userText: 'Synthetic restored retry.' }, { incognito: false } as any), /saved or unrelated job/)
  await assert.rejects(() => privateCompaction.runtime.startThreadRun({ threadId: privateCompaction.thread.id, userText: 'Synthetic restored retry.' }, { incognito: true, threadId: 'different-private-chat', currentThreadId: 'different-private-chat' } as any), /saved or unrelated job/)
  assert.equal(privateCompaction.thread.timeline.length, beforeRefusal, 'unsupported private admission creates no conversation turn')
  assert.equal(modelRequests, beforeRequests, 'unsupported private admission sends no model request')
  for (const transport of ['codex-cli','claude-cli','opencode-cli','cursor-cli','pi-cli'] as const) assert.throws(() => assertIncognitoThreadTransportSupported({ incognito: true }, { transport } as any), /keeps its own session history/)
  assert.throws(() => assertIncognitoThreadTransportSupported({ incognito: true, engine: 'omp' }, provider), /keeps its own session history/)
  assert.doesNotThrow(() => assertIncognitoThreadTransportSupported({ incognito: true }, provider))
  assert.doesNotThrow(() => assertIncognitoThreadTransportSupported({ incognito: false }, { transport: 'codex-cli' }))
  console.log('Actual host Incognito runtime: no journal/checkpoint/automatic memory or skill contribution, positive normal-run controls and real automatic compaction passed.')
} finally {
  hostPluginRuntime.emitAgentEvent = originalPluginEvent
  globalThis.fetch = originalFetch
  Object.assign(store, saved)
  await rm(home, { recursive: true, force: true })
}
