import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-memory-runtime-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { createToolContextRuntime } = await import('./tool-context-runtime')
const { createThreadForkFactory } = await import('./thread-revert-fork')
const { createMissionRuntime } = await import('./mission-runtime')
const { createMission, startMission } = await import('../../agent/missions')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const provider = { id: 'memory-fixture', name: 'Synthetic API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9899/v1', apiKey: '', models: ['synthetic'] }
const skill = { id: 'synthetic-skill', name: 'synthetic-review', description: 'Synthetic fixture', body: 'Inspect the bounded fixture.', enabled: true, createdAt: 1, updatedAt: 1 }
const originalFetch = globalThis.fetch
let scenarioThread: any
let modelRequests = 0, reviewRequests = 0, nextSkill = false
let memoryPrompts: string[] = []
let holdReview: (() => void) | null = null
let reviewStarted: (() => void) | null = null
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates, skills: store.skills, suggestions: store.suggestions, jobs: store.jobs, subAgents: store.subAgents, rooms: store.rooms }
const sse = (delta: unknown, finish = 'stop') => new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'fixture never contacts an external route')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  const request = JSON.parse(String(init.body))
  const text = (request.messages ?? []).map((message: any) => String(message.content ?? '')).join('\n')
  if (/You refine reusable skills/.test(text)) {
    reviewRequests++
    if (reviewStarted) { const callback = reviewStarted; reviewStarted = null; callback(); await new Promise<void>((resolve) => { holdReview = resolve }) }
    return sse({ content: JSON.stringify({ action: 'improve', body: 'Inspect the bounded fixture and report evidence.', rationale: 'Synthetic acceptance.' }) })
  }
  modelRequests++
  memoryPrompts.push(text)
  if (scenarioThread && !scenarioThread.incognito) {
    const durable = (await storage.loadThreads()).find((t) => t.id === scenarioThread.id)
    assert.ok(durable, 'the chat exists in encrypted storage before every first API call')
    assert.deepEqual(durable.memoryPolicy, scenarioThread.memoryPolicy)
  }
  if (/summariz(?:e|ing)|compress.*transcript|verify.*summary/i.test(String(request.messages?.[0]?.content ?? ''))) return sse({ content: 'Synthetic bounded history summary.' })
  if (nextSkill) { nextSkill = false; return sse({ tool_calls: [{ index: 0, id: 'synthetic-skill-call', type: 'function', function: { name: 'skill_view', arguments: JSON.stringify({ name: skill.name }) } }] }, 'tool_calls') }
  return sse({ content: 'Synthetic inspection complete.' })
}) as typeof fetch
async function waitFor(predicate: () => boolean, label: string) { for (let i = 0; i < 600; i++) { if (predicate()) return; await new Promise((r) => setTimeout(r, 5)) }; throw new Error(`Timed out: ${label}`) }
async function scenario(useExisting: boolean, contribute: boolean, options: { compact?: boolean; staleReview?: boolean; incognito?: boolean; chat?: boolean; externalSource?: boolean } = {}) {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', autoCompact: options.compact === true, skillSelfImprovement: true, approvalMode: 'suggest' }
  store.projects = [{ id: 'synthetic-project', name: 'Synthetic project', cwd: home, memory: 'SAVED-PROJECT-AMBER', team: '', createdAt: 1, updatedAt: 1 }]
  store.projectMemory = []; store.projectMemoryCandidates = []; store.skills = [skill as any]; store.suggestions = []; store.jobs = []; store.subAgents = []
  const history = options.compact ? Array.from({ length: 48 }, (_, i) => ({ role: i % 2 ? 'assistant' as const : 'user' as const, content: i === 0 ? 'I always prefer lavenderwidget for synthetic demos.' : `Synthetic history ${i}: ${'bounded context '.repeat(25)}` })) : [{ role: 'user' as const, content: 'CURRENT-CONVERSATION-FACT' }]
  const thread: any = { id: `synthetic-${useExisting}-${contribute}-${JSON.stringify(options)}`, title: 'Synthetic fixture', kind: options.chat ? 'chat' : 'task', projectId: 'synthetic-project', cwd: '/synthetic-workspace', workspace: { kind: 'ssh', hostId: 'inert-fixture', path: '/synthetic-workspace' }, incognito: options.incognito, memoryPolicy: { useExisting, contribute }, providerId: provider.id, providerAccountId: null, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: history, timeline: options.compact ? [{ id: 'old-direct', kind: 'user', text: history[0].content, memorySource: 'user' }] : [] }
  store.threads = [thread]; scenarioThread = thread
  const aborters = new Map<string, AbortController>()
  const guard = new WorkspaceMutationGuard()
  let settled = false, renderedMemory = 0
  const beforeReview = reviewRequests
  memoryPrompts = []; nextSkill = !options.compact
  let startedReview: Promise<void> | null = null
  if (options.staleReview) startedReview = new Promise<void>((resolve) => { reviewStarted = resolve })
  const runtime = createThreadRuntime({
    aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {},
    fullToolContext: (run) => ({ cwd: '/synthetic-workspace', backend: { kind: 'ssh' }, currentThread: thread, memoryPolicy: run?.memoryPolicy, skills: [skill], packagedPlugins: [{ id: 'fixture-disclosure', name: 'Fixture disclosure', manifestPath: '/synthetic/manifest', enabled: true, trusted: true, checksumSha256: 'a'.repeat(64), toolNames: ['skill_view'] }] }), approvalResolver: () => async () => 'allow', isHighRiskTool: () => false,
    claimJob: () => ({ id: `job-${thread.id}`, status: 'running', metadata: {}, pendingApprovals: [] } as any), patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob() { settled = true }, renderProjectMemoryBucket: () => { renderedMemory++; return 'SAVED-PROJECT-AMBER' }, effectiveContextLimit: () => options.compact ? 1500 : 128000,
    captureCheckpoint: async () => ({ ok: true, sha: 'synthetic' }), recordTurnCheckpoint() {},
  })
  await runtime.startThreadRun({ threadId: thread.id, userText: options.compact ? 'Summarize this bounded history.' : 'I always prefer lavenderwidget for synthetic demos. Inspect using the synthetic-review skill.', ...(options.externalSource ? { sourceThreadKey: 'synthetic-connector-source' } : {}) })
  await waitFor(() => settled && !aborters.has(thread.id), 'actual host settlement')
  if (startedReview) {
    await startedReview
    await storage.saveThreadMemoryPolicy(() => store.threads, thread.id, { useExisting: true, contribute: false }, thread.memoryPolicyRevision)
    holdReview!(); holdReview = null
  }
  await new Promise((r) => setTimeout(r, 60))
  assert.equal(renderedMemory, useExisting ? 1 : 0, 'recall-off gate runs before host memory rendering')
  const memoryInjected = useExisting && !options.chat
  assert.equal(memoryPrompts.some((text) => text.includes('SAVED-PROJECT-AMBER')), memoryInjected)
  if (!options.compact) assert.ok(memoryPrompts.some((text) => text.includes('CURRENT-CONVERSATION-FACT')), 'already-seen current context remains available')
  const effectiveContribution = contribute && !options.incognito
  assert.equal(reviewRequests - beforeReview, !options.compact && effectiveContribution ? 1 : 0)
  assert.equal(store.suggestions.length > 0, !options.compact && effectiveContribution && !options.staleReview)
  assert.equal(store.projectMemoryCandidates.length > 0, effectiveContribution && !options.externalSource)
  assert.ok(store.projectMemoryCandidates.every((c) => c.status === 'pending'), 'contribution never auto-approves or trains')
  if (options.compact) assert.ok(thread.compactions?.length, 'real compaction was exercised')
  return { thread, runtime, aborters, guard }
}
try {
  for (const useExisting of [false, true]) for (const contribute of [false, true]) await scenario(useExisting, contribute)
  for (const contribute of [false, true]) await scenario(false, contribute, { compact: true })
  await scenario(false, true, { incognito: true })
  await scenario(true, false, { chat: true })
  await scenario(true, true, { externalSource: true })
  const stale = await scenario(true, true, { staleReview: true })
  assert.equal(store.suggestions.length, 0, 'late successful skill review cannot file after policy revision changes')
  const beforeRequests = modelRequests
  stale.thread.memoryPolicy = { useExisting: false, contribute: false }
  await assert.rejects(stale.runtime.startThreadRun({ threadId: stale.thread.id, userText: 'forged', memoryPolicy: { useExisting: true, contribute: true } } as any), /saved before/)
  store.settings.providers = [{ ...provider, transport: 'codex-cli' }]
  await assert.rejects(stale.runtime.startThreadRun({ threadId: stale.thread.id, userText: 'unsupported' }), /API/)
  assert.equal(modelRequests, beforeRequests)
  store.settings.providers = [provider]
  stale.thread.documentSourceIds = ['synthetic-selected-collection']
  const fork = createThreadForkFactory({ persistThreads() {}, remoteServer: () => ({ publish() {} }), spineHost: () => null })(stale.thread.id, { turn: 1, branch: 'synthetic', conversation: { apiMessages: stale.thread.apiMessages, timeline: [] } } as any)!
  assert.deepEqual(fork.memoryPolicy, { useExisting: false, contribute: false })
  assert.notEqual(fork.memoryPolicy, stale.thread.memoryPolicy)
  assert.deepEqual(fork.documentSourceIds, ['synthetic-selected-collection'])
  assert.notEqual(fork.documentSourceIds, stale.thread.documentSourceIds, 'A fork copies its collection allowlist without a mutable alias to the owner')
  scenarioThread = null
  const parent = stale.thread
  parent.memoryPolicyRevision = 40
  parent.workspace = null; parent.cwd = home
  const contextRuntime = createToolContextRuntime({ sshManager: { backendForWorkspace: () => null }, persistSubAgents() {}, persistThreads() {}, applyEventToThread: (stale.runtime as any).applyEventToThread, emitToRenderer() {}, emitPluginEvent() {}, renderProjectMemoryBucket: () => 'MUST-NOT-RECALL', recordEmailDraft() {}, recordEmailEvent() {}, recordCalendarEvent() {}, recordMessageEvent() {}, runBrowserAutomation() {}, mcpToolContext: () => ({}), cancelThreadRun() {} } as any)
  store.rooms = [{ id: 'room-recall-owner', name: 'Synthetic room', memory: [{ text: 'SAVED-ROOM-MEMORY' }], memorySummary: 'SAVED-ROOM-SUMMARY', sharedMemory: { blocks: { decision: { content: 'SAVED-ROOM-SCRATCH' } } }, finalText: 'SAVED-ROOM-FINAL', events: [{ kind: 'speak', text: 'SAVED-ROOM-CONVERSATION' }, { kind: 'status', text: 'LIVE-ROOM-STATUS' }] } as any]
  const parentController = new AbortController()
  const ctx = contextRuntime.fullToolContext({ thread: parent, provider, model: 'synthetic', signal: parentController.signal })
  assert.doesNotMatch(JSON.stringify(ctx.rooms), /SAVED-ROOM-/)
  assert.match(JSON.stringify(ctx.rooms), /LIVE-ROOM-STATUS/)
  assert.throws(() => ctx.searchSessions({ query: 'AMBER' }), /off/)
  assert.throws(() => ctx.fileSuggestion({ name: 'direct-host', body: 'Synthetic.' }), /off/)
  const child = await ctx.launchSubAgent({ task: 'Inspect a bounded synthetic assignment.' })
  assert.ok(child.output.includes('Synthetic inspection complete'))
  assert.ok(!memoryPrompts.some((text) => text.includes('MUST-NOT-RECALL')), 'actual app-owned child never reads inherited project memory')
  const ownedAgent = store.subAgents[0]
  await contextRuntime.runReadOnlySubAgent(ownedAgent, provider, 'synthetic', { task: 'Direct host child call with omitted policy.', parentLabel: 'Synthetic parent', cwd: home, signal: parentController.signal })
  assert.ok(!memoryPrompts.some((text) => text.includes('MUST-NOT-RECALL')), 'a direct host child call inherits owner flags even when the optional input snapshot is absent')
  parent.memoryPolicy = { useExisting: true, contribute: true }; parent.memoryPolicyRevision++
  await assert.rejects(ctx.launchSubAgent({ task: 'Stale parent.' }), /parent run/)
  assert.throws(() => ctx.fileSuggestion({ name: 'stale-host', body: 'Synthetic.' }), /off|no longer/)
  parent.memoryPolicy = { useExisting: false, contribute: false }; parent.memoryPolicyRevision++
  let mission = startMission(createMission({ title: 'Synthetic linked mission', objective: 'Inspect only synthetic text.', threadId: parent.id, cwd: home, providerId: provider.id, model: 'synthetic', maxRuns: 1 }))
  let missionMemoryReads = 0, missionOutcome = '', missionReason = ''
  const missionRuntime = createMissionRuntime({
    resolveWorkspace: async (cwd: string) => { assert.equal(cwd, home); return home },
    runningMissions: new Set(), missionAborters: new Map(), missionApprovalResolvers: new Map(),
    claimJob: () => ({ id: 'synthetic-mission-job', metadata: {} } as any), patchJob: (job: any, patch: any) => ({ ...job, ...patch }), heartbeatJob: (job: any) => job,
    finishJob: (_job: any, outcome: string, reason: string) => { missionOutcome = outcome; missionReason = reason }, getMission: () => mission, replaceMission: (next: any) => { mission = next; return next },
    fullToolContext: contextRuntime.fullToolContext, renderProjectMemoryBucket: () => { missionMemoryReads++; return 'MUST-NOT-RECALL' }, applyEventToThread: (stale.runtime as any).applyEventToThread,
    timelineErrorSince: () => null, timelineError: () => new Error('Synthetic'), isHighRiskTool: () => false, pendingApprovalMeta: new Map(), persistJobApproval() {}, resolvePersistedJobApproval() { return null }, markRouteErrorRecovered() {}, injectThreadRouteHandoff: () => '', effectiveContextLimit: () => 128000, workspaceMutationGuard: new WorkspaceMutationGuard(),
  } as any)
  await missionRuntime.runMission(mission)
  assert.equal(missionMemoryReads, 0, 'linked mission child gates memory before host rendering')
  assert.equal(missionOutcome, 'completed', `restricted linked mission still completes its ordinary synthetic task: ${missionReason}`)
  assert.ok(!memoryPrompts.some((text) => text.includes('MUST-NOT-RECALL')))
  parent.kind = 'chat'
  const requestsBeforeFailure = modelRequests
  const timelineBeforeFailure = parent.timeline.length, messagesBeforeFailure = parent.apiMessages.length
  const originalRename = fs.rename
  fs.rename = async (from, to) => { if (String(to) === storage.threadsPath()) throw new Error('Synthetic memory durability failure'); return originalRename(from, to) }
  try {
    await assert.rejects(stale.runtime.startThreadRun({ threadId: parent.id, userText: 'Must never reach the model before durable acceptance.' }), /Synthetic memory durability failure/)
    assert.equal(modelRequests, requestsBeforeFailure)
    assert.equal(parent.timeline.length, timelineBeforeFailure)
    assert.equal(parent.apiMessages.length, messagesBeforeFailure)
    assert.equal(stale.aborters.size, 0)
    stale.guard.reserveStart({ threadId: parent.id }).release()
  } finally { fs.rename = originalRename }
  console.log('Actual native host + API-loop + encrypted store: all four policies, compaction/post-run/skill contribution, recall-before-render, current history, Incognito, authored connector provenance, stale reviews, malformed override/CLI refusal, fork and owned-child inheritance passed')
} finally { await new Promise((r) => setTimeout(r, 100)); await Promise.allSettled([...storage.writeChains.values()]); globalThis.fetch = originalFetch; Object.assign(store, saved); storage.closeSessionSearchIndexes(); await rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 }) }
