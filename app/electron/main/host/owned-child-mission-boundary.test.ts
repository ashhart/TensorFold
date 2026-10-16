import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-owned-child-boundary-'))
const previousHome = process.env.VONTRA_HOME
process.env.VONTRA_HOME = home
const { store } = await import('../state')
const storage = await import('../../store')
const { createToolContextRuntime } = await import('./tool-context-runtime')
const { createThreadRuntime } = await import('./thread-runtime')
const { createMissionRuntime, assertMissionWorkspaceSupported } = await import('./mission-runtime')
const { createMission, startMission } = await import('../../agent/missions')
const { createSubAgentFromLaunch, parentForThread } = await import('../../agent/subagents')
const { captureBotConfig } = await import('../../../shared/bot-policy')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const { executeTool } = await import('../../agent/tools')
const previousStore = { ...store }, originalFetch = globalThis.fetch
const provider: any = { id: 'owned-boundary-api', name: 'Synthetic local API', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9899/v1', apiKey: '', models: ['fixture'] }
let requests = 0, projections = 0, recordsSaved = 0, childContexts = 0
globalThis.fetch = (async (_url: any, init?: RequestInit) => {
  if (!init?.body) return Response.json({ data: [{ id: 'fixture' }] })
  requests++
  return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic ordinary child complete.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
const fold = createThreadRuntime({} as any).applyEventToThread
const runtime = createToolContextRuntime({ sshManager: { backendForWorkspace: () => ({ kind: 'ssh' }) }, persistSubAgents() { recordsSaved++ }, persistThreads() {}, applyEventToThread(thread: any, event: any) { childContexts++; fold(thread, event) }, emitToRenderer() { projections++ }, emitPluginEvent() {}, renderProjectMemoryBucket: () => '', recordEmailDraft() {}, recordEmailEvent() {}, recordCalendarEvent() {}, recordMessageEvent() {}, runBrowserAutomation() {}, mcpToolContext: () => ({}), cancelThreadRun() {} } as any)
const owner = (kind: string): any => {
  const bot = kind === 'ordinary' || kind === 'phone' ? undefined : captureBotConfig({ id: `bot_owner_${kind}`, name: 'Synthetic owner', role: 'Inspect synthetic material', lane: 'review', instructions: '', providerId: provider.id, model: 'fixture', executionTarget: kind === 'vm' ? 'vm' : 'workspace', approvalScope: kind === 'review' ? 'review' : 'readonly', ...(kind === 'selected' ? { toolScope: { mode: 'selected', names: [] } } : {}), createdAt: 1, updatedAt: 1 }, [provider])
  if (kind === 'malformed') bot!.toolScope = { mode: 'selected', names: 'damaged' } as any
  if (kind === 'quarantined') bot!.toolScopeProblem = null as any
  return { id: `owner_${kind}`, title: 'Synthetic owner', kind: 'chat', cwd: kind === 'vm' ? '/workspace' : home, workspace: null, bot: kind === 'nullbot' ? null : bot, phoneLocalOnly: kind === 'phone', providerId: provider.id, providerAccountId: null, model: 'fixture', createdAt: 1, updatedAt: 1, status: 'idle', memoryPolicy: { useExisting: false, contribute: false }, memoryPolicyRevision: 0, timeline: [], apiMessages: [{ role: 'user', content: 'OWNED_SYNTHETIC_CONTEXT_AMBER' }] }
}
let createHandler: any
const ipc = await fs.readFile(path.join(process.cwd(), 'electron/main/host/ipc/missions-ipc.ts'), 'utf8')
const tree = ts.createSourceFile('missions-ipc.ts', ipc, ts.ScriptTarget.Latest, true)
let node: ts.Node | undefined
const visit = (current: ts.Node) => { if (ts.isCallExpression(current) && current.expression.getText(tree) === 'handle' && current.arguments[0]?.getText(tree) === "'missions:create'") node = current; ts.forEachChild(current, visit) }
visit(tree); assert.ok(node)
const code = ts.transpileModule(ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, tree), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
let constructed = 0, missionSaved = 0
new Function('handle', 'requireSecurityRole', 'assertMissionWorkspaceSupported', 'store', 'createMission', 'selectedProviderAccountId', 'persistMissions', 'recordMissionCreated', code)(
  (_channel: string, handler: any) => { createHandler = handler }, () => {}, assertMissionWorkspaceSupported, store,
  (input: any) => { constructed++; return createMission(input) }, () => null, () => { missionSaved++ }, () => {},
)
async function missionRun(source: any, canonicalSource?: any) {
  let mission = startMission(createMission({ title: 'Synthetic linked mission', objective: 'OWNED_SYNTHETIC_CONTEXT_AMBER', threadId: source.id, cwd: home, providerId: provider.id, model: 'fixture', maxRuns: 1 }))
  const canonical = canonicalSource ? { ...mission, threadId: canonicalSource.id } : null
  let claimed = 0, workspace = 0, outcome: string | undefined
  const running = new Set<string>(), aborters = new Map<string, AbortController>()
  const missionRuntime = createMissionRuntime({ resolveWorkspace: async () => { workspace++; return home }, runningMissions: running, missionAborters: aborters, missionApprovalResolvers: new Map(), claimJob: () => { claimed++; return { id: 'synthetic-job', metadata: {} } }, patchJob: (job: any, patch: any) => ({ ...job, ...patch }), heartbeatJob: (job: any) => job, finishJob: (_job: any, value: string) => { outcome = value }, getMission: () => canonical ?? mission, replaceMission: (next: any) => { mission = next; return next }, fullToolContext: runtime.fullToolContext, renderProjectMemoryBucket: () => '', applyEventToThread: fold, timelineErrorSince: () => null, timelineError: () => new Error('Synthetic fixture'), isHighRiskTool: () => false, pendingApprovalMeta: new Map(), persistJobApproval() {}, resolvePersistedJobApproval: () => null, markRouteErrorRecovered() {}, injectThreadRouteHandoff: () => '', effectiveContextLimit: () => 128000, workspaceMutationGuard: new WorkspaceMutationGuard() } as any)
  return { invoke: () => missionRuntime.runMission(mission), state: () => ({ claimed, workspace, outcome, running: running.size, aborters: aborters.size }) }
}
try {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'fixture' }
  store.projects = []; store.skills = []; store.commands = []; store.projectMemory = []; store.rooms = []; store.suggestions = []; store.missions = []
  const ordinary = owner('ordinary')
  for (const kind of ['selected', 'vm', 'readonly', 'review', 'malformed', 'quarantined', 'nullbot', 'phone']) {
    const source = owner(kind)
    store.threads = [source, ordinary]; store.subAgents = []
    const before = { requests, projections, recordsSaved, childContexts, constructed, missionSaved }
    const run: any = { thread: source, provider, model: 'fixture', signal: new AbortController().signal, memoryOwner: { id: ordinary.id, revision: 0 } }
    const ctx = runtime.fullToolContext(run)
    run.thread = ordinary
    await assert.rejects(ctx.launchSubAgent({ task: 'OWNED_SYNTHETIC_CONTEXT_AMBER', parentId: ordinary.id } as any), /cannot launch child agents/)
    const originalBot = source.bot, originalPhone = source.phoneLocalOnly
    source.bot = undefined; source.phoneLocalOnly = false
    await assert.rejects(ctx.launchSubAgent({ task: 'Synthetic captured-owner downgrade.' }), /cannot launch child agents/, 'the immutable captured admission bit survives mutation of the same owner object')
    source.bot = originalBot; source.phoneLocalOnly = originalPhone
    assert.equal(store.subAgents.length, 0, 'closed-over guarded owner refuses before creating a subagent; supplied ordinary parent cannot downgrade it')
    const created = createSubAgentFromLaunch([], { source: 'thread_tool', parent: parentForThread(source), task: 'Synthetic assignment', providerId: provider.id, model: 'fixture' })
    store.subAgents = created.agents
    const forged = { ...created.agent, threadId: ordinary.id, parent: parentForThread(ordinary) }
    await assert.rejects(runtime.runReadOnlySubAgent(forged, provider, 'fixture', { task: 'OWNED_SYNTHETIC_CONTEXT_AMBER', parentLabel: 'Forged ordinary parent', cwd: home, signal: new AbortController().signal, memoryOwner: { id: ordinary.id, revision: 0 } }), /cannot launch child agents/)
    assert.equal(store.subAgents.length, 1, 'direct helper neither creates nor settles the pre-existing record when admission is refused')
    await assert.rejects(createHandler({}, { title: 'Synthetic mission', objective: 'Synthetic task', threadId: source.id }), /cannot launch child agents/)
    const mission = await missionRun(source)
    await assert.rejects(mission.invoke(), /cannot launch child agents/)
    assert.deepEqual(mission.state(), { claimed: 0, workspace: 0, outcome: undefined, running: 0, aborters: 0 })
    assert.deepEqual({ requests, projections, recordsSaved, childContexts, constructed, missionSaved }, before, 'all guarded entry seams refuse before provider/context/model, job, workspace, persistence and observer work')
  }
  const realOwner = owner('selected')
  store.threads = [ordinary, realOwner]
  const forgedMission = await missionRun(ordinary, realOwner)
  const requestsBeforeForgery = requests
  await assert.rejects(forgedMission.invoke(), /cannot launch child agents/)
  assert.deepEqual(forgedMission.state(), { claimed: 0, workspace: 0, outcome: undefined, running: 0, aborters: 0 })
  assert.equal(requests, requestsBeforeForgery, 'a caller-supplied ordinary parent cannot downgrade a host-owned guarded mission')
  store.threads = [ordinary]; store.subAgents = []
  const ctx = runtime.fullToolContext({ thread: ordinary, provider, model: 'fixture', signal: new AbortController().signal })
  const before = requests
  const launched = await ctx.launchSubAgent({ task: 'Ordinary synthetic inspection.' })
  assert.equal(launched.agent.status, 'succeeded'); assert.equal(requests, before + 1)
  await runtime.runReadOnlySubAgent(store.subAgents[0], provider, 'fixture', { task: 'Ordinary direct synthetic inspection.', parentLabel: 'Ordinary owner', cwd: home, signal: new AbortController().signal })
  assert.equal(requests, before + 2)
  const created = await createHandler({}, { title: 'Ordinary mission', objective: 'Synthetic inspection', threadId: ordinary.id })
  assert.equal(created.threadId, ordinary.id); assert.equal(constructed, 1); assert.equal(missionSaved, 1)
  const mission = await missionRun(ordinary)
  await mission.invoke()
  assert.equal(mission.state().outcome, 'completed'); assert.equal(mission.state().claimed, 1); assert.equal(requests, before + 3)
  await assert.rejects(runtime.runReadOnlySubAgent({ ...store.subAgents[0], id: 'not-registered' }, provider, 'fixture', { task: 'Synthetic', parentLabel: 'Ordinary', cwd: home, signal: new AbortController().signal }), /unregistered child/)
  const ownAgent = store.subAgents[0]
  const child = { ...ordinary, id: `thread-${ownAgent.id}`, apiMessages: [], timeline: [] }
  const operational = runtime.fullToolContext({ thread: child, provider, model: 'fixture', signal: new AbortController().signal, subAgentId: ownAgent.id, memoryPolicy: { useExisting: false, contribute: false }, memoryOwner: { id: ordinary.id, revision: 0 } })
  assert.deepEqual(operational.readSubAgentMessages({ agentId: ownAgent.id, markRead: false }), [], 'recall-disabled current child can read its own operational inbox')
  assert.equal(operational.sendSubAgentMessage({ agentId: ownAgent.id, kind: 'progress', text: 'Synthetic current worker progress.' }).text, 'Synthetic current worker progress.')
  const ownStatus = await executeTool('subagent_status', { agent_ids: [ownAgent.id] }, { cwd: home, currentThread: child, ...operational })
  assert.equal(ownStatus.output.includes(ownAgent.id), true, 'current worker status remains available without ambient saved-history recall')
  ordinary.phoneLocalOnly = true
  assert.throws(() => operational.readSubAgentMessages({ agentId: ownAgent.id }), /Private sub-agent history/, 'an operational exception never overrides the live canonical parent phone boundary')
  assert.throws(() => operational.sendSubAgentMessage({ agentId: ownAgent.id, text: 'Refused synthetic progress.' }), /Private sub-agent history/)
  console.log('Actual owner-bound child/mission admission passed: eight guarded/damaged owner shapes across closed-over launch, forged direct read-only helper, registered creation and runtime preflight; zero provider/context/job/workspace/model callbacks, ordinary readonly/mission and memory inheritance positives preserved.')
} finally {
  while (storage.writeChains.size) await Promise.allSettled([...storage.writeChains.values()])
  globalThis.fetch = originalFetch; Object.assign(store, previousStore); storage.closeSessionSearchIndexes(); await fs.rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 }); if (previousHome === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = previousHome
}
