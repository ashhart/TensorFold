import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { captureBotConfig } from '../../../shared/bot-policy'
import { buildPersonalRoutine } from '../../../shared/personal-routines'
import type { AgentPreset, Job, Provider, Thread } from '../../agent/types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-bot-effort-runtime-'))
const previousHome = process.env.VONTRA_HOME
process.env.VONTRA_HOME = profile
const requests: Record<string, unknown>[] = []
const server = createServer((request, response) => {
  if (request.method === 'GET') {
    response.writeHead(200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify({ data: [{ id: 'fixture', object: 'model' }] }))
    return
  }
  let raw = ''
  request.on('data', (chunk) => { raw += chunk })
  request.on('end', () => {
    requests.push(JSON.parse(raw))
    response.writeHead(200, { 'Content-Type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic review complete.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
try {
  const { defaultSettings, saveThreads } = await import('../../store')
  const { store } = await import('../state')
  const { createThreadRuntime } = await import('./thread-runtime')
  const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
  const { compactThreadJournal } = await import('../../agent/thread-journal')
  const { createAutomationRuntime } = await import('./automations')
  const provider: Provider = { id: 'effort-fixture', name: 'Local fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: ['fixture'] }
  const preset: AgentPreset = { id: 'bot_effort', name: 'Reviewer', role: 'Review supplied text', lane: 'review', providerId: provider.id, model: 'fixture', reasoningEffort: 'low', toolScope: { mode: 'selected', names: [] }, createdAt: 1, updatedAt: 1 }
  const captured = captureBotConfig(preset, [provider])
  store.settings = { ...defaultSettings(), providers: [provider], agentPresets: [{ ...preset, reasoningEffort: 'high' }], activeProviderId: provider.id, activeModel: 'fixture', reasoningEffort: 'high', autoCompact: false }
  const makeThread = (id: string, bot = captured): Thread => ({ id, title: 'Synthetic review', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'fixture', bot, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] })
  store.threads = [makeThread('captured-low'), makeThread('captured-high', { ...captured, reasoningEffort: 'high' }), makeThread('legacy', { ...captured, reasoningEffort: undefined })]
  await saveThreads(store.threads)
  const aborters = new Map<string, AbortController>()
  let settled: (outcome: string) => void = () => {}
  const runtime = createThreadRuntime({
    aborters, workspaceMutationGuard: new WorkspaceMutationGuard(), pendingApprovalMeta: new Map(), jobLeaseOwner: 'fixture', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {},
    fullToolContext: ({ thread } = {}) => ({ cwd: profile, currentThread: thread }), approvalResolver: () => async () => 'deny', isHighRiskTool: () => false,
    claimJob: () => ({ id: 'fixture-job', status: 'running' } as Job), patchJob: (job) => job, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob: (_job, _thread, outcome) => settled(outcome), renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 8192,
  })
  for (const [id, globalEffort, expected] of [['captured-low', 'high', 'low'], ['captured-high', 'low', 'high'], ['legacy', 'high', 'high']] as const) {
    store.settings.reasoningEffort = globalEffort
    let timer: ReturnType<typeof setTimeout> | undefined
    const finished = new Promise<string>((resolve, reject) => {
      settled = resolve
      timer = setTimeout(() => { runtime.cancelThreadRun(id); reject(new Error('Synthetic bot run did not settle.')) }, 10_000)
    })
    try {
      await runtime.startThreadRun({ threadId: id, userText: 'Review this synthetic text.' })
      const outcome = await finished
      assert.equal(outcome, 'completed', JSON.stringify(store.threads.find((thread) => thread.id === id)?.timeline))
      assert.equal(requests.at(-1)?.reasoning_effort, expected, `${id} sends its captured effort through the actual host and HTTP request`)
      assert.equal(aborters.size, 0)
    } finally {
      clearTimeout(timer)
      await compactThreadJournal(path.join(profile, 'thread-journals'), id, 1).catch(() => {})
    }
  }
  let afterCapture = () => {}
  const automations = createAutomationRuntime({
    threadJobHeartbeatMs: 5000, claimJob: () => ({ id: 'routine-job' } as Job), patchJob: (job) => job, heartbeatJob: (job) => job, finishJob() {}, emitAutomation() {},
    effectiveContextLimit: () => { afterCapture(); return 8192 }, fullToolContext: () => { throw new Error('Personal routines must not request global tools.') },
    applyEventToThread: runtime.applyEventToThread, timelineErrorSince: runtime.timelineErrorSince, timelineError: runtime.timelineError,
    isHighRiskTool: () => false, rememberConnectorEvents: () => { throw new Error('Preview routines must not deliver external messages.') },
  })
  for (const [savedEffort, globalEffort, expected] of [['low', 'high', 'low'], ['high', 'low', 'high'], [undefined, 'high', 'high']] as const) {
    store.settings.agentPresets[0].reasoningEffort = savedEffort
    store.settings.reasoningEffort = globalEffort
    afterCapture = () => { store.settings.agentPresets[0].reasoningEffort = 'medium' }
    const routine = automations.mkAutomation(buildPersonalRoutine({ kind: 'morning-brief', botId: preset.id, providerId: provider.id, model: 'fixture', sourceScope: [], timeZone: 'Etc/UTC' }))
    await automations.runAutomation(routine)
    assert.equal(routine.runs[0].status, 'ok', routine.runs[0].error ?? '')
    assert.equal(requests.at(-1)?.reasoning_effort, expected, 'a routine captures its bot effort before startup awaits or preset edits')
  }
  assert.equal(requests.length, 6)
  console.log('Actual bot/routine host and HTTP checks passed: independent captured efforts, preset edits and legacy fallback.')
} finally {
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
  await (await import('./persistence')).preserveStateForUpdate()
  if (previousHome === undefined) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = previousHome
  await fs.rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 30 })
}
