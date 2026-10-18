import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { Thread } from '../../agent/types'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-telegram-native-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const storage = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { createTelegramRemoteDispatch } = await import('./telegram-remote-dispatch')
const { createTelegramRemoteService } = await import('../../agent/telegram-remote/service')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const { assertGoogleContextTransport } = await import('../../../shared/google-workspace-privacy')
const provider = { id: 'synthetic-native', name: 'Synthetic local transport', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9899/v1', apiKey: '', models: ['synthetic'] }
const previousFetch = globalThis.fetch
const previousRename = fs.rename
const saved = { settings: store.settings, threads: store.threads, projects: store.projects, jobs: store.jobs, subAgents: store.subAgents, skills: store.skills, suggestions: store.suggestions, projectMemory: store.projectMemory, projectMemoryCandidates: store.projectMemoryCandidates }
let modelRequests = 0
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'all real transport calls remain in the synthetic local fixture')
  if (!init?.body) return Response.json({ data: [{ id: 'synthetic' }] })
  const body = JSON.parse(String(init.body))
  assert.ok(!JSON.stringify(body).includes('synthetic-telegram-token'))
  modelRequests++
  return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'Native TensorFold reply to selected chat.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch

try {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', autoCompact: false, skillSelfImprovement: false, approvalMode: 'readonly' }
  store.settings.botControl.telegram = { ...store.settings.botControl.telegram, enabled: true, polling: false, botToken: 'synthetic-telegram-token', allowedChatIds: '7', allowedUserIds: '7',
    remote: { enabled: true, scope: 'selected', threadIds: ['selected'], roomIds: [], revision: 1, sessions: [] } }
  const thread: Thread = { id: 'selected', title: 'Synthetic selected chat', kind: 'chat', cwd: home, providerId: provider.id, providerAccountId: null, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] }
  store.threads = [thread]; store.projects = []; store.jobs = []; store.subAgents = []; store.skills = []; store.suggestions = []; store.projectMemory = []; store.projectMemoryCandidates = []
  const aborters = new Map<string, AbortController>()
  const guard = new WorkspaceMutationGuard()
  const runtime = createThreadRuntime({
    aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {},
    fullToolContext: () => ({ cwd: home, currentThread: thread, threads: store.threads, rooms: [], projects: [], settings: store.settings,
      googlePrivacyGuard: () => { assertGoogleContextTransport(thread, provider); return null } }), approvalResolver: () => async () => 'deny', isHighRiskTool: () => true,
    claimJob: () => ({ id: 'synthetic-job', status: 'running', metadata: {}, pendingApprovals: [] }) as any,
    patchJob: (job, patch) => job ? ({ ...job, ...patch }) : null, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job,
    settleThreadRunJob() {}, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 128000,
  })
  const dispatch = createTelegramRemoteDispatch({ settings: () => store.settings, threads: () => store.threads, rooms: () => [], aborters, requireOperator() {},
    startThreadRun: (payload, admission) => runtime.startThreadRun(payload, undefined, admission), steerRoom() { throw new Error('No room in this fixture') },
    streamChat: async function* () { throw new Error('Selected chat must use the canonical native runtime') }, waitMs: 5_000, pollMs: 5,
  })
  const remote = createTelegramRemoteService({ settings: () => store.settings.botControl.telegram, threads: () => store.threads, rooms: () => [], persist: () => storage.saveSettings(store.settings), dispatchTurn: dispatch })
  const update = (id: number, text: string) => ({ update_id: id, message: { chat: { id: 7, type: 'private' }, from: { id: 7, is_bot: false }, text } })
  assert.match((await remote.ingest(update(1, '/chat selected'))).reply, /selected/)
  const response = await remote.ingest(update(2, 'Respond to this synthetic task.'))
  assert.equal(response.reply, 'Native TensorFold reply to selected chat.')
  assert.equal(modelRequests, 1, 'real host runtime invokes the synthetic HTTP model exactly once')
  assert.equal(thread.timeline.filter((item) => item.kind === 'user').length, 1)
  assert.ok(thread.timeline.some((item) => item.kind === 'assistant' && item.text === response.reply))
  assert.equal(thread.providerId, provider.id); assert.equal(thread.model, 'synthetic'); assert.equal(thread.cwd, home)
  assert.equal(thread.lastRunReceipt?.runId, thread.timeline.find((item) => item.kind === 'assistant')?.trace?.runId)
  assert.equal(aborters.size, 0)

  let release!: () => void
  let entered!: () => void
  const inRename = new Promise<void>((resolve) => { entered = resolve })
  const pause = new Promise<void>((resolve) => { release = resolve })
  fs.rename = async (from, to) => { if (String(to) === storage.threadsPath()) { entered(); await pause } return previousRename(from, to) }
  let allowed = true
  let captured: AbortController | null = null
  const beforeUsers = thread.timeline.filter((item) => item.kind === 'user').length
  const beforeMessages = thread.apiMessages.length
  const attempt = runtime.startThreadRun({ threadId: thread.id, userText: 'Never append this revoked task.' }, undefined, {
    assertAllowed: () => { if (!allowed) throw new Error('Synthetic remote admission revoked') }, onController: (controller) => { captured = controller },
  }).then(() => null, (error) => error as Error)
  await inRename
  allowed = false
  release()
  assert.match((await attempt)?.message ?? '', /admission revoked/)
  assert.ok(captured, 'host receives the exact new controller before asynchronous startup')
  assert.equal(thread.timeline.filter((item) => item.kind === 'user').length, beforeUsers)
  assert.equal(thread.apiMessages.length, beforeMessages)
  assert.equal(modelRequests, 1, 'revocation during protected asynchronous admission reaches no model')
  assert.equal(aborters.size, 0)
  guard.assertThreadStartAllowed(thread.id)
  fs.rename = previousRename
  thread.googleLocalOnly = true
  store.settings.providers = [{ ...provider, kind: 'cloud', baseUrl: 'https://model.example.com/v1' }]
  const beforeGoogleUsers = thread.timeline.filter((item) => item.kind === 'user').length
  await assert.rejects(() => runtime.startThreadRun({ threadId: thread.id, userText: 'Must not export protected Google history.' }), /private Google account data/)
  assert.equal(modelRequests, 1)
  assert.equal(thread.timeline.filter((item) => item.kind === 'user').length, beforeGoogleUsers)
  store.settings.providers = [provider]
  await runtime.startThreadRun({ threadId: thread.id, userText: 'Continue the protected history locally.' })
  for (let i = 0; i < 400 && aborters.has(thread.id); i++) await new Promise((resolve) => setTimeout(resolve, 5))
  assert.equal(aborters.size, 0)
  assert.equal(modelRequests, 2, `protected Google history can still run in the canonical local native model: ${thread.timeline.filter((item) => item.kind === 'error').map((item) => item.text).join('; ')}`)
  console.log('Actual Telegram service to native host, encrypted save and synthetic HTTP model returns the selected final answer; asynchronous admission revocation appends no turn and calls no model')
} finally {
  fs.rename = previousRename
  globalThis.fetch = previousFetch
  await new Promise((resolve) => setTimeout(resolve, 100))
  await Promise.allSettled([...storage.writeChains.values()])
  Object.assign(store, saved)
  storage.closeSessionSearchIndexes()
  await fs.rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 })
}
