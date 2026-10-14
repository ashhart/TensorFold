import assert from 'node:assert/strict'
import { captureBotConfig } from '../../../shared/bot-policy'
import { defaultSettings } from '../../store'
import { store } from '../state'
import { createThreadRuntime } from './thread-runtime'
import { initializeBotVmHost } from './bot-vm-host'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'
import type { Provider, Thread } from '../../agent/types'

const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '', models: ['fixture'] }
const preset = { id: 'bot_fixture', name: 'Guest reviewer', role: 'Review guest files', lane: 'review' as const, providerId: provider.id, model: 'fixture', executionTarget: 'vm' as const, createdAt: 1, updatedAt: 1 }
const bot = captureBotConfig(preset, [provider])
const makeThread = (id: string): Thread => ({ id, title: 'Guest reviewer chat', kind: 'chat', cwd: '/workspace', workspace: null, providerId: provider.id, providerAccountId: null, model: 'fixture', bot: { ...bot }, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] })
const saved = { settings: store.settings, threads: store.threads }
store.settings = { ...defaultSettings(), providers: [provider], agentPresets: [preset], activeProviderId: provider.id, activeModel: 'fixture' }
store.threads = [makeThread('chat-a'), makeThread('chat-b')]
const aborters = new Map<string, AbortController>()
const guard = new WorkspaceMutationGuard()
const runtime = createThreadRuntime({ aborters, workspaceMutationGuard: guard, pendingApprovalMeta: new Map(), jobLeaseOwner: 'fixture', threadJobHeartbeatMs: 5000, sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {}, fullToolContext: () => { throw new Error('Host context cannot be created during VM startup.'); }, approvalResolver: () => async () => 'deny', isHighRiskTool: () => false, claimJob: () => { throw new Error('Readiness tests must stop before model execution.'); }, patchJob: (job) => job, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job, settleThreadRunJob() {}, renderProjectMemoryBucket: () => { throw new Error('Host project memory cannot be read during VM startup.'); }, effectiveContextLimit: () => 4096 })
const vm = initializeBotVmHost({ appHome: '/tmp/tensorfold-bot-start-fixture', supported: false })
const originalReady = vm.requireReady
let entered = 0
let release!: () => void
let gate: Promise<void>
const readyGate = (fail: boolean) => {
  gate = new Promise<void>((resolve) => { release = resolve })
  vm.requireReady = async () => { entered++; await gate; if (fail) throw new Error('Synthetic VM readiness failure.') }
}
try {
  readyGate(true)
  const first = runtime.startThreadRun({ threadId: 'chat-a', userText: 'Read guest notes.' }).then(() => null, (error) => error as Error)
  assert.equal(entered, 1)
  await assert.rejects(() => runtime.startThreadRun({ threadId: 'chat-b', userText: 'Read other guest notes.' }), /already working/, 'another chat cannot enter asynchronous startup for the same VM')
  assert.equal(entered, 1, 'only the admitted run reaches VM readiness')
  guard.assertThreadStartAllowed('chat-b')
  release()
  assert.match((await first)!.message, /Synthetic VM readiness failure/)
  assert.equal(aborters.size, 0)
  guard.assertThreadStartAllowed('chat-a')

  readyGate(false)
  const stopping = runtime.startThreadRun({ threadId: 'chat-b', userText: 'Read guest notes.' }).then(() => null, (error) => error as Error)
  assert.equal(entered, 2, 'failed startup released the per-bot reservation')
  runtime.cancelThreadRun('chat-b')
  assert.equal(aborters.get('chat-b')?.signal.aborted, true)
  release()
  assert.match((await stopping)!.message, /cancelled before VM startup/)
  assert.equal(aborters.size, 0)
  guard.assertThreadStartAllowed('chat-b')

  readyGate(true)
  const afterStop = runtime.startThreadRun({ threadId: 'chat-a', userText: 'Try again.' }).then(() => null, (error) => error as Error)
  assert.equal(entered, 3, 'Stop releases the reservation when startup settles')
  release()
  assert.match((await afterStop)!.message, /Synthetic VM readiness failure/)
  assert.equal(aborters.size, 0)
  assert.equal(store.threads[0].timeline.length, 0, 'refused startup adds no fabricated conversation turn')
  console.log('Actual bot startup checks passed: one VM owner across parallel chats, failure and Stop release reservations, no host execution context.')
} finally {
  vm.requireReady = originalReady
  store.settings = saved.settings
  store.threads = saved.threads
}
