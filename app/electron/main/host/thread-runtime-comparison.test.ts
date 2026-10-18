import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

const home = await mkdtemp(path.join(os.tmpdir(), 'tf-compare-run-admission-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const { defaultSettings } = await import('../../store')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')
const { builtinEngineProvider } = await import('../../agent/engine/provider')
const { EngineService } = await import('../../agent/engine/service')
const { makeHarness, seedModel, waitFor } = await import('../../agent/engine/test-fixtures/harness')
const h = await makeHarness({ FAKE_FIRST_TOKEN_MS: '100' })
const local = { ...builtinEngineProvider(), models: ['Fixture/A'] }
const cloud = { id: 'synthetic-cloud', name: 'Synthetic cloud', kind: 'cloud' as const, transport: 'chat-completions' as const, baseUrl: 'https://example.com/v1', apiKey: '', models: ['synthetic'] }
const saved = { settings: store.settings, threads: store.threads }
store.settings = { ...defaultSettings(), providers: [local, cloud], activeProviderId: local.id, activeModel: 'Fixture/A' }
const makeThread = (id: string, providerId: string | null) => ({ id, title: 'Synthetic comparison admission', kind: 'chat' as const, cwd: '', workspace: null, providerId, model: providerId === cloud.id ? 'synthetic' : 'Fixture/A', createdAt: 1, updatedAt: 1, status: 'idle' as const, apiMessages: [], timeline: [] })
store.threads = [makeThread('bound-local', local.id), makeThread('unbound-local', null), makeThread('cloud-chat', cloud.id)]
const aborters = new Map<string, AbortController>()
const guard = new WorkspaceMutationGuard()
let hostChecks = 0, claims = 0, contextReads = 0
const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, childEnv: h.env, appHome: h.root, emit: () => {}, pollMs: 10, totalMemoryBytes: 32 * 1024 ** 3 })
const runtime = createThreadRuntime({ aborters, workspaceMutationGuard: guard, assertLocalEngineAdmission: () => { hostChecks++; service.assertLocalWorkAdmission() }, pendingApprovalMeta: new Map(), jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000, sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {}, fullToolContext: () => { contextReads++; throw new Error('Synthetic stop before context'); }, approvalResolver: () => async () => 'deny', isHighRiskTool: () => false, claimJob: () => { claims++; throw new Error('Synthetic stop before job or transport'); }, patchJob: (job) => job, replaceJobResult: (result) => result.job!, heartbeatJob: (job) => job, settleThreadRunJob() {}, renderProjectMemoryBucket: () => { throw new Error('No project memory reads'); }, effectiveContextLimit: () => 4096 })
try {
  await Promise.all([seedModel(h.hub, 'Fixture/A'), seedModel(h.hub, 'Fixture/B')]); service.install()
  const input = { id: 'comparison_run_admission_1', models: ['Fixture/A', 'Fixture/B'] as [string, string], prompt: 'Synthetic comparison' }
  const pending = service.compareModels(input)
  await waitFor(() => service.comparisonStatus(input.id)?.rows[0].state === 'running')
  for (const id of ['bound-local', 'unbound-local']) {
    const before = structuredClone(store.threads.find((thread) => thread.id === id))
    await assert.rejects(runtime.startThreadRun({ threadId: id, userText: 'Must remain unsent.' }), /model comparison/)
    assert.deepEqual(store.threads.find((thread) => thread.id === id), before, 'denied local start changes neither history nor model selection')
    assert.equal(aborters.has(id), false); guard.assertThreadStartAllowed(id)
  }
  assert.equal(hostChecks, 2); assert.equal(claims, 0); assert.equal(contextReads, 0)
  // Cloud work reaches the unrelated intentional stop seam rather than comparison admission.
  await assert.rejects(runtime.startThreadRun({ threadId: 'cloud-chat', userText: 'Synthetic cloud task' }), /Synthetic stop before (job or transport|context)/)
  assert.equal(hostChecks, 2, 'comparison admission does not block unrelated cloud transport')
  service.cancelComparison(input.id); await pending
  assert.doesNotThrow(() => service.assertLocalWorkAdmission(), 'finished comparison releases local startup admission')
  assert.equal(aborters.size, 0)
} finally { await service.supervisor.stop(); service.shutdown(); store.settings = saved.settings; store.threads = saved.threads; await h.cleanup(); await rm(home, { recursive: true, force: true }) }
console.log('Actual thread-runtime comparison seam: bound/unbound local starts rejected before history/job/context/aborter mutation; cloud startup unaffected and cancellation releases admission.')
