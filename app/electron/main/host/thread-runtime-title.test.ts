import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { Thread } from '../../agent/types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-thread-title-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
const { defaultSettings, saveThreads } = await import('../../store')
const { enqueueJob } = await import('../../agent/jobs')
const { builtinEngineProvider, setEngineServeHook } = await import('../../agent/engine/provider')
const { store } = await import('../state')
const { createThreadRuntime } = await import('./thread-runtime')
const { WorkspaceMutationGuard } = await import('./workspace-mutation-guard')

const baseUrl = 'http://127.0.0.1:9897/v1'
const model = 'Org/Synthetic'
const provider = { ...builtinEngineProvider(), models: [model] }
const originalFetch = globalThis.fetch

async function waitFor(predicate: () => boolean, label: string) {
  for (let attempt = 0; attempt < 400; attempt++) {
    if (predicate()) return
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error(`Timed out: ${label}`)
}

async function scenario(action: 'finish' | 'cancel' | 'rename' | 'external') {
  const events: string[] = []
  const aborters = new Map<string, AbortController>()
  let heldStream: ReadableStreamDefaultController<string> | undefined
  let settled = false
  const route = action === 'external'
    ? { ...provider, id: 'synthetic-custom-api', source: undefined, baseUrl }
    : provider
  const thread: Thread = {
    id: `synthetic-title-${action}`, title: 'New chat', kind: 'chat', cwd: home,
    providerId: route.id, model, status: 'idle', createdAt: 1, updatedAt: 1,
    timeline: [], apiMessages: [],
  }
  store.settings = { ...defaultSettings(), providers: [route], activeProviderId: route.id, activeModel: model, autoCompact: false, skillSelfImprovement: false }
  store.threads = [thread]
  store.projects = []
  await saveThreads([thread])
  setEngineServeHook(async (requested, options) => {
    assert.equal(requested, model)
    events.push(options.start ? 'turn-lease' : 'title-lease')
    return { model, baseUrl, apiKey: '', release: () => { events.push(options.start ? 'turn-release' : 'title-release') } }
  })
  globalThis.fetch = (async (input, init) => {
    const url = String(input)
    if (url === baseUrl && !init?.body) return new Response(null)
    if (url.endsWith('/models')) return Response.json({ data: [{ id: model }] })
    assert.equal(url, `${baseUrl}/chat/completions`)
    const request = JSON.parse(String(init?.body))
    if (!request.stream) {
      events.push('title-request')
      return Response.json({ choices: [{ message: { content: 'Synthetic Conversation' } }] })
    }
    events.push('turn-request')
    const body = new ReadableStream<string>({ start: (controller) => { heldStream = controller } })
    return new Response(body.pipeThrough(new TextEncoderStream()), { headers: { 'content-type': 'text/event-stream' } })
  }) as typeof fetch
  const runtime = createThreadRuntime({
    aborters, workspaceMutationGuard: new WorkspaceMutationGuard(), pendingApprovalMeta: new Map(),
    jobLeaseOwner: 'synthetic', threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} }, emitToRenderer() {}, persistThreads() {},
    fullToolContext: () => ({ cwd: home }), approvalResolver: () => async () => 'allow', isHighRiskTool: () => false,
    claimJob: (ref) => ({ ...enqueueJob([], ref).job, status: 'running' }),
    patchJob: (job, patch) => job ? { ...job, ...patch } : null,
    replaceJobResult: ({ job }) => job!, heartbeatJob: (job) => job,
    settleThreadRunJob() { settled = true }, renderProjectMemoryBucket: () => '', effectiveContextLimit: () => 32_768,
  })

  try {
    await runtime.startThreadRun({ threadId: thread.id, userText: 'Write one synthetic sentence.' })
    await waitFor(() => heldStream !== undefined, 'foreground stream')
    if (action === 'external') {
      assert.ok(events.includes('title-request'), 'a custom provider still starts background titling immediately')
    } else {
      assert.deepEqual(events, ['turn-lease', 'turn-request'], 'background title must not start while the first turn streams')
    }
    if (action === 'cancel') {
      runtime.cancelThreadRun(thread.id)
    } else {
      if (action === 'rename') thread.title = 'My Chosen Name'
      heldStream!.enqueue('data: {"choices":[{"delta":{"content":"Synthetic answer."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    }
    heldStream!.close()
    await waitFor(() => settled && !aborters.size, 'foreground settlement')
    if (action === 'finish') {
      await waitFor(() => events.includes('title-release'), 'background title settlement')
      assert.deepEqual(events, ['turn-lease', 'turn-request', 'turn-release', 'title-lease', 'title-request', 'title-release'])
      assert.equal(thread.title, 'Synthetic Conversation')
    } else if (action === 'external') {
      await waitFor(() => thread.title === 'Synthetic Conversation', 'custom provider title')
      assert.equal(events.filter((event) => event === 'title-request').length, 1)
    } else {
      assert.deepEqual(events, ['turn-lease', 'turn-request', 'turn-release'])
      assert.equal(thread.title, action === 'rename' ? 'My Chosen Name' : 'New chat')
    }
  } finally {
    if (aborters.has(thread.id)) {
      runtime.cancelThreadRun(thread.id)
      heldStream?.close()
      await waitFor(() => !aborters.size, 'cancelled foreground cleanup')
    }
  }
}

try {
  await scenario('finish')
  await scenario('cancel')
  await scenario('rename')
  await scenario('external')
  console.log('Built-in title follows the foreground lease, preserves cancellation and rename, and leaves custom providers unchanged')
} finally {
  globalThis.fetch = originalFetch
  setEngineServeHook(null)
  await rm(home, { recursive: true, force: true, maxRetries: 3 })
}
