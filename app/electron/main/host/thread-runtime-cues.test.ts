import assert from 'node:assert/strict'
import { assertSshThreadTransportSupported, createThreadRuntime, resolveThreadRunCwd } from './thread-runtime'
import { store } from '../state'
import type { AgentEvent, Thread } from '../../agent/types'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

// Composer-controls #35: pressing Enter while a run is in flight queues a live
// "Guide" cue instead of starting a new run. These tests pin the runtime cue
// mechanics that back that behavior — enqueue while active, drain once, and
// mark undrained cues when the run ends/cancels.

function makeThread(id: string): Thread {
  return {
    id,
    title: 'Test',
    kind: 'task',
    cwd: '/tmp',
    providerId: null,
    model: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'running',
    apiMessages: [],
    timeline: [],
  }
}

function makeRuntime(overrides: Record<string, unknown> = {}) {
  const aborters = new Map<string, AbortController>()
  const emitted: AgentEvent[] = []
  const noop = () => {}
  const deps: any = {
    aborters,
    workspaceMutationGuard: new WorkspaceMutationGuard(),
    pendingApprovalMeta: new Map(),
    jobLeaseOwner: 'test',
    threadJobHeartbeatMs: 5000,
    sshManager: { ensureConnected: async () => {} },
    emitToRenderer: (e: AgentEvent) => { emitted.push(e) },
    persistThreads: noop,
    fullToolContext: () => ({}),
    approvalResolver: () => async () => 'allow' as const,
    isHighRiskTool: () => false,
    claimJob: () => ({} as any),
    patchJob: (j: any) => j,
    replaceJobResult: (r: any) => r.job,
    heartbeatJob: (j: any) => j,
    settleThreadRunJob: noop,
    renderProjectMemoryBucket: () => '',
    effectiveContextLimit: () => 128_000,
    ...overrides,
  }
  return { runtime: createThreadRuntime(deps), aborters, emitted }
}

async function run() {
  // Isolate the shared main-process store for this test.
  const savedThreads = store.threads
  const savedSettings = store.settings
  store.settings = {
    ...(store.settings ?? {}),
    providers: [{ id: 'test-provider', name: 'Test API', kind: 'cloud', transport: 'chat-completions', baseUrl: 'https://example.test/v1', apiKey: '', models: ['test-model'] }],
    activeProviderId: 'test-provider',
    activeModel: 'test-model',
  } as any

  // Test 1: a cue is queued (and added to the timeline) when the thread has an
  // active aborter — i.e. a run is in flight.
  {
    const { runtime, aborters, emitted } = makeRuntime()
    const thread = makeThread('cue-active')
    store.threads = [thread]
    aborters.set(thread.id, new AbortController())

    const result = await runtime.guideThreadRun({ threadId: thread.id, userText: '  focus on errors  ' })
    assert.equal(result.queued, true, 'cue should queue while a run is active')
    assert.equal(result.queueDepth, 1, 'queue depth should reflect the queued cue')
    const cueItem = thread.timeline.find((i) => i.kind === 'cue')
    assert.ok(cueItem, 'a cue timeline item should be added')
    assert.equal(cueItem!.text, 'focus on errors', 'cue text should be trimmed')
    assert.equal(cueItem!.title, 'Guide', 'cue item is labeled Guide')
    assert.equal(cueItem!.state, 'pending', 'cue starts pending')
    assert.ok(emitted.some((e) => e.type === 'item_add'), 'an item_add event should be emitted')
  }

  // Test 2: draining returns the queued cue, and a second drain is empty.
  {
    const { runtime, aborters } = makeRuntime()
    const thread = makeThread('cue-drain')
    store.threads = [thread]
    aborters.set(thread.id, new AbortController())

    await runtime.guideThreadRun({ threadId: thread.id, userText: 'verify input validation' })
    const drained = runtime.drainThreadCues(thread.id)
    assert.equal(drained.length, 1, 'drain returns the queued cue')
    assert.equal(drained[0].text, 'verify input validation', 'drained cue carries its text')
    const second = runtime.drainThreadCues(thread.id)
    assert.equal(second.length, 0, 'second drain returns empty')
  }

  // Test 3: no active run => no queue (the renderer should fall through to a new run).
  {
    const { runtime } = makeRuntime()
    const thread = makeThread('cue-idle')
    store.threads = [thread]

    const result = await runtime.guideThreadRun({ threadId: thread.id, userText: 'will not queue' })
    assert.equal(result.queued, false, 'no cue should queue when there is no active run')
    assert.equal(thread.timeline.length, 0, 'no cue item is added when not queued')
  }

  // Test 4: empty/whitespace cues are ignored even with an active run.
  {
    const { runtime, aborters } = makeRuntime()
    const thread = makeThread('cue-empty')
    store.threads = [thread]
    aborters.set(thread.id, new AbortController())

    const result = await runtime.guideThreadRun({ threadId: thread.id, userText: '   ' })
    assert.equal(result.queued, false, 'whitespace-only cue is not queued')
    assert.equal(thread.timeline.length, 0, 'no timeline item for empty cue')
  }

  // Test 5: undrained cues are marked denied (e.g. on cancel) so the UI does not
  // leave them stuck in "pending".
  {
    const { runtime, aborters, emitted } = makeRuntime()
    const thread = makeThread('cue-undrained')
    store.threads = [thread]
    aborters.set(thread.id, new AbortController())

    await runtime.guideThreadRun({ threadId: thread.id, userText: 'late guide' })
    runtime.markUndrainedCues(thread, 'canceled')
    const cueItem = thread.timeline.find((i) => i.kind === 'cue')!
    assert.equal(cueItem.state, 'denied', 'undrained cue is marked denied')
    assert.equal(cueItem.meta, 'canceled', 'undrained cue carries the cancel reason')
    assert.ok(
      emitted.some((e) => e.type === 'item_update' && (e as any).patch?.state === 'denied'),
      'an item_update should announce the denied state',
    )
    // Already drained by markUndrainedCues — a subsequent drain is empty.
    assert.equal(runtime.drainThreadCues(thread.id).length, 0, 'markUndrainedCues drains the queue')
  }

  // Host-side attachment validation is authoritative for both local IPC and the remote-control
  // REST bridge. In particular, an arbitrary image URL must never reach a provider (SSRF) or be
  // persisted in a cue timeline just because a caller bypassed the renderer's file picker.
  {
    const { runtime, aborters } = makeRuntime()
    const thread = makeThread('cue-host-attachment-policy')
    store.threads = [thread]
    aborters.set(thread.id, new AbortController())

    await assert.rejects(
      () => runtime.guideThreadRun({
        threadId: thread.id,
        userText: 'inspect this',
        attachments: [{ kind: 'image', dataUrl: 'http://127.0.0.1:3000/private' } as any],
      }),
      /supported base64 data URL/i,
      'renderer-bypassing image URLs are rejected at the host boundary',
    )
    assert.equal(thread.timeline.length, 0, 'a rejected attachment does not create a cue timeline item')
  }

  {
    const { runtime, aborters } = makeRuntime()
    const thread = makeThread('start-host-attachment-policy')
    thread.status = 'idle'
    store.threads = [thread]

    await assert.rejects(
      () => runtime.startThreadRun({
        threadId: thread.id,
        userText: 'inspect this',
        attachments: [{ kind: 'image', dataUrl: 'file:///etc/passwd' } as any],
      }),
      /supported base64 data URL/i,
      'local start rejects renderer-bypassing URLs at the host boundary',
    )
    assert.equal(thread.timeline.length, 0, 'a rejected start is not persisted to the timeline')
    assert.equal(thread.apiMessages.length, 0, 'a rejected start is not forwarded to a model')
    assert.equal(aborters.has(thread.id), false, 'a rejected attachment does not reserve a run slot')
  }

  // Test 6: a second startThreadRun for a thread that already has an in-flight run is
  // rejected (rather than overwriting the shared aborters[threadId] entry, which would let
  // the first run's finally delete the live run's controller and corrupt apiMessages).
  {
    const { runtime, aborters } = makeRuntime()
    const thread = makeThread('concurrent-run')
    store.threads = [thread]
    const firstController = new AbortController()
    aborters.set(thread.id, firstController)
    const apiMessagesBefore = thread.apiMessages.length
    const timelineBefore = thread.timeline.length

    await assert.rejects(
      () => runtime.startThreadRun({ threadId: thread.id, userText: 'second run' }),
      /already running/i,
      'a second concurrent run for the same thread is rejected',
    )
    // The rejected start must not touch thread state or clobber the live controller.
    assert.equal(aborters.get(thread.id), firstController, 'the in-flight controller is preserved')
    assert.equal(thread.apiMessages.length, apiMessagesBefore, 'no apiMessages appended by the rejected start')
    assert.equal(thread.timeline.length, timelineBefore, 'no timeline items added by the rejected start')
  }

  // Test 7: the same guard must cover the startup await window. The first run reserves
  // aborters[threadId] before SSH/model setup awaits, so a second start cannot slip in
  // and append a second user message while the first is still preparing.
  {
    let rejectConnect!: (err: Error) => void
    const connect = new Promise((_resolve, reject) => { rejectConnect = reject })
    const { runtime, aborters } = makeRuntime({
      sshManager: { ensureConnected: async () => connect },
    })
    const thread = { ...makeThread('startup-race'), workspace: { kind: 'ssh', hostId: 'pgx', path: '/repo' } } as Thread
    store.threads = [thread]

    const first = runtime.startThreadRun({ threadId: thread.id, userText: 'first run' })
    await Promise.resolve()
    assert.equal(aborters.has(thread.id), true, 'startup reserves the thread before awaited setup')

    await assert.rejects(
      () => runtime.startThreadRun({ threadId: thread.id, userText: 'second run' }),
      /already running/i,
      'a second start during setup is rejected',
    )
    assert.equal(thread.apiMessages.length, 0, 'second start did not append an api message')
    assert.equal(thread.timeline.length, 0, 'second start did not append a timeline item')

    rejectConnect(new Error('offline'))
    await assert.rejects(() => first, /SSH connection failed: offline/)
    assert.equal(aborters.has(thread.id), false, 'failed startup releases the reservation')
  }

  // Test 8: SSH-bound project tasks use the remote workspace path as their run cwd
  // instead of validating the empty local cwd and surfacing "Choose folder".
  {
    const cwd = await resolveThreadRunCwd(
      { kind: 'task', cwd: '' },
      { kind: 'ssh', hostId: 'pgx', path: '/srv/project' },
    )
    assert.equal(cwd, '/srv/project', 'remote workspace path is the task cwd')
    await assert.rejects(
      () => resolveThreadRunCwd({ kind: 'task', cwd: '' }, { kind: 'ssh', hostId: 'pgx', path: '' }),
      /remote workspace folder/i,
      'empty SSH workspace paths produce a remote-specific folder error',
    )
  }

  {
    const ssh = { kind: 'ssh', hostId: 'pgx', path: '/srv/project' } as const
    for (const transport of ['codex-cli', 'claude-cli', 'opencode', 'kilo', 'pi-cli', 'cursor-cli', 'gemini-cli'] as const) {
      assert.throws(
        () => assertSshThreadTransportSupported(ssh, { name: transport, transport } as any),
        /runs on this Mac and cannot safely use an SSH workspace/,
        `${transport} fails closed before a local process can receive the remote cwd`,
      )
    }
    assert.doesNotThrow(() => assertSshThreadTransportSupported(ssh, { name: 'API', transport: 'chat-completions' } as any))
  }

  // Test 9: terminal tool updates get a quiescence receipt attached to the
  // timeline item. This is the event-spine contract the run receipt later exports.
  {
    const { runtime } = makeRuntime()
    const thread = makeThread('event-receipt')
    thread.timeline.push({ id: 'tool-1', kind: 'tool', tool: 'run_command', command: 'npm test', state: 'running', createdAt: 1 })

    runtime.applyEventToThread(thread, {
      runId: 'run-receipts',
      threadId: thread.id,
      type: 'item_update',
      id: 'tool-1',
      patch: { state: 'ok', output: 'passed' },
    } as AgentEvent)

    assert.equal(thread.timeline[0].eventReceipt?.kind, 'tool_result_saved')
    assert.equal(thread.timeline[0].eventReceipt?.runId, 'run-receipts')
    assert.deepEqual(thread.timeline[0].eventReceipt?.evidenceRefs, ['tool-1'])
  }

  store.threads = savedThreads
  store.settings = savedSettings
  console.log('thread-runtime-cues.test: guide-cue queue/drain/undrained + concurrent-run guard tests passed')
}

run().catch((err) => { console.error(err); process.exit(1) })
