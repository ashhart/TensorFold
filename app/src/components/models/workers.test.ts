import assert from 'node:assert/strict'
import type { EngineSnapshot, Project, Settings, Thread } from '../../lib/types'
import { canUseWorkerProject, isLocalWorker, launchLocalWorker, makeWorkerDraft, WorkerLaunchError, workerState, type WorkerLaunchDependencies } from './workers'

const project: Project = { id: 'project', name: 'Example project', cwd: '/projects/example', memory: '', team: '', createdAt: 1, updatedAt: 1 }
const model = 'example/local-coder'
const draft = () => makeWorkerDraft(project, 'Improve search', 'Fix the search and check its behavior.', model, '01234567-89ab-cdef-0123-456789abcdef')
const baseThread = (): Thread => ({
  id: 'worker', title: 'New task', kind: 'task', projectId: project.id, cwd: '/worktrees/example/codex-worker-improve-search-0123456789abcdef',
  providerId: 'cloud', model: 'cloud-model', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [],
})

function harness() {
  const calls: string[] = []
  const thread = baseThread()
  const state = {
    projects: [project], threads: [] as Thread[], status: {} as Record<string, 'thinking' | 'error'>,
    settings: { providers: [{ id: 'tensorfold', name: 'TensorFold', baseUrl: '', apiKey: '', models: [model] }], approvalMode: 'suggest' } as Settings,
  }
  const snapshot = { status: { available: true, state: 'ready', model } } as EngineSnapshot
  let worktreeCreated = false
  const deps: WorkerLaunchDependencies = {
    getState: () => state,
    onThread: (next) => { state.threads = [next, ...state.threads.filter((t) => t.id !== next.id)]; calls.push('mirror') },
    onStarting: (id) => { state.status[id] = 'thinking'; calls.push('starting') },
    onStartFailed: (id) => { state.status[id] = 'error'; calls.push('failed') },
    api: {
      engineSnapshot: async () => { calls.push('engine'); return snapshot },
      createWorktree: async (input) => {
        calls.push('worktree')
        assert.equal(input.mode, 'worktree')
        assert.equal(input.cwd, project.cwd)
        assert.deepEqual(input.workspace, { kind: 'local', path: project.cwd })
        worktreeCreated = true
        return { ok: true, path: thread.cwd, branch: input.name, mode: 'worktree' }
      },
      getWorktrees: async () => { calls.push('worktrees'); return { branches: [draft().branch], current: 'main', worktrees: worktreeCreated ? [{ path: thread.cwd, branch: draft().branch, head: 'abc', detached: false, bare: false }] : [] } },
      getThreads: async () => { calls.push('threads'); return state.threads },
      createThread: async (cwd, kind, projectId, commandId) => {
        calls.push('task')
        assert.equal(cwd, thread.cwd)
        assert.equal(kind, 'task')
        assert.equal(projectId, project.id)
        assert.equal(commandId, draft().commandId)
        state.threads = [thread]
        return thread
      },
      setThreadWorkspace: async (id, workspace, cwd) => {
        calls.push('workspace')
        assert.equal(id, thread.id)
        assert.equal(workspace?.kind, 'local')
        assert.equal(workspace?.path, cwd)
        thread.workspace = workspace; thread.cwd = cwd!
        return thread
      },
      configureThread: async (id, providerId, chosenModel, accountId) => {
        calls.push('configure')
        assert.equal(id, thread.id)
        assert.equal(accountId, null)
        thread.providerId = providerId; thread.model = chosenModel
      },
      renameThread: async (_id, title) => { calls.push('rename'); thread.title = title },
      patchThread: async (_id, patch) => { calls.push('goal'); Object.assign(thread, patch); return thread },
      startRun: async (input) => {
        calls.push('start')
        assert.equal(thread.providerId, 'tensorfold')
        assert.equal(thread.model, model)
        assert.equal(thread.cwd, thread.workspace?.path)
        assert.equal(input.threadId, thread.id)
        assert.equal(input.userText, draft().objective)
        assert.equal(input.approvalMode, 'suggest')
        return { runId: 'run' }
      },
    },
  }
  return { deps, calls, state, thread, snapshot }
}

async function expectFailure(run: () => Promise<unknown>, pattern: RegExp): Promise<WorkerLaunchError> {
  try { await run() } catch (error) {
    assert.ok(error instanceof WorkerLaunchError)
    assert.match(error.message, pattern)
    return error
  }
  throw new Error('Expected worker launch to fail')
}

{
  const h = harness()
  const result = await launchLocalWorker(draft(), h.deps)
  assert.equal(result.runId, 'run')
  assert.equal(result.draft.path, h.thread.cwd)
  assert.equal(result.threadId, h.thread.id)
  assert.ok(h.calls.indexOf('configure') < h.calls.indexOf('start'))
  assert.ok(h.calls.indexOf('workspace') < h.calls.indexOf('start'))
  assert.equal(h.thread.goal, draft().objective)
  assert.equal(h.thread.goalStatus, undefined, 'Setup must not invent objective completion or enable a separate goal loop')
}

{
  for (const state of ['stopped', 'starting', 'failed'] as const) {
    const h = harness()
    h.snapshot.status.state = state
    const error = await expectFailure(() => launchLocalWorker(draft(), h.deps), /Load .* before starting/)
    assert.equal(error.draft.path, null)
    assert.deepEqual(h.calls, ['engine'], 'Readiness must be checked before any worktree or thread mutation')
  }
}

{
  const h = harness()
  h.deps.api.createWorktree = async () => ({ ok: false, branch: draft().branch, mode: 'worktree', path: '', error: 'Make an initial commit first.' })
  await expectFailure(() => launchLocalWorker(draft(), h.deps), /initial commit/)
  assert.ok(!h.calls.includes('task'))
  assert.ok(!h.calls.includes('start'))
}

{
  const h = harness()
  h.deps.api.createWorktree = async () => ({ ok: true, branch: draft().branch, mode: 'worktree', path: project.cwd })
  await expectFailure(() => launchLocalWorker(draft(), h.deps), /isolated worktree/)
  assert.ok(!h.calls.includes('task'), 'A shared checkout must never be used as the fallback')
}

{
  const h = harness()
  h.deps.api.configureThread = async () => { h.calls.push('configure-failed'); throw new Error('Provider temporarily unavailable') }
  const failure = await expectFailure(() => launchLocalWorker(draft(), h.deps), /Provider temporarily unavailable/)
  assert.equal(failure.draft.path, h.thread.cwd)
  assert.equal(failure.draft.threadId, h.thread.id)
  h.deps.api.configureThread = async (_id, providerId, model) => { h.thread.providerId = providerId; h.thread.model = model }
  await launchLocalWorker(failure.draft, h.deps)
  assert.equal(h.calls.filter((call) => call === 'worktree').length, 1, 'Retry reuses the preserved worktree')
  assert.equal(h.calls.filter((call) => call === 'task').length, 1, 'Retry reuses the existing thread')
  assert.equal(h.calls.filter((call) => call === 'start').length, 1)
}

{
  const h = harness()
  h.deps.api.startRun = async () => { throw new Error('The local server disconnected') }
  const failure = await expectFailure(() => launchLocalWorker(draft(), h.deps), /server disconnected/)
  assert.equal(failure.draft.threadId, h.thread.id)
  assert.equal(h.state.status[h.thread.id], 'error')
}

{
  const h = harness()
  h.deps.api.startRun = async () => { h.thread.status = 'running'; throw new Error('IPC response lost') }
  const failure = await expectFailure(() => launchLocalWorker(draft(), h.deps), /this worker is running/)
  assert.equal(h.state.status[h.thread.id], 'thinking', 'An accepted run must not be changed to a failed state because its response was lost')
  await expectFailure(() => launchLocalWorker(failure.draft, h.deps), /already running/)
}

{
  const h = harness()
  const create = h.deps.api.createThread
  let saved: Thread | undefined
  h.deps.api.createThread = async (...args) => {
    saved = await create(...args)
    throw new Error('Create response lost')
  }
  const failure = await expectFailure(() => launchLocalWorker(draft(), h.deps), /Create response lost/)
  assert.equal(failure.draft.threadId, null)
  h.deps.api.createThread = async (_cwd, _kind, _projectId, commandId) => {
    assert.equal(commandId, draft().commandId, 'Retry retains the original create command identity')
    return saved!
  }
  await launchLocalWorker(failure.draft, h.deps)
  assert.equal(h.state.threads.length, 1)
  assert.equal(h.calls.filter((call) => call === 'worktree').length, 1)
}

{
  const h = harness()
  const busy = { ...baseThread(), id: 'other-task', status: 'running' as const }
  h.state.threads = [busy]
  await expectFailure(() => launchLocalWorker(draft(), h.deps), /Another task is using this worktree/)
  assert.ok(!h.calls.includes('task'))
  assert.ok(!h.calls.includes('configure'))
}

{
  const h = harness()
  h.deps.api.patchThread = async (_id, patch) => { Object.assign(h.thread, patch); h.snapshot.status.state = 'stopped'; return h.thread }
  await expectFailure(() => launchLocalWorker(draft(), h.deps), /Load .* before starting/)
  assert.ok(!h.calls.includes('start'), 'A model that stops during setup must not be launched anyway')
}

{
  const h = harness()
  let unblock: (() => void) | undefined
  const gate = new Promise<void>((resolve) => { unblock = resolve })
  h.deps.api.engineSnapshot = async () => { await gate; return h.snapshot }
  const first = launchLocalWorker(draft(), h.deps)
  await expectFailure(() => launchLocalWorker(draft(), h.deps), /Another worker is being set up/)
  unblock!()
  await first
}

{
  assert.equal(canUseWorkerProject(project), true)
  assert.equal(canUseWorkerProject({ ...project, workspace: { kind: 'ssh', path: '/remote' } }), false)
  assert.equal(canUseWorkerProject({ ...project, executionTarget: 'orchestrator' }), false)
  assert.equal(canUseWorkerProject({ ...project, archivedAt: 2 }), false)
  assert.equal(canUseWorkerProject({ ...project, cwd: '' }), false)
  assert.equal(isLocalWorker(baseThread()), true)
  assert.equal(isLocalWorker({ ...baseThread(), cwd: '/worktrees/project/codex/worker-example-01234567' }), true, 'Browser demo worktree path also identifies the worker')
  assert.equal(isLocalWorker({ ...baseThread(), cwd: project.cwd }), false)
  assert.equal(isLocalWorker({ ...baseThread(), orchestratorHostId: 'remote' }), false)
  assert.equal(workerState(baseThread(), 'thinking'), 'running')
  assert.equal(workerState(baseThread(), 'awaiting'), 'waiting')
  assert.equal(workerState(baseThread(), 'thinking', 1), 'waiting')
  assert.equal(workerState(baseThread(), 'error'), 'failed')
  assert.equal(workerState({ ...baseThread(), status: 'running' }), 'ready', 'Persisted running is not live activity after a restart')
  assert.equal(workerState({ ...baseThread(), timeline: [{ kind: 'assistant', id: 'reply', text: 'Done', createdAt: 1 }] }), 'review', 'An assistant reply alone is not proof of completion')
  assert.equal(workerState({ ...baseThread(), goalStatus: 'complete' }), 'completed')
  assert.match(makeWorkerDraft(project, '', 'Fix UTF-8 paths', model, 'abcd1234').branch, /^codex\/worker-fix-utf-8-paths-abcd1234$/)
}

console.log('Worker setup, recovery, isolation, readiness and lifecycle tests passed.')
