import assert from 'node:assert/strict'
import { defaultSettings } from '../../store'
import { store } from '../state'
import { createJobControlRuntime } from './job-control-runtime'
import { sshWorkspaceKey, WorkspaceMutationGuard } from './workspace-mutation-guard'

const saved = {
  projects: store.projects,
  threads: store.threads,
  jobs: store.jobs,
  settings: store.settings,
}

function approval(threadId: string) {
  return {
    callId: 'call-1', threadId, itemId: 'tool-1', tool: 'write_file', resolvedAt: null,
    policy: { key: 'write:file', label: 'Write file', explanation: 'Writes one reviewed file.' },
  } as any
}

function job(threadId: string, pending = approval(threadId)) {
  return {
    id: 'job-1', kind: 'thread', owner: threadId, threadId, currentThreadId: threadId,
    pendingApprovals: [pending],
  } as any
}

try {
  store.settings = defaultSettings()

  // Archived ownership rejects before saving/resolving/running a stale approval.
  {
    const thread = {
      id: 'archived-thread', projectId: 'project', cwd: '/repo', archivedAt: Date.now(),
      timeline: [{ id: 'tool-1', kind: 'tool', tool: 'write_file', args: { path: 'x', content: 'x' } }], apiMessages: [],
    } as any
    store.projects = [{ id: 'project' } as any]
    store.threads = [thread]
    store.jobs = [job(thread.id)]
    let executions = 0
    const runtime = createJobControlRuntime({
      handle: () => {}, aborters: new Map(), workspaceMutationGuard: new WorkspaceMutationGuard(),
      executeTool: async () => { executions++; return { ok: true, output: 'unexpected' } },
      resolvePersistedJobApproval: () => null,
    } as any)
    await assert.rejects(() => runtime.resolveDurableApprovalAfterRestart('call-1', 'allow'), /Restore the archived thread/)
    assert.equal(executions, 0)
    assert.equal(store.jobs[0].pendingApprovals[0].resolvedAt, null, 'stale approval remains pending so the user can restore or deny it')
  }

  // SSH restore path connects first, obtains a real remote backend, and holds a shared run lease
  // across execution. No local fallback can receive the remote cwd.
  {
    const workspace = { kind: 'ssh' as const, hostId: 'pgx', path: '/srv/repo' }
    const thread = {
      id: 'ssh-thread', projectId: 'project', cwd: workspace.path, workspace,
      timeline: [{ id: 'tool-1', kind: 'tool', tool: 'write_file', args: { path: 'x', content: 'x' } }], apiMessages: [],
    } as any
    store.projects = [{ id: 'project', workspace } as any]
    store.threads = [thread]
    store.jobs = [job(thread.id)]
    const guard = new WorkspaceMutationGuard()
    let connected = false
    let executions = 0
    const runtime = createJobControlRuntime({
      handle: () => {},
      aborters: new Map(),
      workspaceMutationGuard: guard,
      sshManager: { ensureConnected: async (hostId: string) => { assert.equal(hostId, 'pgx'); connected = true } },
      fullToolContext: () => {
        assert.equal(connected, true, 'SSH is connected before tool context captures its backend')
        return { backend: { kind: 'ssh' } }
      },
      mcpToolContext: () => ({}),
      executeTool: async () => {
        executions++
        assert.throws(() => guard.acquire({ threadId: thread.id }), /already in progress/, 'thread restore is excluded during approval execution')
        assert.throws(() => guard.acquire({ workspaceKey: sshWorkspaceKey('pgx', '/srv/repo') }), /already in progress/, 'workspace restore is excluded during approval execution')
        return { ok: true, output: 'remote write complete' }
      },
      isHighRiskTool: () => false,
      emitToRenderer: () => {},
      persistThreads: () => {},
      resolvePersistedJobApproval: () => null,
    } as any)

    assert.equal(await runtime.resolveDurableApprovalAfterRestart('call-1', 'allow'), true)
    assert.equal(executions, 1)
    assert.equal(thread.apiMessages.at(-1)?.tool_call_id, 'call-1', 'legacy approvals without a separate raw timeline ID retain their model response ID')
    assert.doesNotThrow(() => guard.acquire({ threadId: thread.id, workspaceKey: sshWorkspaceKey('pgx', '/srv/repo') }).release(), 'lease releases after tool settlement')
  }

  // Even after a successful SSH connection, absence of a remote backend fails closed.
  {
    const workspace = { kind: 'ssh' as const, hostId: 'pgx', path: '/srv/repo' }
    const thread = {
      id: 'missing-backend', projectId: 'project', cwd: workspace.path, workspace,
      timeline: [{ id: 'tool-1', kind: 'tool', tool: 'write_file', args: { path: 'x', content: 'x' } }], apiMessages: [],
    } as any
    store.projects = [{ id: 'project', workspace } as any]
    store.threads = [thread]
    store.jobs = [job(thread.id)]
    let executions = 0
    const guard = new WorkspaceMutationGuard()
    const runtime = createJobControlRuntime({
      handle: () => {}, aborters: new Map(), workspaceMutationGuard: guard,
      sshManager: { ensureConnected: async () => {} },
      fullToolContext: () => ({}),
      executeTool: async () => { executions++; return { ok: true, output: 'unexpected' } },
      isHighRiskTool: () => false, emitToRenderer: () => {}, persistThreads: () => {},
      resolvePersistedJobApproval: () => null,
    } as any)
    assert.equal(await runtime.resolveDurableApprovalAfterRestart('call-1', 'allow'), true)
    assert.equal(executions, 0, 'missing SSH backend never falls through to LocalBackend')
    assert.match(thread.timeline[0].output, /could not obtain an SSH workspace backend/)
    assert.doesNotThrow(() => guard.acquire({ threadId: thread.id }).release(), 'failure releases its reservation')
  }

  // App approval identities remain distinct from the raw model protocol ID on recovery.
  {
    store.settings = { ...defaultSettings(), providers: [], activeProviderId: 'none', activeModel: 'none' }
    const makePending = (threadId: string, itemId: string) => ({ ...approval(threadId), callId: itemId, itemId })
    const makeThread = (id: string, itemId: string) => ({
      id, cwd: '/tmp', timeline: [{ id: itemId, approvalId: itemId, callId: 'raw-shared-call', kind: 'tool', tool: 'write_file', args: { path: 'x', content: 'x' } }],
      apiMessages: [{ role: 'assistant', content: null, tool_calls: [{ id: 'raw-shared-call', type: 'function', function: { name: 'write_file', arguments: '{}' } }] }],
    } as any)
    const first = makeThread('recovered-first', 'opaque-tool-first')
    const second = makeThread('recovered-second', 'opaque-tool-second')
    second.apiMessages.unshift({ role: 'assistant', content: null, tool_calls: [{ id: 'raw-shared-call', type: 'function', function: { name: 'write_file', arguments: '{}' } }] }, { role: 'tool', tool_call_id: 'raw-shared-call', content: 'an earlier turn response' })
    store.projects = []
    store.threads = [first, second]
    store.jobs = [
      { ...job(first.id, makePending(first.id, 'opaque-tool-first')), id: 'recovered-job-first' },
      { ...job(second.id, makePending(second.id, 'opaque-tool-second')), id: 'recovered-job-second' },
    ]
    let executions = 0
    const resolved: string[] = []
    const runtime = createJobControlRuntime({
      handle: () => {}, aborters: new Map(), workspaceMutationGuard: new WorkspaceMutationGuard(),
      fullToolContext: () => ({}), mcpToolContext: () => ({}),
      executeTool: async () => { executions++; return { ok: true, output: 'synthetic recovered output' } },
      isHighRiskTool: () => false, emitToRenderer: () => {}, persistThreads: () => {},
      resolvePersistedJobApproval: (id: string) => { resolved.push(id); return null },
    } as any)
    await Promise.all([
      runtime.resolveDurableApprovalAfterRestart('opaque-tool-first', 'allow'),
      runtime.resolveDurableApprovalAfterRestart('opaque-tool-second', 'deny'),
    ])
    assert.equal(executions, 1, 'independent opaque approvals allow/deny their own restored tool')
    assert.deepEqual(resolved.sort(), ['opaque-tool-first', 'opaque-tool-second'])
    assert.equal(first.apiMessages.at(-1).tool_call_id, 'raw-shared-call', 'allow response uses the raw model ID')
    assert.equal(second.apiMessages.at(-1).tool_call_id, 'raw-shared-call', 'deny response uses the raw model ID')
    assert.equal(second.apiMessages.filter((entry: any) => entry.role === 'tool').length, 2, 'an earlier response with a reused raw ID cannot suppress the current tool-call group')
    assert.doesNotMatch(first.apiMessages.at(-1).tool_call_id, /opaque/, 'app approval identity never enters protocol history')
  }

  // Two UI/mission resolution paths cannot race an allow against a deny and execute twice.
  {
    const thread = {
      id: 'resolution-race', projectId: 'project', cwd: '/repo',
      timeline: [{ id: 'tool-1', kind: 'tool', tool: 'write_file', args: { path: 'x', content: 'x' } }], apiMessages: [],
    } as any
    store.projects = [{ id: 'project' } as any]
    store.threads = [thread]
    store.jobs = [job(thread.id)]
    let release!: () => void
    const blocked = new Promise<void>((resolve) => { release = resolve })
    let entered!: () => void
    const didEnter = new Promise<void>((resolve) => { entered = resolve })
    const runtime = createJobControlRuntime({
      handle: () => {}, aborters: new Map(), workspaceMutationGuard: new WorkspaceMutationGuard(),
      fullToolContext: () => ({}),
      executeTool: async () => { entered(); await blocked; return { ok: true, output: 'done' } },
      isHighRiskTool: () => false, emitToRenderer: () => {}, persistThreads: () => {},
      resolvePersistedJobApproval: () => null,
    } as any)
    const first = runtime.resolveDurableApprovalAfterRestart('call-1', 'allow')
    await didEnter
    await assert.rejects(() => runtime.resolveDurableApprovalAfterRestart('call-1', 'deny'), /already being resolved/)
    release()
    assert.equal(await first, true)
  }

  console.log('job-control-restored-approval.test: archived, SSH backend, and workspace lease boundaries passed')
} finally {
  store.projects = saved.projects
  store.threads = saved.threads
  store.jobs = saved.jobs
  store.settings = saved.settings
}
