import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJobControlRuntime } from './job-control-runtime'
import { store } from '../state'
import type { ApprovalPolicyScope } from '../../agent/approval-policy'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

const originalThreads = store.threads
const originalRole = process.env.VONTRA_SECURITY_ROLE
const originalHome = process.env.VONTRA_HOME
const testHome = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-approval-durability-'))
process.env.VONTRA_SECURITY_ROLE = 'operator'
process.env.VONTRA_HOME = testHome

const policy: ApprovalPolicyScope = {
  scope: 'tool',
  key: 'run:npm-install',
  label: 'Install package',
  explanation: 'This exact package install mutates dependencies.',
} as ApprovalPolicyScope

try {
  store.threads = [{ id: 'thread-1', projectId: 'project-1', cwd: '/repo' } as any]

  // Live allow_always: the parked tool resolver must not run while the bounded saved policy
  // is still being written. This is the actual action boundary, not merely a store-unit test.
  {
    const pendingApprovals = new Map<string, (decision: 'allow' | 'allow_always' | 'deny') => void>()
    const pendingApprovalMeta = new Map([['call-1', { threadId: 'thread-1', itemId: 'item-1', tool: 'run_command', policy }]])
    const save = deferred<any>()
    const decisions: string[] = []
    pendingApprovals.set('call-1', (decision) => { decisions.push(decision) })
    const runtime = createJobControlRuntime({
      handle: () => {},
      pendingApprovals,
      pendingApprovalMeta,
      rememberApprovalRule: () => save.promise,
      aborters: new Map(),
      resolvePersistedJobApproval: () => null,
      spine: () => null,
    } as any)

    let settled = false
    const resolution = runtime.resolveToolApproval('call-1', 'allow_always').then(() => { settled = true })
    await Promise.resolve()
    assert.deepEqual(decisions, [], 'tool execution remains parked before the saved policy is durable')
    assert.equal(settled, false)
    save.resolve({ id: 'rule-1' })
    await resolution
    assert.deepEqual(decisions, ['allow_always'])
  }

  // A save failure leaves the approval parked and rejects the IPC/runtime request. It must not
  // become a one-time allow under the guise of an Always decision.
  {
    const pendingApprovals = new Map<string, (decision: 'allow' | 'allow_always' | 'deny') => void>()
    const pendingApprovalMeta = new Map([['call-2', { threadId: 'thread-1', itemId: 'item-2', tool: 'run_command', policy }]])
    const decisions: string[] = []
    pendingApprovals.set('call-2', (decision) => { decisions.push(decision) })
    const runtime = createJobControlRuntime({
      handle: () => {},
      pendingApprovals,
      pendingApprovalMeta,
      rememberApprovalRule: async () => { throw new Error('disk full') },
      aborters: new Map(),
      resolvePersistedJobApproval: () => null,
      spine: () => null,
    } as any)

    await assert.rejects(() => runtime.resolveToolApproval('call-2', 'allow_always'), /disk full/)
    assert.deepEqual(decisions, [], 'failed policy persistence never releases the tool call')
    assert.equal(pendingApprovals.has('call-2'), true, 'the user can retry or deny the still-pending approval')
    assert.equal(pendingApprovalMeta.has('call-2'), true)
  }
} finally {
  store.threads = originalThreads
  if (originalRole == null) delete process.env.VONTRA_SECURITY_ROLE
  else process.env.VONTRA_SECURITY_ROLE = originalRole
  if (originalHome == null) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = originalHome
  await fs.rm(testHome, { recursive: true, force: true })
}

console.log('job-control-approval-durability.test.ts passed')
