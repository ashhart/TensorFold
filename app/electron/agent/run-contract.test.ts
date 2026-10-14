import assert from 'node:assert/strict'
import {
  agenticSurfaceFromThread,
  buildAgentRunContract,
  buildAgentRunReceipt,
  buildEventReceipt,
  guardPolicyForThread,
} from './run-contract'
import type { Thread } from './types'

const now = 1_700_000_000_000

function thread(overrides: Partial<Thread> = {}): Thread {
  return {
    id: 'thread-1',
    title: 'Ship receipts',
    kind: 'task',
    projectId: 'project-1',
    cwd: '/repo',
    providerId: 'openai',
    providerAccountId: 'acct-1',
    model: 'gpt-5',
    createdAt: now,
    updatedAt: now,
    status: 'idle',
    apiMessages: [],
    timeline: [],
    ...overrides,
  }
}

{
  const t = thread({ kind: 'chat', cwd: '/repo' })
  assert.equal(agenticSurfaceFromThread(t), 'chat')
  const policy = guardPolicyForThread(t)
  assert.equal(policy.maxSteps, 24)
  assert.equal(policy.verifyOnStop, false)
}

{
  const t = thread({ workspace: { kind: 'ssh', hostId: 'mini', path: '/srv/app' }, cwd: '/srv/app' })
  const contract = buildAgentRunContract({
    thread: t,
    runId: 'run-1',
    objective: 'Fix the login flake.',
    provider: { id: 'openai', kind: 'cloud', cliSandbox: 'workspace-write', cliApprovalPolicy: 'on-request' },
    providerAccountId: 'acct-1',
    model: 'gpt-5',
    approvalMode: 'suggest',
    createdAt: now,
  })
  assert.equal(contract.targetKind, 'remote')
  assert.match(contract.targetLabel, /mini/)
  assert.equal(contract.session.source, 'remote')
  assert.ok(contract.permissions.includes('sandbox:workspace-write'))
  assert.equal(contract.guardPolicy.verifyOnStop, true)
}

{
  const t = thread({
    timeline: [
      { id: 'contract', kind: 'notice', title: 'Run contract', text: 'saved', createdAt: now },
      { id: 'edit', kind: 'tool', tool: 'edit_file', path: 'src/App.tsx', state: 'ok', diff: [{ type: 'add', text: '+x' }], eventReceipt: buildEventReceipt({ kind: 'diff_finalized', runId: 'run-1', threadId: 'thread-1', itemId: 'edit', summary: 'diff', createdAt: now + 1 }), createdAt: now + 1 },
    ],
  })
  const contract = buildAgentRunContract({ thread: t, runId: 'run-1', objective: 'edit', provider: null, model: 'gpt-5', approvalMode: 'suggest', createdAt: now })
  const receipt = buildAgentRunReceipt({ thread: t, contract, timelineStart: 0, finishedAt: now + 10 })
  assert.equal(receipt.verdict, 'unverified')
  assert.deepEqual(receipt.changedFiles, ['src/App.tsx'])
  assert.equal(receipt.eventReceipts.length, 1)
  assert.match(receipt.gaps.join('\n'), /fresh test/)
}

{
  const t = thread({
    timeline: [
      { id: 'edit', kind: 'tool', tool: 'edit_file', path: 'src/App.tsx', state: 'ok', createdAt: now + 1 },
      { id: 'test', kind: 'tool', tool: 'run_command', command: 'npm run typecheck', state: 'ok', createdAt: now + 2 },
    ],
  })
  const contract = buildAgentRunContract({ thread: t, runId: 'run-2', objective: 'edit', provider: null, model: 'gpt-5', approvalMode: 'suggest', createdAt: now })
  const receipt = buildAgentRunReceipt({ thread: t, contract, timelineStart: 0, finishedAt: now + 10 })
  assert.equal(receipt.verdict, 'verified')
  assert.deepEqual(receipt.tests, ['npm run typecheck'])
}

{
  const t = thread({
    timeline: [
      { id: 'guard', kind: 'notice', title: 'Guard', text: 'Guardrail: stop repeating this action.', state: 'ok', createdAt: now + 1 },
      { id: 'err', kind: 'error', text: 'provider failed', createdAt: now + 2 },
    ],
  })
  const contract = buildAgentRunContract({ thread: t, runId: 'run-3', objective: 'fail', provider: null, model: 'gpt-5', approvalMode: 'suggest', createdAt: now })
  const receipt = buildAgentRunReceipt({ thread: t, contract, timelineStart: 0, finishedAt: now + 10 })
  assert.equal(receipt.verdict, 'failed')
  assert.match(receipt.warnings.join('\n'), /Guardrail/)
}

console.log('run-contract tests passed')
