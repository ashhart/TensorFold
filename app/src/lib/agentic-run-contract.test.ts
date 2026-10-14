import assert from 'node:assert/strict'
import {
  buildAgenticRunContract,
  guardPolicyForSurface,
  surfaceFromThread,
  type AgenticSurface,
} from './agentic-run-contract'
import type { GitStatus, Settings, Thread } from './types'

const now = 1_700_000_000_000

function thread(overrides: Partial<Thread> = {}): Thread {
  return {
    id: 'thread-1',
    title: 'Implement run receipts',
    kind: 'task',
    projectId: 'project-1',
    cwd: '/repo',
    providerId: 'openai',
    providerAccountId: 'acct-1',
    model: 'gpt-5',
    createdAt: now,
    updatedAt: now + 1000,
    status: 'idle',
    apiMessages: [],
    timeline: [],
    ...overrides,
  }
}

function settings(overrides: Partial<Settings> = {}): BuildSettings {
  return {
    providers: [
      {
        id: 'openai',
        name: 'OpenAI',
        baseUrl: 'https://api.openai.com/v1',
        apiKey: 'sk-test',
        models: ['gpt-5'],
        cliSandbox: 'workspace-write',
        cliApprovalPolicy: 'on-request',
      } as any,
    ],
    activeProviderId: 'openai',
    activeProviderAccountId: 'acct-1',
    activeModel: 'gpt-5',
    approvalMode: 'suggest',
    allowTaskNetwork: false,
    autoCompact: true,
    ...overrides,
  }
}

type BuildSettings = Pick<Settings, 'providers' | 'activeProviderId' | 'activeProviderAccountId' | 'activeModel' | 'approvalMode' | 'allowTaskNetwork' | 'autoCompact'>

const git: GitStatus = {
  branch: 'feature/run-contract',
  files: [{ path: 'src/lib/agentic-run-contract.ts', status: 'M', additions: 42, deletions: 1 }],
  additions: 42,
  deletions: 1,
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      kind: 'chat',
      projectId: null,
      cwd: '/repo-even-though-chat',
      timeline: [{ id: 'u1', kind: 'user', text: 'Talk me through the plan.', createdAt: now }],
    }),
  })
  assert.equal(contract.surface, 'chat')
  assert.equal(contract.guardPolicy.verifyOnStop, false)
  assert.equal(contract.guardPolicy.maxSteps, 24)
  assert.equal(contract.receipt.verdict, 'not_applicable')
  assert.equal(contract.session.source, 'thread')
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      timeline: [
        { id: 'u1', kind: 'user', text: 'Ship VST-002.', createdAt: now },
        { id: 'e1', kind: 'tool', tool: 'edit_file', path: 'src/App.tsx', state: 'ok', diff: [{ type: 'add', text: '+x' }], createdAt: now + 1 },
      ],
    }),
  })
  assert.equal(contract.surface, 'code')
  assert.equal(contract.receipt.verdict, 'unverified')
  assert.equal(contract.receipt.needsReview, true)
  assert.match(contract.receipt.gaps.join('\n'), /verification|fresh test/)
  assert.deepEqual(contract.changedFiles, ['src/App.tsx'])
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      timeline: [
        { id: 'e1', kind: 'tool', tool: 'edit_file', path: 'src/App.tsx', state: 'ok', createdAt: now + 1 },
        { id: 't1', kind: 'tool', tool: 'run_command', command: 'npm run typecheck', state: 'ok', meta: 'exit 0', createdAt: now + 2 },
      ],
    }),
  })
  assert.equal(contract.receipt.verdict, 'verified')
  assert.deepEqual(contract.tests, ['npm run typecheck'])
  assert.equal(contract.receipt.testCount, 1)
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      timeline: [
        { id: 'e1', kind: 'tool', tool: 'edit_file', path: 'src/App.tsx', state: 'ok', createdAt: now + 1 },
        { id: 't1', kind: 'tool', tool: 'run_command', command: 'npm test', state: 'error', output: 'failed', createdAt: now + 2 },
      ],
    }),
  })
  assert.equal(contract.receipt.verdict, 'failed')
  assert.equal(contract.receipt.errorCount, 1)
  assert.equal(contract.receipt.needsReview, true)
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      status: 'running',
      timeline: [{ id: 'a1', kind: 'tool', tool: 'run_command', command: 'rm -rf dist', state: 'awaiting', approval: 'ask', createdAt: now + 1 }],
    }),
    status: 'awaiting',
  })
  assert.equal(contract.receipt.verdict, 'blocked')
  assert.equal(contract.pendingApprovals, 1)
  assert.equal(contract.session.state, 'awaiting_approval')
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      cwd: '/srv/app',
      workspace: { kind: 'ssh', hostId: 'mini', path: '/srv/app' },
    }),
  })
  assert.equal(contract.targetKind, 'remote')
  assert.match(contract.targetLabel, /mini/)
  assert.equal(contract.session.source, 'remote')
}

{
  const contract = buildAgenticRunContract({
    thread: thread({
      status: 'running',
      codexResumeCursor: { threadId: 'codex-session-1' },
      compactions: [{ id: 'c1', at: now, trigger: 'auto', messagesFolded: 8, tokensBefore: 9000, tokensAfter: 2000, summaryExcerpt: 'folded', kindsDropped: ['tool'] }],
      timeline: [
        { id: 'cue-1', kind: 'cue', title: 'Guide', text: 'Focus on tests.', state: 'pending', meta: 'queued', createdAt: now + 1 },
        { id: 'cue-2', kind: 'cue', title: 'Guide', text: 'Use the remote host.', state: 'ok', meta: 'applied', createdAt: now + 2 },
      ],
    }),
    status: 'tool',
  })
  assert.equal(contract.session.state, 'queued_followup')
  assert.equal(contract.session.queueDepth, 1)
  assert.equal(contract.session.guidedCueCount, 1)
  assert.equal(contract.session.resumeCursor, 'codex-session-1')
  assert.equal(contract.session.resumePending, true)
  assert.equal(contract.compactions, 1)
}

{
  assert.equal(surfaceFromThread(thread({ kind: 'chat', cwd: '/repo', projectId: 'project-1' })), 'chat')
  assert.equal(surfaceFromThread(thread({ kind: 'task', projectId: null, cwd: '' })), 'code')
  assert.equal(surfaceFromThread(thread({ timeline: [{ id: 'connector', kind: 'notice', text: 'source', trace: { traceId: 't', surface: 'connector' } }] })), 'assistant')
  assert.equal(surfaceFromThread(thread({ title: 'Kanban ticket: fix bug' })), 'ticket')
}

{
  const surfaces: AgenticSurface[] = ['chat', 'code', 'assistant', 'room', 'ticket', 'automation', 'remote']
  for (const surface of surfaces) {
    const policy = guardPolicyForSurface(surface)
    assert.equal(policy.surface, surface)
    assert.equal(policy.repeatLimit, 3)
    assert.ok(policy.searchLimit > 0)
    assert.ok(policy.noProgressLimit > 0)
    assert.equal(policy.textLoopBreaker, true)
  }
  assert.equal(guardPolicyForSurface('code').verifyOnStop, true)
  assert.equal(guardPolicyForSurface('assistant').verifyOnStop, false)
}

{
  const contract = buildAgenticRunContract({
    thread: thread(),
    git,
    settings: settings({ approvalMode: 'auto', allowTaskNetwork: true }),
  })
  assert.equal(contract.branch, 'feature/run-contract')
  assert.equal(contract.providerId, 'openai')
  assert.equal(contract.providerAccountId, 'acct-1')
  assert.equal(contract.model, 'gpt-5')
  assert.ok(contract.permissions.includes('approval:auto'))
  assert.ok(contract.permissions.includes('network:allowed'))
  assert.ok(contract.permissions.includes('sandbox:workspace-write'))
}

// VST-011: a successful save_artifact registration is receipt-visible evidence.
{
  const thread = {
    id: 't-art', title: 'artifact run', kind: 'task', cwd: '/repo', providerId: 'openai', model: 'gpt-5',
    createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [],
    timeline: [
      { id: 'u1', kind: 'user', text: 'make a report', createdAt: 1 },
      { id: 'a1', kind: 'tool', tool: 'save_artifact', title: 'Artifact: Release report', state: 'ok', createdAt: 2 },
      { id: 'a2', kind: 'tool', tool: 'save_artifact', title: 'Artifact: Broken one', state: 'error', createdAt: 3 },
    ],
  } as any
  const contract = buildAgenticRunContract({ thread, git: null, status: 'idle', settings: null, pendingApprovals: 0 })
  assert.equal(contract.receipt.artifactCount, 1, 'only SUCCESSFUL registrations count')
  assert.deepEqual(contract.artifacts, ['Release report'], 'the receipt names the artifact without the tool prefix')
}

console.log('agentic-run-contract.test.ts passed')
