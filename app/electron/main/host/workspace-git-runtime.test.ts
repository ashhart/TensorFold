import assert from 'node:assert/strict'
import {
  resolveWorkspaceBranch,
  resolveWorkspaceGitStatus,
  resolveWorkspaceGitDiff,
  type GitBackend,
  type WorkspaceGitDeps,
} from './workspace-git-runtime'
import type { WorkspaceLocation } from '../../agent/types'

type Call = { command: string; cwd?: string }

// A fake remote backend that answers git commands like a real repo over SSH would.
function fakeBackend(rootCwd: string, calls: Call[]): GitBackend {
  return {
    rootCwd,
    async exec(command, options) {
      calls.push({ command, cwd: options?.cwd })
      if (command.includes('rev-parse --abbrev-ref HEAD')) return { code: 0, stdout: 'remote-main\n', stderr: '' }
      if (command.includes('status --porcelain')) return { code: 0, stdout: ' M server.ts\n?? new.txt\n', stderr: '' }
      if (command.includes('diff HEAD --numstat')) return { code: 0, stdout: '5\t2\tserver.ts\n', stderr: '' }
      if (command.includes('--no-index')) return { code: 1, stdout: 'untracked body\n', stderr: '' }
      if (command.includes('diff HEAD --no-color')) return { code: 0, stdout: 'tracked body\n', stderr: '' }
      return { code: 0, stdout: '', stderr: '' }
    },
  }
}

const ssh = (path: string): WorkspaceLocation => ({ kind: 'ssh', hostId: 'h1', path })

function localOnlyDeps(overrides: Partial<WorkspaceGitDeps> = {}): WorkspaceGitDeps {
  return {
    resolveBackend: () => undefined,
    localBranch: async () => 'local-branch',
    localStatus: async () => ({ branch: 'local-branch', files: [], additions: 0, deletions: 0 }),
    localDiff: async () => [],
    ...overrides,
  }
}

async function run() {
  // --- remote status routes through the backend (engine), never the local path ---
  {
    const calls: Call[] = []
    const backend = fakeBackend('/srv/default', calls)
    let localHit = false
    const deps: WorkspaceGitDeps = {
      resolveBackend: (ws) => (ws?.kind === 'ssh' ? backend : undefined),
      localBranch: async () => { localHit = true; return 'local' },
      localStatus: async () => { localHit = true; return { branch: 'local', files: [], additions: 0, deletions: 0 } },
      localDiff: async () => { localHit = true; return [] },
    }
    // workspace.path is the canonical remote cwd and must win over the passed cwd.
    const status = await resolveWorkspaceGitStatus({ cwd: '/ignored/local', workspace: ssh('/remote/app') }, deps)
    assert.equal(status.branch, 'remote-main')
    assert.equal(status.additions, 5)
    assert.equal(status.deletions, 2)
    assert.deepEqual(
      status.files.find((f) => f.path === 'server.ts'),
      { path: 'server.ts', status: 'M', additions: 5, deletions: 2 },
    )
    assert.equal(localHit, false, 'remote status must not touch the local path')
    assert.ok(
      calls.some((c) => c.command.includes('status --porcelain') && c.cwd === '/remote/app'),
      'git must run on the backend at the remote workspace path',
    )
  }

  // --- remote branch ---
  {
    const calls: Call[] = []
    const deps = localOnlyDeps({ resolveBackend: () => fakeBackend('/srv', calls) })
    assert.equal(await resolveWorkspaceBranch({ cwd: '/x', workspace: ssh('/remote/repo') }, deps), 'remote-main')
    assert.ok(calls.every((c) => c.cwd === '/remote/repo'))
  }

  // --- remote diff routes through the backend, honoring the path filter ---
  {
    const calls: Call[] = []
    const deps = localOnlyDeps({ resolveBackend: () => fakeBackend('/srv', calls) })
    const diffs = await resolveWorkspaceGitDiff({ cwd: '/x', workspace: ssh('/remote/repo'), paths: ['server.ts'] }, deps)
    assert.equal(diffs.length, 1)
    assert.equal(diffs[0].path, 'server.ts')
    assert.match(diffs[0].diff, /tracked body/)
  }

  // --- remote with no usable path falls back to backend.rootCwd ---
  {
    const calls: Call[] = []
    const deps = localOnlyDeps({ resolveBackend: () => fakeBackend('/default/remote', calls) })
    await resolveWorkspaceGitStatus({ cwd: '', workspace: ssh('') }, deps)
    assert.ok(calls.length > 0)
    assert.ok(calls.every((c) => c.cwd === '/default/remote'), 'empty path uses backend.rootCwd')
  }

  // --- local fallback: no backend → delegate to local impls, never touch a backend ---
  {
    const calls: Call[] = []
    const backend = fakeBackend('/unused', calls)
    let statusCwd = ''
    const deps: WorkspaceGitDeps = {
      resolveBackend: () => undefined, // local workspace (or absent workspace)
      localBranch: async () => 'local-branch',
      localStatus: async (cwd) => {
        statusCwd = cwd
        return { branch: 'local-branch', files: [{ path: 'x.ts', status: 'M', additions: 1, deletions: 0 }], additions: 1, deletions: 0 }
      },
      localDiff: async () => [{ path: 'x.ts', status: 'M', additions: 1, deletions: 0, diff: 'local body' }],
    }
    const status = await resolveWorkspaceGitStatus({ cwd: '/local/app', workspace: { kind: 'local', path: '/local/app' } }, deps)
    assert.equal(status.branch, 'local-branch')
    assert.equal(statusCwd, '/local/app')
    assert.equal(calls.length, 0, 'local path must not touch the backend')
    assert.equal(await resolveWorkspaceBranch({ cwd: '/local/app' }, deps), 'local-branch')
    const diffs = await resolveWorkspaceGitDiff({ cwd: '/local/app' }, deps)
    assert.match(diffs[0].diff, /local body/)
  }

  // --- an explicitly remote workspace never falls back to a same-named local path ---
  {
    let localCalls = 0
    const deps = localOnlyDeps({
      localBranch: async () => { localCalls++; return 'wrong-local-branch' },
      localStatus: async () => { localCalls++; return { branch: 'wrong-local-branch', files: [], additions: 0, deletions: 0 } },
      localDiff: async () => { localCalls++; return [] },
    })
    const args = { cwd: '/same/path/on/this-mac', workspace: ssh('/same/path/on/remote') }
    await assert.rejects(() => resolveWorkspaceBranch(args, deps), /SSH workspace is disconnected/)
    await assert.rejects(() => resolveWorkspaceGitStatus(args, deps), /SSH workspace is disconnected/)
    await assert.rejects(() => resolveWorkspaceGitDiff(args, deps), /SSH workspace is disconnected/)
    assert.equal(localCalls, 0, 'a disconnected SSH workspace cannot invoke any local Git implementation')
  }

  console.log('workspace-git-runtime tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
