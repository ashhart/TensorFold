import assert from 'node:assert/strict'
import { exec, execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { listWorktrees, type GitRunner } from '../../agent/workspace-git'
import { hostGitEnv } from '../../agent/host-git-env'
import { createLocalWorktree } from './workspace-git-runtime'

const run = promisify(exec)
const git = promisify(execFile)
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-worktree-alias-'))
const actual = path.join(root, 'actual profile')
const alias = path.join(root, 'profile alias')
const repo = path.join(actual, 'project')
const env = hostGitEnv({ ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null' })
const runner: GitRunner = {
  exec: async (command, options) => {
    try { return { code: 0, ...(await run(command, { cwd: options?.cwd, env })) } }
    catch (error) {
      const result = error as { code?: number; stdout?: string; stderr?: string }
      return { code: result.code ?? 1, stdout: result.stdout ?? '', stderr: result.stderr ?? '' }
    }
  },
}

try {
  await fs.mkdir(repo, { recursive: true })
  await fs.symlink(actual, alias, 'dir')
  await git('git', ['init', '--initial-branch=main'], { cwd: repo, env })
  await fs.writeFile(path.join(repo, 'README.md'), 'Example project\n')
  await git('git', ['add', 'README.md'], { cwd: repo, env })
  await git('git', ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', '-c', 'commit.gpgsign=false', 'commit', '-m', 'Initial fixture'], { cwd: repo, env })
  const branch = 'codex/worker-check-docs-0123456789abcdef'
  const target = path.join(alias, 'worktrees', 'codex-worker-check-docs-0123456789abcdef')
  const result = await createLocalWorktree(runner, repo, branch, target)
  assert.equal(result.ok, true, result.error)
  const canonical = await fs.realpath(target)
  assert.notEqual(target, canonical, 'Fixture must actually use an aliased path')
  const worktrees = await listWorktrees(runner, repo)
  assert.ok(worktrees.some((worktree) => worktree.path === result.path && worktree.branch === branch && !worktree.bare), 'The created worker path must match Git worktree discovery so frontend isolation verification succeeds')
  assert.equal(result.path, canonical, 'The local host returns the canonical worktree directory')
  assert.equal(await fs.readFile(path.join(result.path, 'README.md'), 'utf8'), 'Example project\n')
  const failed = await createLocalWorktree(runner, repo, branch, target)
  assert.equal(failed.ok, false, 'A failed Git add must retain its failure instead of returning a canonicalized success')
  assert.ok(failed.error)
} finally {
  await fs.rm(root, { recursive: true, force: true })
}

console.log('Local worktree creation returns the same canonical path and branch as real Git discovery through a symlinked profile.')
