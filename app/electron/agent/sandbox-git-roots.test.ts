import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { gitPointerRoots, linuxGitReadRoots } from './sandbox-git-roots'
import { hostGitEnv } from './host-git-env'
import { canUseLinuxSandbox } from './linux-sandbox'
import { canUseWindowsSandbox } from './windows-sandbox'
import { runSandboxedCommand } from './sandbox'

const parent = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-git-view-'))
const repository = path.join(parent, 'repository'), tree = path.join(parent, 'tree'), template = path.join(parent, 'empty-template')
await fs.mkdir(repository); await fs.mkdir(template)
const env = hostGitEnv({ ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: process.platform === 'win32' ? 'NUL' : '/dev/null' })
const git = (...args: string[]) => execFileSync('git', args, { cwd: repository, env, stdio: 'pipe' })
try {
  git('init', '-q', '--template=' + template)
  await fs.writeFile(path.join(repository, 'source.txt'), 'fixture')
  git('add', 'source.txt')
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com', '-c', 'commit.gpgSign=false', 'commit', '-qm', 'fixture')
  git('worktree', 'add', '--detach', tree, 'HEAD')
  const roots = gitPointerRoots(path.join(tree, '.git'))
  assert.equal(roots.common, await fs.realpath(path.join(repository, '.git')))
  assert.deepEqual(linuxGitReadRoots(tree).sort(), [roots.git, roots.common].sort())
  if (canUseLinuxSandbox() || canUseWindowsSandbox()) {
    const result = await runSandboxedCommand('git status --porcelain', { cwd: tree, profile: 'read-only', timeoutMs: 30000 })
    assert.equal(result.code, 0, result.stderr)
    assert.equal(result.stdout.trim(), '')
    const write = await runSandboxedCommand('git config core.fsmonitor unsafe', { cwd: tree, profile: 'build-test', timeoutMs: 30000 })
    assert.notEqual(write.code, 0, 'Linked Git control directories stay read-only')
  }
  await fs.writeFile(path.join(roots.git, 'gitdir'), path.join(parent, 'different/.git') + '\n')
  assert.throws(() => gitPointerRoots(path.join(tree, '.git')), /back to the selected/)
  await fs.writeFile(path.join(tree, '.git'), 'gitdir: ' + os.tmpdir())
  assert.throws(() => gitPointerRoots(path.join(tree, '.git')), /backlink|metadata/)
  console.log('Git metadata views accept real worktree backlinks and reject retargeted or unrelated private directories.')
} finally { await fs.rm(parent, { recursive: true, force: true }) }
