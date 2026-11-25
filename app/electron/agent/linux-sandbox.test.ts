import assert from 'node:assert/strict'
import { chmodSync, mkdtempSync, mkdirSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { canUseLinuxSandbox, linuxSandboxArgv, linuxSandboxBinary, planLinuxMounts } from './linux-sandbox'
import { runSandboxedCommand } from './sandbox'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const root = mkdtempSync(path.join(realpathSync(tmpdir()), 'vontra-linux-sandbox-'))
try {
  const cwd = path.join(root, 'ws')
  mkdirSync(path.join(cwd, '.git', 'hooks'), { recursive: true })
  mkdirSync(path.join(cwd, '.git', 'info'), { recursive: true })
  writeFileSync(path.join(cwd, '.git', 'config'), '[core]\n')

  const buildTest = linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false })
  assert.ok(buildTest.includes('--unshare-all'), 'namespaces unshared')
  assert.ok(!buildTest.includes('--share-net'), 'network stays unshared')
  assert.ok(buildTest.includes('--die-with-parent'))
  const bind = (argv: string[], flag: string, target: string) =>
    argv.some((arg, i) => arg === flag && argv[i + 2] === target)
  assert.ok(bind(buildTest, '--bind', cwd), 'cwd bound writable')
  assert.ok(bind(buildTest, '--ro-bind', path.join(cwd, '.git', 'config')), 'git config pinned read-only')
  assert.ok(bind(buildTest, '--ro-bind', path.join(cwd, '.git', 'hooks')), 'git hooks pinned read-only')
  assert.ok(bind(buildTest, '--bind', path.join(cwd, '.git')), 'git dir pinned against rename')
  assert.equal(buildTest.at(-1), '--', 'argv ends before the command')

  const nested = path.join(cwd, 'vendor', 'sub')
  mkdirSync(path.join(nested, '.git', 'hooks'), { recursive: true })
  writeFileSync(path.join(nested, '.git', 'config'), '[core]\n')
  const nestedArgv = linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false })
  assert.ok(bind(nestedArgv, '--bind', path.join(nested, '.git')), 'nested .git pinned against rename')
  assert.ok(bind(nestedArgv, '--ro-bind', path.join(nested, '.git', 'config')), 'nested git config read-only')
  assert.ok(bind(nestedArgv, '--ro-bind', path.join(nested, '.git', 'hooks')), 'nested git hooks read-only')
  ok('nested repositories are pinned like the workspace root')

  process.env.TENSORFOLD_LINUX_SANDBOX = '1'

  const inModules = path.join(cwd, 'node_modules', 'dep')
  mkdirSync(path.join(inModules, '.git', 'hooks'), { recursive: true })
  writeFileSync(path.join(inModules, '.git', 'config'), '[core]\n')
  const deep = path.join(cwd, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'deep-repo')
  mkdirSync(path.join(deep, '.git', 'hooks'), { recursive: true })
  writeFileSync(path.join(deep, '.git', 'config'), '[core]\n')
  const walked = linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false })
  assert.ok(bind(walked, '--bind', path.join(inModules, '.git')), 'a repository inside node_modules is pinned')
  assert.ok(bind(walked, '--bind', path.join(deep, '.git')), 'a deeply nested repository is pinned')
  ok('dependency folders and deep repositories are pinned, not skipped')

  let chain = path.join(cwd, 'too-deep')
  for (let i = 0; i < 34; i++) chain = path.join(chain, `level-${i}`)
  mkdirSync(chain, { recursive: true })
  assert.throws(() => linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false }), /will not walk below 32 directory levels/)
  rmSync(path.join(cwd, 'too-deep'), { recursive: true, force: true })
  ok('a workspace deeper than the walk bound is refused')

  const innerWorktree = path.join(cwd, 'inner-worktree')
  const innerGitdir = path.join(cwd, '.git', 'worktrees', 'inner')
  mkdirSync(innerGitdir, { recursive: true })
  mkdirSync(innerWorktree, { recursive: true })
  writeFileSync(path.join(innerWorktree, '.git'), 'gitdir: ../.git/worktrees/inner\n')
  const worktreeArgv = linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false })
  assert.ok(bind(worktreeArgv, '--ro-bind', path.join(innerWorktree, '.git')), 'the worktree pointer is read-only')
  assert.ok(bind(worktreeArgv, '--bind', innerGitdir), 'the worktree gitdir stays writable for refs and objects')
  ok('a nested worktree is pinned through its gitdir and its pointer is frozen')

  const unreadable = path.join(cwd, 'unreadable-dir')
  mkdirSync(unreadable)
  chmodSync(unreadable, 0o000)
  assert.throws(() => linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false }), /cannot inspect .*unreadable-dir/)
  chmodSync(unreadable, 0o700)
  rmSync(unreadable, { recursive: true, force: true })
  ok('a directory the walk cannot inspect is refused')

  const sharedWorktree = path.join(cwd, 'shared-worktree')
  const sharedGitdir = path.join(cwd, '.git', 'worktrees', 'shared')
  const outsideShared = path.join(root, 'elsewhere-shared', '.git')
  mkdirSync(path.join(outsideShared, 'hooks'), { recursive: true })
  mkdirSync(sharedGitdir, { recursive: true })
  mkdirSync(sharedWorktree, { recursive: true })
  writeFileSync(path.join(sharedWorktree, '.git'), 'gitdir: ../.git/worktrees/shared\n')
  writeFileSync(path.join(sharedGitdir, 'commondir'), `${path.relative(sharedGitdir, outsideShared)}\n`)
  assert.throws(() => linuxSandboxArgv(cwd, { allowCwdWrite: true, allowNetwork: false }), /shared git directory is outside the workspace/)
  rmSync(path.join(root, 'elsewhere-shared'), { recursive: true, force: true })
  rmSync(sharedWorktree, { recursive: true, force: true })
  rmSync(path.join(cwd, '.git', 'worktrees', 'shared'), { recursive: true, force: true })
  ok('a worktree whose shared directory resolves outside the workspace is refused')

  const outsideWorktree = path.join(root, 'outside-worktree')
  mkdirSync(path.join(root, 'elsewhere', '.git', 'worktrees', 'outside'), { recursive: true })
  mkdirSync(outsideWorktree)
  writeFileSync(path.join(outsideWorktree, '.git'), 'gitdir: ../elsewhere/.git/worktrees/outside\n')
  assert.throws(() => linuxSandboxArgv(outsideWorktree, { allowCwdWrite: true, allowNetwork: false }), /outside the workspace/)
  if (canUseLinuxSandbox()) {
    const refused = await runSandboxedCommand('printf should-not-run', { cwd: outsideWorktree })
    assert.equal(refused.code, 126)
    assert.equal(refused.sandboxed, false)
    assert.match(refused.stderr, /outside the workspace/)
    ok('a worktree pointing outside the workspace is refused before the command runs')
  } else {
    console.log('  (worktree refusal runtime check skipped: bubblewrap unavailable)')
  }

  const plain = path.join(root, 'plain-workspace')
  mkdirSync(plain)
  assert.throws(() => linuxSandboxArgv(plain, { allowCwdWrite: true, allowNetwork: false }), /needs a git workspace/)
  if (canUseLinuxSandbox()) {
    const plainRun = await runSandboxedCommand('printf should-not-run', { cwd: plain })
    assert.equal(plainRun.code, 126)
    assert.match(plainRun.stderr, /needs a git workspace/)
    ok('a workspace without a repository keeps commands refused')
  } else {
    console.log('  (non-repository refusal runtime check skipped: bubblewrap unavailable)')
  }

  const readOnly = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: false })
  assert.ok(bind(readOnly, '--ro-bind', cwd), 'cwd bound read-only')
  assert.ok(!bind(readOnly, '--bind', cwd), 'cwd not writable')
  assert.ok(!readOnly.some((arg, i) => arg === '--ro-bind' && readOnly[i + 2] === path.join(cwd, '.git')), 'no git pinning without cwd writes')
  const network = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: true })
  assert.ok(network.includes('--share-net'), 'network-fetch shares the network namespace')
  ok('draft profile contains requested mount and network arguments (not confinement proof)')

  const tmpCwd = path.join(realpathSync(tmpdir()), 'vontra-tmp-workspace')
  mkdirSync(tmpCwd, { recursive: true })
  const tmpMounts = planLinuxMounts(tmpCwd, { allowCwdWrite: false, allowNetwork: false })
  assert.deepEqual(tmpMounts.filter((mount) => mount.path === tmpCwd).map((mount) => mount.mode), ['ro'], 'read-only workspace stays read-only')
  const tmpIndex = tmpMounts.findIndex((mount) => mount.path === realpathSync(tmpdir()))
  const cwdIndex = tmpMounts.findIndex((mount) => mount.path === tmpCwd)
  assert.ok(tmpIndex >= 0 && cwdIndex > tmpIndex, 'the read-only workspace mount follows its writable temporary parent')
  assert.equal(tmpMounts[cwdIndex].mode, 'ro')
  const writeMounts = planLinuxMounts(tmpCwd, { allowCwdWrite: true, allowNetwork: false })
  assert.equal(writeMounts.find((mount) => mount.path === tmpCwd)?.mode, 'rw')
  const ordered = planLinuxMounts(cwd, { allowCwdWrite: true, allowNetwork: false })
  for (let i = 1; i < ordered.length; i++) assert.ok(ordered[i - 1].path.length <= ordered[i].path.length, 'mounts are parent-first')
  ok('mount plan keeps a read-only workspace decisive under a writable temporary parent')

  const real = path.join(root, 'real-usr-bin')
  const link = path.join(root, 'bin')
  mkdirSync(real)
  symlinkSync(real, link)
  const withLink = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: false, extraReadRoots: [link] })
  assert.ok(withLink.some((arg, i) => arg === '--symlink' && withLink[i + 1] === real && withLink[i + 2] === link), 'symlinked root gets a compat symlink')
  assert.ok(bind(withLink, '--ro-bind', real), 'the resolved root is bound')
  ok('symlinked read roots resolve and keep their original path visible')

  if (canUseLinuxSandbox()) {
    const hello = await runSandboxedCommand('printf hello', { cwd })
    assert.equal(hello.stdout, 'hello')
    assert.equal(hello.implementation, 'linux-bwrap')
    const escape = await runSandboxedCommand('touch /usr/bin/vontra-escape-attempt', { cwd, profile: 'read-only' })
    assert.notEqual(escape.code, 0, 'writes outside the profile fail')
    assert.equal(escape.sandboxed, true)
    const planted = await runSandboxedCommand('printf pwned > .git/hooks/evil', { cwd, profile: 'build-test' })
    assert.notEqual(planted.code, 0, 'git hooks stay read-only under build-test')
    const net = await runSandboxedCommand('curl -sI --max-time 3 https://example.com', { cwd, profile: 'build-test' })
    assert.notEqual(net.code, 0, 'build-test has no network')
    ok('bwrap confinement: fs, git-control and network rules enforced')
  } else {
    console.log('  (draft Linux backend disabled pending qualification; confinement NOT verified)')
  }
} finally {
  rmSync(root, { recursive: true, force: true })
}

console.log(`\n${passed} passed`)
