// A restricted namespace must report a skip rather than claim confinement.
import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import path from 'node:path'
import { runSandboxedCommand } from './sandbox'

process.env.TENSORFOLD_LINUX_SANDBOX = '1'
const { canUseLinuxSandbox, probeLinuxSandbox } = await import('./linux-sandbox')

if (process.platform !== 'linux') {
  console.log('CONFORMANCE: skipped (not Linux)')
} else if (!probeLinuxSandbox(process.cwd())) {
  console.log('CONFORMANCE: skipped (bubblewrap cannot create a namespace on this host)')
} else {
  assert.equal(canUseLinuxSandbox(), true, 'the opt-in enables the backend once the probe passes')
  const root = realpathSync(mkdtempSync(path.join(realpathSync(tmpdir()), 'tf-bwrap-conformance-')))
  const cwd = path.join(root, 'ws')
  mkdirSync(path.join(cwd, '.git', 'hooks'), { recursive: true })
  writeFileSync(path.join(cwd, '.git', 'config'), '[core]\n')

  const server = createServer((socket) => socket.end('ok'))
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as { port: number }).port

  try {
    const hello = await runSandboxedCommand('printf confined', { cwd })
    assert.equal(hello.stdout, 'confined')
    assert.equal(hello.implementation, 'linux-bwrap', 'the Linux backend reported the run')

    const inside = await runSandboxedCommand('printf data > inside.txt', { cwd, profile: 'build-test' })
    assert.equal(inside.code, 0, inside.stderr)
    assert.equal(readFileSync(path.join(cwd, 'inside.txt'), 'utf8'), 'data')

    const outsideTarget = path.join(realpathSync(homedir()), 'tf-bwrap-escape.txt')
    const outside = await runSandboxedCommand(`printf escape > ${JSON.stringify(outsideTarget)}`, { cwd, profile: 'build-test' })
    assert.notEqual(outside.code, 0, 'writes outside the mounted roots fail')
    assert.equal(existsSync(outsideTarget), false, 'nothing appeared outside the workspace')

    const readOnly = await runSandboxedCommand('printf blocked > ro.txt', { cwd, profile: 'read-only' })
    assert.notEqual(readOnly.code, 0, 'read-only profiles cannot write the workspace')
    assert.equal(existsSync(path.join(cwd, 'ro.txt')), false)

    const gitTamper = await runSandboxedCommand('printf pwned > .git/hooks/evil', { cwd, profile: 'build-test' })
    assert.notEqual(gitTamper.code, 0, 'git hooks stay read-only under a writable profile')
    assert.equal(existsSync(path.join(cwd, '.git', 'hooks', 'evil')), false)

    const gitConfig = await runSandboxedCommand('printf pwned >> .git/config', { cwd, profile: 'build-test' })
    assert.notEqual(gitConfig.code, 0, 'git config stays read-only under a writable profile')
    assert.equal(readFileSync(path.join(cwd, '.git', 'config'), 'utf8'), '[core]\n')

    const nested = path.join(cwd, 'vendor', 'sub')
    mkdirSync(path.join(nested, '.git', 'hooks'), { recursive: true })
    writeFileSync(path.join(nested, '.git', 'config'), '[core]\n')
    const nestedHook = await runSandboxedCommand('printf pwned > vendor/sub/.git/hooks/evil', { cwd, profile: 'build-test' })
    assert.notEqual(nestedHook.code, 0, 'a nested repository\'s hooks stay read-only')
    assert.equal(existsSync(path.join(nested, '.git', 'hooks', 'evil')), false)
    const nestedConfig = await runSandboxedCommand('printf pwned >> vendor/sub/.git/config', { cwd, profile: 'build-test' })
    assert.notEqual(nestedConfig.code, 0, 'a nested repository\'s config stays read-only')
    assert.equal(readFileSync(path.join(nested, '.git', 'config'), 'utf8'), '[core]\n')

    const newConfig = await runSandboxedCommand('printf pwned > .git/config.worktree', { cwd, profile: 'build-test' })
    assert.notEqual(newConfig.code, 0, 'a missing control file cannot be created')
    assert.equal(existsSync(path.join(cwd, '.git', 'config.worktree')), true, 'the placeholder exists but stayed read-only')
    assert.equal(readFileSync(path.join(cwd, '.git', 'config.worktree'), 'utf8'), '', 'the placeholder was not written')

    const inModules = path.join(cwd, 'node_modules', 'dep')
    mkdirSync(path.join(inModules, '.git', 'hooks'), { recursive: true })
    writeFileSync(path.join(inModules, '.git', 'config'), '[core]\n')
    const moduleHook = await runSandboxedCommand('printf pwned > node_modules/dep/.git/hooks/evil', { cwd, profile: 'build-test' })
    assert.notEqual(moduleHook.code, 0, 'a repository inside node_modules stays pinned')

    const innerWorktree = path.join(cwd, 'inner-worktree')
    const innerGitdir = path.join(cwd, '.git', 'worktrees', 'inner')
    mkdirSync(innerGitdir, { recursive: true })
    mkdirSync(innerWorktree, { recursive: true })
    writeFileSync(path.join(innerWorktree, '.git'), 'gitdir: ../.git/worktrees/inner\n')
    const repoint = await runSandboxedCommand('printf "gitdir: /tmp/planted\n" > inner-worktree/.git', { cwd, profile: 'build-test' })
    assert.notEqual(repoint.code, 0, 'the worktree pointer stays read-only')
    assert.match(readFileSync(path.join(innerWorktree, '.git'), 'utf8'), /^gitdir: \.\.\/\.git\/worktrees\/inner/, 'the pointer was not rewritten')

    const unreadable = path.join(cwd, 'unreadable-dir')
    mkdirSync(unreadable)
    chmodSync(unreadable, 0o000)
    try {
      const refusedWalk = await runSandboxedCommand('printf should-not-run', { cwd, profile: 'build-test' })
      assert.equal(refusedWalk.code, 126, 'an uninspectable directory refuses the command')
      assert.match(refusedWalk.stderr, /cannot inspect .*unreadable-dir/)
    } finally {
      chmodSync(unreadable, 0o700)
      rmSync(unreadable, { recursive: true, force: true })
    }

    const sharedWorktree = path.join(cwd, 'shared-worktree')
    const sharedGitdir = path.join(cwd, '.git', 'worktrees', 'shared')
    const outsideShared = path.join(root, 'elsewhere-shared', '.git')
    mkdirSync(path.join(outsideShared, 'hooks'), { recursive: true })
    mkdirSync(sharedGitdir, { recursive: true })
    mkdirSync(sharedWorktree, { recursive: true })
    writeFileSync(path.join(sharedWorktree, '.git'), 'gitdir: ../.git/worktrees/shared\n')
    writeFileSync(path.join(sharedGitdir, 'commondir'), `${path.relative(sharedGitdir, outsideShared)}\n`)
    const refusedShared = await runSandboxedCommand('printf should-not-run', { cwd, profile: 'build-test' })
    assert.equal(refusedShared.code, 126, 'a worktree with an outside shared directory is refused')
    assert.match(refusedShared.stderr, /shared git directory is outside the workspace/)
    rmSync(sharedWorktree, { recursive: true, force: true })
    rmSync(path.join(cwd, '.git', 'worktrees', 'shared'), { recursive: true, force: true })
    rmSync(path.join(root, 'elsewhere-shared'), { recursive: true, force: true })

    const outsideWorktree = path.join(root, 'outside-worktree')
    mkdirSync(path.join(root, 'elsewhere', '.git', 'worktrees', 'outside'), { recursive: true })
    mkdirSync(outsideWorktree)
    writeFileSync(path.join(outsideWorktree, '.git'), 'gitdir: ../elsewhere/.git/worktrees/outside\n')
    const refusedWorktree = await runSandboxedCommand('printf should-not-run', { cwd: outsideWorktree, profile: 'build-test' })
    assert.equal(refusedWorktree.code, 126, 'a worktree outside the workspace is refused')
    assert.equal(refusedWorktree.sandboxed, false)
    assert.match(refusedWorktree.stderr, /outside the workspace/)

    const denied = await runSandboxedCommand(`timeout 3 bash -c 'exec 3<>/dev/tcp/127.0.0.1/${port}'`, { cwd, profile: 'build-test' })
    assert.notEqual(denied.code, 0, 'build-test cannot reach a loopback listener')
    const allowed = await runSandboxedCommand(`timeout 3 bash -c 'exec 3<>/dev/tcp/127.0.0.1/${port}'`, { cwd, profile: 'network-fetch' })
    assert.equal(allowed.code, 0, `network-fetch reaches the same listener: ${allowed.stderr}`)

    const controller = new AbortController()
    const started = Date.now()
    const pending = runSandboxedCommand('sleep 30', { cwd, profile: 'build-test', signal: controller.signal })
    setTimeout(() => controller.abort(), 300)
    const cancelled = await pending
    assert.notEqual(cancelled.code, 0, 'cancelled command did not report success')
    assert.ok(Date.now() - started < 10_000, 'cancellation stops the sandboxed command promptly')

    console.log('CONFORMANCE: passed (filesystem, git controls, placeholders, nested repositories, worktree pinning and refusal, shared-directory refusal, uninspectable-path refusal, network and cancellation enforced)')
  } finally {
    server.close()
    rmSync(root, { recursive: true, force: true })
  }
}
