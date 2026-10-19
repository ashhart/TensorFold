// Run: node electron/agent/linux-sandbox.test.ts
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { canUseLinuxSandbox, linuxSandboxArgv, linuxSandboxBinary } from './linux-sandbox'
import { runSandboxedCommand } from './sandbox'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const root = mkdtempSync(path.join(realpathSync(tmpdir()), 'vontra-linux-sandbox-'))
try {
  const cwd = path.join(root, 'ws')
  mkdirSync(path.join(cwd, '.git', 'hooks'), { recursive: true })
  mkdirSync(path.join(cwd, '.git', 'info'), { recursive: true })
  writeFileSync(path.join(cwd, '.git', 'config'), '[core]\n')

  // build-test: cwd writable, network denied, git control pinned.
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

  // read-only: cwd read-only; network-fetch: network shared.
  const readOnly = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: false })
  assert.ok(bind(readOnly, '--ro-bind', cwd), 'cwd bound read-only')
  assert.ok(!bind(readOnly, '--bind', cwd), 'cwd not writable')
  assert.ok(!readOnly.some((arg, i) => arg === '--ro-bind' && readOnly[i + 2] === path.join(cwd, '.git')), 'no git pinning without cwd writes')
  const network = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: true })
  assert.ok(network.includes('--share-net'), 'network-fetch shares the network namespace')
  ok('draft profile contains requested mount and network arguments (not confinement proof)')

  // usrmerge-style symlinked roots get a compatibility symlink, not a broken path.
  const real = path.join(root, 'real-usr-bin')
  const link = path.join(root, 'bin')
  mkdirSync(real)
  symlinkSync(real, link)
  const withLink = linuxSandboxArgv(cwd, { allowCwdWrite: false, allowNetwork: false, extraReadRoots: [link] })
  assert.ok(withLink.some((arg, i) => arg === '--symlink' && withLink[i + 1] === real && withLink[i + 2] === link), 'symlinked root gets a compat symlink')
  assert.ok(bind(withLink, '--ro-bind', real), 'the resolved root is bound')
  ok('symlinked read roots resolve and keep their original path visible')

  // Draft integration cases; the backend remains disabled pending adversarial qualification.
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
