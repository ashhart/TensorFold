// Run: node electron/agent/sandbox-git-control.test.ts
// A writable sandbox must not let a command plant git config, hooks or a fresh .git that
// host-side git (outside the sandbox) would later execute.
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { canUseMacSandbox, gitControlWriteDeny, macSandboxProfile } from './sandbox'
import { hostGitEnv } from './host-git-env'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

{
  const env = hostGitEnv({ PATH: '/usr/bin', GIT_CONFIG_COUNT: '1', GIT_CONFIG_KEY_0: 'user.name', GIT_CONFIG_VALUE_0: 'x' })
  assert.equal(env.GIT_CONFIG_COUNT, '3')
  assert.equal(env.GIT_CONFIG_KEY_0, 'user.name')
  assert.deepEqual([env.GIT_CONFIG_KEY_1, env.GIT_CONFIG_VALUE_1], ['core.fsmonitor', 'false'])
  assert.deepEqual([env.GIT_CONFIG_KEY_2, env.GIT_CONFIG_VALUE_2], ['core.hooksPath', '/dev/null'])
  ok('host git env appends fsmonitor/hooks overrides after existing GIT_CONFIG entries')
}

const dir = mkdtempSync(path.join(os.tmpdir(), 'vontra-sandbox-git-'))
try {
  {
    const readOnly = macSandboxProfile(dir, 'read-only')
    const writable = macSandboxProfile(dir, 'build-test')
    assert.ok(!readOnly.includes('(deny file-write*'), 'read-only profiles need no extra deny')
    assert.ok(writable.includes(gitControlWriteDeny(dir).trim()), 'writable profiles carry the git control deny')
    ok('writable profiles deny git control files; read-only ones are unchanged')
  }

  const env = { PATH: '/usr/bin:/bin', HOME: dir, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@example.com', GIT_COMMITTER_NAME: 't', GIT_COMMITTER_EMAIL: 't@example.com' }
  const gitWorks = spawnSync('/bin/sh', ['-c', 'git init -q . && echo a > a.txt && git add a.txt && git -c commit.gpgsign=false commit -qm init && mkdir -p sub && cd sub && git init -q .'], { cwd: dir, env }).status === 0
  if (!canUseMacSandbox() || !gitWorks) {
    console.log('  - skipped live sandbox checks (no sandbox-exec or no git)')
  } else {
    const profile = macSandboxProfile(dir, 'build-test')
    const allowed = (cmd: string) => spawnSync('/usr/bin/sandbox-exec', ['-p', profile, '/bin/sh', '-c', cmd], { cwd: dir, env }).status === 0
    assert.ok(allowed('echo b > b.txt'), 'source writes still work')
    assert.ok(allowed('git add b.txt && git -c commit.gpgsign=false commit -qm b'), 'git commit still works')
    assert.ok(allowed('git checkout -qb feature && git checkout -q -'), 'git checkout still works')
    ok('normal workspace and git writes are allowed')
    for (const [what, cmd] of [
      ['.git/config append', 'printf "[core]\\n fsmonitor = false\\n" >> .git/config'],
      ['git config', 'git config core.fsmonitor false'],
      ['a hook', 'echo x > .git/hooks/pre-commit'],
      ['info/attributes', 'echo x > .git/info/attributes'],
      ['renaming .git', 'mv .git .git-old'],
      ['a nested repo config', 'echo x >> sub/.git/config'],
      ['creating a nested .git', 'mkdir -p other && mkdir other/.git'],
    ]) assert.ok(!allowed(cmd), `${what} is denied`)
    ok('git config, hooks, info and the .git entry are denied at any depth')
  }
} finally {
  rmSync(dir, { recursive: true, force: true })
}

console.log(`\n${passed} passed`)
