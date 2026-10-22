// Run: npx tsx electron/agent/planning-sandbox.test.ts
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { cliConfinement, cliInstallRoots, cliStateDirs, confineProvider, planningCliArgs, planningCommand, planningProfileFor, planningRoot, preparePlanningDir, removePlanningDir } from './planning-sandbox'
import { canUseMacSandbox } from './sandbox'
import type { Provider } from './types'

const provider = (transport: Provider['transport']): Provider => ({ id: String(transport), name: String(transport), transport, baseUrl: '', apiKey: '', models: [] })

// Each CLI's own switches: no shell/file tools, plugins, MCP or hooks.
{
  const codex = planningCliArgs('codex-cli', ['exec', '-m', 'gpt-5.5', '-s', 'read-only', '-'])
  assert.equal(codex[0], 'exec')
  assert.deepEqual(codex.slice(-5), ['-m', 'gpt-5.5', '-s', 'read-only', '-'])
  assert.ok(codex.includes('features.shell_tool=false'))
  assert.ok(codex.includes('web_search="disabled"'))
  assert.equal(planningCliArgs('codex-cli', ['app-server'])[0], 'app-server')
  const claude = planningCliArgs('claude-cli', ['-p', '--permission-mode', 'plan'])
  assert.deepEqual(claude.slice(0, 2), ['--tools', ''])
  assert.ok(claude.includes('--strict-mcp-config') && claude.includes('--safe-mode'))
  const pi = planningCliArgs('pi-cli', ['--mode', 'rpc', '--tools', 'read', '--session-id', 'abc'])
  assert.ok(pi.includes('--no-tools') && pi.includes('--no-context-files'))
  assert.ok(!pi.includes('--tools') && !pi.includes('read'))
  assert.deepEqual(planningCliArgs('cursor-cli', ['-p']), ['--mode=ask', '-p'])
  assert.deepEqual(planningCliArgs('gemini-cli', ['--approval-mode', 'plan']), ['--approval-mode', 'plan'])
}

// Auth/config dirs follow each CLI's env overrides.
{
  const home = '/Users/example'
  assert.ok(cliStateDirs('codex-cli', {}, home).includes('/Users/example/.codex'))
  assert.ok(cliStateDirs('codex-cli', { CODEX_HOME: '/Users/example/.codex-work' }, home).includes('/Users/example/.codex-work'))
  assert.ok(cliStateDirs('claude-cli', { CLAUDE_CONFIG_DIR: '/Users/example/.claude-work' }, home).includes('/Users/example/.claude-work'))
  assert.ok(cliStateDirs('pi-cli', {}, home).includes('/Users/example/.pi'))
  assert.ok(cliStateDirs('gemini-cli', {}, home).includes('/Users/example/Library/Keychains'))
}

// Planning folders are private mkdtemp dirs in the OS temp dir: no username in the path, one per
// thread for the app's lifetime, fresh for one-shots, stale ones from dead pids swept.
const scratch = realpathSync(mkdtempSync(path.join(os.tmpdir(), 'tf-planning-test-')))
try {
  assert.equal(planningRoot(), realpathSync(os.tmpdir()))
  assert.ok(!planningRoot().startsWith(os.homedir()), 'the planning root is outside the home folder')
  const root = path.join(scratch, 'root')
  mkdirSync(root)
  const stale = path.join(root, 'tf-plan-999999-dead')
  mkdirSync(stale)
  const c = preparePlanningDir('thread-1', root)
  assert.ok(!existsSync(stale), 'folders of dead app runs are swept')
  assert.equal(path.dirname(c.planningDir), realpathSync(root))
  assert.match(path.basename(c.planningDir), new RegExp(`^tf-plan-${process.pid}-`))
  assert.equal(statSync(c.planningDir).mode & 0o777, 0o700)
  assert.ok(existsSync(c.tmpDir))
  assert.equal(preparePlanningDir('thread-1', root), c, 'a thread keeps its folder')
  const oneShot = preparePlanningDir(null, root)
  assert.notEqual(oneShot.planningDir, c.planningDir)
  removePlanningDir({ planningDir: scratch, tmpDir: scratch }, root)
  assert.ok(existsSync(scratch), 'refuses to remove outside the planning root')
  removePlanningDir(c, root)
  removePlanningDir(oneShot, root)
  assert.ok(!existsSync(c.planningDir) && !existsSync(oneShot.planningDir))
  assert.notEqual(preparePlanningDir('thread-1', root).planningDir, c.planningDir, 'a removed folder is not reused')

  // Unconfined providers pass straight through.
  const plain = provider('claude-cli')
  assert.equal(cliConfinement(plain), null)
  assert.deepEqual(planningCommand(plain, '/bin/echo', ['hi'], { cwd: '/x', env: { A: '1' } }), { command: '/bin/echo', args: ['hi'], cwd: '/x', env: { A: '1' } })

  // Install roots include the link's prefix and its target's, but never the home dir itself.
  const fakeHome = path.join(scratch, 'home')
  mkdirSync(path.join(fakeHome, '.local', 'bin'), { recursive: true })
  writeFileSync(path.join(fakeHome, '.local', 'bin', 'tool'), '#!/bin/sh\n')
  const roots = cliInstallRoots(path.join(fakeHome, '.local', 'bin', 'tool'), {}, fakeHome)
  const interpreter = realpathSync('/bin/sh')
  const prefix = path.dirname(path.dirname(interpreter))
  const interpreterRoot = prefix === '/' ? path.dirname(interpreter) : prefix
  assert.deepEqual(roots, [path.join(fakeHome, '.local', 'bin'), '/bin', ...(interpreter !== '/bin/sh' ? [interpreterRoot] : [])])

  // The workspace is denied last, unless it holds the CLI's own dirs (a home-folder workspace).
  {
    const ws = path.join(fakeHome, 'code', 'ws')
    mkdirSync(path.join(fakeHome, '.gemini'), { recursive: true })
    mkdirSync(ws, { recursive: true })
    const confinement = preparePlanningDir('deny', root)
    const profile = (denyRoots: string[]) => planningProfileFor({ transport: 'gemini-cli', command: '/bin/echo', args: [], env: {}, confinement: { ...confinement, denyRoots }, home: fakeHome })
    assert.match(profile([ws]), new RegExp(`\\(deny file-read\\* file-write\\*\\n\\s+\\(subpath "${ws}"\\)`))
    assert.doesNotMatch(profile([fakeHome]), /\(deny file-read\* file-write\*\n/)
  }

  if (process.platform === 'darwin' && canUseMacSandbox()) {
    // Real seatbelt: a CLI-like process (this Node binary) inside the planning profile.
    const workspace = path.join(fakeHome, 'code', 'acme')
    const documents = path.join(fakeHome, 'Documents')
    const authDir = path.join(fakeHome, '.gemini')
    for (const dir of [workspace, documents, authDir]) mkdirSync(dir, { recursive: true })
    writeFileSync(path.join(workspace, 'plan.md'), 'workspace file')
    writeFileSync(path.join(documents, 'notes.txt'), 'documents file')
    writeFileSync(path.join(authDir, 'oauth_creds.json'), '{"token":"placeholder"}')
    const confinement = preparePlanningDir('proof', root)
    const confined = confineProvider(provider('gemini-cli'), confinement)
    const probe = `
      const fs = require('fs'); const out = {}
      const tryRead = (k, p) => { try { fs.readFileSync(p); out[k] = 'read' } catch (e) { out[k] = e.code } }
      const tryWrite = (k, p) => { try { fs.writeFileSync(p, 'x'); out[k] = 'wrote' } catch (e) { out[k] = e.code } }
      tryRead('workspace', ${JSON.stringify(path.join(workspace, 'plan.md'))})
      tryRead('documents', ${JSON.stringify(path.join(documents, 'notes.txt'))})
      try { fs.readdirSync(${JSON.stringify(documents)}); out.listDocuments = 'listed' } catch (e) { out.listDocuments = e.code }
      tryRead('auth', ${JSON.stringify(path.join(authDir, 'oauth_creds.json'))})
      tryWrite('planningWrite', ${JSON.stringify(path.join(confinement.planningDir, 'scratch.txt'))})
      tryWrite('workspaceWrite', ${JSON.stringify(path.join(workspace, 'new.txt'))})
      out.cwd = process.cwd(); out.tmp = require('os').tmpdir()
      out.nested = require('child_process').spawnSync('/usr/bin/sandbox-exec', ['-p', '(version 1)(allow default)', '/usr/bin/true']).status
      console.log(JSON.stringify(out))`
    const run = planningCommand(confined, process.execPath, ['-e', probe], { env: { PATH: '/usr/bin:/bin' }, home: fakeHome })
    assert.equal(run.command, '/usr/bin/sandbox-exec')
    const result = spawnSync(run.command, run.args, { cwd: run.cwd, env: run.env, encoding: 'utf8' })
    assert.equal(result.status, 0, result.stderr)
    const out = JSON.parse(result.stdout.trim().split('\n').pop()!)
    assert.equal(out.workspace, 'EPERM')
    assert.equal(out.documents, 'EPERM')
    assert.equal(out.listDocuments, 'EPERM')
    assert.equal(out.workspaceWrite, 'EPERM')
    assert.equal(out.auth, 'read')
    assert.equal(out.planningWrite, 'wrote')
    assert.equal(out.cwd, confinement.planningDir)
    assert.equal(out.tmp, confinement.tmpDir)
    // Nested seatbelts fail closed (what Codex's own command sandbox hits in planning mode).
    assert.equal(out.nested, 71)
  } else {
    console.log('planning-sandbox: seatbelt proof skipped (no sandbox-exec)')
  }
} finally {
  rmSync(scratch, { recursive: true, force: true })
}

console.log('planning-sandbox tests passed')
