// Run: npx tsx electron/agent/sandbox.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { realpathSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { SANDBOX_PROFILES, buildSandboxArtifactBundle, canUseMacSandbox, captureSandboxArtifactBaseline, summarizeSandboxArtifactBundle, macSandboxProfile, runSandboxedCommand, sandboxCommandEnv, sandboxProfileForCommand, taskSandboxNetworkOverride } from './sandbox'

async function main() {
  // Keep the fixture outside both the repo (mutation snapshots must not see the
  // developer worktree) and TMPDIR (read-only profiles deliberately allow temp writes).
  const cwd = await fs.mkdtemp(path.join(os.homedir(), '.vontra-sandbox-test-'))
  const inside = path.join(cwd, 'inside.txt')
  const outside = path.join(os.homedir(), `.vontra-sandbox-outside-${Date.now()}`)

  const profile = macSandboxProfile(cwd, false)
  assert.match(profile, /deny default/)
  assert.doesNotMatch(profile, /allow network\*/)
  const extraRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-sandbox-extra-'))
  assert.match(macSandboxProfile(cwd, false, [extraRoot]), new RegExp(escapeRegExp(extraRoot)))

  const expectedProfiles = ['read-only', 'build-test', 'package-install', 'browser', 'network-fetch', 'connector-send', 'system-control', 'untrusted-code'] as const
  assert.deepEqual(Object.keys(SANDBOX_PROFILES).sort(), [...expectedProfiles].sort())
  for (const id of expectedProfiles) {
    const sbpl = macSandboxProfile(cwd, id)
    assert.match(sbpl, /deny default/, `${id} denies by default`)
    assert.match(sbpl, new RegExp(escapeRegExp(cwd)), `${id} can read cwd`)
    if (SANDBOX_PROFILES[id].allowNetwork) assert.match(sbpl, /allow network\*/, `${id} allows network`)
    else assert.doesNotMatch(sbpl, /allow network\*/, `${id} denies network`)
    const writeSection = sbpl.slice(sbpl.indexOf('(allow file-write*'))
    if (SANDBOX_PROFILES[id].allowCwdWrite) assert.match(writeSection, new RegExp(escapeRegExp(cwd)), `${id} can write cwd`)
    else assert.doesNotMatch(writeSection, new RegExp(escapeRegExp(cwd)), `${id} cannot write cwd`)
    assert.equal(SANDBOX_PROFILES[id].envSecretsAllowed, false)
  }

  assert.equal(sandboxProfileForCommand('cat package.json'), 'read-only')
  // a compound command with a destructive tail must NOT be classified read-only
  assert.notEqual(sandboxProfileForCommand('cat foo.txt; rm -rf bar'), 'read-only')
  assert.notEqual(sandboxProfileForCommand('ls && curl http://evil'), 'read-only')
  assert.notEqual(sandboxProfileForCommand('cat $(rm -rf bar)'), 'read-only')
  assert.equal(sandboxProfileForCommand('npm test'), 'build-test')
  assert.equal(sandboxProfileForCommand('npm run build'), 'build-test')
  assert.equal(sandboxProfileForCommand('npm install left-pad'), 'package-install')
  assert.equal(sandboxProfileForCommand('curl https://example.com/feed.json'), 'network-fetch')
  assert.equal(sandboxProfileForCommand('sudo launchctl kickstart system/com.example.app'), 'system-control')
  assert.equal(sandboxProfileForCommand('node ./script.js'), 'build-test')
  assert.equal(taskSandboxNetworkOverride('build-test', true), undefined, 'global toggle cannot open network for build/test')
  assert.equal(taskSandboxNetworkOverride('untrusted-code', true), undefined, 'global toggle cannot open network for opaque code')
  assert.equal(taskSandboxNetworkOverride('network-fetch', false), false, 'explicit network commands honor the off toggle')
  assert.equal(taskSandboxNetworkOverride('network-fetch', true), true)
  assert.equal(taskSandboxNetworkOverride('package-install', false), true, 'approved package installs keep their isolated network path')
  // `find` with destructive flags must NOT be classified read-only — when sandbox-exec is
  // unavailable and VONTRA_ALLOW_UNSANDBOXED_COMMANDS=1 is set, a read-only classification
  // would route these straight through runShell with no seatbelt at all.
  assert.notEqual(sandboxProfileForCommand('find ~ -delete'), 'read-only')
  assert.notEqual(sandboxProfileForCommand('find . -name "*.log" -delete'), 'read-only')
  assert.notEqual(sandboxProfileForCommand('find . -exec rm {} \\;'), 'read-only')
  // plain, non-destructive `find` usage should remain read-only
  assert.equal(sandboxProfileForCommand('find . -name "*.log"'), 'read-only')

  const previousSecretEnv = {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
    VONTRA_TOKEN: process.env.VONTRA_TOKEN,
  }
  try {
    process.env.OPENAI_API_KEY = 'sk-test-leak'
    process.env.AWS_SECRET_ACCESS_KEY = 'aws-secret-leak'
    process.env.VONTRA_TOKEN = 'vontra-secret-leak'
    const env = sandboxCommandEnv(cwd)
    assert.equal(env.HOME, cwd)
    assert.equal(env.SHELL, '/bin/zsh')
    assert.equal(env.OPENAI_API_KEY, undefined)
    assert.equal(env.AWS_SECRET_ACCESS_KEY, undefined)
    assert.equal(env.VONTRA_TOKEN, undefined)
    assert.ok(!Object.values(env).some((value) => /secret-leak|sk-test-leak/.test(String(value))))
  } finally {
    restoreEnv('OPENAI_API_KEY', previousSecretEnv.OPENAI_API_KEY)
    restoreEnv('AWS_SECRET_ACCESS_KEY', previousSecretEnv.AWS_SECRET_ACCESS_KEY)
    restoreEnv('VONTRA_TOKEN', previousSecretEnv.VONTRA_TOKEN)
  }

  await fs.writeFile(path.join(cwd, 'package.json'), '{"name":"artifact-test"}\n', 'utf8')
  await fs.mkdir(path.join(cwd, '.vontra-artifacts', 'screens'), { recursive: true })
  await fs.writeFile(path.join(cwd, '.vontra-artifacts', 'old.txt'), 'before\n', 'utf8')
  const baseline = captureSandboxArtifactBaseline(cwd)
  await fs.writeFile(path.join(cwd, '.vontra-artifacts', 'output.txt'), 'after\n', 'utf8')
  await fs.writeFile(path.join(cwd, '.vontra-artifacts', 'screens', 'shot.png'), 'fake image\n', 'utf8')
  await fs.writeFile(path.join(cwd, '.vontra-artifacts', 'old.txt'), 'changed\n', 'utf8')
  const bundle = buildSandboxArtifactBundle({
    cwd,
    command: 'npm test',
    result: { stdout: 'tests ok\n', stderr: '', profile: 'build-test', implementation: 'macos-sandbox-exec' },
    baseline,
    now: 1_000,
  })
  assert.equal(bundle.schema, 'vontra.sandbox-artifacts/1')
  assert.ok(bundle.id.startsWith('sbox-'))
  assert.ok(bundle.inputSnapshot.some((file) => file.path === 'package.json' && file.sha256), 'package manifest captured as input')
  assert.ok(bundle.outputFiles.some((file) => file.path === '.vontra-artifacts/output.txt'), 'output file captured')
  assert.ok(bundle.screenshots.some((file) => file.path === '.vontra-artifacts/screens/shot.png'), 'screenshot captured')
  assert.equal(bundle.logs.find((log) => log.kind === 'stdout')?.bytes, Buffer.byteLength('tests ok\n'))
  assert.ok(bundle.diffs.some((diff) => diff.path === '.vontra-artifacts/output.txt' && diff.status === 'added'), 'added artifact diff captured')
  assert.ok(bundle.diffs.some((diff) => diff.path === '.vontra-artifacts/old.txt' && diff.status === 'changed'), 'changed artifact diff captured')
  assert.match(bundle.patch ?? '', /diff --git a\/\.vontra-artifacts\/old.txt b\/\.vontra-artifacts\/old.txt/)
  assert.match(bundle.patch ?? '', /\+changed/)
  assert.match(bundle.patch ?? '', /diff --git a\/\.vontra-artifacts\/output.txt b\/\.vontra-artifacts\/output.txt/)
  assert.match(bundle.patch ?? '', /\+after/)
  const summary = summarizeSandboxArtifactBundle(bundle)
  assert.equal(summary.id, bundle.id)
  assert.deepEqual(summary.screenshots, ['.vontra-artifacts/screens/shot.png'])
  assert.equal(summary.patch, true)
  assert.equal(summary.logBytes, Buffer.byteLength('tests ok\n'))
  assert.ok(summary.diffs.some((diff) => diff.path === '.vontra-artifacts/old.txt' && diff.status === 'changed'))
  assert.ok(!JSON.stringify(summary).includes('tests ok'), 'summary carries no log contents')

  const packageHost = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-package-host-'))
  const packageJson = '{"name":"package-host","version":"1.0.0"}\n'
  await fs.writeFile(path.join(packageHost, 'package.json'), packageJson, 'utf8')
  const packageInstall = await runSandboxedCommand('npm install --package-lock-only --ignore-scripts --offline left-pad', {
    cwd: packageHost,
    profile: 'package-install',
    allowNetwork: false,
    timeoutMs: 20_000,
  })
  assert.equal(packageInstall.profile, 'package-install')
  assert.ok(packageInstall.artifacts, 'package-install run returns artifact bundle')
  assert.notEqual(packageInstall.artifacts!.cwd, realpathSync(packageHost), 'package install runs away from the host project')
  assert.ok(packageInstall.artifacts!.inputSnapshot.some((file) => file.path === 'package.json'), 'sandbox copy includes package manifest input')
  assert.equal(await fs.readFile(path.join(packageHost, 'package.json'), 'utf8'), packageJson)
  await assert.rejects(() => fs.access(path.join(packageHost, 'node_modules')))
  await assert.rejects(() => fs.access(path.join(packageHost, 'package-lock.json')))
  // the throwaway sandbox cwd must be cleaned up by runSandboxedCommand, not orphaned in TMPDIR
  await assert.rejects(() => fs.access(packageInstall.artifacts!.cwd), 'package-install sandbox temp dir is removed after the run')
  await fs.rm(packageHost, { recursive: true, force: true })

  if (canUseMacSandbox()) {
    const ok = await runSandboxedCommand(`printf ok > ${shellQuote(inside)} && cat ${shellQuote(inside)}`, { cwd })
    assert.equal(ok.code, 0, ok.stderr)
    assert.equal(ok.stdout.trim(), 'ok')
    assert.equal(ok.sandboxed, true)
    assert.equal(ok.profile, 'build-test')
    assert.equal(ok.implementation, 'macos-sandbox-exec')

    const denied = await runSandboxedCommand(`printf nope > ${shellQuote(outside)}`, { cwd, profile: 'build-test' })
    assert.notEqual(denied.code, 0, 'expected home-directory write to be denied by sandbox')
    await assert.rejects(() => fs.access(outside))

    const readOnlyDenied = await runSandboxedCommand(`printf nope > ${shellQuote(inside)}`, { cwd, profile: 'read-only' })
    assert.notEqual(readOnlyDenied.code, 0, 'expected workspace write to be denied by read-only sandbox')
  } else {
    const unavailable = await runSandboxedCommand('printf nope', { cwd })
    assert.equal(unavailable.code, 126)
    assert.equal(unavailable.sandboxed, false)
    assert.equal(unavailable.profile, 'build-test')
    assert.equal(unavailable.implementation, 'unsandboxed')
    assert.match(unavailable.stderr, /No supported command sandbox is available/)
  }

  // maxBuffer overflow with a SIGTERM-ignoring child: the overflow path must clear the
  // original timeout timer and escalate to SIGKILL after a grace period, the same way the
  // normal timeout path does — not just send one SIGTERM and leave the child to linger until
  // the unrelated original timer eventually fires (here set far longer than a prompt kill).
  const trapScript = path.join(cwd, 'trap-sigterm.js')
  await fs.writeFile(trapScript, [
    "process.on('SIGTERM', () => {})",
    "const chunk = 'x'.repeat(65536)",
    "const timer = setInterval(() => { process.stdout.write(chunk) }, 20)",
    "setTimeout(() => { clearInterval(timer); process.exit(0) }, 30000)",
    `require('fs').writeFileSync(${JSON.stringify(path.join(cwd, 'trap.pid'))}, String(process.pid))`,
  ].join('\n'), 'utf8')
  const previousUnsandboxed = process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS
  // Non-macOS CI has no sandbox-exec. Use the existing explicit development escape hatch only
  // for this process-lifecycle test; production Full Auto has no bypass option at all.
  if (!canUseMacSandbox()) process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS = '1'
  const overflowStart = Date.now()
  const overflowResult = await runSandboxedCommand(`node ${shellQuote(trapScript)}`, {
    cwd,
    profile: 'build-test',
    maxBuffer: 32 * 1024,
    timeoutMs: 120_000,
  })
  restoreEnv('VONTRA_ALLOW_UNSANDBOXED_COMMANDS', previousUnsandboxed)
  const overflowElapsed = Date.now() - overflowStart
  const trapPid = Number((await fs.readFile(path.join(cwd, 'trap.pid'), 'utf8')).trim())
  // give the OS a moment to reap the killed process before checking liveness
  await new Promise((resolve) => setTimeout(resolve, 200))
  let trapStillAlive = true
  try { process.kill(trapPid, 0); } catch { trapStillAlive = false }
  assert.equal(trapStillAlive, false, 'SIGTERM-ignoring child must be SIGKILLed after maxBuffer overflow, not left running')
  assert.ok(overflowElapsed < 10_000, `expected prompt SIGKILL escalation on maxBuffer overflow, took ${overflowElapsed}ms (unrelated timeoutMs is 120000ms)`)
  assert.ok(overflowResult.stdout.length > 0, 'overflow result should still include the buffered stdout')

  if (!canUseMacSandbox()) process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS = '1'
  const cancel = new AbortController()
  const cancelStarted = Date.now()
  const cancelledRun = runSandboxedCommand('sleep 30', {
    cwd,
    profile: 'build-test',
    timeoutMs: 120_000,
    signal: cancel.signal,
  })
  setTimeout(() => cancel.abort(), 25)
  const cancelled = await cancelledRun
  assert.equal(cancelled.code, 130, cancelled.stderr)
  assert.match(cancelled.stderr, /Cancelled by the user/)
  assert.ok(Date.now() - cancelStarted < 5_000, 'abort stops the in-flight shell promptly')
  restoreEnv('VONTRA_ALLOW_UNSANDBOXED_COMMANDS', previousUnsandboxed)

  const mutationRepo = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-sandbox-mutations-'))
  assert.equal(spawnSync('git', ['init', '-q'], { cwd: mutationRepo }).status, 0)
  await fs.writeFile(path.join(mutationRepo, 'source.ts'), 'before\n')
  assert.equal(spawnSync('git', ['add', 'source.ts'], { cwd: mutationRepo }).status, 0)
  const mutationUnsandboxed = process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS
  if (!canUseMacSandbox()) process.env.VONTRA_ALLOW_UNSANDBOXED_COMMANDS = '1'
  const failedMutation = await runSandboxedCommand("printf 'after\\n' > source.ts; exit 7", {
    cwd: mutationRepo, profile: 'build-test', timeoutMs: 10_000,
  })
  assert.equal(failedMutation.code, 7)
  assert.deepEqual(failedMutation.changedPaths, ['source.ts'], 'a failing shell still owns its source-file mutation')

  const partial = new AbortController()
  const partialRun = runSandboxedCommand("printf 'partial\\n' > cancelled.ts; sleep 30", {
    cwd: mutationRepo, profile: 'build-test', timeoutMs: 120_000, signal: partial.signal,
  })
  setTimeout(() => partial.abort(), 100)
  const partialResult = await partialRun
  assert.equal(partialResult.code, 130)
  assert.ok(partialResult.changedPaths?.includes('cancelled.ts'), 'a cancelled shell retains partial mutation ownership')
  restoreEnv('VONTRA_ALLOW_UNSANDBOXED_COMMANDS', mutationUnsandboxed)
  await fs.rm(mutationRepo, { recursive: true, force: true })

  await fs.rm(cwd, { recursive: true, force: true })
  await fs.rm(extraRoot, { recursive: true, force: true })
  await fs.rm(outside, { force: true }).catch(() => {})
  console.log('sandbox tests passed')
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function restoreEnv(key: string, value: string | undefined) {
  if (value === undefined) delete process.env[key]
  else process.env[key] = value
}

void main()
