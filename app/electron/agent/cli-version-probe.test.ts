// Run: jiti electron/agent/cli-version-probe.test.ts
import assert from 'node:assert/strict'
import { classifyVersionProbe, shortProblem } from './cli-version-probe.ts'
import { builtinClis, cliToolStatuses } from './cli-registry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-version-probe tests:')

const HOME = '/Users/someone'

{
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: '2.1.280 (Claude Code)\n', stderr: '' }, HOME), { version: '2.1.280 (Claude Code)', problem: null })
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: 'codex-cli 0.142.5\n', stderr: '' }, HOME), { version: 'codex-cli 0.142.5', problem: null })
  // Banner lines before the version are skipped.
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: 'Welcome\nv1.14.20\n', stderr: '' }, HOME), { version: 'v1.14.20', problem: null })
  ok('a clean exit with a version on stdout is a version')
}

{
  // A wrapper script whose venv is gone: bash exits 127 with the error on stderr.
  const run = { code: 127, stdout: '', stderr: `${HOME}/.local/bin/hermes: line 4: ${HOME}/.hermes/venv/bin/python: No such file or directory\n` }
  assert.deepEqual(classifyVersionProbe(run, HOME), { version: null, problem: '~/.local/bin/hermes: line 4: ~/.hermes/venv/bin/python: No such file or directory' })
  ok('a non-zero exit is broken, with the first stderr line and ~ for home')
}

{
  assert.deepEqual(classifyVersionProbe({ code: 2, stdout: '', stderr: '' }, HOME), { version: null, problem: 'Exited with code 2' })
  assert.deepEqual(classifyVersionProbe({ code: null, signal: 'SIGKILL', stdout: '', stderr: '' }, HOME), { version: null, problem: 'Stopped by SIGKILL' })
  // Exit 1 that printed a version is still broken, never a version.
  assert.equal(classifyVersionProbe({ code: 1, stdout: '1.2.3\n', stderr: '' }, HOME).version, null)
  ok('non-zero exits and signals are broken even when stdout looks fine')
}

{
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: '0.9.1\n', stderr: '(node:1) DeprecationWarning: punycode\n' }, HOME), { version: '0.9.1', problem: null })
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: '', stderr: 'error: config at /Users/someone/.tool is invalid\n' }, HOME), { version: null, problem: 'error: config at ~/.tool is invalid' })
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: '', stderr: '' }, HOME), { version: null, problem: 'Printed no version' })
  assert.deepEqual(classifyVersionProbe({ code: 0, stdout: 'Usage: tool [options]\n', stderr: '' }, HOME), { version: null, problem: 'No version in "Usage: tool [options]"' })
  ok('a warning beside a version is fine; no version on stdout is broken, with stderr as the detail')
}

{
  assert.deepEqual(classifyVersionProbe({ code: null, stdout: '', stderr: '', timedOut: true }, HOME), { version: null, problem: 'No answer to --version' })
  assert.deepEqual(classifyVersionProbe({ code: null, stdout: '', stderr: '', error: `spawn ${HOME}/bin/tool EACCES` }, HOME), { version: null, problem: 'spawn ~/bin/tool EACCES' })
  ok('timeouts and spawn errors are broken')
}

{
  const long = shortProblem(`${'x'.repeat(300)}\nsecond line`, HOME)
  assert.equal(long.length, 120)
  assert.ok(long.endsWith('…'))
  assert.equal(shortProblem('  a\t b  \n', HOME), 'a b')
  // A bare "/" home never rewrites every slash.
  assert.equal(shortProblem('/usr/bin/env: python: No such file', '/'), '/usr/bin/env: python: No such file')
  ok('details are one short line')
}

{
  const exists = (p: string) => p === '/usr/local/bin/hermes' || p === '/usr/local/bin/claude'
  const statuses = cliToolStatuses(builtinClis(), [], {
    dirs: ['/usr/local/bin'],
    exists,
    versions: { claude: '2.1.280 (Claude Code)', hermes: 'should be ignored' },
    problems: { hermes: '~/.local/bin/hermes: line 4: no python', gemini: 'not installed, ignored' },
  })
  const by = (id: string) => statuses.find((tool) => tool.id === id)!
  assert.equal(by('claude').state, 'installed')
  assert.equal(by('claude').version, '2.1.280 (Claude Code)')
  assert.equal(by('hermes').state, 'broken')
  assert.equal(by('hermes').version, null, 'a broken CLI never reports a version')
  assert.equal(by('hermes').problem, '~/.local/bin/hermes: line 4: no python')
  assert.equal(by('gemini').state, 'missing')
  assert.equal(by('gemini').problem, null, 'a missing binary has no probe problem')
  assert.equal(by('shell').state, 'installed')
  ok('cliToolStatuses: installed, missing and broken states')
}

console.log(`\n${passed} checks passed.`)
