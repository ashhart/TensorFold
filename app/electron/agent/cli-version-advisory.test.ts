// Run: jiti electron/agent/cli-version-advisory.test.ts
import assert from 'node:assert/strict'
import { parseSemver, compareSemver, formatSemver, versionAdvisory } from './cli-version-advisory'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-version-advisory tests:')

// --- parseSemver over the real CLI --version shapes ---
{
  assert.deepEqual(parseSemver('0.142.5'), { major: 0, minor: 142, patch: 5, prerelease: '' })
  assert.deepEqual(parseSemver('codex-cli 0.142.5'), { major: 0, minor: 142, patch: 5, prerelease: '' })
  assert.deepEqual(parseSemver('2.1.170 (Claude Code)'), { major: 2, minor: 1, patch: 170, prerelease: '' })
  assert.deepEqual(parseSemver('v1.14.20'), { major: 1, minor: 14, patch: 20, prerelease: '' })
  assert.deepEqual(parseSemver('0.80.3'), { major: 0, minor: 80, patch: 3, prerelease: '' })
  assert.deepEqual(parseSemver('1.2.3-beta.4'), { major: 1, minor: 2, patch: 3, prerelease: 'beta.4' })
  assert.deepEqual(parseSemver('1.2'), { major: 1, minor: 2, patch: 0, prerelease: '' }, 'missing patch defaults to 0')
  assert.equal(parseSemver('not a version'), null)
  assert.equal(parseSemver(''), null)
  assert.equal(parseSemver(null), null)
  ok('parseSemver: extracts x.y.z(-pre) from real CLI version strings')
}

// --- compareSemver ordering, incl. prerelease precedence ---
{
  const s = (v: string) => parseSemver(v)!
  assert.equal(compareSemver(s('1.2.3'), s('1.2.4')), -1)
  assert.equal(compareSemver(s('1.3.0'), s('1.2.9')), 1)
  assert.equal(compareSemver(s('2.0.0'), s('1.9.9')), 1)
  assert.equal(compareSemver(s('1.2.3'), s('1.2.3')), 0)
  assert.equal(compareSemver(s('1.2.3-rc.1'), s('1.2.3')), -1, 'a prerelease sorts before the release')
  assert.equal(compareSemver(s('1.2.3-rc.2'), s('1.2.3-rc.10')), -1, 'numeric prerelease identifiers compare numerically')
  assert.equal(compareSemver(s('1.2.3-alpha'), s('1.2.3-beta')), -1, 'alphanumeric prerelease compares lexically')
  assert.equal(compareSemver(null, s('1.0.0')), -1, 'unknown sorts smallest')
  assert.equal(compareSemver(s('1.0.0'), null), 1)
  assert.equal(formatSemver(s('1.2.3-beta.4')), '1.2.3-beta.4')
  ok('compareSemver: numeric + prerelease precedence, nulls last')
}

// --- versionAdvisory decision matrix ---
{
  const outdated = versionAdvisory({ name: 'Codex', installedVersion: 'codex-cli 0.142.5', latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', installed: true })
  assert.equal(outdated.status, 'update_available')
  assert.equal(outdated.canUpdate, true)
  assert.equal(outdated.currentVersion, '0.142.5')
  assert.equal(outdated.latestVersion, '0.150.0')
  assert.ok(outdated.message.includes('npm i -g @openai/codex'), 'the update command is surfaced')
  ok('versionAdvisory: installed < latest → update_available with command')

  const current = versionAdvisory({ name: 'Claude Code', installedVersion: '2.1.170 (Claude Code)', latestVersion: '2.1.170', installed: true })
  assert.equal(current.status, 'current')
  assert.equal(current.canUpdate, false)
  ok('versionAdvisory: installed == latest → current')

  const ahead = versionAdvisory({ name: 'Pi', installedVersion: '0.81.0-dev', latestVersion: '0.80.3', installed: true })
  assert.equal(ahead.status, 'ahead', 'a dev build ahead of latest is informational, not an update')
  assert.equal(ahead.canUpdate, false)
  ok('versionAdvisory: installed > latest → ahead (dev build)')

  const noLatest = versionAdvisory({ name: 'Grok', installedVersion: '1.0.0', latestVersion: null, installed: true })
  assert.equal(noLatest.status, 'unknown')
  assert.equal(noLatest.canUpdate, false)
  assert.ok(noLatest.message.includes('not checked yet'))
  ok('versionAdvisory: latest not known → unknown, never a false "current"')

  const notInstalled = versionAdvisory({ name: 'Kilo', installedVersion: null, latestVersion: '1.0.0', installed: false })
  assert.equal(notInstalled.status, 'unknown')
  assert.ok(notInstalled.message.includes('not installed'))
  ok('versionAdvisory: not installed → unknown (advisory not applicable)')

  const canUpdateNoCmd = versionAdvisory({ name: 'Gemini', installedVersion: '1.0.0', latestVersion: '2.0.0', installed: true })
  assert.equal(canUpdateNoCmd.status, 'update_available')
  assert.equal(canUpdateNoCmd.canUpdate, false, 'update available but no command → cannot auto-update')
  ok('versionAdvisory: update available without a command → canUpdate false')
}

console.log(`\n${passed} passed`)
