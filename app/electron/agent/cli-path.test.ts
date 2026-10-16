import assert from 'node:assert/strict'
import { computeAugmentedPath, resolveCliBinary, knownBinDirs } from './cli-path'

// --- computeAugmentedPath: merge + de-dupe, first-seen order wins ---
assert.equal(computeAugmentedPath('/usr/bin:/bin', '/opt/homebrew/bin:/usr/bin'), '/usr/bin:/bin:/opt/homebrew/bin')
assert.equal(computeAugmentedPath('', '/a', null, '/a:/b'), '/a:/b')
assert.equal(computeAugmentedPath(undefined), '')

// A stripped GUI PATH gets homebrew + ~/.local/bin appended (so codex/claude become findable),
// while the original entries are preserved at the front.
const stripped = '/usr/bin:/bin:/usr/sbin:/sbin'
const merged = computeAugmentedPath(stripped, '', knownBinDirs('/Users/dev').join(':'))
assert.ok(merged.startsWith(stripped), 'existing PATH entries kept first')
assert.ok(merged.split(':').includes('/opt/homebrew/bin'), 'homebrew bin present')
assert.ok(merged.split(':').includes('/Users/dev/.local/bin'), '~/.local/bin present')

// --- resolveCliBinary ---
assert.equal(resolveCliBinary('/custom/codex', ['codex']), '/custom/codex', 'explicit configured path wins')
assert.equal(resolveCliBinary('  ', ['codex'], { dirs: ['/opt/homebrew/bin'], exists: (p) => p === '/opt/homebrew/bin/codex' }), '/opt/homebrew/bin/codex', 'probes known dirs')
assert.equal(resolveCliBinary(null, ['claude'], { dirs: ['/x'], exists: () => false }), 'claude', 'falls back to bare name when not found')
assert.equal(resolveCliBinary(null, ['nope', 'codex'], { dirs: ['/d'], exists: (p) => p === '/d/codex' }), '/d/codex', 'probes names in order')

console.log('cli-path tests passed')
