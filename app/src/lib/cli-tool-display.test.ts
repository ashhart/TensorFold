// Run: npx jiti src/lib/cli-tool-display.test.ts
import assert from 'node:assert/strict'
import { cliToolState, cliToolStatusText, cliVersionLabel } from './cli-tool-display'

assert.equal(cliVersionLabel('2.1.280 (Claude Code)'), 'v2.1.280')
assert.equal(cliVersionLabel('codex-cli 0.142.5'), 'v0.142.5')
assert.equal(cliVersionLabel('v1.14.20'), 'v1.14.20')
assert.equal(cliVersionLabel('1.0.0-beta.2'), 'v1.0.0-beta.2')
assert.equal(cliVersionLabel('/Users/x/.local/bin/hermes: line 4: no such file'), null)
assert.equal(cliVersionLabel(null), null)

const tool = { command: 'hermes', installed: true, version: null, problem: null }
assert.equal(cliToolState({ ...tool, command: null }), 'installed')
assert.equal(cliToolState({ ...tool, installed: false }), 'missing')
// Older hosts send no state: installed alone decides.
assert.equal(cliToolState(tool), 'installed')
assert.equal(cliToolState({ ...tool, state: 'broken' }), 'broken')

assert.deepEqual(cliToolStatusText({ ...tool, state: 'installed', version: '2.1.280 (Claude Code)' }), { text: 'v2.1.280', title: '2.1.280 (Claude Code)' })
assert.deepEqual(cliToolStatusText({ ...tool, state: 'installed' }), { text: 'Installed', title: 'Installed' })
assert.equal(cliToolStatusText({ ...tool, installed: false, state: 'missing' }).text, 'Missing')
assert.deepEqual(cliToolStatusText({ ...tool, state: 'broken', problem: '~/.local/bin/hermes: line 4: no python' }), { text: 'Broken', title: '~/.local/bin/hermes: line 4: no python' })
assert.equal(cliToolStatusText({ ...tool, command: null }).text, 'Built in')

console.log('cli-tool-display: ok')
