import assert from 'node:assert/strict'
import type { PackagedPluginCapability } from './types'
import { executableEvidence, formatMcpArguments, matchesPluginSearch, parseMcpArguments, pluginRegistrationEnabled } from './plugin-library'

assert.ok(matchesPluginSearch('  PDF read ', 'PDF', 'Read files'))
assert.ok(!matchesPluginSearch('PDF read', 'PDF', 'Create files'))
assert.ok(matchesPluginSearch('', undefined, null))
assert.equal(executableEvidence({ signatureVerifiedAt: 1 }), 'Signature recorded', 'a stored timestamp cannot prove current signature verification')
assert.equal(executableEvidence({ checksumSha256: 'a'.repeat(64) }), 'Checksum recorded')
assert.equal(executableEvidence({}), 'No integrity record')
const plugin = { id: 'example', name: 'Example', manifestPath: '/example/plugin.json' } as PackagedPluginCapability
assert.equal(pluginRegistrationEnabled(plugin), false, 'missing enable is not loaded by main')
assert.equal(pluginRegistrationEnabled({ ...plugin, enabled: true, trusted: false }), false)
assert.equal(pluginRegistrationEnabled({ ...plugin, enabled: true, trusted: true }), true)
assert.deepEqual(parseMcpArguments('--path "folder with spaces" --empty ""'), ['--path', 'folder with spaces', '--empty', ''])
assert.deepEqual(parseMcpArguments("--label 'text with spaces' C:\\Projects"), ['--label', 'text with spaces', 'C:\\Projects'])
assert.throws(() => parseMcpArguments('--path "unfinished'), /Close the quote/)
for (const args of [[], ['plain', '--flag'], ['', 'path with spaces', 'quote"value', "apostrophe's", 'C:\\Projects\\example', 'line\nbreak'], ['trailing\\']]) {
  assert.deepEqual(parseMcpArguments(formatMcpArguments(args)), args, 'editing a server must not change argument boundaries')
}
console.log('Plugin search, registration truth, evidence labels and MCP argument preservation checks passed.')
