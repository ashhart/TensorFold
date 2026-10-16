import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const result = spawnSync(process.execPath, [fileURLToPath(new URL('./setup-notarization.mjs', import.meta.url))], { encoding: 'utf8', input: '' })
assert.notEqual(result.status, 0)
assert.match(result.stderr, /interactive local terminal/)
assert.doesNotMatch(result.stdout, /Apple ID|Password|credentials saved/i, 'Automated calls cannot start Apple credential prompts')
console.log('Notarization credential setup requires an interactive local terminal')
