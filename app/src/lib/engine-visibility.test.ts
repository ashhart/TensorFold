// Run: node src/lib/engine-visibility.test.ts
import assert from 'node:assert/strict'
import { engineManagementVisible } from './engine-visibility'

// macOS always manages the bundled engine, even while it reports unavailable (it can be repaired).
assert.equal(engineManagementVisible('darwin', { available: false } as any), true)
assert.equal(engineManagementVisible('darwin', null), true)
// Linux builds ship no engine, so the Models workspace must not appear.
assert.equal(engineManagementVisible('linux', { available: false } as any), false)
assert.equal(engineManagementVisible('linux', null), false)
// A Linux user who supplies their own engine binary gets the workspace back.
assert.equal(engineManagementVisible('linux', { available: true } as any), true)
assert.equal(engineManagementVisible('web-remote', { available: false } as any), false)

console.log('engine management visibility tests passed')
