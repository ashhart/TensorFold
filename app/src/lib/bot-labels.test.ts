// Run: node src/lib/bot-labels.test.ts
import assert from 'node:assert/strict'
import { botVmSupported, localExecutionSource, threadTrustSource } from './bot-labels'

assert.equal(localExecutionSource('darwin'), 'This Mac', 'macOS keeps its established wording')
assert.equal(localExecutionSource('linux'), 'Local', 'a Linux desktop is not a Mac')
assert.equal(localExecutionSource('win32'), 'Local')
assert.equal(localExecutionSource('web-remote'), 'Local')
assert.equal(botVmSupported('darwin'), true)
assert.equal(botVmSupported('linux'), false, 'isolated bot VMs are macOS-only')

assert.equal(threadTrustSource('darwin'), 'On this Mac')
assert.equal(threadTrustSource('linux'), 'Local', 'a Linux desktop is not a Mac')

console.log('bot label tests passed')
