// Run: jiti electron/agent/cli-status.test.ts
import assert from 'node:assert/strict'
import { nextStatus } from './cli-status.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-status tests:')

{
  // Adapted from Clyde's Session.handleOutput: bell => attention, any output => busy, exit => done.
  assert.equal(nextStatus('idle', new Uint8Array([104, 105])), 'busy', 'output makes it busy')
  assert.equal(nextStatus('busy', new Uint8Array([0x07])), 'attention', 'bell (0x07) rings attention')
  assert.equal(nextStatus('attention', new Uint8Array([104])), 'attention', 'stays attention until cleared')
  assert.equal(nextStatus('done', new Uint8Array([104])), 'done', 'a dead session never goes busy again')
  ok('nextStatus: bell -> attention, output -> busy, done is terminal')
}

console.log(`\n${passed} passed`)
