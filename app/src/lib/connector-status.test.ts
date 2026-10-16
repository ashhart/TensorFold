import assert from 'node:assert/strict'
import { plural, formatSyncSummary } from './connector-status.ts'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('connector-status tests:')

{
  assert.equal(plural('source', 1), 'source')
  assert.equal(plural('source', 0), 'sources')
  assert.equal(plural('message', 2), 'messages')
  ok('plural pluralizes everything except exactly one')
}

{
  // The core "nothing happened" fix: a successful sync that finds zero items must
  // read as a confirmed success ("0 messages"), not as nothing.
  assert.equal(formatSyncSummary(1, 0, 'message'), 'Scanned 1 source · 0 messages')
  assert.equal(formatSyncSummary(2, 5, 'message'), 'Scanned 2 sources · 5 messages')
  assert.equal(formatSyncSummary(1, 1, 'document'), 'Scanned 1 source · 1 document')
  assert.equal(formatSyncSummary(3, 12, 'item'), 'Scanned 3 sources · 12 items')
  ok('formatSyncSummary confirms success with explicit source + item counts')
}

{
  // Sync buttons are disabled with zero sources, but guard the message anyway.
  assert.equal(formatSyncSummary(0, 0, 'item'), 'No sources to scan')
  ok('formatSyncSummary handles the no-source edge without a bare "0"')
}

console.log(`\n${passed} checks passed.`)
