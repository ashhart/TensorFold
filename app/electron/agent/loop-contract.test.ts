import assert from 'node:assert/strict'
import { buildRunContract } from './loop-contract'

// solo: from goal + criteria + verify
const solo = buildRunContract({ source: 'solo', mode: 'build', goal: 'reverse stdin CLI', criteria: ['reads stdin', 'reverses'], verify: ['npm test'] })
assert.equal(solo.source, 'solo')
assert.equal(solo.autonomy, 'medium') // default
assert.equal(solo.maxCycles, 5) // build default
assert.deepEqual(solo.verificationCommands, ['npm test'])
assert.ok(solo.acceptanceCriteria.length >= 2)
assert.ok(solo.forbiddenShortcuts.some((s) => /stub|fake|disable/i.test(s))) // injected defaults

// ship: deeper defaults
const ship = buildRunContract({ source: 'solo', mode: 'ship', goal: 'settings screen' })
assert.equal(ship.maxCycles, 7)

// ticket: sources acceptanceCriteria/requiredEvidence from the ticket
const ticket = buildRunContract({ source: 'ticket', mode: 'build', goal: 'add rate limit', ticket: { acceptanceCriteria: ['429 past 100/min'], requiredEvidence: ['test'], allowedCapabilities: ['read', 'write', 'shell'], risk: 'medium' } })
assert.deepEqual(ticket.acceptanceCriteria, ['429 past 100/min'])
assert.deepEqual(ticket.requiredEvidence, ['test'])

// autonomy defaults to medium
assert.equal(ticket.autonomy, 'medium')

// maxCycles hard-capped
const capped = buildRunContract({ source: 'solo', mode: 'build', goal: 'x', maxCycles: 99 })
assert.equal(capped.maxCycles, 12)

console.log('loop-contract tests passed')
