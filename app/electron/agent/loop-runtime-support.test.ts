import assert from 'node:assert/strict'
import { roleSystemPrompt, phaseResultFromTimeline } from './loop-runtime-support'
import type { RunContract } from './loop-types'

const contract: RunContract = {
  source: 'solo', goal: 'reverse stdin CLI', deliverables: ['cli.ts'],
  acceptanceCriteria: ['reads stdin', 'reverses'], verificationCommands: ['npm test'],
  forbiddenShortcuts: ['Do not stub tests.'], requiredEvidence: ['test'], interruptPolicy: [],
  autonomy: 'medium', maxCycles: 5,
}

// role prompt embeds the role + contract goal + forbidden shortcuts
const dev = roleSystemPrompt('dev', contract)
assert.match(dev, /implement|develop|\bdev\b/i)
assert.match(dev, /reverse stdin CLI/)
assert.ok(dev.includes('Do not stub tests.'))
assert.match(dev, /reads stdin/)

// read-only roles are told not to edit
assert.match(roleSystemPrompt('architect', contract), /inspect|read-only|do not (edit|modify|write)/i)
assert.match(roleSystemPrompt('review', contract), /review/i)

// phaseResultFromTimeline counts tool events + extracts the final assistant summary
const items = [
  { kind: 'tool', tool: 'read_file', state: 'ok' },
  { kind: 'tool', tool: 'edit_file', state: 'ok' },
  { kind: 'reasoning', text: 'thinking' },
  { kind: 'assistant', text: 'Implemented reverse().' },
]
const pr = phaseResultFromTimeline('dev', items)
assert.equal(pr.phase, 'dev')
assert.equal(pr.toolEvents, 2)
assert.equal(pr.toolAttempts, 2)
assert.match(pr.summary, /Implemented reverse/)
assert.ok(pr.evidenceRefs.includes('read_file'))
assert.equal(pr.ok, true)

// no tool activity -> toolEvents 0 AND toolAttempts 0 (the "did nothing" case the guard fails)
const empty = phaseResultFromTimeline('dev', [{ kind: 'assistant', text: 'done' }])
assert.equal(empty.toolEvents, 0)
assert.equal(empty.toolAttempts, 0)
// errored tool calls don't count as useful evidence (toolEvents 0) but DO count as an
// attempt (toolAttempts 1) — so the evidence guard treats a failing build as real work.
const errored = phaseResultFromTimeline('dev', [{ kind: 'tool', tool: 'x', state: 'error' }])
assert.equal(errored.toolEvents, 0)
assert.equal(errored.toolAttempts, 1)

console.log('loop-runtime-support tests passed')
