import assert from 'node:assert/strict'
import { buildCriticPrompt, parseVerdict } from './loop-critic'
import type { RunContract } from './loop-types'

const contract: RunContract = { source: 'solo', goal: 'reverse stdin', deliverables: [], acceptanceCriteria: ['reverses input'], verificationCommands: ['npm test'], forbiddenShortcuts: ['no stubbed tests'], requiredEvidence: ['test'], interruptPolicy: [], autonomy: 'medium', maxCycles: 5 }

// prompt embeds contract + evidence + is skeptical/fail-closed
const p = buildCriticPrompt(contract, 'TEST OUTPUT: 3 passed', 'dev: wrote reverse()')
assert.match(p.system, /independent|skeptic|fail-closed/i)
assert.match(p.user, /reverse stdin/)
assert.match(p.user, /3 passed/)

// pass requires passed===true AND confidence>=0.6
assert.equal(parseVerdict('{"passed":true,"confidence":0.9,"summary":"ok","issues":[]}').passed, true)
assert.equal(parseVerdict('{"passed":true,"confidence":0.4,"summary":"meh","issues":[]}').passed, false) // low conf
assert.equal(parseVerdict('{"passed":false,"confidence":0.9,"summary":"no","issues":["tests not run"]}').passed, false)
// unparseable -> fail closed with an issue
const bad = parseVerdict('the work looks great to me!')
assert.equal(bad.passed, false)
assert.ok(bad.issues.length >= 1)
// JSON embedded in prose is still extracted
assert.equal(parseVerdict('Here: {"passed":true,"confidence":0.8,"summary":"ok","issues":[]} done').passed, true)
// a low-conf "pass" surfaces an issue (not empty)
assert.ok(parseVerdict('{"passed":true,"confidence":0.3,"summary":"x","issues":[]}').issues.length >= 1)

const contradictory = parseVerdict('{"passed":true,"confidence":0.99,"summary":"ok","issues":["tests failed"]}')
assert.equal(contradictory.passed, false)
assert.deepEqual(contradictory.issues, ['tests failed'], 'unresolved critic issues cannot disappear behind passed=true')
for (const confidence of [1.01, -1]) {
  assert.equal(parseVerdict(JSON.stringify({ passed: true, confidence, summary: 'ok', issues: [] })).passed, false)
}
assert.equal(parseVerdict('{"passed":true,"confidence":1e999,"summary":"ok","issues":[]}').passed, false)
assert.equal(parseVerdict('{"passed":true,"confidence":1,"summary":"ok"}').passed, false)
assert.equal(parseVerdict('{"passed":true,"confidence":1,"summary":"ok","issues":[123]}').passed, false)

console.log('loop-critic tests passed')
