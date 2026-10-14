import assert from 'node:assert/strict'
import type { Thread, Ticket } from '../../agent/types'
import { summarizeTicketWorkerOutcome } from './ticket-helpers'

// Minimal Ticket / Thread builders. We only populate fields read by
// summarizeTicketWorkerOutcome -> ticketWorkerSignals.
function ticket(): Ticket {
  return { id: 't1', title: 'Demo ticket' } as unknown as Ticket
}

function threadWith(command: string, output = ''): Thread {
  return {
    id: 'thread-1',
    timeline: [
      { id: 'i1', kind: 'tool', command, output, state: 'ok' } as any,
    ],
  } as unknown as Thread
}

let failures = 0
function check(name: string, fn: () => void) {
  try {
    fn()
    console.log(`ok - ${name}`)
  } catch (err) {
    failures += 1
    console.log(`FAIL - ${name}`)
    console.log(String(err))
  }
}

// Bug: an unanchored regex marked any command merely containing the substring
// "test" (latest, fastest, manifest, contest, ...) as a passing test run.
const falsePositives = [
  'git fetch --tags && checkout latest',
  'curl https://example.com/manifest.json',
  'echo fastest path wins',
  'node scripts/contest.js',
]
for (const cmd of falsePositives) {
  check(`non-test command is NOT treated as passing tests: "${cmd}"`, () => {
    const out = summarizeTicketWorkerOutcome(ticket(), threadWith(cmd), 'build', false, null)
    assert.equal(out.testsPassed, false, `expected testsPassed=false for "${cmd}"`)
    assert.ok(
      !out.evidence.some((e) => e.kind === 'test'),
      `expected no fabricated 'test' evidence for "${cmd}"`,
    )
  })
}

// Genuine test commands (word-boundary matches) still register as passing.
const truePositives = [
  'npm test',
  'npx vitest run',
  'npx tsc --noEmit',
  'npm run lint',
  'npx jest src/',
  'npx playwright test',
]
for (const cmd of truePositives) {
  check(`genuine test command IS treated as passing: "${cmd}"`, () => {
    const out = summarizeTicketWorkerOutcome(ticket(), threadWith(cmd), 'build', false, null)
    assert.equal(out.testsPassed, true, `expected testsPassed=true for "${cmd}"`)
  })
}

// A genuine test command that visibly failed must NOT be marked passing.
check('failing test command is not marked passing', () => {
  const out = summarizeTicketWorkerOutcome(
    ticket(),
    threadWith('npm test', 'exit 1\n2 failed'),
    'build',
    true,
    null,
  )
  assert.equal(out.testsPassed, false)
})

for (const [state, output] of [
  ['denied', 'Denied by user.'],
  ['awaiting', 'Awaiting approval.'],
  ['running', 'Still running.'],
  ['error', 'Cancelled by user.'],
] as const) {
  check(`${state} test command is not passing evidence`, () => {
    const thread = threadWith('npm test', output)
    ;(thread.timeline[0] as any).state = state
    const out = summarizeTicketWorkerOutcome(ticket(), thread, 'build', false, null)
    assert.equal(out.testsPassed, false)
    assert.ok(!out.evidence.some((item) => item.kind === 'test'))
    assert.ok(out.signals.blockers.length > 0)
  })
}

check('a successful retry of a previously denied test produces passing evidence', () => {
  const thread = threadWith('npm test', 'Denied by user.')
  ;(thread.timeline[0] as any).state = 'denied'
  thread.timeline.push({ id: 'i2', kind: 'tool', command: 'npm test', output: '12 passed', state: 'ok' } as any)
  const out = summarizeTicketWorkerOutcome(ticket(), thread, 'build', false, null)
  assert.equal(out.testsPassed, true)
  assert.ok(out.evidence.some((item) => item.kind === 'test' && /Test passed: npm test/.test(item.summary)))
})

if (failures > 0) {
  console.log(`\n${failures} test(s) failed`)
  process.exit(1)
}
console.log('\nall ticket-helpers tests passed')
