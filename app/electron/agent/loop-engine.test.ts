import assert from 'node:assert/strict'
import { runLoop, type LoopRuntime } from './loop-engine'
import { buildRunContract } from './loop-contract'
import type { Verdict } from './loop-types'

function runtime(overrides: Partial<LoopRuntime>): LoopRuntime {
  return {
    runPhase: async (phase) => ({ phase, ok: true, toolEvents: 3, toolAttempts: 3, summary: `${phase} ok`, evidenceRefs: [] }),
    runVerification: async contract => ({
      checks: contract.verificationCommands.map(command => ({ command, outcome: 'passed', exitCode: 0, output: '3 passed' })),
      cancelled: false,
    }),
    critique: async () => ({ passed: true, confidence: 0.9, summary: 'ok', issues: [] }),
    checkpoint: async () => 'cp1',
    rollback: async () => {},
    log: () => {},
    ...overrides,
  }
}

async function main() {
  const contract = buildRunContract({ source: 'solo', mode: 'build', goal: 'x', verify: ['npm test'] })

  // happy path: passes on cycle 0
  const r1 = await runLoop(contract, ['dev'], runtime({}))
  assert.equal(r1.status, 'shipped')
  assert.equal(r1.cycles, 1)

  // fail then pass: critic fails cycle 0 (fixable), passes cycle 1
  let n = 0
  const r2 = await runLoop(contract, ['dev'], runtime({
    critique: async (): Promise<Verdict> => (n++ === 0 ? { passed: false, confidence: 0.9, summary: 'no', issues: ['add null check'] } : { passed: true, confidence: 0.9, summary: 'ok', issues: [] }),
  }))
  assert.equal(r2.status, 'shipped')
  assert.equal(r2.cycles, 2)
  assert.ok(r2.checkpoints >= 1)

  // user blocker -> pause
  const r3 = await runLoop(contract, ['dev'], runtime({
    critique: async (): Promise<Verdict> => ({ passed: false, confidence: 0.9, summary: 'no', issues: ['Missing STRIPE_API_KEY'] }),
  }))
  assert.equal(r3.status, 'blocked')
  assert.equal(r3.blockers[0].kind, 'NEEDS_USER_SECRET')

  // cap: never passes -> capped at maxCycles
  const r4 = await runLoop({ ...contract, maxCycles: 3 }, ['dev'], runtime({
    critique: async (): Promise<Verdict> => ({ passed: false, confidence: 0.9, summary: 'no', issues: ['still failing'] }),
  }))
  assert.equal(r4.status, 'capped')
  assert.equal(r4.cycles, 3)

  // Missing implementation evidence remains fixable but prevents completion.
  let textOnlyVerifications = 0
  const r5 = await runLoop(contract, ['dev'], runtime({
    runPhase: async (phase) => ({ phase, ok: true, toolEvents: 0, toolAttempts: 0, summary: 'claimed done', evidenceRefs: [] }),
    runVerification: async c => { textOnlyVerifications++; return runtime({}).runVerification(c) },
  }))
  assert.equal(r5.status, 'capped')
  assert.equal(textOnlyVerifications, contract.maxCycles, 'every text-only cycle still reaches verification instead of hard-failing')
  assert.match(r5.lastVerdict?.issues.join('\n') ?? '', /text-only output twice/)

  // A later cycle can recover; critic optimism cannot override missing execution evidence.
  let recoveryCalls = 0
  const r5Recovered = await runLoop({ ...contract, maxCycles: 3 }, ['dev'], runtime({
    runPhase: async (phase) => {
      recoveryCalls++
      return recoveryCalls <= 2
        ? { phase, ok: true, toolEvents: 0, toolAttempts: 0, summary: 'still planning', evidenceRefs: [] }
        : { phase, ok: true, toolEvents: 1, toolAttempts: 1, summary: 'implemented with evidence', evidenceRefs: ['patch'] }
    },
  }))
  assert.equal(r5Recovered.status, 'shipped')
  assert.equal(r5Recovered.cycles, 2, 'a later implementation cycle recovers normally')
  assert.equal(r5Recovered.checkpoints, 1, 'the incomplete first cycle is checkpointed/retried, not terminally failed')

  // A genuine user blocker reported alongside missing evidence remains an honest blocked result.
  const r5Blocked = await runLoop(contract, ['dev'], runtime({
    runPhase: async (phase) => ({ phase, ok: true, toolEvents: 0, toolAttempts: 0, summary: 'needs access', evidenceRefs: [] }),
    critique: async () => ({ passed: false, confidence: 0.9, summary: 'credential required', issues: ['Missing STRIPE_API_KEY'] }),
  }))
  assert.equal(r5Blocked.status, 'blocked')
  assert.ok(r5Blocked.blockers.some((blocker) => blocker.kind === 'NEEDS_USER_SECRET'))

  // tool-evidence guard is LENIENT for planning roles (architect/product) — a text-only plan is valid
  const r6 = await runLoop(contract, ['architect'], runtime({
    runPhase: async (phase) => ({ phase, ok: false, toolEvents: 0, toolAttempts: 0, summary: 'plan only', evidenceRefs: [] }),
  }))
  assert.equal(r6.status, 'shipped')

  // Failed attempts retain evidence but cannot satisfy an implementation phase.
  const r7 = await runLoop(contract, ['dev'], runtime({
    runPhase: async (phase) => ({ phase, ok: false, toolEvents: 0, toolAttempts: 2, summary: 'build failed', evidenceRefs: [] }),
  }))
  assert.equal(r7.status, 'capped')
  assert.match(r7.lastVerdict?.issues.join('\n') ?? '', /no successful implementation evidence/)

  // regression verdict -> rollback invoked
  let rolledBack = false
  await runLoop({ ...contract, maxCycles: 2 }, ['dev'], runtime({
    critique: async (): Promise<Verdict> => ({ passed: false, confidence: 0.9, summary: 'regression: broke existing tests', issues: ['regression'] }),
    rollback: async () => { rolledBack = true },
  }))
  assert.equal(rolledBack, true)

  console.log('loop-engine tests passed')
}
main().catch((e) => { console.error(e); process.exit(1) })
