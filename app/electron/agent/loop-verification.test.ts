import assert from 'node:assert/strict'
import { runLoop, type LoopRuntime } from './loop-engine'
import { buildRunContract } from './loop-contract'
import { createLoopRuntime } from './loop-runtime'
import { verificationCheck, verificationConfigurationIssue, verificationIssues } from './loop-verification'
import type { VerificationReport } from './loop-types'

const contract = buildRunContract({ source: 'solo', mode: 'build', goal: 'verify', verify: ['test', 'typecheck'], maxCycles: 1 })
const green = (): VerificationReport => ({
  checks: contract.verificationCommands.map(command => ({ command, outcome: 'passed', exitCode: 0, output: 'ok' })),
  cancelled: false,
})

function runtime(report: VerificationReport): LoopRuntime {
  return {
    runPhase: async phase => ({ phase, ok: true, toolEvents: 1, toolAttempts: 1, summary: 'edited', evidenceRefs: [] }),
    runVerification: async () => report,
    critique: async () => ({ passed: true, confidence: 1, summary: 'ship it', issues: [] }),
    checkpoint: async () => 'fixture', rollback: async () => {}, log: () => {},
  }
}

async function main() {
  for (const outcome of ['failed', 'timed_out', 'error'] as const) {
    const report = green()
    report.checks[0] = { ...report.checks[0], outcome, exitCode: outcome === 'failed' ? 1 : null }
    const result = await runLoop(contract, ['dev'], runtime(report))
    assert.notEqual(result.status, 'shipped', outcome)
    assert.equal(result.lastVerdict?.passed, false)
  }
  const cancelled = green()
  cancelled.cancelled = true
  assert.equal((await runLoop(contract, ['dev'], runtime(cancelled))).status, 'stopped')
  for (const incomplete of [
    { ...green(), checks: green().checks.slice(0, 1) },
    { ...green(), checks: [...green().checks].reverse() },
    { ...green(), checks: [...green().checks, green().checks[0]] },
    { ...green(), checks: [{ ...green().checks[0], exitCode: 1 }, green().checks[1]] },
  ]) {
    assert.ok(verificationIssues(contract.verificationCommands, incomplete).length)
    assert.notEqual((await runLoop(contract, ['dev'], runtime(incomplete))).status, 'shipped')
  }
  assert.ok(verificationIssues(contract.verificationCommands, 'All passed' as any).length)
  assert.notEqual((await runLoop(contract, ['dev'], runtime('All passed' as any))).status, 'shipped')

  const absent = await runLoop({ ...contract, verificationCommands: [] }, ['dev'], {
    ...runtime(green()), runPhase: async () => { throw new Error('A blocked setup must not start roles.') },
  })
  assert.equal(absent.status, 'blocked')
  assert.equal(absent.cycles, 0)
  assert.equal(absent.blockers[0].kind, 'NEEDS_PRODUCT_DECISION')
  assert.match(verificationConfigurationIssue({ ...contract, verificationCommands: Array(33).fill('test') })!, /32/)
  assert.match(verificationConfigurationIssue({ ...contract, verificationCommands: ['x'.repeat(2001)] })!, /2000/)
  assert.ok(verificationConfigurationIssue({ ...contract, verificationCommands: [' '] }))
  const invalid = await runLoop({ ...contract, verificationCommands: ['test\0skip'] }, ['dev'], {
    ...runtime(green()), runPhase: async () => { throw new Error('Invalid configuration must not start a role.') },
  })
  assert.equal(invalid.status, 'blocked')
  assert.equal(invalid.cycles, 0)
  assert.match(invalid.blockers[0].issue, /NUL/)

  for (const flags of [{ timedOut: true }, { cancelled: true }, { signal: 'SIGTERM' }]) {
    assert.notEqual(verificationCheck('test', { code: 0, stdout: 'passed', stderr: '', ...flags }).outcome, 'passed')
  }
  assert.notEqual(verificationCheck('test', { code: null, stdout: 'passed', stderr: '' }).outcome, 'passed')

  for (const name of ['AbortError', 'TimeoutError', 'Error']) {
    let calls = 0
    const rt = createLoopRuntime(contract, {
      cwd: '/fixture',
      workerModel: { providerId: 'fixture', model: 'fixture' },
      criticModel: { providerId: 'fixture', model: 'fixture' },
      createRoleThread: async () => ({ id: 'fixture' }),
      runRoleThread: async () => [],
      cleanupThread: async () => {},
      runCritic: async () => '',
      checkpoint: async () => 'fixture',
      rollback: async () => {},
      emit: () => {},
      runShell: async () => { calls++; const error = new Error('command did not finish'); error.name = name; throw error },
    })
    const report = await rt.runVerification(contract)
    assert.equal(report.checks[0].outcome, name === 'AbortError' ? 'cancelled' : name === 'TimeoutError' ? 'timed_out' : 'error')
    assert.equal(report.checks[0].exitCode, null)
    assert.equal(calls, name === 'AbortError' ? 1 : 2)
    assert.ok(verificationIssues(contract.verificationCommands, report).length)
  }
  console.log('Loop verification gates: failures, missing checks, cancellation and execution errors cannot ship.')
}

main().catch(error => { console.error(error); process.exitCode = 1 })
