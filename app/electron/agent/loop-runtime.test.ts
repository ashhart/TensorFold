import assert from 'node:assert/strict'
import { createLoopRuntime, type LoopRuntimeDeps, type RoleTimelineItem } from './loop-runtime'
import { runLoop } from './loop-engine'
import { buildRunContract } from './loop-contract'

function deps(over: Partial<LoopRuntimeDeps> = {}): LoopRuntimeDeps {
  let threadN = 0
  return {
    cwd: '/wt',
    projectId: null,
    workerModel: { providerId: 'p', model: 'worker' },
    criticModel: { providerId: 'p', model: 'critic' },
    createRoleThread: async () => ({ id: `t${++threadN}` }),
    runRoleThread: async (): Promise<RoleTimelineItem[]> => [
      { kind: 'tool', tool: 'edit_file', state: 'ok' },
      { kind: 'assistant', text: 'did the work' },
    ],
    cleanupThread: async () => {},
    runShell: async () => ({ code: 0, stdout: 'PASS 3', stderr: '' }),
    runCritic: async () => '{"passed":true,"confidence":0.9,"summary":"ok","issues":[]}',
    checkpoint: async () => 'cp',
    rollback: async () => {},
    emit: () => {},
    ...over,
  }
}

async function main() {
  const contract = buildRunContract({ source: 'solo', mode: 'build', goal: 'x', verify: ['npm test'] })

  // runPhase derives a PhaseResult from the role thread's items
  const rt = createLoopRuntime(contract, deps())
  const pr = await rt.runPhase('dev', contract, 0)
  assert.equal(pr.toolEvents, 1)
  assert.match(pr.summary, /did the work/)

  // critique parses the critic model's JSON
  assert.equal((await rt.critique(contract, 'evidence')).passed, true)

  // runVerification runs the commands
  const verification = await rt.runVerification(contract)
  assert.equal(verification.checks[0].exitCode, 0)
  assert.equal(verification.checks[0].outcome, 'passed')
  assert.match(verification.checks[0].output, /PASS 3/)

  // cleanupThread is always called (even though run succeeded)
  let cleaned = 0
  const rt2 = createLoopRuntime(contract, deps({ cleanupThread: async () => { cleaned++ } }))
  await rt2.runPhase('dev', contract, 0)
  assert.equal(cleaned, 1)

  // end-to-end: the engine drives this runtime to a ship
  const result = await runLoop(contract, ['dev'], createLoopRuntime(contract, deps()))
  assert.equal(result.status, 'shipped')

  const red = await runLoop({ ...contract, maxCycles: 1 }, ['dev'], createLoopRuntime(contract, deps({
    runShell: async () => ({ code: 1, stdout: 'FAIL: invariant broken', stderr: '' }),
  })))
  assert.equal(red.status, 'capped', 'a confident critic cannot waive a failing required command')
  assert.equal(red.lastVerdict?.passed, false)
  assert.match(red.lastVerdict?.issues.join('\n') ?? '', /npm test.*failed.*exit 1/)

  const recovered = await runLoop({ ...contract, maxCycles: 2 }, ['dev'], createLoopRuntime(contract, deps({
    runShell: (() => { let calls = 0; return async () => ({ code: calls++ ? 0 : 1, stdout: '', stderr: '' }) })(),
  })))
  assert.equal(recovered.status, 'shipped')
  assert.equal(recovered.cycles, 2)

  let cancelled = false
  const stopped = await runLoop(contract, ['dev'], createLoopRuntime(contract, deps({
    isCancelled: () => cancelled,
    runCritic: async () => { cancelled = true; return '{"passed":true,"confidence":1,"summary":"ok","issues":[]}' },
  })))
  assert.equal(stopped.status, 'stopped', 'Stop requested during critique prevents completion')
  assert.equal(stopped.lastVerdict?.passed, false)

  console.log('loop-runtime tests passed')
}
main().catch((e) => { console.error(e); process.exit(1) })
