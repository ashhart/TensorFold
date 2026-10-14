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
  assert.match(await rt.runVerification(contract), /PASS 3/)

  // cleanupThread is always called (even though run succeeded)
  let cleaned = 0
  const rt2 = createLoopRuntime(contract, deps({ cleanupThread: async () => { cleaned++ } }))
  await rt2.runPhase('dev', contract, 0)
  assert.equal(cleaned, 1)

  // end-to-end: the engine drives this runtime to a ship
  const result = await runLoop(contract, ['dev'], createLoopRuntime(contract, deps()))
  assert.equal(result.status, 'shipped')

  console.log('loop-runtime tests passed')
}
main().catch((e) => { console.error(e); process.exit(1) })
