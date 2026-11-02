import assert from 'node:assert/strict'
import path from 'node:path'
import { EngineSupervisor } from './supervisor'
import { engineReadinessProbe, type EngineReadinessProbe } from './readiness'
import { makeHarness, processAlive, waitFor } from './test-fixtures/harness'

async function main() {
  const harness = await makeHarness()
  try {
    for (const stage of ['models', 'context', 'context-error', 'spawn-error'] as const) {
      let pid: number | null = null
      let failedPid: number | null = null
      let readyEvents = 0
      let injectFailure = true
      const terminate = async () => {
        assert.ok(pid)
        failedPid = pid
        process.kill(pid, 'SIGKILL')
        await waitFor(() => pid == null, 5000, 'supervisor to observe child exit')
      }
      const readiness: EngineReadinessProbe = {
        models: async (port, key) => {
          const result = await engineReadinessProbe.models(port, key)
          if (result && injectFailure && stage === 'models') await terminate()
          return result
        },
        context: async (port, key, model, models) => {
          if (injectFailure && stage === 'context-error') {
            failedPid = pid
            throw new Error('Context lookup could not complete.')
          }
          if (injectFailure && stage === 'context') await terminate()
          return engineReadinessProbe.context(port, key, model, models)
        },
      }
      const supervisor = new EngineSupervisor({
        binary: () => injectFailure && stage === 'spawn-error'
          ? { command: path.join(harness.root, 'missing-engine'), args: [] } : harness.binary,
        env: () => harness.env,
        info: async () => ({ family: 'Fixture', modelType: 'qwen3_5', format: 'mlx-q4g64',
          contextTokens: 32768, weightsBytes: 8192, memoryFloorBytes: 8192 }),
        drafterFor: () => null,
        drafterBytes: async () => null,
        totalMemoryBytes: 64 * 1024 ** 3,
        pollMs: 5,
        stopGraceMs: 250,
        readyTimeoutMs: 5000,
        readiness,
        onPid: value => { pid = value },
        onChange: () => { if (supervisor.status().state === 'ready') readyEvents++ },
      })
      try {
        await assert.rejects(() => supervisor.serve('fixture/model'),
          /engine exited|stopped before readiness|Context lookup|Could not start the engine/)
        assert.equal(supervisor.status().state, 'failed', stage)
        assert.equal(supervisor.live(), null, stage)
        assert.equal(supervisor.status().port, null, stage)
        assert.equal(readyEvents, 0, 'a dead or unverified process must never publish ready')
        if (stage !== 'spawn-error') {
          assert.ok(failedPid)
          await waitFor(() => !processAlive(failedPid!), 5000, 'failed startup child to stop')
        }
        injectFailure = false
        const live = await supervisor.serve('fixture/model')
        assert.equal(live.model, 'fixture/model')
        assert.equal(supervisor.status().state, 'ready', 'retry starts and verifies a new process')
        assert.equal(readyEvents, 1)
      } finally {
        await supervisor.stop()
      }
    }
    console.log('Engine readiness: probe exits, lookup errors and failed spawns fail closed; retry recovers.')
  } finally {
    await harness.cleanup()
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
