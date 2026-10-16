// Run: npx jiti electron/agent/delegate-task.not-launched.test.ts
import assert from 'node:assert/strict'
import { delegateTasks, type DelegateTaskDeps, type DelegateTaskInput } from './delegate-task'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

function baseDeps(overrides: Partial<DelegateTaskDeps> = {}): DelegateTaskDeps {
  return {
    provider: { id: 'p', name: 'P', baseUrl: '', apiKey: '', models: [] } as any,
    model: 'gpt-4o',
    cwd: '/tmp/project',
    signal: new AbortController().signal,
    runOneChild: async (input) => ({ result: `done: ${input.task}` }),
    ...overrides,
  }
}

async function run() {
  // --- A child that genuinely fails with the literal sentinel string is NOT
  //     misreported as never-launched. -----------------------------------------
  {
    const inputs: DelegateTaskInput[] = [{ task: 'real failure' }]
    const runOneChild = async () => ({ result: '', error: 'Not launched.' })
    const deps = baseDeps({ runOneChild })
    const results = await delegateTasks(inputs, deps)
    assert.equal(results.length, 1)
    assert.equal(results[0].ok, false)
    // The slot DID run, so the original error must be preserved verbatim — not rewritten
    // into the "aborted before this task started" message.
    assert.equal(results[0].error, 'Not launched.')
    ok('a real child failure with error "Not launched." is preserved, not rewritten as never-launched')
  }

  // --- A child that THROWS an Error whose message is exactly the sentinel is also
  //     preserved. -----------------------------------------------------------------
  {
    const inputs: DelegateTaskInput[] = [{ task: 'throws sentinel' }]
    const runOneChild = async () => { throw new Error('Not launched.') }
    const deps = baseDeps({ runOneChild })
    const results = await delegateTasks(inputs, deps)
    assert.equal(results[0].ok, false)
    assert.equal(results[0].error, 'Not launched.')
    ok('a thrown Error("Not launched.") is captured verbatim, not treated as never-launched')
  }

  // --- A truly never-launched slot (aborted before its turn) still gets the
  //     aborted-before-start message. -------------------------------------------
  {
    const inputs: DelegateTaskInput[] = Array.from({ length: 4 }, (_, i) => ({ task: `t${i}` }))
    const controller = new AbortController()
    const runOneChild = async (input: DelegateTaskInput) => {
      await new Promise((resolve) => setTimeout(resolve, 30))
      return { result: input.task }
    }
    const deps = baseDeps({ runOneChild, maxConcurrent: 1, signal: controller.signal })
    setTimeout(() => controller.abort(), 5)
    const results = await delegateTasks(inputs, deps)
    const notRan = results.filter((r) => !r.ok)
    assert.ok(notRan.length > 0, 'some slots should never have launched after abort')
    for (const r of notRan) {
      assert.match(r.error ?? '', /aborted before this task started/)
    }
    ok('genuinely never-launched slots still report the aborted-before-start message')
  }

  console.log(`\n${passed} checks passed.`)
}

run().catch((e) => {
  console.error(e)
  process.exit(1)
})
