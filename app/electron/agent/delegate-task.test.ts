// Run: npx jiti electron/agent/delegate-task.test.ts
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
  // --- Real concurrency up to maxConcurrent (not sequential) -----------------
  {
    const inputs: DelegateTaskInput[] = [
      { task: 'a' }, { task: 'b' }, { task: 'c' },
    ]
    const starts: number[] = []
    const ends: number[] = []
    const runOneChild = async (input: DelegateTaskInput, childIndex: number) => {
      starts.push(Date.now())
      await new Promise((resolve) => setTimeout(resolve, 60))
      ends.push(Date.now())
      return { result: `${childIndex}:${input.task}` }
    }
    const deps = baseDeps({ runOneChild, maxConcurrent: 3 })
    const t0 = Date.now()
    const results = await delegateTasks(inputs, deps)
    const elapsed = Date.now() - t0

    assert.equal(results.length, 3)
    // All 3 should have overlapped: every start happened before the FIRST end.
    const firstEnd = Math.min(...ends)
    assert.ok(starts.every((s) => s <= firstEnd), `expected all starts (${starts}) <= first end (${firstEnd}) to prove overlap`)
    // Sequential would take ~180ms; concurrent should be close to one slot (~60-150ms).
    assert.ok(elapsed < 150, `expected concurrent run to finish well under sequential time, took ${elapsed}ms`)
    ok('runs up to maxConcurrent children with real time overlap (not sequential)')
  }

  // --- Concurrency is actually LIMITED to maxConcurrent (batches) ------------
  {
    const inputs: DelegateTaskInput[] = Array.from({ length: 6 }, (_, i) => ({ task: `t${i}` }))
    let inFlight = 0
    let maxInFlight = 0
    const runOneChild = async (input: DelegateTaskInput) => {
      inFlight++
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((resolve) => setTimeout(resolve, 30))
      inFlight--
      return { result: input.task }
    }
    const deps = baseDeps({ runOneChild, maxConcurrent: 2 })
    await delegateTasks(inputs, deps)
    assert.equal(maxInFlight, 2, `expected at most 2 children in flight at once, saw ${maxInFlight}`)
    ok('limits concurrency to maxConcurrent (does not launch all at once)')
  }

  // --- Default maxConcurrent when not specified -------------------------------
  {
    const inputs: DelegateTaskInput[] = Array.from({ length: 5 }, (_, i) => ({ task: `t${i}` }))
    let inFlight = 0
    let maxInFlight = 0
    const runOneChild = async (input: DelegateTaskInput) => {
      inFlight++
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((resolve) => setTimeout(resolve, 20))
      inFlight--
      return { result: input.task }
    }
    const deps = baseDeps({ runOneChild }) // no maxConcurrent given
    await delegateTasks(inputs, deps)
    assert.ok(maxInFlight > 1, 'default should still run concurrently')
    assert.ok(maxInFlight <= 3, `default maxConcurrent should be a small sane default (<=3), saw ${maxInFlight}`)
    ok('defaults maxConcurrent to a sane small value (e.g. 3) when unspecified')
  }

  // --- Original input order preserved regardless of completion order ---------
  {
    const inputs: DelegateTaskInput[] = [
      { task: 'slow' }, // finishes last
      { task: 'fast' }, // finishes first
      { task: 'mid' },
    ]
    const delays: Record<string, number> = { slow: 90, fast: 10, mid: 40 }
    const runOneChild = async (input: DelegateTaskInput) => {
      await new Promise((resolve) => setTimeout(resolve, delays[input.task]))
      return { result: `${input.task}-result` }
    }
    const deps = baseDeps({ runOneChild, maxConcurrent: 3 })
    const results = await delegateTasks(inputs, deps)
    assert.deepEqual(results.map((r) => r.task), ['slow', 'fast', 'mid'])
    assert.deepEqual(results.map((r) => r.result), ['slow-result', 'fast-result', 'mid-result'])
    assert.ok(results.every((r) => r.ok))
    ok('collects results in ORIGINAL input order regardless of completion order')
  }

  // --- Abort mid-flight stops launching new children --------------------------
  {
    const inputs: DelegateTaskInput[] = Array.from({ length: 6 }, (_, i) => ({ task: `t${i}` }))
    const controller = new AbortController()
    const started: string[] = []
    const runOneChild = async (input: DelegateTaskInput) => {
      started.push(input.task)
      await new Promise((resolve) => setTimeout(resolve, 30))
      return { result: `${input.task}-done` }
    }
    const deps = baseDeps({ runOneChild, maxConcurrent: 2, signal: controller.signal })
    // Abort almost immediately, before the first batch's children finish.
    setTimeout(() => controller.abort(), 5)
    const results = await delegateTasks(inputs, deps)

    assert.equal(results.length, 6, 'still returns one result entry per input')
    const ran = results.filter((r) => r.ok)
    const notRan = results.filter((r) => !r.ok)
    assert.ok(notRan.length > 0, 'at least some children should not have been launched after abort')
    assert.ok(started.length < inputs.length, `expected fewer children launched than total after abort, started=${started.length}`)
    for (const r of notRan) {
      assert.ok(typeof r.error === 'string' && r.error.length > 0, 'unran entries must carry an error message')
    }
    // Order is still preserved for whatever DID run.
    assert.deepEqual(results.map((r) => r.task), inputs.map((i) => i.task))
    ok('stops launching new children once aborted, returns ok:false placeholders for the rest, preserves order')
  }

  // --- Already-aborted signal launches nothing --------------------------------
  {
    const inputs: DelegateTaskInput[] = [{ task: 'a' }, { task: 'b' }]
    const controller = new AbortController()
    controller.abort()
    let calls = 0
    const runOneChild = async (input: DelegateTaskInput) => { calls++; return { result: input.task } }
    const deps = baseDeps({ runOneChild, signal: controller.signal })
    const results = await delegateTasks(inputs, deps)
    assert.equal(calls, 0, 'no children should run when the signal starts already aborted')
    assert.ok(results.every((r) => !r.ok))
    ok('an already-aborted signal launches zero children')
  }

  // --- Single-task input works -------------------------------------------------
  {
    const inputs: DelegateTaskInput[] = [{ task: 'solo task', maxSteps: 5 }]
    const deps = baseDeps({
      runOneChild: async (input, childIndex) => {
        assert.equal(childIndex, 0)
        assert.equal(input.task, 'solo task')
        assert.equal(input.maxSteps, 5)
        return { result: 'solo result' }
      },
    })
    const results = await delegateTasks(inputs, deps)
    assert.equal(results.length, 1)
    assert.deepEqual(results[0], { task: 'solo task', result: 'solo result', ok: true })
    ok('a single-task input works end to end')
  }

  // --- Empty input ---------------------------------------------------------------
  {
    const deps = baseDeps({ runOneChild: async () => { throw new Error('should not be called') } })
    const results = await delegateTasks([], deps)
    assert.deepEqual(results, [])
    ok('an empty input list returns an empty result list without invoking runOneChild')
  }

  // --- A child error is captured as ok:false with the error message, not thrown --
  {
    const inputs: DelegateTaskInput[] = [{ task: 'will fail' }, { task: 'will pass' }]
    const runOneChild = async (input: DelegateTaskInput) => {
      if (input.task === 'will fail') return { result: '', error: 'boom: child crashed' }
      return { result: 'ok output' }
    }
    const deps = baseDeps({ runOneChild })
    const results = await delegateTasks(inputs, deps)
    assert.equal(results[0].ok, false)
    assert.equal(results[0].error, 'boom: child crashed')
    assert.equal(results[1].ok, true)
    ok('a child reporting an error produces ok:false with the error message (not a thrown rejection)')
  }

  // --- A throwing runOneChild does not crash the whole batch ---------------------
  {
    const inputs: DelegateTaskInput[] = [{ task: 'throws' }, { task: 'fine' }]
    const runOneChild = async (input: DelegateTaskInput) => {
      if (input.task === 'throws') throw new Error('unexpected rejection')
      return { result: 'fine result' }
    }
    const deps = baseDeps({ runOneChild })
    const results = await delegateTasks(inputs, deps)
    assert.equal(results[0].ok, false)
    assert.match(results[0].error ?? '', /unexpected rejection/)
    assert.equal(results[1].ok, true)
    assert.equal(results[1].result, 'fine result')
    ok('a throwing runOneChild is caught per-child and does not abort the rest of the batch')
  }

  console.log(`\n${passed} checks passed.`)
}

run().catch((e) => {
  console.error(e)
  process.exit(1)
})
