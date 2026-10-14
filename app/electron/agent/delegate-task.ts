import type { Provider } from './types'

// Concurrency-limited parallel sub-agent delegation. This is the orchestration layer
// that the explore subagent (electron/agent/loop.ts) was missing: it previously ran
// research children one at a time, awaited sequentially in a for-loop. delegateTasks
// runs up to `maxConcurrent` children at once (a small semaphore over deps.runOneChild),
// while still returning results in the caller's original input order.
//
// Deliberately does NOT drive an LLM itself — that logic (streaming, tool execution,
// approval gating) lives in loop.ts and has heavy dependencies. deps.runOneChild is
// injected so this module stays pure and fully unit-testable with a fake child runner.

export interface DelegateTaskInput {
  task: string
  // Restrict the child to this subset of tool names; absent => caller's safe default
  // (loop.ts defaults to the same read-only set the original Explore subagent used).
  toolNames?: string[]
  maxSteps?: number
}

export interface DelegateTaskDeps {
  provider: Provider
  model: string
  cwd: string
  signal: AbortSignal
  runOneChild: (input: DelegateTaskInput, childIndex: number) => Promise<{ result: string; error?: string }>
  // How many children may run at once. Defaults to 3.
  maxConcurrent?: number
}

export interface DelegateTaskResult {
  task: string
  result: string
  ok: boolean
  error?: string
}

const DEFAULT_MAX_CONCURRENT = 3

/**
 * Run `input.length` child tasks through `deps.runOneChild`, at most `deps.maxConcurrent`
 * in flight at a time, and return one DelegateTaskResult per input IN ORIGINAL INPUT ORDER
 * regardless of which child finishes first.
 *
 * If deps.signal is aborted (before or during the run), no further children are launched;
 * any input that never got to run is reported as `{ ok: false, error: 'aborted...' }` in
 * its original slot. Children already in flight when the abort happens are allowed to
 * settle (deps.runOneChild is responsible for honoring the signal itself if it wants to
 * cancel in-flight work faster).
 */
export async function delegateTasks(inputs: DelegateTaskInput[], deps: DelegateTaskDeps): Promise<DelegateTaskResult[]> {
  const maxConcurrent = Math.max(1, deps.maxConcurrent ?? DEFAULT_MAX_CONCURRENT)
  const results: DelegateTaskResult[] = inputs.map((input) => ({
    task: input.task,
    result: '',
    ok: false,
    error: 'Not launched.',
  }))
  // Track which slots actually ran with a separate flag rather than string-matching the
  // sentinel error text: a child that genuinely fails with the literal message "Not launched."
  // must not be misreported as never-launched.
  const launched: boolean[] = inputs.map(() => false)

  let nextIndex = 0
  const launchNext = async (): Promise<void> => {
    while (true) {
      if (deps.signal.aborted) return
      const index = nextIndex
      if (index >= inputs.length) return
      nextIndex++
      const input = inputs[index]
      launched[index] = true
      try {
        const child = await deps.runOneChild(input, index)
        if (child.error) {
          results[index] = { task: input.task, result: child.result ?? '', ok: false, error: child.error }
        } else {
          results[index] = { task: input.task, result: child.result, ok: true }
        }
      } catch (e: any) {
        results[index] = { task: input.task, result: '', ok: false, error: e?.message ?? String(e) }
      }
    }
  }

  // A small semaphore: spin up `maxConcurrent` workers, each pulling the next unclaimed
  // input off a shared cursor until the queue is drained or the run is aborted. This keeps
  // exactly `maxConcurrent` children in flight at once (not "batches of N awaited together",
  // which would idle a slot until the slowest member of a batch finishes).
  const workerCount = Math.min(maxConcurrent, inputs.length)
  const workers = Array.from({ length: workerCount }, () => launchNext())
  await Promise.all(workers)

  // Anything that never ran was aborted before its turn — flag it by the launched marker,
  // not by matching the placeholder error text (which a real child failure could collide with).
  for (let i = 0; i < results.length; i++) {
    if (!launched[i] && !results[i].ok) {
      results[i] = { ...results[i], error: 'Not launched: run was aborted before this task started.' }
    }
  }

  return results
}
